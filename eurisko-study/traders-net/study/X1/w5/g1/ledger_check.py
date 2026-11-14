"""X1-w5 G-b/G-c/G-g: recompute the G1 ledger from GCE operation receipts, independently of cost-audit.py.
Read-only. Run: python3 ledger_check.py > ledger_check.json"""
import datetime as dt, json, pathlib, re, sys, collections
G1 = pathlib.Path("/home/agent/workspaces/eurisko-study/traders-net/study/G1")
RATE = 0.87
log = (G1 / "GPU-LOG.md").read_text().splitlines()
kv = lambda l: dict(x.split("=", 1) for x in l.split() if "=" in x)
ts = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
vms = collections.OrderedDict()
for l in log:
    m = re.match(r"- (\S+) (\S+) ", l)
    if not m: continue
    t, ev = m.groups(); f = kv(l); vm = f.get("vm")
    if not vm: continue
    d = vms.setdefault(vm, dict(vm=vm, ws=f.get("ws"), events=[]))
    d["events"].append(ev)
    if ev == "CREATE": d["create_log"] = t; d["zone"] = f.get("zone")
    if ev == "READY": d["ready_log"] = t
    if ev == "COST-AUDIT": d["ledger_cost"] = float(f["cost_usd"]); d["basis"] = f.get("basis")
    if ev == "CREATE-FAILED": d["zones_tried"] = l.split("zones=")[1].split(" err=")[0]
rows, optypes = [], collections.Counter()
for vm, d in vms.items():
    p = G1 / "infra/state/audit" / vm / "operations.json"
    ops = json.loads(p.read_text()) if p.exists() else []
    for o in ops: optypes[o["operationType"]] += 1
    ins = [ts(o["insertTime"]) for o in ops if o["operationType"] == "insert" and not o.get("error")]
    ends = [(ts(o["endTime"]), o["operationType"]) for o in ops if o["operationType"] != "insert" and o.get("endTime") and not o.get("error")]
    if ins and ends:
        t0, (t1, kind) = min(ins), max(ends)
        d["recomputed_cost"] = round((t1 - t0).total_seconds() / 3600 * RATE, 6)
        d["alive_from"], d["alive_to"], d["end_kind"] = t0.isoformat(), t1.isoformat(), kind
        d["op_types"] = sorted({o["operationType"] for o in ops})
    else:
        d["recomputed_cost"] = 0.0 if "CREATE-FAILED" in d["events"] else None
    if "create_log" in d and "ready_log" in d:
        d["create_to_ready_min"] = round((ts(d["ready_log"]) - ts(d["create_log"])).total_seconds() / 60, 2)
    d["diff"] = None if d.get("recomputed_cost") is None or "ledger_cost" not in d else round(d["recomputed_cost"] - d["ledger_cost"], 6)
    d["preempted"] = any(k == "compute.instances.preempted" for k in d.get("op_types", []))
    d["run_end_logged"] = "RUN-END" in d["events"]
    rows.append(d)
# concurrency from receipts
iv = [(ts(r["alive_from"]), ts(r["alive_to"]), r["vm"]) for r in rows if "alive_from" in r]
pts = sorted([(a, 1) for a, b, _ in iv] + [(b, -1) for a, b, _ in iv])
cur = mx = 0
for _, s in pts: cur += s; mx = max(mx, cur)
over2 = [(x[2], y[2], z[2]) for i, x in enumerate(iv) for j, y in enumerate(iv) if j > i for k, z in enumerate(iv) if k > j
         if max(x[0], y[0], z[0]) < min(x[1], y[1], z[1])]
pairs = sum(1 for i, x in enumerate(iv) for y in iv[i + 1:] if max(x[0], y[0]) < min(x[1], y[1]))
by_ws = collections.defaultdict(float)
for r in rows: by_ws[r["ws"]] += r.get("ledger_cost") or 0.0
cost_md = (G1 / "COST.md").read_text()
caps = {l.split()[0]: float(l.split()[1]) for l in (G1 / "infra/caps.tsv").read_text().splitlines() if l.strip()}
ready = [r["create_to_ready_min"] for r in rows if "create_to_ready_min" in r]
out = dict(
    checked_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    n_vm_names=len(rows), n_audited=sum("ledger_cost" in r for r in rows),
    max_abs_diff_recomputed_vs_ledger=max(abs(r["diff"]) for r in rows if r["diff"] is not None),
    total_ledger=round(sum(by_ws.values()), 4), by_ws={k: round(v, 4) for k, v in sorted(by_ws.items())},
    smoke_G1=[round(r["ledger_cost"], 4) for r in rows if r["ws"] == "G1"],
    cost_md_total_line=[l for l in cost_md.splitlines() if l.startswith("- **Total")],
    caps=caps, caps_sum=round(sum(caps.values()), 2),
    max_concurrent_vms_from_receipts=mx, overlapping_pairs=pairs, triples_over_cap=over2,
    op_types_seen=dict(optypes),
    preempted_vms=[(r["vm"], r["run_end_logged"]) for r in rows if r["preempted"]],
    create_to_ready_min=dict(n=len(ready), min=min(ready), max=max(ready), median=sorted(ready)[len(ready)//2], values=ready),
    zone_orders_in_create_failed=[r["zones_tried"] for r in rows if "zones_tried" in r],
    zones_created=collections.Counter(r.get("zone") for r in rows if r.get("zone")),
    refused_jobs=sorted(p.name for p in (G1 / "queue/refused").iterdir()),
    rows=rows)
print(json.dumps(out, indent=1, default=str))
