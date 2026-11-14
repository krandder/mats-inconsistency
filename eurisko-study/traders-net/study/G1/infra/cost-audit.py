#!/usr/bin/env python3
"""Verify a traders-net rental is gone and record its operation-based cost (ceiling rate).

Usage: cost-audit.py NAME          audit one VM (must already be deleted)
       cost-audit.py --pending     audit every VM in GPU-LOG.md that has no COST-AUDIT line yet

Adapted from logical-soup/scripts/gpu-rental-audit.py. Evidence (instances/disks/operations JSON)
goes to infra/state/audit/NAME/. Appends one COST-AUDIT line to GPU-LOG.md. Duration = first
successful insert to the last completed delete/preemption receipt. If no insert succeeded and
the log has CREATE-FAILED for the VM, cost is 0 (basis=create-failed-no-insert). Anything else
incomplete exits 1 and leaves the ledger on its DELETE estimate or full reserve (conservative).
"""
import datetime, json, os, pathlib, re, subprocess, sys

INFRA = pathlib.Path(__file__).resolve().parent
G1 = pathlib.Path(os.environ.get("G1_ROOT", INFRA.parent))
LOG = G1 / "GPU-LOG.md"
RATES = {"g2-standard-8": 0.87}
PROJECT = "GCP_PROJECT"


def q(kind, flt, fmt):
    out = subprocess.check_output(["gcloud", "compute", kind, "list", f"--project={PROJECT}",
                                   "--filter=" + flt, f"--format=json({fmt})"], text=True)
    return json.loads(out)


def fields(line):
    return dict(x.split("=", 1) for x in line.split() if "=" in x)


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def audit(name):
    if not re.fullmatch(r"tn-gpu-[a-z0-9-]+", name):
        sys.exit(f"not a traders-net VM: {name}")
    lines = LOG.read_text().splitlines()
    mine = [l for l in lines if f"vm={name} " in l + " "]
    if any(" COST-AUDIT " in l for l in mine):
        print(f"{name}: already audited"); return 0
    ev = G1 / "infra/state/audit" / name
    ev.mkdir(parents=True, exist_ok=True)
    inst = q("instances", f"name={name}", "name,status")
    disks = q("disks", f"name={name}", "name,zone")
    (ev / "instances.json").write_text(json.dumps(inst, indent=2) + "\n")
    (ev / "disks.json").write_text(json.dumps(disks, indent=2) + "\n")
    if inst or disks:
        print(f"{name}: instance or disk still exists; not audited", file=sys.stderr); return 1
    ops = q("operations", f"targetLink~{name}", "operationType,insertTime,startTime,endTime,targetLink,status,error")
    (ev / "operations.json").write_text(json.dumps(ops, indent=2) + "\n")
    creates = [l for l in mine if " CREATE " in l] or [l for l in mine if " PENDING-CREATE " in l]
    ftype = fields(creates[-1]).get("type", "g2-standard-8") if creates else "g2-standard-8"
    ws = fields(creates[-1]).get("ws", "?") if creates else "?"
    rate = RATES[ftype]
    starts = [o for o in ops if o["operationType"] == "insert" and not o.get("error")]
    if not starts:
        if any(" CREATE-FAILED " in l for l in mine):
            cost, dur, basis = 0.0, 0.0, "create-failed-no-insert"
        else:
            print(f"{name}: no successful insert receipt and no CREATE-FAILED line; not inferring zero", file=sys.stderr)
            return 1
    else:
        ends = [o for o in ops if o["operationType"] in ("delete", "compute.instances.preempted",
                                                          "compute.instances.maxRunDurationTermination")
                and o.get("endTime") and not o.get("error")]
        if not ends:
            print(f"{name}: no completed delete/preemption receipt; ledger keeps its estimate", file=sys.stderr)
            return 1
        t0 = min(datetime.datetime.fromisoformat(o["insertTime"]) for o in starts)
        t1 = max(datetime.datetime.fromisoformat(o["endTime"]) for o in ends)
        dur = (t1 - t0).total_seconds()
        cost, basis = dur * rate / 3600, "insert-to-delete-receipt"
    a = dict(vm=name, ws=ws, machine_type=ftype, duration_s=round(dur, 3), rate_usd_h=rate,
             cost_ceiling_usd=round(cost, 6), rate_verified=False, basis=basis, checked_at=now(),
             instances=[], disks=[])
    (ev / "cost-audit.json").write_text(json.dumps(a, indent=2) + "\n")
    with LOG.open("a") as f:
        f.write(f"- {a['checked_at']} COST-AUDIT vm={name} ws={ws} duration_s={dur:.3f} rate_usd_h={rate} "
                f"cost_usd={cost:.6f} basis={basis} UNVERIFIED-ceiling instances=[] disks=[] evidence=infra/state/audit/{name}/\n")
    print(json.dumps(a))
    return 0


def pending():
    if not LOG.exists():
        return 0
    text = LOG.read_text()
    names = sorted(set(re.findall(r"(?:PENDING-CREATE|CREATE|CREATE-FAILED) vm=(tn-gpu-[a-z0-9-]+)", text)))
    done = set(re.findall(r"COST-AUDIT vm=(tn-gpu-[a-z0-9-]+)", text))
    rc = 0
    for n in names:
        if n not in done:
            rc |= audit(n)
    return rc


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    sys.exit(pending() if sys.argv[1] == "--pending" else audit(sys.argv[1]))
