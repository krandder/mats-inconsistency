#!/usr/bin/env python3
"""Actual P15 VM-hours from GCP operations (last successful insert -> preempted/delete), priced at spot $0.52/h
and at the $0.86/h ceiling, next to the watchdog's ledger estimate (which counts until it noticed a VM gone)."""
import subprocess, json, re, datetime as dt, sys
P = "/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/P15/gpu/LEDGER.md"
ops = json.loads(subprocess.run(
    ["gcloud", "compute", "operations", "list", "--project", "GCP_PROJECT",
     "--filter=targetLink~trader-b3-p15- AND (operationType=insert OR operationType=delete OR operationType=compute.instances.preempted)",
     "--format=json"], capture_output=True, text=True, check=True).stdout)
ts = lambda s: dt.datetime.fromisoformat(s).timestamp()
vm = {}
for o in ops:
    n = o["targetLink"].rsplit("/", 1)[1]
    d = vm.setdefault(n, {"ins": [], "end": None, "how": None})
    if o["operationType"] == "insert" and not o.get("error"):
        d["ins"].append(ts(o["endTime"]))
    elif o["operationType"] in ("delete", "compute.instances.preempted"):
        e = ts(o["insertTime"])
        if d["end"] is None or e < d["end"]:
            d["end"], d["how"] = e, ("preempted" if "preempted" in o["operationType"] else "deleted")
led = open(P).read()
tot = 0.0; rows = []
for n in sorted(vm, key=lambda x: int(x.rsplit("-", 1)[1])):
    d = vm[n]
    m = re.findall(rf"CREATE vm={n} .*?epoch=(\d+)", led)
    start = float(m[-1]) if m else (max(d["ins"]) if d["ins"] else None)
    end = d["end"] or dt.datetime.now().timestamp()
    if start is None: continue
    h = max(0.0, end - start) / 3600; tot += h
    rows.append((n, d["how"] or "running", round(h, 3)))
for r in rows: print(*r, sep="\t")
print(f"total VM-h {tot:.3f}  spot $ {tot*0.52:.2f}  ceiling $ {tot*0.86:.2f}")
