#!/usr/bin/env python3
"""R8-E1 audit, check 7: recompute R8-E1's GPU spend from logical-soup/GPU-LOG.md (soup-gpu-r8e1-* lines) with the auditor's own parser,
cross-check it against the GCE operation times (a7_gcloud.txt, read-only gcloud), and time the teardown after the run ended.
Run: python3 a7_cost.py"""
import csv
import io
import json
import os
import re
from datetime import datetime, timezone

OUT = os.path.dirname(os.path.abspath(__file__))
LOG = "/home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md"
RATE, CEIL, GROUP_CAP = 0.87, 0.80, 2.20
rep = {"vms": {}, "r8_group": {}}
for line in open(LOG):
    m = re.search(r"vm=(soup-gpu-(r8e\d)-[a-z0-9-]+)", line)
    if not m:
        continue
    name, lane = m.group(1), m.group(2)
    ev = line.split()[2]
    ts = line[2:19]
    if lane == "r8e1":
        v = rep["vms"].setdefault(name, {"lines": []})
        v["lines"].append(f"{ts} {ev}")
        if ev == "CREATE":
            v["create_min"] = ts
        if ev == "RUN-END":
            v["run_end_min"] = ts
        if ev == "DELETE":
            v["delete_min"] = ts
            v["age_h"] = float(re.search(r"age_h=([0-9.]+)", line).group(1))
            v["cost_usd_logged"] = float(re.search(r"cost_usd=([0-9.]+)", line).group(1))
    if ev == "DELETE":
        rep["r8_group"].setdefault(lane, 0.0)
        rep["r8_group"][lane] += float(re.search(r"cost_usd=([0-9.]+)", line).group(1))
tot_logged = sum(v["cost_usd_logged"] for v in rep["vms"].values())
tot_age = sum(RATE * v["age_h"] for v in rep["vms"].values())
# GCE operations (insert start -> delete end), UTC
txt = open(os.path.join(OUT, "a7_gcloud.txt")).read()
rows = [r for r in csv.DictReader(io.StringIO("\n".join(l for l in txt.splitlines() if not l.startswith("#"))))]
p = lambda s: datetime.fromisoformat(s).astimezone(timezone.utc)
ops = {}
for r in rows:
    o = ops.setdefault(r["target"], {})
    o[r["type"] + "_start"] = p(r["start_time"])
    o[r["type"] + "_end"] = p(r["end_time"])
tot_ops = 0.0
for n, o in ops.items():
    life_h = (o["delete_end"] - o["insert_start"]).total_seconds() / 3600
    to_delete_issue_h = (o["delete_start"] - o["insert_start"]).total_seconds() / 3600
    rep["vms"][n]["gce_insert_start"] = o["insert_start"].strftime("%H:%M:%S")
    rep["vms"][n]["gce_delete_start"] = o["delete_start"].strftime("%H:%M:%S")
    rep["vms"][n]["gce_delete_end"] = o["delete_end"].strftime("%H:%M:%S")
    rep["vms"][n]["life_h_insert_to_delete_end"] = round(life_h, 4)
    rep["vms"][n]["cost_ceiling_insert_to_delete_end"] = round(RATE * life_h, 4)
    rep["vms"][n]["cost_ceiling_insert_to_delete_issue"] = round(RATE * to_delete_issue_h, 4)
    tot_ops += RATE * life_h
rep["total_logged_usd"] = round(tot_logged, 4)
rep["total_rate_x_logged_age_usd"] = round(tot_age, 4)
rep["total_ceiling_rate_insert_to_delete_end_usd"] = round(tot_ops, 4)
rep["ceiling_usd"] = CEIL
rep["within_ceiling_logged"] = tot_logged <= CEIL
rep["within_ceiling_gce_times"] = tot_ops <= CEIL
rep["headroom_logged_usd"] = round(CEIL - tot_logged, 4)
rep["r8_group"] = {k: round(v, 4) for k, v in rep["r8_group"].items()}
rep["r8_group_total_deleted_vms_usd"] = round(sum(rep["r8_group"].values()), 4)
# teardown: the scored run's remote end (driver.txt "remote rc=0") vs the GCE delete start
drv = open("/tmp/r8e1/logs/main/driver.txt").read()
end = re.findall(r"(\S+Z) remote rc=0", drv)[-1]
rend = datetime.strptime(end, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
d2 = ops["soup-gpu-r8e1-main2-1017163715-30043"]
rep["scored_run_remote_end_utc"] = end
rep["teardown_delete_issued_after_remote_end_s"] = round((d2["delete_start"] - rend).total_seconds(), 1)
rep["teardown_delete_done_after_remote_end_s"] = round((d2["delete_end"] - rend).total_seconds(), 1)
d1 = ops["soup-gpu-r8e1-main-1017160722-4433"]
first_crash = datetime(2026, 10, 17, 16, 24, 40, tzinfo=timezone.utc)      # last stdout.txt write of the 3 crashed first attempts
rep["run1_crash_to_delete_issue_min"] = round((d1["delete_start"] - first_crash).total_seconds() / 60, 2)
rep["run1_spend_after_crash_ceiling_usd"] = round(RATE * (d1["delete_end"] - first_crash).total_seconds() / 3600, 4)
print(json.dumps(rep, indent=1, default=str))
json.dump(rep, open(os.path.join(OUT, "a7_cost.json"), "w"), indent=1, default=str)
