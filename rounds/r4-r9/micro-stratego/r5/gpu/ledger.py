"""R5-E2 GPU ledger: rebuilds micro-stratego/r5/GPU-LOG-R5-E2.md from the shared logical-soup/GPU-LOG.md (lines for
vm=soup-gpu-r5e2-*) plus a live gcloud listing. Cost per VM = the DELETE line's cost_usd ($0.87/h list-price ceiling x
age; spot is cheaper; Billing API disabled). VMs still open are counted at their reservation (worst case).
   python3 gpu/ledger.py"""
import re
import subprocess
import time

LOG = "/home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md"
OUT = "/home/agent/workspaces/mats/work/micro-stratego/r5/GPU-LOG-R5-E2.md"
CAP = 9.0
vms = {}
for l in open(LOG):
    m = re.search(r"vm=(soup-gpu-r5e2-\S*)", l)
    if not m:
        continue
    n = m.group(1)
    v = vms.setdefault(n, {"zone": "", "model": "", "reserve": 0.0, "cost": None, "age_h": None, "created": "",
                           "deleted": "", "failed": False, "run": "", "reason": ""})
    ts = l.split()[1] if len(l.split()) > 1 else ""
    if " PENDING-CREATE " in l:
        r = re.search(r"reserve_usd=([\d.]+)", l)
        v["reserve"] = float(r.group(1)) if r else v["reserve"]
        ru = re.search(r"run=(\S+)", l)
        v["run"] = ru.group(1) if ru else ""
    if " CREATE vm=" in l:
        v["created"] = ts
        z = re.search(r"zone=(\S+)", l)
        mo = re.search(r"model=(\S+)", l)
        v["zone"] = z.group(1) if z else ""
        v["model"] = mo.group(1) if mo else ""
    if "CREATE-FAILED" in l:
        v["failed"] = True
    if " COST-AUDIT vm=" in l and "preempted" in l:
        v["reason"] = "preempted"
    if " DELETE vm=" in l or " COST-AUDIT vm=" in l:
        c = re.search(r"cost_usd=([\d.]+)", l)
        a = re.search(r"age_h=([\d.]+)", l)
        rs = re.search(r'reason="([^"]*)"', l)
        if c:
            v["cost"] = float(c.group(1))
        if a:
            v["age_h"] = float(a.group(1))
        if rs:
            v["reason"] = rs.group(1)[:60]
        v["deleted"] = ts
live = subprocess.run(["gcloud", "compute", "instances", "list", "--project", "GCP_PROJECT", "--filter", "name~soup-gpu-r5e2-",
                       "--format", "value(name,status,zone.basename())"], capture_output=True, text=True).stdout.strip()
disks = subprocess.run(["gcloud", "compute", "disks", "list", "--project", "GCP_PROJECT", "--filter", "name~soup-gpu-r5e2-",
                        "--format", "value(name)"], capture_output=True, text=True).stdout.strip()
tot = 0.0
rows = []
for n, v in sorted(vms.items(), key=lambda kv: kv[1]["created"] or "z"):
    if v["failed"] and not v["created"]:
        c, basis = 0.0, "create failed, no instance"
    elif v["cost"] is not None and v["reason"] == "preempted":
        c, basis = v["cost"], f"COST-AUDIT line (spot preempted by GCE, auto-deleted), age {v['age_h']} h x $0.87/h ceiling"
        v["deleted"] = "GCE preemption"
    elif v["cost"] is not None:
        c, basis = v["cost"], f"DELETE line, age {v['age_h']} h x $0.87/h ceiling"
    else:
        c, basis = v["reserve"], "still open: counted at reservation (worst case)"
    tot += c
    rows.append(f"| {n} | {v['run']} | {v['zone']} | {v['model']} | {v['created']} | {v['deleted']} | {c:.2f} | {basis} |")
txt = ["# R5-E2 GPU ledger (budget group r5, group cap $22 shared with R5-E1; own ceiling $9)\n",
       f"Rebuilt {time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())} by r5/gpu/ledger.py from logical-soup/GPU-LOG.md. Rates are",
       "the UNVERIFIED on-demand list-price ceiling ($0.87/h for g2-standard-8 + 1 L4; Cloud Billing API disabled). Spot VMs",
       "(provisioning SPOT) are normally 60-70% cheaper; STANDARD is on-demand. True spend is at most this total.\n",
       f"**Total (ceiling): ${tot:.2f}** of the $9.00 R5-E2 ceiling, over {sum(1 for v in vms.values() if v['created'])} VMs.\n",
       "| VM | run | zone | provisioning | created | deleted | cost USD | basis |", "|---|---|---|---|---|---|---|---|"] + rows
txt.append(f"\nLive r5e2 VMs at rebuild time (gcloud instances list, name~soup-gpu-r5e2-): {live if live else 'none'}")
txt.append(f"\nLeft-over r5e2 disks (gcloud disks list, name~soup-gpu-r5e2-): {disks if disks else 'none'}\n")
open(OUT, "w").write("\n".join(txt))
print(f"total ${tot:.2f}; live: {live if live else 'none'}; disks: {disks if disks else 'none'}")
