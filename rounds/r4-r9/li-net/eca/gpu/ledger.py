"""E-CA GPU ledger: rebuilds eca/GPU-LOG.md from the shared logical-soup/GPU-LOG.md (lines for vm=soup-gpu-eca*) plus a
live gcloud listing. Cost per VM = the DELETE / COST-AUDIT line's cost_usd (ceiling rate $0.87/h x age); VMs without one
are counted at their reservation (worst case). Run: python3 gpu/ledger.py"""
import re
import subprocess
import time

LOG = "/home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md"
OUT = "/home/agent/workspaces/eurisko-study/li-net/eca/GPU-LOG.md"
vms = {}
for l in open(LOG):
    m = re.search(r"vm=(soup-gpu-eca\S*)", l)
    if not m:
        continue
    n = m.group(1)
    v = vms.setdefault(n, {"zone": "", "model": "", "reserve": 0.0, "cost": None, "age_h": None, "created": "", "deleted": "",
                           "failed": False, "run": ""})
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
    if " DELETE vm=" in l or " COST-AUDIT vm=" in l:
        c = re.search(r"cost_usd=([\d.]+)", l)
        a = re.search(r"age_h=([\d.]+)", l)
        if c:
            v["cost"] = float(c.group(1))
        if a:
            v["age_h"] = float(a.group(1))
        v["deleted"] = ts
live = subprocess.run(["gcloud", "compute", "instances", "list", "--project", "GCP_PROJECT", "--filter", "name~soup-gpu-eca",
                       "--format", "value(name,status,zone.basename())"], capture_output=True, text=True).stdout.strip()
tot = 0.0
rows = []
for n, v in sorted(vms.items(), key=lambda kv: kv[1]["created"] or "z"):
    if v["failed"] and not v["created"]:
        c = 0.0
        basis = "create failed (stockout), no instance"
    elif v["cost"] is not None:
        c = v["cost"]
        basis = f"DELETE line, age {v['age_h']} h x $0.87/h ceiling"
    else:
        c = v["reserve"]
        basis = "still open: counted at reservation (worst case)"
    tot += c
    rows.append(f"| {n} | {v['run']} | {v['zone']} | {v['model']} | {v['created']} | {v['deleted']} | {c:.2f} | {basis} |")
txt = [f"# E-CA GPU ledger (budget group overnight-ca, cap $110)\n",
       f"Rebuilt {time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())} by gpu/ledger.py from logical-soup/GPU-LOG.md. Rates are the",
       "UNVERIFIED on-demand list-price ceiling ($0.87/h for g2-standard-8 + 1 L4; Cloud Billing API disabled), so true spend",
       "is at most this. Create attempts that failed on stockout created no instance and cost nothing.\n",
       f"**Total (ceiling): ${tot:.2f}** over {sum(1 for v in vms.values() if v['created'])} VMs.\n",
       "| VM | run | zone | provisioning | created | deleted | cost USD | basis |", "|---|---|---|---|---|---|---|---|"] + rows
txt.append(f"\nLive eca VMs at rebuild time (gcloud): {live if live else 'none'}\n")
txt.append("MATS cluster: not used (opus-mats 01:0xZ: 8/8 L40s allocated, next free slot 11-19); zero jobs submitted.\n")
open(OUT, "w").write("\n".join(txt))
print(f"total ${tot:.2f}; live: {live if live else 'none'}")
