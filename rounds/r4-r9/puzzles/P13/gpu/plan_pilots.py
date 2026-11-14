"""Phase 1 -> budgets (PREREG A4) and the Phase 2A queues: 14 pilots (ECON and ECON-nocap x 7 fixes, seed 100, 0.5 L4-h
of clean throughput each, a resumable checkpoint kept for the 1.0 h extension), each followed by its dev eval on the same
VM (v9, m13, v33, m59 at T = 12n, J option none), plus the M-ref gate eval (m13, m59) on the last pilot VM.
Writes gpu/budgets.json and gpu/queues/<slot>.txt; ctl.sh launches the slots.  Usage: plan_pilots.py [--dry]"""
import json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jobs import G, FIXES, phase1, sec_per_step, trader_job, eval_job, write_queue

H = 3600.0
P = phase1()
S = P["solo"]
for k, v in S.items():
    assert "sec_per_1k" in v, (k, v)
dtr = sec_per_step("dtr", P=P)
B_ref = 40000 * dtr
B = min(B_ref, max(3.0 * H, 25000 * dtr), 3.5 * H)
arms = ["econ", "nocap", "bptt_ds", "bptt_prog"]
main_steps = {f"{arm}:{fix}": int(math.floor(B / sec_per_step(arm, fix, P))) for arm in ["econ", "nocap"] for fix in FIXES}
main_steps.update({f"{arm}:F0": int(math.floor(B / sec_per_step(arm, "F0", P))) for arm in ["bptt_ds", "bptt_prog"]})
main_steps["dtr"] = int(math.floor(B / dtr))
pilot_steps = {f"{arm}:{fix}": int(math.floor(0.5 * H / sec_per_step(arm, fix, P))) for arm in ["econ", "nocap"] for fix in FIXES}

# packing: per-job slowdown when k econ jobs share one L4 (Phase-1 packed timing); VM-hours per job-hour = slowdown_k / k
solo_e = S["econ"]["sec_per_1k"]
pk = P.get("packed", {})
slow = {1: 1.0}
mem = {1: S["econ"].get("peak_mem_mb", 0)}
for k, key in [(2, "econ_x2"), (3, "econ_x3")]:
    g = pk.get(key, {})
    vals = [v["sec_per_1k"] for t, v in g.items() if isinstance(v, dict) and "sec_per_1k" in v]
    if len(vals) == k:
        slow[k] = max(vals) / solo_e
        mem[k] = sum(v.get("peak_mem_mb", 0) for t, v in g.items() if isinstance(v, dict) and "sec_per_1k" in v)
ok = [k for k in slow if mem[k] < 21000 and slow[k] * 0.5 <= 1.6]      # fits in 24 GB, pilot wall <= 1.6 h
kbest = min(ok, key=lambda k: (slow[k] / k, -k))
budget = {"at": P.get("end") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpu": P.get("gpu"), "dtr_sec_per_step": dtr, "B_ref_h": B_ref / H, "B_h": B / H,
          "bptt_seeds": 1 if B > 3.0 * H else 2, "sec_per_1k": {k: v["sec_per_1k"] for k, v in S.items()},
          "main_steps": main_steps, "pilot_steps": pilot_steps, "pack_slowdown": slow, "pack_mem_mb": mem,
          "pilot_pack": kbest, "pilots_per_vm": 2 if kbest == 1 else kbest}
print(json.dumps(budget, indent=1))
if "--dry" in sys.argv:
    sys.exit(0)
json.dump(budget, open(os.path.join(G, "budgets.json"), "w"), indent=1)

jobs = []
for fix in FIXES:
    for arm in ["econ", "nocap"]:
        n = f"pil_{arm}_{fix}"
        jobs.append((n, trader_job(n, arm, fix, pilot_steps[f"{arm}:{fix}"], 100, "--final_ckpt 1"),
                     eval_job(f"ev_{n}", n, "trader", f"out/{n}.pt", "v9,m13,v33,m59")))
# kbest = 1 (packing slower than solo): 2 pilots per VM one after the other (pilot, its eval, pilot, its eval), which keeps
# 7 VMs up under the 10-VM plan and saves 7 boots; kbest >= 2: kbest pilots at once, then their evals
per_vm = 2 if kbest == 1 else kbest
nvm = math.ceil(len(jobs) / per_vm)
slots = [f"trader-b3-p13-{i + 2}" for i in range(nvm)]
for i, s in enumerate(slots):
    mine = jobs[i::nvm]
    if kbest == 1:
        lines = [x for _, j, e in mine for x in (j, e)]
    else:
        lines = [j for _, j, _ in mine] + [e for _, _, e in mine]
    if i == nvm - 1:      # the VM with the fewest pilots
        lines.append(eval_job("ev_mref_gate", "", "mref", "mref/dt_net.pth", "m13,m59"))
    write_queue(s, lines, parallel=kbest)
    print(s, [n for n, _, _ in mine])
