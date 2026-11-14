"""Phase 2A pilot extension to 1.0 L4-h (PREREG): resume each target pilot from its final checkpoint (same --out, constant
learning rate, so resuming equals one uninterrupted 1.0 h run) and re-run its dev eval as ev_<pilot>_x.
  plan_extend.py all [--dry]   all 14 pilots (fewer than 3 candidates per arm reached v9 >= 0.9 at 0.5 h)
  plan_extend.py FIX [--dry]   the selected fix, both arms (learnability gate not met at 0.5 h)
Lines are appended to the pilot's own slot queue: its synced out/ (checkpoint and done markers) is already there, or is
pushed back to the -rN VM that ctl.sh launches for the pending jobs. Each extension waits for the pilot's 0.5 h eval."""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jobs import G, D, FIXES, sec_per_step, trader_job, eval_job, write_queue

target = sys.argv[1]
fixes = list(FIXES) if target == "all" else [target]
assert all(f in FIXES for f in fixes), target
dry = "--dry" in sys.argv
qdir = os.path.join(G, "queues")
queues = {}
for fn in sorted(os.listdir(qdir)):
    if fn.startswith("trader-b3-p13-") and fn.endswith(".txt"):
        queues[fn[:-4]] = [l for l in open(os.path.join(qdir, fn)).read().splitlines() if l.strip() and l.strip() != "END"]
add, done = {}, []
for fix in fixes:
    for arm in ["econ", "nocap"]:
        n = f"pil_{arm}_{fix}"
        slot = [s for s, ls in queues.items() if any(l.split()[0] == n for l in ls)]
        assert len(slot) == 1, (n, slot)
        s = slot[0]
        if any(l.split()[0].split("@")[0] == n + "_x" for l in queues[s]):
            print(n, "already extended on", s); continue
        steps = int(math.floor(3600.0 / sec_per_step(arm, fix)))           # 1.0 L4-h in total
        j = trader_job(n, arm, fix, steps, 100, "--final_ckpt 1")
        j = f"{n}_x@ev_{n}" + j[len(n):]                                  # job name pil_X_x, same --out out/pil_X
        e = eval_job(f"ev_{n}_x", f"{n}_x", "trader", f"out/{n}.pt", "v9,m13,v33,m59")
        add.setdefault(s, []).extend([j, e]); done.append((n, s, steps))
for s, lines in add.items():
    print(s, *lines, sep="\n  ")
    if not dry:
        write_queue(s, queues[s] + lines, parallel=int(open(os.path.join(G, "slots", s)).read().split()[0]))
if not dry and done:
    p = os.path.join(D, "results", "pilot_extension.json")
    old = json.load(open(p)) if os.path.exists(p) else []
    json.dump(old + [{"target": target, "pilots": done}], open(p, "w"), indent=1)
