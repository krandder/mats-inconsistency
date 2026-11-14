"""Phase 2B: the ECON-active ladder (PREREG), only when the selected fix's ECON pilot has frac_capped_pix < 0.10 over its
last 25%. Two candidates with the selected fix, 0.5 L4-h each (ECON timing of that fix), seed 100, one per new slot:
  lad_A1: --W0 0.5;  lad_A2: --bounty lindahl.
Each is followed by its dev eval (v9, m13, v33, m59 at 12n, J option none).  Usage: plan_ladder.py [--dry]"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jobs import D, sec_per_step, trader_job, eval_job, write_queue
from plan_main import next_slots

tag = "1p0" if os.path.exists(os.path.join(D, "results", "pilot_selection_1p0.json")) else "0p5"
sel = json.load(open(os.path.join(D, "results", f"pilot_selection_{tag}.json")))
fix = sel["selected"]
if sel["econ_fix_active"]:
    sys.exit(f"ECON-fix ({fix}) is itself active (frac_capped {sel['econ_sel_frac_capped_last25']:.3f} >= 0.10): no ladder")
steps = int(math.floor(0.5 * 3600 / sec_per_step("econ", fix)))        # 0.5 L4-h, as pre-registered
C = [("lad_A1", "--W0 0.5"), ("lad_A2", "--bounty lindahl")]
slots = next_slots(1)
lines = [trader_job(n, "econ", fix, steps, 100, extra + " --final_ckpt 1") for n, extra in C]
lines += [eval_job(f"ev_{n}", n, "trader", f"out/{n}.pt", "v9,m13,v33,m59") for n, _ in C]
print(slots[0], fix, steps, *lines, sep="\n")
if "--dry" not in sys.argv:
    write_queue(slots[0], lines, parallel=2)
    json.dump({"fix": fix, "steps": steps, "slot": slots[0], "cands": dict(C)}, open(os.path.join(D, "results", "ladder_plan.json"), "w"), indent=1)
