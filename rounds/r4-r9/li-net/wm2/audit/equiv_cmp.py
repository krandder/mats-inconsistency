"""Compare the harness-equivalence pairs written by audit/equiv.sh (and the loop's own scratch pair)."""
import json
import os

PAIRS = [("/tmp/wm2/scratch/sn-k0.json", "/tmp/wm2/scratch/fc-rk0.json", "loop's own check (13:08Z code)"),
         ("/tmp/wm2-audit/eq/sn-k0.json", "/tmp/wm2-audit/eq/fc-rk0.json", "audit rerun, k0, F2 code"),
         ("/tmp/wm2-audit/eq/sn-rbtd8.json", "/tmp/wm2-audit/eq/fc-rrbtd8.json", "audit rerun, rbtd8, F2 code")]
for a, b, what in PAIRS:
    if not (os.path.exists(a) and os.path.exists(b)):
        print(f"{what}: missing {a if not os.path.exists(a) else b}")
        continue
    A, B = json.load(open(a)), json.load(open(b))
    hd = [(x["step"], k) for x, y in zip(A["history"], B["history"]) for k in x if k != "sec" and k in y and x[k] != y[k]]
    vd = [k for k in A["val"] if k in B["val"] and A["val"][k] != B["val"][k]]
    cf = [k for k in A["cfg"] if k in B["cfg"] and A["cfg"][k] != B["cfg"][k]]
    print(f"{what}: steps {A['steps_run']}/{B['steps_run']}, history entries {len(A['history'])}/{len(B['history'])}, "
          f"history fields that differ: {hd or 'none'}, val fields that differ: {vd or 'none'}, cfg keys that differ: {cf or 'none'}, "
          f"flops equal: {A['flops_total'] == B['flops_total']}; val KL {A['val']['kl']:.9f} vs {B['val']['kl']:.9f}, "
          f"log-loss {A['val']['logloss']:.9f} vs {B['val']['logloss']:.9f}")
