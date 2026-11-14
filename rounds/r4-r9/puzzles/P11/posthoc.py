"""P11 post-hoc (NOT pre-registered): every seed's endpoints, and the comparisons restricted to seeds that trained.

Some width-32 recall seeds never learn the task (loss stuck near ln 2 or a dead net), the unreliability Bear et al.
2024 report for DT-R at width 32. The pre-registered analysis (analyze.py) keeps every seed. This script adds:
  - per seed, per arm: in-distribution test9 @1x ('inst' = the spec's training-gate quantity), final training loss,
    and E1-E3 on the arm's scored readout (as chosen in summary.json);
  - a seed 'trained' if its test9 @1x inst exact accuracy is >= 0.90 (the spec's training-gate threshold, per seed);
  - endpoint means over trained seeds, and paired M-budget minus other over seeds where both trained.

  python3 posthoc.py [RESULTS_DIR]      default results/vm -> results/posthoc.json (+ markdown on stdout)
"""
import json
import os
import sys

import numpy as np

from analyze import CELLS, CORE, SECONDARY, acc_at, endpoints, load_stage, paired
from vendored import ci

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "vm")
PARENT = os.path.dirname(RES.rstrip("/"))
S = json.load(open(os.path.join(PARENT, "summary.json")))
P = json.load(open(os.path.join(RES, "params.json")))
GATE = 0.90

rows, EP = {}, {}
for arm in CORE + SECONDARY:
    stage = "core" if arm in CORE else "secondary"
    evs = load_stage(RES, stage, arm, P["hp"][arm])
    if not evs:
        continue
    ro = S["readouts"][arm]["best"]
    ro = ro[0] if isinstance(ro, list) else ro
    rows[arm], EP[arm] = {}, {}
    for s, e in sorted(evs.items()):
        ind = acc_at(e["sets"]["test9"], "inst", 1)
        log = e["train"]["log"]
        ep = endpoints(e, ro)
        rows[arm][s] = {"test9_1x_inst": ind, "trained": ind >= GATE, "readout": ro,
                        "final_loss": log[-1]["loss"] if log else None,
                        "final_train_exact_T": log[-1].get("train_exact_T") if log else None, **ep}
        EP[arm][s] = ep

out = {"rule": f"trained = test9 exact accuracy at 1x ('inst') >= {GATE}; post hoc, not pre-registered",
       "per_seed": rows, "trained_means": {}, "paired_trained": {}}
for arm, r in rows.items():
    tr = [s for s in r if r[s]["trained"]]
    out["trained_means"][arm] = {"n_trained": len(tr), "n": len(r), "seeds": tr,
                                 **{k: ci([r[s][k] for s in tr]) for k in list(CELLS) + ["mean"]}}
if "M-budget" in rows:
    for o in ("M-TV", "M-B", "R-prog", "DT-L"):
        if o not in rows:
            continue
        both = [s for s in rows["M-budget"] if rows["M-budget"][s]["trained"] and rows.get(o, {}).get(s, {}).get("trained")]
        a = {s: EP["M-budget"][s] for s in both}
        b = {s: EP[o][s] for s in both}
        out["paired_trained"][f"M-budget - {o}"] = {k: paired(a, b, k) for k in list(CELLS) + ["mean"]}
json.dump(out, open(os.path.join(PARENT, "posthoc.json"), "w"), indent=1)


def pc(c):
    if not c or c.get("mean") is None:
        return "n/a"
    return f"{100 * c['mean']:.1f}" + (f" ± {100 * c['ci95']:.1f}" if c.get("ci95") is not None else "")


print("### Every seed (exact-maze accuracy, %; scored readout)\n")
print("| arm | seed | test9 @1x inst | final train loss | E1 17@1000x | E2 29@100x | E3 59@100x | mean | trained |")
print("|---|---|---|---|---|---|---|---|---|")
for arm, r in rows.items():
    for s, x in r.items():
        fl = "n/a" if x["final_loss"] is None else f"{x['final_loss']:.3g}"
        print(f"| {arm} | {s} | {100 * x['test9_1x_inst']:.1f} | {fl} | {100 * x['E1']:.1f} | {100 * x['E2']:.1f} | "
              f"{100 * x['E3']:.1f} | {100 * x['mean']:.1f} | {'yes' if x['trained'] else 'NO'} |")
print("\n### Trained seeds only (post hoc; mean ± 95% t-interval)\n")
print("| arm | trained / seeds | E1 | E2 | E3 | mean |")
print("|---|---|---|---|---|---|")
for arm, m in out["trained_means"].items():
    print(f"| {arm} | {m['n_trained']}/{m['n']} | {pc(m['E1'])} | {pc(m['E2'])} | {pc(m['E3'])} | {pc(m['mean'])} |")
print("\n### Paired over seeds where both arms trained (M-budget minus other, points)\n")
for k, c in out["paired_trained"].items():
    m = c["mean"]
    if m["mean"] is None:
        print(f"- {k}: n/a (seeds {m['seeds']})")
        continue
    ci_s = f"[{100 * m['lo']:+.1f}, {100 * m['hi']:+.1f}]" if m["lo"] is not None else "[n/a]"
    print(f"- {k}: mean {100 * m['mean']:+.1f} {ci_s}, {m['n_pos']}+/{m['n_neg']}- of {m['n']} (seeds {m['seeds']}); "
          + ", ".join(f"{e} {100 * c[e]['mean']:+.1f}" for e in CELLS if c[e]["mean"] is not None))
