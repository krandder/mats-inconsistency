"""Print the markdown tables used in RESULTS.md from results/summary.json and results/chart.json.

  python3 tables.py [RESULTS_PARENT]      default results/
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
S = json.load(open(os.path.join(R, "summary.json")))
C = json.load(open(os.path.join(R, "chart.json")))


def pc(c, digits=1):
    """{'mean','ci95'} in [0,1] -> 'm ± h' in percent."""
    if not c or c.get("mean") is None:
        return "n/a"
    h = c.get("ci95")
    return f"{100 * c['mean']:.{digits}f}" + (f" ± {100 * h:.{digits}f}" if h is not None else "")


def pd(c):
    if c.get("mean") is None:
        return "n/a"
    lo, hi = c.get("lo"), c.get("hi")
    ci = f"[{100 * lo:+.1f}, {100 * hi:+.1f}]" if lo is not None else "[n/a]"
    return f"{100 * c['mean']:+.1f} {ci}, {c['n_pos']}+/{c['n_neg']}- of {c['n']}"


print("### Endpoints (exact-maze accuracy, %, mean ± 95% t-interval over seeds)\n")
print("| arm | scored readout | E1 17@1000x | E2 29@100x | E3 59@100x | mean | size 9 @1x |")
print("|---|---|---|---|---|---|---|")
for a, e in S["endpoints"].items():
    ro = S["readouts"][a]["best"]
    print(f"| {a} | {ro} | {pc(e['E1'])} | {pc(e['E2'])} | {pc(e['E3'])} | {pc(e['mean'])} | "
          f"{pc(S['in_dist_test9_1x'].get(a))} |")
    for tag in ("inst", "lmsr", "tv"):
        if tag in e and (tag != "inst" or ro != "inst"):
            nm = S["readouts"][a].get(tag, tag)
            x = e[tag]
            print(f"| {a} | {nm} (eval-only) | {pc(x['E1'])} | {pc(x['E2'])} | {pc(x['E3'])} | {pc(x['mean'])} | |")

print("\n### Accuracy vs depth (%, scored readout; ratio = iterations / 30)\n")
for st in ("test9", "test17", "test29", "test59"):
    keys = [k for k in C["series"] if k.endswith("|" + st) and "|best=" in k]
    if not keys:
        continue
    rat = list(C["series"][keys[0]].keys())
    print(f"{st}:\n")
    print("| arm | " + " | ".join(f"{r}x" for r in rat) + " |")
    print("|---|" + "---|" * len(rat))
    for k in keys:
        a = k.split("|")[0]
        row = C["series"][k]
        print(f"| {a} ({k.split('|')[1].split('=')[1]}) | " + " | ".join(f"{100 * row[r][0]:.1f}" for r in rat) + " |")
    print()

print("### Paired by seed (M-budget minus other, points; t-interval; sign count)\n")
print("| comparison | E1 | E2 | E3 | mean |")
print("|---|---|---|---|---|")
for k, c in S["paired"].items():
    if isinstance(c, dict) and "mean" in c and isinstance(c["mean"], dict):
        print(f"| {k} | {pd(c['E1'])} | {pd(c['E2'])} | {pd(c['E3'])} | {pd(c['mean'])} |")
    else:
        print(f"| {k} | | | | {pd(c)} |")

print("\n### Readouts within a model (hierarchical bootstrap over seeds and mazes, mean over E1-E3, points)\n")
for k, c in S["hboot"].items():
    print(f"- {k}: {100 * c['mean']:+.1f} [{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}]")

print("\n### Mechanism\n")
print("| arm | late correction rate | persistence 10x->1000x test17 | path-indep. spread | rel. state dist |")
print("|---|---|---|---|---|")
for a in S["endpoints"]:
    st = S["stability"].get(a, {})
    pi = S["path_independence"].get(a, {})
    print(f"| {a} | {pc(S['late_correction_rate'].get(a))} | {pc(st.get('persistence_test17'))} | "
          f"{pc(pi.get('spread_exact'))} | {pc(pi.get('rel_state_dist'), 2)} |")
print()
for k, rows in S["budget_use"].items():
    for st, b in rows.items():
        print(f"- budget {k} {st}: exhausted on path {pc(b['exhausted'])}, exhausted while wrong "
              f"{pc(b['exhausted_while_wrong'])}, mean remaining {pc(b['mean_remaining'])}")
print("\n### Gates\n")
g = S["params"]["gates"]
print(f"- training: {json.dumps(g['training'])}")
print(f"- floor: {json.dumps(g['floor'])}")
print(f"- sensitivity: {json.dumps(S['sensitivity_gate'])}")
print(f"- headroom: {json.dumps(S['headroom'])}")
print(f"- endpoint floor (best arm < 20% = inconclusive): {json.dumps(S['floor'])}")
print(f"\n### Verdict\n\n{json.dumps(S['verdict'])}")
