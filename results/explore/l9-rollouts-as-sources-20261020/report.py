"""Print the README tables from summary.json (recipe v1) or summary__v2.json (written by run.py results).

    python3 results/explore/l9-rollouts-as-sources-20261020/report.py [v1|v2]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def ci(s, digits=3):
    return f"{s['mean']:.{digits}f} [{s['ci_low']:.{digits}f}, {s['ci_high']:.{digits}f}]"


def main():
    recipe = sys.argv[1] if len(sys.argv) > 1 else "v1"
    rows = json.loads((HERE / ("summary.json" if recipe == "v1" else f"summary__{recipe}.json")).read_text())["rows"]
    print(f"Recipe {recipe}.\n")
    by = {(r["tag"], r["visibility"]): r for r in rows}

    print("### Table 1. N = 6 rollouts: mean regret to Bayes on the whole context (nats), 95% CI over 5000 dev episodes\n")
    cols = [("imperfect__N6", "full", "imperfect model"), ("exact__N6", "full", "exact model (control)")]
    print("| method | " + " | ".join(c[2] for c in cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    shared = [("ceiling:infinite_rollouts", "infinite-rollouts floor (shared misreading only)"),
              ("ceiling:trace", "trace ceiling (every rollout's trace + true structure)"),
              ("oracle_clearing", "reports ceiling (N reports + true structure)"),
              ("extremized_log_pool", "extremized_log_pool"), ("log_pool", "log_pool"),
              ("linear_pool", "linear_pool"), ("sum_logodds", "sum_logodds")]
    for m, label in shared:
        print(f"| {label} | " + " | ".join(ci(by[(t, v)]["regret"][m]) for t, v, _ in cols) + " |")
    for vis in ("none", "noisy", "full"):
        cells = []
        for t, _, _ in cols:
            r = by.get((t, vis))
            cells.append(ci(r["regret"][f"attn:{vis}:seed_mean"]) if r else "not run")
        print(f"| attention, keys {vis} (3 seeds averaged) | " + " | ".join(cells) + " |")
    for vis in ("none", "full"):
        for ctl in ("untrained_init", "random_network"):
            cells = []
            for t, _, _ in cols:
                r = by.get((t, vis))
                if not r:
                    cells.append("not run")
                    continue
                vals = [r["regret"][f"attn:{vis}:s{s}:control:{ctl}"]["mean"] for s in (0, 1, 2)]
                cells.append(f"{min(vals):.3f} to {max(vals):.3f}")
            print(f"| control {ctl}, keys {vis} (range over 3 seeds) | " + " | ".join(cells) + " |")

    print("\n### Table 2. Paired regret differences, N = 6 (negative = first arm better)\n")
    print("| condition | keys | seed | attn - extremized_log_pool | attn - reports ceiling |")
    print("|---|---|---|---|---|")
    for t, label in (("imperfect__N6", "imperfect"), ("exact__N6", "exact")):
        for vis in ("none", "noisy", "full"):
            r = by.get((t, vis))
            if not r:
                continue
            for s in (0, 1, 2):
                a = f"attn:{vis}:s{s}:trained"
                print(f"| {label} | {vis} | {s} | {ci(r['paired'][f'{a} - extremized_log_pool'])} | "
                      f"{ci(r['paired'][f'{a} - oracle_clearing'])} |")
        r = by[(t, "full")]
        print(f"| {label} | - | - | reports ceiling - extremized: {ci(r['paired']['oracle_clearing - extremized_log_pool'])} | "
              f"trace - reports ceiling: {ci(r['paired']['ceiling:trace - oracle_clearing'])} |")

    print("\n### Table 3. More rollouts (keys full; attention seed 0): mean regret to Bayes (nats), 95% CI\n")
    print("| model | N | reports ceiling | trace ceiling | infinite-rollouts floor | extremized_log_pool | attention | context covered |")
    print("|---|---|---|---|---|---|---|---|")
    for model in ("imperfect", "exact"):
        for n in (1, 2, 4, 6, 12, 24):
            r = by[(f"{model}__N{n}", "full")]
            g = r["regret"]
            print(f"| {model} | {n} | {ci(g['oracle_clearing'])} | {ci(g['ceiling:trace'])} | "
                  f"{ci(g['ceiling:infinite_rollouts'])} | {ci(g['extremized_log_pool'])} | {ci(g['attn:full:s0:trained'])} | {r.get('covered', 0):.1%} |")


if __name__ == "__main__":
    main()
