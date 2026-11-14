#!/usr/bin/env python3
"""Markdown tables for README.md from the result JSONs in this directory (numbers copied from JSON, not typed).

    python3 results/explore/l4-query-voi-20261020/make_tables.py [DIR] [--brief] > tables.md
--brief prints only the regret table and the selection diagnostics.
Also re-validates every JSON with aggbench.results.validate (jsonschema when installed).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from aggbench.results import validate  # noqa: E402

ROWS = [("random+exact", "random order, exact aggregation"),
        ("fixed+exact", "index order, exact aggregation"),
        ("static_voi+exact", "stand-alone VoI ranking (ignores overlap), exact"),
        ("greedy_voi+exact", "exact-Bayes greedy VoI, exact"),
        ("best_subset+exact", "best b-subset in expectation (ceiling), exact"),
        ("hindsight_best+exact", "hindsight-best b-subset (unachievable), exact"),
        ("learned_query+exact:s0", "learned query selection, exact aggregation"),
        ("learned_query+learned:s0", "**learned query + learned aggregation (main arm)**"),
        ("random+learned:s0", "random selection + learned aggregation"),
        ("learned_query+learned:s0:control:random_network", "control: random network (main arm pipeline)"),
        ("learned_query+learned:s0:control:untrained_init", "control: untrained init (main arm pipeline)")]


def cell(m, metric="regret"):
    r = m[metric]
    return f"{r['mean']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]"


def main():
    args = [a for a in sys.argv[1:] if a != "--brief"]
    brief = "--brief" in sys.argv
    d = Path(args[0]) if args else HERE
    for name in ("E1", "E2", "E3-sybil"):
        p = d / f"result-{name}.json"
        if not p.exists():
            continue
        res = json.loads(p.read_text())
        validate(res)
        M, ex = res["methods"], res["extra"]
        B, N = ex["budgets"], ex["n_consultable"]
        seeds = len(ex["learn"])
        print(f"\n### {name} (N = {N} consultable agents; {res['splits']['eval']['n']} dev episodes; "
              f"`{p.name}`, validated)\n")
        print("Mean regret to the global Bayes posterior (nats/episode, 95% bootstrap CI). Learned rows: seed 0.\n")
        print("| arm | " + " | ".join(f"b={b}" for b in B) + " |")
        print("|---|" + "---|" * len(B))
        for key, label in ROWS:
            print(f"| {label} | " + " | ".join(cell(M[f"b{b}/{key}"]) for b in B) + " |")
        print("| learned query + learned, all seeds (means) | " + " | ".join(
            " / ".join(f"{M[f'b{b}/learned_query+learned:s{s}']['regret']['mean']:.3f}" for s in range(seeds))
            for b in B) + " |")
        print("| learned query + exact, all seeds (means) | " + " | ".join(
            " / ".join(f"{M[f'b{b}/learned_query+exact:s{s}']['regret']['mean']:.3f}" for s in range(seeds))
            for b in B) + " |")
        def closed(arm, b):
            r, c = M[f"b{b}/random+exact"]["regret"]["mean"], M[f"b{b}/best_subset+exact"]["regret"]["mean"]
            return (r - M[f"b{b}/{arm}"]["regret"]["mean"]) / (r - c)
        print("| share of the random → best-subset gap closed by learned selection (exact aggregation), "
              "per seed | " + " | ".join(" / ".join(f"{closed(f'learned_query+exact:s{s}', b):.2f}"
                                                    for s in range(seeds)) for b in B) + " |")
        print("| same share for static VoI / greedy VoI | " + " | ".join(
            f"{closed('static_voi+exact', b):.2f} / {closed('greedy_voi+exact', b):.2f}" for b in B) + " |")
        print(f"\nFull information (b = N = {N}): oracle_clearing {cell(M['oracle_clearing'])}, "
              f"extremized_log_pool {cell(M['extremized_log_pool'])}, learned aggregator on all agents (s0) "
              f"{cell(M[f'b{N}/all+learned:s0'])}, its random-network control "
              f"{cell(M[f'b{N}/all+learned:s0:control:random_network'])}.\n")
        if brief:
            print_selection(ex, B)
            continue
        print("Regret to the best-b-subset ceiling: paired difference arm - best_subset+exact (95% CI).\n")
        print("| arm | " + " | ".join(f"b={b}" for b in B) + " |")
        print("|---|" + "---|" * len(B))
        for key in ("random+exact", "static_voi+exact", "greedy_voi+exact", "learned_query+exact:s0",
                    "learned_query+learned:s0", "random+learned:s0",
                    "learned_query+learned:s0:control:random_network"):
            cells = []
            for b in B:
                r = ex["paired"][f"b{b}/{key} - b{b}/best_subset+exact"]
                cells.append(f"{r['mean']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")
            print(f"| {key} | " + " | ".join(cells) + " |")
        print("\nValue of learned selection inside the learned aggregator: learned_query+learned - random+learned "
              "(paired, 95% CI), per seed.\n")
        print("| seed | " + " | ".join(f"b={b}" for b in B) + " |")
        print("|---|" + "---|" * len(B))
        for s in range(seeds):
            for tag in ("", ":control:random_network"):
                cells = []
                for b in B:
                    r = ex["paired"][f"b{b}/learned_query+learned:s{s}{tag} - b{b}/random+learned:s{s}{tag}"]
                    cells.append(f"{r['mean']:+.3f} [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}]")
                print(f"| s{s}{tag} | " + " | ".join(cells) + " |")
        print_selection(ex, B)


def print_selection(ex, B):
    sel = ex["selection"]
    print("\nSelection diagnostics: share of episodes whose chosen set equals the best b-subset; "
          "share holding two agents with identical keys (a Sybil duplicate).\n")
    print("| policy | " + " | ".join(f"b={b} same / dup" for b in B) + " |")
    print("|---|" + "---|" * len(B))
    for pol in ("random", "static_voi", "greedy_voi", "best_subset", "learned_query:s0",
                "learned_query:s0:control:random_network", "learned_query:s0:control:untrained_init"):
        print(f"| {pol} | " + " | ".join(
            f"{sel[f'b{b}'][pol]['same_set_as_best_subset']:.2f} / {sel[f'b{b}'][pol]['holds_sybil_duplicate']:.2f}"
            for b in B) + " |")


if __name__ == "__main__":
    main()
