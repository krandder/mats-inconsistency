"""Build summary.json and the README tables from json/*.json (numbers are copied by this script, not typed).

    python3 results/explore/l3-looped-clearing-20261020/report.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from aggbench.results import validate  # noqa: E402

ARMS = ["indep", "readout_softmax", "stacked_T1", "stacked_T2", "stacked_T4", "stacked_T8", "looped_T2",
        "looped_T4", "looped_T8", "jacobi_T1", "jacobi_T2", "jacobi_T4", "jacobi_T8", "exact"]
BASE = ["E1__full", "E1__noisy", "E2__full", "E2__noisy", "E3-sybil__full", "E3-sybil__noisy",
        "E3-collude__full", "E3-collude__noisy"]
SWEEP_ENVS = ["E2", "E3-sybil"]
SWEEP = [("n_agents=3", "N=3"), ("", "N=5 (default)"), ("n_agents=12", "N=12"),
         ("pieces_per_agent=1-3", "1-3 pieces (low overlap)"), ("", "2-6 pieces (default)"),
         ("pieces_per_agent=6-10", "6-10 pieces (high overlap)")]


def fmt(s, nd=3):
    return f"{s['mean']:.{nd}f} [{s['ci_low']:.{nd}f}, {s['ci_high']:.{nd}f}]"


def load():
    out = {}
    for p in sorted((HERE / "json").glob("*.json")):
        r = json.loads(p.read_text())
        validate(r)
        out[p.stem] = r
    return out


def row_values(r, arm):
    m = r["methods"]
    pd = r["extra"]["paired_regret_difference"]
    key = f"{arm}:trained" if f"{arm}:trained" in m else arm
    return {
        "regret_to_bayes": m[key]["regret"],
        "regret_to_oracle_clearing": pd[key]["oracle_clearing"],
        "minus_extremized_log_pool": pd[key]["extremized_log_pool"],
    }


def main():
    res = load()
    summary = {"track": "EXPLORATORY", "jobs": {}}
    for tag, r in res.items():
        rows = {}
        for arm in ARMS:
            v = row_values(r, arm)
            meta = r["extra"]["arms"][arm]
            v.update({"T": meta["T"], "width": meta["width"], "params": meta["params"],
                      "control_random_network_regret": r["methods"][f"{arm}:control:random_network"]["regret"],
                      "control_untrained_init_regret": r["methods"][f"{arm}:control:untrained_init"]["regret"]})
            rows[arm] = v
        for b in ("oracle_clearing", "extremized_log_pool", "sum_logodds", "log_pool"):
            rows[b] = {"regret_to_bayes": r["methods"][b]["regret"]}
            if b != "oracle_clearing" and b != "extremized_log_pool":
                rows[b]["regret_to_oracle_clearing"] = r["extra"]["paired_regret_difference"][b]["oracle_clearing"]
        summary["jobs"][tag] = {"env": r["env"]["name"], "visibility": r["visibility"],
                                "overrides": r["extra"]["overrides"], "rows": rows,
                                "union_identified_dev_frac": r["extra"]["union_identified_dev_frac"],
                                "harness_commit": r["harness"]["git_commit"]}
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")

    lines = []
    base = [b for b in BASE if b in res]

    def table(title, metric, tags, header):
        lines.append(f"\n**{title}**\n")
        lines.append("| arm | params | " + " | ".join(header) + " |")
        lines.append("|---|---|" + "---|" * len(header))
        for arm in ARMS + ["extremized_log_pool", "sum_logodds", "oracle_clearing"]:
            cells = []
            for t in tags:
                row = summary["jobs"][t]["rows"].get(arm, {})
                cells.append(fmt(row[metric]) if metric in row else "-")
            par = summary["jobs"][tags[0]]["rows"][arm].get("params", "-") if arm in ARMS else "-"
            lines.append(f"| {arm} | {par} | " + " | ".join(cells) + " |")

    table("Regret to bayes (nats/episode, 95% bootstrap CI, 2000 dev episodes), default configs",
          "regret_to_bayes", base, [b.replace("__", " ") for b in base])
    table("Regret to oracle_clearing (paired difference arm - oracle_clearing, 95% CI), default configs",
          "regret_to_oracle_clearing", base, [b.replace("__", " ") for b in base])
    for env in SWEEP_ENVS:
        tags = [f"{env}__{o}__full" if o else f"{env}__full" for o, _ in SWEEP]
        if not all(t in res for t in tags):
            continue
        table(f"{env}, full keys: regret to oracle_clearing across agents and overlap", "regret_to_oracle_clearing",
              tags, [h for _, h in SWEEP])

    # depth question: paired difference against one self-attention layer (same dev episodes)
    ptags = [t for t in ["E3-sybil__full", "E3-sybil__noisy", "E3-sybil__n_agents=3__full",
                         "E3-sybil__n_agents=12__full", "E3-sybil__pieces_per_agent=1-3__full",
                         "E3-sybil__pieces_per_agent=6-10__full", "E2__full", "E3-collude__full"] if t in res]
    lines.append("\n**Paired regret difference arm - stacked_T1 (one self-attention layer), 95% CI; negative = "
                 "better than one layer**\n")
    lines.append("| arm | " + " | ".join(t.replace("__", " ") for t in ptags) + " |")
    lines.append("|---|" + "---|" * len(ptags))
    for arm in [a for a in ARMS if a != "stacked_T1"]:
        cells = [fmt(res[t]["extra"]["paired_regret_difference"][f"{arm}:trained"]["stacked_T1:trained"])
                 for t in ptags]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")

    # controls: does any random-network / untrained control come within the trained arm's CI?
    lines.append("\n**Controls** (random-network and untrained-init, same predict pipeline): number of "
                 "(config, arm) pairs where the control's mean regret to bayes is <= the trained arm's CI upper "
                 "bound; and the best control regret per config.\n")
    n_pairs, n_viol, viol = 0, 0, []
    best = []
    for t, j in summary["jobs"].items():
        bmin = None
        for arm in ARMS:
            row = j["rows"][arm]
            for c in ("control_random_network_regret", "control_untrained_init_regret"):
                n_pairs += 1
                if row[c]["mean"] <= row["regret_to_bayes"]["ci_high"]:
                    n_viol += 1
                    viol.append(f"{t}:{arm}:{c.split('_')[1]}")
                if bmin is None or row[c]["mean"] < bmin[0]:
                    bmin = (row[c]["mean"], arm, c)
        best.append(f"{t} {bmin[0]:.3f} ({bmin[1]}, {bmin[2].replace('control_', '').replace('_regret', '')})")
    lines.append(f"- control <= trained CI upper bound: {n_viol} of {n_pairs} pairs"
                 + (f" ({', '.join(viol)})" if viol else ""))
    lines.append("- best (lowest) control regret per config: " + "; ".join(best))
    text = "\n".join(lines) + "\n"
    (HERE / "tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
