"""Markdown tables for RESULTS-R2.md from results-r2/results.json and eval_checkpoints.csv (run analyze.py first).
Every cell names its key in results.json; CIs are 95% t (df = 2, t = 4.303) over seeds 101-103."""
import csv
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json")))
ARMS = [("fused", "Fused"), ("tokmask", "Token-Masked"), ("lir", "LIR-Masked")]


def f(x, d=3):
    return "n/a" if x is None else f"{x:.{d}f}"


def cell(a, k, d=3, pct=False):
    c = R["arms"][a][k]
    if not c.get("n"):
        return "n/a"
    m = 100 * c["mean"] if pct else c["mean"]
    if c["n"] < 2:
        return f"{m:.{d}f} (n=1)"
    lo, hi = (100 * c["lo"], 100 * c["hi"]) if pct else (c["lo"], c["hi"])
    return f"{m:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def vals(a, k, d=3):
    c = R["arms"][a][k]
    return " / ".join(f(v, d) for v in c.get("vals", [])) if c.get("n") else "n/a"


out = []
out.append("### Table 1. Headline metrics at post_rl, 3-seed mean [95% t-CI] (results.json: arms.<arm>.<key>)\n")
rows = [("gate_corr_bayes", "pre-RL gate: Pearson(E[q], exact Bayes)", 3, False),
        ("env_reward_pre", "eval env reward, pre-RL", 3, False),
        ("env_reward_post", "eval env reward, post-RL", 3, False),
        ("reward_rise", "reward rise (post/pre - 1), %", 1, True),
        ("train_rise", "training-curve rise (last 50 vs first 50 steps), %", 1, True),
        ("syc_false_note_p5_pre", "false-note sycophancy, pre-RL", 3, False),
        ("syc_false_note_p5_post", "false-note sycophancy, post-RL", 3, False),
        ("belief_drift_max_abs_post", "belief drift, max abs vs pre-RL", 4, False),
        ("belief_drift_mean_abs_post", "belief drift, mean abs vs pre-RL", 4, False),
        ("delta_desire_pre", "Delta_desire, pre-RL", 4, False),
        ("delta_desire_post", "Delta_desire, post-RL", 4, False),
        ("delta_desire_change", "Delta_desire change (post - pre)", 4, False),
        ("faith_flip_rate_post", "faith_swap flip rate (exact-Bayes judged), post-RL", 3, False),
        ("swap_belief_change_max_post", "real_belief change under the swap (max abs)", 4, False)]
out.append("| metric | " + " | ".join(n for _, n in ARMS) + " |")
out.append("|---|" + "---|" * len(ARMS))
for k, name, d, pct in rows:
    out.append(f"| {name} (`{k}`) | " + " | ".join(cell(a, k, d, pct) for a, _ in ARMS) + " |")
out.append("\n### Table 2. Per-seed values (101 / 102 / 103)\n")
out.append("| metric | " + " | ".join(n for _, n in ARMS) + " |")
out.append("|---|" + "---|" * len(ARMS))
for k, name, d, pct in rows:
    out.append(f"| `{k}` | " + " | ".join(vals(a, k, d if not pct else 3) for a, _ in ARMS) + " |")
out.append("\n### Table 3. Paired differences by seed, mean [95% t-CI] (results.json: paired)\n")
out.append("| metric | " + " | ".join(R["paired"].keys()) + " |")
out.append("|---|" + "---|" * len(R["paired"]))
for k in ("syc_false_note_p5_post", "delta_desire_post", "delta_desire_change", "env_reward_post", "reward_rise",
          "faith_flip_rate_post", "belief_drift_mean_abs_post"):
    cs = []
    for p in R["paired"].values():
        c = p[k]
        cs.append("n/a" if not c.get("n") else (f"{c['mean']:.4f} [{c['lo']:.4f}, {c['hi']:.4f}]" if c["n"] > 1 else f"{c['mean']:.4f} (n=1)"))
    out.append(f"| `{k}` | " + " | ".join(cs) + " |")
# checkpoint table (belief drift and sycophancy over training)
by = defaultdict(list)
for row in csv.DictReader(open(os.path.join(HERE, "eval_checkpoints.csv"))):
    by[(row["arm"], row["tag"], int(row["step"]))].append(row)
out.append("\n### Table 4. Belief drift, sycophancy and Delta_desire at each checkpoint, mean over seeds (eval_checkpoints.csv)\n")
out.append("| arm | checkpoint | n seeds | eval env reward | belief corr with Bayes | false-note sycophancy | belief drift max | belief drift mean | Delta_desire | flip rate |")
out.append("|---|---|---|---|---|---|---|---|---|---|")
for a, n in ARMS:
    for (aa, tag, step), rs in sorted(by.items(), key=lambda t: (t[0][0], t[0][2])):
        if aa != a:
            continue
        m = lambda k: sum(float(r[k]) for r in rs if r[k] not in ("", "None")) / max(1, len([r for r in rs if r[k] not in ("", "None")]))
        out.append(f"| {n} | {tag} | {len(rs)} | {m('env_reward'):.3f} | {m('gate_corr_bayes'):.3f} | {m('syc_false_note_p5'):.3f} | "
                   f"{m('belief_drift_max_abs'):.4f} | {m('belief_drift_mean_abs'):.4f} | {m('delta_desire'):.4f} | {m('faith_flip_rate'):.3f} |")
open(os.path.join(HERE, "tables.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
