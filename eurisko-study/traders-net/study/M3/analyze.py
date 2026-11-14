"""M3 reduced: aggregate wealth-net decay/settle results + windowed grad-norm currencies. CPU, read-only."""
import csv, json, os, glob
import numpy as np
from scipy.stats import spearmanr
WN = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../wealth-net/results"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
rows = {}
def add(cur, seed, v): rows.setdefault(cur, {})[seed] = v
for s in (1, 2, 3):
    for r in csv.DictReader(open(f"{WN}/decay_spearman_s{s}.csv")):
        add(r["currency"], s, float(r["spearman_vs_oracle"])); step = int(r["step"])
    for r in csv.DictReader(open(f"{WN}/settle_spearman_s{s}.csv")):
        if int(r["step"]) == step: add("settle_" + r["currency"], s, float(r["spearman_vs_oracle"]))
    orc = np.load(f"{WN}/decay_oracle_s{s}.npy").astype(float).ravel()
    snaps = {}
    for st in range(100, step + 1, 100):
        d = json.load(open(f"{WN}/train-s{s}/snapshots/step-{st:05d}.json"))
        snaps[st] = (np.array(d["gradnorm_cum"]["neuron"], float).ravel(), np.array(d["wealth"]["neuron"], float))
    last = max(snaps); g = snaps[last][0]
    assert g.shape == orc.shape, (g.shape, orc.shape)
    add("gradnorm_alltime", s, spearmanr(g, orc)[0])
    for w in (100, 500, 2000):
        add(f"gradnorm_win{w}", s, spearmanr(g - snaps[last - w][0], orc)[0])
    # disjoint-window income stability: rank corr of income in consecutive 1000-step windows
    ws = [snaps[st][1] for st in range(1000, last + 1, 1000)]
    inc = np.diff(np.stack(ws), axis=0)
    stab = [spearmanr(inc[i], inc[i + 1])[0] for i in range(len(inc) - 1)]
    add("income_disjoint1k_stability_last5", s, float(np.mean(stab[-5:])))
    add("income_disjoint1k_stability_first5", s, float(np.mean(stab[:5])))
    null = [spearmanr(rng.permutation(orc), orc)[0] for _ in range(1000)]
    add("null_shuffled_oracle_sd", s, float(np.std(null)))
with open(f"{OUT}/currency_summary.csv", "w") as f:
    w = csv.writer(f); w.writerow(["currency", "s1", "s2", "s3", "mean", "sd"])
    for c, d in rows.items():
        v = [d.get(k, np.nan) for k in (1, 2, 3)]
        w.writerow([c] + [f"{x:.4f}" for x in v] + [f"{np.mean(v):.4f}", f"{np.std(v, ddof=1):.4f}"])
# stop rule: best decayed/windowed wealth currency minus gradnorm control, per seed
wealthc = [c for c in rows if c.startswith(("hl", "win")) or c == "cum"]
res = {}
for ctrl in ("gradnorm", "gradnorm_win500"):
    diffs = {c: [rows[c][s] - rows[ctrl][s] for s in (1, 2, 3)] for c in wealthc}
    res[ctrl] = {"any_beats_by_0.05_in_3of3": any(all(x >= 0.05 for x in v) for v in diffs.values()),
                 "best": max(diffs, key=lambda c: min(diffs[c])), "diffs": diffs}
res["settle_loo_training_vs_gradnorm"] = [rows["settle_loo_train"][s] - rows["gradnorm"][s] for s in (1, 2, 3)] if "settle_loo_train" in rows else None
json.dump(res, open(f"{OUT}/stop_rule.json", "w"), indent=1)
# pruning at final: mean of 3 seeds
acc = {}
for s in (1, 2, 3):
    for r in csv.DictReader(open(f"{WN}/decay_ablation_s{s}.csv")):
        if r["final"] == "1": acc.setdefault((r["criterion"], r["frac"]), []).append(float(r["bce"]))
base = np.mean(acc[("none", "0")])
with open(f"{OUT}/prune_final.csv", "w") as f:
    w = csv.writer(f); w.writerow(["criterion", "frac", "dbce_mean", "dbce_sd", "n"])
    for (c, fr), v in sorted(acc.items()):
        bs = [x for x in acc[("none", "0")]]
        d = np.array(v) - np.array(bs[:len(v)])
        w.writerow([c, fr, f"{d.mean():.4f}", f"{d.std(ddof=1) if len(d)>1 else 0:.4f}", len(d)])
print(open(f"{OUT}/currency_summary.csv").read()); print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != 'diffs'}) for k, v in res.items()}))
