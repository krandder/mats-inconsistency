"""Paired bootstrap CIs (over boards) for the Gate 0 pool comparisons: D-fit (stakes cross-fitted over boards) minus
the prior, the mean and the vote, in Brier vs exact Bayes. Descriptive only (the pre-registered gate is on raw
informativeness). Reads runs/gate0-q{06,17}/gate0_chains.jsonl; writes gate_pool_ci.json."""
import json, os
import numpy as np
from pool import Board, Claims, fit_stakes, price
from refocus import make_board, open_mask
from env.shiphunt import PRESETS
HERE = os.path.dirname(os.path.abspath(__file__))
B = Board(PRESETS["medium"])
rng = np.random.default_rng(3)
out = {}
for model in ("gate0-q06", "gate0-q17"):
    recs = [json.loads(l) for l in open(os.path.join(HERE, "runs", model, "gate0_chains.jsonl"))]
    for L in (64, 256):
        by = {}
        for r in recs:
            if r["cell"] == "gate0" and r["L"] == L:
                by.setdefault(r["seed"], []).append(r)
        eps = []
        for seed, lst in sorted(by.items()):
            env = make_board(B.cfg, seed); z = B.truth(env)
            ch, st, q, w = [], [], [], []
            for r in lst:
                if r["c"]:
                    for j, (p, c) in zip(r["t"], r["c"]):
                        ch.append(r["cid"]); st.append(j); q.append(p); w.append(c)
            eps.append({"claims": Claims(np.array(ch, int), np.array(st, int), np.array(q), np.array(w)), "y": B.F[z].copy(),
                        "zstar": z, "open": open_mask(B, env), "bayes": B.bayes(env) @ B.F})
        fits = {f: fit_stakes(B, [e for k, e in enumerate(eps) if k % 2 != f], "stmt") for f in (0, 1)}
        se = {m: [] for m in ("prior", "mean", "vote", "D-fit")}
        for k, e in enumerate(eps):
            for m in se:
                p, _ = price(B, e["claims"], m, fits[k % 2] if m == "D-fit" else None)
                se[m].append(float(np.mean((p[e["open"]] - e["bayes"][e["open"]]) ** 2)))
        res = {"boards": len(eps), "brier_vs_bayes": {m: float(np.mean(v)) for m, v in se.items()}}
        for base in ("prior", "mean", "vote"):
            d = np.array(se["D-fit"]) - np.array(se[base])
            bs = d[rng.integers(len(d), size=(4000, len(d)))].mean(1)
            res[f"Dfit_minus_{base}"] = [float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        out[f"{model}|L{L}"] = res
        print(model, L, {k: [round(1000 * x, 2) for x in v] for k, v in res.items() if k.startswith("Dfit")})
json.dump(out, open(os.path.join(HERE, "gate_pool_ci.json"), "w"), indent=1)
