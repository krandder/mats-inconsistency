"""Correlation of gradient / update share with dSyco across A1 + A4 runs (writes share_corr.json)."""
import json, os
import numpy as np
E2 = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(E2, "results.json")))
rs = [r for r in d["rows"] if r["arm"].startswith("A4") or r["arm"] == "A1"]
rk = lambda x: np.argsort(np.argsort(x))
out = {"runs": [{k: r[k] for k in ("name", "grad_share_mean", "update_share_mean", "d.main.syco_note_bias", "rl_reward_last20")} for r in rs]}
for sub, name in [(rs, "A1+A4"), ([r for r in rs if r["arm"].startswith("A4")], "A4_only_alpha_gt_0")]:
    g = np.array([r["grad_share_mean"] for r in sub]); u = np.array([r["update_share_mean"] for r in sub])
    y = np.array([r["d.main.syco_note_bias"] for r in sub])
    out[name] = {"n": len(sub), "spearman_grad_share": float(np.corrcoef(rk(g), rk(y))[0, 1]),
                 "spearman_update_share": float(np.corrcoef(rk(u), rk(y))[0, 1]),
                 "pearson_grad_share": float(np.corrcoef(g, y)[0, 1]), "pearson_update_share": float(np.corrcoef(u, y)[0, 1])}
json.dump(out, open(os.path.join(E2, "share_corr.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))
