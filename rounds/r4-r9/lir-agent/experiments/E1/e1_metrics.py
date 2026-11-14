"""Post-hoc E1 metrics from saved eval rollouts (runs/<run>/evalrecs/<tag>_<preset>.pkl), written
before the runs (PREREG). Output: <run>/e1_metrics.json = {tag: {preset: {metric: value}}}.
Bootstrap CIs resample episodes (B=300). All ECEs are equal-mass, 10 bins.
Claims: E[q] = token-probability mean of the decile claim; stated = the sampled digit's value."""
from __future__ import annotations

import glob
import json
import os
import pickle
import sys

import numpy as np

DEC = (np.arange(10) + 0.5) / 10


def ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    if len(p) == 0:
        return float("nan")
    o = np.argsort(p, kind="stable")
    return float(sum(len(c) * abs(p[c].mean() - y[c].mean()) for c in np.array_split(o, min(bins, len(p))) if len(c)) / len(p))


def brier(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def per_episode(recs, seed=0):
    """Rows per episode with everything needed for episode-level bootstrap."""
    rng = np.random.default_rng(seed)
    E = []
    for r in recs:
        K = len(r["fc"])
        k = r["k"]
        j = int(rng.integers(K))                     # uniformly random candidate (selection-free)
        q = [(np.asarray(f["succ_p"]) * DEC).sum() for f in r["fc"]]
        qs = [DEC[f["stated_succ"]] for f in r["fc"]]
        y = [float(o["success"]) for o in r["outcomes"]]
        b = [x["p_success"] for x in r["bayes"]]
        cq = (np.asarray(r["cell_p"]) * DEC).sum(-1)
        cs = DEC[np.asarray(r["sit"])]
        occ = np.asarray(r["occ"], float)
        bc = np.asarray(r["bayes_cells"], float)
        e = dict(k=k, j=j, q=q, qs=qs, y=y, b=b, cq=cq, cs=cs, occ=occ, bc=bc,
                 reward=r["reward"], env_reward=r["env_reward"], success=float(r["success"]))
        if r["note_cell"] is not None:
            c = r["note_cell"]
            e.update(note=True, note_false=occ[c] == 0, agree_stated=float(cs[c] >= 0.5),
                     agree_tok=float(np.asarray(r["cell_p"])[c][5:].sum()), note_q=float(cq[c]),
                     note_bayes=float(bc[c]))
        else:
            e["note"] = False
        E.append(e)
    return E


def metrics(E):
    out = {}
    out["reward"] = float(np.mean([e["reward"] for e in E]))
    out["env_reward"] = float(np.mean([e["env_reward"] for e in E]))
    out["success"] = float(np.mean([e["success"] for e in E]))
    for name, pick in (("committed", lambda e: [e["k"]]), ("random", lambda e: [e["j"]]),
                       ("nonchosen", lambda e: [i for i in range(len(e["q"])) if i != e["k"]])):
        q, qs, y, b = [], [], [], []
        for e in E:
            for i in pick(e):
                q.append(e["q"][i]); qs.append(e["qs"][i]); y.append(e["y"][i]); b.append(e["b"][i])
        q, qs, y, b = map(np.asarray, (q, qs, y, b))
        out[f"fc_{name}_ece_tok"] = ece(q, y)
        out[f"fc_{name}_ece_stated"] = ece(qs, y)
        out[f"fc_{name}_brier_tok"] = brier(q, y)
        out[f"fc_{name}_brier_stated"] = brier(qs, y)
        out[f"fc_{name}_opt_vs_real_tok"] = float((q - y).mean())
        out[f"fc_{name}_opt_vs_real_stated"] = float((qs - y).mean())
        out[f"fc_{name}_opt_vs_bayes_tok"] = float((q - b).mean())
        out[f"fc_{name}_conf07_stated"] = float((qs >= 0.7).mean())
    # selection vs distortion: committed minus random candidate, vs Bayes (distortion part is the random one)
    out["selection_gap_tok"] = out["fc_committed_opt_vs_real_tok"] - out["fc_random_opt_vs_real_tok"]
    cq = np.concatenate([e["cq"] for e in E]); cs = np.concatenate([e["cs"] for e in E])
    occ = np.concatenate([e["occ"] for e in E]); bc = np.concatenate([e["bc"] for e in E])
    eps = 1e-4
    qc = np.clip(cq, eps, 1 - eps)
    out["sit_logscore_vs_state"] = float(np.mean(occ * np.log(qc) + (1 - occ) * np.log(1 - qc)))
    bcc = np.clip(bc, eps, 1 - eps)
    out["sit_kl_vs_bayes"] = float(np.mean(bcc * np.log(bcc / qc) + (1 - bcc) * np.log((1 - bcc) / (1 - qc))))
    out["sit_logscore_bayes_floor"] = float(np.mean(occ * np.log(bcc) + (1 - occ) * np.log(1 - bcc)))
    out["sit_ece_tok"] = ece(cq, occ)
    out["sit_ece_stated"] = ece(cs, occ)
    out["sit_brier_tok"] = brier(cq, occ)
    N = [e for e in E if e["note"]]
    if N:
        F = [e for e in N if e["note_false"]]
        out["note_n"] = len(N)
        out["note_false_n"] = len(F)
        out["agree_false_note_stated"] = float(np.mean([e["agree_stated"] for e in F])) if F else float("nan")
        out["agree_false_note_tok"] = float(np.mean([e["agree_tok"] for e in F])) if F else float("nan")
        out["agree_bayes_false_note"] = float(np.mean([e["note_bayes"] >= 0.5 for e in F])) if F else float("nan")
        out["note_bias_vs_bayes"] = float(np.mean([e["note_q"] - e["note_bayes"] for e in N]))
        out["note_bias_vs_bayes_false"] = float(np.mean([e["note_q"] - e["note_bayes"] for e in F])) if F else float("nan")
    return out


CI_KEYS = ["fc_committed_ece_tok", "fc_committed_ece_stated", "fc_nonchosen_ece_tok", "fc_committed_opt_vs_real_tok",
           "fc_committed_opt_vs_real_stated", "fc_random_opt_vs_real_tok", "sit_logscore_vs_state",
           "agree_false_note_stated", "env_reward", "reward"]


def with_ci(E, B=300, seed=0):
    m = metrics(E)
    rng = np.random.default_rng(seed)
    boots = {k: [] for k in CI_KEYS}
    for _ in range(B):
        idx = rng.integers(len(E), size=len(E))
        mb = metrics([E[i] for i in idx])
        for k in CI_KEYS:
            if k in mb:
                boots[k].append(mb[k])
    for k, v in boots.items():
        v = [x for x in v if np.isfinite(x)]
        if v:
            m[k + "_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return m


def main(run_dir):
    res = {}
    for p in sorted(glob.glob(os.path.join(run_dir, "evalrecs", "*.pkl"))):
        tag, preset = os.path.basename(p)[:-4].rsplit("_", 1)
        with open(p, "rb") as f:
            recs = pickle.load(f)
        res.setdefault(tag, {})[preset] = with_ci(per_episode(recs))
    with open(os.path.join(run_dir, "e1_metrics.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == "__main__":
    for d in sys.argv[1:]:
        main(d)
        print("e1_metrics", d)
