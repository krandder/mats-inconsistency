"""Post-hoc baseline for L7 (EXPLORATORY), added after the L4 run: online Bayesian logistic regression.

Why added: conj_bayes (generative, generic prior) gets worse at small H on train tasks, so it is a weak
"online Bayesian" reference. This one is the discriminative counterpart: logit = l0 + w . (r - l0) + b with a
Gaussian prior centred on the fixed extremized log pool (w0 = alpha / N, b0 = 0), MAP by Newton on the H
past episodes, MacKay probit-corrected predictive. Prior scales are fitted once per env on train tasks
(mean log loss over H in {4, 16, 64}), not per H. Same dev tasks and histories as l7_icl.py (same seeds and
salts), so its numbers sit next to the main run's. With --icl-forecasts (forecasts.npz of the main run) it also
records paired icl_attn - bayes_logreg regret differences on the same dev episodes.

Usage: PYTHONPATH=. python3 results/explore/l7-in-context-reliability-20261020/posthoc_logreg.py --out DIR
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from l7_icl import ENVS, HMAX, HS, TRAIN_ROWS, histories  # noqa: E402

from aggbench import make_env  # noqa: E402
from aggbench.baselines import all_baselines  # noqa: E402
from aggbench.metrics import bernoulli_logloss, paired_difference, summarize  # noqa: E402
from aggbench.results import make_result, split_record, write_result  # noqa: E402
from aggbench.verify import oracle_check  # noqa: E402


def sig(x):
    return 0.5 * (1 + np.tanh(x / 2))


def logreg_predict(hist, hy, q, H, l0, alpha, sw, sb, iters=25):
    M, _, N = hist.shape
    m0 = np.concatenate([np.full(N, alpha / N), [0.0]])
    P0 = np.diag(np.concatenate([np.full(N, sw ** -2), [sb ** -2]]))
    Phi = np.concatenate([hist[:, :H] - l0, np.ones((M, H, 1))], -1)
    th = np.tile(m0, (M, 1))
    y = hy[:, :H].astype(float)
    for _ in range(iters if H else 0):
        z = np.einsum("mhd,md->mh", Phi, th)
        p = sig(z)
        g = np.einsum("mhd,mh->md", Phi, y - p) - (th - m0) @ P0
        Hs = np.einsum("mhd,mh,mhe->mde", Phi, p * (1 - p), Phi) + P0
        th = th + np.linalg.solve(Hs, g[..., None])[..., 0]
    p = sig(np.einsum("mhd,md->mh", Phi, th)) if H else np.zeros((M, 0))
    Hs = np.einsum("mhd,mh,mhe->mde", Phi, p * (1 - p), Phi) + P0
    S = np.linalg.inv(Hs)
    phq = np.concatenate([q - l0, np.ones((M, 1))], -1)
    mean = np.einsum("md,md->m", phq, th)
    var = np.einsum("md,mde,me->m", phq, S, phq)
    return l0 + mean / np.sqrt(1 + np.pi * var / 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=30000)
    ap.add_argument("--n-dev", type=int, default=2000)
    ap.add_argument("--n-tune", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default="l7-in-context-reliability-20261020")
    ap.add_argument("--icl-forecasts", default=None, help="forecasts.npz from l7_icl.py: adds paired icl - logreg")
    args = ap.parse_args()
    icl = np.load(args.icl_forecasts) if args.icl_forecasts else None
    for name in ENVS:
        env = make_env(name)
        l0 = env.public()["prior_logit"]
        tr, dv = env.sample("train", args.n_train), env.sample("dev", args.n_dev)
        rows, ys = histories(_head(tr, args.n_tune), TRAIN_ROWS, salt=1)  # same RNG seeds as l7_icl.py
        dv_hist, dv_hy = histories(dv, HMAX, salt=2)
        forecasts, alpha = all_baselines(tr, dv, "none")
        best = None
        for sw in [0.01, 0.03, 0.1, 0.3, 1.0]:
            for sb in [0.03, 0.1, 0.3, 1.0]:
                ll = np.mean([bernoulli_logloss(logreg_predict(rows[:, :HMAX], ys[:, :HMAX], rows[:, HMAX], H, l0,
                                                               alpha, sw, sb), ys[:, HMAX]).mean() for H in (4, 16, 64)])
                if best is None or ll < best[0]:
                    best = (ll, sw, sb)
        _, sw, sb = best
        keep = {k: forecasts[k] for k in ("bayes", "oracle_clearing", "extremized_log_pool")}
        for H in HS:
            keep[f"bayes_logreg@H{H}"] = {"logit": logreg_predict(dv_hist, dv_hy, dv.reports["logit"], H, l0, alpha, sw, sb)}
        methods = {k: summarize(f, dv) for k, f in keep.items()}
        icl_paired = {}
        if icl is not None:
            blocks = list(icl["blocks"])
            for H in HS:
                i = blocks.index(f"{name}@H{H}")
                f = {"logit": icl["trained"][i * args.n_dev:(i + 1) * args.n_dev]}
                icl_paired[str(H)] = paired_difference(f, keep[f"bayes_logreg@H{H}"], dv)
        paired = {k: {"vs_extremized_log_pool": paired_difference(f, keep["extremized_log_pool"], dv),
                      "vs_oracle_clearing": paired_difference(f, keep["oracle_clearing"], dv)}
                  for k, f in keep.items() if "@H" in k}
        result = make_result(
            run_id=args.run_id + "-posthoc-logreg", track="EXPLORATORY", env=env, visibility="none",
            splits={"train": split_record(tr, "train"), "eval": split_record(dv, "dev")},
            oracle_check=oracle_check(env), methods=methods,
            extra={"what": "post-hoc online Bayesian logistic regression baseline (see posthoc_logreg.py docstring)",
                   "prior": {"w0": f"alpha/N, alpha={alpha:.4f} (extremized_log_pool fitted on train)", "sw": sw,
                             "sb": sb, "train_logloss_H4_16_64": float(best[0])},
                   "paired_regret_differences": paired,
                   "icl_attn_minus_bayes_logreg": icl_paired,
                   "runner_sha256": __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest()})
        write_result(args.out, result, name=f"posthoc-logreg-{name}.json")
        print(name, {k: round(v["regret"]["mean"], 4) for k, v in methods.items()}, "sw", sw, "sb", sb, flush=True)


def _head(batch, n):
    """First n episodes of a harness Batch (same seeds), as a Batch-like object for histories()."""
    import copy
    b = copy.copy(batch)
    b.seeds = batch.seeds[:n]
    b.oracle = {k: v[:n] for k, v in batch.oracle.items()}
    return b


if __name__ == "__main__":
    main()
