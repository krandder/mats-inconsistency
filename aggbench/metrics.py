"""Metrics against the truth and against the exact Bayes posterior.

A forecast is a dict: {"logit": (n,)} for bernoulli envs, {"mean": (n, d), "prec": (n, d, d)}
for gaussian envs. Per episode:
  logloss  -log q(truth)                 (Gaussian: negative log density of theta)
  brier    (q - y)^2                     (Gaussian: squared error of the mean)
  kl       KL(Bayes posterior || q)
  regret   logloss(q) - logloss(Bayes);  its expectation equals E[kl], kl is the
           lower-variance estimate of the same quantity.
"""
import numpy as np

METRICS = ("logloss", "brier", "kl", "regret")


def logsig(x):
    return -np.logaddexp(0.0, -x)


def sigmoid(x):
    return np.exp(logsig(x))


def bernoulli_logloss(logit, y):
    return -(y * logsig(logit) + (1 - y) * logsig(-logit))


def gaussian_nll(mean, prec, theta):
    diff = theta - mean
    d = theta.shape[-1]
    quad = np.einsum("ni,nij,nj->n", diff, prec, diff)
    return 0.5 * (quad - np.linalg.slogdet(prec)[1] + d * np.log(2 * np.pi))


def gaussian_kl(m0, p0, m1, p1):
    """KL(N(m0, p0^-1) || N(m1, p1^-1)) per episode."""
    d = m0.shape[-1]
    tr = np.einsum("nij,nji->n", p1, np.linalg.inv(p0))
    diff = m1 - m0
    quad = np.einsum("ni,nij,nj->n", diff, p1, diff)
    return 0.5 * (tr + quad - d + np.linalg.slogdet(p0)[1] - np.linalg.slogdet(p1)[1])


def per_episode(forecast, batch):
    """Dict metric -> (n,) array."""
    if batch.kind == "bernoulli":
        y, q, b = batch.truth, forecast["logit"], batch.bayes["logit"]
        ll, llb = bernoulli_logloss(q, y), bernoulli_logloss(b, y)
        pb = sigmoid(b)
        kl = pb * (logsig(b) - logsig(q)) + (1 - pb) * (logsig(-b) - logsig(-q))
        brier = (sigmoid(q) - y) ** 2
    else:
        th, b = batch.truth, batch.bayes
        ll = gaussian_nll(forecast["mean"], forecast["prec"], th)
        llb = gaussian_nll(b["mean"], b["prec"], th)
        kl = gaussian_kl(b["mean"], b["prec"], forecast["mean"], forecast["prec"])
        brier = ((forecast["mean"] - th) ** 2).sum(-1)
    return {"logloss": ll, "brier": brier, "kl": kl, "regret": ll - llb}


def bootstrap_ci(x, n_boot=2000, level=0.95, seed=0):
    """Percentile bootstrap CI of the mean over episodes."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for start in range(0, n_boot, 200):  # chunks keep memory small for large n
        idx = rng.integers(0, len(x), (min(200, n_boot - start), len(x)))
        means[start:start + len(idx)] = x[idx].mean(1)
    alpha = (1 - level) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def summarize_values(values, n_boot=2000, seed=0):
    lo, hi = bootstrap_ci(values, n_boot, seed=seed)
    return {"mean": float(np.mean(values)), "ci_low": lo, "ci_high": hi, "n": int(len(values))}


def summarize(forecast, batch, n_boot=2000, seed=0):
    """Dict metric -> {mean, ci_low, ci_high, n} with 95% bootstrap CIs over episodes."""
    return {k: summarize_values(v, n_boot, seed) for k, v in per_episode(forecast, batch).items()}


def paired_difference(forecast_a, forecast_b, batch, metric="regret", n_boot=2000, seed=0):
    """Mean and CI of metric(a) - metric(b) on the same episodes (use this to compare two arms)."""
    da = per_episode(forecast_a, batch)[metric]
    db = per_episode(forecast_b, batch)[metric]
    return summarize_values(da - db, n_boot, seed)
