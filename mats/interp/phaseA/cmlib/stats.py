"""Small statistics helpers: ESS, AUC, ECE, tail diagnostics (PSIS k-hat, Hill), bootstrap CIs."""
import numpy as np
from scipy.stats import rankdata


def logsumexp(a, axis=None):
    a = np.asarray(a, float)
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis) if axis is not None else float(out.ravel()[0])


def kish_ess(w):
    """Kish effective sample size (sum w)^2 / sum w^2 of non-negative weights."""
    w = np.asarray(w, float)
    s2 = np.sum(w * w)
    return float(np.sum(w) ** 2 / s2) if s2 > 0 else 0.0


def kish_ess_log(logw):
    logw = np.asarray(logw, float)
    return kish_ess(np.exp(logw - np.max(logw)))


def auc(score_pos, score_neg):
    """Mann-Whitney AUC: P(score_pos > score_neg), ties count 1/2."""
    s = np.concatenate([score_pos, score_neg])
    r = rankdata(s)
    n1, n0 = len(score_pos), len(score_neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_se(a, n1, n0):
    """Hanley-McNeil standard error of an AUC."""
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    return float(np.sqrt((a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0)))


def ece(prob, label, nbins=15, weight=None):
    """Expected calibration error with equal-width bins; optional per-sample weights (for class balancing)."""
    prob = np.asarray(prob, float); label = np.asarray(label, float)
    weight = np.ones_like(prob) if weight is None else np.asarray(weight, float)
    bins = np.minimum((prob * nbins).astype(int), nbins - 1)
    tot = weight.sum(); e = 0.0
    for b in range(nbins):
        m = bins == b
        if not m.any():
            continue
        wb = weight[m]
        e += wb.sum() / tot * abs(np.average(prob[m], weights=wb) - np.average(label[m], weights=wb))
    return float(e)


def _gpdfit(x):
    """Zhang & Stephens (2009) generalized-Pareto fit with the weak prior used by PSIS (Vehtari et al.).
    x: sorted ascending positive exceedances. Returns (k, sigma); k > 0 = heavy tail, 1/k = tail index."""
    n = len(x)
    prior_bs, prior_k = 3, 10
    m_est = 30 + int(n ** 0.5)
    b = 1 - np.sqrt(m_est / (np.arange(1, m_est + 1) - 0.5))
    b /= prior_bs * x[int(n / 4 + 0.5) - 1]
    b += 1 / x[-1]
    k_arr = np.log1p(-b[:, None] * x).mean(axis=1)
    len_scale = n * (np.log(-(b / k_arr)) - k_arr - 1)
    wts = 1 / np.exp(len_scale - len_scale[:, None]).sum(axis=1)
    ok = wts >= 10 * np.finfo(float).eps
    wts, b = wts[ok], b[ok]
    wts /= wts.sum()
    b_post = np.sum(b * wts)
    k_post = np.log1p(-b_post * x).mean()
    sigma = -k_post / b_post
    k_post = (n * k_post + prior_k * 0.5) / (n + prior_k)
    return float(k_post), float(sigma)


def psis_khat(logw):
    """Pareto k-hat of the upper tail of the weights exp(logw) (PSIS diagnostic).
    < 0.5: finite variance plausible; 0.5-0.7: usable with care; > 0.7: IS estimate unreliable."""
    lw = np.sort(np.asarray(logw, float))
    n = len(lw)
    M = int(min(n / 5, 3 * np.sqrt(n)))
    if M < 5:
        return float('nan')
    lw = lw - lw[-1]
    tail = np.exp(lw[-M:]); cut = np.exp(lw[-M - 1])
    ex = tail - cut
    if np.all(ex <= 0):
        return -np.inf
    ex = np.maximum(ex, 1e-300)
    return _gpdfit(ex)[0]


def hill_tail_index(x, frac=0.02):
    """Hill estimate of the tail index alpha of |x| from the top `frac` order statistics (alpha < 2: infinite variance)."""
    a = np.sort(np.abs(np.asarray(x, float)))[::-1]
    k = max(int(len(a) * frac), 10)
    if a[k] <= 0:
        return float('inf')
    return float(1.0 / np.mean(np.log(a[:k] / a[k])))


def boot_mean_t(d, B=2000, rng=None, level=0.95):
    """Studentised bootstrap CI for the mean of d (1-D). Returns (mean, se, lo, hi)."""
    rng = np.random.default_rng(0) if rng is None else rng
    d = np.asarray(d, float); n = len(d)
    m = d.mean(); se = d.std(ddof=1) / np.sqrt(n)
    idx = rng.integers(0, n, size=(B, n))
    db = d[idx]
    mb = db.mean(1); seb = db.std(1, ddof=1) / np.sqrt(n)
    t = (mb - m) / np.maximum(seb, 1e-300)
    a = (1 - level) / 2
    tlo, thi = np.quantile(t, [a, 1 - a])
    return float(m), float(se), float(m - thi * se), float(m - tlo * se)


def boot_percentile(stat_fn, n, B=1000, rng=None, level=0.95, strata=None):
    """Percentile bootstrap of stat_fn(idx) over n units (optionally stratified). Returns (lo, hi, sd, samples)."""
    rng = np.random.default_rng(0) if rng is None else rng
    out = []
    for _ in range(B):
        if strata is None:
            idx = rng.integers(0, n, n)
        else:
            idx = np.concatenate([rng.choice(np.flatnonzero(strata == s), size=int((strata == s).sum()))
                                  for s in np.unique(strata)])
        out.append(stat_fn(idx))
    out = np.asarray(out, float)
    a = (1 - level) / 2
    lo, hi = np.quantile(out, [a, 1 - a], axis=0)
    return lo, hi, out.std(0, ddof=1), out
