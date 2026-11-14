"""P10 stats: tie-aware average ranks (numpy only), Spearman, partial Spearman, within-group
Spearman, hierarchical bootstrap, t-intervals. Checked against scipy.stats by `python3 stats.py`."""
import math
import numpy as np

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def rankdata(a):
    """Average ranks (1-based), ties get the mean of their positions (= scipy.stats.rankdata 'average')."""
    a = np.asarray(a, float)
    n = len(a)
    order = np.argsort(a, kind='mergesort')
    s = a[order]
    ranks = np.empty(n, float)
    i = 0
    # vectorised tie groups
    change = np.concatenate([[True], s[1:] != s[:-1], [True]])
    starts = np.flatnonzero(change[:-1])
    ends = np.flatnonzero(change[1:]) + 1
    avg = (starts + ends + 1) / 2.0  # mean of 1-based positions starts+1..ends
    grp = np.repeat(avg, ends - starts)
    ranks[order] = grp
    return ranks


def pearson(x, y):
    x = np.asarray(x, float) - np.mean(x)
    y = np.asarray(y, float) - np.mean(y)
    d = math.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else float('nan')


def spearman(a, b):
    return pearson(rankdata(a), rankdata(b))


def partial_from_r(rxy, rxz, ryz):
    den = math.sqrt(max(1e-12, (1 - rxz ** 2) * (1 - ryz ** 2)))
    return (rxy - rxz * ryz) / den


def partial_spearman(x, y, z):
    """Spearman of x,y controlling for z (and z may be a 2-D array of several controls):
    residualise ranks of x and y on ranks of the controls, then Pearson."""
    rx, ry = rankdata(x), rankdata(y)
    z = np.asarray(z, float)
    if z.ndim == 1:
        z = z[:, None]
    Z = np.column_stack([np.ones(len(rx))] + [rankdata(z[:, j]) for j in range(z.shape[1])])
    bx = np.linalg.lstsq(Z, rx, rcond=None)[0]
    by = np.linalg.lstsq(Z, ry, rcond=None)[0]
    return pearson(rx - Z @ bx, ry - Z @ by)


def within_ranks(v, g):
    """Within-group average ranks, centred and scaled to [-0.5, 0.5] per group (groups of size 1 -> 0)."""
    v = np.asarray(v, float)
    g = np.asarray(g)
    out = np.zeros(len(v))
    for k in np.unique(g):
        m = g == k
        n = m.sum()
        if n > 1:
            out[m] = (rankdata(v[m]) - (n + 1) / 2) / (n - 1)
    return out


def within_spearman(score, target, group, min_n=3):
    """Per-group tie-aware Spearman; returns array of per-group values (NaN groups dropped)."""
    score = np.asarray(score, float)
    target = np.asarray(target, float)
    group = np.asarray(group)
    out = []
    for k in np.unique(group):
        m = group == k
        if m.sum() < min_n:
            continue
        r = spearman(score[m], target[m])
        if not math.isnan(r):
            out.append(r)
    return np.array(out)


def within_partial(x, y, z, group):
    """Within-group partial Spearman: pooled Pearson of within-group ranks of x and y after regressing
    out within-group ranks of the controls z (1-D or 2-D)."""
    rx, ry = within_ranks(x, group), within_ranks(y, group)
    z = np.asarray(z, float)
    if z.ndim == 1:
        z = z[:, None]
    Z = np.column_stack([np.ones(len(rx))] + [within_ranks(z[:, j], group) for j in range(z.shape[1])])
    bx = np.linalg.lstsq(Z, rx, rcond=None)[0]
    by = np.linalg.lstsq(Z, ry, rcond=None)[0]
    return pearson(rx - Z @ bx, ry - Z @ by)


def tci(xs):
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    n = len(xs)
    if n == 0:
        return {"mean": None, "ci95": None, "n": 0}
    m = float(np.mean(xs))
    if n == 1:
        return {"mean": m, "ci95": None, "n": 1}
    sd = float(np.std(xs, ddof=1))
    return {"mean": m, "ci95": T975.get(n - 1, 1.96) * sd / math.sqrt(n), "n": n,
            "lo": m - T975.get(n - 1, 1.96) * sd / math.sqrt(n), "hi": m + T975.get(n - 1, 1.96) * sd / math.sqrt(n)}


def boot_ci(x, B=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return {"mean": float(x.mean()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)), "n": len(x)}


def hier_boot(per_seed, B=4000, seed=0):
    """per_seed: list of 1-D arrays (per-state values for each seed). Resample seeds, then states within
    each drawn seed; statistic = mean over drawn seeds of per-seed means. Returns mean and 95% interval."""
    rng = np.random.default_rng(seed)
    S = len(per_seed)
    stats = np.empty(B)
    for b in range(B):
        ss = rng.integers(0, S, S)
        vals = []
        for s in ss:
            x = per_seed[s]
            vals.append(x[rng.integers(0, len(x), len(x))].mean())
        stats[b] = np.mean(vals)
    m = float(np.mean([np.mean(x) for x in per_seed]))
    return {"mean": m, "lo": float(np.quantile(stats, 0.025)), "hi": float(np.quantile(stats, 0.975)), "n_seeds": S}


def paired_boot_diff(a, b, B=4000, seed=0):
    """Bootstrap CI of mean(a-b) for paired per-item values (e.g. solved flags)."""
    return boot_ci(np.asarray(a, float) - np.asarray(b, float), B, seed)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj


def t_pvalue_two_sided(xs):
    """Two-sided one-sample t-test p-value against 0 (uses scipy if available, else normal approx)."""
    xs = np.asarray(xs, float)
    n = len(xs)
    sd = xs.std(ddof=1)
    if sd == 0:
        return 0.0 if xs.mean() != 0 else 1.0
    t = xs.mean() / (sd / math.sqrt(n))
    try:
        from scipy import stats as st
        return float(2 * st.t.sf(abs(t), n - 1))
    except Exception:
        return float(math.erfc(abs(t) / math.sqrt(2)))


if __name__ == "__main__":
    import scipy.stats as st
    rng = np.random.default_rng(1)
    for trial in range(200):
        n = rng.integers(3, 60)
        a = rng.integers(0, 5, n).astype(float)
        b = a + rng.integers(0, 3, n)
        assert np.allclose(rankdata(a), st.rankdata(a)), "rank mismatch"
        r1, r2 = spearman(a, b), st.spearmanr(a, b).statistic
        assert (math.isnan(r1) and math.isnan(r2)) or abs(r1 - r2) < 1e-9, (r1, r2)
    print("stats.py: rankdata and spearman match scipy.stats", st.__name__, "on 200 tied samples")
