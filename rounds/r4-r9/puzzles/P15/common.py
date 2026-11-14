"""Shared helpers for P7: discrete information measures, lookup probes, t-intervals."""
import math
import numpy as np

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def ci(xs):
    xs = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    n = len(xs)
    if n == 0:
        return None
    m = float(np.mean(xs))
    if n == 1:
        return {"mean": m, "ci95": None, "n": 1}
    sd = float(np.std(xs, ddof=1))
    return {"mean": m, "ci95": T975.get(n - 1, 1.96) * sd / math.sqrt(n), "n": n}


def _H(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def entropy(a):
    _, c = np.unique(a, return_counts=True)
    return _H(c.astype(float))


def mutual_info(a, b):
    a = np.asarray(a).astype(np.int64)
    b = np.asarray(b).astype(np.int64)
    ab = a * (int(b.max()) + 1) + b
    return entropy(a) + entropy(b) - entropy(ab)


def cond_mutual_info(a, b, c):
    """I(a;b|c) = sum_c p(c) I(a;b|C=c)."""
    c = np.asarray(c)
    tot = 0.0
    for v in np.unique(c):
        m = c == v
        tot += m.mean() * mutual_info(np.asarray(a)[m], np.asarray(b)[m])
    return tot


def cond_entropy(b, c):
    c = np.asarray(c)
    tot = 0.0
    for v in np.unique(c):
        m = c == v
        tot += m.mean() * entropy(np.asarray(b)[m])
    return tot


def frac(num, den):
    return float(num / den) if den > 1e-9 else float("nan")


def probe_acc(z, g, seed=0):
    """Best lookup-table readout of g from discrete z (the linear probe on one-hot z),
    fitted on a random half, scored on the other half. Returns (acc, majority baseline)."""
    z = np.asarray(z).astype(np.int64)
    g = np.asarray(g).astype(np.int64)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(z))
    fit, ev = idx[: len(z) // 2], idx[len(z) // 2:]
    G = int(g.max()) + 1
    Z = int(z.max()) + 1
    tab = np.zeros((Z, G))
    np.add.at(tab, (z[fit], g[fit]), 1)
    glob = np.bincount(g[fit], minlength=G).argmax()
    lut = np.where(tab.sum(1) > 0, tab.argmax(1), glob)
    acc = float((lut[z[ev]] == g[ev]).mean())
    base = float((g[ev] == glob).mean())
    return acc, base


def alignment(z, g, y, seed=0):
    """How much of ground-truth variable g the latent z carries, and whether it only copies y.
    nmi_g      = I(z;g)/H(g)
    probe_g    = lookup-probe accuracy for g (baseline = majority)
    extra_g    = I(z;g|y)/H(g|y): information about g beyond the output (0 = collapsed onto y)
    nmi_y      = I(z;y)/H(y)
    """
    z = np.asarray(z)
    g = np.asarray(g)
    y = np.asarray(y)
    pa, pb = probe_acc(z, g, seed)
    return {
        "nmi_g": frac(mutual_info(z, g), entropy(g)),
        "probe_g": pa,
        "probe_g_base": pb,
        "extra_g": frac(cond_mutual_info(z, g, y), cond_entropy(g, y)),
        "nmi_y": frac(mutual_info(z, y), entropy(y)),
        "n_states": int(len(np.unique(z))),
    }


def spearman(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    d = math.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def auroc(score, label):
    score = np.asarray(score, float)
    label = np.asarray(label).astype(bool)
    npos, nneg = label.sum(), (~label).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    r = np.argsort(np.argsort(score)).astype(float) + 1
    return float((r[label].sum() - npos * (npos + 1) / 2) / (npos * nneg))
