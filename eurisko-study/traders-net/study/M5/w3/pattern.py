"""M5 worker 3: pattern-2/4/8 shadow traders (soup rule, KT table, price-reading corrector).

Shadow traders have wealth -> 0 and never move the price, so the traded price equals the
clearing price (T1 lemma 5) and no fixed point is needed.

Interface (numpy only):
    bits  [Q, Tb] uint8   full outcome sequence per question
    W     int             position i predicts bits[:, i+W]; history = bits[:, :i+W]
    z     [n_pos, Q]      market logits (NaN allowed outside the scored range)
    lo, hi                scored positions [lo, hi)
    fam   [Q] int, n_fam
run_class(bits, W, z, lo, hi, fam, n_fam) -> dict name -> {"log": [hi-lo, n_fam] or None,
                                                          "share": [hi-lo, n_fam]}
(sums over the questions of each family, per scored position).
"""
import numpy as np

KS = (2, 4, 8)
EPS = 1e-4


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def contexts(bits, W, n_pos, k):
    """ctx[i, q] = last k revealed bits before bits[q, i+W] as an int (age 0 = LSB)."""
    Q = bits.shape[0]
    ctx = np.zeros((n_pos, Q), np.int64)
    for a in range(k):  # age a -> bits[:, i+W-1-a]
        ctx |= bits[:, W - 1 - a: W - 1 - a + n_pos].T.astype(np.int64) << a
    return ctx


def soup_pred(bits, W, lo, hi, k, L=32):
    """Soup rule (logical-soup/docs/shadow.md): most recent j in 1..L-k with ages j..j+k-1 ==
    ages 0..k-1 -> predict age j-1. Returns [hi-lo, Q] in {0,1} or -1 (abstain)."""
    Q = bits.shape[0]
    n = hi - lo
    # H[i, q, a] = age-a outcome = bits[q, lo+i+W-1-a]; need lo+W-L >= 0
    assert lo + W - L >= 0, "need L bits of history"
    H = np.stack([bits[:, lo + W - 1 - a: lo + W - 1 - a + n].T for a in range(L)], -1)  # [n,Q,L]
    pred = np.full((n, Q), -1, np.int8)
    head = H[..., :k]
    for j in range(1, L - k + 1):
        m = (H[..., j:j + k] == head).all(-1) & (pred < 0)
        pred[m] = H[..., j - 1][m]
    return pred


def fam_sum(x, fam, n_fam):
    out = np.zeros((x.shape[0], n_fam))
    for f in range(n_fam):
        sel = fam == f
        if sel.any():
            out[:, f] = x[:, sel].sum(1)
    return out


def run_class(bits, W, z, lo, hi, fam, n_fam, lr=0.05, clip=4.0, ks=KS):
    bits = np.asarray(bits, np.uint8)
    Q = bits.shape[0]
    n_pos = z.shape[0]
    y = bits[:, W:W + n_pos].T.astype(np.float64)          # [n_pos, Q]
    zs = np.asarray(z[lo:hi], np.float64)
    ys = y[lo:hi]
    p = np.clip(_sig(zs), EPS, 1 - EPS)
    logp = np.where(ys > 0, np.log(p), np.log1p(-p))
    out = {}
    for k in ks:
        # soup rule, one-share currency only
        sp = soup_pred(bits, W, lo, hi, k)
        d = np.where(sp < 0, 0.0, np.sign(sp - p))
        out[f"soup-{k}"] = {"log": None, "share": fam_sum(d * (ys - p), fam, n_fam)}
        # KT table (learns on all positions from 0) and corrector (learns on scored positions)
        ctx = contexts(bits, W, n_pos, k)
        n1 = np.zeros((Q, 2 ** k)); nn = np.zeros((Q, 2 ** k))
        delta = np.zeros((Q, 2 ** k))
        qi = np.arange(Q)
        qkt = np.empty((hi - lo, Q)); qco = np.empty((hi - lo, Q))
        for i in range(hi):
            c = ctx[i]
            if i >= lo:
                r = i - lo
                qkt[r] = (n1[qi, c] + 0.5) / (nn[qi, c] + 1.0)
                qc = _sig(zs[r] + delta[qi, c])
                qco[r] = qc
                delta[qi, c] = np.clip(delta[qi, c] + lr * (ys[r] - qc), -clip, clip)
            n1[qi, c] += y[i]; nn[qi, c] += 1.0
        for name, qb in ((f"kt-{k}", qkt), (f"corr-{k}", qco)):
            qb = np.clip(qb, EPS, 1 - EPS)
            logq = np.where(ys > 0, np.log(qb), np.log1p(-qb))
            d = np.sign(qb - p)
            out[name] = {"log": fam_sum(logq - logp, fam, n_fam),
                         "share": fam_sum(d * (ys - p), fam, n_fam)}
    mkt = {"bce": fam_sum(-logp, fam, n_fam), "brier": fam_sum((p - ys) ** 2, fam, n_fam)}
    return out, mkt


def slope(series):
    """Least-squares slope of the cumulative sum of a per-position series (per position)."""
    c = np.cumsum(series)
    t = np.arange(len(c), dtype=np.float64)
    t -= t.mean()
    return float((t * (c - c.mean())).sum() / (t * t).sum())


def ec_stats(per_pos, Q):
    """per_pos: [n] total income per position (all questions). Returns slopes per question-position."""
    n = len(per_pos)
    h = n // 2
    return {"slope_h1": slope(per_pos[:h]) / Q, "slope_h2": slope(per_pos[h:]) / Q,
            "phi_last20": slope(per_pos[int(n * 0.8):]) / Q, "E_end": float(per_pos.sum()) / Q}
