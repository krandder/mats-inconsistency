"""Gap estimators.

Notation (METHOD.md §3): Delta = L(q) - L(p) = E_q[G] - E_p[G] = -Cov_q(G, w), w = p/q.
Kelvin's tempering: p_alpha ∝ q w^alpha; L(alpha) = E_{p_alpha}[G]; L(0) - L(1) = Delta.

- direct_gap      paired / unpaired direct Delta with studentised bootstrap CI and Hill tail index (no weights)
- cell_rms        unbiased E_c[Delta(c)^2] from two independent inner halves; Delta_rms (cells = episodes)
- group_rms       the same over a few discrete groups z with many draws each
- cov_qw          q-side identity: Cov_q(G, w) (plain IS and self-normalised), per group, with bootstrap CI
- temper_q        Kelvin's SNIS from q with weights w^alpha, normalised per group
- temper_bridge   bridge path on the pooled p+q sample, weights w^alpha / (pi_p w + pi_q), normalised per group;
                  plus dL/dalpha = Cov_{p_alpha}(G, log w) and its trapezoid integral
- region_split    exact localisation Delta = sum_A Delta_A with mass / within-region parts
"""
import numpy as np

from .stats import boot_mean_t, hill_tail_index, kish_ess


def _groups(g, n):
    return np.zeros(n, int) if g is None else np.asarray(g)


# ----------------------------------------------------------------------------- direct
def direct_gap(Gq, Gp, paired=True, B=2000, seed=0):
    """Direct Delta = E_q G - E_p G.
    paired: Gq shape (n, M) or (n,), Gp shape (n,) from the same episodes (common random numbers).
    unpaired: independent samples; CI by Welch normal approximation plus bootstrap."""
    rng = np.random.default_rng(seed)
    Gq = np.asarray(Gq, float); Gp = np.asarray(Gp, float)
    if paired:
        gq = Gq.mean(1) if Gq.ndim == 2 else Gq
        d = gq - Gp
        m, se, lo, hi = boot_mean_t(d, B=B, rng=rng)
        return dict(delta=m, se=se, lo=lo, hi=hi, n=len(d), hill=hill_tail_index(d - d.mean()))
    gq = Gq.ravel()
    m = gq.mean() - Gp.mean()
    se = np.sqrt(gq.var(ddof=1) / len(gq) + Gp.var(ddof=1) / len(Gp))
    return dict(delta=float(m), se=float(se), lo=float(m - 1.96 * se), hi=float(m + 1.96 * se),
                n=(len(gq), len(Gp)))


def cell_rms(Gq, Gp, B=1000, seed=0):
    """Per-cell RMS gap. Gq: (n_cells, Mq), Gp: (n_cells, Mp), draws at fixed cell c_i, Mq, Mp >= 2.
    E_c[Delta(c)^2] is estimated without bias by mean_i DeltaA_i * DeltaB_i (independent halves)."""
    Gq = np.asarray(Gq, float); Gp = np.asarray(Gp, float)
    hq, hp = Gq.shape[1] // 2, Gp.shape[1] // 2
    dA = Gq[:, :hq].mean(1) - Gp[:, :hp].mean(1)
    dB = Gq[:, hq:].mean(1) - Gp[:, hp:].mean(1)
    prod = dA * dB
    rng = np.random.default_rng(seed)
    n = len(prod)
    bs = prod[rng.integers(0, n, (B, n))].mean(1)
    ms = prod.mean()
    return dict(ms=float(ms), ms_se=float(prod.std(ddof=1) / np.sqrt(n)),
                rms=float(np.sqrt(max(ms, 0.0))),
                rms_lo=float(np.sqrt(max(np.quantile(bs, 0.025), 0))),
                rms_hi=float(np.sqrt(max(np.quantile(bs, 0.975), 0))),
                delta=float((Gq.mean(1) - Gp.mean(1)).mean()))


def group_rms(Gq, zq, Gp, zp, group_probs=None, B=500, seed=0):
    """RMS gap over a few discrete groups z (many draws per group): Delta(z) per group, and
    E_z Delta(z)^2 estimated without bias by the product of two independent halves within each group.
    Returns per-group deltas, rms, delta, and bootstrap replicates (resampling draws within groups)."""
    Gq, Gp, zq, zp = (np.asarray(a) for a in (Gq, Gp, zq, zp))
    ks = np.unique(zp)
    if group_probs is None:
        group_probs = {k: float(np.mean(zp == k)) for k in ks}
    rng = np.random.default_rng(seed)
    iq = {k: rng.permutation(np.flatnonzero(zq == k)) for k in ks}
    ip = {k: rng.permutation(np.flatnonzero(zp == k)) for k in ks}

    def point(sel_q, sel_p):
        ms = dl = 0.0; per = {}
        for k in ks:
            a, b = sel_q[k], sel_p[k]
            ha, hb = len(a) // 2, len(b) // 2
            dA = Gq[a[:ha]].mean() - Gp[b[:hb]].mean(); dB = Gq[a[ha:]].mean() - Gp[b[hb:]].mean()
            ms += group_probs[k] * dA * dB
            per[k] = Gq[a].mean() - Gp[b].mean(); dl += group_probs[k] * per[k]
        return ms, dl, per
    ms, dlt, per = point(iq, ip)
    if B == 0:
        return dict(ms=float(ms), rms=float(np.sqrt(max(ms, 0))), delta=float(dlt), per_group=per)
    bs = []
    for _ in range(B):
        bq = {k: rng.choice(v, len(v)) for k, v in iq.items()}
        bp = {k: rng.choice(v, len(v)) for k, v in ip.items()}
        bs.append(point(bq, bp)[:2])
    bs = np.asarray(bs)
    rms_b = np.sqrt(np.maximum(bs[:, 0], 0))
    return dict(ms=float(ms), rms=float(np.sqrt(max(ms, 0))), delta=float(dlt), per_group=per,
                rms_lo=float(np.quantile(rms_b, .025)), rms_hi=float(np.quantile(rms_b, .975)),
                rms_se=float(rms_b.std(ddof=1)), delta_se=float(bs[:, 1].std(ddof=1)))


# ----------------------------------------------------------------------------- q-side identity
def _cov_point(G, lw, g, gw):
    out_plain, out_sn = 0.0, 0.0
    for k, pk in gw.items():
        m = g == k
        w = np.exp(lw[m]); Gk = G[m]
        out_plain += pk * (np.mean(Gk * w) - Gk.mean() * w.mean())
        out_sn += pk * (np.sum(Gk * w) / np.sum(w) - Gk.mean())
    return out_plain, out_sn


def cov_qw(Gq, lw_q, groups=None, group_probs=None, B=500, seed=0):
    """Cov_q(G, w) from synthetic draws only (Kelvin's identity E_p G - E_q G = Cov_q(G, w)).
    Within each group z: plain Cov = mean(Gw) - mean(G)mean(w); SNIS version = sum(Gw)/sum(w) - mean(G).
    Groups are combined with weights p(z) (group_probs, default: frequencies in the q sample).
    Returns both, bootstrap CIs (resampling within groups; the ratio model is held fixed), Kish ESS, E_q w."""
    Gq = np.asarray(Gq, float); lw_q = np.asarray(lw_q, float)
    g = _groups(groups, len(Gq))
    ks = np.unique(g)
    if group_probs is None:
        group_probs = {k: float(np.mean(g == k)) for k in ks}
    plain, sn = _cov_point(Gq, lw_q, g, group_probs)
    rng = np.random.default_rng(seed)
    idx_by = {k: np.flatnonzero(g == k) for k in ks}
    bs = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(v, len(v)) for v in idx_by.values()])
        bs.append(_cov_point(Gq[idx], lw_q[idx], g[idx], group_probs))
    bs = np.asarray(bs)
    w = np.exp(lw_q)
    return dict(cov_plain=plain, cov_snis=sn,
                plain_lo=float(np.quantile(bs[:, 0], .025)), plain_hi=float(np.quantile(bs[:, 0], .975)),
                plain_se=float(bs[:, 0].std(ddof=1)),
                snis_lo=float(np.quantile(bs[:, 1], .025)), snis_hi=float(np.quantile(bs[:, 1], .975)),
                snis_se=float(bs[:, 1].std(ddof=1)),
                ess=kish_ess(w), mean_w=float(w.mean()))


# ----------------------------------------------------------------------------- tempering
def _snis_path(G, logomega, g, gw, alphas_unused=None):
    """Per-group self-normalised mean of G under weights exp(logomega); combined with p(z)."""
    L = 0.0; ess = 0.0
    for k, pk in gw.items():
        m = g == k
        lo = logomega[m]; om = np.exp(lo - lo.max())
        L += pk * np.sum(om * G[m]) / np.sum(om)
        ess += kish_ess(om)
    return L, ess


def temper_q(Gq, lw_q, alphas=np.linspace(0, 1, 11), groups=None, group_probs=None, B=300, seed=0):
    """Kelvin's tempered reweighting from q: p_alpha ∝ q w^alpha, L(alpha) = sum w^alpha G / sum w^alpha
    within each group. Variance at alpha governed by E_q[w^{2 alpha}] (chi^2 at alpha = 1)."""
    Gq = np.asarray(Gq, float); lw_q = np.asarray(lw_q, float)
    g = _groups(groups, len(Gq)); ks = np.unique(g)
    if group_probs is None:
        group_probs = {k: float(np.mean(g == k)) for k in ks}
    rng = np.random.default_rng(seed)
    idx_by = {k: np.flatnonzero(g == k) for k in ks}
    L = np.array([_snis_path(Gq, a * lw_q, g, group_probs)[0] for a in alphas])
    ess = np.array([_snis_path(Gq, a * lw_q, g, group_probs)[1] for a in alphas])
    bs = []
    for _ in range(B):
        idx = np.concatenate([rng.choice(v, len(v)) for v in idx_by.values()])
        bs.append([_snis_path(Gq[idx], a * lw_q[idx], g[idx], group_probs)[0] for a in alphas])
    bs = np.asarray(bs)
    d = bs[:, 0] - bs[:, -1]
    return dict(alphas=np.asarray(alphas), L=L, lo=np.quantile(bs, .025, 0), hi=np.quantile(bs, .975, 0),
                se=bs.std(0, ddof=1), ess=ess, gap=float(L[0] - L[-1]), gap_se=float(d.std(ddof=1)),
                gap_lo=float(np.quantile(d, .025)), gap_hi=float(np.quantile(d, .975)))


def _bridge_point(G, lw, isp, g, gw, alphas):
    """Pooled bridge per group. Pool = pi_p p + pi_q q, so p_alpha / pool ∝ w^alpha / (pi_p w + pi_q)."""
    Ls = np.zeros(len(alphas)); dLs = np.zeros(len(alphas)); ess = np.zeros(len(alphas))
    for k, pk in gw.items():
        m = g == k
        lwk, Gk = lw[m], G[m]
        pp = isp[m].mean()
        logmix = np.logaddexp(np.log(pp) + lwk, np.log(1 - pp))
        for i, a in enumerate(alphas):
            lo = a * lwk - logmix
            om = np.exp(lo - lo.max()); om /= om.sum()
            Lk = np.sum(om * Gk)
            Ls[i] += pk * Lk
            dLs[i] += pk * np.sum(om * (Gk - Lk) * (lwk - np.sum(om * lwk)))
            ess[i] += 1.0 / np.sum(om * om)
    return Ls, dLs, ess


def temper_bridge(Gp, lw_p, Gq, lw_q, alphas=np.linspace(0, 1, 11), groups_p=None, groups_q=None,
                  group_probs=None, B=300, seed=0):
    """Bridge tempering path (METHOD §3.4), normalised per group so p(z) (hence the labels) stays fixed.
    Weights are bounded for every alpha, so variance is governed by the Renyi affinity, not chi^2.
    Returns L(alpha), CIs (bootstrap over draws within groups, ratio held fixed), dL/dalpha, its trapezoid
    integral, and Kish ESS per alpha."""
    G = np.concatenate([Gp, Gq]).astype(float); lw = np.concatenate([lw_p, lw_q]).astype(float)
    isp = np.concatenate([np.ones(len(Gp)), np.zeros(len(Gq))])
    g = np.concatenate([_groups(groups_p, len(Gp)), _groups(groups_q, len(Gq))])
    ks = np.unique(g)
    if group_probs is None:
        gp = g[isp == 1]
        group_probs = {k: float(np.mean(gp == k)) for k in ks}
    alphas = np.asarray(alphas, float)
    L, dL, ess = _bridge_point(G, lw, isp, g, group_probs, alphas)
    rng = np.random.default_rng(seed)
    cells = [(k, s) for k in ks for s in (0, 1)]
    idx_by = {c: np.flatnonzero((g == c[0]) & (isp == c[1])) for c in cells}
    bs, bint = [], []
    for _ in range(B):
        idx = np.concatenate([rng.choice(v, len(v)) for v in idx_by.values() if len(v)])
        Lb, dLb, _ = _bridge_point(G[idx], lw[idx], isp[idx], g[idx], group_probs, alphas)
        bs.append(Lb); bint.append(np.trapezoid(dLb, alphas))
    bs = np.asarray(bs); d = bs[:, 0] - bs[:, -1]
    integ = float(np.trapezoid(dL, alphas))
    return dict(alphas=alphas, L=L, lo=np.quantile(bs, .025, 0), hi=np.quantile(bs, .975, 0),
                se=bs.std(0, ddof=1), dL=dL, integral=integ, integral_se=float(np.std(bint, ddof=1)),
                gap=float(L[0] - L[-1]), gap_se=float(d.std(ddof=1)),
                gap_lo=float(np.quantile(d, .025)), gap_hi=float(np.quantile(d, .975)), ess=ess)


# ----------------------------------------------------------------------------- localisation
def region_split(Gq, inAq, Gp, inAp):
    """Exact Delta_A = E_q[G 1_A] - E_p[G 1_A] = mass part + within part (METHOD §3.5)."""
    Gq, Gp = np.asarray(Gq, float), np.asarray(Gp, float)
    inAq, inAp = np.asarray(inAq, bool), np.asarray(inAp, bool)
    Pq, Pp = inAq.mean(), inAp.mean()
    EpA = Gp[inAp].mean() if inAp.any() else 0.0
    EqA = Gq[inAq].mean() if inAq.any() else 0.0
    return dict(delta_A=float(np.mean(Gq * inAq) - np.mean(Gp * inAp)), mass=float((Pq - Pp) * EpA),
                within=float(Pq * (EqA - EpA)), Pq=float(Pq), Pp=float(Pp))
