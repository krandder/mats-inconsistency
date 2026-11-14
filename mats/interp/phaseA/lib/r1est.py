"""Round-1 estimators on (cells x M) arrays, the per-token-type q wrapper, and Gaussian subspace conditioning.

Paired design: draw j of cell i is a full episode x'_ij ~ p(x | c_i); the real draw is the network on x'_ij and the
synthetic draw replaces the site's tokens by q(. | phi(x'_ij)) with the rest of x'_ij real. So
Delta(c) = E_{u|c}[G_q - G_p] and Delta_rms^2 = E_c Delta(c)^2 is estimated by the product of two independent halves."""
import math
import numpy as np
from scipy.special import expit

import sys
sys.path.insert(0, '/home/agent/workspaces/mats/work/cm-interp')
from cmlib.qfam import LinearGaussian            # noqa: E402
from cmlib.stats import boot_mean_t, hill_tail_index, kish_ess, psis_khat, auc, auc_se  # noqa: E402
from cmlib.ratio import fit_ratio, ratio_diagnostics  # noqa: E402
from . import tp                                 # noqa: E402


# ----------------------------------------------------------------------------- direct
def direct(Gq, Gp, B=1000, seed=0):
    """Gq, Gp: (n, M) paired. Returns Delta with studentised bootstrap CI over cells, Delta_rms (two halves) with
    percentile CI, signed per-cell gaps, Hill index of per-cell differences."""
    d = Gq - Gp
    dc = d.mean(1)
    m, se, lo, hi = boot_mean_t(dc, B=B, rng=np.random.default_rng(seed))
    h = d.shape[1] // 2
    prod = d[:, :h].mean(1) * d[:, h:].mean(1)
    rng = np.random.default_rng(seed + 1)
    bs = prod[rng.integers(0, len(prod), (B, len(prod)))].mean(1)
    ms = prod.mean()
    return dict(delta=m, se=se, lo=lo, hi=hi, rms=math.sqrt(max(ms, 0)), ms=float(ms), ms_se=float(prod.std(ddof=1) / math.sqrt(len(prod))),
                rms_lo=math.sqrt(max(np.quantile(bs, .025), 0)), rms_hi=math.sqrt(max(np.quantile(bs, .975), 0)),
                Lq=float(Gq.mean()), Lp=float(Gp.mean()), hill=hill_tail_index(dc - dc.mean()))


def rms_diff(dq1, dq2, B=500, seed=0):
    """Joint bootstrap of rms(q1) - rms(q2) where dq = (n, M) paired differences G_q - G_p."""
    h = dq1.shape[1] // 2
    p1 = dq1[:, :h].mean(1) * dq1[:, h:].mean(1); p2 = dq2[:, :h].mean(1) * dq2[:, h:].mean(1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(p1), (B, len(p1)))
    b = np.sqrt(np.maximum(p1[idx].mean(1), 0)) - np.sqrt(np.maximum(p2[idx].mean(1), 0))
    return float(math.sqrt(max(p1.mean(), 0)) - math.sqrt(max(p2.mean(), 0))), float(b.std(ddof=1))


# ----------------------------------------------------------------------------- ratio-based (output space)
def crossfit_cells(Fp, Fq, cell, n_cells, reps=2, seed=0, family='mlp'):
    """Out-of-fold log w for every real (Fp) and synthetic (Fq) draw; folds split by cell (2 folds).
    Fp, Fq: (N, k) classifier inputs (output vector + c features). Returns list over reps of (lwp, lwq, diag)."""
    out = []
    for r in range(reps):
        rng = np.random.default_rng(seed + 100 * r)
        fold_c = rng.integers(0, 2, n_cells); fold = fold_c[cell]
        lwp, lwq = np.empty(len(Fp)), np.empty(len(Fq)); dg = []; models = []
        for k in range(2):
            tr = fold != k
            rm, d = fit_ratio(Fp[tr], Fq[tr], family=family, seed=seed + 100 * r + k)
            lwp[~tr] = rm.log_w(Fp[~tr]); lwq[~tr] = rm.log_w(Fq[~tr]); dg.append(d); models.append((rm, ~tr))
        diag = ratio_diagnostics(lwp, lwq)
        diag['auc_se'] = auc_se(diag['auc'], len(lwp), len(lwq))
        diag['ece_in'] = float(np.mean([x['ece'] for x in dg]))
        diag['train_minus_test_bce'] = float(np.mean([x['train_minus_test_bce'] for x in dg]))
        out.append((lwp, lwq, diag, models))
    return out


def _bridge_cells(Gp, lwp, Gq, lwq, alphas):
    G = np.concatenate([Gp, Gq], 1); lw = np.concatenate([lwp, lwq], 1)
    lc = -np.logaddexp(0, -lw); l1c = -np.logaddexp(0, lw)                     # log c, log (1 - c), c = w/(1+w)
    Lc, dLc, ess = [], [], []
    for a in alphas:
        lo = a * lc + (1 - a) * l1c
        om = np.exp(lo - lo.max(1, keepdims=True)); om /= om.sum(1, keepdims=True)
        La = (om * G).sum(1)
        Lc.append(La)
        dLc.append((om * (G - La[:, None]) * (lw - (om * lw).sum(1, keepdims=True))).sum(1))
        ess.append(1 / (om * om).sum(1))
    return np.stack(Lc, 1), np.stack(dLc, 1), np.stack(ess, 1)               # (n, A)


def ratio_block(Gp, Gq, lwp, lwq, alphas=np.linspace(0, 1, 11), B=500, seed=0):
    """Kelvin's identity and tempering from one set of out-of-fold log w (n, M arrays).
    cov_plain = E_c [mean_j G_q w - mean_j G_q mean_j w]  (Cov_q(G, w) per cell; = E_p G - E_q G if w exact)
    ident     = E_q[G w] (should equal E_p G);  cov_snis = E_c[sum G w / sum w - mean G] per cell
    bridge    = per-cell normalised pooled path L(alpha), dL/dalpha, integral, Kish ESS."""
    wq = np.exp(lwq)
    cov_c = (Gq * wq).mean(1) - Gq.mean(1) * wq.mean(1)
    sn_c = (Gq * wq).sum(1) / wq.sum(1) - Gq.mean(1)
    id_c = (Gq * wq).mean(1)
    Lc, dLc, ess = _bridge_cells(Gp, lwp, Gq, lwq, alphas)
    rng = np.random.default_rng(seed)
    n = len(Gp)
    idx = rng.integers(0, n, (B, n))
    bsL = np.stack([Lc[i].mean(0) for i in idx])
    gap_b = bsL[:, 0] - bsL[:, -1]
    integ_c = np.trapezoid(dLc, alphas, axis=1)
    L = Lc.mean(0)
    flat_w = wq.ravel()
    Lk = []
    for a in alphas:
        lo = a * lwq; om = np.exp(lo - lo.max(1, keepdims=True)); om /= om.sum(1, keepdims=True)
        Lk.append((om * Gq).sum(1).mean())
    return dict(alphas=list(alphas), L=L.tolist(), L_kelvin_q=list(map(float, Lk)), L_lo=np.quantile(bsL, .025, 0).tolist(), L_hi=np.quantile(bsL, .975, 0).tolist(),
                L_se=bsL.std(0, ddof=1).tolist(),
                gap=float(L[0] - L[-1]), gap_se=float(gap_b.std(ddof=1)),
                dL=dLc.mean(0).tolist(), integral=float(integ_c.mean()), integral_se=float(integ_c.std(ddof=1) / math.sqrt(n)),
                cov_plain=float(cov_c.mean()), cov_plain_se=float(cov_c.std(ddof=1) / math.sqrt(n)),
                cov_snis=float(sn_c.mean()), cov_snis_se=float(sn_c.std(ddof=1) / math.sqrt(n)),
                ident=float(id_c.mean()), ident_se=float(id_c.std(ddof=1) / math.sqrt(n)),
                ess_q_frac=kish_ess(flat_w) / len(flat_w), ess_q_cell=float((wq.sum(1) ** 2 / (wq ** 2).sum(1)).mean()),
                bridge_ess_min_pooled=float(ess.sum(0).min()), bridge_ess_cell_mean=ess.mean(0).tolist(),
                Eq_w=float(wq.mean()), Ep_invw=float(np.exp(-lwp).mean()), khat_q=psis_khat(lwq.ravel()))


def combine_reps(blocks):
    """Mean over classifier refits; total SE = sqrt(boot/cell SE^2 + sd across refits^2)."""
    out = {}
    for k in ('gap', 'integral', 'cov_plain', 'cov_snis', 'ident', 'ess_q_frac', 'ess_q_cell', 'Eq_w', 'Ep_invw', 'khat_q',
              'bridge_ess_min_pooled'):
        v = np.array([b[k] for b in blocks])
        out[k] = float(v.mean()); out[k + '_refit_sd'] = float(v.std(ddof=1)) if len(v) > 1 else 0.0
        if k + '_se' in blocks[0]:
            out[k + '_se'] = float(math.sqrt(np.mean([b[k + '_se'] ** 2 for b in blocks]) + out[k + '_refit_sd'] ** 2))
    Ls = np.array([b['L'] for b in blocks])
    out['L'] = Ls.mean(0).tolist()
    out['L_se'] = np.sqrt(np.mean(np.array([b['L_se'] for b in blocks]) ** 2, 0) + (Ls.std(0, ddof=1) ** 2 if len(Ls) > 1 else 0)).tolist()
    out['dL'] = np.array([b['dL'] for b in blocks]).mean(0).tolist()
    out['alphas'] = blocks[0]['alphas']
    return out


# ----------------------------------------------------------------------------- q per token type
class QTok:
    """q(h_t | phi_t) = N(phi A + b, Sigma[_g]) for one token type at one cut, in quotient coordinates."""

    def __init__(self, ttype, fset, extra=(), norm=False, hetero=False, lam=None):
        self.ttype, self.fset, self.extra, self.norm, self.hetero = ttype, fset, tuple(extra), norm, hetero
        self.lam = lam

    def phi(self, d, cs):
        return tp.token_features(d, cs, self.ttype, self.fset, self.extra)

    def groups(self, cs, ntok):
        if not self.hetero:
            return None
        g = np.digitize(cs['bcv'], self.edges)
        return np.repeat(g, ntok)

    def fit(self, d, cs, H):
        """H: (n, ntok, dmodel) real states of this type (already in quotient coordinates)."""
        F = self.phi(d, cs)
        n, nt, k = F.shape
        if self.hetero:
            self.edges = np.quantile(cs['bcv'], [.2, .4, .6, .8])
        kw = dict(cov_type='full')
        if self.lam is not None:
            kw['lambdas'] = (self.lam,)
        self.m = LinearGaussian(**kw)
        Phi = F.reshape(n * nt, k) if k else np.zeros((n * nt, 1))
        self.k = k
        self.m.fit(H.reshape(n * nt, -1).astype(np.float64), Phi, cov_groups=self.groups(cs, nt),
                   cv_groups=np.repeat(np.arange(n), nt))
        return self

    def _phi_flat(self, d, cs):
        F = self.phi(d, cs)
        n, nt, k = F.shape
        return (F.reshape(n * nt, k) if k else np.zeros((n * nt, 1))), n, nt

    def mean_cov(self, d, cs):
        Phi, n, nt = self._phi_flat(d, cs)
        mu, covs, g = self.m.mean_cov(Phi, cov_groups=self.groups(cs, nt))
        return mu.reshape(n, nt, -1), covs, (g.reshape(n, nt) if g is not None else np.zeros((n, nt), int))

    def sample(self, d, cs, rng):
        Phi, n, nt = self._phi_flat(d, cs)
        Hs = self.m.sample(Phi, self.groups(cs, nt), rng=rng).reshape(n, nt, -1)
        return tp.to_q(Hs, self.norm)

    def logpdf_proj(self, d, cs, H, U):
        """log density of the real states projected on U (d x r) under q's projected marginal; (n*nt,)"""
        mu, covs, g = self.mean_cov(d, cs)
        z = (H - mu).reshape(-1, H.shape[-1]) @ U
        g = g.ravel(); out = np.empty(len(z))
        for key, C in covs.items():
            m = g == key
            Cu = U.T @ C @ U
            Lc = np.linalg.cholesky(Cu + 1e-10 * np.eye(len(Cu)))
            w = np.linalg.solve(Lc, z[m].T)
            out[m] = -0.5 * (w * w).sum(0) - np.log(np.diag(Lc)).sum() - 0.5 * len(Cu) * math.log(2 * math.pi)
        return out

    def n_struct_groups(self):
        base = {'Rmarg': 0, 'Rin': 4, 'q0': 10, 'q0noID': 8, 'q0U': 8}[self.fset]
        if self.ttype == 'QT':
            base = {'Rmarg': 0, 'Rin': 1, 'q0': 7, 'q0noID': 6, 'q0U': 5}[self.fset]
        return base + len(self.extra) + (1 if self.hetero else 0)


def cond_gauss_sample(H, mu, C, A, rng):
    """H, mu: (N, d) in some coordinates; C (d,d). Resample coordinates A given the rest (knock-out)."""
    d = H.shape[1]; A = np.asarray(A); Bc = np.setdiff1d(np.arange(d), A)
    K = C[np.ix_(A, Bc)] @ np.linalg.inv(C[np.ix_(Bc, Bc)])
    cm = mu[:, A] + (H[:, Bc] - mu[:, Bc]) @ K.T
    Sg = C[np.ix_(A, A)] - K @ C[np.ix_(Bc, A)]
    Sg = 0.5 * (Sg + Sg.T) + 1e-10 * np.eye(len(A))
    out = H.copy()
    out[:, A] = cm + rng.standard_normal((len(H), len(A))) @ np.linalg.cholesky(Sg).T
    return out


def subspace_knock(q, d, cs, H, U, rng, mode='knockout'):
    """Real states H (n, nt, dm) of q's type. Knock-out: the coordinates along U (orthonormal, dm x r) are drawn from
    q(. | coordinates along U-perp, phi); knock-in: U kept real, U-perp drawn from q given U."""
    mu, covs, g = q.mean_cov(d, cs)
    dm = H.shape[-1]
    Q_, _ = np.linalg.qr(np.concatenate([U, np.random.default_rng(0).standard_normal((dm, dm))], 1))
    Bm = Q_[:, :dm]; Bm[:, :U.shape[1]] = U                                     # basis [U, rest]
    Bm, _ = np.linalg.qr(Bm)
    r = U.shape[1]
    A = np.arange(r) if mode == 'knockout' else np.arange(r, dm)
    Hf = H.reshape(-1, dm); muf = mu.reshape(-1, dm); gf = g.ravel()
    out = Hf.copy()
    for key, C in covs.items():
        m = gf == key
        Z = Hf[m] @ Bm; Zm = muf[m] @ Bm; Cz = Bm.T @ C @ Bm
        Zs = cond_gauss_sample(Z, Zm, Cz, A, rng)
        out[m] = Zs @ Bm.T
    return tp.to_q(out.reshape(H.shape), q.norm)
