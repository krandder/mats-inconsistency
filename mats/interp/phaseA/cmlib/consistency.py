"""Cross-layer consistency (METHOD §6).

tier i  (identity): Delta_l(q_l) = Delta_{l+1}(F#q_l) exactly, checked sample by sample (plumbing only).
tier ii (Kelvin's condition): q~_{l+1} = Proj(F#q_l) is q_{l+1}'s family refitted on pushed samples only;
        compare with q^fit_{l+1} fitted on real h_{l+1}:
          Gamma_proj = Delta(q~) - Delta(F#q_l)          what the family loses summarising the push-forward
          Gamma_fit  = rms(q~) - rms(q^fit)              structure the real-fitted description has and the derived one lacks
          AUC(q~ vs q^fit) by a fresh classifier (both synthetic: unlimited n, both have densities)
        pass: Gamma_fit <= max(2 SE, tol) and AUC <= auc_max (METHOD rule). Also reported: the one-sided Gamma
        rule alone, and a two-sided rule (|Gamma_fit| and |Delta(q~) - Delta(q^fit)| within 2 SE, plus AUC).
        Validation (iii) has a case where q~ is wrong but has a *smaller* rms gap than q^fit, so the
        one-sided Gamma rule alone misses it.
Gaps are per latent group z (discrete; many draws per group; group_rms). For continuous cells c (the tiny PFN),
use estimators.cell_rms with the exact conditional sampler instead.
"""
import numpy as np

from .estimators import group_rms
from .ratio import fit_ratio


def identity_check(H_l, z, F, G_next, G_l):
    """max |G_l(h) - G_next(F(h))| over draws h: zero to float error if F and G_l are the same computation."""
    return float(np.max(np.abs(G_l(H_l, z) - G_next(F(H_l, z), z))))


def _onehot(z, keys):
    return (np.asarray(z)[:, None] == np.asarray(keys)[None, :]).astype(float)


def consistency_check(sample_ql, F, fit_family, sample_real_next, G_next, z_pool, n_fit=100000, n_eval=100000,
                      tol=0.0, auc_max=0.55, n_auc=40000, seed=0, clf_family='quad', B=300, refits=8):
    """sample_ql(z, rng) -> h_l draws from the layer-l description given latents z;
    F(h, z) -> actual layer-(l+1) computation; fit_family() -> unfitted q family with fit(H, z) / sample(z, rng);
    sample_real_next(z, rng) -> real h_{l+1} given z; G_next(h_{l+1}, z) -> per-draw downstream quantity;
    z_pool: latent values to draw from (its empirical distribution is p(z)).
    refits: half-sampling refits of both families; their spread (variance from fitting q~ and q^fit on finite
    data) is added in quadrature to the evaluation-bootstrap SEs. Without it the null false-fail rate was 2/6."""
    rng = np.random.default_rng(seed)
    keys = np.unique(z_pool)
    zf = rng.choice(z_pool, n_fit); Hpush_f = F(sample_ql(zf, rng), zf)
    q_til = fit_family().fit(Hpush_f, zf)
    zf2 = rng.choice(z_pool, n_fit); Hreal_f = sample_real_next(zf2, rng)
    q_fit = fit_family().fit(Hreal_f, zf2)

    ze = rng.choice(z_pool, n_eval)
    Gp = G_next(sample_real_next(ze, rng), ze)
    Gpush = G_next(F(sample_ql(ze, rng), ze), ze)
    Gtil = G_next(q_til.sample(ze, rng=rng), ze)
    Gfit = G_next(q_fit.sample(ze, rng=rng), ze)
    r_push = group_rms(Gpush, ze, Gp, ze, B=50, seed=seed)
    r_til, r_fit = group_rms(Gtil, ze, Gp, ze, B=50, seed=seed), group_rms(Gfit, ze, Gp, ze, B=50, seed=seed)
    # joint bootstrap of Gamma_fit (shared real draws)
    bs = []
    idx_by = {k: np.flatnonzero(ze == k) for k in keys}
    for _ in range(B):
        idx = np.concatenate([rng.choice(v, len(v)) for v in idx_by.values()])
        ms_t = ms_f = 0.0
        for k in keys:
            m = ze[idx] == k; pk = np.mean(ze == k)
            gp, gt, gf = Gp[idx][m], Gtil[idx][m], Gfit[idx][m]
            h = len(gp) // 2
            ms_t += pk * (gt[:h].mean() - gp[:h].mean()) * (gt[h:].mean() - gp[h:].mean())
            ms_f += pk * (gf[:h].mean() - gp[:h].mean()) * (gf[h:].mean() - gp[h:].mean())
        bs.append(np.sqrt(max(ms_t, 0)) - np.sqrt(max(ms_f, 0)))
    gamma_fit = r_til['rms'] - r_fit['rms']
    se_gamma = float(np.std(bs, ddof=1))
    # signed paired gap difference Delta(q~) - Delta(q^fit) (the real draws cancel)
    gd = []
    for k in keys:
        m = ze == k
        gd.append((np.mean(m), Gtil[m].mean() - Gfit[m].mean(), (np.var(Gtil[m]) + np.var(Gfit[m])) / m.sum()))
    gamma_delta = float(sum(p * d for p, d, _ in gd))
    gamma_delta_se = float(np.sqrt(sum(p * p * v for p, _, v in gd)))
    # fit variance by half-sampling refits of both families (evaluated against the same real draws)
    fit_sd_rms = fit_sd_delta = 0.0
    if refits:
        gr, dr = [], []
        for _ in range(refits):
            i1 = rng.permutation(n_fit)[:n_fit // 2]; i2 = rng.permutation(n_fit)[:n_fit // 2]
            qt = fit_family().fit(Hpush_f[i1], zf[i1]); qf = fit_family().fit(Hreal_f[i2], zf2[i2])
            gt, gf = G_next(qt.sample(ze, rng=rng), ze), G_next(qf.sample(ze, rng=rng), ze)
            gr.append(group_rms(gt, ze, Gp, ze, B=0, seed=seed)['rms'] - group_rms(gf, ze, Gp, ze, B=0, seed=seed)['rms'])
            dr.append(sum(np.mean(ze == k) * (gt[ze == k].mean() - gf[ze == k].mean()) for k in keys))
        fit_sd_rms, fit_sd_delta = float(np.std(gr, ddof=1)), float(np.std(dr, ddof=1))
        se_gamma = float(np.hypot(se_gamma, fit_sd_rms)); gamma_delta_se = float(np.hypot(gamma_delta_se, fit_sd_delta))

    za = rng.choice(z_pool, n_auc)
    _, dg = fit_ratio(q_til.sample(za, rng=rng), q_fit.sample(za, rng=rng), _onehot(za, keys), _onehot(za, keys),
                      family=clf_family, seed=seed)
    return dict(delta_push=r_push['delta'], delta_tilde=r_til['delta'], delta_fit=r_fit['delta'],
                rms_push=r_push['rms'], rms_tilde=r_til['rms'], rms_fit=r_fit['rms'],
                gamma_proj=r_til['delta'] - r_push['delta'], gamma_fit=float(gamma_fit), gamma_fit_se=se_gamma,
                gamma_delta=gamma_delta, gamma_delta_se=gamma_delta_se, auc=dg['auc'],
                fit_sd_rms=fit_sd_rms, fit_sd_delta=fit_sd_delta,
                passed=bool(gamma_fit <= max(2 * se_gamma, tol) and dg['auc'] <= auc_max),
                passed_gamma_only=bool(gamma_fit <= max(2 * se_gamma, tol)),
                passed_two_sided=bool(abs(gamma_fit) <= max(2 * se_gamma, tol)
                                      and abs(gamma_delta) <= max(2 * gamma_delta_se, tol)
                                      and dg['auc'] <= auc_max),
                q_tilde=q_til, q_fit=q_fit)
