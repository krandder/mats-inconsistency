"""E26 estimator checks on synthetic functions with planted components (no network involved; separate seed).
usage: python e26_check.py DATA.npz   (DATA built by: python e26_data.py DATA.npz SEED N)
Checks:
  1. a planted function D = a y* + b sum_s e_s + q sum_s y* e_s + h sum_s e_s u_s0 + m sum_{s<t} e_s e_t + p sum_{s!=t} u_s1 e_t
     (each term an ANOVA-pure interaction of independent factors) -> planted context, own-sufficient, own-pattern,
     tangled-sufficient and tangled-pattern variances, recovered by e26_summ's estimators;
  2. a purely additive function with large main effects -> the 2x2 tangledness estimate is exactly 0;
  3. KL-exact pattern cost on Gaussians with planted direction-dependent mean shifts (expected eps^2 / 8 after the
     16/15 correction), through e26_run.set_stats on the same grid as the real runs.
"""
import sys
import numpy as np
import e26_data as ED
import e26_summ as SM
import e26_run as RU

EV = float(np.mean(ED.W.VLEV))


def factors(z, k):
    res, err = z['res'][:, k], z['err'][:, k]
    rho = np.linalg.norm(res, axis=2, keepdims=True)
    return z['y'][:, ED.QL], err, res / rho


def planted(z, k, a, b, q, h, m, p):
    ys, e, u = factors(z, k)
    D = a * ys + b * e.sum(1) + q * (ys[:, None] * e).sum(1) + h * (e * u[:, :, 0]).sum(1)
    D = D + m * (e[:, 0] * e[:, 1] + e[:, 0] * e[:, 2] + e[:, 1] * e[:, 2])
    D = D + p * sum(u[:, s, 1] * e[:, t] for s in range(3) for t in range(3) if s != t)
    return D


def main():
    z = np.load(sys.argv[1]); n = len(z['y'])
    a, b, q, h, m, p = 0.3, 0.4, 0.2, 0.5, 0.25, 0.35
    scale = 1 + np.abs(ED.DELTAS)                                      # a delta-dependent scale, as in real log-odds
    Lall = np.stack([planted(z, k, a, b, q, h, m, p) for k in range(ED.NVAR)], 1)[:, :, None] * scale[None, None]
    tu = Lall[:, ED.PAT].var(1, ddof=1); tsp = Lall[:, ED.SP].var(1, ddof=1)
    zeros = np.zeros(n)
    C = SM.columns(Lall[:, :12], tu, tsp, zeros, zeros + 1, zeros, zeros + 1, np.zeros_like(Lall[:, :12]), SM.DSEL_ALL)
    c = {k: float(v[0]) for k, v in SM.comps_from_means(np.ones((1, n)) / n @ C, 6, n).items()}
    s2 = float((scale ** 2).sum())
    plant = dict(Vc=a ** 2, own_suff=3 * b ** 2 * EV + 3 * q ** 2 * EV, Vu_own=3 * h ** 2 * EV / 4,
                 X_suff=3 * m ** 2 * EV ** 2, Vu_cross=6 * p ** 2 * EV / 4)
    plant = {k: v * s2 for k, v in plant.items()}
    plant['Vtot'] = sum(plant.values()); plant['VX'] = plant['X_suff'] + plant['Vu_cross']; plant['Tu'] = plant['Vu_own'] + plant['Vu_cross']
    print('check 1 (planted components; recovered / planted):')
    for k in ('Vtot', 'Vc', 'own_suff', 'Vu_own', 'X_suff', 'Vu_cross', 'VX', 'Tu'):
        print('  %-9s planted %.4f  recovered %.4f  ratio %.3f' % (k, plant[k], c[k], c[k] / plant[k]))
    print('  VX from 2x2 cuts %.4f (planted pairwise only, so it should equal VX)' % c['VX2'])
    print('  Tsp %.4f (planted terms use u_s0, u_s1, which change under sign/assignment redraws)' % c['Tsp'])
    # check 2: additive with big main effects
    ys, e, u = factors(z, 0)
    Ladd = np.stack([(lambda f: 2.0 * f[0] + 3.0 * np.tanh(f[1]).sum(1) + 1.5 * (f[1] ** 2).sum(1))(factors(z, k)) for k in range(ED.NVAR)], 1)[:, :, None] * scale[None, None]
    C2 = SM.columns(Ladd[:, :12], Ladd[:, ED.PAT].var(1, ddof=1), Ladd[:, ED.SP].var(1, ddof=1), zeros, zeros + 1, zeros, zeros + 1,
                    np.zeros_like(Ladd[:, :12]), SM.DSEL_ALL)
    c2 = {k: float(v[0]) for k, v in SM.comps_from_means(np.ones((1, n)) / n @ C2, 6, n).items()}
    print('check 2 (additive, large main effects): Vtot %.3f  VX(pick-freeze) %.4f  VX(2x2) %.2e  Tu %.2e' % (c2['Vtot'], c2['VX'], c2['VX2'], c2['Tu']))
    # check 3: KL-exact pattern cost with planted Gaussian shifts
    eps = 0.4; nb = min(n, 2000)
    g = z['bmean'][:nb, None] + RU.GRID[None]
    mm, vv = z['bmean'][:nb], z['bvar'][:nb]
    lpB = -0.5 * np.log(2 * np.pi * vv)[:, None] - 0.5 * (g - mm[:, None]) ** 2 / vv[:, None]
    hds = []
    for k in ED.PAT:
        _, _, u = factors(z, k)
        mu = (mm + eps * u[:nb, 0, 0] * np.sqrt(vv))[:, None]
        hds.append((mu, 0.5 * np.log(vv)[:, None], np.zeros((nb, 1))))
    kl, pc, mass = RU.set_stats(hds, lpB, g)
    print('check 3 (pattern cost): planted %.5f  recovered (x16/15) %.5f  mean KL %.5f (planted %.5f)  min mass %.6f'
          % (eps ** 2 / 8, pc.mean() * 16 / 15, kl.mean(), eps ** 2 / 8, mass.min()))


if __name__ == '__main__':
    main()
