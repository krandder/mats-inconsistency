"""Post hoc re-analysis after the GPT-6 review (findings 1, 8, 11) -> results/posthoc_review_fixes.json.
(1) Gram chain without selecting on replica 0's final radius: unconditional annealed E delta vs the L0 fixed point
    averaged over the same chains' radii; the direction order D_u (all chains) vs L0 evaluated with the same
    ratio-of-means averaging over the stationary radius law; bins of the shared pre-window radius Q0 (legitimate:
    independent of the window noise); unconditional 4-replica PR_u.
(8) Flip response restricted to paths that had a unit within one noise SD (the registered population).
"""
import json, glob, math, sys
import numpy as np

D = '/home/agent/workspaces/mats/work/stationary-numerics'


def kappa(c):
    c = min(max(c, -1.0), 1.0)
    return (math.sqrt(max(1 - c * c, 0.0)) + (math.pi - math.acos(c)) * c) / math.pi


def delta_star(Q):
    rho = 1 - 2.23 * Q ** (-2 / 3) if Q > 3 else 0.0
    for _ in range(200000):
        r2 = Q * kappa(rho) / (Q + 1)
        if abs(r2 - rho) < 1e-13:
            break
        rho = r2
    return 1 - rho


def single_sum(Q, J):
    """L0: sum over ages of delta_t after one unshared layer (shared noise afterwards), t = 0..J-1."""
    rho = Q / (Q + 1)
    s = 0.0
    for t in range(J):
        s += 1 - rho
        rho = (Q * kappa(rho) + 1) / (Q + 1)
    return s


out = {'gram': {}, 'flip': {}}
for n in (8, 16, 32, 64, 128, 256, 512, 1024):
    w = json.load(open(f'{D}/runs/gram_window_n{n}.json'))
    ch = w['chains']
    Qf = np.array(ch['Qf']); Q0 = np.array(ch['Q0'])
    d01u = np.array(ch['d01']['u']); d02u = np.array(ch['d02']['u'])
    delta = d01u / 2  # |u1-u2|^2 = 2(1 - rho)
    # L0 on a Q grid, interpolated
    qg = np.exp(np.linspace(np.log(max(Qf.min(), 0.5)), np.log(Qf.max()), 60))
    ds = np.array([delta_star(q) for q in qg])
    J = 4 * n
    ss = np.array([single_sum(q, J) for q in qg])
    dsQ = np.interp(np.log(Qf), np.log(qg), ds)
    ssQ = np.interp(np.log(Qf), np.log(qg), ss)
    r = dict(E_delta_exact=float(delta.mean()), E_delta_L0=float(dsQ.mean()), ratio=float(delta.mean() / dsQ.mean()),
             Du_exact=float(J * d02u.mean() / d01u.mean()), Du_L0_same_averaging=float(ssQ.mean() / dsQ.mean()))
    r['Du_ratio'] = r['Du_exact'] / r['Du_L0_same_averaging']
    # bins of the shared pre-window radius
    bins = []
    for lo, hi in ((0, 0.25), (0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 1e9)):
        m = (Q0 / n >= lo) & (Q0 / n < hi)
        if m.sum() >= 50:
            bins.append(dict(lo=lo, hi=hi, count=int(m.sum()), E_delta=float(delta[m].mean()), E_delta_L0=float(dsQ[m].mean()),
                             Du=float(J * d02u[m].mean() / d01u[m].mean())))
    r['by_Q0_bin'] = bins
    out['gram'][str(n)] = r
    print(n, 'E delta exact/L0 %.3f   D_u exact %.2f vs L0(same averaging) %.2f (ratio %.2f)' % (r['ratio'], r['Du_exact'], r['Du_L0_same_averaging'], r['Du_ratio']))
pr = {}
for n in (16, 32, 64, 128, 256, 512):
    p = json.load(open(f'{D}/runs/gram_pr_n{n}.json'))
    H = sum(1.0 / k for k in range(1, n + 1))
    harm = (H - 1) ** 2 / (math.pi ** 2 / 6 - 1)
    pr[str(n)] = dict(PR_u=p['PR_u'], PR_u_se=p['PR_u_se'], harmonic=harm, ratio=p['PR_u'] / harm)
    print(n, 'unconditional annealed PR_u %.2f +- %.2f, harmonic %.2f, ratio %.2f' % (p['PR_u'], p['PR_u_se'], harm, p['PR_u'] / harm))
out['pr_unconditional'] = pr
for n in (8, 16, 32, 64, 128, 256):
    rs = [json.load(open(f)) for f in sorted(glob.glob(f'{D}/runs/D_n{n}_s*.json'))]
    cas, life, elig, marg = [], [], [], []
    per_net = []
    for r in rs:
        e = np.array(r['flip_nd']['had_candidate'], bool)
        c = np.array(r['flip_nd']['cascade']); L = np.array(r['flip_nd']['lifetime_r2'])
        elig.append(e.mean())
        cas.append(c[e]); life.append(L[e] / n)
        per_net.append(c[e].mean() / n if e.any() else np.nan)
        marg.append(np.array(r['flip_nd']['margin_sd'])[~e])
    allm = np.concatenate(marg) if marg else np.array([])
    life = np.concatenate(life)
    out['flip'][str(n)] = dict(eligible_fraction=float(np.mean(elig)), cascade_over_n_eligible_median_over_nets=float(np.nanmedian(per_net)),
                               lifetime100x_over_n_eligible_quartiles=[float(np.quantile(life, q)) for q in (0.25, 0.5, 0.75)],
                               fallback_margin_max=float(allm.max()) if allm.size else None)
    print(n, 'flip: eligible %.3f, cascade/n (eligible) %.3f, lifetime/n quartiles %s' % (
        np.mean(elig), np.nanmedian(per_net), np.round(np.quantile(life, [0.25, 0.5, 0.75]), 2)))
json.dump(out, open(f'{D}/results/posthoc_review_fixes.json', 'w'), indent=1)
