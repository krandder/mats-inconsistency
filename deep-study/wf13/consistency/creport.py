"""creport.py OUTDIR [BADCHECKS] : all metrics of one world from the arms' test forecasts and the exact references.

EVALUATION CODE: joins OUTDIR/*.npz (written by ctrain.py, cdirect.py) with OUTDIR/exact.npz. Writes OUTDIR/summary.json.
Metrics (K = number of trajectories per forecast):
 t1  excess = mean KL(exact P(x_c | x_S) || forecast) over all (context, outside check); real = log loss on the real
     outcome; excT = the same KL against the score's exact answer; by context size; on BADCHECKS and the rest.
 t2  pair mismatch D = mean over triples of sum_{x,y} w (log J1 - log J2)^2, w = (J1 + J2) / 2, J1 = A(X|C) A(Y|C,X),
     J2 = A(Y|C) A(X|C,Y); noise floor = the same between two independent trajectory draws of ONE order; KL of the
     exact joint to each order's joint.
 t3  V_ord = variance over 4 orders of the summed log step predictions for one real terminal vector (mean over
     vectors); V_noise = the same between two trajectory draws of one order; msq_true = mean (sum - exact log P*(x))^2;
     var_score = variance over (vector, order) of (sum - log R-hat(x)), the quantity form iis trains on.
 drag  per triple, the order whose joint is closer to the exact joint under the data-only run is "better"; change of
     each order's KL in the arm, minus the same change in a second data-only run with another initialisation (placebo).
"""
import glob
import json
import os
import sys

import numpy as np

LEVELS = [1, 4, 16, 64]


def sig(z):
    return 1.0 / (1.0 + np.exp(-z.astype(np.float64)))


def lsig(z):
    z = z.astype(np.float64)
    return -np.logaddexp(0.0, -z)


def klb(p, q):
    q = np.clip(q, 1e-6, 1 - 1e-6)
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def joints(z):
    """z (n,6) logits -> log J1, log J2 (n,2,2) indexed [x,y]."""
    l = lambda a: np.stack([lsig(-a), lsig(a)], -1)
    lx, ly = l(z[:, 0]), l(z[:, 1])
    lyx = np.stack([l(z[:, 2]), l(z[:, 3])], 1)
    lxy = np.stack([l(z[:, 4]), l(z[:, 5])], 2)
    return lx[:, :, None] + lyx, ly[:, None, :] + lxy


def exact_joint(E):
    """Exact P*(X = x, Y = y | C) (n,2,2) from the five contexts' exact numbers."""
    ps, X, Y = E['t2_pstar'], E['t2_X'], E['t2_Y']
    n = len(X)
    ar = np.arange(n)
    px = ps[ar, X]
    py0, py1 = ps[n + ar, Y], ps[2 * n + ar, Y]
    J = np.zeros((n, 2, 2))
    J[:, 0, 0], J[:, 0, 1] = (1 - px) * (1 - py0), (1 - px) * py0
    J[:, 1, 0], J[:, 1, 1] = px * (1 - py1), px * py1
    return J


def run_metrics(A, E, bad):
    out = dict(t1={}, t2={}, t3={})
    mk = ~E['t1_mask']
    ps, pt, x = E['t1_pstar'], E['t1_pTu'], E['t1_x'].astype(np.float64)
    size = E['t1_mask'].sum(1)
    Jx = exact_joint(E)
    per = {}
    for K in LEVELS:
        if 't1_K%d' % K not in A:
            continue
        p = np.clip(sig(A['t1_K%d' % K]), 1e-6, 1 - 1e-6)
        kl = klb(ps, p)
        ctx = (kl * mk).sum(1) / np.maximum(mk.sum(1), 1)            # per-context mean excess
        per['t1_K%d' % K] = ctx
        ll = -(x * np.log(p) + (1 - x) * np.log(1 - p))
        d = dict(excess=float(kl[mk].mean()), real=float(ll[mk].mean()), excT=float(klb(pt, p)[mk].mean()))
        for nm, lo, hi in (('small', 0, 9), ('mid', 10, 19), ('large', 20, 29)):
            s = mk & ((size >= lo) & (size <= hi))[:, None]
            d['exc_' + nm] = float(kl[s].mean())
        if bad:
            b = np.zeros(30, dtype=bool)
            b[bad] = True
            d['exc_bad'], d['exc_good'] = float(kl[mk & b[None, :]].mean()), float(kl[mk & ~b[None, :]].mean())
        out['t1'][K] = d
        l1, l2 = joints(A['t2_K%d_r0' % K])
        w = 0.5 * (np.exp(l1) + np.exp(l2))
        k1 = (Jx * (np.log(np.clip(Jx, 1e-300, 1)) - l1)).sum((1, 2))
        k2 = (Jx * (np.log(np.clip(Jx, 1e-300, 1)) - l2)).sum((1, 2))
        per['t2_K%d' % K] = np.stack([k1, k2], 1)
        d = dict(D=float((w * (l1 - l2) ** 2).sum((1, 2)).mean()), KL1=float(k1.mean()), KL2=float(k2.mean()),
                 absdiff=float(np.abs(l1 - l2).mean()))
        if 't2_K%d_r1' % K in A:
            m1, m2 = joints(A['t2_K%d_r1' % K])
            d['D_noise'] = float(0.5 * ((0.5 * (np.exp(l1) + np.exp(m1)) * (l1 - m1) ** 2).sum((1, 2)).mean()
                                        + (0.5 * (np.exp(l2) + np.exp(m2)) * (l2 - m2) ** 2).sum((1, 2)).mean()))
        out['t2'][K] = d
        perm, xv = E['t3_perm'], E['t3_xvec'].astype(np.int64)
        val = np.take_along_axis(np.broadcast_to(xv[:, None, :], perm.shape), perm, 2)
        L = lsig(A['t3_K%d_r0' % K] * (2 * val - 1)).sum(-1)                    # (Ns, O)
        d = dict(V_ord=float(L.var(1, ddof=1).mean()), msq_true=float(((L - E['t3_logPstar'][:, None]) ** 2).mean()),
                 bias_true=float((L - E['t3_logPstar'][:, None]).mean()),
                 var_score=float((L - E['t3_logRhat'][:, None]).var()), msq_T=float(((L - E['t3_logT'][:, None]) ** 2).mean()))
        if 't3_K%d_r1' % K in A:
            L1 = lsig(A['t3_K%d_r1' % K] * (2 * val - 1)).sum(-1)
            d['V_noise'] = float(((L - L1) ** 2 / 2).mean())
        out['t3'][K] = d
    out['t1_avg'] = float(np.mean([out['t1'][K]['excess'] for K in out['t1']]))
    out['real_avg'] = float(np.mean([out['t1'][K]['real'] for K in out['t1']]))
    return out, per


def boot(v, nb=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(nb, len(v)))
    m = v[idx].mean(1)
    return [float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


if __name__ == '__main__':
    odir = sys.argv[1]
    bad = [int(v) for v in sys.argv[2].split(',')] if len(sys.argv) > 2 and sys.argv[2] else []
    E = dict(np.load(odir + '/exact.npz'))
    if not bad:                                                   # registered rule: checks where the score T itself is
        mk0 = ~E['t1_mask']                                       # more than 0.05 nats from the exact posterior
        kT = klb(E['t1_pstar'], E['t1_pTu']) * mk0
        per_check = kT.sum(0) / np.maximum(mk0.sum(0), 1)
        bad = [int(c) for c in np.where(per_check > 0.05)[0]]
        print('checks the score gets wrong (excess of T > 0.05):', bad, np.round(per_check, 3).tolist())
    runs, pers = {}, {}
    for f in sorted(glob.glob(odir + '/*.npz')):
        tag = os.path.basename(f)[:-4]
        if tag == 'exact':
            continue
        A = dict(np.load(f))
        meta = json.load(open(f[:-4] + '.json')) if os.path.exists(f[:-4] + '.json') else {}
        if tag.startswith('direct'):                              # one-pass baseline: t1 only, no K
            p = np.clip(A['t1'].astype(np.float64), 1e-6, 1 - 1e-6)
            mk, x = ~E['t1_mask'], E['t1_x'].astype(np.float64)
            runs[tag] = dict(arm='direct', n=meta.get('n'), d=meta.get('d'), t1_avg=float(klb(E['t1_pstar'], p)[mk].mean()),
                             real_avg=float(-(x * np.log(p) + (1 - x) * np.log(1 - p))[mk].mean()))
            continue
        m, per = run_metrics(A, E, bad)
        m.update({k: meta.get(k) for k in ('arm', 'lam', 'n', 'd', 'r', 'best_step', 'best_val', 'secs')})
        runs[tag], pers[tag] = m, per
    # one-pass references from the exact file (M and the two other sizes)
    mk = ~E['t1_mask']
    ref = dict(score_T=float(klb(E['t1_pstar'], E['t1_pTu'])[mk].mean()))
    xr = E['t1_x'].astype(np.float64)
    ref['real_truth'] = float(-(xr * np.log(E['t1_pstar']) + (1 - xr) * np.log(1 - E['t1_pstar']))[mk].mean())
    for k in ('p_one', 'p_one_S', 'p_one_M', 'p_one_L'):
        if 't1_' + k in E:
            ref[k] = float(klb(E['t1_pstar'], E['t1_' + k])[mk].mean())
    if 't3_logT' in E:                                            # how good the score's sampled reference value is
        dR = (E['t3_logRhat'] - E['t3_logT']).astype(np.float64)
        ref.update(rhat_abs_err=float(np.abs(dR).mean()), rhat_bias=float(dR.mean()), rhat_sd=float(dR.std()),
                   rhat_max_abs=float(np.abs(dR).max()), logT_minus_logPstar=float((E['t3_logT'] - E['t3_logPstar']).mean()),
                   logPstar_mean=float(E['t3_logPstar'].mean()))
    # paired differences against the data-only run with the same (n, d, r = 0), and the drag analysis
    cmp_ = {}
    for tag, m in runs.items():
        if m.get('arm') in (None, 'data', 'direct') or not m.get('n'):
            continue
        base = 'data_n%d_d%d_r0' % (m['n'], m['d'])
        plac = 'data_n%d_d%d_r1' % (m['n'], m['d'])
        if base not in pers or m['r'] != 0:
            continue
        c = {}
        dv = np.mean([pers[tag]['t1_K%d' % K] - pers[base]['t1_K%d' % K] for K in LEVELS if 't1_K%d' % K in pers[tag]], 0)
        c['d_excess'] = boot(dv)
        for K in (1, 4, 16):
            if 't2_K%d' % K not in pers[tag] or plac not in pers:
                continue
            b, a, pl = pers[base]['t2_K%d' % K], pers[tag]['t2_K%d' % K], pers[plac]['t2_K%d' % K]
            better = b.argmin(1)
            ar = np.arange(len(b))
            db, dw = a[ar, better] - b[ar, better], a[ar, 1 - better] - b[ar, 1 - better]
            pb, pw = pl[ar, better] - b[ar, better], pl[ar, 1 - better] - b[ar, 1 - better]
            c['drag_K%d' % K] = dict(better=boot(db), worse=boot(dw), placebo_better=boot(pb), placebo_worse=boot(pw),
                                     adj_better=boot(db - pb), adj_worse=boot(dw - pw),
                                     gap_base=float((b[ar, 1 - better] - b[ar, better]).mean()))
        cmp_[tag] = c
    json.dump(dict(runs=runs, ref=ref, cmp=cmp_, bad=bad, n_t1_ctx=int(E['t1_mask'].shape[0]), n_t1_queries=int(mk.sum()),
                   n_t2=int(len(E['t2_X'])), n_t3=int(E['t3_xvec'].shape[0])), open(odir + '/summary.json', 'w'), indent=0)
    print('summary written: %d runs; one pass of M %.4f, the score T %.4f' % (len(runs), ref.get('p_one', -1), ref['score_T']))
    for tag in sorted(runs):
        m = runs[tag]
        if m.get('t2'):
            K = max(m['t2'])
            print('%-28s excess %.4f real %.4f | pair D(K=%d) %.4f | V_ord %.3f' % (tag, m['t1_avg'], m['real_avg'], K, m['t2'][K]['D'], m['t3'][K]['V_ord']))
