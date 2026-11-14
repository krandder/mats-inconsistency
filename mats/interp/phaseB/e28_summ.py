"""E28 analysis (PREREG-B amendment 36).
Costs (KL removed by averaging over a family's 9 samples; uncorrected, as E26's review asked; x 9/8 also reported):
  odd share = cost(negate_ind) / cost(flip_ind), pooled over nets of a group.
Mean-shift regression on the negation pairs: for every base and source s, dm_s = m(base) - m(base with source s negated);
regressors X1 = 2 rbar_s w_s (own reading's weight in the net, base version) and X2 = 2 rbar_s (w_1 + w_2 + w_3) / 3 (a mean error
pooled over the sources and applied to every reading); OLS without intercept (dm is odd in rbar_s). R^2 = 1 - SSR / sum dm^2.
A bias reading of a source's own mean error that shifts its reading by -c rbar_s gives dm_s = -c X1 (c > 0: correction).
Reliability change: dw_s = w_s(base) - w_s(negated), regressed on 2 rbar_s (descriptive).
usage: python e28_summ.py -> results/e28.json, results/e28_tables.md
"""
import json, os
import numpy as np
import e26_summ as SM
import e28_data as E8

F = E8.FAMILIES


OUT = os.environ.get('E28_OUT', 'results/e28')                 # plumbing checks only: E28_DATA, E28_OUT, E28_RES
RES = os.environ.get('E28_RES', 'results/e28')


def load(set_name, tag):
    f = os.path.join(OUT, set_name, tag + '.npz')
    return np.load(f) if os.path.exists(f) else None


def costs(z):
    n = int(z['nb']); W = SM.boot_weights(n)
    M = W @ np.stack([z['kl_flip_ind']] + [z['pc_' + f] for f in F], 1)
    return M                                                              # (B+1, 1 + len(F))


def regress(zs, rbar, n):
    """Pooled OLS over the given nets' negation pairs. Returns coefficients (2,), R^2, per-net own-only coefficients."""
    Y, X = [], []
    per = []
    for z in zs:
        w = z['w_base']; dm = np.stack([z['m_base'] - z['m_neg%d' % s] for s in range(3)], 1)
        x1 = 2 * rbar[:n] * w; x2 = 2 * rbar[:n] * w.sum(1, keepdims=True) / 3
        Y.append(dm.ravel()); X.append(np.stack([x1.ravel(), x2.ravel()], 1))
        c1 = float((x1 * dm).sum() / (x1 ** 2).sum())
        b, *_ = np.linalg.lstsq(np.stack([x1.ravel(), x2.ravel()], 1), dm.ravel(), rcond=None)
        r2 = 1 - ((dm.ravel() - np.stack([x1.ravel(), x2.ravel()], 1) @ b) ** 2).sum() / (dm.ravel() ** 2).sum()
        per.append(dict(own_only=c1, own=float(b[0]), pooled=float(b[1]), r2=float(r2),
                        rms_dm=float(np.sqrt((dm ** 2).mean()))))
    Y = np.concatenate(Y); X = np.concatenate(X)
    b, *_ = np.linalg.lstsq(X, Y, rcond=None)
    r2 = 1 - ((Y - X @ b) ** 2).sum() / (Y ** 2).sum()
    b1 = float((X[:, 0] * Y).sum() / (X[:, 0] ** 2).sum()); r2_1 = 1 - ((Y - b1 * X[:, 0]) ** 2).sum() / (Y ** 2).sum()
    return dict(own=float(b[0]), pooled=float(b[1]), r2=float(r2), own_only=b1, r2_own_only=float(r2_1)), per


def dw_regress(zs, rbar, n):
    out = []
    for z in zs:
        dw = np.stack([z['w_base'][:, s] - z['w_neg%d' % s][:, s] for s in range(3)], 1)
        x = 2 * rbar[:n]
        out.append(dict(slope=float((x * dw).sum() / (x ** 2).sum()), r2=float(1 - ((dw - x * (x * dw).sum() / (x ** 2).sum()) ** 2).sum() / (dw ** 2).sum()),
                        mean_w=float(z['w_base'].mean())))
    return out


def main():
    rows = json.load(open('results/e26_nets.json'))
    dat = np.load(os.environ.get('E28_DATA', 'results/e28_data.npz')); rbar = dat['rbar']
    nets = {}
    for s in ('M', 'G'):
        for r in rows:
            if r['set'] == s:
                z = load(s, r['tag'])
                if z is not None:
                    nets[r['tag']] = (s, z)
    groups = dict(A=SM.A_BASES, B=SM.B_BASES, M=[t for t, (s, _) in nets.items() if s == 'M'],
                  G=[t for t, (s, _) in nets.items() if s == 'G'])
    res = dict(groups={}, per_net={})
    for g, tags in groups.items():
        zs = [nets[t][1] for t in tags if t in nets]
        if not zs:
            continue
        n = int(zs[0]['nb'])
        C = sum(costs(z) for z in zs)
        o = dict(n=len(zs), mean_KL=SM.ci(C[:, 0] / len(zs)))
        for i, f in enumerate(F):
            o['share_' + f] = SM.ci(C[:, 1 + i] / C[:, 0])
        o['odd_share'] = SM.ci(C[:, 2] / C[:, 1]); o['joint_over_ind_negation'] = SM.ci(C[:, 3] / C[:, 2])
        reg, per = regress(zs, rbar, n)
        o['regression'] = reg
        o['dw'] = dw_regress(zs, rbar, n)
        res['groups'][g] = o
        for t, pr_, dwr in zip([t for t in tags if t in nets], per, o['dw']):
            res['per_net'][t] = dict(set=nets[t][0], **pr_, dw_slope=dwr['slope'], dw_r2=dwr['r2'])
    pr = {}
    if 'B' in res['groups']:
        b = res['groups']['B']
        pr['P144'] = dict(odd_share=b['odd_share'], holds=bool(b['odd_share'][0] >= 0.5))
        pr['P145'] = dict(r2=b['regression']['r2'], holds=bool(b['regression']['r2'] >= 0.3))
        own = [res['per_net'][t]['own'] for t in SM.B_BASES if t in res['per_net']]
        npos = sum(x > 0 for x in own); nneg = sum(x < 0 for x in own)
        pr['P146'] = dict(own_coefficients=own, holds=bool(max(npos, nneg) >= 5 and all(abs(x) >= 0.1 for x in own if (x > 0) == (npos >= nneg))))
    res['predictions'] = pr
    md = ['# E28 tables (generated by e28_summ.py)', '',
          '| group | nets | mean KL | flip_ind / KL | negate_ind / KL | negate_joint / KL | odd share | joint/ind negation | regression own | pooled | R^2 | own only | R^2 own only |',
          '|---' * 13 + '|']
    for g, o in res['groups'].items():
        r = o['regression']
        md.append('| %s | %d | %.4f | %.3f | %.3f | %.3f | %.2f [%.2f, %.2f] | %.2f | %.3f | %.3f | %.2f | %.3f | %.2f |' % (
            g, o['n'], o['mean_KL'][0], o['share_flip_ind'][0], o['share_negate_ind'][0], o['share_negate_joint'][0], *o['odd_share'],
            o['joint_over_ind_negation'][0], r['own'], r['pooled'], r['r2'], r['own_only'], r['r2_own_only']))
    md += ['', '| net | set | own | pooled | R^2 | own only | rms dm | dw slope | dw R^2 |', '|---' * 9 + '|']
    for t, o in res['per_net'].items():
        md.append('| %s | %s | %.3f | %.3f | %.2f | %.3f | %.4f | %.4f | %.2f |' % (t, o['set'], o['own'], o['pooled'], o['r2'], o['own_only'],
                                                                             o['rms_dm'], o['dw_slope'], o['dw_r2']))
    md += ['', '## Predictions', ''] + ['- %s: %s' % (k, json.dumps(v)) for k, v in pr.items()]
    json.dump(res, open(RES + '.json', 'w'), indent=1, default=float)
    open(RES + '_tables.md', 'w').write('\n'.join(md) + '\n')
    print(json.dumps(pr, indent=1, default=float))


if __name__ == '__main__':
    main()
