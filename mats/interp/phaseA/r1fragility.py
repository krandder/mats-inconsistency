"""POST HOC (METHOD §4.2 fragility control, not in PREREG): is q0's gap missing structure or the network's response to
noise of q0's residual shape? On fresh eval B, per site:
  q0        : h' ~ N(mu(phi), Sigma)                     (as in round 1)
  noise     : h' = h_real + eps, eps ~ N(0, Sigma)       (real structure kept, same noise added)
  mean-only : h' = mu(phi)                               (q0's mean, no noise; not a cover of p, diagnostic only)
  q0-half   : h' ~ N(mu(phi), Sigma/4)                   (noise sd halved)
usage: python r1fragility.py MODEL"""
import json, sys
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/cm-interp')
from lib import tp
from lib import r1est as E
NAME = sys.argv[1]
model, meta, ev = tp.load(NAME); L = meta['depth']
want = {('post', k): None for k in range(1, L)}; want[('post', L)] = [tp.QI]


def prep(seed, n, M_):
    d0 = tp.sample_episodes(n, seed); cs0 = tp.cell_stats(d0)
    if M_:
        d = tp.cond_draws(d0, M_, seed + 1)
    else:
        d = d0; d['cell'] = np.arange(n)
    X, V = tp.tokens(d)
    head, st, _ = tp.run_real(model, X, V, want)
    GQ = tp.GQuad(cs0)
    return dict(d=d, cs=tp.take(cs0, d['cell']), V=V, st=st, GQ=GQ, cell=d['cell'], n=n, M=max(M_, 1),
                Gp=GQ(head, d['cell']).reshape(n, max(M_, 1)))


F = prep(60000, 20000, 0); B = prep(63000, 6000, 8)
mid = ('post', L - 1); fin = ('post', L)
sites = [(mid, ('LAB', 'QQ', 'QT'), 'ALL'), (mid, ('QQ',), 'QQ'), (mid, ('QT',), 'QT'), (fin, ('QT',), 'QT')]
rng = np.random.default_rng(3)
out = {}
for cut, tys, lab in sites:
    isfin = cut == fin
    qs = {}
    for t in tys:
        ix = [0] if isfin else tp.GROUPS[t]
        qs[t] = E.QTok(t, 'q0', norm=tp.needs_norm(model, cut, t)).fit(F['d'], F['cs'], tp.to_q(F['st'][cut][:, ix].astype(np.float64), tp.needs_norm(model, cut, t)))
    res = {}
    for mode in ('q0', 'noise', 'mean', 'q0half'):
        h = B['st'][cut].copy()
        for t in tys:
            ix = [0] if isfin else tp.GROUPS[t]
            norm = tp.needs_norm(model, cut, t)
            mu, covs, _ = qs[t].mean_cov(B['d'], B['cs'])
            C = covs[0]; n_, nt, dm = mu.shape
            eps = rng.standard_normal((n_ * nt, dm)) @ np.linalg.cholesky(C).T
            eps = eps.reshape(n_, nt, dm)
            real = tp.to_q(B['st'][cut][:, ix].astype(np.float64), norm)
            if mode == 'q0':
                hs = mu + eps
            elif mode == 'noise':
                hs = real + eps
            elif mode == 'mean':
                hs = mu
            else:
                hs = mu + 0.5 * eps
            h[:, ix] = tp.to_q(hs, norm)
        head = tp.forward_qt(model, h, cut) if isfin else tp.forward_from(model, h, cut, B['V'])
        G = B['GQ'](head, B['cell']).reshape(6000, 8)
        r = E.direct(G, B['Gp'], B=300)
        res[mode] = dict(delta=r['delta'], se=r['se'], rms=r['rms'])
    key = f"h{cut[1]}-{lab}"
    out[key] = res
    print(NAME, key, {k: round(v['delta'], 4) for k, v in res.items()}, flush=True)
json.dump(out, open(f'/home/agent/workspaces/mats/work/cm-interp/r1/fragility_{NAME}.json', 'w'))
