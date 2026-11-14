"""Exact-Bayes targets for the PROBE and C4 sets + checks (bayes_ext vs predictive; forward vs harness)."""
import pickle, sys, time
import numpy as np
import torch
import mech as Mx

which = sys.argv[1]          # probe | c4 | fwdcheck
if which == 'fwdcheck':
    ep, d = Mx.c4_set()
    for name in ('D4_s0', 'Q6_s0'):
        m = Mx.load_net(name)
        X = torch.as_tensor(ep['X'][:40]); M = torch.as_tensor(ep['mask'][:40])
        (mean, lsd, (w, mu, v)), _, _ = Mx.forward(m, X, M, keep_sites=False)
        p = Mx.Hn.predict(m, ep['X'][:40], ep['mask'][:40])
        hm, hv = Mx.Hn.moments(p)
        print(name, 'max |mean diff|', float(np.abs(mean.numpy() - hm).max()),
              'max |logsd diff|', float(np.abs(lsd.numpy() - 0.5 * np.log(hv)).max()))
    sys.exit()
t0 = time.time()
if which == 'probe':
    ep = Mx.probe_set()
else:
    ep, d = Mx.c4_set()
eps = ep['eps']
out = []
for i, e in enumerate(eps):
    r = Mx.bayes_ext(e)
    out.append(r)
    if i < 30:
        w, m, s = Mx.W.predictive(e, 'bayes')
        qm = (w * m).sum()
        assert abs(qm - r['qmean']) < 1e-6, (i, qm, r['qmean'])
        assert abs((r['beta'] * e.y).sum() - r['qmean']) < 1e-6, (i, (r['beta'] * e.y).sum(), r['qmean'])
if which == 'c4':
    assert np.allclose([r['qmean'] for r in out], d['mean_bayes'], atol=1e-6)
mx = max(abs((r['beta'] * e.y).sum() - r['qmean']) for r, e in zip(out, eps))
print(which, len(out), 'episodes', f'{time.time() - t0:.1f}s', 'max |sum beta y - mean|', mx)
pickle.dump(out, open(f'out/bayes_{which}.pkl', 'wb'))
