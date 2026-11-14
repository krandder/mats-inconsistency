"""E54 plumbing (disclosed in amendment 62): synthetic per-net results with planted values for e54_summ.py, on the plumbing pairs
(/tmp/e54_plumb/pairs_d{1,2}.npz, seed 54,900, outside the registered data). Planted:
  ctrl D=1: dls -0.05 in 11 nets and +0.01 in one, lam 0.2      -> P212 part holds (11 of 12), P213 part holds;
  ctrl D=2: dls -0.03 in all, lam +0.1 in 6 nets and -0.1 in 6  -> P213 fails;
  canon D=1: dls +0.02; the other 7 arms dls -0.02               -> 9 of 10 arms negative, P214 holds;
  log-sd slope on log f: b = 0.5 everywhere (so u = 1 - exp(4 dls)).
usage: python e54_synth.py -> /tmp/e54_synth/res/*.npz; then python e54_summ.py /tmp/e54_synth/res /tmp/e54_plumb/pairs_d%d.npz
"""
import os
import numpy as np

OUT = '/tmp/e54_synth/res'
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)
for D in (1, 2):
    P = np.load('/tmp/e54_plumb/pairs_d%d.npz' % D)
    n = len(P['dshift']); mB = (P['bayes_wt'] * P['bayes_mu']).sum(1)
    for fmt in ('ctrl', 'canon', 'prod', 'sgnprod', 'pre'):
        for s in range(12):
            if fmt == 'ctrl' and D == 1:
                dls, lam = (-0.05 if s < 11 else 0.01), 0.2
            elif fmt == 'ctrl' and D == 2:
                dls, lam = -0.03, (0.1 if s < 6 else -0.1)
            elif fmt == 'canon' and D == 1:
                dls, lam = 0.02, 0.0
            else:
                dls, lam = -0.02, 0.0
            ls_spr = rng.normal(0, 0.1, n)
            m_spr = mB + rng.normal(0, 0.01, n)
            out = dict(fmt=('ctrl' if fmt == 'pre' else fmt), D=D, seed=s, variant=('pre' if fmt == 'pre' else ''), run='synthetic',
                       ls_spr=ls_spr, ls_big=ls_spr + dls, ls_f60=ls_spr + 0.5 * np.log(0.6), ls_f80=ls_spr + 0.5 * np.log(0.8),
                       m_spr=m_spr, m_big=m_spr + lam * P['dshift'], m_f60=m_spr, m_f80=m_spr,
                       kl_spr=np.full(n, 0.01), kl_big=np.full(n, 0.012), mass_big=np.ones(n), mass_spr=np.ones(n))
            np.savez(os.path.join(OUT, 'e%s_d%d%s_s%d.npz' % ('49' if fmt == 'pre' else '40', D, fmt, s)), **out)
print('synthetic results written to', OUT)
