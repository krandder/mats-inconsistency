"""Reviewer spot check (not part of round 1): at the final query-token cut the head reads LN(h), so the state is
only defined up to shift AND scale. Round 1 fits q0 on centred (not scale-normalised) QT states there. Does fitting
the same q0 (same features, same family) in the LN-normalised gauge remove the 'quadratic in the predictive mean'
residual? Eval on round-1 eval B (seed 63000, 8 draws). usage: python ln_check.py MODEL"""
import json, sys, time
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/cm-interp')
from lib import tp
from lib import r1est as E
NAME = sys.argv[1]
T0 = time.time()
model, meta, ev = tp.load(NAME); L = meta['depth']
fin = ('post', L)
want = {fin: [tp.QI]}


def prep(seed, n, M_):
    d0 = tp.sample_episodes(n, seed); cs0 = tp.cell_stats(d0)
    if M_:
        d = tp.cond_draws(d0, M_, seed + 1)
    else:
        d = d0; d['cell'] = np.arange(n)
    X, V = tp.tokens(d)
    head, st, _ = tp.run_real(model, X, V, want)
    GQ = tp.GQuad(cs0)
    return dict(d=d, cs=tp.take(cs0, d['cell']), st=st, GQ=GQ, cell=d['cell'], n=n, M=max(M_, 1),
                Gp=GQ(head, d['cell']).reshape(n, max(M_, 1)))


F = prep(60000, 20000, 0); B = prep(63000, 6000, 8)
print(NAME, 'prep %.0fs' % (time.time() - T0), 'KL_B %.5f' % B['Gp'].mean(), flush=True)
rng = np.random.default_rng(11)
out = {}
for lab, norm, extra, mean_only in [('q0_centred', False, (), False), ('q0_LNgauge', True, (), False),
                                     ('q0_centred_meanonly', False, (), True), ('q0_LNgauge_meanonly', True, (), True),
                                     ('q0+quadlat_LNgauge', True, ('quadlat',), False)]:
    Hf = tp.to_q(F['st'][fin][:, [0]].astype(np.float64), norm)
    q = E.QTok('QT', 'q0', extra, norm=norm).fit(F['d'], F['cs'], Hf)
    if mean_only:
        mu, _, _ = q.mean_cov(B['d'], B['cs']); Hs = tp.to_q(mu, norm)
    else:
        Hs = q.sample(B['d'], B['cs'], rng)
    head = tp.forward_qt(model, Hs, fin)
    G = B['GQ'](head, B['cell']).reshape(B['n'], B['M'])
    r = E.direct(G, B['Gp'], B=300)
    out[lab] = dict(delta=r['delta'], se=r['se'], rms=r['rms'], rms_lo=r['rms_lo'], rms_hi=r['rms_hi'])
    print(NAME, lab, 'delta %.4f+-%.4f rms %.4f [%.4f, %.4f]' % (r['delta'], r['se'], r['rms'], r['rms_lo'], r['rms_hi']),
          '%.0fs' % (time.time() - T0), flush=True)
json.dump(dict(model=NAME, cut=list(fin), kl_B=float(B['Gp'].mean()), res=out),
          open(f'/tmp/review-interp/ln_check_{NAME}.json', 'w'), indent=1)
