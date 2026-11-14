"""POST HOC (after seeing round-1 naming results): split the winning candidates into sub-groups to sharpen the names.
QQ sites: cross -> cross_o (o*_{-s}, log SS_{-s}) vs cross_p (full-Bayes p_B,s, E_B,s - E_lab,s).
QT sites: quadlat -> meansq (squared predictive means) vs full quadlat. Direct Delta_rms on fresh eval B.
usage: python r1posthoc.py MODEL"""
import json, sys, time
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/cm-interp')
from lib import tp
from lib import r1est as E
NAME = sys.argv[1]
model, meta, ev = tp.load(NAME); L = meta['depth']
T0 = time.time()
want = {('post', k): None for k in range(1, L)}; want[('post', L)] = [tp.QI]
NFIT, NC, M = 20000, 6000, 8


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


F = prep(60000, NFIT, 0); B = prep(63000, NC, M)
out = {}
sites = [(('post', k), 'QQ') for k in range(1, L)] + [(('post', L), 'QT'), (('post', L - 1), 'QT')]
rng = np.random.default_rng(5)
for cut, t in sites:
    fin = cut == ('post', L)
    ix = [0] if fin else tp.GROUPS[t]
    norm = tp.needs_norm(model, cut, t)
    Hf = tp.to_q(F['st'][cut][:, ix].astype(np.float64), norm)
    cands = [(), ('cross',), ('cross_o',), ('cross_p',)] if t == 'QQ' else [(), ('quadlat',), ('meansq',), ('betaQ',)]
    res = {}
    for ex in cands:
        q = E.QTok(t, 'q0', ex, norm=norm).fit(F['d'], F['cs'], Hf)
        h = B['st'][cut].copy(); h[:, ix] = q.sample(B['d'], B['cs'], rng)
        head = tp.forward_qt(model, h, cut) if fin else tp.forward_from(model, h, cut, B['V'])
        G = B['GQ'](head, B['cell']).reshape(NC, M)
        r = E.direct(G, B['Gp'], B=300)
        res['+'.join(ex) or 'q0'] = dict(delta=r['delta'], rms=r['rms'], rms_lo=r['rms_lo'], rms_hi=r['rms_hi'], n_params=q.m.n_params())
    out[f"{'h'}{cut[1]}-{t}"] = res
    print(NAME, cut, t, {k: round(v['rms'], 4) for k, v in res.items()}, f'{time.time() - T0:.0f}s', flush=True)
json.dump(out, open(f'/home/agent/workspaces/mats/work/cm-interp/r1/posthoc_{NAME}.json', 'w'))
