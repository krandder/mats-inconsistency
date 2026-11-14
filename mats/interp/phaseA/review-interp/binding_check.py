"""Reviewer spot check (not part of round 1): does 'q0U beats q0' at h1-QQ isolate forgotten source binding, or
q0U's extra content (all-source sorted log SS, mean Q, blind-reference moments)? Nested feature sets, same
LinearGaussian family, fit split seed 60000 (20k), eval B seed 63000 (6000 x 8). usage: python binding_check.py MODEL"""
import json, sys, time
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/cm-interp')
from lib import tp
from lib import r1est as E
NAME = sys.argv[1]
T0 = time.time()
model, meta, ev = tp.load(NAME); L = meta['depth']
cut = ('post', 1); t = 'QQ'
want = {cut: None}


def prep(seed, n, M_):
    d0 = tp.sample_episodes(n, seed); cs0 = tp.cell_stats(d0)
    if M_:
        d = tp.cond_draws(d0, M_, seed + 1)
    else:
        d = d0; d['cell'] = np.arange(n)
    X, V = tp.tokens(d)
    head, st, _ = tp.run_real(model, X, V, want)
    GQ = tp.GQuad(cs0)
    return dict(d=d, cs=tp.take(cs0, d['cell']), st=st, V=V, GQ=GQ, cell=d['cell'], n=n, M=max(M_, 1),
                Gp=GQ(head, d['cell']).reshape(n, max(M_, 1)))


def bc(a, nt):  # (n, k) -> (n, nt, k)
    return np.repeat(a[:, None, :], nt, 1)


def feats(kind, d, cs):
    nt = len(tp.GROUPS[t])
    if kind == 'q0':
        return tp.token_features(d, cs, t, 'q0')
    if kind == 'q0U':
        return tp.token_features(d, cs, t, 'q0U')
    if kind == 'q0+q0U':
        return np.concatenate([tp.token_features(d, cs, t, 'q0'), tp.token_features(d, cs, t, 'q0U')], 2)
    rin = tp.token_features(d, cs, t, 'Rin')
    mq = bc(cs['ostar'].mean(1, keepdims=True), nt)
    if kind == 'Rin+meanQ':
        return np.concatenate([rin, mq], 2)
    if kind == 'Rin+meanQ+sortedSS':
        return np.concatenate([rin, mq, bc(np.sort(cs['lss'], 1), nt)], 2)
    raise ValueError(kind)


class QC(E.QTok):
    def __init__(self, kind, norm):
        super().__init__(t, 'q0', norm=norm); self.kind = kind

    def phi(self, d, cs):
        return feats(self.kind, d, cs)


F = prep(60000, 20000, 0); B = prep(63000, 6000, 8)
norm = tp.needs_norm(model, cut, t)
ix = tp.GROUPS[t]
Hf = tp.to_q(F['st'][cut][:, ix].astype(np.float64), norm)
rng = np.random.default_rng(13)
out = {}
for kind in ('q0', 'q0U', 'q0+q0U', 'Rin+meanQ', 'Rin+meanQ+sortedSS'):
    q = QC(kind, norm).fit(F['d'], F['cs'], Hf)
    h = B['st'][cut].copy(); h[:, ix] = q.sample(B['d'], B['cs'], rng)
    head = tp.forward_from(model, h, cut, B['V'])
    G = B['GQ'](head, B['cell']).reshape(B['n'], B['M'])
    r = E.direct(G, B['Gp'], B=300)
    out[kind] = dict(delta=r['delta'], se=r['se'], rms=r['rms'], rms_lo=r['rms_lo'], rms_hi=r['rms_hi'], k=int(q.k))
    print(NAME, kind, 'k=%d delta %.4f+-%.4f rms %.4f [%.4f, %.4f]' % (q.k, r['delta'], r['se'], r['rms'], r['rms_lo'], r['rms_hi']),
          '%.0fs' % (time.time() - T0), flush=True)
json.dump(dict(model=NAME, site='h1-QQ', kl_B=float(B['Gp'].mean()), res=out), open(f'/tmp/review-interp/binding_check_{NAME}.json', 'w'), indent=1)
