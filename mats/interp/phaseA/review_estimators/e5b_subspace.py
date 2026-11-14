"""E5: is the 'G-relevant 4-d subspace' result specific to q0's missing structure, or just where downstream reads?
Control subspace U_G = top-4 eigenvectors of E_p[g g^T], g = gradient of G itself w.r.t. the token's quotient coordinates
on REAL draws (no classifier, no q0). Knock-in (keep U real, rest from q0 given U) exactly as r1est.subspace_knock.
Compare with round-1's Uout knock-in fraction for the same net/site (r1/<net>.json, eval A).
usage: python e5_subspace.py MODEL NC"""
import json, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/cm-interp')
from lib import tp                       # noqa: E402
from lib import r1est as E               # noqa: E402

NAME, NC, FSET = sys.argv[1], int(sys.argv[2]), sys.argv[3]
NFIT, M = 20000, 8
T0 = time.time()
log = lambda *a: print(f'[{NAME} {time.time() - T0:5.0f}s]', *a, flush=True)
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


F = prep(60000, NFIT, 0); B = prep(64500, NC, M)
log('data ready')
R1 = json.load(open(f'/home/agent/workspaces/mats/work/cm-interp/r1/{NAME}.json'))
r1 = R1['localisation_A']
fin = ('post', L)
sites = [(('post', k), 'QQ') for k in range(1, L)] + [(fin, 'QT')]
rng = np.random.default_rng(77)
out = {}


def G_torch(head, cell):
    mu, ls, lg = head
    g = torch.tensor(B['GQ'].g[cell]); pw = torch.tensor(B['GQ'].pw[cell])
    lgn = torch.log_softmax(lg.double(), 1)
    lq = lgn[:, None, :] - ls.double()[:, None, :] - 0.5 * math.log(2 * math.pi) \
        - 0.5 * (g[:, :, None] - mu.double()[:, None, :]) ** 2 / torch.exp(2 * ls.double())[:, None, :]
    lq = torch.logsumexp(lq, 2)
    return torch.tensor(B['GQ'].plp[cell]) - (pw * lq).sum(1)


for cut, t in sites:
    isfin = cut == fin
    ix = [0] if isfin else tp.GROUPS[t]
    norm = tp.needs_norm(model, cut, t)
    key = f"h{cut[1]}-{t}"
    q0 = E.QTok(t, FSET, norm=norm).fit(F['d'], F['cs'], tp.to_q(F['st'][cut][:, ix].astype(np.float64), norm))
    # gradient of G on real draws w.r.t. the token states (raw coordinates), then into quotient coordinates
    sel = rng.permutation(NC * M)[:6000]
    h = torch.tensor(B['st'][cut][sel], dtype=torch.float32, requires_grad=True)
    if isfin:
        x = h[:, 0]
    else:
        al = tp._allow(torch.from_numpy(B['V'][sel])); x = h
        for kk in range(cut[1] + 1, L + 1):
            x, _ = model.blocks[kk - 1](x, al)
        x = x[:, tp.QI]
    y = model.out(model.ln(x)); K = model.mixk
    head = (y[:, K:2 * K], y[:, 2 * K:].clamp(-6, 3), y[:, :K])
    G_torch(head, B['cell'][sel]).sum().backward()
    g = h.grad.numpy()[:, ix].astype(np.float64)
    if norm:
        raw = B['st'][cut][sel][:, ix].astype(np.float64)
        g = g * (raw - raw.mean(-1, keepdims=True)).std(-1, keepdims=True)
    g = g.reshape(-1, g.shape[-1]); dm = g.shape[1]
    one = np.ones((dm, 1)) / math.sqrt(dm); Pm = np.eye(dm) - one @ one.T
    ev_, U = np.linalg.eigh(Pm @ (g.T @ g / len(g)) @ Pm)
    UG = U[:, ::-1][:, :4]; evs = ev_[::-1]
    Ur = np.linalg.qr(np.random.default_rng(3).standard_normal((dm, 4)))[0]
    Ur = np.linalg.qr(Ur - one @ (one.T @ Ur))[0]
    Hr = tp.to_q(B['st'][cut][:, ix].astype(np.float64), norm)
    res = dict(G_sens_eig_top8_frac=[float(x) for x in (evs[:8] / evs.sum())])
    base = None
    for lab, Uu in (('q0', None), ('UG4', UG), ('rand4', Ur)):
        hh = B['st'][cut].copy()
        Hs = q0.sample(B['d'], B['cs'], rng) if Uu is None else E.subspace_knock(q0, B['d'], B['cs'], Hr, Uu, rng, 'knockin')
        hh[:, ix] = Hs
        hd = tp.forward_qt(model, hh, cut) if isfin else tp.forward_from(model, hh, cut, B['V'])
        G = B['GQ'](hd, B['cell']).reshape(NC, M)
        r = E.direct(G, B['Gp'], B=300)
        if base is None:
            base = r['delta']
        res[lab] = dict(delta=r['delta'], se=r['se'], frac_of_q0=r['delta'] / base)
        log(key, lab, 'delta %.4f +- %.4f frac %.3f' % (r['delta'], r['se'], r['delta'] / base))
    k1 = f'subspace_{key}'
    if k1 in r1:
        o = r1[k1]; dq0 = R1['direct_A'][f'{key}|q0']['delta']
        res['round1_A_frac'] = {lab: o[f'{lab}|knockin']['delta'] / dq0 for lab in ('Uout4', 'Uh4', 'rand4')}
        log(key, 'round-1 (eval A) knock-in fractions', res['round1_A_frac'])
    out[key] = res
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'e5_{NAME}_{FSET}.json'), 'w'), indent=1)
log('done')
