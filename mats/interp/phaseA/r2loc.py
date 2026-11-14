"""Round 2 readout control for localisation (estimator review fix 2; PREREG amendment 4, P18).
Round 1 claimed (a) 4 G-relevant directions carry 92-98% of q0's gap and (b) the MLP write carries 98-100% of it at the
first post-block cut. Both lacked a readout control. Here, on every net, the same knock-in localisations are run for q0
AND for the weaker rung R_in, with U_G = top-4 eigenvectors of E_p[grad G grad G^T] on real draws (no classifier, no q):
  - subspace knock-in: U_G kept real, rest from the rung given U_G (and a random 4-d control for q0);
  - component knock-in: one residual-stream write [prev, head_0, head_1, mlp] kept real, the others from the rung's
    stacked Gaussian given it (components centred, NOT rescaled; their sum is then put in quotient coordinates).
A localisation is about q0 only where q0's remaining fraction differs from R_in's.
usage: python r2loc.py MODEL     writes r2/loc_MODEL.json"""
import json, math, sys, time
import numpy as np
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/cm-interp')
from lib import tp                       # noqa: E402
from lib import r1est as E               # noqa: E402

torch.set_num_threads(1)
NAME = sys.argv[1]
NFIT, NC, M = 20000, 3000, 8
T0 = time.time()
log = lambda *a: print(f'[{NAME} {time.time() - T0:6.0f}s]', *a, flush=True)
model, meta, ev = tp.load(NAME)
L = meta['depth']
MULTI = [('post', k) for k in range(1, L)]
FPOST = ('post', L)
want = {c: None for c in MULTI}; want[FPOST] = [tp.QI]
comp_cuts = {c: None for c in MULTI}; comp_cuts[FPOST] = [tp.QI]
cname = lambda c: f'h{c[1]}'


def prep(seed, n, M_):
    d0 = tp.sample_episodes(n, seed); cs0 = tp.cell_stats(d0)
    if M_:
        d = tp.cond_draws(d0, M_, seed + 1)
    else:
        d = d0; d['cell'] = np.arange(n)
    X, V = tp.tokens(d)
    head, st, comp = tp.run_real(model, X, V, want, comp_cuts, bs=1000)
    GQ = tp.GQuad(cs0) if M_ else None
    Gp = GQ(head, d['cell'], chunk=1000).reshape(n, M_) if M_ else None
    return dict(d=d, cs=tp.take(cs0, d['cell']), V=V, st=st, comp=comp, GQ=GQ, cell=d['cell'], Gp=Gp, n=n, M=max(M_, 1))


F = prep(60000, NFIT, 0)
B = prep(74000, NC, M)
log('data ready')
tok_idx = lambda cut, t: [0] if cut == FPOST else tp.GROUPS[t]
types_at = lambda cut: ['QT'] if cut == FPOST else ['LAB', 'QQ', 'QT']
norm_of = lambda cut, t: tp.needs_norm(model, cut, t)


def eval_site(cut, repl):
    h = B['st'][cut].copy()
    for t, Hs in repl.items():
        h[:, tok_idx(cut, t)] = Hs
    head = tp.forward_qt(model, h, cut) if cut == FPOST else tp.forward_from(model, h, cut, B['V'], bs=1000)
    return B['GQ'](head, B['cell'], chunk=1000).reshape(NC, M)


def dlt(G):
    return float((G - B['Gp']).mean())


rows = {}
rng = np.random.default_rng(11)
# ----------------------------------------------------------------------------- component knock-in (stacked Gaussian)
for cut in MULTI + [FPOST]:
    tys = types_at(cut)
    qs = {}
    for r in ('q0', 'Rin'):
        for t in tys:
            C = F['comp'][cut][:, tok_idx(cut, t)].astype(np.float64)
            C = C - C.mean(-1, keepdims=True)
            n_, nt, nc, dm = C.shape
            q = E.QTok(t, r)
            Phi = q.phi(F['d'], F['cs']).reshape(n_ * nt, -1)
            m = E.LinearGaussian(cov_type='full', shrink=1e-5)
            m.fit(C.reshape(n_ * nt, nc * dm), Phi, cv_groups=np.repeat(np.arange(n_), nt))
            qs[(r, t)] = (q, m, nc, dm)
    COMP = ['prev'] + [f'head{h}' for h in range(meta['heads'])] + ['mlp']
    for r in ('q0', 'Rin'):
        def draw(ci, t):
            q, m, nc, dm = qs[(r, t)]
            Cb = B['comp'][cut][:, tok_idx(cut, t)].astype(np.float64)
            Cb = Cb - Cb.mean(-1, keepdims=True)
            n_, nt = Cb.shape[:2]
            Phi = q.phi(B['d'], B['cs']).reshape(n_ * nt, -1)
            if ci is None:
                Hs = m.sample(Phi, rng=rng)
            else:
                Hs = m.subset_sample(Cb.reshape(n_ * nt, nc * dm), np.arange(ci * dm, (ci + 1) * dm), Phi, rng=rng, mode='knockin')
            return tp.to_q(Hs.reshape(n_, nt, nc, dm).sum(2), norm_of(cut, t))
        d_all = dlt(eval_site(cut, {t: draw(None, t) for t in tys}))
        for ci, cn in enumerate(COMP):
            dk = dlt(eval_site(cut, {t: draw(ci, t) for t in tys}))
            key = f'{cname(cut)}-{"ALL" if cut != FPOST else "QT"}|comp:{cn}'
            o = rows.setdefault(key, dict(site=f'{cname(cut)}-{"ALL" if cut != FPOST else "QT"}', part=f'component {cn}'))
            o[f'{r}_delta'] = d_all; o[f'{r}_frac'] = dk / d_all if d_all != 0 else float('nan')
        log('components', cname(cut), r, 'stack Δ %.4f' % d_all, {cn: round(rows[f'{cname(cut)}-{"ALL" if cut != FPOST else "QT"}|comp:{cn}'][f'{r}_frac'], 3) for cn in COMP})


# ----------------------------------------------------------------------------- U_G subspace knock-in
def G_torch(head, cell):
    mu, ls, lg = head
    g = torch.tensor(B['GQ'].g[cell]); pw = torch.tensor(B['GQ'].pw[cell])
    lgn = torch.log_softmax(lg.double(), 1)
    lq = lgn[:, None, :] - ls.double()[:, None, :] - 0.5 * math.log(2 * math.pi) \
        - 0.5 * (g[:, :, None] - mu.double()[:, None, :]) ** 2 / torch.exp(2 * ls.double())[:, None, :]
    return torch.tensor(B['GQ'].plp[cell]) - (pw * torch.logsumexp(lq, 2)).sum(1)


for cut, t in [(c, 'QQ') for c in MULTI] + [(c, 'QT') for c in MULTI] + [(FPOST, 'QT')]:
    ix = tok_idx(cut, t); nm = norm_of(cut, t)
    sel = rng.permutation(NC * M)[:6000]
    h = torch.tensor(B['st'][cut][sel], dtype=torch.float32, requires_grad=True)
    if cut == FPOST:
        x = h[:, 0]
    else:
        al = tp._allow(torch.from_numpy(B['V'][sel])); x = h
        for kk in range(cut[1] + 1, L + 1):
            x, _ = model.blocks[kk - 1](x, al)
        x = x[:, tp.QI]
    y = model.out(model.ln(x)); K = model.mixk
    G_torch((y[:, K:2 * K], y[:, 2 * K:].clamp(-6, 3), y[:, :K]), B['cell'][sel]).sum().backward()
    g = h.grad.numpy()[:, ix].astype(np.float64)
    if nm:
        raw = B['st'][cut][sel][:, ix].astype(np.float64)
        g = g * (raw - raw.mean(-1, keepdims=True)).std(-1, keepdims=True)
    g = g.reshape(-1, g.shape[-1]); dm = g.shape[1]
    one = np.ones((dm, 1)) / math.sqrt(dm); Pm = np.eye(dm) - one @ one.T
    ev_, U = np.linalg.eigh(Pm @ (g.T @ g / len(g)) @ Pm)
    UG = U[:, ::-1][:, :4]; evs = ev_[::-1]
    Ur = np.linalg.qr(np.random.default_rng(3).standard_normal((dm, 4)))[0]; Ur = np.linalg.qr(Ur - one @ (one.T @ Ur))[0]
    Hr = tp.to_q(B['st'][cut][:, ix].astype(np.float64), nm)
    key = f'{cname(cut)}-{t}|U_G4'
    o = rows.setdefault(key, dict(site=f'{cname(cut)}-{t}', part='U_G top-4', G_eig_top4_frac=float(evs[:4].sum() / evs.sum())))
    for r in ('q0', 'Rin'):
        q = E.QTok(t, r, norm=nm).fit(F['d'], F['cs'], tp.to_q(F['st'][cut][:, ix].astype(np.float64), nm))
        d_all = dlt(eval_site(cut, {t: q.sample(B['d'], B['cs'], rng)}))
        dk = dlt(eval_site(cut, {t: E.subspace_knock(q, B['d'], B['cs'], Hr, UG, rng, 'knockin')}))
        o[f'{r}_delta'] = d_all; o[f'{r}_frac'] = dk / d_all
        if r == 'q0':
            o['rand_frac_q0'] = dlt(eval_site(cut, {t: E.subspace_knock(q, B['d'], B['cs'], Hr, Ur, rng, 'knockin')})) / d_all
    log('U_G', key, {k: round(v, 3) for k, v in o.items() if isinstance(v, float)})
json.dump(dict(model=NAME, sizes=dict(NFIT=NFIT, NC=NC, M=M), rows=rows, seconds=time.time() - T0),
          open(f'/home/agent/workspaces/mats/work/cm-interp/r2/loc_{NAME}.json', 'w'), indent=1)
log('done')
