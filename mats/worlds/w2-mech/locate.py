"""Step 1 (locate) on the PROBE set: ridge probes per site/token type (trained vs untrained vs shuffled) and attention
statistics A1-A3 on held-out episodes. usage: python locate.py NET  (D4_s0 | Q6_s0) -> out/locate_NET.json"""
import json, pickle, sys, time
import numpy as np
import torch
import mech as Mx

NET = sys.argv[1]
E_TR = 1500
MAXTOK = 48
t0 = time.time()
ep = Mx.probe_set()
B = pickle.load(open('out/bayes_probe.pkl', 'rb'))
eps = ep['eps']
E = len(eps); T = ep['X'].shape[1]; QI = T - 1

# ---------------------------------------------------------------- token selection and targets
rng = np.random.default_rng(0)
types = {k: ([], [], []) for k in ('CTX', 'UNK', 'SPOTB', 'SPOT', 'QT')}   # (episode, token, target dict)
TGT = dict(CTX=('bmean', 'logprec', 'vfun'), UNK=('pH', 'logprec', 'bmean'), SPOTB=('resid', 'resid2', 'vfun', 'bmean'),
           SPOT=('vfun',), QT=('qmean', 'qlogvar'))
for i, (e, b) in enumerate(zip(eps, B)):
    n = e.n
    src = e.q_src; typ = e.ev_typ[e.q_ev]
    tok = dict(bmean=b['bmean'][src], logprec=b['logprec'][src], vfun=b['vfun'], pH=b['pH'][src], resid=b['resid'],
               resid2=b['resid'] ** 2)
    sel = dict(CTX=np.arange(n), UNK=np.flatnonzero(e.unk[src]), SPOTB=np.flatnonzero(typ != 1),
               SPOT=np.flatnonzero(typ == 0))
    for k, idx in sel.items():
        if len(idx) > MAXTOK:
            idx = np.sort(rng.choice(idx, MAXTOK, replace=False))
        types[k][0].append(np.full(len(idx), i)); types[k][1].append(idx)
        types[k][2].append(np.stack([tok[t][idx] for t in TGT[k]], 1))
    types['QT'][0].append(np.array([i])); types['QT'][1].append(np.array([QI]))
    types['QT'][2].append(np.array([[b['qmean'], np.log(b['qvar'])]]))
types = {k: tuple(np.concatenate(v) for v in vv) for k, vv in types.items()}


# ---------------------------------------------------------------- activations
def acts(model):
    out = None
    X = torch.as_tensor(ep['X']); M = torch.as_tensor(ep['mask'])
    for a in range(0, E, 100):
        _, s, _ = Mx.forward(model, X[a:a + 100], M[a:a + 100])
        s = {k: v.numpy() for k, v in s.items()}
        out = s if out is None else {k: np.concatenate([out[k], s[k]]) for k in out}
    return out


nets = {'T': Mx.load_net(NET), 'U': Mx.load_net(NET, trained=False)}
depth = len(nets['T'].blocks)
SITES = Mx.site_names(depth)
A = {k: acts(m) for k, m in nets.items()}
print(f'acts done {time.time() - t0:.0f}s', flush=True)

# ---------------------------------------------------------------- probes
probes = {}
perm_rng = np.random.default_rng(0)
for ty, (ei, ti, Y) in types.items():
    tr = ei < E_TR; te = ~tr
    Ytr = Y[tr]; Yte = Y[te]
    ok = np.isfinite(Y).all(1)
    tr &= ok; te &= ok
    perm = perm_rng.permutation(tr.sum())
    for site in SITES:
        res = {}
        for net in ('T', 'U'):
            F = A[net][site][ei, ti]
            res[net] = Mx.ridge_probe(F[tr], Y[tr], F[te], Y[te], ei[tr]).tolist()
        F = A['T'][site][ei, ti]
        res['S'] = Mx.ridge_probe(F[tr], Y[tr][perm], F[te], Y[te], ei[tr]).tolist()
        for j, tname in enumerate(TGT[ty]):
            r, ru, rs = res['T'][j], res['U'][j], res['S'][j]
            probes[f'{ty}|{tname}|{site}'] = dict(R2=r, R2_U=ru, R2_S=rs, n_tr=int(tr.sum()), n_te=int(te.sum()),
                                                  positive=bool(r >= 0.5 and r - max(ru, rs) >= 0.10),
                                                  strong=bool(r >= 0.8 and r - max(ru, rs) >= 0.10))
    print(f'probes {ty} done {time.time() - t0:.0f}s', flush=True)


# ---------------------------------------------------------------- attention stats (held-out episodes)
def attn_stats(model):
    X = torch.as_tensor(ep['X']); M = torch.as_tensor(ep['mask'])
    nl, nh = len(model.blocks), model.blocks[0].h
    a1x = [[[] for _ in range(nh + 1)] for _ in range(nl)]; a1y = []
    acc = {k: np.zeros((nl, nh + 1, 2)) for k in ('src', 'src_unk', 'ovl')}   # (sum of per-token ratios, count)
    pool = {k: np.zeros((nl, nh + 1, 4)) for k in ('src', 'ovl')}            # mass_in, n_in, mass_out, n_out
    for a in range(E_TR, E, 50):
        _, _, att = Mx.forward(model, X[a:a + 50], M[a:a + 50], keep_sites=False, keep_attn=True)
        att = [torch.cat([x, x.mean(1, keepdim=True)], 1).numpy() for x in att]   # head mean as last
        for bi in range(att[0].shape[0]):
            i = a + bi; e = eps[i]; b = B[i]; n = e.n
            bt = np.abs(b['beta']); bt = bt / max(bt.sum(), 1e-12)
            a1y.append(bt)
            src = e.q_src
            same = (src[:, None] == src[None, :]) & ~np.eye(n, dtype=bool)
            other = src[:, None] != src[None, :]
            it = e.ev_items[e.q_ev]
            inc = np.zeros((n, e.K)); rr, cc = np.nonzero(it >= 0); inc[rr, it[rr, cc]] = 1
            ov = (inc @ inc.T) > 0
            ovl_in = other & ov; ovl_out = other & ~ov
            unkq = e.unk[src]
            for l in range(nl):
                for h in range(nh + 1):
                    Am = att[l][bi, h, :n, :n]
                    a1x[l][h].append(att[l][bi, h, QI, :n])
                    for key, mi, mo, rows in (('src', same, other, None), ('src_unk', same, other, unkq),
                                              ('ovl', ovl_in, ovl_out, None)):
                        ni, no = mi.sum(1), mo.sum(1)
                        okr = (ni > 0) & (no > 0)
                        if rows is not None: okr &= rows
                        if okr.sum() == 0: continue
                        ri = (Am * mi).sum(1)[okr] / ni[okr]; ro = (Am * mo).sum(1)[okr] / no[okr]
                        acc[key][l, h] += [np.mean(ri / np.maximum(ro, 1e-12)), 1]
                        if key in pool:
                            pool[key][l, h] += [(Am * mi).sum(), ni.sum(), (Am * mo).sum(), no.sum()]
    y = np.concatenate(a1y)
    out = dict(A1_r=[[float(np.corrcoef(np.concatenate(a1x[l][h]), y)[0, 1]) for h in range(nh + 1)] for l in range(nl)])
    for k, v in acc.items():
        out['A_' + k] = (v[..., 0] / np.maximum(v[..., 1], 1)).tolist()
    for k, v in pool.items():
        out['P_' + k] = ((v[..., 0] / v[..., 1]) / (v[..., 2] / v[..., 3])).tolist()
    return out


att = {k: attn_stats(m) for k, m in nets.items()}
print(f'attn done {time.time() - t0:.0f}s', flush=True)
json.dump(dict(net=NET, sites=SITES, targets=TGT, probes=probes, attn=att), open(f'out/locate_{NET}.json', 'w'), indent=1)
