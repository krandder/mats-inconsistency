"""Amendment 1: edge ablation (block 2/3, Aq<-AEV, Bq<-BEV), read-out equalisation (QT->Aq/Bq), probe transfer.
usage: python intervene.py NET -> out/intervene_NET.json"""
import json, pickle, sys, time
import numpy as np
import torch
import mech as Mx

NET = sys.argv[1]
READ = 3                    # block 4 (0-indexed 3) is the read block in both nets
t0 = time.time()
ep, d = Mx.c4_set()
eps = ep['eps']
X = torch.as_tensor(ep['X']); M = torch.as_tensor(ep['mask'])
E, T, _ = X.shape; QI = T - 1
mem = d['meta_member'].astype(int); pair = d['meta_pair'].astype(int)
i0 = np.flatnonzero(mem == 0); i1 = np.flatnonzero(mem == 1)
pos = {k: np.zeros(E, int) for k in ('Aq', 'Bq')}
AEV = np.zeros((E, T), bool); BEV = np.zeros((E, T), bool); AB = np.zeros((E, T), bool); CTX = np.zeros((E, T), bool)
for i, e in enumerate(eps):
    A, Bs = e.meta['A'], e.meta['B']
    onq = e.ev_items[e.q_ev][:, 0] == e.query
    isA = e.q_src == A; isB = e.q_src == Bs
    pos['Aq'][i] = np.flatnonzero(isA & onq)[0]; pos['Bq'][i] = np.flatnonzero(isB & onq)[0]
    AEV[i, :e.n] = isA & ~onq; BEV[i, :e.n] = isB & ~onq; AB[i, :e.n] = isA | isB; CTX[i, :e.n] = True
# control keys: 12 random non-A/B context tokens per row, same for both members of a pair
CA = np.zeros((E, T), bool); CB = np.zeros((E, T), bool)
for p in np.unique(pair):
    r = np.random.default_rng(1000 + int(p))
    ii = np.flatnonzero(pair == p)
    cand = np.flatnonzero(CTX[ii[0]] & ~AB[ii[0]])
    ca = r.choice(cand, 12, replace=False); cb = r.choice(cand, 12, replace=False)
    CA[ii[:, None], ca[None]] = True; CB[ii[:, None], cb[None]] = True

pb = Mx.Hn._unragged(d)
g = (np.ones((E, 1)), d['mean_freq_only'][:, None], d['var_freq_only'][:, None])
klf = Mx.Hn.kl_bayes_model(pb, g)


def drop_mask(keysA, keysB, sl):
    Dm = torch.zeros(len(sl), T, T, dtype=torch.bool)
    ar = torch.arange(len(sl))
    Dm[ar, torch.as_tensor(pos['Aq'][sl])] = torch.as_tensor(keysA[sl])
    Dm[ar, torch.as_tensor(pos['Bq'][sl])] = torch.as_tensor(keysB[sl])
    return Dm


def run(model, edit_factory=None):
    ms, preds = [], []
    for a in range(0, E, 100):
        sl = np.arange(a, min(E, a + 100))
        ed = edit_factory(sl) if edit_factory else None
        (m, ls, (w, mu, v)), _, _ = Mx.forward(model, X[sl], M[sl], keep_sites=False, attn_edit=ed)
        ms.append(m.numpy()); preds.append((w.numpy(), mu.numpy(), v.numpy()))
    m = np.concatenate(ms)
    pm = tuple(np.concatenate([p[k] for p in preds]).astype(np.float64) for k in range(3))
    return m, pm


def ablate(blocks, keysA, keysB):
    def fac(sl):
        Dm = drop_mask(keysA, keysB, sl)[:, None]

        def ed(li, a):
            if li not in blocks:
                return a
            a = a.masked_fill(Dm, 0.0)
            return a / a.sum(-1, keepdim=True)
        return ed
    return fac


def equalise(block):
    def fac(sl):
        ar = torch.arange(len(sl)); pa = torch.as_tensor(pos['Aq'][sl]); pq = torch.as_tensor(pos['Bq'][sl])

        def ed(li, a):
            if li != block:
                return a
            a = a.clone()
            mA = a[ar, :, QI, pa]; mB = a[ar, :, QI, pq]
            a[ar, :, QI, pa] = 0.5 * (mA + mB); a[ar, :, QI, pq] = 0.5 * (mA + mB)
            return a
        return ed
    return fac


model = Mx.load_net(NET)
base_m, base_p = run(model)
dm = base_m[i0] - base_m[i1]
keep = np.abs(dm) >= 0.05


def score(m, p):
    kl = Mx.Hn.kl_bayes_model(pb, p)
    clos = 1 - kl / klf
    sf = (m[i0] - m[i1])[keep] / dm[keep]
    return dict(shift_frac_median=float(np.median(sf)), shift_frac_q25=float(np.percentile(sf, 25)),
                shift_frac_q75=float(np.percentile(sf, 75)), c4_closure_frac=float((clos >= 0.5).mean()),
                mean_kl=float(kl.mean()))


res = dict(net=NET, n_pairs_kept=int(keep.sum()), base=score(base_m, base_p))
nb = len(model.blocks)
conds = {
    'edge_b3': ablate({2}, AEV, BEV), 'ctrl_b3': ablate({2}, CA, CB),
    'edge_b2': ablate({1}, AEV, BEV), 'ctrl_b2': ablate({1}, CA, CB),
    'edge_b2b3': ablate({1, 2}, AEV, BEV), 'ctrl_b2b3': ablate({1, 2}, CA, CB),
    'edge_b1b2b3': ablate({0, 1, 2}, AEV, BEV),
    'equalise_read_b4': equalise(READ),
}
if nb > 4:
    conds['equalise_b5'] = equalise(4)
    def both(sl):
        f1, f2 = equalise(READ)(sl), equalise(4)(sl)
        return lambda li, a: f2(li, f1(li, a))
    conds['equalise_b4b5'] = both
for k, fac in conds.items():
    res[k] = score(*run(model, fac))
    print(k, res[k], f'{time.time() - t0:.0f}s', flush=True)

# ---------------- probe transfer
pe = Mx.probe_set()
Bp = pickle.load(open('out/bayes_probe.pkl', 'rb'))
peps = pe['eps']
rng = np.random.default_rng(0)
U_e, U_t, U_y, S_e, S_t, S_y = [], [], [], [], [], []
for i, (e, b) in enumerate(zip(peps, Bp)):
    src = e.q_src; typ = e.ev_typ[e.q_ev]
    u = np.flatnonzero(e.unk[src]); s = np.flatnonzero(typ != 1)
    if len(u) > 48: u = np.sort(rng.choice(u, 48, replace=False))
    if len(s) > 48: s = np.sort(rng.choice(s, 48, replace=False))
    U_e.append(np.full(len(u), i)); U_t.append(u); U_y.append(np.stack([b['pH'][src[u]], b['logprec'][src[u]]], 1))
    S_e.append(np.full(len(s), i)); S_t.append(s); S_y.append(np.stack([b['resid'][s] ** 2, np.abs(b['resid'][s])], 1))
U_e, U_t, U_y, S_e, S_t, S_y = map(np.concatenate, (U_e, U_t, U_y, S_e, S_t, S_y))
Xp = torch.as_tensor(pe['X']); Mp = torch.as_tensor(pe['mask'])
sites_p = None
for a in range(0, len(peps), 100):
    _, s, _ = Mx.forward(model, Xp[a:a + 100], Mp[a:a + 100])
    s = {k: v.numpy() for k, v in s.items()}
    sites_p = s if sites_p is None else {k: np.concatenate([sites_p[k], s[k]]) for k in s}
sites_c = None
for a in range(0, E, 100):
    _, s, _ = Mx.forward(model, X[a:a + 100], M[a:a + 100])
    s = {k: v.numpy() for k, v in s.items()}
    sites_c = s if sites_c is None else {k: np.concatenate([sites_c[k], s[k]]) for k in s}
Bc = pickle.load(open('out/bayes_c4.pkl', 'rb'))
ar = np.arange(E)
pHA = np.array([b['pH'][e.meta['A']] for e, b in zip(eps, Bc)]); pHB = np.array([b['pH'][e.meta['B']] for e, b in zip(eps, Bc)])
lpA = np.array([b['logprec'][e.meta['A']] for e, b in zip(eps, Bc)]); lpB = np.array([b['logprec'][e.meta['B']] for e, b in zip(eps, Bc)])
ev_idx = [np.flatnonzero(AEV[i] | BEV[i]) for i in range(E)]
ev_r2 = np.concatenate([Bc[i]['resid'][ev_idx[i]] ** 2 for i in range(E)])
trp = []
for site in Mx.site_names(nb):
    trU = U_e < 1500; trS = S_e < 1500
    fU = Mx.ridge_fit(sites_p[site][U_e[trU], U_t[trU]], U_y[trU], U_e[trU])
    fS = Mx.ridge_fit(sites_p[site][S_e[trS], S_t[trS]], S_y[trS], S_e[trS])
    dA = fU(sites_c[site][ar, pos['Aq']]); dB = fU(sites_c[site][ar, pos['Bq']])
    evf = np.concatenate([sites_c[site][i, ev_idx[i]] for i in range(E)])
    dS = fS(evf)
    row = dict(site=site,
               r_pH=float(np.corrcoef(np.r_[dA[:, 0], dB[:, 0]], np.r_[pHA, pHB])[0, 1]),
               r_logprec=float(np.corrcoef(np.r_[dA[:, 1], dB[:, 1]], np.r_[lpA, lpB])[0, 1]),
               pHA_m0_minus_m1=float(dA[i0, 0].mean() - dA[i1, 0].mean()),
               bayes_pHA_m0_minus_m1=float(pHA[i0].mean() - pHA[i1].mean()),
               r_resid2_EV=float(np.corrcoef(dS[:, 0], ev_r2)[0, 1]))
    trp.append(row)
    print(row, flush=True)
res['probe_transfer'] = trp
json.dump(res, open(f'out/intervene_{NET}.json', 'w'), indent=1)
print('done', f'{time.time() - t0:.0f}s')
