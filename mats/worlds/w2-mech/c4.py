"""C4 set: paired attention (A4), C4 attention structure from Aq, and activation patching between pair members.
usage: python c4.py NET -> out/c4_NET.json"""
import json, pickle, sys, time
import numpy as np
import torch
import mech as Mx

NET = sys.argv[1]
t0 = time.time()
ep, d = Mx.c4_set()
B = pickle.load(open('out/bayes_c4.pkl', 'rb'))
eps = ep['eps']
X = torch.as_tensor(ep['X']); M = torch.as_tensor(ep['mask'])
E, T, _ = X.shape; QI = T - 1
mem = d['meta_member'].astype(int); pair = d['meta_pair'].astype(int)
i0 = np.flatnonzero(mem == 0); i1 = np.flatnonzero(mem == 1)
assert (pair[i0] == pair[i1]).all()

# groups (positions aligned across members; check that only EV y-values differ)
G = {k: np.zeros((E, T), bool) for k in ('QQ', 'EV', 'SRC', 'QT', 'CTX', 'Aq', 'Bq', 'AEV', 'BEV')}
for i, e in enumerate(eps):
    A, Bs = e.meta['A'], e.meta['B']
    it0 = e.ev_items[e.q_ev][:, 0]
    isA = e.q_src == A; isB = e.q_src == Bs; onq = it0 == e.query
    G['Aq'][i, :e.n] = isA & onq; G['Bq'][i, :e.n] = isB & onq
    G['AEV'][i, :e.n] = isA & ~onq; G['BEV'][i, :e.n] = isB & ~onq
    G['QQ'][i] = G['Aq'][i] | G['Bq'][i]; G['EV'][i] = G['AEV'][i] | G['BEV'][i]
    G['SRC'][i] = G['QQ'][i] | G['EV'][i]; G['QT'][i, QI] = True; G['CTX'][i, :e.n] = True
assert G['Aq'].sum(1).max() == 1 and G['Bq'].sum(1).min() == 1
diff = (X[i0] != X[i1]).any(-1).numpy()
assert not (diff & ~G['EV'][i0]).any(), 'members differ outside EV'
print('pair alignment ok; EV tokens differing:', diff.sum(1).mean(), flush=True)

betaA = np.array([b['beta'][G['Aq'][i, :eps[i].n]][0] for i, b in enumerate(B)])
betaB = np.array([b['beta'][G['Bq'][i, :eps[i].n]][0] for i, b in enumerate(B)])
qmB = np.array([b['qmean'] for b in B])
res = dict(net=NET)


def run_all(model, patch=None, keep_attn=False, idx=None):
    idx = np.arange(E) if idx is None else idx
    outs = []
    for a in range(0, len(idx), 100):
        sl = idx[a:a + 100]
        p = None if patch is None else {l: (torch.as_tensor(pos[a:a + 100]), st[a:a + 100]) for l, (pos, st) in patch.items()}
        (m, ls, _), s, att = Mx.forward(model, X[sl], M[sl], keep_sites=patch is None and not keep_attn, keep_attn=keep_attn,
                                        patch=p)
        outs.append((m.numpy(), ls.numpy(), s, att))
    return outs


for tag, trained in (('T', True), ('U', False)):
    model = Mx.load_net(NET, trained=trained)
    nl, nh = len(model.blocks), model.blocks[0].h
    # ---------------- attention: A4 and Aq -> own-source structure
    att_out = run_all(model, keep_attn=True)
    att = [np.concatenate([o[3][l].numpy() for o in att_out]) for l in range(nl)]        # (E, h, T, T)
    att = [np.concatenate([a, a.mean(1, keepdims=True)], 1) for a in att]
    lb = np.log(np.abs(betaA)) - np.log(np.abs(betaB))
    dbeta = lb[i0] - lb[i1]
    A4 = []; AQ = []
    for l in range(nl):
        rowl, rowq = [], []
        for h in range(nh + 1):
            aq = att[l][:, h, QI]
            dd = np.log(aq[G['Aq']]) - np.log(aq[G['Bq']])
            ddp = dd[i0] - dd[i1]
            rowl.append(dict(sign_agree=float((np.sign(ddp) == np.sign(dbeta)).mean()),
                             r=float(np.corrcoef(ddp, dbeta)[0, 1]) if ddp.std() > 0 else float('nan'),
                             mean_ddp=float(ddp.mean()), qt_mass_QQ=float(aq[G['QQ']].sum() / E),
                             qt_mass_EV=float(aq[G['EV']].sum() / E)))
            # Aq as attention query: per-key mass on own-source other quotes (AEV) vs other tokens (ex. self, Bq)
            ar = att[l][:, h][G['Aq']]                                  # (E, T) row of Aq
            own = (ar * G['AEV']).sum(1) / G['AEV'].sum(1)
            oth_mask = G['CTX'] & ~G['AEV'] & ~G['Aq']
            oth = (ar * oth_mask).sum(1) / oth_mask.sum(1)
            br = att[l][:, h][G['Bq']]
            ownB = (br * G['BEV']).sum(1) / G['BEV'].sum(1)
            othB_mask = G['CTX'] & ~G['BEV'] & ~G['Bq']
            othB = (br * othB_mask).sum(1) / othB_mask.sum(1)
            # same-item: Aq -> Bq per-key vs Aq -> other tokens
            rowq.append(dict(Aq_ownsrc_ratio=float(np.mean(own / oth)), Bq_ownsrc_ratio=float(np.mean(ownB / othB)),
                             Aq_to_Bq_ratio=float(np.mean(ar[G['Bq']] / oth)),
                             Aq_ownsrc_mass=float(own.mean() * 12)))
        A4.append(rowl); AQ.append(rowq)
    res[f'A4_{tag}'] = A4; res[f'AQ_{tag}'] = AQ
    print(tag, 'attention done', f'{time.time() - t0:.0f}s', flush=True)
    del att, att_out
    # ---------------- patching
    base = run_all(model)
    mean0 = np.concatenate([o[0] for o in base]); ls0 = np.concatenate([o[1] for o in base])
    sites = {}
    for k in ['L0'] + [f'L{l}' for l in range(1, nl + 1)]:
        sites[k] = torch.cat([o[2][k] for o in base])
    del base
    other = np.empty(E, int); other[i0] = i1; other[i1] = i0
    dm = mean0[other] - mean0; dl = ls0[other] - ls0
    keep = np.abs(dm) >= 0.05
    patch_res = {}
    for g in ('QQ', 'EV', 'SRC', 'QT', 'CTX', 'Aq', 'Bq'):
        pos = G[g]
        rows = []
        for l in range(0, nl + 1):
            st = sites['L0' if l == 0 else f'L{l}'][other]
            out = run_all(model, patch={l: (pos, st)})
            mp = np.concatenate([o[0] for o in out]); lp = np.concatenate([o[1] for o in out])
            f = (mp - mean0) / np.where(keep, dm, np.nan)
            fl = (lp - ls0) / np.where(np.abs(dl) >= 0.05, dl, np.nan)
            rows.append(dict(layer=l, f_mean_median=float(np.nanmedian(f)), f_mean_q25=float(np.nanpercentile(f, 25)),
                             f_mean_q75=float(np.nanpercentile(f, 75)), f_logsd_median=float(np.nanmedian(fl)),
                             n=int(keep.sum()), n_sd=int(np.isfinite(fl).sum())))
        patch_res[g] = rows
    res[f'patch_{tag}'] = patch_res
    # model shift vs Bayes shift
    res[f'shift_{tag}'] = dict(model_shift_median=float(np.median(mean0[i0] - mean0[i1])),
                               bayes_shift_median=float(np.median(qmB[i0] - qmB[i1])),
                               frac=float(np.median((mean0[i0] - mean0[i1]) / (qmB[i0] - qmB[i1]))),
                               n_keep=int(keep.sum()))
    print(tag, 'patching done', f'{time.time() - t0:.0f}s', flush=True)
res['bayes'] = dict(betaA_m0=float(np.median(betaA[i0])), betaB_m0=float(np.median(betaB[i0])),
                    betaA_m1=float(np.median(betaA[i1])), betaB_m1=float(np.median(betaB[i1])))
json.dump(res, open(f'out/c4_{NET}.json', 'w'), indent=1)
