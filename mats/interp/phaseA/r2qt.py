"""Amendment 6: information-matched query-token oracle O_q0m on one net. usage: python r2qt.py MODEL
q0 at QT reads six permutation-invariant features f(c) that do not determine c (r2qtinfo.py), so O_c over-informs it.
Here c'' = k-th nearest neighbour (k = 1..8) of f(c') among 4,000,000 prior episodes (standardised features); x'' is
exact given c'' (qemb_4 kept; other IDs, labelled y, residual directions fresh). O_q0m = neighbour 1; M_q0m = mean
of the 8 neighbours' QT states + N(0, (1 - 1/8) Sigma_w) (Sigma_w from 2,000 fit episodes matched the same way).
Evaluated at every QT site and (O_q0m, with LAB/QQ from r2run's O_q0 draws, same seeds) at the full cuts.
Writes r2/qt_MODEL.json and r2/qt_MODEL_D.npz (per-draw G_V - G_p, same eval draws as r2run)."""
import json, sys, time
import numpy as np
import torch
from scipy.spatial import cKDTree
sys.path.insert(0, '/home/agent/workspaces/mats/work/cm-interp')
from lib import tp, oracle as O          # noqa: E402
from lib import r1est as E               # noqa: E402
from r2qtpool import feats_c             # noqa: E402

torch.set_num_threads(1)
NAME = sys.argv[1]
NC, M, NW, K = 6000, 8, 2000, 8
T0 = time.time()
log = lambda *a: print(f'[{NAME} {time.time() - T0:6.0f}s]', *a, flush=True)
model, meta, ev = tp.load(NAME)
L = meta['depth']
MULTI = [('post', k) for k in range(1, L)]
FINAL = [('attn', L), ('post', L)]
FPOST = ('post', L)
ALLCUTS = MULTI + FINAL
want = {c: None for c in MULTI}; want.update({c: [tp.QI] for c in FINAL})
cname = lambda c: f"{'h' if c[0] == 'post' else 'a'}{c[1]}"


def prep(seed, n, M_, with_G=True):
    d0 = tp.sample_episodes(n, seed); cs0 = tp.cell_stats(d0)
    if M_:
        d = tp.cond_draws(d0, M_, seed + 1)
    else:
        d = d0; d['cell'] = np.arange(n)
    X, V = tp.tokens(d)
    head, st, _ = tp.run_real(model, X, V, want)
    GQ = tp.GQuad(cs0) if with_G else None
    Gp = GQ(head, d['cell'], chunk=1000).reshape(n, max(M_, 1)) if with_G else None
    return dict(d=d, V=V, st=st, GQ=GQ, cell=d['cell'], Gp=Gp, n=n)


E_ = prep(70000, NC, M)                                    # identical to r2run's eval draws
F0 = tp.sample_episodes(20000, 60000)
Fsub = {k: v[:NW] for k, v in F0.items()}
pool = np.load('/tmp/r2scratch/qtpool.npz')
PQ, PS, Pf = pool['Q'], pool['S'], pool['f']
mu, sd = Pf.mean(0), Pf.std(0)
tree = cKDTree((Pf - mu) / sd)
log('data + pool ready')


def matched_states(d, seed):
    """K matched QT donors per draw of d -> list of {cut: (N, d) float32} and matching distances."""
    Qx, Sx, _ = O.cvars(d)
    dist, idx = tree.query((feats_c(Qx, Sx) - mu) / sd, k=K)
    out = []
    for k in range(K):
        rng = np.random.default_rng([seed, k])
        N = len(Qx)
        semb = rng.standard_normal((N, tp.S, tp.DS)) / 4
        qemb = rng.standard_normal((N, tp.Q, tp.DS)) / 4
        qemb[:, tp.QL] = d['qemb'][:, tp.QL]
        y = rng.standard_normal((N, tp.Q)); y[:, tp.QL] = 0.0
        x2 = O.build(PQ[idx[:, k]], PS[idx[:, k]], y, semb, qemb, rng)
        X, V = tp.tokens(x2)
        out.append(O.forward_token(model, X, V, tp.QI, ALLCUTS, bs=1000))
    return out, dist


def qc(cut, h, ln=False):
    return tp.to_q(np.asarray(h, np.float64), True if ln else False)


res = dict(model=NAME, direct={}, checks={})
D = {}


def eval_site(cut, idx, Hs):
    h = E_['st'][cut].copy()
    h[:, idx] = Hs
    head = tp.forward_qt(model, h, cut) if cut in FINAL else tp.forward_from(model, h, cut, E_['V'], bs=1000)
    return E_['GQ'](head, E_['cell'], chunk=1000).reshape(NC, M)


def record(key, G):
    D[key] = G - E_['Gp']
    r = E.direct(G, E_['Gp'], B=400)
    res['direct'][key] = r
    return r


# Sigma_w on the fit subset (within-match covariance of the QT state, quotient coordinates)
drF, distF = matched_states(Fsub, 76000)
SIGW = {}
for cut in ALLCUTS:
    for ln in ((False, True) if cut == FPOST else (False,)):
        H = np.stack([qc(cut, dr[cut], ln) for dr in drF], 0)              # (K, N, dm)
        R_ = H - H.mean(0, keepdims=True)
        C = np.einsum('kni,knj->ij', R_, R_) / (R_.shape[1] * (K - 1))
        SIGW[(cut, ln)] = 0.5 * (C + C.T)
res['checks']['sigma_w_trace'] = {f'{cname(c)}{"_LN" if ln else ""}': float(np.trace(C)) for (c, ln), C in SIGW.items()}
del drF
drE, distE = matched_states(E_['d'], 76100)
res['checks']['match_dist'] = dict(median=float(np.median(distE[:, 0])), q90=float(np.quantile(distE[:, 0], .9)),
                                   median_k8=float(np.median(distE[:, -1])))
log('matched donors ready', res['checks'])
rngM = np.random.default_rng(76200)
for cut in ALLCUTS:
    site = f'{cname(cut)}-QT'
    idx = [tp.QI] if cut in MULTI else [0]
    record(f'{site}|O_q0m', eval_site(cut, idx, drE[0][cut][:, None]))
    for ln in ((False, True) if cut == FPOST else (False,)):
        mean = np.mean([qc(cut, dr[cut], ln) for dr in drE], 0)
        C = SIGW[(cut, ln)] * (1 - 1 / K)
        Lc = np.linalg.cholesky(C + 1e-9 * np.trace(C) / len(C) * np.eye(len(C)))
        noisy = mean + rngM.standard_normal(mean.shape) @ Lc.T
        sfx = '_LN' if ln else ''
        record(f'{site}|M_q0m{sfx}', eval_site(cut, idx, tp.to_q(noisy, ln)[:, None]))
        record(f'{site}|M_q0mmean{sfx}', eval_site(cut, idx, tp.to_q(mean, ln)[:, None]))
    log(site, {k.split('|')[1]: round(v['delta'], 4) for k, v in res['direct'].items() if k.startswith(site)})

# full cuts: LAB/QQ from r2run's O_q0 draws (seed 70300, want_own=True: identical RNG use), QT from O_q0m
bx = O.beta_of(*O.cvars(E_['d'])[:2])
for cut in MULTI:
    h_all = {}
    for t in range(tp.T - 1):
        rng_t = np.random.default_rng([70300, t])
        x2, _ = O.draw(E_['d'], t, 'q0', rng_t, K=32, beta_x=bx[:, tp.TOK_SRC[t]], want_own=True)
        X, V = tp.tokens(x2)
        h_all[t] = O.forward_token(model, X, V, t, [cut], bs=1000)[cut]
    Hs = np.stack([h_all[t] for t in range(tp.T - 1)] + [drE[0][cut]], 1)
    record(f'{cname(cut)}-ALL|O_q0m', eval_site(cut, list(range(tp.T)), Hs))
    log(f'{cname(cut)}-ALL', round(res['direct'][f'{cname(cut)}-ALL|O_q0m']['delta'], 4))
res['seconds'] = time.time() - T0
json.dump(res, open(f'/home/agent/workspaces/mats/work/cm-interp/r2/qt_{NAME}.json', 'w'), default=float)
np.savez_compressed(f'/home/agent/workspaces/mats/work/cm-interp/r2/qt_{NAME}_D.npz', **{k: v.astype(np.float32) for k, v in D.items()})
log('done')
