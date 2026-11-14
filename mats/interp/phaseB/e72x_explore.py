"""E72 exploration (eighth agent; DEV NETS ONLY, disclosed in the E72 amendment; not a registered test): constrained-Bayes (CB)
readers of block 1's own-source pooled statistic, as a ladder that removes one idealisation at a time.
For a net and its pooling heads (E63: summary-row mass on the source's own labelled tokens >= 0.5):
  M0  uniform pool, even capped feature only: T = mean_j g_j with g_j = (1 + r_j^2 / c^2)^(-1/2), c the net's median cap scale;
  M1  reduced pooling model (e68_rpm.py): per pooling head h, w_j = softmax_j((L_h + K_h r_j) g_j), T_h = (sum w g, sum w r g),
      with L_h, K_h the head's mean norm-scaled own logit and signed coefficient (E63) and c as in M0;
  M2  real weights: the net's block-1 attention of the summary row over its four own labelled keys, renormalised to sum 1, and the
      real per-token g_j = A_j / n_j (lngeom); per pooling head T_h = (sum w g, sum w r g);
  M3  real pooled values: per pooling head the renormalised own-key pooled value vector (dh numbers).
The reader of T: p(v | T) from a classifier (histogram gradient boosting, uniform prior over the three levels) trained on simulated
sources; for M2/M3 the simulated sources sit in full simulated episodes (random codes, the net's own block 1). CB combination over the
27 variance combinations with the exact query-reading marginal (e53_cb); the effective weight of the pair's source is d mean / d o*
by central differences with T held fixed (the statistic's own dependence on o* through the summary row's query is ignored and is
measured separately: dT_do = the RMS change of T per unit o*, against T's standard deviation over episodes).
Output per net: the net's rom (registered evaluation file), the four models' CB rom on E12's pairs (large, middle), heads, c, rho.
usage: python e72x_explore.py OUTJSON TAG=RUNDIR:EVALFILE [...]
"""
import json, math, os, sys, time
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import world as W  # noqa: E402
import e63_run as E63  # noqa: E402
import e68_rpm as RPM  # noqa: E402
import lngeom as LG  # noqa: E402
import pfn_b1hw as BH  # noqa: E402

BH.install()
torch.set_num_threads(1)
S, Q, QL, T = 3, 5, 4, 16
SUMM, LABS = E63.SUMM, E63.LABS
OC, YC = 32, 33
NSIM = int(os.environ.get('E72_NSIM', 200_000))


def heads_of(model, arm):
    z = np.load('results/e27_data.npz'); sl = slice(0, 2000)
    X, v = E63.F.make_X(z['o'][sl, 0], z['y'][sl], z['semb'][sl], z['qemb'][sl], arm)
    A, qh, blk = E63.block1_parts(model, X); H = A.shape[1]
    P = np.stack([A[:, :, SUMM[s], LABS[s]].sum(-1) for s in range(S)], 1).mean((0, 1))
    L, ratio, odd = E63.h3_logits(model, X, qh, blk)
    own_i = [[QL * s + q for q in range(QL)] for s in range(S)]
    Lown = np.stack([L[:, :, s, own_i[s]] for s in range(S)], 1); K = np.stack([odd[:, :, s, own_i[s]] for s in range(S)], 1)
    g = LG.geom(model, X)
    pool = [h for h in range(H) if P[h] >= 0.5]
    return dict(P_h=[float(x) for x in P], pool=pool, L=[float(Lown[:, :, h].mean()) for h in range(H)], K=[float(K[:, :, h].mean()) for h in range(H)],
                c=g['c_median'], rho_abs=g['rho_abs_median'], A_cv=g['A_cv'])


def block1_own(model, X):
    """Per source: renormalised own-key attention (B, S, H, 4), real g (B, S, 4), r (B, S, 4), own-key values (B, S, H, 4, dh)."""
    blk = model.blocks[0]; Xt = torch.from_numpy(X).float(); B = X.shape[0]
    allow = torch.ones(B, T, T, dtype=torch.bool); allow[:, :, T - 1] = False; allow[:, T - 1, T - 1] = True
    with torch.no_grad():
        x = model.inp(Xt); h = blk.ln1(x)
        q, k, v = blk.qkv(h).split(blk.h * blk.dh, -1)
        sp = lambda t: t.view(B, T, blk.h, blk.dh).transpose(1, 2)
        q, k, v = sp(q), sp(k), sp(v)
        a = (q @ k.transpose(-1, -2) / math.sqrt(blk.dh)).masked_fill(~allow[:, None], float('-inf')).softmax(-1).numpy()
        v = v.numpy()
    Wm = model.inp.weight.detach().numpy().astype(np.float64); bm = model.inp.bias.detach().numpy().astype(np.float64)
    lab = [j for s in range(S) for j in LABS[s]]
    Xl = X[:, lab, :].astype(np.float64); X0 = Xl.copy(); X0[..., OC] = X0[..., YC]
    a0 = X0 @ Wm.T + bm; a0 = a0 - a0.mean(-1, keepdims=True); x1 = Xl @ Wm.T + bm; x1 = x1 - x1.mean(-1, keepdims=True)
    g = (np.linalg.norm(a0, axis=-1) / np.linalg.norm(x1, axis=-1)).reshape(B, S, QL)
    r = (Xl[..., OC] - Xl[..., YC]).reshape(B, S, QL)
    w = np.stack([a[:, :, SUMM[s], LABS[s]] for s in range(S)], 1)                         # (B, S, H, 4)
    w = w / w.sum(-1, keepdims=True)
    vals = np.stack([v[:, :, LABS[s], :] for s in range(S)], 1)                             # (B, S, H, 4, dh)
    return w, g, r, vals


def feats(model, X, pool, kind):
    out = []
    for i in range(0, len(X), 4000):
        w, g, r, vals = block1_own(model, X[i:i + 4000])
        if kind == 'M2':
            f = [np.stack([(w[:, :, h] * g).sum(-1), (w[:, :, h] * r * g).sum(-1)], -1) for h in pool]
        else:
            f = [(w[:, :, h, :, None] * vals[:, :, h]).sum(-2) for h in pool]
        out.append(np.concatenate(f, -1))
    return np.concatenate(out)                                                              # (n, S, F)


def sim_tokens(n, seed, arm):
    rng = np.random.default_rng(seed)
    d = W.sample_np(n, rng)
    se = rng.standard_normal((n, S, 16)) / 4; qe = rng.standard_normal((n, Q, 16)) / 4
    X, _ = E63.F.make_X(d['o'], d['y'], se, qe, arm)
    return X, d['vidx']


def fit(Xf, y):
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, random_state=0)
    clf.fit(Xf, y)
    return clf


def cb_rom_feats(clf, F0, F1, z):
    """CB rom from per-source features of the pair episodes (n, S, F), versions 0 (spread) and 1 (big miss)."""
    src, rg = z['src'], z['range']; rows = np.arange(len(src)); ws = []
    for Fk, k in ((F0, 0), (F1, 1)):
        n = len(Fk); p = clf.predict_proba(Fk.reshape(n * S, -1)).reshape(n, S, W.NV)
        Lsrc = np.log(np.maximum(p, 1e-12))
        o = z['d%d_o' % k].astype(np.float64); ostar = o[:, :, QL]
        llc = sum(Lsrc[:, s, W.COMB[:, s]] for s in range(S))

        def mean_at(os_):
            quadq = np.einsum('ns,pst,nt->np', os_, W.SIGQI, os_)
            lw = W.LOGPRIOR[None] + llc - 0.5 * quadq - 0.5 * W.LOGDETQ[None]
            lw = lw - lw.max(1, keepdims=True); ww = np.exp(lw); ww /= ww.sum(1, keepdims=True)
            return (ww * (np.einsum('ps,ns->np', 1.0 / W.VC, os_) / W.PREC[None])).sum(1)
        hh = 1e-3; op = ostar.copy(); om = ostar.copy(); op[rows, src] += hh; om[rows, src] -= hh
        ws.append((mean_at(op) - mean_at(om)) / (2 * hh))
    Wb = z['WB0'][rows, src]
    return {rr: float((ws[1] - ws[0])[rg == rr].sum() / Wb[rg == rr].sum()) for rr in ('large', 'middle')}


def main():
    outp = sys.argv[1]; res = json.load(open(outp)) if os.path.exists(outp) else {}
    z = np.load('results/e12_pairs.npz')
    for spec in sys.argv[2:]:
        tag, rest = spec.split('=', 1); run, evf = rest.split(':', 1)
        if tag in res:
            continue
        t0 = time.time(); model, arm, meta = E63.load(run, None); evz = np.load(evf, allow_pickle=True)
        hd = heads_of(model, arm); o = dict(net_rom_large=float(evz['rom_large']), net_rom_middle=float(evz['rom_middle']), gap=float(evz['gap']), **hd)
        c = hd['c']; pool = hd['pool']
        PX = []
        for k in (0, 1):                                                              # the pairs' tokens in the net's format
            ps, pq = E63.F.ids_from_X(z['X%d' % k])
            PX.append(E63.F.make_X(z['d%d_o' % k].astype(np.float64), z['d%d_y' % k].astype(np.float64), ps, pq, arm)[0])
        o['M0'] = RPM.rom_of(RPM.fit_reader(lambda r: RPM.rpm_stat(r, [(0.0, 0.0)], c, 'ctrl')[..., :1]), z)
        o['M1'] = RPM.rpm_rom([(hd['L'][h], hd['K'][h]) for h in pool], c, 'ctrl', z) if pool else None
        if pool:
            Xs, vidx = sim_tokens(NSIM, 7201, arm)
            for kind in ('M2', 'M3'):
                Fs = feats(model, Xs, pool, kind)
                clf = fit(Fs.reshape(NSIM * S, -1), vidx.reshape(-1))
                F0 = feats(model, PX[0], pool, kind); F1 = feats(model, PX[1], pool, kind)
                o[kind] = cb_rom_feats(clf, F0, F1, z)
                if kind == 'M2':                                                      # the statistic's dependence on o*
                    Xp = PX[0].copy()
                    for s in range(S):
                        Xp[:, SUMM[s], OC] += 0.1
                    Fp = feats(model, Xp, pool, kind)
                    o['dT_do_over_sd'] = [float(np.sqrt(((Fp - F0)[..., j] ** 2).mean()) / 0.1 / Fs[..., j].std()) for j in range(Fs.shape[-1])]
        o['seconds'] = time.time() - t0
        res[tag] = o
        json.dump(res, open(outp, 'w'), indent=1)
        print(tag, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items() if k not in ('P_h',)}), flush=True)


if __name__ == '__main__':
    main()
