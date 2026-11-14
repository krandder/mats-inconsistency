"""E72 Part 1 (PREREG-B amendment 80): constrained-Bayes (CB) readers of block 1's own-source pooled statistic, per net.
Block 1's attention is taken from the net's own forward (its masks or fixed weights: nocons, own, uni as built), on E27's first 2,000
bases for the pooling-head classification (P_h = summary-row mass on the source's own four labelled tokens >= 0.5) and on the
simulated and pair episodes for the pooled statistics. Models (a ladder; each a statistic T of one source):
  M0  even capped feature only, uniform pool: T = mean_j (1 + r_j^2 / c^2)^(-1/2), c the net's median cap scale (lngeom);
  M1  reduced pooling model (e68_rpm.rpm_stat): per pooling head w_j = softmax_j((L_h + K_h s_j) g_j), g_j = (1 + r_j^2 / c^2)^(-1/2),
      T_h = (sum w g, sum w s g), s = r (|r| in the mag format), L_h and K_h the head's mean norm-scaled own logit and signed
      coefficient over own keys (e63_run.h3_logits); uniform-pool nets: one head with L = K = 0;
  M2  real weights: per pooling head the net's block-1 attention of the summary row over its four own labelled keys, renormalised to
      sum 1, w_j; real g_j = A_j / n_j (the token's centred embedding norm at r = 0 over its norm); T_h = (sum w g, sum w s g);
  M3  real pooled values: per pooling head sum_j w_j v_hj, the renormalised own-key pooled value vector (dh numbers; for dh > 16 its
      top 16 principal components, fitted on the simulated sources).
Reader: p(v | T) from a histogram-gradient-boosting classifier (300 iterations, learning rate 0.1, 63 leaves; uniform prior over the
three levels) trained on simulated sources: M0 / M1 on 200,000 sources per level (e68_rpm), M2 / M3 on the 600,000 sources of
200,000 simulated episodes (world prior, random codes, the net's own format and block 1; numpy seed 7201). CB combination over the
27 variance combinations with the exact query-reading marginal (e53_cb); effective weight of the pair's source d mean / d o* by
central differences (h = 1e-3) with T held fixed; rom on E12's 5,968 matched pairs (large, middle) as E12.
usage: python e72_run.py OUTDIR TAG=RUNDIR:EVALFILE [...] -> OUTDIR/<tag>.json   (plumbing only: E72_NSIM overrides the episode
count, E72_FORCEPOOL=h,h' the pooling heads)
"""
import json, os, sys, time
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
import pfn_l2fix  # noqa: E402

pfn_l2fix.install()                      # E70 b1h4's block 2 and E64's nocons mask (loader fixes, found in E72)
torch.set_num_threads(1)
S, Q, QL, T = 3, 5, 4, 16
SUMM, LABS = E63.SUMM, E63.LABS
OC, YC = 32, 33
NSIM = int(os.environ.get('E72_NSIM', 200_000))
NPC = 16


def attn1(model, X):
    """Block 1's attention (B, H, T, T) from the net's own forward."""
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 4000):
            Xt = torch.from_numpy(X[i:i + 4000]).float(); v = torch.ones(Xt.shape[:2], dtype=torch.bool)
            out.append(model(Xt, v, want=True)[-1][0].numpy())
    return np.concatenate(out)


def heads_of(model, arm, meta):
    z = np.load('results/e27_data.npz'); sl = slice(0, 2000)
    X, v = E63.F.make_X(z['o'][sl, 0], z['y'][sl], z['semb'][sl], z['qemb'][sl], arm)
    A = attn1(model, X); H = A.shape[1]
    P = np.stack([A[:, :, SUMM[s], LABS[s]].sum(-1) for s in range(S)], 1).mean((0, 1))
    _, qh, blk = E63.block1_parts(model, X)
    L, ratio, odd = E63.h3_logits(model, X, qh, blk)
    own_i = [[QL * s + q for q in range(QL)] for s in range(S)]
    Lown = np.stack([L[:, :, s, own_i[s]] for s in range(S)], 1); K = np.stack([odd[:, :, s, own_i[s]] for s in range(S)], 1)
    g = LG.geom(model, X)
    return dict(variant=meta.get('variant'), P_h=[float(x) for x in P], pool=[h for h in range(H) if P[h] >= 0.5],
                L=[float(Lown[:, :, h].mean()) for h in range(H)], K=[float(K[:, :, h].mean()) for h in range(H)],
                c=g['c_median'], rho_abs=g['rho_abs_median'], A_cv=g['A_cv'])


def own_parts(model, X):
    A = attn1(model, X); blk = model.blocks[0]; B = X.shape[0]
    with torch.no_grad():
        q, k, v = blk.qkv(blk.ln1(model.inp(torch.from_numpy(X).float()))).split(blk.h * blk.dh, -1)
        v = v.view(B, T, blk.h, blk.dh).transpose(1, 2).numpy()
    Wm = model.inp.weight.detach().numpy().astype(np.float64); bm = model.inp.bias.detach().numpy().astype(np.float64)
    lab = [j for s in range(S) for j in LABS[s]]
    Xl = X[:, lab, :].astype(np.float64); X0 = Xl.copy(); X0[..., OC] = X0[..., YC]
    a0 = X0 @ Wm.T + bm; a0 = a0 - a0.mean(-1, keepdims=True); x1 = Xl @ Wm.T + bm; x1 = x1 - x1.mean(-1, keepdims=True)
    g = (np.linalg.norm(a0, axis=-1) / np.linalg.norm(x1, axis=-1)).reshape(B, S, QL)
    sgn = (Xl[..., OC] - Xl[..., YC]).reshape(B, S, QL)
    w = np.stack([A[:, :, SUMM[s], LABS[s]] for s in range(S)], 1); w = w / w.sum(-1, keepdims=True)    # (B, S, H, 4)
    vals = np.stack([v[:, :, LABS[s], :] for s in range(S)], 1)                                              # (B, S, H, 4, dh)
    return w, g, sgn, vals


def feats(model, X, pool):
    m2, m3 = [], []
    for i in range(0, len(X), 4000):
        w, g, s, vals = own_parts(model, X[i:i + 4000])
        m2.append(np.concatenate([np.stack([(w[:, :, h] * g).sum(-1), (w[:, :, h] * s * g).sum(-1)], -1) for h in pool], -1))
        m3.append(np.stack([(w[:, :, h, :, None] * vals[:, :, h]).sum(-2) for h in pool], 2))           # (b, S, P, dh)
    return np.concatenate(m2), np.concatenate(m3)


def fit(Xf, y):
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, random_state=0)
    return clf.fit(Xf, y)


def cb_rom_feats(clf, F0, F1, z):
    src, rg = z['src'], z['range']; rows = np.arange(len(src)); ws = []
    for Fk, k in ((F0, 0), (F1, 1)):
        n = len(Fk); p = clf.predict_proba(Fk.reshape(n * S, -1)).reshape(n, S, W.NV)
        Lsrc = np.log(np.maximum(p, 1e-12))
        ostar = z['d%d_o' % k].astype(np.float64)[:, :, QL]
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
    outdir = sys.argv[1]; os.makedirs(outdir, exist_ok=True)
    z = np.load('results/e12_pairs.npz')
    for spec in sys.argv[2:]:
        tag, rest = spec.split('=', 1); run, evf = rest.split(':', 1); outp = os.path.join(outdir, tag + '.json')
        if os.path.exists(outp):
            continue
        t0 = time.time(); model, arm, meta = E63.load(run, None); evz = np.load(evf, allow_pickle=True)
        hd = heads_of(model, arm, meta)
        o = dict(tag=tag, arm=arm, net_rom_large=float(evz['rom_large']), net_rom_middle=float(evz['rom_middle']), gap=float(evz['gap']),
                 type='A' if float(evz['rom_middle']) < 0 else 'B', **hd)
        uni = meta.get('variant') in ('uni', 'unif')
        fmt = 'mag' if arm in ('mag', 'both') else 'ctrl'
        c = hd['c']; pool = hd['pool']
        if os.environ.get('E72_FORCEPOOL'):                                  # plumbing only (step-0 weights have no pooling head)
            pool = [int(x) for x in os.environ['E72_FORCEPOOL'].split(',')]
        o['M0'] = RPM.rom_of(RPM.fit_reader(lambda r: RPM.rpm_stat(r, [(0.0, 0.0)], c, fmt)[..., :1]), z)
        h1 = [(0.0, 0.0)] if uni else [(hd['L'][h], hd['K'][h]) for h in pool]
        o['M1'] = RPM.rom_of(RPM.fit_reader(lambda r: RPM.rpm_stat(r, h1, c, fmt)), z) if h1 else None
        if pool:
            PX = []
            for k in (0, 1):
                ps, pq = E63.F.ids_from_X(z['X%d' % k])
                PX.append(E63.F.make_X(z['d%d_o' % k].astype(np.float64), z['d%d_y' % k].astype(np.float64), ps, pq, arm)[0])
            rng = np.random.default_rng(7201); d = W.sample_np(NSIM, rng)
            se = rng.standard_normal((NSIM, S, 16)) / 4; qe = rng.standard_normal((NSIM, Q, 16)) / 4
            Xs, _ = E63.F.make_X(d['o'], d['y'], se, qe, arm)
            m2s, m3s = feats(model, Xs, pool); m20, m30 = feats(model, PX[0], pool); m21, m31 = feats(model, PX[1], pool)
            y = d['vidx'].reshape(-1)
            o['M2'] = cb_rom_feats(fit(m2s.reshape(NSIM * S, -1), y), m20, m21, z)
            dh = m3s.shape[-1]
            if dh > NPC:                                                     # per pooling head, top NPC principal components
                proj = []
                for p in range(len(pool)):
                    Y = m3s[:, :, p].reshape(-1, dh); mu = Y.mean(0); _, _, Vt = np.linalg.svd(Y[::10] - mu, full_matrices=False)
                    proj.append((mu, Vt[:NPC].T))
                pr = lambda M: np.concatenate([(M[:, :, p] - proj[p][0]) @ proj[p][1] for p in range(len(pool))], -1)
                m3s, m30, m31 = pr(m3s), pr(m30), pr(m31)
            else:
                m3s, m30, m31 = (M.reshape(M.shape[0], S, -1) for M in (m3s, m30, m31))
            o['M3'] = cb_rom_feats(fit(m3s.reshape(NSIM * S, -1), y), m30, m31, z)
        o['seconds'] = time.time() - t0
        json.dump(o, open(outp + '.tmp', 'w'), indent=1); os.replace(outp + '.tmp', outp)
        print(tag, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items() if k not in ('P_h', 'L', 'K')}), flush=True)


if __name__ == '__main__':
    main()
