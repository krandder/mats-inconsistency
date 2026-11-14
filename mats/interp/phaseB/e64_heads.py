"""E64 head measures (PREREG-B amendment 72): E63's block-1 summary-head measures (amendment 71, e63_run.py) on E64's nets, with the
variant's own block-1 computation: for nocons the summary rows' mask; for codeln and both the LayerNorm of the content part only, the
code part added after it. Per net on E27's first 2,000 bases:
  M1 P_h, self, osum, olab, C_h, C_max, consensus, n_pool (as e63_run.py);
  H3 for the pooling heads: L_own (norm-scaled own logit) and dlogit(|r| = 3), with alpha, A, U and rho taken from the part of the
     token that the block-1 LayerNorm normalises (the whole embedding for nocons; the content part for codeln and both), and K_odd;
     for codeln and both also the code logit q.W_K(x_code) / sqrt(dh) of own and of other sources' labelled keys (gap, descriptive);
  M3 the per-source normaliser A's CV across episodes (the LayerNorm's input part, as above).
usage: python e64_heads.py RUN [RUN ...] -> results/e64_heads/<tag>.json
"""
import json, math, os, sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import pfn_e64 as P64  # noqa: E402

P2.build = lambda meta: P64.build_any(meta, P2, P2.fin_of(meta['ds']))
import e30_eval as E30  # noqa: E402
import e48_fmt as F  # noqa: E402

E30.F = F
torch.set_num_threads(1)
S, Q, QL, T, DS, NCODE = 3, 5, 4, 16, 16, 32
SUMM = [s * Q + QL for s in range(S)]
LABS = [[s * Q + q for q in range(QL)] for s in range(S)]
NB = 2000
OUT = os.environ.get('E64_HEADS_OUT', 'results/e64_heads')          # override: plumbing only


def main():
    os.makedirs(OUT, exist_ok=True)
    z = np.load('results/e27_data.npz'); sl = slice(0, NB)
    for run in sys.argv[1:]:
        model, arm, meta = E30.load(run); v = meta.get('variant')
        codeln = v in ('codeln', 'both'); nocons = v in ('nocons', 'both')
        X, valid = F.make_X(z['o'][sl, 0], z['y'][sl], z['semb'][sl], z['qemb'][sl], arm)
        Xt = torch.from_numpy(X).float(); B = NB
        blk = model.blocks[0]
        Wt = model.inp.weight.detach(); bt = model.inp.bias.detach()
        with torch.no_grad():
            if codeln:
                h = blk.ln1(Xt[..., NCODE:] @ Wt[:, NCODE:].T + bt) + Xt[..., :NCODE] @ Wt[:, :NCODE].T
            else:
                h = blk.ln1(model.inp(Xt))
            q, k, _ = blk.qkv(h).split(blk.h * blk.dh, -1)
            sp = lambda t: t.view(B, T, blk.h, blk.dh).transpose(1, 2)
            qh, kh = sp(q), sp(k)
            allow = torch.ones(B, T, T, dtype=torch.bool); allow[:, :, T - 1] = False; allow[:, T - 1, T - 1] = True
            if nocons:
                for s in range(S):
                    for t in range(S):
                        if t != s:
                            allow[:, SUMM[s], SUMM[t]] = False
            sc = (qh @ kh.transpose(-1, -2) / math.sqrt(blk.dh)).masked_fill(~allow[:, None], float('-inf'))
            A = sc.softmax(-1).numpy(); qh = qh.numpy()
        H = A.shape[1]
        P = np.stack([A[:, :, SUMM[s], LABS[s]].sum(-1) for s in range(S)], 1)
        slf = np.stack([A[:, :, SUMM[s], SUMM[s]] for s in range(S)], 1)
        osum = np.stack([A[:, :, SUMM[s], [SUMM[t] for t in range(S) if t != s]].sum(-1) for s in range(S)], 1)
        olab = np.stack([A[:, :, SUMM[s], [j for t in range(S) if t != s for j in LABS[t]]].sum(-1) for s in range(S)], 1)
        o = dict(tag=os.path.basename(run), variant=v, P_h=[float(P[:, :, i].mean()) for i in range(H)], self_h=[float(slf[:, :, i].mean()) for i in range(H)],
                 osum_h=[float(osum[:, :, i].mean()) for i in range(H)], olab_h=[float(olab[:, :, i].mean()) for i in range(H)])
        o['C_h'] = [o['self_h'][i] + o['osum_h'][i] for i in range(H)]; o['C_max'] = float(max(o['C_h'])); o['consensus'] = bool(o['C_max'] >= 0.5)
        pool = [i for i in range(H) if o['P_h'][i] >= 0.5]; o['n_pool'] = len(pool); o['pool_heads'] = pool
        W = Wt.numpy().astype(np.float64); b = bt.numpy().astype(np.float64)
        gam = blk.ln1.weight.detach().numpy().astype(np.float64)
        Wk = blk.qkv.weight.detach().numpy().astype(np.float64)[blk.h * blk.dh:2 * blk.h * blk.dh]
        d = W.shape[0]
        lab = [j for s in range(S) for j in LABS[s]]
        X0 = X[:, lab, :].astype(np.float64).copy(); X0[..., 32] = X0[..., 33]
        if codeln:
            X0n = X0.copy(); X0n[..., :NCODE] = 0                                  # the normalised part: content only
            a = X0n @ W.T + b
        else:
            a = X0 @ W.T + b
        a = a - a.mean(-1, keepdims=True)
        u = W[:, 32] - W[:, 32].mean(); Aj = np.linalg.norm(a, axis=-1); U = np.linalg.norm(u); rho = (a @ u) / (Aj * U)
        ka = np.einsum('ed,bjd->bje', Wk, gam[None, None] * a).reshape(B, len(lab), blk.h, blk.dh)
        ku = (Wk @ (gam * u)).reshape(blk.h, blk.dh)
        L = np.zeros((B, H, S, len(lab))); odd = np.zeros((B, H, S, len(lab)))
        for s in range(S):
            qs = qh[:, :, SUMM[s], :]
            L[:, :, s, :] = math.sqrt(d / blk.dh) * np.einsum('bhe,bjhe->bhj', qs, ka) / Aj[:, None, :]
            odd[:, :, s, :] = math.sqrt(d / blk.dh) * np.einsum('bhe,he->bh', qs, ku)[:, :, None] / Aj[:, None, :]
        r = 3.0
        ratio = 0.5 * (Aj / np.sqrt(Aj ** 2 + 2 * r * Aj * U * rho + r ** 2 * U ** 2) + Aj / np.sqrt(Aj ** 2 - 2 * r * Aj * U * rho + r ** 2 * U ** 2))
        own_i = [[QL * s + qq for qq in range(QL)] for s in range(S)]
        Lown = np.stack([L[:, :, s, own_i[s]] for s in range(S)], 1); rat = np.stack([ratio[:, own_i[s]] for s in range(S)], 1)
        Kodd = np.stack([odd[:, :, s, own_i[s]] for s in range(S)], 1)
        o['L_own_h'] = [float(Lown[:, :, i].mean()) for i in range(H)]
        o['dlogit_r3_h'] = [float((Lown[:, :, i] * (rat - 1.0)).mean()) for i in range(H)]
        o['Kodd_mean_h'] = [float(Kodd[:, :, i].mean()) for i in range(H)]; o['Kodd_rms_h'] = [float(np.sqrt((Kodd[:, :, i] ** 2).mean())) for i in range(H)]
        if codeln:
            kc = np.einsum('ed,bjd->bje', Wk, X[:, lab, :NCODE].astype(np.float64) @ W[:, :NCODE].T).reshape(B, len(lab), blk.h, blk.dh)
            cl = np.zeros((B, H, S, len(lab)))
            for s in range(S):
                cl[:, :, s, :] = np.einsum('bhe,bjhe->bhj', qh[:, :, SUMM[s], :], kc) / math.sqrt(blk.dh)
            co = np.stack([cl[:, :, s, own_i[s]] for s in range(S)], 1)
            ct = np.stack([cl[:, :, s, [i for t in range(S) if t != s for i in own_i[t]]] for s in range(S)], 1)
            o['code_logit_own_h'] = [float(co[:, :, i].mean()) for i in range(H)]; o['code_logit_oth_h'] = [float(ct[:, :, i].mean()) for i in range(H)]
        As = Aj.reshape(B, S, QL).mean(-1); o['A_src_cv'] = float(As.std() / As.mean())
        json.dump(o, open(os.path.join(OUT, o['tag'] + '.json'), 'w'), indent=1)
        print(json.dumps({k: (round(x, 4) if isinstance(x, float) else x) for k, x in o.items()}), flush=True)


if __name__ == '__main__':
    main()
