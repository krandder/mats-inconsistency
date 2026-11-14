"""E75 (PREREG-B amendment 83): does the source's own reading, through the block-1 query of its summary row, carry information the
nets use? Evaluation-time interventions on nets with pfn2's standard forward (any head count and depth; pfn_b1hw and pfn_l2fix loaders).
In block 1, the queries of the summary rows (tokens s*5+4) in a chosen set of heads are recomputed from an input in which that token's
reading is replaced by the same source's reading in another episode (a fixed derangement: episode i takes episode i+1's); every key,
value, residual and later block uses the real input. Head sets: the pooling heads (P_h >= 0.5) and the consensus heads (C_h >= 0.5),
classified on E27's first 2,000 bases as E63. Measures, in the net's own format: test-NLL gap on tiny-pfn's 16,000 test episodes and
rom (large, middle) on E12's pairs, for base, shuf_pool and shuf_cons; and, per pooling head, the correlation over (episode, source)
of the source-mean signed coefficient K (E63's h3 decomposition) with the source's reading o*.
usage: python e75_run.py OUTDIR TAG=RUNDIR:EVALFILE [...] -> OUTDIR/<tag>.json
"""
import json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn-interp/heads')
import pfn2 as P2  # noqa: E402
import e63_run as E63  # noqa: E402
import pfn_b1hw as BH  # noqa: E402

BH.install()
import pfn_l2fix  # noqa: E402

pfn_l2fix.install()
torch.set_num_threads(1)
S, Q, T, OC = 3, 5, 16, 32
SUMM = [4, 9, 14]; QI = 15
TP = '/home/agent/workspaces/mats/work/tiny-pfn'


def allow_mask(B):
    a = torch.ones(B, T, T, dtype=torch.bool); a[:, :, QI] = False; a[:, QI, QI] = True
    return a


def forward(model, Xt, Xq, heads):
    B = Xt.shape[0]; blk = model.blocks[0]; al = allow_mask(B)
    x = model.inp(Xt); q, k, v = blk.qkv(blk.ln1(x)).split(blk.h * blk.dh, -1)
    if Xq is not None:
        qq = blk.qkv(blk.ln1(model.inp(Xq))).split(blk.h * blk.dh, -1)[0]
        q = q.clone()
        for hh in heads:
            q[:, SUMM, hh * blk.dh:(hh + 1) * blk.dh] = qq[:, SUMM, hh * blk.dh:(hh + 1) * blk.dh]
    sp = lambda t, b: t.view(B, T, b.h, b.dh).transpose(1, 2)
    q, k, v = sp(q, blk), sp(k, blk), sp(v, blk)
    a = ((q @ k.transpose(-1, -2)) / math.sqrt(blk.dh)).masked_fill(~al[:, None], float('-inf')).softmax(-1)
    x = x + blk.o((a @ v).transpose(1, 2).reshape(B, T, blk.h * blk.dh)); x = x + blk.mlp(blk.ln2(x))
    for b2 in model.blocks[1:]:
        q2, k2, v2 = b2.qkv(b2.ln1(x)).split(b2.h * b2.dh, -1)
        q2, k2, v2 = sp(q2, b2), sp(k2, b2), sp(v2, b2)
        a2 = ((q2 @ k2.transpose(-1, -2)) / math.sqrt(b2.dh)).masked_fill(~al[:, None], float('-inf')).softmax(-1)
        x = x + b2.o((a2 @ v2).transpose(1, 2).reshape(B, T, b2.h * b2.dh)); x = x + b2.mlp(b2.ln2(x))
    y = model.out(model.ln(x[:, QI])); K = model.mixk
    return y[:, K:2 * K], y[:, 2 * K:].clamp(-6, 3), y[:, :K]


def shuf(X):
    Xq = X.copy()
    for s in SUMM:
        Xq[:, s, OC] = np.roll(X[:, s, OC], -1)
    return Xq


def gap(model, X, tz, heads):
    nll = []
    with torch.no_grad():
        for i in range(0, len(X), 2000):
            Xi = X[i:i + 2000]; Xq = None if heads is None else torch.from_numpy(shuf(Xi)).float()
            hd = forward(model, torch.from_numpy(Xi).float(), Xq, heads or [])
            nll.append(P2.head_nll_np(tuple(t.numpy() for t in hd), tz['ystar'][i:i + 2000]))
    return float((np.concatenate(nll) - tz['nll_bayes']).mean())


def rom(model, PX, z, heads):
    src, rg = z['src'], z['range']; Wb = z['WB0'][np.arange(len(src)), src]; ws = []
    for k in (0, 1):
        X = PX[k]; w = []
        for i in range(0, len(src), 1000):
            Xi = X[i:i + 1000]; Xt = torch.from_numpy(Xi).float().requires_grad_(True)
            Xq = None if heads is None else torch.from_numpy(shuf(Xi)).float()
            mu, ls, lg = forward(model, Xt, Xq, heads or [])
            g, = torch.autograd.grad((torch.softmax(lg, -1) * mu).sum(), Xt)
            w.append(g[torch.arange(len(Xt)), torch.tensor([SUMM[s] for s in src[i:i + 1000]]), OC].numpy().astype(np.float64))
        ws.append(np.concatenate(w))
    return {rr: float((ws[1] - ws[0])[rg == rr].sum() / Wb[rg == rr].sum()) for rr in ('large', 'middle')}


def main():
    outdir = sys.argv[1]; os.makedirs(outdir, exist_ok=True)
    tz = np.load(os.path.join(TP, 'runs/testset.npz')); z = np.load('results/e12_pairs.npz')
    ts, tq = E63.F.ids_from_X(tz['X'])
    zz = np.load('results/e27_data.npz'); sl = slice(0, 2000)
    for spec in sys.argv[2:]:
        tag, rest = spec.split('=', 1); run, evf = rest.split(':', 1); outp = os.path.join(outdir, tag + '.json')
        if os.path.exists(outp):
            continue
        t0 = time.time(); model, arm, meta = E63.load(run, None); evz = np.load(evf, allow_pickle=True)
        assert not meta.get('variant'); model.requires_grad_(False)
        X, _ = E63.F.make_X(tz['o'].astype(np.float64), tz['y'].astype(np.float64), ts, tq, arm); X = X.astype(np.float32)
        PX = []
        for k in (0, 1):
            ps, pq = E63.F.ids_from_X(z['X%d' % k])
            PX.append(E63.F.make_X(z['d%d_o' % k].astype(np.float64), z['d%d_y' % k].astype(np.float64), ps, pq, arm)[0].astype(np.float32))
        Xb, vb = E63.F.make_X(zz['o'][sl, 0], zz['y'][sl], zz['semb'][sl], zz['qemb'][sl], arm)
        A, qh, blk = E63.block1_parts(model, Xb)
        P = np.stack([A[:, :, E63.SUMM[s], E63.LABS[s]].sum(-1) for s in range(S)], 1).mean((0, 1))
        C = np.stack([A[:, :, E63.SUMM[s], E63.SUMM].sum(-1) for s in range(S)], 1).mean((0, 1))
        pool = [h for h in range(len(P)) if P[h] >= 0.5]; cons = [h for h in range(len(C)) if C[h] >= 0.5]
        L, ratio, odd = E63.h3_logits(model, Xb, qh, blk)
        own = [[4 * s + q for q in range(4)] for s in range(S)]
        Ks = np.stack([odd[:, :, s, own[s]].mean(-1) for s in range(S)], 1); ost = zz['o'][sl, 0][:, :, 4]
        o = dict(tag=tag, arm=arm, net_rom_large=float(evz['rom_large']), net_rom_middle=float(evz['rom_middle']), net_gap=float(evz['gap']),
                 type='A' if float(evz['rom_middle']) < 0 else 'B', pool=pool, cons=cons,
                 corr_K_ostar={h: float(np.corrcoef(Ks[:, :, h].ravel(), ost.ravel())[0, 1]) for h in pool})
        o['base'] = dict(gap=gap(model, X, tz, None), **rom(model, PX, z, None))
        for nm, hs in (('shuf_pool', pool), ('shuf_cons', cons)):
            if hs:
                o[nm] = dict(gap=gap(model, X, tz, hs), **rom(model, PX, z, hs))
        o['seconds'] = time.time() - t0
        json.dump(o, open(outp + '.tmp', 'w'), indent=1); os.replace(outp + '.tmp', outp)
        print(tag, json.dumps({k: v for k, v in o.items() if k not in ('corr_K_ostar',)}), flush=True)


if __name__ == '__main__':
    main()
