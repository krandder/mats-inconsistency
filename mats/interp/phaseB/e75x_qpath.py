"""E75 exploration (eighth agent; DEV NETS ONLY, disclosed in a later amendment): is the reading's path through its own pool useful?
In block 1, only the summary rows' queries (tokens s*5+4, every head) are recomputed from an input in which that token's reading o*_s is
  zero    replaced by 0 (the prior mean of a reading), or
  shuf    replaced by the same source's reading in another episode of the set (a fixed derangement: episode i takes i+1's),
while every key, value, residual and later computation uses the real input. Measures (test set: tiny-pfn's 16,000 test episodes; pairs:
E12's): the test-NLL gap and rom (large, middle), for base, zero and shuf. Standard-forward nets only (pfn2 blocks, any head count).
usage: python e75x_qpath.py OUTJSON TAG=RUNDIR [...]
"""
import json, math, os, sys
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


HEADS = None                                                            # heads whose summary-row queries are replaced (None: all)


def forward(model, Xt, Xq):
    """Xq: the input used for the summary rows' block-1 queries (same as Xt except their reading)."""
    B = Xt.shape[0]; blk = model.blocks[0]; al = allow_mask(B)
    x = model.inp(Xt); h = blk.ln1(x)
    q, k, v = blk.qkv(h).split(blk.h * blk.dh, -1)
    if Xq is not None:
        qq = blk.qkv(blk.ln1(model.inp(Xq))).split(blk.h * blk.dh, -1)[0]
        q = q.clone()
        for hh in (range(blk.h) if HEADS is None else HEADS):
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


def qinput(X, mode):
    if mode == 'base':
        return None
    Xq = X.copy()
    for s in SUMM:
        Xq[:, s, OC] = 0.0 if mode == 'zero' else np.roll(X[:, s, OC], -1)
    return Xq


def gap(model, tz, mode):
    nll = []
    with torch.no_grad():
        for i in range(0, len(tz['X']), 2000):
            X = tz['X'][i:i + 2000]; Xq = qinput(X, mode)
            hd = forward(model, torch.from_numpy(X).float(), None if Xq is None else torch.from_numpy(Xq).float())
            nll.append(P2.head_nll_np(tuple(t.numpy() for t in hd), tz['ystar'][i:i + 2000]))
    return float((np.concatenate(nll) - tz['nll_bayes']).mean())


def rom(model, z, mode):
    src, rg = z['src'], z['range']; Wb = z['WB0'][np.arange(len(src)), src]; ws = []
    for k in (0, 1):
        X = z['X%d' % k]; Xq = qinput(X, mode); w = []
        for i in range(0, len(src), 1000):
            Xt = torch.from_numpy(X[i:i + 1000]).float().requires_grad_(True)
            if Xq is None:
                Xqt = None
            else:                                                       # the query's reading follows the input (zero) or not (shuf)
                Xqt = torch.from_numpy(Xq[i:i + 1000]).float()
            mu, ls, lg = forward(model, Xt, Xqt)
            g, = torch.autograd.grad((torch.softmax(lg, -1) * mu).sum(), Xt)
            w.append(g[torch.arange(len(Xt)), torch.tensor([SUMM[s] for s in src[i:i + 1000]]), OC].numpy().astype(np.float64))
        ws.append(np.concatenate(w))
    return {rr: float((ws[1] - ws[0])[rg == rr].sum() / Wb[rg == rr].sum()) for rr in ('large', 'middle')}


def main():
    outp = sys.argv[1]; res = json.load(open(outp)) if os.path.exists(outp) else {}
    tz = np.load(os.path.join(TP, 'runs/testset.npz')); z = np.load('results/e12_pairs.npz')
    for spec in sys.argv[2:]:
        tag, run = spec.split('=', 1)
        if tag in res:
            continue
        model, arm, meta = E63.load(run, None); assert not meta.get('variant') and arm == 'ctrl'
        global HEADS
        if os.environ.get('E75X_SPLIT'):                                # per head type: pooling heads only, consensus heads only
            zz = np.load('results/e27_data.npz'); sl = slice(0, 2000)
            X, v = E63.F.make_X(zz['o'][sl, 0], zz['y'][sl], zz['semb'][sl], zz['qemb'][sl], arm)
            A, qh, blk = E63.block1_parts(model, X)
            P = np.stack([A[:, :, E63.SUMM[s_], E63.LABS[s_]].sum(-1) for s_ in range(S)], 1).mean((0, 1))
            C = np.stack([A[:, :, E63.SUMM[s_], E63.SUMM].sum(-1) for s_ in range(S)], 1).mean((0, 1))
            pool = [h_ for h_ in range(len(P)) if P[h_] >= 0.5]; cons = [h_ for h_ in range(len(C)) if C[h_] >= 0.5]
            o = dict(pool=pool, cons=cons, base=dict(gap=gap(model, tz, 'base'), **rom(model, z, 'base')))
            for nm, hs in (('shuf_pool', pool), ('shuf_cons', cons)):
                if hs:
                    HEADS = hs; o[nm] = dict(gap=gap(model, tz, 'shuf'), **rom(model, z, 'shuf'))
            HEADS = None
        else:
            o = {m: dict(gap=gap(model, tz, m), **rom(model, z, m)) for m in ('base', 'zero', 'shuf')}
        res[tag] = o; json.dump(res, open(outp, 'w'), indent=1)
        print(tag, json.dumps(o), flush=True)


if __name__ == '__main__':
    main()
