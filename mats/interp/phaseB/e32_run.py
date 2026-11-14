"""E32 per-net measures (PREREG-B amendment 40): does block 1's cross-source same-question attention carry the sign dependence?
Patch: in block 1, for every labelled token (s, q) and every head, the attention weights on the other two sources' tokens of the same
question q, (t, q) with t != s, are set to zero and the row is renormalised; everything else is unchanged.
On the first 3,000 bases of E28's data (Bayes-invariant flip and negation families), unpatched and patched:
  the KL-exact (uncorrected) costs of independent sign flips and of whole-vector negation, the mean KL (base and family
  variants); and the natural marker xq (mean attention mass per labelled token on those tokens, read before the patch, which
  sets that mass to zero).
usage: python e32_run.py SET I0 I1 -> results/e32/<SET>/<tag>.npz   (sets and nets as in results/e26_nets.json)
"""
import json, math, os, sys, time
import numpy as np
import torch

import e26_run as RU
import e28_data as E8
import world as W
import pfn as P

torch.set_num_threads(1)
NB = 3000
LAB = [(s, q) for s in range(3) for q in range(4)]


def cross_mask(T=16):
    M = torch.zeros(T, T, dtype=torch.bool)
    for s, q in LAB:
        for t in range(3):
            if t != s:
                M[5 * s + q, 5 * t + q] = True
    return M


CM = cross_mask()


def forward(model, X, valid, patch):
    """pfn.PFN forward with an optional patch of block 1's attention; returns the head tuple and the xq marker."""
    B, T, _ = X.shape
    qi = T - 1
    ctx = valid.clone(); ctx[:, qi] = False
    allow = ctx[:, None, :].expand(B, T, T).clone(); allow[:, qi, qi] = True
    eye = torch.eye(T, dtype=torch.bool)[None]; allow = allow | (eye & ~valid[:, :, None])
    x = model.inp(X)
    for li, blk in enumerate(model.blocks):
        d = x.shape[-1]
        q, k, v = blk.qkv(blk.ln1(x)).split(d, -1)
        sp = lambda t: t.view(B, T, blk.h, d // blk.h).transpose(1, 2)
        q, k, v = sp(q), sp(k), sp(v)
        s = q @ k.transpose(-1, -2) / math.sqrt(d // blk.h)
        s = s.masked_fill(~allow[:, None], float('-inf'))
        a = s.softmax(-1)
        if li == 0:
            xq = float(a[:, :, CM].sum(-1).mean() / len(LAB))       # mean mass per labelled row (2 entries each)
            if patch:
                a = a.masked_fill(CM[None, None], 0.0)
                a = a / a.sum(-1, keepdim=True)
        x = x + blk.o((a @ v).transpose(1, 2).reshape(B, T, d))
        if not blk.attn_only:
            x = x + blk.mlp(blk.ln2(x))
        if li == 0:
            xq_out = xq
    y = model.out(model.ln(x[:, qi]))
    K = model.mixk
    head = (y[:, 0], y[:, 1].clamp(-6, 3)) if K == 0 else (y[:, K:2 * K], y[:, 2 * K:].clamp(-6, 3), y[:, :K])
    return head, xq_out


def heads(model, X, valid, patch, bs=1500):
    outs, xs = [], []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            h, xq = forward(model, torch.from_numpy(X[i:i + bs]), torch.from_numpy(valid[i:i + bs]), patch)
            outs.append([t.numpy() for t in h]); xs.append(xq)
    h = tuple(np.concatenate([o[j] for o in outs]) for j in range(len(outs[0])))
    if len(h) == 2:
        return (h[0][:, None].astype(np.float64), h[1][:, None].astype(np.float64), np.zeros((len(h[0]), 1))), float(np.mean(xs))
    mu, ls, lg = [t.astype(np.float64) for t in h]
    lg = lg - lg.max(1, keepdims=True); lg = lg - np.log(np.exp(lg).sum(1, keepdims=True))
    return (mu, ls, lg), float(np.mean(xs))


def main():
    which, i0, i1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    rows = [r for r in json.load(open('results/e26_nets.json')) if r['set'] == which][i0:i1]
    z = np.load('results/e28_data.npz')
    D = {k: z[k][:NB] for k in ('y', 'semb', 'qemb', 'o')}
    g = z['bmean'][:NB, None] + RU.GRID[None]
    lpB = RU.ED.mix_logp_points(z['bmeans'][:NB], z['bvars'], z['bwts'][:NB], g)
    os.makedirs(os.path.join('results/e32', which), exist_ok=True)
    idx = sorted(set(E8.fam_idx('flip_ind') + E8.fam_idx('negate_ind')))
    for row in rows:
        p = os.path.join('results/e32', which, row['tag'] + '.npz')
        if os.path.exists(p):
            continue
        t0 = time.time(); model, kind = RU.load(row); assert kind == 'clean'
        out = {}
        for patch in (False, True):
            tag = 'p' if patch else 'u'; hd = {}
            for k in idx:
                X, valid = W.assemble(D['semb'], D['qemb'], D['o'][:, k], D['y'])
                hd[k], xq = heads(model, X, valid, patch)
                if k == 0:
                    out['xq_' + tag] = xq
            for f in ('flip_ind', 'negate_ind'):
                kl, pc, m = RU.set_stats([hd[k] for k in E8.fam_idx(f)], lpB, g)
                out['kl_%s_%s' % (f, tag)], out['pc_%s_%s' % (f, tag)] = float(kl.mean()), float(pc.mean())
            kl0, _, _ = RU.set_stats([hd[0]], lpB, g)
            out['kl_base_' + tag] = float(kl0.mean())
        np.savez(p + '.tmp.npz', **out); os.replace(p + '.tmp.npz', p)
        print(which, row['tag'], {k: round(v, 5) for k, v in out.items()}, '%.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
