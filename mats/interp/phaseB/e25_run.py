"""E25 (PREREG-B amendment 33): sharpening E24's localisation to the source's own pooling.
In block 1 the attention output of the modified source's query-question token (token 5s+4), per head, is a sum of terms
a_t v_t over tokens t. In the big-miss version, the terms of that source's four labelled tokens (5s..5s+3) are replaced, at that
token only, by the spread version's terms (its own attention weights times its own values); every other token's attention output
and every other term are unchanged. Variants: qq_all (all four terms), qq_big (only the term at the big-error position), qq_small
(only the other three terms). The attention row of token 5s+4 is computed without gradient in every variant and in the reference,
as in E24. Effective weight of s by autograd; deterministic.
usage: python e25_run.py LIST_JSON I0 I1 -> results/e25/<tag>.npz
"""
import json, math, os, sys, time
import numpy as np
import torch
import pfn as P
import nnoise as N
import e15_run as R

torch.set_num_threads(1)
VAL = 32


def block1(blk, x, allow, src, mask=None, terms=None, want=False):
    """mask (B, 4) bool: which of the source's labelled-token terms to replace at its query-question token; terms (B, h, 4, dh)."""
    B, T, d = x.shape; h = blk.h; dh = d // h
    q, k, v = blk.qkv(blk.ln1(x)).split(d, -1)
    sp = lambda t: t.view(B, T, h, dh).transpose(1, 2)
    q, k, v = sp(q), sp(k), sp(v)
    a = (q @ k.transpose(-1, -2) / math.sqrt(dh)).masked_fill(~allow[:, None], float('-inf')).softmax(-1)
    bi = torch.arange(B); qi = 5 * src + 4
    a = a.clone(); a[bi, :, qi, :] = a[bi, :, qi, :].detach()
    cols = torch.stack([5 * src + j for j in range(4)], 1)                                   # (B, 4)
    own = torch.stack([a[bi, :, qi, cols[:, j]][..., None] * v[bi, :, cols[:, j], :] for j in range(4)], 2)   # (B, h, 4, dh)
    if want:
        return own.detach()
    o = a @ v                                                                                 # (B, h, T, dh)
    if mask is not None:
        m = mask[:, None, :, None].to(own.dtype)
        delta = (m * (terms - own)).sum(2)                                                    # (B, h, dh)
        o = o.clone(); o[bi, :, qi, :] = o[bi, :, qi, :] + delta
    x = x + blk.o(o.transpose(1, 2).reshape(B, T, d))
    if not blk.attn_only:
        x = x + blk.mlp(blk.ln2(x))
    return x


def head_mean(model, X, valid, src, mask=None, terms=None):
    allow = N.NoisyPFN.allow_mask(valid); T = X.shape[1]
    x = block1(model.blocks[0], model.inp(X), allow, src, mask, terms)
    for blk in model.blocks[1:]:
        x, _ = blk(x, allow)
    y = model.out(model.ln(x[:, T - 1]))
    head = (y[:, 0], y[:, 1].clamp(-6, 3)) if model.mixk == 0 else (y[:, model.mixk:2 * model.mixk], y[:, 2 * model.mixk:].clamp(-6, 3), y[:, :model.mixk])
    return R.pred_mean(head)


def big_pos(z, rows):
    """position (0-3) of the big error in the big-miss version: the largest |error| of the modified source."""
    src = z['src'][rows]; e = z['d1_o'][rows][np.arange(len(rows)), src, :4] - z['d1_y'][rows][:, :4]
    return np.abs(e).argmax(1)


def run_net(model, z, bs=500):
    src_all = z['src']; out = {k: [] for k in ('ref', 'qq_all', 'qq_big', 'qq_small', 'spread')}
    for i in range(0, len(src_all), bs):
        rows = np.arange(i, min(i + bs, len(src_all))); s = torch.as_tensor(src_all[rows]); n = len(rows); bi = torch.arange(n)
        X0, X1 = torch.from_numpy(z['X0'][rows]), torch.from_numpy(z['X1'][rows]); v0, v1 = torch.from_numpy(z['valid0'][rows]), torch.from_numpy(z['valid1'][rows])
        with torch.no_grad():
            terms0 = block1(model.blocks[0], model.inp(X0), N.NoisyPFN.allow_mask(v0), s, want=True)
        bp = torch.as_tensor(big_pos(z, rows))
        allm = torch.ones(n, 4, dtype=torch.bool); bigm = torch.zeros(n, 4, dtype=torch.bool); bigm[bi, bp] = True
        for name, Xb, vb, mk in (('ref', X1, v1, None), ('qq_all', X1, v1, allm), ('qq_big', X1, v1, bigm), ('qq_small', X1, v1, ~bigm),
                                 ('spread', X0, v0, None)):
            Xi = Xb.clone().requires_grad_(True)
            m = head_mean(model, Xi, vb, s, mk, terms0 if mk is not None else None)
            gr, = torch.autograd.grad(m.sum(), Xi)
            out[name].append(gr[bi, 5 * s + 4, VAL].numpy().astype(np.float64))
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    rows = json.load(open(sys.argv[1]))[int(sys.argv[2]):int(sys.argv[3])]
    os.makedirs('results/e25', exist_ok=True); z = np.load('results/e12_pairs.npz')
    for r in rows:
        p = f"results/e25/{r['tag']}.npz"
        if os.path.exists(p):
            continue
        t0 = time.time(); meta = json.load(open(os.path.join(r['run'], 'meta.json')))
        m = P.build(meta); m.load_state_dict(torch.load(os.path.join(r['run'], 'model.pt'), map_location='cpu')); m.eval(); m.requires_grad_(False)
        np.savez(p, **run_net(m, z)); print(r['tag'], 'done %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
