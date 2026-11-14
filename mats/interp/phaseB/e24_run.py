"""E24 (PREREG-B amendment 31): on-distribution transplants that split the big-miss weight difference by route, in clean 2-layer nets.
For every E12 pair (modified source s) the big-miss version is run with parts of block 1 taken from its own spread version:
  att   the attention row of s's query-question token (token 5s+4), all heads, replaced by the spread version's row;
  val   the block-1 value vectors of s's four labelled tokens (5s..5s+3) replaced by the spread version's;
  both  att and val together.
Transplanted quantities are constants (no gradient). For comparability every variant, including the unpatched reference ("ref"),
computes the attention row of s's query-question token without gradient (so the reading acts through values and the residual only).
Effective weight of s = d m / d o*_s by autograd; deterministic. Also stored: the spread version's weight under the same convention.
usage: python e24_run.py LIST_JSON I0 I1 -> results/e24/<tag>.npz
"""
import json, math, os, sys, time
import numpy as np
import torch
import pfn as P
import nnoise as N
import e15_run as R

torch.set_num_threads(1)
VAL = 32


def block1(blk, x, allow, src, att_row=None, vals=None, want=False):
    B, T, d = x.shape; h = blk.h; dh = d // h
    q, k, v = blk.qkv(blk.ln1(x)).split(d, -1)
    sp = lambda t: t.view(B, T, h, dh).transpose(1, 2)
    q, k, v = sp(q), sp(k), sp(v)
    s = (q @ k.transpose(-1, -2) / math.sqrt(dh)).masked_fill(~allow[:, None], float('-inf'))
    a = s.softmax(-1)
    bi = torch.arange(B); qi = 5 * src + 4                                    # (B,) query-question token of the modified source
    row = a[bi, :, qi, :].detach() if att_row is None else att_row           # (B, h, T), no gradient in every variant
    a = a.clone(); a[bi, :, qi, :] = row
    if vals is not None:
        for j in range(4):
            v = v.clone(); v[bi, :, 5 * src + j, :] = vals[:, :, j, :]
    if want:
        return row, torch.stack([v[bi, :, 5 * src + j, :] for j in range(4)], 2).detach()
    x = x + blk.o((a @ v).transpose(1, 2).reshape(B, T, d))
    if not blk.attn_only:
        x = x + blk.mlp(blk.ln2(x))
    return x


def head_mean(model, X, valid, src, att_row=None, vals=None):
    allow = N.NoisyPFN.allow_mask(valid); B, T, _ = X.shape
    x = block1(model.blocks[0], model.inp(X), allow, src, att_row, vals)
    for blk in model.blocks[1:]:
        x, _ = blk(x, allow)
    y = model.out(model.ln(x[:, T - 1]))
    head = (y[:, 0], y[:, 1].clamp(-6, 3)) if model.mixk == 0 else (y[:, model.mixk:2 * model.mixk], y[:, 2 * model.mixk:].clamp(-6, 3), y[:, :model.mixk])
    return R.pred_mean(head)


def run_net(model, z, bs=500):
    src = z['src']; out = {k: [] for k in ('ref', 'att', 'val', 'both', 'spread')}
    for i in range(0, len(src), bs):
        s = torch.as_tensor(src[i:i + bs]); X0 = torch.from_numpy(z['X0'][i:i + bs]); X1 = torch.from_numpy(z['X1'][i:i + bs])
        v0 = torch.from_numpy(z['valid0'][i:i + bs]); v1 = torch.from_numpy(z['valid1'][i:i + bs]); bi = torch.arange(len(s))
        with torch.no_grad():
            row0, vals0 = block1(model.blocks[0], model.inp(X0), N.NoisyPFN.allow_mask(v0), s, want=True)
        for name, Xb, vb, ar, vl in (('ref', X1, v1, None, None), ('att', X1, v1, row0, None), ('val', X1, v1, None, vals0),
                                    ('both', X1, v1, row0, vals0), ('spread', X0, v0, None, None)):
            Xi = Xb.clone().requires_grad_(True)
            m = head_mean(model, Xi, vb, s, ar, vl)
            gr, = torch.autograd.grad(m.sum(), Xi)
            out[name].append(gr[bi, 5 * s + 4, VAL].numpy().astype(np.float64))
    return {k: np.concatenate(v) for k, v in out.items()}


def main():
    rows = json.load(open(sys.argv[1]))[int(sys.argv[2]):int(sys.argv[3])]
    os.makedirs('results/e24', exist_ok=True); z = np.load('results/e12_pairs.npz')
    for r in rows:
        p = f"results/e24/{r['tag']}.npz"
        if os.path.exists(p):
            continue
        t0 = time.time(); meta = json.load(open(os.path.join(r['run'], 'meta.json')))
        m = P.build(meta); m.load_state_dict(torch.load(os.path.join(r['run'], 'model.pt'), map_location='cpu')); m.eval(); m.requires_grad_(False)
        np.savez(p, **run_net(m, z)); print(r['tag'], 'done %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
