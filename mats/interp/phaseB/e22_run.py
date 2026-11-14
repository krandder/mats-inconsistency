"""E22 (PREREG-B amendment 29): the toy's attention intervention inside the real 2-layer clean nets.
In block 1, for each source's query-question token (tokens 4, 9, 14), in every head, the attention weights on that source's four
labelled tokens (5s .. 5s+3) are replaced by their mean (uniform within the group, same total mass); all other weights, and
everything else in the net, unchanged. Effective weights on the E12 pairs are then recomputed through the patched forward
(autograd, deterministic), unpatched and patched.
usage: python e22_run.py LIST_JSON I0 I1 -> results/e22/<tag>.npz
"""
import json, math, os, sys, time
import numpy as np
import torch
import pfn as P
import e15_run as R

torch.set_num_threads(1)
QQ = [4, 9, 14]


def block1_patched(blk, x, allow):
    B, T, d = x.shape
    q, k, v = blk.qkv(blk.ln1(x)).split(d, -1)
    sp = lambda t: t.view(B, T, blk.h, d // blk.h).transpose(1, 2)
    q, k, v = sp(q), sp(k), sp(v)
    s = (q @ k.transpose(-1, -2) / math.sqrt(d // blk.h)).masked_fill(~allow[:, None], float('-inf'))
    a = s.softmax(-1)
    rows = []
    for src, qi in enumerate(QQ):
        cols = list(range(5 * src, 5 * src + 4))
        grp = a[:, :, qi, cols]                                         # (B, h, 4)
        rows.append((qi, cols, grp.mean(-1, keepdim=True).expand_as(grp)))
    a2 = a.clone()
    for qi, cols, val in rows:
        a2[:, :, qi, cols] = val
    x = x + blk.o((a2 @ v).transpose(1, 2).reshape(B, T, d))
    if not blk.attn_only:
        x = x + blk.mlp(blk.ln2(x))
    return x


def forward(model, X, valid, patch):
    if not patch:
        return model(X, valid)
    B, T, _ = X.shape
    allow = __import__('nnoise').NoisyPFN.allow_mask(valid)
    x = model.inp(X)
    x = block1_patched(model.blocks[0], x, allow)
    for blk in model.blocks[1:]:
        x, _ = blk(x, allow)
    y = model.out(model.ln(x[:, T - 1]))
    if model.mixk == 0:
        return (y[:, 0], y[:, 1].clamp(-6, 3))
    K = model.mixk
    return (y[:, K:2 * K], y[:, 2 * K:].clamp(-6, 3), y[:, :K])


def eff_w(model, X, valid, patch, bs=1000):
    W_ = []
    for i in range(0, len(X), bs):
        Xi = torch.from_numpy(X[i:i + bs]).clone().requires_grad_(True)
        m = R.pred_mean(forward(model, Xi, torch.from_numpy(valid[i:i + bs]), patch))
        gr, = torch.autograd.grad(m.sum(), Xi); W_.append(gr[:, QQ, 32].numpy().astype(np.float64))
    return np.concatenate(W_)


def main():
    rows = json.load(open(sys.argv[1]))[int(sys.argv[2]):int(sys.argv[3])]
    os.makedirs('results/e22', exist_ok=True); z = np.load('results/e12_pairs.npz')
    for r in rows:
        p = f"results/e22/{r['tag']}.npz"
        if os.path.exists(p):
            continue
        t0 = time.time(); meta = json.load(open(os.path.join(r['run'], 'meta.json')))
        m = P.build(meta); m.load_state_dict(torch.load(os.path.join(r['run'], 'model.pt'), map_location='cpu')); m.eval(); m.requires_grad_(False)
        out = {}
        for patch in (0, 1):
            for k in (0, 1):
                out[f'ws{k}_p{patch}'] = eff_w(m, z[f'X{k}'], z[f'valid{k}'], patch)
        np.savez(p, **out); print(r['tag'], 'done %.0fs' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
