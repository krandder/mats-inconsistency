"""E8 design aid (descriptive, before pre-registration of E8.2): layer-2 query-row attention per net and head:
mass on the query-question (QQ) tokens, on labelled tokens, on itself; correlation of per-source mass with Bayes source weights.
One noise draw, 2,048 fit-distribution episodes. usage: python qattn_look.py -> results/qattn_look.json
"""
import json, math, os
import numpy as np
import torch
import nnoise as N
import world as W
import qbayes as QB
import qw
torch.set_num_threads(1)
NETS = ['L2_s0.05', 'L2seed1_s0.05', 'L2mx0_s0.05', 'L2mx1_s0.05', 'L2mx2_s0.05', 'L2nb6_s0.05', 'L2seed2_s0.05', 'L2ws0_s0.05', 'L2ws3_s0.05', 'L2mx4_s0.05', 'L2nb5_s0.05', 'L2nb7_s0.05']
rng = np.random.default_rng(98_000); d = W.sample_np(2048, rng); X, v = W.tokens_np(d, rng, 16)
b = QB.bayes_features(d); Wb = b[:, 20:23]
out = {}
for nm in NETS:
    model, meta = N.load_run(os.path.join('runs', nm)); model.requires_grad_(False); sg = meta['sigma']
    st, nz, allow = qw.trace_nz(model, X, v, sg, torch.Generator().manual_seed(1))
    blk = model.blocks[1]; x = st['r1']
    with torch.no_grad():
        q, k, vv = blk.qkv(blk.ln1(x)).split(blk.h * blk.dh, -1); sp = lambda t: t.view(len(x), 16, blk.h, blk.dh).transpose(1, 2)
        A = ((sp(q) @ sp(k).transpose(-1, -2)) / math.sqrt(blk.dh)).masked_fill(~allow[:, None], float('-inf')).softmax(-1)[:, :, 15]   # (n, h, 16)
    A = A.numpy(); r = {}
    for h in range(blk.h):
        src = np.stack([A[:, h, 5 * s:5 * s + 5].sum(1) for s in range(3)], 1); qq = A[:, h, [4, 9, 14]]
        lab = src - qq
        cs = float(np.mean([np.corrcoef(src[:, s] / src.sum(1), Wb[:, s] / Wb.sum(1))[0, 1] for s in range(3)]))
        cq = float(np.mean([np.corrcoef(qq[:, s] / np.maximum(qq.sum(1), 1e-9), Wb[:, s] / Wb.sum(1))[0, 1] for s in range(3)]))
        r[f'head{h}'] = dict(mass_QQ=float(qq.sum(1).mean()), mass_labelled=float(lab.sum(1).mean()), mass_self=float(A[:, h, 15].mean()),
                             corr_source_mass_vs_Bayes_weight=cs, corr_QQ_mass_vs_Bayes_weight=cq)
    out[nm] = r; print(nm, {h: {k_: round(v_, 2) for k_, v_ in x_.items()} for h, x_ in r.items()}, flush=True)
json.dump(out, open('results/qattn_look.json', 'w'), indent=1)
