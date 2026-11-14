"""E15 post hoc (written after seeing E15's registered results): normalised layer-1 encoding strength for clean nets.
The registered M3 differentiates the raw residual h1 after block 1. Layer 2 reads it through LayerNorm, and the noisy nets'
g (E13) was measured on a sphere-normalised residual. Here g_sph = |d sphere(h1)[QQ_src] / d o*_src| with
sphere(v) = v sqrt(d) / ||v|| (per token), the closest analogue; acquired nets, best checkpoints, both pair versions.
usage: python e15_gsph.py -> results/e15/<tag>__best_gsph.npz, results/e15_gsph.json
"""
import json, math, os
import numpy as np
import torch
from torch.func import jvp
import nnoise as N
import pfn as P

torch.set_num_threads(1)
QQ = [4, 9, 14]


def g_sph(model, X, valid, src, vi, bs=1000):
    out = []; qq = torch.as_tensor(QQ); d = model.ln.normalized_shape[0]
    for i in range(0, len(X), bs):
        Xi = torch.from_numpy(X[i:i + bs]); n = len(Xi); s = torch.as_tensor(src[i:i + bs])
        al = N.NoisyPFN.allow_mask(torch.from_numpy(valid[i:i + bs]))
        tg = torch.zeros_like(Xi); tg[torch.arange(n), qq[s], vi] = 1.0
        def f(Xa):
            h = model.blocks[0](model.inp(Xa), al)[0]
            return h * (math.sqrt(d) / h.norm(dim=-1, keepdim=True))
        _, t1 = jvp(f, (Xi,), (tg,))
        out.append(t1[torch.arange(n), qq[s]].norm(dim=-1).detach().numpy().astype(np.float64))
    return np.concatenate(out)


def main():
    nets = json.load(open('results/e15_nets.json'))
    z16, z4 = np.load('results/e12_pairs.npz'), np.load('results/e15_pairs_ds4.npz'); src, rg = z16['src'], z16['range']
    res = {}
    for r in nets:
        q = np.load(os.path.join('results/e15', f"{r['tag']}__best.npz"))
        if r['depth'] != 2 or float(q['gap']) >= 0.10:
            continue
        p = os.path.join('results/e15', f"{r['tag']}__best_gsph.npz")
        if not os.path.exists(p):
            meta = json.load(open(os.path.join(r['run'], 'meta.json')))
            model = P.build(meta); model.load_state_dict(torch.load(os.path.join(r['run'], 'model.pt'), map_location='cpu')); model.eval(); model.requires_grad_(False)
            pz = z16 if r['ds'] == 16 else z4; vi = 2 * r['ds']
            np.savez(p, g0=g_sph(model, pz['X0'], pz['valid0'], src, vi), g1=g_sph(model, pz['X1'], pz['valid1'], src, vi))
        g = np.load(p)
        res[r['tag']] = dict(cfg=r['cfg'], heads=r['heads'], attn_only=r['attn_only'],
                             **{rr: float((g['g1'] - g['g0'])[rg == rr].sum() / g['g0'][rg == rr].sum()) for rr in ('large', 'middle')})
    json.dump(res, open('results/e15_gsph.json', 'w'), indent=1)
    by = {}
    for t, v in res.items():
        by.setdefault(v['cfg'], []).append((v['large'], v['middle']))
    for c, v in sorted(by.items()):
        v = np.array(v); print(f'{c:24s} n {len(v)}  Δg_sph/g large {v[:, 0].mean():+.3f}  middle {v[:, 1].mean():+.3f}')


if __name__ == '__main__':
    main()
