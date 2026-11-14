"""E8.2 (PREREG-B amendment 15): patches of the layer-2 query-row attention.
Variants per head (query row t = 15; other rows and all other computation real):
  base; Pbayes (source masses proportional to the Bayes weights W_s = d mu_B / d o*_s, clipped at 0 and renormalised;
  total context mass and within-source split kept); PbayesQQ (only the three QQ-token masses proportional to W_s, their
  total kept); Ppost / PpostQQ (the same with the Bayes posterior precision share, qbayes features 20:23); Ptrue (source masses
  proportional to true precisions 1/v); Puniform (equal source masses); Pmean (query row = its fit-episode average).
K = 64 noise draws (layer 1 shared across variants within a draw), first 4,000 test episodes.
usage: python qpatch.py NET [NET ...] -> runs/NET/qpatch.npz
"""
import math, os, sys, time
import numpy as np
import torch
import nnoise as N
import world as W
import devpred as DP
import qw
import qbayes as QB
import qweights as QWT

torch.set_num_threads(1)
T, QT = 16, 15
SRC = [list(range(5 * s, 5 * s + 5)) for s in range(3)]; QQ = [4, 9, 14]
VARIANTS = ['base', 'Pbayes', 'PbayesQQ', 'Ppost', 'PpostQQ', 'Ptrue', 'Puniform', 'Pmean']


def layer2_attn(b, x, allow, variant, wts=None, Abar=None):
    n = len(x); q, k, v = b.qkv(b.ln1(x)).split(b.h * b.dh, -1)
    sp = lambda t: t.view(n, T, b.h, b.dh).transpose(1, 2)
    A = ((sp(q) @ sp(k).transpose(-1, -2)) / math.sqrt(b.dh)).masked_fill(~allow[:, None], float('-inf')).softmax(-1)
    if variant != 'base':
        A = A.clone(); row = A[:, :, QT]                                       # (n, h, T)
        if variant == 'Pmean':
            row = Abar[None].expand(n, -1, -1).clone()
        elif variant.endswith('QQ'):
            qq = row[:, :, QQ]; tot = qq.sum(-1, keepdim=True)
            row[:, :, QQ] = tot * wts[:, None, :]
        else:
            m = torch.stack([row[:, :, S].sum(-1) for S in SRC], -1); tot = m.sum(-1, keepdim=True)   # (n, h, 3)
            f = tot * wts[:, None, :] / m.clamp_min(1e-12)
            for s, S in enumerate(SRC): row[:, :, S] = row[:, :, S] * f[:, :, s:s + 1]
        A[:, :, QT] = row
    return b.o((A @ sp(v)).transpose(1, 2).reshape(n, T, b.h * b.dh))


def run(nm, K=64, n=4000, out_name='qpatch.npz'):
    t0 = time.time(); model, meta = N.load_run(os.path.join('runs', nm)); model.requires_grad_(False); sg = meta['sigma']; d_ = model.d
    Tz = np.load(os.path.join(DP.TP, 'runs/testset.npz'))
    Tt = {k: Tz[k][:n] for k in ('comp_means', 'w_bayes', 'mean_bayes', 'var_bayes')}; Tt['comp_vars'] = Tz['comp_vars']
    dT = dict(y=Tz['y'][:n], o=Tz['o'][:n], v=Tz['v'][:n], vidx=Tz['vidx'][:n])
    wb = QB.bayes_features(dT)[:, 20:23].astype(np.float64); wb = wb / wb.sum(1, keepdims=True)
    wd = np.clip(QWT.bayes_weights({k: a.copy() for k, a in dT.items()}), 0, None); sd_ = wd.sum(1, keepdims=True)
    wd = np.where(sd_ > 1e-9, wd / np.maximum(sd_, 1e-12), wb)
    pt = 1.0 / dT['v']; pt = pt / pt.sum(1, keepdims=True)
    W_ = {'Pbayes': torch.as_tensor(wd, dtype=torch.float32), 'PbayesQQ': torch.as_tensor(wd, dtype=torch.float32),
          'Ppost': torch.as_tensor(wb, dtype=torch.float32), 'PpostQQ': torch.as_tensor(wb, dtype=torch.float32),
          'Ptrue': torch.as_tensor(pt, dtype=torch.float32), 'Puniform': torch.full((n, 3), 1.0 / 3)}
    # mean query row over fit episodes (one noise draw)
    rng = np.random.default_rng(98_500); dF = W.sample_np(4096, rng); XF, vF = W.tokens_np(dF, rng, 16)
    st, nz, allowF = qw.trace_nz(model, XF, vF, sg, torch.Generator().manual_seed(3))
    b = model.blocks[1]
    with torch.no_grad():
        xF = st['r1']; q, k, v = b.qkv(b.ln1(xF)).split(b.h * b.dh, -1); sp = lambda t: t.view(len(xF), T, b.h, b.dh).transpose(1, 2)
        Abar = ((sp(q) @ sp(k).transpose(-1, -2)) / math.sqrt(b.dh)).masked_fill(~allowF[:, None], float('-inf')).softmax(-1)[:, :, QT].mean(0)
    Xt = torch.from_numpy(Tz['X'][:n]); al = model.allow_mask(torch.from_numpy(Tz['valid'][:n]))
    heads = {v_: ([], [], []) for v_ in VARIANTS}; g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for kk in range(K):
            nzd = {s: torch.randn((n, T, (2 * d_ if s[0] == 'm' else d_)), generator=g) for s in model.site_names()}
            x = model.f_emb(Xt) + sg * nzd['emb']
            xa = model.f_a(1, x, al) + sg * nzd['a1']; m = model.f_m(1, xa) + sg * nzd['m1']; x1 = model.f_r(1, xa, m) + sg * nzd['r1']
            for vr in VARIANTS:
                xa2 = N.sphere(x1 + layer2_attn(b, x1, al, vr, W_.get(vr), Abar)) + sg * nzd['a2']
                m2 = model.f_m(2, xa2) + sg * nzd['m2']; r2 = model.f_r(2, xa2, m2) + sg * nzd['r2']
                mu, ls, lg = model.head(r2)
                heads[vr][0].append(mu); heads[vr][1].append(ls); heads[vr][2].append(torch.log_softmax(lg, -1) - math.log(K))
    out = {}
    for vr in VARIANTS:
        h = tuple(torch.cat(t, 1).numpy() for t in heads[vr]); dv = qw.deviations_t(h, Tt)
        for f_ in ('mean', 'logsd', 'kl'): out[f'{vr}_{f_}'] = dv[f_]
    np.savez(os.path.join('runs', nm, out_name), **out)
    print(nm, 'done %.0fs' % (time.time() - t0), {vr: round(float((out[f'{vr}_mean'] ** 2).mean() / (out['base_mean'] ** 2).mean()), 2) for vr in VARIANTS}, flush=True)


if __name__ == '__main__':
    for nm in sys.argv[1:]:
        if not os.path.exists(os.path.join('runs', nm, 'qpatch.npz')): run(nm)
