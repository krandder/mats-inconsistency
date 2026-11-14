"""E9 (PREREG-B amendment 16): where do the layer-2 attention masses become the net's effective source weights?
E9.1  exact path split of the effective weights w_s = d m / d o*_s (noise-marginal predictive mean, K = 8, autograd):
      layer-2 query-row inputs R (the query token's own residual x1_q), M (attention logits: queries and keys) and
      V (values), each split into the direct route and the route through MLP2. Sum of the six parts = full gradient.
E9.2  patches of the value path and of MLP2, and both together (K = 64 deviations; K = 8 effective weights):
      PVg  QQ-token values: the coordinate along each value's fitted reading direction set to the reading itself
           (removes episode-dependent gain on the reading; everything orthogonal kept);
      PV   QQ-token values replaced by their fitted linear function of the reading (a + b o*_s);
      PM   MLP2 (query row) replaced by its least-squares linear approximation, keeping the same noise draw;
      PVgPM, PVPM  both together.
Fits use the 8,192 E8 fit episodes (one noise draw). Evaluation: first 4,000 test episodes; noise generator seed 1 for
K = 64 (as E8), seeds 2 + batch for K = 8.
usage: python qe9.py NET [NET ...] -> runs/NET/qfit9.npz, runs/NET/qe9.npz
"""
import math, os, sys, time
import numpy as np
import torch
import nnoise as N
import devpred as DP
import qw
import qweights as QWT
from qpatch import T, QT, QQ

torch.set_num_threads(1)
VAL = 32
VARIANTS = ['base', 'PVg', 'PV', 'PM', 'PVgPM', 'PVPM']
PATHS = {'R_direct': (1, 0, 0, 'direct'), 'R_mlp': (1, 0, 0, 'mlp'), 'M_direct': (0, 1, 0, 'direct'), 'M_mlp': (0, 1, 0, 'mlp'),
         'V_direct': (0, 0, 1, 'direct'), 'V_mlp': (0, 0, 1, 'mlp'), 'full': (1, 1, 1, 'both')}
QQMASK = torch.zeros(T, dtype=torch.bool); QQMASK[QQ] = True


def noise_list(n, d_, g):                              # emb, a1, m1, r1, a2, m2, r2
    return [torch.randn((n, T, w), generator=g) for w in (d_, d_, 2 * d_, d_, d_, 2 * d_, d_)]


def layer1(model, X, al, sg, z):
    x = model.f_emb(X) + sg * z[0]; xa = model.f_a(1, x, al) + sg * z[1]; m = model.f_m(1, xa) + sg * z[2]
    return model.f_r(1, xa, m) + sg * z[3]


def fwd(model, X, al, sg, z, P=None, variant='base', cfg=None, vmask=None):
    n = len(X); b = model.blocks[1]; x1 = layer1(model, X, al, sg, z)
    q, k, v = b.qkv(b.ln1(x1)).split(b.h * b.dh, -1)
    sp = lambda t: t.view(n, T, b.h, b.dh).transpose(1, 2)
    q, k, v = sp(q), sp(k), sp(v)                                            # (n, h, T, dh)
    if variant in ('PVg', 'PV', 'PVgPM', 'PVPM'):
        o = X[:, QQ, VAL][:, None, :, None]                                  # (n, 1, 3, 1)
        a_, b_ = P['a'][None], P['b'][None]                                  # (1, h, 3, dh)
        vq = v[:, :, QQ, :]
        if variant.startswith('PVg'):
            proj = ((vq - a_) * b_).sum(-1, keepdim=True) / (b_ ** 2).sum(-1, keepdim=True)
            vq = vq + b_ * (o - proj)
        else:
            vq = a_ + b_ * o
        full = torch.zeros_like(v); full[:, :, QQ, :] = vq
        v = torch.where(QQMASK[None, None, :, None], full, v)
    res, mass, val, route = cfg if cfg is not None else (1, 1, 1, 'both')
    qq_, kk_ = (q, k) if mass else (q.detach(), k.detach()); vv_ = v if val else v.detach()
    if vmask is not None: vv_ = torch.where(vmask[None, None, :, None], v, v.detach())
    A = ((qq_ @ kk_.transpose(-1, -2)) / math.sqrt(b.dh)).masked_fill(~al[:, None], float('-inf')).softmax(-1)
    att = b.o((A @ vv_).transpose(1, 2).reshape(n, T, b.h * b.dh))
    xa2 = N.sphere((x1 if res else x1.detach()) + att) + sg * z[4]
    if variant in ('PM', 'PVgPM', 'PVPM'):
        u = xa2 + xa2 @ P['L'].T + P['c'] + sg * (z[5] @ b.mlp[2].weight.T)
    elif route == 'both':
        u = xa2 + b.mlp[2](model.f_m(2, xa2) + sg * z[5])
    elif route == 'direct':
        u = xa2 + b.mlp[2](model.f_m(2, xa2.detach()) + sg * z[5])
    else:
        u = xa2.detach() + b.mlp[2](model.f_m(2, xa2) + sg * z[5])
    r2 = N.sphere(u) + sg * z[6]
    return model.head(r2), A[:, :, QT]


def fit_params(model, sg, nm):
    p = os.path.join('runs', nm, 'qfit9.npz')
    if os.path.exists(p):
        z = np.load(p); return {k: torch.as_tensor(z[k], dtype=torch.float32) for k in ('a', 'b', 'L', 'c')}, {k: z[k] for k in z.files}
    fit, _ = QWT.worlds(); X = torch.from_numpy(fit['X']); al = model.allow_mask(torch.from_numpy(fit['v'])); n = len(X); b = model.blocks[1]
    z = noise_list(n, model.d, torch.Generator().manual_seed(31))
    with torch.no_grad():
        x1 = layer1(model, X, al, sg, z); q, k, v = b.qkv(b.ln1(x1)).split(b.h * b.dh, -1)
        sp = lambda t: t.view(n, T, b.h, b.dh).transpose(1, 2); q, k, v = sp(q), sp(k), sp(v)
        vq = v[:, :, QQ, :].double(); o = X[:, QQ, VAL].double()                               # (n, h, 3, dh), (n, 3)
        om, vm = o.mean(0), vq.mean(0)
        bb = ((o - om)[:, None, :, None] * (vq - vm)).mean(0) / ((o - om) ** 2).mean(0)[None, :, None]
        aa = vm - bb * om[None, :, None]
        fitted = aa[None] + bb[None] * o[:, None, :, None]
        r2_lin = 1 - ((vq - fitted) ** 2).sum((0, 3)) / ((vq - vm) ** 2).sum((0, 3))                # (h, 3) share of value variance
        A = ((q @ k.transpose(-1, -2)) / math.sqrt(b.dh)).masked_fill(~al[:, None], float('-inf')).softmax(-1)
        att = b.o((A @ v).transpose(1, 2).reshape(n, T, b.h * b.dh))
        xa2 = (N.sphere(x1 + att) + sg * z[4])[:, QT].double()
        tgt = b.mlp[2](model.f_m(2, (N.sphere(x1 + att) + sg * z[4])))[:, QT].double()
        Z = torch.cat([xa2, torch.ones(n, 1, dtype=torch.float64)], 1)
        sol = torch.linalg.lstsq(Z, tgt).solution                                                 # (d + 1, d)
        L, c = sol[:-1].T, sol[-1]
        r2_mlp = float(1 - ((tgt - Z @ sol) ** 2).sum() / ((tgt - tgt.mean(0)) ** 2).sum())
    out = dict(a=aa.float().numpy(), b=bb.float().numpy(), L=L.float().numpy(), c=c.float().numpy(), r2_value_linear=r2_lin.numpy(), r2_mlp_linear=np.array(r2_mlp))
    tmp = p + '.tmp.npz'; np.savez(tmp, **out); os.replace(tmp, p)
    return {k: torch.as_tensor(out[k], dtype=torch.float32) for k in ('a', 'b', 'L', 'c')}, out


def eff_weights(model, X, valid, sg, P, variant='base', cfg=None, K=8, bs=500, vmask=None):
    Ws, Ms = [], []
    for bi, i in enumerate(range(0, len(X), bs)):
        g = torch.Generator().manual_seed(2 + 1000 * bi)
        Xi = torch.from_numpy(X[i:i + bs]).clone().requires_grad_(True); al = model.allow_mask(torch.from_numpy(valid[i:i + bs]))
        m = 0; mass = 0
        for _ in range(K):
            z = noise_list(len(Xi), model.d, g)
            (mu, ls, lg), Aq = fwd(model, Xi, al, sg, z, P, variant, cfg, vmask)
            m = m + (torch.softmax(lg, -1) * mu).sum(-1) / K; mass = mass + Aq[:, :, QQ].detach() / K
        gr, = torch.autograd.grad(m.sum(), Xi)
        Ws.append(gr[:, QQ, VAL].numpy().astype(np.float64)); Ms.append(mass.numpy())
    return np.concatenate(Ws), np.concatenate(Ms)


def deviations(model, nm, sg, P, K=64, n=4000):
    Tz = np.load(os.path.join(DP.TP, 'runs/testset.npz'))
    Tt = {k: Tz[k][:n] for k in ('comp_means', 'w_bayes', 'mean_bayes', 'var_bayes')}; Tt['comp_vars'] = Tz['comp_vars']
    Xt = torch.from_numpy(Tz['X'][:n]); al = model.allow_mask(torch.from_numpy(Tz['valid'][:n]))
    heads = {v_: ([], [], []) for v_ in VARIANTS}; g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for _ in range(K):
            z = noise_list(n, model.d, g)
            for vr in VARIANTS:
                (mu, ls, lg), _ = fwd(model, Xt, al, sg, z, P, vr)
                heads[vr][0].append(mu); heads[vr][1].append(ls); heads[vr][2].append(torch.log_softmax(lg, -1) - math.log(K))
    out = {}
    for vr in VARIANTS:
        h = tuple(torch.cat(t, 1).numpy() for t in heads[vr]); dv = qw.deviations_t(h, Tt)
        for f_ in ('mean', 'logsd', 'kl'): out[f'{vr}_{f_}'] = dv[f_]
    return out


def run(nm, n=4000, K=64, out_name='qe9.npz', quiet=False):
    t0 = time.time(); model, meta = N.load_run(os.path.join('runs', nm)); model.requires_grad_(False); sg = meta['sigma']
    P, info = fit_params(model, sg, nm)
    Tz = np.load(os.path.join(DP.TP, 'runs/testset.npz')); X, valid = Tz['X'][:n], Tz['valid'][:n]
    out = {}
    for name, cfg in PATHS.items():
        w, mass = eff_weights(model, X, valid, sg, P, 'base', cfg); out[f'path_{name}'] = w
        if name == 'full': out['masses_qq'] = mass
    for vr in VARIANTS[1:]:
        out[f'w_{vr}'], _ = eff_weights(model, X, valid, sg, P, vr)
    out.update(deviations(model, nm, sg, P, K=K, n=n))
    np.savez(os.path.join('runs', nm, out_name), **out)
    chk = np.abs(sum(out[f'path_{k}'] for k in PATHS if k != 'full') - out['path_full']).max()
    if not quiet:
        print(nm, 'done %.0fs' % (time.time() - t0), 'path-sum check %.2e' % chk, 'value-fit R2', np.round(info['r2_value_linear'], 2).tolist(),
              'MLP2 linear R2 %.3f' % float(info['r2_mlp_linear']), flush=True)
    return chk


if __name__ == '__main__':
    for nm in sys.argv[1:]:
        if not os.path.exists(os.path.join('runs', nm, 'qe9.npz')): run(nm)
