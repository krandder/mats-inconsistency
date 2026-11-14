"""E8.2b (post hoc, PREREG-B amendment 15 addendum 3): dose-response and single-head versions of the layer-2 Bayes patch.
Per-source masses of the query row m'_s = (1 - a) m_s + a * M * Wt_s (M = the head's total source mass, Wt = W_s clipped at
0 and renormalised; within-source split kept). Variants: base; PbA10 / PbA25 / PbA50 (a = 0.1 / 0.25 / 0.5, both heads);
PbH / PbH50 (a = 1 / 0.5, only the head with the largest average query-row mass on the QQ tokens, chosen on fit episodes).
K = 64, first 4,000 test episodes, the same noise draws as qpatch.py (generator seed 1).
usage: python qpatch2.py NET [NET ...] -> runs/NET/qpatch2.npz
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
from qpatch import T, QT, SRC, QQ

torch.set_num_threads(1)
SPEC = {'base': None, 'PbA10': (0.1, None), 'PbA25': (0.25, None), 'PbA50': (0.5, None), 'PbH': (1.0, 'qq'), 'PbH50': (0.5, 'qq')}


def layer2_mix(b, x, allow, spec, wts, hq):
    n = len(x); q, k, v = b.qkv(b.ln1(x)).split(b.h * b.dh, -1)
    sp = lambda t: t.view(n, T, b.h, b.dh).transpose(1, 2)
    A = ((sp(q) @ sp(k).transpose(-1, -2)) / math.sqrt(b.dh)).masked_fill(~allow[:, None], float('-inf')).softmax(-1)
    if spec is not None:
        a, which = spec; A = A.clone(); row = A[:, :, QT].clone()                       # (n, h, T)
        m = torch.stack([row[:, :, S].sum(-1) for S in SRC], -1); tot = m.sum(-1, keepdim=True)
        new = (1 - a) * m + a * tot * wts[:, None, :]
        f = new / m.clamp_min(1e-12)
        heads = range(b.h) if which is None else [hq]
        for h in heads:
            for s, S in enumerate(SRC): row[:, h, S] = row[:, h, S] * f[:, h, s:s + 1]
        A[:, :, QT] = row
    return b.o((A @ sp(v)).transpose(1, 2).reshape(n, T, b.h * b.dh))


def run(nm, K=64, n=4000, out_name='qpatch2.npz'):
    t0 = time.time(); model, meta = N.load_run(os.path.join('runs', nm)); model.requires_grad_(False); sg = meta['sigma']; d_ = model.d
    Tz = np.load(os.path.join(DP.TP, 'runs/testset.npz'))
    Tt = {k: Tz[k][:n] for k in ('comp_means', 'w_bayes', 'mean_bayes', 'var_bayes')}; Tt['comp_vars'] = Tz['comp_vars']
    dT = dict(y=Tz['y'][:n].copy(), o=Tz['o'][:n].copy(), v=Tz['v'][:n].copy(), vidx=Tz['vidx'][:n].copy())
    wb = QB.bayes_features(dT)[:, 20:23].astype(np.float64); wb = wb / wb.sum(1, keepdims=True)
    wd = np.clip(QWT.bayes_weights({k: a.copy() for k, a in dT.items()}), 0, None); sd_ = wd.sum(1, keepdims=True)
    wd = torch.as_tensor(np.where(sd_ > 1e-9, wd / np.maximum(sd_, 1e-12), wb), dtype=torch.float32)
    rng = np.random.default_rng(98_500); dF = W.sample_np(4096, rng); XF, vF = W.tokens_np(dF, rng, 16)
    st, nz, allowF = qw.trace_nz(model, XF, vF, sg, torch.Generator().manual_seed(3))
    b = model.blocks[1]
    with torch.no_grad():
        xF = st['r1']; q, k, v = b.qkv(b.ln1(xF)).split(b.h * b.dh, -1); sp = lambda t: t.view(len(xF), T, b.h, b.dh).transpose(1, 2)
        Abar = ((sp(q) @ sp(k).transpose(-1, -2)) / math.sqrt(b.dh)).masked_fill(~allowF[:, None], float('-inf')).softmax(-1)[:, :, QT].mean(0)
    hq = int(Abar[:, QQ].sum(-1).argmax()); qqmass = [float(x) for x in Abar[:, QQ].sum(-1)]
    Xt = torch.from_numpy(Tz['X'][:n]); al = model.allow_mask(torch.from_numpy(Tz['valid'][:n]))
    heads = {v_: ([], [], []) for v_ in SPEC}; g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for kk in range(K):
            nzd = {s: torch.randn((n, T, (2 * d_ if s[0] == 'm' else d_)), generator=g) for s in model.site_names()}
            x = model.f_emb(Xt) + sg * nzd['emb']
            xa = model.f_a(1, x, al) + sg * nzd['a1']; m = model.f_m(1, xa) + sg * nzd['m1']; x1 = model.f_r(1, xa, m) + sg * nzd['r1']
            for vr, spec in SPEC.items():
                xa2 = N.sphere(x1 + layer2_mix(b, x1, al, spec, wd, hq)) + sg * nzd['a2']
                m2 = model.f_m(2, xa2) + sg * nzd['m2']; r2 = model.f_r(2, xa2, m2) + sg * nzd['r2']
                mu, ls, lg = model.head(r2)
                heads[vr][0].append(mu); heads[vr][1].append(ls); heads[vr][2].append(torch.log_softmax(lg, -1) - math.log(K))
    out = dict(qq_head=hq, qq_mass=np.array(qqmass))
    for vr in SPEC:
        h = tuple(torch.cat(t, 1).numpy() for t in heads[vr]); dv = qw.deviations_t(h, Tt)
        for f_ in ('mean', 'logsd', 'kl'): out[f'{vr}_{f_}'] = dv[f_]
    np.savez(os.path.join('runs', nm, out_name), **out)
    print(nm, 'done %.0fs' % (time.time() - t0), 'qq head', hq, 'qq mass', [round(x, 2) for x in qqmass], flush=True)


if __name__ == '__main__':
    for nm in sys.argv[1:]:
        if not os.path.exists(os.path.join('runs', nm, 'qpatch2.npz')): run(nm)
