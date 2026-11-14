"""E3 (scoped): tiny-PFN layer expansion on q's support, trained (noise-trained) vs random init, plus the Gaussian-family
deductive chain's moments.

Layer maps between residual cuts: emb -> a1 (attention sublayer), a1 -> r1 (MLP sublayer; the hidden noise m1 is held at
its sampled value), ... For each map F and states h on the support of q (Gaussian chain) or p (real noisy states):
  expansion  = ||J||_F^2 / D (Hutchinson, 16 probes), top singular value (power iteration, 15 steps).
expansion > 1: the layer spreads noise-scale perturbations, so a description's support grows (waste can compound).

usage: python e3.py RUN_DIR   -> RUN_DIR/e3.json
"""
import json, math, os, sys, time
import numpy as np
import torch
from torch.func import jvp, vjp
import nnoise as N
import world as W
from scipy.stats import qmc, norm

torch.set_num_threads(1)


def layer_maps(model, valid, sigma, gen):
    """List of (name, F) where F maps the flattened residual state (B, T*d) -> next residual state (pre-noise)."""
    allow = model.allow_mask(valid)
    d = model.d; T = valid.shape[1]
    maps = []
    for l in range(1, model.depth + 1):
        def fa(x, l=l):
            return model.f_a(l, x.view(-1, T, d), allow[:x.shape[0]]).reshape(x.shape[0], -1)
        maps.append((f'r{l - 1 if l > 1 else 0}->a{l}' if l > 1 else 'emb->a1', fa))
        def fr(x, l=l):
            xa = x.view(-1, T, d)
            mm = model.f_m(l, xa)
            return model.f_r(l, xa, mm).reshape(x.shape[0], -1)          # hidden noise omitted inside the map (mean map)
        maps.append((f'a{l}->r{l}', fr))
    return maps


def expansion(F, H, probes=16, iters=15):
    H = H.detach(); D = H.shape[1]
    fro = torch.zeros(len(H))
    for _ in range(probes):
        v = torch.randn_like(H)
        _, jv = jvp(F, (H,), (v,))
        fro += (jv ** 2).sum(1)
    fro = fro / probes / D
    u = torch.randn_like(H); u = u / u.norm(dim=1, keepdim=True)
    _, vj = vjp(F, H)
    for _ in range(iters):
        _, ju = jvp(F, (H,), (u,))
        (w,) = vj(ju)
        u = w / w.norm(dim=1, keepdim=True).clamp_min(1e-12)
    _, ju = jvp(F, (H,), (u,))
    top = ju.norm(dim=1)
    return float(fro.mean()), float(top.mean())


def run(run_dir, n=48):
    model, meta = N.load_run(run_dir); sg = meta['sigma']
    rnd = N.build(meta); torch.manual_seed(123)
    rnd = N.NoisyPFN(rnd.inp.in_features, meta['depth'], meta['width'], meta['heads'], meta['mixk'], sg)   # random init
    rng = np.random.default_rng(90_000)
    d = W.sample_np(4096, rng); X, valid = W.tokens_np(d, rng, meta['ds'])
    Xt = torch.from_numpy(X); vt = torch.from_numpy(valid)
    out = dict(sigma=sg, depth=meta['depth'], rows=[])
    for tag, mdl in (('trained', model), ('random', rnd)):
        gen = torch.Generator().manual_seed(7)
        with torch.no_grad():
            _, _, _, st = mdl(Xt, vt, want=True, gen=gen)
        maps = layer_maps(mdl, vt, sg, gen)
        src = ['emb'] + [x for l in range(1, meta['depth'] + 1) for x in (f'a{l}',)]
        # p-support (real noisy states) and q-support (Gaussian chain: moments of K q pushed from Gaussian q)
        cuts = []
        for l in range(1, meta['depth'] + 1):
            cuts += [('emb' if l == 1 else f'r{l - 1}'), f'a{l}']
        mq = None
        for (name, F), cut in zip(maps, cuts):
            Hp = st[cut][:n].reshape(n, -1).double().float()
            fp, tp = expansion(F, Hp)
            # Gaussian q for this cut: moments of the real states would be p-based; instead use the chain:
            # q_cut = N(m, S) with m, S from pushing the previous q's QMC nodes (first cut: from the exact prior)
            if mq is None:
                with torch.no_grad():
                    base = mdl.f_emb(Xt).reshape(len(Xt), -1)
                m = base.mean(0); C = torch.cov(base.T) + sg ** 2 * torch.eye(base.shape[1])
            else:
                m, C = mq
            L = torch.linalg.cholesky(C.double() + 1e-6 * torch.eye(C.shape[0], dtype=torch.float64)).float()
            z = torch.as_tensor(norm.ppf(np.clip(qmc.Sobol(C.shape[0], scramble=True, seed=1).random(512), 1e-9, 1 - 1e-9)), dtype=torch.float32)
            Hq = m + z @ L.T
            fq, tq = expansion(F, Hq[:n])
            with torch.no_grad():
                Y = F(Hq) + sg * torch.randn(Hq.shape, generator=gen)
            mq = (Y.mean(0), torch.cov(Y.T))
            row = dict(net=tag, map=name, exp_p=fp, top_p=tp, exp_q=fq, top_q=tq,
                       q_tr=float(torch.trace(C)) / C.shape[0], p_var=float(Hp.var(0).mean()))
            out['rows'].append(row); print(row, flush=True)
            json.dump(out, open(os.path.join(run_dir, 'e3.json'), 'w'), indent=1)


if __name__ == '__main__':
    run(sys.argv[1])
