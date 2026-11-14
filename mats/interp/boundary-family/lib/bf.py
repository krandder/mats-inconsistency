"""Boundary-family lane (2026-10-29): shared helpers.

Nets are ARC's family exactly as in work/arc-noisy/anet.py (copied here): x ~ N(0, I_n), z_l = W_l h_{l-1},
a_l = ReLU(z_l), h_l = a_l + s_l eps_l, W_l iid N(0, 2/n), s_l = sigma r_l with r_l the RMS of a_l in the noisy net
(seeded 2^12-path pilot). Width 64 nets use the same weight generator with n = 64 (different nets from the width-256
ones of the same seed); this lane uses seeds >= 500 at width 64 (dev 500-509, eval 520-559).
"""
import json, math, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import anet  # noqa: E402

RT = os.path.join(HERE, 'rtable_bf.json')


def get_net(seed, n=64, L=32, sigma=0.001):
    """anet.Net with the noise scales cached in rtable_bf.json (width 256 entries fall back to the ARC table)."""
    key = f"{n}_{L}_{seed}_{sigma:g}"
    tab = json.load(open(RT)) if os.path.exists(RT) else {}
    if key not in tab and sigma > 0:
        arc = anet.load_rtable()
        if key in arc:
            r = arc[key]
        else:
            W = anet.make_weights(seed, n, L)
            r = anet.pilot_rms(W, sigma, seed=seed)
        tab[key] = r
        tmp = RT + f'.tmp{os.getpid()}'
        json.dump(tab, open(tmp, 'w')); os.replace(tmp, RT)
    r = tab.get(key)
    return anet.Net(seed, n=n, L=L, sigma=sigma, r=r)


@torch.no_grad()
def mc_stats(net, N, seed=0, chunk=1 << 16, L=None, out_moments=True, cuts=(), keep_n=0, dtype=torch.float32):
    """Plain Monte Carlo over x and the noise. Returns dict with
    mean[l] (n,) = E[a_l], sq[l] = E[a_l^2] for l = 1..L (index l-1), zmom (4, n) raw moments of z_L = W_L h_{L-1},
    and samples of z_m (pre-activation) at each cut m (first keep_n paths)."""
    L = L or net.L
    n = net.n
    g = torch.Generator().manual_seed(seed)
    W = [w.to(dtype) for w in net.W[:L]]
    s = [float(v) for v in net.s[:L]]
    S1 = torch.zeros(L, n, dtype=torch.float64); S2 = torch.zeros(L, n, dtype=torch.float64)
    ZM = torch.zeros(4, n, dtype=torch.float64)
    keep = {m: [] for m in cuts}; kept = 0
    done = 0
    while done < N:
        b = min(chunk, N - done)
        h = torch.randn(b, n, generator=g, dtype=dtype)
        for l in range(L):
            z = h @ W[l].T
            if l == L - 1 and out_moments:
                zd = z.double()
                p = zd.clone()
                for k in range(4):
                    ZM[k] += p.sum(0)
                    if k < 3:
                        p.mul_(zd)
            if (l + 1) in keep and kept < keep_n:
                keep[l + 1].append(z[:max(0, keep_n - kept)].clone())
            a = torch.relu(z)
            ad = a.double()
            S1[l] += ad.sum(0); S2[l] += (ad * ad).sum(0)
            if l < L - 1 and net.sigma > 0:
                h = a + s[l] * torch.randn(b, n, generator=g, dtype=dtype)
            else:
                h = a
        kept += b
        done += b
    out = dict(N=N, mean=(S1 / N), sq=(S2 / N), zmom=(ZM / N))
    out['cuts'] = {m: torch.cat(v, 0) for m, v in keep.items() if v}
    return out


@torch.no_grad()
def propagate_from(net, z_m, m, L, seed=1, chunk=1 << 16, dtype=torch.float32, ret_z=False):
    """Given samples of the pre-activation z_m (layer m, 1-based), apply ReLU, the noise after layer m (if m < L),
    and the exact layers m+1..L. Returns E[a_L] estimate (n,) (float64) and optionally raw moments of z_L."""
    g = torch.Generator().manual_seed(seed)
    n = net.n
    S1 = torch.zeros(n, dtype=torch.float64); ZM = torch.zeros(4, n, dtype=torch.float64)
    N = z_m.shape[0]
    for a0 in range(0, N, chunk):
        z = z_m[a0:a0 + chunk].to(dtype)
        b = z.shape[0]
        for l in range(m - 1, L):       # l is the 0-based index of the layer whose pre-activation z is
            if l > m - 1:
                z = h @ net.W[l].to(dtype).T
            if l == L - 1:
                zd = z.double(); p = zd.clone()
                for k in range(4):
                    ZM[k] += p.sum(0)
                    if k < 3:
                        p.mul_(zd)
            a = torch.relu(z)
            if l == L - 1:
                S1 += a.double().sum(0)
                break
            h = a + float(net.s[l]) * torch.randn(b, n, generator=g, dtype=dtype) if net.sigma > 0 else a
    out = S1 / N
    return (out, ZM / N) if ret_z else out


def nmse(est, truth_mean, truth_var, n_paths_truth=None):
    """ARC's nMSE: mean_j (est_j - truth_j)^2 / mean_j Var(a_L,j); optionally subtract the truth's own 1/N."""
    v = float(truth_var.mean())
    e = float(((est - truth_mean) ** 2).mean()) / v
    if n_paths_truth:
        e -= 1.0 / n_paths_truth
    return e
