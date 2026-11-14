"""Check the radial Rao-Blackwellised tail estimator against plain MC at large N, with and without noise, on one
residual net (n = 16, L = 64, seed 0): is its ~5-10% offset a bias of the noise approximation?"""
import sys, os, math
import numpy as np
import torch
from scipy.stats import chi
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import nd

torch.set_num_threads(1)
DT = nd.DT
for sigma in (0.0, 0.001):
    net = nd.Net('res', 0, 16, 64, sigma)
    g = torch.Generator().manual_seed(5)
    norms = []
    for i in range(32):
        norms.append(net.forward(torch.randn(1 << 16, 16, generator=g, dtype=DT), gen=g).norm(dim=1))
    norms = torch.cat(norms)
    for q in (1e-3, 1e-4):
        t = float(torch.quantile(norms[:1 << 20].float(), 1 - q))
        p_mc = float((norms > t).double().mean())
        EX = nd.chi_mean(16)
        sv = net.s.clone(); net.s = net.s / EX
        G = []
        for i in range(8):
            u = torch.randn(1 << 16, 16, generator=g, dtype=DT); u = u / u.norm(dim=1, keepdim=True)
            G.append(net.forward(u, gen=g).norm(dim=1))
        net.s = sv
        G = torch.cat(G)
        rb = chi.sf((t / G).numpy(), 16)
        print(f"sigma={sigma} q={q}: MC p={p_mc:.4g} (rel SE {math.sqrt(p_mc / len(norms)) / p_mc:.3f}) RB p={rb.mean():.4g} "
              f"(rel SE {rb.std() / math.sqrt(len(rb)) / rb.mean():.3f})", flush=True)
