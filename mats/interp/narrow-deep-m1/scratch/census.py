"""Census of plain width-8 depth-256 nets (lib nd.Net, same family as the narrow-deep lane).
usage: python census.py SEED_FROM SEED_TO OUTJSON [N=8] [L=256]
Per seed:
- noiseless: 2^16 inputs; fraction of inputs that have passed an all-off layer by depth 16/32/64/128/256
  ("dead" net: every sampled input has; then the noiseless output is exactly 0 for all of them);
- noisy (sigma = 1e-3, 1e-2): pilot RMS r_l (min over layers), output tr Var, share of output variance explained by
  the input (two independent noise draws per input: Cov(out1, out2) = Var(E[out | x])), fraction of paths with an
  all-off layer.
No estimator is run here; this only classifies nets (healthy / dying) before any estimation result.
"""
import sys, os, json, math, time
import numpy as np
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep/lib')
import nd

torch.set_num_threads(1)
S0, S1, OUT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 8
L = int(sys.argv[5]) if len(sys.argv) > 5 else 256
DT = nd.DT
P = 1 << 16
CH = [16, 32, 64, 128, 256, 512, 1024, 2048]
res = {}
for seed in range(S0, S1):
    t0 = time.time()
    row = dict(seed=seed)
    net0 = nd.Net('plain', seed, N, L, 0.0)
    g = torch.Generator().manual_seed(31_000_003 + seed)
    x = torch.randn(P, N, generator=g, dtype=DT)
    h = x; dead = torch.zeros(P, dtype=torch.bool); dfrac = {}
    for l in range(L):
        h = torch.relu(h @ net0.W[l].T)
        dead |= (h.abs().sum(1) == 0)
        if (l + 1) in CH:
            dfrac[l + 1] = float(dead.double().mean())
    row['dead0'] = dfrac
    row['out0_rms'] = float(h.pow(2).sum(1).mean().sqrt())
    for sig in (1e-3, 1e-2):
        net = nd.Net('plain', seed, N, L, sig)
        r = net.r.numpy()
        y1 = net.forward(x, gen=g); y2 = net.forward(x, gen=g)
        m1 = y1.mean(0)
        v = ((y1 - m1) ** 2).mean(0).sum()
        cv = ((y1 - m1) * (y2 - y2.mean(0))).mean(0).sum()
        # dead fraction in the noisy net (all units off at the output layer)
        row[f's{sig:g}'] = dict(rmin=float(r.min()), rmin_layer=int(r.argmin()) + 1, r_last=float(r[-1]),
                               trvar=float(v), mean_sq=float((m1 ** 2).sum()),
                               input_share=float(cv / v) if v > 0 else None,
                               out_zero_frac=float((y1.abs().sum(1) == 0).double().mean()))
    res[seed] = row
    s3 = row['s0.001']
    print(f"seed {seed}: dead0 L64/128/256 = {dfrac[64]:.3f}/{dfrac[128]:.3f}/{dfrac[L]:.3f}; s=1e-3 trvar {s3['trvar']:.3g} "
          f"input share {s3['input_share']}, out0 frac {s3['out_zero_frac']:.3f}, rmin {s3['rmin']:.3g}@{s3['rmin_layer']}; "
          f"s=1e-2 share {row['s0.01']['input_share']} ({time.time()-t0:.1f}s)", flush=True)
json.dump(dict(N=N, L=L, P=P, res=res), open(OUT, 'w'), indent=1)
