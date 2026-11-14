"""Monte Carlo truths on GPU (also runs on CPU for small checks).
usage: truth_gpu.py OUTDIR LOG2N SIGMA[,SIGMA..] SEED[,SEED..] [--L 32] [--batch 16] [--m2 8,16,24,32]

Per (seed, sigma): sums of a_l and a_l^2 over N = 2^LOG2N noisy paths for every layer l = 1..L (float64
accumulators of float32 batch sums; TF32 off), plus at layers 8 and L the Rao-Blackwellised readout
g_l = E[a_l | a_{l-1}] (noise after a_{l-1} integrated per neuron) and its square; optional second-moment
matrices sum a_l a_l^T at the --m2 layers. Saved to OUTDIR/truth_<n>_<L>_s<sigma>_seed<seed>.npz with the
weights checksum, which the analysis compares with anet.weights_checksum.
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anet

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('log2n', type=int); ap.add_argument('sigmas'); ap.add_argument('seeds')
ap.add_argument('--L', type=int, default=32); ap.add_argument('--n', type=int, default=256)
ap.add_argument('--batch', type=int, default=14); ap.add_argument('--m2', default='')
ap.add_argument('--rb', default='8,32')
args = ap.parse_args()
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
os.makedirs(args.out, exist_ok=True)
rtab = anet.load_rtable()
m2_layers = [int(v) for v in args.m2.split(',') if v]
rb_layers = [int(v) for v in args.rb.split(',') if v and int(v) <= args.L]
SQ2PI = math.sqrt(2 * math.pi)


def g_relu(mu, sd):
    t = mu / sd
    return mu * torch.special.ndtr(t) + sd * torch.exp(-0.5 * t * t) / SQ2PI


for seed in [int(v) for v in args.seeds.split(',')]:
    W64 = anet.make_weights(seed, args.n, args.L)
    chk = anet.weights_checksum(W64)
    W = [w.float().to(dev) for w in W64]
    WT = [w.T.contiguous() for w in W]
    for sigma in [float(v) for v in args.sigmas.split(',')]:
        path = os.path.join(args.out, f"truth_{args.n}_{args.L}_s{sigma:g}_seed{seed}.npz")
        if os.path.exists(path):
            print('skip', path, flush=True); continue
        t0 = time.time()
        if sigma > 0:
            key = f"{args.n}_32_{seed}_{sigma:g}"
            r = rtab[key][:args.L] if key in rtab else anet.pilot_rms(W64, sigma, seed=seed)
        else:
            r = [1.0] * args.L
        s = [sigma * v for v in r]
        sd_rb = {l: (s[l - 2] * W[l - 1].norm(dim=1)).double() for l in rb_layers if l >= 2 and sigma > 0}
        n, L, B = args.n, args.L, 1 << args.batch
        S1 = torch.zeros(L, n, dtype=torch.float64, device=dev); S2 = torch.zeros_like(S1)
        R1 = {l: torch.zeros(n, dtype=torch.float64, device=dev) for l in sd_rb}
        R2 = {l: torch.zeros(n, dtype=torch.float64, device=dev) for l in sd_rb}
        M2 = {l: torch.zeros(n, n, dtype=torch.float64, device=dev) for l in m2_layers}
        gen = torch.Generator(device=dev).manual_seed(424_242 + 1000 * seed + int(round(sigma * 1e6)))
        nb = 1 << max(0, args.log2n - args.batch)
        zbuf = torch.empty(B, n, device=dev, dtype=torch.float32)
        hbuf = torch.empty(B, n, device=dev, dtype=torch.float32)
        ebuf = torch.empty(B, n, device=dev, dtype=torch.float32)
        tb = time.time()
        for b in range(nb):
            torch.randn(B, n, generator=gen, out=hbuf)
            h = hbuf
            for l in range(L):
                a = torch.mm(h, WT[l], out=zbuf)
                torch.relu_(a)
                S1[l] += a.sum(0).double()
                S2[l] += torch.linalg.vector_norm(a, dim=0).double().pow(2)
                if (l + 1) in M2:
                    M2[l + 1] += (a.T @ a).double()
                if l + 2 in sd_rb:                      # RB readout of layer l+2 from the pre-noise a_{l+1}
                    gl = g_relu((a @ WT[l + 1]).double(), sd_rb[l + 2][None, :])
                    R1[l + 2] += gl.sum(0); R2[l + 2] += (gl * gl).sum(0)
                if l < L - 1:
                    if sigma > 0:
                        torch.randn(B, n, generator=gen, out=ebuf)
                        a.add_(ebuf, alpha=s[l])
                    hbuf.copy_(a); h = hbuf
            if b == 3 and seed == int(args.seeds.split(',')[0]):
                torch.cuda.synchronize() if dev.type == 'cuda' else None
                print(f"  timing: {(time.time() - tb) / 4 * 1000:.1f} ms per batch of 2^{args.batch}", flush=True)
        N = nb * B
        out = dict(S1=S1.cpu().numpy(), S2=S2.cpu().numpy(), N=N, r=np.array(r), sigma=sigma, seed=seed, chk=chk,
                   n=n, L=L, secs=time.time() - t0, device=str(dev))
        for l in sd_rb:
            out[f'R1_{l}'] = R1[l].cpu().numpy(); out[f'R2_{l}'] = R2[l].cpu().numpy()
        for l in M2:
            out[f'M2_{l}'] = M2[l].cpu().numpy()
        np.savez_compressed(path, **out)
        mu = S1[-1] / N; var = S2[-1] / N - mu * mu
        print(f"seed {seed} sigma {sigma:g} N=2^{int(math.log2(N))} mean|a_L| {mu.mean():.4f} var {var.mean():.5f} "
              f"[{time.time() - t0:.0f}s]", flush=True)
