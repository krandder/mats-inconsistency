"""Sampling baselines on the noisy nets (GPU; also runs on CPU for small checks).
usage: gpu_samplers.py OUTDIR SEEDS SIGMAS METHODS R [--L 8,32] [--budgets ...]

For each (seed, sigma, L, method) and replicate r = 0..R-1, draws the method's design for the largest budget and
records prefix sums (float64) of a_L and of the control layers a_m at every budget checkpoint N_b = floor(B / c),
c = the method's flopscope cost per path. Also fits, on a separate 4,096-path plain-MC pilot, the least-squares map
a_L ~ [a_m, 1] for each control layer m (stored, so linear controls with any closure mean can be scored offline).
Saved: OUTDIR/samp_<seed>_s<sigma>_L<L>.npz with arrays sum_<method>_<layer> (R, n_checkpoints, n), N_<method>,
cost_<method>, A_<m>, b_<m> (pilot regression), and the budgets.
"""
import argparse, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anet, samplers

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('seeds'); ap.add_argument('sigmas'); ap.add_argument('methods'); ap.add_argument('R', type=int)
ap.add_argument('--L', default='8,32')
ap.add_argument('--budgets', default='1e9,2e9,5e9,1e10,2e10,5e10,6.8e10,1e11,2e11,2.72e11,3e11')
ap.add_argument('--chunk', type=int, default=14)
args = ap.parse_args()
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cuda.matmul.allow_tf32 = False
torch.set_float32_matmul_precision('highest')
os.makedirs(args.out, exist_ok=True)
budgets = np.array([float(b) for b in args.budgets.split(',')])
CTRL = {8: [2, 4], 32: [4, 8, 12, 16, 24]}
Bk = None
if 'mub' in args.methods:  # also mubanti
    import kerdock
    Bk = kerdock.build().astype(np.float64)
n = 256


def path_cost(net, L, method):
    c = net.flops_path(L)
    if method in ('mub', 'mubanti'):
        c = c - n * anet.F_NORMAL - anet.mm_flops(1, n, n) + n * n + n + 1600
    if method == 'mubanti':
        c = 2 * c                       # two paths per design point (first layer shared: counted twice, conservative)
    return c


@torch.no_grad()
def forward_from_z1(z1, WT, s, L, gen, keep, sign=None, noises=None):
    a = torch.relu(z1); acts = {}
    if 1 in keep:
        acts[1] = a
    h = a
    for l in range(1, L):
        if s[l - 1] > 0:
            e = noises[l - 1] if noises is not None else torch.randn(a.shape, generator=gen, device=a.device, dtype=a.dtype)
            h = a + (sign if sign is not None else 1.0) * s[l - 1] * e
        a = torch.relu(h @ WT[l])
        if (l + 1) in keep:
            acts[l + 1] = a
    return acts


def run_points(z1, WT, s, L, gen, keep, sums, idx, cps, start, anti=False):
    """Forward a chunk whose first-layer pre-activations are z1; accumulate prefix sums at checkpoints.
    anti=True: two paths per point with noise +e and -e, averaged (antithetic noise)."""
    if anti:
        noises = [torch.randn(z1.shape, generator=gen, device=z1.device, dtype=z1.dtype) for _ in range(L - 1)]
        A1 = forward_from_z1(z1, WT, s, L, gen, keep, 1.0, noises)
        A2 = forward_from_z1(z1, WT, s, L, gen, keep, -1.0, noises)
        acts = {ly: (A1[ly] + A2[ly]) / 2 for ly in keep}
    else:
        acts = forward_from_z1(z1, WT, s, L, gen, keep)
    m = z1.shape[0]
    while idx[0] < len(cps) and cps[idx[0]] <= start + m:
        off = int(cps[idx[0]] - start)
        for ly in keep:
            sums[ly]['out'][idx[0]] = sums[ly]['run'] + acts[ly][:off].sum(0, dtype=torch.float64)
        idx[0] += 1
    for ly in keep:
        sums[ly]['run'] += acts[ly].sum(0, dtype=torch.float64)


for seed in [int(v) for v in args.seeds.split(',')]:
    W64 = anet.make_weights(seed)
    for sigma in [float(v) for v in args.sigmas.split(',')]:
        for L in [int(v) for v in args.L.split(',')]:
            path = os.path.join(args.out, f"samp_{seed}_s{sigma:g}_L{L}.npz")
            if os.path.exists(path):
                print('skip', path, flush=True); continue
            t0 = time.time()
            net = anet.Net(seed, sigma=sigma, W=W64).truncate(L)
            WT = [w.float().T.contiguous().to(dev) for w in net.W]
            s = [float(v) for v in net.s]
            keep = [L] + CTRL[L]
            res = dict(budgets=budgets, sigma=sigma, seed=seed, L=L)
            # pilot regression a_L ~ [a_m, 1] from 4096 plain paths (separate stream)
            gp = torch.Generator(device=dev).manual_seed(5_550_000 + seed)
            x = torch.randn(4096, n, generator=gp, device=dev)
            acts = {}
            a = torch.relu(x @ WT[0]); hcur = a
            for l in range(L):
                if l > 0:
                    a = torch.relu(hcur @ WT[l])
                acts[l + 1] = a
                if l < L - 1:
                    hcur = a + s[l] * torch.randn(a.shape, generator=gp, device=dev) if s[l] > 0 else a
            for mm in CTRL[L]:
                X = torch.cat([acts[mm].double(), torch.ones(4096, 1, device=dev, dtype=torch.float64)], 1)
                sol = torch.linalg.lstsq(X.cpu(), acts[L].double().cpu()).solution
                res[f'A_{mm}'] = sol[:-1].T.numpy().astype(np.float32); res[f'b_{mm}'] = sol[-1].numpy()
            res['pilot_cost'] = 4096 * net.flops_path(L)
            for method in args.methods.split(','):
                c = path_cost(net, L, method)
                Nb = np.floor(budgets / c).astype(np.int64)
                if method in ('mub', 'mubanti'):
                    Nb = (Nb // 2) * 2
                Nb = np.maximum(Nb, 2)
                cps = np.unique(Nb); Nmax = int(cps[-1])
                res[f'N_{method}'] = cps; res[f'cost_{method}'] = c
                out = {ly: np.zeros((args.R, len(cps), n)) for ly in keep}
                for r in range(args.R):
                    gen = torch.Generator(device=dev).manual_seed(1_000_000 * (r + 1) + 1000 * seed + int(sigma * 1e5))
                    rng = np.random.default_rng(77_000 + 1000 * r + seed)
                    sums = {ly: dict(run=torch.zeros(n, device=dev, dtype=torch.float64),
                                     out=torch.zeros(len(cps), n, device=dev, dtype=torch.float64)) for ly in keep}
                    idx = [0]; start = 0; CH = 1 << args.chunk
                    if method == 'sobol':
                        Xall = samplers.sobol_points(Nmax, n, seed=10_000 * seed + r, device=dev)
                    if method in ('mub', 'mubanti'):
                        Z1all, _ = samplers.MUBDesign(Bk, net.W[0], n).z1_points(Nmax, gen, rng, dev, radius='strat')
                    while start < Nmax:
                        m = min(CH, Nmax - start)
                        if method == 'mc':
                            z1 = torch.randn(m, n, generator=gen, device=dev) @ WT[0]
                        elif method == 'sobol':
                            z1 = Xall[start:start + m] @ WT[0]
                        elif method in ('mub', 'mubanti'):
                            z1 = Z1all[start:start + m]
                        run_points(z1, WT, s, L, gen, keep, sums, idx, cps, start, anti=(method == 'mubanti'))
                        start += m
                    for ly in keep:
                        out[ly][r] = sums[ly]['out'].cpu().numpy()
                for ly in keep:
                    res[f'sum_{method}_{ly}'] = out[ly].astype(np.float64)
            np.savez_compressed(path, **res)
            print(f"seed {seed} sigma {sigma:g} L {L}: done [{time.time() - t0:.0f}s]", flush=True)
