"""C-Kd (PREREG-TAILS section 3): part 1's Kerdock-MUB design (antipodal pairs over the 129 MUBs of R^256, Haar rotation
per block folded into W_1, stratified chi radius per pair) as the input design for the tail events, and plain Monte
Carlo at the same N for reference. 16 replicates of 2^16 points per (net, sigma); exceedance counts of every event at
L = 4, 8 and 32 from one depth-32 pass.
python tails/kerdock_tail.py SEEDS(comma) SIGMA THRDIR OUTDIR [R] [LOG2N]"""
import json, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from tl import TNet
import samplers as SP

seeds = [int(v) for v in sys.argv[1].split(',')]; sigma = float(sys.argv[2]); THR, OUT = sys.argv[3], sys.argv[4]
R = int(sys.argv[5]) if len(sys.argv) > 5 else 16
lg = int(sys.argv[6]) if len(sys.argv) > 6 else 16
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = False
B_np = np.load(os.path.join(os.path.dirname(HERE), 'kerdock129.npy'))
os.makedirs(OUT, exist_ok=True)
for seed in seeds:
    thr = json.load(open(os.path.join(THR, f'thr_{seed}.json')))['thr']
    net = TNet(seed, 32, sigma, device=dev)
    T = {L: torch.tensor(thr[str(L)], device=dev) for L in (4, 8, 32)}       # (8, 5)
    gen = torch.Generator(device=dev).manual_seed(95_000 + 97 * seed + int(round(1e4 * sigma)))
    rng = np.random.default_rng(95_000 + 97 * seed + int(round(1e4 * sigma)))
    design = SP.MUBDesign(B_np, net.W[0].double().cpu())
    res = dict(seed=seed, sigma=sigma, N=1 << lg, R=R, kerdock={}, mc={})
    t0 = time.time()
    for name in ('kerdock', 'mc'):
        counts = {L: [] for L in (4, 8, 32)}; setup = 0
        for r in range(R):
            if name == 'kerdock':
                z1, fl = design.z1_points(1 << lg, gen, rng, dev); setup += fl
            else:
                z1 = torch.randn(1 << lg, 256, generator=gen, device=dev) @ net.W[0].T
            c = {L: torch.zeros(8, 5, device=dev) for L in (4, 8, 32)}
            for a in range(0, 1 << lg, 1 << 15):
                z = z1[a:a + (1 << 15)].to(torch.float32)
                for l in range(1, 32):
                    h = torch.relu(z)
                    h = h + net.s[l - 1] * torch.randn(h.shape, generator=gen, device=dev)
                    z = h @ net.W[l].T
                    if l + 1 in (4, 8, 32):
                        c[l + 1] += (z[:, :8, None] > T[l + 1][None]).float().sum(0)
            for L in (4, 8, 32):
                counts[L].append(c[L].cpu().numpy().tolist())
        res[name] = dict(counts={str(L): v for L, v in counts.items()}, setup_flops=setup)
    json.dump(res, open(os.path.join(OUT, f'kd_{seed}_s{sigma:g}.json'), 'w'))
    print(seed, f'{time.time() - t0:.0f}s', flush=True)
