"""Exploration X3 / oracle: raw moments E[z^k], k = 1..4, of the last pre-activation at depths 4, 8 and 32 (one depth-32
pass gives all three truncations), all 256 neurons, by plain Monte Carlo in float64 accumulation.
python tails/x3.py SEEDS(comma) SIGMA LOG2N OUTDIR"""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tl import TNet

seeds = [int(v) for v in sys.argv[1].split(',')]; sigma = float(sys.argv[2]); lg = int(sys.argv[3]); OUT = sys.argv[4]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = False
DEPTHS = (4, 8, 32)
os.makedirs(OUT, exist_ok=True)
for seed in seeds:
    net = TNet(seed, 32, sigma, device=dev)
    gen = torch.Generator(device=dev).manual_seed(40_000 + seed)
    S = {L: torch.zeros(4, 256, dtype=torch.float64, device=dev) for L in DEPTHS}
    B = 1 << 16; t0 = time.time()
    with torch.no_grad():
        for _ in range((1 << lg) // B):
            h = torch.randn(B, 256, generator=gen, device=dev)
            for l in range(32):
                z = h @ net.W[l].T
                if l + 1 in DEPTHS:
                    zd = z.double()
                    p = zd.clone()
                    for k in range(4):
                        S[l + 1][k] += p.sum(0)
                        if k < 3:
                            p *= zd
                a = torch.relu(z)
                h = a + net.s[l] * torch.randn(a.shape, generator=gen, device=dev) if sigma > 0 else a
    out = dict(seed=seed, sigma=sigma, N=1 << lg, moments={str(L): (S[L] / (1 << lg)).cpu().tolist() for L in DEPTHS})
    json.dump(out, open(os.path.join(OUT, f'x3_{seed}_s{sigma:g}.json'), 'w'))
    print(seed, f'{time.time() - t0:.0f}s', flush=True)
