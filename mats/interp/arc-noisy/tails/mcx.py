"""Plain Monte Carlo pass (one depth-32 pass gives L = 4, 8, 32): exceedance counts of every event (truth cross-check
(b), PREREG-TAILS section 2) and raw moments E[z^p], p = 1..4, of all 256 outputs (C-MCx and U3).
python tails/mcx.py SEEDS(comma) SIGMA THRDIR OUTDIR LOG2N TAG [REPLICATES]"""
import json, os, sys, time, zlib
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tl import TNet

seeds = [int(v) for v in sys.argv[1].split(',')]; sigma = float(sys.argv[2]); THR, OUT = sys.argv[3], sys.argv[4]
lg, TAG = int(sys.argv[5]), sys.argv[6]
REP = int(sys.argv[7]) if len(sys.argv) > 7 else 1
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = False
DEPTHS = (4, 8, 32)
os.makedirs(OUT, exist_ok=True)
for seed in seeds:
    thr = json.load(open(os.path.join(THR, f'thr_{seed}.json')))['thr']
    T = {L: torch.tensor(thr[str(L)], device=dev) for L in DEPTHS}
    net = TNet(seed, 32, sigma, device=dev)
    reps = []
    t0 = time.time()
    for rep in range(REP):
        gen = torch.Generator(device=dev).manual_seed(1_000_000 * (zlib.crc32(TAG.encode()) % 100_000) + 97 * seed + 1009 * rep + int(round(1e4 * sigma)))
        S = {L: torch.zeros(4, 256, dtype=torch.float64, device=dev) for L in DEPTHS}
        C = {L: torch.zeros(8, 5, dtype=torch.float64, device=dev) for L in DEPTHS}
        B = min(1 << 16, 1 << lg)
        with torch.no_grad():
            for _ in range((1 << lg) // B):
                h = torch.randn(B, 256, generator=gen, device=dev)
                for l in range(32):
                    z = h @ net.W[l].T
                    if l + 1 in DEPTHS:
                        zd = z.double(); p = zd.clone()
                        for k in range(4):
                            S[l + 1][k] += p.sum(0)
                            if k < 3:
                                p *= zd
                        C[l + 1] += (z[:, :8, None] > T[l + 1][None]).double().sum(0)
                    if l == 31:
                        break
                    a = torch.relu(z)
                    h = a + net.s[l] * torch.randn(a.shape, generator=gen, device=dev) if sigma > 0 else a
        reps.append(dict(moments={str(L): (S[L] / (1 << lg)).cpu().tolist() for L in DEPTHS},
                         counts={str(L): C[L].cpu().tolist() for L in DEPTHS}))
    json.dump(dict(seed=seed, sigma=sigma, N=1 << lg, tag=TAG, reps=reps),
              open(os.path.join(OUT, f'mc_{TAG}_{seed}_s{sigma:g}.json'), 'w'))
    print(seed, f'{time.time() - t0:.0f}s', flush=True)
