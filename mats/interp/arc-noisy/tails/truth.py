"""Truth (PREREG-TAILS section 2) for one (net, sigma): for L = 4, 8, 32 and outputs 0..7,
tuning pilot (N = 1000, adaptive levels with the five thresholds as stops, rho = 0.1, scales adapted; own seed), then R
frozen runs of N = 2000 (R = 32 at L = 32, 16 at L = 4, 8), 5 pCN moves per level; plus cross-check (c): an independent
splitting estimator with rho = 0.3 (pilot N = 1000, R = 8 frozen runs of N = 1000) for outputs 0 and 1.
python tails/truth.py SEED SIGMA THRDIR OUTDIR [DEPTHS]"""
import json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tl import TNet
import bgs

seed, sigma, THR, OUT = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
DEPTHS = [int(v) for v in sys.argv[5].split(',')] if len(sys.argv) > 5 else [4, 8, 32]
# optional sizes (part 3, CPU): R_FROZEN N_FROZEN N_PILOT R_XC N_XC
SZ = [int(v) for v in sys.argv[6].split(',')] if len(sys.argv) > 6 else None
if not torch.cuda.is_available():
    torch.set_num_threads(int(os.environ.get('TH_THREADS', '2')))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = False
thr_all = json.load(open(os.path.join(THR, f'thr_{seed}.json')))
os.makedirs(OUT, exist_ok=True)
MAXP = 131072                                                   # particles per chunk at L = 32 (memory)


def frozen_chunked(net, idx, Lv, Bs, N, R, gen):
    """R frozen runs for each system in idx; returns (len(idx), R, K) log estimates and evaluations."""
    per_chunk = max(1, int(MAXP * 32 / net.L) // (N * R))       # systems (outputs) per chunk
    logs = []; nev = 0
    for a in range(0, len(idx), per_chunk):
        sl = slice(a, min(len(idx), a + per_chunk))
        n_sys = sl.stop - sl.start
        o = bgs.frozen(net, bgs.neuron_scores(net, idx[sl].repeat_interleave(R)), N, gen,
                       Lv[sl].repeat_interleave(R, 0), Bs[sl].repeat_interleave(R, 0), n_moves=5)
        logs.append(o['logp'].reshape(n_sys, R, -1)); nev += o['n_eval']
    return torch.cat(logs, 0).numpy(), nev


res = dict(seed=seed, sigma=sigma, by_L={})
t0 = time.time()
for L in DEPTHS:
    thr = thr_all['thr'][str(L)]; NN = len(thr)
    net = TNet(seed, L, sigma, device=dev)
    gen = torch.Generator(device=dev).manual_seed(60_000 + 97 * seed + 7 * L + int(round(1e4 * sigma)))
    idx = torch.arange(NN, device=dev)
    # main estimator
    p = bgs.pilot(net, bgs.neuron_scores(net, idx), NN, SZ[2] if SZ else 1000, gen, thr, rho=0.1)
    Lv, Bs = bgs.pad(p['levels'], p['bs'], 5)
    R = SZ[0] if SZ else (32 if L == 32 else 16)
    NF = SZ[1] if SZ else 2000
    logs, nev = frozen_chunked(net, idx, Lv, Bs, NF, R, gen)
    cols = [[int(np.where(np.abs(Lv[b].numpy() - t) < 1e-12)[0][0]) for t in thr[b]] for b in range(NN)]
    est = [[logs[b, :, c].tolist() for c in cols[b]] for b in range(NN)]
    out = dict(thr=thr, levels=[l for l in p['levels']], n_levels_to=[[c + 1 for c in cols[b]] for b in range(NN)],
               pilot_n_eval=p['n_eval'], frozen_n_eval=nev, N=NF, R=R, logp=est)
    # cross-check (c): rho = 0.3, outputs 0 and 1
    idx2 = idx[:2]
    gen2 = torch.Generator(device=dev).manual_seed(70_000 + 97 * seed + 7 * L + int(round(1e4 * sigma)))
    p2 = bgs.pilot(net, bgs.neuron_scores(net, idx2), 2, SZ[4] if SZ else 1000, gen2, thr[:2], rho=0.3)
    Lv2, Bs2 = bgs.pad(p2['levels'], p2['bs'], 5)
    logs2, nev2 = frozen_chunked(net, idx2, Lv2, Bs2, SZ[4] if SZ else 1000, SZ[3] if SZ else 8, gen2)
    cols2 = [[int(np.where(np.abs(Lv2[b].numpy() - t) < 1e-12)[0][0]) for t in thr[b]] for b in range(2)]
    out['xcheck'] = dict(logp=[[logs2[b, :, c].tolist() for c in cols2[b]] for b in range(2)], n_eval=nev2 + p2['n_eval'],
                         n_levels=[len(l) for l in p2['levels']])
    res['by_L'][str(L)] = out
    print(f'L={L} done {time.time() - t0:.0f}s', flush=True)
json.dump(res, open(os.path.join(OUT, f'truth_{seed}_s{sigma:g}.json'), 'w'))
print('done', f'{time.time() - t0:.0f}s', flush=True)
