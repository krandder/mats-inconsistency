"""Exploration X1 (development nets only): per-neuron splitting efficiency and amortised splitting on the top mode.
python tails/x1.py SEED L SIGMA NNEUR OUTDIR
1. threshold pilot per neuron (adaptive, rho = 0.1, 12 levels): thresholds = levels 4, 6, 8, 10, 12 (p ~ 1e-4 .. 1e-12);
2. independent tuning pilot with those thresholds as stops, then R = 16 frozen runs of N = 1000 per neuron;
3. mode score S = h_{L-1} . v (top eigenvector of E[h h^T] from 2^14 paths): pilot (20 levels) + 16 frozen runs of
   N = 2000 with a readout of every neuron's 5 events.
Writes OUTDIR/x1_{seed}_L{L}_s{sigma}.json."""
import json, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tl import TNet
import bgs

seed, L, sigma, NN, OUT = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = False
net = TNet(seed, L, sigma, device=dev)
gen = torch.Generator(device=dev).manual_seed(20_000 + 97 * seed + L)
neur = list(range(NN)); idx = torch.tensor(neur, device=dev)
t0 = time.time(); res = dict(seed=seed, L=L, sigma=sigma, neurons=neur)
# 1. thresholds
p1 = bgs.pilot(net, bgs.neuron_scores(net, idx), NN, 2000, gen, [[float('inf')]] * NN, max_levels=13)
thr = [[p1['levels'][b][k] for k in (3, 5, 7, 9, 11)] for b in range(NN)]
res['thr'] = thr; res['t_thr'] = time.time() - t0
print('thresholds done', f'{time.time() - t0:.0f}s', flush=True)
# 2. per-neuron frozen GS
p2 = bgs.pilot(net, bgs.neuron_scores(net, idx), NN, 1000, gen, thr)
Lv, Bs = bgs.pad(p2['levels'], p2['bs'], 5)
R = 16
CH = max(1, int(65536 // (1000 * R) * (32 // L)))                 # neurons per chunk (memory)
logs = []
for a in range(0, NN, CH):
    sl = slice(a, min(NN, a + CH))
    oo = bgs.frozen(net, bgs.neuron_scores(net, idx[sl].repeat_interleave(R)), 1000, gen, Lv[sl].repeat_interleave(R, 0),
                    Bs[sl].repeat_interleave(R, 0), n_moves=5)
    logs.append(oo['logp'])
o = dict(logp=torch.cat(logs, 0))
per = []
for b in range(NN):
    rows = o['logp'][b * R:(b + 1) * R].numpy()
    lv = Lv[b].numpy(); kk = [int(np.argmin(np.abs(lv - t))) for t in thr[b]]
    nlev = [int(np.sum(lv[:k + 1] < t - 1e-12)) + 1 for k, t in zip(kk, thr[b])]   # levels used up to the target
    per.append(dict(logp=[rows[:, k].tolist() for k in kk], nlev=nlev, nlev_total=len(p2['levels'][b])))
res['per_neuron'] = per; res['pilot2_n_eval'] = p2['n_eval'] / NN; res['frozen_N'] = 1000; res['R'] = R
res['t_per'] = time.time() - t0
print('per-neuron GS done', f'{time.time() - t0:.0f}s', flush=True)
# 3. mode score
with torch.no_grad():
    H = net.hidden(bgs.draw(net, 1, 1 << 14, gen)[0]).double()
    M2 = H.T @ H / H.shape[0]; ev, V = torch.linalg.eigh(M2)
    v = V[:, -1]; v = v * torch.sign(v.sum()); vf = v.to(net.dtype)
    c = (net.W[-1].double() @ v).cpu()
    S = H @ v; Zall = H @ net.W[-1].double().T
    sc = S - S.mean(); zc = Zall - Zall.mean(0)
    r2 = ((sc[:, None] * zc).mean(0) ** 2 / ((sc ** 2).mean() * (zc ** 2).mean(0))).cpu()
res['mode_share'] = float(ev[-1] / ev.sum()); res['c'] = c[:NN].tolist(); res['r2'] = r2[:NN].tolist()


def mode_score(Z):
    B, N = Z.shape[:2]
    return (net.hidden(Z.reshape(B * N, *Z.shape[2:])).reshape(B, N, -1) @ vf)


T = torch.tensor(thr, device=dev, dtype=net.dtype)              # (NN, 5)
W4 = net.W[-1][idx]


def readout(Z, Y, g):
    B, N = Z.shape[:2]
    h = net.hidden(Z.reshape(B * N, *Z.shape[2:])).reshape(B, N, -1)
    z = h @ W4.T                                                # (B, N, NN)
    hit = (z[:, :, :, None] > T[None, None]).float()            # (B, N, NN, 5)
    band = (Y.double() <= g[:, None]).float()
    return (hit * band[:, :, None, None]).mean(1).reshape(B, -1)


for sgn in (1, -1):
    sf = (lambda Z: mode_score(Z)) if sgn == 1 else (lambda Z: -mode_score(Z))
    pm = bgs.pilot(net, sf, 1, 2000, gen, [[float('inf')]], max_levels=24)
    Lm, Bm = bgs.pad(pm['levels'], pm['bs'], 5)
    Lm = Lm[:, :-1]; Bm = Bm[:, :Lm.shape[1] - 1] if Lm.shape[1] > 1 else Bm
    Rm = 16
    om = bgs.frozen(net, sf, 2000, gen, Lm.repeat(Rm, 1), Bm.repeat(Rm, 1, 1), n_moves=5, readout=readout)
    E = om['est'].numpy().reshape(Rm, NN, 5)
    res[f'mode{sgn:+d}'] = dict(levels=Lm[0].tolist(), est=E.tolist(), n_eval_per_run=om['n_eval'] / Rm,
                                pilot_n_eval=pm['n_eval'])
    print(f'mode {sgn:+d} done', f'{time.time() - t0:.0f}s', flush=True)
res['secs'] = time.time() - t0
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(os.path.join(OUT, f'x1_{seed}_L{L}_s{sigma:g}.json'), 'w'))
print('done', f'{time.time() - t0:.0f}s')
