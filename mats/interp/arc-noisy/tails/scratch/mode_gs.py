"""Exploration (dev net, depth 32): splitting on the top mode of h_{L-1} (one run for all neurons) with a readout of each
neuron's events, against per-neuron splitting. python mode_gs.py SEED SIGMA L N R NLEV"""
import math, sys, time, json
import numpy as np
import torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy/tails')
from tl import TNet
import gs as GS

torch.set_num_threads(4)
seed, sigma, L, N, R, NLEV = [float(v) if '.' in v else int(v) for v in sys.argv[1:7]]
net = TNet(seed, L, sigma)
gen = torch.Generator().manual_seed(9500 + seed)
draw = lambda m: net.draw(m, gen)
H = net.hidden(draw(1 << 14)).double()
M2 = H.T @ H / H.shape[0]
ev, V = torch.linalg.eigh(M2)
v = V[:, -1]; v = v * torch.sign(v.sum())
print('top mode share of E|h|^2: %.3f; next %.3f' % (float(ev[-1] / ev.sum()), float(ev[-2] / ev.sum())))
vf = v.float()
score = lambda Z: net.hidden(Z) @ vf
S = (H @ v)
Zall = H @ net.W[-1].double().T
c = net.W[-1].double() @ v
sc = S - S.mean(); zc = Zall - Zall.mean(0)
r2 = ((sc[:, None] * zc).mean(0) ** 2 / ((sc ** 2).mean() * (zc ** 2).mean(0)))
print('R2 of z_i on S: median %.2f; frac > 0.5: %.2f; frac c>0 %.2f' % (float(r2.median()), float((r2 > 0.5).double().mean()), float((c > 0).double().mean())))
lev = {0: [1.796, 2.235, 2.632, 3.028, 3.432], 1: [0.467, 0.644, 0.804, 0.954, 1.069], 2: [1.235, 1.549, 1.878, 2.188, 2.489], 3: [2.763, 3.337, 3.833, 4.330, 4.778]}
for i in lev:
    print(f'neuron {i}: c={float(c[i]):+.3f} R2={float(r2[i]):.2f}')
neur = list(lev)
T = torch.tensor([lev[i] for i in neur], dtype=torch.float32)   # (4, 5)
W4 = net.W[-1][neur]                                             # (4, n)


def readout(Z, Y, g_next):
    z = net.hidden(Z) @ W4.T                                     # (N, 4)
    hit = (z[:, :, None] > T[None]).float()                      # (N, 4, 5)
    band = (Y <= g_next).float()
    return (hit * band[:, None, None]).mean(0).reshape(-1)


t0 = time.time()
pil = GS.gs_run(score, draw, 2000, gen, float('inf'), rho=0.1, n_moves=5, max_levels=NLEV)
levs = pil['levels'][:-1]; bs = GS.pad_bs(pil['bs'], len(levs) + 1, 5)
print('S levels', np.round(levs, 2).tolist(), f'pilot {time.time() - t0:.0f}s, acc last {np.mean(pil["acc"][-1]):.2f} b {pil["bs"][-1][-1]:.2f}', flush=True)
ests = []; nev = 0
for r in range(R):
    e, cum, ne = GS.gs_readout(score, draw, N, gen, levs, bs, 5, readout)
    ests.append(e.numpy()); nev += ne
    print(f'run {r}: {time.time() - t0:.0f}s', flush=True)
E = np.array(ests).reshape(R, 4, 5)
for a, i in enumerate(neur):
    m = E[:, a].mean(0); s = E[:, a].std(0, ddof=1)
    print(f'neuron {i}: amortized log10 p ' + ' '.join('%.2f' % (math.log10(x) if x > 0 else -99) for x in m)
          + ' | rel var per run ' + ' '.join('%.2f' % (sv ** 2 / max(mv, 1e-300) ** 2) for sv, mv in zip(s, m)) + f' | evals/run {nev / R:.0f}')
# per-neuron frozen GS references for the same 4 neurons
for i in neur:
    sc_i = lambda Z, i=i: net.z(Z, i)
    p2 = GS.gs_run(sc_i, draw, 1000, gen, max(lev[i]), rho=0.1, n_moves=5, stops=lev[i])
    lv = p2['levels']; b2 = GS.pad_bs(p2['bs'], len(lv), 5)
    runs, ne = GS.frozen_runs(sc_i, draw, lv, b2, gen, 1000, 8, 5)
    js = [int(np.argmin(np.abs(np.array(lv) - t))) for t in lev[i]]
    out = [GS.summarize(runs[:, j]) for j in js]
    print(f'neuron {i}: per-neuron GS log10 p ' + ' '.join('%.2f' % (o['logp'] / math.log(10)) for o in out)
          + ' | rel var per run ' + ' '.join('%.2f' % ((o['rel_se'] * math.sqrt(8)) ** 2) for o in out) + f' | evals/run {ne / 8:.0f}', flush=True)
