"""U1 (PREREG-TAILS section 3): splitting on the top mode of the last hidden layer, one run for all outputs, with a
Feynman-Kac readout of every output's events. Up run (S = v . h) and down run (-S); output i uses the up run if
v . W_L[i] > 0. Levels: adaptive pilot on S (N = 2000, rho = 0.1, 24 levels; the last, infinite one dropped), scales
frozen from the pilot; R frozen runs of N = 2000 with the readout.
python tails/mode.py SEED SIGMA L THRDIR OUTDIR [R]"""
import json, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tl import TNet
import bgs

seed, sigma, L, THR, OUT = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5]
R = int(sys.argv[6]) if len(sys.argv) > 6 else 8
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = False
thr = json.load(open(os.path.join(THR, f'thr_{seed}.json')))['thr'][str(L)]
NN = len(thr)
net = TNet(seed, L, sigma, device=dev)
gen = torch.Generator(device=dev).manual_seed(80_000 + 97 * seed + 7 * L + int(round(1e4 * sigma)))
t0 = time.time()
M0 = 1 << 14
with torch.no_grad():
    H = net.hidden(bgs.draw(net, 1, M0, gen)[0]).double()
    ev, V = torch.linalg.eigh(H.T @ H / M0)
    v = V[:, -1]; v = v * torch.sign(v.sum())
c = (net.W[-1].double() @ v).cpu().numpy()
vf = v.to(net.dtype)
T = torch.tensor(thr, device=dev, dtype=net.dtype)                 # (NN, 5)
W8 = net.W[-1][:NN]


def score_up(Z):
    B, N = Z.shape[:2]
    return net.hidden(Z.reshape(B * N, *Z.shape[2:])).reshape(B, N, -1) @ vf


def readout(Z, Y, g):
    B, N = Z.shape[:2]
    h = net.hidden(Z.reshape(B * N, *Z.shape[2:])).reshape(B, N, -1)
    hit = ((h @ W8.T)[:, :, :, None] > T[None, None]).float()       # (B, N, NN, 5)
    band = (Y.double() <= g[:, None]).float()
    return (hit * band[:, :, None, None]).mean(1).reshape(B, -1)


res = dict(seed=seed, sigma=sigma, L=L, c=c[:NN].tolist(), mode_share=float(ev[-1] / ev.sum()), M0=M0, R=R, runs={})
for name, sgn in (('up', 1.0), ('down', -1.0)):
    sf = (lambda Z: score_up(Z)) if sgn > 0 else (lambda Z: -score_up(Z))
    pm = bgs.pilot(net, sf, 1, 2000, gen, [[float('inf')]], max_levels=24)
    Lm, Bm = bgs.pad(pm['levels'], pm['bs'], 5)
    Lm = Lm[:, :-1]; Bm = Bm[:, :Lm.shape[1] - 1]
    om = bgs.frozen(net, sf, 2000, gen, Lm.repeat(R, 1), Bm.repeat(R, 1, 1), n_moves=5, readout=readout)
    res['runs'][name] = dict(levels=Lm[0].tolist(), est=om['est'].numpy().reshape(R, NN, 5).tolist(),
                             n_eval_per_run=om['n_eval'] / R, pilot_n_eval=pm['n_eval'])
    print(name, f'{time.time() - t0:.0f}s', flush=True)
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(os.path.join(OUT, f'mode_{seed}_L{L}_s{sigma:g}.json'), 'w'))
print('done', f'{time.time() - t0:.0f}s')
