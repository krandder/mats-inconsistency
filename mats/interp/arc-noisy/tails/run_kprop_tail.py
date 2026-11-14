"""CPU driver: ARC's tail method (heaviside) and U2's moments, one propagation per (net, L, sigma, variant).
python tails/run_kprop_tail.py SEED SIGMA L VARIANT(k2|k3f|k3fa) THRDIR OUTDIR"""
import json, os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tl import TNet
from kprop_tail import kprop_all

torch.set_num_threads(int(os.environ.get('KP_THREADS', '2')))
seed, sigma, L, var, THR, OUT = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5], sys.argv[6]
cfg = dict(k2=(2, False, 'simple'), k3f=(3, True, 'simple'), k3fa=(3, True, 'augment'))[var]
thr = json.load(open(os.path.join(THR, f'thr_{seed}.json')))['thr'][str(L)]
net = TNet(seed, L, sigma)
ths = []
for k in range(5):
    t = torch.full((256,), 3.0 * 2 ** 0.5, dtype=torch.float64)       # outputs without an event get ARC's threshold
    for b in range(len(thr)):
        t[b] = thr[b][k]
    ths.append(t)
t0 = time.time()
o = kprop_all(net, ths, k_max=cfg[0], factor=cfg[1], kind=cfg[2])
res = dict(seed=seed, sigma=sigma, L=L, variant=var, secs=time.time() - t0, flops_shared=o['flops_shared'],
           flops_heavi_each=o['flops_heavi_each'], flops_moments=o['flops_moments'],
           heavi=[[float(o['heavi'][k][b]) for k in range(5)] for b in range(len(thr))],
           moments=o['moments'].numpy().tolist())
os.makedirs(OUT, exist_ok=True)
json.dump(res, open(os.path.join(OUT, f'kp_{seed}_L{L}_s{sigma:g}_{var}.json'), 'w'))
print(seed, sigma, L, var, f'{time.time() - t0:.0f}s shared {o["flops_shared"]:.2e} heavi {o["flops_heavi_each"][0]:.2e} mom {o["flops_moments"]:.2e}', flush=True)
