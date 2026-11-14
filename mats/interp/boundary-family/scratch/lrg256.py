"""LRG vs exact Gaussian closure at width 256 (ARC dev nets, ARC truths), sigma = 0.001."""
import json, os, sys, time, numpy as np, torch
sys.path.insert(0, 'lib')
torch.set_num_threads(int(os.environ.get('TH', '3')))
import bf, gmprop, lrg
ARC = '/home/agent/workspaces/mats/work/arc-noisy'
seed = int(sys.argv[1]); L = int(sys.argv[2]) if len(sys.argv) > 2 else 32
net = bf.get_net(seed, 256, 32, 0.001)
d = np.load(f'{ARC}/truth/truth_256_32_s0.001_seed{seed}.npz'); N = float(d['N'])
TM = d['S1'] / N; TV = d['S2'] / N - TM ** 2
def err(e, l):
    e = e.numpy() if torch.is_tensor(e) else np.asarray(e)
    tm = TM[l - 1]; dd = e - tm; u = tm / np.linalg.norm(tm)
    par = float(dd @ u); tot = float((dd ** 2).sum())
    return ((dd ** 2).mean() / TV[l - 1].mean() - 1 / N), par ** 2 / tot, par / np.linalg.norm(tm)
t0 = time.time()
_, fll, R = lrg.run(net, L)
t1 = time.time()
_, flg, G = gmprop.run(net, L, mode='none', keep=True)
t2 = time.time()
print(f'seed {seed}: flops LRG {fll:.2e} ({t1-t0:.0f}s) GC {flg:.2e} ({t2-t1:.0f}s)')
kp = {}
for kf in ('k2', 'k2a', 'k3f'):
    p = f'{ARC}/results/kprop_{seed}_s0.001_L32_{kf}.json'
    if os.path.exists(p):
        kp[kf] = json.load(open(p))['means']
for l in (2, 4, 8, 12, 16, 20, 24, 28, 32):
    if l > L: break
    eg = err(G[l - 1], l); er = err(R[l - 1], l)
    ks = ' '.join(f'{k} {err(np.array(v[l - 1]), l)[0]:.2e}' for k, v in kp.items())
    print(f'l={l:2d} GC {eg[0]:.2e} (scale {eg[2]:+.4f} frac {eg[1]:.2f})  LRG {er[0]:.2e} (scale {er[2]:+.4f} frac {er[1]:.2f})  {ks}', flush=True)
