"""LCG (k, B) at width 256 depth 32 on an ARC dev net vs the ARC truth, GC, K2, K3f (ARC lane results)."""
import json, os, sys, time, numpy as np, torch
sys.path.insert(0, 'lib')
torch.set_num_threads(int(os.environ.get('TH', '3')))
import bf, lcg
ARC = '/home/agent/workspaces/mats/work/arc-noisy'
seed, k, B = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
net = bf.get_net(seed, 256, 32, 0.001)
d = np.load(f'{ARC}/truth/truth_256_32_s0.001_seed{seed}.npz'); N = float(d['N'])
TM = d['S1'] / N; TV = d['S2'] / N - TM ** 2
err = lambda e, l: float(((np.asarray(e) - TM[l - 1]) ** 2).mean() / TV[l - 1].mean() - 1 / N)
t0 = time.time()
_, _, per = lcg.run(net, 32, k=k, B=B)
print(f'LCG k={k} B={B} seed {seed}: {time.time() - t0:.0f}s', flush=True)
kp = {}
for kf in ('k2', 'k3f'):
    p = f'{ARC}/results/kprop_{seed}_s0.001_L32_{kf}.json'
    if os.path.exists(p):
        kp[kf] = json.load(open(p))
res = {}
for l in (8, 16, 24, 32):
    res[l] = dict(lcg=err(per[l - 1].numpy(), l), **{kf: err(np.array(v['means'][l - 1]), l) for kf, v in kp.items()})
    print(l, json.dumps({a: float(f'{b:.3g}') for a, b in res[l].items()}), flush=True)
print('kprop flops', {kf: v.get('flops') for kf, v in kp.items()})
os.makedirs('runs/lcg256', exist_ok=True)
json.dump(res, open(f'runs/lcg256/lcg_{seed}_k{k}_B{B}.json', 'w'), indent=1)
