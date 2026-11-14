import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
import rnet as R, rare_gs as GS

gen = torch.Generator().manual_seed(5)
net = R.RNet(16, 4, 'tanh', 0.75, 0.3, 0)
t0 = time.time()
lev, fl = GS.level_curve(net, 4096, gen, rho=0.1, n_levels=20)
print('adaptive levels (prob ~10^-k):', [round(x, 3) for x in lev], f'{time.time() - t0:.0f}s', flush=True)
for t, ref in ((2.882, -9.98), (4.143, -30.06), (5.115, None)):
    lv = [x for x in lev if x < t]
    ests = []
    t0 = time.time()
    for r in range(6):
        lpr, _, fl = GS.splitting(net, t, 4096, gen, levels=lv)
        ests.append(lpr)
    e = np.array(ests); p = np.exp(e - e.max())
    print(f't={t}: GS log2 p = {(np.log(p.mean()) + e.max()) / math.log(2):.2f} (ref {ref}), rel sd {p.std(ddof=1) / p.mean():.2f}, '
          f'per-run FLOPs {fl:.2e}, {time.time() - t0:.0f}s', flush=True)
