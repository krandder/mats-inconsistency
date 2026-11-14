import json, os, sys, torch
sys.path.insert(0, 'lib')
torch.set_num_threads(2)
import bf, gmprop, lrg
seed, L, tfile = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
lsw = int(sys.argv[4]) if len(sys.argv) > 4 else 2
net = bf.get_net(seed, 64, 32, 0.001)
tr = torch.load(tfile); NT = tr['N']
def err(e, l):
    tm = tr['mean'][l - 1]; tv = tr['sq'][l - 1] - tm ** 2
    d = e - tm; u = tm / tm.norm(); par = float(d @ u); tot = float((d ** 2).sum())
    return bf.nmse(e, tm, tv, NT), par ** 2 / max(tot, 1e-300), par / float(tm.norm())
_, flg, G = gmprop.run(net, L, mode='none', keep=True)
_, fll, R = lrg.run(net, L, l_switch=lsw)
print('flops GC %.2e LRG %.2e' % (flg, fll))
for l in range(2, L + 1, 2):
    eg = err(G[l - 1], l); er = err(R[l - 1], l)
    print(f'l={l:2d} GC {eg[0]:.2e} (scale {eg[2]:+.4f}, frac {eg[1]:.2f})   LRG {er[0]:.2e} (scale {er[2]:+.4f}, frac {er[1]:.2f})')
