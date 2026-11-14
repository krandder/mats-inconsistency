import json, os, sys, torch
sys.path.insert(0, 'lib')
torch.set_num_threads(2)
import bf, gmprop
seed, L, tfile = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
net = bf.get_net(seed, 64, 32, 0.001)
tr = torch.load(tfile)
kp = json.load(open(f'runs/kprop64/kprop64_{seed}.json'))
est, fl, per = gmprop.run(net, L, mode='none', keep=True)
for l in range(2, L + 1, 2):
    tm = tr['mean'][l - 1]; tv = tr['sq'][l - 1] - tm ** 2; v = float(tv.mean())
    for name, e in (('GC', per[l - 1]), ('K2', torch.tensor(kp['K2']['means'][l - 1], dtype=torch.float64)),
                    ('K3f', torch.tensor(kp['K3f']['means'][l - 1], dtype=torch.float64))):
        d = e - tm
        u = tm / tm.norm()
        par = float((d @ u)); tot = float((d ** 2).sum())
        print(f'l={l:2d} {name:4s} nMSE {tot / len(d) / v:.2e}  frac along mean {par ** 2 / tot:.2f}  rel scale err {par / float(tm.norm()):+.4f}', flush=True)
