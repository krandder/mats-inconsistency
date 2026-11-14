import json, os, sys, torch, time
sys.path.insert(0, 'lib')
torch.set_num_threads(int(os.environ.get('TH', '2')))
import bf, gmprop
seed, L, tfile = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
net = bf.get_net(seed, 64, 32, 0.001)
tr = torch.load(tfile); NT = tr['N']
def err(e, l):
    tm = tr['mean'][l - 1]; tv = tr['sq'][l - 1] - tm ** 2
    return bf.nmse(e, tm, tv, NT)
for cfg in sys.argv[4:]:
    k, ls = [int(v) for v in cfg.split(':')[:2]]
    t0 = time.time()
    _, fl, per = gmprop.run_grid(net, L, k, ls)
    print(f'k={k} l_s={ls}: nMSE(L) {err(per[-1], L):.2e} flops {fl:.2e} {time.time()-t0:.0f}s  per-layer ' + ' '.join(f'{err(per[l], l + 1):.1e}' for l in range(0, L, 2)), flush=True)
