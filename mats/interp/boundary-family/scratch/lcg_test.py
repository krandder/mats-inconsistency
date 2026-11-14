import json, os, sys, time, torch
sys.path.insert(0, 'lib')
torch.set_num_threads(int(os.environ.get('TH', '2')))
import bf, lcg
seed, L, tfile = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
net = bf.get_net(seed, 64, 32, 0.001)
tr = torch.load(tfile); NT = tr['N']
def err(e, l):
    tm = tr['mean'][l - 1]; tv = tr['sq'][l - 1] - tm ** 2
    return bf.nmse(e, tm, tv, NT)
for cfg in sys.argv[4:]:
    k, B, l0 = [int(v) for v in cfg.split(':')]
    t0 = time.time()
    _, _, per = lcg.run(net, L, k=k, B=B, l0=l0)
    print(f'LCG k={k} B={B} l0={l0}: nMSE(L) {err(per[-1], L):.2e} {time.time()-t0:.0f}s per-layer ' + ' '.join(f'{err(per[l], l + 1):.1e}' for l in range(0, L, 2)), flush=True)
