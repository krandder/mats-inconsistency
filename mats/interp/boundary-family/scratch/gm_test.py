"""Quick dev-net check of the boundary-aware Gaussian-mixture propagation (gmprop) against the MC truth and kprop.
python scratch/gm_test.py SEED L TRUTHFILE [configs...]   config = mode:Cmax[:keep_frac[:l0]]"""
import json, math, os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
torch.set_num_threads(int(os.environ.get('TH', '2')))
import bf, gmprop

seed, L, tfile = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
cfgs = sys.argv[4:] or ['none:1']
net = bf.get_net(seed, 64, 32, 0.001)
tr = torch.load(tfile)
NT = tr['N']
kp = None
kpf = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs', 'kprop64', f'kprop64_{seed}.json')
if os.path.exists(kpf):
    kp = json.load(open(kpf))


def err(est, l):
    tm = tr['mean'][l - 1]; tv = tr['sq'][l - 1] - tm ** 2
    return bf.nmse(est, tm, tv, NT)


if kp:
    for name in ('K2', 'K2a', 'K3f', 'K3fa'):
        e = err(torch.tensor(kp[name]['means'][L - 1], dtype=torch.float64), L)
        print(f'kprop {name}: nMSE {e:.3e}  flops(L) {kp[name].get("flops_by_L", {}).get(str(L))}', flush=True)
for cfg in cfgs:
    p = cfg.split(':')
    mode, Cmax = p[0], int(p[1])
    kf = float(p[2]) if len(p) > 2 else 0.5
    l0 = int(p[3]) if len(p) > 3 else 1
    t0 = time.time()
    est, fl, per = gmprop.run(net, L, Cmax=Cmax, mode=mode, keep=True, keep_frac=kf, l0=l0)
    errs = [err(per[l], l + 1) for l in range(L)]
    print(f'{cfg}: nMSE(L={L}) {errs[-1]:.3e} flops {fl:.3e} secs {time.time() - t0:.1f}  per-layer '
          + ' '.join(f'{e:.1e}' for e in errs), flush=True)
