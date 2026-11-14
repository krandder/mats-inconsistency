"""ARC kprop (K=2, K=3 factorized) on this lane's ARC nets (all layers kept). Uses an unedited copy of
work/arc-noisy/kprop_noisy.py and ARC's mlp_kprop source in work/arc-noisy/ext (read only).
usage: run_kprop.py N SEED SIGMA VARIANTS(k2,k3f) OUTDIR"""
import json, os, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy/ext/mlp_cumulant_propagation/src')
sys.path.insert(0, os.path.join(HERE, '..', 'ext'))
import anet_copy as A
import kprop_noisy_copy as KN
torch.set_num_threads(int(os.environ.get('NT', '2')))
n, seed, sigma = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
outdir = sys.argv[5]; os.makedirs(outdir, exist_ok=True)
SPEC = dict(k2=(2, False, 'simple'), k3f=(3, True, 'simple'), k2a=(2, False, 'augment'))
tab = json.load(open(os.path.join(HERE, '..', 'ext', 'rtable_sc.json')))
net = A.Net(seed, n=n, L=32, sigma=sigma, r=tab[f"{n}_32_{seed}_{sigma:g}"])
for v in sys.argv[4].split(','):
    out = os.path.join(outdir, f'kprop_n{n}_seed{seed}_s{sigma:g}_{v}.json')
    if os.path.exists(out):
        continue
    k, fac, kind = SPEC[v]
    t0 = time.time()
    means, fl, sec = KN.kprop_noisy(net, 32, k, factor=fac, kind=kind, keep=True)
    json.dump(dict(n=n, seed=seed, sigma=sigma, variant=v, flops=fl, secs=sec, r=net.r.tolist(),
                   means=[m.tolist() for m in means]), open(out, 'w'))
    print(n, seed, sigma, v, f"flops {fl:.3e} secs {sec:.0f}", flush=True)
