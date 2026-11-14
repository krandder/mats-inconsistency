"""Run ARC kprop variants on one net, all layers up to L, save results/kprop_<seed>_s<sigma>_L<L>_<variant>.json.
usage: run_kprop.py SEED SIGMA L VARIANT[,VARIANT..]   VARIANT in k1, k2, k2a, k3, k3f, k3fa, k4f, k4fa, k2m (avg_metric off)"""
import json, os, sys, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anet, kprop_noisy as KN
torch.set_num_threads(1)
seed, sigma, L = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
SPEC = dict(k1=(1, False, 'simple', True), k2=(2, False, 'simple', True), k2a=(2, False, 'augment', True),
            k3=(3, False, 'simple', True), k3f=(3, True, 'simple', True), k3fa=(3, True, 'augment', True),
            k4f=(4, True, 'simple', True), k4fa=(4, True, 'augment', True), k2m=(2, False, 'simple', False),
            k3fm=(3, True, 'simple', False))
net = anet.Net(seed, sigma=sigma)
os.makedirs(os.path.join(anet.HERE, 'results'), exist_ok=True)
for v in sys.argv[4].split(','):
    out = os.path.join(anet.HERE, 'results', f'kprop_{seed}_s{sigma:g}_L{L}_{v}.json')
    if os.path.exists(out):
        continue
    k, fac, kind, avg = SPEC[v]
    t0 = time.time()
    means, fl, sec = KN.kprop_noisy(net, L, k, factor=fac, kind=kind, keep=True, avg_metric=avg)
    # FLOPs per layer prefix: rerun is too costly; record total and per-layer means
    json.dump(dict(seed=seed, sigma=sigma, L=L, variant=v, k=k, factor=fac, kind=kind, avg_metric=avg, flops=fl, secs=sec,
                   means=[m.tolist() for m in means]), open(out, 'w'))
    print(v, f"flops {fl:.3e} secs {sec:.0f}", flush=True)
