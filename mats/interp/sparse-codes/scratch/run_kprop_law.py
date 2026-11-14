"""ARC kprop (K=2, K=3f) on this lane's ARC nets under the E4 input laws (lib/nets.input_law).
usage: run_kprop_law.py N SEED SIGMA LAW VARIANTS OUTDIR"""
import json, os, sys, time, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy/ext/mlp_cumulant_propagation/src')
sys.path.insert(0, os.path.join(HERE, '..', 'ext')); sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import anet_copy as A
import kprop_noisy_law as KN
import nets
torch.set_num_threads(int(os.environ.get('NT', '2')))
n, seed, sigma, law = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
outdir = sys.argv[6]; os.makedirs(outdir, exist_ok=True)
SPEC = dict(k2=(2, False, 'simple'), k3f=(3, True, 'simple'))
tab = json.load(open(os.path.join(HERE, '..', 'ext', 'rtable_sc.json')))
net = A.Net(seed, n=n, L=32, sigma=sigma, r=tab[f"{n}_32_{seed}_{sigma:g}"])
m0, L0 = nets.input_law(law, n, seed)
mu0 = torch.tensor(m0, dtype=torch.float64); C0 = torch.tensor(L0 @ L0.T, dtype=torch.float64)
for v in sys.argv[5].split(','):
    out = os.path.join(outdir, f'kprop_n{n}_seed{seed}_s{sigma:g}_{law}_{v}.json')
    if os.path.exists(out):
        continue
    k, fac, kind = SPEC[v]
    means, fl, sec = KN.kprop_noisy(net, 32, k, factor=fac, kind=kind, keep=True, mu0=mu0, C0=C0)
    json.dump(dict(n=n, seed=seed, sigma=sigma, law=law, variant=v, flops=fl, secs=sec,
                   means=[m.tolist() for m in means]), open(out, 'w'))
    print(n, seed, sigma, law, v, f"flops {fl:.3e} secs {sec:.0f}", flush=True)
