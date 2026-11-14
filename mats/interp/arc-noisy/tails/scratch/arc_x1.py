"""ARC K3f heaviside (and K2) on the X1 neurons/thresholds of dev nets; scored against the X1 references."""
import sys, json, math, glob, os
import numpy as np, torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy/tails')
from tl import TNet
from kprop_tail import kprop_tail
torch.set_num_threads(4)
RD = sys.argv[1]
out = {}
for f in sorted(glob.glob(os.path.join(RD, 'x1', 'x1_*.json'))):
    r = json.load(open(f)); NN = len(r['neurons'])
    net = TNet(r['seed'], r['L'], r['sigma'])
    ths = []
    for k in range(5):
        t = torch.full((256,), 1e3, dtype=torch.float64)
        for b in range(NN): t[b] = r['thr'][b][k]
        ths.append(t)
    ref = np.zeros((NN, 5))
    for b, pn in enumerate(r['per_neuron']):
        for k in range(5):
            v = np.array(pn['logp'][k]); m = v.max(); ref[b, k] = math.log(np.exp(v - m).mean()) + m
    for (km, fac) in ((2, False), (3, True)):
        o, fl, secs, sh = kprop_tail(net, ths, k_max=km, factor=fac)
        est = np.array([[float(o[k][b]) for k in range(5)] for b in range(NN)])
        rel = est / np.exp(ref) - 1
        lg = np.where(est > 0, np.log10(np.clip(est, 1e-300, None)) - ref / math.log(10), np.nan)
        print(f"{os.path.basename(f)} K{km}{'f' if fac else ''} flops {fl:.2e}: within50 " + ' '.join('%.2f' % v for v in np.mean(np.abs(rel) <= 0.5, 0))
              + ' | med|relerr| ' + ' '.join('%.2f' % v for v in np.median(np.abs(rel), 0)) + ' | med log10 ratio ' + ' '.join('%+.2f' % v for v in np.nanmedian(lg, 0))
              + ' | frac<=0 ' + ' '.join('%.2f' % v for v in np.mean(est <= 0, 0)), flush=True)
        out[f"{os.path.basename(f)}_K{km}"] = dict(est=est.tolist(), flops=fl)
json.dump(out, open(os.path.join(RD, 'arc_x1.json'), 'w'))
