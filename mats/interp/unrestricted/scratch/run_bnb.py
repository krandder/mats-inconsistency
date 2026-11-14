"""Run the certified B&B on one noiseless net and record its estimates at budget checkpoints (2^k forward-pass
equivalents). Writes runs/bnb_n{n}_L{L}_seed{seed}.json.
Usage: run_bnb.py n L seed [--kmax 22] [--max_leaves 400000] [--batch 128]
"""
import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))
from nets import Net  # noqa: E402
from bnb import CertifiedBnB  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('n', type=int); ap.add_argument('L', type=int); ap.add_argument('seed', type=int)
ap.add_argument('--kmax', type=int, default=22)
ap.add_argument('--max_leaves', type=int, default=400000)
ap.add_argument('--batch', type=int, default=128)
ap.add_argument('--max_seconds', type=float, default=3000)
ap.add_argument('--priority', default='gap')
ap.add_argument('--split', default='mid')
ap.add_argument('--out', default=None)
a = ap.parse_args()
t0 = time.time()
net = Net(a.seed, a.n, a.L)
bb = CertifiedBnB(net.W, batch=a.batch, priority=a.priority, split=a.split)
cps = [2 ** k for k in range(4, a.kmax + 1)]
recs = []
ci = 0
stop = 'budget'
while True:
    while ci < len(cps) and bb.flops / bb.pf >= cps[ci]:
        st = bb.state(); st['checkpoint'] = cps[ci]; st['seconds'] = time.time() - t0
        recs.append(st); ci += 1
    if bb.flops / bb.pf >= cps[-1]:
        break
    if bb.n_leaves >= a.max_leaves:
        stop = 'max_leaves'; break
    if time.time() - t0 > a.max_seconds:
        stop = 'max_seconds'; break
    if not bb.step():
        stop = 'all_certified'; break
st = bb.state(); st['checkpoint'] = 'final'; st['seconds'] = time.time() - t0; recs.append(st)


def js(r):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}


res = dict(n=a.n, L=a.L, seed=a.seed, stop=stop, priority=a.priority, split=a.split, q=bb.q, batch=a.batch, pass_flops=bb.pf, records=[js(r) for r in recs])
out = a.out or os.path.join(os.path.dirname(__file__), '..', 'runs', f'bnb_{a.priority}-{a.split}_n{a.n}_L{a.L}_seed{a.seed}.json')
with open(out, 'w') as f:
    json.dump(res, f)
f = recs[-1]
print(f'n={a.n} L={a.L} seed={a.seed} stop={stop} passes={f["passes"]:.3g} leaves={f["leaves"]} '
      f'exact_mass={f["mass_exact"]:.4f} halfwidth_max={np.max(np.asarray(f["ub"]) - np.asarray(f["lb"])) / 2:.3e} '
      f'{time.time() - t0:.0f}s', flush=True)
