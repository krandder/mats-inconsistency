"""P8's registered quantity: entropy of the query token's 64 layer-2 GELU gates in the PFN (all 64, not only the 52
relevant ones), same estimator panel as run.py. usage: pfn_query_entropy.py OUT.json"""
import sys, os, json
import numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as RUN
net = RUN.build('pfn:../cm-interp-B/runs/L2_s0.05')
Gs = []
for i in range(8):
    st = net.sample(1 << 12, seed=1000 + i)          # the same entropy-sample seeds as run.py (chunks 0..7)
    Gs.append(net.gates(st, 2)[:, (net.T - 1) * net.dm:])
G = np.concatenate(Gs)
sz = dict(made_hidden=True)
p = RUN.entropy_panel(G, 64, sz, allow_made=True)
out = {k: p[k] for k in ('N', 'm', 'K_obs', 'plugin', 'mm', 'indep_upper', 'H2', 'H2_zero', 'upper')}
out['nsb'] = p['nsb']; out['fvsbn'] = p['fvsbn']; out['made'] = p.get('made')
json.dump(out, open(sys.argv[1], 'w'), indent=1)
print(out)
