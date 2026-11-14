"""Code check: answers recomputed by sub() with the actual gate pattern equal the forward pass's answers."""
import sys, numpy as np, torch
sys.argv = ['x']
import run as RUN
for spec in ['rnn:../algzoo-noisy/runs/gpu/w64_s0_sig0.01.npz', 'arc:64:32:0.01:0', 'tanh:3:64:32:0.01:0', 'tanh:0.25:64:8:0.1:0', 'pfn:../cm-interp-B/runs/L2_s0.05']:
    net = RUN.build(spec)
    st = net.sample(512, seed=3)
    out = []
    for l in net.layers:
        G = torch.from_numpy(net.gates(st, l))
        a = net.sub(st, l, np.arange(512), G)
        b = st['ans']
        if a.dtype == torch.bool:
            out.append(float((a != b).float().mean()))
        else:
            out.append(float((a - b).abs().max()))
    print(spec, 'max discrepancy per layer:', ['%.2g' % v for v in out])
import json
r = json.load(open('results/smoke_pfn____cm-interp-B_runs_L2_s0_05.json'))
print('pfn l2 non-query flip changes (must be 0):', r['layers']['2'].get('pfn_l2_nonquery_check_changes'))
