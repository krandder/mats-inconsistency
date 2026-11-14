"""Recompute E3 (merged entropy) with the current tree for a finished run (Amendment 1). usage: e3_rerun.py SPEC TAG
Re-samples the same flip set (seed 1) and held-out set (seed 2), recomputes the single-flip matrix, runs the tree,
and writes E3 into results/TAG.json (old E3 kept as E3_v1)."""
import sys, os, json, time
import numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as RUN, analyze as AN
spec, tag = sys.argv[1], sys.argv[2]
p = os.path.join(RUN.HERE, 'results', tag + '.json')
res = json.load(open(p))
net = RUN.build(spec); sz = res['sizes']
st_f = net.sample(sz['N_flip'], seed=1)
if net.kind == 'pfn':
    dev = st_f['ans'] - st_f['bayes_mean']; net.tol_abs = float(net.tolfrac * dev.pow(2).mean().sqrt())
st_h = net.sample(sz['N_ho'], seed=2)
for l in net.layers:
    t0 = time.time()
    L = res['layers'][str(l)]
    if net.kind == 'pfn' and l == 2:
        m = net.m[l]; qcols = np.arange((net.T - 1) * net.dm, net.T * net.dm)
        F = np.zeros((sz['N_flip'], m), dtype=bool)
        Fq, _ = RUN.flip_matrix_cols(net, st_f, l, np.arange(sz['N_flip']), qcols); F[:, qcols] = Fq
    else:
        F, _ = AN.flip_matrix(net, st_f, l, np.arange(sz['N_flip']))
    assert np.allclose(F.mean(0), np.array(L['flip_rate'])), 'flip rates differ from the original run'
    eps_list = (0.01, 0.002) if net.kind == 'rnn' else (0.01,)
    if 'E3_v1' not in L: L['E3_v1'] = L['E3']
    L['E3'] = AN.merged_entropy(net, st_f, st_h, l, F, eps_list, max_splits=sz['max_splits'], log_every=False)
    print(time.strftime('%H:%M:%S'), tag, 'l', l, {k: v for k, v in L['E3'].items() if k.startswith(('H_merged', 'final'))}, f'{time.time() - t0:.0f}s', flush=True)
    json.dump(res, open(p, 'w'))
print('e3 rerun done', tag)
