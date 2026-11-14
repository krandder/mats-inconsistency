"""Fresh-data validation of the merged entropy E3 (review finding 1). usage: e3_fresh.py SPEC TAG
For each layer and tolerance eps where E3(eps) was reached: rebuild the same tree (deterministic, flip set seed 1),
freeze it at the selected checkpoint (first one meeting eps on the original held-out set, seed 2), and evaluate the
frozen tree on fresh samples (seed 3): change rate with a one-sided 95% Clopper-Pearson upper limit, leaf entropy
(plug-in and Miller-Madow) and the number of leaves used. Writes results/TAG_fresh.json."""
import sys, os, json, time, math, copy
import numpy as np, torch
from scipy.stats import beta
torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run as RUN, analyze as AN, ent

spec, tag = sys.argv[1], sys.argv[2]
res = json.load(open(os.path.join(RUN.HERE, 'results', tag + '.json')))
net = RUN.build(spec); sz = res['sizes']
st_f = net.sample(sz['N_flip'], seed=1)
if net.kind == 'pfn':
    dev = st_f['ans'] - st_f['bayes_mean']; net.tol_abs = float(net.tolfrac * dev.pow(2).mean().sqrt())
n_fresh = 1 << 16 if net.kind == 'rnn' else 1 << 15
st_n = net.sample(n_fresh, seed=3)
out = dict(spec=spec, tag=tag, n_fresh=n_fresh, layers={})
for l in net.layers:
    L = res['layers'][str(l)]; E3 = L['E3']
    targets = {}
    for key in [k for k in E3 if k.startswith('H_merged_') and E3[k] is not None]:
        H = E3[key]
        row = next(r for r in E3['path'] if abs(r['H_leaves'] - H) < 1e-12)
        targets[key] = row['splits']
    if not targets:
        continue
    t0 = time.time()
    F, _ = AN.flip_matrix(net, st_f, l, np.arange(sz['N_flip']))
    assert np.allclose(F.mean(0), np.array(L['flip_rate']))
    G_tr = net.gates(st_f, l)
    snaps = {}
    def on_cp(tree, splits):
        if splits in targets.values():
            snaps[splits] = (list(tree.gate), [list(c) for c in tree.child], [None if r is None else r.copy() for r in tree.rep])
    AN.build_tree_path(G_tr, F, max_splits=max(targets.values()), checkpoints=sorted(set(targets.values())), on_checkpoint=on_cp)
    G_n = net.gates(st_n, l); idx = np.arange(n_fresh)
    base = net.sub(st_n, l, idx, torch.from_numpy(G_n))
    rows = {}
    for key, k in targets.items():
        tr = AN.Tree(G_tr.shape[1]); tr.gate, tr.child, reps = copy.deepcopy(snaps[k][0]), copy.deepcopy(snaps[k][1]), snaps[k][2]
        leaf = tr.assign(G_n)
        R = np.stack([r if r is not None else np.zeros(G_tr.shape[1], bool) for r in reps])
        a1 = net.sub(st_n, l, idx, torch.from_numpy(R[leaf]))
        ch = net.changed(base, a1).numpy(); x = int(ch.sum())
        up95 = float(beta.ppf(0.95, x + 1, n_fresh - x)) if x < n_fresh else 1.0
        c = np.bincount(leaf); c = c[c > 0]
        row = dict(splits=k, leaves_in_tree=k + 1, leaves_used_fresh=int(len(c)), H_fresh=ent.plugin(c), H_fresh_mm=ent.miller_madow(c),
                   change_rate_fresh=x / n_fresh, change_upper95=up95, H_selected=E3[key])
        if hasattr(net, 'bias_nmse'):
            d = (a1 - base).double()
            mean_d = d.mean(0); se = d.std(0) / math.sqrt(n_fresh)
            row['bias_nmse_fresh'] = net.bias_nmse(base, a1, n_fresh)
            row['bias_nmse_noise_floor'] = float((se ** 2).mean() / (net.scale ** 2 / d.shape[1]))
        rows[key] = row
    out['layers'][str(l)] = rows
    print(time.strftime('%H:%M:%S'), tag, 'l', l, {k: (round(v['H_fresh_mm'], 2), round(100 * v['change_rate_fresh'], 3), round(100 * v['change_upper95'], 3), v['leaves_in_tree']) for k, v in rows.items()}, f'{time.time() - t0:.0f}s', flush=True)
    json.dump(out, open(os.path.join(RUN.HERE, 'results', tag + '_fresh.json'), 'w'), indent=1)
print('e3 fresh done', tag)
