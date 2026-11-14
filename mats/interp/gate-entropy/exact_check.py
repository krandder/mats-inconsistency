"""Estimator check on real gate data: T16 per-step patterns (16 gates). Reference = plug-in / Miller-Madow on N = 2^21
(<= 65,536 cells, bias < 0.02 bits); estimators on disjoint subsamples of 2^10 .. 2^16 (3 replicates each)."""
import sys, os, json, math, time
import numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ent, nets
net = nets.RNNNet('../algzoo-noisy/runs/ft_16_10_ex_s0.001.npz', name='T16')
N = 1 << 21; chunk = 1 << 16
Gs = {l: [] for l in net.layers}
for i in range(0, N, chunk):
    st = net.sample(chunk, seed=50_000 + i // chunk)
    for l in net.layers: Gs[l].append(net.gates(st, l))
out = {}
for l in net.layers:
    G = np.concatenate(Gs[l]); Gs[l] = None
    c, _ = ent.pattern_ids(G)
    ref = dict(plugin=ent.plugin(c), mm=ent.miller_madow(c), K_obs=int(len(c)))
    rows = []
    rng = np.random.default_rng(l)
    perm = rng.permutation(N)
    for n in (1 << 10, 1 << 12, 1 << 14, 1 << 16):
        for rep in range(3):
            S = G[perm[rep * n:(rep + 1) * n]]
            cs, _ = ent.pattern_ids(S)
            r = dict(n=n, rep=rep, plugin=ent.plugin(cs), mm=ent.miller_madow(cs), nsb=ent.nsb(cs, 16)['H'], H2=ent.collision(cs)[0],
                     indep=ent.indep_upper(S))
            if n >= 1 << 12:
                r['fvsbn'] = ent.made_upper(S, hidden=0, seed=rep)['H_up']
                r['made'] = ent.made_upper(S, hidden=64, seed=rep)['H_up']
            rows.append(r)
    out[l] = dict(ref=ref, rows=rows)
    agg = {}
    for n in (1 << 10, 1 << 12, 1 << 14, 1 << 16):
        rr = [r for r in rows if r['n'] == n]
        agg[n] = {k: round(float(np.mean([r[k] for r in rr if k in r])) - ref['mm'], 3) for k in ('plugin', 'mm', 'nsb', 'H2', 'indep', 'fvsbn', 'made') if any(k in r for r in rr)}
    out[l]['bias_vs_ref'] = agg
    print(time.strftime('%H:%M:%S'), 'step', l, 'ref', ref, 'bias by n:', agg, flush=True)
    json.dump(out, open('results/exact_check_T16.json', 'w'), indent=1)
