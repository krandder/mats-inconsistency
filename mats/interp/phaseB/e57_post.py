"""E57 post-review check (not registered): how well c's one-parameter reading holds along training. For each E47 lineage and
checkpoint (as e57_run.py), lngeom.py's diagnostics: A's coefficient of variation over labelled tokens (the error-free part's
norm; the one-parameter cap assumes a narrow A) and the median |rho| (the cross term between the error direction and the rest;
the even cap assumes rho near 0); also the median A and U separately (c = A / U can move through either).
Medians over lineages by step and final type (E47's lineage rules via e57_summ.lineages).
usage: python e57_post.py -> results/e57_post.json
"""
import glob, json, os, re, sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
import e48_fmt as F  # noqa: E402
import lngeom as G  # noqa: E402
import e57_summ as S  # noqa: E402

z = np.load('results/e27_data.npz')
X, _ = F.make_X(z['o'][:500, 0], z['y'][:500], z['semb'][:500], z['qemb'][:500], 'ctrl')
diag = {}
for tag in json.load(open('results/e57_c.json')):
    run = os.path.join('runs', tag.split('_')[0], tag); m = {}
    for ck in sorted(glob.glob(os.path.join(run, 'ckpt/step_*.pt'))):
        st = int(re.findall(r'step_(\d+)', ck)[0])
        if st in S.STEPS:
            sd = torch.load(ck, map_location='cpu'); g = G.geom_wb(sd['inp.weight'].numpy(), sd['inp.bias'].numpy(), X)
            m[st] = dict(A_cv=g['A_cv'], rho=g['rho_abs_median'], A=float(np.median(g['A'])), U=g['U'])
    diag[tag] = m
L = S.lineages({t: {s: 0 for s in m} for t, m in diag.items()})
out = {}
for st in S.STEPS:
    o = {}
    for typ, sel in (('A', lambda x: not x['B']), ('B', lambda x: x['B'])):
        v = [diag[x['tag']][st] for x in L if sel(x) and st in x['c']]
        o[typ] = {k: float(np.median([d[k] for d in v])) for k in ('A_cv', 'rho', 'A', 'U')}
    out[st] = o
json.dump(out, open('results/e57_post.json', 'w'), indent=1)
for st, o in out.items():
    print(st, ' | '.join('%s: A_cv %.3f |rho| %.3f A %.2f U %.3f' % (t, d['A_cv'], d['rho'], d['A'], d['U']) for t, d in o.items()))
