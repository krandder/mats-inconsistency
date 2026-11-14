"""E55 (PREREG-B amendment 63; CPU, weights only): block 1's LayerNorm cap scale c of world-1 nets (lngeom.py), from each net's
input embedding (inp.weight, inp.bias of model.pt) on the labelled tokens of E27's first 500 bases built in the net's own format.
Nets (cohorts as in e55_summ.py): exploration E37 ctrl and mag; confirmatory E38 (w64, w128, d3; ctrl, mag), E30 fix, E48 sq,
E43 (7 arms), E44 (6 swapped groups), E46 (3 perturbation sizes), E53 uni and unif. Training dynamics: every checkpoint of E37's
pure ctrl and mag runs (ckpt/step_*.pt).
No behaviour is computed here.
usage: python e55_run.py -> results/e55_geom.json, results/e55_traj.json
"""
import glob, json, os, re, sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
import e48_fmt as F  # noqa: E402
import lngeom as G  # noqa: E402

COHORTS = [  # (cohort, glob of run dirs, results subdirectory, exploration?)
    ('x_e37_ctrl', 'runs/e37/e37_ctrl_s*', 'e37', True), ('x_e37_mag', 'runs/e37/e37_mag_s*', 'e37', True),
    ('e38_w64ctrl', 'runs/e38/e38_w64ctrl_s*', 'e38', False), ('e38_w64mag', 'runs/e38/e38_w64mag_s*', 'e38', False),
    ('e38_w128ctrl', 'runs/e38/e38_w128ctrl_s*', 'e38', False), ('e38_w128mag', 'runs/e38/e38_w128mag_s*', 'e38', False),
    ('e38_d3ctrl', 'runs/e38/e38_d3ctrl_s*', 'e38', False), ('e38_d3mag', 'runs/e38/e38_d3mag_s*', 'e38', False),
    ('e30_fix', 'runs/e30/fix_s*', 'e30', False), ('e48_sq', 'runs/e48/e48_w1sq_s*', 'e48w1', False)]
COHORTS += [('e43_%s' % a, 'runs/e43/e43_%s_s*' % a, 'e43', False) for a in ('br0', 'br2k', 'br3k', 'br5k', 'br10k', 'br20k', 'newinit')]
COHORTS += [('e44_%s' % g, 'runs/e44/e44_%s_r*' % g, 'e44', False) for g in ('inp', 'b1att', 'b1mlp', 'b2att', 'b2mlp', 'head')]
COHORTS += [('e46_eps%s' % e, 'runs/e46/e46_eps%s_d*' % e, 'e46', False) for e in ('0.003', '0.03', '0.3')]
COHORTS += [('e53_uni', 'runs/e53/e53_uni_s*', 'e53', False), ('e53_unif', 'runs/e53/e53_unif_s*', 'e53', False)]
NB = 500


def tokens(fmt, z):
    X, _ = F.make_X(z['o'][:NB, 0], z['y'][:NB], z['semb'][:NB], z['qemb'][:NB], fmt)
    return X


def geom_of(sd, X):
    g = G.geom_wb(sd['inp.weight'].numpy(), sd['inp.bias'].numpy(), X)
    return dict(c=g['c_median'], A_cv=g['A_cv'], rho=g['rho_abs_median'], U=g['U'])


def main():
    z = np.load('results/e27_data.npz'); toks = {}
    out = {}
    for cohort, pat, sub, expl in COHORTS:
        for run in sorted(d for d in glob.glob(pat) if os.path.isdir(d) and os.path.exists(os.path.join(d, 'meta.json'))):
            tag = os.path.basename(run); meta = json.load(open(os.path.join(run, 'meta.json'))); fmt = meta['fmt']
            if fmt not in toks:
                toks[fmt] = tokens(fmt, z)
            sd = torch.load(os.path.join(run, 'model.pt'), map_location='cpu')
            out[tag] = dict(cohort=cohort, exploration=expl, sub=sub, fmt=fmt, width=meta['width'], depth=meta['depth'],
                            seed=meta['seed'], **geom_of(sd, toks[fmt]))
    json.dump(out, open('results/e55_geom.json', 'w'), indent=1)
    tr = {}
    for arm in ('ctrl', 'mag'):
        for run in sorted(d for d in glob.glob('runs/e37/e37_%s_s*' % arm) if os.path.isdir(d)):
            tag = os.path.basename(run); steps, cs = [], []
            for ck in sorted(glob.glob(os.path.join(run, 'ckpt/step_*.pt'))):
                steps.append(int(re.findall(r'step_(\d+)', ck)[0]))
                cs.append(geom_of(torch.load(ck, map_location='cpu'), toks.get(arm) if arm in toks else tokens(arm, z))['c'])
            tr[tag] = dict(step=steps, c=cs)
    json.dump(tr, open('results/e55_traj.json', 'w'), indent=1)
    print('geometry for %d nets, %d trajectories' % (len(out), len(tr)))


if __name__ == '__main__':
    main()
