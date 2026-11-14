"""E57 (PREREG-B amendment 65; CPU, weights only): the LayerNorm cap scale c (lngeom.py, as e55_run.py: E27's first 500 bases,
ctrl format, labelled tokens) at every checkpoint of E47's lineages (E37's ctrl runs, E43's branches and new initialisations,
E44's hybrids, E46's perturbed runs), at steps 0, 2,000, 3,000, 5,000, 10,000, 20,000, 40,000, 60,000, 80,000 and 90,000.
No behaviour is computed here.
usage: python e57_run.py -> results/e57_c.json  ({tag: {step: c}})
"""
import glob, json, os, re, sys
import numpy as np
import torch

sys.path.insert(0, '/home/agent/workspaces/mats/work/tiny-pfn')
import e48_fmt as F  # noqa: E402
import lngeom as G  # noqa: E402

STEPS = [0, 2000, 3000, 5000, 10000, 20000, 40000, 60000, 80000, 90000]


def main():
    z = np.load('results/e27_data.npz')
    X, _ = F.make_X(z['o'][:500, 0], z['y'][:500], z['semb'][:500], z['qemb'][:500], 'ctrl')
    out = {}
    tags = [os.path.basename(f)[:-4] for f in sorted(glob.glob('results/e47/*.npz'))]
    for tag in tags:
        sub = tag.split('_')[0]
        run = os.path.join('runs', sub, tag)
        if not os.path.isdir(run):
            continue
        m = {}
        for ck in sorted(glob.glob(os.path.join(run, 'ckpt/step_*.pt'))):
            st = int(re.findall(r'step_(\d+)', ck)[0])
            if st in STEPS:
                sd = torch.load(ck, map_location='cpu')
                m[st] = G.geom_wb(sd['inp.weight'].numpy(), sd['inp.bias'].numpy(), X)['c_median']
        out[tag] = m
    json.dump(out, open('results/e57_c.json', 'w'), indent=1)
    print('c trajectories for %d lineages' % len(out))


if __name__ == '__main__':
    main()
