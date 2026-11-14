"""sdiag.py -- POST HOC diagnosis: where do the openings go? For each policy, on the 100 hard test puzzles (closed
trees): openings on states that still contain the solution, openings inside dead subtrees (no solution below), the
number of wrong branches entered and their mean size. Usage: sdiag.py SEED [TAG] -> out/sdiag_s{SEED}{TAG}.out"""
import sys
import numpy as np, torch
sys.dont_write_bytecode = True
import sk, spol
ROOT = '/home/agent/workspaces/mats/work/deep-study/'
HERE = ROOT + 'wf13/loop/'
seed = int(sys.argv[1]); tag = sys.argv[2] if len(sys.argv) > 2 else ''
order = 'confmask' if tag == '_mask' else 'conf'
rules = sk.Rules(np.load(HERE + 'out/sudoku_rules.npz')['allowed'])
sd = torch.load(HERE + 'ckpt/s_s%d%s.pt' % (seed, tag), map_location='cpu')
d = np.load(ROOT + 'wf7/sudoku/data/k6_test_min.npz')
hard = d['puz'][np.where(d['rating'] == 2)[0][150:250]]
pols = [('round%s' % k, spol.NetPolicy(spol.SNet(48, 3), order=order)) for k in ('1', '2', '5')] + [('oneshot', spol.NetPolicy(spol.SNet(48, 3), order=order))]
for (k, p) in pols:
    p.net.load_state_dict(sd[k.replace('round', '')]); p.net.eval()
pols.append(('hand_mrv', spol.Hand('mrv')))
out = ['policy | openings | on states that contain the solution | inside dead subtrees | wrong branches entered | mean size of a wrong branch | largest']
for name, pol in pols:
    tot = on = dead = nb = 0
    sizes = []
    for i, p in enumerate(hard):
        rng = np.random.default_rng(7000 * (i + 1) + seed)
        T = sk.Tree(rules, p, 6 * 2048 + 64)
        sk.grow(T, pol, 2048, rng, rho=0.5, eps=0.02)
        n = T.N; st = T.status[:n]
        has = np.zeros(n, dtype=bool)
        for dn in np.flatnonzero(st == 2):
            j = dn
            while j >= 0 and not has[j]:
                has[j] = True; j = T.parent[j]
        opened = st == 1
        tot += int(opened.sum()); on += int((opened & has).sum()); dead += int((opened & ~has).sum())
        # size of each wrong branch: opened dead states grouped by their highest dead ancestor
        root = np.full(n, -1)
        for j in range(n):
            if not has[j]:
                pj = T.parent[j]
                root[j] = j if (pj < 0 or has[pj]) else root[pj]
        r = root[opened & ~has]
        if len(r):
            u, c = np.unique(r, return_counts=True)
            nb += len(u); sizes += c.tolist()
    out.append('%-9s | %6.1f | %6.1f | %6.1f | %5.2f | %6.1f | %d' % (name, tot / len(hard), on / len(hard), dead / len(hard), nb / len(hard),
                                                                 np.mean(sizes) if sizes else 0, max(sizes) if sizes else 0))
open(HERE + 'out/sdiag_s%d%s.out' % (seed, tag), 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
