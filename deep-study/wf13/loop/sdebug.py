"""sdebug.py -- reproduces the two interval violations of the first Sudoku scoring (one-shot arm, seed 1) and shows
that they come from floating-point cancellation in the old formula for UN. -> out/sdebug.out"""
import sys
import numpy as np, torch
sys.dont_write_bytecode = True
import sk, spol, sexact
ROOT = '/home/agent/workspaces/mats/work/deep-study/'
HERE = ROOT + 'wf13/loop/'
rules = sk.Rules(np.load(HERE + 'out/sudoku_rules.npz')['allowed'])
sd = torch.load(HERE + 'ckpt/s_s1.pt', map_location='cpu')
ARM = sys.argv[1] if len(sys.argv) > 1 else 'oneshot'
net = spol.SNet(48, 3); net.load_state_dict(sd['oneshot' if ARM == 'oneshot' else '5']); net.eval()
pol = spol.NetPolicy(net, 'ok' if ARM == 'oneshot' else 'avoid', order='conf')
RHO, EPS = (0.5, 0.02) if ARM == 'oneshot' else (0.0, 0.0)
d = np.load(ROOT + 'wf7/sudoku/data/k6_test_min.npz')
sets = dict(hard=d['puz'][np.where(d['rating'] == 2)[0][150:250]], mixed=d['puz'][1000:1100],
            multi=np.load(ROOT + 'wf7/sudoku/data/k6_test_multi.npz')['puz'][100:160])
marks = [16, 32, 64, 128, 256, 512, 1024, 2048]
out = []
tot_old = tot_new = 0
for sn, puzzles in sets.items():
    for i, p in enumerate(puzzles):
        rng = np.random.default_rng(7000 * (i + 1) + 1)
        sols = sexact.solutions(p); exm, emp = sexact.marginals(sols), p < 0
        T = sk.Tree(rules, p, 6 * 2048 + 64)
        for mk in marks:
            sk.grow(T, pol, mk, rng, rho=RHO, eps=EPS)
            bk = sk.books(T)                                   # fixed formula
            st = T.status[:T.N]; fr = st == 0
            U = np.exp(T.logU[:T.N][fr]); b, c = T.board[:T.N][fr], T.cand[:T.N][fr]
            isv = np.stack([(b == v) for v in range(6)], -1); und = (b < 0)[:, :, None] & c
            UA = np.tensordot(U, isv.astype(float), 1); Uu = np.tensordot(U, und.astype(float), 1)
            UN_old = U.sum() - UA - Uu
            ZA = np.stack([(bk['sol'] == v).sum(0) for v in range(6)], -1).astype(float) if bk['Z'] else np.zeros((36, 6))
            with np.errstate(invalid='ignore', divide='ignore'):
                lo_old = np.nan_to_num(ZA / (ZA + (bk['Z'] - ZA) + UN_old + Uu), nan=0.0)
            v_old = int(((exm < lo_old - 1e-9) | (exm > bk['hi'] + 1e-9))[emp].sum())
            v_new = int(((exm < bk['lo'] - 1e-9) | (exm > bk['hi'] + 1e-9))[emp].sum())
            tot_old += v_old; tot_new += v_new
            if v_old:
                j = np.argwhere(((exm < lo_old - 1e-9) & emp[:, None]))[0]
                out.append('%s puzzle %d at %d openings: old formula %d violation(s); e.g. cell %d value %d: exact %.4f, old lower '
                           'bound %.4f, fixed lower bound %.6f; U.sum() = %.3e, UN_old = %.3e (true UN = %.3e), found = %d'
                           % (sn, i, mk, v_old, j[0], j[1], exm[j[0], j[1]], lo_old[j[0], j[1]], bk['lo'][j[0], j[1]], U.sum(),
                              UN_old[j[0], j[1]], np.tensordot(U, (~isv & ~und).astype(float), 1)[j[0], j[1]], bk['Z']))
out.append('%s arm, seed 1, all 260 test puzzles, all marks: violations with the old formula %d, with the fixed formula %d' % (ARM, tot_old, tot_new))
open(HERE + 'out/sdebug_%s.out' % ARM, 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
