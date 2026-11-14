"""spilot.py -- pilot on TRAINING-orbit puzzles only: tree sizes under hand policies. Not a registered run."""
import sys, time
import numpy as np
import sk, spol, sexact
ROOT = '/home/agent/workspaces/mats/work/deep-study/'
rules = sk.Rules(np.load(ROOT + 'wf13/loop/out/sudoku_rules.npz')['allowed'])
d = np.load(ROOT + 'wf7/sudoku/data/%s.npz' % sys.argv[1])
n = int(sys.argv[2]); cap = int(sys.argv[3])
puz = d['puz'][:n]
print(sys.argv[1], 'clues per puzzle: mean %.1f' % (puz >= 0).sum(1).mean())
for name, pol, rho in [('mrv', spol.Hand('mrv'), 0.5), ('random', spol.Hand('random'), 0.5), ('random rho0', spol.Hand('random'), 0.0)]:
    t0 = time.time(); op, ff, cl, bad = [], [], [], 0
    for i in range(n):
        rng = np.random.default_rng(i)
        T = sk.Tree(rules, puz[i], 6 * cap + 64)
        sk.grow(T, pol, cap, rng, rho=rho)
        bk = sk.books(T)
        sols = sexact.solutions(puz[i])
        ex = sexact.marginals(sols)
        bad += int(((ex < bk['lo'] - 1e-9) | (ex > bk['hi'] + 1e-9)).sum())
        op.append(T.opened); ff.append(T.first_found); cl.append(bk['closed'])
    op, ff, cl = np.array(op), np.array(ff), np.array(cl)
    print('%-12s closed %.2f  opened to close: median %.0f mean %.0f max %d | first solution at: median %.0f (found %.2f) | exact answers outside interval: %d | %.1fs'
          % (name, cl.mean(), np.median(op[cl]) if cl.any() else -1, op[cl].mean() if cl.any() else -1, op.max(), np.median(ff[ff > 0]) if (ff > 0).any() else -1, (ff > 0).mean(), bad, time.time() - t0))
