"""c7_guards.py -- checker's re-run of the guard arms on fresh data (claims 3 and 11): which guard keeps an avoiding
policy from leaving the interval open. Parity: 30 fresh instances per world (my seed), audit width at 256 / 2,048.
Sudoku: 100 hard wf7 test puzzles outside the author's ranges, share closed inside 2,048 openings and C."""
import sys, time
import numpy as np
import torch
sys.dont_write_bytecode = True
ROOT = '/home/agent/workspaces/mats/work/deep-study/'
sys.path.insert(0, ROOT + 'wf13/loop')
import pdata, pw, ptree, ppol, ploop, sk, spol          # noqa: E402

torch.set_num_threads(2)
AV = list(range(5))
for ws in (101, 102, 103):
    tab = pdata.load_tab(ws)
    W = pdata.world(ws, tab)
    _, X = W.sample(30, ws * 1000 + 944)
    sd = torch.load(ROOT + 'wf13/loop/ckpt/p_w%d_s1.pt' % ws, map_location='cpu')
    g = ppol.GNN(32, 3)
    g.load_state_dict(sd['5'])
    g.eval()
    arms = [('round 5 itself', ppol.NetPolicy(g, order='greedy1'), 0.5, 0.02),
            ('round 5, no share', ppol.NetPolicy(g, order='greedy1'), 0.0, 0.02),
            ('avoid, no guard', ppol.NetPolicy(g, 'avoid', AV, order='greedy1'), 0.0, 0.0),
            ('avoid, floor only', ppol.NetPolicy(g, 'avoid', AV, order='greedy1'), 0.0, 0.02),
            ('avoid, share only', ppol.NetPolicy(g, 'avoid', AV, order='greedy1'), 0.5, 0.0)]
    for an, pol, rho, eps in arms:
        w = {256: [], 1024: [], 2048: []}
        for i, x in enumerate(X):
            inst = pw.Inst(tab, x.astype(np.int64))
            rng = np.random.default_rng(77 * (i + 1) + ws)
            T = ploop.new_tree(inst, 2 * 2048 + 8)
            for mk in w:
                ptree.grow(T, pol, mk, rng, rho=rho, eps=eps)
                bk = ptree.books(T)
                au = ptree.audit(T, bk, 4096, rng)
                w[mk].append(float((au['hi'] - au['lo']).mean()))
        print('parity %d %-18s audit width at 256 / 1,024 / 2,048 openings: %.3f / %.3f / %.3f' % (ws, an, np.mean(w[256]), np.mean(w[1024]), np.mean(w[2048])), flush=True)
rules = sk.Rules(np.load(ROOT + 'wf13/loop/out/sudoku_rules.npz')['allowed'])
d = np.load(ROOT + 'wf7/sudoku/data/k6_test_min.npz')
hi_ = np.where(d['rating'] == 2)[0]
hard = d['puz'][np.concatenate([hi_[250:300], hi_[16:66]])]
for s in (1, 2):
    sd = torch.load(ROOT + 'wf13/loop/ckpt/s_s%d.pt' % s, map_location='cpu')
    net = spol.SNet(48, 3)
    net.load_state_dict(sd['5'])
    net.eval()
    for an, pol, rho, eps in [('avoid, no guard', spol.NetPolicy(net, 'avoid'), 0.0, 0.0), ('avoid, floor only', spol.NetPolicy(net, 'avoid'), 0.0, 0.02),
                              ('avoid, share only', spol.NetPolicy(net, 'avoid'), 0.5, 0.0), ('wrong goal', spol.NetPolicy(net, 'perm'), 0.5, 0.02)]:
        Cs = []
        for i, p in enumerate(hard):
            rng = np.random.default_rng(123 * (i + 1) + s)
            T = sk.Tree(rules, p, 6 * 2048 + 64)
            sk.grow(T, pol, 2048, rng, rho=rho, eps=eps)
            Cs.append(T.opened if T.n_open() == 0 else np.nan)
        Cs = np.array(Cs)
        print('sudoku seed %d %-18s closes %.2f of hard puzzles; C (2,048 if not closed) %.0f' % (s, an, np.isfinite(Cs).mean(), np.nan_to_num(Cs, nan=2048).mean()), flush=True)
print('done')
