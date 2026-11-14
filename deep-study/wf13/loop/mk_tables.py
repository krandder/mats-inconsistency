"""mk_tables.py -- read the one score's numbers from the trained world models on disk and cache them.

Parity worlds (wf7/probworld, wf11/learn-the-score): a_i = M's number for assumption i with an empty context,
f_c = M's number for check c given only its three parents (wf7 arms.score_table, unchanged). 1 + 8m calls of M per world.
Sudoku 6x6 (wf7/sudoku): allowed[i, u, j, v] = [log M(X_j = v | only X_i = u) >= theta], theta from wf7 out/pair_meta.json
(set there from real validation grids). 216 calls of M.
No exact inference here. Usage: mk_tables.py  -> out/tab_w{seed}.npz, out/sudoku_rules.npz, out/mk_tables.out
"""
import json
import sys

import numpy as np
import torch

sys.dont_write_bytecode = True
ROOT = '/home/agent/workspaces/mats/work/deep-study/'
HERE = ROOT + 'wf13/loop/'
log = open(HERE + 'out/mk_tables.out', 'w')


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    log.write(s + '\n')
    log.flush()


def parity():
    sys.path.insert(0, ROOT + 'wf7/probworld')
    import model as MM
    import arms as A
    from world import World
    todo = [(7, 20, 30, ROOT + 'wf7/probworld/ckpt/M_w7_S.pt', 'S'),
            (101, 20, 30, ROOT + 'wf7/probworld/ckpt/M_w101_M.pt', 'M'),
            (102, 20, 30, ROOT + 'wf7/probworld/ckpt/M_w102_M.pt', 'M'),
            (103, 20, 30, ROOT + 'wf7/probworld/ckpt/M_w103_M.pt', 'M'),
            (311, 20, 30, ROOT + 'wf11/learn-the-score/ckpt/M_w311_M.pt', 'M'),
            (201, 40, 60, ROOT + 'wf7/probworld/ckpt/Mscale_w201_M.pt', 'M')]
    for ws, n, m, path, size in todo:
        W = World(ws, n=n, m=m)
        ck = torch.load(path, map_location='cpu')
        net = MM.build(size, n + m)
        net.load_state_dict(ck['state'])
        net.eval()
        A.N_H = n
        S = A.score_table(net, torch.tensor(W.checks), torch.device('cpu'))
        loga, logf = S.loga.double().numpy(), S.logf.double().numpy()
        # distance of the table from the true mechanism (a property of M, reported for the reader; evaluation only)
        par = np.array([bin(v).count('1') & 1 for v in range(8)])
        tf = np.where(par == 1, 1 - W.eps, W.eps)
        say('world %d: n=%d m=%d rank=%d  calls of M=%d  max|a - prior|=%.4f  max|f - true f|=%.4f  checks with error > 0.1: %s'
            % (ws, n, m, W.gf2_rank(), S.calls, np.abs(np.exp(loga[:, 1]) - W.prior).max(),
               np.abs(np.exp(logf[:, :, 1]) - tf[None]).max(),
               np.where(np.abs(np.exp(logf[:, :, 1]) - tf[None]).max(1) > 0.1)[0].tolist()))
        np.savez(HERE + 'out/tab_w%d.npz' % ws, loga=loga, logf=logf, checks=W.checks, prior=W.prior, eps=W.eps,
                 calls=S.calls)
    sys.path.pop(0)
    for k in ['model', 'arms', 'world']:
        sys.modules.pop(k, None)


def sudoku():
    sys.path.insert(0, ROOT + 'wf7/sudoku')
    import core
    theta = json.load(open(ROOT + 'wf7/sudoku/out/pair_meta.json'))['theta']
    m, _ = core.load_model(ROOT + 'wf7/sudoku/ckpt/mA6.pt')
    M = core.Net(m, 6, 'mA6')
    N, K = 36, 6
    lp = np.zeros((N, K, N, K), dtype=np.float32)
    for i in range(N):
        s = torch.full((K, N), -1, dtype=torch.long)
        s[torch.arange(K), i] = torch.arange(K)
        lp[i] = M.logp(s).numpy()
    allowed = lp >= theta
    for i in range(N):
        allowed[i, :, i, :] = np.eye(K, dtype=bool)          # a cell agrees with itself only
    # a property of M, for the reader (evaluation only): compare with the true rules
    r, c = np.arange(N) // K, np.arange(N) % K
    b = (r // 2) * 2 + c // 3
    peer = ((r[:, None] == r[None]) | (c[:, None] == c[None]) | (b[:, None] == b[None])) & ~np.eye(N, dtype=bool)
    true = np.ones((N, K, N, K), dtype=bool)
    for u in range(K):
        true[:, u, :, u] = ~peer
    for i in range(N):
        true[i, :, i, :] = np.eye(K, dtype=bool)
    say('sudoku 6x6: theta=%.3f  calls of M=%d  rule table equals the true rules: %s (%d of %d entries differ); '
        'smallest margin to theta: %.2f nats' % (theta, M.calls, bool((allowed == true).all()), int((allowed != true).sum()),
                                                  allowed.size, float(np.abs(lp - theta).min())))
    np.savez(HERE + 'out/sudoku_rules.npz', allowed=allowed, theta=theta, calls=M.calls)


if __name__ == '__main__':
    parity()
    sudoku()
