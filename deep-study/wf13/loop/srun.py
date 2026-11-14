"""srun.py -- 6x6 Sudoku driver: the loop (amplify with the tree at a fixed budget per puzzle, distil, repeat), the
one-shot control, then scoring (sscore.py). Usage: srun.py SEED [key=value ...] -> out/s_s{S}{tag}.json, ckpt/, logs/.
Training puzzles: wf7's k6_train_min (training orbits; only the puzzle is read, never its stored solution).
"""
import copy
import json
import sys
import time

import numpy as np
import torch

sys.dont_write_bytecode = True
import sk, spol          # noqa: E402

ROOT = '/home/agent/workspaces/mats/work/deep-study/'
HERE = ROOT + 'wf13/loop/'
CFG = dict(R=5, n_train=200, B_train=512, rho=0.5, eps=0.02, S=24, steps=400, bs=256, lr=2e-3, order='conf',
           n_hard=100, n_mixed=100, n_multi=60, cap=2048, marks=[16, 32, 64, 128, 256, 512, 1024, 2048], K_is=64,
           d=48, L=3, tag='', offset=1, arms='all')


def main():
    seed = int(sys.argv[1])
    cfg = dict(CFG)
    for kv in sys.argv[2:]:
        k, v = kv.split('=')
        cfg[k] = type(CFG[k])(v) if not isinstance(CFG[k], list) else [int(z) for z in v.split(',')]
    name = 's_s%d%s' % (seed, cfg['tag'])
    logf = open(HERE + 'logs/%s.log' % name, 'a')

    def log(s):
        logf.write('[%s] %s\n' % (time.strftime('%H:%M:%S'), s))
        logf.flush()
    torch.set_num_threads(2)
    rules = sk.Rules(np.load(HERE + 'out/sudoku_rules.npz')['allowed'])
    train_puz = np.load(ROOT + 'wf7/sudoku/data/k6_train_min.npz')['puz']
    log('start %s cfg=%s' % (name, json.dumps(cfg)))

    def fresh(r):
        a = (seed * 1000 + r * cfg['n_train']) % (len(train_puz) - cfg['n_train'])
        return train_puz[a:a + cfg['n_train']]

    def amplify(pol, puzzles, rseed):
        buf, st = [], []
        for i, p in enumerate(puzzles):
            rng = np.random.default_rng(rseed * 100003 + i)
            T = sk.Tree(rules, p, 6 * cfg['B_train'] + 64)
            sk.grow(T, pol, cfg['B_train'], rng, rho=cfg['rho'], eps=cfg['eps'])
            st.append([T.n_open() == 0, (T.status[:T.N] == 2).any(), T.opened])
            d = spol.collect(T, cfg['S'], rng)
            if d is not None:
                buf.append(d)
        return buf, np.array(st, dtype=float).mean(0)

    torch.manual_seed(seed)
    net0 = spol.SNet(cfg['d'], cfg['L'])
    nets = {0: copy.deepcopy(net0)}
    res = dict(cfg=cfg, seed=seed, loop=[], n_params=sum(p.numel() for p in net0.parameters()))
    net, buf, buf0 = copy.deepcopy(net0), [], None
    for r in range(cfg['R']):
        t0 = time.time()
        new, st = amplify(spol.NetPolicy(net, order=cfg['order']), fresh(r), seed * 1000 + r)
        buf += new
        if r == 0:
            buf0 = list(new)
        loss = spol.train(net, buf, cfg['steps'], cfg['bs'], cfg['lr'], seed * 77 + r, log) if buf else float('nan')
        nets[r + 1] = copy.deepcopy(net)
        res['loop'].append(dict(round=r, closed=st[0], found=st[1], opened=st[2], states=sum(len(b[0]) for b in buf),
                                loss=loss, sec=time.time() - t0))
        log('round %d: at the training budget: interval closed on %.2f of puzzles, a solution found on %.2f, %.0f openings; '
            'buffer %d states; %.0fs' % (r, st[0], st[1], st[2], res['loop'][-1]['states'], time.time() - t0))
    # one-shot control: the same puzzles and the same number of gradient steps, every puzzle amplified by the
    # UNTRAINED policy at the same budget
    t0 = time.time()
    net1, buf1 = copy.deepcopy(net0), list(buf0)
    for r in range(1, cfg['R']):
        new, st = amplify(spol.NetPolicy(net0, order=cfg['order']), fresh(r), seed * 1000 + r)
        buf1 += new
    if buf1:
        spol.train(net1, buf1, cfg['steps'] * cfg['R'], cfg['bs'], cfg['lr'], seed * 77 + 99, log)
    nets['oneshot'] = net1
    res['oneshot'] = dict(states=sum(len(b[0]) for b in buf1), sec=time.time() - t0)
    log('one-shot control trained on %d states; %.0fs' % (res['oneshot']['states'], time.time() - t0))
    torch.save({str(k): v.state_dict() for k, v in nets.items()}, HERE + 'ckpt/%s.pt' % name)
    json.dump(res, open(HERE + 'out/%s.json' % name, 'w'))
    import sscore
    sscore.score_all(seed, cfg, rules, nets, res, name, log)


if __name__ == '__main__':
    main()
