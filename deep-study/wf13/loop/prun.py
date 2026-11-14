"""prun.py -- parity world driver: the loop (amplify with the tree at a fixed budget, distil, repeat), the one-shot
control, and all scoring. Usage: prun.py WORLD SEED [key=value ...]  -> out/p_w{W}_s{S}{tag}.json, ckpt/, logs/.
Exact answers (pexact) are computed in score_all() only and are passed to scoring functions after each search.
"""
import copy
import json
import sys
import time

import numpy as np
import torch

sys.dont_write_bytecode = True
import pdata, pw, ptree, ppol, ploop          # noqa: E402

HERE = pdata.HERE
CFG = dict(R=5, n_train=128, B_train=256, rho=0.5, eps=0.02, S=48, steps=500, bs=256, lr=2e-3, order='greedy1',
           n_test=60, marks=[16, 32, 64, 128, 256, 512, 1024, 2048], kmarks=[1, 4, 16, 64, 256], n_draw=4096,
           d=32, T=3, tag='', stage='all', hand='greedy1', n_tsamp=32)


def main():
    ws, seed = int(sys.argv[1]), int(sys.argv[2])
    cfg = dict(CFG)
    for kv in sys.argv[3:]:
        k, v = kv.split('=')
        cfg[k] = type(CFG[k])(v) if not isinstance(CFG[k], list) else [int(z) for z in v.split(',')]
    name = 'p_w%d_s%d%s' % (ws, seed, cfg['tag'])
    logf = open(HERE + 'logs/%s.log' % name, 'a')

    def log(s):
        logf.write('[%s] %s\n' % (time.strftime('%H:%M:%S'), s))
        logf.flush()
    torch.set_num_threads(2)
    tab = pdata.load_tab(ws)
    log('start %s cfg=%s' % (name, json.dumps(cfg)))

    def amplify(net_pol, xs, rseed):
        """One round of amplification at the fixed budget: returns distillation tuples and the books' own summary."""
        buf, stats = [], []
        for i, x in enumerate(xs):
            inst = pw.Inst(tab, x)
            rng = np.random.default_rng(rseed * 100003 + i)
            T = ploop.new_tree(inst, 2 * cfg['B_train'] + 8)
            ptree.grow(T, net_pol, cfg['B_train'], rng, rho=cfg['rho'], eps=cfg['eps'])
            bk = ptree.books(T)
            au = ptree.audit(T, bk, cfg['n_draw'], rng)
            stats.append([bk['unfound_ub'], float((au['hi'] - au['lo']).mean()), bk['n_closed']])
            d = ploop.collect(T, cfg['S'], rng, au.get('draws'))
            if d is not None:
                buf.append(d)
        return buf, np.array(stats).mean(0)

    def fresh(r):
        return pdata.fresh_x(ws, tab, cfg['n_train'], seed * 50 + r)

    inc = ppol.incidence(pw.Inst(tab, np.zeros(tab['checks'].shape[0], dtype=np.int64)))
    torch.manual_seed(seed)
    net0 = ppol.GNN(cfg['d'], cfg['T'])
    nets = {0: copy.deepcopy(net0)}
    res = dict(cfg=cfg, world=ws, seed=seed, loop=[], n_params=sum(p.numel() for p in net0.parameters()))
    # ------------------------------------------------------------------ the loop
    net, buf = copy.deepcopy(net0), []
    for r in range(cfg['R']):
        t0 = time.time()
        new, st = amplify(ppol.NetPolicy(net, order=cfg['order']), fresh(r), seed * 1000 + r)
        buf += new
        if r == 0:
            buf0 = list(new)
        loss = ploop.train(net, buf, cfg['steps'], cfg['bs'], cfg['lr'], seed * 77 + r, inc, log)
        nets[r + 1] = copy.deepcopy(net)
        res['loop'].append(dict(round=r, unfound_ub=st[0], w_aud=st[1], n_closed=st[2], states=sum(len(b[0]) for b in buf),
                                loss=loss, sec=time.time() - t0))
        log('round %d: books at the training budget: unfound upper bound %.4f, audit width %.4f, closed states %.1f; '
            'buffer %d states; %.0fs' % (r, st[0], st[1], st[2], res['loop'][-1]['states'], time.time() - t0))
        torch.save({k: v.state_dict() for k, v in nets.items()}, HERE + 'ckpt/%s.pt' % name)
    # ------------------------------------------------------------------ one-shot control: same instances, same total
    # number of amplified instances and of gradient steps, but every instance is amplified by the UNTRAINED policy
    t0 = time.time()
    net1, buf1 = copy.deepcopy(net0), []
    for r in range(cfg['R']):
        if r == 0:
            new = buf0                                   # round 0 of the loop is the same amplification, same seeds
        else:
            new, st = amplify(ppol.NetPolicy(net0, order=cfg['order']), fresh(r), seed * 1000 + r)
        buf1 += new
    ploop.train(net1, buf1, cfg['steps'] * cfg['R'], cfg['bs'], cfg['lr'], seed * 77 + 99, inc, log)
    nets['oneshot'] = net1
    res['oneshot'] = dict(states=sum(len(b[0]) for b in buf1), sec=time.time() - t0)
    log('one-shot control trained on %d states; %.0fs' % (res['oneshot']['states'], time.time() - t0))
    torch.save({str(k): v.state_dict() for k, v in nets.items()}, HERE + 'ckpt/%s.pt' % name)
    json.dump(res, open(HERE + 'out/%s.json' % name, 'w'))
    import pscore
    pscore.score_all(ws, seed, cfg, tab, nets, res, name, log)


if __name__ == '__main__':
    main()
