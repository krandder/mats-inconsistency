"""c4_sloop.py -- checker's replication of the Sudoku loop with a FRESH seed (registered settings of srun.py CFG; the
author's sk.py / spol.py are the code under test; this driver and the scoring are mine; outputs stay in check/out).
Scored on hard wf7 test puzzles the author's runs never scored. Usage: c4_sloop.py SEED N_TEST"""
import copy, json, sys, time
import numpy as np
import torch
sys.dont_write_bytecode = True
ROOT = '/home/agent/workspaces/mats/work/deep-study/'
sys.path.insert(0, ROOT + 'wf13/loop')
import sk, spol          # noqa: E402

OUT = ROOT + 'wf13/loop/check/out/'
seed, n_test = int(sys.argv[1]), int(sys.argv[2])
R, n_train, B_train, rho, eps, S, steps, bs, lr = 5, 200, 512, 0.5, 0.02, 24, 400, 256, 2e-3
torch.set_num_threads(2)
rules = sk.Rules(np.load(ROOT + 'wf13/loop/out/sudoku_rules.npz')['allowed'])
train_puz = np.load(ROOT + 'wf7/sudoku/data/k6_train_min.npz')['puz']
d = np.load(ROOT + 'wf7/sudoku/data/k6_test_min.npz')
hi_ = np.where(d['rating'] == 2)[0]
test = d['puz'][np.concatenate([hi_[250:300], hi_[16:150]])[:n_test]]
sols = d['sol'][np.concatenate([hi_[250:300], hi_[16:150]])[:n_test]]          # scoring only


def fresh(r):
    a = (seed * 1000 + r * n_train) % (len(train_puz) - n_train)
    return train_puz[a:a + n_train]


def amplify(pol, puzzles, rseed):
    buf, st = [], []
    for i, p in enumerate(puzzles):
        rng = np.random.default_rng(rseed * 100003 + i)
        T = sk.Tree(rules, p, 6 * B_train + 64)
        sk.grow(T, pol, B_train, rng, rho=rho, eps=eps)
        st.append([T.n_open() == 0, (T.status[:T.N] == 2).any(), T.opened])
        dd = spol.collect(T, S, rng)
        if dd is not None:
            buf.append(dd)
    return buf, np.array(st, dtype=float).mean(0)


def say(s):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), s), flush=True)


torch.manual_seed(seed)
net0 = spol.SNet(48, 3)
nets = {'0': copy.deepcopy(net0)}
net, buf, buf0 = copy.deepcopy(net0), [], None
for r in range(R):
    new, st = amplify(spol.NetPolicy(net), fresh(r), seed * 1000 + r)
    buf += new
    if r == 0:
        buf0 = list(new)
    if buf:
        spol.train(net, buf, steps, bs, lr, seed * 77 + r)
    nets[str(r + 1)] = copy.deepcopy(net)
    say('round %d at the training budget: closed %.2f, openings %.0f, buffer %d states' % (r, st[0], st[2], sum(len(b[0]) for b in buf)))
net1, buf1 = copy.deepcopy(net0), list(buf0)
for r in range(1, R):
    new, st = amplify(spol.NetPolicy(net0), fresh(r), seed * 1000 + r)
    buf1 += new
if buf1:
    spol.train(net1, buf1, steps * R, bs, lr, seed * 77 + 99)
nets['oneshot'] = net1
say('one-shot control: %d states' % sum(len(b[0]) for b in buf1))
torch.save({k: v.state_dict() for k, v in nets.items()}, OUT + 'c4_s%d.pt' % seed)
res = {}
arms = [(k, spol.NetPolicy(nets[k])) for k in ['0', '1', '2', '3', '4', '5', 'oneshot']] + [('hand_mrv', spol.Hand('mrv'))]
for an, pol in arms:
    Cs, okset, succ = [], 0, []
    for i, p in enumerate(test):
        rng = np.random.default_rng(31337 * (i + 1) + seed)
        T = sk.Tree(rules, p, 6 * 2048 + 64)
        sk.grow(T, pol, 2048, rng, rho=rho, eps=eps)
        closed = T.n_open() == 0
        Cs.append(T.opened if closed else 2048)
        found = T.board[:T.N][T.status[:T.N] == 2]
        okset += int(closed and len(found) == 1 and (found[0] == sols[i]).all())
        b, lw, _ = sk.run_is(rules, p, pol, 64, rng, eps)
        succ.append(float(np.isfinite(lw).mean()))
    res[an] = dict(C_mean=float(np.mean(Cs)), C_median=float(np.median(Cs)), closed=float(np.mean(np.array(Cs) < 2048)),
                   exact_set=okset, rollout_success=float(np.mean(succ)), per=Cs)
    say('scored %-8s hard C mean %.1f median %.1f | closed with exactly the true solution %d of %d | rollouts ending in the solution %.2f'
        % (an, np.mean(Cs), np.median(Cs), okset, len(test), np.mean(succ)))
    json.dump(res, open(OUT + 'c4_s%d.json' % seed, 'w'))
say('final / one-shot = %.2f ; final / hand = %.2f ; round1 / round0 = %.3f' % (res['5']['C_mean'] / res['oneshot']['C_mean'],
    res['5']['C_mean'] / res['hand_mrv']['C_mean'], res['1']['C_mean'] / res['0']['C_mean']))
