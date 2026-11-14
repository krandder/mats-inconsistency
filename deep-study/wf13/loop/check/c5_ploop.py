"""c5_ploop.py -- checker's replication of the parity loop with a FRESH seed (registered settings of prun.py CFG; the
author's pw / ptree / ppol / ploop are the code under test; driver, instances, exact answers and scoring are mine).
Usage: c5_ploop.py WORLD SEED N_INST"""
import copy, json, sys, time
import numpy as np
import torch
sys.dont_write_bytecode = True
ROOT = '/home/agent/workspaces/mats/work/deep-study/'
sys.path.insert(0, ROOT + 'wf13/loop')
import pdata, pw, ptree, ppol, ploop          # noqa: E402

OUT = ROOT + 'wf13/loop/check/out/'
ws, seed, n_inst = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
R, n_train, B_train, rho, eps, S, steps, bs, lr, n_draw = 5, 128, 256, 0.5, 0.02, 48, 500, 256, 2e-3, 4096
tab = pdata.load_tab(ws)
n, m = tab['loga'].shape[0], tab['checks'].shape[0]
W = pdata.world(ws, tab)
torch.set_num_threads(2)


def say(s):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), s), flush=True)


def amplify(pol, xs, rseed):
    buf = []
    for i, x in enumerate(xs):
        inst = pw.Inst(tab, x)
        rng = np.random.default_rng(rseed * 100003 + i)
        T = ploop.new_tree(inst, 2 * B_train + 8)
        ptree.grow(T, pol, B_train, rng, rho=rho, eps=eps)
        bk = ptree.books(T)
        au = ptree.audit(T, bk, n_draw, rng)
        dd = ploop.collect(T, S, rng, au.get('draws'))
        if dd is not None:
            buf.append(dd)
    return buf


def fresh(r):
    return pdata.fresh_x(ws, tab, n_train, seed * 50 + r)


inc = ppol.incidence(pw.Inst(tab, np.zeros(m, dtype=np.int64)))
torch.manual_seed(seed)
net0 = ppol.GNN(32, 3)
nets = {'0': copy.deepcopy(net0)}
net, buf, buf0 = copy.deepcopy(net0), [], None
for r in range(R):
    new = amplify(ppol.NetPolicy(net, order='greedy1'), fresh(r), seed * 1000 + r)
    buf += new
    if r == 0:
        buf0 = list(new)
    ploop.train(net, buf, steps, bs, lr, seed * 77 + r, inc)
    nets[str(r + 1)] = copy.deepcopy(net)
    say('round %d trained, buffer %d states' % (r, sum(len(b[0]) for b in buf)))
net1, buf1 = copy.deepcopy(net0), list(buf0)
for r in range(1, R):
    buf1 += amplify(ppol.NetPolicy(net0, order='greedy1'), fresh(r), seed * 1000 + r)
ploop.train(net1, buf1, steps * R, bs, lr, seed * 77 + 99, inc)
nets['oneshot'] = net1
say('one-shot control trained on %d states' % sum(len(b[0]) for b in buf1))
torch.save({k: v.state_dict() for k, v in nets.items()}, OUT + 'c5_w%d_s%d.pt' % (ws, seed))

# fresh instances and my enumeration
_, xm = W.sample(50000, ws * 1000 + 1)
seen = set(pdata.pack(xm).tolist())
for r in range(R):
    seen |= set(pdata.pack(fresh(r)).tolist())
_, xs = W.sample(4 * n_inst, ws * 1000 + 942)
X = np.array([x for x, k in zip(xs, pdata.pack(xs).tolist()) if k not in seen][:n_inst])
z = np.arange(2 ** n, dtype=np.int64)
bits = ((z[:, None] >> np.arange(n)[None]) & 1).astype(np.float64)
cfg = np.zeros((2 ** n, m), dtype=np.int64)
for t in range(3):
    cfg += (((z[:, None] >> tab['checks'][None, :, t]) & 1) << t)


def exact(x):
    lr = bits @ (tab['loga'][:, 1] - tab['loga'][:, 0]) + tab['loga'][:, 0].sum()
    for c in range(m):
        lr = lr + tab['logf'][c, :, int(x[c])][cfg[:, c]]
    w = np.exp(lr - lr.max())
    return (w @ bits) / w.sum()


PS = [exact(x) for x in X]
insts = [pw.Inst(tab, x) for x in X]
res = {}
for k in ['0', '1', '2', '3', '4', '5', 'oneshot']:
    e_is, w_aud, e_aud, out = [], [], [], 0
    for i, inst in enumerate(insts):
        rng = np.random.default_rng(606 * (i + 1) + seed)
        logw, bel, _ = ptree.run_is(inst, ppol.NetPolicy(nets[k], order='conf'), 64, rng, eps)
        w = np.exp(logw - logw.max())
        e_is.append(float(np.abs((w[:, None] * bel).sum(0) / w.sum() - PS[i]).mean()))
        T = ploop.new_tree(inst, 2 * 256 + 8)
        ptree.grow(T, ppol.NetPolicy(nets[k], order='greedy1'), 256, rng, rho=rho, eps=eps)
        bk = ptree.books(T)
        au = ptree.audit(T, bk, n_draw, rng)
        out += int(((PS[i] < bk['lo'] - 1e-9) | (PS[i] > bk['hi'] + 1e-9)).sum())
        w_aud.append(float((au['hi'] - au['lo']).mean()))
        e_aud.append(float(np.abs(np.clip(au.get('pt', bk['pt']), au['lo'], au['hi']) - PS[i]).mean()))
    res[k] = dict(E_IS64=float(np.mean(e_is)), se=float(np.std(e_is) / np.sqrt(len(e_is))), W_aud256=float(np.mean(w_aud)),
                  E_aud256=float(np.mean(e_aud)), outside=out)
    say('round %-7s E_IS(64) %.4f (s.e. %.4f) | audit width at 256 openings %.3f | audit point error %.4f | exact answers outside the certain interval %d'
        % (k, res[k]['E_IS64'], res[k]['se'], res[k]['W_aud256'], res[k]['E_aud256'], out))
    json.dump(res, open(OUT + 'c5_w%d_s%d.json' % (ws, seed), 'w'))
say('H5: final/round0 = %.3f, final/one-shot = %.3f | H4: W_aud final/round0 = %.3f, final/one-shot = %.3f'
    % (res['5']['E_IS64'] / res['0']['E_IS64'], res['5']['E_IS64'] / res['oneshot']['E_IS64'],
       res['5']['W_aud256'] / res['0']['W_aud256'], res['5']['W_aud256'] / res['oneshot']['W_aud256']))
