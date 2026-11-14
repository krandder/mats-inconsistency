"""pscale.py -- item (e): where the interval stays wide. World 201 (40 assumptions, 60 checks, wf7's trained M).
The number of HIDDEN assumptions is 20, 30 or 40: the others are shown (real values of the instance) and enter the
root state as evidence. Arms: the specification alone (hand order, rho = 1), a rollout policy distilled in ANOTHER
world (20 assumptions; the network reads the graph, so it can be moved), and a random policy.
Exact answers exist for 20 hidden (enumeration, scoring only). Usage: pscale.py NET_CKPT N_INST [WORLD MARKS] -> out/pscale.json (world 201) or out/pscale_dev_w{W}.json"""
import json, sys, time
import numpy as np
import torch
sys.dont_write_bytecode = True
import pdata, pw, ptree, ppol, ploop, pexact

HERE = pdata.HERE
ck, n_inst = sys.argv[1], int(sys.argv[2])
ws = int(sys.argv[3]) if len(sys.argv) > 3 else 201
marks = [int(v) for v in sys.argv[4].split(',')] if len(sys.argv) > 4 else [64, 256, 1024, 4096, 16384]
OUT = 'out/pscale.json' if ws == 201 else 'out/pscale_dev_w%d.json' % ws
tab = pdata.load_tab(ws)
n = tab['loga'].shape[0]
x, h = pdata.test_x(ws, tab, n_inst)
sd = torch.load(ck, map_location='cpu')
net = ppol.GNN(32, 3)
net.load_state_dict(sd[sorted(k for k in sd if k != 'oneshot')[-1]])
net.eval()
torch.set_num_threads(2)
arms = [('spec_alone', ppol.Hand('greedy1', 'books'), 1.0), ('moved_net', ppol.NetPolicy(net, order='greedy1'), 0.5),
        ('random', ppol.Hand('random', 'uniform'), 0.5)]
res = dict(marks=marks, n_inst=n_inst, hidden={})
for hid in (n // 2, 3 * n // 4, n):
    out = {a[0]: [] for a in arms}
    for i in range(n_inst):
        rs = np.random.default_rng(555 + i)
        root = np.full(n, 2, dtype=np.int8)
        shown = rs.choice(n, size=n - hid, replace=False)
        root[shown] = h[i][shown]
        inst = pw.Inst(tab, x[i])
        op = root == 2
        pstar = lz = None
        if hid <= 20:
            pstar, lz = pexact.EnumSub(tab['checks'], root).answer(tab['loga'], tab['logf'], x[i])
        for an, pol, rho in arms:
            rng = np.random.default_rng(31 * i + 7)
            T = ploop.new_tree(inst, 2 * max(marks) + 8, root)
            rows, prev = [], None
            for mk in marks:
                t0 = time.time()
                ptree.grow(T, pol, mk, rng, rho=rho, eps=0.02)
                bk = ptree.books(T)
                a1 = ptree.audit(T, bk, 4096, rng)
                a2 = ptree.audit(T, bk, 65536, rng)
                r = dict(w_cert=float((bk['hi'] - bk['lo'])[op].mean()), w_aud4k=float((a1['hi'] - a1['lo'])[op].mean()),
                         w_aud64k=float((a2['hi'] - a2['lo'])[op].mean()), unfound_ub=float(bk['unfound_ub']),
                         log10_UF_over_found=float(np.log10(max(bk['UF'], 1e-300) / max(bk['Z'], 1e-300))) if bk['Z'] > 0 else None,
                         n_closed=bk['n_closed'], opened=T.opened,
                         nested=bool(prev is None or ((bk['lo'] >= prev[0] - 1e-12).all() and (bk['hi'] <= prev[1] + 1e-12).all())))
                prev = (bk['lo'], bk['hi'])
                if pstar is not None:
                    ps = pstar[op]
                    r.update(viol_cert=int(((ps < bk['lo'][op] - 1e-9) | (ps > bk['hi'][op] + 1e-9)).sum()),
                             viol_Z=int(not (bk['logZ_lo'] - 1e-7 <= lz <= bk['logZ_hi'] + 1e-7)),
                             miss4k=int(((ps < a1['lo'][op] - 1e-9) | (ps > a1['hi'][op] + 1e-9)).sum()),
                             miss64k=int(((ps < a2['lo'][op] - 1e-9) | (ps > a2['hi'][op] + 1e-9)).sum()),
                             err_aud64k=float(np.abs(np.clip(a2.get('pt', bk['pt']), a2['lo'], a2['hi'])[op] - ps).mean()))
                rows.append(r)
            out[an].append(rows)
        print('hidden', hid, 'instance', i, 'done', time.strftime('%H:%M:%S'), flush=True)
        res['hidden'][str(hid)] = out
        json.dump(res, open(HERE + OUT, 'w'))
print('done')
