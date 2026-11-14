"""pilot2.py -- DEV WORLD ONLY: ceiling of a perfectly distilled value head (the exact conditional as a stand-in for
the policy's q; evaluation-only numbers used as a POLICY here, so this is a pilot and never a reported arm)."""
import sys, time
import numpy as np
import pdata, pw, ptree, ppol, pexact

ws = int(sys.argv[1]); n_inst = int(sys.argv[2]); marks = [int(v) for v in sys.argv[3].split(',')]
tab = pdata.load_tab(ws)
x, h = pdata.test_x(ws, tab, n_inst)
EN = pexact.Enum(tab['checks'], tab['loga'].shape[0])
cur = {}
fn = lambda inst, e: EN.cond(cur['lr'], e['A'])
arms = [('hand complete,rho1', ppol.Hand('complete'), 1.0), ('hand greedy1,rho1', ppol.Hand('greedy1'), 1.0),
        ('hand near,rho1', ppol.Hand('near'), 1.0),
        ('exactq conf,rho1', ppol.Table(fn, 'conf'), 1.0), ('exactq conf,rho.5', ppol.Table(fn, 'conf'), 0.5),
        ('exactq greedy1,rho.5', ppol.Table(fn, 'greedy1'), 0.5)]
res = {a[0]: {mk: [] for mk in marks} for a in arms}
t0 = time.time()
for i in range(n_inst):
    inst = pw.Inst(tab, x[i])
    cur['lr'] = EN.log_R(tab['loga'], tab['logf'], x[i])
    pstar, lz = EN.marg(cur['lr'])
    for name, pol, rho in arms:
        rng = np.random.default_rng(1000 * i + 7)
        T = ptree.Tree(inst, 2 * max(marks) + 8)
        T.add(inst.ev(np.full((1, inst.n), 2, dtype=np.int8)), -1, 0.0)
        for mk in marks:
            ptree.grow(T, pol, mk, rng, rho=rho)
            bk = ptree.books(T); au = ptree.audit(T, bk, 4096, rng)
            res[name][mk].append([np.mean(bk['hi'] - bk['lo']), np.mean(au['hi'] - au['lo']), np.abs(bk['pt'] - pstar).mean(),
                                  bk['unfound_ub'], np.exp(np.log(max(bk['Z'], 1e-300)) + T.logU[0] - lz), np.abs(au['pt'] - pstar).mean()])
    print('instance', i, 'done %.0fs' % (time.time() - t0), flush=True)
print('columns: certain width | audit width | |point - exact| | unfound upper bound | found share (exact) | |audit point - exact|')
for name, _, _ in arms:
    for mk in marks:
        print('%-24s B=%5d  ' % (name, mk) + '  '.join('%.4f' % v for v in np.array(res[name][mk]).mean(0)))
