"""pilot1.py -- DEV WORLDS ONLY (7, 311): sanity of the books and a first look at interval widths. Not a registered run."""
import sys, time
import numpy as np
import pdata, pw, ptree, ppol, pexact

ws = int(sys.argv[1]); n_inst = int(sys.argv[2]); marks = [int(v) for v in sys.argv[3].split(',')]
tab = pdata.load_tab(ws)
x, h = pdata.test_x(ws, tab, n_inst)
EN = pexact.Enum(tab['checks'], tab['loga'].shape[0])
arms = [('complete,rho1', ppol.Hand('complete'), 1.0), ('conf,rho1', ppol.Hand('conf'), 1.0),
        ('complete,rho.5', ppol.Hand('complete'), 0.5), ('complete,rho0', ppol.Hand('complete'), 0.0),
        ('random-uniform,rho.5', ppol.Hand('random', 'uniform'), 0.5), ('random-uniform,rho0', ppol.Hand('random', 'uniform'), 0.0)]
res = {a[0]: {mk: [] for mk in marks} for a in arms}
viol = 0; t0 = time.time()
for i in range(n_inst):
    inst = pw.Inst(tab, x[i])
    lr = EN.log_R(tab['loga'], tab['logf'], x[i])
    pstar, lz = EN.marg(lr)
    for name, pol, rho in arms:
        rng = np.random.default_rng(1000 * i + 7)
        T = ptree.Tree(inst, 2 * max(marks) + 8)
        e = inst.ev(np.full((1, inst.n), 2, dtype=np.int8)); T.add(e, -1, 0.0)
        for mk in marks:
            ptree.grow(T, pol, mk, rng, rho=rho)
            bk = ptree.books(T)
            au = ptree.audit(T, bk, 4096, rng)
            bad = ((pstar < bk['lo'] - 1e-9) | (pstar > bk['hi'] + 1e-9)).sum()
            badz = not (bk['logZ_lo'] - 1e-9 <= lz <= bk['logZ_hi'] + 1e-9)
            viol += int(bad) + int(badz)
            res[name][mk].append([np.mean(bk['hi'] - bk['lo']), np.mean(au['hi'] - au['lo']),
                                  np.mean((pstar < au['lo'] - 1e-9) | (pstar > au['hi'] + 1e-9)),
                                  np.abs(bk['pt'] - pstar).mean(), bk['unfound_ub'], np.exp(np.log(max(bk['Z'],1e-300)) + T.logU[0] - lz),
                                  T.opened, np.abs(au['pt'] - pstar).mean()])
print('world', ws, 'instances', n_inst, 'violations of the certain interval:', viol, 'time %.0fs' % (time.time() - t0))
print('columns: certain width | audit width | audit miss rate | |point - exact| | unfound upper bound | found share (exact) | opened | |audit point - exact|')
for name, _, _ in arms:
    for mk in marks:
        r = np.array(res[name][mk]).mean(0)
        print('%-22s B=%5d  %.4f  %.4f  %.4f  %.4f  %.4f  %.4f  %6.0f  %.4f' % (name, mk, *r))
