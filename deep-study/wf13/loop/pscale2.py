"""pscale2.py -- POST HOC for item (e). World 201, all 40 assumptions hidden, the specification alone (hand order,
rho = 1). Two things the registered run could not show:
 (1) a cross-check of the audit where no exact answer exists: the audit interval at 4,096 openings against the CERTAIN
     interval of the same tree at 16,384 openings (disjoint = a proven miss) and against the audit point there;
 (2) fewer observed checks: only the first M_OBS of the 60 checks are observed (the others are summed out, exactly,
     because every check factor sums to one), so each assumption is touched by fewer checks.
Usage: pscale2.py M_OBS N_INST -> out/pscale2_m{M_OBS}.json"""
import json, sys, time
import numpy as np
sys.dont_write_bytecode = True
import pdata, pw, ptree, ppol, ploop

HERE = pdata.HERE
mobs, n_inst = int(sys.argv[1]), int(sys.argv[2])
marks = [1024, 4096, 16384]
tab = pdata.load_tab(201)
x, h = pdata.test_x(201, tab, n_inst)
sub = dict(loga=tab['loga'], logf=tab['logf'][:mobs], checks=tab['checks'][:mobs])
pol = ppol.Hand('greedy1', 'books')
rows = []
for i in range(n_inst):
    inst = pw.Inst(sub, x[i][:mobs])
    rng = np.random.default_rng(31 * i + 7)
    T = ploop.new_tree(inst, 2 * max(marks) + 8)
    rec, keep = {}, {}
    for mk in marks:
        ptree.grow(T, pol, mk, rng, rho=1.0, eps=0.02)
        bk = ptree.books(T)
        a1, a2 = ptree.audit(T, bk, 4096, rng), ptree.audit(T, bk, 65536, rng)
        keep[mk] = (bk, a1, a2)
        rec[mk] = dict(w_cert=float((bk['hi'] - bk['lo']).mean()), w_aud4k=float((a1['hi'] - a1['lo']).mean()),
                       w_aud64k=float((a2['hi'] - a2['lo']).mean()), n_closed=bk['n_closed'], n_open=bk['n_open'],
                       unfound_ub=float(bk['unfound_ub']), opened=T.opened,
                       log10_UF_over_found=float(np.log10(bk['UF'] / bk['Z'])) if bk['Z'] > 0 and bk['UF'] > 0 else None)
    bkL, _, aL = keep[marks[-1]]
    ref = np.clip(aL.get('pt', bkL['pt']), aL['lo'], aL['hi'])
    for mk in marks[:-1]:
        for key, a in (('4k', keep[mk][1]), ('64k', keep[mk][2])):
            rec[mk]['proven_miss_' + key] = int(((a['hi'] < bkL['lo'] - 1e-9) | (a['lo'] > bkL['hi'] + 1e-9)).sum())
            rec[mk]['ref_outside_' + key] = int(((ref < a['lo'] - 1e-9) | (ref > a['hi'] + 1e-9)).sum())
    rec['ref_width'] = float((aL['hi'] - aL['lo']).mean())
    rows.append(rec)
    print('instance', i, {mk: (round(rec[mk]['w_cert'], 3), round(rec[mk]['w_aud4k'], 3), round(rec[mk]['w_aud64k'], 3)) for mk in marks},
          time.strftime('%H:%M:%S'), flush=True)
    json.dump(dict(mobs=mobs, marks=marks, rows=rows), open(HERE + 'out/pscale2_m%d.json' % mobs, 'w'))
print('done')
