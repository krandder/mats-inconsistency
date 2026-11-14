"""c6_scale.py -- checker's re-run of the scale claim on FRESH instances of world 201 (40 assumptions, 60 checks),
the specification alone (hand order, rho = 1), as in the author's pscale2.py. Mine: instances (my seed), counting, and
for 20 hidden assumptions my own enumeration. Usage: c6_scale.py N_INST"""
import json, sys, time
import numpy as np
sys.dont_write_bytecode = True
ROOT = '/home/agent/workspaces/mats/work/deep-study/'
sys.path.insert(0, ROOT + 'wf13/loop')
import pdata, pw, ptree, ppol, ploop          # noqa: E402

OUT = ROOT + 'wf13/loop/check/out/'
n_inst = int(sys.argv[1])
tab = pdata.load_tab(201)
n, m = tab['loga'].shape[0], tab['checks'].shape[0]
W = pdata.world(201, tab)
h, x = W.sample(n_inst, 201 * 1000 + 943)
pol = ppol.Hand('greedy1', 'books')
res = {}
# ---- 40 hidden: 60 checks, and only the first 30 checks observed
marks = [1024, 4096, 16384]
for mobs in (60, 30):
    sub = dict(loga=tab['loga'], logf=tab['logf'][:mobs], checks=tab['checks'][:mobs])
    rows = {mk: [] for mk in marks}
    cross = dict(pairs=0, disjoint=0, ref_outside=0, not_nested=0)
    for i in range(n_inst):
        inst = pw.Inst(sub, x[i][:mobs].astype(np.int64))
        rng = np.random.default_rng(1009 * (i + 1))
        T = ploop.new_tree(inst, 2 * max(marks) + 8)
        keep, prev = {}, None
        for mk in marks:
            ptree.grow(T, pol, mk, rng, rho=1.0, eps=0.02)
            bk = ptree.books(T)
            a1, a2 = ptree.audit(T, bk, 4096, rng), ptree.audit(T, bk, 65536, rng)
            rows[mk].append([float((bk['hi'] - bk['lo']).mean()), float((a1['hi'] - a1['lo']).mean()), float((a2['hi'] - a2['lo']).mean())])
            if prev is not None:
                cross['not_nested'] += int(((bk['lo'] < prev[0] - 1e-12) | (bk['hi'] > prev[1] + 1e-12)).sum())
            prev = (bk['lo'], bk['hi'])
            keep[mk] = (bk, a1, a2)
        bkL, _, aL = keep[marks[-1]]
        ref = np.clip(aL.get('pt', bkL['pt']), aL['lo'], aL['hi'])
        for mk in marks[:-1]:
            for a in keep[mk][1:]:
                cross['pairs'] += n
                cross['disjoint'] += int(((a['hi'] < bkL['lo'] - 1e-9) | (a['lo'] > bkL['hi'] + 1e-9)).sum())
                cross['ref_outside'] += int(((ref < a['lo'] - 1e-9) | (ref > a['hi'] + 1e-9)).sum())
        print('checks %d instance %d %s %s' % (mobs, i, {mk: [round(v, 3) for v in rows[mk][-1]] for mk in marks}, time.strftime('%H:%M:%S')), flush=True)
    med = {mk: np.median(np.array(rows[mk]), 0).round(4).tolist() for mk in marks}
    rng_ = {mk: [np.array(rows[mk]).min(0).round(3).tolist(), np.array(rows[mk]).max(0).round(3).tolist()] for mk in marks}
    res['hidden40_checks%d' % mobs] = dict(median_cert_aud4k_aud64k=med, min_max=rng_, cross=cross)
    print('40 hidden, %d checks: median width (certain, audit 4,096 draws, audit 65,536 draws) %s | min/max %s | cross-check %s'
          % (mobs, med, rng_, cross), flush=True)
    json.dump(res, open(OUT + 'c6_scale.json', 'w'))
# ---- 20 hidden (the other 20 shown): my enumeration over the hidden ones
out = dict(intervals=0, outside=0, aud_miss=0)
for i in range(min(n_inst, 6)):
    rs = np.random.default_rng(8800 + i)
    root = np.full(n, 2, dtype=np.int8)
    shown = rs.choice(n, size=n - 20, replace=False)
    root[shown] = h[i][shown]
    opn = np.flatnonzero(root == 2)
    z = np.arange(2 ** 20, dtype=np.int64)
    full = np.repeat(np.where(root == 2, 0, root).astype(np.int64)[None], 2 ** 20, 0)
    full[:, opn] = (z[:, None] >> np.arange(20)[None]) & 1
    lr = full @ (tab['loga'][:, 1] - tab['loga'][:, 0]) + tab['loga'][:, 0].sum()
    for c in range(m):
        cf = full[:, tab['checks'][c, 0]] + 2 * full[:, tab['checks'][c, 1]] + 4 * full[:, tab['checks'][c, 2]]
        lr = lr + tab['logf'][c, :, int(x[i][c])][cf]
    w = np.exp(lr - lr.max())
    ps = (w @ full) / w.sum()
    inst = pw.Inst(tab, x[i].astype(np.int64))
    rng = np.random.default_rng(2003 * (i + 1))
    T = ploop.new_tree(inst, 2 * 1024 + 8, root)
    for mk in (64, 256, 1024):
        ptree.grow(T, pol, mk, rng, rho=1.0, eps=0.02)
        bk = ptree.books(T)
        au = ptree.audit(T, bk, 4096, rng)
        out['intervals'] += 20
        out['outside'] += int(((ps < bk['lo'] - 1e-9) | (ps > bk['hi'] + 1e-9))[opn].sum())
        out['aud_miss'] += int(((ps < au['lo'] - 1e-9) | (ps > au['hi'] + 1e-9))[opn].sum())
res['hidden20'] = out
print('20 hidden: %s' % out, flush=True)
json.dump(res, open(OUT + 'c6_scale.json', 'w'))
print('done', flush=True)
