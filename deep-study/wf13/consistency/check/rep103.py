"""rep103.py WORLD DATADIR OUTDIR T1NAME T2NAME LEVELS : metrics of the checker's fresh-world runs (tr.py).
EVALUATION ONLY. Own exact posterior by enumeration (same function as recheck.py)."""
import glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/deep-study/wf7/probworld'); sys.dont_write_bytecode = True
from world import World
w, ddir, odir, t1n, t2n = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
LEV = [int(v) for v in sys.argv[6].split(',')]
W = World(w); torch.set_num_threads(2)


def exact_post(x, mask, bs=32):
    z = np.arange(2 ** 20)
    bits = ((z[:, None] >> np.arange(20)[None, :]) & 1).astype(np.float64)
    par = torch.tensor((bits[:, W.checks].sum(-1) % 2))
    lp = torch.tensor(bits @ np.log(W.prior / (1 - W.prior)) + np.log(1 - W.prior).sum())
    la, le = np.log(1 - W.eps), np.log(W.eps)
    ps = []
    x, mask = torch.tensor(x.astype(np.float64)), torch.tensor(mask.astype(np.float64))
    for i in range(0, len(x), bs):
        a1, a0 = (mask[i:i + bs] * x[i:i + bs]).T, (mask[i:i + bs] * (1 - x[i:i + bs])).T
        agree = par @ a1 + (1 - par) @ a0
        lw = lp[:, None] + agree * la + (mask[i:i + bs].sum(1)[None, :] - agree) * le
        pp = torch.softmax(lw, 0).T @ par
        ps.append(pp * (1 - W.eps) + (1 - pp) * W.eps)
    return torch.cat(ps).numpy()


T1 = {k: v.numpy() for k, v in torch.load('%s/%s.pt' % (ddir, t1n)).items()}
T2 = {k: v.numpy() for k, v in torch.load('%s/%s.pt' % (ddir, t2n)).items()}
cache = '%s/myexact_%s_%s.npz' % (odir, t1n, t2n)
if os.path.exists(cache):
    X = dict(np.load(cache))
else:
    X = dict(t1=exact_post(T1['x'], T1['mask']), t2=exact_post(T2['x'], T2['mask']))
    np.savez_compressed(cache, **X)
sig = lambda z: 1 / (1 + np.exp(-z.astype(np.float64)))
lsig = lambda z: -np.logaddexp(0.0, -z.astype(np.float64))
def kl(p, q):
    q = np.clip(q, 1e-6, 1 - 1e-6); p = np.clip(p, 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
mk = ~T1['mask']; size = T1['mask'].sum(1); ps1 = X['t1']; xr = T1['x'].astype(np.float64)
n2 = len(T2['X']); ar = np.arange(n2); Xc, Yc = T2['X'], T2['Y']; ps2 = X['t2']
px = ps2[ar, Xc]; py0, py1 = ps2[n2 + ar, Yc], ps2[2 * n2 + ar, Yc]
J = np.stack([np.stack([(1 - px) * (1 - py0), (1 - px) * py0], -1), np.stack([px * (1 - py1), px * py1], -1)], 1)
BINS = (('s', 0, 9), ('m', 10, 19), ('l', 20, 29))


def joints(z):
    l = lambda a: np.stack([lsig(-a), lsig(a)], -1)
    return l(z[:, 0])[:, :, None] + np.stack([l(z[:, 2]), l(z[:, 3])], 1), l(z[:, 1])[:, None, :] + np.stack([l(z[:, 4]), l(z[:, 5])], 2)


def t1m(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); k = kl(ps1, p)
    o = dict(exc=float(k[mk].mean()), ctx=(k * mk).sum(1) / np.maximum(mk.sum(1), 1), real=float(-(xr * np.log(p) + (1 - xr) * np.log(1 - p))[mk].mean()),
             to_one={nm: float(np.abs(p - T1['p_one'])[mk & ((size >= lo) & (size <= hi))[:, None]].mean()) for nm, lo, hi in BINS})
    o.update({'exc_' + nm: float(k[mk & ((size >= lo) & (size <= hi))[:, None]].mean()) for nm, lo, hi in BINS})
    return o


def metrics(A):
    o = {}
    for K in LEV:
        o[K] = t1m(sig(A['t1_K%d' % K]))
        l1, l2 = joints(A['t2_K%d_r0' % K])
        o[K]['D'] = float((0.5 * (np.exp(l1) + np.exp(l2)) * (l1 - l2) ** 2).sum((1, 2)).mean())
        o[K]['kl'] = np.stack([(J * (np.log(J) - l1)).sum((1, 2)), (J * (np.log(J) - l2)).sum((1, 2))], 1)
        if 't2_K%d_r1' % K in A:
            m1, m2 = joints(A['t2_K%d_r1' % K])
            o[K]['Dn'] = float(0.5 * ((0.5 * (np.exp(l1) + np.exp(m1)) * (l1 - m1) ** 2).sum((1, 2)).mean() + (0.5 * (np.exp(l2) + np.exp(m2)) * (l2 - m2) ** 2).sum((1, 2)).mean()))
            o[K]['draw'] = float(np.abs(sig(A['t2_K%d_r0' % K]) - sig(A['t2_K%d_r1' % K])).mean())
            # cross-order mismatch of the MEAN joint over the two draws minus nothing: an unbiased estimate of the squared expected mismatch
            o[K]['Dx'] = float((0.25 * (np.exp(l1) + np.exp(l2) + np.exp(m1) + np.exp(m2)) * (l1 - l2) * (m1 - m2)).sum((1, 2)).mean())
    o['exc'] = float(np.mean([o[K]['exc'] for K in LEV])); o['ctx'] = np.mean([o[K]['ctx'] for K in LEV], 0)
    return o


def boot(v, nb=2000, seed=777):
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), size=(nb, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


M = {os.path.basename(f)[:-4]: metrics(dict(np.load(f))) for f in sorted(glob.glob(odir + '/*_r[0-9]*.npz'))}
Kt = max(LEV)
one = t1m(T1['p_one']); half = t1m(np.full_like(ps1, 0.5))
# one pass of M on the triples: its own pair mismatch
z1 = np.log(T2['p_one'] / (1 - T2['p_one'])); zz = np.stack([z1[ar, Xc], z1[ar, Yc], z1[n2 + ar, Yc], z1[2 * n2 + ar, Yc], z1[3 * n2 + ar, Xc], z1[4 * n2 + ar, Xc]], 1)
l1, l2 = joints(zz)
print('refs: half %.4f  one pass %.4f (0-9 %.3f, 10-19 %.3f, 20-29 %.3f)  one-pass pair mismatch D %.4f  queries %d  truth real-loss %.4f' % (
    half['exc'], one['exc'], one['exc_s'], one['exc_m'], one['exc_l'], float((0.5 * (np.exp(l1) + np.exp(l2)) * (l1 - l2) ** 2).sum((1, 2)).mean()), int(mk.sum()),
    float(-(xr * np.log(ps1) + (1 - xr) * np.log(1 - ps1))[mk].mean())))
res = {}
for tag in sorted(M):
    m = M[tag]
    arm, n, d, r = tag.split('_')[0], int(tag.split('_')[1][1:]), int(tag.split('_')[2][1:]), int(tag.split('_')[3][1:])
    ph = '_ph' in tag
    base = 'data_n%d_d%d_r0%s' % (n, d, tag[tag.index('_ph'):] if ph else '')
    line = dict(exc=round(m['exc'], 4), excK={K: round(m[K]['exc'], 4) for K in LEV}, bins=[round(m[Kt]['exc_' + b], 3) for b in 'sml'],
                D={K: round(m[K]['D'], 3) for K in LEV}, Dn={K: round(m[K].get('Dn', float('nan')), 3) for K in LEV}, Dx={K: round(m[K].get('Dx', float('nan')), 4) for K in LEV},
                draw={K: round(m[K].get('draw', float('nan')), 4) for K in LEV}, to_one=[round(m[Kt]['to_one'][b], 3) for b in 'sml'])
    if base in M and base != tag:
        c = boot(m['ctx'] - M[base]['ctx'])
        line['gain'] = '%+.0f%%%s' % (100 * (1 - m['exc'] / M[base]['exc']), '*' if c[1] > 0 or c[2] < 0 else '')
        plac = 'data_n%d_d%d_r1' % (n, d)
        if plac in M and not ph:
            for K in LEV[1:]:
                b, a, pl = M[base][K]['kl'], m[K]['kl'], M[plac][K]['kl']
                bt = b.argmin(1); i = np.arange(len(b))
                cc = boot((a[i, bt] - b[i, bt]) - (pl[i, bt] - b[i, bt]))
                line['drag_K%d' % K] = '%+.3f%s' % (cc[0], '*' if cc[1] > 0 or cc[2] < 0 else '')
    res[tag] = line
    print(tag, json.dumps(line))
json.dump(res, open(odir + '/report.json', 'w'), indent=0)
