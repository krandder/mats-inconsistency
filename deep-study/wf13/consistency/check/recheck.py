"""recheck.py WORLD : checker's independent recomputation of the note's tables from the arms' stored test forecasts.
Own exact posterior (numpy enumeration over 2^20, written from world.py's definition; wf7's exact.py and the work's
cexact.py / creport.py are not imported). Compares own exact numbers with out/wW/exact.npz, then recomputes Tables
A-E and the mechanism numbers. Output: check/out/recheck_wW.json and printed tables."""
import glob, json, os, sys
import numpy as np, torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/deep-study/wf7/probworld'); sys.dont_write_bytecode = True
from world import World
R = '/home/agent/workspaces/mats/work/deep-study/wf13/consistency'
w = int(sys.argv[1]); W = World(w)
torch.set_num_threads(2)


def exact_post(x, mask, W, bs=32):
    """P*(x_c = 1 | x_S) for all 30 checks, and log P*(x_S). x, mask (N,30)."""
    z = np.arange(2 ** 20)
    bits = ((z[:, None] >> np.arange(20)[None, :]) & 1).astype(np.float64)
    par = torch.tensor((bits[:, W.checks].sum(-1) % 2))                     # (2^20, 30) float64
    lp = torch.tensor(bits @ np.log(W.prior / (1 - W.prior)) + np.log(1 - W.prior).sum())
    la, le = np.log(1 - W.eps), np.log(W.eps)
    ps, lz = [], []
    x, mask = torch.tensor(x.astype(np.float64)), torch.tensor(mask.astype(np.float64))
    for i in range(0, len(x), bs):
        a1, a0 = (mask[i:i + bs] * x[i:i + bs]).T, (mask[i:i + bs] * (1 - x[i:i + bs])).T
        agree = par @ a1 + (1 - par) @ a0
        lw = lp[:, None] + agree * la + (mask[i:i + bs].sum(1)[None, :] - agree) * le
        lz.append(torch.logsumexp(lw, 0))
        pp = torch.softmax(lw, 0).T @ par
        ps.append(pp * (1 - W.eps) + (1 - pp) * W.eps)
    return torch.cat(ps).numpy(), torch.cat(lz).numpy()


def ld(name):
    fs = sorted(glob.glob('%s/exactwork/w%d/%s*.pt' % (R, w, name)))
    ps = [torch.load(f) for f in fs]
    return {k: torch.cat([p[k] for p in ps], 0).numpy() for k in ps[0]}


T1, T2, T3 = ld('t1'), ld('t2'), ld('t3')
E = np.load('%s/out/w%d/exact.npz' % (R, w))
cache = '%s/check/out/myexact_w%d.npz' % (R, w)
if os.path.exists(cache):
    X = dict(np.load(cache))
else:
    X = {}
    for nm, T in (('t1', T1), ('t2', T2)):                     # t3 needs no exact number for V_ord
        X[nm + '_ps'], X[nm + '_lz'] = exact_post(T['x'], T['mask'], W)
    np.savez_compressed(cache, **X)
rep = dict(exact_maxdiff={nm: float(np.abs(X[nm + '_ps'] - E[nm + '_pstar']).max()) for nm in ('t1', 't2') if nm + '_ps' in X},
           same_testsets=bool((E['t1_x'] == T1['x']).all() and (E['t1_mask'] == T1['mask']).all() and (E['t2_X'] == T2['X']).all()))
if 't3_ps' in X:
    rep['exact_maxdiff']['t3'] = float(np.abs(X['t3_ps'] - E['t3_pstar']).max())
print('own exact vs exact.npz:', rep)

sig = lambda z: 1 / (1 + np.exp(-z.astype(np.float64)))
lsig = lambda z: -np.logaddexp(0.0, -z.astype(np.float64))
def kl(p, q):
    q = np.clip(q, 1e-6, 1 - 1e-6); p = np.clip(p, 1e-12, 1 - 1e-12)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))
mk = ~T1['mask']; size = T1['mask'].sum(1); ps1 = X['t1_ps']; xr = T1['x'].astype(np.float64)
n2 = len(T2['X']); ar = np.arange(n2); Xc, Yc = T2['X'], T2['Y']; ps2 = X['t2_ps']
px = ps2[ar, Xc]; py0, py1 = ps2[n2 + ar, Yc], ps2[2 * n2 + ar, Yc]
J = np.stack([np.stack([(1 - px) * (1 - py0), (1 - px) * py0], -1), np.stack([px * (1 - py1), px * py1], -1)], 1)   # [x,y]
py = ps2[ar, Yc]; px0, px1 = ps2[3 * n2 + ar, Xc], ps2[4 * n2 + ar, Xc]                # exact joint the other way round
J2x = np.stack([np.stack([(1 - py) * (1 - px0), py * (1 - px1)], -1), np.stack([(1 - py) * px0, py * px1], -1)], 1)
rep['exact_joint_orders_agree'] = float(np.abs(J - J2x).max())
perm = T3['perm']; val = np.take_along_axis(np.broadcast_to(T3['xvec'][:, None, :].astype(np.int64), perm.shape), perm, 2)


def joints(z):
    l = lambda a: np.stack([lsig(-a), lsig(a)], -1)
    lx, ly = l(z[:, 0]), l(z[:, 1])
    return lx[:, :, None] + np.stack([l(z[:, 2]), l(z[:, 3])], 1), ly[:, None, :] + np.stack([l(z[:, 4]), l(z[:, 5])], 2)


def metrics(A):
    o = {}
    for K in (1, 4):
        p = np.clip(sig(A['t1_K%d' % K]), 1e-6, 1 - 1e-6); k = kl(ps1, p)
        o['exc%d' % K] = float(k[mk].mean()); o['ctx%d' % K] = (k * mk).sum(1) / np.maximum(mk.sum(1), 1)
        o['real%d' % K] = float(-(xr * np.log(p) + (1 - xr) * np.log(1 - p))[mk].mean())
        for nm, lo, hi in (('s', 0, 9), ('m', 10, 19), ('l', 20, 29)):
            o['exc%d%s' % (K, nm)] = float(k[mk & ((size >= lo) & (size <= hi))[:, None]].mean())
        if 't2_K%d_r0' % K not in A:
            continue
        l1, l2 = joints(A['t2_K%d_r0' % K]); m1, m2 = joints(A['t2_K%d_r1' % K])
        o['D%d' % K] = float((0.5 * (np.exp(l1) + np.exp(l2)) * (l1 - l2) ** 2).sum((1, 2)).mean())
        o['Dn%d' % K] = float(0.5 * ((0.5 * (np.exp(l1) + np.exp(m1)) * (l1 - m1) ** 2).sum((1, 2)).mean() + (0.5 * (np.exp(l2) + np.exp(m2)) * (l2 - m2) ** 2).sum((1, 2)).mean()))
        o['kl%d' % K] = np.stack([(J * (np.log(J) - l1)).sum((1, 2)), (J * (np.log(J) - l2)).sum((1, 2))], 1)
        L = lsig(A['t3_K%d_r0' % K] * (2 * val - 1)).sum(-1); L1 = lsig(A['t3_K%d_r1' % K] * (2 * val - 1)).sum(-1)
        o['V%d' % K] = float(L.var(1, ddof=1).mean()); o['Vn%d' % K] = float(((L - L1) ** 2 / 2).mean())
        o['draw%d' % K] = float(np.abs(sig(A['t2_K%d_r0' % K]) - sig(A['t2_K%d_r1' % K])).mean())
    o['exc'] = (o['exc1'] + o['exc4']) / 2; o['ctx'] = (o['ctx1'] + o['ctx4']) / 2; o['real'] = (o['real1'] + o['real4']) / 2
    return o


def boot(v, nb=2000, seed=12345):
    rng = np.random.default_rng(seed)
    m = v[rng.integers(0, len(v), size=(nb, len(v)))].mean(1)
    return float(v.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def loaddir(d):
    out = {}
    for f in sorted(glob.glob(d + '/*.npz')):
        tag = os.path.basename(f)[:-4]
        if tag == 'exact' or tag.startswith('direct'):
            if tag.startswith('direct'):
                out[tag] = dict(exc=float(kl(ps1, np.clip(np.load(f)['t1'].astype(np.float64), 1e-6, 1 - 1e-6))[mk].mean()))
            continue
        out[tag] = metrics(dict(np.load(f)))
    return out


M = loaddir('%s/out/w%d' % (R, w))
NS = [64, 256, 1024, 4096, 20480]
res = dict(rep=rep, A={}, B={}, C={}, D={}, mech={}, E={})
one = M['fixed-one_n0_d0_r0']
res['refs'] = dict(one=one['exc'], one_l=one['exc4l'], half=float(kl(ps1, np.full_like(ps1, 0.5))[mk].mean()), exu1=M['fixed-exu_n0_d0_r0']['exc1'], exu4=M['fixed-exu_n0_d0_r0']['exc4'],
                   one_D=one['D4'], one_V=one['V4'], real_truth=float(-(xr * np.log(ps1) + (1 - xr) * np.log(1 - ps1))[mk].mean()), n_queries=int(mk.sum()),
                   exu_D=[M['fixed-exu_n0_d0_r0'][k] for k in ('D1', 'Dn1', 'D4', 'Dn4', 'V1', 'Vn1', 'V4', 'Vn4')])
ARMS = ['data', 'i@1', 'i@10', 'iio@1', 'iio@10', 'iis@0.1', 'iis@1', 'iis@10', 'iii@1', 'iiisym@10', 'synth@1']
for a in ARMS:
    for n in NS:
        ds = [d for d in (0, 1) if '%s_n%d_d%d_r0' % (a, n, d) in M and 'data_n%d_d%d_r0' % (n, d) in M]
        if not ds:
            continue
        ea = np.mean([M['%s_n%d_d%d_r0' % (a, n, d)]['exc'] for d in ds]); eb = np.mean([M['data_n%d_d%d_r0' % (n, d)]['exc'] for d in ds])
        cis = [boot(M['%s_n%d_d%d_r0' % (a, n, d)]['ctx'] - M['data_n%d_d%d_r0' % (n, d)]['ctx']) for d in ds]
        star = all(c[1] > 0 for c in cis) or all(c[2] < 0 for c in cis)
        res['A']['%s n=%d' % (a, n)] = round(float(ea), 4)
        res['B']['%s n=%d' % (a, n)] = ('%+.0f%%' % (100 * (1 - ea / eb))) + ('*' if star and a != 'data' else '') + (' [%d runs]' % len(ds))
        m0 = M['%s_n%d_d0_r0' % (a, n)]
        res['C']['%s n=%d' % (a, n)] = [round(m0[k], 3) for k in ('D1', 'Dn1', 'D4', 'Dn4', 'V1', 'Vn1', 'V4', 'Vn4')]
        res['mech']['%s n=%d' % (a, n)] = dict(draw=round(m0['draw4'], 4), exc_large=round(m0['exc4l'], 4), exc_large_K1=round(m0['exc1l'], 4))
        if a != 'data' and 'data_n%d_d0_r1' % n in M:
            b, aa, pl = M['data_n%d_d0_r0' % n]['kl4'], m0['kl4'], M['data_n%d_d0_r1' % n]['kl4']
            bt = b.argmin(1); i = np.arange(len(b))
            adj = (aa[i, bt] - b[i, bt]) - (pl[i, bt] - b[i, bt])
            c = boot(adj)
            res['D']['%s n=%d' % (a, n)] = ('%+.3f' % c[0]) + ('*' if c[1] > 0 or c[2] < 0 else '') + ' gap %.3f' % float((b[i, 1 - bt] - b[i, bt]).mean())
for a in ('i@1', 'iio@1', 'iis@1', 'synth@1', 'iii@1'):
    for r in (0, 1):
        t = '%s_n0_d0_r%d' % (a, r)
        if t in M:
            res['A'][t] = round(M[t]['exc'], 4)
res['direct'] = {n: round(float(np.mean([M['direct_n%d_d%d' % (n, d)]['exc'] for d in (0, 1) if 'direct_n%d_d%d' % (n, d) in M])), 4) for n in NS}
res['real'] = {'%s n=%d' % (a, n): round(float(np.mean([M['%s_n%d_d%d_r0' % (a, n, d)]['real'] for d in (0, 1) if '%s_n%d_d%d_r0' % (a, n, d) in M])), 4)
               for a in ('data', 'i@10', 'iis@1', 'synth@1') for n in (64, 20480)}
res['init_noise'] = {n: round(abs(M['data_n%d_d0_r0' % n]['exc'] - M['data_n%d_d0_r1' % n]['exc']), 4) for n in NS}
res['amp_gain_K'] = {n: [round(float(np.mean([M['data_n%d_d%d_r0' % (n, d)]['exc%d' % K] for d in (0, 1)])), 4) for K in (1, 4)] for n in NS}
for sub in ('ph25', 'phdb'):
    d = '%s/out/w%d_%s' % (R, w, sub)
    if os.path.isdir(d):
        P = loaddir(d)
        res['E' if sub == 'ph25' else 'F'] = {t: [round(P[t][k], 3) for k in ('exc4', 'exc4s', 'exc4m', 'exc4l')] + [round(P[t]['exc'], 4)] for t in P}
json.dump(res, open('%s/check/out/recheck_w%d.json' % (R, w), 'w'), indent=1)
for k in ('refs', 'A', 'B', 'C', 'D', 'mech', 'E', 'F', 'direct', 'real', 'init_noise', 'amp_gain_K'):
    print('==', k)
    v = res.get(k, {})
    for kk in v:
        print('  ', kk, v[kk])
