"""Score PREREG Part 1 (P1.1-P1.20) from runs/p1a_*, p1b_*, p1c_*, p1chain_* -> results/part1_scores.json + tables.
Rules exactly as registered (PREREG.md Part 1, hash 21782e63); "median" = median over the 4 nets of a cell."""
import glob, json, math, os, sys
import numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/input-bits/lib')
import ib

D_ = '/home/agent/workspaces/mats/work/input-bits'
NS = [32, 64, 128, 256]
SEEDS = [4000, 4001, 4002, 4003]


def load(pat):
    out = {}
    for f in glob.glob(f'{D_}/runs/{pat}'):
        r = json.load(open(f))
        out[(r['n'], r.get('seed'))] = r
    return out


A = load('p1a_n*_s*.json'); B = load('p1b_n*_s*.json'); C = load('p1c_n*_s*.json')
CH = {r['n']: r for r in (json.load(open(f)) for f in glob.glob(f'{D_}/runs/p1chain_n*.json'))}


def J2(t):
    return 3 * math.sin(t) * math.cos(t) + (math.pi - t) * (1 + 2 * math.cos(t) ** 2)


def kern(Lmax):
    rho = 0.0; Lam = 0.0; b1 = 1.0; out = {}
    for l in range(1, Lmax + 1):
        th = math.acos(rho)
        Lam += 6 - (2 / math.pi) * J2(th)
        b1 *= (math.pi - th) / math.pi
        ham_l = th / math.pi
        rho = float(ib.kappa(rho))
        out[l] = dict(delta=1 - rho, Lambda=Lam, b1=b1, hamfrac=ham_l)
    return out


K = kern(2048)
med = lambda v: float(np.median(v)) if len(v) else float('nan')
S = {}
rows = []


def depthsA(n):
    return sorted({max(1, n // 8), n // 4, n // 2, n, 2 * n, 32} | ({4 * n} if n <= 64 else set()))


def cell(n, L, key, src='split'):
    v = []
    for s in SEEDS:
        r = A.get((n, s))
        if r and str(L) in r[src]:
            v.append(r[src][str(L)][key])
    return v


# P1.1
v_amp, v_dir, v_rad = cell(256, 32, 's_amp'), cell(256, 32, 's_dir'), cell(256, 32, 's_rad')
S['P1.1'] = dict(hit=(0.40 <= med(v_amp) <= 0.75) and (0.20 <= med(v_dir) <= 0.55) and med(v_rad) <= 0.06,
                 s_amp=v_amp, s_dir=v_dir, s_rad=v_rad)
# P1.2
ok_le, ok_ge, det = [], [], []
for n in [64, 128, 256]:
    for L in depthsA(n):
        rat = [a / b for a, b in zip(cell(n, L, 'V_dir'), cell(n, L, 'V_amp')) if b > 0]
        if not rat:
            continue
        pred = 4 * n * K[L]['delta'] / K[L]['Lambda']
        m = med(rat)
        if L <= n // 2:
            ok_le.append(0.5 <= m / pred <= 2.0)
        if L >= n:
            ok_ge.append(m < pred)
        det.append(dict(n=n, L=L, median_ratio=m, kernel=pred, q=m / pred))
S['P1.2'] = dict(hit=all(ok_le) and all(ok_ge), le_half=f'{sum(ok_le)}/{len(ok_le)}', ge_n=f'{sum(ok_ge)}/{len(ok_ge)}', cells=det)
# P1.3
sd32 = [med(cell(n, 32, 's_dir')) for n in NS]
sdh = [med(cell(n, n // 2, 's_dir')) for n in NS]
S['P1.3'] = dict(hit=all(np.diff(sd32) > 0) and all(np.diff(sdh) < 0), s_dir_L32=sd32, s_dir_half=sdh)
# P1.4
ok, det = [], []
for n in [64, 128, 256]:
    for L in depthsA(n):
        if L > n:
            continue
        v = [x for x in cell(n, L, 'logA_var') if x is not None]
        q = med(v) / (K[L]['Lambda'] / (4 * n))
        ok.append(0.6 <= q <= 2.0); det.append(dict(n=n, L=L, ratio=q))
S['P1.4'] = dict(hit=all(ok), n_ok=f'{sum(ok)}/{len(ok)}', cells=det)
# P1.5
ok, det = [], []
for n in [64, 128, 256]:
    if n not in CH:
        continue
    for L in depthsA(n):
        if L > n:
            continue
        Vd, Vt = cell(n, L, 'V_dir'), cell(n, L, 'V')
        pooled = sum(Vd) / sum(Vt)
        ch = CH[n]['chainA'][str(L)]
        q1 = ch['s_dir'] / pooled if pooled > 0 else float('inf')
        q2 = ch['V'] / np.mean(Vt)
        ok.append((1 / 1.5 <= q1 <= 1.5) and (1 / 1.5 <= q2 <= 1.5))
        det.append(dict(n=n, L=L, chain_s_dir=ch['s_dir'], pooled_s_dir=pooled, q_sdir=q1, q_V=q2))
S['P1.5'] = dict(hit=all(ok), n_ok=f'{sum(ok)}/{len(ok)}', cells=det)
# P1.6
ok, det = [], []
for n in NS:
    Tm = (4 * n / 5) * math.log(1 / (1e-3 * math.sqrt(0.4 * n)))
    for L in depthsA(n):
        if L > Tm / 2:
            continue
        for s in SEEDS:
            r = A.get((n, s))
            if not r:
                continue
            z = r['noise'][str(L)]
            c = (z['coupled_diff_var'] <= 0.01 and abs(z['sig']['s_amp'] - z['zero']['s_amp']) <= 0.01
                 and abs(z['sig']['s_dir'] - z['zero']['s_dir']) <= 0.01)
            ok.append(c)
            if not c:
                det.append(dict(n=n, L=L, seed=s, dvar=z['coupled_diff_var'], d_amp=z['sig']['s_amp'] - z['zero']['s_amp'],
                                d_dir=z['sig']['s_dir'] - z['zero']['s_dir']))
S['P1.6'] = dict(hit=all(ok), n_ok=f'{sum(ok)}/{len(ok)}', failures=det)


# bits
def layer_med(n, key):
    per = []
    for s in SEEDS:
        r = A.get((n, s))
        if r:
            per.append([x[key] for x in r['layers']])
    if not per:
        return None
    m = min(len(p) for p in per)
    return np.median(np.array([p[:m] for p in per]), 0)


ok7a, ok7b, det7 = [], [], []
for n in NS:
    hm = layer_med(n, 'hamming')
    if hm is None:
        continue
    for l in range(2, n // 4 + 1):
        pred = n * K[l]['hamfrac']
        ok7a.append(abs(hm[l - 1] / pred - 1) <= 0.25)
        if abs(hm[l - 1] / pred - 1) > 0.25:
            det7.append(dict(n=n, l=l, med=float(hm[l - 1]), kernel=pred))
    if n in CH:
        for x in CH[n]['ham_layers']:
            l = x['l']
            if l <= len(hm) and hm[l - 1] >= 2:
                ok7b.append(abs(x['ham'] / hm[l - 1] - 1) <= 0.25)
S['P1.7'] = dict(hit=all(ok7a) and all(ok7b), kernel=f'{sum(ok7a)}/{len(ok7a)}', chain=f'{sum(ok7b)}/{len(ok7b)}', kernel_misses=det7[:40])
sl = {}
for n in [128, 256]:
    hm = layer_med(n, 'hamming')
    ls = np.arange(8, n // 4 + 1)
    sl[n] = float(np.polyfit(np.log(ls), np.log(hm[ls - 1]), 1)[0])
S['P1.8'] = dict(hit=all(-1.25 <= v <= -0.6 for v in sl.values()), slopes=sl)
ok, det = [], []
for n in NS:
    per = []
    for s in SEEDS:
        r = A.get((n, s))
        if r:
            per.append([x['H_marg'] / x['hamming'] if x['hamming'] > 0 else np.nan for x in r['layers']])
    hm = layer_med(n, 'hamming')
    m = min(len(p) for p in per)
    rat = np.nanmedian(np.array([p[:m] for p in per]), 0)
    for l in range(8, n // 2 + 1):
        if hm[l - 1] >= 4:
            ok.append(2.1 <= rat[l - 1] <= 2.5)
            if not (2.1 <= rat[l - 1] <= 2.5):
                det.append(dict(n=n, l=l, ratio=float(rat[l - 1])))
S['P1.9'] = dict(hit=all(ok), n_ok=f'{sum(ok)}/{len(ok)}', misses=det[:30])
v = []
for s in SEEDS:
    r = A.get((256, s))
    if r:
        x = r['layers'][30]
        v.append(dict(seed=s, H=x['H_marg'], ham=x['hamming'], l=x['l']))
S['P1.10'] = dict(hit=all(30 <= x['H'] <= 65 and 14 <= x['ham'] <= 28 for x in v), nets=v)


# 1b
def bcell(n, L, key):
    return [B[(n, s)]['by_depth'][str(L)][key] for s in SEEDS if (n, s) in B and str(L) in B[(n, s)]['by_depth']]


S['P1.11'] = dict(hit=6 <= med(bcell(256, 32, 'D')) <= 18, D=bcell(256, 32, 'D'))
q = med(bcell(256, 128, 'D')) / med(bcell(64, 32, 'D'))
S['P1.12'] = dict(hit=q >= 1.5, ratio=q, D256_128=bcell(256, 128, 'D'), D64_32=bcell(64, 32, 'D'))
okm, okg, det = [], [], []
for n in NS:
    for L in sorted({max(1, n // 8), n // 4, n // 2, n, 32}):
        if L < n // 4 or L > 2 * n:
            continue
        w1 = bcell(n, L, 'W1')
        if not w1:
            continue
        okm.append(0.12 <= med(w1) <= 0.45)
        mn = ib.mean_radius(n)
        for s in SEEDS:
            if (n, s) in B and (n, s) in A:
                Vb = B[(n, s)]['by_depth'][str(L)]['V']
                EA = A[(n, s)]['split'][str(L)]['EA']
                gp = mn * mn * K[L]['b1'] * EA * EA / Vb
                okg.append(B[(n, s)]['by_depth'][str(L)]['W1'] > 2 * gp)
                det.append(dict(n=n, L=L, seed=s, W1=B[(n, s)]['by_depth'][str(L)]['W1'], gp=gp))
S['P1.13'] = dict(hit=all(okm) and all(okg), median_band=f'{sum(okm)}/{len(okm)}', gp=f'{sum(okg)}/{len(okg)}', cells=det)
ok, det = [], []
for (n, s), r in B.items():
    for L, x in r['by_depth'].items():
        if (n, s) in A and L in A[(n, s)]['split']:
            sr = A[(n, s)]['split'][L]['s_rad']
            d = x['S1'] - x['W1']
            c = sr - 0.03 <= d <= sr + 0.06
            ok.append(c)
            if not c:
                det.append(dict(n=n, seed=s, L=int(L), S1_minus_W1=d, s_rad=sr))
S['P1.14'] = dict(hit=all(ok), n_ok=f'{sum(ok)}/{len(ok)}', misses=det)
ok, det = [], []
for n in NS:
    for L in sorted({max(1, n // 8), n // 4, n // 2, n, 32}):
        v = bcell(n, L, 's_even')
        if v and L >= n // 8:
            ok.append(0.35 <= med(v) <= 0.65); det.append(dict(n=n, L=L, s_even=med(v)))
S['P1.15'] = dict(hit=all(ok), n_ok=f'{sum(ok)}/{len(ok)}', cells=det)
ok1, ok2, det = [], [], []
for n in NS:
    for L in sorted({n // 2, n} | ({32} if 32 >= n // 2 else set())):
        if L < n // 2:
            continue
        rr = [B[(n, s)]['by_depth'][str(L)] for s in SEEDS if (n, s) in B and str(L) in B[(n, s)]['by_depth']]
        if not rr:
            continue
        ratio = med([x['rot_W2_off'] / x['W2_off'] if x['W2_off'] > 0 else np.inf for x in rr])
        gain = med([x['rot_S1'] - x['S1'] - 0.5 * (x['W2_off'] - x['rot_W2_off']) for x in rr])
        ok1.append(ratio <= 0.6); ok2.append(gain >= 0)
        det.append(dict(n=n, L=L, rot_off_ratio=ratio, excess=gain, S1=med([x['S1'] for x in rr]), rot_S1=med([x['rot_S1'] for x in rr]),
                        W2_off=med([x['W2_off'] for x in rr]), rot_W2_off=med([x['rot_W2_off'] for x in rr])))
S['P1.16'] = dict(hit=all(ok1) and all(ok2), off_ratio_ok=f'{sum(ok1)}/{len(ok1)}', s1_ok=f'{sum(ok2)}/{len(ok2)}', cells=det)


# 1c
def ccell(n, l, f):
    v = []
    for s in SEEDS:
        r = C.get((n, s))
        if r and str(l) in r['layers']:
            v.append(f(r['layers'][str(l)]))
    return v


ok17, ok18a, ok18b, ok20, det = [], [], [], [], []
sl19, g19 = {}, []
for n in [64, 256]:
    marks = [2 ** i for i in range(0, 12) if 2 ** i <= 2 * n]
    for l in marks:
        if 2 <= l <= n // 4:
            q = med(ccell(n, l, lambda x: x['finite']['ang_rel'])) / K[l]['delta']
            ok17.append(0.5 <= q <= 2.0); det.append(dict(n=n, l=l, ang_rel_over_delta=q))
        if 4 <= l <= n // 2:
            ok18a.append(med(ccell(n, l, lambda x: x['tangent']['ang_PR'])) <= n / (2 * l) + 2)
        if l <= n // 2:
            ok18b.append(med(ccell(n, l, lambda x: x['finite']['ang_PR'])) >= med(ccell(n, l, lambda x: x['tangent']['ang_PR'])))
            g = med(ccell(n, l, lambda x: x['Gamma_mean']))
            g19.append(0.3 <= g <= 3)
        if l in (n, 2 * n):
            ok20.extend([x >= 0.9 for x in ccell(n, l, lambda x: x['finite']['radial_share'])])
    ls = [l for l in marks if 4 <= l <= n // 2]
    sr = [med(ccell(n, l, lambda x: x['sens_ratio_mean'])) for l in ls]
    sl19[n] = float(np.polyfit(np.log(ls), np.log(sr), 1)[0])
S['P1.17'] = dict(hit=all(ok17), n_ok=f'{sum(ok17)}/{len(ok17)}', cells=det)
S['P1.18'] = dict(hit=all(ok18a) and all(ok18b), tangent_PR=f'{sum(ok18a)}/{len(ok18a)}', finite_ge_tangent=f'{sum(ok18b)}/{len(ok18b)}')
S['P1.19'] = dict(hit=all(v >= 1.0 for v in sl19.values()) and all(g19), slopes=sl19, gamma_ok=f'{sum(g19)}/{len(g19)}')
S['P1.20'] = dict(hit=all(ok20), n_ok=f'{sum(ok20)}/{len(ok20)}')
json.dump(S, open(f'{D_}/results/part1_scores.json', 'w'), indent=1, default=float)
for k, v in S.items():
    print(k, 'HIT' if v['hit'] else 'MISS', json.dumps({kk: vv for kk, vv in v.items() if kk not in ('cells', 'hit')}, default=float)[:400])

# ---------------- post hoc (after astra round 2): P1.19 on the bug-fixed Gamma reruns runs/p1cfix_* (same nets and seeds);
# reported alongside the registered score, which stays as computed above.
CF = {}
for f in glob.glob(f'{D_}/runs/p1cfix_n*_s*.json'):
    r = json.load(open(f)); CF[(r['n'], r['seed'])] = r
if len(CF) == 8:
    def cfcell(n, l, fn):
        return [fn(CF[(n, s)]['layers'][str(l)]) for s in SEEDS if (n, s) in CF and str(l) in CF[(n, s)]['layers']]
    sl2, g2 = {}, []
    for n in [64, 256]:
        marks = [2 ** i for i in range(0, 12) if 2 ** i <= 2 * n]
        for l in marks:
            if l <= n // 2:
                g2.append(0.3 <= med(cfcell(n, l, lambda x: x['Gamma_mean'])) <= 3)
        ls = [l for l in marks if 4 <= l <= n // 2]
        sl2[n] = float(np.polyfit(np.log(ls), np.log([med(cfcell(n, l, lambda x: x['sens_ratio_mean'])) for l in ls]), 1)[0])
    S['P1.19_fixed'] = dict(hit=all(v >= 1.0 for v in sl2.values()) and all(g2), slopes=sl2, gamma_ok=f'{sum(g2)}/{len(g2)}',
                            gamma_medians={n: {l: med(cfcell(n, l, lambda x: x['Gamma_mean'])) for l in [2 ** i for i in range(0, 12) if 2 ** i <= 2 * n]} for n in (64, 256)})
    json.dump(S, open(f'{D_}/results/part1_scores.json', 'w'), indent=1, default=float)
    print('P1.19_fixed', 'HIT' if S['P1.19_fixed']['hit'] else 'MISS', json.dumps({k: v for k, v in S['P1.19_fixed'].items() if k != 'hit'}, default=float)[:600])
else:
    print('P1.19_fixed pending:', len(CF), 'of 8 reruns')
