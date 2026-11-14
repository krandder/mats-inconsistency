"""Part 2 scoring (PREREG-TAILS sections 2-4 and amendments 1-3, with the code-review fixes): truth checks, every
method's per-event and amortised FLOPs to 50% relative error (B50), fractions within 50%, better-than-zero, wins.
Every table is printed twice: all events, and validated events only (provisional ones removed for every method).
python tails/analyze_tails.py TRUTHDIR KPDIR MCDIR KDDIR [--mode MODEDIR] [--sigmas ..] [--json OUT]"""
import argparse, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deduce as D
from common import c_path, c_move, run_evals, load_truth

ap = argparse.ArgumentParser()
ap.add_argument('truth'); ap.add_argument('kp'); ap.add_argument('mc'); ap.add_argument('kd')
ap.add_argument('--mode', default=None)
ap.add_argument('--sigmas', default='0.001,0.003,0.01'); ap.add_argument('--seeds', default=','.join(map(str, range(20))))
ap.add_argument('--depths', default='4,8,32')
ap.add_argument('--json', default=None); ap.add_argument('--boot', type=int, default=10000)
A = ap.parse_args()
SIG = [float(v) for v in A.sigmas.split(',')]; SEEDS = [int(v) for v in A.seeds.split(',')]
DEPTHS = [int(v) for v in A.depths.split(',')]
KS = (4, 6, 8, 10, 12); NN = 8; n = 256
rng = np.random.default_rng(12345)
INF = np.inf
POSTHOC = {'U2', 'U2r', 'U2a', 'U2ar'}
MOM_ACC = 2 * 4 * n            # accumulating four powers of 256 outputs, per path per depth
COMBINED = ('C-MCx-gauss', 'C-MCx-edge', 'U3', 'U3r')


def rel_mse(E, p):
    """E (R, ...) replicate estimates, p (...) truth -> relative MSE of one replicate, normalised residuals."""
    return ((E / p - 1.0) ** 2).mean(0)


def ev(L, sigma):
    out = dict(seeds=[], p=[], prov=[], flags=[])
    meth = {}
    cm, cp = c_move(L), c_path(L)
    for s in SEEDS:
        tf = os.path.join(A.truth, f'truth_{s}_s{sigma:g}.json')
        if not os.path.exists(tf):
            continue
        T = load_truth(tf)[L]
        p = np.exp(T['lp'])
        flags = np.zeros((NN, 5), dtype=int)
        flags[~(T['rse'] <= 0.25)] |= 1
        flags[~(np.abs(T['half']) <= 1)] |= 2
        # (b) Monte Carlo at k = 4 (missing -> unverified; zero count -> 95% bound 3/N)
        mf = os.path.join(A.mc, f'mc_xc_{s}_s{sigma:g}.json')
        if os.path.exists(mf):
            m = json.load(open(mf)); N = m['N']; cnt = np.array(m['reps'][0]['counts'][str(L)])[:, 0]
            for i in range(NN):
                if cnt[i] == 0:
                    bad = p[i, 0] > 3.0 / N
                else:
                    pm = cnt[i] / N; se_m = math.sqrt((1 - pm) / (pm * N))
                    bad = abs(math.log(pm) - T['lp'][i, 0]) > 3 * math.sqrt(T['rse'][i, 0] ** 2 + se_m ** 2)
                if bad:
                    flags[i, 0] |= 4
        else:
            flags[:, 0] |= 16
        # (c) independent splitting, outputs 0-1
        bad = ~(np.abs(T['xc'] - T['lp'][:2]) <= 3 * np.sqrt(T['rse'][:2] ** 2 + T['xc_rse'] ** 2))
        flags[:2][bad] |= 8
        out['seeds'].append(s); out['p'].append(p); out['prov'].append(flags > 0); out['flags'].append(flags)
        kidx = np.arange(5)[None, :].repeat(NN, 0)
        # ---------------- C-MC (analytic)
        N50 = 4 * (1 - p) / p
        meth.setdefault('C-MC', []).append(dict(kind='rand', rmse=np.full((NN, 5), 0.25), b50=N50 * cp,
                                                b50am=N50 * cp / (n * (kidx + 1))))
        # ---------------- C-GS from the truth's frozen runs (tested budget = the R pooled runs)
        nlev, rv1, R = T['nlev'], T['rv1'], T['R']
        b = run_evals(500, nlev) * cm + run_evals(2000, nlev) * cm * rv1 / 0.25
        b = np.where(rv1 / R < 1, b, INF)
        meth.setdefault('C-GS', []).append(dict(kind='rand', rmse=rv1, b50=b, b50am=b))
        # ---------------- kprop-based (deterministic)
        for var in ('k2', 'k3f', 'k3fa'):
            kf = os.path.join(A.kp, f'kp_{s}_L{L}_s{sigma:g}_{var}.json')
            if not os.path.exists(kf):
                continue
            kp = json.load(open(kf))
            sh, hv, mo = kp['flops_shared'], kp['flops_heavi_each'][0], kp['flops_moments']
            est = np.array(kp['heavi'])
            name = {'k2': 'C-K2', 'k3f': 'C-K3f', 'k3fa': 'C-K3fa'}[var]
            meth.setdefault(name, []).append(dict(kind='det', rel=np.abs(est / p - 1), cost=sh + hv,
                                                  costam=sh / (5 * n) + hv / n, est=est))
            if var in ('k3f', 'k3fa'):
                M = np.array(kp['moments']); tag = {'k3f': 'U2', 'k3fa': 'U2a'}[var]
                e4 = np.zeros((NN, 5)); er = np.zeros((NN, 5)); fb = np.zeros((NN, 5))
                for i in range(NN):
                    kk = D.cumulants_from_moments(M[:, i])
                    for k in range(5):
                        t = T['thr'][i, k]
                        v = D.saddle(t, kk, 4)
                        if not np.isfinite(v):
                            fb[i, k] = 1; v = D.saddle(t, kk, 3)
                        if not np.isfinite(v):
                            fb[i, k] = 2; v = D.gauss(t, kk)
                        e4[i, k] = v
                        er[i, k] = D.radial(t, M[:, i], order=4)
                ro = D.F_SADDLE + D.F_CUMULANTS
                rr = D.F_RADIAL_NODES * (D.F_SADDLE + 10) + D.F_CUMULANTS
                for nm, e, rcost in ((tag, e4, ro), (tag + 'r', er, rr)):
                    rel = np.where(np.isfinite(e), np.abs(e / p - 1), INF)
                    meth.setdefault(nm, []).append(dict(kind='det', rel=rel, cost=sh + mo + rcost,
                                                        costam=(sh + mo) / (5 * n) + rcost, est=e, fallback=fb))
        # ---------------- Monte Carlo moments: C-MCx (Gaussian, Edgeworth) and U3 (saddlepoint, radial)
        for tagM in ('m20', 'm22'):
            mf = os.path.join(A.mc, f'mc_{tagM}_{s}_s{sigma:g}.json')
            if not os.path.exists(mf):
                continue
            m = json.load(open(mf)); Mn = m['N']
            ests = {k_: [] for k_ in COMBINED}
            fbs = []
            for rep in m['reps']:
                Mo = np.array(rep['moments'][str(L)])
                eg = np.zeros((NN, 5)); ee = np.zeros((NN, 5)); es = np.zeros((NN, 5)); er = np.zeros((NN, 5))
                fb = np.zeros((NN, 5))
                for i in range(NN):
                    kk = D.cumulants_from_moments(Mo[:, i])
                    for k in range(5):
                        t = T['thr'][i, k]
                        eg[i, k] = D.gauss(t, kk); ee[i, k] = D.edgeworth(t, kk)
                        v = D.saddle(t, kk, 4)
                        if not np.isfinite(v):
                            fb[i, k] = 1; v = D.saddle(t, kk, 3)
                        if not np.isfinite(v):
                            fb[i, k] = 2; v = D.gauss(t, kk)
                        es[i, k] = v; er[i, k] = D.radial(t, Mo[:, i], order=4)
                for nm, e in (('C-MCx-gauss', eg), ('C-MCx-edge', ee), ('U3', es), ('U3r', er)):
                    ests[nm].append(np.where(np.isfinite(e), e, 0.0))
                fbs.append(fb)
            cost = Mn * (cp + MOM_ACC)
            ro = {'C-MCx-gauss': D.F_GAUSS, 'C-MCx-edge': D.F_EDGE, 'U3': D.F_SADDLE,
                  'U3r': D.F_RADIAL_NODES * (D.F_SADDLE + 10)}
            for nm, E in ests.items():
                rm = rel_mse(np.array(E), p)
                meth.setdefault(f'{nm}@{tagM}', []).append(dict(kind='biased', rmse=rm, cost=cost + ro[nm] + D.F_CUMULANTS,
                                                               costam=cost / (5 * n) + ro[nm] + D.F_CUMULANTS,
                                                               fallback=np.array(fbs).mean(0) if nm == 'U3' else None))
        # ---------------- U1: mode splitting (L = 32), two reach rules (amendment 3)
        if A.mode and L == 32:
            mf = os.path.join(A.mode, f'mode_{s}_L{L}_s{sigma:g}.json')
            if os.path.exists(mf):
                mo = json.load(open(mf))
                M0 = mo['M0']
                setup = M0 * c_path(L) + 2 * M0 * n * n + 10 * n ** 3 + sum(ru['pilot_n_eval'] * cm for ru in mo['runs'].values())
                per = 0.0
                for ru in mo['runs'].values():
                    K = len(ru['levels'])
                    per += run_evals(2000, K + 1) * cm                                  # moves
                    per += (K + 1) * 2000 * ((L - 1) * (n * (2 * n - 1) + 3 * n))      # readout recomputes h (as implemented)
                    per += (K + 1) * 2000 * (n * (2 * n - 1) + n * 5)                  # readout: all outputs, 5 thresholds
                rm = np.zeros((NN, 5)); nz = np.zeros((NN, 5), dtype=bool)
                for i in range(NN):
                    E = np.array(mo['runs']['up' if mo['c'][i] > 0 else 'down']['est'])[:, i, :]
                    rm[i] = rel_mse(E, p[i]); nz[i] = (E > 0).any(0)
                tot = setup + per * rm / 0.25
                for tag, ok in (('U1', nz & (rm < 1)), ('U1m', nz & (rm <= 0.81))):
                    meth.setdefault(tag, []).append(dict(kind='rand', rmse=rm, b50=np.where(ok, tot, INF),
                                                         b50am=np.where(ok, tot / (5 * n), INF)))
        # ---------------- Kerdock design and simulated Monte Carlo (k = 4, 6)
        kf = os.path.join(A.kd, f'kd_{s}_s{sigma:g}.json')
        if os.path.exists(kf):
            kd = json.load(open(kf)); Nk = kd['N']; Rk = kd['R']
            for nm, key in (('C-Kd', 'kerdock'), ('C-MCsim', 'mc')):
                Cn = np.array(kd[key]['counts'][str(L)]) / Nk                  # (R, 8, 5)
                rm = rel_mse(Cn, p)
                setup = kd[key]['setup_flops'] / Rk                              # per replicate: rotations + basis products
                # Kerdock: no input draw; first-layer products are inside setup (z1_points); radii ~100 FLOPs per pair
                per = Nk * (cp - (16 * n + n * (2 * n - 1) - 50 if key == 'kerdock' else 0))
                ok = (Cn > 0).any(0) & (rm / Rk < 1)
                b = np.where(ok, (setup + per) * rm / 0.25, INF)
                b[:, 2:] = INF
                meth.setdefault(nm, []).append(dict(kind='rand', rmse=rm, b50=b, b50am=b / (n * (kidx + 1))))
    # combine moment budgets: B50 = cheapest tested budget whose relative MSE <= 0.25 (amendment 1)
    for base in COMBINED:
        vs = [k for k in meth if k.startswith(base + '@')]
        if len(vs) != 2 or any(len(meth[v]) != len(out['seeds']) for v in vs):
            continue
        comb = []
        for j in range(len(out['seeds'])):
            ds = sorted([meth[v][j] for v in vs], key=lambda d: d['cost'])
            b = np.full((NN, 5), INF); bam = np.full((NN, 5), INF)
            for d in reversed(ds):
                ok = d['rmse'] <= 0.25
                b = np.where(ok, d['cost'], b); bam = np.where(ok, d['costam'], bam)
            comb.append(dict(kind='rand', rmse=ds[-1]['rmse'], b50=b, b50am=bam))
        meth[base] = comb
    return out, meth


def b50_of(d, am):
    if d['kind'] == 'det':
        return np.where(d['rel'] <= 0.5, d['costam'] if am else d['cost'], INF)
    if d['kind'] == 'biased':
        return np.where(d['rmse'] <= 0.25, d['costam'] if am else d['cost'], INF)
    return d['b50am'] if am else d['b50']


def relerr_of(d):
    return d['rel'] if d['kind'] == 'det' else np.sqrt(d['rmse'])


def q(a, f):
    return float(np.quantile(a, f, method='inverted_cdf'))


def boot_pooled(V, mask):
    """V, mask: (nets, NN) for one k. Median over the (masked) events pooled over nets, with a bootstrap 95% CI from
    resampling nets. Order-statistic quantiles, so infinities stay infinite."""
    vals = [V[j][mask[j]] for j in range(len(V))]
    allv = np.concatenate(vals) if vals else np.array([])
    if allv.size == 0:
        return float('nan'), float('nan'), float('nan')
    med = q(allv, 0.5)
    nets = len(vals); meds = np.empty(A.boot)
    for bb in range(A.boot):
        pick = rng.integers(0, nets, nets)
        x = np.concatenate([vals[j] for j in pick])
        meds[bb] = q(x, 0.5) if x.size else np.nan
    meds = meds[~np.isnan(meds)]
    return med, q(meds, 0.025), q(meds, 0.975)


def boot_median_nets(v):
    v = np.asarray(v, float); m = len(v)
    meds = np.array([q(v[rng.integers(0, m, m)], 0.5) for _ in range(A.boot)])
    return q(v, 0.5), q(meds, 0.025), q(meds, 0.975)


def fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 'nan'
    return 'inf' if not np.isfinite(x) else ('%.2g' % x)


summary = {}
for sigma in SIG:
    for L in DEPTHS:
        out, meth = ev(L, sigma)
        if not out['seeds']:
            continue
        P = np.array(out['p']); PROV = np.array(out['prov']); FL = np.array(out['flags']); nets = len(out['seeds'])
        print(f'\n######## L = {L}, sigma = {sigma}: {nets} nets; provisional events {int(PROV.sum())}/{PROV.size} '
              f'(by k: {[int(PROV[:, :, k].sum()) for k in range(5)]}); flags: rse {int(((FL & 1) > 0).sum())}, '
              f'halves {int(((FL & 2) > 0).sum())}, MC {int(((FL & 4) > 0).sum())}, MC missing {int(((FL & 16) > 0).sum())}, '
              f'xcheck {int(((FL & 8) > 0).sum())}')
        print('   median truth log10 p by k: ' + ' '.join('%.2f' % np.median(np.log10(P[:, :, k])) for k in range(5)))
        for valid in (False, True):
            mask_all = ~PROV if valid else np.ones_like(PROV, dtype=bool)
            vt = 'validated events only' if valid else 'all events'
            for am in (False, True):
                print(f'  -- [{vt}] {"amortised over the 256 outputs" if am else "per event"}: median B50 [95% CI] | fraction reached | median rel err (or rel RMSE)')
                for name, lst in meth.items():
                    if '@' in name or len(lst) != nets:
                        continue
                    B = np.array([b50_of(d, am) for d in lst]); RE = np.array([relerr_of(d) for d in lst])
                    row = []
                    for k in range(5):
                        if name in ('C-Kd', 'C-MCsim') and k >= 2:
                            row.append('-'); continue
                        mk = mask_all[:, :, k]
                        m, lo, hi = boot_pooled(B[:, :, k], mk)
                        fr = float(np.isfinite(B[:, :, k][mk]).mean()) if mk.any() else float('nan')
                        re = float(np.median(RE[:, :, k][mk])) if mk.any() else float('nan')
                        row.append(f'{fmt(m)} [{fmt(lo)},{fmt(hi)}] {fr:.2f} {re:.2f}')
                        summary.setdefault(f'{sigma}|{L}|{name}|{"am" if am else "ev"}|{"valid" if valid else "all"}', {})[KS[k]] = dict(
                            b50=m, lo=lo, hi=hi, reached=fr, relerr=re)
                    lab = ' (post hoc)' if name in POSTHOC else ''
                    print(f'   {name + lab:>22}: ' + ' | '.join(row))
            # better than zero: deterministic methods at their own cost; moment methods and U1 at their tested budgets
            res_lines = []
            for name, lst in meth.items():
                if len(lst) != nets or name in ('C-MC', 'C-GS', 'C-Kd', 'C-MCsim') or name in COMBINED:
                    continue
                RE = np.array([relerr_of(d) for d in lst]); flags_ = []
                for k in range(5):
                    m, lo, hi = boot_pooled(np.minimum(RE[:, :, k], 1e6), mask_all[:, :, k])
                    flags_.append('Y' if (m <= 0.9 and hi < 1) else 'n')
                res_lines.append(f'{name}{" (post hoc)" if name in POSTHOC else ""} ' + ''.join(flags_))
            print(f'  -- [{vt}] better than zero (median rel err or rel RMSE <= 0.9, CI upper < 1), k = 4..12: ' + '; '.join(res_lines))
            # randomised methods at stated per-event budgets: Monte Carlo (analytic) and splitting (scaled from N = 2000)
            ARC = {4: 3.45e9, 8: 1.34e10, 32: 2.09e11}   # one threshold vector: shared propagation + one heaviside step (kp files)
            for Bud in (ARC[L], 3e11):
                txt = []
                Nmc = Bud / c_path(L)
                REmc = np.sqrt((1 - P) / (P * Nmc))
                rows = []
                for j, s_ in enumerate(out['seeds']):
                    T = load_truth(os.path.join(A.truth, f'truth_{s_}_s{sigma:g}.json'))[L]
                    cpil = run_evals(500, T['nlev']) * c_move(L); crun = run_evals(2000, T['nlev']) * c_move(L)
                    avail = Bud - cpil
                    rows.append(np.where(avail > 0, np.sqrt(T['rv1'] * crun / np.maximum(avail, 1e-300)), INF))
                for nm, RE in (('C-MC', REmc), ('C-GS', np.array(rows))):
                    fl_ = []
                    for k in range(5):
                        m, lo, hi = boot_pooled(np.minimum(RE[:, :, k], 1e6), mask_all[:, :, k])
                        fl_.append('Y' if (m <= 0.9 and hi < 1) else 'n')
                    txt.append(f'{nm} ' + ''.join(fl_))
                print(f'  -- [{vt}] better than zero at {Bud:.3g} FLOPs per event: ' + '; '.join(txt))
            # sampling/hybrid category: ours vs the best competitor (median over nets of the per-net B50 ratio)
            comps = [c for c in ('C-MC', 'C-GS', 'C-Kd', 'C-MCx-gauss', 'C-MCx-edge') if c in meth and len(meth[c]) == nets]
            for ours in ('U3', 'U3r', 'U1', 'U1m'):
                if ours not in meth or len(meth[ours]) != nets:
                    continue
                for am in (False, True):
                    row = []
                    for k in range(5):
                        mk = mask_all[:, :, k]

                        def pernet(name):
                            B = np.array([b50_of(d, am) for d in meth[name]])[:, :, k]
                            return np.array([q(B[j][mk[j]], 0.5) if mk[j].any() else np.nan for j in range(nets)])
                        best, bestm = None, INF
                        for c in comps:
                            if c == 'C-Kd' and k >= 2:
                                continue
                            Bc_ = np.array([b50_of(d, am) for d in meth[c]])[:, :, k][mk]
                            m_ = q(Bc_, 0.5) if Bc_.size else INF
                            if best is None or m_ < bestm:
                                best, bestm = c, m_
                        Bo, Bc = pernet(ours), pernet(best)
                        ok = ~np.isnan(Bo) & ~np.isnan(Bc)
                        with np.errstate(divide='ignore', invalid='ignore'):
                            ratio = np.where(np.isinf(Bo[ok]) & np.isinf(Bc[ok]), 1.0, Bc[ok] / Bo[ok])
                        m, lo, hi = boot_median_nets(ratio) if ratio.size else (np.nan, np.nan, np.nan)
                        row.append(f'{best} {fmt(m)} [{fmt(lo)},{fmt(hi)}]{" WIN" if lo > 1 else ""}')
                        summary.setdefault(f'{sigma}|{L}|win|{ours}|{"am" if am else "ev"}|{"valid" if valid else "all"}', {})[KS[k]] = dict(
                            best=best, ratio=m, lo=lo, hi=hi)
                    print(f'  -- [{vt}] {ours} vs best sampling/hybrid competitor ({"amortised" if am else "per event"}): competitor/ours, median over nets [CI]: ' + ' | '.join(row))
            # descriptive (post hoc) comparison of the killed U2 variants with ARC's readouts
            for ours, comp in (('U2', 'C-K3f'), ('U2r', 'C-K3f'), ('U2a', 'C-K3fa')):
                if ours in meth and comp in meth and len(meth[ours]) == len(meth[comp]) == nets:
                    Ro = np.array([d['rel'] for d in meth[ours]]); Rc = np.array([d['rel'] for d in meth[comp]])
                    co = meth[ours][0]['cost'] / meth[comp][0]['cost']; ca = meth[ours][0]['costam'] / meth[comp][0]['costam']
                    row = []
                    for k in range(5):
                        mk = mask_all[:, :, k]
                        dfr = np.array([np.mean(Ro[j, :, k][mk[j]] <= 0.5) - np.mean(Rc[j, :, k][mk[j]] <= 0.5) if mk[j].any() else np.nan for j in range(nets)])
                        dfr = dfr[~np.isnan(dfr)]
                        mm = np.array([dfr[rng.integers(0, len(dfr), len(dfr))].mean() for _ in range(A.boot)])
                        row.append(f'{dfr.mean():+.2f} [{q(mm, 0.025):+.2f},{q(mm, 0.975):+.2f}]')
                    print(f'  -- [{vt}] POST HOC (killed; no claim) {ours} - {comp}: per-net fraction within 50%, mean difference [CI] '
                          f'(cost ratio per event {co:.2f}, amortised {ca:.2f}): ' + ' | '.join(row))
            # fallbacks of the saddlepoint (U2 and U3)
            for name in ('U2', 'U3@m20', 'U3@m22'):
                if name in meth and len(meth[name]) == nets and meth[name][0].get('fallback') is not None:
                    F = np.array([d['fallback'] for d in meth[name]])
                    print(f'  -- [{vt}] {name} saddlepoint fallbacks (fraction of events not using order 4), k = 4..12: '
                          + ' '.join('%.2f' % np.mean((F[:, :, k] > 0)[mask_all[:, :, k]]) for k in range(5)))
if A.json:
    json.dump(summary, open(A.json, 'w'), indent=1, default=float)
