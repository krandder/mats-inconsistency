"""Score PREREG Part B (PB1-PB8, HB) from runs/{truth,mlmc,cv,anti,rqmc,tail}_n*_s*.json."""
import glob, json, math, os
import numpy as np
from scipy.stats import spearmanr

D = '/home/agent/workspaces/mats/work/stationary'
NS = [8, 16, 32, 64, 128, 256]; SEEDS = [1000, 1001]
ld = lambda p: json.load(open(p)) if os.path.exists(p) else None
rows = []
for n in NS:
    for s in SEEDS:
        T = ld(f'{D}/runs/truth_n{n}_s{s}.json'); ML = ld(f'{D}/runs/mlmc_n{n}_s{s}.json')
        CV0 = ld(f'{D}/runs/cv_n{n}_s{s}.json'); CVF = ld(f'{D}/runs/cvfix_n{n}_s{s}.json'); AN = ld(f'{D}/runs/anti_n{n}_s{s}.json')
        CV = CVF or CV0   # bug fix 1 rerun (identical except the guarded RB+CV); original kept as cv_rb_orig
        RQ = ld(f'{D}/runs/rqmc_n{n}_s{s}.json'); TL = ld(f'{D}/runs/tail_n{n}_s{s}.json')
        r = dict(n=n, seed=s)
        B = 8 * n
        if ML:
            bias = ML['bias']
            B6 = next((b['T'] for b in bias if b['bias2_over_trvar'] <= 1e-6), None)   # registered rule: point estimate
            r['B6'] = B6
            r['K_MC'] = (B6 + 1) if B6 else None
            r['K_ML'] = ML['mlmc_var_cost_over_trvar']; r['K_ML_over_n'] = r['K_ML'] / n
            sel = [b for b in bias if 2 * n <= b['T'] <= 8 * n and b['bias2_over_trvar'] > 1e-14]
            if len(sel) >= 2:
                x = np.array([b['T'] for b in sel], float); y = np.log([b['bias2_over_trvar'] for b in sel])
                r['bias_rate_times_n'] = float(-np.polyfit(x, y, 1)[0] * n)
            r['bias_table'] = [(b['T'], b['bias2_over_trvar']) for b in bias]
        Kref = r.get('K_MC') or (B + 1)
        if CV:
            r['cv'] = CV['cv_var_ratio']; r['cv_rb'] = CV['rb_cv_var_ratio']; r['rb'] = CV['rb_var_ratio']; r['cv_hard'] = CV['cv_hard_var_ratio']
            r['cv_rb_orig'] = CV0['rb_cv_var_ratio'] if CV0 else None
            r['gain_cv_rb'] = Kref / (2 * Kref * CV['rb_cv_var_ratio'])
            r['gain_rb'] = 1.0 / CV['rb_var_ratio']
            r['setup_over_Kref'] = CV['setup_mv'] / Kref
            if T:
                tm = np.array(T['mean']); tv = T['trvar']
                r['nmse_K2'] = float(np.sum((np.array(CV['EK2']) - tm) ** 2) / tv)
                r['nmse_K2_truth_res'] = 1.0 / T['M']
                r['nmse_cvest'] = float(np.sum((np.array(CV['cv_est']) - tm) ** 2) / tv)
        if AN:
            r['anti'] = AN['pair_var_over_half_trvar']; r['gain_anti'] = 1.0 / AN['pair_var_over_half_trvar']
        if RQ and T:
            tv = T['trvar']
            pts = [(q['N'], q['var_sum']) for q in RQ['rows']]
            r['rqmc_ratio'] = [(N, (tv / N) / v) for N, v in pts]
            Ns = np.array([p[0] for p in pts], float); vs = np.array([p[1] for p in pts])
            r['rqmc_exponent'] = float(-np.polyfit(np.log(Ns), np.log(vs), 1)[0])
            r['gain_rqmc'] = r['rqmc_ratio'][-1][1]
        if TL:
            p = TL['p_truth']; mcrv = TL['mc_rel_var']
            tl = []
            for q in TL['rows']:
                kind = q.get('kind', 'radial')
                se_is = math.sqrt(max(q['rel_var_per_sample'], 1e-300) / TL['M']) * p
                se_t = T['tails'][TL.get('ptail', '0.01')]['se'] if T else 0.0
                tl.append(dict(kind=kind, scale=q['scale'], gain=mcrv / max(q['rel_var_per_sample'], 1e-300),
                               rel_err=q['p_est'] / p - 1, z=(q['p_est'] - p) / math.sqrt(se_is ** 2 + se_t ** 2)))
            r['tail'] = tl; r['p'] = p
        rows.append(r)

# per-method summaries
def med(xs):
    xs = [x for x in xs if x is not None]
    return float(np.median(xs)) if xs else None

out = {}
by_n = {n: [r for r in rows if r['n'] == n] for n in NS}
out['PB1'] = dict(hit=all(4 <= r['K_ML_over_n'] <= 25 for r in rows if 'K_ML_over_n' in r) and
                  sum(1 for r in rows if r.get('K_ML') and r.get('K_MC') and r['K_ML'] > r['K_MC']) >= len([r for r in rows if r.get('K_ML')]) / 2,
                  K_ML_over_n={n: [round(r['K_ML_over_n'], 2) for r in by_n[n] if 'K_ML_over_n' in r] for n in NS},
                  K_MC={n: [r.get('K_MC') for r in by_n[n]] for n in NS})
out['PB2'] = dict(hit=all(1.5 <= r['bias_rate_times_n'] <= 3.5 for r in rows if 'bias_rate_times_n' in r),
                  rates={n: [round(r['bias_rate_times_n'], 2) for r in by_n[n] if 'bias_rate_times_n' in r] for n in NS})
cvr = [(r['n'], r['cv_rb']) for r in rows if 'cv_rb' in r]
sp3 = spearmanr([c[0] for c in cvr], [c[1] for c in cvr]).correlation if len(cvr) > 2 else None
out['PB3'] = dict(spearman=sp3, hit=(sp3 is not None and sp3 <= -0.6 and all(r['cv_rb'] <= 0.2 for r in by_n[256] if 'cv_rb' in r)),
                  refuted=(sp3 is not None and (sp3 > -0.3 or any(r['cv_rb'] > 0.35 for r in by_n[256] if 'cv_rb' in r))),
                  cv_rb={n: [round(r['cv_rb'], 3) for r in by_n[n] if 'cv_rb' in r] for n in NS},
                  cv={n: [round(r['cv'], 3) for r in by_n[n] if 'cv' in r] for n in NS},
                  cv_hard={n: [round(r['cv_hard'], 3) for r in by_n[n] if 'cv_hard' in r] for n in NS})
k2 = [(r['n'], r['nmse_K2']) for r in rows if 'nmse_K2' in r]
sp4 = spearmanr([c[0] for c in k2], [c[1] for c in k2]).correlation if len(k2) > 2 else None
out['PB4'] = dict(spearman=sp4, hit=(sp4 is not None and sp4 <= -0.6 and all(r['nmse_K2'] <= 5e-4 for r in by_n[256] if 'nmse_K2' in r)),
                  nmse_K2={n: [f"{r['nmse_K2']:.2e}" for r in by_n[n] if 'nmse_K2' in r] for n in NS},
                  truth_res={n: [f"{r['nmse_K2_truth_res']:.1e}" for r in by_n[n] if 'nmse_K2_truth_res' in r] for n in NS})
def rq_at(r, N):
    for NN, v in r.get('rqmc_ratio', []):
        if NN == N:
            return v
    return None
pb5a = all((rq_at(r, 1024) or 0) >= 2 for n in (8, 16) for r in by_n[n] if 'rqmc_ratio' in r)
pb5b = all((rq_at(r, 1024) or 99) < 2 for r in by_n[128] if 'rqmc_ratio' in r)
pb5c = all(1.0 <= r['rqmc_exponent'] <= 1.3 for n in (32, 64, 128) for r in by_n[n] if 'rqmc_exponent' in r)
out['PB5'] = dict(hit=pb5a and pb5b and pb5c, a=pb5a, b=pb5b, c=pb5c,
                  ratio_1024={n: [round(rq_at(r, 1024), 2) if rq_at(r, 1024) else None for r in by_n[n]] for n in NS},
                  exponent={n: [round(r['rqmc_exponent'], 3) for r in by_n[n] if 'rqmc_exponent' in r] for n in NS})
rb = [(r['n'], r['rb']) for r in rows if 'rb' in r]
sp6 = spearmanr([c[0] for c in rb], [c[1] for c in rb]).correlation if len(rb) > 2 else None
out['PB6'] = dict(hit=all(0.4 <= c[1] <= 0.9 for c in rb) and sp6 is not None and sp6 >= 0.6, spearman=sp6,
                  rb={n: [round(r['rb'], 3) for r in by_n[n] if 'rb' in r] for n in NS})
out['PB7'] = dict(hit=all(r['anti'] <= 0.7 for r in rows if 'anti' in r), anti={n: [round(r['anti'], 3) for r in by_n[n] if 'anti' in r] for n in NS})
tl_ok = []
for n in NS:
    good = 0; tot = 0; rad_worse = 0
    for r in by_n[n]:
        if 'tail' not in r:
            continue
        tot += 1
        gr = [q for q in r['tail'] if q['kind'] == 'grad']; ra = [q for q in r['tail'] if q['kind'] == 'radial']
        bg = max(gr, key=lambda q: q['gain']) if gr else None; br = max(ra, key=lambda q: q['gain']) if ra else None
        if bg and bg['gain'] >= 5 and abs(bg['z']) <= 3:   # registered: within 3 combined SE
            good += 1
        if bg and br and br['gain'] < bg['gain']:
            rad_worse += 1
    tl_ok.append((n, good, tot, rad_worse))
out['PB8'] = dict(rows=tl_ok, hit=all(g >= t / 2 for _, g, t, _ in tl_ok if t) and sum(rw for *_, rw in tl_ok) >= sum(t for _, _, t, _ in tl_ok) / 2,
                  detail={f"{r['n']}_{r['seed']}": [(q['kind'], q['scale'], round(q['gain'], 2), round(q['rel_err'], 3)) for q in r.get('tail', [])] for r in rows})
G = {}
for n in NS:
    gs = []
    for r in by_n[n]:
        cand = [r.get('gain_cv_rb'), r.get('gain_rb'), r.get('gain_anti'), r.get('gain_rqmc')]
        cand = [c for c in cand if c is not None]
        if cand:
            gs.append(max(cand))
    G[n] = med(gs)
out['HB'] = dict(G=G, hit=(G.get(256) is not None and G.get(16) is not None and G[256] < 3 * G[16]))
out['gains'] = {f"{r['n']}_{r['seed']}": dict(cv_rb=r.get('gain_cv_rb'), rb=r.get('gain_rb'), anti=r.get('gain_anti'),
                                              rqmc=r.get('gain_rqmc'), K_MC=r.get('K_MC'), K_ML=r.get('K_ML')) for r in rows}
json.dump(dict(out=out, rows=rows), open(f'{D}/results/partB_scores.json', 'w'), indent=1, default=float)
for k, v in out.items():
    print(k, json.dumps(v, default=float)[:1500])
