"""Score PREREG Part A (PA1-PA12, PR1-PR2) from runs/*.json; write results/partA_scores.json and partA_tables.md."""
import glob, json, math, os, sys
import numpy as np
from scipy.stats import spearmanr

D = '/home/agent/workspaces/mats/work/stationary'
R = lambda p: json.load(open(p))
NS = [8, 16, 32, 64, 128, 256]
out = {}; md = []


def med(x):
    x = [v for v in x if v is not None]
    return float(np.median(x)) if x else None


# PA1/PA2 radius
rows = []
for n in NS:
    p = f'{D}/runs/radius_n{n}.json'
    if not os.path.exists(p):
        continue
    r = R(p)
    rows.append(dict(n=n, ratio=r['ratio_net_chain'], chain_med_over_n=r['chain_median_over_n'], hill_net=r['hill_net_1pct'],
                     hill_chain=r['hill_chain_1pct'], chain_q=r['chain_q'], net_q=r['net_q'], duf_q=r['dufresne_q']))
pa1 = all(all(abs(x - 1) <= 0.2 for x in rw['ratio'][:3]) for rw in rows if rw['n'] <= 64)
pa1_10 = all(all(abs(x - 1) <= 0.1 for x in rw['ratio'][:3]) for rw in rows)
cm = [rw['chain_med_over_n'] for rw in rows]
pa2a = rows[-1]['n'] == 256 and abs(rows[-1]['chain_med_over_n'] / (2 / (5 * math.log(2))) - 1) <= 0.1
pa2b = all(0.7 <= rw['hill_net'] <= 1.4 for rw in rows if rw['n'] >= 16)
out['PA1'] = dict(hit_20pct_n_le_64=pa1, all_within_10pct=pa1_10, rows=rows)
out['PA2'] = dict(median_hit=pa2a, monotone=bool(np.all(np.diff(cm) * np.sign(0.577 - cm[0]) >= 0)),
                  hill_hit=pa2b, chain_med_over_n=cm, hill_net=[rw['hill_net'] for rw in rows])
md.append('| n | net/chain q10, q50, q90, q99 | chain median/n | Hill (nets, 1%) | Hill (chain) |')
md.append('|---|---|---|---|---|')
for rw in rows:
    md.append(f"| {rw['n']} | {', '.join(f'{x:.3f}' for x in rw['ratio'])} | {rw['chain_med_over_n']:.4f} | {rw['hill_net']:.2f} | {rw['hill_chain']:.2f} |")

# PA3 lyap, PA4 couple
lrows = []
for n in NS[1:]:
    L = [R(p) for p in sorted(glob.glob(f'{D}/runs/lyap_n{n}_s*.json'))]
    C = [R(p) for p in sorted(glob.glob(f'{D}/runs/couple_n{n}_s*.json'))]
    if not L:
        continue
    lam1 = [l['lam_times_n'][0] for l in L]
    gaps = [float(np.mean(-np.diff(l['lam_times_n'][:9]))) for l in L]
    rates = [c['rate_times_n'] for c in C]
    lrows.append(dict(n=n, lam1n=lam1, gap_mean=gaps, couple_rate_n=rates,
                      lam_first8=[[round(x, 2) for x in l['lam_times_n'][:8]] for l in L]))
out['PA3'] = dict(lam1_hit_every_net=all(-1.6 <= x <= -1.0 for r_ in lrows for x in r_['lam1n']),
                  gap_hit_every_net=all(0.7 <= g <= 2.0 for r_ in lrows if r_['n'] >= 32 for g in r_['gap_mean']),
                  # registered aggregation rule ("medians over nets"), applied after astra's round-2 audit:
                  lam1_hit=all(-1.6 <= med(r_['lam1n']) <= -1.0 for r_ in lrows),
                  gap_hit=all(0.7 <= med(r_['gap_mean']) <= 2.0 for r_ in lrows if r_['n'] >= 32),
                  lam1_medians=[med(r_['lam1n']) for r_ in lrows], gap_medians=[med(r_['gap_mean']) for r_ in lrows], rows=lrows)
out['PA4'] = dict(hit=all(x is not None and -1.6 <= x <= -1.0 for r_ in lrows for x in r_['couple_rate_n']))
md.append('\n| n | n lambda_1 (2 nets) | mean gap n(l_k - l_k+1), k<=8 | coupling rate x n | n lambda_1..8 (net 1) |')
md.append('|---|---|---|---|---|')
for r_ in lrows:
    md.append(f"| {r_['n']} | {', '.join(f'{x:.2f}' for x in r_['lam1n'])} | {', '.join(f'{x:.2f}' for x in r_['gap_mean'])} | "
              f"{', '.join(f'{x:.2f}' if x is not None else 'NA' for x in r_['couple_rate_n'])} | {r_['lam_first8'][0]} |")

# PA5-PA7, PA11, descriptive: cloud
crows = []
for n in NS:
    Cs = [R(p) for p in sorted(glob.glob(f'{D}/runs/cloud_n{n}_s100?.json'))]
    if not Cs:
        continue
    slopes = [c['evC_slope_2_n8'] for c in Cs]
    fslopes_p = [c['ev_frozen_path_slope'] for c in Cs]; fslopes_a = [c.get('ev_frozen_attr_slope') for c in Cs]
    ratio = [float(np.median(np.array(c['evC'][1:8]) / np.array(c['coord_pred'][1:8]))) for c in Cs]
    ratio1 = [c['evC'][0] / c['coord_pred'][0] for c in Cs]
    rad = [c.get('radial_share') for c in Cs]
    skew_top = [abs(c['skew_top8'][0]) for c in Cs]; kurt_top = [abs(c['exkurt_top8'][0]) for c in Cs]
    skew_rest = [float(np.median(np.abs(c['skew_top8'][1:8]))) for c in Cs]
    crows.append(dict(n=n, slope=slopes, frozen_path_slope=fslopes_p, frozen_attr_slope=fslopes_a, ratio_k2_8=ratio,
                      ratio_k1=ratio1, radial_share=rad, skew_top=skew_top, kurt_top=kurt_top, skew_rest=skew_rest,
                      X_med=[c['X_quantiles'][1] for c in Cs], ang_total_rel=[c.get('ang_total_rel') for c in Cs],
                      cos_u_med=[c['cos_to_u_quantiles'][2] if 'cos_to_u_quantiles' in c else None for c in Cs],
                      R_cv=[c['R_cv'] for c in Cs], gateH=[c['out_gate_entropy_bits'] for c in Cs],
                      frac_unc=[c['out_frac_uncertain'] for c in Cs], logR_sd=[c['logR_sd_within'] for c in Cs],
                      trC_over_trS2=[c['tr_C'] / c['tr_S2'] if 'tr_S2' in c else None for c in Cs]))
pa5 = all(-1.3 <= s <= -0.7 for r_ in crows if r_['n'] >= 32 for s in r_['slope'])
pa5b = all(s <= -1.4 for r_ in crows if r_['n'] >= 32 for s in r_['frozen_path_slope'] + [x for x in r_['frozen_attr_slope'] if x is not None])
meds6 = [med(r_['ratio_k2_8']) for r_ in crows]
sp6 = spearmanr([r_['n'] for r_ in crows], meds6).correlation if len(crows) > 2 else None
pa6 = (sp6 is not None and sp6 >= 0.8 and all(m >= 2 for r_, m in zip(crows, meds6) if r_['n'] >= 128))
pa6_refuted = any(m <= 1.5 for r_, m in zip(crows, meds6) if r_['n'] == 256)
meds7 = [med(r_['radial_share']) for r_ in crows]
pa7 = bool(np.all(np.diff(meds7) > -0.02)) and crows[-1]['n'] == 256 and meds7[-1] >= 0.85
pa11 = all(med(r_['skew_rest']) < 0.3 and (sum(1 for s, k in zip(r_['skew_top'], r_['kurt_top']) if s > 0.5 or k > 0.8) >= len(r_['skew_top']) / 2)
           for r_ in crows if r_['n'] >= 64)
out['PA5'] = dict(hit=pa5, frozen_slopes_le_m1p4=pa5b)
out['PA6'] = dict(hit=pa6, refuted=pa6_refuted, spearman=sp6, medians=meds6)
out['PA7'] = dict(hit=pa7, medians=meds7)
out['PA11'] = dict(hit=pa11)
out['cloud_rows'] = crows
md.append('\n| n | X_med (nets) | slope evC k=2..n/8 | frozen slopes path/attr | evC/coord k=2-8 (median) | evC/coord k=1 | radial share | cos(h,u*) median | ang. var / radial | R cv within net | out gate entropy (bits) | uncertain out gates |')
md.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
f2 = lambda v: ', '.join(f'{x:.2f}' if x is not None else 'NA' for x in v)
for r_ in crows:
    md.append(f"| {r_['n']} | {f2(r_['X_med'])} | {f2(r_['slope'])} | {f2(r_['frozen_path_slope'])} / {f2(r_['frozen_attr_slope'])} | {f2(r_['ratio_k2_8'])} | {f2(r_['ratio_k1'])} | "
              f"{f2(r_['radial_share'])} | {f2(r_['cos_u_med'])} | {f2(r_['ang_total_rel'])} | {f2(r_['R_cv'])} | {f2(r_['gateH'])} | {f2(r_['frac_unc'])} |")

# PA8/PA9 births
brows = []
for n in NS:
    Bs = [R(p) for p in sorted(glob.glob(f'{D}/runs/births_n{n}_s*.json'))]
    if not Bs:
        continue
    devs = []; devc = []
    for b in Bs:
        for bn in b['bins']:
            if bn['X_med'] >= 1:
                devs.append(bn['flips_mean'] / bn['flips_pred'] - 1)
            if bn['X_med'] >= 10:
                devc.append(bn['near_mean'] / bn['near_pred_coord'] - 1)
    fm = [b['flips_mean'] for b in Bs]
    brows.append(dict(n=n, max_abs_dev=float(np.max(np.abs(devs))) if devs else None, med_dev=float(np.median(devs)) if devs else None,
                      coord_max_abs_dev=float(np.max(np.abs(devc))) if devc else None, flips_per_sqrt_n=[x / math.sqrt(n) for x in fm],
                      X_med=[b['X_med'] for b in Bs]))
pa8 = all(r_['max_abs_dev'] is not None and r_['max_abs_dev'] <= 0.15 for r_ in brows)
pa8c = all(r_['coord_max_abs_dev'] is None or r_['coord_max_abs_dev'] <= 0.2 for r_ in brows)
fps = [med(r_['flips_per_sqrt_n']) for r_ in brows if r_['n'] >= 16]
pa9 = all(0.4 <= x <= 1.2 for x in fps) and max(fps) / min(fps) < 1.6
out['PA8'] = dict(hit=pa8, coord_hit=pa8c, rows=brows)
out['PA9'] = dict(hit=pa9, flips_per_sqrt_n_medians=fps)
md.append('\n| n | flips vs n arccos(X/(X+1))/pi: median dev, max abs dev (bins X>=1) | coordinator count, max abs dev (X>=10) | flips per layer / sqrt(n) (4 nets) |')
md.append('|---|---|---|---|')
for r_ in brows:
    md.append(f"| {r_['n']} | {r_['med_dev']:+.3f}, {r_['max_abs_dev']:.3f} | {r_['coord_max_abs_dev'] if r_['coord_max_abs_dev'] is None else round(r_['coord_max_abs_dev'], 3)} | {f2(r_['flips_per_sqrt_n'])} |")

# PA10 age
arows = []
for n in NS:
    As = [R(p) for p in sorted(glob.glob(f'{D}/runs/age_n{n}_s*.json'))]
    if not As:
        continue
    for A in As:
        pr = A['profile']; js = np.array([p['j'] for p in pr], float); T = np.array([p['T_h'] for p in pr]); Ta = np.array([p['T_a'] for p in pr])
        tot = float(np.trapezoid(T, js))   # registered: trapezoid over sampled ages
        tota = float(np.trapezoid(Ta, js))
        sel_old = js >= n
        old = float(np.trapezoid(T[sel_old], js[sel_old])) / tot if sel_old.sum() > 1 else 0.0
        rad = [p.get('radial') for p in pr if n / 2 <= p['j'] <= n]
        pr4 = [p['PR'] for p in pr if p['j'] == 4]; pr8 = [p['PR'] for p in pr if p['j'] == 8]
        sel = (js >= n) & (T > 0)
        rate = float(-np.polyfit(js[sel], np.log(T[sel]), 1)[0] * n) if sel.sum() >= 3 else None
        arows.append(dict(n=n, seed=A['seed'], sumT_h=tot, sumT_a=tota, old_share=old, radial_half_to_n=rad,
                          PR4=pr4[0] if pr4 else None, PR8=pr8[0] if pr8 else None, coord_PR4=n / 8 + 1, coord_PR8=n / 16 + 1,
                          decay_rate_times_n=rate, T0=float(T[0]), profile=[(int(j), round(float(t), 5)) for j, t in zip(js, T)]))
pa10a = all(all(x >= 0.4 for x in r_['radial_half_to_n'] if x is not None) for r_ in arows if r_['n'] >= 32)
pa10b = all((r_['PR4'] is None or 0.5 <= r_['PR4'] / r_['coord_PR4'] <= 2) and (r_['PR8'] is None or 0.5 <= r_['PR8'] / r_['coord_PR8'] <= 2) for r_ in arows)
pa10c = all(r_['decay_rate_times_n'] is not None and 1 <= r_['decay_rate_times_n'] <= 4 for r_ in arows)
pa10d = all(r_['sumT_h'] > 1.5 for r_ in arows if r_['n'] >= 16)
pa10e = all(r_['old_share'] < 0.05 for r_ in arows)
# registered aggregation ("medians over nets") for (b), applied after astra's round-2 audit
pa10b_med = True
for n_ in NS:
    rr = [r_ for r_ in arows if r_['n'] == n_]
    if rr:
        m4 = med([r_['PR4'] / r_['coord_PR4'] for r_ in rr if r_['PR4']]); m8 = med([r_['PR8'] / r_['coord_PR8'] for r_ in rr if r_['PR8']])
        pa10b_med &= (m4 is None or 0.5 <= m4 <= 2) and (m8 is None or 0.5 <= m8 <= 2)
out['PA10'] = dict(a=pa10a, b_every_net=pa10b, b=pa10b_med, c=pa10c, d=pa10d, e=pa10e, rows=arows)
md.append('\n| n | seed | sum of total-effect shares (h, a) | share of ages >= n | radial share at ages n/2..n | PR(4) / coord | PR(8) / coord | tail decay rate x n |')
md.append('|---|---|---|---|---|---|---|---|')
for r_ in arows:
    md.append(f"| {r_['n']} | {r_['seed']} | {r_['sumT_h']:.2f}, {r_['sumT_a']:.2f} | {r_['old_share']:.3f} | {f2(r_['radial_half_to_n'])} | "
              f"{r_['PR4']:.1f} / {r_['coord_PR4']:.1f} | {r_['PR8']:.1f} / {r_['coord_PR8']:.1f} | {r_['decay_rate_times_n'] if r_['decay_rate_times_n'] is None else round(r_['decay_rate_times_n'], 2)} |")

# PA12 scale
base = f'{D}/runs/cloud_n32_s1000.json'
if os.path.exists(base):
    b = R(base); devs = []
    for sg in ('0.001', '0.01', '0.1'):
        p = f'{D}/runs/cloud_n32_s1000_sig{sg}.json'
        if os.path.exists(p):
            c = R(p)
            devs.append(max(abs(c['X_quantiles'][i] / b['X_quantiles'][i] - 1) for i in range(3)))
            devs.append(float(np.max(np.abs(np.array(c['evC']) / np.array(b['evC']) - 1))))
    out['PA12'] = dict(hit=bool(devs) and max(devs) <= 1e-8, max_rel_dev=max(devs) if devs else None)

# PR1/PR2
prows = []
for n in (32, 128):
    for s in (1000, 1001):
        rs = {}
        for sg in ('0.001', '0.01', '0.1'):
            p = f'{D}/runs/partR_n{n}_s{s}_sig{sg}.json'
            if os.path.exists(p):
                rs[sg] = R(p)
        if not rs:
            continue
        for L in (4 * n, 16 * n):
            row = dict(n=n, seed=s, L=L)
            for sg, r in rs.items():
                at = r['at'][str(L)]
                row[sg] = dict(births=at['births_mean'], pred=at['births_pred'], ang=at['ang_var_rel_mean'], slope=at['evC_slope'],
                               logR_sd=at['logR_sd'], ev1=at['evC_rel'][0], ev2=at['evC_rel'][1])
            prows.append(row)
pr1 = all(abs(r_[sg]['births'] / r_[sg]['pred'] - 1) <= 0.25 for r_ in prows for sg in ('0.01', '0.1') if sg in r_)
pr2a = all(50 <= r_['0.01']['ang'] / r_['0.001']['ang'] <= 200 for r_ in prows if '0.01' in r_ and '0.001' in r_)
pr2b = all(-1.3 <= r_[sg]['slope'] <= -0.7 for r_ in prows if r_['L'] == 16 * r_['n'] for sg in ('0.001', '0.01', '0.1') if sg in r_)
out['PR1'] = dict(hit=pr1); out['PR2'] = dict(ratio_hit=pr2a, slope_hit=pr2b, rows=prows)
md.append('\n| n | seed | L | sigma: births / pred, angular var (rel), ev1, ev2 (rel), slope, sd log R |')
md.append('|---|---|---|---|')
for r_ in prows:
    cells = '; '.join(f"{sg}: {r_[sg]['births']:.3f}/{r_[sg]['pred']:.3f}, {r_[sg]['ang']:.2e}, {r_[sg]['ev1']:.2e}, {r_[sg]['ev2']:.2e}, {r_[sg]['slope']:.2f}, {r_[sg]['logR_sd']:.3f}"
                      for sg in ('0.001', '0.01', '0.1') if sg in r_)
    md.append(f"| {r_['n']} | {r_['seed']} | {r_['L']} | {cells} |")

os.makedirs(f'{D}/results', exist_ok=True)
json.dump(out, open(f'{D}/results/partA_scores.json', 'w'), indent=1, default=float)
open(f'{D}/results/partA_tables.md', 'w').write('\n'.join(md) + '\n')
for k in ['PA1', 'PA2', 'PA3', 'PA4', 'PA5', 'PA6', 'PA7', 'PA8', 'PA9', 'PA10', 'PA11', 'PA12', 'PR1', 'PR2']:
    v = out.get(k, {})
    print(k, {kk: vv for kk, vv in v.items() if kk not in ('rows',)})
print('\n'.join(md))
