"""Score the registered theory measurements (PREREG P10-P13) and summarise M5 and the post-hoc kernel test.
Writes results/theory_scores.json and prints a summary."""
import json, math, os, glob
import numpy as np

R = os.path.join(os.path.dirname(__file__), '..')
out = {}
# P10 noise stability
p10 = {}
for f in sorted(glob.glob(os.path.join(R, 'runs', 'nstab_n*_K48.json'))):
    r = json.load(open(f))
    rows = r['rows']
    def slope(lo, hi):
        xs = [math.log(x['delta']) for x in rows if lo <= x['delta'] <= hi]
        ys = [math.log(x['D']) for x in rows if lo <= x['delta'] <= hi]
        return float(np.polyfit(xs, ys, 1)[0])
    D = {x['delta']: x['D'] for x in rows}
    p10[f"n{r['n']}_L{r['L']}"] = dict(slope_1e4_1e2=slope(1e-4, 1e-2), slope_1e5_1e3=slope(1e-5, 1e-3), D01=D[0.1],
                                       D1=D[1.0], tail4=r['degree_bounds']['4']['tail_lower'],
                                       worth={k: v['worth_samples_max'] for k, v in r['degree_bounds'].items()})
deep = [v for k, v in p10.items() if k.endswith('_L16') or k.endswith('_L32')]
shallow = [v for k, v in p10.items() if k.endswith('_L4')]
out['P10a'] = dict(slopes=[v['slope_1e4_1e2'] for v in deep], hit=all(0.5 <= v['slope_1e4_1e2'] <= 0.8 for v in deep))
out['P10b'] = dict(slopes=[v['slope_1e5_1e3'] for v in shallow], hit=all(v['slope_1e5_1e3'] >= 0.8 for v in shallow))
out['P10c'] = dict(D01=[v['D01'] for v in deep], tail4=[v['tail4'] for v in deep],
                   hit=all(v['D01'] >= 0.25 and v['tail4'] >= 0.1 for v in deep))
out['P10_detail'] = p10
# P11 harmonic
p11 = {}
for f in sorted(glob.glob(os.path.join(R, 'runs', 'harm_n*_s5000.json'))):
    r = json.load(open(f))
    nets = r['nets']
    fs = [x['f1_J'] + x['f2_J'] for x in nets]
    ceil = [1 / max(1e-9, 1 - x) for x in fs]
    p11[f"n{r['n']}_L{r['L']}"] = dict(f1=[x['f1'] for x in nets], f2=[x['f2_J'] for x in nets], f12=fs,
                                       ceiling=ceil, median_ceiling=float(np.median(ceil)))
ok_range = all(0.3 <= f <= 0.95 for v in p11.values() for f in v['f12'])
c16 = p11.get('n16_L4', {}).get('median_ceiling'); c256 = p11.get('n256_L64', {}).get('median_ceiling')
out['P11'] = dict(detail=p11, in_range=ok_range, c16=c16, c256=c256,
                  hit=bool(ok_range and c16 and c256 and max(c16, c256) / min(c16, c256) < 2))
# P12 spin glass (registered: concave for every L)
sg = json.load(open(os.path.join(R, 'runs', 'spin_glass.json')))
out['P12'] = dict(concave_by_L={k: v['concave'] for k, v in sg.items()}, hit=all(v['concave'] for v in sg.values()))
# P13 regions
fn = os.path.join(R, 'runs', 'regions_n3.json')
if os.path.exists(fn):
    rg = json.load(open(fn))
    Ls = sorted(set(x['L'] for x in rg))
    med = {L: float(np.median([x['patterns'] for x in rg if x['L'] == L])) for L in Ls}
    sl = float(np.polyfit(np.log(Ls), np.log([med[L] for L in Ls]), 1)[0]) if len(Ls) > 1 else None
    out['P13'] = dict(median_patterns=med, loglog_slope=sl, hit=(sl is not None and 1 <= sl <= 3), raw=rg)
# M5 noise effect
fn = os.path.join(R, 'runs', 'noise_effect.json')
if os.path.exists(fn):
    ne = json.load(open(fn))
    out['M5'] = ne
# post-hoc kernel test
ke = {}
for f in sorted(glob.glob(os.path.join(R, 'runs', 'keven_n*.json'))):
    r = json.load(open(f))
    ke[f"n{r['n']}_L{r['L']}"] = dict(even_ge4_over_ang=r['even_ge4_over_ang'], design_gain_bound=r['design_gain_bound'],
                                      ens=r['ens'])
out['posthoc_kernel'] = ke
json.dump(out, open(os.path.join(R, 'results', 'theory_scores.json'), 'w'), indent=1)
for k in ('P10a', 'P10b', 'P10c', 'P11', 'P12', 'P13'):
    if k in out:
        v = {kk: vv for kk, vv in out[k].items() if kk not in ('detail', 'raw')}
        print(k, json.dumps(v)[:400])
if 'M5' in out:
    for x in out['M5']:
        print(f"M5 n={x['n']} L={x['L']} seed={x['seed']}: noise effect nMSE {x['effect_nmse']:.2e} (SE^2 {x['se2_nmse']:.1e})")
for k, v in ke.items():
    print('kernel', k, json.dumps(v)[:300])
