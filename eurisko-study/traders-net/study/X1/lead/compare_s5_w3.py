#!/usr/bin/env python3
"""X1 lead: seed-5 tie-break, derived from X1/w3/compare_w3.py (only the seed and paths changed). Original doc: seed-4 reproduction vs the claimed seeds 1-3 (mean +- 2 sd, sample sd;
paired sd for paired quantities). Seed-1..3 values are recomputed from the original per-seed files
with the workers' own formulas and must equal the claimed aggregates (formula check). Seed-4 values
come from the fresh runs under /tmp/traders-net/X1w3. Output: results/repro.json, results/repro.csv.
usage: python3 compare_w3.py [STUDY_ROOT] [RERUN_ROOT]"""
import csv, json, math, os, sys, datetime
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ST = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath(os.path.join(HERE, '..', '..'))
RR = sys.argv[2] if len(sys.argv) > 2 else '/tmp/traders-net/X1lead'
CL = f'{RR}/clean/study'; OUT = f'{RR}/out'
def _fallback(p):
    """Seed-4 files: use /tmp/traders-net/X1w3 when present, else the synced copy in results/seed4/."""
    if os.path.exists(p) or not str(p).startswith('/tmp/traders-net/X1w3/'): return p
    b = os.path.basename(p); d = os.path.dirname(p)
    pre = {'M6/w1/results': 'M6w1-', 'out/M6w2': 'M6w2-', 'out/M6w4': 'M6w4-', 'out/M6w5': 'M6w5-', 'out/M7w2': 'M7w2-', 'M5/w4/out': ''}
    for k, v in pre.items():
        if d.endswith(k): return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'seed4', v + b)
    return p
J = lambda p: json.load(open(_fallback(p)))
def orig(*p): return os.path.join(ST, *p)

def ms(v):
    v = np.array(v, float); return float(v.mean()), float(v.std(ddof=1))

rows = []
def row(rid, what, s123, s4_getter, claimed_mean, claimed_sd=None, qual=None, qual_text='', evidence='', rerun=''):
    m, sd = ms(s123)
    formula_ok = abs(m - claimed_mean) <= max(1e-4 * abs(claimed_mean), 5e-5) and (claimed_sd is None or abs(sd - claimed_sd) <= max(0.02 * claimed_sd, 5e-5))
    try:
        v4 = float(s4_getter())
    except Exception as e:
        rows.append(dict(id=rid, what=what, seeds123=s123, mean=m, sd=sd, formula_ok=formula_ok, seed4=None,
                         within_2sd=None, z=None, qual_holds=None, qual=qual_text, status='unverified',
                         note='seed-4 run missing: ' + repr(e)[:160], evidence=evidence, rerun=rerun)); return
    within = abs(v4 - m) <= 2 * sd
    z = (v4 - m) / sd if sd > 0 else float('inf')
    q = bool(qual(v4)) if qual else None
    if within and q is not False: st = 'verified'
    elif q is False and not within: st = 'refuted'
    else: st = 'partly verified'
    rows.append(dict(id=rid, what=what, seeds123=[float(x) for x in s123], mean=m, sd=sd, formula_ok=formula_ok,
                     seed4=v4, within_2sd=bool(within), z=float(z), qual_holds=q, qual=qual_text, status=st,
                     evidence=evidence, rerun=rerun))

# ---- M5 probe (w4 aggregate.py: phiL = probes.L.phi_last_half_per_qstep; phiP = probes.P...) ----
for ck, cm, cs, pm, psd in [('P4', 0.0920325, 0.0486378, 0.0016413, 0.000191462), ('P16', 0.11787, 6.63e-4, 0.00173798, 7.9e-5)]:
    f = lambda s, k: J(orig('M5', 'w4', 'out', f'preq-{ck}-win-s{s}.json'))['probes'][k]['phi_last_half_per_qstep']
    g = lambda k: J(f'{CL}/M5/w4/out/preq-{ck}-win-s5.json')['probes'][k]['phi_last_half_per_qstep']
    rr = f'python3 S0/w1/run_w1.py {ck} 4; python3 M5/w4/probes.py --source preq --ckpt {ck} --seed 4'
    row(f'R-M5{"a" if ck == "P4" else "b"}', f'M5 online MLP probe L phi on prequential net {ck} (nats/q-step)',
        [f(s, 'L') for s in (1, 2, 3)], lambda: g('L'), cm, cs, qual=lambda v: v > 0.02,
        qual_text='probe L phi > 0.02 (net exploitable, >> price-only)', evidence=f'M5/w4/out/preq-{ck}-win-s{{1,2,3}}.json; X1/w3/results/seed4/preq-{ck}-win-s4.json', rerun=rr)
    row(f'R-M5c.{ck}', f'M5 price-only probe P phi, {ck}', [f(s, 'P') for s in (1, 2, 3)], lambda: g('P'), pm, None,
        qual=lambda v: 0 < v < 0.1 * 0.09, qual_text='price-only phi positive and < 10% of probe L', evidence=f'M5/w4/out/preq-{ck}-win-s*.json', rerun=rr)

# ---- M5 E_C log-log exponent b (w4 aggregate.py: E_C.loglog) ----
for ck, bm, bs in [('P4', 1.64123, 0.082641), ('P16', 1.27896, 0.0388675)]:
    row(f'R-M5d.{ck}', f'M5 E_C(probes) log-log exponent b, {ck}', [J(orig('M5', 'w4', 'out', f'preq-{ck}-win-s{s}.json'))['E_C']['loglog'] for s in (1, 2, 3)],
        lambda ck=ck: J(f'{CL}/M5/w4/out/preq-{ck}-win-s5.json')['E_C']['loglog'], bm, bs, qual=lambda v: v >= 1.0,
        qual_text='b >= 1 (exploitation linear, not sublinear)', evidence=f'M5/w4/out/preq-{ck}-win-s*.json (E_C.loglog)',
        rerun=f'python3 S0/w1/run_w1.py {ck} 4; python3 M5/w4/probes.py --source preq --ckpt {ck} --seed 4')

# ---- M6 w1 (summarize.py: score.learn_bce, score.humility; ES = A.score_es) ----
w1o = lambda reg, arm, s: J(orig('M6', 'w1', 'results', f'{reg}-{arm}-s{s}.json'))
w1n = lambda reg, arm: J(f'{CL}/M6/w1/results/{reg}-{arm}-s5.json')
d = lambda o, n, k='learn_bce', ko='score', kn='score': o[kn][k] - n[ko][k]
row('R-M6a', 'M6 w1 (b)-(a) learnable BCE, P regime (paired)',
    [w1o('P', 'B', s)['score']['learn_bce'] - w1o('P', 'A', s)['score']['learn_bce'] for s in (1, 2, 3)],
    lambda: w1n('P', 'B')['score']['learn_bce'] - w1n('P', 'A')['score']['learn_bce'], 0.0074, 0.0107,
    qual=lambda v: True, qual_text='tie claim; no sign asserted', evidence='M6/w1/results/P-{A,B}-s*.json',
    rerun='cd M6/w1 && python3 run_w1.py P A 4 && python3 run_w1.py P B 4')
row('R-M6b.B', 'M6 w1 (b)-(a) learnable BCE, ME regime (paired)',
    [w1o('ME', 'B', s)['score']['learn_bce'] - w1o('ME', 'A', s)['score']['learn_bce'] for s in (1, 2, 3)],
    lambda: w1n('ME', 'B')['score']['learn_bce'] - w1n('ME', 'A')['score']['learn_bce'], 0.0008, 0.0012,
    qual=lambda v: True, qual_text='tie claim; no sign asserted', evidence='M6/w1/results/ME-{A,B}-s*.json',
    rerun='cd M6/w1 && python3 run_w1.py ME A 4 && python3 run_w1.py ME B 4')
row('R-M6b.BOOST', 'M6 w1 ME BOOST noise excess (humility)', [w1o('ME', 'BOOST', s)['score']['humility'] for s in (1, 2, 3)],
    lambda: w1n('ME', 'BOOST')['score']['humility'], 0.0017, 0.0008,
    qual=lambda v: v < w1n('ME', 'A')['score']['humility'], qual_text='BOOST noise below (a) on the same seed',
    evidence='M6/w1/results/ME-BOOST-s*.json', rerun='cd M6/w1 && python3 run_w1.py ME BOOST 4')
row('R-M6b.A', 'M6 w1 ME (a) noise excess (humility)', [w1o('ME', 'A', s)['score']['humility'] for s in (1, 2, 3)],
    lambda: w1n('ME', 'A')['score']['humility'], 0.0108, 0.0006, qual=lambda v: v > 0.005, qual_text='(a) over-confident on noise',
    evidence='M6/w1/results/ME-A-s*.json', rerun='cd M6/w1 && python3 run_w1.py ME A 4')
row('R-M6b.ES', 'M6 w1 ME ES minus (a), learnable BCE (paired)',
    [w1o('ME', 'A', s)['score_es']['learn_bce'] - w1o('ME', 'A', s)['score']['learn_bce'] for s in (1, 2, 3)],
    lambda: w1n('ME', 'A')['score_es']['learn_bce'] - w1n('ME', 'A')['score']['learn_bce'], -0.0050, 0.0004,
    qual=lambda v: v < 0, qual_text='ES beats (a)', evidence='M6/w1/results/ME-A-s*.json (score_es)', rerun='cd M6/w1 && python3 run_w1.py ME A 4')

# ---- M6 w2 harness (lead/merge.py w2rec) ----
FAM = ["periodic", "lfsr16", "ca30", "ca110", "prime", "prng", "and", "or", "periodic_ho"]
LEARN = [FAM.index(f) for f in ["periodic", "lfsr16", "ca110", "prime", "and", "or", "periodic_ho"]]
NOISE = [FAM.index("ca30"), FAM.index("prng")]
def w2rec(p):
    b = np.array(J(p)['bce_all']); return float(np.nanmean(b[LEARN])), float(np.max(b[NOISE] - math.log(2))), float(b[FAM.index('lfsr16')])
cg = [w2rec(orig('M6', 'w2', 'results', f'cg-s{s}.json')) for s in (1, 2, 3)]
bo = [w2rec(orig('M6', 'lead', 'results', f'boost-s{s}.json')) for s in (1, 2, 3)]
cg4 = lambda: w2rec(f'{OUT}/M6w2/cg-s5.json'); bo4 = lambda: w2rec(f'{OUT}/M6w2/boost-s5.json')
row('R-M6c.cg', 'M6 w2 cg noise excess', [x[1] for x in cg], lambda: cg4()[1], float(np.mean([x[1] for x in cg])),
    qual=lambda v: v < bo4()[1], qual_text='cg noise below BOOST on the same seed', evidence='M6/w2/results/cg-s*.json',
    rerun='cd M6/w2 && python3 pool.py --arm cg --seed 4 --T 60000')
row('R-M6c.boost', 'M6 w2-harness BOOST noise excess', [x[1] for x in bo], lambda: bo4()[1], 0.0036075497934602665, 0.0003543899049597863,
    evidence='M6/lead/results/boost-s*.json', rerun='cd M6/lead && python3 pool_kc.py --arm boost --seed 4 --T 60000')
row('R-M6c.paired', 'M6 w2 cg minus BOOST noise excess (paired)', [a[1] - b[1] for a, b in zip(cg, bo)], lambda: cg4()[1] - bo4()[1],
    float(np.mean([a[1] - b[1] for a, b in zip(cg, bo)])), qual=lambda v: v < 0, qual_text='cg < BOOST (the survival margin)',
    evidence='M6/w2/results/cg-s*.json, M6/lead/results/boost-s*.json', rerun='both of the above')

# ---- M6 w4 (summarize.py: fp_vs_pass2_M3_rel on li stream; Newton iters on base) ----
w4o = lambda st, arm, s: J(orig('M6', 'w4', 'results', f'{st}-{arm}-s{s}.json'))
w4n = lambda st, arm: J(f'{OUT}/M6w4/{st}-{arm}-s5.json')
row('R-M6d', 'M6 w4 fixed point vs 2-pass, M3 D-coherence rel (li stream)',
    [w4o('li', 'fp', s)['M3_D_coherence'] / w4o('li', 'pass2', s)['M3_D_coherence'] - 1 for s in (1, 2, 3)],
    lambda: w4n('li', 'fp')['M3_D_coherence'] / w4n('li', 'pass2')['M3_D_coherence'] - 1, float(np.mean([0.0016580873861067325, 0.0012143830036852599, -0.0008212613904159394])),
    qual=lambda v: v > -0.5, qual_text='fixed point NOT kept (needs <= -50%)', evidence='M6/w4/results/li-{fp,pass2}-s*.json',
    rerun='cd M6/w4 && python3 fp_layer.py li fp 4 && python3 fp_layer.py li pass2 4')
row('R-M6d.newton', 'M6 w4 Newton iterations (base)', [w4o('base', 'fp', s)['fp']['iters_mean'] for s in (1, 2, 3)],
    lambda: w4n('base', 'fp')['fp']['iters_mean'], 2.05333, 0.0793872, qual=lambda v: v < 8, qual_text='converges below the 8-iteration cap',
    evidence='M6/w4/results/base-fp-s*.json', rerun='cd M6/w4 && python3 fp_layer.py base fp 4')

# ---- M6 w5 (aggregate.py: rel_bce_vs_U = Cm.learn_bce / U.learn_bce - 1) ----
w5o = lambda arm, s: J(orig('M6', 'w5', 'results', f'{arm}-s{s}.json'))
w5n = lambda arm: J(f'{OUT}/M6w5/{arm}-s5.json')
row('R-M6e.rel', 'M6 w5 Cm vs U learnable BCE rel (paired)', [w5o('Cm', s)['learn_bce'] / w5o('U', s)['learn_bce'] - 1 for s in (1, 2, 3)],
    lambda: w5n('Cm')['learn_bce'] / w5n('U')['learn_bce'] - 1, float(np.mean([-0.0030294682207481216, -0.013099275330847093, -0.009303250738804225])),
    qual=lambda v: v < 0.01, qual_text='Cm not worse than U by 1% (not dropped)', evidence='M6/w5/results/{Cm,U}-s*.json',
    rerun='cd M6/w5 && python3 commit.py --arm Cm --seed 4 && python3 commit.py --arm U --seed 4')
row('R-M6e.corr', 'M6 w5 Cm head clone correlation', [w5o('Cm', s)['head_corr_mean'] for s in (1, 2, 3)],
    lambda: w5n('Cm')['head_corr_mean'], 0.9934161345480419, 0.001486874377824803, qual=lambda v: v > 0.95,
    qual_text='heads are clones (corr > 0.95)', evidence='M6/w5/results/Cm-s*.json', rerun='cd M6/w5 && python3 commit.py --arm Cm --seed 4')

# ---- M7 w2 (merge_m7.py: d = dsfw - ds3 per seed, bce_all) ----
m7o = lambda arm, E, s: J(orig('M7', 'w2', 'results', 'all', f'{arm}-E{E}-s{s}.json'))
m7n = lambda arm, E: J(f'{OUT}/M7w2/{arm}-E{E}-s5.json')
for E, cm, cs, rid in [(8, -0.00512, 0.00095, 'R-M7a'), (64, -0.00724, 0.00492, 'R-M7b')]:
    row(rid, f'M7 dsfw minus ds3 bce_all, E={E} (paired)', [m7o('dsfw', E, s)['bce_all'] - m7o('ds3', E, s)['bce_all'] for s in (1, 2, 3)],
        lambda E=E: m7n('dsfw', E)['bce_all'] - m7n('ds3', E)['bce_all'], cm, cs, qual=lambda v: v < 0,
        qual_text='dsfw beats DeepSeek ds3 on the seed', evidence=f'M7/w2/results/all/{{dsfw,ds3}}-E{E}-s*.json',
        rerun=f'cd M7/w2 && python3 moe_w2.py --arm dsfw --E {E} --seed 4 --T 24000 --B 8 && python3 moe_w2.py --arm ds3 --E {E} --seed 4 --T 24000 --B 8')
    row(rid + '.postshift', f'M7 dsfw minus ds3 post-shift BCE, E={E} (paired)', [m7o('dsfw', E, s)['bce_post_shift'] - m7o('ds3', E, s)['bce_post_shift'] for s in (1, 2, 3)],
        lambda E=E: m7n('dsfw', E)['bce_post_shift'] - m7n('ds3', E)['bce_post_shift'],
        float(np.mean([m7o('dsfw', E, s)['bce_post_shift'] - m7o('ds3', E, s)['bce_post_shift'] for s in (1, 2, 3)])),
        qual=lambda v: v > 0, qual_text='dsfw worse after the shift (cost claim)', evidence=f'M7/w2/results/all/{{dsfw,ds3}}-E{E}-s*.json', rerun='as above')

now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
json.dump(dict(checked_at=now, rule='seed4 within mean+-2sd of seeds 1-3 (sample sd) and qualitative claim holds', rows=rows),
          open(os.path.join(HERE, 'results', 'repro_s5_w3.json'), 'w'), indent=1)
with open(os.path.join(HERE, 'results', 'repro_s5_w3.csv'), 'w') as fh:
    w = csv.writer(fh); w.writerow(['id', 'status', 'mean', 'sd', 'seed4', 'z', 'within_2sd', 'qual_holds', 'formula_ok', 'what'])
    for r in rows:
        w.writerow([r['id'], r['status'], f"{r['mean']:.6g}", f"{r['sd']:.3g}", '' if r['seed4'] is None else f"{r['seed4']:.6g}",
                    '' if r['z'] is None else f"{r['z']:.2f}", r['within_2sd'], r['qual_holds'], r['formula_ok'], r['what']])
for r in rows:
    print(f"{r['id']:16s} {r['status']:16s} m={r['mean']:.5g} sd={r['sd']:.3g} s4={r['seed4']} z={r['z']} f_ok={r['formula_ok']}")
