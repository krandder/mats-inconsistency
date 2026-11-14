#!/usr/bin/env python3
"""X1 worker 3, part A: re-read every M5/M6/M7 number quoted in the orchestrator
summary from the file it cites. Read-only. Output: results/static_check.json.

Usage: python3 check_w3.py [STUDY_ROOT]   (default: ../../ relative to this file)
Tolerance: half a unit in the last quoted digit (the quote is a rounding), plus 1e-12.
"""
import csv, json, os, re, sys, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.abspath(os.path.join(HERE, '..', '..'))
P = lambda *a: os.path.join(ROOT, *a)

def tol_of(s):
    s = s.strip().lstrip('+-')
    m = re.match(r'^(\d*)(?:\.(\d*))?(?:e([+-]?\d+))?$', s)
    dec = len(m.group(2) or '')
    exp = int(m.group(3) or 0)
    return 0.5 * 10 ** (exp - dec) + 1e-12

_c = {}
def rows(path):
    if path not in _c:
        _c[path] = list(csv.DictReader(open(P(path))))
    return _c[path]
def js(path):
    if path not in _c:
        _c[path] = json.load(open(P(path)))
    return _c[path]
def csvget(path, col, **kw):
    hit = [r for r in rows(path) if all(str(r[k]) == str(v) for k, v in kw.items())]
    assert len(hit) == 1, (path, kw, len(hit))
    return float(hit[0][col])
def hl(i, col='mean'):
    return csvget('M5/results/headline.csv', col, id=i)

C = []  # (id, ws, text, claimed_str, getter, scale)
def add(cid, ws, text, claimed, getter, scale=1.0):
    C.append((cid, ws, text, claimed, getter, scale))

# ---------------- M5 ----------------
add('M5.P4.mlp.phi', 'M5', 'P4 online MLP probe phi mean', '0.092', lambda: hl('P4.mlp.phi'))
add('M5.P4.mlp.phi.sd', 'M5', 'P4 online MLP probe phi sd', '0.0486', lambda: hl('P4.mlp.phi', 'sd'))
add('M5.P16.mlp.phi', 'M5', 'P16 online MLP probe phi mean', '0.118', lambda: hl('P16.mlp.phi'))
add('M5.P16.mlp.phi.sd', 'M5', 'P16 online MLP probe phi sd', '6.63e-04', lambda: hl('P16.mlp.phi', 'sd'))
add('M5.P4.mlp.b', 'M5', 'P4 E_C log-log b', '1.64', lambda: hl('P4.mlp.b'))
add('M5.P16.mlp.b', 'M5', 'P16 E_C log-log b', '1.28', lambda: hl('P16.mlp.b'))
add('M5.P4.cal.phi', 'M5', 'P4 cal-online-fam phi', '8.26e-04', lambda: hl('P4.cal.phi'))
add('M5.P4.cal.phi.sd', 'M5', 'P4 cal phi sd', '1.11e-04', lambda: hl('P4.cal.phi', 'sd'))
add('M5.P16.cal.phi', 'M5', 'P16 cal phi', '0.00142', lambda: hl('P16.cal.phi'))
add('M5.P16.cal.phi.sd', 'M5', 'P16 cal phi sd', '8.89e-05', lambda: hl('P16.cal.phi', 'sd'))
add('M5.P4.cal.b', 'M5', 'P4 fixed-bin b', '0.535', lambda: hl('P4.cal.b'))
add('M5.P16.cal.b', 'M5', 'P16 fixed-bin b', '0.475', lambda: hl('P16.cal.b'))
add('M5.P4.mlp.P', 'M5', 'P4 price-only probe', '0.00164', lambda: hl('P4.mlp.P'))
add('M5.P16.mlp.P', 'M5', 'P16 price-only probe', '0.00174', lambda: hl('P16.mlp.P'))
add('M5.P4.pat.phi', 'M5', 'P4 best pattern phi', '-0.00151', lambda: hl('P4.pat.phi'))
add('M5.P4.pat.phi.sd', 'M5', 'P4 pattern phi sd', '3.27e-04', lambda: hl('P4.pat.phi', 'sd'))
add('M5.P16.pat.phi', 'M5', 'P16 best pattern phi', '-0.00108', lambda: hl('P16.pat.phi'))
add('M5.P16.pat.phi.sd', 'M5', 'P16 pattern phi sd', '2.42e-04', lambda: hl('P16.pat.phi', 'sd'))
add('M5.P4.pat.share', 'M5', 'P4 one-share pattern', '0.00761', lambda: hl('P4.pat.share'))
add('M5.P16.pat.share', 'M5', 'P16 one-share pattern', '0.00922', lambda: hl('P16.pat.share'))
for k, v, s in [('taut_or', '0.477', '0.183'), ('impl', '0.545', '0.0758'), ('contra', '0.536', '0.0689')]:
    add(f'M5.li-net.{k}.pre', 'M5', f'net {k} violation before resolution', v, lambda k=k: hl(f'li-net.{k}.pre'))
    add(f'M5.li-net.{k}.pre.sd', 'M5', f'net {k} violation sd', s, lambda k=k: hl(f'li-net.{k}.pre', 'sd'))
    add(f'M5.li-logistic.{k}.pre', 'M5', f'logistic {k} violation (uninformed)', '0.5', lambda k=k: hl(f'li-logistic.{k}.pre'))
for k, b1, b2 in [('and', '0.13', '0.231'), ('or', '0.298', '0.387'), ('ex3', '0.196', '0.193'), ('twin', '0.74', '0.586')]:
    add(f'M5.li-net.{k}.B1', 'M5', f'{k} bundle arb B1', b1, lambda k=k: hl(f'li-net.{k}.B1'))
    add(f'M5.li-net.{k}.B2', 'M5', f'{k} bundle arb B2', b2, lambda k=k: hl(f'li-net.{k}.B2'))
for i, v in [('WN8k.cal.phi', '0.126'), ('WN16k.cal.phi', '0.16'), ('WN30k.cal.phi', '0.192'),
             ('WN8k.pat.phi', '0.179'), ('WN20k.pat.phi', '0.281'), ('WNfinal.pat.phi', '0.339'),
             ('WN8k.mlp.phi', '0.305'), ('WN24k.mlp.phi', '0.41'), ('WNfinal.mlp.phi', '0.44'),
             ('WNfinal.mlp.P', '0.439')]:
    add('M5.' + i, 'M5', 'frozen wealth-net ' + i, v, lambda i=i: hl(i))
for i, v in [('clip.bce_raw', '0.972'), ('clip.bce_p1e-7', '0.886'), ('clip.bce_p1e-4', '0.73'),
             ('clip.bce_logit8', '0.687'), ('clip.bce_raw_2nd', '1.07')]:
    add('M5.' + i, 'M5', 'held-out BCE, clip variant ' + i, v, lambda i=i: hl(i))
add('M5.clip.bce_raw.sd', 'M5', 'raw BCE sd', '0.0219', lambda: hl('clip.bce_raw', 'sd'))
for i, v in [('soup.k12-f12.pat4', '0.128'), ('soup.k12-f10.pat4', '0.1'), ('soup.k14-f10.pat4', '0.00671'),
             ('soup.k14-f8.pat4', '-0.0148')]:
    add('M5.' + i, 'M5', 'soup sh_pat4 ' + i, v, lambda i=i: hl(i))
add('M5.soup.arb.min', 'M5', 'soup sh_arb min over cells', '0.0468',
    lambda: min(float(r['mean']) for r in rows('M5/results/headline.csv') if r['id'].startswith('soup.') and r['id'].endswith('.arb')))
add('M5.soup.arb.max', 'M5', 'soup sh_arb max over cells', '0.175',
    lambda: max(float(r['mean']) for r in rows('M5/results/headline.csv') if r['id'].startswith('soup.') and r['id'].endswith('.arb')))

# ---------------- M6 ----------------
V = lambda: js('M6/lead/M6-verdict.json')
def paired(arm_pair, metric):
    return [r for r in rows('M6/w1/results/PAIRED-w1.csv')]
add('M6.gateB', 'M6', 'base_arms_survive', '1', lambda: float(bool(V()['base_arms_survive'])))
def surv_count():
    v = V()
    for k in ('surviving', 'survivors', 'survive'):
        if k in v: return float(len(v[k]))
    return float(sum(1 for a in v['arms'].values() if isinstance(a, dict) and a.get('survives', a.get('survive')) is True))
add('M6.n_survive', 'M6', 'arms surviving', '10', surv_count)
add('M6.n_dropped', 'M6', 'arms dropped', '5', lambda: float(sum(1 for a in V()['arms'].values() if a.get('dropped') is True)))
add('M6.n_undecided', 'M6', 'arms undecided', '0', lambda: float(len(V()['undecided'])))
C_M6_W1 = 'M6/w1/results/PAIRED-w1.csv'
M6_BCE_LIST = []  # filled by probing file below
def w1p(stream, contrast, col):
    rs = rows(C_M6_W1)
    hit = [r for r in rs if r.get('stream') == stream and contrast in (r.get('contrast') or r.get('comparison') or '')]
    assert len(hit) >= 1, (stream, contrast)
    return float(hit[0][col])
# PAIRED-w1.csv layout probed at run time (see M6_W1_ROWS below)
pw = lambda reg, arm, metric, col: csvget('M6/w1/results/PAIRED-w1.csv', col, regime=reg, arm=arm, metric=metric)
sw = lambda reg, arm, metric, col: csvget('M6/w1/results/SUMMARY-w1.csv', col, regime=reg, arm=arm, metric=metric)
add('M6.w1.P.b_minus_a', 'M6', '(b)-(a) learnable BCE, P', '+0.0074', lambda: pw('P', 'B', 'learn_bce', 'diff_mean'))
add('M6.w1.P.b_minus_a.sd', 'M6', '(b)-(a) sd, P', '0.0107', lambda: pw('P', 'B', 'learn_bce', 'diff_sd'))
add('M6.w1.ME.b_minus_a', 'M6', '(b)-(a) learnable BCE, ME', '+0.0008', lambda: pw('ME', 'B', 'learn_bce', 'diff_mean'))
add('M6.w1.ME.b_minus_a.sd', 'M6', '(b)-(a) sd, ME', '0.0012', lambda: pw('ME', 'B', 'learn_bce', 'diff_sd'))
add('M6.w1.ME.BOOST.noise', 'M6', 'ME BOOST noise excess (humility)', '0.0017', lambda: sw('ME', 'BOOST', 'humility', 'mean'))
add('M6.w1.ME.BOOST.noise.sd', 'M6', 'ME BOOST noise excess sd', '0.0008', lambda: sw('ME', 'BOOST', 'humility', 'sd'))
add('M6.w1.ME.A.noise', 'M6', 'ME (a) noise excess', '0.0108', lambda: sw('ME', 'A', 'humility', 'mean'))
add('M6.w1.ME.A.noise.sd', 'M6', 'ME (a) noise excess sd', '0.0006', lambda: sw('ME', 'A', 'humility', 'sd'))
add('M6.w1.ME.ES.learn', 'M6', 'ME ES learnable vs (a)', '-0.0050', lambda: pw('ME', 'ES', 'learn_bce', 'diff_mean'))
add('M6.w1.ME.ES.learn.sd', 'M6', 'ME ES learnable vs (a) sd', '0.0004', lambda: pw('ME', 'ES', 'learn_bce', 'diff_sd'))
add('M6.w1.ME.ES.3of3', 'M6', 'ES below (a) in n seeds', '3', lambda: float(sum(1 for v in [r for r in rows('M6/w1/results/PAIRED-w1.csv') if r['regime']=='ME' and r['arm']=='ES' and r['metric']=='learn_bce'][0]['values'].split('/') if float(v) < 0)))
va = lambda *ks: (lambda d: [d := d[k] for k in ks][-1])(V())
for j, v in enumerate(['0.0019', '0.0023', '0.0023']):
    add(f'M6.w2.cg.noise.s{j+1}', 'M6', 'cg noise per seed', v, lambda j=j: va('arms', 'cg', 'noise_per_seed')[j])
for j, v in enumerate(['0.0033', '0.0035', '0.0040']):
    add(f'M6.w2.boost.noise.s{j+1}', 'M6', 'w2-harness BOOST noise per seed', v, lambda j=j: va('w2_harness_kill_controls', 'boost', 'per_seed_noise')[j])
add('M6.w2.cg_below_boost_3of3', 'M6', 'cg noise below BOOST in n seeds', '3', lambda: float(sum(a < b for a, b in zip(va('arms', 'cg', 'noise_per_seed'), va('w2_harness_kill_controls', 'boost', 'per_seed_noise')))))
for j, v in enumerate(['35', '40', '44']):
    add(f'M6.w2.cg.rel_vs_a.s{j+1}', 'M6', 'cg learnable BCE vs (a), %', v, lambda j=j: 100 * va('arms', 'cg', 'rel_vs_a')[j])
for k, v in [('learn_mean_sd', ('0.195', '0.055')), ('noise_mean_sd', ('0.0094', '0.0006'))]:
    add(f'M6.w2.wd.{k}', 'M6', 'w2-harness WD ' + k, v[0], lambda k=k: va('w2_harness_kill_controls', 'wd', k)[0])
    add(f'M6.w2.wd.{k}.sd', 'M6', 'w2-harness WD ' + k + ' sd', v[1], lambda k=k: va('w2_harness_kill_controls', 'wd', k)[1])
for k, v in [('learn_mean_sd', ('0.215', '0.039')), ('noise_mean_sd', ('0.0036', '0.0004'))]:
    add(f'M6.w2.boost.{k}', 'M6', 'w2-harness BOOST ' + k, v[0], lambda k=k: va('w2_harness_kill_controls', 'boost', k)[0])
    add(f'M6.w2.boost.{k}.sd', 'M6', 'w2-harness BOOST ' + k + ' sd', v[1], lambda k=k: va('w2_harness_kill_controls', 'boost', k)[1])
for arm, vals in [('cg', ['0.1779', '0.1339', '0.1561']), ('boost', ['0.1806', '0.1359', '0.1528'])]:
    for j, v in enumerate(vals):
        add(f'M6.w2.exlfsr.{arm}.s{j+1}', 'M6', f'{arm} learnable BCE excluding LFSR16', v, lambda arm=arm, j=j: va('w2_harness_exlfsr', arm, 'learn_exlfsr')[j])
w3s = lambda a, k, j: js('M6/w3/SUMMARY-w3.json')['arms'][a][k][j]
add('M6.w3.elog.noise', 'M6', 'elog noise excess mean', '0.0140', lambda: w3s('elog', 'noise_excess', 0))
add('M6.w3.elog.noise.sd', 'M6', 'elog noise excess sd', '0.0013', lambda: w3s('elog', 'noise_excess', 1))
add('M6.w3.eexp.noise', 'M6', 'eexp noise excess mean', '0.0150', lambda: w3s('eexp', 'noise_excess', 0))
add('M6.w3.eexp.noise.sd', 'M6', 'eexp noise excess sd', '0.0004', lambda: w3s('eexp', 'noise_excess', 1))
add('M6.w3.fb01.noise', 'M6', 'fb01 noise excess mean', '0.0031', lambda: w3s('fb01', 'noise_excess', 0))
add('M6.w3.fb01.noise.sd', 'M6', 'fb01 noise excess sd', '0.0002', lambda: w3s('fb01', 'noise_excess', 1))
for j, v in enumerate(['8.1', '6.7', '7.3']):
    add(f'M6.w3.fb01.rel_vs_b.s{j+1}', 'M6', 'fb01 BCE vs (b), %', v,
        lambda j=j: 100 * js('M6/lead/w3_detail.json')['fb01']['rel_vs_b'][j])
w4v = lambda: js('M6/w4/VERDICT.json')
for j, v in enumerate(['0.17', '0.12', '-0.08']):
    add(f'M6.w4.fp_vs_pass2.s{j+1}', 'M6', 'fixed point vs 2-pass violations, %', v,
        lambda j=j: 100 * w4v()['fp_vs_pass2_M3_rel'][j])
add('M6.w4.keep_fp', 'M6', 'keep_fp flag', '0', lambda: float(bool(w4v()['keep_fp'])))
add('M6.w4.newton', 'M6', 'Newton iterations mean', '2.05', lambda: csvget('M6/w4/SUMMARY.csv', 'mean', stream='base', arm='fp', metric='fp:iters_mean'))
add('M6.w4.newton.sd', 'M6', 'Newton iterations sd', '0.08', lambda: csvget('M6/w4/SUMMARY.csv', 'sd', stream='base', arm='fp', metric='fp:iters_mean'))
sr = lambda a, j: 100 * js('M6/w5/results/stoprule.json')[a]['rel_bce_vs_U'][j]
for a in ['Cm', 'Cw']:
    for j, v in enumerate(['-0.3', '-1.3', '-0.9']):
        add(f'M6.w5.{a}.rel_vs_U.s{j+1}', 'M6', f'{a} BCE vs U, %', v, lambda a=a, j=j: sr(a, j))
for j, v in enumerate(['1.8', '1.3', '2.0']):
    add(f'M6.w5.Cs.rel_vs_U.s{j+1}', 'M6', 'Cs BCE vs U, %', v, lambda j=j: sr('Cs', j))
add('M6.w5.Cm.head_corr', 'M6', 'Cm head clone correlation', '0.9934', lambda: csvget('M6/w5/results/summary.csv', 'mean', arm='Cm', metric='head_corr_mean'))
add('M6.w5.Cm.head_corr.sd', 'M6', 'Cm head clone correlation sd', '0.0015', lambda: csvget('M6/w5/results/summary.csv', 'sd', arm='Cm', metric='head_corr_mean'))
add('M6.w5.Cm.spearman_K', 'M6', 'Spearman wealth vs marginal value (Cm, K)', '-0.0016', lambda: csvget('M6/w5/results/summary.csv', 'mean', arm='Cm', metric='measures_vs_oracle.K.spearman'))
add('M6.w5.Cm.spearman_K.sd', 'M6', 'Spearman sd', '0.0221', lambda: csvget('M6/w5/results/summary.csv', 'sd', arm='Cm', metric='measures_vs_oracle.K.spearman'))

# ---------------- M7 ----------------
def arms(col, **kw):
    kw = {k: str(v) for k, v in kw.items()}
    return csvget('M7/merge/arms.csv', col, **kw)
add('M7.dsfw.E8.rel', 'M7', 'dsfw vs ds3 bce_all E8, %', '-1.30', lambda: arms('rel_vs_ds_pct', worker='w2', metric='bce_all', E=8, arm='dsfw'))
add('M7.dsfw.E8.sd', 'M7', 'dsfw vs ds3 paired sd E8', '0.0009', lambda: arms('d_vs_ds_sd', worker='w2', metric='bce_all', E=8, arm='dsfw'))
add('M7.dsfw.E8.wins', 'M7', 'dsfw wins E8', '3', lambda: arms('wins_vs_ds', worker='w2', metric='bce_all', E=8, arm='dsfw'))
add('M7.dsfw.E64.rel', 'M7', 'dsfw vs ds3 bce_all E64, %', '-1.87', lambda: arms('rel_vs_ds_pct', worker='w2', metric='bce_all', E=64, arm='dsfw'))
add('M7.dsfw.E64.sd', 'M7', 'dsfw vs ds3 paired sd E64', '0.0049', lambda: arms('d_vs_ds_sd', worker='w2', metric='bce_all', E=64, arm='dsfw'))
add('M7.dsfw.E64.wins', 'M7', 'dsfw wins E64', '3', lambda: arms('wins_vs_ds', worker='w2', metric='bce_all', E=64, arm='dsfw'))
add('M7.dsfw.E8.maxload', 'M7', 'dsfw max load worst E8', '0.24', lambda: arms('max_load_worst_mean', worker='w2', metric='bce_all', E=8, arm='dsfw'))
add('M7.dsfw.E64.maxload', 'M7', 'dsfw max load worst E64', '0.11', lambda: arms('max_load_worst_mean', worker='w2', metric='bce_all', E=64, arm='dsfw'))
add('M7.dsfw.last10k.E64.rel', 'M7', 'dsfw bce_last10k E64, %', '-1.33', lambda: arms('rel_vs_ds_pct', worker='w2', metric='bce_last10k', E=64, arm='dsfw'))
add('M7.dsfw.last10k.E64.wins', 'M7', 'dsfw bce_last10k E64 wins', '2', lambda: arms('wins_vs_ds', worker='w2', metric='bce_last10k', E=64, arm='dsfw'))
add('M7.dsfw.E64.wins_vs_none', 'M7', 'dsfw vs none E64 wins', '2', lambda: arms('wins_vs_none', worker='w2', metric='bce_all', E=64, arm='dsfw'))
def ps(E, fn):
    v = [float(r['diff']) for r in rows('M7/w2/results/paired.csv') if r['arm'] == 'dsfw' and r['E'] == str(E) and r['metric'] == 'bce_post_shift']
    assert len(v) == 3
    return fn(v)
add('M7.dsfw.postshift.E8.min', 'M7', 'dsfw post-shift delta min E8', '0.041', lambda: ps(8, min))
add('M7.dsfw.postshift.E8.max', 'M7', 'dsfw post-shift delta max E8', '0.054', lambda: ps(8, max))
add('M7.dsfw.postshift.E64.min', 'M7', 'dsfw post-shift delta min E64', '0.047', lambda: ps(64, min))
add('M7.dsfw.postshift.E64.max', 'M7', 'dsfw post-shift delta max E64', '0.092', lambda: ps(64, max))
add('M7.exp3.E8.maxload', 'M7', 'exp3 max load worst E8', '0.775', lambda: arms('max_load_worst_mean', worker='w3', metric='bce_2nd_half', E=8, arm='exp3'))
def w3m(arm, E, metric, col):
    return csvget('M7/w3/results/summary.csv', col, arm=arm, E=E, metric=metric)
add('M7.exp3.E8.maxload.sd', 'M7', 'exp3 max load worst sd E8 (w3 summary)', '0.074', lambda: w3m('exp3', 8, 'max_load_worst_layer', 'sd'))
add('M7.exp3.E8.maxload.w3', 'M7', 'exp3 max load worst E8 (w3 summary)', '0.775', lambda: w3m('exp3', 8, 'max_load_worst_layer', 'mean'))
add('M7.ds.E8.maxload', 'M7', 'DeepSeek max load worst E8 (w3 summary)', '0.368', lambda: w3m('deepseek', 8, 'max_load_worst_layer', 'mean'))
add('M7.ds.E8.maxload.sd', 'M7', 'DeepSeek max load worst sd E8 (w3 summary)', '0.054', lambda: w3m('deepseek', 8, 'max_load_worst_layer', 'sd'))
add('M7.exp3.E8.collapsed', 'M7', 'exp3 collapsed seeds E8', '3', lambda: arms('collapsed_seeds', worker='w3', metric='bce_2nd_half', E=8, arm='exp3'))
add('M7.global.E8.maxload', 'M7', 'global prior b1000 max load E8', '1.00', lambda: csvget('M7/w4/summary.csv', 'max_load_mean', E=8, arm='global', beta='1000.0'))
add('M7.global.E64.maxload', 'M7', 'global prior b1000 max load E64', '1.00', lambda: csvget('M7/w4/summary.csv', 'max_load_mean', E=64, arm='global', beta='1000.0'))
add('M7.global.E8.mi', 'M7', 'global prior b1000 MI bits E8', '0', lambda: csvget('M7/w4/summary.csv', 'mi_bits_mean', E=8, arm='global', beta='1000.0'))
add('M7.global.E64.mi', 'M7', 'global prior b1000 MI bits E64', '0', lambda: csvget('M7/w4/summary.csv', 'mi_bits_mean', E=64, arm='global', beta='1000.0'))
add('M7.bb.E8.collapsed', 'M7', 'bb collapsed seeds E8', '3', lambda: arms('collapsed_seeds', worker='w5', metric='bce_2nd_half', E=8, arm='bb'))
add('M7.bb.E8.maxload', 'M7', 'bb max load E8', '0.57', lambda: arms('max_load_worst_mean', worker='w5', metric='bce_2nd_half', E=8, arm='bb'))
add('M7.bb.E64.rel', 'M7', 'bb vs DeepSeek E64, %', '+4.54', lambda: arms('rel_vs_ds_pct', worker='w5', metric='bce_2nd_half', E=64, arm='bb'))
add('M7.bbfresh.E64.rel', 'M7', 'bb-fresh vs DeepSeek E64, %', '+2.82', lambda: arms('rel_vs_ds_pct', worker='w5', metric='bce_2nd_half', E=64, arm='bb-fresh'))
add('M7.none.E8.rel.w1', 'M7', 'none vs DeepSeek E8 (w1), %', '-0.40', lambda: arms('rel_vs_ds_pct', worker='w1', metric='bce_2nd_half', E=8, arm='none'))
add('M7.none.E8.wins.w1', 'M7', 'none wins E8 (w1)', '3', lambda: arms('wins_vs_ds', worker='w1', metric='bce_2nd_half', E=8, arm='none'))
add('M7.none.E64.rel.w4', 'M7', 'none vs DeepSeek E64 (w4), %', '-0.71', lambda: arms('rel_vs_ds_pct', worker='w4', metric='bce_last', E=64, arm='none'))
add('M7.none.E64.wins.w4', 'M7', 'none wins E64 (w4)', '3', lambda: arms('wins_vs_ds', worker='w4', metric='bce_last', E=64, arm='none'))
ctl = lambda arm, col: csvget('M7/w1/results/controls.csv', col, arm=arm, E=8)
add('M7.dense.E8', 'M7', 'dense FLOP-matched E8 bce_2nd_half (w1 controls)', '0.3083', lambda: ctl('dense', 'bce_2nd_half_mean'))
add('M7.dense.E8.sd', 'M7', 'dense sd (w1 controls)', '0.0236', lambda: ctl('dense', 'bce_2nd_half_sd'))
add('M7.ds.E8.bce2h', 'M7', 'DeepSeek u1e-3 E8 bce_2nd_half (w1 controls)', '0.3314', lambda: ctl('deepseek-u1e-3', 'bce_2nd_half_mean'))
add('M7.ds.E8.bce2h.sd', 'M7', 'DeepSeek u1e-3 E8 bce_2nd_half sd (w1 controls)', '0.0197', lambda: ctl('deepseek-u1e-3', 'bce_2nd_half_sd'))
def stage0_max():
    d = js('M7/w1/results/stage0.json')
    vals = []
    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == 'max_rel_err' and isinstance(v, (int, float)):
                    vals.append(v)
                walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(d)
    return max(vals)
add('M7.stage0.max_rel_err', 'M7', 'stage 0 max rel err', '1.13e-15', stage0_max)
def gate():
    return js('M7/merge/gate.json')
def cost_sum(ws, lo, hi):
    lines = open(P('G1/COST.md')).read().splitlines()
    tot = 0.0
    for ln in lines[lo - 1:hi]:
        cells = [c.strip() for c in ln.strip('|').split('|')]
        if len(cells) > 4 and cells[2] == ws:
            tot += float(cells[4].lstrip('$'))
    return tot
add('M7.gpu.rows32_41', 'M7', 'M7 GPU audited total, COST.md rows 32-41', '1.41', lambda: cost_sum('M7', 32, 41))
add('M7.gpu.table', 'M7', 'M7 GPU total in COST.md table', '1.41',
    lambda: float(re.search(r'^\| M7 \| \$([0-9.]+)', open(P('G1/COST.md')).read(), re.M).group(1)))
add('M6.gpu.w4row', 'M6', 'M6 GPU base pass (w4 fp job) audited', '0.316', lambda: cost_sum('M6', 31, 31))

# ---------- run ----------
out = []
for cid, ws, text, claimed, getter, scale in C:
    try:
        got = getter() * scale
        cv = float(claimed)
        tol = tol_of(claimed)
        ok = abs(got - cv) <= tol
        out.append(dict(id=cid, ws=ws, what=text, claimed=claimed, file_value=got, tol=tol,
                        status='match' if ok else 'MISMATCH'))
    except Exception as e:
        out.append(dict(id=cid, ws=ws, what=text, claimed=claimed, file_value=None, tol=None,
                        status='ERROR', error=repr(e)[:300]))
now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
summ = {ws: {s: sum(1 for o in out if o['ws'] == ws and o['status'] == s) for s in ('match', 'MISMATCH', 'ERROR')} for ws in ('M5', 'M6', 'M7')}
res = dict(checked_at=now, study_root=ROOT, n=len(out), summary=summ, checks=out)
os.makedirs(os.path.join(HERE, 'results'), exist_ok=True)
json.dump(res, open(os.path.join(HERE, 'results', 'static_check.json'), 'w'), indent=1)
print(json.dumps(summ))
for o in out:
    if o['status'] != 'match':
        print(o['status'], o['id'], 'claimed', o['claimed'], 'file', o['file_value'], o.get('error', ''))
