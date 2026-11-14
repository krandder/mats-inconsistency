"""Amendments 8-9: collect the width sweep (training criterion + sparsity measures) into results/sweep/summary.json and
results/sweep/table.md. Series S: CPU, 2^16 steps; L: GPU, 2^20 steps; Z: ARC's noiseless zoo models.
usage: python sweep_table.py"""
import json, glob, os, re
import numpy as np

rows = []
for series, pat in (('S', 'runs/sweep/S_w*_s*_sig*.json'), ('L', 'runs/gpu/w*_s*_sig*.json'), ('Z', 'runs/sweep/Z_w*_s*_sig*.json')):
    for f in sorted(glob.glob(pat)):
        mm = re.search(r'w(\d+)_s(\d+)_sig([0-9.]+)\.json$', f)
        if not mm: continue
        d_, s_, sg = int(mm.group(1)), int(mm.group(2)), float(mm.group(3))
        if series == 'L' and sg not in (0.0, 0.01): continue
        tr = json.load(open(f)); e = tr['eval_refs']['16:10:ex']; e2 = tr['eval_refs'].get('16:10:2', {})
        row = dict(series=series, d=d_, seed=s_, sigma=sg, L=e['L_var'], L_orig=e['L_orig'], rel_gap=e['rel_gap'],
                   A=e['A_var'], A_orig=e['A_orig'], meets=e['meets'], meets_seed2=e2.get('meets'))
        af = f'results/sweep/{series}_w{d_}_s{s_}_sig{sg}.json'
        if os.path.exists(af):
            a = json.load(open(af))
            row.update(E=a['n_edges'], M1=a['M1_mean']['frac'], M1K=a['M1_mean']['kept_total'], M2=a['M2_resample']['frac'],
                       M2K=a['M2_resample']['kept_total'], M3=a['M3_alg']['frac'], M3K=a['M3_alg']['kept_total'],
                       M4=a['M4_dirs']['frac'], M4k=a['M4_dirs']['k'], M5=a['M5_single']['share_ge_0p1pct'],
                       M5any=a['M5_single']['share_any'], p_net=a['p_net'], noise_flip=a['noise_flip'],
                       alg_readout_err=a['alg_readout_err'])
            if 'R_redundancy' in a:
                rr = a['R_redundancy']
                row.update(R_exists=rr['exists'], R_flip=rr['flip_ablate_M1_set'], R_frac=rr.get('frac'), R_K=rr.get('kept_total'),
                           S_jac=a['S_stability']['jaccard'], S_sizes=a['S_stability']['sizes'], B_frac=a['B_random']['frac'],
                           B_K=a['B_random']['kept_total'])
        rows.append(row)
json.dump(rows, open('results/sweep/summary.json', 'w'), indent=1)


def pct(v):
    return '%.0f%%' % (100 * v)


def cell(g, key, f=pct):
    v = [r_[key] for r_ in g if r_.get(key) is not None]
    if not v: return '-'
    return '%s (%s)' % (f(np.median(v)), ', '.join(f(x) for x in v)) if len(v) > 1 else f(v[0])


lines = ['| series | width | sigma | passes | loss (example 0.147) | M1 mean-ablation share | M1 kept edges | M2 resample | M3 alg.-relative | M4 directions (k) | M5 single-edge | disjoint DAG exists (its share) | repeat Jaccard | random order |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
groups = {}
for r_ in rows: groups.setdefault((r_['series'], r_['d'], r_['sigma']), []).append(r_)
for (se, d_, sg), g in sorted(groups.items(), key=lambda kv: ({'S': 0, 'L': 1, 'Z': 2}[kv[0][0]], kv[0][2], kv[0][1])):
    g = sorted(g, key=lambda r_: r_['seed'])
    an = [r_ for r_ in g if 'M1' in r_]
    loss = ', '.join('%.3f%s' % (r_['L'], '' if r_['meets'] else '*') for r_ in g)
    kcell = cell(an, 'M1K', lambda v: '%d' % v)
    m4 = cell(an, 'M4') + (' [%s]' % ', '.join(str(r_['M4k']) for r_ in an) if an else '')
    red = ', '.join(('yes (%s)' % pct(r_['R_frac'])) if r_.get('R_exists') else ('no (%.1f%%)' % (100 * r_['R_flip']) if 'R_flip' in r_ else '-') for r_ in an) or '-'
    jac = ', '.join('%.2f-%.2f' % (min(r_['S_jac']), max(r_['S_jac'])) for r_ in an if 'S_jac' in r_) or '-'
    lines.append('| %s | %d | %g | %d of %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
        se, d_, sg, sum(r_['meets'] for r_ in g), len(g), loss, cell(an, 'M1'), kcell, cell(an, 'M2'), cell(an, 'M3'), m4,
        cell(an, 'M5'), red, jac, cell(an, 'B_frac')))
open('results/sweep/table.md', 'w').write('\n'.join(lines) + '\n')
print('\n'.join(lines))
