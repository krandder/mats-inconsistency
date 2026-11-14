"""E64b analysis (PREREG-B amendment 72, addendum 1): content routing with split normalisation of block 1's input (split: the content and
the codes normalised separately, so the error feature's normaliser has no code and the routing codes' scale has no error), and split
with nocons (splitnc), retrained from the ctrl initialisation and episodes of each seed (world 1, ctrl format).
Inputs: results/e64b/e64_<arm>_s<seed>.npz (e64b_eval.py), results/e64b_heads/<tag>.json (e64b_heads.py), results/e64b_codes/<arm>.npz
(e64b_codes.py); comparisons as e64_summ.py (E37's ctrl, E60's own, E63's ctrl pooling heads, E59's ctrl code share).
Predictions (registered in amendment 72, addendum 1):
  P251  split: (a) the median over its pooling heads of L_own (content part) < 0; (b) paired mean rom_large(split) <= 0.7 x ctrl's;
  P252  splitnc: paired mean KL(splitnc) <= 1.3 x own's, and mean rom_large(splitnc) <= 0.06;
  P253  paired mean KL(split) / mean KL(ctrl) <= 0.75;
  P254  code share(split) and code share(splitnc) each <= 0.5 x code share(ctrl);
  P255  >= 75% of the acquired split nets are A-type.
usage: python e64b_summ.py -> results/e64b.json, results/e64b_tables.md   (E64_ROOT overrides the input root for plumbing)
"""
import glob, json, os
import numpy as np

ROOT = os.environ.get('E64_ROOT', '.')
rng = np.random.default_rng(641)
ARMS = ('split', 'splitnc')


def ev(path):
    if not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=True)
    return {k: (float(z[k]) if z[k].ndim == 0 and np.issubdtype(z[k].dtype, np.number) else z[k]) for k in z.files}


def arm_rows(arm):
    rows = {}
    for p in glob.glob(os.path.join(ROOT, 'results/e64b/e64_%s_s*.npz' % arm)):
        s = int(p.rsplit('_s', 1)[1][:-4]); rows[s] = ev(p)
    return rows


def ref_rows(kind):
    rows = {}
    pat = 'results/e37/e37_ctrl_s%d.npz' if kind == 'ctrl' else 'results/e60/e60_own_s%d.npz'
    for s in range(12):
        r = ev(pat % s)
        if r is not None:
            rows[s] = r
    return rows


def acq(r):
    return r is not None and r['gap'] < 0.10


def paired_ratio(a, b, key):
    seeds = sorted(s for s in a if s in b and acq(a[s]) and acq(b[s]))
    if not seeds:
        return None
    x = np.array([a[s][key] for s in seeds]); y = np.array([b[s][key] for s in seeds])
    pt = float(x.mean() / y.mean()); bs = []
    for _ in range(2000):
        j = rng.integers(0, len(seeds), len(seeds)); bs.append(x[j].mean() / y[j].mean())
    return [pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(seeds), float(x.mean()), float(y.mean())]


def share(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)
    pt = float(num.sum() / den.sum()); bs = []
    for _ in range(2000):
        j = rng.integers(0, len(num), len(num)); bs.append(num[j].sum() / den[j].sum())
    return [pt, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def main():
    ctrl, own = ref_rows('ctrl'), ref_rows('own')
    heads = {}
    for p in glob.glob(os.path.join(ROOT, 'results/e64b_heads/*.json')):
        o = json.load(open(p)); heads[o['tag']] = o
    e63 = json.load(open('results/e63/ctrl.json'))
    L_ctrl = float(np.median([o['L_own_h'][h] for o in e63.values() for h in o.get('pool_heads', [])]))
    cc = np.load('results/e59/ctrl.npz'); code_ctrl = float(cc['code_pc'].sum() / cc['code_kl'].sum())
    res = dict(arms={}, predictions={}, refs=dict(L_own_ctrl_median=L_ctrl, code_share_ctrl=code_ctrl))
    R = {}
    for arm in ARMS:
        rows = arm_rows(arm); R[arm] = rows
        acqs = sorted(s for s, r in rows.items() if acq(r))
        o = dict(n=len(rows), acquired=len(acqs), seeds=acqs)
        if acqs:
            o['kl'] = float(np.mean([rows[s]['gap'] for s in acqs])); o['kl_sd'] = float(np.std([rows[s]['gap'] for s in acqs]))
            o['rom_large'] = float(np.mean([rows[s]['rom_large'] for s in acqs])); o['rom_middle'] = float(np.mean([rows[s]['rom_middle'] for s in acqs]))
            o['A'] = int(sum(rows[s]['rom_middle'] < 0 for s in acqs)); o['B'] = len(acqs) - o['A']
            if all('sign_pc' in rows[s] for s in acqs):
                o['sign_share'] = share([rows[s]['sign_pc'] for s in acqs], [rows[s]['sign_kl'] for s in acqs])
            o['xq'] = float(np.mean([rows[s]['xq'] for s in acqs])); o['lab_last'] = float(np.mean([rows[s]['lab_last'] for s in acqs]))
            hd = [heads.get('e64_%s_s%d' % (arm, s)) for s in acqs]
            if all(h is not None for h in hd):
                o['n_pool'] = [int(sum(h['n_pool'] == k for h in hd)) for k in (0, 1, 2)]
                o['consensus'] = int(sum(h['consensus'] for h in hd))
                o['two_pool_and_A'] = int(sum(h['n_pool'] == 2 and rows[s]['rom_middle'] < 0 for h, s in zip(hd, acqs)))
                Lp = [h['L_own_h'][i] for h in hd for i in h['pool_heads']]
                o['L_own_median'] = float(np.median(Lp)) if Lp else None
                o['dlogit_median'] = float(np.median([h['dlogit_r3_h'][i] for h in hd for i in h['pool_heads']])) if Lp else None
                o['Kodd_pairs_opposite'] = int(sum(h['n_pool'] == 2 and h['Kodd_mean_h'][h['pool_heads'][0]] * h['Kodd_mean_h'][h['pool_heads'][1]] < 0 for h in hd))
                o['A_src_cv'] = float(np.median([h['A_src_cv'] for h in hd]))
                if 'code_logit_own_h' in hd[0]:
                    o['code_logit_gap'] = float(np.mean([h['code_logit_own_h'][i] - h['code_logit_oth_h'][i] for h in hd for i in h['pool_heads']])) if Lp else None
        cp = os.path.join(ROOT, 'results/e64b_codes/%s.npz' % arm)
        if os.path.exists(cp):
            z = np.load(cp)
            o['code_share'] = share(z['code_pc'], z['code_kl'])
            for k in (7, 9):
                if 'ens%d_kl' % k in z.files:
                    mem = z['ens%d_member_kl' % k]; red = z['ens%d_pc' % k]; o['c_seed%d' % k] = float(np.mean(red / mem))
        o['vs_ctrl_kl'] = paired_ratio(rows, ctrl, 'gap'); o['vs_ctrl_rom'] = paired_ratio(rows, ctrl, 'rom_large'); o['vs_own_kl'] = paired_ratio(rows, own, 'gap')
        res['arms'][arm] = o
    A = res['arms']; pr = res['predictions']
    sp, nc = A['split'], A['splitnc']
    if sp.get('L_own_median') is not None and sp.get('vs_ctrl_rom'):
        pr['P251'] = dict(L_own_median=sp['L_own_median'], rom_ratio=sp['vs_ctrl_rom'], a_holds=bool(sp['L_own_median'] < 0),
                          b_holds=bool(sp['vs_ctrl_rom'][0] <= 0.7))
        pr['P251']['holds'] = bool(pr['P251']['a_holds'] and pr['P251']['b_holds'])
    if nc.get('vs_own_kl'):
        pr['P252'] = dict(kl_ratio_own=nc['vs_own_kl'], rom_large=nc['rom_large'], holds=bool(nc['vs_own_kl'][0] <= 1.3 and nc['rom_large'] <= 0.06))
    if sp.get('vs_ctrl_kl'):
        pr['P253'] = dict(split=sp['vs_ctrl_kl'], holds=bool(sp['vs_ctrl_kl'][0] <= 0.75))
    if 'code_share' in sp and 'code_share' in nc:
        pr['P254'] = dict(split=sp['code_share'], splitnc=nc['code_share'], bar=0.5 * code_ctrl,
                          holds=bool(sp['code_share'][0] <= 0.5 * code_ctrl and nc['code_share'][0] <= 0.5 * code_ctrl))
    if sp.get('acquired'):
        pr['P255'] = dict(A=sp['A'], acquired=sp['acquired'], holds=bool(sp['A'] >= 0.75 * sp['acquired']))
    md = ['# E64b tables (generated by e64b_summ.py)', '', '| arm | acquired | A / B | mean KL (sd) | rom large | rom middle | sign share | code share | c_seed(9) | pooling heads 0/1/2 | consensus | L_own median | dlogit median | K_odd pairs opposite |',
          '|---' * 14 + '|']
    for arm, o in A.items():
        if not o.get('acquired'):
            md.append('| %s | 0 of %d |' % (arm, o['n'])); continue
        f = lambda k: ('%.3f [%.3f, %.3f]' % tuple(o[k][:3])) if k in o else '-'
        md.append('| %s | %d of %d | %d / %d | %.4f (%.4f) | %.3f | %.3f | %s | %s | %s | %s | %s | %s | %s | %s |' % (
            arm, o['acquired'], o['n'], o['A'], o['B'], o['kl'], o['kl_sd'], o['rom_large'], o['rom_middle'], f('sign_share'), f('code_share'),
            '%.2f' % o['c_seed9'] if 'c_seed9' in o else '-', '/'.join(str(x) for x in o['n_pool']) if 'n_pool' in o else '-',
            o.get('consensus', '-'), '%.2f' % o['L_own_median'] if o.get('L_own_median') is not None else '-',
            '%.2f' % o['dlogit_median'] if o.get('dlogit_median') is not None else '-', o.get('Kodd_pairs_opposite', '-')))
    md += ['', '| pair | KL ratio [95% CI] (seeds) | rom large ratio [95% CI] |', '|---|---|---|']
    for arm, o in A.items():
        for k, lab in (('vs_ctrl_kl', '%s / ctrl' % arm), ('vs_own_kl', '%s / own' % arm)):
            if o.get(k):
                x = o[k]; rr = o.get('vs_ctrl_rom') if k == 'vs_ctrl_kl' else None
                md.append('| %s | %.2f [%.2f, %.2f] (%d) | %s |' % (lab, x[0], x[1], x[2], x[3], ('%.2f [%.2f, %.2f]' % tuple(rr[:3])) if rr else '-'))
    md += ['', '| seed | ' + ' | '.join(ARMS) + ' | ctrl | own |', '|---' * (len(ARMS) + 3) + '|']
    for s in range(12):
        cells = []
        for arm in ARMS:
            r = R[arm].get(s); hd = heads.get('e64_%s_s%d' % (arm, s))
            cells.append('-' if r is None else ('%.4f %s %.3f%s' % (r['gap'], 'stall' if not acq(r) else ('A' if r['rom_middle'] < 0 else 'B'), r['rom_large'],
                                                                   (' p%d' % hd['n_pool']) if hd else '')))
        for ref in (ctrl, own):
            r = ref.get(s)
            cells.append('-' if r is None else '%.4f %s %.3f' % (r['gap'], 'stall' if not acq(r) else ('A' if r['rom_middle'] < 0 else 'B'), r['rom_large']))
        md.append('| %d | %s |' % (s, ' | '.join(cells)))
    md += ['', 'Cells: test KL, type (or stall), large-range rom, number of pooling heads (pN).', '', '## Predictions', ''] + \
          ['- %s: %s' % (k, json.dumps(v)) for k, v in pr.items()]
    json.dump(res, open(os.path.join(ROOT, 'results/e64b.json'), 'w'), indent=1, default=float)
    open(os.path.join(ROOT, 'results/e64b_tables.md'), 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
