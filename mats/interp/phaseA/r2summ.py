"""Round 2 summary tables (PREREG amendment 4): reads r2/<net>.json, r2/<net>_D.npz (and r2/loc_<net>.json if present),
writes r2/tables.md, r2/predictions.md and r2/summary.json. usage: python r2summ.py"""
import json, math, os, sys
import numpy as np
from scipy import stats as sst

R2 = os.environ.get('R2DIR', '/home/agent/workspaces/mats/work/cm-interp/r2')
NETS2 = [f'2L_s{s}' for s in range(5)]
NETS3 = [f'3L_s{s}' for s in range(10, 15)]
NETS = NETS2 + NETS3 + ['plat_s3']
ONEPOOL, TWOPOOL = ['2L_s0', '2L_s2', '2L_s3'], ['2L_s1', '2L_s4']
B = 200
rng = np.random.default_rng(0)

res, Dz = {}, {}
for n in NETS:
    if os.path.exists(f'{R2}/{n}.json') and os.path.exists(f'{R2}/{n}_D.npz'):
        r_ = json.load(open(f'{R2}/{n}.json'))
        if 'js_binding' not in r_:                 # run not finished
            continue
        res[n] = r_
        Dz[n] = dict(np.load(f'{R2}/{n}_D.npz'))
        if os.path.exists(f'{R2}/qt_{n}.json') and os.path.exists(f'{R2}/qt_{n}_D.npz'):   # amendment 6
            q_ = json.load(open(f'{R2}/qt_{n}.json'))
            res[n]['direct'].update(q_['direct']); res[n]['qt_checks'] = q_['checks']
            Dz[n].update(dict(np.load(f'{R2}/qt_{n}_D.npz')))
NETS = [n for n in NETS if n in res]
loc = {n: json.load(open(f'{R2}/loc_{n}.json')) for n in NETS if os.path.exists(f'{R2}/loc_{n}.json')}


def final_cuts(n):
    L = res[n]['depth']
    return (f'a{L}', f'h{L}')


def is_final(n, site):
    return site.split('-')[0] in final_cuts(n)


def alias(n, site, v):
    """variant name actually evaluated for (site, v). QT sites: O_q0 = O_own = the matched oracle O_q0m (amendment 6;
    O_c if r2qt has not run), M_q0* = M_q0m*; O_q0ID = O_cIDr = O_cID; at final cuts O_cID = J and O_lat = O_c in
    distribution. Full cuts: O_q0 = O_q0m (QT token matched) if present; 'O_q0c' = r2run's O_q0 (QT at c-information)."""
    raw = lambda x: f'{site}|{x}' in Dz[n]
    if v == 'O_q0c':
        return 'O_q0'
    if v == 'O_q0' and raw('O_q0m'):
        return 'O_q0m'
    if site.endswith('-QT') and v.startswith('M_q0') and raw(v.replace('M_q0', 'M_q0m')):
        return v.replace('M_q0', 'M_q0m')
    if site.endswith('-QT') and v == 'O_own' and raw('O_q0m'):
        return 'O_q0m'
    if site.endswith('-QT'):
        if v in ('O_q0', 'O_own'):
            v = 'O_c'
        if v in ('O_q0ID', 'O_cIDr'):
            v = 'O_cID'
        if is_final(n, site) and v == 'O_cID':
            v = 'J'
        if is_final(n, site) and v == 'O_lat':
            v = 'O_c'
    return v


def has(n, site, v):
    return f'{site}|{alias(n, site, v)}' in Dz[n]


def cellmean(n, site, v):
    return Dz[n][f'{site}|{alias(n, site, v)}'].astype(np.float64).mean(1)


def dstat(n, site, v):
    return res[n]['direct'][f'{site}|{alias(n, site, v)}']


def eps(n):
    return 0.1 * res[n]['kl_model_E']


def boot_steps(n, site, pairs, denom=('q0', None)):
    """pairs: {name: (va, vb)} -> step = mean_c[d_a - d_b] (vb None: d_a alone). Returns {name: dict(est, lo, hi,
    share, share_lo, share_hi)} with a paired cell bootstrap (same indices for all steps)."""
    arrs = {}
    for k, (a, b) in pairs.items():
        x = cellmean(n, site, a)
        arrs[k] = x - (cellmean(n, site, b) if b else 0)
    den = cellmean(n, site, denom[0]) - (cellmean(n, site, denom[1]) if denom[1] else 0)
    nc = len(den)
    idx = rng.integers(0, nc, (B, nc))
    dm = den[idx].mean(1)
    out = {}
    for k, x in arrs.items():
        se = float(x.std(ddof=1) / math.sqrt(nc))
        bm = x[idx].mean(1)
        out[k] = dict(est=float(x.mean()), se=se, lo=float(x.mean() - 1.96 * se), hi=float(x.mean() + 1.96 * se),
                      share=float(x.mean() / den.mean()),
                      share_lo=float(np.quantile(bm / dm, .025)), share_hi=float(np.quantile(bm / dm, .975)))
    return out


def tq(k):
    return sst.t.ppf(0.975, k - 1) if k > 1 else float('nan')


def seedagg(vals):
    v = np.array([x for x in vals if x is not None and np.isfinite(x)])
    if len(v) == 0:
        return '–'
    if len(v) == 1:
        return f'{v[0]:.3f}'
    return f'{v.mean():.3f} ± {tq(len(v)) * v.std(ddof=1) / math.sqrt(len(v)):.3f}'


def f3(x, d=3):
    return '–' if x is None or not np.isfinite(x) else f'{x:.{d}f}'


SITES_MAIN = {'2L': ['h1-LAB', 'h1-QQ', 'h1-QT', 'h1-ALL', 'a2-QT', 'h2-QT'],
              '3L': ['h1-LAB', 'h1-QQ', 'h1-QT', 'h1-ALL', 'h2-LAB', 'h2-QQ', 'h2-QT', 'h2-ALL', 'a3-QT', 'h3-QT'],
              'plat': ['h1-LAB', 'h1-QQ', 'h1-QT', 'h1-ALL', 'a2-QT', 'h2-QT']}
grp = lambda n: '2L' if n.startswith('2L') else ('3L' if n.startswith('3L') else 'plat')
RUNGS = ['q0', 'q0x', 'q1', 'M_q0', 'M_q0mean', 'M_c', 'M_cmean', 'O_lat', 'O_own', 'O_q0', 'O_q0ID', 'O_c', 'O_U', 'O_cID', 'O_cIDr', 'J']

# ----------------------------------------------------------------------------- per site decomposition
DEC = {}
for n in NETS:
    for site in SITES_MAIN[grp(n)]:
        if not has(n, site, 'q0'):
            continue
        fin = is_final(n, site)
        pairs = {'form': ('q0', 'O_q0'), 'mean': ('q0', 'M_q0'), 'shape': ('M_q0', 'O_q0')}
        if site.endswith('-ALL') and has(n, site, 'O_q0c') and alias(n, site, 'O_q0') == 'O_q0m':
            pairs['shape'] = ('M_q0', 'O_q0c')                       # QT token at c-information in both
            pairs['qtinfo'] = ('O_q0', 'O_q0c')                      # what the QT token's missing information costs
        if site.endswith('-QT'):
            pairs['form_bound'] = ('q0', 'O_c')                      # O_c over-informs q0 at QT: upper bound on form
        if not fin:
            pairs.update({'cross': ('O_q0', 'O_c'), 'ids': ('O_c', 'O_cID'), 'ownres': ('O_cID', 'O_cIDr'),
                          'rest': ('O_cIDr', None), 'routing': ('O_lat', 'O_c')})
            if has(n, site, 'q0x'):
                pairs.update({'form_B': ('q0x', 'O_c'), 'cross_B': ('q0', 'q0x')})
            if has(n, site, 'M_c'):
                pairs.update({'mean_c': ('q0x', 'M_c'), 'shape_c': ('M_c', 'O_c')})
            pairs.update({'ids_B': ('O_q0', 'O_q0ID'), 'cross_B2': ('O_q0ID', 'O_cID')})
        else:
            pairs.update({'rest': ('O_c', None)})
        pairs.update({'binding': ('O_U', 'O_c'), 'missing': ('O_q0', None)})
        if site.endswith('h2-QT') and has(n, site, 'q0_LN'):
            pairs.update({'mean_LN': ('q0_LN', 'M_q0_LN'), 'form_LN': ('q0_LN', 'O_c')})
        if site.split('-')[0] == f"h{res[n]['depth']}" and has(n, site, 'q0_LN'):
            pairs.update({'mean_LN': ('q0_LN', 'M_q0_LN'), 'form_LN': ('q0_LN', 'O_c')})
        DEC[(n, site)] = boot_steps(n, site, pairs)

# ----------------------------------------------------------------------------- tables
L_ = []
w = L_.append
w('# Round 2 tables (generated by r2summ.py)\n')
w(f'Nets present: {", ".join(NETS)}\n')
w('Δ = L(description) − L(p) in nats on fresh eval E (6,000 cells × 8 exact conditional draws). Oracles O_* replace the '
  'site\'s tokens by the network\'s real states on fresh world episodes drawn given the named information (PREREG '
  'amendment 4). At query-token sites O_q0 = O_own = O_c and O_q0ID = O_cIDr = O_cID; at final cuts O_cID = J.\n')

w('## Model KL and checks\n')
w('| net | KL_model (E) | eval.json | ε | β-match \\|err\\| median / q99 | U: P(assignment = identity) mean | JS blind max | JS grid check |')
w('|---|---|---|---|---|---|---|---|')
for n in NETS:
    c = res[n]['checks']; js = res[n].get('js_binding', {})
    be = c.get('beta_err_q0', {})
    w(f"| {n} | {res[n]['kl_model_E']:.4f} | {res[n]['eval_kl']:.4f} | {eps(n):.5f} | {f3(be.get('median'), 4)} / {f3(be.get('q99'), 3)} | "
      f"{f3(c.get('U_p_identity', {}).get('mean'), 2)} | {js.get('js_blind_max', float('nan')):.1e} | {js.get('grid_check_bayes_rel', float('nan')):.1e} |")
w('')

for g, nets in (('2L', [n for n in NETS if grp(n) == '2L']), ('3L', [n for n in NETS if grp(n) == '3L']), ('plat', [n for n in NETS if grp(n) == 'plat'])):
    if not nets:
        continue
    w(f'## Ladder, signed Δ (seed mean ± t-interval half-width over {len(nets)} seed(s)) — {g}\n')
    w('| site | ' + ' | '.join(RUNGS) + ' |')
    w('|---|' + '---|' * len(RUNGS))
    for site in SITES_MAIN[g]:
        row = []
        for v in RUNGS:
            vals = [dstat(n, site, v)['delta'] if has(n, site, v) else None for n in nets]
            row.append(seedagg(vals) if any(x is not None for x in vals) else '–')
        w(f'| {site} | ' + ' | '.join(row) + ' |')
    w('')
    w(f'## Ladder, Δ_rms (closure metric; seed mean) — {g}\n')
    w('| site | ' + ' | '.join(RUNGS) + ' |')
    w('|---|' + '---|' * len(RUNGS))
    for site in SITES_MAIN[g]:
        row = []
        for v in RUNGS:
            vals = [dstat(n, site, v)['rms'] if has(n, site, v) else None for n in nets]
            row.append(seedagg(vals) if any(x is not None for x in vals) else '–')
        w(f'| {site} | ' + ' | '.join(row) + ' |')
    w('')

w('## Per-net decomposition of Δ(q0), fixed nested order (signed nats; share of Δ(q0) in brackets)\n')
w('Order: q0 →form→ O_q0 →cross-source→ O_c →other IDs→ O_cID →own residual vector + labels→ O_cIDr →rest→ (all of x). '
  'Form = mean [q0 → M_q0] + shape/off-manifold [M_q0 → O_q0]. **Order-dependent**: see the order-check table.\n')
w('| net | site | Δ(q0) | form | (mean / shape) | cross | IDs | own res. | rest | binding O_U−O_c | routing O_lat−O_c |')
w('|---|---|---|---|---|---|---|---|---|---|---|')
for (n, site), dd in DEC.items():
    dq = dstat(n, site, 'q0')['delta']
    fmt = lambda k: (f"{dd[k]['est']:+.3f} [{dd[k]['share']:+.2f}]" if k in dd else '–')
    fmt2 = lambda k: (f"{dd[k]['est']:+.3f}" if k in dd else '–')
    w(f"| {n} | {site} | {dq:.3f} | {fmt('form')} | {fmt2('mean')} / {fmt2('shape')} | {fmt('cross')} | {fmt('ids')} | "
      f"{fmt('ownres')} | {fmt('rest')} | {fmt2('binding')} | {fmt2('routing')} |")
w('')

w('## Order checks (signed nats): the attribution depends on the order in which form and variables are changed\n')
w('(form, cross): A = q0 →form→ O_q0 →cross→ O_c; B = q0 →cross (linear, q0x = q0 + `cross`)→ q0x →form→ O_c. '
  'Shapley = mean of the two orders. (cross, IDs): A = O_q0 → O_c → O_cID; B = O_q0 → O_q0ID → O_cID.\n')
w('| net | site | form A | form B | form Shapley | cross A | cross B | cross Shapley | IDs A | IDs B |')
w('|---|---|---|---|---|---|---|---|---|---|')
for (n, site), dd in DEC.items():
    if 'form_B' not in dd:
        continue
    fa, fb, ca, cb = dd['form']['est'], dd['form_B']['est'], dd['cross']['est'], dd['cross_B']['est']
    w(f"| {n} | {site} | {fa:+.3f} | {fb:+.3f} | {(fa + fb) / 2:+.3f} | {ca:+.3f} | {cb:+.3f} | {(ca + cb) / 2:+.3f} | "
      f"{dd['ids']['est']:+.3f} | {dd['ids_B']['est']:+.3f} |")
w('')

w('## Seed means of the decomposition (signed nats; Shapley form/cross where q0x exists)\n')
w('| group | site | Δ(q0) | form (A) | Shapley form | mean | shape | cross (A) | Shapley cross | IDs | own res. | rest | binding |')
w('|---|---|---|---|---|---|---|---|---|---|---|---|---|')
GROUPS = [('2L one-pool', ONEPOOL), ('2L two-pool', TWOPOOL), ('3L', NETS3), ('plateau', ['plat_s3'])]
for gname, gn in GROUPS:
    gn = [n for n in gn if n in NETS]
    if not gn:
        continue
    for site in SITES_MAIN[grp(gn[0])]:
        if (gn[0], site) not in DEC:
            continue
        get = lambda k: [DEC[(n, site)][k]['est'] if k in DEC[(n, site)] else None for n in gn]
        shap = lambda a, b: [((DEC[(n, site)][a]['est'] + DEC[(n, site)][b]['est']) / 2) if b in DEC[(n, site)] else None for n in gn]
        w(f"| {gname} | {site} | {seedagg([dstat(n, site, 'q0')['delta'] for n in gn])} | {seedagg(get('form'))} | "
          f"{seedagg(shap('form', 'form_B'))} | {seedagg(get('mean'))} | {seedagg(get('shape'))} | {seedagg(get('cross'))} | "
          f"{seedagg(shap('cross', 'cross_B'))} | {seedagg(get('ids'))} | {seedagg(get('ownres'))} | {seedagg(get('rest'))} | {seedagg(get('binding'))} |")
w('')

w('## Joint manifold: out-of-fold AUC, whole flattened layer, real vs stitched (main cut; 3,000 cells × 8)\n')
CV = ['q0', 'O_q0', 'O_c', 'O_cID', 'O_cIDr', 'J']
w('| net | site | ' + ' | '.join(CV) + ' |')
w('|---|---|' + '---|' * len(CV))
for n in NETS:
    jc = res[n].get('joint_classifier', {})
    for site in sorted({k.split('|')[0] for k in jc}):
        w(f'| {n} | {site} | ' + ' | '.join(f3(jc.get(f'{site}|{v}', {}).get('auc')) for v in CV) + ' |')
w('')

w('## Whole-input binding (no stitching): B_JS = E[JS_net] / E[JS_Bayes] over the six SS-to-source assignments\n')
w('| net | E JS_net | E JS_Bayes | B_JS [95% CI] | B_JS, top half by JS_Bayes | corr(JS_net, JS_Bayes) | JS blind (null) max |')
w('|---|---|---|---|---|---|---|')
for n in NETS:
    js = res[n].get('js_binding')
    if js:
        w(f"| {n} | {js['js_net']:.4f} | {js['js_bayes']:.4f} | {js['B_JS']:.3f} [{js['B_JS_lo']:.3f}, {js['B_JS_hi']:.3f}] | "
          f"{js['B_JS_tophalf']:.3f} | {js['corr_net_bayes']:.3f} | {js['js_blind_max']:.1e} |")
w('')

w('## Per-net rung values: Δ ± se; Δ_rms [95% CI]; Hill; 1%-trimmed mean\n')
for n in NETS:
    w(f'### {n}\n')
    w('| site | variant | Δ | se | Δ_rms | rms 95% CI | Hill | trim1 |')
    w('|---|---|---|---|---|---|---|---|')
    for site in SITES_MAIN[grp(n)]:
        for v in RUNGS + ['q0_LN', 'M_q0_LN', 'M_q0mean_LN']:
            k = f'{site}|{v}'
            if k in res[n]['direct']:
                r = res[n]['direct'][k]
                w(f"| {site} | {v} | {r['delta']:.4f} | {r['se']:.4f} | {r['rms']:.4f} | [{r['rms_lo']:.4f}, {r['rms_hi']:.4f}] | {r['hill']:.1f} | {r['trim1']:.4f} |")
    w('')

if loc:
    w('## Readout control for localisation (r2loc.py)\n')
    w('Fraction of each rung\'s own Δ that remains when the named part is kept real and the rest is drawn from the rung '
      '(knock-in). U_G = top-4 eigenvectors of E_p[∇G ∇Gᵀ]; components = residual-stream writes at the cut.\n')
    w('| net | site | part | q0: Δ, remaining fraction | R_in: Δ, remaining fraction | random 4-d (q0) |')
    w('|---|---|---|---|---|---|')
    for n in NETS:
        if n not in loc:
            continue
        for key, o in loc[n]['rows'].items():
            w(f"| {n} | {o['site']} | {o['part']} | {o['q0_delta']:.3f}, {f3(o['q0_frac'], 3)} | {o['Rin_delta']:.3f}, {f3(o['Rin_frac'], 3)} | {f3(o.get('rand_frac_q0'), 3)} |")
    w('')
open(f'{R2}/tables.md', 'w').write('\n'.join(L_))

# ----------------------------------------------------------------------------- predictions
P = []
p = P.append


def ok(b):
    return 'HOLDS' if b else 'FAILS'


near = [n for n in NETS if n != 'plat_s3']
full_cuts = lambda n: (['h1-ALL'] if res[n]['depth'] == 2 else ['h1-ALL', 'h2-ALL'])
fin_sites = lambda n: [f'{c}-QT' for c in final_cuts(n)]
# C1
c1 = []
for n in NETS:
    for site in full_cuts(n):
        r = dstat(n, site, 'J'); c1.append((n, site, 'J', abs(r['delta']) <= 2 * r['se'] and r['rms_hi'] <= eps(n), r['delta'], r['se'], r['rms_hi']))
    for site in fin_sites(n):
        r = dstat(n, site, 'O_c'); c1.append((n, site, 'O_c', abs(r['delta']) <= 2 * r['se'] and r['rms_hi'] <= eps(n), r['delta'], r['se'], r['rms_hi']))
p(f"- **C1 (V6)**: {sum(x[3] for x in c1)}/{len(c1)} pass. Fails: " + ('; '.join(f'{a} {b} {c}: Δ {d:+.4f} ± {e:.4f}, rms_hi {f:.4f} (ε {eps(a):.4f})' for a, b, c, ok_, d, e, f in c1 if not ok_) or 'none'))
c2 = [(n, res[n]['joint_classifier'].get(f"{'h1' if res[n]['depth'] == 2 else 'h2'}-ALL|J", {}).get('auc')) for n in NETS if 'joint_classifier' in res[n]]
p(f"- **C2** (joint-classifier AUC of J at the full cut ≤ 0.52): {sum(1 for _, a in c2 if a is not None and a <= 0.52)}/{len(c2)}; values " + ', '.join(f'{n} {f3(a)}' for n, a in c2))
c3 = [(n, res[n]['js_binding']['js_blind_max'], res[n]['js_binding']['grid_check_bayes_rel']) for n in NETS if 'js_binding' in res[n]]
p(f"- **C3** (JS null and grid): {sum(1 for _, a, b in c3 if a < 1e-6 and b < 0.01)}/{len(c3)} pass")
c4 = [(n, res[n]['checks'].get('beta_err_q0', {})) for n in NETS]
p(f"- **C4** (β match): {sum(1 for _, b in c4 if b and b['median'] <= 0.005 and b['q99'] <= 0.05)}/{len(c4)} pass; medians " + ', '.join(f"{n} {f3(b.get('median'), 4)}" for n, b in c4))
# P9
p9 = [(n, s, dstat(n, s, 'O_q0')['rms'], dstat(n, s, 'q0')['rms'], alias(n, s, 'O_q0')) for n in NETS for s in fin_sites(n)]
p(f"- **P9** (final QT: Δ_rms(O_q0) ≤ 0.1·Δ_rms(q0); O_q0 = matched oracle where run, amendment 6): {ok(all(a <= 0.1 * b for _, _, a, b, _ in p9))} "
  f"({sum(a <= 0.1 * b for _, _, a, b, _ in p9)}/{len(p9)}); ratios " + ', '.join(f'{n} {s} {a / b:.2f} ({v})' for n, s, a, b, v in p9))
p9c = [(n, s, dstat(n, s, 'O_c')['rms'] / dstat(n, s, 'q0')['rms']) for n in NETS for s in fin_sites(n)]
p(f"  - with O_c in place of O_q0 (the amendment-4 definition; O_c over-informs q0): max ratio {max(r for _, _, r in p9c):.3f}")
# P10
p10s = [(n, s) for n in NETS for s in (['h1-QQ', 'h1-ALL'] if res[n]['depth'] == 2 else ['h2-QQ', 'h2-ALL'])]
sh = {(n, s): DEC[(n, s)]['form']['share'] for n, s in p10s}
a = sum(v >= 0.5 for v in sh.values())
b_ = [(n, s, sh[(n, s)]) for n in TWOPOOL if n in NETS for s in ('h1-QQ', 'h1-ALL')]
p(f"- **P10** (form share ≥ 0.5 in ≥ 80% of read-heavy context site-nets; ≥ 0.8 in s1, s4): {a}/{len(sh)} ≥ 0.5 "
  f"({ok(a >= 0.8 * len(sh))}); two-pool: " + ', '.join(f'{n} {s} {v:.2f}' for n, s, v in b_) + f" ({ok(all(v >= 0.8 for _, _, v in b_))}). Shares: "
  + ', '.join(f'{n} {s} {v:.2f}' for (n, s), v in sh.items()))
# P11
p11 = {n: DEC[(n, 'h1-QQ')]['cross']['est'] for n in NETS2 if n in NETS}
p11r = {n: dstat(n, 'h1-QQ', 'O_q0')['rms'] - dstat(n, 'h1-QQ', 'O_c')['rms'] for n in NETS2 if n in NETS}
h11 = all(p11[n] >= 0.05 for n in ONEPOOL if n in p11) and all(p11[n] <= 0.02 for n in TWOPOOL if n in p11)
h11r = all(p11r[n] >= 0.1 for n in ONEPOOL if n in p11r) and all(p11r[n] <= 0.02 for n in TWOPOOL if n in p11r)
p(f"- **P11** (D2 motif; cross step at 2L h1-QQ ≥ 0.05 one-pool, ≤ 0.02 two-pool): {ok(h11)}: " + ', '.join(f'{n} {v:+.3f}' for n, v in p11.items())
  + f". Reviewer's Δ_rms version (≥ 0.1 / ≤ 0.02): {ok(h11r)}: " + ', '.join(f'{n} {v:+.3f}' for n, v in p11r.items()))
# P12
p12 = {n: DEC[(n, 'h1-QQ')]['binding']['est'] for n in NETS2 + ['plat_s3'] if n in NETS}
h12 = all(p12[n] >= 0.1 for n in NETS2 if n in p12) and ('plat_s3' not in p12 or p12['plat_s3'] <= 0.02)
p(f"- **P12** (D3 layer binding at h1-QQ: ≥ 0.1 near-Bayes 2L, ≤ 0.02 plateau): {ok(h12)}: " + ', '.join(f'{n} {v:+.3f}' for n, v in p12.items()))
# P13
p13 = {n: res[n]['js_binding']['B_JS'] for n in NETS if 'js_binding' in res[n]}
h13 = all(p13[n] >= 0.75 for n in near if n in p13) and ('plat_s3' not in p13 or p13['plat_s3'] <= 0.05)
p(f"- **P13** (B_JS ≥ 0.75 near-Bayes, ≤ 0.05 plateau): {ok(h13)}: " + ', '.join(f'{n} {v:.3f}' for n, v in p13.items()))
# P14
p14 = []
for n in near:
    for s in fin_sites(n) + [('h1-ALL' if res[n]['depth'] == 2 else 'h2-ALL')]:
        dd = DEC[(n, s)]
        if dd['form']['est'] > 2 * dd['form']['se']:
            p14.append((n, s, dd['mean']['est'] / dd['form']['est']))
a14 = sum(r >= 0.6 for _, _, r in p14)
p(f"- **P14** (mean part ≥ 0.6 of form at final QT + main full cut, ≥ 80% of near-Bayes site-nets): {ok(a14 >= 0.8 * len(p14))} "
  f"({a14}/{len(p14)}): " + ', '.join(f'{n} {s} {r:.2f}' for n, s, r in p14))
# P15
p15 = [(n, s, DEC[(n, s)]['ids']['est']) for n in near for s in SITES_MAIN[grp(n)] if (n, s) in DEC and 'ids' in DEC[(n, s)]]
p(f"- **P15** (|O_c − O_cID| ≤ 0.02 everywhere near-Bayes): {ok(all(abs(v) <= 0.02 for _, _, v in p15))} "
  f"({sum(abs(v) <= 0.02 for _, _, v in p15)}/{len(p15)}); largest: " + ', '.join(f'{n} {s} {v:+.3f}' for n, s, v in sorted(p15, key=lambda x: -abs(x[2]))[:4]))
# P16
p16 = {n: DEC[(n, 'h1-QQ')]['routing']['est'] for n in NETS2 if n in NETS}
p(f"- **P16** (O_lat − O_c ≥ 0.1 at 2L h1-QQ): {ok(all(v >= 0.1 for v in p16.values()))}: " + ', '.join(f'{n} {v:+.3f}' for n, v in p16.items()))
# P17
p17 = []
for n in NETS:
    jc = res[n].get('joint_classifier', {})
    site = f"{'h1' if res[n]['depth'] == 2 else 'h2'}-ALL"
    A = {v: jc.get(f'{site}|{v}', {}).get('auc') for v in ('q0', 'O_q0', 'O_c', 'J')}
    se = {v: jc.get(f'{site}|{v}', {}).get('auc_se', 0) for v in ('q0', 'O_q0', 'O_c', 'J')}
    if None in A.values():
        continue
    p17.append((n, A['q0'] > A['O_q0'] and A['O_q0'] >= A['O_c'] - 2 * math.hypot(se['O_q0'], se['O_c']) and A['O_c'] > A['J'], A))
p(f"- **P17** (AUC order q0 > O_q0 ≥ O_c > J at the full cut in ≥ 8/11): {ok(sum(x[1] for x in p17) >= 8)} ({sum(x[1] for x in p17)}/{len(p17)}): "
  + '; '.join(f"{n} " + '/'.join(f3(v) for v in A.values()) for n, _, A in p17))
if loc:
    ug = [(n, o['site'], o['q0_frac'], o['Rin_frac']) for n in loc for k, o in loc[n]['rows'].items() if k.endswith('U_G4')]
    ug_both = [x for x in ug if x[2] <= 0.1]
    h18a = all(x[3] <= 0.1 for x in ug_both)
    mlp = [(n, o['q0_frac'], o['Rin_frac']) for n in loc for k, o in loc[n]['rows'].items() if k == 'h1-ALL|comp:mlp']
    h18b = sum(x[2] <= 0.2 for x in mlp) >= 8
    p(f"- **P18** (readout control): (a) U_G knock-in removes ≥ 90% of R_in's gap wherever it removes ≥ 90% of q0's: {ok(h18a)} "
      f"({sum(x[3] <= 0.1 for x in ug_both)}/{len(ug_both)}; q0 remaining {min(x[2] for x in ug):.3f}–{max(x[2] for x in ug):.3f}, "
      f"R_in remaining {min(x[3] for x in ug):.3f}–{max(x[3] for x in ug):.3f}). (b) MLP-write knock-in at the first post-block cut removes ≥ 80% of "
      f"R_in's gap in ≥ 8/11 nets: {ok(h18b)} ({sum(x[2] <= 0.2 for x in mlp)}/{len(mlp)}; R_in remaining " + ', '.join(f'{n} {x:.3f}' for n, _, x in mlp) + ')')
open(f'{R2}/predictions.md', 'w').write('# Round 2 controls and predictions (generated by r2summ.py)\n\n' + '\n'.join(P) + '\n')
json.dump({f'{n}|{s}': v for (n, s), v in DEC.items()}, open(f'{R2}/summary.json', 'w'), indent=0)
print('\n'.join(P))
