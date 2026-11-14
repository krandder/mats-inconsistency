"""Turn results/validation.json into results/tables.md."""
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'results', 'validation.json')
rows = json.load(open(src))
out = []
P = out.append


def by(case, *keys):
    g = defaultdict(list)
    for r in rows:
        if r['case'] == case:
            g[tuple(r[k] for k in keys)].append(r)
    return dict(sorted(g.items(), key=lambda kv: tuple(str(x) for x in kv[0])))


def f(x, p=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return '–'
    return f'{x:.{p}g}' if abs(x) < 1e-3 and x != 0 else f'{x:.{p}f}'


def est_stats(rs, key, truth):
    v = np.array([r[key] for r in rs], float)
    se = np.array([r.get(key + '_se', np.nan) for r in rs], float)
    cov = [r.get(key + '_cov') for r in rs]
    cov = np.mean(cov) if cov[0] is not None else np.nan
    sd = v.std(ddof=1) if len(v) > 1 else np.nan
    tot = [r[key + '_se_tot'] for r in rs if key + '_se_tot' in r]
    ctot = [r[key + '_cov_tot'] for r in rs if key + '_cov_tot' in r]
    return dict(mean=v.mean(), bias=v.mean() - truth, sd=sd, se=np.nanmean(se), ratio=sd / np.nanmean(se),
                cover=cov, rmse=np.sqrt(np.mean((v - truth) ** 2)),
                se_tot=np.mean(tot) if tot else np.nan, cover_tot=np.mean(ctot) if ctot else np.nan)


EST = [('direct', 'direct'), ('cov_plain', 'Cov_q(G,w) plain IS'), ('cov_snis', 'Cov_q SNIS'),
       ('temper_q', 'L(0)-L(1), SNIS from q'), ('bridge', 'L(0)-L(1), bridge'), ('bridge_int', '-∫dL/dα, bridge')]

# ------------------------------------------------------------------ (i)
P('## (i) p = q (gap 0, w = 1)\n')
P('Rates are the fraction of replicates whose 95% CI excludes 0 (nominal 0.05).\n')
P('| world | classifier | R | AUC mean ± sd | AUC ≠ 0.5 at 2 SE | KL plug-in | sd log ŵ | E_q ŵ | ECE | '
  'CI misses 0: direct | Cov plain | Cov SNIS | L SNIS | bridge | mean abs(Cov) / direct SE | '
  'gated Cov: CI misses 0 | refit-SE CI misses 0: Cov / bridge (R) |')
P('|' + '---|' * 17)
for (w, fam), rs in by('i', 'world', 'family').items():
    a = np.array([r['auc'] for r in rs])
    miss = lambda k: 1 - np.mean([r[k + '_cov'] for r in rs])
    rr = [r for r in rs if 'cov_plain_cov_tot' in r]
    refit = (f"{1 - np.mean([r['cov_plain_cov_tot'] for r in rr]):.2f} / "
             f"{1 - np.mean([r['bridge_cov_tot'] for r in rr]):.2f} ({len(rr)})") if rr else '–'
    P(f"| {w} | {fam} | {len(rs)} | {a.mean():.4f} ± {a.std(ddof=1):.4f} | {np.mean([r['auc_sig'] for r in rs]):.2f} | "
      f"{np.mean([r['kl_plugin'] for r in rs]):.2g} | {np.mean([r['logw_sd'] for r in rs]):.3f} | "
      f"{np.mean([r['norm_q'] for r in rs]):.4f} | {np.mean([r['ece'] for r in rs]):.4f} | "
      f"{miss('direct'):.2f} | {miss('cov_plain'):.2f} | {miss('cov_snis'):.2f} | {miss('temper_q'):.2f} | "
      f"{miss('bridge'):.2f} | {np.mean([abs(r['cov_plain']) / r['direct_se'] for r in rs]):.2f} | "
      f"{1 - np.mean([r['gated_cov_cov'] for r in rs]):.2f} | {refit} |")
P('')

# ------------------------------------------------------------------ (ii)
P('## (ii) Non-Gaussian p with analytic ratio (MixWorld d=8, K=3, a=2; q = moment-matched conditional Gaussian)\n')
any_ii = [r for r in rows if r['case'] == 'ii' and r['mode'] == 'true']
if any_ii:
    P(f"True KL(p‖q) = {any_ii[0]['kl_true']:.4f} nats, χ²(p‖q) = {any_ii[0]['chi2_true']:.4f}.\n")
P('| w used | G | truth Δ | estimator | mean | bias | emp. sd | mean SE (ratio fixed) | sd/SE | 95% CI coverage | '
  'SE incl. refit | coverage incl. refit |')
P('|---|---|---|---|---|---|---|---|---|---|---|---|')
for (mode, g), rs in by('ii', 'mode', 'g').items():
    truth = rs[0]['truth']
    for k, name in EST:
        if k not in rs[0]:
            continue
        s = est_stats(rs, k, truth)
        P(f"| {mode} (R={len(rs)}) | {g} | {truth:.4f} | {name} | {s['mean']:.4f} | {s['bias']:+.4f} | {s['sd']:.4f} | "
          f"{f(s['se'], 3)} | {f(s['ratio'], 2)} | {f(s['cover'], 2)} | {f(s['se_tot'], 3)} | {f(s['cover_tot'], 2)} |")
    if 'T1' in rs[0]:
        P(f"| | | | trust checks: T1 pass rate {np.mean([r['T1'] for r in rs]):.2f}, "
          f"T2 pass rate {np.mean([r['T2'] for r in rs]):.2f} | | | | | | | | |")
P('')
P('Classifier diagnostics (case ii):\n')
P('| classifier | AUC | corr(log ŵ, log w) | RMSE log ŵ | KL plug-in (true 0.172) | DV bound | ECE | E_q ŵ | E_p 1/ŵ |')
P('|---|---|---|---|---|---|---|---|---|')
for (mode,), rs in by('ii', 'mode').items():
    if mode not in ('mlp', 'quad'):
        continue
    rs = [r for r in rs if r['g'] == 'quartic']
    m = lambda k: np.mean([r[k] for r in rs])
    P(f"| {mode} | {m('auc'):.4f} | {m('corr_logw'):.3f} | {m('rmse_logw'):.3f} | {m('kl_plugin'):.4f} | {m('dv'):.4f} | "
      f"{m('ece'):.4f} | {m('norm_q'):.4f} | {m('norm_p'):.4f} |")
P('')
P('Tempered path L(α), G = quartic (mean over replicates):\n')
P('| α | exact | bridge, true w | SNIS from q, true w | bridge, MLP ŵ | SNIS from q, MLP ŵ |')
P('|---|---|---|---|---|---|')
gr = by('ii', 'mode', 'g')
tr, ml = gr.get(('true', 'quartic')), gr.get(('mlp', 'quartic'))
if tr and ml:
    ex = tr[0]['L_exact']
    mb = lambda rs, k: np.mean([r[k] for r in rs], 0)
    for i, a in enumerate(np.linspace(0, 1, len(ex))):
        if i % (len(ex) // 5):
            continue
        P(f"| {a:.1f} | {ex[i]:.4f} | {mb(tr, 'L_bridge')[i]:.4f} | {mb(tr, 'L_q')[i]:.4f} | {mb(ml, 'L_bridge')[i]:.4f} | "
          f"{mb(ml, 'L_q')[i]:.4f} |")
P('')

# ------------------------------------------------------------------ (iii)
P('## (iii) Hand-built net that ignores structure (IgnoreNet; knock-out of a subset by Gaussian conditioning)\n')
P('Reference: paired direct Δ at n = 400k with the same fitted q (per replicate). "err" = estimate − reference, '
  'mean ± sd over replicates. MLP classifiers.\n')
P('| subset swapped | Δ reference | direct err | h-space AUC | h-space −Cov err | h-space bridge err | '
  'output-space AUC | output-space −Cov err | output-space bridge err | output-space gated −Cov err | ptp L(α), output space |')
P('|---|---|---|---|---|---|---|---|---|---|---|')
for (s,), rs in by('iii', 'subset').items():
    m = lambda k: np.mean([r[k] for r in rs])
    e = lambda k: f"{np.mean([r[k] - r['ref'] for r in rs]):+.4f} ± {np.std([r[k] - r['ref'] for r in rs], ddof=1):.4f}"
    P(f"| {s} | {m('ref'):+.4f} | {e('delta')} | {m('h_auc'):.3f} | {e('h_cov')} | {e('h_bridge')} | "
      f"{m('o_auc'):.3f} | {e('o_cov')} | {e('o_bridge')} | {e('o_gated_cov')} | {m('o_Lflat'):.4f} |")
P('')
P('Cross-layer consistency (layer l = h, layer l+1 = ReLU layer; q_l = conditional Gaussian; family at l+1 = conditional Gaussian):\n')
P('| layer-l structure read by F | R | AUC(q~ vs q^fit) | Γ_fit (rms) ± SE | Δ(q~) − Δ(q^fit) ± SE | Γ_proj | '
  'pass rate: METHOD rule | Γ rule only | two-sided |')
P('|---|---|---|---|---|---|---|---|---|')
for (cb,), rs in by('iii_cons', 'c_bimodal').items():
    m = lambda k: np.mean([r[k] for r in rs])
    P(f"| {'yes (c1 bimodal)' if cb else 'no (only the ignored b1 is non-Gaussian)'} | {len(rs)} | {m('auc'):.3f} | "
      f"{m('gamma_fit'):+.4f} ± {m('gamma_fit_se'):.4f} | {m('gamma_delta'):+.4f} ± {m('gamma_delta_se'):.4f} | "
      f"{m('gamma_proj'):+.4f} | {m('passed'):.2f} | {m('passed_gamma_only'):.2f} | {m('passed_two_sided'):.2f} |")
P('')

# ------------------------------------------------------------------ (iv)
P('## (iv) q → p: MixWorld q_t (t=0 moment-matched Gaussian, t=1 equals p), G = quartic, MLP classifier\n')
P('| t | truth Δ | KL(p‖q) | χ² | AUC | KL plug-in | T1 pass (MLP) | estimator | bias | rel. bias | emp. sd | sd/SE | '
  'coverage | coverage incl. refit | RMSE |')
P('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
for (t,), rs in by('iv_mix', 't').items():
    tr = rs[0]['truth']
    first = True
    for k, name in [('direct', 'direct'), ('true_cov_plain', 'Cov plain, true w'), ('true_bridge', 'bridge, true w'),
                    ('mlp_cov_plain', 'Cov plain, MLP ŵ'), ('mlp_bridge', 'bridge, MLP ŵ'),
                    ('mlp_bridge_int', '−∫dL/dα, MLP ŵ')]:
        s = est_stats(rs, k, tr)
        head = (f"| {t} | {tr:.4f} | {rs[0]['kl_true']:.4f} | {rs[0]['chi2_true']:.4f} | "
                f"{np.mean([r['auc'] for r in rs]):.4f} | {np.mean([r['kl_plugin'] for r in rs]):.4f} | "
                f"{np.mean([r['mlp_T1'] for r in rs]):.2f} |") if first else '| | | | | | | |'
        first = False
        rel = s['bias'] / tr if tr else np.nan
        P(f"{head} {name} | {s['bias']:+.4f} | {f(rel, 2)} | {s['sd']:.4f} | {f(s['ratio'], 2)} | {f(s['cover'], 2)} | "
          f"{f(s['cover_tot'], 2)} | {s['rmse']:.4f} |")
P('')
P('## (iv-b) Scale family in d = 31: p = N(0, I), q = N(0, κ²I), G = |h|²/d, true w (and quadratic classifier where marked)\n')
P('| κ | truth Δ | χ²(p‖q) | k̂ (q side) | ESS_q/n | min bridge ESS/pool | estimator | bias | emp. sd | sd/SE | coverage |')
P('|---|---|---|---|---|---|---|---|---|---|---|')
for (k, clf), rs in by('iv_scale', 'k', 'clf').items():
    tr = rs[0]['truth']; first = True
    for key, name in [('direct', 'direct'), ('cov_plain', 'Cov plain IS'), ('cov_snis', 'Cov SNIS'),
                      ('temper_q', 'L(0)-L(1) SNIS from q'), ('bridge', 'bridge')]:
        s = est_stats(rs, key, tr)
        lab = f"{k} (quad ŵ, R={len(rs)}, AUC {np.mean([r['auc'] for r in rs]):.3f})" if clf else f"{k} (R={len(rs)})"
        head = (f"| {lab} | {tr:.4f} | {rs[0]['chi2_true']:.3g} | {np.mean([r['khat_q'] for r in rs]):.2f} | "
                f"{np.mean([r['ess_q'] for r in rs]):.3f} | {np.mean([r['ess_bridge_min'] for r in rs]):.3f} |") \
            if first else '| | | | | | |'
        first = False
        P(f"{head} {name} | {s['bias']:+.4f} | {f(s['sd'], 3)} | {f(s['ratio'], 2)} | {f(s['cover'], 2)} |")
P('')
dst = os.path.join(HERE, 'results', 'tables_quick.md' if 'quick' in src else 'tables.md')
open(dst, 'w').write('\n'.join(out))
print('\n'.join(out))
