"""K1 worker3 analysis: reads K1/runs/w3-*/metrics.csv -> out/summary_N<N>.csv, out/summary.json."""
import csv, glob, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(HERE), 'runs')
OUT = os.path.join(HERE, 'out'); os.makedirs(OUT, exist_ok=True)


def load(p):
    r = list(csv.DictReader(open(p)))
    f = lambda k: np.array([float(x[k]) if x.get(k, '') not in ('', 'nan') else np.nan for x in r])
    return r, f


def logit(s):
    s = np.clip(s, 1e-12, 1 - 1e-12); return np.log(s / (1 - s))


def run_stats(p):
    r, f = load(p)
    h = len(r) // 2
    sec = lambda k: float(np.nanmean(f(k)[h:]))
    d = {'brier2': sec('brier_all'), 'brier2_null': sec('brier_all_null'), 'bce2': sec('bce_all'),
         'brier2_oracle': sec('brier_oracle_all'), 'eff_species_q2': sec('eff_species_q'),
         'eff_leaves_q2': sec('eff_leaves_q'), 'gini_T': f('gini_traders')[-1], 'gini_null_T': f('gini_null_traders')[-1],
         'top_clade_T': f('top_clade_share')[-1], 'eff_clades_T': f('eff_clades')[-1], 'alive_T': f('alive')[-1],
         'bankrupt_T': f('bankrupt')[-1], 'births_T': f('births')[-1], 'W_T': f('W_traders')[-1],
         'cons_resid_max': float(np.nanmax(f('cons_resid_ledger')))}
    for tag in ('', 'null_'):
        sf = f(f'{tag}share_fund')
        d[f'{tag}premium_agg'] = float(logit(sf[-1]) - logit(sf[0]))
        d[f'{tag}share_fund_T'] = float(sf[-1])
        d[f'{tag}premium_late'] = float(logit(sf[-1]) - logit(sf[h]))   # exploratory: second half only
        d[f'{tag}fund_alive_T'] = float(f(f'{tag}alive_fund_frac')[-1])
        d[f'{tag}bhgrp_alive_T'] = float(f(f'{tag}alive_bhgrp_frac')[-1])
        d[f'{tag}premium_median'] = float(f(f'{tag}premium_median')[-1])
    return d


def main():
    rows = []
    for p in sorted(glob.glob(os.path.join(RUNS, 'w3-*', 'metrics.csv'))):
        rid = os.path.basename(os.path.dirname(p))
        _, rule, topo, N, s = rid.split('-')
        d = run_stats(p); d.update(rule=rule, topo=topo, N=int(N[1:]), seed=int(s[1:]), run=rid, path=p)
        rows.append(d)
    res = {}
    for N in sorted({r['N'] for r in rows}):
        flat = {r['seed']: r for r in rows if r['N'] == N and r['rule'] == 'bh' and r['topo'] == 'within'}
        out = []
        for rule in ('bh', 'crp', 'eg', 'fof'):
            for topo in ('within', 'cross'):
                rs = sorted([r for r in rows if r['N'] == N and r['rule'] == rule and r['topo'] == topo], key=lambda r: r['seed'])
                if not rs:
                    continue
                a = {'N': N, 'arm': f'{rule}-{topo}', 'seeds': len(rs)}
                keys = [k for k in rs[0] if isinstance(rs[0][k], float) or isinstance(rs[0][k], (int, np.floating)) and k not in ('N', 'seed')]
                for k in keys:
                    if k in ('N', 'seed'):
                        continue
                    v = np.array([r[k] for r in rs], float)
                    a[k] = float(v.mean()); a[k + '_sd'] = float(v.std(ddof=1)) if len(v) > 1 else float('nan')
                dv = np.array([r['brier2'] - flat[r['seed']]['brier2'] for r in rs if r['seed'] in flat])
                a['dbrier_vs_flat'] = float(dv.mean()) if len(dv) else float('nan')
                a['dbrier_vs_flat_sd'] = float(dv.std(ddof=1)) if len(dv) > 1 else float('nan')
                a['dbrier_vs_flat_per_seed'] = dv.tolist()
                a['premium_agg_per_seed'] = [r['premium_agg'] for r in rs]
                a['premium_late_per_seed'] = [r['premium_late'] for r in rs]
                a['runs'] = [r['path'] for r in rs]
                out.append(a)
        res[N] = out
        cols = ['arm', 'seeds', 'brier2', 'brier2_sd', 'dbrier_vs_flat', 'dbrier_vs_flat_sd', 'brier2_oracle', 'brier2_null',
                'premium_agg', 'premium_agg_sd', 'null_premium_agg', 'null_premium_agg_sd', 'premium_late', 'premium_late_sd', 'null_premium_late', 'null_premium_late_sd', 'fund_alive_T', 'fund_alive_T_sd',
                'bhgrp_alive_T', 'bhgrp_alive_T_sd', 'share_fund_T', 'eff_species_q2', 'eff_species_q2_sd',
                'eff_leaves_q2', 'eff_leaves_q2_sd', 'gini_T', 'gini_T_sd', 'gini_null_T', 'top_clade_T', 'top_clade_T_sd',
                'eff_clades_T', 'eff_clades_T_sd', 'bankrupt_T', 'births_T', 'W_T', 'cons_resid_max']
        with open(os.path.join(OUT, f'summary_N{N}.csv'), 'w', newline='') as fh:
            w = csv.writer(fh); w.writerow(cols)
            for a in out:
                w.writerow([a.get(c) if isinstance(a.get(c), str) else ('%.6g' % a[c] if c in a else '') for c in cols])
    json.dump({'per_run': rows, 'arms': {str(k): v for k, v in res.items()}}, open(os.path.join(OUT, 'summary.json'), 'w'),
              indent=1, default=float)
    for N, out in res.items():
        print(f'N={N}')
        for a in out:
            print(f"  {a['arm']:12s} brier2 {a['brier2']:.4f}±{a['brier2_sd']:.4f} dB {a['dbrier_vs_flat']:+.4f}±{a['dbrier_vs_flat_sd']:.4f}"
                  f" prem {a['premium_agg']:+.2f}±{a['premium_agg_sd']:.2f} (null {a['null_premium_agg']:+.2f})"
                  f" fundAlive {a['fund_alive_T']:.2f} bhAlive {a['bhgrp_alive_T']:.2f} effSp {a['eff_species_q2']:.2f}±{a['eff_species_q2_sd']:.2f}"
                  f" gini {a['gini_T']:.2f}/{a['gini_null_T']:.2f} top {a['top_clade_T']:.3f} cons {a['cons_resid_max']:.1e}")


if __name__ == '__main__':
    main()
