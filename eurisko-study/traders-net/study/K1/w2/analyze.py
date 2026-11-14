"""Aggregate K1 worker2 runs -> results/{shift_runs,csweep_runs}.csv, results/summary.json, PNGs."""
import csv, glob, json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(os.path.dirname(HERE), 'runs')
OUT = os.path.join(HERE, 'results')
SHIFT = 10000
CSTAR = {'10k': 0.04, '1k-nn': 0.000625}


def load(rid):
    d = os.path.join(RUNS, rid)
    rows = list(csv.DictReader(open(os.path.join(d, 'metrics.csv'))))
    cols = {k: np.array([float(r[k]) if r[k] not in ('', 'None') else np.nan for r in rows]) for k in rows[0]}
    return cols, json.load(open(os.path.join(d, 'w2extra.json')))


def at(c, lo, hi, k):
    m = (c['step'] > lo) & (c['step'] <= hi)
    return float(np.nanmean(c[k][m]))


def last(c, hi, k):
    i = np.flatnonzero(c['step'] <= hi)[-1]
    return float(c[k][i])


def block(c, hi):
    """end-of-window economy metrics at step hi (tail = last 10% of the run window)."""
    lo = hi * 0.9
    r = {k: last(c, hi, k) for k in ('alive', 'gini_traders', 'gini_null_traders', 'top_clade_share', 'eff_clades',
                                     'clades_1pct', 'top_species_share', 'alive_null', 'top_clade_share_null',
                                     'W_traders', 'W_traders_null', 'births', 'bankrupt', 'cons_resid_ledger',
                                     'cons_resid_settle')}
    r['gini_gap'] = r['gini_traders'] - r['gini_null_traders']
    r['brier'] = at(c, lo, hi, 'brier_all')
    r['regret'] = r['brier'] - at(c, lo, hi, 'brier_oracle_all')
    r['regret_null'] = at(c, lo, hi, 'brier_all_null') - at(c, lo, hi, 'brier_oracle_all')
    r['bce'] = at(c, lo, hi, 'bce_all')
    r['monoculture'] = r['top_clade_share'] >= 0.9
    return r


def shift_metrics(c):
    s, e = c['step'], c['excess_regret_trk']
    pre = at(c, SHIFT - 1000, SHIFT, 'excess_regret_trk') if np.isfinite(e[(s > SHIFT - 1000) & (s <= SHIFT)]).any() else float('nan')
    post = s > SHIFT
    ep = e[post] - (pre if np.isfinite(pre) else 0)
    pk = float(np.nanmax(ep))
    rec = float('nan')
    ok = ep <= 0.1 * pk
    for i in range(len(ok) - 4):
        if ok[i:i + 5].all():
            rec = float(s[post][i] - SHIFT)
            break
    r = dict(excess_pre=pre, excess_peak=pk, recovery_steps=rec,
             excess_last2k=at(c, 18000, 20000, 'excess_regret_trk') - (pre if np.isfinite(pre) else 0),
             regret_trk_first500=at(c, SHIFT, SHIFT + 500, 'regret_trk'),
             regret_trk_last2k=at(c, 18000, 20000, 'regret_trk'),
             regret_untrk_last2k=at(c, 18000, 20000, 'regret_untrk'))
    if 'newborn_share_trk' in c:
        r['newborn_share_trk_end'] = last(c, 20000, 'newborn_share_trk')
        r['wshare_trk_pre'] = last(c, SHIFT - 100, 'wshare_trk') if np.isfinite(last(c, SHIFT - 100, 'wshare_trk')) else float('nan')
        r['wshare_trk_end'] = last(c, 20000, 'wshare_trk')
    return r


def ms(vals):
    v = np.array([x for x in vals if x is not None], float)
    return [float(np.nanmean(v)), float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else float('nan')]


def main():
    os.makedirs(OUT, exist_ok=True)
    summ = {'shift': {}, 'csweep': {}}
    srows, crows = [], []
    for tag in CSTAR:
        for sh in ('on', 'off'):
            arm = f'S-{sh}{tag}'
            per = []
            for seed in (1, 2, 3):
                rid = f'w2-{arm}-s{seed}'
                if not os.path.exists(os.path.join(RUNS, rid, 'w2extra.json')):
                    continue
                c, x = load(rid)
                r = dict(arm=arm, seed=seed, run=rid, **block(c, 20000), **shift_metrics(c),
                         lockin_spearman_clade=x['lockin_spearman_clade'])
                r['pre_alive'] = last(c, SHIFT, 'alive'); r['pre_gini_gap'] = last(c, SHIFT, 'gini_traders') - last(c, SHIFT, 'gini_null_traders')
                per.append(r); srows.append(r)
            if per:
                summ['shift'][arm] = {k: ms([p[k] for p in per]) for k in per[0] if isinstance(per[0][k], (int, float, bool, np.floating)) and k != 'seed'}
                summ['shift'][arm]['n_seeds'] = len(per)
        for cr in (0, 0.0025, 0.01, 0.03, 0.1, 0.3, 1, 3):
            arm = f'S-off{tag}' if cr == 0.0025 else f'C{cr:g}-{tag}'
            per = []
            for seed in (1, 2, 3):
                rid = f'w2-{arm}-s{seed}'
                if not os.path.exists(os.path.join(RUNS, rid, 'w2extra.json')):
                    continue
                c, x = load(rid)
                r = dict(tag=tag, c_over_cstar=cr, c=cr * CSTAR[tag], seed=seed, run=rid, **block(c, 10000))
                per.append(r); crows.append(r)
            if per:
                summ['csweep'][f'{tag}@{cr:g}'] = {k: ms([p[k] for p in per]) for k in per[0] if isinstance(per[0][k], (int, float, bool, np.floating)) and k not in ('seed',)}
                summ['csweep'][f'{tag}@{cr:g}']['n_seeds'] = len(per)
    for name, rows in (('shift_runs.csv', srows), ('csweep_runs.csv', crows)):
        if rows:
            keys = list(rows[0])
            with open(os.path.join(OUT, name), 'w', newline='') as f:
                w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    allmono = all(r['monoculture'] for r in srows + crows) if srows + crows else None
    summ['stop_rule_monoculture_every_arm'] = allmono
    json.dump(summ, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
    plots()
    print(json.dumps(summ, indent=1)[:6000])


def plots():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    col = {'on': '#2a78d6', 'off': '#eb6834'}
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, tag, ttl in zip(axs, CSTAR, ('N=10k, Q=64, niches on', 'N=1k, Q=64, niches off')):
        for sh in ('on', 'off'):
            for seed in (1, 2, 3):
                rid = f'w2-S-{sh}{tag}-s{seed}'
                p = os.path.join(RUNS, rid, 'w2extra.json')
                if not os.path.exists(p):
                    continue
                c, _ = load(rid)
                ax.plot(c['step'], c['excess_regret_trk'], color=col[sh], lw=1.5, alpha=0.8,
                        label=f'shift {sh}' if seed == 1 else None)
        ax.axvline(SHIFT, color='#888', lw=1, ls=':')
        ax.set_title(ttl, fontsize=10); ax.set_xlabel('step'); ax.set_ylabel('Brier regret, shifted - other questions')
        ax.spines[['top', 'right']].set_visible(False); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'shift_excess_regret.png'), dpi=110); plt.close(fig)
    rows = list(csv.DictReader(open(os.path.join(OUT, 'csweep_runs.csv')))) if os.path.exists(os.path.join(OUT, 'csweep_runs.csv')) else []
    if not rows:
        return
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for tag, cc, lab in (('10k', '#2a78d6', 'N=10k niches on'), ('1k-nn', '#eb6834', 'N=1k niches off')):
        R = [r for r in rows if r['tag'] == tag]
        xs = sorted(set(float(r['c_over_cstar']) for r in R))
        xp = [max(x, 1e-3) for x in xs]
        for ax, k, yl in zip(axs, ('alive', 'gini_gap', 'regret'), ('survivors (alive)', 'Gini - Gini(shuffled null)', 'Brier regret (tail)')):
            m = [np.mean([float(r[k]) for r in R if float(r['c_over_cstar']) == x]) for x in xs]
            s = [np.std([float(r[k]) for r in R if float(r['c_over_cstar']) == x], ddof=1) for x in xs]
            ax.errorbar(xp, m, yerr=s, color=cc, lw=2, marker='o', ms=6, capsize=3, label=lab)
            ax.set_xscale('log'); ax.set_xlabel('c / c*  (0 plotted at 1e-3)'); ax.set_ylabel(yl)
            ax.spines[['top', 'right']].set_visible(False)
    axs[0].legend(frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'csweep.png'), dpi=110); plt.close(fig)


if __name__ == '__main__':
    main()
