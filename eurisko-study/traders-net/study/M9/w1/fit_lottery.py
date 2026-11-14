"""M9(a) worker 1: lottery-model fit of the soup's per-family Brier on K/Q.
Reads logical-soup/runs/phase3/*/metrics.csv (read-only). Writes CSV/JSON/PNG next to this file.
Run: PYTHONPATH=/tmp/traders-net/M9/w1/pylib python3 fit_lottery.py
"""
import csv, json, math, os, glob
import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
SOUP = os.path.abspath(os.path.join(HERE, '../../../../logical-soup/runs/phase3'))
FAMS = ['periodic', 'prime', 'and', 'or', 'lfsr', 'ca30', 'ca110', 'pi']
PRICED = ['periodic', 'prime', 'and']
RNG = np.random.default_rng(20260928)


def load():
    rows = []
    for d in sorted(glob.glob(os.path.join(SOUP, 'k*-f*-s*'))):
        name = os.path.basename(d)
        cfg = json.load(open(os.path.join(d, 'config.json')))
        with open(os.path.join(d, 'metrics.csv')) as fh:
            recs = list(csv.DictReader(fh))
        last = recs[-50:]
        parts = name.split('-')
        variant = parts[3] if len(parts) > 3 else 'base'
        K = cfg['K']; Q = 2 ** cfg['logq']
        r = dict(run=name, cell='-'.join(parts[:2]), seed=int(parts[2][1:]), variant=variant,
                 K=K, Q=Q, x=K / Q, epochs=int(recs[-1]['epoch']) + 1)
        for f in FAMS:
            r[f] = float(np.mean([float(z['brier_' + f]) for z in last])) / 1e6
        rows.append(r)
    return rows


def lot(p, x, b0=0.25):
    d, lp = p[0], p[1]
    pi = 1 / (1 + math.exp(-lp))
    return b0 - d * (1 - np.exp(x * np.log1p(-pi)))


LPGRID = np.linspace(-25, 10, 1401)


def fit_L2(x, y, refine=True):
    """Profile Delta out analytically on a logit(pi) grid, then refine locally."""
    pis = 1 / (1 + np.exp(-LPGRID))
    G = 1 - np.exp(np.outer(np.log1p(-pis), x))          # (grid, n)
    t = 0.25 - y
    gg = (G * G).sum(1); gt = G @ t
    d = np.clip(np.where(gg > 0, gt / np.maximum(gg, 1e-300), 0), -1, 1)
    sse = ((t[None, :] - d[:, None] * G) ** 2).sum(1)
    i = int(np.argmin(sse)); d0, lp0, s0 = d[i], LPGRID[i], sse[i]
    if refine:
        r = least_squares(lambda p: lot(p, x) - y, [d0, lp0], bounds=([-1, -30], [1, 15]))
        if 2 * r.cost < s0:
            d0, lp0, s0 = r.x[0], r.x[1], 2 * r.cost
    return dict(delta=float(d0), pi=float(1 / (1 + math.exp(-lp0))), sse=float(s0), k=2)


def fit_L3(x, y):
    best = None
    for lp0 in np.linspace(-12, 4, 17):
        for d0 in (-0.05, 0.05, 0.2):
            r = least_squares(lambda p: lot(p[:2], x, p[2]) - y, [d0, lp0, 0.25],
                              bounds=([-1, -30, 0], [1, 15, 0.5]))
            if best is None or r.cost < best.cost:
                best = r
    d, lp, b0 = best.x
    return dict(delta=d, pi=1 / (1 + math.exp(-lp)), b0=b0, sse=2 * best.cost, k=3)


def fit_log(x, y):
    A = np.c_[np.ones_like(x), np.log(x)]
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    return dict(a=c[0], b=c[1], sse=float(((A @ c - y) ** 2).sum()), k=2)


def fit_mm(x, y):
    """Saturating alternative: 0.25 - D*x/(x+h) (Michaelis-Menten), profile D on an h grid."""
    best = None
    for h in np.logspace(-2, 4, 1201):
        g = x / (x + h); t = 0.25 - y
        d = float(g @ t / (g @ g)); sse = float(((t - d * g) ** 2).sum())
        if best is None or sse < best['sse']:
            best = dict(delta=d, h=float(h), sse=sse, k=2)
    return best


def fit_L2K(x, K, y):
    """Lottery on K/Q plus a log K term: does K matter beyond K/Q?"""
    L2 = fit_L2(x, y)
    lp0 = math.log(L2['pi'] / (1 - L2['pi']))
    r = least_squares(lambda p: lot(p[:2], x) + p[2] * np.log(K / 16384) - y, [L2['delta'], lp0, 0.0],
                      bounds=([-1, -30, -1], [1, 15, 1]))
    return dict(delta=r.x[0], pi=1 / (1 + math.exp(-r.x[1])), c_logK=r.x[2], sse=2 * r.cost, k=3)


def stats(y, sse, k):
    n = len(y); sst = float(((y - y.mean()) ** 2).sum())
    return dict(r2=1 - sse / sst if sst > 0 else float('nan'),
                aic=n * math.log(max(sse, 1e-300) / n) + 2 * k)


def x_for_fraction(pi, frac):
    return math.log(1 - frac) / math.log1p(-pi)


def main():
    rows = load()
    with open(os.path.join(HERE, 'soup_runs.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    base = [r for r in rows if r['variant'] == 'base']
    var = [r for r in rows if r['variant'] != 'base']
    x = np.array([r['x'] for r in base]); K = np.array([r['K'] for r in base], float)
    Q = np.array([r['Q'] for r in base], float); seeds = np.array([r['seed'] for r in base])
    out, fitrows = {}, []
    for f in FAMS:
        y = np.array([r[f] for r in base])
        sst = float(((y - y.mean()) ** 2).sum())
        m = {}
        L2 = fit_L2(x, y); m['L2'] = {**L2, **stats(y, L2['sse'], 2)}
        L3 = fit_L3(x, y); m['L3'] = {**L3, **stats(y, L3['sse'], 3)}
        LG = fit_log(x, y); m['LOG'] = {**LG, **stats(y, LG['sse'], 2)}
        LK = fit_L2(K / 4096, y); m['LK'] = {**LK, **stats(y, LK['sse'], 2)}
        LQ = fit_L2(4096 / Q, y); m['LQ'] = {**LQ, **stats(y, LQ['sse'], 2)}
        MM = fit_mm(x, y); m['MM'] = {**MM, **stats(y, MM['sse'], 2)}
        LKK = fit_L2K(x, K, y); m['L2K'] = {**LKK, **stats(y, LKK['sse'], 3)}
        m['NULL'] = {'sse': sst, 'k': 1, **stats(y, sst, 1)}
        # per-seed fits
        ps = []
        for s in (1, 2, 3):
            sel = seeds == s
            ps.append(fit_L2(x[sel], y[sel]))
        m['L2_per_seed'] = dict(pi=[p['pi'] for p in ps], delta=[p['delta'] for p in ps],
                                log10pi_mean=float(np.mean([math.log10(p['pi']) for p in ps])),
                                log10pi_sd=float(np.std([math.log10(p['pi']) for p in ps], ddof=1)),
                                delta_mean=float(np.mean([p['delta'] for p in ps])),
                                delta_sd=float(np.std([p['delta'] for p in ps], ddof=1)))
        # bootstrap CI on pi (resample runs)
        bpi = []
        for _ in range(1000):
            idx = RNG.integers(0, len(y), len(y))
            if len(set(x[idx])) < 3:
                continue
            bpi.append(fit_L2(x[idx], y[idx], refine=False)['pi'])
        m['L2_boot_pi_ci95'] = [float(np.percentile(bpi, 2.5)), float(np.percentile(bpi, 97.5))]
        # shuffled soup table: permute K/Q labels over runs
        nr2 = []
        for _ in range(1000):
            xp = RNG.permutation(x)
            nr2.append(1 - fit_L2(xp, y, refine=False)['sse'] / sst)
        nr2 = np.array(nr2)
        m['shuffle_null_r2_mean'] = float(nr2.mean()); m['shuffle_null_r2_p95'] = float(np.percentile(nr2, 95))
        m['shuffle_p'] = float((1 + (nr2 >= m['L2']['r2']).sum()) / (1 + len(nr2)))
        # out-of-sample variants at K/Q = 4
        vy = np.array([r[f] for r in var]); vx = np.array([r['x'] for r in var])
        m['variants_mean_abs_err'] = float(np.abs(lot([0, 0], vx) * 0 + (0.25 - L2['delta'] * (1 - (1 - L2['pi']) ** vx)) - vy).mean())
        # per-cell observed means
        m['cells'] = {}
        for xv in sorted(set(x)):
            yy = y[x == xv]; m['cells'][str(xv)] = [float(yy.mean()), float(yy.std(ddof=1)), int(len(yy))]
        # K/Q for half / 90% of the fitted gain
        if L2['delta'] > 0.005:
            m['x_half'] = x_for_fraction(L2['pi'], 0.5); m['x_90'] = x_for_fraction(L2['pi'], 0.9)
        out[f] = m
        fitrows.append(dict(family=f, L2_delta=L2['delta'], L2_pi=L2['pi'], L2_r2=m['L2']['r2'], L2_aic=m['L2']['aic'],
                            L3_b0=L3['b0'], L3_r2=m['L3']['r2'], L3_aic=m['L3']['aic'],
                            LOG_r2=m['LOG']['r2'], LOG_aic=m['LOG']['aic'], LK_r2=m['LK']['r2'], LK_aic=m['LK']['aic'],
                            LQ_r2=m['LQ']['r2'], LQ_aic=m['LQ']['aic'],
                            MM_r2=m['MM']['r2'], MM_aic=m['MM']['aic'], L2K_clogK=m['L2K']['c_logK'], L2K_aic=m['L2K']['aic'], NULL_aic=m['NULL']['aic'],
                            pi_seed_log10_mean=m['L2_per_seed']['log10pi_mean'], pi_seed_log10_sd=m['L2_per_seed']['log10pi_sd'],
                            delta_seed_mean=m['L2_per_seed']['delta_mean'], delta_seed_sd=m['L2_per_seed']['delta_sd'],
                            pi_ci_lo=m['L2_boot_pi_ci95'][0], pi_ci_hi=m['L2_boot_pi_ci95'][1],
                            shuffle_r2_p95=m['shuffle_null_r2_p95'], shuffle_p=m['shuffle_p'],
                            x_half=m.get('x_half', ''), x_90=m.get('x_90', ''), variants_mae=m['variants_mean_abs_err']))
    json.dump(out, open(os.path.join(HERE, 'lottery_fit.json'), 'w'), indent=1, default=float)
    with open(os.path.join(HERE, 'lottery_fit.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(fitrows[0].keys())); w.writeheader()
        for r in fitrows:
            w.writerow({k: (f'{v:.5g}' if isinstance(v, float) else v) for k, v in r.items()})
    # plot
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 4, figsize=(14, 6.5), sharex=True)
    xs = np.logspace(-0.3, 2.3, 200)
    for ax, f in zip(axs.flat, FAMS):
        y = np.array([r[f] for r in base])
        ax.scatter(x * np.exp(RNG.normal(0, 0.04, len(x))), y, s=14, color='#4a6fa5', label='run (last 50 ep)')
        L2 = out[f]['L2']
        ax.plot(xs, 0.25 - L2['delta'] * (1 - (1 - L2['pi']) ** xs), color='#c0504d',
                label=f"lottery R²={L2['r2']:.2f}")
        ax.axhline(0.25, color='#999', lw=0.8, ls=':')
        ax.set_xscale('log'); ax.set_title(f); ax.legend(fontsize=7, frameon=False)
    for ax in axs[1]:
        ax.set_xlabel('traders per question K/Q')
    axs[0, 0].set_ylabel('Brier'); axs[1, 0].set_ylabel('Brier')
    fig.suptitle('Soup phase 3: per-family Brier vs K/Q with the lottery fit 0.25 − Δ(1−(1−π)^{K/Q})')
    fig.tight_layout(); fig.savefig(os.path.join(HERE, 'lottery_fit.png'), dpi=110)
    for f in FAMS:
        L2 = out[f]['L2']
        print(f"{f:9s} d={L2['delta']:+.4f} pi={L2['pi']:.3g} R2={L2['r2']:.3f} | LOG R2={out[f]['LOG']['r2']:.3f} "
              f"LK R2={out[f]['LK']['r2']:.3f} LQ R2={out[f]['LQ']['r2']:.3f} | AIC L2 {L2['aic']:.1f} LOG {out[f]['LOG']['aic']:.1f} "
              f"MM {out[f]['MM']['aic']:.1f} L2K {out[f]['L2K']['aic']:.1f} c={out[f]['L2K']['c_logK']:+.4f} NULL {out[f]['NULL']['aic']:.1f} | shufp={out[f]['shuffle_p']:.3f} x_half={out[f].get('x_half','-')}")


if __name__ == '__main__':
    main()
