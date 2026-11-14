"""POST HOC diagnostics of the ARC2/ARC3 readouts (not pre-registered): per-(front, back) relative errors on test fronts,
fraction of blow-ups, and errors after excluding them. usage: arc3_diag.py runs/NAME.npz ..."""
import sys
import numpy as np
for path in sys.argv[1:]:
    d = np.load(path); F = int(d['F'])
    T = 0.5 * (d['MB'] + d['MC']); te = slice(F // 2, F)
    s_f = np.sqrt(d['E2'].mean(axis=(1, 2)))
    ok = s_f[te] > 0
    Tt = T[te][ok] / s_f[te][ok, None, None]
    V = np.clip(d['E2'] - T ** 2, 0, None)[te][ok] / s_f[te][ok, None, None] ** 2
    Vbar = V.mean()
    out = [path.split('/')[-1]]
    for key in ('MF', 'K2', 'A2', 'A3'):
        X = d[key][te][ok] / s_f[te][ok, None, None]
        if np.all(np.isnan(X)):
            continue
        e = ((Tt - X) ** 2).mean(axis=2)                       # per (front, back)
        bad = e > 100 * Vbar
        med = np.nanmedian(e) / Vbar
        trim = np.nanmean(np.where(bad, np.nan, e)) / Vbar
        out.append(f'{key}: median {med:.3g}, blow-ups (>100 Vbar) {bad.mean():.3%}, mean excluding blow-ups {trim:.3g}')
    print(' | '.join(out))
