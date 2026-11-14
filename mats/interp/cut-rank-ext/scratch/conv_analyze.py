"""Amendment 2 convergence check: ARC2 and K2X with cut moments from 8,192 against 65,536 directions on 64 test fronts.
Scale-normalised Err/Vbar on those fronts (as nl_analyze.py), and the RMS change between the two estimates relative to
sqrt(Vbar). usage: conv_analyze.py n c d ..."""
import sys
import numpy as np
args = sys.argv[1:]
for i in range(0, len(args), 3):
    n, c, d = args[i:i + 3]
    base = f'/home/agent/workspaces/mats/work/cut-rank-ext/runs/nl_n{n}_c{c}_d{d}'
    D = np.load(base + '.npz'); C = np.load(base + '_conv.npz')
    fr = C['fronts']
    MB, MC, E2 = D['MB'], D['MC'], D['E2']
    T = 0.5 * (MB + MC)
    s_f = np.sqrt(E2.mean(axis=(1, 2)))
    alive = s_f > 0
    t_b = np.sqrt((E2[alive] / s_f[alive, None, None] ** 2).mean(axis=(0, 2)))
    sc = (s_f[fr, None, None] * t_b[None, :, None])
    ok = alive[fr]
    Tn = (T[fr] / sc)[ok]
    Vn = (np.clip(E2 - T ** 2, 0, None)[fr] / sc ** 2)[ok]
    Vbar = Vn.mean()
    out = [f'(n,c,d)=({n},{c},{d}), {ok.sum()} fronts']
    for k in ('A2', 'K2'):
        e8 = np.mean((Tn - (C[k + '_8k'] / sc)[ok]) ** 2) / Vbar
        e64 = np.mean((Tn - (C[k + '_64k'] / sc)[ok]) ** 2) / Vbar
        dlt = np.mean(((C[k + '_8k'] - C[k + '_64k']) / sc)[ok] ** 2) / Vbar
        out.append(f'{k}: Err/Vbar 8k {e8:.3g}, 64k {e64:.3g}, change {dlt:.2g}')
    print(' | '.join(out))
