import json, sys, glob, numpy as np
for tf in sorted(glob.glob('runs/dev/truth_n*_seed*.json')):
    t = json.load(open(tf)); n, L, s = t['n'], t['L'], t['seed']
    tr = np.array(t['truth']); V = t['V']
    rq = {int(m): v['nmse'] for m, v in t['rqmc'].items()}
    qd = {int(m): v['nmse'] for m, v in t['qmc_det'].items()}
    print(f"== n={n} L={L} seed={s} V={V:.3g} rho_rb={t['rho_rb']:.3f} truth_nmse={t['truth_nmse']:.1e}")
    print('   RQMC: ' + ' '.join(f"2^{m}:{v:.1e}" for m, v in rq.items() if m % 2 == 0))
    print('   QMCd: ' + ' '.join(f"2^{m}:{v:.1e}" for m, v in qd.items() if m % 2 == 0))
    for pr in ('gap', 'instab'):
        f = f'runs/dev/bnb_{pr}_n{n}_L{L}_seed{s}.json'
        try: b = json.load(open(f))
        except Exception: continue
        line = []
        for r in b['records']:
            if r['checkpoint'] == 'final' or (isinstance(r['checkpoint'], int) and (r['checkpoint'].bit_length() - 1) % 2 == 0 and r['checkpoint'] >= 2**10):
                p = np.array(r['point']); lb = np.array(r['lb']); ub = np.array(r['ub'])
                e = np.sum((p - tr) ** 2) / V; c = np.sum(((ub - lb) / 2) ** 2) / V
                N = r['passes']
                # RQMC interpolated at equal passes (log-log)
                ms = sorted(rq); xs = np.log2([2.0 ** m for m in ms]); ys = np.log(np.array([rq[m] for m in ms]))
                rqi = float(np.exp(np.interp(np.log2(N), xs, ys)))
                line.append(f"{N:.1e}p:pt{e:.1e}/cert{c:.1e}/ex{r['mass_exact']:.2f}/RQMC{rqi:.1e}")
        print(f'   BnB-{pr}: ' + ' '.join(line))
