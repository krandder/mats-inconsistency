"""Regime sweep for the rewiring audit (PREREG-FA.md): one line of JSON per configuration.
usage: python sweep_fa.py b|a OUT.jsonl"""
import itertools, json, sys, time
import audit_fa as A

kind, out = sys.argv[1], sys.argv[2]
grid = []
if kind == 'b':
    for delta, sig2, rho in itertools.product((0.5, 1.0, 2.0, 4.0), (2.0, 4.0, 8.0), (0.0, 0.3, 0.6)):
        grid.append((dict(prior='b', b_hyper=[(delta, sig2, rho, 1.0)], T_train=(0, 32)), 3, 16, 3000))
    for delta, sig2 in itertools.product((1.0, 2.0, 4.0), (2.0, 4.0, 8.0)):       # is there a common source?
        grid.append((dict(prior='b', b_hyper=[(delta, sig2, 0.0, 0.5), (delta, sig2, 0.6, 0.5)], T_train=(0, 32)), 3, 16, 3000))
elif kind == "a":
    for scale, pf in itertools.product((0.5, 1.0, 2.0, 4.0), ('flat', 'high')):
        pfv = None if pf == 'flat' else (1, 1, 1, 2, 3)
        grid.append((dict(prior='a', a_m=[x * scale for x in (0.5, 1.5, 4.0)], a_pf=pfv, T_train=(0, 32)), 3, 7, 600))
if kind == 'a2':                      # structural levers: bigger blocks, more near-copies, noisier traders
    for scale, al, pf in itertools.product((2.0, 4.0, 8.0), ((1, 1, 1, 1), (1, 1, 2, 4)), ((1, 1, 1, 2, 3), (1, 1, 1, 3, 8))):
        grid.append((dict(prior='a', a_m=[x * scale for x in (0.5, 1.5, 4.0)], a_alpha=list(al), a_pf=list(pf),
                          T_train=(0, 32)), 3, 7, 600))
if kind == 'a4':                      # signed loadings (contrarian exposure to a shared source)
    base = dict(prior='a', a_alpha=[1, 1, 2, 4], a_pf=[1, 1, 1, 3, 8], T_train=(0, 32))
    for scale, ps in itertools.product((1.0, 2.0, 3.0), (0.25, 0.4)):
        grid.append((dict(base, a_m=[x * scale for x in (0.5, 1.5, 4.0)], a_psign=ps), 3, 7, 500))
if kind == 'a3':                      # accurate-looking shared blocks; signed loadings
    base = dict(prior='a', a_alpha=[1, 1, 2, 4], a_pf=[1, 1, 1, 3, 8], T_train=(0, 32))
    for scale, pmb, ps in itertools.product((1.0, 2.0), (None, (4, 1, 1)), (0.0, 0.25)):
        grid.append((dict(base, a_m=[x * scale for x in (0.5, 1.5, 4.0)], a_pm_block=pmb, a_psign=ps), 3, 7, 500))
with open(out, 'a') as fh:
    for cfg, nmin, nmax, E in grid:
        t = time.time()
        r = A.audit(cfg, nmin, nmax, E)
        s = r['all']
        row = dict(cfg=cfg, nmin=nmin, nmax=nmax, E=E, miss_R2_diag=s['miss_R2_diag_bayes'], miss_mse_diag=s['miss_mse_diag_bayes'],
                   gap_diag=s['gap_diag_bayes'][0], by_T_diag={k: round(v['miss_R2_diag_bayes'], 3) for k, v in r['by_T'].items()},
                   miss_R2_prec=s['miss_R2_prec'], miss_mse_prec=s['miss_mse_prec'],
                   miss_R2_equal=s['miss_R2_equal'], gap_prec=s['gap_prec'][0], gap_T0=s['gap_bayes_T0'][0],
                   gap_gls=s['gap_gls_plugin'][0], gap_oracle=s['gap_oracle'][0],
                   by_T={k: round(v['miss_R2_prec'], 3) for k, v in r['by_T'].items()}, sec=time.time() - t)
        fh.write(json.dumps(row) + '\n'); fh.flush()
        print(json.dumps(row), flush=True)
