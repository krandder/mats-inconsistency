"""sscore.py -- 6x6 Sudoku: SCORING ONLY. Runs every policy on held-out puzzles (test orbits) and compares the books
with the exact solution sets (sexact: independent enumeration under the true rules), after each search has finished."""
import json
import time

import numpy as np

import sk, spol, sexact

ROOT = '/home/agent/workspaces/mats/work/deep-study/'
HERE = ROOT + 'wf13/loop/'
K, N = 6, 36


def route_logprob(rules, puz, sol, pol, rng, eps):
    """log of the policy's probability of producing the solution `sol` along its own order (value floor eps)."""
    b, c = puz.astype(np.int8).copy()[None], rules.cand_of(puz)[None]
    lp = 0.0
    while (b < 0).any():
        q, j = pol(b, c, rng)
        j = int(j[0])
        cj = c[0, j]
        qj = np.where(cj, q[0, j], 0.0)
        qj = qj / max(qj.sum(), 1e-300)
        qj = np.where(cj, (1 - eps) * qj + eps / max(cj.sum(), 1), 0.0)
        v = int(sol[j])
        lp += np.log(max(qj[v], 1e-300))
        b[0, j] = v
        c[0] &= rules.allowed[j, v]
    return lp


def score_all(seed, cfg, rules, nets, res, name, log):
    d = np.load(ROOT + 'wf7/sudoku/data/k6_test_min.npz')
    o = cfg['offset']       # registered runs use offset 1: puzzles never touched by the smoke runs (PREREG section 6)
    hard = d['puz'][np.where(d['rating'] == 2)[0][150 * o:150 * o + cfg['n_hard']]]
    mixed = d['puz'][1000 * o:1000 * o + cfg['n_mixed']]
    multi = np.load(ROOT + 'wf7/sudoku/data/k6_test_multi.npz')['puz'][100 * o:100 * o + cfg['n_multi']]
    sets = dict(hard=hard, mixed=mixed, multi=multi)
    exact = {k: [sexact.solutions(p) for p in v] for k, v in sets.items()}
    res['sets'] = {k: dict(n=len(v), clues=float((v >= 0).sum(1).mean()), solutions=float(np.mean([len(s) for s in exact[k]])))
                   for k, v in sets.items()}
    R, rho, eps, od = cfg['R'], cfg['rho'], cfg['eps'], cfg['order']
    fin = nets[R]
    arms = [('round%d' % r, spol.NetPolicy(nets[r], order=od), rho, eps) for r in range(R + 1)]
    arms += [('oneshot', spol.NetPolicy(nets['oneshot'], order=od), rho, eps),
             ('hand_mrv', spol.Hand('mrv'), rho, eps), ('bad_random', spol.Hand('random'), rho, eps),
             ('bad_perm', spol.NetPolicy(fin, 'perm', order=od), rho, eps),
             ('bad_avoid_noguard', spol.NetPolicy(fin, 'avoid', order=od), 0.0, 0.0),
             ('bad_avoid_floor', spol.NetPolicy(fin, 'avoid', order=od), 0.0, eps),
             ('bad_avoid_share', spol.NetPolicy(fin, 'avoid', order=od), rho, 0.0),
             ('final_rho0', spol.NetPolicy(fin, order=od), 0.0, eps), ('final_eps0', spol.NetPolicy(fin, order=od), rho, 0.0),
             ('final_mrvorder', spol.NetPolicy(fin, order='mrv'), rho, eps)]
    if cfg.get('arms', 'all') == 'min':            # post hoc runs score the loop, the control and the hand order only
        arms = [a for a in arms if a[0].startswith(('round', 'oneshot', 'hand'))]
    marks = cfg['marks']
    out = {}
    for an, pol, rh, ep in arms:
        t0 = time.time()
        out[an] = dict(rho=rh, eps=ep)
        for sn, puzzles in sets.items():
            rows, W, C = [], [], []
            for i, p in enumerate(puzzles):
                rng = np.random.default_rng(7000 * (i + 1) + seed)
                sols = exact[sn][i]
                exm, emp = sexact.marginals(sols), p < 0
                T = sk.Tree(rules, p, 6 * cfg['cap'] + 64)
                w, c, viol = [], [], 0
                for mk in marks:
                    sk.grow(T, pol, mk, rng, rho=rh, eps=ep)
                    bk = sk.books(T)
                    w.append(float((bk['hi'] - bk['lo'])[emp].mean()))
                    c.append(bk['closed'])
                    viol += int(((exm < bk['lo'] - 1e-9) | (exm > bk['hi'] + 1e-9))[emp].sum())
                same = bool(bk['closed'] and len(bk['sol']) == len(sols)
                            and {tuple(s) for s in bk['sol'].tolist()} == {tuple(s) for s in sols.tolist()})
                wrong = int(len({tuple(s) for s in bk['sol'].tolist()} - {tuple(s) for s in sols.tolist()}))
                b, lw, calls = sk.run_is(rules, p, pol, cfg['K_is'], rng, ep)
                ok = np.isfinite(lw)
                if ok.any():
                    ww = np.exp(lw[ok] - lw[ok].max())
                    mg = np.stack([(ww[:, None] * (b[ok] == v)).sum(0) / ww.sum() for v in range(K)], -1)
                    is_err = float(np.abs(mg - exm)[emp].mean())
                else:
                    is_err = float('nan')
                q, _ = pol(p[None].astype(np.int8), rules.cand_of(p)[None], rng)
                onepass = bool((q[0].argmax(-1) == sols[0])[emp].all()) if len(sols) == 1 else False
                lp = route_logprob(rules, p, sols[0], pol, rng, ep) if len(sols) == 1 else float('nan')
                rows.append([T.opened if bk['closed'] else np.nan, T.first_found, bk['closed'], same, wrong, viol,
                             ok.mean(), is_err, calls / cfg['K_is'], onepass, lp, T.calls])
                W.append(w)
                C.append(c)
            A = np.array(rows, dtype=float)
            W, C = np.array(W), np.array(C, dtype=float)
            cl = A[:, 2] > 0
            out[an][sn] = dict(closed_within_cap=float(cl.mean()), opened_to_close_median=float(np.nanmedian(A[:, 0])) if cl.any() else None,
                               opened_to_close_mean=float(np.nanmean(A[:, 0])) if cl.any() else None,
                               opened_censored_mean=float(np.where(cl, A[:, 0], cfg['cap']).mean()),
                               first_solution_median=float(np.median(A[:, 1][A[:, 1] > 0])) if (A[:, 1] > 0).any() else None,
                               solution_found=float((A[:, 1] > 0).mean()), same_set_as_exact=float(A[:, 3].mean()),
                               wrong_solutions=int(A[:, 4].sum()), violations=int(A[:, 5].sum()),
                               is_success_per_rollout=float(A[:, 6].mean()), is_any_success=float((A[:, 6] > 0).mean()),
                               is_err=float(np.nanmean(A[:, 7])) if np.isfinite(A[:, 7]).any() else None,
                               is_calls_per_rollout=float(A[:, 8].mean()), onepass_solved=float(A[:, 9].mean()),
                               neglog_p_solution=float(np.nanmean(-A[:, 10])) if np.isfinite(A[:, 10]).any() else None,
                               frac_p_solution_below_1pct=float(np.nanmean(A[:, 10] < np.log(0.01))) if np.isfinite(A[:, 10]).any() else None,
                               width_at_marks=W.mean(0).tolist(), closed_at_marks=C.mean(0).tolist(),
                               per_puzzle_opened=np.where(cl, A[:, 0], -1).tolist())
        o = out[an]
        log('scored %-18s rho=%.2f eps=%.2f | hard: closed %.2f, C %.0f, median %s, first solution %s, 1-pass %.2f, '
            'IS success/rollout %.3f | mixed: C %.0f | multi: closed %.2f IS err %s | violations %d wrong %d | %.0fs'
            % (an, rh, ep, o['hard']['closed_within_cap'], o['hard']['opened_censored_mean'], o['hard']['opened_to_close_median'], o['hard']['first_solution_median'],
               o['hard']['onepass_solved'], o['hard']['is_success_per_rollout'], o['mixed']['opened_censored_mean'],
               o['multi']['closed_within_cap'], o['multi']['is_err'],
               sum(o[s]['violations'] for s in sets), sum(o[s]['wrong_solutions'] for s in sets), time.time() - t0))
        res['arms'] = out
        json.dump(res, open(HERE + 'out/%s.json' % name, 'w'))
    log('done')
