"""pscore.py -- parity world: SCORING ONLY. Runs every policy on the held-out test instances and compares the books,
the audit, the importance form and the one-pass beliefs with the exact answers of the same score (pexact, enumeration).
The searches run first; the exact numbers are used only to score what they returned."""
import json
import time

import numpy as np

import pdata, pw, ppol, ploop, pexact

HERE = pdata.HERE
SCAL = ['w_cert', 'w_aud', 'unfound_ub', 'viol_cert', 'viol_Z', 'miss_aud', 'err_pt', 'err_aud', 'found', 'calls', 'n_open']


def score_all(ws, seed, cfg, tab, nets, res, name, log):
    n = tab['loga'].shape[0]
    x, h = pdata.test_x(ws, tab, cfg['n_test'])
    EN = pexact.Enum(tab['checks'], n)
    ta, tf = pexact.true_table(n, tab['checks'].shape[0], tab['prior'], float(tab['eps']))
    ex = []
    rs = np.random.default_rng(424242 + ws)
    for i in range(len(x)):
        lr = EN.log_R(tab['loga'], tab['logf'], x[i])
        pstar, lz = EN.marg(lr)
        ptrue, _ = EN.marg(EN.log_R(ta, tf, x[i]))
        w = np.exp(lr - lr.max())
        idx = rs.choice(len(w), size=cfg['n_tsamp'], p=w / w.sum())
        ex.append(dict(pstar=pstar, lz=lz, ptrue=ptrue, H=EN.bits[idx].astype(np.int64), logT=lr[idx] - lz))
    insts = [pw.Inst(tab, x[i]) for i in range(len(x))]
    R, rho, eps, od = cfg['R'], cfg['rho'], cfg['eps'], cfg['order']
    fin = nets[R]
    av = list(range(5))
    arms = []                                # name, policy, rho, eps, importance form?, narrowness?
    for r in range(R + 1):
        arms.append(('round%d' % r, ppol.NetPolicy(nets[r], order=od), rho, eps, True, True))
    arms += [('oneshot', ppol.NetPolicy(nets['oneshot'], order=od), rho, eps, True, True),
             ('hand_spec', ppol.Hand(cfg['hand'], 'books'), 1.0, eps, False, False),
             ('hand_rho', ppol.Hand(cfg['hand'], 'books'), rho, eps, True, False),
             ('bad_random', ppol.Hand('random', 'uniform'), rho, eps, True, False),
             ('bad_flip', ppol.NetPolicy(fin, 'flip', order=od), rho, eps, True, False),
             ('bad_avoid_noguard', ppol.NetPolicy(fin, 'avoid', av, order=od), 0.0, 0.0, True, False),
             ('bad_avoid_floor', ppol.NetPolicy(fin, 'avoid', av, order=od), 0.0, eps, True, False),
             ('bad_avoid_share', ppol.NetPolicy(fin, 'avoid', av, order=od), rho, 0.0, False, False),
             ('final_rho0', ppol.NetPolicy(fin, order=od), 0.0, eps, False, False),
             ('final_rho1', ppol.NetPolicy(fin, order=od), 1.0, eps, False, False),
             ('final_eps0', ppol.NetPolicy(fin, order=od), rho, 0.0, True, True),
             ('final_conforder', ppol.NetPolicy(fin, order='conf'), rho, eps, True, False)]
    marks, km = cfg['marks'], cfg['kmarks']
    out, keep = {}, {}
    for an, pol, rh, ep, do_is, do_nar in arms:
        t0 = time.time()
        # importance form: a network policy fills in its own order (most confident first); the tree uses cfg['order']
        ispol = ppol.NetPolicy(pol.net, pol.mode, pol.avoid, order='conf') if isinstance(pol, ppol.NetPolicy) else pol
        M = {k: np.zeros((len(x), len(marks))) for k in SCAL}
        IS = np.zeros((len(x), len(km), 3))
        onep, nar = np.zeros((len(x), 2)), np.zeros((len(x), 2))
        pts = {mk: [] for mk in marks}
        for i, inst in enumerate(insts):
            rng = np.random.default_rng(9000 * (i + 1) + seed)
            o, T = ploop.score_tree(inst, pol, marks, rng, ex[i]['pstar'], ex[i]['lz'], rh, ep, cfg['n_draw'])
            for a, mk in enumerate(marks):
                for k in SCAL:
                    M[k][i, a] = o[mk][k]
                pts[mk].append(np.stack([o[mk]['lo'], o[mk]['hi'], o[mk]['pt'], o[mk]['alo'], o[mk]['ahi'], o[mk]['apt']]))
            if do_is:
                s = ploop.score_is(inst, ispol, km, rng, ex[i]['pstar'], ep)
                IS[i] = [[s[K]['err'], s[K]['ess'], s[K]['calls']] for K in km]
            b1 = ploop.one_pass(inst, pol)
            onep[i] = [np.abs(b1 - ex[i]['pstar']).mean(), pexact.kl_bern(ex[i]['pstar'], b1).mean()]
            if do_nar:
                lp = ploop.route_logprob(inst, ispol, ex[i]['H'], rng, ep)
                nar[i] = [np.mean(ex[i]['logT'] - lp), np.mean(lp - ex[i]['logT'] < np.log(0.01))]
        keep[an] = {mk: np.stack(v) for mk, v in pts.items()}                    # [inst, 6, n]
        out[an] = dict(rho=rh, eps=ep, marks=marks, kmarks=km,
                       **{k: M[k].mean(0).tolist() for k in SCAL}, viol_cert_total=int(M['viol_cert'].sum()),
                       viol_Z_total=int(M['viol_Z'].sum()), miss_aud_total=int(M['miss_aud'].sum()),
                       per_inst_w_cert=M['w_cert'].tolist(), per_inst_w_aud=M['w_aud'].tolist(),
                       per_inst_err_aud=M['err_aud'].tolist(), per_inst_found=M['found'].tolist(),
                       is_err=IS[:, :, 0].mean(0).tolist(), is_ess=IS[:, :, 1].mean(0).tolist(),
                       is_calls=IS[:, :, 2].mean(0).tolist(), per_inst_is_err=IS[:, :, 0].tolist(),
                       onepass_abs=float(onep[:, 0].mean()), onepass_kl=float(onep[:, 1].mean()),
                       kl_T_policy=float(nar[:, 0].mean()), frac_100x_under=float(nar[:, 1].mean()),
                       sec=time.time() - t0)
        log('scored %-18s rho=%.2f eps=%.2f | width cert/audit @%d: %.3f/%.3f @%d: %.3f/%.3f | err audit-pt @%d %.4f | IS err @%d %.4f | '
            'violations cert %d, Z %d, audit misses %d | %.0fs'
            % (an, rh, ep, marks[len(marks) // 2], out[an]['w_cert'][len(marks) // 2], out[an]['w_aud'][len(marks) // 2],
               marks[-1], out[an]['w_cert'][-1], out[an]['w_aud'][-1], marks[len(marks) // 2],
               out[an]['err_aud'][len(marks) // 2], km[-1], out[an]['is_err'][-1], out[an]['viol_cert_total'],
               out[an]['viol_Z_total'], out[an]['miss_aud_total'], time.time() - t0))
        res['arms'] = out
        json.dump(res, open(HERE + 'out/%s.json' % name, 'w'))
    # ---- do the answers stay the same across rounds? (same instance, same budget, different rollout policy)
    same = {}
    for mk in marks:
        st = np.stack([keep['round%d' % r][mk] for r in range(R + 1)])           # [rounds, inst, 6, n]
        same[mk] = dict(cert_intervals_intersect=float((st[:, :, 0].max(0) <= st[:, :, 1].min(0) + 1e-9).mean()),
                        audit_intervals_intersect=float((st[:, :, 3].max(0) <= st[:, :, 4].min(0) + 1e-9).mean()),
                        max_spread_audit_point=float((st[:, :, 5].max(0) - st[:, :, 5].min(0)).max()),
                        mean_spread_audit_point=float((st[:, :, 5].max(0) - st[:, :, 5].min(0)).mean()),
                        mean_spread_found_point=float((st[:, :, 2].max(0) - st[:, :, 2].min(0)).mean()))
    res['same_answers'] = same
    # ---- baselines on the same instances: one pass of M, the prior, T against reality
    op = np.load(HERE + 'out/onepass_w%d.npz' % ws)
    assert (op['x'][:len(x)] == x).all()
    pst = np.stack([e['pstar'] for e in ex])
    ptr = np.stack([e['ptrue'] for e in ex])
    res['baselines'] = dict(onepass_M_abs=float(np.abs(op['b'][:len(x)] - pst).mean()),
                            onepass_M_kl=float(pexact.kl_bern(pst, op['b'][:len(x)]).mean()),
                            prior_abs=float(np.abs(np.exp(tab['loga'][:, 1])[None] - pst).mean()),
                            T_vs_truth_abs=float(np.abs(pst - ptr).mean()),
                            T_vs_truth_kl=float(pexact.kl_bern(ptr, pst).mean()), M_flops=float(op['flops']))
    json.dump(res, open(HERE + 'out/%s.json' % name, 'w'))
    log('done')
