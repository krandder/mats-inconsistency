"""summ.py -- compact digest of the registered outputs for NOTE.md (-> out/summ.out)."""
import glob, json
import numpy as np
HERE = '/home/agent/workspaces/mats/work/deep-study/wf13/loop/'
L = []
def P(s=''):
    L.append(s)
# ---------------- Sudoku
for fn in sorted(glob.glob(HERE + 'out/s_s[12]*.json')):
    r = json.load(open(fn))
    if 'arms' not in r or 'hand_mrv' not in r['arms'] or 'prefix' in fn or 'partial' in fn:
        continue
    A, R = r['arms'], r['cfg']['R']
    P('SUDOKU %s (B_train %d, order %s): loop log closed/opened %s; one-shot states %d' % (
        fn.split('/')[-1], r['cfg']['B_train'], r['cfg']['order'],
        ['%.2f/%.0f' % (x['closed'], x['opened']) for x in r['loop']], r['oneshot']['states']))
    for sn in ('hard', 'mixed'):
        P('  %s C: rounds %s | oneshot %.0f | hand %.0f | random %s | final_mrvorder %s' % (
            sn, [round(A['round%d' % k][sn]['opened_censored_mean']) for k in range(R + 1)], A['oneshot'][sn]['opened_censored_mean'],
            A['hand_mrv'][sn]['opened_censored_mean'], round(A['bad_random'][sn]['opened_censored_mean']) if 'bad_random' in A else 'n/a',
            round(A['final_mrvorder'][sn]['opened_censored_mean']) if 'final_mrvorder' in A else 'n/a'))
        P('  %s median: rounds %s | oneshot %s | hand %s' % (sn, [A['round%d' % k][sn]['opened_to_close_median'] for k in range(R + 1)],
                                                            A['oneshot'][sn]['opened_to_close_median'], A['hand_mrv'][sn]['opened_to_close_median']))
    P('  hard IS success/rollout: rounds %s | oneshot %.3f | hand %.3f' % (
        ['%.3f' % A['round%d' % k]['hard']['is_success_per_rollout'] for k in range(R + 1)], A['oneshot']['hard']['is_success_per_rollout'], A['hand_mrv']['hard']['is_success_per_rollout']))
    P('  hard -logP(sol): rounds %s | oneshot %s ; P(sol)<1%%: rounds %s | oneshot %s' % (
        ['%.2f' % A['round%d' % k]['hard']['neglog_p_solution'] for k in range(R + 1)], '%.2f' % A['oneshot']['hard']['neglog_p_solution'],
        ['%.2f' % A['round%d' % k]['hard']['frac_p_solution_below_1pct'] for k in range(R + 1)], '%.2f' % A['oneshot']['hard']['frac_p_solution_below_1pct']))
    P('  multi IS err: rounds %s | hand %s ; one-pass solved hard: %s' % (
        ['%.3f' % (A['round%d' % k]['multi']['is_err'] or float('nan')) for k in range(R + 1)], A['hand_mrv']['multi']['is_err'],
        [A['round%d' % k]['hard']['onepass_solved'] for k in range(R + 1)]))
    for an in A:
        if an.startswith(('bad', 'final')):
            a = A[an]
            P('  %-18s rho %.1f eps %.2f | hard C %.0f closed %.2f | mixed C %.0f closed %.2f | multi closed %.2f | IS succ %.3f | IS err multi %s | P(sol)<1%% %s'
              % (an, a['rho'], a['eps'], a['hard']['opened_censored_mean'], a['hard']['closed_within_cap'], a['mixed']['opened_censored_mean'],
                 a['mixed']['closed_within_cap'], a['multi']['closed_within_cap'], a['hard']['is_success_per_rollout'], a['multi']['is_err'],
                 a['hard']['frac_p_solution_below_1pct']))
    tot = {k: sum(A[an][s][k] for an in A for s in ('hard', 'mixed', 'multi')) for k in ('violations', 'wrong_solutions')}
    nq = sum(r['sets'][s]['n'] for s in r['sets']) * len(A)
    same = [(an, s, A[an][s]['same_set_as_exact'], A[an][s]['closed_within_cap']) for an in A for s in ('hard', 'mixed', 'multi')]
    P('  totals: violations %d, wrong solutions %d over %d (arm, puzzle) trees x %d budgets; closed trees with the exact set: %s'
      % (tot['violations'], tot['wrong_solutions'], nq, len(r['cfg']['marks']), all(abs(a - b) < 1e-9 for _, _, a, b in same)))
    P('  trees closed in cap, summed over arms: %.0f' % sum(A[an][s]['closed_within_cap'] * r['sets'][s]['n'] for an in A for s in r['sets']))
# ---------------- parity
for fn in sorted(glob.glob(HERE + 'out/p_w10[1-3]_s1.json')):
    r = json.load(open(fn))
    if 'baselines' not in r:
        continue
    A, mk, km, R = r['arms'], r['cfg']['marks'], r['cfg']['kmarks'], r['cfg']['R']
    i = {m: mk.index(m) for m in mk}
    b = r['baselines']
    P('PARITY world %d: one-pass M %.4f, prior %.4f, T vs truth %.4f' % (r['world'], b['onepass_M_abs'], b['prior_abs'], b['T_vs_truth_abs']))
    for key, lab in (('w_aud', 'W_aud@256'), ('w_cert', 'W_cert@256'), ('found', 'found@256'), ('err_aud', 'E_aud@256'), ('err_pt', 'E_found@256')):
        P('  %-11s rounds %s | oneshot %.4f | hand_spec %.4f | hand_rho %.4f | random %.4f | flip %.4f' % (
            lab, ['%.4f' % A['round%d' % k][key][i[256]] for k in range(R + 1)], A['oneshot'][key][i[256]], A['hand_spec'][key][i[256]],
            A['hand_rho'][key][i[256]], A['bad_random'][key][i[256]], A['bad_flip'][key][i[256]]))
    for K in (16, 64, 256):
        P('  E_IS@K=%-3d  rounds %s | oneshot %.4f | hand_rho %.4f | flip %.4f | avoid_noguard %.4f | avoid_floor %.4f | conforder n/a' % (
            K, ['%.4f' % A['round%d' % k]['is_err'][km.index(K)] for k in range(R + 1)], A['oneshot']['is_err'][km.index(K)],
            A['hand_rho']['is_err'][km.index(K)], A['bad_flip']['is_err'][km.index(K)], A['bad_avoid_noguard']['is_err'][km.index(K)],
            A['bad_avoid_floor']['is_err'][km.index(K)]))
    P('  one-pass err rounds %s | oneshot %.4f ; KL(T,pol) rounds %s | oneshot %.2f | eps0 %.2f ; under100x rounds %s | oneshot %.3f | eps0 %.3f' % (
        ['%.3f' % A['round%d' % k]['onepass_abs'] for k in range(R + 1)], A['oneshot']['onepass_abs'],
        ['%.2f' % A['round%d' % k]['kl_T_policy'] for k in range(R + 1)], A['oneshot']['kl_T_policy'], A['final_eps0']['kl_T_policy'],
        ['%.3f' % A['round%d' % k]['frac_100x_under'] for k in range(R + 1)], A['oneshot']['frac_100x_under'], A['final_eps0']['frac_100x_under']))
    P('  IS eps0 final: %s ; calls per rollout %.1f' % (['%.4f' % v for v in A['final_eps0']['is_err']], A['round%d' % R]['is_calls'][-1] / km[-1]))
    for an in ('round%d' % R, 'hand_spec', 'bad_random', 'bad_flip', 'bad_avoid_noguard', 'bad_avoid_floor', 'bad_avoid_share', 'final_rho0', 'final_rho1', 'final_conforder'):
        a = A[an]
        P('  %-18s W_cert %s | W_aud %s | E_aud %s' % (an, ' '.join('%.3f' % a['w_cert'][i[m]] for m in (64, 256, 1024, 2048)),
                                                      ' '.join('%.3f' % a['w_aud'][i[m]] for m in (64, 256, 1024, 2048)),
                                                      ' '.join('%.4f' % a['err_aud'][i[m]] for m in (64, 256, 1024, 2048))))
    nq = r['cfg']['n_test'] * 20 * len(mk)
    P('  violations: cert %d, Z %d; audit misses %d of %d queries (%d arms)' % (
        sum(a['viol_cert_total'] for a in A.values()), sum(a['viol_Z_total'] for a in A.values()),
        sum(a['miss_aud_total'] for a in A.values()), nq * len(A), len(A)))
    sa = r['same_answers']
    P('  same answers: cert intersect %.4f/%.4f, audit intersect %.4f/%.4f, audit point spread mean %.4f (256) %.5f (2048), found-point spread %.4f (256) %.5f (2048)' % (
        sa['256']['cert_intervals_intersect'], sa['2048']['cert_intervals_intersect'], sa['256']['audit_intervals_intersect'],
        sa['2048']['audit_intervals_intersect'], sa['256']['mean_spread_audit_point'], sa['2048']['mean_spread_audit_point'],
        sa['256']['mean_spread_found_point'], sa['2048']['mean_spread_found_point']))
open(HERE + 'out/summ.out', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
