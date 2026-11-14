"""B2 budget tables (PREREG Amendments 3, 3b)."""
import json
import numpy as np

R = [json.load(open(f'out/budget_main_s{s}.json')) for s in range(5)]
k4 = json.load(open('out/k4floor.json'))['k4_floor_kl']


def col(f):
    return [f(r) for r in R]


kl0 = np.array(col(lambda r: r['kl']['N:none']['kl']))


def row(name, v, frac=False):
    v = np.array(v)
    cells = [f'{x:.4f}' + (f' ({x / k:.0%})' if frac else '') for x, k in zip(v, kl0)]
    tail = f'{v.mean():.4f}' + (f' ({(v / kl0).mean():.0%})' if frac else '')
    print(f'| {name} | ' + ' | '.join(cells) + f' | {tail} |')


print('### Part 1: nuisance symmetrisation (held-out KL; fraction of KL_model)\n')
print('| quantity | s0 | s1 | s2 | s3 | s4 | mean |'); print('|---|---|---|---|---|---|---|')
row('KL_model', kl0)
for k in ('N:id', 'N:rot', 'N:y', 'N:id+rot+y'):
    row(f'KL after averaging {k[2:]}', col(lambda r: r['kl'][k]['kl']), True)
for x in ('id', 'rot', 'y'):
    row(f'Shapley share removed: {x}', col(lambda r: r['shapley_kl_removed'][x]), True)
print('\nBayes invariance check (max abs change of Bayes mean/var):', max(max(r['bayes_invariance_maxabs'].values()) for r in R))

print('\n### Output-moment split (post hoc)\n')
print('| quantity | s0 | s1 | s2 | s3 | s4 | mean |'); print('|---|---|---|---|---|---|---|')
for nm in ('N:none', 'N:id+rot+y'):
    k = np.array(col(lambda r: r['kl'][nm]['kl']))
    mf = np.array(col(lambda r: r['moment_split'][nm]['kl_meanfix']))
    mv = np.array(col(lambda r: r['moment_split'][nm]['kl_mvfix']))
    row(f'{nm}: mean part', k - mf, True); row(f'{nm}: variance part', mf - mv, True); row(f'{nm}: shape part (rest)', mv, True)
print(f'\nK=4 head floor (best 4-component mixture to Bayes, held-out mean KL): {k4:.5f}')

print('\n### Part 2: layer-2 query-token weights vs Bayes beta (log-weights centred over sources)\n')
print('| head | slope a | R^2 beta | R^2 beta_lab | resid on top-beta source by coupling quintile | by spread quintile | corr(resid, outlyingness) | corr(resid, log SS) |')
print('|---|---|---|---|---|---|---|---|')
for s, r in enumerate(R):
    for h in (0, 1):
        p = r['part2'][f'H{h}_beta']; pl = r['part2'][f'H{h}_beta_lab']
        print(f"| s{s} H{h} | {p['slope']:.2f} | {p['r2']:.2f} | {pl['r2']:.2f} | "
              + ', '.join(f'{x:+.3f}' for x in p['resid_top_by_coupling_q']) + ' | '
              + ', '.join(f'{x:+.3f}' for x in p['resid_top_by_spread_q'])
              + f" | {p['corr_resid_outlyingness']:+.2f} | {p['corr_resid_logSS']:+.2f} |")

print('\n### Parts 3-4: mechanism idealisations\n')
print('| replacement | KL alone, mean [min-max] | KL with nuisances averaged | mean MSE / model | mean MSE averaged / averaged model | NLL cost vs model |')
print('|---|---|---|---|---|---|')
mse0 = np.array(col(lambda r: r['kl']['N:none']['mse_mean'])); mseN = np.array(col(lambda r: r['kl']['N:id+rot+y']['mse_mean']))
nll0 = np.array(col(lambda r: r['kl']['N:none']['nll']))
print(f"| none (model) | {kl0.mean():.4f} | {np.mean(col(lambda r: r['kl']['N:id+rot+y']['kl'])):.4f} | 1 | 1 | 0 |")
for nm in ('a_beta', 'a_beta_H0', 'a_beta_H1', 'a_beta_lab', 'b_suff', 'b_faith', 'a_beta+b_suff'):
    k = np.array(col(lambda r: r['kl'][nm]['kl'])); kn = np.array(col(lambda r: r['kl'][nm + '|N:all']['kl']))
    ms = np.array(col(lambda r: r['kl'][nm]['mse_mean'])); msn = np.array(col(lambda r: r['kl'][nm + '|N:all']['mse_mean']))
    nl = np.array(col(lambda r: r['kl'][nm]['nll']))
    print(f'| {nm} | {k.mean():.4f} [{k.min():.4f}-{k.max():.4f}] | {kn.mean():.4f} | {(ms / mse0).mean():.2f} | {(msn / mseN).mean():.2f} | {(nl - nll0).mean():+.4f} |')
print('\nMLP1 fit R^2 (suff, faith):', [(round(r['part3_fit_r2']['suff'], 2), round(r['part3_fit_r2']['faith'], 2)) for r in R])
print('Fitted a (H0, H1):', [(round(r['a_fit']['0_beta'], 2), round(r['a_fit']['1_beta'], 2)) for r in R])
