"""Side-by-side table of the interpretation battery across models. usage: python compare.py TAG1 TAG2 ...
Tags: results/{noisesens_TAG_own, noisesens_TAG_s0.01, lowrank_TAG, dag_TAG, fam_TAG, backward_TAG, devpred_TAG, estim_TAG}.json
"""
import sys, json, os
import numpy as np

H = os.path.dirname(os.path.abspath(__file__))


def ld(name):
    p = os.path.join(H, 'results', name + '.json')
    return json.load(open(p)) if os.path.exists(p) else None


rows = []
for tag in sys.argv[1:]:
    r = dict(tag=tag)
    ns = ld(f'noisesens_{tag}_own'); ns1 = ld(f'noisesens_{tag}_s0.01')
    if ns:
        r['sigma'] = ns['sigma_test']; r['err'] = ns['err_measured']; r['err0'] = ns['err_noiseless']
        r['m_over_s_med_own'] = ns['abs_margin_over_noise_quantiles']['0.5']
        sh = np.array(ns['neuron_share']); r['d'] = len(sh)
        srt = np.sort(sh)[::-1]; r['neurons_for_80pct_noise'] = int(np.searchsorted(np.cumsum(srt), 0.8) + 1)
        r['top_neuron_share'] = float(srt[0])
    if ns1:
        r['m_over_s_med_at_0.01'] = ns1['abs_margin_over_noise_quantiles']['0.5']; r['err_at_0.01'] = ns1['err_measured']
    lr = ld(f'lowrank_{tag}')
    if lr:
        q = {x['rank']: x['flip'] for x in lr['rows'] if x['kind'] == 'q'}; f = {x['rank']: x['flip'] for x in lr['rows'] if x['kind'] == 'f'}
        r['dirs_flip_le_1pct_q'] = min([k for k, v in q.items() if v <= 0.01], default=None)
        r['dirs_flip_le_1pct_pca'] = min([k for k, v in f.items() if v <= 0.01], default=None)
    dg = ld(f'dag_{tag}')
    if dg:
        g = [x for x in dg['greedy'] if abs(x['tau'] - 0.01) < 1e-9]
        if g: r['dag_edges_frac_flip1pct'] = g[0]['K'] / dg['n_edges_total']; r['dag_test_flip'] = g[0]['test']['flip']
        r['noise_flip'] = dg['noise_flip']
    fm = ld(f'fam_{tag}')
    if fm:
        for k in ('G', 'AC', 'CL', 'CLX'):
            if k in fm['error_pred']: r[f'fam_{k}_pred'] = fm['error_pred'][k]['p_family']
            r[f'fam_{k}_bits_t10'] = fm['steps'][k][-1]['unexplained_bits']; r[f'fam_{k}_R2_t10'] = fm['steps'][k][-1]['R2_mean']
        r['fam_p_net'] = fm['error_pred']['G']['p_net']
    bw = ld(f'backward_{tag}')
    if bw:
        I = np.array(bw['I_E_state']) / bw['H_E']; r['info_frac_t5_t9'] = [round(float(I[5]), 2), round(float(I[9]), 2)]
        Ia = np.array(bw['I_E_alg']) / bw['H_E']; r['alg_info_frac_t10'] = round(float(Ia[-1]), 2)
    dv = ld(f'devpred_{tag}')
    if dv:
        r['R2_dag'] = dv['R2_dag']; r['scen_hits'] = sum(int(v.get('hit', False)) for k, v in dv['scenarios'].items())
        r['S1_rate'] = dv['scenarios']['S1']['net']['rate']; r['S4_rate'] = dv['scenarios']['S4']['net']['rate']
    es = ld(f'estim_{tag}')
    if es and 'est' in es and es['est'].get('mc') and es['est'].get('rqmc'):
        mc = {x['N']: x['mse'] for x in es['est']['mc']}; rq = {x['N']: x['mse'] for x in es['est']['rqmc']}
        r['mc_mse_16k'] = mc.get(16384); r['rqmc_gain_16k'] = mc.get(16384) / rq.get(16384) if rq.get(16384) else None
        if es['est'].get('adf'): a = es['est']['adf'][-1]; r['adf_bias'] = a['bias']; r['adf_flops'] = a['flops']
    rows.append(r)
keys = []
for r in rows:
    for k in r:
        if k not in keys: keys.append(k)
print('| quantity | ' + ' | '.join(r['tag'] for r in rows) + ' |')
print('|---|' + '---|' * len(rows))
for k in keys[1:]:
    def fmt(v):
        if v is None: return '-'
        if isinstance(v, float): return f'{v:.4g}'
        return str(v)
    print(f'| {k} | ' + ' | '.join(fmt(r.get(k)) for r in rows) + ' |')
json.dump(rows, open(os.path.join(H, 'results', 'compare_' + '_'.join(sys.argv[1:])[:80] + '.json'), 'w'), indent=1)
