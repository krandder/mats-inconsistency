"""Scores the PREREG-SF predictions from m3_subst.json, m4_bottleneck.json, m5_sink.json."""
import json, os
import numpy as np

W = '/home/agent/workspaces/mats/work/scale-free-theory/'
LEARNED = [f'w32_h2_L2_s{s}' for s in range(5)] + [f'w32_h2_L2_mix4_s{s}' for s in (0, 1, 2, 4)]
STUCK = 'w32_h2_L2_mix4_s3'
out = {}

if os.path.exists(W + 'm3_subst.json'):
    m3 = json.load(open(W + 'm3_subst.json'))
    nets = [n for n in LEARNED if n in m3]
    ladder = rb = ob = suff = 0
    rows = []
    for n in nets:
        K = {l: {v: m3[n][f'layer{l}'][v]['kl_net_variant'] for v in ('fit', 'aligned', 'misaligned')} for l in (1, 2)}
        KB = {l: {v: m3[n][f'layer{l}'][v]['kl_bayes_variant'] for v in ('fit', 'aligned', 'misaligned')} for l in (1, 2)}
        lad = all(K[l]['fit'] < K[l]['aligned'] < K[l]['misaligned'] for l in (1, 2))
        rho = {l: K[l]['misaligned'] - K[l]['aligned'] for l in (1, 2)}
        om = {l: K[l]['aligned'] - K[l]['fit'] for l in (1, 2)}
        sf = all(K[l]['fit'] <= 0.25 * K[l]['aligned'] for l in (1, 2))
        ladder += lad; rb += rho[1] > rho[2]; ob += om[2] > om[1]; suff += sf
        rows.append((n, K, KB, rho, om, m3[n]['kl_bayes_net']))
    out['M3'] = {'n': len(nets), 'P-M3a_ladder_count': ladder, 'P-M3b_rho1>rho2_count': rb, 'P-M3b_omega2>omega1_count': ob,
                 'P-M3c_fit<=0.25aligned_count': suff}
    print('M3', out['M3'])
    for n, K, KB, rho, om, kbn in rows:
        print(f"  {n:22s} L1 fit {K[1]['fit']:.4f} al {K[1]['aligned']:.4f} mis {K[1]['misaligned']:.4f} | "
              f"L2 fit {K[2]['fit']:.4f} al {K[2]['aligned']:.4f} mis {K[2]['misaligned']:.4f} | "
              f"rho {rho[1]:.3f}/{rho[2]:.3f} omega {om[1]:.3f}/{om[2]:.3f} | KL(B||net) {kbn:.4f} "
              f"KL(B||fit) {KB[1]['fit']:.4f}/{KB[2]['fit']:.4f}")
    if STUCK in m3:
        s = m3[STUCK]
        print('  stuck', {f'L{l}': {v: round(s[f"layer{l}"][v]['kl_net_variant'], 4) for v in ('fit', 'aligned', 'misaligned')} for l in (1, 2)})

if os.path.exists(W + 'm4_bottleneck.json'):
    m4 = json.load(open(W + 'm4_bottleneck.json'))
    for extra in ('m4_bottleneck_mix4.json', 'm4_bottleneck_mix4_s2.json', 'm4_bottleneck_mix4_s4.json'):
        if os.path.exists(W + extra):
            m4.update({k: v for k, v in json.load(open(W + extra)).items() if 'k8' in v})
    nets = [n for n in LEARNED if n in m4]
    ks = [0, 1, 2, 3, 4, 6, 8]
    cnt_a = cnt_b = 0
    for n in nets:
        B = {k: m4[n][f'k{k}']['kl_net_bott'] for k in ks if f'k{k}' in m4[n]}
        kmin = next((k for k in ks if k in B and B[k] <= 0.01), None)
        cnt_a += (kmin is not None and kmin <= 4); cnt_b += B.get(0, 0) >= 0.05
        print(f"  {n:22s} " + ' '.join(f"k{k}:{B[k]:.4f}" for k in ks if k in B) + f" | kmin {kmin} | KL(B||net) {m4[n]['kl_bayes_net']:.4f} "
              + ' '.join(f"B{k}:{m4[n][f'k{k}']['kl_bayes_bott']:.4f}" for k in (2, 4) if f'k{k}' in m4[n]))
    out['M4'] = {'n': len(nets), 'P-M4a_kmin<=4_count': cnt_a, 'P-M4b_B0>=0.05_count': cnt_b}
    print('M4', out['M4'])

if os.path.exists(W + 'm5_sink.json'):
    m5 = json.load(open(W + 'm5_sink.json'))
    def best(rec):
        hs = sorted(rec.values(), key=lambda h: -h['r_bayes_var'])
        return hs[0]
    cl = [n for n in LEARNED if n in m5['clean']]
    ca = sum(best(m5['clean'][n])['r_bayes_var'] >= 0.5 for n in cl)
    no = list(m5['noisy'])
    cb = sum(best(m5['noisy'][n])['r_bayes_var'] >= 0.5 for n in no)
    out['M5'] = {'clean_n': len(cl), 'P-M5a_count': ca, 'noisy_n': len(no), 'P-M5b_count': cb}
    print('M5', out['M5'])
    for grp, names in [('clean', list(m5['clean'])), ('noisy', no)]:
        for n in names:
            b = best(m5[grp][n])
            print(f"  {grp} {n:22s} best r(log a_self, log VarBayes) {b['r_bayes_var']:.3f} slope {b['slope_bayes_var']:.3f} "
                  f"r(net var) {b['r_net_var']:.3f} mean a_self {b['mean_aself']:.3f}")
json.dump(out, open(W + 'sf_scores.json', 'w'), indent=1)
