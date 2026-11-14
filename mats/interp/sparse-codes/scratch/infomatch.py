"""Descriptive (critique finding 7): one-step error of sparse vs dense latents at matched latent description size.
Sparse bits = H(support, independent bound) + amplitude bits; dense bits = sum_{i<=k} 1/2 log2(1 + lambda_i / s^2).
usage: infomatch.py RUNDIR_OF_NET N SEED   (needs results/latent_bits_n{N}_seed{SEED}.json from scratch/latent_bits.py)"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
netdir, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
LB = json.load(open(os.path.join(HERE, '..', 'results', f'latent_bits_n{n}_seed{seed}.json')))
rows_out = []
for l in sorted(set(r['layer'] for r in LB)):
    p = os.path.join(netdir, f'e2_L{l}.json')
    if not os.path.exists(p):
        continue
    E2 = {r['model']: r for r in json.load(open(p))['rows']}
    dense_bits = next(r['bits'] for r in LB if r['layer'] == l and r['kind'] == 'dense')
    dense = []
    for k, b in sorted(((int(k), v) for k, v in dense_bits.items())):
        errs = [E2[m]['conf']['mean_nmse'] for m in (f'DENSE{k}', f'DENSEQ{k}') if m in E2]
        if errs:
            dense.append((k, b, min(errs)))
    G = E2['G']['conf']['mean_nmse']
    for r in LB:
        if r['layer'] != l or r['kind'] != 'sparse':
            continue
        name = f"{'SPARSE' if r['origin'] == 'mean' else 'SPARSEZ'}_F{r['F']}_k{r['k']}"
        if name not in E2:
            continue
        es = E2[name]['conf']['mean_nmse']; bs = r['total_bits']
        lo = max((d for d in dense if d[1] <= bs), key=lambda d: d[1], default=None)
        hi = min((d for d in dense if d[1] >= bs), key=lambda d: d[1], default=None)
        rows_out.append(dict(layer=l, model=name, bits=bs, H=r['H_support_indep'], amp=r['amp_bits'], err=es, G=G,
                             dense_lo=lo, dense_hi=hi))
        print(f"L{l} {name}: bits {bs:.0f} (support {r['H_support_indep']:.0f} + amp {r['amp_bits']:.0f}) err {es:.1e} | "
              f"dense below: {('k=%d %.0f bits %.1e' % lo) if lo else '-'} | dense above: {('k=%d %.0f bits %.1e' % hi) if hi else '-'} | G {G:.1e}")
json.dump(rows_out, open(os.path.join(HERE, '..', 'results', f'infomatch_n{n}_seed{seed}.json'), 'w'), indent=1)
