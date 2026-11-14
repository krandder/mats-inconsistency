"""Table of the GPU from-scratch models (train_batched.py outputs) against both references (example; zoo seed 2).
usage: python gpu_table.py DIR -> prints markdown and writes results/gpu_table.json"""
import sys, json, glob, os
D = sys.argv[1]
rows = []
for f in sorted(glob.glob(os.path.join(D, 'w*_s*_sig*.json'))):
    if '_ck' in f: continue
    j = json.load(open(f)); ev = j['eval_refs']
    row = dict(file=os.path.basename(f), d=j['d'], seed=j['seed'], sigma=j['sigma'], L_marg=j['marginal']['L_marg'])
    for ref, e in ev.items():
        tag = 'ex' if ref.endswith('ex') else 'z2'
        row.update({f'{tag}_L_var': e['L_var'], f'{tag}_L_orig': e['L_orig'], f'{tag}_gap': e['rel_gap'], f'{tag}_A_var': e['A_var'],
                    f'{tag}_A_orig': e['A_orig'], f'{tag}_dA_se': e['dA_se'], f'{tag}_meets': e['meets'], f'{tag}_ci': e['dL_ci95']})
    rows.append(row)
rows.sort(key=lambda r: (r['d'], r['sigma'], r['seed']))
print('| width d | sigma | seed | loss (noise on) | gap vs example (0.147) | gap vs zoo seed 2 (0.166) | accuracy (example 0.9532 / seed 2 0.9486) | meets: example / seed 2 | 16-draw marginal loss |')
print('|---|---|---|---|---|---|---|---|---|')
for r in rows:
    print(f"| {r['d']} | {r['sigma']} | {r['seed']} | {r['ex_L_var']:.4f} | {100 * r['ex_gap']:+.1f}% | {100 * r['z2_gap']:+.1f}% | {r['ex_A_var']:.4f} | "
          f"{'yes' if r['ex_meets'] else 'no'} / {'yes' if r['z2_meets'] else 'no'} | {r['L_marg']:.4f} |")
json.dump(rows, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'gpu_table.json'), 'w'), indent=1)
