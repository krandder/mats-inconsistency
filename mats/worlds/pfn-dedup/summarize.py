"""Aggregate eval.json files into markdown tables (mean +- se over seeds) and score the prereg levels."""
import glob, json, os, sys
import numpy as np

root = sys.argv[1]
R = {}
for f in glob.glob(os.path.join(root, 'L*_k*_s*/eval.json')):
    e = json.load(open(f)); m = e['meta']
    R.setdefault((m['depth'], m['keys']), []).append(e)

SPL = ['all', 'dup_singletons', 'singletons_nodup', 'group2', 'group3', 'group4plus', 'shared_any']


def ms(v):
    v = np.array(v); return f'{v.mean():.4f}' + (f' ± {v.std(ddof=1) / np.sqrt(len(v)):.4f}' if len(v) > 1 else '')


print('# PFN dedup vs depth: NLL gap to the arm\'s exact Bayes (nats; mean ± se over seeds)\n')
print('Bayes = exact partition posterior (no keys) or GLS with the true partition (keys).\n')
for k in (0, 1):
    print(f'\n## keys = {k}\n')
    print('| depth | conv | ' + ' | '.join(SPL) + ' |')
    print('|' + '---|' * (len(SPL) + 2))
    for L in (1, 2, 4, 8):
        es = R.get((L, k), [])
        if not es: continue
        conv = sum(e['meta']['converged'] for e in es)
        cells = [ms([e[s]['gap_model'] for e in es]) if s in es[0] else '-' for s in SPL]
        print(f'| {L} | {conv}/{len(es)} | ' + ' | '.join(cells) + ' |')
    ref = next((R[(L2, k)][0] for L2 in (1, 2, 4, 8) if (L2, k) in R), None)
    if ref is None: continue
    for nm, kk in (('(a) count-all', 'gap_a'), ('(b) iid-distinct', 'gap_b')) + ((('(c) GLS-oracle', 'gap_c'),) if k == 0 else ()):
        print(f'| ref {nm} | | ' + ' | '.join(f"{ref[s][kk]:.4f}" if s in ref else '-' for s in SPL) + ' |')

print('\n## Levels (prereg): fraction of reference gap closed, mean over seeds\n')
print('| depth | keys | L1 dup (vs a) | L2 singletons gap | L3 shared (vs b) |')
print('|---|---|---|---|---|')
for (L, k), es in sorted(R.items()):
    l1 = np.mean([1 - e['dup_singletons']['gap_model'] / e['dup_singletons']['gap_a'] for e in es])
    l2 = np.mean([e['singletons_nodup']['gap_model'] for e in es])
    l3 = np.mean([1 - e['shared_any']['gap_model'] / e['shared_any']['gap_b'] for e in es])
    tag = lambda ok: ' ✓' if ok else ''
    print(f'| {L} | {k} | {l1:.2f}{tag(l1 >= 0.75)} | {l2:.4f}{tag(l2 <= 0.02)} | {l3:.2f}{tag(l3 >= 0.75)} |')

print('\n## Mechanism (seed 0)\n')
for (L, k), es in sorted(R.items()):
    e = [x for x in es if x['meta']['seed'] == 0]
    if not e or 'mech' not in e[0]: continue
    mm = e[0]['mech']
    print(f'\n### depth {L}, keys {k}\n')
    print('probe R2 per layer (0 = input): residual o-y ' + ', '.join(f'{v:.2f}' for v in mm['probe_residual']))
    print('\nprobe R2 per layer: source group size ' + ', '.join(f'{v:.2f}' for v in mm['probe_groupsize']))
    print('\nprobe R2 per layer: query token -> Bayes mean ' + ', '.join(f'{v:.2f}' for v in mm['probe_query_bayesmean']))
    print('\n| ablate ctx-ctx at layer | gap | mse→a | mse→b | mse→c | mse→d |')
    print('|---|---|---|---|---|---|')
    for ab in mm['ablation']:
        print(f"| {ab['layer']} | {ab['gap']:.4f} | {ab['mse_to_a']:.4f} | {ab['mse_to_b']:.4f} | {ab['mse_to_c']:.4f} | {ab['mse_to_d']:.4f} |")
    print('\n| layer | query attn on query-question tokens | corr(per-source attn, GLS weight) | attn by group size m | GLS weight by m |')
    print('|---|---|---|---|---|')
    for qa in mm['query_attn']:
        print(f"| {qa['layer']} | {qa['frac_on_query_question']:.2f} | {qa['corr_with_gls_weight_nodup']:.2f} | "
              f"{ {k2: round(v, 3) for k2, v in qa['mass_by_groupsize'].items()} } | { {k2: round(v, 3) for k2, v in qa['gls_by_groupsize'].items()} } |")
