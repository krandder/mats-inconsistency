import json, sys
r = json.load(open(f'out/locate_{sys.argv[1]}.json'))
S = r['sites']; P = r['probes']
print('probe R2 trained (U, S) ; * positive, ** strong')
for ty, tg in r['targets'].items():
    for t in tg:
        cells = []
        for s in S:
            p = P[f'{ty}|{t}|{s}']
            mk = '**' if p['strong'] else ('*' if p['positive'] else '')
            cells.append(f"{p['R2']:.2f}({p['R2_U']:.2f}){mk}")
        print(f'{ty:5s} {t:8s}', ' '.join(f'{s}:{c}' for s, c in zip(S, cells)))
for net in ('T', 'U'):
    a = r['attn'][net]
    print('--- attention', net, '(per layer: heads 0..h-1, mean)')
    for k in ('A1_r', 'A_src', 'A_src_unk', 'A_ovl', 'P_src', 'P_ovl'):
        print(k, ' | '.join(' '.join(f'{x:.2f}' for x in row) for row in a[k]))
