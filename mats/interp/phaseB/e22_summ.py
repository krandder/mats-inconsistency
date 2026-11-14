"""E22 analysis (PREREG-B amendment 29): does making layer-1 attention uniform within each source's labelled tokens raise the departure
in the real nets, as it does in the toy (E20.3)?
usage: python e22_summ.py list -> results/e22_nets.json ; python e22_summ.py -> results/e22.json, results/e22_tables.md
"""
import json, os, sys
import numpy as np


def make_list():
    nets = json.load(open('results/e15_nets.json')); out = []
    for r in nets:
        q = np.load(f"results/e15/{r['tag']}__best.npz")
        if r['depth'] == 2 and r['ds'] == 16 and float(q['gap']) < 0.10:
            out.append(dict(tag=r['tag'], run=r['run'], cfg=r['cfg'], heads=r['heads'], attn_only=r['attn_only']))
    json.dump(out, open('results/e22_nets.json', 'w'), indent=1); print(len(out), 'nets')


def main():
    z = np.load('results/e12_pairs.npz'); src, rg = z['src'], z['range']; rows = np.arange(len(src)); Wb = z['WB0'][rows, src]
    nets = json.load(open('results/e22_nets.json')); per = {}
    for r in nets:
        q = np.load(f"results/e22/{r['tag']}.npz"); d = dict(cfg=r['cfg'], heads=r['heads'])
        for p in (0, 1):
            dw = q[f'ws1_p{p}'][rows, src] - q[f'ws0_p{p}'][rows, src]
            for rr in ('large', 'middle'):
                s = rg == rr; d[f'{rr}_p{p}'] = float(dw[s].sum() / Wb[s].sum())
        per[r['tag']] = d
    n = len(per)
    up_l = sum(v['large_p1'] > v['large_p0'] for v in per.values()); up_m = sum(v['middle_p1'] > v['middle_p0'] for v in per.values())
    A = [k for k, v in per.items() if v['middle_p0'] < 0]; flip = sum(per[k]['middle_p1'] > 0 for k in A)
    out = dict(per_net=per, predictions=dict(
        P128=dict(n_up=int(up_l), n=n, holds=bool(up_l >= 0.8 * n)),
        P129=dict(n_up=int(up_m), n=n, n_A=len(A), n_A_flipped=int(flip), holds=bool(up_m >= 0.8 * n and len(A) > 0 and flip >= (2 / 3) * len(A)))),
        means=dict(large_p0=float(np.mean([v['large_p0'] for v in per.values()])), large_p1=float(np.mean([v['large_p1'] for v in per.values()])),
                   middle_p0=float(np.mean([v['middle_p0'] for v in per.values()])), middle_p1=float(np.mean([v['middle_p1'] for v in per.values()]))))
    by = {}
    for v in per.values():
        by.setdefault(v['cfg'], []).append(v)
    out['by_config'] = {c: {k: float(np.mean([x[k] for x in L])) for k in ('large_p0', 'large_p1', 'middle_p0', 'middle_p1')} | dict(n=len(L)) for c, L in by.items()}
    json.dump(out, open('results/e22.json', 'w'), indent=1)
    print(out['predictions']); print(out['means'])
    T = ['| config | nets | rom large: unpatched → uniform within source | rom middle: unpatched → uniform |', '|---|---|---|---|']
    for c, v in sorted(out['by_config'].items()):
        T.append(f"| {c} | {v['n']} | {v['large_p0']:+.3f} → {v['large_p1']:+.3f} | {v['middle_p0']:+.3f} → {v['middle_p1']:+.3f} |")
    open('results/e22_tables.md', 'w').write('\n'.join(T) + '\n')


if __name__ == '__main__':
    make_list() if len(sys.argv) > 1 and sys.argv[1] == 'list' else main()
