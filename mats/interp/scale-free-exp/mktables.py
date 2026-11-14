"""Markdown tables for the report, all from one snapshot of run directories (review item 2).
usage: python mktables.py DIR [DIR ...] > results/tables.md ; also writes results/snapshot.json (every number used)."""
import glob, json, os, sys
from collections import defaultdict
import numpy as np

DIRS = sys.argv[1:]
PATHS = {}
for D in DIRS:
    for p in sorted(glob.glob(os.path.join(D, '*'))):
        if os.path.isdir(p) and os.path.exists(os.path.join(p, 'meta.json')):
            PATHS[os.path.basename(p)] = p
R = {}
for n, p in PATHS.items():
    for f in ('eval_test_model.json', 'eval_test.json'):
        fp = os.path.join(p, f)
        if os.path.exists(fp):
            ev = json.load(open(fp))
            if 'skipped' in ev:
                continue
            R[n] = dict(ev=ev, meta=json.load(open(os.path.join(p, 'meta.json'))), path=p)
arms = defaultdict(list)
for n, r in R.items():
    arms[n.rsplit('_s', 1)[0]].append((int(n.rsplit('_s', 1)[1]), r))
snap = {}


def kl(arm, key='kl'):
    rs = sorted(arms.get(arm, []))
    v = [r['ev'][key] for _, r in rs]
    snap[arm] = dict(seeds=[s for s, _ in rs], vals=v)
    return v


def fmt(v, nd=4, rng=True):
    if not v:
        return '-'
    m = float(np.mean(v))
    s = f'{m:.{nd}f}'
    if rng and len(v) > 1:
        s += f' [{min(v):.{nd}f}, {max(v):.{nd}f}]'
    return s + (f' (n={len(v)})' if len(v) < 2 else '')


blind = json.load(open('results/blind_ref.json'))
out = []
P = out.append
P('### W0 test KL (mean [min, max] over seeds)\n')
cols = ['tree1', 'tree2', 'tree4', 'tree8', 'tree16', 'tree32', 'set', 'setnc', 'tok']
P('| N | ' + ' | '.join(cols) + ' | two-pass 3+2 / 3+4 / 3+8 | one-pass h96 m 7 / 11 | history-blind Bayes |')
P('|' + '---|' * (len(cols) + 4))
for N in (3, 6, 12, 24, 64, 256):
    row = [fmt(kl(f'e1w0_{c}_N{N}'), 4) for c in cols]
    tp = ' / '.join(fmt(kl(f'a3_tree2m{m}_N{N}'), 4, False) for m in (2, 4, 8))
    mt = ' / '.join(fmt(kl(f'a6_tree{m}h96_N{N}'), 4, False) for m in (7, 11))
    P(f'| {N} | ' + ' | '.join(row) + f' | {tp} | {mt} | {blind[f"indep_N{N}"]["blind_gap"]:.3f} |')
P('\n### W0 parameters')
for a in ('e1w0_tree4_N12', 'e1w0_tree32_N12', 'e1w0_set_N12', 'e1w0_tok_N12', 'a3_tree2m8_N24', 'a6_tree11h96_N24'):
    rs = arms.get(a)
    if rs:
        P(f'- {a}: {rs[0][1]["meta"]["params"]} parameters')
P('\n### W0 steps to val gap <= 0.05 (per seed; None = not reached)\n')
for N in (3, 6, 12, 24):
    for c in ('tree16', 'set', 'tok'):
        rs = sorted(arms.get(f'e1w0_{c}_N{N}', []))
        st = []
        for _, r in rs:
            hit = [s for s, g, *_ in r['meta']['curve'] if g <= 0.05]
            st.append(hit[0] if hit else None)
        P(f'- N={N} {c}: {st}')
P('\n### W1 test KL\n')
cols = ['tree32', 'hiersum_m4', 'hiersum_m16', 'hierattn_m4', 'hierattn_m16', 'setg', 'tokg']
P('| N | coupling-blind | ' + ' | '.join(cols) + ' |')
P('|' + '---|' * (len(cols) + 2))
for N in (4, 8, 16, 24):
    ib = None
    for c in cols:
        rs = arms.get(f'e1w1_{c}_N{N}')
        if rs:
            ib = rs[0][1]['ev'].get('indep_gap'); break
    P(f'| {N} | {ib:.3f} | ' + ' | '.join(fmt(kl(f'e1w1_{c}_N{N}'), 3, True) for c in cols) + ' |')
P('\n### W2 test KL and recovered share\n')
cols = ['tree8', 'tree32', 'tree128', 'set', 'tok']
P('| N | coupling-blind | ' + ' | '.join(cols) + ' |')
P('|' + '---|' * (len(cols) + 2))
for N in (4, 6, 8):
    ib = arms[f'e1w2_set_N{N}'][0][1]['ev']['indep_gap']
    cells = []
    for c in cols:
        v = kl(f'e1w2_{c}_N{N}')
        cells.append(f'{np.mean(v):.3f} ({min(1 - x / ib for x in v):.0%}-{max(1 - x / ib for x in v):.0%})' if v else '-')
    P(f'| {N} | {ib:.3f} | ' + ' | '.join(cells) + ' |')
P('\n### E3 test KL\n')
for N in (3, 12):
    P(f'- W0 N={N}: ' + '; '.join(f'{k} {fmt(kl(f"e3w0_{k}_N{N}"), 4)}' for k in ('A', 'D', 'C', 'Bp', 'B')))
for N in (8, 16):
    P(f'- W1 N={N}: ' + '; '.join(f'{k} {fmt(kl(f"e3w1_{k}_N{N}"), 4)}' for k in ('A4', 'G4', 'G4u')))
P('\n### E2 cut cost: KL(cut) - KL(control) per seed; control KL\n')
for N, w in ((3, 'w0'), (12, 'w0'), (24, 'w0'), (8, 'w1')):
    for s in (0, 1):
        pre = 'e2w1r' if w == 'w1' else f'e2{w}'                 # W1 reruns (base nets were on sfx-b3)
        ctl = R.get(f'{pre}_ctl_N{N}_s{s}')
        if not ctl:
            continue
        cuts = [('xsrc', (0, 1, 2, 4)), ('toq', (1, 2, 4, 8)), ('labout', (0, 1, 2, 4))] if w == 'w0' else \
            [('xsrc', (0, 2)), ('xgrp', (0, 2))]
        row = []
        for c, ks in cuts:
            for k in ks:
                r = R.get(f'{pre}_{c}{k}_N{N}_s{s}')
                row.append(f'{c}{k} {r["ev"]["kl"] - ctl["ev"]["kl"]:+.4f}' if r else f'{c}{k} -')
        base = R.get(f'e1{w}_tok{"g" if w == "w1" else ""}_N{N}_s{s}')
        snap[f'e2{w}_N{N}_s{s}'] = dict(ctl=ctl['ev']['kl'], row=row)
        P(f'- {w} N={N} s{s}: base {base["ev"]["kl"] if base else float("nan"):.4f}, control {ctl["ev"]["kl"]:.4f}; ' + ', '.join(row))
P('\n### M4 (base frozen, cut on every edge out of a source): KL to Bayes; KL(net || bottlenecked)\n')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import klpair
for N in (3, 12, 24):
    for s in (0, 1):
        base = R.get(f'e1w0_tok_N{N}_s{s}')
        if not base:
            continue
        cells = []
        for k in (1, 2, 4, 8):
            r = R.get(f'a6_m4both{k}_N{N}_s{s}')
            if r and os.path.exists(os.path.join(r['path'], 'pred_test.npz')):
                kk = klpair.kl(klpair.load(base['path']), klpair.load(r['path']))
                cells.append(f'k={k}: {r["ev"]["kl"]:.4f} / {kk.mean():.4f}')
        P(f'- N={N} s{s} base KL {base["ev"]["kl"]:.4f}: ' + '; '.join(cells))
P('\n### E4 (World FA) fraction of the history value recovered (test), gap to Bayes\n')
for k in ('flat', 'flat1', 'told', 'toldu', 'learned'):
    v = [(s, r['ev']['frac'], r['ev']['gap']) for s, r in sorted(arms.get(f'e4r_{k}', []))]
    snap[f'e4r_{k}'] = v
    P(f'- {k}: ' + ', '.join(f's{s} {f:+.3f} (gap {g:.4f})' for s, f, g in v))
print('\n'.join(out))
json.dump(snap, open('results/snapshot.json', 'w'), indent=1, default=float)
