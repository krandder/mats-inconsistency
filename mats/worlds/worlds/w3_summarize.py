"""Summarise the W3 C2 pilot: KL(Bayes||model) per l per arm (seed mean +- se), information captured, reach,
looped test-time iterations, and the pre-registered readings R1-R4 (PREREG.md, W3)."""
import glob, json, math, os, re, sys
import numpy as np

OUT = sys.argv[1]
THR = 0.05
LS = list(range(9))


def load(path):
    s = json.load(open(path))
    kl = {}
    for c in s['cells']:
        kl[(c['policy'], c['N'])] = c['splits']['all']['kl_model'][0]
        for sp in ('exact_chain', 'noisy_chain'):
            if sp in c['splits']:
                kl[(c['policy'] + ':' + sp, c['N'])] = c['splits'][sp]['kl_model'][0]
    return kl


def info():
    out = {}
    for f in glob.glob(os.path.join(OUT, 'evalset', 'cell_*.npz')):
        m = re.match(r'cell_N(\d+)_(\w+)\.npz', os.path.basename(f))
        d = np.load(f)
        out[(m.group(2), int(m.group(1)))] = float((d['nll_prior'] - d['nll_bayes']).mean())
        out[(m.group(2) + ':nearest', int(m.group(1)))] = float((d['nll_nearest_only'] - d['nll_bayes']).mean())
    return out


runs = {}
for d in sorted(glob.glob(os.path.join(OUT, '*_s*'))):
    if not os.path.isdir(d):
        continue
    arm, seed = re.match(r'(.+)_s(\d+)$', os.path.basename(d)).groups()
    for e in glob.glob(os.path.join(d, 'eval*', 'summary.json')):
        tag = os.path.basename(os.path.dirname(e))
        it = int(tag[7:]) if tag.startswith('eval_it') else None
        runs.setdefault((arm, it), {})[int(seed)] = load(e)

INFO = info()


def stat(arm, it, pol, l):
    v = [r[(pol, l)] for r in runs.get((arm, it), {}).values() if (pol, l) in r]
    if not v:
        return None, None, 0
    return float(np.mean(v)), (float(np.std(v, ddof=1) / math.sqrt(len(v))) if len(v) > 1 else 0.0), len(v)


def reach(arm, it, pol='std'):
    r = -1
    for l in LS:
        m, _, n = stat(arm, it, pol, l)
        if m is None or m >= THR:
            break
        r = l
    return r


def rounds(arm, it):
    if arm.startswith('U'):
        return int(arm[1:])
    return 2 * it


print('# W3 C2 pilot summary\n')
print('KL(Bayes||prior) available per l (std): ' + ' '.join(f'{l}:{INFO.get(("std", l), float("nan")):.3f}' for l in LS))
print('KL(Bayes||nearest-only) per l (std):  ' + ' '.join(f'{l}:{INFO.get(("std:nearest", l), float("nan")):.3f}' for l in LS))
main = [('U2', None), ('U4', None), ('U6', None)] + [(a, int(a[4:])) for a in ('LP2x2', 'LP2x3')]
for pol in ('std', 'far', 'std:exact_chain', 'std:noisy_chain'):
    print(f'\n## KL(Bayes||model), policy {pol}: seed mean (se), n seeds\n')
    print('| arm (rounds) | ' + ' | '.join(f'l={l}' for l in LS) + ' | reach |')
    print('|---|' + '---|' * (len(LS) + 1))
    for arm, it in main + [(a, k) for a in ('LP2x2', 'LP2x3') for k in range(1, 7) if k != int(a[4:])]:
        if (arm, it) not in runs:
            continue
        cells = []
        for l in LS:
            m, se, n = stat(arm, it, pol, l)
            cells.append('' if m is None else f'{m:.3f} ({se:.3f})')
        name = arm + (f' it{it}' if it else '')
        print(f'| {name} ({rounds(arm, it)}) | ' + ' | '.join(cells) + f' | {reach(arm, it, pol)} |')

print('\n## Fraction of available information captured, 1 - KL/KL(Bayes||prior) (std, seed mean)\n')
print('| arm | ' + ' | '.join(f'l={l}' for l in LS) + ' |')
print('|---|' + '---|' * len(LS))
for arm, it in main:
    if (arm, it) not in runs:
        continue
    print(f'| {arm} | ' + ' | '.join(f'{1 - stat(arm, it, "std", l)[0] / INFO[("std", l)]:.2f}' for l in LS) + ' |')

print('\n## Pre-registered readings\n')
m, se, n = stat('U2', None, 'std', 6)
if m is not None:
    print(f'- R1: U2 KL at l=6 = {m:.3f} (se {se:.3f}, {n} seeds) -> ' + ('C2 REFUTED for W3' if m < THR else 'not refuted'))
rr = {a: reach(a, None) for a in ('U2', 'U4', 'U6')}
print(f'- R2: reach U2={rr["U2"]} U4={rr["U4"]} U6={rr["U6"]} (l in 0..8; -1 = fails at l=0)')
for T, U in ((2, 'U4'), (3, 'U6')):
    a = f'LP2x{T}'
    diffs = []
    for l in range(5):
        m1, s1, _ = stat(a, T, 'std', l); m2, s2, _ = stat(U, None, 'std', l)
        if m1 is not None and m2 is not None:
            diffs.append((l, m1 - m2, 2 * math.sqrt(s1 ** 2 + s2 ** 2)))
    if diffs:
        ok = all(abs(d) <= 0.02 or abs(d) <= s for _, d, s in diffs)
        print(f'- R3: {a} - {U} on l<=4: ' + ' '.join(f'{l}:{d:+.3f}' for l, d, _ in diffs) + f' -> {"match" if ok else "differ"}')
if ('LP2x3', 3) in runs:
    seeds = runs[('LP2x3', 3)].keys()
    lines = []
    allbetter = True
    for k in (4, 5, 6):
        if ('LP2x3', k) not in runs:
            continue
        for l in (5, 6, 7, 8):
            per = [runs[('LP2x3', k)][s][('std', l)] - runs[('LP2x3', 3)][s][('std', l)] for s in seeds
                   if s in runs[('LP2x3', k)]]
            allbetter &= all(p < 0 for p in per)
            lines.append(f'it{k} l{l}: {np.mean(per):+.3f}')
    print('- R4: LP2x3 extra iterations minus 3 iterations (KL, std): ' + '; '.join(lines) +
          f' -> {"every seed better at every l>=5" if allbetter else "not uniformly better"}')
m, _, _ = stat('U6', None, 'std', 2)
ok = all((stat('U6', None, 'std', l)[0] or 1) < THR for l in range(3))
print(f'- Training validity: U6 KL < 0.05 on l<=2: {ok}')
