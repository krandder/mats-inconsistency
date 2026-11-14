"""Scores the PBO addendum against PREREG-PBO.md.
python analyze_pbo.py <dir with e1_256_64_*.json> [<dir with e2_256_64_*.json>]"""
import glob, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
pre = json.load(open(os.path.join(HERE, '..', 'PREREG-PBO-predictions.json')))
FLOOR = 2e-6
d1 = sys.argv[1]
d2 = sys.argv[2] if len(sys.argv) > 2 else None
tab = {}
for fn in sorted(glob.glob(os.path.join(d1, 'e1_256_64_*.json'))):
    r = json.load(open(fn))
    for key, rec in r['runs'].items():
        rep, cut, fam = key.split('/')
        tab.setdefault(int(cut), {}).setdefault(fam, {})[r['seed']] = rec['nmse'][-1]


def kstar(row, prefix, ks):
    for k in ks:
        v = row.get(f'{prefix}{k}')
        if v is not None and v <= 1e-5:
            return k
    return None


def gmean(xs):
    return float(np.exp(np.mean(np.log(xs))))


for cut in sorted(tab):
    T = tab[cut]
    seeds = sorted({s for f in T.values() for s in f})
    print(f"\n=== cut {cut}: nMSE of E[a_64] (per net {seeds}; mean) ===")
    order = ['EMP'] + [f'PB{k}' for k in (16, 32, 48, 64)] + [f'PBW{k}' for k in (16, 32)] + \
            [f'PBO{k}' for k in (8, 16, 24, 32, 48)]
    for fam in order:
        if fam in T:
            v = [T[fam].get(s, np.nan) for s in seeds]
            pr = [pre['pred'].get(f'{cut}/{s}', {}).get(fam) for s in seeds]
            ps = ' (pred ' + ' '.join(f'{p:.1e}' for p in pr) + ')' if all(p is not None for p in pr) else ''
            print(f"  {fam:6s} " + ' '.join(f'{x:9.2e}' for x in v) + f"   mean {np.nanmean(v):9.2e}{ps}")
    # T-A
    kO = {s: kstar({f: T[f][s] for f in T if s in T[f]}, 'PBO', (8, 16, 24, 32, 48)) for s in seeds}
    kP = {s: kstar({f: T[f][s] for f in T if s in T[f]}, 'PB', (16, 32, 48, 64)) for s in seeds}
    nO = sum(1 for s in seeds if kO[s] is not None and kO[s] <= 32)
    nLess = sum(1 for s in seeds if kO[s] is not None and (kP[s] is None or kO[s] < kP[s]))
    nBad = sum(1 for s in seeds if kO[s] is None or kO[s] >= 48)
    nGE = sum(1 for s in seeds if kO[s] is None or (kP[s] is not None and kO[s] >= kP[s]))
    TA = 'SUPPORTED' if (nO >= 3 and nLess >= 3) else ('REFUTED' if (nBad >= 3 or nGE >= 3) else 'MIXED')
    print(f"  T-A: k*_PBO {[kO[s] for s in seeds]}, k*_PB {[kP[s] for s in seeds]} -> {TA}")
    for name, other, thr in (('T-B', 'PB', 1.5), ('T-C', 'PBW', 1.3)):
        gms = {}
        for k in (16, 32):
            rs = [max(T[f'{other}{k}'][s], FLOOR) / max(T[f'PBO{k}'][s], FLOOR) for s in seeds
                  if s in T.get(f'{other}{k}', {}) and s in T.get(f'PBO{k}', {})]
            gms[k] = gmean(rs) if rs else float('nan')
        verdict = 'SUPPORTED' if all(g >= thr for g in gms.values()) else (
            'REFUTED' if any(g <= 1 for g in gms.values()) else 'MIXED')
        print(f"  {name}: geometric-mean ratio {other}/PBO at k = 16: {gms[16]:.2f}, k = 32: {gms[32]:.2f} -> {verdict}")
    # T-D
    res = []
    for s in seeds:
        for k in (8, 16, 24, 32, 48):
            m = T.get(f'PBO{k}', {}).get(s)
            p = pre['pred'].get(f'{cut}/{s}', {}).get(f'PBO{k}')
            if m is not None and p is not None and m > FLOOR:
                res.append(math.log10(m / p))
    if res:
        frac = np.mean(np.abs(res) <= 0.5)
        TD = 'SUPPORTED' if frac >= 0.75 else ('REFUTED' if frac < 0.5 else 'MIXED')
        print(f"  T-D: {len(res)} PBO points above the floor; log10(measured/predicted) mean {np.mean(res):+.2f}, "
              f"range {min(res):+.2f}..{max(res):+.2f}; within 0.5: {frac:.0%} -> {TD}")

if d2:
    print("\n=== composition (re-projection at every layer from 32), nMSE at L = 64 ===")
    for fn in sorted(glob.glob(os.path.join(d2, 'e2_256_64_*.json'))):
        r = json.load(open(fn))
        C = {}
        for key, rec in r['runs'].items():
            lnc, tag, fam = key.split('/')
            C[(int(lnc), fam)] = rec['nmse'][-1]
        fams = sorted({f for (_, f) in C})
        print(f"  net {r['seed']}: " + '; '.join(
            f"{f} C(2^20) {C.get((20, f), float('nan')):.2e} C(2^21) {C.get((21, f), float('nan')):.2e} "
            f"B {2 * C.get((21, f), float('nan')) - C.get((20, f), float('nan')):.2e}" for f in fams))
