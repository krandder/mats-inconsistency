"""Learning-curve tables for the W2b arms: python w2b_curves.py OUT  (reads OUT/*/*/curve.json)."""
import glob, json, os, sys
OUT = sys.argv[1]
cols = ['val_gap', 'gap_n128', 'c4_frac_closure', 'c4_shift_frac', 'c7_dedup_D4/T4', 'c7_dedup_HR4/HI4', 'c7_dedup_D/T']
for f in sorted(glob.glob(os.path.join(OUT, '*', '*', 'curve.json'))):
    cur = json.load(open(f))
    name = os.path.relpath(os.path.dirname(f), OUT)
    print(f'## {name}\n')
    have = [c for c in cols if c in cur[0]]
    print('| step | stage | ' + ' | '.join(have) + ' |'); print('|' + '---|' * (2 + len(have)))
    for r in cur:
        print(f"| {r['step']} | {r.get('stage')} | " + ' | '.join(f'{r[c]:.3f}' for c in have) + ' |')
    print()
