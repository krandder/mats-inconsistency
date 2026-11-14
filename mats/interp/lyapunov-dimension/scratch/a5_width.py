"""A5: width at fixed L/n = 1/4. Net-mean PC tails (n7) against the nongauss PB ladder (read-only): the conversion
factor C = nMSE(PB_k) / tail_PC(k), and the k needed for tail <= target at each width.
python a5_width.py"""
import glob, json
import numpy as np

NG = {128: '/home/agent/workspaces/mats/work/nongauss-family/runs/gpu/ngfam3/a3/e1_128_32_{s}.json',
      256: '/home/agent/workspaces/mats/work/nongauss-family/runs/gpu/ngfam2c/a1/e1_256_64_{s}.json',
      512: '/home/agent/workspaces/mats/work/nongauss-family/runs/gpu/ngfam3/a3/e1_512_128_{s}.json'}
R1_256 = {32: {'PB8': 1.1e-4, 'PB16': 4.4e-5, 'PB32': 1.6e-5}}      # nongauss summary, net means, cut 32
rows = {}
for fn in sorted(glob.glob('runs/n7_*.json')):
    r = json.load(open(fn))
    rows.setdefault((r['n'], r['L'], r['cut']), []).append(r)
print("size cut  nets  PR(state)  tail_PC at k = 8, 16, 32, 64, 128          state tail at k = 16, 32, 64")
for key in sorted(rows):
    R = rows[key]
    t = {k: np.mean([x['tail_PC'][str(k)] for x in R if str(k) in x['tail_PC']]) for k in (8, 16, 32, 64, 128)
         if all(str(k) in x['tail_PC'] for x in R)}
    st = {k: np.mean([x['tail_state'][str(k)] for x in R]) for k in (16, 32, 64) if all(str(k) in x['tail_state'] for x in R)}
    print(f"{key[0]}x{key[1]} {key[2]:3d}  {len(R)}  {np.mean([x['pr_state'] for x in R]):6.2f}   "
          + " ".join(f"{t[k]:.2e}" for k in t) + "   " + " ".join(f"{st[k]:.2e}" for k in st))
print("\nconversion factor C = mean nMSE(PB_k) / mean tail_PC(k)   (needs both)")
for key in sorted(rows):
    n, L, cut = key
    R = rows[key]; seeds = sorted(x['seed'] for x in R)
    ng = {}
    for s in seeds:
        try:
            d = json.load(open(NG[n].format(s=s)))
        except (FileNotFoundError, KeyError):
            continue
        for kk, rec in d['runs'].items():
            rep, c, fam = kk.split('/')
            if int(c) == cut and fam.startswith('PB') and fam[2:].isdigit():
                ng.setdefault(fam, []).append(rec['nmse'][-1])
    if n == 256:
        for fam, v in R1_256.get(cut, {}).items():
            ng.setdefault(fam, [v])
    out = []
    for fam, v in sorted(ng.items(), key=lambda kv: int(kv[0][2:])):
        k = int(fam[2:])
        if not all(str(k) in x['tail_PC'] for x in R):
            continue
        tail = np.mean([x['tail_PC'][str(k)] for x in R])
        out.append(f"{fam}: nMSE {np.mean(v):.2e} / tail {tail:.2e} = C {np.mean(v) / tail:.2e}")
    print(f"{n}x{L} cut {cut}: " + "; ".join(out))
