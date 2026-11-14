"""Quick look at score logs: python peek.py <log> [keys]"""
import ast, sys
keys = sys.argv[2].split(',') if len(sys.argv) > 2 else ['rel_KLc', 'rel_Dinf', 'dec_KLc', 'dec_Dinf', 'smc_KLc', 'smc_KLw', 'smc_c']
for line in open(sys.argv[1]):
    if not line[:2] in ('G ', 'NC', 'GC', 'KN'):
        continue
    fam, rest = line.split(' ', 1)
    try:
        d = ast.literal_eval(rest.replace('np.float64(', '(').strip())
    except Exception:
        continue
    print(f"{fam:3s} L{d['layer']}: " + ' '.join(f"{k}={d[k]:.2f}" for k in keys if k in d))
