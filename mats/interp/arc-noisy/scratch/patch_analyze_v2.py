"""One-off patch: analyze.py sampler lookup -> measured checkpoints with exact costs (v2 files in samp_v2/, else v1
files with corrected costs), largest checkpoint whose total cost <= B."""
p = '/home/agent/workspaces/mats/work/arc-noisy/analyze.py'
s = open(p).read()
a = s.index("    sub, key = ('sampanti', 'mubanti') if method == 'mubanti' else ('samp', method)")
b = s.index("def collect(L, sigma):")
new = '''    key = method
    T = truth(seed, sigma)
    if T is None:
        return None
    d = None
    p2 = os.path.join(HERE, 'samp_v2', f'samp_{seed}_s{sigma:g}_L{L}.npz')
    if os.path.exists(p2):
        d2 = np.load(p2)
        if f'N_{key}' in d2.files:
            d = d2; N = d[f'N_{key}']; C = d[f'costs_{key}']
    if d is None:                                         # version-1 files: sparse checkpoints, costs corrected
        sub = 'sampanti' if method == 'mubanti' else 'samp'
        p1 = os.path.join(HERE, sub, f'samp_{seed}_s{sigma:g}_L{L}.npz')
        if not os.path.exists(p1):
            return None
        d = np.load(p1)
        if f'N_{key}' not in d.files:
            return None
        N = d[f'N_{key}']; c1 = float(d[f'cost_{key}'])
        setup = 6 * 256 ** 3 + 256 * 256 * anet.F_NORMAL
        if key == 'mub':
            C = N * (c1 - 1500) + np.ceil(N / 66048) * setup
        elif key == 'mubanti':
            C = N * (c1 - 3000) + np.ceil(N / 66048) * setup
        elif key == 'sobol':
            C = N * (c1 - 256 * anet.F_NORMAL + 256 * 100)
        else:
            C = N * c1
    S = d[f'sum_{key}_{L}']
    out = {}
    for bi, B in enumerate(BUDGETS):
        j = int(np.searchsorted(C, B, side='right') - 1)
        if j >= 0:
            out[bi] = float(np.mean([truthlib.nmse(S[r, j] / N[j], T, L) for r in range(S.shape[0])]))
    return out


'''
s = s[:a] + new + s[b:]
open(p, 'w').write(s)
print('patched')
