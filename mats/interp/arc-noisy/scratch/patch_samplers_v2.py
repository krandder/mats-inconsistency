"""One-off patch: gpu_samplers.py -> version 2 (dense checkpoints with exact costs, per-block MUB setup, sigma=0 fix,
fit cost recorded). Keeps the previous file as gpu_samplers_v1.py."""
import shutil
p = '/home/agent/workspaces/mats/work/arc-noisy/gpu_samplers.py'
shutil.copyfile(p, p.replace('gpu_samplers.py', 'gpu_samplers_v1.py'))
s = open(p).read()
old_doc = s[s.index('For each (seed, sigma, L, method)'):s.index('"""', 10)]
new_doc = '''Version 2 (2026-10-29 ~03:10Z, after the GPT-6 code review): records prefix sums (float64) of a_L and of the control
layers a_m at a DENSE geometric grid of checkpoints (48 values up to the largest affordable N), together with the exact
total flopscope cost of each checkpoint (costs_<method>), so that any method or hybrid can be compared at a measured
(not rescaled) sample count for its budget. Kerdock-MUB setup (Haar draw + QR + folding into W1, 6 n^3 + 16 n^2 FLOPs) is
charged once per started 66,048-point block; the per-point charge covers the basis product (n^2), negation, ReLU and a chi
quantile (100 per point). Sobol points are charged 100 FLOPs per coordinate (generation + ndtri). The 4,096-path pilot
regression a_L ~ [a_m, 1] is stored for offline linear controls; its path cost and its least-squares fit cost
(fit_cost, per control layer) are recorded for the hybrid scorer to charge. Version 1 (sparse checkpoints, setup
amortised as +1600 FLOPs per point) is kept as gpu_samplers_v1.py for the files it produced.
Saved: OUTDIR/samp_<seed>_s<sigma>_L<L>.npz with sum_<method>_<layer> (R, n_checkpoints, n), N_<method>,
costs_<method>, A_<m>, b_<m>, pilot_cost, fit_cost, and the budgets.
'''
s = s.replace(old_doc, new_doc)
old_cost = s[s.index('def path_cost(net, L, method):'):s.index('@torch.no_grad()')]
new_cost = '''MUB_BLOCK = 129 * 2 * 256
MUB_SETUP = 6 * n ** 3 + n * n * anet.F_NORMAL


def path_cost(net, L, method):
    """Per-point cost (FLOPs) excluding per-block setup."""
    c = net.flops_path(L)
    if method in ('mub', 'mubanti'):
        c = c - n * anet.F_NORMAL - anet.mm_flops(1, n, n) + n * n + n + 100
    if method == 'mubanti':
        c = 2 * c                       # two paths per design point (first layer shared: counted twice, conservative)
    if method == 'sobol':
        c = c - n * anet.F_NORMAL + n * 100
    return c


def total_cost(N, net, L, method):
    N = np.asarray(N, dtype=np.float64)
    tot = N * path_cost(net, L, method)
    if method in ('mub', 'mubanti'):
        tot = tot + np.ceil(N / MUB_BLOCK) * MUB_SETUP
    return tot


def checkpoints(net, L, method, Bmax, k=48):
    lo, hi = 2, 4_000_000
    while hi - lo > 1:                                  # largest N with total cost <= Bmax
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if total_cost(mid, net, L, method) <= Bmax else (lo, mid)
    grid = np.unique(np.round(np.geomspace(32, lo, k)).astype(np.int64))
    if method in ('mub', 'mubanti'):
        grid = np.unique(np.maximum(2, grid // 2 * 2))
    return grid


'''
s = s.replace(old_cost, new_cost)
s = s.replace('''    h = a
    for l in range(1, L):
        if s[l - 1] > 0:''', '''    for l in range(1, L):
        h = a
        if s[l - 1] > 0:''')
s = s.replace("""            res['pilot_cost'] = 4096 * net.flops_path(L)""", """            res['pilot_cost'] = 4096 * net.flops_path(L)
            P, p1 = 4096, n + 1                                     # Householder QR + Q^T y + back substitution
            res['fit_cost'] = 2 * P * p1 ** 2 - (2 / 3) * p1 ** 3 + 4 * P * p1 * n + p1 ** 2 * n""")
old_cp = s[s.index("                c = path_cost(net, L, method)\n"):s.index("                out = {ly: np.zeros((args.R, len(cps), n)) for ly in keep}")]
new_cp = """                cps = checkpoints(net, L, method, budgets.max()); Nmax = int(cps[-1])
                res[f'N_{method}'] = cps; res[f'costs_{method}'] = total_cost(cps, net, L, method)
"""
s = s.replace(old_cp, new_cp)
open(p, 'w').write(s)
print('patched')
