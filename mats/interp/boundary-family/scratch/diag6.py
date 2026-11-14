"""Diagnostic 6 (width 256, ARC dev nets): gate compression at a cut m (Kelvin: families that line up with the gate
hyperplanes and compress similar / functionally correlated gates).

Near-boundary gates: D_raw = #{i: entropy of the gate of z_m,i > 0.1 bit} (fit set, 2^NFIT paths).
Groupings (single-linkage at cosine/phi threshold c; offsets within delta = 0.25 sd for geometry):
  geo   unit normals U_i/|U_i| and offsets mu_i/|U_i| in the relevance subspace U (top-3 eigenvectors of E[g g^T])
  sw    phi coefficient (correlation) of the gate indicators
  eff   cosine of the downstream effect vectors E[d a_32 / d z_m,i] (finite differences, common noise)
D_eff(c) = number of groups. Error cost (representation, exact remaining network, against the independent truth):
cells = sign patterns of D boundaries, Gaussian per cell truncated to its pattern; boundaries = (a) the D_eff group
representatives (highest-entropy member) for the sw grouping at c = 0.9, (b) the top-D raw gates by relevance with
D = D_eff, (c) the top-2D raw gates. Patterns with < 600 fit paths are pooled (untruncated).
python scratch/diag6.py SEED M NFIT NREP"""
import json, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
torch.set_num_threads(int(os.environ.get('TH', '3')))
import bf, gmprop

seed, m = int(sys.argv[1]), int(sys.argv[2])
NFIT, NREP = 1 << int(sys.argv[3]), 1 << int(sys.argv[4])
L = 32; n = 256
ARC = '/home/agent/workspaces/mats/work/arc-noisy'
net = bf.get_net(seed, n, 32, 0.001)
d = np.load(f'{ARC}/truth/truth_256_32_s0.001_seed{seed}.npz'); NT = float(d['N'])
TM = torch.tensor(d['S1'] / NT); TV = torch.tensor(d['S2'] / NT) - TM ** 2
err = lambda e: float(((e - TM[L - 1]) ** 2).mean() / TV[L - 1].mean()) - 1 / NT
t0 = time.time()
smp = bf.mc_stats(net, NFIT, seed=80_000 + seed, L=m, cuts=(m,), keep_n=NFIT, out_moments=False)
Z = smp['cuts'][m].double()
W = [w.double() for w in net.W]; s = [float(v) for v in net.s]
res = dict(seed=seed, m=m)


def tail(zm, gen, grad=False):
    h = torch.relu(zm) + s[m - 1] * torch.randn(zm.shape, generator=gen)
    for l in range(m, L):
        a = torch.relu(h @ W[l].float().T)
        h = a + s[l] * torch.randn(a.shape, generator=gen) if l < L - 1 else a
    return h


# relevance subspace
g = torch.Generator().manual_seed(5)
NA = 1 << 13
zz = Z[:NA].float().clone().requires_grad_(True)
r = torch.randn(NA, n, generator=g)
(tail(zz, g) * r).sum().backward()
Gr = zz.grad.double(); Mx = Gr.T @ Gr / NA
evM, UA = torch.linalg.eigh(Mx); UA = UA.flip(1)
U = UA[:, :3]
mz = Z.mean(0); Cz = torch.cov(Z.T); sdz = Cz.diagonal().sqrt()
G = (Z > 0)
pon = G.double().mean(0)
ent = -(pon * pon.clamp_min(1e-12).log2() + (1 - pon) * (1 - pon).clamp_min(1e-12).log2())
near = torch.nonzero(ent > 0.1).flatten()
Draw = int(near.numel())
res['D_raw'] = Draw; res['entropy_sum_near'] = float(ent[near].sum())
relev = ent * Mx.diagonal().sqrt()
print(f'D_raw {Draw}  sum entropy {float(ent[near].sum()):.1f} bits  {time.time() - t0:.0f}s', flush=True)

# geometry in the relevance subspace: y = U^T (z - mu); gate i: mu_i + U_i . y = 0
Ui = U[near]; nU = Ui.norm(dim=1).clamp_min(1e-12)
nrm = Ui / nU[:, None]
Ys = ((Z - mz) @ U).std(0)                              # sd of y along each axis
off = (mz[near] / nU) / float(Ys.mean())                # offset in sd units of y
cosg = nrm @ nrm.T
# co-switching
Gn = G[:, near].double(); Gc = Gn - Gn.mean(0)
phi = (Gc.T @ Gc) / (Gc.std(0)[:, None] * Gc.std(0)[None, :] * Gn.shape[0]).clamp_min(1e-12)
# downstream effect vectors (finite differences on a batch with common noise)
B = 512; eps = 1e-3
zb = Z[:B].float()
gen0 = torch.Generator().manual_seed(9); base = tail(zb, gen0)
eff = torch.zeros(Draw, n, dtype=torch.float64)
for j, i in enumerate(near.tolist()):
    zp = zb.clone(); zp[:, i] += eps * float(sdz[i])
    genj = torch.Generator().manual_seed(9)
    eff[j] = ((tail(zp, genj) - base).double().mean(0)) / (eps * float(sdz[i]))
effn = eff / eff.norm(dim=1, keepdim=True).clamp_min(1e-12)
cose = effn @ effn.T
print(f'effects done {time.time() - t0:.0f}s', flush=True)


def groups(A, c, extra=None):
    """single-linkage components of the graph A > c (and extra mask)."""
    adj = A > c
    if extra is not None:
        adj &= extra
    lab = list(range(A.shape[0]))
    def find(x):
        while lab[x] != x:
            lab[x] = lab[lab[x]]; x = lab[x]
        return x
    idx = torch.nonzero(torch.triu(adj, 1))
    for a, b in idx.tolist():
        ra, rb = find(a), find(b)
        if ra != rb:
            lab[ra] = rb
    roots = [find(x) for x in range(A.shape[0])]
    return roots


res['D_eff'] = {}
offclose = (off[:, None] - off[None, :]).abs() < 0.25
for c in (0.8, 0.9, 0.95):
    res['D_eff'][str(c)] = dict(geo=len(set(groups(cosg, c, offclose))), sw=len(set(groups(phi, c))),
                                eff=len(set(groups(cose, c))))
print('D_eff', json.dumps(res['D_eff']), flush=True)
# summary of pairwise similarity
iu = torch.triu_indices(Draw, Draw, 1)
res['pair_median'] = dict(cos_geo=float(cosg[iu[0], iu[1]].abs().median()), phi=float(phi[iu[0], iu[1]].abs().median()),
                          cos_eff=float(cose[iu[0], iu[1]].abs().median()))
res['pair_frac_gt_0.9'] = dict(phi=float((phi[iu[0], iu[1]].abs() > 0.9).double().mean()),
                               cos_eff=float((cose[iu[0], iu[1]].abs() > 0.9).double().mean()))


def rep_cells(bidx):
    D = len(bidx)
    lab = ((Z[:, bidx] > 0).long() * (1 << torch.arange(D))).sum(1)
    cells = []; rest = torch.zeros(Z.shape[0], dtype=torch.bool)
    for cc in torch.unique(lab):
        S = lab == cc
        if int(S.sum()) < 600:
            rest |= S; continue
        cells.append((float(S.double().mean()), Z[S].mean(0), torch.cov(Z[S].T), ((int(cc) >> torch.arange(D)) & 1).bool()))
    if rest.any():
        cells.append((float(rest.double().mean()), Z[rest].mean(0), torch.cov(Z[rest].T), None))
    gg = torch.Generator().manual_seed(11)
    parts = []
    for w, mm, C, pat in cells:
        k = int(round(NREP * w))
        if k == 0:
            continue
        ev, V = torch.linalg.eigh(C); Lh = V * ev.clamp_min(0).sqrt()
        got = []; need = k
        for _ in range(400):
            dd = mm + torch.randn(max(2 * need, 2048), n, generator=gg, dtype=torch.float64) @ Lh.T
            if pat is not None:
                dd = dd[((dd[:, bidx] > 0) == pat).all(1)]
            got.append(dd[:need]); need -= got[-1].shape[0]
            if need <= 0:
                break
        parts.append(torch.cat(got, 0).float())
    e = err(bf.propagate_from(net, torch.cat(parts, 0), m, L, seed=7))
    return e, len(cells), sum(1 for c in cells if c[0] > 1e-4)


order = torch.argsort(relev, descending=True)
roots = groups(phi, 0.9)
reps = {}
for j, rt in enumerate(roots):
    i = int(near[j])
    if rt not in reps or ent[i] > ent[reps[rt]]:
        reps[rt] = i
rep_list = sorted(reps.values(), key=lambda i: -float(relev[i]))
res['err'] = {}
for D in (4, 6, 8):
    if D > len(rep_list):
        break
    e1 = rep_cells(torch.tensor(rep_list[:D]))
    e2 = rep_cells(order[:D])
    res['err'][str(D)] = dict(grouped=e1, raw=e2)
    print(D, 'grouped-reps', e1, 'raw-top', e2, f'{time.time() - t0:.0f}s', flush=True)
os.makedirs('runs/diag6', exist_ok=True)
json.dump(res, open(f'runs/diag6/diag6_{seed}_m{m}.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != 'err'}), flush=True)
print(f'done {time.time() - t0:.0f}s', flush=True)
