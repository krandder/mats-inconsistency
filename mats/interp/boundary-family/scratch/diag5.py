"""Diagnostic 5 (width 256, ARC dev nets, independent ARC 2^25-path truths): representation error vs push error of
cell families at a cut m (the critique's decisive measurement, with Kelvin's relevance axes).

Cells are fitted on one Monte Carlo set (2^NFIT paths of z_m); errors are against the independent truth at layer 32.
Families (Gaussian per cell, true within-cell moments):
  G        one Gaussian
  PCAk     quantile bins of the top-k PCA modes of z_m (16 bins for k = 1, 4^k for k > 1)
  ASk      the same along the top-k eigenvectors of M = E[g g^T], g = d(r . a_32)/d z_m with r ~ N(0, I) per path
           (the active subspace of the all-outputs mean question)
  GATEd    sign patterns of the d most relevant gates of layer m (relevance = gate entropy x sqrt(M_ii)); patterns
           with < 600 fit paths pooled into one remainder cell
  GATETd   the same cells, each Gaussian truncated to its sign pattern (rejection) in the representation test
Per family: rep = draw 2^NREP paths from the fitted mixture, exact remaining layers (with noise) -> E[a_32];
            push = per-cell exact Gaussian closure to layer 32 -> E[a_32].
python scratch/diag5.py SEED M NFIT NREP [FAMILIES]"""
import json, math, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
torch.set_num_threads(int(os.environ.get('TH', '3')))
import bf, gmprop

seed, m = int(sys.argv[1]), int(sys.argv[2])
NFIT, NREP = 1 << int(sys.argv[3]), 1 << int(sys.argv[4])
FAM = sys.argv[5].split(',') if len(sys.argv) > 5 else ['G', 'PCA1', 'PCA3', 'AS1', 'AS3', 'GATE6', 'GATET6']
L = 32; n = 256
ARC = '/home/agent/workspaces/mats/work/arc-noisy'
net = bf.get_net(seed, n, 32, 0.001)
d = np.load(f'{ARC}/truth/truth_256_32_s0.001_seed{seed}.npz'); NT = float(d['N'])
TM = torch.tensor(d['S1'] / NT); TV = torch.tensor(d['S2'] / NT) - TM ** 2
err = lambda e: float(((e - TM[L - 1]) ** 2).mean() / TV[L - 1].mean()) - 1 / NT
t0 = time.time()
smp = bf.mc_stats(net, NFIT, seed=70_000 + seed, L=m, cuts=(m,), keep_n=NFIT, out_moments=False)
Z = smp['cuts'][m].double()
print(f'fit samples {time.time() - t0:.0f}s', flush=True)
W = [w.double() for w in net.W]; s = [float(v) for v in net.s]
res = dict(seed=seed, m=m, NFIT=NFIT, NREP=NREP)

# active subspace of the all-outputs mean question at z_m
g = torch.Generator().manual_seed(5)
NA = 1 << 13
zz = Z[:NA].float().clone().requires_grad_(True)
h = torch.relu(zz) + s[m - 1] * torch.randn(zz.shape, generator=g)
for l in range(m, L):
    zl = h @ W[l].float().T
    a = torch.relu(zl)
    h = a + s[l] * torch.randn(a.shape, generator=g) if l < L - 1 else a
r = torch.randn(NA, n, generator=g)
(h * r).sum().backward()
G_ = zz.grad.double()
M = G_.T @ G_ / NA
evM, UA = torch.linalg.eigh(M); evM = evM.flip(0); UA = UA.flip(1)
mz = Z.mean(0); Cz = torch.cov(Z.T)
evP, UP = torch.linalg.eigh(Cz); evP = evP.flip(0); UP = UP.flip(1)
sv = torch.linalg.svdvals(UP[:, :3].T @ UA[:, :3])
res['overlap_top3_cos'] = [float(v) for v in sv]
res['AS_top_frac'] = [float(v) for v in (evM[:8] / evM.sum())]
res['PCA_top_frac'] = [float(v) for v in (evP[:8] / evP.sum())]
# variance captured by AS axes and gradient energy captured by PCA axes
res['var_in_AS3'] = float((UA[:, :3].T @ Cz @ UA[:, :3]).trace() / Cz.trace())
res['grad_in_PCA3'] = float((UP[:, :3].T @ M @ UP[:, :3]).trace() / M.trace())
print('axes', json.dumps({k: res[k] for k in ('overlap_top3_cos', 'var_in_AS3', 'grad_in_PCA3')}),
      'AS top', [round(v, 3) for v in res['AS_top_frac'][:4]], 'PCA top', [round(v, 3) for v in res['PCA_top_frac'][:4]],
      flush=True)
sd = Cz.diagonal().sqrt(); pon = (Z > 0).double().mean(0)
ent = -(pon * pon.clamp_min(1e-12).log2() + (1 - pon) * (1 - pon).clamp_min(1e-12).log2())
relev = ent * M.diagonal().sqrt()
order = torch.argsort(relev, descending=True)
res['gate_entropy_sum'] = float(ent.sum()); res['n_gates_entropy_gt_0.1'] = int((ent > 0.1).sum())


def bins(Y, k):
    nb = 16 if k == 1 else 4
    lab = torch.zeros(Y.shape[0], dtype=torch.long)
    for j in range(k):
        qs = torch.quantile(Y[:, j], torch.linspace(0, 1, nb + 1, dtype=torch.float64)[1:-1])
        lab = lab * nb + torch.bucketize(Y[:, j].contiguous(), qs)
    return lab


def cells_from_labels(lab, minc=0):
    out = []; rest = torch.zeros(lab.shape[0], dtype=torch.bool)
    for c in torch.unique(lab):
        S = lab == c
        if int(S.sum()) < minc:
            rest |= S; continue
        out.append((S, gmprop.Cell(float(S.double().mean()), Z[S].mean(0), torch.cov(Z[S].T))))
    if rest.any():
        out.append((rest, gmprop.Cell(float(rest.double().mean()), Z[rest].mean(0), torch.cov(Z[rest].T))))
    return out


def rep_error(cells, trunc=None):
    """Sample the mixture (optionally truncated: trunc = (idx, pattern per cell)), propagate exactly."""
    gg = torch.Generator().manual_seed(11)
    parts = []
    for j, (S, c) in enumerate(cells):
        k = int(round(NREP * c.w))
        if k == 0:
            continue
        ev, V = torch.linalg.eigh(c.S); Lh = V * ev.clamp_min(0).sqrt()
        if trunc is None:
            parts.append((c.m + torch.randn(k, n, generator=gg, dtype=torch.float64) @ Lh.T).float())
        else:
            idx, pats = trunc
            got = []; need = k
            for _ in range(400):
                dd = c.m + torch.randn(max(2 * need, 2048), n, generator=gg, dtype=torch.float64) @ Lh.T
                if pats[j] is not None:
                    dd = dd[((dd[:, idx] > 0) == pats[j]).all(1)]
                got.append(dd[:need]); need -= got[-1].shape[0]
                if need <= 0:
                    break
            parts.append(torch.cat(got, 0).float())
    Zs = torch.cat(parts, 0)
    return err(bf.propagate_from(net, Zs, m, L, seed=7))


def push_error(cells):
    cs = [c for _, c in cells]
    for l in range(m, L):
        cs = [gmprop.push(c, W[l], s[l - 1] ** 2)[0] for c in cs]
    return err(gmprop.readout_mean(cs))


for fam in FAM:
    t1 = time.time()
    trunc = None
    if fam == 'G':
        cells = [(torch.ones(Z.shape[0], dtype=torch.bool), gmprop.Cell(1.0, mz, Cz))]
    elif fam.startswith('PCA'):
        k = int(fam[3:]); cells = cells_from_labels(bins((Z - mz) @ UP[:, :k], k))
    elif fam.startswith('AS'):
        k = int(fam[2:]); cells = cells_from_labels(bins((Z - mz) @ UA[:, :k], k))
    elif fam.startswith('SPLIT'):
        # deductive control: the single Gaussian with the fitted moments, split 3^k ways along its top-k PCA modes
        # (moment-preserving, gmprop.grid_split); no information beyond the Gaussian enters
        k = int(fam[5:])
        kids, _ = gmprop.grid_split(gmprop.Cell(1.0, mz, Cz), k)
        cells = [(None, c) for c in kids]
    elif fam.startswith('GATE'):
        T = fam.startswith('GATET'); D = int(fam[5:] if T else fam[4:])
        idx = order[:D]
        lab = ((Z[:, idx] > 0).long() * (1 << torch.arange(D))).sum(1)
        cells = cells_from_labels(lab, minc=600)
        if T:
            pats = []
            for S, c in cells:
                ls = torch.unique(lab[S])
                pats.append(((ls[0] >> torch.arange(D)) & 1).bool() if len(ls) == 1 else None)
            trunc = (idx, pats)
    r = dict(cells=len(cells), rep=rep_error(cells, trunc))
    if not fam.startswith('GATET'):
        r['push'] = push_error(cells)
    res[fam] = r
    print(fam, json.dumps({k: (float(f'{v:.4g}') if isinstance(v, float) else v) for k, v in r.items()}),
          f'{time.time() - t1:.0f}s', flush=True)
os.makedirs('runs/diag5', exist_ok=True)
json.dump(res, open(f'runs/diag5/diag5_{seed}_m{m}.json', 'w'), indent=1)
print(f'done {time.time() - t0:.0f}s', flush=True)
