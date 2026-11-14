"""c8_fixed.py (checker) -- 4x4, exact: what a SMALL class of routes learns from n real grids.
The exact trust optimum (c2_4x4.py full) is a table with one free choice per state and memorises its sample. Here the
class is 256 random fixed orders (the same 256 as in c2_4x4_sym.out): (a) pick the single order with the lowest mean
surprise on n real grids drawn from the true law; (b) fit mixture weights over the 256 orders by EM on the same n grids.
Both are then scored exactly against the whole law (all 288 grids). 30 repetitions per n."""
import json, math
import numpy as np, torch
import ck

K, N = 4, 16; LN = math.log(288)
G = np.load(f"{ck.W}/data/k4.npz")["G"].astype(np.int64); Gt = torch.from_numpy(G); net = ck.M("mA4", 4)
rng = np.random.default_rng(20261110); L = []
for r in range(256):
    perm = rng.permutation(N); rank = torch.empty(N); rank[torch.as_tensor(perm)] = torch.arange(N).float()
    L.append(ck.walk(net, lambda s, lp, rank=rank: rank.expand(s.shape[0], -1).clone(), torch.full_like(Gt, -1), X=Gt)["logp"].numpy())
L = np.array(L)                                              # [256 orders, 288 grids] log T_order(grid), exact
np.save("out/c8_fixed_logT.npy", L)
kl = -L.mean(1) - LN
print(f"256 fixed orders, exact KL to the true law: best {kl.min():.4f}, median {np.median(kl):.4f}, worst {kl.max():.4f}; uniform mixture {-np.log(np.exp(L).mean(0)).mean() - LN:.4f}")


def em(Ls, iters=300):
    w = np.full(Ls.shape[0], 1 / Ls.shape[0])
    for _ in range(iters):
        a = Ls.T + np.log(w + 1e-300); a -= a.max(1, keepdims=True); r = np.exp(a); r /= r.sum(1, keepdims=True); w = r.mean(0)
    return w


OUT = {}; r2 = np.random.default_rng(99)
print(f"{'n real grids':>12s} {'best single order picked on the sample':>40s} {'mixture, weights by EM on the sample':>40s}   (exact KL to the true law, mean and range over 30 samples)")
for n in (9, 18, 36, 72, 144, 288, 1152):
    a, b = [], []
    for rep in range(30):
        idx = r2.integers(288, size=n); pick = int(np.argmax(L[:, idx].mean(1))); a.append(kl[pick])
        w = em(L[:, idx]); b.append(float(-np.log((np.exp(L) * w[:, None]).sum(0)).mean() - LN))
    OUT[n] = dict(single_mean=float(np.mean(a)), single_min=float(np.min(a)), single_max=float(np.max(a)), mix_mean=float(np.mean(b)), mix_min=float(np.min(b)), mix_max=float(np.max(b)))
    print(f"{n:12d} {np.mean(a):22.4f} [{np.min(a):.4f}, {np.max(a):.4f}] {np.mean(b):22.4f} [{np.min(b):.4f}, {np.max(b):.4f}]")
w = em(L); print(f"   with the whole law: best single {kl.min():.4f}; EM mixture {float(-np.log((np.exp(L) * w[:, None]).sum(0)).mean() - LN):.4f} (orders with weight > 1%: {int((w > 0.01).sum())})")
OUT["whole"] = dict(single=float(kl.min()), mix=float(-np.log((np.exp(L) * w[:, None]).sum(0)).mean() - LN))
json.dump(OUT, open("out/c8_fixed.json", "w"), indent=1)
