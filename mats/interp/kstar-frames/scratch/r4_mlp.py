"""Run D (PREREG Q1d): learned map from the state direction u = z_c/|z_c| to a frame. MLP (2 hidden layers of width 4n,
GELU) -> n x k matrix A(u), output layer initialised at 0; G = orth(D_c (G_fix + A(u))), G_fix = optimal fixed frame
(top-k of Mt from the reference paths). Loss -mean CS on the first N_train paths of A (90% fit, 10% early stopping);
scored on the held-out paths B (O, CS, T).
nice -n 19 ionice -c3 python r4_mlp.py --n 64 --seed 400 --L 32 --c 16 --ntrain 4096"""
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'lib'))
import kf

ap = argparse.ArgumentParser()
ap.add_argument('--n', type=int, default=64); ap.add_argument('--seed', type=int, default=400)
ap.add_argument('--L', type=int, default=32); ap.add_argument('--c', type=int, default=16)
ap.add_argument('--ntrain', type=int, default=4096); ap.add_argument('--threads', type=int, default=3)
ap.add_argument('--epochs', type=int, default=150); ap.add_argument('--patience', type=int, default=10)
ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--batch', type=int, default=256)
ap.add_argument('--runs', default=os.path.join(HERE, '..', 'runs'))
ap.add_argument('--out', default=None, help='directory for the result json (default: --runs)')
A = ap.parse_args()
torch.set_num_threads(A.threads)
torch.manual_seed(1234 + A.seed + A.c)
n, L, seed, c = A.n, A.L, A.seed, A.c
tag = f"n{n}_s{seed}_L{L}"
d = np.load(os.path.join(A.runs, f'B_{tag}_c{c}.npz'))
k = int(d['k'])
NA = int(d['NA'])
Nt = min(A.ntrain, NA)
nfit = int(round(0.9 * Nt))
f32 = torch.float32


def prep(z, V, s):
    z = torch.from_numpy(z).to(f32).to(kf.DEV)
    return (z / z.norm(dim=-1, keepdim=True), (z > 0).to(f32), torch.from_numpy(V).to(f32).to(kf.DEV),
            torch.from_numpy(s).to(f32).to(kf.DEV))


uA, DA, VA, sA = prep(d['zA'][:Nt], d['VA'][:Nt], d['sA'][:Nt])
uB, DB, VB, sB = prep(d['zB'], d['VB'], d['sB'])
EB = torch.from_numpy(d['EB']).to(kf.DEV)
Gfix = torch.from_numpy(d['GM']).to(f32).to(kf.DEV)
H = 4 * n


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(n, H), nn.GELU(), nn.Linear(H, H), nn.GELU(), nn.Linear(H, n * k))
        nn.init.zeros_(self.f[-1].weight); nn.init.zeros_(self.f[-1].bias)

    def forward(self, u, D):
        Am = self.f(u).reshape(-1, n, k) + Gfix
        Q, _ = torch.linalg.qr(D[:, :, None] * Am)
        return Q


def cs_of(G, V, s):
    proj = V.transpose(-1, -2) @ G
    w = (proj ** 2).sum(-1)
    s2 = s ** 2
    return (s2 * w).sum(-1) / s2[:, :k].sum(-1)


model = Net().to(kf.DEV)
nparams = sum(p.numel() for p in model.parameters())
opt = torch.optim.Adam(model.parameters(), lr=A.lr, weight_decay=1e-5)
best, best_state, bad = -1.0, None, 0
t0 = time.time()
idx_fit = torch.arange(nfit, device=kf.DEV)
for ep in range(A.epochs):
    model.train()
    perm = idx_fit[torch.randperm(nfit, device=kf.DEV)]
    for i in range(0, nfit, A.batch):
        b = perm[i:i + A.batch]
        loss = -cs_of(model(uA[b], DA[b]), VA[b], sA[b]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        val = float(cs_of(model(uA[nfit:], DA[nfit:]), VA[nfit:], sA[nfit:]).mean())
        tr = float(cs_of(model(uA[:min(nfit, 2048)], DA[:min(nfit, 2048)]), VA[:min(nfit, 2048)],
                         sA[:min(nfit, 2048)]).mean())
    if val > best + 1e-5:
        best, bad = val, 0
        best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
    else:
        bad += 1
    if ep % 5 == 0 or bad == 0:
        print(f"[{tag} c={c} N={Nt}] epoch {ep} train CS {tr:.4f} val CS {val:.4f} best {best:.4f} {time.time()-t0:.0f}s",
              flush=True)
    if bad >= A.patience:
        break
model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    G = torch.cat([model(uB[i:i + 512], DB[i:i + 512]) for i in range(0, uB.shape[0], 512)]).double()
    G0 = kf.project_active(Gfix.double().expand(uB.shape[0], n, k), DB.double())
O, CS, T = kf.scores(G, VB.double(), sB.double(), k, EB)
O0, CS0, T0 = kf.scores(G0, VB.double(), sB.double(), k, EB)
res = {'n': n, 'seed': seed, 'L': L, 'c': c, 'k': k, 'ntrain': Nt, 'params': nparams, 'epochs': ep + 1,
       'val_best': best, 'train_last': tr,
       'mlp': {'O': float(O.mean()), 'CS': float(CS.mean()), 'T': float(T.mean())},
       'fixed_proj': {'O': float(O0.mean()), 'CS': float(CS0.mean()), 'T': float(T0.mean())}}
print(f"[{tag} c={c} N={Nt}] held-out: MLP O={res['mlp']['O']:.4f} CS={res['mlp']['CS']:.4f} T={res['mlp']['T']:.4f} | "
      f"fixed(proj) O={res['fixed_proj']['O']:.4f} CS={res['fixed_proj']['CS']:.4f} | params {nparams} "
      f"epochs {ep+1} {time.time()-t0:.0f}s", flush=True)
res['epochs_cap'] = A.epochs; res['patience'] = A.patience
json.dump(res, open(os.path.join(A.out or A.runs, f'D_{tag}_c{c}_N{Nt}.json'), 'w'), indent=1)
