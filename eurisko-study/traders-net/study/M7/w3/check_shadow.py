"""M7 w3 check: (1) the fullinfo shadow credit c_e = ghat_e (r_e - R) equals the autograd router gradient
-dL/dz_e when e is added to the routed set S (renormalised softmax over S u {e}), up to O(ghat^2) because r is
read at the mix without e; (2) the exp3 estimator (one uniform unrouted sample, / q) is unbiased for fullinfo.
Writes results/check_shadow.json. CPU, seconds."""
import json
import os

import torch
torch.set_num_threads(1)

torch.manual_seed(0)
N, d, E, k = 64, 16, 8, 2
x = torch.randn(N, d)
W = torch.randn(E, d, d) / d ** 0.5
tgt = torch.randn(N, d)
z = torch.randn(N, E) * 2


def loss_of(h):
    return ((h - tgt) ** 2).sum(1).mean()


Eo = torch.einsum("nd,edf->nef", x, W)
S = z.topk(k, 1).indices
routed = torch.zeros(N, E, dtype=torch.bool).scatter(1, S, True)
g = torch.softmax(z.gather(1, S), 1)
h = (g[..., None] * Eo.gather(1, S[..., None].expand(-1, -1, d))).sum(1).requires_grad_(True)
loss_of(h).backward()
rev = -torch.einsum("ned,nd->ne", Eo, h.grad)
R = (torch.zeros(N, E).scatter(1, S, g) * rev).sum(1)
lseS = torch.logsumexp(z.masked_fill(~routed, float("-inf")), 1, keepdim=True)
ghat = torch.exp(z - torch.logaddexp(z, lseS))
c_full = (~routed).float() * ghat * (rev - R[:, None])

# autograd with e added to S, one e at a time
err, ref = [], []
for n in range(N):
    for e in range(E):
        if routed[n, e]:
            continue
        idx = torch.cat([S[n], torch.tensor([e])])
        zz = z[n, idx].clone().requires_grad_(True)
        gg = torch.softmax(zz, 0)
        hh = (gg[:, None] * Eo[n, idx]).sum(0)
        (((hh - tgt[n]) ** 2).sum() / N).backward()
        a = -zz.grad[-1].item()
        ref.append(a)
        err.append(abs(a - c_full[n, e].item()))
ref_t = torch.tensor(ref)
err_t = torch.tensor(err)
corr = torch.corrcoef(torch.stack([ref_t, c_full[~routed]]))[0, 1].item()

# unbiasedness of the exp3 sample
M = 20000
acc = torch.zeros(N, E)
gen = torch.Generator().manual_seed(1)
for _ in range(M):
    u = torch.rand(N, E, generator=gen)
    u[routed] = -1
    e = u.argmax(1)
    mask = torch.zeros(N, E).scatter(1, e[:, None], 1.0)
    acc += mask * c_full * (E - k)
est = acc / M
rel_bias = ((est - c_full).abs().sum() / c_full.abs().sum()).item()

out = dict(N=N, E=E, k=k,
           autograd_vs_formula_corr=corr,
           autograd_vs_formula_median_abs_err=float(err_t.median()),
           autograd_median_abs=float(ref_t.abs().median()),
           exp3_mean_rel_bias_20k_samples=rel_bias)
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"), exist_ok=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "check_shadow.json"), "w"), indent=1)
print(json.dumps(out))
