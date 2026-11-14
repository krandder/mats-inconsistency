"""Conservation test on a tiny model: exact with frozen attention, small residual live."""
import torch, torch.nn.functional as F
from model import Net
from books import account, N_FAM
from streams import make_questions

torch.manual_seed(0)
bits, fam, _ = make_questions(16, 600, 3)
Q, W = 16, 32
b = torch.tensor(bits, dtype=torch.float64)
wins = b.unfold(1, W, 1)
idx = torch.randint(0, 500, (8,))
win, y = wins[:, idx].permute(1, 0, 2), b[:, idx + W].T
m = Net(Q, W, d=16, H=4, L=2).double()
with torch.no_grad():
    for p in m.parameters():
        p.add_(0.1 * torch.randn_like(p))     # non-trivial biases / LN shifts
oh = F.one_hot(torch.tensor(fam), N_FAM).double()
for fa in (True, False):
    r = account(m, win, y, oh, freeze_attn=fa)
    rl = account(m, win, y, oh, freeze_attn=fa, batched=False)
    R = float(r["R"].sum())
    print(f"freeze_attn={fa} R={R:.6f} cut resid/R={(r['cons_resid']/R).numpy().round(8)} "
          f"sum-rule resid={r['sum_resid'].numpy().max():.2e} batched-vs-loop={max(float((r[k]-rl[k]).abs().max()) for k in ['head','neuron','comp']):.2e}")
    if fa:
        assert (r["cons_resid"].abs() / abs(R)).max() < 1e-9 and r["sum_resid"].abs().max() < 1e-9
# out-layer revenue equals R exactly, family split sums to total
assert abs(float(r["out"].sum()) - R) < 1e-9
print("books OK")
