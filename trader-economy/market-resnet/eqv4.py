"""Gradient check: economy train_step (voc on/off) vs BPTT --tbptt 1 --train_pick random --ds --covsup
--vocsup --ptopt --bidsup (the replica arm), same weights, batch and trader sequence, wealth scale 1,
lr 0 and clipping off. Prints per-trader cosine and max relative difference."""
import sys, types, torch
sys.path.insert(0, "/home/agent/workspaces/calibrated-ai/trader-economy/market-resnet")
import econ2
from econ2 import Economy, BPTT
from mresnet import prefix_batch
torch.nn.utils.clip_grad_norm_ = lambda *a, **k: torch.tensor(0.0)
base = dict(N=4, width=32, lr=0.0, cost=0.0, W0=20.0, frac=0.5, wealth_cap=1, stream="cov", lam_c=1.0, lam_v=0.3,
            rent=1e-3, self_pay=0, train_halt=-1, settle="eps", eps=0.1, iters=10, prog=0, bounty="", clamp=0, D=12)
for voc in (1, 0):
    torch.manual_seed(0)
    E = Economy(types.SimpleNamespace(**base, mode="econ", voc=voc, train_pick=""), "cpu")
    E.wealth = [1e12] * 4
    Bm = BPTT(types.SimpleNamespace(**base, mode="bptt", voc=0, train_pick="random", tbptt=1, ds=1, covsup=1,
                                    vocsup=voc, ptopt=1, bidsup=1, alpha=1.0), "cpu")
    for i in range(4):
        Bm.traders[i].load_state_dict(E.traders[i][0].state_dict())
    g = torch.Generator().manual_seed(5); x, y, m = prefix_batch(8, 8, 20, g, "cpu")
    E.train_step(x, y, m, 12, torch.Generator().manual_seed(7))
    Bm.train_step(x, y, m, 12, torch.Generator().manual_seed(7))
    for i in range(4):
        ge = torch.cat([p.grad.flatten() for p in E.traders[i][0].parameters() if p.grad is not None])
        gb = torch.cat([p.grad.flatten() for p in Bm.traders[i].parameters() if p.grad is not None])
        if ge.numel() != gb.numel():
            print("voc", voc, "trader", i, "numel differ", ge.numel(), gb.numel()); continue
        cos = torch.nn.functional.cosine_similarity(ge, gb, 0).item()
        print(f"voc {voc} trader {i}: cos {cos:.7f} max abs diff {(ge - gb).abs().max().item():.2e} |g| {ge.norm().item():.3e}")
