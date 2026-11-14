"""P11 models: DT-Recall 2D (width 32) for the R and M arms, DT-L 2D (vendored P6 port).

R arms: f empty, logit_t = head(phi_t).
M arms (closed loop): G reads concat[phi, x, tanh(sg(b)/4), remaining/initial budget]; the head emits u_t;
b_t = sg(b_{t-1}) + Delta(u_t). BPTT runs through phi only; b and the float64 ledger are detached.
  M-B      Delta = u                                  (uncapped; budget channel = 1)
  M-budget Delta = LMSR bound(b, remaining W) * tanh(u)  (per-pixel worst-case-loss spending budget)
  M-TV     Delta = remaining V * tanh(u)               (per-pixel logit total-variation budget)
  M-cap    Delta = LMSR bound(b, eps) * tanh(u)        (per-step cap; budget channel = 1)
  M-clip   Delta = c * tanh(u)                         (per-step logit clip; budget channel = 1)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import caps
from vendored import DTL

R_ARMS = ("R-prog", "R-final", "DT-L")
M_ARMS = ("M-B", "M-budget", "M-TV", "M-cap", "M-clip")


class BasicBlock(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.conv1 = nn.Conv2d(w, w, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(w, w, 3, 1, 1, bias=False)

    def forward(self, x):
        return F.relu(self.conv2(F.relu(self.conv1(x))) + x)


class RecallDT(nn.Module):
    """dt_net_recall_2d (Bansal et al. 2022) at width w: projection conv + ReLU, recall conv on
    cat(h, x, f) with no activation, 2 BasicBlocks (5 convs per iteration), head w -> 16 -> 8 -> 1. No biases."""
    def __init__(self, w=32, nf=0):
        super().__init__()
        self.nf = nf
        self.proj = nn.Conv2d(3, w, 3, 1, 1, bias=False)
        self.recall = nn.Conv2d(w + 3 + nf, w, 3, 1, 1, bias=False)
        self.blocks = nn.Sequential(BasicBlock(w), BasicBlock(w))
        self.head = nn.Sequential(nn.Conv2d(w, 16, 3, 1, 1, bias=False), nn.ReLU(),
                                  nn.Conv2d(16, 8, 3, 1, 1, bias=False), nn.ReLU(),
                                  nn.Conv2d(8, 1, 3, 1, 1, bias=False))

    def init_state(self, x):
        return F.relu(self.proj(x))

    def step(self, h, x, f=None):
        z = torch.cat([h, x] if f is None else [h, x, f], 1)
        return self.blocks(self.recall(z))

    def out(self, h):
        return self.head(h)[:, 0]


class DTL2(DTL):
    """Vendored DT-L with the common step(h, x, f) signature (f unused)."""
    def step(self, h, x, f=None):
        return super().step(h, x)


def build(arm, w=32):
    if arm == "DT-L":
        return DTL2(w)
    return RecallDT(w, nf=2 if arm in M_ARMS else 0)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


class MState:
    """Belief b and float64 ledger for an M arm. hp: W (budget), V (TV), eps (cap), c (clip)."""
    def __init__(self, arm, hp, shape, dev):
        self.arm, self.hp = arm, hp
        self.b = torch.zeros(shape, device=dev)
        if arm == "M-budget":
            self.init = float(hp["W"])
        elif arm == "M-TV":
            self.init = float(hp["V"])
        else:
            self.init = None
        self.rem = None if self.init is None else torch.full(shape, self.init, device=dev, dtype=torch.float64)
        self.last_cost = None

    def feat(self):
        fb = torch.tanh(self.b.detach() / 4)
        fr = torch.ones_like(fb) if self.rem is None else (self.rem / self.init).float()
        return torch.stack([fb, fr], 1)

    def delta(self, u):
        a, b = self.arm, self.b.detach()
        if a == "M-B":
            return u
        if a == "M-budget":
            return caps.lmsr_delta(b, u, self.rem.float())
        if a == "M-TV":
            return caps.tv_delta(u, self.rem.float())
        if a == "M-cap":
            return caps.lmsr_delta(b, u, torch.full_like(b, float(self.hp["eps"])))
        if a == "M-clip":
            return caps.clip_delta(u, float(self.hp["c"]))
        raise ValueError(a)

    def advance(self, u):
        """-> new b (graph through u only); updates the ledger with the realised cost."""
        b_old = self.b.detach()
        b_new = b_old + self.delta(u)
        with torch.no_grad():
            if self.arm == "M-budget":
                self.rem, self.last_cost = caps.charge_lmsr(self.rem, b_old, b_new)
            elif self.arm == "M-TV":
                self.rem, self.last_cost = caps.charge_tv(self.rem, b_old, b_new)
        self.b = b_new
        return b_new
