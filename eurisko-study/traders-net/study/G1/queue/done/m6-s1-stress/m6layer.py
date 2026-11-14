"""M6 output market layer (worker 1 harness; siblings may import).

MarketNet(Q, W, arm, nu=0.1): S0 trunk (model.Net body, d=64, H=4, L=2) -> final LN residual y [B,Q,d];
market price l0 = per-question logistic (model.Logistic), detached inside the trader layer;
TraderLayer: n_heads trader heads per question; head i: g_i = relu(A_i y + c_i) (width hw, shared over
questions), d_{q,i} = r_{q,i}.g_i + a_{q,i}*l0 + b_{q,i}.

forward(win) -> dict(z=combined logit [B,Q], l0=market logit [B,Q] (with grad, for its own loss),
                     d=per-head corrections [B,Q,n_heads], y=residual).
Arms (combine()): 'A' z = sum d; 'B' z = l0 + sum d; 'BOOST' z = l0 + nu*sum d.
Siblings: subclass and override combine(l0_detached, d, out) for pools / bounded bets / fixed points.
Loss for training = BCE(z) + BCE(l0) (l0 detached in z, so the two losses train disjoint parameters).
"""
import math
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.normpath(os.path.join(HERE, "..", "..", "S0", "substrate"))
if SUB not in sys.path:
    sys.path.insert(0, SUB)
from model import Net, Logistic  # noqa: E402


def trunk(net, win):
    """model.Net forward up to the final LayerNorm (returns y [B,Q,d]); readout rw/rb unused."""
    B = win.shape[0]
    x = net.win(2.0 * win - 1.0) + net.qid.unsqueeze(0).expand(B, -1, -1)
    for blk in net.blocks:
        h = blk.ln1(x)
        q, k, v = blk.qkv(h).view(B, net.Q, 3, net.H, blk.dh).permute(2, 0, 3, 1, 4)
        A = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(blk.dh), dim=-1)
        x = x + blk.o((A @ v).transpose(1, 2).reshape(B, net.Q, net.d))
        x = x + blk.fc2(F.relu(blk.fc1(blk.ln2(x))))
    return net.lnf(x)


class TraderLayer(nn.Module):
    def __init__(self, Q, d, n_heads=16, hw=8, init=0.01):
        super().__init__()
        self.n, self.hw = n_heads, hw
        self.A = nn.Linear(d, n_heads * hw)
        self.r = nn.Parameter(torch.randn(Q, n_heads, hw) * init)
        self.a = nn.Parameter(torch.zeros(Q, n_heads))
        self.b = nn.Parameter(torch.zeros(Q, n_heads))

    def forward(self, y, l0d):
        B, Q, _ = y.shape
        g = F.relu(self.A(y)).view(B, Q, self.n, self.hw)
        return (g * self.r).sum(-1) + self.a * l0d.unsqueeze(-1) + self.b      # [B,Q,n]


class MarketNet(nn.Module):
    def __init__(self, Q, W, arm="A", nu=0.1, n_heads=16, d=64, H=4, L=2):
        super().__init__()
        self.arm, self.nu = arm, nu
        self.net = Net(Q, W, d=d, H=H, L=L)
        self.net.rw.requires_grad_(False); self.net.rb.requires_grad_(False)
        self.market = Logistic(Q, W)
        self.traders = TraderLayer(Q, d, n_heads)

    def net_params(self):
        return [p for n, p in self.named_parameters() if p.requires_grad and not n.startswith("market.")]

    def combine(self, l0d, d, out):
        s = d.sum(-1)
        if self.arm == "A":
            return s
        if self.arm == "B":
            return l0d + s
        if self.arm == "BOOST":
            return l0d + self.nu * s
        raise ValueError(self.arm)

    def forward(self, win):
        y = trunk(self.net, win)
        l0 = self.market(win)
        l0d = l0.detach()
        d = self.traders(y, l0d)
        out = {"l0": l0, "d": d, "y": y}
        out["z"] = self.combine(l0d, d, out)
        return out


def masked_bce(z, y, m):
    l = F.binary_cross_entropy_with_logits(z, y, reduction="none")
    return (l * m).sum() / m.sum().clamp_min(1.0)


def make_opts(model, arm_wd=0.0, lr=1e-3, lr_market=1e-2):
    if arm_wd > 0:
        o = torch.optim.AdamW(model.net_params(), lr, weight_decay=arm_wd)
    else:
        o = torch.optim.Adam(model.net_params(), lr)
    return o, torch.optim.Adam(model.market.parameters(), lr_market)
