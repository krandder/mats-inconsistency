"""Small transformer over Q question tokens, plus a per-question logistic baseline.

Each token is one question: embedding = Linear(window of W outcomes as +-1) + question-id
embedding. Pre-LN blocks with 4 heads and a ReLU MLP (4x). One logit per question from a
per-question readout vector. `forward(..., rec=dict)` records the intermediates the books need.

freeze_ln: LayerNorm denominators are detached (values unchanged). Needed for the books:
  LayerNorm is scale-invariant, so activation x gradient through it would be ~0.
freeze_attn: attention patterns are detached (values unchanged); used only for the exact
  conservation check. The real books run with live attention.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LN(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(d))
        self.b = nn.Parameter(torch.zeros(d))
        self.eps = eps

    def forward(self, x, freeze=False):
        xc = x - x.mean(-1, keepdim=True)
        s = torch.sqrt(xc.pow(2).mean(-1, keepdim=True) + self.eps)
        if freeze:
            s = s.detach()
        return xc / s * self.g + self.b


class Block(nn.Module):
    def __init__(self, d, H, mlp):
        super().__init__()
        self.H, self.dh = H, d // H
        self.ln1 = LN(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ln2 = LN(d)
        self.fc1 = nn.Linear(d, mlp * d)
        self.fc2 = nn.Linear(mlp * d, d)


class Net(nn.Module):
    def __init__(self, Q, W, d=64, H=4, L=2, mlp=4):
        super().__init__()
        self.Q, self.W, self.d, self.H, self.L = Q, W, d, H, L
        self.win = nn.Linear(W, d)
        self.qid = nn.Parameter(torch.randn(Q, d) * 0.1)
        self.blocks = nn.ModuleList([Block(d, H, mlp) for _ in range(L)])
        self.lnf = LN(d)
        self.rw = nn.Parameter(torch.randn(Q, d) / math.sqrt(d))
        self.rb = nn.Parameter(torch.zeros(Q))

    def forward(self, win, rec=None, freeze_ln=False, freeze_attn=False, mask=None):
        """mask: optional {'neuron': [L,N], 'head': [L,H]} multipliers (0 = ablated)."""
        B = win.shape[0]
        e_w = self.win(2.0 * win - 1.0)                     # [B,Q,d]
        e_q = self.qid.unsqueeze(0).expand(B, -1, -1)
        x = e_w + e_q
        if rec is not None:
            rec["e_w"], rec["e_q"], rec["cuts"] = e_w, e_q, [x]
            rec["attn_out"], rec["mlp_out"], rec["hv"], rec["act"] = [], [], [], []
        for li, blk in enumerate(self.blocks):
            h = blk.ln1(x, freeze_ln)
            q, k, v = blk.qkv(h).view(B, self.Q, 3, self.H, blk.dh).permute(2, 0, 3, 1, 4)
            A = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(blk.dh), dim=-1)
            if freeze_attn:
                A = A.detach()
            hv = A @ v                                        # [B,H,Q,dh]
            if mask is not None:
                hv = hv * mask['head'][li].view(1, -1, 1, 1)
            a_out = blk.o(hv.transpose(1, 2).reshape(B, self.Q, self.d))
            x = x + a_out
            if rec is not None:
                rec["hv"].append(hv); rec["attn_out"].append(a_out); rec["cuts"].append(x)
            act = F.relu(blk.fc1(blk.ln2(x, freeze_ln)))      # [B,Q,4d]
            if mask is not None:
                act = act * mask['neuron'][li]
            m_out = blk.fc2(act)
            x = x + m_out
            if rec is not None:
                rec["act"].append(act); rec["mlp_out"].append(m_out); rec["cuts"].append(x)
        y = self.lnf(x, freeze_ln)
        z = (y * self.rw).sum(-1) + self.rb                   # [B,Q]
        if rec is not None:
            rec["z"] = z
        return z

    def bias_params(self):
        """Constant-input parameters (biases, LN shifts) with the cut index they enter after.
        A bias 'enters after cut k' if it lies downstream of residual cut k."""
        out = [("win.bias", self.win.bias, -1)]
        for l, blk in enumerate(self.blocks):
            c_attn, c_mlp = 2 * l, 2 * l + 1
            out += [(f"L{l}.ln1.b", blk.ln1.b, c_attn), (f"L{l}.qkv.bias", blk.qkv.bias, c_attn),
                    (f"L{l}.o.bias", blk.o.bias, c_attn), (f"L{l}.ln2.b", blk.ln2.b, c_mlp),
                    (f"L{l}.fc1.bias", blk.fc1.bias, c_mlp), (f"L{l}.fc2.bias", blk.fc2.bias, c_mlp)]
        out += [("lnf.b", self.lnf.b, 2 * self.L), ("rb", self.rb, 2 * self.L)]
        return out


class Logistic(nn.Module):
    """Baseline: logit_q = w_q . (2*window_q - 1) + b_q, independent per question."""

    def __init__(self, Q, W):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(Q, W))
        self.b = nn.Parameter(torch.zeros(Q))

    def forward(self, win):
        return ((2.0 * win - 1.0) * self.w).sum(-1) + self.b
