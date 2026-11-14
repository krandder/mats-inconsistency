"""P9 networks.

DTR: continuous DT-recall (structure of P0/dtnet.py DTRecall, itself copied from aks2203/deep-thinking,
Schwarzschild/Bansal et al., MIT): projection conv, recall conv on cat(h, x) (no ReLU after it, as upstream
2D), BasicBlocks, conv head; no norms, no biases. nblocks and head widths are arguments so P9a can be ~20k
parameters at w=24 (upstream 2 blocks + [w,32,8,2] head is 37k).

BinNet (N1): the ONLY carried state is K binary straight-through channels.
    e = relu(embed(x));  a = conv3(relu(conv2(relu(conv1([e, z_{t-1}])))));  z_t = 1[a>0] + s(a) - sg(s(a))
    z_0 = 0; the head reads z_t.
Clamps: (mask, value, t_from) tensors; at absolute step t, z_t = value where mask & t >= t_from.
"""
import os

import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.conv1 = nn.Conv2d(w, w, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(w, w, 3, 1, 1, bias=False)

    def forward(self, x):
        return F.relu(self.conv2(F.relu(self.conv1(x))) + x)


class DTR(nn.Module):
    def __init__(self, cin, w=24, nblocks=1, head=(16, 8)):
        super().__init__()
        self.cin, self.w = cin, w
        self.projection = nn.Sequential(nn.Conv2d(cin, w, 3, 1, 1, bias=False), nn.ReLU())
        self.recall = nn.Conv2d(w + cin, w, 3, 1, 1, bias=False)
        self.blocks = nn.Sequential(*[BasicBlock(w) for _ in range(nblocks)])
        ch = [w, *head, 2]
        hl = []
        for i in range(len(ch) - 1):
            hl.append(nn.Conv2d(ch[i], ch[i + 1], 3, 1, 1, bias=False))
            if i < len(ch) - 2:
                hl.append(nn.ReLU())
        self.head = nn.Sequential(*hl)

    def init(self, x):
        return self.projection(x)

    def step(self, h, x):
        return self.blocks(self.recall(torch.cat([h, x], 1)))

    def readout(self, h):
        o = self.head(h)
        return o[:, 1] - o[:, 0]  # binary logit (B,n,n)


BIN_VARIANT = os.environ.get("P9_BIN", "rms")  # gate retune 2026-09-29: rms (pilot "sig" collapsed)
GRAD_CLIP = float(os.environ.get("P9_CLIP", "0"))


class BinNet(nn.Module):
    """N1 backbone. hybrid=True: continuous h (DT-recall style) plus K binary event channels (gate fallback)."""

    def __init__(self, cin, w=24, K=8, head_w=24, variant=None):
        super().__init__()
        self.cin, self.w, self.K = cin, w, K
        # gate retune (spec: retuning allowed only at the gate). "sig": sigmoid straight-through (pilot, collapsed:
        # |a|~70, dead units). "rms": per-cell RMS-normalise a over the K channels before the threshold, so the
        # surrogate stays in its live range (sign, hence the forward binary state, is scale-invariant).
        self.variant = variant or BIN_VARIANT
        self.embed = nn.Conv2d(cin, w, 3, 1, 1)
        self.conv1 = nn.Conv2d(w + K, w, 3, 1, 1)
        self.conv2 = nn.Conv2d(w, w, 3, 1, 1)
        self.conv3 = nn.Conv2d(w, K, 3, 1, 1)
        self.head1 = nn.Conv2d(K, head_w, 3, 1, 1)
        self.head2 = nn.Conv2d(head_w, 1, 1)

    def init(self, x):
        e = F.relu(self.embed(x))
        B, _, n, m = x.shape
        return e, x.new_zeros(B, self.K, n, m)

    def pre(self, z, e):
        return self.conv3(F.relu(self.conv2(F.relu(self.conv1(torch.cat([e, z], 1))))))

    @staticmethod
    def st(a):
        s = torch.sigmoid(a)
        return (a > 0).float() + s - s.detach()

    def norm(self, a):
        if "rms" in self.variant:
            a = a * torch.rsqrt(a.pow(2).mean(1, keepdim=True) + 1e-4) * 2.0
        return a

    def step(self, z, e):
        return self.st(self.norm(self.pre(z, e)))

    def readout(self, z):
        return self.head2(F.relu(self.head1(z)))[:, 0]


def n_params(m):
    return sum(p.numel() for p in m.parameters())
