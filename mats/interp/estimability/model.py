"""Small standard pre-norm transformer for the shrunken source world (no positional encoding, set-style).

Context tokens attend to all context tokens; the query token attends to the context and itself; nothing attends
to the query token. Output from the query token: Gaussian mean mu and log-std logs (no clamp); loss = NLL.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import world_small as W

LN_EPS = 1e-5


def allow_mask(T=W.T):
    m = torch.zeros(T, T, dtype=torch.bool)
    m[:T - 1, :T - 1] = True
    m[T - 1, :] = True
    return m


class Block(nn.Module):
    def __init__(self, d, h, mlp=4):
        super().__init__()
        self.h = h
        self.ln1 = nn.LayerNorm(d, eps=LN_EPS); self.ln2 = nn.LayerNorm(d, eps=LN_EPS)
        self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.fc1 = nn.Linear(d, mlp * d); self.fc2 = nn.Linear(mlp * d, d)

    def attn(self, y, allow):
        B, T, d = y.shape
        q, k, v = self.qkv(y).split(d, -1)
        sp = lambda t: t.view(B, T, self.h, d // self.h).transpose(1, 2)
        q, k, v = sp(q), sp(k), sp(v)
        s = q @ k.transpose(-1, -2) / math.sqrt(d // self.h)
        s = s.masked_fill(~allow, float('-inf'))
        a = s.softmax(-1)
        return self.o((a @ v).transpose(1, 2).reshape(B, T, d)), a

    def forward(self, x, allow, want=False):
        att, a = self.attn(self.ln1(x), allow)
        x1 = x + att
        x2 = x1 + self.fc2(Fn.gelu(self.fc1(self.ln2(x1))))
        return x1, x2, (a if want else None)


class Net(nn.Module):
    def __init__(self, depth, d=32, h=2, mlp=4, fin=W.F):
        super().__init__()
        self.d, self.h, self.depth = d, h, depth
        self.inp = nn.Linear(fin, d)
        self.blocks = nn.ModuleList([Block(d, h, mlp) for _ in range(depth)])
        self.lnf = nn.LayerNorm(d, eps=LN_EPS); self.head = nn.Linear(d, 2)
        self.register_buffer('allow', allow_mask())

    def forward(self, X, want=False, upto=None):
        """Returns mu, logs, list of residual streams [x0, after attn 1, after mlp 1, ...], attention maps."""
        x = self.inp(X)
        res, attns = [x], []
        nb = len(self.blocks) if upto is None else upto
        for blk in self.blocks[:nb]:
            x1, x, a = blk(x, self.allow, want)
            res += [x1, x]; attns.append(a)
        y = self.head(self.lnf(x[:, -1]))
        return y[:, 0], y[:, 1], res, attns


def nll(mu, logs, y):
    return logs + 0.5 * math.log(2 * math.pi) + 0.5 * ((y - mu) * torch.exp(-logs)) ** 2
