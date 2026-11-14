"""DT-L (Bear, Pruegel-Bennett & Hare 2024, arXiv:2410.23451), 1D, our reimplementation.

No official code exists (Jay-Bear/rethinking-deep-thinking was empty on 29 Sep 2026), so
this follows the paper's description, approximately:
  - recurrence phi' = F(phi, x): recall = SN-conv(phi) + conv(x) (with bias), ELU, then two
    learnable convex residual blocks z <- (1 - g) z + g Block(z), g = sigmoid(per-channel
    parameter), Block = SN-conv, ELU, SN-conv, ELU;
  - spectral normalisation (power iteration on the weight flattened to Cout x Cin*k) on
    every conv acting on the latent inside the recurrence; the conv on x is not normalised;
  - ELU activations; BatchNorm only outside the recurrence (projection and head, not the last
    layer); biases on the recall, the recurrent block convs and the last layer.
Width is the largest w with at most 160,000 parameters (asserted).
"""
import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

MAX_PARAMS = 160_000


class ResBlockL(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.c1 = spectral_norm(nn.Conv1d(w, w, 3, 1, 1))
        self.c2 = spectral_norm(nn.Conv1d(w, w, 3, 1, 1))
        self.g = nn.Parameter(torch.zeros(w))  # sigmoid(0) = 0.5
        self.act = nn.ELU()

    def forward(self, z):
        g = torch.sigmoid(self.g).view(1, -1, 1)
        return (1 - g) * z + g * self.act(self.c2(self.act(self.c1(z))))


class DTL1D(nn.Module):
    def __init__(self, width, in_channels=3):
        super().__init__()
        w = self.width = width
        self.dim = 1
        self.projection = nn.Sequential(nn.Conv1d(in_channels, w, 3, 1, 1, bias=False), nn.BatchNorm1d(w), nn.ELU())
        self.recur_h = spectral_norm(nn.Conv1d(w, w, 3, 1, 1, bias=False))
        self.recur_x = nn.Conv1d(in_channels, w, 3, 1, 1, bias=True)
        self.recur_blocks = nn.Sequential(ResBlockL(w), ResBlockL(w))
        self.head = nn.Sequential(nn.Conv1d(w, w, 3, 1, 1, bias=False), nn.BatchNorm1d(w), nn.ELU(),
                                  nn.Conv1d(w, w // 2, 3, 1, 1, bias=False), nn.BatchNorm1d(w // 2), nn.ELU(),
                                  nn.Conv1d(w // 2, 2, 3, 1, 1, bias=True))
        self.act = nn.ELU()

    def init(self, x):
        return self.projection(x)

    def step(self, h, x):
        return self.recur_blocks(self.act(self.recur_h(h) + self.recur_x(x)))

    def readout(self, h):
        return self.head(h)

    def macs_per_pos(self):
        w, c = self.width, 3
        step = w * w * 3 + c * w * 3 + 4 * w * w * 3
        head = w * w * 3 + w * (w // 2) * 3 + (w // 2) * 2 * 3
        return step, head


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def pick_width():
    w = 64
    while n_params(DTL1D(w + 1)) <= MAX_PARAMS:
        w += 1
    return w


def make_dtl():
    w = pick_width()
    m = DTL1D(w)
    assert n_params(m) <= MAX_PARAMS
    return m
