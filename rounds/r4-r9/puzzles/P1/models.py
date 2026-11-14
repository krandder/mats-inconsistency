"""Backbones behind a common stepper API:
  st = model.init_state(x); st, logit, h = model.step_once(st)
logit: (B, *S) Bernoulli logit per position; h: (B, C, *S) latent for head features.
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
MRES = os.path.join(HERE, "..", "..", "market-resnet")
sys.path.insert(0, MRES)
from mresnet import MarketResNet  # noqa: E402


class T2mNet(MarketResNet):
    """MarketResNet on float multi-channel input (5 ch: onehot x (3), marker, blocker)."""

    def __init__(self, w=64, nin=5):
        super().__init__(w=w, nin=nin)

    def encode(self, x):
        return self.embed(x)

    def init_state(self, x):
        B, _, L = x.shape
        e = self.encode(x)
        mask = torch.ones(B, L, device=x.device)
        return (torch.zeros(B, self.w, L, device=x.device), torch.zeros(B, L, device=x.device), e, mask)

    def step_once(self, st):
        h, b, e, mask = st
        h, b = self.step(h, b, e, mask)
        return (h, b, e, mask), b, h


class PrefixNet(MarketResNet):
    """The pilot backbone (int input, nin=3) with the stepper API (T1 smoke test)."""

    def init_state(self, x):
        B, L = x.shape
        e = self.encode(x)
        mask = torch.ones(B, L, device=x.device)
        return (torch.zeros(B, self.w, L, device=x.device), torch.zeros(B, L, device=x.device), e, mask)

    def step_once(self, st):
        h, b, e, mask = st
        h, b = self.step(h, b, e, mask)
        return (h, b, e, mask), b, h


def load_pilot(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = PrefixNet(w=ck["args"]["width"], nin=3)
    m.load_state_dict(ck["model"])
    return m.to(device).eval()


# ---------------------------------------------------------------- DT-Recall 2D (Bansal et al. 2022)
class BasicBlock2D(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.conv1 = nn.Conv2d(w, w, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(w, w, 3, padding=1, bias=False)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out) + x
        return F.relu(out)


class DTRecall2D(nn.Module):
    """dt_net_recall_2d(width): projection, recur_block = [conv(w+3 -> w), 2 BasicBlocks], head."""

    def __init__(self, width=128, in_channels=3):
        super().__init__()
        self.w = width
        self.projection = nn.Sequential(nn.Conv2d(in_channels, width, 3, padding=1, bias=False), nn.ReLU())
        self.recur_block = nn.Sequential(nn.Conv2d(width + in_channels, width, 3, padding=1, bias=False),
                                         nn.Sequential(BasicBlock2D(width), BasicBlock2D(width)))
        self.head = nn.Sequential(nn.Conv2d(width, 32, 3, padding=1, bias=False), nn.ReLU(),
                                  nn.Conv2d(32, 8, 3, padding=1, bias=False), nn.ReLU(),
                                  nn.Conv2d(8, 2, 3, padding=1, bias=False))

    def init_state(self, x):
        return (self.projection(x), x)

    def step_once(self, st):
        h, x = st
        h = self.recur_block(torch.cat([h, x], 1))
        out = self.head(h)
        return (h, x), out[:, 1] - out[:, 0], h


def load_dtrecall(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k.replace("module.", "", 1): v for k, v in ck["net"].items()}
    m = DTRecall2D(128)
    m.load_state_dict(sd)
    return m.to(device).eval()


def backbone_macs_per_pos(kind, w):
    if kind == "t2m" or kind == "prefix":
        return (2 * w + 1) * w * 3 + w * w * 3 + w * (w // 2) + w // 2
    if kind == "maze":
        return (w + 3) * w * 9 + 4 * w * w * 9 + w * 32 * 9 + 32 * 8 * 9 + 8 * 2 * 9
    raise ValueError(kind)
