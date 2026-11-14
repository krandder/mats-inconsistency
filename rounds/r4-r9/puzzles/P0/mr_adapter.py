"""Adapters exposing init / step / readout for the P0 harness.

MRAdapter wraps MarketResNet's per-step update (read-only import of
market-resnet/mresnet.py): state = (h, b), x = (e, mask) with e = encode(tokens)
computed once; step = MarketResNet.step; readout = b (the belief logit).
DTAdapter wraps dtnet.DTRecall / dtl.DTL1D; readout = logit1 - logit0.
Every adapter returns z = l1 - l0 flattened to (B, positions); p(1) = sigmoid(z).
"""
import os, sys
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
MR_DIR = os.environ.get("P0_MR_DIR", os.path.join(HERE, "../../market-resnet"))
sys.path.insert(0, MR_DIR)
import mresnet  # noqa: E402


class DTAdapter:
    def __init__(self, net, task):
        self.net, self.task = net, task
        s, hd = net.macs_per_pos()
        self.flops_per_pos = 2 * (s + hd)

    def prep(self, batch):
        if self.task == "maze":
            return batch["x"]
        return F.one_hot(batch["tok"].long(), 3).float().transpose(1, 2).contiguous()

    def init(self, x):
        return self.net.init(x)

    def step(self, h, x):
        return self.net.step(h, x)

    def readout(self, h):
        o = self.net.readout(h)
        return (o[:, 1] - o[:, 0]).flatten(1)

    def latent(self, h):
        return h.flatten(2)

    def flop_probe(self, x):
        h = self.init(x)
        return lambda: self.readout(self.step(h, x))


class MRAdapter:
    def __init__(self, model):
        self.net = model
        w = model.w
        step = (2 * w + 1) * w * 3 + sum(c.weight.numel() for c in model.layers[1:])
        head = w * (w // 2) + (w // 2)
        self.flops_per_pos = 2 * (step + head)

    def prep(self, batch):
        tok = batch["tok"].long()
        return self.net.encode(tok), batch["mask"].float()

    def init(self, x):
        e, m = x
        B, _, L = e.shape
        return (torch.zeros(B, self.net.w, L, device=e.device), torch.zeros(B, L, device=e.device))

    def step(self, s, x):
        e, m = x
        return self.net.step(s[0], s[1], e, m)

    def readout(self, s):
        return s[1]

    def latent(self, s):
        return s[0]

    def flop_probe(self, x):
        s = self.init(x)
        s = self.step(s, x)
        return lambda: self.readout(self.step(s, x))


def load_mr(path, device):
    s = torch.load(path, map_location=device, weights_only=False)
    a = s["args"]
    m = mresnet.MarketResNet(w=a["width"]).to(device).eval()
    m.load_state_dict(s["model"])
    return m, a


class MRTrainWrapper(torch.nn.Module):
    """Lets mresnet.loss_fn('B') run verbatim on a DT backbone (arm P6):
    forward(x, mask, T) returns bs = [b_0..b_T] with b_t = readout(h_t) as l1 - l0."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.cf_pays = []

    def forward(self, x, mask, T, **kw):
        xo = F.one_hot(x.long(), 3).float().transpose(1, 2)
        h = self.net.init(xo)
        o = self.net.readout(h)
        bs = [(o[:, 1] - o[:, 0]) * mask]
        for _ in range(T):
            h = self.net.step(h, xo)
            o = self.net.readout(h)
            bs.append((o[:, 1] - o[:, 0]) * mask)
        return bs, [], [], []
