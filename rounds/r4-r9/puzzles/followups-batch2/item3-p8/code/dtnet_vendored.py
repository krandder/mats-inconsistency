# Vendored verbatim from puzzles/P0/dtnet.py, sha256 fdcbcaa257589eb8c330a27c07d642e69d84d5b07df24596f05e5b62be5ca65b (read-only source; copied so the GPU VM can import it).
"""DT-Recall in 1D and 2D.

Copied (structure, module names and init) from aks2203/deep-thinking
deepthinking/models/dt_net_{1d,2d}.py and blocks.py (Schwarzschild, Borgnia,
Bansal, Emam, 2021; MIT licence), read on 29 Sep 2026. Changes: the forward
loop is split into init(x) -> h0, step(h, x) -> h' and readout(h) -> logits so
the P0 harness streams iterations and never builds all_outputs; the 1D input has
3 channels (one-hot 0, 1, PAD) instead of Bansal's 1 channel.

Quirks kept from upstream: the 2D recall conv has no ReLU after it, the 1D one
does; no norm layers; no biases.
"""
import torch
from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, dim, planes):
        super().__init__()
        Conv = nn.Conv2d if dim == 2 else nn.Conv1d
        self.conv1 = Conv(planes, planes, 3, 1, 1, bias=False)
        self.gn1 = nn.Sequential()
        self.conv2 = Conv(planes, planes, 3, 1, 1, bias=False)
        self.gn2 = nn.Sequential()
        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + x
        return F.relu(out)


class DTRecall(nn.Module):
    """dt_net_recall_{1d,2d}: projection conv, recall conv on cat(h, x), 2 BasicBlocks, 3-conv head."""

    def __init__(self, width, dim=2, in_channels=3):
        super().__init__()
        self.dim, self.width, self.in_channels = dim, width, in_channels
        Conv = nn.Conv2d if dim == 2 else nn.Conv1d
        self.projection = nn.Sequential(Conv(in_channels, width, 3, 1, 1, bias=False), nn.ReLU())
        rec = [Conv(width + in_channels, width, 3, 1, 1, bias=False)]
        if dim == 1:
            rec.append(nn.ReLU())
        rec.append(nn.Sequential(BasicBlock(dim, width), BasicBlock(dim, width)))
        self.recur_block = nn.Sequential(*rec)
        if dim == 2:
            chans = [width, 32, 8, 2]
        else:
            chans = [width, width, width // 2, 2]
        self.head = nn.Sequential(Conv(chans[0], chans[1], 3, 1, 1, bias=False), nn.ReLU(),
                                  Conv(chans[1], chans[2], 3, 1, 1, bias=False), nn.ReLU(),
                                  Conv(chans[2], chans[3], 3, 1, 1, bias=False))

    def init(self, x):
        return self.projection(x)

    def step(self, h, x):
        return self.recur_block(torch.cat([h, x], 1))

    def readout(self, h):
        return self.head(h)

    def forward(self, x, iters):
        """Upstream-equivalent forward: (last logits, last h)."""
        h = self.init(x)
        for _ in range(iters):
            h = self.step(h, x)
        return self.readout(h), h

    def macs_per_pos(self):
        """Multiply-accumulates per spatial position: one step and one readout."""
        k = 9 if self.dim == 2 else 3
        w, c = self.width, self.in_channels
        step = (w + c) * w * k + 4 * w * w * k
        head = sum(m.weight.shape[0] * m.weight.shape[1] * k for m in self.head if isinstance(m, (nn.Conv1d, nn.Conv2d)))
        return step, head


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def load_mref(path, device="cpu"):
    """Knutson et al. 2024 dt_net.pth (DT-Recall-2D w=128), strict load."""
    sd = torch.load(path, map_location=device, weights_only=True)["net"]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    m = DTRecall(128, dim=2, in_channels=3)
    m.load_state_dict(sd, strict=True)
    return m.to(device).eval()


def prefix_onehot(tokens, nin=3):
    """tokens (B, L) in {0,1,2=PAD} -> (B, 3, L) float one-hot."""
    return F.one_hot(tokens.long(), nin).float().transpose(1, 2).contiguous()
