"""P18 models.

DTRecallK(width, K): DT-Recall 2D (dtnet.DTRecall, copied from P13, itself copied from aks2203/deep-thinking) with K
untied recurrent blocks and ONE shared projection and head. Step t (1-based) applies block i_t; i_t is chosen by the
trainer (random or round robin) and is round robin at test: i_t = (t - 1) mod K. Every block has DT-Recall's exact
shape (recall conv (w+3 -> w, no ReLU in 2D) + 2 BasicBlocks), so the FLOPs of one step equal DT-Recall's.

Construction order (projection, K blocks, head) matches DTRecall's, so with K = 1 and the same seed the initial weights
are bit-identical to DTRecall's (tests_cpu.py (a)); only the state-dict names differ (recur_blocks.0.* vs recur_block.*).
Parameter names contain "recur" for the recurrent blocks, so the trainer's lr/T throttle applies to all K blocks.
"""
import torch
from torch import nn

from dtnet import BasicBlock, n_params  # noqa: F401


class DTRecallK(nn.Module):
    def __init__(self, width, K=4, in_channels=3):
        super().__init__()
        self.width, self.K, self.in_channels, self.dim = width, K, in_channels, 2
        self.projection = nn.Sequential(nn.Conv2d(in_channels, width, 3, 1, 1, bias=False), nn.ReLU())
        blocks = []
        for _ in range(K):
            blocks.append(nn.Sequential(nn.Conv2d(width + in_channels, width, 3, 1, 1, bias=False),
                                        nn.Sequential(BasicBlock(2, width), BasicBlock(2, width))))
        self.recur_blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(nn.Conv2d(width, 32, 3, 1, 1, bias=False), nn.ReLU(),
                                  nn.Conv2d(32, 8, 3, 1, 1, bias=False), nn.ReLU(),
                                  nn.Conv2d(8, 2, 3, 1, 1, bias=False))

    def init(self, x):
        return self.projection(x)

    def step(self, h, x, i):
        return self.recur_blocks[i](torch.cat([h, x], 1))

    def readout(self, h):
        return self.head(h)

    def macs_per_pos(self):
        k, w, c = 9, self.width, self.in_channels
        step = (w + c) * w * k + 4 * w * w * k
        head = sum(m.weight.shape[0] * m.weight.shape[1] * k for m in self.head if isinstance(m, nn.Conv2d))
        return step, head


def dtr_to_k1_state(sd):
    """DTRecall state dict -> DTRecallK(K=1) state dict (renames recur_block.* to recur_blocks.0.*)."""
    return {k.replace("recur_block.", "recur_blocks.0."): v for k, v in sd.items()}
