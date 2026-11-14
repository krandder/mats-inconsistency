"""FLOPs of one training iteration per arm (batch 64, D = 64, 9x9), counted by torch's FlopCounterMode (forward +
backward, all passes: rollout, reverse sweep, IG points, TD targets). Writes <out> as JSON."""
import json
import sys

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

import mazes
import p5

out = sys.argv[1] if len(sys.argv) > 1 else "flops.json"
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ops = mazes.load_train()
res = {}
for arm in ["A1", "A1w", "A2", "A2t", "A3", "A4", "A5", "A6", "A7"]:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = p5.build(arm).to(dev)
    x, y, mask, dist, e = p5.to_dev(mazes.make_batch(ops, rng, 64), dev)
    ex = dict(phi=1.0, dense=False, ema=None, onpath=e[:, 0])
    st = dict(wabs_self=0.0, wabs_all=0.0)
    fc = FlopCounterMode(display=False)
    with fc:
        if arm in ("A1", "A1w"):
            p5.step_A1(model, x, y, mask, 64, 0.01, rng)
        elif arm == "A2":
            p5.step_A2(model, x, y, mask, 64, 0.01, rng)
        elif arm == "A2t":
            p5.step_A2t(model, x, y, mask, 64)
        elif arm == "A5":
            p5.step_A5(model, arm, x, y, mask, 64, ex, st)
        else:
            p5.step_online(model, arm, x, y, mask, 64, ex, st)
    res[arm] = dict(flops=int(fc.get_total_flops()), params=sum(p.numel() for p in model.parameters()))
    print(arm, res[arm], flush=True)
for arm in res:
    res[arm]["ratio_to_A1"] = res[arm]["flops"] / res["A1"]["flops"]
json.dump(res, open(out, "w"), indent=1)
