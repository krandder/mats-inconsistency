"""M6 lead: kill controls (wd, boost nu=0.1) in w2's harness. Imports w2/pool.py unchanged."""
import sys, os, types
sys.path.insert(0, "/home/agent/workspaces/eurisko-study/traders-net/study/M6/w2")
import torch
import pool

arm = sys.argv[sys.argv.index("--arm") + 1]
assert arm in ("wd", "boost")
_Market = pool.Market


class MarketKC(_Market):
    def __init__(self, Q, W, arm_, base, alpha=0.0):
        super().__init__(Q, W, "a", base, alpha)
        self.kc = arm_

    def forward(self, win):
        z = super().forward(win)
        if self.kc == "boost":
            l0 = self.base(win).detach()
            return l0 + 0.1 * z
        return z


pool.Market = MarketKC
if arm == "wd":
    _Adam = torch.optim.Adam
    calls = {"n": 0}

    def AdamFirst(params, *a, **k):
        calls["n"] += 1
        if calls["n"] == 1:   # the net optimiser; the second call is the market maker's
            return torch.optim.AdamW(params, *a, weight_decay=0.1, **k)
        return _Adam(params, *a, **k)
    pool.torch.optim.Adam = AdamFirst
if "--out" not in sys.argv:
    sys.argv += ["--out", "/home/agent/workspaces/eurisko-study/traders-net/study/M6/lead/results"]
pool.main()
