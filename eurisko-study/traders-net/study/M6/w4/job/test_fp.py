"""Checks for fp_layer: FP solves F(l*)=l*, implicit gradient matches finite differences,
pass1/pass2 are F(l0), F(F(l0)). usage: python3 test_fp.py"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from fp_layer import MarketLayer
from model import Logistic
torch.set_num_threads(1); torch.manual_seed(0)
torch.set_default_dtype(torch.float64)
Q, W = 6, 8
mk = Logistic(Q, W)
with torch.no_grad():
    mk.w.normal_(0, 0.5)
ml = MarketLayer(Q, W, 16, 1, "fp", mk)
with torch.no_grad():
    for p in (ml.ub, ml.uk, ml.M):
        p.normal_(0, 0.4)
win = torch.rand(3, Q, W).round()
l = ml(win)
from fp_layer import trunk
h = trunk(ml.net, win); bh, kh = ml._bk(h); l0 = mk(win)
f = ml.F(l, l0, bh, kh)[0]
print("fp residual", float((f - l).abs().max()), "iters", ml.stats["iters"])
assert (f - l).abs().max() < 1e-6
# implicit grad vs finite differences on a scalar loss wrt ml.M and ml.cb
w = torch.randn_like(l)
loss = (l * w).sum(); loss.backward()
for name in ("M", "cb", "uk"):
    P = getattr(ml, name); g = P.grad.clone()
    idx = tuple(torch.randint(0, s, (1,)).item() for s in P.shape)
    eps = 1e-6
    with torch.no_grad():
        P[idx] += eps; lp = (ml(win) * w).sum(); P[idx] -= 2 * eps; lm_ = (ml(win) * w).sum(); P[idx] += eps
    fd = (lp - lm_) / (2 * eps)
    print(name, idx, "implicit", float(g[idx]), "fd", float(fd))
    assert abs(float(g[idx]) - float(fd)) < 1e-4 * max(1, abs(float(fd)))
ml.arm = "pass1"; l1 = ml(win); assert torch.allclose(l1, ml.F(l0, l0, bh, kh)[0])
ml.arm = "pass2"; l2 = ml(win); assert torch.allclose(l2, ml.F(l1, l0, bh, kh)[0])
print("PASS")
