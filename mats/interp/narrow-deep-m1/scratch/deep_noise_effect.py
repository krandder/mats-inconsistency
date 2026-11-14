"""Diagnostic: size of the deep part's noise effect, paired (no truth needed). States at c_d sampled from the noisy net
(2^16); compare per-state MC of the noisy deep part (layers c_d+1..L, R noise draws per state averaged) with the
noiseless deep output and with the K2-Stein table correction (bvf.Deep). nMSE of the mean differences / tr Var(a_L)."""
import sys, json, torch, math, time
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep/lib'); sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep-m1/lib')
import nd, bvf
torch.set_num_threads(1)
DT = torch.float64
SEED, SIG, c_d = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
net = nd.Net('plain', SEED, 8, 256, SIG); n, L = 8, 256
g = torch.Generator().manual_seed(13)
P = 1 << 16
h = torch.randn(P, n, generator=g, dtype=DT)
for l in range(c_d):
    a = torch.relu(h @ net.W[l].T); h = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[l]
deep = bvf.Deep(net, c_d, h[:256], torch.ones(256, dtype=DT), bvf.Flops())
noisy = torch.zeros(P, n, dtype=DT)
R = 4
for r in range(R):
    hh = h.clone()
    for l in range(c_d, L):
        a = torch.relu(hh @ net.W[l].T); hh = a + torch.randn(a.shape, generator=g, dtype=DT) * net.s[l] if l < L - 1 else a
    noisy += hh / R
trv = float(((noisy - noisy.mean(0)) ** 2).sum(1).mean())     # approx tr Var (noise-averaged; slight underestimate)
V0 = bvf.noiseless_pass(net, h, c_d, L, bvf.Flops())
V1 = deep.value(h, bvf.Flops())
e0 = float((V0.mean(0) - noisy.mean(0)).pow(2).sum() / trv)
e1 = float((V1.mean(0) - noisy.mean(0)).pow(2).sum() / trv)
se = float((noisy - V0).pow(2).sum(1).mean() / P / trv)
print(f"seed {SEED} sig {SIG:g} c_d={c_d}: deep noise effect (noiseless vs MC) nMSE {e0:.3g}; K2-table residual {e1:.3g}; "
      f"MC resolution ~{se:.2g}; ref class share {deep.class_share:.3f} ({deep.nclass} classes)", flush=True)
