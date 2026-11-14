import sys, time, torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/narrow-deep/lib')
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
import nd
import kprop_noisy as KN
torch.set_num_threads(1)
net = nd.Net('plain', 103, 8, 256, 1e-3)
# sampled truth of E[a_c] for c = 1..8
g = torch.Generator().manual_seed(5)
x = torch.randn(1 << 20, 8, generator=g, dtype=torch.float64)
h = x; tm = []; tv = []
for l in range(8):
    a = torch.relu(h @ net.W[l].T); tm.append(a.mean(0)); tv.append(a.var(0).sum())
    h = a + torch.randn(a.shape, generator=g, dtype=torch.float64) * net.s[l]
for k, fac, kind in [(1, False, 'simple'), (2, False, 'simple'), (2, False, 'augment'), (3, True, 'simple'), (3, False, 'simple'), (3, True, 'augment'), (4, True, 'simple'), (4, False, 'simple')]:
    try:
        t0 = time.time()
        means, fl, sec = KN.kprop_noisy(net, 8, k, factor=fac, kind=kind, keep=True)
        errs = [float((means[c] - tm[c]).pow(2).sum() / tv[c]) for c in range(8)]
        print(f"K={k} factor={fac} kind={kind}: flops {fl:.3g} ({sec:.1f}s): nMSE of E[a_c], c=1..8: " + ' '.join(f'{e:.2g}' for e in errs), flush=True)
    except Exception as e:
        print(f"K={k} factor={fac} kind={kind}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
