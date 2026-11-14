import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import rnet as R, rare as RA

gen = torch.Generator().manual_seed(11)
net = R.RNet(16, 4, 'tanh', 0.75, 0.3, 0)
# brute force reference for moderate t
N = 1 << 21
lp_mc = {}
ys = []
for a in range(0, N, 1 << 16):
    x = torch.randn(1 << 16, 16, generator=gen, dtype=R.DT)
    h = x
    for l in range(1, 4):
        h = net.f(l, h) + 0.3 * torch.randn(h.shape, generator=gen, dtype=R.DT)
    ys.append(net.f(4, h) @ net.w)
mL = torch.cat(ys)
for t in (2.0, 3.0, 3.5):
    p = float(torch.special.ndtr(-(t - mL) / 0.3).mean())     # RB last layer
    print(f't={t}: brute force (RB) p = {p:.3e} = 2^{math.log2(p):.2f}', flush=True)
    for name in ('none', 'bulk', 'tail', 'iapf', 'ms'):
        t0 = time.time()
        if name == 'none':
            tw, fl = None, 0
        elif name == 'bulk':
            tw, fl = RA.twists_bulk(net, t)
        elif name in ('tail', 'ms'):
            xi, obj, fl = RA.instanton(net, t)
            if name == 'tail':
                tw, f2 = RA.twists_tail(net, t, xi); fl += f2
        elif name == 'iapf':
            tw, fl = RA.twists_iapf(net, t, 4096, gen, iters=4)
        ests = []
        for r in range(8):
            if name == 'ms':
                lp, f3 = RA.meanshift_is(net, t, xi, 4096, gen)
            else:
                lp, f3 = RA.smc(net, t, tw, 4096, gen)
            ests.append(lp)
        e = torch.tensor(ests).exp()
        print(f'   {name:5s}: mean {float(e.mean()):.3e} relsd {float(e.std() / e.mean()):.3f}  log2 mean {math.log2(float(e.mean())):.2f}'
              f'  setup {fl:.2e} FLOPs, {time.time() - t0:.1f}s', flush=True)
print('K2 tail:', [RA.k2_tail(net, t)[0] / math.log(2) for t in (2.0, 3.0, 3.5)])
