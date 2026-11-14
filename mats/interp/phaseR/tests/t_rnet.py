import sys, time, math, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import rnet as R

torch.manual_seed(0)
# Jacobian check
net = R.RNet(8, 3, 'tanh', 0.75, 0.3, 0)
h = torch.randn(4, 8, dtype=R.DT)
J = net.jac(2, h)
Ja = torch.stack([torch.autograd.functional.jacobian(lambda v: net.f(2, v[None])[0], h[i]) for i in range(4)])
print('jac err', float((J - Ja).abs().max()))
# K2 vs MC moments of h_L
for act, g in (('tanh', 0.75), ('tanh', 3.0), ('relu', math.sqrt(2))):
    for d in (16, 64, 256):
        net = R.RNet(d, 4, act, g, 0.3, 0)
        gen = torch.Generator().manual_seed(1)
        t = time.time(); n = 1 << 17
        S1, S2, _ = net.mc_outputs(n, gen)
        tm = time.time() - t
        Ey = S1 / n; Vy = S2 / n - Ey ** 2
        t = time.time(); mu, S = R.k2_propagate(net); tk = time.time() - t
        mu0, S0 = R.k2_propagate(net, ln_corr=False)
        Ek = net.Wout @ mu; Ek0 = net.Wout @ mu0
        nm = float((((Ek - Ey) ** 2) / Vy).mean()); nm0 = float((((Ek0 - Ey) ** 2) / Vy).mean())
        print(f'{act} g={g:.2f} d={d}: normMSE K2 {nm:.2e} (no LN corr {nm0:.2e}); MC floor 1/n={1/n:.1e}; '
              f'MC {n} samples {tm:.1f}s ({net.flops_sample()*n/tm/1e9:.1f} GF/s), K2 {tk:.2f}s; '
              f'FLOP-equiv N of K2 = {net.flops_k2()/net.flops_sample():.0f}', flush=True)
