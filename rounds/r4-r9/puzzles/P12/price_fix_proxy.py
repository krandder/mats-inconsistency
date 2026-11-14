"""Check after the 2026-09-29 price fix: does decode(u + delta) match decode(theta + delta) on the cpu_phase proxy
for the chosen BP variant? Output: results/price_fix_proxy.txt."""
import sys, torch
sys.path.insert(0, '.')
import maze, flow
torch.set_num_threads(4)
import cpu_phase as cp
pool = maze.load('data/pool_W9.npz')
o, pa, S, G, cost, _ = cp.tens(pool, slice(0, 64))
for sigma in (2.0, 3.0):
    for (mode, c, tau, damp) in (("logprob", 0.3, 0.3, 0.5), ("logodds", 0.0, 1.0, 0.5)):
        g = torch.Generator().manual_seed(0); B, n = o.shape[:2]; y = pa[..., :2].float()
        uE = 3*(2*y[...,0]-1)+sigma*torch.randn(B,n,n,generator=g); uS = 3*(2*y[...,1]-1)+sigma*torch.randn(B,n,n,generator=g)
        u = flow.es_to_slots(uE, uS); bp = flow.FlowBP(o, S, G, tau=tau, damp=damp); st = bp.init_state(); th = flow.unary(u, mode, c)
        for _ in range(30): st = bp.step(st, th)
        dl = bp.delta(st).clamp(-30, 30)
        r = {k: round(float(flow.exact(flow.decode(v, o), pa).float().mean()), 3) for k, v in (("th+d", th + dl), ("u+d", u + dl), ("u", u))}
        print(sigma, mode, c, tau, damp, r, flush=True)
