import sys, os, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import rnet as R, fam as F, kq as KQ
gen = torch.Generator().manual_seed(3)

def node_logkq(net, l, q, H, N, chunk=1 << 15):
    s2 = net.sigma ** 2; d = net.d; acc = []
    for a in range(0, N, chunk):
        C = net.f(l, q.sample(chunk, gen))
        D2 = (H * H).sum(1)[:, None] + (C * C).sum(1)[None] - 2 * H @ C.T
        acc.append(torch.logsumexp(-0.5 * D2 / s2, 1))
    return torch.logsumexp(torch.stack(acc, 1), 1) - math.log(N) - 0.5 * d * (math.log(2 * math.pi * s2))

if __name__ == "__main__":
  for act, g, d, sig in (('tanh', 0.75, 4, 0.5), ('tanh', 0.75, 8, 0.3), ('tanh', 3.0, 8, 1.0), ('relu', 1.414, 8, 0.5)):
      net = R.RNet(d, 3, act, g, sig, 0)
      X = torch.randn(1 << 15, d, generator=gen, dtype=R.DT)
      H1 = net.f(1, X) + sig * torch.randn(1 << 15, d, generator=gen, dtype=R.DT)
      for qname, q in (('G', F.Gauss.fit(H1)), ('GC', F.GC.fit_from_s(net.f(1, X), H1, sig))):
          Hp = net.f(2, q.sample(32, gen)) + sig * torch.randn(32, d, generator=gen, dtype=R.DT)
          Hq = q.sample(32, gen)
          for nm, H in (('Kq-pts', Hp), ('q-pts', Hq)):
              t = time.time(); li, ns = KQ.log_kq_smc(net, 2, q, H, gen, Np=256); ti = time.time() - t
              ln = node_logkq(net, 2, q, H, 1 << 21)
              e = li - ln
              print(f'{act} g={g} d={d} s={sig} {qname} {nm}: SMC-node mean {float(e.mean()):+.3f} sd {float(e.std()):.3f} '
                    f'maxabs {float(e.abs().max()):.3f} nats; steps median {float(ns.median()):.0f}; {ti:.1f}s', flush=True)
