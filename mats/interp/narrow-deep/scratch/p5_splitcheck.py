"""Sanity checks of lib/k2mix.py, run after the section-10 runs.
(1) k = 0 reproduces lib/k2.py's K2 (sorter and RNN).
(2) One split at layer 0 of the sorter (exactly Gaussian input) against MC (2^20 paths) for the moments of h_1,
    for the whole mixture and for each piece against MC conditioned on the gate."""
import sys, os
import torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import tnets, k2, k2mix

torch.set_num_threads(1)
DT = torch.float64

for kind, npz in (('mlp', 'train/gpu_out/mlp_res_L64_s0_sig0.0.npz'), ('rnn', 'train/gpu_out/rnn_T64_s0_sig0.0.npz'),
                  ('mlp', 'train/gpu_out/mlp_res_L64_s0_sig0.01.npz'), ('rnn', 'train/gpu_out/rnn_T64_s1_sig0.01.npz')):
    net = tnets.MLPNet(npz) if kind == 'mlp' else tnets.RNNNet(npz)
    if kind == 'mlp':
        mu, C, _ = k2.res_prop(net, torch.zeros(net.n, dtype=DT), torch.eye(net.n, dtype=DT), 0, net.L, True)
        comps, _ = k2mix.res_mix(net, {}, True)
    else:
        mu, C, _ = k2.rnn_prop(net, torch.zeros(net.d, dtype=DT), torch.zeros(net.d, net.d, dtype=DT), 0, net.T, True)
        comps, _ = k2mix.rnn_mix(net, {}, True)
    print(os.path.basename(npz), 'k=0 vs K2: max |dmu|', float((comps[0][1] - mu).abs().max()),
          'max |dC|', float((comps[0][2] - C).abs().max()), flush=True)

net = tnets.MLPNet('train/gpu_out/mlp_res_L64_s0_sig0.0.npz')
g = torch.Generator().manual_seed(5)
x = torch.randn(1 << 20, net.n, generator=g, dtype=DT)
z = x @ net.W[0].T
h1 = x + torch.relu(z) @ net.V[0].T
m_mc = h1.mean(0)
C_mc = torch.cov(h1.T)
se = (C_mc.diagonal() / h1.shape[0]).sqrt()


class One:
    pass


one = One()
one.n = net.n; one.L = 1; one.W = net.W[:1]; one.V = net.V[:1]; one.sigma = 0.0; one.s = net.s
k0, _ = k2mix.res_mix(one, {}, True)
for i in (0, 5, 11):
    k1, _ = k2mix.res_mix(one, {0: [i]}, True)
    mm = k2mix.mix_mean(k1)
    Wt = sum(w for w, _, _ in k1)
    Cm = sum(w * (Cc + torch.outer(m - mm, m - mm)) for w, m, Cc in k1) / Wt
    print(f"split at layer 0 unit {i}: weights {[round(w, 4) for w, _, _ in k1]} "
          f"(MC P(z_i > 0) {float((z[:, i] > 0).double().mean()):.4f}); "
          f"max mean error in MC SEs: K2 {float(((k0[0][1] - m_mc).abs() / se).max()):.2f}, "
          f"split {float(((mm - m_mc).abs() / se).max()):.2f}; "
          f"max cov error / max cov: K2 {float((k0[0][2] - C_mc).abs().max() / C_mc.abs().max()):.2e}, "
          f"split {float((Cm - C_mc).abs().max() / C_mc.abs().max()):.2e}", flush=True)
    on = z[:, i] > 0
    for (w, m, Cc), mask in zip(k1, (on, ~on)):
        hm = h1[mask].mean(0)
        sem = (h1[mask].var(0) / mask.sum()).sqrt()
        print(f"    piece P = {w:.4f}: mean error vs MC conditioned on the gate: max {float(((m - hm).abs() / sem).max()):.1f} SEs,"
              f" {float(((m - hm).abs() / h1[mask].std(0)).max()):.3f} sd", flush=True)
