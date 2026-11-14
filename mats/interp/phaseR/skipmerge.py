"""Skip-merge check (PREREG-R amendment 1, S5). usage: python skipmerge.py d act g bias -> results/skipmerge_<net>.json

Two-layer block h_1 -> h_3 (layers 2 and 3 of a 3-layer net, so the block's input is non-Gaussian):
real kernel h_3 = f_3(f_2(h_1) + s e_2) + s e_3; merged Gaussian N(f_3(f_2(h_1)), s^2 (I + J J^T)), J = df_3 at f_2(h_1);
naive merge N(f_3(f_2(h_1)), s^2 I). For M outer h_1 from real paths, K inner samples give the moment-matched Gaussian of the
real kernel; KL(N(m_hat, C_hat) || merged) is a lower bound on KL(real kernel || merged) (Pythagoras for Gaussian Q).
The estimation floor is measured by the same statistic on K samples drawn from the merged Gaussian itself.
"""
import json, math, os, sys
import torch
import rnet as R

torch.set_num_threads(1)
DT = torch.float64
LOG2 = math.log(2)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def gkl(m0, C0, m1, C1):
    """KL(N(m0, C0) || N(m1, C1)) in nats (batched over the first dim)."""
    d = m0.shape[-1]
    L1 = torch.linalg.cholesky(C1)
    Sol = torch.cholesky_solve(C0, L1)
    tr = Sol.diagonal(dim1=-2, dim2=-1).sum(-1)
    dm = (m1 - m0)[..., None]
    quad = (dm.transpose(-1, -2) @ torch.cholesky_solve(dm, L1))[..., 0, 0]
    ld1 = 2 * torch.log(L1.diagonal(dim1=-2, dim2=-1)).sum(-1); ld0 = torch.linalg.slogdet(C0)[1]
    return 0.5 * (tr + quad - d + ld1 - ld0), 0.5 * quad


def run(d, act, g, bias, sigma=0.3, M=96, K=None, seed=0):
    K = K or (2048 if d <= 16 else 8192)
    net = R.RNet(d, 3, act, g, sigma, seed, bias=bias)
    gen = torch.Generator().manual_seed(77 + seed)
    x = torch.randn(M, d, generator=gen, dtype=DT)
    h1 = net.f(1, x) + sigma * torch.randn(M, d, generator=gen, dtype=DT)
    out = dict(d=d, act=act, g=g, bias=bias, sigma=sigma, M=M, K=K, rows=[])
    I = torch.eye(d, dtype=DT)
    kl_m, kl_n, kl_floor, mean_term, cov_ratio = [], [], [], [], []
    for i in range(M):
        hi = h1[i:i + 1].expand(K, d)
        h2 = net.f(2, hi) + sigma * torch.randn(K, d, generator=gen, dtype=DT)
        h3 = net.f(3, h2) + sigma * torch.randn(K, d, generator=gen, dtype=DT)
        mh = h3.mean(0); Ch = torch.cov(h3.T)
        u = net.f(2, h1[i:i + 1]); Fm = net.f(3, u)[0]; J = net.jac(3, u)[0]
        Cm = sigma ** 2 * (I + J @ J.T)
        k1, q1 = gkl(mh, Ch, Fm, Cm); k0, _ = gkl(mh, Ch, Fm, sigma ** 2 * I)
        # floor: same statistic for K samples from the merged Gaussian itself
        Lm = torch.linalg.cholesky(Cm); Y = Fm + torch.randn(K, d, generator=gen, dtype=DT) @ Lm.T
        kf, _ = gkl(Y.mean(0), torch.cov(Y.T), Fm, Cm)
        kl_m.append(float(k1)); kl_n.append(float(k0)); kl_floor.append(float(kf)); mean_term.append(float(q1))
        cov_ratio.append(float(Ch.trace() / Cm.trace()))
    t = torch.tensor
    out.update(kl_merged_bits=float(t(kl_m).mean()) / LOG2, kl_naive_bits=float(t(kl_n).mean()) / LOG2,
               floor_bits=float(t(kl_floor).mean()) / LOG2, mean_term_bits=float(t(mean_term).mean()) / LOG2,
               trace_ratio=float(t(cov_ratio).mean()))
    json.dump(out, open(os.path.join(OUT, f'skipmerge_d{d}_{act}{g:g}_b{bias:g}.json'), 'w'), indent=1)
    print(f"d={d} {act} {g}: merged KL >= {out['kl_merged_bits']:.2f} bits (floor {out['floor_bits']:.2f}, mean part "
          f"{out['mean_term_bits']:.2f}), naive s2 I merge {out['kl_naive_bits']:.2f}, tr(C_real)/tr(C_merged) {out['trace_ratio']:.2f}",
          flush=True)


if __name__ == '__main__':
    a = sys.argv
    run(int(a[1]), a[2], float(a[3]), float(a[4]))
