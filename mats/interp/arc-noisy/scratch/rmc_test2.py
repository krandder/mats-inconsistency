import sys, time, torch, numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
import anet, truthlib, rmc
torch.set_num_threads(2)
seed, sigma, L, logN, reps, mode = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]; K = int(sys.argv[7]) if len(sys.argv) > 7 else 8
net = anet.Net(seed, sigma=sigma); T = truthlib.load(seed, sigma); N = 1 << logN
er = []; em = []
for rep in range(reps):
    g = torch.Generator().manual_seed(1000 + rep)
    est, plain, fl, info = rmc.rmc(net, L, N, g, keep=True, cov_corr=mode, k_modes=K)
    er.append([truthlib.nmse(est[l - 1], T, l, debias=False) for l in range(1, L + 1)]); em.append(truthlib.nmse(plain, T, L, debias=False))
er = np.array(er); em = np.array(em)
print(f"{mode:6s} k={K} L={L} N=2^{logN}: RMC {er[:, -1].mean():.3e}  MC(same particles) {em.mean():.3e}  ratio {em.mean() / er[:, -1].mean():.1f}  flops {fl:.2e} ({fl / net.flops_path(L):.0f} paths)")
print("   by layer:", ' '.join(f"{v:.1e}" for v in er.mean(0)), flush=True)
