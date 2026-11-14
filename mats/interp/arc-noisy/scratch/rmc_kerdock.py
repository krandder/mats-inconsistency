"""RMC on Kerdock-MUB particles vs the plain Kerdock-MUB mean on the same particles (and vs iid MC)."""
import sys, time, torch, numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
import anet, truthlib, rmc, samplers
torch.set_num_threads(2)
seed, sigma, L, nb, reps, mode = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
net = anet.Net(seed, sigma=sigma); T = truthlib.load(seed, sigma)
B = np.load('/home/agent/workspaces/mats/work/arc-noisy/kerdock129.npy').astype(np.float64)
n = 256; N = nb * 2 * n
er, ek = [], []
for rep in range(reps):
    g = torch.Generator().manual_seed(500 + rep); rng = np.random.default_rng(900 + rep)
    Q = samplers.haar(n, g, 'cpu')
    xs = []
    for k in rng.permutation(129)[:nb]:
        U = torch.from_numpy(B[k]) @ Q.T                      # rows: Q b
        rad = torch.from_numpy(samplers.chi_radii(n, n, rng, 'strat'))
        X = U * rad[:, None]
        xs.append(torch.stack([X, -X], 1).reshape(-1, n))
    x = torch.cat(xs)
    t0 = time.time()
    est, plain, fl, info = rmc.rmc(net, L, N, g, x=x, keep=False, cov_corr=mode)
    er.append(truthlib.nmse(est, T, L)); ek.append(truthlib.nmse(plain, T, L))
    print(f"rep {rep}: RMC-{mode} on Kerdock {er[-1]:.3e}   Kerdock mean {ek[-1]:.3e}   (1/N {1/N:.1e}) [{time.time()-t0:.0f}s]", flush=True)
print(f"L={L} N={N} mode={mode}: RMC/Kerdock mean nMSE {np.mean(er):.3e}   Kerdock {np.mean(ek):.3e}   ratio {np.mean(ek)/np.mean(er):.2f}   "
      f"rmc flops {fl:.2e} = {fl / net.flops_path(L):.0f} paths")
