import sys, time, torch, numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
import anet, truthlib, rmc
torch.set_num_threads(2)
seed, sigma, L, logN, reps = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
net = anet.Net(seed, sigma=sigma)
T = truthlib.load(seed, sigma)
assert abs(T['chk'] - anet.weights_checksum(net.W)) < 1e-6
N = 1 << logN
res = []
for rep in range(reps):
    g = torch.Generator().manual_seed(1000 + rep)
    t0 = time.time()
    est, plain, fl, info = rmc.rmc(net, L, N, g, keep=True)
    e_rmc = [truthlib.nmse(est[l - 1], T, l, debias=False) for l in range(1, L + 1)]
    e_mc = truthlib.nmse(plain, T, L, debias=False)
    res.append((e_rmc, e_mc))
    print(f"rep {rep}: RMC nMSE@L {e_rmc[-1]:.3e}  plain MC same particles {e_mc:.3e}  (1/N = {1/N:.1e})  flops {fl:.3e} = {fl / net.flops_path(L):.0f} paths  [{time.time()-t0:.0f}s]", flush=True)
er = np.array([r[0] for r in res]); em = np.array([r[1] for r in res])
print(f"mean over reps: RMC {er[:, -1].mean():.3e}   MC {em.mean():.3e}   ratio MC/RMC {em.mean() / er[:, -1].mean():.1f}")
print("RMC nMSE by layer (mean over reps):", ' '.join(f"{v:.1e}" for v in er.mean(0)))
