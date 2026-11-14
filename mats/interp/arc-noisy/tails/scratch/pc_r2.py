import sys, torch, numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy/tails')
from tl import TNet
torch.set_num_threads(4)
for L in (8, 32):
    net = TNet(100, L, 0.001)
    g = torch.Generator().manual_seed(5)
    H = net.hidden(net.draw(1 << 15, g)).double()
    Z = H @ net.W[-1].double().T
    Hc = H - H.mean(0); Zc = Z - Z.mean(0)
    C = Hc.T @ Hc / H.shape[0]
    ev, V = torch.linalg.eigh(C); ev, V = ev.flip(0), V.flip(1)
    pr = float(ev.sum() ** 2 / (ev ** 2).sum())
    print(f'L={L}: PR of centered h_(L-1) {pr:.2f}; top eig shares {np.round((ev[:6] / ev.sum()).numpy(), 3)}')
    for k in (1, 2, 3, 5, 10, 20):
        P = Hc @ V[:, :k]
        beta = torch.linalg.lstsq(P, Zc).solution
        r2 = 1 - ((Zc - P @ beta) ** 2).mean(0) / (Zc ** 2).mean(0)
        print(f'   top-{k:2d} PCs: R2 of z_i median {float(r2.median()):.3f} [10% {float(r2.quantile(0.1)):.3f}, 90% {float(r2.quantile(0.9)):.3f}]')
