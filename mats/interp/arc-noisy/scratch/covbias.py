"""Structure of the Gaussian-closure covariance bias at layer 8 (seed 100, sigma 0.001): closure (exact Gaussian
moments, mean corrected by particles) vs the true covariance from the 2^26-sample truth."""
import sys, torch, numpy as np
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
import anet, truthlib, rmc, gauss_relu as GR
torch.set_num_threads(2)
net = anet.Net(100, sigma=0.001); T = truthlib.load(100, 0.001); d = T['d']; N = T['N']
for Lc in (8, 16):
    m = d['S1'][Lc - 1] / N; C_true = d[f'M2_{Lc}'] / N - np.outer(m, m)
    # pure Gaussian closure (no particle corrections) and mean-corrected closure
    mu = torch.zeros(256, dtype=torch.float64); Sig = torch.eye(256, dtype=torch.float64)
    for l in range(Lc):
        W = net.W[l]; mz = W @ mu; Cz = W @ Sig @ W.T
        mu, Ca, _ = GR.relu_moments(mz, Cz)
        Sig = Ca + float(net.s[l]) ** 2 * torch.eye(256, dtype=torch.float64)
    C_cl = Ca.numpy(); mu_cl = mu.numpy()
    D = C_cl - C_true
    print(f"layer {Lc}: pure Gaussian closure mean nMSE {((mu_cl - m) ** 2).mean() / C_true.diagonal().mean():.3e}")
    print(f"   |C_true|_F {np.linalg.norm(C_true):.3e}  |C_cl - C_true|_F {np.linalg.norm(D):.3e}  diag part {np.linalg.norm(np.diag(np.diag(D))):.3e}")
    ev, V = np.linalg.eigh(C_true); V = V[:, ::-1]
    for k in (1, 2, 4, 8, 16, 32):
        P = V[:, :k]; Dk = P @ (P.T @ D @ P) @ P.T
        print(f"   bias energy in top-{k:2d} true modes block: {np.linalg.norm(Dk)**2 / np.linalg.norm(D)**2:.3f}")
    ed = np.linalg.eigvalsh(D); ed = ed[np.argsort(-np.abs(ed))]
    print("   top |eigs| of bias:", np.round(ed[:6], 5), " sum|eig| share top8:", np.abs(ed[:8]).sum() / np.abs(ed).sum())
    # effect on next-layer variances s^2 = diag(W D W^T) relative to true s^2
    W = net.W[Lc].numpy(); ds2 = np.einsum('ij,jk,ik->i', W, D, W); s2 = np.einsum('ij,jk,ik->i', W, C_true, W)
    print(f"   next-layer rel. variance error: median {np.median(np.abs(ds2 / s2)):.3e}  max {np.abs(ds2 / s2).max():.3e}")
