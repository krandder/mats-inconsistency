import sys, numpy as np, torch
sys.path.insert(0, '/home/agent/workspaces/mats/work/arc-noisy')
import gauss_relu as GR
from scipy.stats import multivariate_normal
rng = np.random.default_rng(0)
# 1) bvnu vs scipy
errs = []
for _ in range(400):
    h, k = rng.normal(size=2) * 2; r = rng.uniform(-0.999, 0.999) if rng.random() < 0.7 else rng.choice([-1, 1]) * rng.uniform(0.9, 0.99999)
    ref = multivariate_normal(mean=[0, 0], cov=[[1, r], [r, 1]]).cdf([-h, -k])   # P(X<-h, Y<-k) = P(X>h,Y>k) by symmetry
    got = float(GR.bvnu(torch.tensor(h, dtype=torch.float64), torch.tensor(k, dtype=torch.float64), torch.tensor(r, dtype=torch.float64)))
    errs.append(abs(ref - got))
print('bvnu max abs err vs scipy:', max(errs))
# 2) relu_moments vs MC on a random Gaussian with strong correlations
n = 12
A = rng.normal(size=(n, 3)); C = A @ A.T + 0.05 * np.eye(n); m = rng.normal(size=n) * 1.2
z = rng.multivariate_normal(m, C, size=4_000_000); a = np.maximum(z, 0)
mu, cov, _ = GR.relu_moments(torch.tensor(m), torch.tensor(C))
se_mu = a.std(0) / np.sqrt(len(a))
print('mean max |err|/se:', np.max(np.abs(mu.numpy() - a.mean(0)) / se_mu))
cv = np.cov(a.T)
print('cov max abs err:', np.max(np.abs(cov.numpy() - cv)), ' (MC se ~', np.sqrt(np.mean(cv.diagonal())**2 / len(a)) , ')')
ac = a - a.mean(0)
se = np.zeros((n, n))
for i in range(n):
    se[i] = (ac[:, i:i+1] * ac).std(0) / np.sqrt(len(a))
zsc = np.abs(cov.numpy() - cv) / se
print('cov max |err|/se:', zsc.max(), ' mean |err|/se:', zsc.mean())
