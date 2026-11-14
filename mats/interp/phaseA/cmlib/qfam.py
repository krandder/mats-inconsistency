"""Simple q families.

CondGaussian    q(h | z) = N(mu_z, Sigma_z) per discrete latent group z (or one group).
LinearGaussian  q(h | phi) = N(A phi + b, Sigma[_g]), ridge mean (lambda by grouped K-fold CV),
                residual covariance full / diag / iso / lowrank:r (PPCA), optionally per group (heteroscedastic).
Both expose mean_cov(...), sample(...), logpdf(...), params() (named arrays: means / ridge maps and Cholesky
factors, for description-length quantisation), n_params(), and subset sampling (knock-out / knock-in) by Gaussian conditioning.
"""
import numpy as np


def _fit_cov(R, cov_type, shrink=1e-6):
    d = R.shape[1]
    C = np.cov(R, rowvar=False, bias=False).reshape(d, d) if len(R) > 1 else np.eye(d)
    if cov_type == 'full':
        C = C + shrink * np.trace(C) / d * np.eye(d); k = d * (d + 1) // 2
    elif cov_type == 'diag':
        C = np.diag(np.diag(C)); k = d
    elif cov_type == 'iso':
        C = np.trace(C) / d * np.eye(d); k = 1
    elif cov_type.startswith('lowrank:'):
        r = int(cov_type.split(':')[1])
        ev, U = np.linalg.eigh(C); ev, U = ev[::-1], U[:, ::-1]
        s2 = ev[r:].mean() if r < d else 1e-8
        W = U[:, :r] * np.sqrt(np.maximum(ev[:r] - s2, 0))
        C = W @ W.T + s2 * np.eye(d); k = d * r + 1
    else:
        raise ValueError(cov_type)
    return C, k


class _GaussBase:
    def _chol(self, C):
        return np.linalg.cholesky(C)

    def logpdf(self, H, *cond):
        mu, covs, gi = self.mean_cov(*cond, n=len(H))
        out = np.empty(len(H))
        for k, C in covs.items():
            m = gi == k
            L = np.linalg.cholesky(C)
            z = np.linalg.solve(L, (H[m] - mu[m]).T)
            out[m] = -0.5 * (z * z).sum(0) - np.log(np.diag(L)).sum() - 0.5 * len(C) * np.log(2 * np.pi)
        return out

    def sample(self, *cond, rng=None, n=None):
        rng = np.random.default_rng() if rng is None else rng
        mu, covs, gi = self.mean_cov(*cond, n=n)
        out = mu.copy()
        for k, C in covs.items():
            m = gi == k
            out[m] += rng.standard_normal((m.sum(), len(C))) @ np.linalg.cholesky(C).T
        return out

    def subset_sample(self, H, A, *cond, rng=None, mode='knockout'):
        """Replace the coordinates A of real H by draws from q conditioned on the other coordinates and on cond.
        mode='knockout': h_A ~ q(h_A | h_-A, cond), rest real (structure in A that q cannot predict from the rest).
        mode='knockin' : h_-A ~ q(h_-A | h_A, cond), A real."""
        rng = np.random.default_rng() if rng is None else rng
        d = H.shape[1]
        A = np.asarray(A)
        if mode == 'knockin':
            A = np.setdiff1d(np.arange(d), A)
        Bc = np.setdiff1d(np.arange(d), A)
        mu, covs, gi = self.mean_cov(*cond, n=len(H))
        out = H.copy()
        for k, C in covs.items():
            m = gi == k
            if len(Bc) == 0:
                cm = mu[m][:, A]; S = C
            else:
                K = C[np.ix_(A, Bc)] @ np.linalg.inv(C[np.ix_(Bc, Bc)])
                cm = mu[m][:, A] + (H[m][:, Bc] - mu[m][:, Bc]) @ K.T
                S = C[np.ix_(A, A)] - K @ C[np.ix_(Bc, A)]
            S = 0.5 * (S + S.T) + 1e-12 * np.eye(len(A))
            out[np.ix_(np.flatnonzero(m), A)] = cm + rng.standard_normal((m.sum(), len(A))) @ np.linalg.cholesky(S).T
        return out


class CondGaussian(_GaussBase):
    """q(h | z) = N(mu_z, Sigma_z). groups=None fits a single Gaussian."""

    def __init__(self, cov_type='full', shrink=1e-6):
        self.cov_type, self.shrink = cov_type, shrink

    def fit(self, H, groups=None):
        g = np.zeros(len(H), int) if groups is None else np.asarray(groups)
        self.keys = np.unique(g)
        self.mu, self.cov, self._k = {}, {}, 0
        for k in self.keys:
            Hk = H[g == k]
            self.mu[k] = Hk.mean(0)
            self.cov[k], kc = _fit_cov(Hk, self.cov_type, self.shrink)
            self._k += H.shape[1] + kc
        return self

    def mean_cov(self, groups=None, n=None):
        g = np.zeros(n, int) if groups is None else np.asarray(groups)
        mu = np.stack([self.mu[k] for k in g]) if len(g) else np.zeros((0, len(self.mu[self.keys[0]])))
        return mu, {k: self.cov[k] for k in self.keys if np.any(g == k)}, g

    def n_params(self):
        return self._k

    def params(self):
        out = {}
        for k in self.keys:
            out[f'mu_{k}'] = self.mu[k]; out[f'chol_{k}'] = np.linalg.cholesky(self.cov[k])
        return out

    def set_params(self, P):
        for k in self.keys:
            self.mu[k] = P[f'mu_{k}']; L = np.tril(P[f'chol_{k}']); self.cov[k] = L @ L.T + 1e-12 * np.eye(len(L))


class LinearGaussian(_GaussBase):
    """q(h | phi) = N(phi @ A + b, Sigma_g). Ridge lambda chosen by K-fold CV (grouped by cv_groups if given)."""

    def __init__(self, cov_type='full', lambdas=(1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100), folds=5, shrink=1e-6):
        self.cov_type, self.lambdas, self.folds, self.shrink = cov_type, lambdas, folds, shrink

    @staticmethod
    def _ridge(X, Y, lam):
        Xa = np.concatenate([X, np.ones((len(X), 1))], 1)
        R = lam * np.eye(Xa.shape[1]); R[-1, -1] = 0
        return np.linalg.solve(Xa.T @ Xa + R, Xa.T @ Y)

    def fit(self, H, Phi, cov_groups=None, cv_groups=None, seed=0):
        Phi = np.asarray(Phi, float)
        self.pm, self.ps = Phi.mean(0), Phi.std(0) + 1e-12
        X = (Phi - self.pm) / self.ps
        rng = np.random.default_rng(seed)
        if cv_groups is None:
            fold = rng.integers(0, self.folds, len(H))
        else:
            u, inv = np.unique(cv_groups, return_inverse=True)
            fold = rng.integers(0, self.folds, len(u))[inv]
        best = None
        for lam in self.lambdas:
            err = 0.0
            for f in range(self.folds):
                tr = fold != f
                Bm = self._ridge(X[tr], H[tr], lam)
                Xa = np.concatenate([X[~tr], np.ones(((~tr).sum(), 1))], 1)
                err += ((H[~tr] - Xa @ Bm) ** 2).sum()
            if best is None or err < best[0]:
                best = (err, lam)
        self.lam = best[1]
        self.B = self._ridge(X, H, self.lam)
        R = H - self._mean(Phi)
        g = np.zeros(len(H), int) if cov_groups is None else np.asarray(cov_groups)
        self.keys = np.unique(g); self.cov = {}; kc_tot = 0
        for k in self.keys:
            self.cov[k], kc = _fit_cov(R[g == k], self.cov_type, self.shrink); kc_tot += kc
        self._k = self.B.size + kc_tot
        return self

    def _mean(self, Phi):
        X = (np.asarray(Phi, float) - self.pm) / self.ps
        return np.concatenate([X, np.ones((len(X), 1))], 1) @ self.B

    def mean_cov(self, Phi, cov_groups=None, n=None):
        g = np.zeros(len(Phi), int) if cov_groups is None else np.asarray(cov_groups)
        return self._mean(Phi), {k: self.cov[k] for k in self.keys if np.any(g == k)}, g

    def n_params(self):
        return self._k

    def params(self):
        out = {'B': self.B}
        for k in self.keys:
            out[f'chol_{k}'] = np.linalg.cholesky(self.cov[k])
        return out

    def set_params(self, P):
        self.B = P['B']
        for k in self.keys:
            L = np.tril(P[f'chol_{k}']); self.cov[k] = L @ L.T + 1e-12 * np.eye(len(L))


def gaussian_kl(mu0, C0, mu1, C1):
    """KL(N0 || N1) in nats."""
    d = len(mu0)
    C1i = np.linalg.inv(C1)
    dm = mu1 - mu0
    return 0.5 * (np.trace(C1i @ C0) + dm @ C1i @ dm - d + np.linalg.slogdet(C1)[1] - np.linalg.slogdet(C0)[1])
