"""Density-ratio estimation w = p/q by a calibrated probabilistic classifier (real = 1, synthetic = 0).

Classes are balanced by sample weights, so the calibrated logit estimates log p(h|z)/q(h|z) directly.
Splits: train (fit) / cal (model selection + Platt calibration) / test (all reported diagnostics).
Conditioning variables z (latents, labels, cell features) enter as extra inputs; the classifier then
estimates the conditional ratio w_z(h) provided real and synthetic draws share the same z distribution.
"""
import math
import numpy as np
import torch
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from .stats import auc, auc_se, ece, psis_khat, kish_ess_log, logsumexp

LOG2 = math.log(2.0)


# ----------------------------------------------------------------------------- feature maps
class QuadFeatures:
    """Whitened quadratic features [u, u_i u_j (i<=j)] with optional z interactions.
    interact: None (z additive only), 'linear' (z x u), 'full' (z x [u, uu])."""

    def __init__(self, interact='full'):
        self.interact = interact

    def fit(self, H, Z=None):
        mu = H.mean(0); C = np.cov(H, rowvar=False).reshape(H.shape[1], H.shape[1])
        C = C + 1e-6 * np.trace(C) / len(C) * np.eye(len(C))
        ev, U = np.linalg.eigh(C)
        self.mu, self.W = mu, U / np.sqrt(ev)
        d = H.shape[1]; self.iu = np.triu_indices(d)
        F = self._raw(H, Z)
        self.fm, self.fs = F.mean(0), F.std(0) + 1e-8
        return self

    def _raw(self, H, Z):
        u = (H - self.mu) @ self.W
        uu = (u[:, :, None] * u[:, None, :])[:, self.iu[0], self.iu[1]]
        base = np.concatenate([u, uu], 1)
        if Z is None:
            return base
        Z = np.asarray(Z, float).reshape(len(H), -1)
        parts = [base, Z]
        if self.interact == 'linear':
            parts.append((Z[:, :, None] * u[:, None, :]).reshape(len(H), -1))
        elif self.interact == 'full':
            parts.append((Z[:, :, None] * base[:, None, :]).reshape(len(H), -1))
        return np.concatenate(parts, 1)

    def __call__(self, H, Z=None):
        return (self._raw(H, Z) - self.fm) / self.fs


# ----------------------------------------------------------------------------- classifiers
class _QuadLogit:
    def __init__(self, Cs=(0.01, 0.1, 1.0, 10.0), interact='full', seed=0):
        self.Cs, self.interact, self.seed = Cs, interact, seed

    def fit(self, Xtr, ytr, swtr, Xca, yca, swca, Ztr=None, Zca=None):
        self.feat = QuadFeatures(self.interact).fit(Xtr, Ztr)
        Ftr, Fca = self.feat(Xtr, Ztr), self.feat(Xca, Zca)
        best = None
        for C in self.Cs:
            m = LogisticRegression(C=C, max_iter=3000, tol=1e-6)
            m.fit(Ftr, ytr, sample_weight=swtr)
            s = m.decision_function(Fca)
            bce = _wbce(s, yca, swca)
            if best is None or bce < best[0]:
                best = (bce, C, m)
        self.C, self.m = best[1], best[2]
        return self

    def score(self, X, Z=None):
        return self.m.decision_function(self.feat(X, Z))


class _MLP(torch.nn.Module):
    def __init__(self, din, width, depth):
        super().__init__()
        layers, d = [], din
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.SiLU()]; d = width
        layers += [torch.nn.Linear(d, 1)]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class _MLPLogit:
    def __init__(self, width=64, depth=2, lr=3e-3, wd=1e-4, epochs=80, batch=512, patience=6, seed=0):
        self.__dict__.update(width=width, depth=depth, lr=lr, wd=wd, epochs=epochs, batch=batch,
                             patience=patience, seed=seed)

    def _in(self, X, Z):
        X = (X - self.mu) / self.sd
        if Z is not None:
            X = np.concatenate([X, np.asarray(Z, float).reshape(len(X), -1)], 1)
        return torch.as_tensor(X, dtype=torch.float32)

    def fit(self, Xtr, ytr, swtr, Xca, yca, swca, Ztr=None, Zca=None):
        torch.manual_seed(self.seed)
        self.mu, self.sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        xt, xc = self._in(Xtr, Ztr), self._in(Xca, Zca)
        yt, yc = torch.as_tensor(ytr, dtype=torch.float32), torch.as_tensor(yca, dtype=torch.float32)
        wt, wc = torch.as_tensor(swtr, dtype=torch.float32), torch.as_tensor(swca, dtype=torch.float32)
        self.net = _MLP(xt.shape[1], self.width, self.depth)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        best, bad, state = float('inf'), 0, None
        g = torch.Generator().manual_seed(self.seed)
        n = len(xt)
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch):
                b = perm[i:i + self.batch]
                loss = torch.nn.functional.binary_cross_entropy_with_logits(self.net(xt[b]), yt[b], weight=wt[b])
                opt.zero_grad(); loss.backward(); opt.step()
            sched.step()
            self.net.eval()
            with torch.no_grad():
                v = float(torch.nn.functional.binary_cross_entropy_with_logits(self.net(xc), yc, weight=wc))
            if v < best - 1e-5:
                best, bad = v, 0
                state = {k: t.clone() for k, t in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(state)
        self.epochs_run = ep + 1
        return self

    def score(self, X, Z=None):
        with torch.no_grad():
            return self.net(self._in(X, Z)).double().numpy()


def _wbce(s, y, sw):
    s = np.asarray(s, float)
    l = np.logaddexp(0, -s) * y + np.logaddexp(0, s) * (1 - y)
    return float(np.sum(sw * l) / np.sum(sw))


def _balanced_weights(y):
    n1, n0 = y.sum(), (1 - y).sum()
    return np.where(y == 1, 0.5 * len(y) / n1, 0.5 * len(y) / n0)


def _platt(s, y, sw):
    """Fit log-odds' = a*s + b by weighted logistic regression (no regularisation)."""
    m = LogisticRegression(C=1e6, max_iter=1000).fit(s[:, None], y, sample_weight=sw)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


# ----------------------------------------------------------------------------- main API
class RatioModel:
    """Calibrated log-ratio estimator. log_w(H, Z) = a * score + b estimates log p(h|z)/q(h|z)."""

    def __init__(self, clf, a, b):
        self.clf, self.a, self.b = clf, a, b
        self.gated = False

    def log_w(self, H, Z=None):
        if self.gated:
            return np.zeros(len(H))
        return self.a * self.clf.score(H, Z) + self.b


def fit_ratio(Hp, Hq, Zp=None, Zq=None, family='quad', split=(0.6, 0.2, 0.2), seed=0, clf_kw=None,
              calibrate=True, null_gate=False):
    """Fit a calibrated classifier ratio estimator on real (Hp) vs synthetic (Hq) draws.

    Returns (RatioModel, diag) with diag computed on the held-out test split:
      auc, bce (balanced, nats), js_lb = log2 - bce, train_minus_test_bce, ece,
      kl_plugin = E_p log w, dv = E_p log w - log E_q w  (Donsker-Varadhan lower bound on KL),
      norm_q = E_q w, norm_p = E_p 1/w (both 1 if calibrated), khat_q, khat_p,
      chi2 = E_q w^2 - 1 (only if khat_q < 0.5, else nan), ess_q_frac, auc_se (Hanley-McNeil).
    null_gate: if the held-out AUC is within 2 SE of 0.5 ("indistinguishable at this n", METHOD T6), the
      returned model gives log w = 0 everywhere (removes classifier-noise covariance with G; diag['gated']).
    """
    rng = np.random.default_rng(seed)
    clf_kw = clf_kw or {}

    def splits(n):
        perm = rng.permutation(n); a = int(split[0] * n); b = a + int(split[1] * n)
        return perm[:a], perm[a:b], perm[b:]

    ip, iq = splits(len(Hp)), splits(len(Hq))
    def take(k):
        X = np.concatenate([Hp[ip[k]], Hq[iq[k]]])
        y = np.concatenate([np.ones(len(ip[k])), np.zeros(len(iq[k]))])
        Z = None if Zp is None else np.concatenate([np.asarray(Zp)[ip[k]], np.asarray(Zq)[iq[k]]])
        return X, y, Z, _balanced_weights(y)
    Xtr, ytr, Ztr, swtr = take(0); Xca, yca, Zca, swca = take(1); Xte, yte, Zte, swte = take(2)

    if family == 'quad':
        clf = _QuadLogit(seed=seed, **clf_kw)
    elif family == 'mlp':
        clf = _MLPLogit(seed=seed, **clf_kw)
    else:
        raise ValueError(family)
    clf.fit(Xtr, ytr, swtr, Xca, yca, swca, Ztr, Zca)
    a, b = _platt(clf.score(Xca, Zca), yca, swca) if calibrate else (1.0, 0.0)
    rm = RatioModel(clf, a, b)

    s_tr = rm.log_w(Xtr, Ztr); s_te = rm.log_w(Xte, Zte)
    diag = ratio_diagnostics(s_te[yte == 1], s_te[yte == 0])
    diag['train_minus_test_bce'] = _wbce(s_tr, ytr, swtr) - diag['bce']
    diag['ece'] = ece(expit(s_te), yte, weight=swte)
    diag['platt_a'], diag['platt_b'] = a, b
    diag['auc_se'] = auc_se(diag['auc'], int((yte == 1).sum()), int((yte == 0).sum()))
    diag['gated'] = bool(null_gate and abs(diag['auc'] - 0.5) < 2 * diag['auc_se'])
    rm.gated = diag['gated']
    if family == 'quad':
        diag['C'] = clf.C
    else:
        diag['epochs'] = clf.epochs_run
    return rm, diag


def ratio_diagnostics(lw_p, lw_q):
    """Diagnostics from log-ratio values on held-out real (lw_p) and synthetic (lw_q) draws."""
    lw_p, lw_q = np.asarray(lw_p, float), np.asarray(lw_q, float)
    bce = 0.5 * (np.mean(np.logaddexp(0, -lw_p)) + np.mean(np.logaddexp(0, lw_q)))
    kq = psis_khat(lw_q)
    lnq = logsumexp(lw_q) - math.log(len(lw_q))
    return dict(
        auc=auc(lw_p, lw_q), bce=float(bce), js_lb=float(LOG2 - bce),
        kl_plugin=float(lw_p.mean()), dv=float(lw_p.mean() - lnq),
        norm_q=float(math.exp(lnq)), norm_p=float(np.exp(logsumexp(-lw_p) - math.log(len(lw_p)))),
        khat_q=kq, khat_p=psis_khat(-lw_p),
        chi2=float(np.exp(logsumexp(2 * lw_q) - math.log(len(lw_q))) - 1) if kq < 0.5 else float('nan'),
        ess_q_frac=kish_ess_log(lw_q) / len(lw_q),
    )


def crossfit_log_w(Hp, Hq, Zp=None, Zq=None, K=2, family='quad', seed=0, clf_kw=None):
    """Out-of-fold log w for every real and synthetic point (K folds), so the same draws can be used
    both to fit the ratio and to estimate Cov / L(alpha) without in-sample overfitting bias."""
    rng = np.random.default_rng(seed)
    fp, fq = rng.integers(0, K, len(Hp)), rng.integers(0, K, len(Hq))
    lp, lq = np.empty(len(Hp)), np.empty(len(Hq)); diags = []
    for k in range(K):
        sel = lambda Z, m: None if Z is None else np.asarray(Z)[m]
        rm, dg = fit_ratio(Hp[fp != k], Hq[fq != k], sel(Zp, fp != k), sel(Zq, fq != k), family=family,
                           seed=seed + k, clf_kw=clf_kw)
        lp[fp == k] = rm.log_w(Hp[fp == k], sel(Zp, fp == k))
        lq[fq == k] = rm.log_w(Hq[fq == k], sel(Zq, fq == k))
        diags.append(dg)
    return lp, lq, diags


def refit_spread(Hp, Hq, Zp, Zq, estimate_fn, B=20, family='quad', seed=0, clf_kw=None):
    """Classifier-refit variance by half-sampling (delete-n/2 jackknife): refit the ratio on B random halves of
    the training draws (no duplicates, so no train/test leakage) and re-evaluate estimate_fn(RatioModel) on the
    fixed estimation draws. The spread of the B estimates estimates the refit variance at the full training n.
    Add it to the (ratio-fixed) bootstrap variance of the estimator: SE_total^2 = SE_boot^2 + var(spread).
    Returns the array of B estimates."""
    rng = np.random.default_rng(seed)
    out = []
    for b in range(B):
        ip = rng.permutation(len(Hp))[:len(Hp) // 2]; iq = rng.permutation(len(Hq))[:len(Hq) // 2]
        sel = lambda Z, i: None if Z is None else np.asarray(Z)[i]
        rm, _ = fit_ratio(Hp[ip], Hq[iq], sel(Zp, ip), sel(Zq, iq), family=family, seed=seed + 1 + b, clf_kw=clf_kw)
        out.append(estimate_fn(rm))
    return np.asarray(out, float)
