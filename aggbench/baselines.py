"""Non-learned baselines. All take a View (reports only) except oracle_clearing.

Bernoulli envs (reports are log-odds l_i, prior log-odds l0):
  linear_pool   logit of the mean probability
  log_pool      mean of log-odds
  sum_logodds   sum of log-odds minus (N-1) l0: the product of posteriors if agents were
                independent (equals the plain sum when the prior is 0.5)
  extremized    l0 + alpha (mean_i l_i - l0), alpha fitted on train by log loss
Gaussian env (reports are posterior mean m_i, precision P_i; prior N(0, I)):
  linear_pool   moment-matched Gaussian of the equal-weight mixture of the agents' posteriors
  log_pool      normalized geometric mean: precision mean_i P_i, information mean_i P_i m_i
  sum_logodds   independent product: precision sum_i P_i - (N-1) I, information sum_i P_i m_i
  extremized    precision I + alpha (mean_i P_i - I), information alpha mean_i P_i m_i
oracle_clearing  exact posterior given the reports AND the true structure (coverage, piece
                 strengths, noise correlations, adversaries' distortions). It equals the global
                 Bayes posterior in E1 and whenever the union is identified from the reports in
                 E2/E3 (see envs.union_identified); otherwise it is the reports-level ceiling.
                 E4: the exact posterior given the rollouts' reports and the true error structure
                 (orders, calibration, recency, shared and private noise sds); see envs.RolloutEnv.
                 Structure zoo (Z*): the exact posterior given the reports and the true structure, by
                 enumeration over the finite world (symbols with equal reports merged); see aggbench.zoo.
"""
import numpy as np

from .envs import RolloutConfig, gaussian_global, report_posterior_logit, rollout_report_posterior_logit
from .metrics import bernoulli_logloss, gaussian_nll, logsig


def _gauss(P, h):
    return {"mean": np.linalg.solve(P, h[..., None])[..., 0], "prec": P}


def _info(view):
    prec = view.reports["prec"]
    return prec, np.einsum("naij,naj->nai", prec, view.reports["mean"])


def linear_pool(view):
    if view.kind == "bernoulli":
        l = view.reports["logit"]
        n_ag = l.shape[1]
        logp = np.logaddexp.reduce(logsig(l), axis=1) - np.log(n_ag)
        log1mp = np.logaddexp.reduce(logsig(-l), axis=1) - np.log(n_ag)
        return {"logit": logp - log1mp}
    m = view.reports["mean"]
    cov = np.linalg.inv(view.reports["prec"])
    mix_mean = m.mean(1)
    second = (cov + np.einsum("nai,naj->naij", m, m)).mean(1)
    mix_cov = second - np.einsum("ni,nj->nij", mix_mean, mix_mean)
    return {"mean": mix_mean, "prec": np.linalg.inv(mix_cov)}


def extremized(view, alpha):
    if view.kind == "bernoulli":
        l0 = view.public["prior_logit"]
        return {"logit": l0 + alpha * (view.reports["logit"].mean(1) - l0)}
    prec, h = _info(view)
    eye = np.eye(prec.shape[-1])
    return _gauss(eye + alpha * (prec.mean(1) - eye), alpha * h.mean(1))


def log_pool(view):
    return extremized(view, 1.0)


def sum_logodds(view):
    n_ag = view.keys.shape[1]
    if view.kind == "bernoulli":
        l0 = view.public["prior_logit"]
        return {"logit": view.reports["logit"].sum(1) - (n_ag - 1) * l0}
    prec, h = _info(view)
    return _gauss(prec.sum(1) - (n_ag - 1) * np.eye(prec.shape[-1]), h.sum(1))


def _mean_logloss(forecast, kind, truth):
    if kind == "bernoulli":
        return bernoulli_logloss(forecast["logit"], truth).mean()
    return gaussian_nll(forecast["mean"], forecast["prec"], truth).mean()


def fit_extremize(view, truth, lo=0.05, hi=20.0):
    """alpha minimizing mean log loss on (train) episodes: log grid, then golden-section refine."""
    loss = lambda la: _mean_logloss(extremized(view, np.exp(la)), view.kind, truth)
    grid = np.linspace(np.log(lo), np.log(hi), 61)
    j = int(np.argmin([loss(g) for g in grid]))
    a, b = grid[max(j - 1, 0)], grid[min(j + 1, len(grid) - 1)]
    g = (np.sqrt(5) - 1) / 2
    for _ in range(40):
        c, d = b - g * (b - a), a + g * (b - a)
        if loss(c) < loss(d):
            b = d
        else:
            a = c
    return float(np.exp((a + b) / 2))


def oracle_clearing(batch):
    """Exact posterior from reports + true structure (see module docstring)."""
    o = batch.oracle
    if "report_logit" in o:  # structure zoo: exact reports ceiling computed by enumeration (aggbench.zoo)
        return {"logit": np.asarray(o["report_logit"], float).copy()}
    if "W" in o:  # E4 rollouts
        c, l0 = RolloutConfig(**batch.config), batch.public["prior_logit"]
        logit = [rollout_report_posterior_logit(batch.reports["logit"][e], o["W"][e], o["length"][e], o["mu"][e],
                                                c, l0) for e in range(len(batch))]
        return {"logit": np.array(logit)}
    if batch.kind == "bernoulli":
        l0 = batch.public["prior_logit"]
        logit = [report_posterior_logit(batch.reports["logit"][e], o["cov"][e], o["mu"][e],
                                        o["a"][e], o["b"][e], l0) for e in range(len(batch))]
        return {"logit": np.array(logit)}
    means, precs = [], []
    for e in range(len(batch)):
        A, Bf, sig = o["A"][e], o["Bf"][e], o["sig"][e]
        N, m, d = A.shape
        s = np.zeros((N, m))
        for i in range(N):  # recover s_i from the report: P_i m_i = A_i^T R_i^{-1} s_i
            R = Bf[i] @ Bf[i].T + sig[i] ** 2 * np.eye(m)
            h = batch.reports["prec"][e, i] @ batch.reports["mean"][e, i]
            s[i] = np.linalg.lstsq(A[i].T @ np.linalg.inv(R), h, rcond=None)[0]
        mean, prec = gaussian_global(A, Bf, sig, s)
        means.append(mean)
        precs.append(prec)
    return {"mean": np.array(means), "prec": np.array(precs)}


def all_baselines(train_batch, eval_batch, visibility="full"):
    """Dict name -> forecast on eval_batch; extremizing alpha is fitted on train_batch. Also returns alpha."""
    ev = eval_batch.view(visibility)
    alpha = fit_extremize(train_batch.view(visibility), train_batch.truth)
    out = {
        "bayes": dict(eval_batch.bayes),
        "oracle_clearing": oracle_clearing(eval_batch),
        "linear_pool": linear_pool(ev),
        "log_pool": log_pool(ev),
        "sum_logodds": sum_logodds(ev),
        "extremized_log_pool": extremized(ev, alpha),
    }
    return out, alpha
