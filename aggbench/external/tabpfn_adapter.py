"""TabPFN (Hollmann et al. 2025, Nature, doi:10.1038/s41586-024-08328-6; PriorLabs pip package `tabpfn`) applied
directly as an in-context Bayesian aggregator: no training on aggbench, the train-split episodes are its context.

Model: TabPFN v2 weights (`ModelVersion.V2`; the default v2.5+ weights need a PriorLabs licence token, the v2 weights
download without one). On a GPU: the package's default ensemble size (E1 regressors: 1 estimator, a cost cap) and up
to 10000 context rows (TabPFN v2's pretraining limit); on a CPU: 1 estimator and 1000 context rows (TabPFN's CPU
default limit). `config()` records the values used.

Encoding (one row per episode). Rich (default): the input is aggbench.transformer.full_context (the tokens
full_context_transformer reads: every agent's raw observation, present mask, own sensor model, provenance key and
report), so TabPFN reads all the information the agents hold; scalar-report baseline: a View, and the row holds the reports and view keys only.
Agents are sorted (E2/E3 by report, E1 by trace of the reported precision) to remove the arbitrary agent order; the
per-agent tokens are flattened, followed by pooled summaries (E2/E3: sum and mean of the centred log-odds; E1: sum
of the information vectors and of the precisions). E2/E3: TabPFNClassifier on the outcome, forecast = logit of
P(Y=1). E1: one TabPFNRegressor per coordinate of theta; forecast mean = the predictive mean, precision =
diag(1 / predictive variance) (the cross-coordinate correlation is not modelled).

Controls (aggbench rule 5, adapted to an in-context learner): "control:shuffled_labels" fits the same context with
the outcomes permuted (no information about the labels), "control:random_network" is the fitted model with every
weight tensor replaced by scale-matched random values (aggbench.controls.random_weights).
"""
import numpy as np
import torch

from .learned import tokens


def _reports(x):
    """The agents' reports of a View, or as carried at the end of each FullContext token (aggbench.transformer)."""
    if hasattr(x, "reports"):
        return x.reports
    t = np.asarray(x.tokens, float)
    if x.kind == "bernoulli":
        return {"logit": 4 * t[..., -1]}                      # full_context stores log-odds / 4
    d = x.public["d"]
    return {"mean": t[..., -(d + d * d):-(d * d)], "prec": t[..., -(d * d):].reshape(t.shape[:2] + (d, d))}


def _sorted_tokens(x):
    rep = _reports(x)
    if x.kind == "bernoulli":
        score = rep["logit"]
    else:
        score = np.einsum("naii->na", rep["prec"])
    order = np.argsort(-score, axis=1, kind="stable")
    return np.take_along_axis(tokens(x), order[..., None], axis=1)


def features(x):
    rows = _sorted_tokens(x)
    n = len(rows)
    flat = rows.reshape(n, -1)
    rep = _reports(x)
    if x.kind == "bernoulli":
        c = rep["logit"] - x.public["prior_logit"]
        extra = np.stack([c.sum(1), c.mean(1)], 1)
    else:
        P = rep["prec"]
        d = P.shape[-1]
        iu = np.triu_indices(d)
        h = np.einsum("naij,naj->ni", P, rep["mean"])
        extra = np.concatenate([h, P.sum(1)[:, iu[0], iu[1]]], 1)
    return np.concatenate([flat, extra], 1)


def config(device=None, kind="bernoulli"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        return {"device": "cpu", "n_context": 1000, "n_estimators": 1, "weights": "v2"}
    # GPU: package-default ensemble (None), except E1's four regressors (x3 arms), capped at 1 estimator for cost
    return {"device": device, "n_context": 10000, "n_estimators": 1 if kind == "gaussian" else None, "weights": "v2"}


class TabPFNAggregator:
    def __init__(self, seed=0, n_context=None, n_estimators="auto", shuffle_labels=False, device=None):
        self.device = config(device)["device"]
        self.n_context_arg, self.n_estimators_arg = n_context, n_estimators
        self.seed, self.shuffle_labels = seed, shuffle_labels
        self.module = None
        self.kind = None

    def _new(self, cls):
        from tabpfn.constants import ModelVersion
        kw = {} if self.n_estimators is None else {"n_estimators": self.n_estimators}
        return cls.create_default_for_version(ModelVersion.V2, device=self.device, random_state=self.seed,
                                              ignore_pretraining_limits=True, **kw)

    def fit(self, view, truth):
        from tabpfn import TabPFNClassifier, TabPFNRegressor
        self.kind = view.kind
        c = config(self.device, view.kind)
        self.n_context = self.n_context_arg or c["n_context"]
        self.n_estimators = c["n_estimators"] if self.n_estimators_arg == "auto" else self.n_estimators_arg
        X = features(view)[:self.n_context]
        y = np.asarray(truth)[:self.n_context]
        if self.shuffle_labels:
            y = y[np.random.default_rng(self.seed + 7).permutation(len(y))]
        if view.kind == "bernoulli":
            self.models = [self._new(TabPFNClassifier).fit(X, y.astype(int))]
        else:
            self.models = [self._new(TabPFNRegressor).fit(X, y[:, j]) for j in range(y.shape[1])]
        self.module = torch.nn.ModuleList([m.models_[0] for m in self.models])
        return self

    def predict(self, view, chunk=500):
        X = features(view)
        if self.kind == "bernoulli":
            p = np.concatenate([self.models[0].predict_proba(X[i:i + chunk])[:, 1] for i in range(0, len(X), chunk)])
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return {"logit": np.log(p) - np.log1p(-p)}
        means, vars_ = [], []
        for m in self.models:
            mu, var = [], []
            for i in range(0, len(X), chunk):
                o = m.predict(X[i:i + chunk], output_type="full")
                mu.append(np.asarray(o["mean"], float))
                var.append(o["criterion"].variance(o["logits"]).detach().cpu().double().numpy())
            means.append(np.concatenate(mu))
            vars_.append(np.concatenate(var))
        mean, var = np.stack(means, 1), np.clip(np.stack(vars_, 1), 1e-6, None)
        return {"mean": mean, "prec": np.einsum("ni,ij->nij", 1 / var, np.eye(mean.shape[1]))}


def run_tabpfn_with_controls(train_view, train_truth, eval_view, seed=0, **kw):
    """Dict arm -> forecast for the trained (in-context) arm and its two controls."""
    from ..controls import random_weights
    agg = TabPFNAggregator(seed=seed, **kw).fit(train_view, train_truth)
    shuffled = TabPFNAggregator(seed=seed, shuffle_labels=True, **kw).fit(train_view, train_truth)
    return {
        "trained": agg.predict(eval_view),
        "control:shuffled_labels": shuffled.predict(eval_view),
        "control:random_network": random_weights(agg, seed + 1).predict(eval_view),
    }
