# FORECAST hawk sanity sim (CPU, numpy, one process, ~20 s).
# Question: with committed-only supervision, what goes wrong on non-committed candidates,
# and does a full-support commit rule with importance weights (Chen-Kash-Ruberry-Shnayder 2011,
# Thm 1) recover what branching gives?  Also: a phrasing channel the policy can use to inflate
# forecasts, and the selection artefact in a "martingale" check on committed plans.
import numpy as np, json
sg = lambda z: 1/(1+np.exp(-z))
C, K, T, SEEDS = 50, 8, 20000, 5           # context types, candidates, episodes, seeds
def run(arm, param, seed, eps=0.1, lr=0.2):
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1.5, (C, K)) - 1.0    # true logits per (context, plan)
    F = rng.normal(0, 1, 4)                  # plan features shared across contexts (linear arm)
    feat = rng.normal(0, 1, (C, K, 4))
    if param == 'linear':                    # truth = linear in features + idiosyncratic part
        Z = feat @ F + 0.7*rng.normal(0, 1, (C, K)) - 1.0
    P = sg(Z)
    # forecaster: tabular logit table, or linear over features + per-cell residual table (shared)
    W = np.zeros((C, K)) + rng.normal(0, 1.0, (C, K))   # init noise -> some pessimistic errors
    w = np.zeros(4)
    puff = 1.0                                # weight on 'puffed phrasing' feature: forecaster initially fooled
    def fc(c, k, ph):
        z = W[c, k] + (feat[c, k] @ w if param == 'linear' else 0) + puff*ph
        return sg(z)
    regret = 0.0; npuff = 0
    for t in range(T):
        c = rng.integers(C)
        f_plain = np.array([fc(c, k, 0) for k in range(K)])
        f_puff = np.array([fc(c, k, 1) for k in range(K)])
        # policy: chooses plan and phrasing that maximise the forecast it reads (greedy reader)
        ph_best = (f_puff > f_plain).astype(int)
        f_best = np.maximum(f_puff, f_plain)
        kstar = int(np.argmax(f_best))
        if arm == 'ips' and rng.random() < eps:
            k = int(rng.integers(K))
        else:
            k = kstar
        prop = np.full(K, eps/K) if arm == 'ips' else np.zeros(K)
        prop[kstar] += (1-eps) if arm == 'ips' else 1.0
        ph = ph_best[k]; npuff += ph
        y = float(rng.random() < P[c, k])
        regret += P[c].max() - P[c, k]
        # updates
        labelled = range(K) if arm == 'branch' else [k]
        for j in labelled:
            yj = y if j == k else float(rng.random() < P[c, j])   # branch = independent counterfactual draw
            phj = ph_best[j]
            wt = (1/prop[j]) / K if arm == 'ips' else 1.0          # IPS weight, normalised to mean 1
            wt = min(wt, 10.0)
            g = (fc(c, j, phj) - yj) * wt * lr
            W[c, j] -= g
            if param == 'linear': w -= g * feat[c, j] * 0.1
            puff -= g * phj * 0.05
    # evaluation against the exact truth on ALL candidates (oracle targets, no sampling noise)
    fp = np.array([[fc(c, k, 0) for k in range(K)] for c in range(C)])
    chosen = np.zeros((C, K), bool)
    for c in range(C): chosen[c, int(np.argmax(np.maximum(fp[c], [fc(c, k, 1) for k in range(K)])))] = True
    err = fp - P
    return dict(regret_per_ep=regret/T, brier_vs_truth_all=float(np.mean(err**2)),
                signed_err_nonchosen=float(err[~chosen].mean()),
                abs_err_nonchosen=float(np.abs(err[~chosen]).mean()),
                signed_err_chosen=float(err[chosen].mean()),
                frac_best_plan_forecast_below_truth_by_0p1=float(np.mean([(fp[c].argmax()!=P[c].argmax()) and (fp[c, P[c].argmax()] < P[c, P[c].argmax()]-0.1) for c in range(C)])),
                puff_weight_final=float(puff), frac_commits_puffed=npuff/T)
out = {}
for param in ['tabular', 'linear']:
    for arm in ['committed', 'ips', 'branch']:
        rs = [run(arm, param, s) for s in range(SEEDS)]
        out[f'{param}/{arm}'] = {k: [float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))] for k in rs[0]}
        print(param, arm, {k: f'{v[0]:.4f}±{v[1]:.4f}' for k, v in out[f'{param}/{arm}'].items()}, flush=True)
# martingale selection artefact: perfect Bayesian forecaster, K candidates, two signals before commit
rng = np.random.default_rng(1); N = 200000
z = rng.normal(-1, 1.5, (N, K)); p = sg(z)
# prior forecast p0 = E[p] given nothing about the plan (common to all), signal 1 reveals p exactly
p0 = float(sg(rng.normal(-1, 1.5, 10**6)).mean())
k = p.argmax(1); pc = p[np.arange(N), k]; y = (rng.random(N) < pc).astype(float)
art = dict(prior_forecast=p0, mean_revision_prior_to_commit_given_committed=float((pc - p0).mean()),
           mean_outcome_minus_commit_forecast_given_committed=float((y - pc).mean()),
           se_outcome_minus_commit=float((y-pc).std()/np.sqrt(N)))
print('martingale artefact', art)
out['martingale_selection_artefact'] = art
json.dump(out, open('/home/agent/workspaces/eurisko-study/lir-agent/hawks/FORECAST-sim.json', 'w'), indent=1)
