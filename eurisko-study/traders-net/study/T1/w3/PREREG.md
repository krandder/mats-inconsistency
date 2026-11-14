# PREREG, T1 worker 3: lemmas 6 (fund of funds) and 7 (top-k router gradient, broker gate)

Written 2026-09-27 before any run. Substrate: S0's verbatim copy of wealth-net streams
(`study/S0/substrate/streams.py`, 8 families, Q=64). Seeds 1, 2, 3 (stream seed = model/init seed).
Scripts: `check_L6.py` (numpy), `check_L7.py` (torch float64 from wealth-net/.pylib), outputs `out/*.json|csv`.
Every reported number is mean ± sd over the 3 seeds unless it is an exact identity (then: max error over seeds).

## Lemma 6 (theorem): fixed-fraction fund of funds = flat mixture; power only from rebalancing

Leaves: per-question online predictors on the S0 stream (KT context counters k = 0,1,2,3,4,6,8;
online logistic on a 16-step window; two miscalibrated constants 0.9 / 0.1): 10 leaves.
Kelly traders against an exogenous price pi_t (order-0 KT running base rate): W_i,t+1 = W_i,t p_i(y)/pi(y).
Funds: random DAG over leaves (3 levels, shared children), plus a cyclic cross-holding variant (spectral radius < 1).

| check | arm | metric | pass |
|---|---|---|---|
| C6a | buy-and-hold DAG vs flat mixture, path-sum (Leontief) weights | max abs log-wealth difference over t, q, seeds | < 1e-9 |
| C6b | constant-rebalanced DAG vs flat CRP, path-product weights | same | < 1e-9 |
| C6c | endogenous wealth-weighted price: market of buy-and-hold funds vs market of flattened leaves | max abs price difference | < 1e-9; report the same diff for rebalancing funds (expected > 0) |
| C6d | growth vs market (nats/step), per family: best leaf (hindsight), buy-and-hold uniform, uniform CRP, online EG CRP (causal, eta=0.05), best CRP in hindsight | mean ± sd over seeds | descriptive; predicted BH - best leaf in [-ln(10)/T, 0]; best CRP >= best leaf |
| C6e | analytic toy: truth 0.5, price 0.3, leaves 0.9 and 0.1 | growth rates | leaf/BH = -0.424, 50/50 CRP = +0.087 |

## Lemma 7 (theorem + known): top-k router gradient = gate-weighted relative revenue, censored

Tiny MoE on S0 windows: x = random projection (d=32) of the +-1 window (W=16) of one question,
N=8 experts (ReLU MLP, hidden 16), readout to one logit, BCE on the next outcome. Batch 256 tokens.
Gate conventions: (i) softmax over top-k, renormalised (k=2); (ii) softmax over all N, top-k unnormalised (k=2);
(iii) sigmoid then normalised over top-k (DeepSeek-V3 style, k=2); (iv) top-1 renormalised; (v) top-1 unnormalised (Switch).

| check | metric | pass |
|---|---|---|
| C7a | max abs error of the closed-form -dL/dz vs autograd, per convention | < 1e-10 (float64) |
| C7b | zero-sum: sum_i dL/dz_i; (ii) routed experts gain R(1 - sum_S g), outsiders lose g_i R | < 1e-10 |
| C7c | tangent books at the gate: (sum g dR/dg + sum <E, dR/dE>) / R; broker and Euler conventions conserve; SwiGLU doubling | = 2 within 1e-10; residual < 1e-10 |
| C7d | Exp3 shadow-credit estimator vs full-information (dense softmax) gradient, 2e5 Monte Carlo draws | relative error within 4 MC standard errors |
| C7e | lock-in toy (numpy bandit): N=16 experts, mean revenues mu_i ~ U(0.1,1), bias-only router, SGD, 5k steps, 100 instances per seed; arms: top-1 unnormalised (censored), top-2 renormalised (censored), top-1 + Exp3 shadow credit | fraction of instances whose final argmax is the best expert, mean ± sd over 3 seeds | descriptive; predicted censored top-1 ~ 1/16-level (locks on first positive earner), Exp3 near 1 |

Stop rule: an exact check (C6a-c, C7a-c) failing its tolerance means the lemma as stated is false:
record it, fix the statement (not the tolerance), rerun once. C6d, C7d, C7e are descriptive; no tuning
after seeing results (eta, T, N fixed above). CPU only, at most 3 processes, 1 thread each.
