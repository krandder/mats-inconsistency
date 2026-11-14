# T1 worker 1 pre-registration: numerical checks for lemmas 1-3

Written 2026-09-27 before any run. Scope: THEORY.md lemmas 1 (books identity), 2 (exact LMSR
settlement), 3 (trunk cost sharing). Each lemma is tagged theorem / known (cited) / conjecture and
gets one check script under `study/T1/w1/`, output under `study/T1/w1/out/` (JSON only).

Substrate: S0 does not exist yet at 22:10Z (`study/S0/` has no substrate). Toys are therefore
(i) a float64 copy of wealth-net's Net (Q=16, W=8, d=16, H=2, L=2) on wealth-net streams
(`streams.py` copied, seeds 1-3), (ii) a one-hidden-layer ReLU MLP / logistic regression on
synthetic data with a known posterior. If S0 lands before the final, rerun L1(b) on it.

Seeds: 1, 2, 3 for every stochastic check; report mean ± sd over seeds.

## Arms and metrics

L1a Euler identity (theorem). Per unit (readout, MLP neuron in-side and out-side, head out-side),
  books income a·dR/da vs -θ·dL/dθ in the same graph (frozen-LN). Metric: max relative error.
  Pass: < 1e-9 (float64). Also report the gap to the true live-LN loss gradient (LN residual).
L1b Norm growth. Train the toy 2000 steps, arms SGD / SGD+momentum 0.9 / Adam, books on the same
  batch pre-update (as wealth-net). Metrics: per-step identity residual of
  income = (Δ‖θ‖² - η²‖g‖²)/(2η) under SGD (pass < 1e-9 relative); R² across units of cumulative
  wealth vs Δ‖θ‖²/2η, per arm, mean ± sd over seeds; R² of books (frozen LN) vs -θ·∇L (live LN).
L1c Weight-decay rent. SGD + coupled L2 λ ∈ {0, 1e-3, 1e-2} on a non-separable toy to
  stationarity. Metrics: time-mean income vs λ‖θ‖² - (η/2)‖g̃‖² over the second half; fraction of
  units with positive income; Spearman(income, ‖θ‖²).
L1d Zero-mean tangent. (i) Predictor = true posterior on synthetic data: mean tangent income per
  unit and total, with standard error, n = 2e5 per seed; pass: |mean| < 3 se. (ii) Overfit
  logistic regression (signal + noise features) at convergence: in-sample per-feature income ≈ 0,
  prequential per-feature income: sign on signal vs noise features.
L2a LMSR identity (theorem, known): exact payoff = log p1(y) - log p0(y) = tangent + KL(p1‖p0);
  max abs error on random draws. Pass < 1e-12.
L2b IG split. Closed form share_i = c_i(y - p̄), p̄ = (softplus ℓ1 - softplus ℓ0)/(ℓ1 - ℓ0), vs
  16-step midpoint and trapezoid (M1's planned path length). Metric: max and 99th pct relative
  error of shares for |Δℓ| ≤ 5, 10, 20.
L2c Expected exact settlement from the base-rate prior = I(X;Y) for a calibrated posterior;
  mean ± sd over seeds vs numerical integral. Pass: within 3 se.
L2d Coalition gap: IG share of a coalition vs its exact ablation value; report gap vs bound.
L3a Euler for costs: for a degree-r homogeneous C, Σ x_i ∂C/∂x_i = r·C (marginal pricing collects
  r·C); Aumann-Shapley balances the budget for any smooth C. Check on random instances.
L3b K=8 traders on a concave shared-trunk cost: Shapley (exact, 2^8), Aumann-Shapley (path
  integral), marginal cost. Metric: budget residual, per-trader L1 distance between rules.
L3c FLOP budget tables for M7 (E ∈ {8, 64}, top-2, wealth-net dims) and K1 (pattern-k and online
  logistic traders sharing window features). Deliverable: `costshare.py` (importable) + JSON.

## Stop rule

Each check runs once per seed (deterministic). If a "theorem" check fails its pass threshold, the
lemma is re-tagged conjecture/false and the failure is reported, not tuned away. Total CPU
< 30 min, at most 3 processes at once, torch threads = 2.
