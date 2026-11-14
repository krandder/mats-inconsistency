# PREREG, T1 worker 2: lemma 4 (delayed settlement) and lemma 5 (finite-population LI)

Written 2026-09-27 before any run. Substrate: `study/S0/substrate/streams_ext.make_questions`
(wealth-net streams with seeded PRNG-parity instead of pi; 8 families, Q=64). CPU only, numpy,
one process at a time, OMP_NUM_THREADS=1. Seeds 1, 2, 3; report mean ± sd over seeds.
Outputs: `study/T1/w2/results/*.json|csv`; scripts `check_delay.py`, `check_li.py`.

## Check 4: delayed settlement (`check_delay.py`)

Toy market: per-question online logistic regression (window W=16 of revealed bits as ±1, bias),
plain SGD, lr 0.05, T=8000 steps. Delay D ∈ {0, 64, 512}: the forecast at t uses bits revealed
before t (y_{t-D-W} .. y_{t-D-1}); y_t is revealed at t+D and the cached trade-time gradient is
applied then (stale-gradient SGD). Prior for exact books: ℓ0 = logit of the Laplace running base
rate of revealed labels. Units: the 16 weights and the bias of each question.

Quantities (per family, per D):
- 4a telescoping: |Σ mark-to-market increments − settlement at reveal|, marks = the current
  model re-priced on the cached trade input every 16 steps. Prediction: ≤ 1e-9 (identity).
- 4b hindsight gap: exact-LMSR income booked with the reveal-time model minus income booked at the
  trade price, per contract. Prediction: 0 at D=0; > 0 and increasing in D on learnable families
  (periodic, and, or, prime); |gap| within 2 sd of 0 on prng and on shuffled-label controls.
- 4c stale-gradient norm identity: Δ‖θ‖²/2η = W_trade − S_stale + (η/2)Σ‖g‖², exact to 1e-8
  relative; report S_stale / |W_trade| vs D (prediction: 0 at D=0, grows with D).
- 4d self-resolution: settle on y' ~ Bernoulli(p_t) instead of y. Prediction: tangent income mean
  within 3 SE of 0 on every family; exact income mean = mean KL(p_t‖p0_t) > 0 on every family,
  including prng, where true-label exact income is ≤ 0.
- 4e (identity only, not an RL run): TD errors on a random 20-state Markov reward chain with a
  wrong critic telescope to G_t − V(s_t) (λ=1) to 1e-9; one-step (λ=0) books are biased by the
  critic error.

## Check 5: finite-population LI (`check_li.py`)

Market per question: Kelly (log-utility) traders, price = wealth-weighted mean of beliefs (BLP),
wealth ← wealth · q_i(y)/p(y). Residents: pattern-0/1/2 (KT counts on the last k revealed bits).
Class C: pattern-4, pattern-8, and 20 smooth price-reading calibration traders
q = σ(logit p ± 1·bump_b(p)) (10 triangular bins). Extra family "diag" (8 questions):
y_t = 1[p_t < 1/2] on the market's own clearing price (Garrabrant-style paradox). T=8000, Q=72.
E_T(t) = Σ_{s≤t} log(q_T(y_s)/p_s(y_s)) at the clearing price (exact-LMSR P&L of T), per question,
averaged within family; E_C = max over T ∈ C. Invasion fitness φ_T = E_T(T)/T at wealth → 0.

Arms: N (residents only, C shadow); a (C in population at ε=1e-3, one pass: price-readers see the
residents' pre-price); b (a + fixed-point clearing p = Σ w_i q_i(p), bisection 50 iters);
c (N + births: per step per question with prob ρ ∈ {1e-3, 1e-2} a trader drawn uniformly from
C ∪ 9 constant-belief junk traders enters at wealth ε taken pro rata); d (residents + pattern-8
only; shadow pattern-4 — "covering" instead of membership); a-D64 (arm a with D=64 delay,
D+1 interleaved sub-markets).

Predictions / stop rule (pass/fail per theorem, no tuning after seeing data):
- P5.1 (theorem) arms b: max_T∈C E_T(t) ≤ ln(1/ε) for all t, all questions, all seeds.
- P5.2 (theorem) arm a: E_T ≤ ln(1/ε) for pattern traders; price-reading traders measured at the
  clearing price may exceed it, most on diag.
- P5.3 (theorem) arm a-D64: E_T ≤ (D+1) ln(1/ε).
- P5.4 (conjecture support) arm N: E_C second-half slope > 0 (3/3 seeds) on periodic and diag;
  arm c: E_C plateaus (second-half slope < 0.2 × first-half) at ρ=1e-2; arm d: shadow
  pattern-4's E grows sublinearly (second-half slope < 0.5 × first-half) on periodic.
A violated theorem prediction means a bug or a false lemma: fix or retract, never re-tune.
Stop: when all rows above are measured on 3 seeds (wall cap 3 h).

## Addendum 2026-09-27T22:17:59Z, after seeing check-4 data (exploratory, not preregistered)
At lr 0.05 the stale-gradient learner is unstable for D ≥ 64 (exact income strongly negative on
every family, also on shuffled labels), so prediction 4b is confounded. Exploratory rerun with a
delay-scaled step η_D = 0.05·16/(16+D) (`check_delay.py --eta-scale`, results/delay_stable.json).
Reported as exploratory; the preregistered 4b verdict stays as measured.
