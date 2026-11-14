# THEORY, part 2 (T1 worker 2): lemmas 4 and 5

Checked 2026-09-27 ~22:37Z. Scripts and outputs are in `study/T1/w2/`: `PREREG.md`, `check_delay.py`,
`check_li.py` and `summarize.py`. Results are `results/delay.json`, `results/delay_stable.json`,
`results/li.json`, the tables `results/summary_delay.csv`, `results/summary_delay_stable.csv`,
`results/summary_li.csv` and `results/li_curves.csv`, and the aggregate `results/summary.json`.
Substrate: `study/S0/substrate/streams_ext.make_questions` (8 families, Q=64, seeded PRNG-parity); its
sha256 prefix is recorded in every results file. Seeds 1-3. Numbers are mean ± sd over the 3 seeds, or, for
exact identities, the maximum error over seeds. CPU only, one process, GPU $0.
To rerun: `cd study/T1/w2; export OMP_NUM_THREADS=1; python3 check_delay.py; python3 check_delay.py --eta-scale --out results/delay_stable.json; python3 check_li.py; python3 summarize.py`
(about 2 min, 2 min and 20 min under load 40).

Notation. A contract on question q is opened at trade time t at logit ℓ_t (price p_t = σ(ℓ_t)) against
a prior logit ℓ0_t. Its outcome y is revealed at t+D. Books: tangent ℓ_t(y − p_t), and exact
LMSR log p_t(y) − log p0_t(y) (lemma 2). A mark m ∈ [0,1] is a probability for y=1 used to value an
open contract before reveal.

---

## Lemma 4: delayed settlement. Book at the trade price, mark to market in between, settle at reveal

Toy (`check_delay.py`, PREREG check 4): one online logistic model per question on the last W=16
revealed bits, with plain SGD, lr 0.05 and T=8000. With delay D ∈ {0, 64, 512} the forecast at t sees only
bits revealed before t, and the trade-time gradient is cached and applied at t+D (stale-gradient
SGD). The exact-book prior is the Laplace running base rate. Every contract caches x, θ, ℓ, ℓ0 and a
self-sampled outcome y′ in a ring buffer. Open contracts are re-marked every 16 steps and at D/2.

**4.1 Telescoping. [theorem, elementary]** A contract's value at mark m is linear in m:
tangent V(m) = ℓ_t(m − p_t), and exact V(m) = m·[log p_t(1) − log p0_t(1)] + (1−m)·[log p_t(0) − log p0_t(0)].
For any mark sequence p_t = m_0, m_1, …, m_n = y, the mark-to-market increments sum to V(y) − V(m_0).
- Tangent: this is exactly the settlement.
- Exact: V(y) is the settlement, and V(p_t) = KL(p_t‖p0_t) is the "day-one" value a trader books if it marks at its own trade price.

Check: the max residual over contracts, families, D and seeds is 3.3e-14 for the tangent books (`results/delay.json`, `tel_tan_max`). For exact books the check only confirms V(y) = settlement (0.0, by construction).

**4.2 Martingale marks, and marks from the trader's own model. [theorem; known]** If the marks are the
conditional probabilities m_k = P(y=1 | F_k) for some filtration, V(m_k) is a martingale, because V is linear. So
E[settlement | F_k] = V(m_k), and income can be paid at every mark without bias. Real marks come from
a model, and a mark taken from the trader's *own* current model is biased toward its own trade: at m = p_t
it books KL(p_t‖p0_t) ≥ 0 whatever the truth (see 4.5). Measured at D/2, with bias = final settlement minus the mid-life mark:
- **Mid-life marks were optimistic.** The bias was −11.2 to −0.011 at lr 0.05 (`results/summary_delay.csv`, `mtm_bias_ex`) and −0.059 to +0.002 with the delay-scaled step (`results/summary_delay_stable.csv`).
- **The marks tracked settlement poorly.** corr(mid mark, settlement) was ≤ 0.18 at lr 0.05 and ≤ 0.48 with the scaled step.

Rule: mark open contracts with a price that the trader does not control (the market's clearing price or a
held-out model), never with the trader's own model.

**4.3 Hindsight gap: cache at trade, settle at reveal. [theorem: identity; measured: partial]**
Suppose a contract is booked at reveal with the reveal-time model p′ on the cached input, instead of at the trade price. The
credit then changes by gap = log p′(y) − log p_t(y). This is learning between t and t+D on that input (p′ has
been updated on the labels of the contracts opened in the D steps before t, though not on y itself). It is a prequential leak, not trade income. At D=0 the gap is
exactly 0 (checked: 0.0 in every D=0 cell). Rule: store ℓ_t, ℓ0_t and the per-unit shares at trade time, and apply the settlement to
those cached books at reveal. Never re-price at reveal.
- *Preregistered 4b fails as stated* (`results/summary_delay.csv`). At lr 0.05 the gap on periodic is +0.0051 ± 0.00006
  (D=64) and +0.046 ± 0.0008 (D=512). But it is negative on ca110 at D=64 and far from 0 on shuffled labels
  (periodic D=512 shuffled −0.57 ± 0.16). The learner is unstable for D ≥ 64 at this step: exact income on
  prng falls from −0.061 (D=0) to −1.78 (D=512).
- *Exploratory, logged in PREREG addendum* (η_D = 0.05·16/(16+D), `results/summary_delay_stable.csv`).
  All four preregistered learnable families are positive and increase with D. From D=64 to D=512:
  - periodic: 0.0053 → 0.0474
  - prime: 0.0021 → 0.0296
  - and: 0.0028 → 0.0307
  - or: 0.0013 → 0.0120

  On lfsr16 and prng, |gap| ≤ 4.9e-4 (the largest is lfsr16 at D=64, −0.00049 ± 0.00021; ≤ 1.5e-4
  at D=512 and for prng; lead correction 22:45Z). The shuffled controls stay positive at D=512 (periodic 0.017, prime
  0.027, and 0.028): shuffling keeps each question's base rate, and the model still learns it. So the gap measures
  *all* learning, and the "shuffled ≈ 0" clause was a wrong prediction, not a bug.
- **Size.** At D=512 (scaled step), hindsight booking adds 0.030 on prime and 0.031 on "and", where the trade-price
  income is −0.004 and −0.012. It flips the sign of those families' income.

**4.4 Stale-gradient norm identity (the delay residual to lemma 1.2). [theorem]** Plain SGD applies at
step k the gradient g_τ computed at θ_τ (k = τ + D). Then
Δ‖θ‖²/2η = W_trade − S_stale + (η/2)Σ‖g_τ‖², with W_trade = −Σ θ_τ·g_τ (the trade-time books of lemma 1)
and S_stale = Σ (θ_{τ+D} − θ_τ)·g_τ. Check: the max relative error is 2.7e-13 (`results/summary.json`,
`delay.norm_rel_err_max`). The residual is as large as the income:
- At lr 0.05, S_stale/|W_trade| is −0.747 ± 0.004 at D=64 and −1.197 ± 0.066 at D=512.
- With the scaled step, the ratio is −0.97 ± 0.03 at D=64. At D=512, W_trade = −1890 ± 1544 crosses 0, so the ratio is meaningless there; S_stale = −19153 ± 1842.

So under delay (delayed labels, asynchronous data parallelism, pipeline staleness) "wealth = norm growth" does not
hold even with SGD. The books must be the cached W_trade.

**4.5 Self-resolution pays confidence, not accuracy. [theorem]** Settle on y′ ~ Bernoulli(p_t), the model's own
sample. Then E[ℓ_t(y′ − p_t)] = 0 and E[log p_t(y′) − log p0_t(y′)] = KL(p_t‖p0_t) ≥ 0.
Check over all 48 cells of both runs:
- |mean tangent|/SE ≤ 1.63.
- Exact mean vs KL: relative difference ≤ 3.7% at lr 0.05 and ≤ 14% in the scaled run, where KL is as small as 0.0017 (`results/summary_delay*.csv`).

Example, prng at D=0: true-label exact income −0.061, self-resolved +0.056, KL 0.055.

**4.6 What changes for LM next-token and RL. [statements; not run]**
- *LM next-token pretraining:* the label is the next data token, so D=0 per token. Books settle at once
  and lemmas 1-2 apply unchanged. There are two exceptions:
  - With stale updates (async data parallelism, pipeline or accumulation delay), 4.4's S_stale term appears.
  - With labels from the model's own samples (self-training, distillation on its own generations), 4.5 holds: books pay KL(p‖p0), not accuracy. They need an external resolver (a verifier, or later real data).
- *Sequence-level resolution (RL, verifier reward at the end):* open a per-token contract at s_t with the critic's mark
  V(s_t) and settle it at the return G_t. With λ=1 the TD errors telescope: Σ_k γ^k δ_{t+k} = G_t − V(s_t)
  (4.1; `mrp` in `results/delay.json`, max residual 5.8e-15, on a 20-state reward chain with H=50, γ=0.9 and a wrong critic).
  One-step books (λ=0) mark at the critic's next value. They differ from books marked at the true value by exactly
  γ·e(s_{t+1}), where e = V − v_true (identity residual 1.6e-15; mean bias 0.28-0.76 per seed for e ~ N(0.5, 1)). So a
  biased critic taxes or subsidizes every step of a trajectory. Known: TD(λ), Sutton 1988, and Holland's
  bucket brigade 1985 (both unverified). Marking with the actor's own critic is 4.2's own-model mark bias; mark with
  the population's price.

**4.7 Deductive settlement (ties to M5's D-prove). [known: LI]** In Logical Induction
(https://arxiv.org/abs/1609.03543), holdings in φ are worth 1 or 0 once the deductive process proves φ or ¬φ,
and are valued at market prices before that. Settling at proof time is 4.1 with the reveal time set to the proof time.
A bundle that is incoherent at the market prices (A and ¬A bought for less than 1) has a guaranteed value as soon as A∨¬A is
proven, before A is ever decided. For M5's D-prove family (never sampled), an arbitrageur's E_C should
therefore be booked at proof time; waiting for A's label books nothing, because that label never comes.

---

## Lemma 5: finite-population logical induction. E_C(t), invasion fitness, and what makes E_C sublinear

Market (`check_li.py`, PREREG check 5): one Kelly market per question, with price p = Σ w_i q_i / Σ w_i
(Beygelzimer, Langford and Pennock, https://arxiv.org/abs/1201.6655, from the scope doc) and update
w_i ← w_i q_i(y)/p(y), which conserves Σw. The traders:
- residents: pattern-0/1/2 (KT counts);
- class C: pattern-4, pattern-8 and 20 smooth price-reading calibration traders, q = σ(logit p ± bump_b(p)) over 10 bins;
- 9 constant "junk" traders, used only in the birth pool.

The family "diag" has 8 extra questions with y_t = 1[p_t < 1/2] on the market's own price. E_T(t) = Σ_s log q_T(y_s)/p_s(y_s) at
the clearing price, and E_C = max over T ∈ C. The invasion fitness φ_T is the slope of E_T for a shadow T (wealth → 0).
ε = 1e-3 and T = 8000. Conservation error is ≤ 1.6e-11 in every arm (`results/summary.json`, `li.*.max_conservation_err`).

**5.1 Members are bounded. [theorem; known: Bayes mixture / BLP]** If T trades in the population
from wealth share ε, then w_T(t) = ε·exp(E_T(t)) ≤ 1, so E_T(t) ≤ ln(1/ε) = 6.9078 for all t.
Check (P5.1 pass): the max member E is
- arm b: 6.90739;
- arm a: 6.907755278977;
- arm d: 6.907755278982234, against a bound of 6.907755278982137 (a 1e-13 float overshoot).

(`results/summary.json`, `li.{a,b,d}.max_member_traded_E`)

**5.2 Price readers need fixed-point clearing. [theorem]** A trader whose belief q_T(p) reads the price, cleared in
one pass, trades on the *pre*-price. Its traded E (belief at the pre-price, scored at p_clear) obeys 5.1. The LI-style
test, belief at p_clear scored at p_clear, is unbounded. With fixed-point clearing p* = Σ w_i q_i(p*)/Σw, a
root exists by the intermediate value theorem for continuous q_i and is found by bisection (60 iterations; PREREG said 50). Then the two
coincide and 5.1 covers the reader. This is the finite version of LI's Brouwer step and of the continuity
requirement in smooth calibration (Foster and Hart, https://arxiv.org/abs/2210.07152).
Check (P5.2 pass): arm a readers at clearing reach 48.4, 36.4 and 26.9 by seed (> 6.91), while arm b readers reach ≤ 6.907
(`li.a.max_reader_clearing_E_per_seed`, `li.b...`).

**5.3 Delay costs a factor (D+1). [theorem; known technique]** With labels D steps late, run D+1 interleaved
wealth vectors, where sub-market j trades the steps t ≡ j (mod D+1) and so only ever trades on settled wealth. 5.1 per
sub-market gives E_T ≤ (D+1)·ln(1/ε) (Weinberger and Ordentlich 2002, unverified; Joulani, György and Szepesvári,
https://arxiv.org/abs/1306.0686). Check (P5.3 pass): arm a-D64 max 447.81 against a bound of 449.00, which is 99.7% of the bound, so the bound
is nearly tight (`li.a-D64`).

**5.4 Births bound E_C by the wait until birth. [theorem for the bound; measured for the size]** A trader born
at τ with wealth ε′ has E_T(t) − E_T(τ) ≤ ln(1/ε′) from then on (5.1). With births drawn uniformly from a pool of n
at rate ρ per step, τ is geometric with mean n/ρ, so E_T ≲ φ_T·min(t, n/ρ) + ln(1/ε′). Repeated births restart a
trader that went broke before a shift, which is the fixed-share mechanism (Herbster and Warmuth 1998, unverified).

Here n = 31, so n/ρ = 3100 steps at ρ = 1e-2 and 31000 > T at ρ = 1e-3 (`results/summary_li.csv`):
- periodic, ρ = 1e-2: E_C = 472 ± 236, against a prediction of 0.157·3100 + 7 ≈ 494; arm N reached 1239 ± 188.
- diag, ρ = 1e-2: E_C = 1065 ± 444, against arm N 4535. The prediction is 897-1786, depending on whether one or both of pattern-4 and pattern-8 must be born.
- ρ = 1e-3: little change (periodic 1173, diag 3367), as predicted.

The preregistered plateau test (second-half slope < 0.2 × first-half slope at ρ = 1e-2) passes on diag (0.228 → 0.040, ratio 0.18)
and fails on periodic (0.092 → 0.027 ± 0.034, ratio 0.29): T is only about 2.6 birth waits. Fixed-point births (cfp arms) equal the c arms
to 4 digits on periodic, because pattern traders do the exploiting, not price readers.

**5.5 Covering: membership is not necessary. [theorem]** If a member R satisfies
log q_R(y_s) ≥ log q_T(y_s) − c_s on the realized path with Σ_{s≤t} c_s = o(t), then E_T ≤ E_R + Σ c ≤ ln(1/ε_R) + o(t).
Check (arm d: residents plus pattern-8, with pattern-4 as a shadow):
- periodic: E_pat4 = −907 ± 16, with slopes −0.114 → −0.116. It never exploits, which meets P5.4-d.
- diag: E_pat4 = 141 with slopes 0.0286 → 0.0065 (ratio 0.23). This is sublinear but positive, because pattern-8 only approximately covers pattern-4 there (256 contexts to fill against 16).

**5.6 Refined conjecture. [conjecture]** For a finite population of Kelly traders under BLP clearing,
with C made of price-continuous traders, E_C(t) = o(t) on every stream if and only if every T ∈ C with φ_T > 0 is
eventually *covered* at the price actually used. Covering here means one of: membership (5.1), a covering member (5.5), or a birth
process that samples T or a cover at a positive rate (5.4). A price reader counts as covered only under fixed-point
clearing (5.2); under delay, bounds scale by (D+1) (5.3).
- The "if" direction follows from 5.1-5.5.
- The "only if" direction is the conjecture. The evidence is arm N, where all three routes fail and E_C grows linearly: periodic slope 0.156 → 0.157, positive on 3/3 seeds; diag 0.544 → 0.603 with market log-loss 0.694; prng stays near 0 (E_C 0.9), since there is nothing to learn.
- Known risk: "covered" is close to a restatement. The testable content is the *rates*: φ·n/ρ for births, (D+1)·ln(1/ε) for delay, and the size of the reader gap without a fixed point.

**Implications.**
- **M5.** Report E_C both at the clearing price and as traded. Use fixed-point clearing whenever C contains price readers (calibration bins). Book D-delay with D+1 interleaved books, or report the (D+1) scale. Book D-prove at proof time (4.7).
- **K1.** Births from a pool bound E_C at about φ·n/ρ + ln(1/ε). With a 100k-trader pool, uniform births wait n/ρ steps, so births must target (for example, mutations of wealthy traders) or the plateau never arrives within a run. Put the diag family in K1 as the cheapest detector of one-pass clearing.
- **S0 and M1.** Books under any delay must be cached at trade time (4.3); norm growth is not wealth under stale updates (4.4); self-generated labels pay KL (4.5).

**What this does not show.**
- There are no LM or RL runs; 4.6 is argument plus the reward-chain identity.
- The toy learner is a per-question logistic model, not wealth-net.
- 4b passes only in an exploratory arm.
- The plateau test in 5.4 fails on periodic at T = 8000.
- 5.6's "only if" direction is unproven.

## Prior art (checked = the page was opened in this session)
- Garrabrant, Benson-Tilsen, Critch, Soares and Taylor, Logical Induction: https://arxiv.org/abs/1609.03543 (checked)
- Joulani, György and Szepesvári 2013, online learning under delayed feedback: https://arxiv.org/abs/1306.0686 (checked)
- Foster and Hart, smooth calibration and fixed points (GEB 109:271-293): https://arxiv.org/abs/2210.07152 (checked)
- Beygelzimer, Langford and Pennock 2012, learning performance of prediction markets with Kelly bettors: https://arxiv.org/abs/1201.6655 (from the scope doc; not opened here)
- Chen and Vaughan 2010, cost-function markets and no-regret learning: https://arxiv.org/abs/1003.0034 (from the scope doc; not opened here)
- Weinberger and Ordentlich 2002, delayed-feedback prediction via interleaving (unverified)
- Sutton 1988, TD(λ); Holland 1985, bucket brigade (unverified)
- Herbster and Warmuth 1998, tracking the best expert with fixed share (unverified)
- Blume and Easley 2006, market selection (from the scope doc; unverified)
