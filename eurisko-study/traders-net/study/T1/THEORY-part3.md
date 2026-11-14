# THEORY, part 3 (T1 worker 3): lemmas 6 and 7

Checked 2026-09-27 ~22:35Z. Scripts and outputs: `study/T1/w3/` (`PREREG.md`, `check_L6.py`,
`check_L7.py`, `aggregate.py`; results in `out/L6-s{1,2,3}.json`, `out/L7-s{1,2,3}.json`,
aggregated in `out/summary.json` and `out/L6_growth.csv`). Stream: S0's verbatim copy of the
wealth-net streams (`study/S0/substrate/streams.py`), seeds 1-3. Numbers: mean ± sd over the 3 seeds,
or, for exact identities, the maximum error over seeds.
Rerun: `cd study/T1/w3; for s in 1 2 3; do python3 check_L6.py --seed $s; PYTHONPATH=../../../../wealth-net/.pylib python3 check_L7.py --seed $s; done; python3 aggregate.py`

---

## Lemma 6: a fund of funds with fixed fractions is a flat mixture; only rebalancing adds power

**Setup.** Outcomes y_t; a price pi_t over outcomes (Arrow securities, exogenous or endogenous).
A leaf trader l with belief p_l,t bets full Kelly: its wealth relative is x_l,t = p_l,t(y_t)/pi_t(y_t).
A fund f holds claims on children c (leaves or funds) with fractions A[f,c] of its capital.

**6.1 Buy-and-hold is a flat Bayes mixture. [theorem]** If every fund buys at t = 0 and never
trades, then for every node f and every t

    W_f,t / W_f,0 = sum_l lambda_fl * W_l,t / W_l,0,   lambda_f = row f of (I - A)^-1, restricted to leaves,

that is, the sum over paths of the products of edge fractions. This also holds for cyclic
cross-holdings when the spectral radius of the fund-to-fund block is below 1 (the Leontief
inverse). So f is one Kelly trader whose belief is the Bayesian posterior mixture
p_f,t = sum_l w_l,t p_l,t, with w_l,t proportional to lambda_l prod_{s<t} p_l,s(y_s).
Consequences: log-regret to leaf l is at most ln(1/lambda_l) (Beygelzimer, Langford, Pennock 2012,
https://arxiv.org/abs/1201.6655), and the asymptotic growth rate is the best leaf's. With an
endogenous wealth-weighted price, a market of buy-and-hold funds has exactly the price path of the
flat market of their leaves: consolidation does not move prices.
*Proof:* the value of held shares is linear in the children's values; induct over the DAG, or solve
V = A V for the cyclic case.

**6.2 Constant rebalancing is a flat CRP. [theorem]** If every fund rebalances to fixed fractions of
its current value at every step, then x_f,t = sum_l lambda_fl x_l,t with the same path-product
weights. The root is one constant-rebalanced portfolio over the leaves. Its belief is the fixed
linear opinion pool sum_l lambda_l p_l,t. Hierarchy again adds nothing beyond the flat weights.

**6.3 Where power comes from. [theorem for the toy; known for CRPs]** A fixed-fraction hierarchy
never adds power; only the rebalancing rule changes growth. A CRP can strictly beat every leaf and
every buy-and-hold fund. Toy (truth 0.5, price 0.3, leaves 0.9 and 0.1): each leaf and the
buy-and-hold fund grow at -0.424 nats/step, while the 50/50 CRP grows at +0.087. The best CRP in
hindsight is at least as good as the best leaf, and causal universal or EG portfolios track it
(Cover 1991, "Universal portfolios", unverified; Helmbold, Schapire, Singer, Warmuth 1998, EG
portfolio, unverified). Hierarchy matters only when an inner node's allocation is a *nonlinear*
function of its own children's history (adaptive rebalancing, for example a nested Fixed Share,
Herbster and Warmuth 1998, unverified). Even then, the recursive-Bayes case is a flat mixture under
a structured prior (Context Tree Weighting, Willems, Shtarkov, Tjalkens 1995, unverified).
**[conjecture]** A nested Fixed Share over a tree equals a flat switching mixture whose transition
prior is shaped by the tree.

**Checks** (`out/summary.json`, `out/L6_growth.csv`; 10 leaves: KT context counters k = 0,1,2,3,4,6,8,
an online logistic on a 16-step window, and constants 0.9 and 0.1; price = order-0 KT base rate; T = 6000, Q = 64):

| check | result | pass |
|---|---|---|
| C6a random 3-level DAG, buy-and-hold vs flat mixture | max \|Δ log W\| 8.9e-16 | yes |
| C6a root as ONE Kelly trader with the Bayes-mixture belief vs flat | 1.8e-11 (over 6000 steps, log W up to ~2600 nats) | yes |
| C6a cyclic cross-holdings (ρ = 0.24) vs Leontief weights | 1.8e-15 | yes |
| C6b constant-rebalanced DAG vs flat CRP | 9.1e-13 | yes |
| C6c endogenous price: buy-and-hold funds vs flat leaves | max \|Δπ\| 6.7e-16 | yes |
| C6c the same with rebalancing funds | max \|Δπ\| 0.73 ± 0.08 (prices differ, as predicted) | yes |
| C6d buy-and-hold uniform minus best leaf, per question | min -3.84e-4, max -2.7e-4 ± 0.3e-4 nats/step; bound -ln(10)/T = -3.84e-4 | yes |

C6d, growth per step against the market (nats/step, mean ± sd over seeds; the full table is
`out/L6_growth.csv`):

| family | best leaf (hindsight) | buy-and-hold | online EG rebalancing | EG − best leaf | best CRP (hindsight) |
|---|---|---|---|---|---|
| ca110 | 0.134 ± 0.020 | 0.134 ± 0.020 | **0.147 ± 0.021** | **+0.0132 ± 0.0012** (96% of questions) | 0.143 ± 0.020 |
| periodic | 0.436 ± 0.012 | 0.436 ± 0.012 | 0.432 ± 0.012 | -0.0043 ± 0.0001 | 0.436 |
| prime | 0.044 ± 0.009 | 0.044 ± 0.009 | 0.040 ± 0.009 | -0.0039 ± 0.0001 | 0.044 |
| and / or | 0.017 / 0.017 | 0.016 / 0.016 | 0.015 / 0.016 | -0.0011 / -0.0003 | 0.018 / 0.019 |
| ca30 | 0.00006 | -0.0003 | -0.0025 | -0.0026 ± 0.0005 | 0.0025 (in-sample only) |
| lfsr16, pi | 0.0000 | -0.0004 | -0.003 | -0.0036 / -0.0027 | 0.0004-0.0005 (in-sample only) |

Reading: buy-and-hold is always within ln(N)/T of the best leaf, as 6.1 says. Causal rebalancing
adds real power on CA110 only, where different context lengths win on different stretches, and it
pays an O(√T)-type cost elsewhere. On CA30, LFSR and pi, the best CRP in hindsight beats every leaf
on 100% of CA30 questions, but that is in-sample memorisation: the causal EG loses there. This is the
settled premise "in-sample books pay memorisation" in portfolio form.

**What this means for the study.**
- Kelvin's "a big trader's wealth is the sum of its constituents", with fixed hierarchical or partial
  ownership, is case 6.1. Consolidation is accounting: it changes no prediction, no price and no
  selection. It is a view, not a mechanism. Additive a·dR/da books consolidate the same way.
- **K1**, funds of funds: fixed fractions must reproduce the flat economy exactly. Use
  `w3/check_L6.py:bh_fund_values` and `flat_weights` as an assertion (tolerance 1e-9). Only rebalancing
  funds can change survivors or Brier; report EG or universal rebalancing, not only uniform
  fractions, which cost growth (the uniform CRP gets 0.20 vs 0.44 on periodic).
- **M8** gating and any per-step wealth-driven allocation are rebalancing rules, so this is where any
  power has to come from. A wealth-proportional *linear* pool in probability space is exactly
  buy-and-hold (Bayes), with nothing beyond Bayes. **M6**(d) pools in log-odds (a geometric pool) and is
  outside this lemma. A neural logit sum is a product of experts, not a mixture.

---

## Lemma 7: the top-k router gradient is gate-weighted relative revenue under censored feedback; the router is a broker

**Setup.** h = sum_{i in S} g_i E_i(x), S = top-k(z) (piecewise constant, held fixed a.e.).
r_i = <E_i(x), -dL/dh> is the revenue expert i *would* earn at unit gate; it is defined for every
expert, but observed only for i in S. R = sum_S g_i r_i = <h, -dL/dh> is the revenue paid to the block.

**7.1 Router gradient. [theorem; two-line softmax product rule]** u_i := -dL/dz_i equals:

| convention | u_i |
|---|---|
| (i) softmax renormalised over S | g_i (r_i − R) for i in S; 0 outside |
| (ii) softmax over all N, gates of S not renormalised | g^N_i (r_i·1[i∈S] − R): outsiders get −g^N_i R, their revenue imputed as 0 (censored) |
| (iii) sigmoid, normalised over S (the DeepSeek-V3 form, unverified detail of https://arxiv.org/abs/2412.19437) | (1 − σ(z_i)) g_i (r_i − R) for i in S |
| (iv) top-1 renormalised | 0 (the gate is identically 1; the router never learns) |
| (v) top-1 not renormalised (Switch, https://arxiv.org/abs/2101.03961) | selected: g(1−g) r_sel; outsiders: −g^N_i g_sel r_sel |
| dense softmax (full information) | g_i (r_i − R): Hedge / replicator step on revenue |

Check C7a: the closed form matches autograd to at most 8.6e-16 relative error for all six conventions,
3 seeds (`out/summary.json`, keys `C7a_*`).

**7.2 Zero-sum and mass transfer. [theorem]** For softmax gates, sum_i u_i = 0 (shift invariance),
so router updates are pure reallocation (C7b: ≤ 1.0e-18, 3 seeds). In (ii) the routed experts jointly gain
R(1 − sum_S g) and the outsiders jointly lose the same (checked ≤ 4.3e-19). When the routed
portfolio earns (R > 0), logit mass flows from *every* unrouted expert to the routed ones, whatever
the outsiders' own r_i. Rich-get-richer is an identity, not a tendency. Sigmoid gates are not
shift-invariant and are not zero-sum (|sum| 3.3e-4 to 7.9e-4 against a gradient scale of 4.4e-4 to 8.1e-4, 3 seeds). This refinement
was written into `check_L7.py` before the run; PREREG's C7b criterion applies to softmax
conventions only.

**7.3 The router bias is a censored Hedge log-wealth. [theorem under SGD]** With z = W_r x + b and
plain SGD at rate η, b_i(T) − b_i(0) = η Σ_t Σ_tokens u_i: cumulative gate-weighted relative revenue,
with unobserved revenue imputed as zero. Under Adam it is sign-normalised and not a wealth. The
DeepSeek loss-free bias (https://arxiv.org/abs/2408.15664) enters only the top-k selection and is
updated by load, not revenue (recollection, unverified). It is a capacity price, orthogonal to this ledger.

**7.4 Censoring and lock-in. [theorem for a bias-only router with top-1 not renormalised; the rest is
empirical]** Under (v), if the incumbent's revenue stays positive, its logit rises and every
outsider's falls, so the router keeps the *first* expert with positive revenue forever. Toy C7e
(16 experts, μ_i ~ U(0.1, 1), noise 0.3, η = 0.1, 5k steps, 100 instances per seed; `out/summary.json` key `C7e`):

| arm | final argmax = best expert | best routed, last 1k steps | regret, last 1k |
|---|---|---|---|
| top-1 not renormalised, censored | 6.7 ± 2.5% (chance 6.25%) | 6.7 ± 2.5% | 0.378 ± 0.014 |
| top-1 + Exp3 shadow credit | 52 ± 4% (the rest are near-ties) | 52 ± 4% | **0.023 ± 0.002** |
| top-2 renormalised, censored | 88 ± 3% | 98 ± 2% | 0.089 ± 0.002 (includes the 2nd slot; not comparable) |

So collapse from censoring is sharp for top-1 and for non-renormalised gates while R > 0. From
(ii), two incumbents can both rise while their r_i > R, so the same lock-in holds (derived, not run).
It is not sharp for renormalised top-k ≥ 2 with a stationary bias-only router: relative revenue inside
S evicts the weaker incumbent (**exploration by eviction**). Context-dependent routers and experts
that keep learning are M7's question, not this toy's.

**7.5 Exp3 shadow credit is unbiased for the full-information gradient. [theorem; standard
importance weighting]** Sample one unrouted expert J with probability q_J and run it at zero gate
(+1/k FLOPs, no effect on h). Set r̂ = r on S, r̂_J = r_J/q_J, and 0 elsewhere. Then
û_i = g^N_i (r̂_i − Σ_j g^N_j r̂_j) has expectation equal to the dense-softmax gradient. C7d: max |z|
over 8 experts = 1.17, 1.47, 1.82 (seeds 1-3; 1000 draws × 256 tokens). The censored top-2 gradient
is 43 ± 23% away (relative L2; 35%, 68%, 24% by seed) from the full-information one. Exp3: Auer, Cesa-Bianchi, Freund,
Schapire 2002 (unverified).

**7.6 Gate convention: the router is a commission-free broker. [theorem (Euler) + convention]** At
each gate, the tangent books sum to Σ_i g_i ∂R/∂g_i + Σ_i ⟨E_i, ∂R/∂E_i⟩ = 2R, because the product is
homogeneous of degree 2 in (g, E). C7c: ratio in [2 − 1.6e-13, 2 + 7e-14] for every convention; the
SwiGLU (gate·up) factor books are 2.0 × the product books. **Convention:**
- **(a) Broker, adopted.** Expert i is paid g_i r_i. This conserves: the sum is R (residual ≤ 2.6e-18),
  and it keeps conserving into the expert's neurons plus output bias (residual ≤ 2.6e-18). The router
  earns nothing from the product. Its ledger is the zero-sum reallocation u_i of 7.1-7.2: routing
  preference, not wealth.
- **(b) Euler half split, sensitivity only.** It conserves (≤ 2.2e-18) but pays the router g_i r_i/2
  of *level* revenue for the expert's work.
- **Router skill score, not wealth.** The router's tangent logit book z·(−∂L/∂z) = Cov_g(z, r) over S
  (identity checked ≤ 2.6e-18). It is positive exactly when higher logits went to higher-revenue
  experts, and it lies outside the conserved flow.
- **Scope.** The same doubling hits GLU FFNs and QK attention. Wealth-net's ReLU MLP avoids it.

**7.7 Routed-expert wealth is sleeping-expert wealth. [known]** Expert books accrue only on routed
tokens. This is the specialists / sleeping-experts setting (Freund, Schapire, Singer, Warmuth 1997,
unverified). Cross-expert wealth comparisons (M7 bankruptcy, K1) must normalise by routed-token count
or use shadow credit.

**For M7.** Stage 0 is done here: `w3/check_L7.py` (functions `gates`, `closed_form`, `check_exp3`)
can be imported. Use convention (a) with (b) as sensitivity. Expect lock-in in no-balancing arms only
for top-1 or non-renormalised gates; top-2 renormalised explores by eviction. The Exp3 arm estimates
the dense-softmax gradient without bias.

**Prior art found (search, 2026-09-27; summaries not verified by reading the papers).**
- "When Are Experts Misrouted?" (https://arxiv.org/abs/2605.07260) frames the missing counterfactual
  losses of unrouted experts, the closest match to "censored feedback".
- ProbMoE (https://arxiv.org/abs/2606.01509) notes that gradients reach only the selected softmax
  weights.
- Router-noise exploration: https://arxiv.org/abs/2609.13058.
- Limited-observation experts regret (Kale 2014, via https://arxiv.org/abs/2606.14929).
- Noisy top-k gating (Shazeer et al. 2017, https://arxiv.org/abs/1701.06538, unverified id).
- SparseMixer (https://arxiv.org/abs/2310.00811, unverified id).

No source found states 7.2's exact mass-transfer identity, 7.6's broker convention or the
eviction/lock-in split (search only; L1 should confirm).
