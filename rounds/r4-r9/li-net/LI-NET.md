# Toward a logical induction network (eurisko, 2026-10-01)

Sources read: Garrabrant et al., Logical Induction, abridged (sections 3-4) and extended
(section 5, the construction: MarketMaker 5.1, Budgeter 5.2, TradingFirm 5.3, LIA 5.4).
Local copies: `LogicalInductionAbridged.pdf`, `LogicalInduction.pdf`, text in `abridged.txt`, `full.txt`.

## 1. The criterion, in plain terms

- Objects: sentences φ; a **market** = a sequence of pricings P_1, P_2, ... with P_n(φ) ∈ [0,1]
  (day n's probability of φ); a **deductive process** D_1 ⊆ D_2 ⊆ ... (the theorems proven by day n).
- **Trader**: on day n, sees the price history P_≤n and outputs buy/sell orders for sentences.
  The orders must be *continuous* functions of today's prices and *polynomial-time* computable.
- **Plausible worth**: a trader's holdings are valued by every truth assignment W that is
  propositionally consistent with D_n (W(φ)=1 for proven φ, Boolean algebra respected, everything
  else free). This is how unproven sentences get valued: by the worst and best worlds still open.
- **Exploits**: the trader's plausible worth over all days and all plausible worlds is bounded
  below but unbounded above (unbounded gain from bounded risk).
- **Criterion**: no efficiently computable trader exploits the market. (Def. 3.0.1)
- Consequences (sec. 4): convergence; limit coherence; e.c. sequences of theorems get price → 1
  in a timely manner (P_n(φ_n) → 1, before proofs arrive); pseudorandom sequences get their
  frequency; linear constraints that hold in all consistent worlds are respected in a timely manner
  (affine coherence); introspection, paradox resistance, no expected net update, self-trust.

## 2. How the paper's algorithm works (the part that matters for a network)

LIA_n := MarketMaker_n( TradingFirm_n(P_≤n−1), P_≤n−1 ). Three pieces:

1. **TradingFirm** (5.3): enumerate all e.c. traders T^k; trader k is switched on from day k
   (S^k_n = T^k_n if n ≥ k, else 0); each is wrapped by Budgeter with budget b and weighted
   2^−k−b, summed into one supertrader. More traders are active as n grows.
2. **Budgeter** (5.2): if a trader's plausible worth ever fell to −b, its trades become 0 forever;
   otherwise its day-n trade is scaled down so that no plausible world can push it below −b.
   Wealth is "b plus worst-case plausible P&L", and a bankrupt trader is silenced. This is the
   only "learning": traders are not changed, they are weighted and killed.
3. **MarketMaker** (5.1): given the supertrader's strategy T_n (a continuous function of today's
   prices), set today's prices at a **fixed point** of
   fix(V)(φ) = clip(V(φ) + T_n(P_≤n−1, V)[φ]) : raise the price of anything the trader would buy,
   lower anything it would sell, until the trader would only buy at ~1 or sell at ~0. At that point
   the trade gains at most 2^−n in *every* world. Existence by Brouwer; the paper finds it by brute
   force (impractical); any fixed-point solver works in practice.

So the market **never lets a trader profit**: prices are set *before* the trade so the trade is
worthless. Traders are not paid for being right; they are made redundant by having their
information priced in. The inductor is the fixed point of its own trader population.

This is the same mechanism as Vovk's defensive forecasting (Forecaster picks forecasts so that
no continuous Skeptic strategy gains; Schauder fixed point) and Foster-Vohra calibration; the
community notes defensive forecasting as a precursor (sources: arXiv cs/0505083, cs/0506007,
2506.11848 "In Defense of Defensive Forecasting", Perdomo & Recht 2025). The only public code is
Alex Flint's `logicalinduction` (github.com/alexflint/logical-induction, "simplicity over
efficiency", one trader at a time, hand-coded features). No published neural version exists
(searched 2026-10-01).

## 3. Mapping to a network whose repeated blocks are time

| paper | network |
|---|---|
| day n | block application n (same block, applied n times; depth = time) |
| belief state P_n | the block's output: a vector of prices over the current sentence set S_n |
| deductive process D_n | an external input stream: which sentences became proven by block n |
| sentence set S_n (finite, growing) | a growing set of question tokens with structure (A, ¬A, A∧B, "prg(i)=k", ...) |
| trader T^k | a small parametric sub-network reading P_≤n (and sentence features), emitting orders; continuous by construction |
| trader k active from day k | trader slots unlocked as depth grows (more traders at deeper blocks) |
| Budgeter | per-trader wealth w_k = b − worst-case plausible loss so far; trades scaled by min(1, w_k / plausible loss); w_k ≤ 0 silences |
| MarketMaker | a fixed-point layer: solve P_n = clip(P_n + Σ_k T^k(P_≤n)) by Newton/DEQ; differentiable implicitly |
| plausible worlds PC(D_n) | an LP/ILP over truth assignments consistent with proven sentences (small) or sampled worlds |
| exploitation | max over traders of plausible-worth growth with bounded minimum |

**What is fixed by the criterion and what is free.** Fixed: prices come from the fixed point
against the traders, traders are budgeted, worth is plausible worth over PC(D_n). Free: *where
the traders come from.* The paper enumerates all of them; a network can only have finitely many,
so the gap to the criterion is exactly the gap between the trader class the network hosts and
"all e.c. traders". That gap is what progress means.

## 4. What "progress toward the criterion" means, measurably

Define a trader class C (e.g. all small networks of a given size that read prices and sentence
features). Exploitability by C at depth N:
E_C(N) = sup_{T ∈ C} [ max plausible worth of T over days ≤ N ] subject to its plausible worth
never below −1. Progress = E_C(N) grows sublinearly (bounded) for larger and larger C. Concretely:
- **The market is a defender; traders are attackers.** Train attackers (by gradient, on *their*
  plausible worth: this is allowed, the criterion is about the market, not how traders are found)
  to exploit the current market. Set the market by the fixed point against the attackers. If a
  trained attacker still gains unboundedly, the market is not an inductor over C; add it to the
  population (it becomes T^{k+1}) and re-solve. This is the LIA with learned traders instead of
  enumerated ones, and the same shape as adversarial training / Vovk's Skeptic.
- Property tests, each a theorem of sec. 4 turned into a metric at finite depth: timely provability
  induction (P_n(φ_n) for an e.c. theorem sequence revealed late by D), pseudorandom frequency,
  affine coherence on constraint families (prg0+prg1+prg2 = 1; "3 of 7 true"), convergence of
  P_n(φ) in n, no expected net update (P_n(φ) vs E_n P_{f(n)}(φ)).

Important difference from everything we ran so far: **traders are not paid by settlement.** In
LIA no money changes hands at all; the market pre-empts them. Budgets only ever go down. The soup
and wealth-net paid traders for being right; an inductor silences the ones that are wrong and
prices in the ones that are right before they can collect. That is why the previous experiments
were not tests of this criterion.

## 5. Three experiment options (Kelvin picks)

**A. Faithful LIA with learned attackers (closest to the criterion).**
Sentence set: propositional atoms with e.c. structure (theorem sequences, pseudorandom families,
constraint families, a few deductive rules), D reveals proofs late. Traders: a population of tiny
nets (price readers), budgeted exactly as in 5.2, fixed-point market exactly as in 5.1 (Newton on
the clip map). Attacker loop: at each depth, train a new tiny net to exploit; add it. Metrics:
E_C(N) curve; the sec. 4 property tests. Cost: CPU, days. Risk: fixed-point solving with many
traders is the hard part; the paper's brute force is replaced by Newton/DEQ, which may fail on
non-monotone traders (continuity gives existence, not uniqueness).

**B. Defensive-forecasting network (the engineering-friendly version of A).**
Same as A but the market is a single network trained to minimise the *maximum plausible gain of
the attacker population* at the fixed point (a GAN: market = forecaster, traders = skeptics), with
the block repeated over days and the trader population growing with depth. Weaker guarantees
(approximate, not exact pre-emption) but scales and uses standard tooling. Metrics as in A.

**C. Property-first (cheapest, diagnostic).**
Skip exploitability; build only the fixed-point market layer over a fixed hand-written trader
class (the paper's example traders: ¬¬φ vs φ arbitrage, theorem-sequence buyers, frequency
traders, constraint traders) and measure which sec. 4 properties appear at finite depth and how
fast. Tells us what the fixed-point layer alone buys before any learning. Cost: hours.

My recommendation: C first (two days, no GPU), then A. B only if A's fixed point does not scale.

## 6. Things I am not sure about

- Whether Newton/DEQ finds the Brouwer fixed point reliably for non-monotone (momentum-like)
  traders; may need the continuity slack the paper uses (traders with finite price precision).
- How to make D (the deductive process) rich enough to matter without a theorem prover: a
  propositional D (unit propagation over a growing clause set) is the honest minimum.
- Plausible worlds grow exponentially with atoms; the LP relaxation over PC(D_n) is what the
  paper's worst-case uses, and whether the relaxation is tight enough for exploitation measurement
  is untested.
