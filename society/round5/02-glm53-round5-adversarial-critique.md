# Society of Thinkers — Round 5 Adversarial Red-Team
**Econometrica vs. ICLR Smoking-Gun Audit & R17 Decentralized Market-Training Protocol**

---

## 0. Verdict (TL;DR)

1. **Theorem R5.1 as stated is false as a DSIC claim.** It is provably only an *ε-DSIC* (approximate clearing), *asymptotic-DSIC* (behaviorally verified $\hat G$), *substitutes-regime* (submodular loss game) result. Under complementarity (the XOR pairs you yourself propose in R17-E2) it runs a budget deficit exactly equal to the complementarity surplus and is shill-exploitable — this is Ausubel–Milgrom, and no $\rho$-tuning fixes it. The theorem worth proving is the *characterization* (R5.1′, §1.5), which is honest and, notably, the Green–Laffont trilemma corner you occupy is forced, not chosen.
2. **R17-E2 is the only plausible ICLR smoking gun, and its central risk is a theorem-level obstruction you have not named:** XOR payoffs are the canonical *non-separable* security (Ostrovsky 2012); cold-start failure is not an implementation bug, it is the predicted non-aggregation regime. The fix (heterogeneous seeded priors + information bounties + repeated wealth reallocation) must be pre-registered or the experiment measures nothing.
3. **R17-E3 as currently specified mathematically guarantees infant mortality:** with flat rent, $t^* = W_m(0)/c_{\text{rent}}$ is a certain-death clock for any module whose discovery time exceeds it. Flat rent and deep-feature discovery are incompatible; a value-of-information rent (§3.2) is mandatory, not optional.

---

## 1. Critique of Theorem R5.1 (VCG-Gershgorin-LOO)

Settlement rule under audit:
$$W_m \;=\; \beta_0\big[\ell(\mu^*_{-m}, y) - \ell(\mu^*_{\text{full}}, y)\big] - c_{\text{rent}}, \qquad w^*_{-m} = \big(\hat\Lambda_{-m} + I + \rho\,\hat G_{\text{off},-m}\big)^{-1}\mathbf{1}.$$

### 1.1 What DSIC actually requires, and where each condition fails

Exact DSIC of a Clarke-pivot settlement requires **all five** of:

- **(C1) Quasilinear utility.** Fails in R17-E3: with a bankruptcy floor, module utility is S-shaped in wealth — risk-averse mid-range, *risk-seeking* near bankruptcy. Near-dead modules take maximal-variance positions (limited-liability asset substitution, Jensen–Meckling 1976). The settlement is DSIC only for modules far from the floor. Fix: payouts as equity shares, not cash transfers, plus posted collateral bonds.
- **(C2) Direct revelation = executed behavior.** Holds only if every payment-relevant statistic is recomputed by the clearinghouse from behavior. See §1.2 — currently violated by $\hat k_m$.
- **(C3) Exact clearing.** The Gershgorin linear solve is a *one-step linearization* of a nonlinear re-clearing fixed point. Approximate VCG with clearing residual $\varepsilon$ yields manipulation gain $\le \mathrm{Lip}_\mu \cdot \varepsilon \cdot \beta_{\max}$: this is **ε-DSIC, not DSIC**. Exact re-clearing of a general module economy is NP-hard (combinatorial), so ε-DSIC is the honest ceiling.
- **(C4) Submodularity of the loss game.** $L(S \cup \{m\}) - L(S)$ nonincreasing in $S$. Clarke payments are individually rational and weakly budget-balanced **iff** the substitutes condition holds. XOR-complementary module pairs violate it (§1.3).
- **(C5) Unilateral deviations only.** DSIC says nothing about coalitions with transfers; VCG's failure here is structural (Ausubel–Milgrom 2006: revenue non-monotonicity, shill-bidding).

**Sharpest possible statement of the current gap:** R5.1 conflates two orthogonal claims smuggled under one theorem. The Gershgorin preconditioner gives *numerical stability of clearing* (§1.4); DSIC comes *entirely* from the pivot payment structure plus C1–C5. Proving invertibility of $(\hat\Lambda + I + \rho \hat G_{\text{off}})^{-1}$ proves nothing about incentives.

### 1.2 Key-direction falsification and the behavioral-$\hat G$ fix

**The attack.** $\hat k_m = |\hat\delta_m| / \|\hat\delta_m\|_2$ is a *declared* statistic. An agent reports $\hat k_m$ orthogonal to all competitors ($\hat G_{mj} \to 0$), escapes the crowding penalty $\rho \hat G_{\text{off}}$, receives inflated weight $w_m$, while actually correlating on-target. Worse — the **hold-up/sabotage amplification**: payment is $\ell(\mu^*_{-m}) - \ell(\mu^*)$. A module that behaves badly *while looking good* degrades $\ell(\mu^*_{\text{full}})$, and since that term enters with negative sign, **sabotage increases the payment**. Key-falsification is therefore not merely profitable; it converts the pivot term into a hostage mechanism. If $\hat G$ (or $\hat\Lambda$) is self-reported at all, the mechanism is not even Bayesian-IC, because the lie is undetectable.

**The fix (necessary and essentially sufficient).** Close the declaration channel entirely: every payment-relevant statistic must be a clearinghouse *measurement* on behavior:

$$\hat G_{mj} \;=\; \widehat{\mathrm{Corr}}_{\mathcal{D}_{\text{audit}}}\big(r_m(x),\, r_j(x)\big), \qquad r_m(x) \;:=\; \mu^*(x) - \mu^*_{-m}(x),$$

i.e., the correlation of *influence vectors* (perturbation probes: inject $\delta_m + \epsilon\zeta$ with random $\zeta$, read the market output change — black-box, no weight access needed). Note the implementation synergy: the same $M$-sweep LOO pass that computes payments also yields $r_m$, hence $\hat G_{mj}$ *and* self-quality $\hat\Lambda_{mm} = \widehat{\mathrm{Var}}(r_m)$-type statistics, at zero marginal compute. Update with shrinkage $\hat G^{(t+1)} = (1-\alpha)\hat G^{(t)} + \alpha\,\widehat{\mathrm{Corr}}_{\text{audit}}$ so persistent falsification is detected within $O(\log(1/\epsilon)/\alpha)$ rounds. One-shot falsification gain is then bounded by the estimator's detection half-life, and the honest claim degrades correctly to **asymptotic DSIC**: strategyproof in the limit of repeated audited play. With behavior-only statistics, sabotage self-punishes: bad behavior lowers measured $\hat\Lambda_{mm}$, weight $\to 0$, payment $\to -c_{\text{rent}}$, bankruptcy. That closes the loop.

"Verifiable readout-salience over $V \cup Z_{\text{aux}}$" is acceptable *only* if the salience regression is fit on clearinghouse-chosen audit states the module cannot preview; otherwise modules can overfit the audit probe — the meta-gaming failure of every evaluator-fixed benchmark.

### 1.3 Clones vs. shills — the distinction that decides R17-E1

- **Duplicate clones (substitutes).** Clone annihilation is real and correct: if $A' \approx A$, then $\mu^*_{-A} \approx \mu^*$ (the clone re-clears into the vacancy), so $W_A \to -c_{\text{rent}} < 0$. The pivot term annihilates substitutes for free. *Conditional on behavioral $\hat G$ detecting $G_{AA'} \approx 1$.* With self-reported keys, two clones simply report orthogonal keys and the annihilation never fires.
- **Complementary shills (the Vickrey complementarity problem).** Two XOR agents $A$ (detects $x_1 {\oplus} x_2$), $B$ (detects $x_3 {\oplus} x_4$), $Y = A \oplus B$. Each alone: marginal product $a$; joint: $2a + \gamma$, $\gamma > 0$ the complementarity surplus. Pivot payment to each is $\approx a + \gamma$; total payout $2a + 2\gamma$ against total surplus $2a + \gamma$:

$$\text{Budget deficit} \;=\; \gamma \quad \text{— the surplus is exactly double-counted.}$$

  Consequences: (i) the clearinghouse runs a structural loss on every complementary pair; (ii) an agent splitting into two complementary shills extracts $\gamma$ more than truth (VCG revenue non-monotonicity — splitting *increases* total payments); (iii) this is precisely Yokoo–Sakurai–Yokoo false-name-proofness failure: **clone annihilation restores false-name-proofness in the duplicates regime and does nothing in the complements regime.** R17-E1 must test both arms separately or it tests only the easy half.
- **Impossibility boundary (the honest theorem).** Green–Laffont (1977): DSIC + efficiency + weak budget balance cannot coexist for general quasilinear preferences. d'Aspremont–Gérard-Varet: drop DSIC for Bayesian-IC and BB is achievable. Cavallo (2006): keep DSIC + efficiency and redistribute *positive surplus* — but positive surplus exists **only in the substitutes regime**. So R17's corner of the trilemma is forced: **either the module economy is restricted to substitutes (then Cavallo-style redistribution gives budget-neutral DSIC), or complementary modules must be internalized by an explicit fusion/merger operation** (mitosis's dual), or you concede a soft budget constraint and cap payments at the deficit bound $\gamma$.

### 1.4 What the Gershgorin preconditioner actually buys

If $0 \le \hat G_{mj} \le 1$ enters as a penalty and $\rho \max_m \sum_{j \ne m} \hat G_{mj} < 1 + \min_m \hat\Lambda_{mm}$ (strict diagonal dominance, Levy–Desplanques/Gershgorin), then $A = \hat\Lambda + I - \rho\hat G_{\text{off}}$ is a nonsingular M-matrix, $A^{-1} \ge 0$ elementwise (weights stay positive), and the Neumann expansion $\sum_k \rho^k (D^{-1}\hat G_{\text{off}})^k$ converges, giving the first-order crowding correction $w_m \approx w_m^{(0)} - \rho\sum_j [D^{-1}\hat G_{\text{off}}]_{mj} w_j^{(0)} + O(\rho^2)$. Note the sign: as written in the audit ($+\rho\hat G_{\text{off}}$), you need to state whether off-diagonals are subtracted; an M-matrix structure (penalty sign) is what guarantees $w^* > 0$. This is a *stability and positivity* claim about the clearing. It is a lemma, not a theorem about incentives, and the paper must be structured as: Lemma (Gershgorin stability) + Theorem (payment/IC under C1–C5) + Impossibility (complements).

### 1.5 R5.1′ — the theorem you can actually prove

> **Theorem (R5.1′).** Consider a module economy with loss game $L: 2^M \to \mathbb{R}$, behavioral-only statistics (C2), clearing residual $\varepsilon$ (C3), quasilinear utilities (C1). Then: **(i)** the re-cleared LOO settlement is ε-DSIC with manipulation gain $\le \mathrm{Lip}_\mu\,\varepsilon\,\beta_{\max}$; **(ii)** if $L$ is submodular (substitutes), payments are individually rational, weakly budget balanced, and false-name-proof against duplicate shills; **(iii)** if $L$ has complementarity surplus $\gamma$ on any pair, total payments exceed total surplus by exactly $\gamma$ and the mechanism is not false-name-proof against complementary shills; **(iv)** no settlement rule in this class achieves DSIC + efficiency + weak budget balance on complements (Green–Laffont).

Additionally, the separation from the Wolpert–Tumer identity is real but must be stated precisely: $\nabla_{\theta_m}[L(M_{-m}) - L(M)] = -\nabla_{\theta_m} L(M)$ holds when $M_{-m}$ is a *dead* market; the re-cleared counterfactual makes the payment a **finite difference over two distinct fixed points**, which is why it is not a gradient and escapes both the TBPTT(1) identity and the sign census (§3.1). This finite-difference lemma — not the market language — is the load-bearing mathematical content.

---

## 2. Critique of R17-E2 (Zero-Global-Backprop Decentralized Training)

### 2.1 The Adjoint-Necessity Lemma — restate the separation honestly

Any mechanism that assigns credit through a composition $f_M \circ \cdots \circ f_1$ must estimate sensitivities $\partial L / \partial h_m$ *somehow*. The backward pass solves $(I - J^\top)\lambda = \nabla_y L$ by Gauss-elimination-in-topological-order; a price relaxation that iterates quotes to a fixed point solves *the same linear system* by Jacobi/Gauss–Seidel iteration; equilibrium propagation (Scellier–Bengio 2017) approximates the same adjoint by contrastive fixed points; synthetic gradients (Jaderberg et al. 2017) *learn* the same adjoint. **There is no escape from the adjoint — only different estimators with different bias/variance/communication/incentive profiles.** Therefore D6 ("separation from backprop") cannot mean "does something backprop cannot"; it can only mean:

1. **Locality:** each module observes only *prices/scalars* of its inputs/outputs — no autograd graph, no per-sample gradients of others, no weight access. (Architectural claim, testable by communication-bandwidth accounting.)
2. **Incentives:** modules optimize local profit $\ne$ global loss; the research question is *when local profit-seeking aligns with global loss reduction* (answer: when prices are strictly proper scoring-rule based and the security space is separable — Ostrovsky 2012 — see §2.3).
3. **Rate:** scalar/price-only feedback is bandit-like; expect $\sqrt{T}$-type slowdown vs. full gradient (Flaxman–Kalai–McMahan bandit regime). Pre-register the sample-complexity gap rather than being caught by it in review.

If the paper does not concede this lemma up front, one sharp reviewer note ("your market is Jacobi-decomposed backprop with extra steps") kills it.

### 2.2 The exact mechanism for learning $I(X_m; Y) = 0$ features

Marginal MI is zero, so no signal exists in the $Y$-contract for module $m$ with inputs $(x_1, x_2)$. The *only* way detached modules learn the parity bit $z = x_1 \oplus x_2$ is if **$z$ is itself a traded asset**: $H(z \mid X_m) = 0$, so module $m$ can mint certain profit on the $z$-market; downstream traders who need $z$ for $Y$ bid its price off the prior; the price trajectory of $z$ is then a *sufficient statistic* for the usefulness of $z$ — that price is the credit-assignment channel that replaces the backward pass. The gradient signal for $\theta_m$ is entirely local: $\nabla_{\theta_m} \ell_{\text{local}}(\text{quote}_m(X_m), z_{\text{price}})$ — no cross-boundary autograd anywhere. This is the correct and only mechanism; it should be stated as the core of M3, and it implies **M3 must create and fund auxiliary markets on $Z_{\text{aux}}$** — which couples R17-E2 to R17-E3 (asset discovery by mitosis). Pre-register the dependency.

### 2.3 Cold start is a theorem-level obstruction, not a bug

All auxiliary prices at zero prior $\Rightarrow$ no demand from downstream (who haven't discovered they need $z$) $\Rightarrow$ no supply from upstream (who earn nothing quoting $z$) $\Rightarrow$ degenerate no-trade equilibrium. This is not an accident: **XOR payoffs are the canonical non-separable security**; Ostrovsky (2012) proves non-separable markets need not converge to informative prices. Your headline task (Hidden-Parity) is, formally, in the regime where prediction markets are *known to fail* in one-shot play. The known escape (Forsythe–Lundholm 1990; Ostrovsky's dynamic extensions): **belief heterogeneity + repeated rounds + wealth reallocation**. Concrete, pre-registrable fixes:

1. **Heterogeneous seeded priors:** initialize each auxiliary market's LMSR liquidity parameter from a small random probe ensemble (random projections of the state). This is the market analog of random init breaking symmetry — but note it decides *which* assets get funded, so seed diversely and let bankruptcy prune.
2. **Information bounties:** pay the first trader to move an auxiliary price by $\Delta$ a bonus proportional to the *conditional information gain* $I(Z_{\text{aux}}; Y \mid \text{trade}) - I(Z_{\text{aux}}; Y)$. This funds exploration of $Z_{\text{aux}}$ exactly where it matters. (Budgeted; see §3.2 for the anti-farming cap.)
3. **Two-sided quoting obligations** on newborn modules during an incubation window (they must post prices on their output assets, forcing price discovery).

### 2.4 Two-timescale stability and the pinned shadow market

- **Borkar (1997) two-timescale condition.** $\theta_m$ (module weights, fast) and $\beta_m$ (stakes/wealth, slow) must satisfy $\gamma_t / \alpha_t \to 0$ with $\sum\alpha_t = \infty$, $\sum\alpha_t^2 < \infty$: e.g., $\alpha_t = \alpha_0 t^{-2/3}$, $\gamma_t = \gamma_0 t^{-1}$. If $\beta$ tracks stale $\theta$ (right ratio wrong constants), wealth lags quality; if $\beta$ moves at $\theta$-speed, wealth chases minibatch noise and oscillates — signature: wealth oscillation period consistent with the $\alpha/\gamma$ lag, worth pre-registering as a diagnostic.
- **The STUDY-LIR §3b–3d result is binding on M3:** descending on raw PDG inconsistency collapses $\beta_i \to 0$; the exact local outcome-gradient requires the *pinned copy* of the cleared market $\mu^*$ attended with negative weight $-\beta_i$. M3's spec must include the pinned shadow market explicitly, or M3 will reproduce §3b's stake collapse and the experiment measures the wrong thing.
- **Multi-step relaxation is non-negotiable.** THEORY-4's cosine-1.0000000 identity (1-step spot-price LMSR $\equiv$ TBPTT(1)) means M3 must run the price fixed point $K \ge$ task-depth iterations per update. Pre-register a diagnostic: M2 (naive LMSR, 1-step, detached) should replicate the TBPTT(1) identity (trajectory cosine vs. TBPTT(1) $\ge 0.99$); if it does, the harness is calibrated; M3's departure from M2 is then attributable to relaxation depth + IC settlement, cleanly.

### 2.5 Failure-mode checklist (pre-register detectors for each)

| Mode | Detector |
|---|---|
| Degenerate no-trade equilibrium | aux-price entropy flat from step 0 |
| Stake collapse (§3b recurrence) | median $\beta_m \to 0$ while loss plateaus |
| Price oscillation (two-timescale) | wealth PSD peak at predicted $\alpha/\gamma$ lag |
| Shadow-clone credit theft | behavioral $\hat G_{mj} > 0.9$ pairs with asymmetric payouts |
| Settlement noise swamps signal | paired-CRN variance of $\ell_{\mu^*_{-m}} - \ell_{\mu^*}$ > payment scale |

---

## 3. Critique of R17-E3 (Market-Driven Architecture Learning)

### 3.1 Why LOO wealth escapes the 25.6% sign flip — and what it costs

$|w_i \cdot g_i|$ fails on (a) saturated units ($|\sigma'| \to 0$ shrinks the gradient of causally important units — a magnitude error), and (b) substitute co-adaptation (removing $i$ lets $j$ take over, so $w_ig_i$ over-credits, and co-adapted gradients can even point the wrong way — the sign error). The re-cleared LOO payment is a **global finite difference between two fixed points**: it never evaluates $\sigma'$ (immune to saturation) and it *includes re-equilibration* (immune to substitutes by construction — the substitute re-clears into the vacancy and the measured loss gap is small). Two non-optional conditions:

1. **Common random numbers:** evaluate $\ell(\mu^*_{-m})$ and $\ell(\mu^*)$ on *identical* minibatches, else the difference of two noisy losses has variance $\approx 2\sigma^2/n$ and small modules' wealth is pure noise.
2. **Compute-matched baselines.** LOO sweeps cost $O(M)$ forwards per settlement. Comparing against $|w\cdot g|$ at unequal compute is referee bait; the fair baselines are **OBD (LeCun 1990) / OBS (Hassibi–Stork 1993)** — second-order saliency $\approx \frac{1}{2} w_i^2 / [H^{-1}]_{ii}$, the classical finite-difference-flavored pruner — and Carreira-Perpiñán-style one-shot compression. If LOO wealth doesn't beat OBD/OBS at matched compute, the market framing adds nothing over 1990s saliency.

Also account honestly: a central clearinghouse running $M$ re-clears per round is a *centralized bottleneck*. The honest claim is "zero global backprop," not "fully decentralized." A sampled-LOO variant (random knockout per round, importance-weighted, Monte Carlo VCG) restores decentralization at variance cost — pre-register one arm.

### 3.2 Infant mortality — the certain-death calculation and the mandatory fix

Pre-discovery, a parity module's expected payment $\approx 0$ (it hasn't found the feature), so
$$W_m(t) \;\approx\; W_m(0) - t\,c_{\text{rent}} \quad\Longrightarrow\quad \text{certain death at } t^* = W_m(0)/c_{\text{rent}}.$$
The module survives iff $t_{\text{disc}} < t^*$. Since $t_{\text{disc}}$ grows with task depth/complexity, **flat rent selects against exactly the deep/bilinear modules the protocol exists to discover.** Three fixes, in increasing rigor:

1. **Rent ramp** $c_{\text{rent}}(t) = c_\infty \min(t/T_{\text{mat}}, 1)$ — necessary but gameable (perpetual re-fragmentation into newborns to stay in the ramp). Requires:
2. **Non-renewable lineage endowment:** endowment is minted once per lineage; mitosis *transfers* parent wealth (never mints); no fresh subsidies to children. This is the anti-Sybil condition — without it the incubator is a junk-printing press.
3. **Value-of-information rent (the principled fix).** Charge effective rent
$$\tilde c_m(t) \;=\; c_{\text{rent}} - \underbrace{\lambda_0\, e^{-t/\tau_{\text{mat}}}\,\sqrt{\widehat{\mathrm{Var}}(\overline{\mathrm{LOO}}_m)}}_{\text{exploration bonus }=\text{VoI}_m(t)}$$
where $\widehat{\mathrm{Var}}$ is the posterior variance of the module's LOO flow rate. Deep modules in discovery produce *uninformative* evaluations → variance stays high → bonus keeps them alive. Junk modules produce consistent zero/negative LOO → variance collapses → bonus decays → they pay full rent and die. This is exactly UCB optimism applied to module survival, and it is the correct separation between "unproven" and "junk."
4. **Bankruptcy as a sequential test, not a wealth floor.** Run a Wald-style SPRT on the LOO cash-flow stream: bankrupt when the posterior odds of $\mathbb{E}[\mathrm{LOO}] \le 0$ cross a boundary chosen for target false-kill rate $\alpha^*$ (controls infant mortality *statistically*) — a fixed wealth threshold controls nothing.

**Junk-control budget:** total incubator subsidy capped per lineage and globally; information bounties (§2.3) paid from the same capped budget. Without caps, R17-E3's pruning recall is meaningless because farming outpaces pruning.

### 3.3 Mitosis without churn or subsidy farming

If children are near-duplicates, the substitutes pivot annihilates one within a settlement cycle → pure compute churn. Mitosis acceptance must require **demonstrated decorrelation and complementarity**: behavioral $\hat G_{c_1 c_2} < \theta_{\text{mit}}$ *and* estimated joint LOO of the pair > parent LOO (a complementarity test). Symmetrically, add the **fusion** operator (§1.3): pairs with measured $\hat G_{c_1c_2}$ high *and* complementary function get merged to internalize $\gamma$ — otherwise the economy persistently overpays complementary pairs at the clearinghouse's expense.

---

## 4. Scorecard (0–10)

| Dimension | R16 (Metaphor) | Proj-R17 (current spec) | R17′ (post-fix design) | Bar |
|---|---|---|---|---|
| D1 Econometrica IC & Sybil resistance | 1 | 4 (DSIC claim false as stated; sabotage attack open) | 7 (ε/asymptotic DSIC + clone annihilation; shill-boundary conceded via impossibility) | 8 |
| D2 Zero-global-backprop multi-hop | 1 | 4 (cold-start obstruction unnamed; M2 ≡ TBPTT(1) unaddressed) | 6 (separable-asset fix + bounties; sample-cost conceded) | 7 |
| D3 Architecture discovery | 2 | 6 (LOO finite difference is right; baselines unfair) | 7 (pending OBD/OBS compute-matched win) | 7 |
| D4 Multi-agent strategic aggregation | 2 | 6.5 (wealth dynamics + strategic shading; risk posture unmodeled near bankruptcy) | 7 | 8 |
| D5 Budget balance & infant mortality | 1 | 3 (flat rent = guaranteed infant mortality; deficit $\gamma$ unaddressed) | 6 (VoI rent + SPRT + caps; fusion) | 7 |
| D6 Rigor & separation from backprop | 3 (THEORY-4 kills are real work) | 6 (finite-difference lemma genuine; adjoint-necessity not conceded) | 7 | 8 |

R16 is correctly dead as a paper. R17 as currently specified would survive neither an Econometrica referee (D1/D5) nor an ICLR reviewer attack on D6. R17′ is plausibly ICLR-competitive; it is *not* Econometrica-competitive as a simulation-only design paper (see §6).

---

## 5. Pre-Registered [PRIOR] Forecasts (80% CIs) and Falsifiers

### R17-E1 — Strategic IC & Clone Annihilation

| Quantity | [PRIOR] 80% CI | Falsifier |
|---|---|---|
| Inflation ratio $\hat\beta_m / \beta^{\text{true}}_m$, Naive-LMSR | **[1.40, 2.60]** | < 1.2 ⇒ naive LMSR already near-IC; audit harness suspect |
| Inflation ratio, VCG-Gershgorin-LOO (behavioral $\hat G$, $\alpha = 0.1$) | **[0.97, 1.18]** | > 1.3 ⇒ behavioral-$\hat G$ insufficient, §1.2 fix fails |
| Same, but self-reported $\hat G$ (control arm) | [1.25, 2.10] | — run this arm; it isolates the declaration-channel effect |
| Clone-cartel payout share (2 duplicates), Naive-LMSR | **[35%, 80%]** | — |
| Clone-cartel payout share, VCG-LOO + behavioral $\hat G$ | **[1%, 10%]** | > 15% ⇒ clone annihilation oversold |
| Complementary-shill pair: total payout / joint surplus | **[1.70, 2.30]** (theory pins ≈ 2) | ≤ 1.5 ⇒ empirical submodularity — *interesting either way* |
| Clearinghouse deficit on complements | occurs with $P \ge 0.9$ absent cap/fusion | — |

### R17-E2 — Hidden-Parity (depth 4) & Equiv-SAT (3-hop), test accuracy / KL (nats)

| Model | Hidden-Parity acc | KL (nats) | Equiv-SAT acc |
|---|---|---|---|
| M1 Local greedy (detached) | **[0.50, 0.56]** | [0.60, 0.693] | [0.50, 0.58] |
| M2 Naive LMSR (detached, 1-step) | **[0.50, 0.68]** | [0.25, 0.69] | [0.50, 0.60] |
| M3 VCG-Gershgorin-LIR (detached, $K \ge 4$ relaxation, pinned shadow market, seeded bounties) | **[0.58, 0.92]** (median 0.78 *conditional on fixes*; 0.55 without) | [0.03, 0.50] | [0.52, 0.88] |
| M4 Global backprop ceiling | **[0.97, 1.00]** | [0.00, 0.03] | [0.90, 0.99] |
| Sample-complexity ratio $T_{M3}/T_{M4}$ | **[2.5, 12]** | — | — |

Calibration diagnostics: M2 trajectory cosine vs. TBPTT(1) ≥ 0.99 (replicates THEORY-4 identity); M3 ≤ 0.60 on Hidden-Parity *with all fixes* falsifies the price-mediated-credit hypothesis at depth 4 — that is the smoking-gun-or-negative result, publishable either way as a clean negative.

### R17-E3 — Bankruptcy & Mitosis

| Quantity | [PRIOR] 80% CI | Falsifier |
|---|---|---|
| Junk/clone pruning recall (≤5% false-prune), VCG-LOO | **[82%, 97%]** | < 75% ⇒ settlement noise dominates |
| Same, $\vert w\cdot g\vert$ | [55%, 85%] | — |
| Same, OBD/OBS (compute-matched) | [75%, 93%] | **LOO ≤ OBD at matched compute ⇒ market adds nothing over saliency — kill D3 claim** |
| Sign agreement with ablation ground truth, VCG-LOO | **[93%, 99.5%]** (vs. 74.4% pinned by prior census) | < 90% ⇒ §3.1 mechanism wrong |
| XOR-bottleneck survival, flat rent | **[3%, 30%]** (pre-register as the *expected failure*) | — |
| XOR-bottleneck survival, VoI rent + SPRT + lineage caps | **[55%, 92%]** | < 40% ⇒ infant-mortality fix insufficient |
| Mitosis acceptance (decorrelation + complementarity gate) | [15%, 55%]; junk-mitosis ≤ 5% with non-renewable lineage endowment (else [20%, 45%]) | — |
| Final-acc delta vs. GradMag on saturation/co-adaptation tasks | **[+1, +6 pts]**; benign tasks [−1, +1.5] (settlement noise cost) | — |

---

## 6. Publication Routing — Blunt Assessment

**Econometrica: not with R17 as constituted.** The publishable kernel is R5.1′ — a clean characterization (ε-DSIC under approximate clearing; WBB + false-name-proofness iff submodular; deficit $= \gamma$ on complements; Green–Laffont corner forced). That is a mechanism-design theory result whose natural homes are *Econometrica*-adjacent only with an application, identification, or field evidence; absent that, its honest venues are **EC/SODA/TE/JET**. "We ran a simulation of module markets" does not clear Econometrica's bar and claiming it would damage credibility.

**ICLR: R17-E2 is the flagship, R17-E3 the companion.** E2 clears if M3 ≥ 0.85 on depth-4 Hidden-Parity with compute-matched baselines, the M2 ≡ TBPTT(1) calibration shown, and the adjoint-necessivity conceded and reframed as locality + incentives + rate. E3 clears only if LOO wealth beats **OBD/OBS at matched compute** — beating $|w\cdot g|$ alone is beating a strawman.

**Missing related work a referee will cite against you:** Ostrovsky 2012 (separability — *the* cold-start theorem); Hanson LMSR properness; peer prediction (Miller–Resnick–Zeckhauser 2005; Dasgupta–Ghosh 2013) for truthful elicitation without verification; Yokoo et al. false-name-proofness; Ausubel–Milgrom 2006; Cavallo 2006; Green–Laffont 1977; d'Aspremont–Gérard-Varet 1979; Jaderberg et al. 2017 (synthetic gradients); Scellier–Bengio 2017 (equilibrium prop); Bai et al. 2019 (DEQ); LeCun 1990 / Hassibi–Stork 1993 (OBD/OBS); Borkar 1997 (two-timescale); Forsythe–Lundholm 1990 (belief heterogeneity); Jensen–Meckling 1976 (near-bankruptcy risk-seeking); Wolpert–Tumer COIN. Absorbing these *before* submission is the difference between "markets as unifying principle" landing and it reading as rediscovery.

**Bottom line:** the finite-difference-over-re-cleared-fixed-points lemma and the substitutes/complements characterization are the two genuine mathematical contributions. Everything else — DSIC as stated, flat-rent survival, cold-start-free aux markets — is currently wrong or unproven, and the fixes above are the pre-registration discipline that makes either acceptance or clean rejection a result worth having.
