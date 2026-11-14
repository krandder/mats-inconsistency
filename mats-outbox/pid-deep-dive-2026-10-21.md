# Williams & Beer 2010 and the aggregation agenda (deep dive, 2026-10-21)

**TL;DR.** Williams & Beer (WB) give us vocabulary: redundancy R, unique information U_i, synergy S, and a lattice of "obtainable from A or B" atoms. Almost every result that carries weight below turns out to be a Shannon quantity that needs no PID measure. One PID quantity does real work: BROJA synergy for two sources, which is the information about Y that calibrated per-agent forecasts cannot certify. Ranked recommendation: (1) the *listed-securities gap* G(Σ), which measures the value of your Part 2 exactly in aggbench; (2) separating *credit* from *learning signal* on parity. Nothing has been trained yet; all numbers are exact toy computations.

References: arXiv ids were opened on export.arxiv.org and DOIs checked on Crossref by the research and critique passes today. (recall) = not checked. Scripts: `/home/agent/workspaces/mats/work/pid-diag/` (aggbench diagnostics, nats) and `.../pid-diag/from-tmp/` (toy scripts, bits; copied out of /tmp).

## 1. What Williams & Beer showed (arXiv:1004.2515, preprint)

- Sources X_1..X_n, target Y. Shannon gives I(Y;A) for every source set A but cannot say how I(Y;X1,X2) splits into shared, own and joint-only parts.
- They index the parts by antichains α = {A_1..A_k} ("obtainable from A_1, or A_2, ..."). Each set's MI is the sum of the atoms at or below its node. Atom count is the Dedekind number minus 2: 4, 18, 166, 7,579 for n = 2..5.
- Their axioms (symmetry, self-redundancy, monotonicity) plus a redundancy function I_∩ fix all atoms by Möbius inversion. Their I_∩ is I_min = Σ_y p(y) min_i D_KL(p(a_i|y)||p(a_i)). Main theorem: I_min gives nonnegative atoms for any n.
- Two sources, the equation that matters:
  I(Y;X1) = R+U1,  I(Y;X2) = R+U2,  I(Y;X1,X2) = R+U1+U2+S,  so co-information = R − S.
  Three equations, four unknowns; the redundancy measure supplies the fourth.
- Examples (X1, X2 independent uniform bits, bits): XOR: S = 1, all else 0 (every measure). AND: I_min and BROJA give R = 0.311, U = 0, S = 0.5. COPY, Y = (X1,X2): I_min gives R = 1, S = 1; the intuitive answer R = 0, U1 = U2 = 1 is what BROJA gives. COPY is the standard proof that I_min measures "same amount", not "same information".

## 2. What changed since 2010, and what to use now

- **Criticism.** Harder, Salge & Polani 2013 (doi:10.1103/PhysRevE.87.012130, arXiv:1207.2080): the COPY failure, and the identity axiom. Rauh et al. 2014 (arXiv:1404.3146, doi:10.1109/ISIT.2014.6875230): for n ≥ 3 no measure has nonnegative atoms on the WB lattice plus identity. Example: X1, X2 iid bits, X3 = X1⊕X2, Y = (X1,X2,X3): identity forces 3 bits into a 2-bit total. Kolchinsky 2022 (doi:10.3390/e24030403, arXiv:1908.08642) drops the lattice sum and defines redundancy via a channel preorder. Consequence: atoms are measure-relative; report the measure with every number.
- **Measure choice.**
  - 2 discrete sources, operational meaning: BROJA (Bertschinger et al. 2014, doi:10.3390/e16042161, arXiv:1311.2852). U1 = min over Q with the (Y,X1),(Y,X2) marginals fixed of I_Q(Y;X1|X2); S = I(Y;X1X2) − min_Q I_Q(Y;X1X2). Convex program; satisfies identity.
  - 3 to 4 sources or a differentiable objective: I_sx (Makkeh, Gutknecht & Wibral 2021, doi:10.1103/PhysRevE.103.032149, arXiv:2002.03356); alternatives I± (doi:10.3390/e20040297) and I_ccs (doi:10.3390/e19070318, can be negative).
  - Gaussian, univariate target: all standard measures reduce to MMI, R = min_i I(Y;X_i) (Barrett 2015, arXiv:1411.2832). Vector targets: Venkatesh et al. 2023 (arXiv:2307.10515, preprint).
  - n ≥ 5: no lattice. Use O-information (Rosas et al. 2019, arXiv:1902.11239; negative = synergy dominates), synergistic disclosure (arXiv:2001.10387) or representational complexity (Ehrlich et al., arXiv:2209.10438).
  - Do not use I_min as a primary measure.
- **Estimators and code.** dit (doi:10.21105/joss.00738, exact discrete, n ≤ 3-4). BROJA-2PID (arXiv:1802.02485). IDTxl (doi:10.21105/joss.01081). Neural, 2 sources: Liang et al. CVX/BATCH (arXiv:2302.12247, github.com/pliang279/PID); Gaussian PID with normalizing flows (Zhao et al., arXiv:2510.04417). HOI for O-information on GPU (doi:10.21105/joss.07360).
- **Applied neighbours.** Synergy bounds from disagreement of single-source predictors (Liang et al., arXiv:2306.04539). Local PID objectives per neuron ("infomorphic", Makkeh et al., PNAS 2025, doi:10.1073/pnas.2408125122). Synergy as the gap between the true posterior and a pooled one (van Enk 2023, arXiv:2302.02251): closest prior work to our regret framing. No one has applied PID to forecast aggregation or prediction-market benchmarks (literature search, not proof of absence).

## 3. Four directions, after adversarial critique

Notation: O_i or X_i is agent i's evidence, M_i its message, C the consensus it reads, R_i = p(Y|O_i) its Part-1 forecast, q the aggregate.

### 3a. Diagnostic: what part of aggregator regret is unavoidable for Part-1 messages
- **Survives.** For any message vector M and aggregator q = f(M): regret = I(Y;O|M) + E KL(p(Y|M)||q). Chain rule, not new, but it splits regret cleanly. Measured on aggbench (nats): the report gap G_R = I(Y;O|R) equals the whole I(Y;O) on Z6 (0.258) and Z9 k=3, 4 (0.496, 0.453), and is ≈0 to 0.017 on E2, E3-sybil, Z1, Z5, Z8. This explains why every report-only arm sits at the prior on parity.
- **Your Part 2 made exact.** Let Σ be the securities the consensus already lists and G(Σ) = I(Y;O | {p(Z|O_i): Z ∈ Σ, all i}). Part 2 proposes a new Z; its value is G(Σ) − G(Σ ∪ {Z}). XOR: G({Y}) = 1 bit; listing X_1 drops it to 0. Exactly computable in aggbench and independent of which PID measure is chosen.
- **Does not survive.** "Redundancy predicts log-pool damage" (Spearman 1.0) uses ΣI_i/I − 1, which is co-information, not PID, on 6 non-independent cells. Log-pool regret is not a function of the atoms (a shared Y-irrelevant noise bit changes the terms but not the regret). The planned regression is circular (regret = G_R + pool term exactly). G_R is a floor only for one-shot reports: repeated rounds can reveal more (Geanakoplos & Polemarchakis 1982, doi:10.1016/0022-0531(82)90099-0; Ostrovsky 2012, doi:10.3982/ECTA8479), although not on XOR. The diag table's "logpool" and "excess over oracle" columns disagree on E2, Z8, E3 and need a defined baseline.
- **Kept PID fact.** BROJA S is not the report gap. AND: S = 0.5 bit but G_R = 0, because forecasts there are one-to-one in the evidence and a decoder that knows the joint recovers everything. BROJA S is what a decoder knowing only each agent's calibrated channel p(y,x_i), not how evidence co-varies, cannot certify.
- **First step (about 1 h CPU).** Compute G(Σ) for Σ = {Y}, {Y, X_i}, {Y, pairwise parities} on Z6, Z9 k=3,4,6, Z5, E3-sybil and an AND cell.

### 3b. Message design, including your two-part message
- **Survives (Shannon).** Cut-set bound: H(M_i) ≥ I(Y;X_i|X_{-i}) (= U_i + S for two sources). Redundancy is paid once, unique information by its owner, synergy by every holder: XOR costs 2 bits for 1 bit of information. For agent i speaking after C, the coarsest lossless message T_i groups x_i values that no later evidence can tell apart.
- **When Part 2 is needed:** exactly when T_i is not a function of (R_i, C). Sufficient for Part 2 = 0: X_i ⊥ X_{>i} | (Y, C), or | (W, C) for a security set W that determines Y. So Part 2's job is to add variables to the consensus until later evidence is conditionally independent given them. This is a precise version of "the consensus knowing what is supposed to be important".
- **Does not survive.** "Part 2 is needed exactly when S > R" is false both ways: AND has S − R = 0.189 > 0 and needs no Part 2; Y = A⊕B, X1 = (A,Z), X2 = (B,Z), Z = Y flipped w.p. 0.1, has S − R = −0.062 and still has forecast-only regret 0.469 bit (`from-tmp/cx.py`). The two-part message is not minimal (T_i need not determine R_i); keeping Part 1 is justified only if the consensus must be readable after every message, and the design should say so. Duplicates are free one-shot when the structure is known; their cost comes from hidden provenance (E3-sybil, Z8). The criterion is vacuous without a cost on W (W = X always works); the right object is Wyner common information (doi:10.1109/TIT.1975.1055346) restricted to W that determine Y.
- **Order matters, as a trade-off.** 3-bit parity, 6 readers: grouped order costs 4.19 bits total with 3.06 bits of Part 2; interleaved costs 4.70 with 2.0 (`from-tmp/seq.py`; 1 reading per agent, so not the aggbench Z9 cell, which stalls at 0.457 rather than 0.429).
- **First step (about 1 day CPU, exact).** Port seq.py into aggbench as a "message audit". Add the two counterexample cells and a `pieces_per_agent` knob. Test which predicts measured Part 2 = H(T_i|R_i,C) across the zoo: S − R, BROJA S, or conditional-independence violation given (Y,C). Prediction: only the last works.

### 3c. Learned securities
- **Survives (Shannon).** Proposition B: let T = (Y,Z) and pool Part-1 forecasts log-linearly, q(t|x) ∝ p(t)^{1−n} Π_i p(t|x_i). Then
  E_x KL(p(y|x)||q(y|x)) ≤ TC(X|T) − KL(p(x)||r(x)) ≤ TC(X|T),  TC(X|T) = Σ_i H(X_i|T) − H(X|T),
  with r(x) = Σ_t p(t) Π_i p(x_i|t). Part 1 gives exact Bayes when evidence is conditionally independent given T; Part 2 picks Z to get there. Toy (bits, `from-tmp/tc.py`): common cause TC 0.190 → 0 with Z = C; duplicate source 0.610 → 0; 3-bit parity, 3 readers/bit: 1.724 → 1.423 → 0 while regret on Y stays 0.423 until the last step.
- **Known.** This is the KL form of the naive-Bayes pooling error; adding structure by scoring I(X_i;X_j|C) is TAN (Friedman et al. 1997, doi:10.1023/A:1007465528199); hidden-variable versions are hierarchical naive Bayes (recall). CorEx (arXiv:1406.1222) is the unsupervised analogue.
- **Does not survive.** The PID labels add nothing (TC = S − R + I(X1;X2) for every measure). "Synergy plus shared noise" is not a split into nonnegative parts (I(X1;X2) − R = −0.311 on AND). TC's first drop on parity comes from explaining duplicate readers, not parity progress, and TC also rewards Y-irrelevant shared latents. No single agent can compute TC; it is a centralized training loss. The Ostrovsky link is a restatement: for a binary security, non-separable iff some prior gives I(Y;X_i) = 0 for every i.
- **Open.** cost(Z). Z = X recovers the full combinatorial market, whose pricing is hard (recall: Chen, Fortnow, Lambert, Pennock & Wortman 2008).
- **First step (CPU, exact).** Greedy security search on Z1, Z8, E3-sybil, Z9 k=3 (1 and 3 readers/bit) with decoys (Y-irrelevant shared latents, random parities, reader keys). Score steps by ΔTC, Δ(pooled regret on Y), ΔS_BROJA. Add a TC loss to learnsec arm C only if TC beats regret-guided search.

### 3d. Credit assignment and learning signal
- **Survives.** Two sources: a sequential log market scoring rule pays agent i I(Y;M_i|C_t) = U_i + S, so redundancy goes to whoever is first and synergy to whoever completes it. Shapley = I(Y;M_i) + (S − R)/2, no PID measure needed. Under log score, pairwise "substitutes" means R ≥ S; Chen & Waggoner (arXiv:1703.08636) already prove substitutes aggregate immediately and complements late. AND: sequential [0.311, 0.5], LOO [0.5, 0.5], Shapley [0.406, 0.406] bits.
- **Credit does not create learning signal (exact).** ∇_{θ_i}[v(N) − v(N∖i)] = ∇_{θ_i} v(N), so leave-one-out has the same expected gradient as team reward and only lowers variance. On k-parity with 1-bit messages of bias ρ_j, v ≈ (Πρ_j)²/(2 ln 2), so the gradient is ∝ ρ^{2k−1} near 0: 5.3e-4 bits of value at ρ = 0.3, k = 3; 3.8e-7 at k = 6. This matches the overnight stall of learned encoders and the 6-bit trader on Z9.
- **Two separate difficulties.** (i) Known support, cold start: a saddle with escape time ~ρ_0^{−(2k−2)} (information-exponent effect, Ben Arous et al., arXiv:2003.10409); a warm start solves it. (ii) Unknown support, agents reading m > b bits: gradient learners pay a statistical-query cost ~C(m,k) (Blum et al. 1994, doi:10.1145/195058.195147; Kearns 1998, doi:10.1145/293347.293351). Exact outcome data do identify a sparse parity in O(k log n) samples by brute force; the barrier is for gradient learners, so "what is important" must come from the query, priors or reading the evidence.
- **Does not survive.** "Reward unique + synergistic, penalize redundant + consensus-unique" collapses to I(M;X_i|C) − I(M;C): it pays for distractor bits and penalizes Part 1. Use the conditional information bottleneck max I(M;Y|C) − β I(M;X_i|C) (CEB, arXiv:2002.05379). Synergy is gameable (M1 = X1⊕K, M2 = K), so atoms are diagnostics, not rewards. LOO is not budget-balanced; Shapley is not replication-robust (arXiv:1805.08125).
- **First step (CPU).** Each agent reads m = 4 bits (some distractors), sends b = 1 bit; Y = 3-parity across agents. Arms: team reward, LOO (also log gradient variance), Shapley, conditional IB, oracle partial-parity auxiliary securities, warm start. Prediction: only auxiliary securities and warm start escape, with cost growing like C(m,k). If only the auxiliary arm escapes, the bottleneck is which securities to list, which is the learned-securities question.

## 4. Ranked recommendation

1. **Listed-securities gap G(Σ) and the value of a proposal, G(Σ) − G(Σ ∪ {Z})** (3a + 3c). It turns your two-part message into one number per environment: how much Y-information Part-1 forecasts on the current securities leave out, and how much a proposed security recovers. Exact in aggbench, independent of PID measure choice, cheap (1 h CPU), and it is the target any learned-securities arm should hit. TC(X|T) and the conditional-independence criterion of 3b are its training-time proxies; test them against G, with decoys.
2. **Credit versus learning signal on parity** (3d). It explains the overnight stalls exactly and the planned run separates three causes: cold start, support search, missing auxiliary securities. Result decides whether learnsec needs a curriculum of securities or just a different init.
3. Message audit (3b) next, mainly to measure Part 2 = H(T_i|R_i,C) per cell. PID atoms (BROJA on agent-versus-rest splits, I_sx on ≤ 4 sources) go into every table as labels, not as bounds or losses.

What WB adds to the agenda, honestly: vocabulary, the warning that n ≥ 3 atoms depend on the measure, and BROJA S as the two-source answer to "what can calibrated per-agent forecasts not certify".

## 5. Limits

- Everything is exact enumeration on toys or aggbench cells with known structure; no learned system was trained against these quantities.
- G_R and G(Σ) assume a decoder that knows the joint structure; under unknown keys the real floor is higher.
- One-shot analysis only; dynamic market rounds can beat the one-shot floor in non-XOR cases.
- PID atoms for n ≥ 3 are measure-relative (Rauh et al.); pairwise atoms are all 0 on Z9 k ≥ 3 and cannot see higher-order parity.
- E1 (vector Gaussian target) not computed; pairwise BROJA on Z1 did not finish.
- Novelty claims rest on short Exa searches; van Enk 2023 and Chen & Waggoner are close prior work.
- The scripts in `from-tmp/` were written by research agents and checked only by the critique pass (tc.py and the credit arithmetic were re-run; seq.py was not).
