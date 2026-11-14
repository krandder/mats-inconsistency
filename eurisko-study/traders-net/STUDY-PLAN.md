# Study plan: traders inside a neural network (workflow 2)

Planner, 2026-09-27, from `CONTEXT.md`, `research/scope-*.md` and `scope-critique.md`.
Wealth-net (checked about 22:05Z, `../wealth-net/results/train-s1.log`): step 12.5k of 60k,
checkpoints at 8k/10k/12k; `ablate.py` is the wealth-net lane's. Budget: 30-60 Opus 5.5 agents,
about 24 h, this box (16 CPU, 62 GB, ~100 GB free), spot L4s, at most $50 of GPU.

**Rules for every agent.**
- Write only under `traders-net/study/<ws>/` (≤20 GB in total, no git worktrees).
- Wealth-net and the soup are read-only; never rerun wealth-net's ablation or touch its VM.
- At most 12 CPU jobs at once; all GPU time goes through G1.
- Write `PREREG.md` (arms, metric, seeds, stop rule) before the first run; 3 seeds, mean ± sd.
- Coordinate through files, not messages. Effort medium; T1 and X1 high.

**RETARGET (eurisko, 2026-09-28 00:25Z, after Kelvin confirmed the sign-flip diagnosis; overrides the workstream briefs below where they conflict).**
- M1 becomes the *negative-result* workstream: (a) formalise the sign relation between backprop credit and market settlement (a toy where it is provable: a unit whose signal is absorbed by others gets negative gradient credit while a committed trader with the same signal earns); (b) measure it on the wealth-net checkpoints (fraction of oracle-important units with negative gradient credit); (c) still test the three-rules settlement for attention/gates only, since those have an internal outcome. CURRENCY.lock now says: "no bookkeeping currency for generic neurons; committed-trader settlement only". Gate A passes when (a)-(c) are written.
- M3 (currency lock-in) is reduced to one agent's short note referencing wealth-net's decay results; M8 (wealth-gated dense net) is reduced to the sign test only; their remaining agents' effort goes to M5, M6, M7, which already treat traders as designed objects (committed log-odds corrections with confidence, settled against price), and to a new item in M6: units become traders only by emitting a committed correction, and the experiment measures whether making internal units commit (heads/experts as sub-market traders) changes what wealth means.
- T1 adds Lemma 8: conditions under which first-order attribution credit has the opposite sign of market settlement.
- D1 leads with this finding for Jessica Taylor: "attribution flow is not settlement; the sign evidence".

## 1. Goal and questions (priority order)

**Goal.** Find out whether "units as traders with wealth" adds anything beyond known tools, and
show Kelvin wealth evolving in a small network.

**Premises (settled by the critique; do not re-derive):** a·dR/da is signed Taylor/movement
importance (about norm growth under SGD); in-sample books pay memorisation; tangent settlement has
zero mean at calibration; books leak at attention/LN and gates pay twice; a convex pool cannot
add independent evidence; the DeepSeek bias is already a capacity price; cumulative rank
stability is an artefact.

**Questions**
1. **Q1 Currency.** Settled like a market (prequential, exact LMSR from a named prior), does
   wealth measure more than signed movement pruning? (M1, M2, T1)
2. **Q2 Novelty.** What is new relative to Balduzzi, Storkey, Hayek/Economy of Minds, MoB and
   movement pruning? This gives the one-line claim for Jessica. (L1)
3. **Q3 Induction.** Is a net of traders exploitable by a fixed trader class? Is it coherent
   *before* resolution on delayed and deductive questions? (M5)
4. **Q4 Humility.** Do anchored, bounded, fixed-point traders beat a plain net, early stopping,
   weight decay and boosting? (M6)
5. **Q5 MoE.** Can wealth take over the anti-collapse role of load balancing without a tuned
   coefficient? (M7)
6. **Q6 Lock-in.** Which currency tracks current value, and does gating by it survive a regime
   shift? (M3, M8)
7. **Q7 K/Q.** Is the soup's K/Q law about markets or about search? What does shared computation
   buy at equal FLOPs? (M9)
8. **Q8 Economy.** Do thousands of Kelly traders coexist or collapse, as seen live? (K1, V1)

**Timeline**
- H0-H4: S0, L1, T1, M1(a), K1 build, M9(a), V1, G1.
- **Gate A (H5):** M1 writes `CURRENCY.lock`; every later arm imports it.
- H5-H12: M2, M3, M5, M6, M7 stages 0-1, M9(b), K1 sweeps.
- **Gate B (H12):** M8 full grid only if M3 beats recent |a·g| or M2 shows a coalition win
  (else its reduced arm); M7 stage 2 only if a wealth arm beats DeepSeek; M6 stress only if its
  base arms survive their stop rule.
- H12-H20: M8, M7 stage 2, M9(c), M6 stress.
- H18-H24: X1, then D1.

## 2. Workstreams

| id | name | kind | agents | depends on | deliverable |
|---|---|---|---:|---|---|
| S0 | Substrate + metric panel | measurement | 4 | none | `substrate/`, tests, `SUBSTRATE.md` |
| L1 | Literature novelty memo | literature | 4 | none | `NOVELTY.md`, `NOVELTY-FLAGS.md` |
| T1 | Theory memo | theory | 3 | none | `THEORY.md` + check scripts |
| M1 | Books v2 + audit | measurement | 4 | S0 (b-d) | `books_v2.py`, `AUDIT.md`, `CURRENCY.lock` |
| M2 | Validity bake-off | measurement | 4 | M1 | `oracle.py`, `BAKEOFF.md` |
| M3 | Currency and lock-in | experiment | 3 | M1, M2 | `CURRENCY.md` |
| M5 | Exploitability/coherence harness | measurement | 4 | S0, T1 | `exploit.py`, `EXPLOIT.md` |
| M6 | Humble fixed-point market layer | experiment | 5 | S0, M1, M5 | `M6.md` |
| M7 | Priced MoE, staged | experiment | 5 | S0, M1, L1 | `M7.md` |
| M8 | Wealth-gated dense net | experiment | 4 | M1, M2, M3 (Gate B) | `M8.md` |
| M9 | Soup-to-net transfer, K/Q | experiment | 3 | S0, M6 | `M9.md` |
| K1 | Kelly-pool economy | experiment | 3 | S0, T1 | `K1.md` |
| V1 | Live wealth view | measurement | 2 | S0 | Grafana `traders-net` + check |
| G1 | GPU broker, cost ledger | measurement | 1 | none | `infra/`, `COST.md` |
| X1 | Skeptic + reproduction | measurement | 5 | M1-M9, K1 | `CLAIMS.md` |
| D1 | Delivery | literature | 2 | X1 | `IDEAS-NET.md`, HTML, hub PR |

**Total: 56 agents.**

**M1 addendum (eurisko 23:40Z).** Test the 'three rules' settlement from CONTEXT.md (Euler for linear nodes, half/half for bilinear products, softmax-as-LMSR with values paid at the prior and keys paid (a_j − π_j)·u_j) as a candidate for CURRENCY.lock; first check is exact conservation on the wealth-net checkpoints with live attention; second is whether pruning-by-wealth and lock-in change. T1: state it as a lemma (conservation identity) and its relation to Aumann-Shapley/IG.

**S0.** Copy `../wealth-net/{streams,model}.py` into `substrate/`, then:
- On-the-fly prequential single pass (test, then train), plus a multi-epoch stress mode.
- Seeded PRNG-parity instead of pi (no shared offsets); a held-out periodic family released
  mid-run; a regime shift (every 10k steps a random quarter of questions switch family).
- Three LI families: **D-delay** (labels revealed D ∈ {0, 64, 512} steps late, booked at the
  trade price, settled at reveal); **D-prove** (A∨¬A, A∧¬A, (A∧B)→A, exclusive-exhaustive
  triples, never sampled, revealed late by a deductive process); **D-expensive** (a CA110 cell
  or LFSR bit 64-1024 steps ahead, delay growing with compute cost).

`panel.py`, the same for every run:
- prequential BCE and Brier per family, and ECE;
- AND/OR/D-prove violations before resolution;
- E_C(t) slopes (from M5);
- wealth Gini against a shuffled-label null;
- dead/bankrupt fraction, cumulative FLOPs, conservation residual.

`runlog.py` writes one CSV per run for V1. Tests: seeded determinism, an assert that train and
eval never overlap, and known panel values on a toy.

**L1.** Three cluster readers and one synthesizer. For each paper: fetch it, verify the ID, and
write 5 lines covering what was built, the overlap with ideas 1-6, and what it did not do.
- **(A) Markets in ML:**
  - Balduzzi https://arxiv.org/abs/1401.1465, https://arxiv.org/abs/1411.6191
  - Storkey https://arxiv.org/abs/1106.4509, https://arxiv.org/abs/1206.6443
  - Lay-Barbu https://arxiv.org/abs/1102.1465
  - Kwee et al. https://arxiv.org/abs/cs/0105025 and the Hayek machine
  - Chang https://arxiv.org/abs/2007.02382
  - survey https://arxiv.org/abs/2503.05828
  - Economy of Minds https://arxiv.org/abs/2606.02859
  - GLN https://arxiv.org/abs/1910.01526
- **(B) Attribution and pruning:**
  - movement pruning https://arxiv.org/abs/2005.07683
  - Taylor https://arxiv.org/abs/1906.10771
  - IG https://arxiv.org/abs/1703.01365
  - conductance https://arxiv.org/abs/1805.12233
  - AttnLRP https://arxiv.org/abs/2402.05602
  - Neuron Shapley https://arxiv.org/abs/2002.09815
  - Hydra https://arxiv.org/abs/2307.15771
  - LCA https://arxiv.org/abs/1909.01440
  - lottery ticket https://arxiv.org/abs/1803.03635
- **(C) Routing, LI and online learning:**
  - https://arxiv.org/abs/2408.15664, https://arxiv.org/abs/2512.03915
  - MoB https://arxiv.org/abs/2512.10969
  - https://arxiv.org/abs/2604.00801, https://arxiv.org/abs/2604.04230
  - PEER https://arxiv.org/abs/2407.04153
  - https://arxiv.org/abs/2310.00811
  - LI https://arxiv.org/abs/1609.03543
  - https://arxiv.org/abs/1201.6655, https://arxiv.org/abs/1003.0034
  - https://arxiv.org/abs/2101.01739
  - Blume-Easley 2006
  - the hub wiki `consistency-ft.md` (Exploit LP)

The synthesizer writes `NOVELTY.md` (claim → nearest prior art → delta, plus the one-line
claim) and, by H3, `NOVELTY-FLAGS.md` (planned arms that prior art already ran).

**T1.** Lemmas, each tagged theorem, known (cited) or conjecture, each with a numerical check on
an S0 toy. Draft by H6, final by H12.
1. Books identity: income = −θ·∇L; norm growth under SGD (state the Adam residual); rent under
   weight decay; zero-mean tangent.
2. Exact LMSR settlement = log-score gain = IG from the prior.
3. **A cost-sharing rule for the shared trunk** (Aumann-Shapley over usage vs marginal cost). It
   gives the FLOP budgets for M7 and K1.
4. **Delayed settlement** (trade price → resolution, with mark-to-market in between). State what
   changes for LM next-token and RL; these are not run.
5. Finite-population LI: E_C(t) and invasion fitness. Conjecture: sublinear E_C needs one of
   C ⊆ population, fixed-point prices, or births that sample C.
6. A fund of funds with fixed fractions is a flat mixture; only rebalancing adds power.
7. The top-k router gradient is relative revenue under censored feedback; router-as-broker gate
   convention.

**M1.**
- (a) Now, read-only on wealth-net snapshots: regress wealth on Δ‖θ_out‖²/2η and on the
  movement score; report R² and the residual.
- (b) Rerun one seed with SGD and one with SGD plus weight decay.
- (c) `books_v2.py` with three books: prequential tangent; exact LMSR (a 16-step path from prior
  ℓ0 = the running base rate); mean-baseline (a−ā)·g.
- (d) Split the conservation residual by source (QK, softmax, LN); try AttnLRP-rule books.

`CURRENCY.lock` names the books every later arm imports.

**M2.** `oracle.py` runs exact held-out ablations (zero and mean) of every neuron, head and
readout at 4 checkpoints, plus 200 random and 50 structured coalitions. It reads wealth-net's
`ablation.csv` and does not recompute it.
- Predictors: books v1 and v2; accumulated |a·g|; movement; ‖w_out‖; gradient norm;
  IG/conductance; sampled Shapley (64 permutations, heads and readouts).
- Metrics: Spearman against true Δloss per granularity; the coalition additivity gap; the pruning
  frontier.
- tracr ground truth only if time allows.

**M3.** Currencies: cumulative, EMA (half-lives 100/1k/10k), depreciation δ, fixed-share α,
Kelly-multiplicative, disjoint-window income, on M1's prequential and shift runs. Metrics:
Spearman with oracle Δloss at +1k/+5k, disjoint-window stability, and lag after a shift until
income reaches the units needed later (vs gradient norm).

**M5.** Class C of shadow traders: calibration bins (10 buckets × sign), AND/OR/exclusive/D-prove
arbitrageurs, pattern-2/4/8, online MLP probes on (features, price) at 3 sizes.

Report E_C(t) slopes and invasion fitness at wealth ε (reuse consistency-ft's Exploit LP if
reachable). Baselines: wealth-net checkpoints, the logistic model, and the soup via the `sh_*`
columns of `runs/phase3/*/metrics.csv` (per-question replay only if `soup.dump` exposes prices;
≤1 h to check).

**M6.** An output market layer: 16 trader heads per question read the residual and the market
price ℓ0 (the logistic baseline, detached). Arms:
- (a) plain logit sum;
- (b) additive corrections on ℓ0;
- (c) convex pool, k learned;
- (d) convex pool, k from prequential wealth;
- (e) log-utility vs exponential-utility pool;
- (f) bounded bets;
- (g) a fixed-point layer: traders read p and emit d_i(p); Newton solves Σd_i(p*)=0 in at most 8
  iterations; implicit gradients (DEQ, https://arxiv.org/abs/1909.01377). Compare it with 1-pass
  and 2-pass layers.

Stress tests: the multi-epoch regime, an independent-evidence family, 50% injected noise
traders, and the D families.

**M7.**
- **Stage 0:** a numerical check that the router gradient is g_i(r_i − Σg·r); fix the gate
  convention.
- **Stage 1**, on S0 with 8 and 64 experts, top-2. Arms:
  - no balancing, Switch aux loss, DeepSeek bias, expert choice;
  - Exp3 shadow credit (one sampled unrouted expert at zero gate);
  - DeepSeek bias + per-family wealth prior;
  - global wealth prior (expected leak);
  - bankruptcy and birth.
- **Stage 2**, only if some wealth arm beats DeepSeek at stage 1: TinyStories, 6 layers, d=384,
  32 experts top-2; DeepSeek vs the best wealth arm vs the Switch aux loss; 2 seeds.

**M8.** Gate each MLP neuron by w_i ∝ exp(η·income) in M3's currency. Add a fixed-share floor α,
an exit priced at c per activation, and rebirth. Run on the shift stream. Sweep η ∈ {0.1, 1, 10},
α ∈ {0, 1e-3, 1e-2}, and 3 values of c. If M2 ended in a tie, run only the |a·g|-gated arm.

**M9.**
- (a) Fit 1−(1−π_f)^{K/Q} per family to `runs/phase3`, predict the K/Q that LFSR and CA need,
  and state the lottery-ticket link.
- (b) Confirm that wealth-net prices LFSR and CA110 at 60k, then plot FLOPs-to-Brier for the net
  against the soup's virtual instructions (assumption: 1 VI = 1 op).
- (c) Sweep K/Q ∈ {1, 4, 16, 64} in M6's layer, at fixed total parameters and at fixed trader
  size.

**K1.** A numpy economy: 1k/10k/100k Kelly traders (pattern-k, online logistic), a
wealth-weighted price, inflow = maker subsidy, sink = compute price c per question traded, split
and die, Q ∈ {1, 64, 4096} niches, a mid-run shift. Arms: niches on/off, shift on/off, a c sweep,
and funds of funds with fixed fractions vs rebalancing.

**V1.** Clone `logical-soup/infra/grafana` (serve.py + Infinity datasource) on a new port;
leave the soup dashboard alone. Panels: wealth by family and scale, Gini vs null, rank churn,
births/bankruptcies, conservation residual, E_C, price per family; CSVs re-read on each refresh;
a check script proves they load.

**G1.** Copy the soup's infra scripts (6-hour auto-delete, watchdog). Keep a job queue and a
ledger priced at $0.87 per L4-hour (a ceiling, unverified). Refuse any job that would push the
ledger above $45, and check at the end that no VMs are left.

## 3. Experiments: controls, metrics, stop rules, cost

| exp | controls | primary metric | stop rule | cap |
|---|---|---|---|---|
| M1 | frozen and random books; SGD vs Adam | R² of wealth vs norm growth; Spearman of in-sample vs prequential; PRNG wealth sign | Prequential ranks ≥0.9 of in-sample: keep the cheap books. Otherwise use prequential books everywhere. | $3 |
| M2 | random, Taylor, movement, magnitude | Spearman vs true Δloss; additivity gap | A tie with Taylor or movement within seed sd at every granularity: report it and stop the pruning work. | CPU |
| M3 | shuffled-label books | Spearman with future Δloss | Nothing beats recent \|a·g\| by ≥0.05 in 3/3 seeds: gate by \|a·g\|, and wealth is accounting only. | CPU |
| M5 | logistic model; soup shadow traders | E_C slope; coherence before resolution | none (harness) | CPU |
| M6 | early stopping, weight decay, boosting with shrinkage (market as offset) | learnable-family BCE; max(BCE−ln2) on PRNG/CA30; D coherence | (c)-(g) ≥ (a)+1% BCE in 3/3 seeds with no protection beyond the kill controls: drop the pool. Keep the fixed point only if it cuts violations ≥50% against 2-pass. | $7 |
| M7 | Switch, DeepSeek, expert choice | loss per active FLOP; dead experts; max load; expert×family MI; leak | Max load >50% in 2/3 seeds, or loss >2% worse than DeepSeek at every E: stop. No wealth arm beats DeepSeek: skip stage 2 and report that "DeepSeek already built the price". | $5 + $17 |
| M8 | magnitude, movement and Taylor pruning; random reinit; continual backprop (https://arxiv.org/abs/2306.13812) | BCE at matched active units; loss in the 2k steps after a shift | No gain over random reinit in 3 seeds: stop. | $5 |
| M9 | shuffled soup table | lottery-model R²; FLOPs to Brier 0.1 | Two sweep points with no trend: report the law as absent. | $4 |
| K1 | beliefs shuffled | survivors; Gini against the null; Brier | Monoculture in every arm: report Blume-Easley and stop. | CPU |

X1 reproduction gets $4 and there is a $5 reserve, for $50 in total. Wall-time caps: M6 and M7
10 h each; everything else 6-8 h.

## 4. Exciting vs expected

| Q | expected | exciting |
|---|---|---|
| Q1 | Wealth ≈ movement pruning; prequential books flip PRNG wealth negative. | Exact-LMSR books predict *coalition* ablations better than Taylor: "big trader = sum" holds. |
| Q3 | Coherent on AND, not on OR or D. | The fixed point makes D-prove prices coherent before resolution; arbitrageur E_C becomes sublinear. |
| Q4 | Weight decay or early stopping matches the humble layer. | The humble layer holds PRNG and CA30 at ln 2 in the multi-epoch regime, keeps LFSR, and beats boosting. |
| Q5 | Wealth arms tie DeepSeek; the global prior leaks. | Per-family wealth plus Exp3 matches DeepSeek with no coefficient, and beats it after a shift. |
| Q6 | Cumulative wealth freezes. | Wealth-gated rebirth beats continual backprop. |
| Q7 | The law is absent in gradient markets. | The K/Q collapse reappears in the net. |
| Q8 | Monoculture without niches. | Coexistence, with a Gini near 0.8 that beats the null. |

## 5. Out of scope

- **Harberger tax, self-assessment:** Kelvin asked for none now; no slot is contested yet.
- **Fractal equity, ownership matrices:** books leak 45-70%; T1 lemma 6 covers the theory.
- **Groves budget balance, strategic price-making units:** no unit here optimises its own wealth.
- **SAE feature books, IOI/copy suppression, grokking:** good interpretability, not Kelvin's
  questions.
- **MoD, logit-lens sequential settlement:** off the main line; unverified premise.
- **OLMoE measurement:** too much of the $50.
- **LM next-token and RL delayed-credit runs:** T1 defines the rule only.
- **New soup/Bend2 runs, wealth-net's own ablation:** other lanes own them.
- **Real calibrated-ai questions:** after a held-out family transfers.

## 6. Verification and delivery

- **Checks.** X1 checks each result against its `PREREG.md`.
- **Skeptics.** Five, at high effort, 2-3 workstreams each: hunt for leakage (train/eval
  overlap, the old pi-offset overlap), check seed variance and shuffled-label controls, ask
  whether a kill control explains the result, and confirm the stop rules were honoured.
- **Reproduction.** Rerun each headline number from the committed scripts on a fresh seed, in a
  clean copy; it must land within 2 sd.
- **Claims register.** `CLAIMS.md` has one row per claim: verified, refuted or unverified, with
  `checked_at`, the evidence path and the rerun command (policy 15).
- **Delivery (D1):** `IDEAS-NET.md` (3-line TL;DR, answers to Q1-Q8, the one-line novelty claim
  for Jessica); an HTML page at http://[ip-removed]:8765/traders-net/ (demos: soup vs net on
  one compute axis, wealth treemap with an in-sample/prequential toggle, humble-layer price
  trajectories, the expert economy); the Grafana dashboard `traders-net`; a hub PR adding
  `wiki/traders-net.md` (plain language, no AI attribution); a Kelvin handoff of at most three
  lines.
