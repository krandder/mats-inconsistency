# Scope critique: completeness and taste (2026-09-27)

Critic over the six scoping views (theorist, moe-systems, interpretability,
economist, experimentalist, kelvin-taste). Inputs read: CONTEXT.md,
REPORT-SOUP.md sections 1-3, the six scope-*.md summaries, and
wealth-net/results/summary.json. wealth-net/RESULTS.md does not exist yet.

State check (read-only, checked at 2026-09-27T21:58Z, `results/summary.json` and
`train-s1.log`): wealth-net is at step 11.5k of 60k. The cumulative conservation
residual at the first three cuts is still -0.70/-0.51/-0.51 (it was
-0.64/-0.45/-0.45 at 8.8k, so it is getting worse). The run and its planned
ablation test belong to the wealth-net lane. The larger study must reuse its
checkpoints and must not duplicate that ablation test.

## 1. Consensus (all six agree, so treat it as settled scope, not open questions)

- The ledger `a * dR/da` is first-order Taylor attribution. Summed over training
  it is Molchanov/Sanh movement-pruning importance with the sign kept, and under
  plain SGD it is roughly squared-norm growth over 2*eta. Anything new has to
  come from settlement (prequential, exact LMSR), from dynamics (depreciation,
  birth, gating) or from multi-scale conservation, not from the ledger itself.
- In-sample books pay memorisation: pi has the largest readout wealth and the
  worst held-out BCE. Prequential settlement is a prerequisite for everything.
- The tangent settlement z(y-p) has zero mean at calibration. Exact LMSR
  (log-score gain, integrated from a named prior) is the value measure.
- Books do not conserve across attention/LN, and gates double-pay (Euler, degree 2).
- The convex pool (l0+sum k l)/(1+sum k) cannot add independent evidence.
- DeepSeek loss-free bias is already the capacity price. Wealth can only take
  over the anti-collapse role, and compounding wealth causes collapse unless it is
  paired with exploration or birth.
- Cumulative wealth rank stability (0.98) is an artefact of cumulative sums.

These points should go in the study plan as premises. They do not need to be
re-derived.

## 2. Missing (what Kelvin or Jessica Taylor would ask)

1. **Deductive settlement, the heart of logical induction.** Every proposal
   settles on sampled outcomes from streams. LI prices sentences *before* they
   resolve, and it is judged on coherence with a deductive process (provable
   sentences go to 1, refutable ones to 0, exclusive-exhaustive sets sum to 1).
   It is not judged on log loss. Jessica's first question will be: what happens
   with questions that resolve late or never, or that are settled by proof (for
   example "A or not A", or an implication between two open questions)? Add a
   question family with delayed or deductive resolution, and report whether
   prices become coherent before resolution. Garrabrant et al.
   https://arxiv.org/abs/1609.03543 (id from memory, unverified).
2. **Fixed-point prices inside a layer.** Several proposals note that LI's power
   comes from prices being a fixed point of price-reading traders. No proposal
   builds one. experimentalist E7 does one extra round, which is not a fixed
   point. The minimal build is a DEQ-style market layer: traders read price p
   and emit demand d_i(p); solve sum d_i(p*) = 0 by Newton; use implicit
   gradients. Compare it with one-pass and two-pass layers on AND/OR coherence.
   DEQ: https://arxiv.org/abs/1909.01377 (unverified).
3. **Who pays for shared computation.** Kelvin's "compute = wealth" and "share
   core computation" pull against each other, and no proposal says how the FLOPs
   of a shared trunk get charged to the traders that use it. This is a
   cost-sharing problem: Aumann-Shapley or Shapley cost sharing over usage, or
   marginal-cost pricing. It is needed before any wealth budget in FLOPs can
   mean anything. Friedman and Moulin 1999 (unverified).
4. **Late-resolving credit and delayed settlement.** In LMs (next token) and in
   RL, the outcome arrives late or is shared. Say how the books settle with
   delay, or explicitly scope this out.
5. **One shared substrate and metric panel.** Six views propose six substrates
   and metrics. The study needs one fixed panel that every arm reports:
   held-out/prequential BCE per family, ECE, AND/OR coherence violations,
   exploitability slope E_C(t), wealth Gini against a null model, FLOPs per
   Brier, and dead or bankrupt unit fraction. Use one substrate: the soup
   streams with the pi family replaced by a seeded PRNG-parity family with no
   shared offsets, plus a small text corpus only for the MoE arm.
6. **Wealth over time as a live view.** Kelvin said he wants to "get a small
   network and see how the wealth works or evolves over time". The proposals
   treat this as a demo afterthought. It should be a deliverable: Grafana panels
   per run (wealth by family and scale, Gini, rank churn, births and bankruptcies,
   conservation residual), with a re-run check, following fleet policy 15 and
   the Grafana-over-HTML preference.
7. **Boosting reading of idea 2.** Traders that correct the market's log-odds
   with an offset are stacking or gradient boosting with the market as the
   offset. No one checks whether "humble traders" are just boosting with
   shrinkage. That comparison is the natural kill control for the humble layer.
   Friedman 2001 gradient boosting (unverified).
8. **Lottery-ticket framing.** The soup's K/Q law and wealth lock-in both look
   like lottery-ticket effects (Frankle and Carbin,
   https://arxiv.org/abs/1803.03635, unverified). The K/Q item should state that
   link explicitly.
9. **Stop rules and budget.** Only the experimentalist gives kill criteria. Every
   item in the larger study needs a stop rule and a cap. The soup spent $10 of
   $200. The MoE items as proposed total $100-300.

## 3. Drop or defer

- **Wealth Gini as a universal** (kelvin-taste): drop. The economist shows that
  Gini of the positive part is fragile with signed stocks and different
  population sizes. Keep only a null-model footnote in the metric panel.
- **Fractal equity / Leontief ownership / Owen-value ownership** (economist
  fractal-equity and ownership-rule; kelvin-taste Q10): defer until the books
  conserve. Ownership over books that leak 50-70% is meaningless. Keep one
  theory paragraph: fixed-fraction recursion reduces to a flat mixture, and
  power comes only from rebalancing (theorist). That item is answered in-scope,
  so no study time is needed.
- **Market-priced depth / MoD** (moe E6): drop. It is optional and off the main line.
- **Call vs sequential settlement along the logit lens** (theorist E3): defer.
  Intermediate readouts are uncalibrated, so the premise needs a tuned-lens
  check first. Low value for its cost.
- **Budget-balance impossibility (Holmström/Groves) and Nash of price-making
  units** (economist Q4, theorist Q9): merge them, then defer to a theory memo
  of a few hours. They matter only once units optimise their own wealth, which
  no experiment here does.
- **Feature-level books on SAEs** (interp E7) and the **IOI/copy-suppression
  sign test in GPT-2** (interp E6 part): defer. They are good interpretability
  work but do not answer Kelvin's questions. Keep the cheaper tracr/InterpBench
  ground-truth check as part of the validity bake-off if time allows.
- **Eisenberg-Gale / Fisher-market router** (theorist Q7): fold it into the
  priced-MoE literature step as one arm, only if the DeepSeek bias baseline
  leaves room.
- **Wealth-proportional expert capacity** (moe E2): keep it only as an arm in
  priced MoE. The critics expect it to lock in, and it makes stragglers worse at scale.

## 4. Merges (duplicates)

| merged item | sources |
|---|---|
| M1 Books v2 + audit | theorist E1+E2, interp E1+E4, economist books-audit, experimentalist E0, kelvin-taste Q1 |
| M2 Validity bake-off (single + coalition ablation vs LOO; baselines Taylor, movement, magnitude, IG, sampled Shapley) | interp E3, experimentalist Q2+E1, kelvin-taste E2, economist Hulten Q |
| M3 Currency definition and lock-in (cumulative / EMA / depreciated / fixed-share / Kelly; regime shift) | experimentalist E2, economist depreciation Q, interp Q3+E5, experimentalist E6 |
| M4 Close the books (QK, softmax, LN, gates; IG/conductance/AttnLRP) | interp E2, kelvin-taste Q2, moe gate-split Q, economist ledger Q |
| M5 Exploitability and coherence harness | theorist Q4+E6, experimentalist Q-coherence+E7, kelvin-taste E1 coherence part |
| M6 Humble market layer (pool, utility axis, bounded bets, fixed point) | experimentalist E3, kelvin-taste E1, theorist utility Q, economist tatonnement Q |
| M7 Priced MoE | theorist Q6+E5, moe E0/E1/E3/E4, economist router-as-market, experimentalist E5, kelvin-taste E3 |
| M8 Wealth-gated execution in a dense net | theorist E4, economist compute-price-exit, experimentalist E6 (reinit arm) |
| M9 Soup-to-net transfer (lottery fit, K/Q in nets, shared computation vs selection) | theorist E7, moe E5, experimentalist E4, kelvin-taste Q4+Q7 |
| M10 Literature novelty memo | theorist lit, moe lit, economist lit Q, kelvin-taste Q9 |

## 5. Top ten, ordered

1. **M1 Books v2 + audit** (prequential and exact-LMSR/path settlement,
   market-mean baseline, regression on norm growth with an SGD rerun). Every
   later item inherits the currency. Hours on CPU with existing checkpoints.
2. **M10 Literature novelty memo** (Balduzzi 1401.1465, Storkey, Economy of Minds
   2606.02859, MoB 2512.10969, movement pruning, Chang 2007.02382, primal-dual
   loss-free 2512.03915). Hours. It stops re-derivation and gives the one-line
   "what is new" claim that Jessica will ask for first.
3. **M2 Validity bake-off against true ablation, including coalitions.** It
   decides whether wealth is more than signed Taylor pruning. Stop rule: a tie
   with Taylor or movement pruning means "report it and stop pruning work".
4. **M5 Exploitability and coherence harness, with delayed/deductive questions
   (missing item 1).** It gives one LI-style metric across soup, net and MoE, and
   is the lens Jessica will judge everything by.
5. **M6 Humble fixed-point market layer** (convex vs additive pool, log vs
   exponential utility, DEQ fixed point, bounded bets; kill controls: early
   stopping, weight decay, boosting with shrinkage). This is Kelvin's "improve on
   the market here-and-there" plus the missing arbitrageur, tested together.
6. **M7 Priced MoE, staged**: router-gradient = relative-revenue identity, then
   DeepSeek bias baseline vs shadow-exploration (Exp3) credit vs per-market
   wealth prior vs bankruptcy/birth, on a small substrate before any LM run.
   This answers his MoE and balancing-loss question. The kill result "DeepSeek
   already built the price" is itself a finding.
7. **M3 Currency definition and lock-in under regime shift.** Wealth ranks
   freeze while income churns, so depreciation or fixed-share decides whether
   gating by wealth can work at all. Cheap: books only, on M1 runs.
8. **M8 Wealth-gated execution in a dense net** (Hedge gate with a Fixed-Share
   floor, compute-price exit, rebirth), compared with pruning at matched active
   units. This is step 2 of Kelvin's "smallest modification". Run only after
   M1-M3 pass.
9. **M9 Soup-to-net transfer**: fit the lottery model to the soup table first
   (hours), then compare net vs soup at equal FLOPs on LFSR/CA110 and sweep
   active traders per question. It tests whether the K/Q law is about markets
   or about evolutionary search, and it quantifies "share core computation".
10. **Kelly-pool minimal economy with a live wealth view** (economist
    kelly-pool, plus missing item 6). It asks whether thousands of traders
    coexist or collapse to one (Blume-Easley), and it is the literal "small
    network, watch wealth evolve" that Kelvin asked for.

Just below the cut: M4 close the books (needed for any fractal or ownership
claim; do it after M1 shows the path ledger is worth keeping), and the
shared-compute cost-sharing rule (missing item 3), a theory memo of a few hours.

## 6. Notes on proposals that may be wrong

- theorist and economist both assert the norm-growth identity as exact under
  plain SGD. It holds per step only to first order, and only for units whose
  income is -<w_out, g_out>. With Adam and a moving input it is approximate.
  E1 should report the residual, not assume the identity.
- The moe view says the softmax router logit "already is a Hedge log-wealth".
  That holds for the gate-weighted relative-revenue gradient, but the router is
  context-dependent (a linear function of the token), so its "wealth" is not a
  per-expert stock. The per-market vs global comparison (moe E3) is the right test.
- The experimentalist's "LFSR priced by the net" came from a partial run. Confirm
  it at 60k before anyone builds M9 on it.
