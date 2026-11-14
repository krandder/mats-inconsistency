# MATS research inventory (3 November 2026)

Read-only inventory written 3 November 2026 by a subagent of opus-mats, for Kelvin. Nothing was run, edited or sent.
Every claim names its source. Abbreviations: GH = https://github.com/futarchy-fi/inconsistency-search/blob/main/
(repo, `main`, last MATS sync PR #23 on 1 Nov, commit c15146c); OUT = /home/agent/workspaces/mats/outbox/; WORK =
/home/agent/workspaces/mats/work/; CFT = /home/agent/workspaces/consistency-ft-next-run-20260919/docs/; CAI =
/home/agent/workspaces/calibrated-ai/. A GH link is given when the file is in the repo; files dated 1-2
November after 17:00Z are not in the repo yet (checked with `gh api .../contents/mats/reports` on 3 Nov).

Status words used below: reviewed (names the reviewer), unreviewed, registered (predictions hashed before the run),
exploratory, positive, negative, parked.

## TL;DR

- 8 groups. Groups 1-4 (September to 25 October) are LLM-forecast coherence, markets inside networks, learned search,
  and learned aggregation against exact Bayes. Groups 5-8 (26 October to 2 November) are the mechanism of in-context
  Bayes in small transformers (M2), ARC mean estimation measured (M1), ARC mean estimation theory (M1), and scale-free
  combination. Since 26 October nearly all compute has gone to groups 5-7.
- About 390 individual experiments, counted loosely (section 1.9). Most are exploratory. Since about 20 October most
  lanes pre-register predictions and get a GPT-6 or Gemini review before a claim.
- [removed]

## 1. Experiment groups

### 1.1 Coherence of LLM forecasts and consistency fine-tuning (13 Sep - 10 Oct; Market Pulse prep in Aug)
- Question: does making forecasts consistent, at answer time or through training, make them more accurate?
- Run: projection onto a coherent joint (24 Sep, CFT coherence-projection-20260924.md); structure vs averaging and
  balanced 16-question triangles (24-27 Sep); eliciting weights (CFT eliciting-weights-20260927.md); weekly world
  model (CFT weekly-worldmodel-20260927.md); strength scaling, registered (CFT strength-scaling-20260930.md); MiniBench
  347-question backtest (30 Sep); 0.6B consistency loss, 8 arms (23 Sep); distillation v1/v2 (CFT
  consistency-distill-v2-20260929.md); GRPO pilot, registered (8-10 Oct, CFT grpo-consistency-pilot-20261008.md);
  LIR E1-E6 (2-3 Oct); MiniBench bot live since 5 Oct. Owner: the calibrated-ai lane, not opus-mats
  (OUT OPUS-MATS-CONTEXT.md, "Research direction and coordination").
- Result (site text, quoted in CAI/experiments_page.py): "projecting Qwen3-32B's answers onto the logic cut log loss
  from 0.611 to 0.541 with no training ... and was not significant on 347 real MiniBench questions"; "Fine-tuning on
  consistent targets gives a small edge (-0.013 nats at 0.6B; -0.006 at 1.7B with the 235B teacher); both RL runs
  stopped before a test." Status: the lane's own reports, no independent audit found; mostly exploratory, two
  registered. The first 0.6B run is void (prompt bug), per the wiki page ekelvin-research-page.md (via `recall
  --read-wiki`). Mixed: projection positive on synthetic forms, no significant gain on real questions.
- Follow-on: the real-dataset plan (RL with a coherence term at Qwen3-8B, about $15-20k) is proposed, not run. [removed]
- Links: mats.ekelvin.com/experiments/ (streams "Coherence of forecasts", "Training and running forecasters");
  mats.ekelvin.com/forecast/ (live bot); CFT reports above; OUT market-pulse-*.md (Aug, no real data used).

### 1.2 Markets inside networks (September; PARKED 30 Sep)
- Question: can a network's parts be paid like traders, and does a market help it learn or think longer?
- Run: neurons (13 entries), paying the parts (33), Market ResNet and puzzles over depth (35), evolving traders /
  soup (8); catalogue CAI/trader-economy/CATALOGUE.md.
- Result (site text): "training on it never beat backprop, and the 100x depth result came from random training order
  ... 19 pre-registered experiments, zero gains." Negative. Parked: /home/agent/workspaces/trader-theory/
  THEORY-4.md is titled "Round 4: stop" (30 Sep).
- Links: the site's archived "Markets inside networks" stream (ARCHIVE_STREAMS in CAI/experiments_page.py).

### 1.3 Learned search over a world model ("inconsistency search") (5 Oct - 2 Nov)
- Question: can a net trained on outcomes learn a world model, and does searching its own beliefs, or a learned
  search that minimises inconsistency, improve forecasts and generalise?
- Run:
  - ShipHunt (5-11 Oct): reveal-chain TD reaches per-cell training's 10k-board accuracy "with about 2.8k boards
    (0.28x, CI 0.20-0.37)" (/home/agent/workspaces/eurisko-study/li-net/smallnet/RESULTS.md). Positive,
    audited.
  - Overnight rounds 1-3 (15 Oct; GH docs/papers/overnight-results-paper-2026-10-15.md) and rounds 4-9 (16-17 Oct; 18
    registered 3-seed studies; GH docs/papers/experiments-rounds-4-9-paper-v3.md). Board-opus audit of v2: "not fit to
    publish yet ... 12 wrong numbers" (GH docs/audits/experiments-rounds-4-9-paper-v2-audit-2026-10-18.md); v3 fixes
    them and was not re-audited. Site text: "Search helped only once the net knew the local rule and one pass fell
    short".
  - Gemini's GLC-AOM campaign, R10-R25 (18-19 Oct): a 59-agent critique found, of 165 claims, 87 artefacts and 12
    supported, and "not one capability claim that passes a pre-registered, unchanged rule on a genuine run" (GH
    docs/audits/gemini-campaign-critique-2026-10-19.md). Negative. R26 parked by Kelvin on 19 Oct (OUT
    OPUS-MATS-CONTEXT.md line 461).
  - R27A/B/C (20 Oct, frozen under GH docs/PROTOCOL.md, board-opus audits GH docs/audits/R27{A,B,C}-audit.md): Sudoku
    "solves 0 of 256 boards"; N-Queens passes weakly; R27C "the proximity edge matters, the network adds nothing".
    Negative.
  - Generalisation diagnosis (1 Nov, OUT generalization-diagnosis-2026-11-01.md, reviewed by a critique pass): R27A's
    N-Queens rule holds with board size but valid boards fall 22% to 5% (10x10 to 16x16), because nothing finishes
    near-solutions.
  - Toy search (1-2 Nov): DNAR pilot learns BFS exactly to 1,600 nodes when the design supplies the step (OUT
    toy-search-generalization-2026-11-01.md); follow-up, registered, astra-reviewed twice: a plain recurrent GNN trained
    on the consistency energy alone, no labels, "builds a valid BFS tree on every test graph in 2 of 3 seeds" up to
    1,600 nodes, without learning the exact tie-break (OUT toy-search-followup-2026-11-02.md). Positive, small (3
    seeds, BFS only).
- Links: site streams "Search over a learned world model" and "Searching any-order world models"; GH docs/specs/,
  GH docs/papers/, GH docs/audits/, GH results/; direction note GH docs/direction/research-direction-note-v7-2026-10-19.md
  (also mats.ekelvin.com/direction/ and Google Doc [removed], per OUT
  OPUS-MATS-CONTEXT.md line 425).
- Parked sub-track, Löb/PrudentBot cooperation (1-2 Nov): an energy with depth-indexed beliefs has PrudentBot as its
  unique zero, but "the depth-0 default does all the selecting", so inconsistency minimisation "can host Löbian
  cooperation but does not produce it" (OUT lob-neural-2026-11-01.md, unreviewed as a whole; OUT
  energy-prudentbot-2026-11-01.md, unreviewed; OUT lob-toys-2026-11-01.md, Toy 1 only, unreviewed). Parked by Kelvin
  on 2 Nov "until fable-5.5 gives us more insights" (OUT overnight-report-2026-10-29.md, 01:32Z entry).

### 1.4 Learned aggregation against exact Bayes (20-29 Oct)
- Question: can a learned aggregator over agents' rich messages, or a small PFN transformer reading the evidence,
  match exact Bayes on worlds where Bayes is computable? Kelvin's bar: a compressed consensus that learns as well as
  one transformer reading everything, on the same data (OUT stage1-design-fit-review-2026-10-21.md).
- Run, stage 1 (20-25 Oct; harness GH aggbench/README.md; runs GH results/explore/): lanes L1-L10, role keys (240
  runs), rich agents (378), query-dependent agents (69), ForecastWorld, compression batches 4-10, atom and GLS ordering
  energies; 16 adversarial "gate" reviews by Opus subagents (OUT gate-*.md).
- Result, stage 1: "Kelvin's challenge is still not met blind (+0.051 at b=6, +0.024 at b=8)" (OUT
  gate-batch8-2026-10-22.md). Role keys land "within 0.0044 nats of the joint ceiling", gate PASS with fixes (OUT
  role-keys-learned-2026-10-22.md, OUT gate-role-keys-2026-10-22.md); rich agents match within 0.001 nats but "two
  headline readings FAIL" at the gate (OUT gate-rich-agents-2026-10-23.md). Exploratory, gate-checked.
- Run, PFN worlds (26-29 Oct; GH mats/worlds/): deduplication by depth, tiny PFN on a three-source world, W1 junta, W2
  market (pilot, W2b, W2c), W3 chains, World FA, estimability penalty, C6 note, astra's binding-plateau study.
- Result, PFN worlds (GH mats/README.md status table): every model removes part of an exact copy, none all of it (two
  adversarial reviews, GH mats/reports/pfn-dedup-depth-2026-10-26.md); one tiny-PFN config is within 0.02 nats of
  Bayes on test NLL (no dedicated review, GH mats/reports/tiny-pfn-2026-10-27.md); W2b passes in-context source
  calibration at N = 96 and fails at N = 512 (unreviewed). Plateau paper: four pages, Codex-reviewed
  (GH mats/reports/binding-plateau-paper-2026-10-28/).
- Parked here: World FA (Kelvin 29 Oct, "We can park the World FA for now"; binding in flat tokens is the blocker; GH
  mats/README.md); W3 chains ("Parked. No model learns even one hop", same table); estimability penalty (registered
  SUCCESS at a task cost of 25-38% of the remaining gap, GH mats/reports/estimability-penalty-2026-10-27.md; parked by
  Kelvin 27 Oct: "I don't think a loss that makes activations more gaussian is the way to go", OUT
  pending-discussion-kelvin.md); wealth-based proposition market (OUT parked-proposition-market-2026-10-22.md).
- Links: mats.ekelvin.com/fa/ (World FA explainer, GH mats/pages/fa-page/). Stage-1 reports are local only (OUT) and
  have no site page.

### 1.5 How small transformers approximate Bayes in context, and why they deviate (M2; 27 Oct - 2 Nov)
- Question (target M2, OUT cm-interp-program-2026-10-28.md, "The bar"): a general account of how transformers
  approximate Bayes in context and why they deviate, predicting the departure from the architecture.
- Run: tiny-PFN interpretability step 1 (unreviewed); W2 mechanism (Codex review: routing holds, algorithm
  underdetermined); noisy-channel phase A rounds 1-2 (two adversarial reviews and astra's audit of round 1); phase B
  episodes E1-E86 (each registered with a hashed amendment and Codex-reviewed before a claim; OUT
  cm-interp-phaseB-2026-10-28.md, handoff at top); phase R on random noisy MLPs (Codex-reviewed, revised).
- Result: about 29% of the nets' KL to Bayes is dependence on the pattern of a source's past errors, mostly signs; a
  sign-free input cuts the KL to a third to a half of control; a net's solution type (A or B) comes from its
  initialisation and tracks the head count in block 1 (GH mats/README.md, phase B row). Latest: E85 and E86 support
  "one head occupying the summaries job early keeps these initialisations B" by registered rule (OUT
  cm-interp-phaseB-2026-10-28.md handoff; OUT overnight-report-2026-10-29.md, 18:59Z and 22:56Z on 2 Nov). Status:
  reviewed episode by episode; positive and mechanistic, on toy worlds; no claim yet across tasks and LLMs (M2's full
  bar). Nothing in flight (phase B handoff, 22:54Z 2 Nov).
- Links: GH mats/interp/phaseA/, GH mats/interp/phaseB/, GH mats/interp/phaseR/, GH
  mats/reports/cm-interp-phaseB-2026-10-28.md (repo copy through E76; E77-E86 local only);
  mats.ekelvin.com/interp/ (proposal page, GH mats/pages/interp-page/).

### 1.6 ARC mean estimation, measured: can anything beat sampling on ARC's deep random MLPs? (M1; 26 Oct - 2 Nov)
- Question (target M1, OUT cm-interp-program-2026-10-28.md): a sample-free estimator that matches or beats sampling on
  deep random MLPs at depth growing with width (e.g. 256 x 32), or estimates AlgZoo M_{16,10} as well as sampling.
  Since 2 Nov ~04:10Z: "a major advance on ARC 256x32", either a positive or "a major argument that it seems hopeless
  and ARC should update" (OUT overnight-report-2026-10-29.md, 04:14Z on 2 Nov).
- Run (lanes; reports in GH mats/reports/ unless marked local): arc-noisy (Parts 1-3b), algzoo-noisy,
  boundary-family, nongauss-family, sparse-codes, narrow-deep, narrow-deep-m1, unrestricted-estimators (to 31 Oct, in
  repo); nd-exact, arc-update, scale-mode, future-aware, preint, randomness (2 Nov, local OUT *-2026-11-02.md);
  adaptive-regions (certified-linear probe, learned linear experts, cone tree; running on 2 Nov evening, WORK
  adaptive-regions/); a check of the Phase-1 leaderboard (WORK phase1-board/FINDINGS.md).
- Result: "No positive. Every deductive method tested at 256 x 32 wins at most a constant factor, in a budget window
  that closes within ~5% of the contest budget; at the budget RQMC beats those that fit by 7-65x, and hybrids built on
  RQMC reach 0.8-1.0x of it" (OUT synthesis-draft-2026-11-02.md, TL;DR; draft, reviewed by main, held). ARC's own
  code: order 2 wins a ~2x window near 450 passes on 4 of 5 nets; RQMC is 7-40x ahead at the budget (OUT
  arc-update-2026-11-02.md). Board-fable adjudication: "bar NOT reached as 'hopeless, ARC should update': restricted
  negative" (OUT overnight-report-2026-10-29.md, 10:24Z on 2 Nov). The public board's #1 entry, which looked 10-50x
  below MC, comes from "fitting the 50 public leaderboard MLPs using the per-MLP errors every submission page
  publishes" (WORK phase1-board/FINDINGS.md, verified for 5-6 Aug, inferred for 10 Aug). The one confirmed positive is
  on tails: a sample-free readout of ARC's augmented description puts +0.32 [0.23, 0.41] more depth-8 1e-8 events
  within 50% than ARC's readout, on 20 fresh nets (registered, GPT-6-reviewed; GH
  mats/reports/arc-noisy-deep-2026-10-29.md). Pre-integration and structured designs: no (OUT preint-2026-11-02.md,
  three astra rounds; OUT randomness-2026-11-02.md, astra-reviewed: "Nothing beats RQMC exponent"). Learned-expert
  results of 2 Nov were withdrawn ("everything before ... does not count", Kelvin, OUT overnight-report-2026-10-29.md
  18:56Z); on 4x4 nets an exact-cell estimator reaches the truth's noise floor (21:23Z, exploratory).
- Status: negative for the methods tested, reviewed lane by lane; the overall negative is restricted (synthesis §6).
- Links: GH mats/interp/{arc-noisy,algzoo-noisy,boundary-family,nongauss-family,sparse-codes,narrow-deep,
  narrow-deep-m1,unrestricted}/; OUT synthesis-draft-2026-11-02.md (the best single summary; not in repo).

### 1.7 ARC mean estimation, theory: hardness and the bits that matter at a layer (M1; 29 Oct - 2 Nov)
- Question: why is beating sampling hard on these nets, and what information at layer l the output needs (Kelvin's
  "do correctly compressed bits get scrambled", OUT OPUS-MATS-CONTEXT.md, 04:15Z update of 2 Nov).
- Run: hardness, lyapunov-dimension, random-phases, gate-entropy, cut-rank, cut-rank-ext, kstar-frames,
  tractable-gates, stationary, stationary-numerics, input-bits (in repo); gram-chain, relevance-accounting,
  bits-bounds, bit-accounting phases 1-2, bit-accounting-arc, backward-bits (1-2 Nov, local OUT).
- Result: with only runs of the net, sampling is optimal up to about 4x (proven); worst case, no deterministic method
  matches sampling under #NSETH (proven); for random nets with depth proportional to width no general lower bound is
  possible with current tools (GH mats/reports/hardness-mean-estimation-2026-10-29.md, Gemini and GPT-6 reviews). The
  exp(c k*) premise is refuted; one-way linear summaries need (1/eps)^Theta(n) numbers (GH
  mats/reports/cut-rank-lower-bound-2026-10-30.md, GH mats/reports/summary-2026-10-31.md). Theorem F7 (unreviewed
  when the synthesis was drafted; the preint report is final after three gpt-6-astra rounds): rules linear in f, fixed independently of the weights,
  cannot beat MC by more than 2/c (OUT preint-2026-11-02.md; synthesis §3). Bits: consecutive-layer codes map almost
  linearly (R^2 0.98-1.00 from L/4), so codes are not scrambled layer to layer; the bits are scrambled relative to the
  coordinates in which the law is computable (synthesis §5; OUT bit-accounting-2026-11-02.md, phase-2 report v3,
  astra r3). A heuristic relevance accounting predicts subspace counts within 2x on 55-56 of 56 cells of eight fresh
  ARC nets (registered, three GPT-6 rounds; OUT relevance-accounting-2026-11-01.md). Status: mixed proofs (some under
  stated hypotheses) and registered measurements, reviewed; explanatory, not an estimator.
- Links: GH mats/interp/{hardness,lyapunov-dimension,cut-rank,cut-rank-ext,kstar-frames,tractable-gates,stationary,
  stationary-numerics,input-bits,random-phases,gate-entropy}/; OUT gram-chain-explainer-2026-11-01.md (explainer).

### 1.8 Scale-free information combination (29-30 Oct)
- Question: does combining evidence need a tangled dense network, or can a hierarchy of sources that interact only
  through sums do it, and how does this relate to Ngo's scale-free agency?
- Run: scale-free theory (+ registered addendum M3-M5 on phase B nets); scale-free experiments (trees against dense
  transformers); Ngo comparison; an explainer page with brute-force checks.
- Result: told which tokens belong to which source, a sum tree plus one broadcast round matches the dense transformer
  at equal data and reaches KL 0.001 at N = 12 with 4x data, about 3x below the dense net; dense attention is needed
  for routing and for finding hidden shared errors (GH mats/reports/scale-free-exp-2026-10-29.md, two GPT-6 reviews).
  Five of seven registered predictions held; "Layer 1 routes, layer 2 weights" failed the causal test (OUT
  morning-summary-2026-10-30.md). Ngo link: "in a log market scoring rule (LMSR) market over the open latents, a
  myopic Bayesian trader buys its log-likelihood"; his subagents are strategic and ours are not (GH
  mats/reports/scale-free-agency-ngo-2026-10-29.md, two GPT-6 reviews). Positive, toy scale.
- Links: GH mats/interp/{scale-free-theory,scale-free-exp,scale-free-ngo}/; GH mats/pages/scale-free-explainer/ (not
  deployed to the site; a copy was opened on Kelvin's Mac, OUT morning-summary-2026-10-30.md).

### 1.9 Counts and how they were counted
- Groups: 8 (1.1-1.8). Parked tracks inside them: markets inside networks, R26, World FA, W3 chains, estimability,
  proposition market, Löb/PrudentBot (with the GAN-like description game and binary nets, OUT OPUS-MATS-CONTEXT.md
  line 15), tensor-transformer exploration (OUT pending-discussion-kelvin.md, 26 Oct).
- Individual experiments, roughly 390:
  - Groups 1-3 to 20 Oct: about 226 site entries (coherence + forecasters + readings side experiments 50, markets 89,
    search and glc streams 87), counted from the STREAMS table of CAI/experiments_page.py (232 rows; readings and
    non-experiments excluded). An entry is sometimes a family of runs, sometimes a note.
  - Group 1.3 after 20 Oct: about 6 (generalisation diagnosis, toy-search pilot and follow-up, three Löb reports).
  - Group 1.4: about 20 stage-1 experiments (named lanes and batches; the 16 gate reviews are reviews, not counted)
    plus about 12 PFN-world experiments (rows of GH mats/README.md).
  - Group 1.5: 86 registered phase B episodes (E1-E86) plus phase A (2 rounds), phase R, tiny-PFN interp, W2 mech:
    about 91.
  - Groups 1.6-1.7: about 33 lanes, each a registered experiment set with several predictions (rows of GH
    mats/README.md plus the 1-2 Nov reports in OUT).
  - Group 1.8: 4.
- Counting a lane or site entry as one experiment undercounts runs (one lane holds dozens of registered cells) and
  overcounts notes. Use "about 8 groups, a few hundred experiments, about 35 lanes since 26 October" in conversation.

## 2-3. [removed]

The sections on contacts and correspondence were left out of the public copy.
