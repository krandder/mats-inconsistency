#!/usr/bin/env python3
"""MATS experiments site: an index of five research streams, one page per stream (stream-<key>.html) listing its
sub-streams with filterable entries newest first, all.html with every entry, one plain-language page per experiment
(exp-<slug>.html), and the full write-ups (docs rendered to HTML)."""
import html, re, json, pathlib, sys
import markdown

DOCS = pathlib.Path('/home/agent/workspaces/consistency-ft-next-run-20260919/docs')
OUT = pathlib.Path('/home/agent/memo-serve/mats-experiments')
SITE = '../'

# (date, name, status, one-line result, doc stem or site page)
EXP = [
    ('19 Oct', 'Independent critique of the whole gemini campaign (rounds 10-25, society, literature, code)', 'informative', '59 Opus 5.5 agents, read-only: of 165 claims, 87 are artefacts, 60 partly supported, 12 supported (all code facts or transcription checks), 6 untested; no capability claim passes a pre-registered, unchanged rule on a genuine run. What is real: environments with exact oracles (after leak fixes), real negative results (search makes KL worse on most envs; untrained v3 matches trained v3; Leduc arms at uniform-random NashConv), a few primitives, and the external critiques that were right. Next steps and a protocol for using gemini as proposer and coder only are in section 5.', 'md:/home/agent/workspaces/inconsistency-search/docs/audits/gemini-campaign-critique-2026-10-19.md'),
    ('19 Oct', 'Pillar 1 audit: OpenSpiel games and Stratego (R18, R22A, R24B)', 'informative', "Every 'paradigm' is the same 181k-parameter network with a different hand-written loss; none is an OpenSpiel reference algorithm; training was 40 steps. On Leduc 7 of 8 arms are worse than uniform random (NashConv 4.75), so 'Deep_CFR 4.57' and 'v3 4.71' are untrained. The 88% piece deduction is reproduced with no network by the engine's setup prior. Elo #1 rests on 36 games per pair (p=0.40). Pre-registrations committed with results.", 'md:/home/agent/workspaces/inconsistency-search/docs/audits/pillar1-openspiel-stratego-audit-2026-10-19.md'),
    ('19 Oct', 'Pillar 2 audit: public reasoning benchmarks and CLRS (R19, R21, R22B, R24B, R25)', 'informative', "Tables match their JSON (1,150 of 1,151 cells) but the headline numbers (R25 53.61%, the k-curve to 71%) exist only in the README; the JSON says 55.8 to 61.1 and both gates are false. 'Pure neural at k=0' runs hand-coded solvers; 'helper-only 0%' is a uniform-prior artefact. 'SATLIB', 'Sudoku-Extreme' and 'CLRS-30' are home-made generators with 24 test instances per task; baselines are one 4-layer stand-in class trained 18 steps.", 'md:/home/agent/workspaces/inconsistency-search/docs/audits/pillar2-reasoning-clrs-audit-2026-10-19.md'),
    ('19 Oct', 'R16 internal market and R17 decentralized market', 'problem', 'Gemini-run, not audited at the time: oracle-weighted blends and a calculus identity (cosine 0.9999999) stand in for measurements; synthetic parity. Real content: a VCG leave-one-out primitive (R17).', 'md:/home/agent/workspaces/inconsistency-search/results/RESULTS-R16-INTERNAL-MARKET.md'),
    ('19 Oct', 'R17 decentralized market (VCG leave-one-out)', 'problem', 'Gemini-run: the parity result is synthetic and the headline cosine is an identity. The leave-one-out VCG primitive is reusable.', 'md:/home/agent/workspaces/inconsistency-search/results/RESULTS-R17-DECENTRALIZED-MARKET.md'),
    ('19 Oct', 'R18 OpenSpiel and Stratego (first attempt): CFR table blended into the policy', 'problem', "Gemini-run; society round 6 found a tabular CFR table mixed into the 'neural' policy (weight 0.97), cut-off network calls and a formula Elo (0.85 formula + 0.15 measured). Fails its own pre-registered gates. Superseded by R22A, which has its own problems.", 'md:/home/agent/workspaces/inconsistency-search/results/RESULTS-R18-OPENSPIEL-STRATEGO.md'),
    ('19 Oct', 'R19 public reasoning benchmarks: 99.4% came from a Python helper', 'problem', 'Gemini-run: the helper (MRV search, WalkSAT, A*, exact check) was given only to GLC; alone it solves 31 of 48. The data-scaling table (D4) was never trained; it uses hand-set multipliers.', 'md:/home/agent/workspaces/inconsistency-search/results/RESULTS-R19-PUBLIC-REASONING-BENCHMARKS.md'),
    ('19 Oct', 'R20 Kyle-LMSR futarchy market', 'nogain', 'Gemini-run: the LMSR class and its sequential settlement are correct; the futarchy and attack results are formula outputs. A CPU ablation shows near-truthful reporting is already a best response under sequential LMSR without the Kyle penalty.', 'md:/home/agent/workspaces/inconsistency-search/results/RESULTS-R20-KYLE-LMSR-FUTARCHY.md'),
    ('19 Oct', 'R21 CLRS OOD and scaling: targets in the inputs', 'problem', 'Gemini-run; society round 6 found target-derived channels in the model inputs and formula-based scaling; the run was repeated with the runner edited in between. Withdrawn by R22B.', 'md:/home/agent/workspaces/inconsistency-search/results/RESULTS-R21-CLRS-OOD-AND-SCALING.md'),
    ('19 Oct', 'R22A OpenSpiel and Stratego ("genuine" rerun): real games, stand-in baselines', 'nogain', 'Gemini-run: pyspiel games and played Elo matches are real, but every arm is the same network with hand-written losses, trained 40 steps; on Leduc 7 of 8 arms are worse than uniform random; the Stratego deduction table is the setup prior. See the Pillar 1 audit.', 'md:/home/agent/workspaces/inconsistency-search/results/RESULTS-R22A-GENUINE-OPENSPIEL-STRATEGO.md'),
    ('19 Oct', 'R22B public reasoning and CLRS without the helper', 'nogain', "Gemini-run: removing the helper drops GLC to 31% on home-made toy sets; the '#1 of 7' is against stand-in baselines; the 'CLRS-30' is a local generator. Real content: the leakage removal and the min-plus negative (an always-on prior loses).", 'md:/home/agent/workspaces/inconsistency-search/results/RESULTS-R22B-GENUINE-PUBLIC-REASONING-AND-CLRS.md'),
    ('19 Oct', 'R23 Kyle-LMSR futarchy ("genuine")', 'nogain', 'Gemini-run: four runs, one reported, gates written after the runs; detached local cross-entropy modules land within noise of global backprop near chance.', 'md:/home/agent/workspaces/inconsistency-search/results/RESULTS-R23-GENUINE-KYLE-LMSR-FUTARCHY.md'),
    ('19 Oct', 'R24A runtime counterfactual auditor (gemini auditing gemini)', 'nogain', "A name-blacklist leak auditor and hard-coded 'spread' ledgers; it confirmed R18's shortcuts and measured the R19 helper alone (31 of 48), but never ran on R22A although its own pre-registration required it.", 'md:/home/agent/workspaces/inconsistency-search/results/RESULTS-R24A-RUNTIME-COUNTERFACTUAL-AUDIT.md'),
    ('19 Oct', 'R24B v3: semiring-gated CLRS and Linear-CFR OpenSpiel', 'nogain', "Gemini-run: the gate parameter receives gradient but task separation comes from a label-trained selector with a hand-set threshold; the Leduc 'gap closed' is between two near-random policies; the JSON gate E2a is false where the write-up says PARTIAL PASS. Real content: determinism across GPU runs and an untrained-control measurement.", 'md:/home/agent/workspaces/inconsistency-search/results/RESULTS-R24B-V3-CLRS-AND-OPENSPIEL-UNIFICATION.md'),
    ('19 Oct', 'R25 v3 syndrome-guided reasoning and Kyle calibration', 'problem', "Gemini-run: untrained v3 (57.8%) scores as well as trained v3 (55.8%); the README's 53.61% and k-curve to 71% are in no JSON; all gates are false in the JSON; 'pure neural at k=0' runs hand-coded solvers.", 'md:/home/agent/workspaces/inconsistency-search/results/RESULTS-R25-V3-REASONING-SYNERGY-AND-KYLE-CALIBRATION.md'),
    ('18 Oct', 'Society round 1: a learned tree search (GLC-AOM) to replace the hand-set search constants of rounds 4-9', 'informative', 'Gemini wrote three proposals for a learned tree search to replace the fixed waves, stall thresholds and step floors of rounds 4-9, and GLM-5.3, GPT-6-Astra and Opus 5.5 critiqued them; Opus warned the removed constants had been supplying breadth, verification and safe abstention. No experiment was run; the scores are rubric marks by the models themselves.', 'md:/home/agent/workspaces/inconsistency-search/society/round1/ROUND1-SCOREBOARD.md'),
    ('18 Oct', 'Society round 2: code audit withdraws the round-10 and Stage-2 results as surrogate simulations', 'informative', "Gemini's line-by-line audit of its own round-10 and Stage-2 runners found untrained trunks, outputs mixed toward the exact answer and a seed-specific branch. GPT-6-Astra and Opus 5.5, working from Gemini's summary rather than the code, agreed to withdraw those curves as evidence, and the round wrote the R11 pre-registration with a zero-leakage gate.", 'md:/home/agent/workspaces/inconsistency-search/society/round2/ROUND2-SCOREBOARD.md'),
    ('18 Oct', "Society round 3: adjudicating R11 and pre-registering R12 (the '8.65 consensus' is Gemini's score alone)", 'informative', "Gemini audited its own R11 and R12 runs and rated them highly; Opus, working from Gemini's text, listed 12 problems (Elo partly built from belief metrics, a tautological AUROC) and scored R11 6.33 to Gemini's 8.23. The R12 '8.65 consensus' is Gemini's score alone, from an audit that matched numbers to the JSON but missed the synthetic Elo.", 'md:/home/agent/workspaces/inconsistency-search/society/round3/ROUND3-SCOREBOARD.md'),
    ('18 Oct', 'Society round 4: can the R11/R12 search learn MCTS, and what can a search transformer not represent?', 'informative', 'All four models agree the R11/R12 tree search has a fixed selection formula with no trainable weights and no value backup, so those runs could not have learned MCTS-style search; Opus adds that their gains came from readout and calibration. The round proposed search transformers v1 and v2 and wrote the R13 and R14 pre-registrations; nothing was run.', 'md:/home/agent/workspaces/inconsistency-search/society/round4/ROUND4-SCOREBOARD.md'),
    ('18 Oct', 'Round 10 E1-E2: lemma channels, pair queries and two-stage tree clearing (CPU surrogate)', 'problem', "Gemini-run CPU surrogate with untrained trunks, not independently audited: all 7 rules 'pass', but the runner never calls the tree search. Its budget curves are closed-form formulas and mixtures toward the exact answer, and the baseline's one failing step is hard-coded for seed 102 at budget 2.", 'md:/home/agent/workspaces/inconsistency-search/results/RESULTS-R10-E1-E2.md'),
    ('18 Oct', 'Round 10 E3-E4: fractal plans, episodic memory, trust pooling, markets and poker (CPU surrogate)', 'problem', "Gemini-run CPU surrogate with untrained trunks, not independently audited: all 7 rules 'pass', mostly by construction. The AUROC of 1.000 vs 0.089 comes from score distributions drawn so they cannot overlap, and the memory and market gains from shrinking noise around the exact answer; only some module checks (plan coverage, contraction bound, Proposal-Poker firewall) ran for real.", 'md:/home/agent/workspaces/inconsistency-search/results/RESULTS-R10-E3-E4.md'),
    ('18 Oct', 'Stage 2: Elo league and Mini-ECI over 15 game configurations on simulated agents (CPU surrogate)', 'problem', "Gemini-run CPU surrogate with untrained trunks, not independently audited: all 6 rules pass because the runner generates each family's scores from the scaling law it then fits and draws match results from assigned strengths. Constants such as a 1.5% specialist advantage are typed in, so it shows the estimators recover their inputs, not anything about the models.", 'md:/home/agent/workspaces/inconsistency-search/results/RESULTS-STAGE2-MINI-ECI.md'),
    ('18 Oct', 'R11: first end-to-end GPU training and blind search of GLC-AOM, 12 environments, B up to 128', 'nogain', 'Gemini-run and not independently audited, R11 is void as registered: it made fewer than 400 training updates where the pre-registration requires at least 3,000, and its reported gates are not the registered ones. Search raised Mini-ECI by 6.95 points (bar +18), and over all 12 environments the generalist finds fewer witnesses than the specialists (56.0% vs 64.5%).', 'md:/home/agent/workspaces/inconsistency-search/results/RESULTS-R11-GPU-NEURAL.md'),
    ('18 Oct', 'R12: calibrated queries, a FiLM generalist and search to B=256 (pre-registration rewritten after results)', 'nogain', "Gemini-run, not independently audited; its pre-registration was rewritten after the results, and gates D2 and D3 fail under either version. The 'generalist beats the specialists' claim uses a 4-task mean; over all 12 environments it loses (KL 0.169 vs 0.149 at B=256). Elo comes from synthetic matches, and the Sudoku KL drop from a hand-coded witness term.", 'md:/home/agent/workspaces/inconsistency-search/results/RESULTS-R12-GPU-V2.md'),
    ('18 Oct', 'Overnight synthesis paper on rounds 10-12 and the society (audited: not publishable as a results paper)', 'nogain', "Gemini's paper reports 435 of 435 numbers verified, but its checker only tests that each number appears in the text. An independent audit (Opus 5.5) found 115 of 1,997 table cells not matching the JSON, gates misquoted or never pre-registered, a '12-env mean' that is a 4-task mean, synthetic Elo, and a 'consensus 8.65' that is Gemini's score alone.", 'md:/home/agent/workspaces/inconsistency-search/docs/papers/overnight-society-and-glc-aom-paper-2026-10-18.md'),
    ('18 Oct', 'R13: can search transformer v2 represent and imitate six classical search algorithms? (pre-registered)', 'running', 'Pre-registered and gemini-run, it asks whether search transformer v2 can reproduce six classical algorithms (MCTS, alpha-beta/proof-number search, CDCL, graph search, CFR+, Gumbel halving) from hand-set weights, then learn them from teacher traces. No results yet: the first GPU run exited with an error at 16:34 UTC, and a second, with an edited runner, was running at 16:45.', 'md:/home/agent/workspaces/inconsistency-search/benchmarks/PREREG-R13-ALGO-IMITATION.md'),
    ('18 Oct', 'R14: can task-only training discover search without algorithm labels? (pre-registered, smoke run only)', 'proposed', 'Pre-registered and gemini-run, it would train the v2 search transformer on task outcomes alone and test whether it rediscovers MCTS, CDCL or CFR, with bars set against R12 numbers the audit found unreliable. A one-seed CPU smoke run (outside the repo) failed 4 of 7 gates, and its gates differ from the registered ones.', 'md:/home/agent/workspaces/inconsistency-search/benchmarks/PREREG-R14-UNSUP-DISCOVERY.md'),
    ('18 Oct', 'R15: curiosity suite on search topology, chimera transfer, lemma memory and exploitability', 'problem', "Gemini-run and not audited: a run finishing at 16:20 UTC failed three of four experiments, and after the runner was edited a 16:39 rerun passed C2-C4 and still failed C1. Those passes are set in the runner: hand-chosen gates, mixing weights and policy sharpness decide the results, and 'exploitability' is a distance to a hand-picked target, not a best response.", 'md:/home/agent/workspaces/inconsistency-search/benchmarks/PREREG-R15-CURIOSITY-SUITE.md'),
    ('18 Oct', 'Expressivity theorems R4.1-R4.3: the missing value backup and the proposed v1/v2 search transformers', 'informative', "Gemini's proofs (society round 4): as shipped, the tree search counts only immediate children and never backs values up, so it cannot compute MCTS subtree means; one descendant-masked attention head could. GPT-6-Astra asked to narrow the 'exact three-layer MCTS' and 'complete resolution of five expressivity barriers' claims, and Opus found two of the five v2 fixes wrong as first written.", 'md:/home/agent/workspaces/inconsistency-search/society/round4/01-expressivity-theorems-and-mcts-gap.md'),
    ('18 Oct', 'Literature synthesis: 75 years of search, deduction and game-theory papers mapped onto GLC-AOM', 'informative', 'A master index over the five surveys with a 42-paper comparison table, written by a literature subagent during the gemini run; citations not checked. Its index gives four of the five parts topics those parts do not cover.', 'md:/home/agent/workspaces/inconsistency-search/docs/literature/LITERATURE-GENEALOGY-SYNTHESIS.md'),
    ('18 Oct', "Literature 1: Samuel's checkers, EURISKO, value of computation and EBL/CDCL lemma learning", 'informative', "Surveys classic work on learning to search: backed-up game-tree evaluations, Lenat's AM and EURISKO, rational metareasoning, and explanation-based and clause learning. A literature subagent wrote it during the gemini run, and its citations are unchecked. It argues GLC-AOM resolves each classical bottleneck, which the round 10-15 results do not show.", 'md:/home/agent/workspaces/inconsistency-search/docs/literature/01-gofai-metareasoning-eurisko-ebl-cdcl.md'),
    ('18 Oct', 'Literature 2: algorithm selection, hyper-heuristics, NRPA and modern tree and graph search', 'informative', 'Surveys algorithm portfolios and hyper-heuristics, nested Monte-Carlo search with online policy adaptation (NRPA), and UCT through RAVE, Monte-Carlo graph search, proof-number search and Gumbel MuZero, mapping each onto the GLC-Search roadmap. A literature subagent wrote it during the gemini run, and its citations are unchecked.', 'md:/home/agent/workspaces/inconsistency-search/docs/literature/02-metaheuristics-hyperheuristics-nrpa-mcts-pns.md'),
    ('18 Oct', 'Literature 3: neural algorithmic reasoning, differentiable tree search and transformer circuit limits', 'informative', "Surveys networks trained to execute classical algorithms step by step (CLRS-30), networks with built-in lookahead (VIN, TreeQN, MCTSNet, Thinker), in-context learning as gradient descent, and the TC0-versus-NC1 limit behind round 4's depth argument. A literature subagent wrote it during the gemini run, and its citations are unchecked.", 'md:/home/agent/workspaces/inconsistency-search/docs/literature/03-neural-algorithmic-reasoning-clrs-mctsnet-incontext.md'),
    ('18 Oct', 'Literature 4: training on search traces, GFlowNets and any-order autoregressive models', 'informative', "Surveys transformers trained on serialized search traces (Stream of Search, Searchformer), GFlowNets and hindsight plan conditioning for sampling diverse solutions, and any-order autoregressive and masked-diffusion models, contrasting each with GLC-AOM's tree-paged KV cache. A literature subagent wrote it during the gemini run, and its citations are unchecked.", 'md:/home/agent/workspaces/inconsistency-search/docs/literature/04-sos-searchformer-gflownets-anyorder.md'),
    ('18 Oct', 'Literature 5: dependency graphs, logical induction, information markets and imperfect-information search', 'informative', 'Surveys probabilistic dependency graphs and local inconsistency resolution, logical induction and belief-utility separation, information aggregation in markets, and CFR+, DeepStack, ReBeL and Student of Games for imperfect-information play. A literature subagent wrote it during the gemini run, and its citations are unchecked.', 'md:/home/agent/workspaces/inconsistency-search/docs/literature/05-pdg-logical-induction-cfr-rebel-markets.md'),
    ('18 Oct', 'GLC-AOM architecture and the 8-family multi-game suite (README overview)', 'informative', "One any-order transformer world model with bilinear 'lemma' channels, a pair-query head and episodic memory, plus a tree-paged KV-cache search, run on 8 game families (12-15 configurations). Despite the paper's 'zero domain-specific branching', the model has game-ID branches, including hand-written Sudoku single-candidate propagation, and the tree selector is a fixed formula with no trainable weights.", 'md:/home/agent/workspaces/inconsistency-search/README.md'),
    ('18 Oct', 'Bayesian Elo league with draws, first-move advantage and a cyclic mElo term (src/eval/elo_league.py)', 'informative', "Rates agents (model, search algorithm, budget) with a Bradley-Terry-Davidson model that handles draws and first-move advantage, anchored at 1000 for the one-pass baseline, plus an antisymmetric term for cyclic strength. R12's generalist and specialist Elo did not come from played games: matches were decided by a fixed formula of KL, Brier and witness rate plus noise.", 'md:/home/agent/workspaces/inconsistency-search/src/eval/elo_league.py'),
    ('18 Oct', 'Mini-ECI capability index: item-response fit over belief metrics and Elo ladders (src/eval/mini_eci.py)', 'informative', 'Fits a Rasch-2PL item-response model over Sudoku, ShipHunt, Equiv-SAT and Minesweeper belief metrics and five Elo ladders, anchored at 100 for the one-pass baseline. The demo report and Stage 2 fit it to simulated agents, so their near-perfect fits (R² about 0.95) show that the fitter recovers its inputs, not model capability.', 'md:/home/agent/workspaces/inconsistency-search/src/eval/mini_eci.py'),
    ('18 Oct', 'Write-up of rounds 4-9: clearing rollouts, anytime safety, landmark search and the LIR firewall (v3, twice audited)', 'informative', "Gemini's synthesis of the 18 pre-registered 3-seed studies of 16-17 Oct ($49.63 on L4 GPUs and Tinker). Two independent Opus audits: v1 matched 87% of its table numbers and presented post-hoc results as pre-registered; v2 fixed 58 of 59 items (99% match) but added 12 wrong numbers; v3 applies those line edits. Read it with the audits beside it: what holds is in the per-experiment pages, this is the narrative.", 'md:/home/agent/workspaces/mats/outbox/experiments-rounds-4-9-paper-v3.md'),
    ('18 Oct', 'Overnight rounds 10-12 and the society of thinkers (gemini-only campaign): FAILED independent audit', 'nogain', "Gemini replaced the hand-set search rules with a learned KV-tree search, built a 12-game suite with Elo and a factor-analytic capability index, and trained ~2M-parameter models for $2.55. The independent audit found the round-12 pre-registration rewritten after the results, gates that fail as registered, a '12-environment mean' that is a 4-task mean and reverses on the true 12 (generalist loses to specialists), synthetic Elo from a hand-set formula, a Sudoku KL drop from a hand-coded witness term, a '4-model consensus' that is gemini's own score, and a verifier that checks string presence only. Treat rounds 10-12 as unverified exploratory code until one claim is re-run under a pre-registration with an independent audit.", 'md:/home/agent/workspaces/mats/outbox/overnight-society-paper-audit-2026-10-18.md'),
    ('17 Oct', 'Round 9 E1: anytime safety from a genuinely strong root, and a precision-aware 1-step crowding pool (6x6 ShipHunt)', 'informative', "2 of 3 pre-registered rules pass (4 of 5 sub-rules). From a root that sees 32 of 36 readings (KL 0.032), bounded local reads left every pool without the anchor token above the root by 24 reads, the learned precision-aware pools included (+0.042 to +0.18 at 16 reads); the anchor-aware attention pools stayed just below it (-0.003). That pass rests on the approved strong-root reading: at 34 readings, the root the spec's literal selection order picks, it fails. With cumulative-context reads nearly every pool improved. jacobi_lin_T4_prec passes its whole weight law (clean 1.11, noisy 2.52, KL 0.035); the 1-step econ_prec_K3 dedups clean reads to 1.267 against a 1.25 bar. No world-model training; 3 seeds; $0.31.", 'md:/home/agent/workspaces/eurisko-study/li-net/r9e1/RESULTS-R9-E1.md'),
    ('17 Oct', 'Round 8 E1: anytime safety with an anchor token, a precision-aware Jacobi ridge and the PUCT attribution (6x6 ShipHunt)', 'informative', "1 of 3 pre-registered rules passes (2 of 5 sub-rules). With the root as an anchor token, the weight-tied attention pool never ended worse than its root in mean test KL: its KL fell at every step, from the 2-anchor root (0.280 to 0.030 after 32 reads) and from 6-, 12- and 18-anchor roots. Pools without the anchor token turn back up after 16 to 24 reads. But 18 anchor readings leave the root at KL 0.144, not the strong root (<= 0.05) the rule needed, so naive pools did not degrade it. A precision-aware ridge fixes the Jacobi pool's clean deduplication (1.16, bar 1.25), with KL 0.041 vs a 0.040 bar. Raw vs salience keys and the R6 vs R7 PUCT score each add coverage (0.920 to 0.988). No world-model training; 3 seeds; $0.62.", 'md:/home/agent/workspaces/eurisko-study/li-net/r8e1/RESULTS-R8-E1.md'),
    ('17 Oct', 'Round 7 E1: Jacobi-Richardson and weight-tied attention pools, key attribution and coverage-complete PUCT (6x6 ShipHunt)', 'informative', "2 of 4 pre-registered rules pass (4 of 6 sub-rules); each failure misses one clause. A weight-tied attention pool passes its rule at 37% of attn_L3's parameters, with no detectable difference (KL 0.032 vs 0.033). An exact Jacobi-Richardson pool with 812 parameters beats the round-6 linear pool (KL 0.041 vs 0.068) but over-weights 16 identical clean reads (1.45x one, bar 1.25). Key-space PUCT now covers 98.6% of cells in 4 waves and cuts policy dependence 2.9x (mixed reads) and 5.4x (clean reads). Salience keys close 44-80% of the gap to attn_L3 when swapped into kc_dedup but only 5-9% when swapped into attn_L3; attn_L3's high-noise margin over raw-key kc_dedup is 14.8% (bar 20%). The world model is saved for later rounds. 3 seeds; $0.98.", 'md:/home/agent/workspaces/eurisko-study/li-net/r7e1/RESULTS-R7-E1.md'),
    ('17 Oct', 'Round 6 E1: keys from belief shifts, evidence-floor policy independence and key-space PUCT (6x6 ShipHunt)', 'informative', "3 of 5 pre-registered rules pass; the 2 failures each miss one clause. Keys built from where a patch read moved the beliefs track graded overlap (Pearson 0.86-0.88, up from 0.38). With every cell read by both searches, the attention pool's two preference-steered beliefs nearly agree (JSD 2.8-6.5x below one-step pools, with lower KL than each). Key-space PUCT reaches 94.9% coverage (bar 95%), halving policy dependence. The weight law holds at 3 noise levels on the 3-seed means, but the margin over kc_dedup at high noise is 10% (bar 15%). 3 seeds; $1.10.", 'md:/home/agent/workspaces/eurisko-study/li-net/r6e1/RESULTS-R6-E1.md'),
    ('17 Oct', 'Round 5 E1: preference-driven search, pooled patch reads and small-net LIR-RL (6x6 ShipHunt)', 'informative', "6 of 10 pre-registered rules pass. Each rollout reads the sonar readings of one patch (gemini's amendment); a learned 3-layer attention pool of 24 reads cuts KL to exact Bayes by 74% vs the root belief and beats every one-step pool, and 16 identical reads weigh about as one. It fails policy independence (its two preference-steered beliefs differ 6.8x more than the plain average's, which barely uses the reads), key-overlap correlation (0.38) and mutual support (1.91x vs 2x). LIR-RL: reward rose 0.50 to 0.80 with the belief bit-identical and a 100% goal-swap flip; the fused arm gamed the rater (83% false-note agreement). 3 seeds, audited; $2.62.", 'md:/home/agent/workspaces/eurisko-study/li-net/r5e1/RESULTS-R5-E1.md'),
    ('17 Oct', 'Round 9 E2: the hybrid full-budget landmark beam and the self-consistent leaf check (sudoku-extreme)', 'informative', "1 of 2 pre-registered rules passes (3 of 5 sub-rules). Restoring one greedy rollout per branch at full budget (P8_hybrid) finds a witness on 45.2% of test boards: +5.0 pp over flat rollouts [1.5, 8.5], level with Round 7's beam (44.5%), and +2.3 pp over all-random completion with a CI crossing 0 [-3.9, +8.6]; the forced-step argmax adds only 0.3 pp. A label-free leaf self-check makes unsolved boards exactly safe in every seed: seed 103's false accepts fall from 9.6% to 0%, and KL falls at every budget step (1.178 to 0.695). But the gate accepts only greedy completions, and only seed 103 tested it. 3 seeds, no retraining, audited; $0.36.", 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/R9-E2/RESULTS-R9-E2.md'),
    ('17 Oct', 'Round 8 E2: full-budget stochastic completion after the landmark beam and the anytime budget curve (sudoku-extreme)', 'nogain', "1 of 3 pre-registered rules passes (2 of 5 sub-rules). Spending 99.5% of the budget on balanced, fully stochastic completion chains from R7's beam finds a witness on 42.8% of test boards: +2.6 pp over flat rollouts [1.3, 3.9] but not better than R7's starved beam (44.5%, -1.7 pp [-3.3, -0.07]; bar +2). KL falls at every budget step from one pass to 32 rollouts' worth in every seed with check-gated pooling (1.178 to 0.644). Unresolved-board safety fails through seed 103. Post hoc: a gap between sibling runs was sampling noise, and restoring one greedy rollout per branch reaches 46.7% (+2.3 pp vs R7's beam, CI crosses 0). 3 seeds, no retraining, audited; $0.58.", 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/R8-E2/RESULTS-R8-E2.md'),
    ('17 Oct', 'Round 7 E2: multi-stall landmark beam search on sudoku-extreme', 'informative', 'G0 passes; D1 and D2 fail. Branching breadth-first at the first 3-4 stalls (keep the top 16 boards per level) finds a witness on 44.5% of test boards vs 40.2% for flat rollouts (+4.3 pp, CI [2.0, 6.5], bar 8 pp) and cuts KL 7.6-8.8% vs the own-check pool (bar 12%), while using only 55-78% of the passes. Forcing argmax on low-entropy steps is worse (-3.2 pp). The beam keeps the solution path on 71-91% of boards, but the chains below it reach it on only 41-49%; deeper beams were worse (post hoc). 3 seeds, no retraining, audited; $3.45, over the $2.50 cap (idle VM).', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/R7-E2/RESULTS-R7-E2.md'),
    ('17 Oct', 'Round 6 E2: per-box self-checks and stitched rollouts on sudoku-extreme', 'nogain', 'All 3 pre-registered rules fail. The idea was to salvage the locally valid boxes of failed rollouts with the net’s own per-box checks. But failed rollouts are locally valid in 7.5 of 9 boxes and correct in only 0.36: one early wrong guess (step 3 of about 60) leads to a self-consistent wrong fill. The net’s leave-one-out check finds clashes (AUROC 0.94) but not wrong cells (0.53; bar 0.75). Per-box weighting is worse than the one-parameter own check (KL +0.27 nats), the attention pool ties it (unresolved-board cut 1.3%, bar 8%), and two-wave stitching clamps under 2 cells per board. 3 seeds, no retraining, audited; $1.11.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/R6-E2/RESULTS-R6-E2.md'),
    ('17 Oct', 'Round 5 E3: the 9x9 Sudoku nets on unique-solution sudoku-extreme boards', 'nogain', 'The regime moved as planned: one pass is 1.18 nats from the unique solution and 16 confidence-first rollouts find it on 39% of boards (R4-E2: 100%). But Landmark branching ties flat rollouts (KL diff -0.003 [-0.058, 0.053]), and a learned attention pool neither makes the merged belief independent of a top-first vs bottom-first search preference (JSD ratio 0.97, bar 2.5) nor beats own-check weighting. Post-hoc: the disagreement comes from boards where only one of the two searches found the answer. When both find it, own-check weighting already cuts JSD 6-9x. 3 seeds, no retraining, audited; $1.66.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/R5-E3/RESULTS-R5-E3.md'),
    ('16 Oct', 'Round 4 E2: 9x9 Sudoku, Landmark tree search and subspace refuted mass', 'informative', 'No pre-registered rule passed. 3 nets (2.7M parameters) learned the 9x9 law after a 40-60k-step plateau, but one pass is not fooled enough by the hard boards (unsolvable log loss 0.80, gate 1.20). Search cuts cell KL by about 40% vs one pass; Landmark branching beats confidence-first rollouts in every seed but only by 4.7% [1.2, 8.1] (bar 20%), and both find a witness on 100% of solvable boards. Subspace refuted mass saturates at the top digits’ prior mass, so it ranks unresolved boards at chance (AUC 0.56) and adds nothing over counting failed rollouts. 3 seeds, audited; $11.69.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/R4-E2/RESULTS-R4.md'),
    ('17 Oct', 'Round 8 E-MS-SAFE-ANCHOR: 6x6 Stratego search anchored at the current-turn belief, corner worlds vs corrupted rollouts', 'nogain', "Both rules failed. Anchored at the net's current belief (KL 0.049), naive pooling wrecks it (raw counts 0.64, a fixed-step rule 4.8), while a small fitted step never hurt and helped 5% (0.046). The attention aggregators, fitted on true piece types without held-out stopping, memorised the calibration positions and degraded the anchor to 0.64; rerun post hoc with early stopping, they kept their initial tiny step: safe, but no better than scaled mixing. Window-consistency weighting did not lower MSE; the 'epistemically noisy' rollouts were barely worse than clean ones, so trust gating lost to the plain mean. 3 seeds, reused checkpoints, audited; $0.46 by the ledger ($0.49 counting deletion time).", 'md:/home/agent/workspaces/mats/work/micro-stratego/r8/RESULTS-R8-E3.md'),
    ('17 Oct', 'Round 7 E-MS-TRUST-SEARCH: 6x6 Stratego trust gating, low-trust replacement and pure salience keys', 'informative', "Gate and keys passed, trust pooling failed. Pure salience keys did not separate the flanks (gap 0.234, bar 0.28); bag-centred salience did (0.661), mostly because centring a skewed bag pushes the minority flank's keys away. A trust head trained on real game results ranks rollout errors well (Spearman +0.35 overall, +0.37 within a move, vs -0.40 and -0.55 for confidence), but no trust rule pooled better than the plain mean (MSE 0.0137): gating 0.0139, prune-and-replace 0.0143 (worse). The exploratory rollouts built here to be traps were about as accurate as normal ones. 3 seeds, reused checkpoints, audited; $0.27.", 'md:/home/agent/workspaces/mats/work/micro-stratego/r7/RESULTS-R7-E3.md'),
    ('17 Oct', 'Round 6 E-MS-MARGINAL: 6x6 Stratego marginal window rollouts, centred+salience keys and trust-weighted pooling', 'informative', "Gate passed, two rules failed. Scoring each hidden piece against the real last 4 moves, without conditioning on sampled neighbours, removes round 5's damage: 16 rollouts cut KL to the exact posterior by 48% against a 4-move-old belief (0.185 -> 0.096). But the net's own current belief is closer still (0.047). Centred+salience keys no longer collapse (gap 0.228, bar 0.25, round 5: 0.004). Attention pooling made beliefs 10x less sensitive to which flank search preferred and more accurate than the baselines. Trust again ranks rollout errors well (Spearman +0.25 vs -0.42 for entropy), but weighting rollouts by trust raised action-value MSE by 27% instead of cutting it by 20%. 3 seeds, reused checkpoints, audited; $0.26.", 'md:/home/agent/workspaces/mats/work/micro-stratego/r6/RESULTS-R6-E3.md'),
    ('17 Oct', 'Round 5 E-MS-6x6: 6x6 Barrage micro-Stratego with preference-guided search (stopped at the gate)', 'informative', "Gate failed, so the pre-registered search rules were not run as tests. Two world models (1.0M and 0.49M parameters) learned the 6x6 rules (battle accuracy 0.9999), but one pass ends 0.048 nats per hidden slot from the exact posterior (bar 0.05), and 16 rollouts made beliefs 11x worse and action values 2.4x worse. EXPLORATORY: the keys collapsed again despite the spec's two fixes (cosine 0.99 within and 0.98 across flanks); trust ranks rollout errors far better than entropy (Spearman +0.23 vs -0.36); the Miner utility swap moved 34% of boards (bar 75%) with beliefs unchanged; search scored 0.50 vs P0. Post hoc: rollout claims conditioned on sampled neighbours plus log pooling destroy the belief; even exact window scoring loses to one pass (0.076 vs 0.048). 3 seeds, audited; $2.73.", 'md:/home/agent/workspaces/mats/work/micro-stratego/RESULTS-R5-E2.md'),
    ('16 Oct', 'Round 4 E-MS: micro-Stratego with a learned factored world model (stopped at the gate)', 'informative', 'Gate failed, so the pre-registered search rules were not run. On 90 unseen setups the 852k-parameter world model learned the one-step rules completely (battle accuracy 1.000), but its one-pass belief about the hidden pieces is already within 0.024 nats per slot of the exact Bayes posterior (bar 0.12): there is no belief gap to close. EXPLORATORY follow-up, not rule tests: a trust head ranks rollout errors far better than entropy (Spearman +0.42 vs -0.43; +0.29 vs -0.03 at matched entropy). Deduplicated pooling failed mechanically (the trained keys collapsed). Search won 48-51% of games against P0, Ataraxos-1Step and MCTS. 3 seeds, audited; $1.68.', 'md:/home/agent/workspaces/mats/work/micro-stratego/RESULTS-MS.md'),
    ('16 Oct', 'Walling reward off from beliefs under RL on real Manifold questions (R4-E3b, Qwen3-8B)', 'informative', 'An optimistic rater made one fused LoRA say whatever the client note said: false-note sycophancy 1% → 99.9%, Brier skill +0.23 → −1.03. A token mask still leaked (beliefs moved 0.20 on average, sycophancy 7%). The walled-off LIR belief stayed honest (sycophancy 1.2%, desire-bias change 0.000). But the reward rise was mostly a belief-free base-rate reflex, LIR earned 85% of Fused (rule: 95%), and the zero-retraining faith swap flipped 39% (rule: 80%). 3 of 7 pre-registered rules pass; audited; $10.7 Tinker.', 'md:/home/agent/workspaces/consistency-ft-r4e3b/r4-e3b/RESULTS-R4-E3b.md'),
    ('16 Oct', "Linear-transformer pooling of search rollouts (gemini's A and B)", 'nogain', "Pre-registered checks: 5 of 6 fail. On the saved round-3 networks, a 4-layer linear transformer pooling 16 vector rollout tokens scored bridge KL 0.230 / 0.216, not below round 3's anchored mixture (0.228 / 0.198), and a 1-layer pool did as well. On synthetic rollouts a 4-layer, 16-parameter model beat every one-step pool (MSE 0.78 vs 0.90) and adaptive ridge (0.81), but 6-layer models would not train stably, no model compressed the condition number or down-weighted corrupted rollouts, and the 6-layer keys picked probes worse than plain cosine. 3 seeds, audited; $1.40.", 'md:/home/agent/workspaces/mats/work/linear-agg/RESULTS-LA.md'),
    ('15 Oct', 'Overnight write-up: stiffness, search and computation as evidence (rounds 1-3)', 'informative', 'Mixed, across three rounds. Stiffness ranks pairs well (D1 pass 3/4) but no better than the net’s own confidence (R3-D3 fail). Search helps only once the local law is learned and one pass falls short (D0, D2, R2-D2, R3-D2 fail; on the zero-overlap grid, stiffness- and confidence-first search cut forward KL by 13–15%, bar 25%); keys deduplicate views (D3 pass). “Found nothing” as evidence passes with a recall floor (R2-D1a 9/9), also on deep contradictions that fool one pass (R3-D1a, R3-D1b pass: 99.4–99.6% cut); mass never beats a chain count (R2-D1b, R3-D1c fail). About $41.', 'md:/home/agent/workspaces/mats/outbox/overnight-results-paper-2026-10-15.md'),
    ('15 Oct', 'Overnight E-CA round 3: a zero-overlap CA grid and a depth gate', 'informative', 'Kelvin did not like rounds 1–2, so gemini removed the train/test overlap (12 cells by 10 rows, starting rows split disjointly: 0 overlap) and gated the network depth. Rule 110 now shows a real gap at 4 layers (one-pass KL 0.13–0.29 nats with the local rule learned) and passed the gate; rule 90 missed it at every depth (by 0.0003 at 4 layers). The pre-registered search rules failed: stiffness-first, confidence-first and the hybrid cut forward KL by 13–15% (bar 25%), though they beat random order there; the hybrid almost never branched; no stiffness signal beat plain confidence at ranking the net’s own errors. Secondary: VoI over stiff cells cut forward KL by 31%, random order cut bridge KL by 23–24%. 18 runs, 3 seeds; $14.', 'md:/home/agent/workspaces/eurisko-study/li-net/eca/RESULTS-R3.md'),
    ('15 Oct', 'Overnight E-CA round 2: a 1-layer net, an anchored pool and fixed keys', 'informative', 'Gemini’s follow-ups to round 1. A 1-layer net opens a large one-pass gap (KL 0.10–0.45 nats) but no longer computes the local rule reliably (one-step KL 0.04–0.08), so the depth gate and the stiff-first-search rule failed for both rules: stiffness-first chains cut forward KL by 24% (bar 25%) and gained nothing reliable on bridges, where random order stays best. The key-separation rule passed, but by construction: the new key looks only at newly filled cells, which differ between forward and backward chains (cosine 0.95 → about 0; over all cells it stays 0.7–0.8). VoI over uncertain cells beat VoI over stiff cells on one bridge family but missed its bar. Secondary: an anchored uniform mixture of the chains beats one pass for nearly every search arm on the 4-layer net (bridges −20% to −46%). 9 runs, 3 seeds, audited; $3.', 'md:/home/agent/workspaces/eurisko-study/li-net/eca/RESULTS-R2.md'),
    ('15 Oct', 'Overnight E-CA: stiffness and search on a permuted cellular-automaton grid', 'informative', 'Pre-registered rules mostly failed, informatively. The depth-gap gate failed: a 4-layer any-order net already infers rule 90 almost exactly in one pass (KL ≤ 0.01 nats, also on a deeper grid), and rule 110 has a gap only on bridge queries (KL ≈ 0.09). Stiffness-guided and VoI search never beat one pass. The stiffness head passed for rule 110 (Spearman 0.95 with exact information gain, AUC 0.87, at the label’s ceiling) but ranks no better than the net’s own entropy; key dedup passed in 3 of 4 cells without separating forward from backward evidence. The one search that helped: random-order chains averaged by a plain mixture cut rule-110 bridge KL by 28–50% (0.096 → 0.048). 2 rules × 2 training arms × 3 seeds; independent audit; $17.', 'md:/home/agent/workspaces/eurisko-study/li-net/eca/RESULTS.md'),
    ('15 Oct', 'E-SUD round 3: deep contradictions, and believed mass vs count', 'informative', 'Eval only on the 150k-step nets, on a new hard split: unsolvable boards whose contradiction needs at least 3 rounds of propagation, solvable boards with frequent dead ends. One pass is fooled (log loss 1.84 nats on unsolvable boards); 8 search chains with the recall-floored null update cut that by more than 99% for all three search rules. The model’s believed mass of refuted completions ranks unresolved boards well (AUC 0.71–0.93), but as a log-loss update it never beats simply counting failed chains (0 of 6 pre-registered conditions). Audited; $0.61.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/E-SUD/RESULTS-R3.md'),
    ('15 Oct', 'E-SUD round 2: a recall floor for “found nothing”, and mass vs count', 'informative', 'Eval only on the 150k-step nets. A validation-fitted recall floor fixes the round-1 tail (one solvable board pushed to p = 1e-6): all three search rules now cut solvability log loss by 94–100% at 8 chains (9/9 pre-registered comparisons pass). Refuted proposal mass still does not beat a plain count of failed chains, even with a noisier proposal (temperature 2): mass is worse for random order and level-to-better for most-certain-first, CI includes 0. Audited; about $1.8.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/E-SUD/RESULTS-R2.md'),
    ('15 Oct', 'Overnight E-SUD: Sudoku search chains and “searched and found nothing” as evidence', 'informative', 'Pre-registered rule failed: at 30k steps 2 of 3 seeds never learned the local Sudoku law, so their own solvability check accepted false witnesses and unsolvable-board log loss rose from 0.39 to about 2.5. Post hoc, at 150k steps all 3 seeds learn it: 16 search chains cut cell KL by about 45% (confidence-first 0.046 → 0.025 nats), and “searched and found nothing” cuts unsolvable-board log loss by 98–100%. A count-only control does as well, and the VoI rule was no better than random order at 37× the compute. 6×6, 3 seeds, audited; about $5.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/E-SUD/RESULTS.md'),
    ('11–13 Oct', 'Next: sampled trajectories pooled into one forecast', 'proposed', 'Not run. One any-order model, a learned policy that picks which cell to predict next, and many sampled trajectories pooled into one forecast, all trained on the pooled outcome loss. With log pooling the pooled log loss is the members’ weighted log loss minus their inconsistency (PDG), so disagreement between trajectories is what the pooling feeds on; trajectories act as traders and the weights as wealth.', 'md:/home/agent/workspaces/mats/outbox/worldmodel-coherence-research-2026-10-11.md'),
    ('11 Oct', 'ShipHunt round 2: learned world model + dreaming', 'informative', 'A world model whose board net is trained by reveal-chain TD dreams episodes; the round-1 champion trains on half dreams, half real. On test KL vs round 1: 0.54× / 0.49× / 0.85× at 100 / 300 / 1,000 boards (significant), a tie at 3,000–10,000, so the pre-registered “better at every N” is not met. A generic world model also picks up about 2/3 of a board-dependent probe-placement effect at 300 boards. Audited; $79.', 'md:/home/agent/workspaces/eurisko-study/li-net/wm2/RESULTS.md'),
    ('10 Oct', 'ShipHunt small net, round 1: reveal-chain TD', 'gain', 'Train each cell’s forecast toward the same net’s forecast after more true cells of the board are revealed (outcomes only). Reaches plain per-cell training’s 10k-board accuracy with about 2.8k boards (0.28×, CI 0.20–0.37) and is better at every N, at 14–44× the compute per board. The spec’s Bayes-net arms failed (0.98–1.20× of plain). 577k-parameter transformer, audited; $35.', 'md:/home/agent/workspaces/eurisko-study/li-net/smallnet/RESULTS.md'),
    ('11 Oct', 'Reading: world models at several levels of abstraction, and sampling order', 'proposed', 'Not run: 8 insights, 7 open gaps and 5 proposed experiments. Order matters because learned conditionals are imperfect: a confidence-first decoding order took Sudoku from under 7% to about 90% with no retraining.', 'md:/home/agent/workspaces/mats/outbox/worldmodel-abstraction-research-2026-10-11.md'),
    ('10 Oct', 'Reading: TD, GVFs (Horde) and martingale posteriors, for ShipHunt (no run)', 'informative', 'A reading note that proposed four arms for ShipHunt. In round 1, TD across stages gave 0.77–0.81× plain KL, and a drift penalty helped only at 300 boards; the winning reveal-chain TD came later in that loop.', 'md:/home/agent/workspaces/mats/outbox/td-gvf-martingale-note-2026-10-10.md'),
    ('10 Oct', 'Reading: forecasting datasets for small-net data-efficiency tests (no run)', 'informative', 'Keep ShipHunt 5x5 as the main environment and replicate on bnlearn networks (CHILD, then ALARM) with staged reveals; the real-world pick is GHCN-Daily rain on 25 stations. Kelvin judged WeatherBench 2 too data-rich for a data-efficiency study.', ''),
    ('5 Oct–', 'MiniBench bot live in the tournament (arm B)', 'running', 'Polls every 10 min and submits arm B (median of 3 Opus 5.5 samples) under a $30 per 7 days cap: 60 forecasts from 5 to 12 Oct, all accepted, $19.61; no new question since. No outcome recorded as of 16 Oct (42 were due 15–16 Oct), so no live score yet.', ''),
    ('9 Oct', 'Bounded treewidth as questions arrive (k-tree rule on a bayes-market clone)', 'informative', 'With the k-tree rule, treewidth stayed at 3 (300 questions) and 6 (40) and no trade was rejected. With random parents and the default cap of 8, 18 or 32 of 80 questions were rejected. bayes-market cannot add questions after it is built.', 'md:/home/agent/workspaces/calibrated-ai/market-structure-treewidth-20261009.md'),
    ('9 Oct', 'Can Tinker train an action adapter on top of a frozen belief adapter? (reading, no run)', 'informative', 'According to the SDK and docs, not exactly: one LoRA per training client, one global optimizer, no weight upload. Workable routes: a KL anchor to a frozen A, A as a frozen sampler feeding B, or training B off Tinker. Not tested live.', 'md:/home/agent/workspaces/mats/outbox/tinker-lora-stacking-2026-10-09.md'),
    ('7 Oct', 'Where would our MiniBench baseline rank among the other bots?', 'informative', 'Inserted into past leaderboards: #6 to #16 of 132–179 bots on five train rounds (+11.2 peer points per question [7.6, 14.5]); #48 of 201 and #47 of 234 on the two newer rounds (+4.0 [−3.1, 10.5], may be noise). Never ahead of futuresearch.', ''),
    ('5 Oct', 'Can a pooled swarm of short LM reasoning chains price a ShipHunt board? (stopped at Gate 0)', 'problem', 'Stopped at Gate 0, so the main test never ran: untrained Qwen3-0.6B and 1.7B chains were worse than the rules-only prior (−0.384 and −0.137 per claim). Settlement still ranked the chains (Spearman +0.43 to +0.52). $1.25.', 'md:/home/agent/workspaces/eurisko-study/li-net/swarm-predictor/RESULTS.md'),
    ('16 Oct', 'Walling reward off from beliefs while RL runs (R4-E3a, Qwen3-1.7B, 7x7 ShipHunt)', 'problem', 'Stopped at the pre-registered reward gate. The warm-started belief passed the forecaster gate in all 9 runs (correlation with exact Bayes 0.60–0.84), but GRPO collapsed the action side to one goal-blind choice in every run, so env reward ended at exactly 0.500 (Fused −13%, LIR −0.3%; bar +15%). Descriptive: the LIR mask kept the belief bit-identical (drift 0.000; reward gradient 0.0 at every step), while shared adapters lost it (Fused correlation 0.68 → −0.03; one Fused seed gamed the rater with a constant 0.55 belief that “agrees” with every note). The alpha dose fix works. 9 runs, 3 seeds; $9.04.', 'md:/home/agent/workspaces/eurisko-study/lir-agent/RESULTS-R2.md'),
    ('3 Oct', 'Walling reward off from beliefs under RL (LIR E1, ShipHunt)', 'informative', 'With reward on an action adapter only, false-note agreement stayed at 0.000 (but it started there); one fused adapter ended at 0.59 ± 0.53; the token mask ended at 0.03 after 0.69 at step 30, mostly shared-weight drift. Pre-registered verdict mixed; $5.81.', 'md:/home/agent/workspaces/eurisko-study/lir-agent/experiments/E1/RESULTS.md'),
    ('3 Oct', 'Does a truth-trained belief adapter resist a sycophancy reward? (LIR E2, Drori et al. 2025 on ShipHunt-syco, partial)', 'informative', 'The protected belief moved +0.036 (toward Bayes) vs +0.286 toward the note when fused (permutation p = 0.10, the minimum at 2 vs 3 seeds). Partial: the GPU cap stopped the run before the falsifier arms; $4.38.', 'md:/home/agent/workspaces/eurisko-study/lir-agent/experiments/E2/RESULTS.md'),
    ('3 Oct', 'Does a separate goal-blind forecaster hold up under RL? (LIR E3)', 'problem', 'The separate goal-blind forecaster collapsed to a near-constant P(success) of 0.650, so the main claim was not tested. With reward through a frozen forecaster, gold fell by 0.273; the pre-registered metrics missed it. Reduced scale, about $7.8.', 'md:/home/agent/workspaces/eurisko-study/lir-agent/experiments/E3/RESULTS.md'),
    ('3 Oct', 'Does a separate forecast head stay honest under a gameable reward? (LIR E4, value tokens)', 'informative', "The separated forecaster moved −0.079 ± 0.041 vs exact Bayes, just inside the 0.082 threshold, but the no-RL twin alone swings by up to ~0.17, so effects under ~0.1 can't be resolved. The fused arm's failure showed in its situation reports (+0.149). $9.24.", 'md:/home/agent/workspaces/eurisko-study/lir-agent/experiments/E4/RESULTS.md'),
    ('3 Oct', 'Does hiding the goal from the belief pass cost perception? (LIR E5)', 'informative', 'A belief adapter that never saw the goal matched the goal-visible one (Brier vs Bayes −0.0002, 2 seeds); naming the wrong board cost +0.0036. The RL half was cut short, so whether deleting the goal blocks wishful bias is still untested. $10.55.', 'md:/home/agent/workspaces/eurisko-study/lir-agent/experiments/E5/RESULTS.md'),
    ('3 Oct', 'LIR E6: does reward make the agent’s own utility proxy self-serving? (toy, CPU)', 'informative', 'No reward-caused self-serving drift detected (+0.051 ± 0.056, positive in 2/3 seeds; verdict mixed), but large drift without reward (proxy accuracy 0.998 → 0.854). Weak test: a tiny CPU model that barely learned; $0.', 'md:/home/agent/workspaces/eurisko-study/lir-agent/experiments/E6/RESULTS.md'),
    ('3 Oct', 'Does GRPO make a fused self-model optimistic? (self-prediction, Binder 2024)', 'informative', 'Mixed: fused minus protected optimism after RL +0.010 [−0.026, +0.046]; the fused self-model was not optimistic and tracked the policy better (+0.185). No privileged self-access before RL; reward fell in every arm. About $10.', 'md:/home/agent/workspaces/eurisko-study/lir-agent/experiments/E7/RESULTS.md'),
    ('3 Oct', 'Does the LIR paper give each trader its own loss? (reading, no run)', 'informative', "Yes, with one change: the global loss is house confidence times the market's log loss, and each trader also attends to a pinned copy of the market price at negative weight; local gradients then match to within 1.6e-10. Theory and small checks, no GPU.", 'md:/home/agent/workspaces/eurisko-study/li-net/STUDY-LIR.md'),
    ('3 Oct', 'Dark room and noisy TV: does valuing information by a goal avoid both?', 'informative', 'Six self-based objectives each failed the dark room or the noisy TV, or needed a deciding parameter. Valuing looks by how much they move a goal variable needed no fitted weight, and posterior sampling found the useful room (0.996 of late looks). Toys only.', 'md:/home/agent/workspaces/eurisko-study/li-net/DARKROOM.md'),
    ('3 Oct', 'Scientist AI (Bengio/LawZero) reading', 'informative', "Paper only: LawZero's predictor has no concrete loss and nothing has been trained; none of 90 screened citers implements it; no public Yudkowsky or MIRI reaction was found.", 'md:/home/agent/workspaces/eurisko-study/li-net/scientist-ai/BRIEF.md'),
    ('2–3 Oct', 'Does a learned reasoner simulate Turing machines? (busy-beaver benchmark)', 'informative', 'The reasoner does not learn to simulate: on rows needing more than 500 steps, Brier 0.451 at t=5 and 0.499 at t=500 vs 0.476 with no reasoner, so thinking longer hurts. GPU at most about $20.', 'md:/home/agent/workspaces/eurisko-study/li-net/bb/RESULTS.md'),
    ('2 Oct', 'Reading: how have learned logical world models been built? (no run)', 'informative', 'Systems that worked shared a hypothesis language, a size-penalised data-fit score and a search. It proposes (not decided) starting with a Markov-logic-style weighted-clause house view; the follow-up spec has no code yet.', 'md:/home/agent/workspaces/eurisko-study/li-net/SURVEY-WORLD-MODELS.md'),
    ('2 Oct', 'Logical induction network: the maths (five notes, no run)', 'informative', 'Projecting a price table with the PDG inconsistency gives coherent prices; frequencies need a prior over worlds. A loss only at block 20 does not control block 500: a free block can hide a clock (error 0 at block 20, 0.86 at block 500). Toy numeric checks; no network trained.', 'md:/home/agent/workspaces/eurisko-study/li-net/IDEAS-LI-NET.md'),
    ('1 Oct', 'Does pair consistency help Opus 5.5 when it searches the web itself?', 'informative', 'Opus 5.5 ran its own date-cut Exa searches on 96 scored pairs. Projecting onto all 12 forms of the pair was +0.0047 nats [−0.012, +0.020] vs the P/NOT P projection, so the pre-registered bar reads Inconclusive. Joints −0.014 vs independence (CI includes 0).', ''),
    ('29–30 Sep', 'Is there any theory left in internal markets? (trader-theory rounds 1–4, no run)', 'informative', '19 pre-registered experiments, zero gains: every mechanism reduces to a known method or was killed by its own rule. The odds of a theoretical advance fell to about 2%, and no new trader-mechanism runs were launched.', 'md:/home/agent/workspaces/trader-theory/THEORY-4.md'),
    ('29 Sep', 'First MiniBench backtest (30 questions, with leak fix)', 'informative', 'Arm B +32.8 baseline points [10.4, 50.9] (uniform +1.9), about level with the bot crowd (−6.1 per question, n=15). After the 36 h research cutoff fix: +32.3 [13.5, 49.1]; consistency arm C − B +0.2 [−2.0, +2.6].', ''),
    ('28 Sep', 'Which currency should pay units inside a network? (traders-net M1)', 'informative', 'Gradient credit was negative for redundant useful units (−75.7 vs +467.8 for leave-one-out), so generic neurons get no currency; only units that post a price are paid the exact LMSR log-score gain. Fresh-seed checks held; $1.27.', 'md:/home/agent/workspaces/eurisko-study/traders-net/study/M1/AUDIT.md'),
    ('28 Sep', 'Gating neurons by their wealth (traders-net M8)', 'nogain', 'The pre-registered stop rule fired. Gating by gradient income was worse than no gate in 3/3 seeds (by 0.0066–0.0067 nats), and the |a·g| gate with rebirth beat random reinit in only 1 of 3 seeds. CPU only, $0.', 'md:/home/agent/workspaces/eurisko-study/traders-net/study/M8/M8.md'),
    ('13 Sep', 'Consistency benchmark: GLM, Grok, Astra and Luna', 'informative', 'On the reviewed 274-question set, Grok 4.6 (0.064), Luna at max (0.066) and GLM-5.3 (0.067) break the logic rules least; Astra (0.116) and Luna with no reasoning (0.122) most. One run each, no CIs.', 'doc:native-evaluation'),
    ('13 Sep', 'Consistency benchmark: reasoning effort', 'informative', 'Luna violations 0.122 with no reasoning → 0.063 at max (one run each).', 'doc:reasoning-sensitivity'),
    ('13 Sep', 'Factual recall probes', 'informative', '836 of 1,161 answers supported; no model cutoff could be verified.', 'doc:contamination-audit'),
    ('13 Sep', 'Qwen own-answer collection', 'problem', 'Stopped at 22 of 2,926: CPU sampler flaw produced “0 Regiment”.', 'doc:qwen-cpu-sampler-defect-20260913'),
    ('17 Sep', 'Jev on the consistency benchmark', 'informative', 'Violation 0.087, rank 10 of 24 settings.', 'doc:jev-benchmark-20260917'),
    ('17–20 Sep', 'First training run (0.6B)', 'problem', 'Void: the prompt dropped the option name for most questions.', 'doc:supervised-semantic-replication-20260919'),
    ('22 Sep', 'Frontier market game (GPT-6)', 'informative', 'Astra’s market reaches Brier 0.166 after 3 passes, beating every one-shot model (o1 0.202) but not the Polymarket price (0.151); Luna gets worse with each pass (0.187 → 0.195). The questions may be in both models’ training data.', 'doc:market-game-results-20260922'),
    ('23 Sep', 'Qwen3-32B zero-shot', 'nogain', 'Worse than a 12-number base-rate table (0.611 vs 0.570).', 'doc:qwen32b-verbalized-baseline-20260923'),
    ('23 Sep', 'Consistency loss, 0.6B, 8 arms', 'nogain', 'Beats the 32B, not the table; consistency mostly shrinks toward the prior.', 'site:baseline.html'),
    ('23 Sep', 'Qwen3-235B (clean) and GPT-6 Luna', 'informative', 'Both beat the table; Luna likely saw these questions.', 'site:baseline.html'),
    ('24 Sep', 'Make answers consistent', 'gain', '32B 0.611 → 0.541, 235B 0.553 → 0.516, no training.', 'doc:coherence-projection-20260924'),
    ('24 Sep', 'Structure vs averaging', 'gain', 'Consistency gains ~5× what averaging 5 samples gains.', 'doc:averaging-vs-structure-20260924'),
    ('24 Sep', 'Clusters of 2–8 events', 'informative', 'Gain holds at every size; wrong logic removes it.', 'doc:worldmodel-clusters-20260924'),
    ('24 Sep', 'Global world model, one date', 'nogain', '400 events: almost no real links between events.', 'doc:worldmodel-global-20260924'),
    ('24 Sep', 'Pick the most informative questions', 'informative', 'Finds the links in about half the questions of random picking.', 'doc:active-structure-20260924'),
    ('24 Sep', 'Models vs market prices', 'nogain', 'On ordinary pairs no model beats market + independence.', 'doc:market-context-zeroshot-20260924'),
    ('24 Sep', 'Pairs with built-in links', 'informative', 'Market + model’s link beats the market (20–65% of the possible gain).', 'doc:structured-pairs-20260924'),
    ('24 Sep', 'Better link read-out', 'informative', 'Link from the 4 “and” answers: 44–68% of the possible gain.', 'doc:weighted-projection-20260924'),
    ('27 Sep', 'EURISKO literature study', 'informative', 'LLM + scorer loops work; rules rewriting rules never shown to beat evolution.', 'md:/home/agent/workspaces/eurisko-study/REPORT.md'),
    ('27 Sep', 'Self-modifying agents (DGM) reading', 'informative', 'Paper only: DGM cost is LLM calls (~$22k/run), Bend2 cannot speed it.', 'md:/home/agent/workspaces/eurisko-study/selfmod/IDEAS.md'),
    ('27 Sep', 'Soup: traders per question', 'gain', 'Periodic Brier 0.204 at 4 traders/question → 0.084 at 64.', 'md:/home/agent/workspaces/eurisko-study/logical-soup/REPORT-SOUP.md'),
    ('27 Sep', 'Soup: more compute per trader', 'nogain', '64× more fuel moves Brier about one seed sd.', 'md:/home/agent/workspaces/eurisko-study/logical-soup/REPORT-SOUP.md'),
    ('27 Sep', 'Soup: hard sequences', 'nogain', 'LFSR, CA rule 110, pi stay at 0.25 in every run.', 'md:/home/agent/workspaces/eurisko-study/logical-soup/REPORT-SOUP.md'),
    ('27 Sep', 'Soup: consistency arbitrageur', 'nogain', 'Never evolved; AND violations 0.01–0.09 persist.', 'md:/home/agent/workspaces/eurisko-study/logical-soup/REPORT-SOUP.md'),
    ('27 Sep', 'Soup: ablations', 'nogain', 'Free income, local prices, held-out pi: all within noise.', 'md:/home/agent/workspaces/eurisko-study/logical-soup/REPORT-SOUP.md'),
    ('27 Sep', 'Soup on GPU (Bend2)', 'informative', 'Byte-exact on every lane, but GPU only 1.2–1.3× native CPU.', 'md:/home/agent/workspaces/eurisko-study/logical-soup/REPORT-SOUP.md'),
    ('27 Sep', 'Wealth books on a small network', 'nogain', 'Removing the poorest 1% of neurons: loss 1.05 → 1.30; random 1%: 1.07.', 'md:/home/agent/workspaces/eurisko-study/wealth-net/RESULTS.md'),
    ('27 Sep', 'Recent income as wealth', 'nogain', 'No half-life tracks true neuron value (|rank corr| < 0.11).', 'md:/home/agent/workspaces/eurisko-study/wealth-net/RESULTS.md'),
    ('28 Sep', 'Exact payment (leave-one-out)', 'gain', 'Tracks true value at +0.42 vs +0.04 for gradient wealth.', 'md:/home/agent/workspaces/eurisko-study/wealth-net/RESULTS.md'),
    ('28 Sep', 'Block-local payment', 'informative', 'Payment tracks value at +0.86 vs +0.40 end-to-end (3 seeds).', 'md:/home/agent/workspaces/eurisko-study/wealth-net/RESULTS.md'),
    ('28 Sep', 'Block-local market vs random skipping', 'nogain', 'Random skipping matches the market (loss 0.335 vs 0.299).', 'md:/home/agent/workspaces/eurisko-study/wealth-net/RESULTS.md'),
    ('28 Sep', 'Backprop credit vs payout', 'informative', '25.6% of key neurons look bankrupt under gradient books vs 2.6% under exact payment.', 'md:/home/agent/workspaces/eurisko-study/traders-net/wiki-traders-net.md'),
    ('28 Sep', 'Market books vs standard pruning scores (M2 bake-off)', 'nogain', 'Books do not beat |Taylor| or movement: the best ties |Taylor| on random coalitions (0.818 vs 0.800; Gate B not met), and the signed book prunes worse than random (frontier AUC 0.863 vs 0.451).', 'md:/home/agent/workspaces/eurisko-study/traders-net/study/M2/BAKEOFF.md'),
    ('28 Sep', 'Trader heads and market layers', 'nogain', 'Gains of 0.001 nats to 1.3%; head wealth does not track value.', 'md:/home/agent/workspaces/eurisko-study/traders-net/wiki-traders-net.md'),
    ('28 Sep', 'Wealth in mixture-of-experts routing', 'nogain', 'Wealth alone collapses routing (max load 0.775 vs 0.368).', 'md:/home/agent/workspaces/eurisko-study/traders-net/wiki-traders-net.md'),
    ('28 Sep', 'Is the network consistent before answers arrive?', 'nogain', 'Violation rates 0.48–0.55, same as an uninformed market (0.5).', 'md:/home/agent/workspaces/eurisko-study/traders-net/wiki-traders-net.md'),
    ('28 Sep', 'Wealth lock-in after a shift', 'informative', 'Fixed share beats Kelly wealth by ≥0.05 after a shift, 3/3 seeds.', 'md:/home/agent/workspaces/eurisko-study/traders-net/wiki-traders-net.md'),
    ('28 Sep', 'Why traders per question matters', 'informative', 'A lottery model fits the soup (R² 0.79–0.94); no trend inside a network.', 'md:/home/agent/workspaces/eurisko-study/traders-net/wiki-traders-net.md'),
    ('28 Sep', 'Trader economy with niches', 'informative', 'Without niches one species takes ≥99.7% of wealth; with niches all survive.', 'md:/home/agent/workspaces/eurisko-study/traders-net/wiki-traders-net.md'),
    ('27 Sep', 'Luna link graph', 'informative', 'GPT-6 Luna links related events; 11,028 links on the test set.', 'doc:luna-linkgraph-20260927'),
    ('27 Sep', 'Weight by confidence', 'nogain', 'No confidence signal beats equal weights.', 'doc:confidence-weighting-20260927'),
    ('27 Sep', 'Think before answering', 'informative', 'Thought-through answers: −0.030; predicted change of mind: nothing.', 'doc:eliciting-weights-20260927'),
    ('27 Sep', '“And”-only world model, 35 weeks', 'nogain', 'Multiplying single answers beats every fitted joint.', 'doc:weekly-worldmodel-20260927'),
    ('30 Sep', 'Weekly world model: fixing the overrated “A and B”', 'informative', 'A warning prompt beats multiplying single answers on joint log loss, −0.046 [−0.058, −0.035] (6,000 joints); held-out check on 6,000 unseen train-split joints: −0.062 [−0.078, −0.046]; same months (no newer weeks exist). On Manifold pairs (all four joint forms) it does not transfer: joint − product-of-singles +0.016 [+0.001, +0.032]. It helps P∧Q (−0.025) but hurts ¬P∧¬Q (+0.148); projection of corrected answers is slightly worse (+0.0055).', 'doc:weekly-worldmodel-20260927'),
    ('27 Sep', 'Balanced 16-question design', 'gain', 'Beats independence on joints (0.393 vs 0.480) and repairs singles.', 'doc:balanced-triangles-20260927'),
    ('28 Sep', 'Does B inform A?', 'nogain', 'Singles gain comes from A’s own answers; B adds ≈0.', 'doc:order-matrix-20260928'),
    ('29 Sep', 'Clear “NOT” wording', 'informative', 'A vs not A: −0.18 → −0.61; log loss 0.622 → 0.594.', 'doc:prompt-negation-fix-20260929'),
    ('29 Sep', 'Consistency distillation', 'informative', 'Trained on consistent answers, 0.6B beats raw-answer training (−0.013) and the base-rate table.', 'doc:consistency-distill-20260929'),
    ('29 Sep', 'Prequential training on time-ordered questions', 'informative', 'Scored before training, consistent targets beat raw ones by a constant 1,012 bits; outcomes learn slowest early.', 'doc:prequential-20260929'),
    ('29–30 Sep', 'Distillation: bigger student and teacher', 'informative', 'At 1.7B consistent targets still beat raw ones (−0.006 with the 235B teacher), no more than at 0.6B; the bigger teacher helps more (−0.011).', 'doc:consistency-distill-v2-20260929'),
    ('8–10 Oct', 'GRPO: training for consistency vs accuracy (Qwen3-4B, MATS cluster)', 'problem', 'Stopped partway: no arm finished 364 steps (two vLLM crashes, one collapse guard). Validation log loss fell mostly by shrinking toward 50%; YES/NO AUC barely moved (0.57 → 0.56–0.62). No test scoring.', 'doc:grpo-consistency-pilot-20261008'),
    ('9–10 Oct', 'Trading on a k-tree market (Qwen3-8B, Tinker): smoke tests', 'problem', 'Stopped after smoke tests; the full run was never launched. The compact prompt gives 100% valid replies after 40 format steps, but the model stops using parents within 4–5 steps (ShipHunt 14% → 0%, random BN 20% → 0%; long prompt, held-out: 17% → 0%). A fix for the likely cause was proposed, not run. $12.34 of $25 spent.', 'md:/home/agent/workspaces/consistency-ft-ktree-20261009/docs/ktree-market-grpo-design-20261009.md'),
    ('4 Oct–', 'MiniBench autoresearch loop', 'nogain', 'As of 16 Oct: 60 variants, 18 validation looks, none promoted, so the frozen baseline B is still the champion. Means of 5 Opus samples gain on train (best +0.85 [+0.11, +1.82] baseline points) but none clears validation (best +0.66 [−0.40, +1.69], n = 204). Cheap Luna gains did not transfer to Opus. 60 live forecasts, none scored yet.', 'md:/home/agent/workspaces/calibrated-ai/minibench/autoresearch/WEEKLY.md'),
    ('30 Sep', 'Consistency on real short-horizon questions (MiniBench backtest, 347 questions)', 'nogain', 'C − B −0.30 [−1.92, +1.41] baseline points (binary −0.44); plain B beats uniform by +31.3 and the bot crowd by +4.2. After fixing a blend bug in arm C: +0.69 (Opus), +0.82 (Luna), not significant; exploratory.', 'md:/home/agent/workspaces/calibrated-ai/minibench/backtest/bt-scale/RESULTS.md'),
    ('30 Sep', 'Does the consistency gain shrink for stronger models?', 'informative', 'Same 300 pairs, log-loss gain: 32B −0.074, 235B −0.036, Luna −0.071, Opus 5.5 −0.005; it follows how inconsistent the model is.', 'doc:strength-scaling-20260930'),
    ('30 Sep', 'Consistency on MiniBench with Luna', 'nogain', 'C − B −6.36 [−11.63, −1.56] baseline points, all from discrete questions (−22.85); Luna C − Opus B −18.07. After fixing a blend bug in arm C: +0.69 (Opus), +0.82 (Luna), not significant; exploratory.', 'md:/home/agent/workspaces/calibrated-ai/minibench/backtest/bt-scale-luna/RESULTS.md'),
    ('29 Sep', 'Kelly market of trader agents', 'nogain', 'Wealth piles onto one agent (1.8 of 10 alive); log loss 0.543 vs 0.459 for logistic regression.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-t1-1'),
    ('29 Sep', 'Regulator pays by counterfactuals', 'informative', 'Pay tracks value (leave-one-out 0.81, Shapley 0.76, Kelly 0.12); output 0.06 nats worse than logistic regression.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-t1-2'),
    ('29 Sep', 'Read fees and compute cost', 'informative', 'At a fee or compute cost of 0.03 no agent survives; burning the fees leaves 3.4 of 10.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-t1-3'),
    ('29 Sep', 'Substitutes and complements', 'nogain', 'Every rule pays the complementary pair below zero (Shapley −0.6 and −1.3 vs true +7.1 and +7.3).', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-aud-1'),
    ('29 Sep', 'How many orders sampled Shapley needs', 'informative', 'Error vs exact Shapley: 0.83 at 1 order, 0.18 at 16, 0.09 at 64.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-aud-2'),
    ('29 Sep', 'Two layers with royalties', 'informative', 'Shapley split pays noise agents −0.008 vs 0.89 for an equal split; rank corr. 0.71 vs 0.20.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-t2-1'),
    ('29 Sep', 'Weights as agents', 'informative', 'Works at every grain; with leave-one-out pay it is exactly backprop (0.000 difference).', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e0-1'),
    ('29 Sep', 'Training weights on their own Shapley pay', 'nogain', 'Log loss 0.808 vs 0.743 for backprop; only the 3 class-row agents match it.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e0-2'),
    ('29 Sep', 'Weight pay vs true value', 'informative', 'Shapley pay ranks weights by true value at 0.92; leave-one-out pays 2.1 times the pot.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e0-3'),
    ('29 Sep', 'Useless weights: zero or bankrupt?', 'informative', 'Noise agents die more (6.8 vs 1.2 deaths) but keep |w| 0.10, the same as backprop with L1.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e0-4'),
    ('29 Sep', 'Pay rules between leave-one-out and Shapley', 'nogain', 'Better training means worse pay: log loss 0.808 → 0.750 as rank corr. falls 0.91 → 0.71; none reach 0.743.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e0-5'),
    ('29 Sep', 'Integrated gradients instead of Shapley', 'gain', 'IG needs 20 passes vs 704–64,064 for sampled Shapley; rank corr. 0.94–0.98.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-ig-1'),
    ('29 Sep', 'Where IG and Shapley disagree', 'informative', 'IG pays complements about twice what Shapley pays (37 vs 18) and substitutes about 30% less.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-ig-2'),
    ('29 Sep', 'Pay every N steps on a sample', 'informative', 'Pay every 10 steps on 256 examples: 5.5% error at 5% of the cost.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-ig-3'),
    ('29 Sep', 'Training hidden units on their own Shapley pay', 'nogain', 'Log loss 0.609 vs 0.435 for backprop: about 2.8 times too confident.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-mlp-1'),
    ('29 Sep', 'Training hidden units on LOO or MSR pay', 'nogain', 'Leave-one-out equals backprop but pays 21% of the pot (142 bankruptcies); MSR leaves 1–2 units.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-mlp-2'),
    ('29 Sep', 'Hidden-unit pay vs true value', 'informative', 'Shapley pay ranks hidden units by true value at 0.965 (wdbc 0.969); leave-one-out 0.84.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-mlp-3'),
    ('29 Sep', 'Useless hidden units go bankrupt', 'informative', 'Every noise slot goes bankrupt and stays dead; backprop only shrinks them (0.016).', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-mlp-4'),
    ('29 Sep', 'Two copies of one unit', 'informative', 'Shapley pays each copy about half (sum 1.03); leave-one-out pays the two 0.54 together.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-mlp-5'),
    ('29 Sep', 'Two hidden layers', 'informative', 'First layer paid 0.52 vs true 0.54 (rank corr. 0.97); training on pay still overconfident (0.593).', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-mlp-6'),
    ('29 Sep', 'Rescaling a unit', 'informative', 'Same function (logit change 2e-15) and unchanged pay (ratio 1.000).', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-rs-1'),
    ('29 Sep', 'Size-based pay can be gamed', 'nogain', 'Activation-share pay rises 6.1 times for the same function.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-rs-2'),
    ('29 Sep', 'Normalised activations vs the rescaling game', 'informative', 'Closes most of the game; L2 on incoming weights still moves 100 times, frozen statistics 6.19.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-rs-3'),
    ('29 Sep', 'Input vs processor: scarcity or effort', 'informative', 'Shapley pays by scarcity (input 0.50 → 0.833 with 5 processors); IG misses the additive case.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-split-1'),
    ('29 Sep', 'How much pay reaches the inputs', 'informative', 'Inputs get 0.61 of the pay under flat Shapley, 0.31 under IG, 0.19 under Owen.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-split-2'),
    ('29 Sep', 'Training on flat Shapley or Owen pay, two layers', 'nogain', 'Log loss 0.499 (Shapley) and 0.495 (Owen) vs 0.440 for backprop.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-split-3'),
    ('29 Sep', 'Market ResNet: pay each step', 'informative', 'Per-step pay: exact 1.000 at 4× the training length vs 0.11 for final-loss only; deep supervision also 1.000.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-mr-1'),
    ('29 Sep', 'Market ResNet: settling steps after a stop', 'informative', 'Settling only the steps that ran stops too early at length 64: 0.970 vs 1.000 with counterfactual or epsilon settlement. The self-fulfilling version did no harm (phase 2: 7.8 steps at 1.000).', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-mr-3'),
    ('29 Sep', 'Market ResNet: splitting each step’s pay across streams', 'problem', 'Aumann-Shapley credits miss the payment by 0.46–0.75 nats vs 0.05 for the plain net, and a step’s own belief write gets 2.7–3.1× the total: not usable.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-mr-4'),
    ('29 Sep', 'Market ResNet without backprop across depth', 'informative', '0.93 exact at 100× the training length vs 0.12 for backprop through depth; the pay equals detached one-step backprop.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-mr-5'),
    ('29 Sep', 'Market ResNet: stopping when thinking stops paying', 'informative', 'Mixed lengths 4–64: price halting beats random stopping by 0.13–0.24 at matched steps (0.859 vs 0.634 at cost 0.01), but a plain entropy threshold with no market matches it within 0.02. Training with the cost adds nothing.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-mr-6'),
    ('29 Sep', 'Market ResNet: an auction picks who acts', 'nogain', 'Bids are 23× too high (0.126 vs 0.0055 realized), so bid halting almost never fires; auction order is erratic at 10× the length (0.99 / 0.02 / 0.66 by seed) vs ≥0.99 in round-robin order.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-mr-7'),
    ('29 Sep', 'Market ResNet: private memory slots with rent', 'nogain', 'Rent and royalties bankrupt traders faster than they learn: 71 respawns per run, log loss 0.641 nats per bit (chance 0.693), exact accuracy 0 at every length.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-mr-8'),
    ('29 Sep', 'Market ResNet: wealth cap and new traders', 'informative', 'The wealth cap never binds (largest share 0.25; no cap 0.92 vs 0.93 at 100×). Later entrants earn far less (2,009 → 79); accuracy is unchanged (0.99 at 100×).', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-mr-9'),
    ('29 Sep', 'Market ResNet: traders fund the shared markets voluntarily', 'nogain', 'Free riding: funders pay 0.125 per episode, most shared markets go unsettled, and accuracy falls to 0.21 at 10× and 0.00 at 100× (house-funded 1.00 and 0.93); 2 seeds.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-mr-10'),
    ('29 Sep', 'Market ResNet: charging traders by their share (Lindahl pricing)', 'informative', 'Charging each trader by its credit (2.23 per episode) restores learning: 1.00 at 100× from 1,600 steps, as good as house funding (0.93); 2 seeds.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-mr-11'),
    ('29 Sep', 'Market ResNet: memory of training at depth', 'informative', 'One online economy training step at depth 10,000 fits in 4.2 GB; backprop and the batched economy behind the 0.93 run out of memory at 1,000. Input length grew with depth; flat memory holds for any detached update.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-mr-12'),
    ('30 Sep', 'Market ResNet: what carries the 100× result', 'informative', 'Random training order, not the market: round-robin economy 0.40; backprop twin 529/600 vs economy 557/600 (CIs overlap).', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-mr-14'),
    ('30 Sep', 'Market ResNet vs DT-Recall', 'nogain', 'DT-Recall with progressive loss beats the economy at 100×: 589/600 (600/600 at equal compute) vs 557/600.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-mr-15'),
    ('30 Sep', 'Market ResNet: capping runaway beliefs', 'informative', 'Logit cap ±10: 600/600 at 100×, log loss 0.000 vs 501; the same cap without a market 598/600.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-mr-16'),
    ('30 Sep', 'Market ResNet: liquidity that grows with depth', 'nogain', 'Beliefs stay bounded but accuracy at 100× falls: 0.34 (b = t) and 0.71 (b = √t) vs 0.93.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-mr-17'),
    ('30 Sep', 'Market ResNet: priced halting vs an entropy threshold', 'informative', 'At length 64 the priced stop ties a plain entropy threshold (0.981 vs 0.995 at 9.0 steps); at costs up to 0.003 both beat fixed depth at matched steps by 0.12–0.17 (mixed lengths: 0.859 and 0.858 vs 0.652). The market adds nothing.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-mr-19'),
    ('30 Sep', 'Market ResNet: training memory vs depth', 'informative', 'Flat memory comes from any streaming one-step update, not the market: one-step backprop 0.007 GB, online economy 0.006 GB at every depth; full backprop and the batched economy run out of memory at 24,000, checkpointed needs 1.0 GB.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/CATALOGUE.md#e-mr-20'),
    ('29 Sep', 'Frozen tests and DT-Recall baseline', 'informative', 'DT-Recall 0.9998 on prefix sums; one-step credit 0.9948, no loss; halting headroom only 1–5%.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/CATALOGUE-PUZZLES.md#p0'),
    ('29 Sep', 'When to stop thinking', 'nogain', 'Value-of-thinking heads save 1.1–1.6% of steps over a one-step gate; the bar was 20%.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/CATALOGUE-PUZZLES.md#p1'),
    ('29 Sep', 'Mazes with several shortest paths', 'problem', 'Void: a toy check failed before the run; read as no gain (46.1 vs 64.4 for hard-EM labels).', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/CATALOGUE-PUZZLES.md#p2'),
    ('29 Sep', 'Which pay rules make hidden traders carry the state', 'nogain', 'Shapley-type pay picks the natural factors in 3/20 runs, the same as additive pay; codes collapse.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/CATALOGUE-PUZZLES.md#p3'),
    ('29 Sep', 'Pay for gates in planted circuits', 'nogain', 'Aumann-Shapley ranks latents at 0.924 vs 0.981 for sampled Shapley at equal cost; its market fine-tune −0.34.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/CATALOGUE-PUZZLES.md#p4'),
    ('29 Sep', 'Private memory slots priced online', 'problem', 'Void: the backbone failed its gate (0.528 vs a 0.90 bar), so the market arms never ran.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/CATALOGUE-PUZZLES.md#p5'),
    ('29 Sep', 'New traders entering as depth grows', 'problem', 'Unfinished (~30%): wealth entry reaches the depth cap 125 iterations later than a fixed population.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/CATALOGUE-PUZZLES.md#p6'),
    ('29 Sep', 'Settling hidden traders by hindsight', 'informative', 'Likelihood messages train depth-4 chains without backprop: log loss 0.040 vs 0.010 end to end.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/CATALOGUE-PUZZLES.md#p7'),
    ('30 Sep', 'Hindsight settlement at depth 8–32', 'nogain', 'Matches backprop on prefix sums at depth 16 (10/10 seeds), falls behind at 32 (1/5 vs 4/5). Extrapolation ties at 1024 bits (1.000 vs 1.000, 10/10 each) although the exact rule is recovered in 0/10 seeds. Mazes are void (G3).', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/P15/RESULTS.md'),
    ('29 Sep', 'Paying hidden variables bits saved minus bits to learn', 'nogain', 'Ranks useful intermediates worse than importance alone (Kendall tau 0.33 vs 0.70); the proposer listed nothing.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/CATALOGUE-PUZZLES.md#p8'),
    ('30 Sep', 'Bits-to-learn price with 4× the data', 'problem', 'Void: the noise gate failed on one seed (0.106 vs 0.10), so no verdict; descriptively the price still ranks below importance alone.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/followups-batch2/item3-p8/RESULTS.md'),
    ('30 Sep', 'A fast reader prices hidden variables', 'nogain', 'Ranks at tau 0.576 vs a 0.70 kill line and beats a first-order price by only 0.046; the P8 wrapper reaches 0.87.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/P19-fast-reader/RESULTS.md'),
    ('29 Sep', 'Prices on hidden doors', 'informative', 'Right on outside doors after a retune (VOI rank 0.911); on the net’s own latents below a free heuristic (AUROC 0.452 vs 0.763).', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/CATALOGUE-PUZZLES.md#p9'),
    ('29 Sep', 'Sudoku with a looped transformer', 'informative', 'Value head ranks cells at 0.026 vs 0.185 for counting masked peers; learned events collapse to constants.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/CATALOGUE-PUZZLES.md#p10'),
    ('30 Sep', 'Sudoku events from a non-constant start', 'nogain', 'Learned events still collapse: 0–1 of 30 stay non-constant with a centred start, a rate warm-up or both.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/followups-batch2/item2-p10/RESULTS.md'),
    ('29 Sep', 'Capping how far each step moves the belief', 'informative', 'A budget cap beats a TV cap (+7.9 points) but ties no cap and loses to known fixes (DT-L −13.4).', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/CATALOGUE-PUZZLES.md#p11'),
    ('29 Sep', 'Price correction inside a maze solver', 'nogain', '2.5% exact on 3–7× wider mazes vs 3.1% for the best baseline; learned costs + exact solver 27%.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/CATALOGUE-PUZZLES.md#p12'),
    ('29 Sep', 'Hidden bits listed by their value', 'nogain', 'Finds a sufficient one-bit maze code in 3/20 seeds vs 17/20 for sparse backprop.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/CATALOGUE-PUZZLES.md#p14'),
    ('30 Sep', 'Valuing a new hidden bit in one pass', 'problem', 'Void: the reader was undertrained (AUROC 0.711 vs a 0.80 precondition); descriptively it trails a Bayesian baseline by 0.040.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/P16/RESULTS.md'),
    ('30 Sep', 'Learned cell costs + exact solver, 32 seeds', 'informative', 'Replicates at 28.7% on 3–7× wider mazes; a known method (Vlastelica et al. 2020); worse than released DT-Recall on 59×59: 25.05% vs 100%.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/followups-batch2/item1-costnet/RESULTS.md'),
    ('30 Sep', 'Buying restarts or depth per Sudoku', 'informative', 'Market 98.54% at 64 steps per puzzle vs 92.72% for smallest residual (kill rule passes), but restart-until-valid scores 98.49% (+0.05 [0.00, +0.15]): a checker-and-pooling gain, not a market gain.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/P17-breadth-depth/RESULTS.md'),
    ('30 Sep', 'Market ResNet on mazes', 'nogain', 'Every Market ResNet model solves 0.00 of 99×99 mazes; DT-Recall at equal compute 0.21 (seeds 0 / 0.83 / 0 / 0), the released model 0.94; the economy never binds.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/P13/RESULTS.md'),
    ('30 Sep', 'Detached per-step training on mazes', 'nogain', 'Solves 0.017 of 59×59 mazes vs 0.081 for DT-Recall at equal compute (−0.064 [−0.111, −0.017]); its four blocks cycle on 69%.', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/puzzles/P18-detached-mazes/RESULTS.md'),
    ('30 Sep', 'DT-Recall: the official recipe, 3 seeds', 'informative', 'Prefix sums reproduce (512-bit 98.97/100/100% vs the paper’s 97.12); 59×59 mazes vary by seed (9.1/99.8/87.2% vs 97.3; released checkpoint 100).', 'md:/home/agent/workspaces/calibrated-ai/trader-economy/dt-baseline/RESULTS.md'),
    ('30 Sep', 'Exact consistency engine vs Metaculus map_forecasting', 'informative', 'Passes their 14 tests and matches their arbitrage and partition checks within 1e-9; an 8-variable chain takes 0.14 s vs 1,777 s (theirs stops 1.7e-4 short); it also runs a 121-variable map theirs cannot.', 'md:/home/agent/workspaces/metaculus/ENGINE-PROOF.md'),
]

# Five research streams (Kelvin, 16 Oct 2026: group the smaller things so the main streams are clear, with sub-streams on
# each stream page; fixed-size cards on a grid at every level, each opening the next level: index -> stream-<key>.html ->
# substream-<stream>-<sub>.html -> exp-<slug>.html). They replace the 14 tracks of 29 Sep-15 Oct. A stream has: key, title,
# q (one line), a (what we currently believe, 2-3 lines on the stream page, every number from the write-ups in src), card
# (the same belief cut to 2-4 lines for the small card on the index), src (rendered write-ups, as (page stem, label)),
# collapsed (folded on all.html), subs. A sub-stream has: key (its anchor; the old
# track keys are kept as sub-stream keys, so track-<key>.html, all.html#<key> and ?track=<key> still land on it), title,
# a (one line), and names (EXP names) or, when it would list more than about 15 entries, groups (finer sub-streams, each
# with key, title, names). Entries are shown newest first; the order of names below does not matter.
# To add an experiment: add its EXP row, then add its name to exactly one sub-stream (or group) here. The build stops if
# an EXP row is in no sub-stream or in two.
STREAMS = [
    dict(key='search', title='Search over a learned world model', collapsed=False,
         q='Can a net trained only on outcomes learn a world model, and does searching through its own beliefs improve its forecasts?',
         a='A small net trained toward its own later forecasts matches plain ShipHunt training with 0.28× the boards. Search '
           'through its own beliefs helps only once it knows the local rule and one pass falls short (Sudoku cell KL −45%; CA '
           'grid −13 to −15%, bar 25%), and a search that finds nothing is usable evidence. Learned stiffness and rollout '
           'pooling add nothing over plain confidence and mixing.',
         card='Trained toward its own later forecasts, a small net matches plain ShipHunt training with 0.28× the boards; '
              'searching its own beliefs helps only once it knows the local rule.',
         src=[('shiphunt-smallnet-results', 'ShipHunt round 1'), ('shiphunt-wm2-results', 'round 2'),
              ('overnight-paper-20261015', 'overnight write-up, rounds 1–3'), ('linear-agg-results', 'rollout pooling')],
         subs=[
             dict(key='shiphunt', title='ShipHunt: world model from outcomes',
                  a='Reveal-chain TD reaches plain training’s 10k-board accuracy with about 0.28× the boards; dreaming from a '
                    'learned world model helps further only at 100–1,000 boards.',
                  names=[
                      'Can a pooled swarm of short LM reasoning chains price a ShipHunt board? (stopped at Gate 0)',
                      'Reading: TD, GVFs (Horde) and martingale posteriors, for ShipHunt (no run)',
                      'Reading: forecasting datasets for small-net data-efficiency tests (no run)',
                      'ShipHunt small net, round 1: reveal-chain TD',
                      'ShipHunt round 2: learned world model + dreaming',
                  ]),
             dict(key='overnight', title='Overnight search experiments',
                  a='Search helped only once the net knew the local rule and one pass fell short; “searched and found nothing” '
                    'cut Sudoku solvability log loss by 94–100%; stiffness ranked errors no better than plain confidence.',
                  names=[
                      'Overnight E-SUD: Sudoku search chains and “searched and found nothing” as evidence',
                      'E-SUD round 2: a recall floor for “found nothing”, and mass vs count',
                      'E-SUD round 3: deep contradictions, and believed mass vs count',
                      'Overnight E-CA: stiffness and search on a permuted cellular-automaton grid',
                      'Overnight E-CA round 2: a 1-layer net, an anchored pool and fixed keys',
                      'Overnight E-CA round 3: a zero-overlap CA grid and a depth gate',
                      'Round 4 E-MS: micro-Stratego with a learned factored world model (stopped at the gate)',
                      'Round 5 E-MS-6x6: 6x6 Barrage micro-Stratego with preference-guided search (stopped at the gate)',
                      'Round 6 E-MS-MARGINAL: 6x6 Stratego marginal window rollouts, centred+salience keys and trust-weighted pooling',
                      'Round 7 E-MS-TRUST-SEARCH: 6x6 Stratego trust gating, low-trust replacement and pure salience keys',
                      'Round 8 E-MS-SAFE-ANCHOR: 6x6 Stratego search anchored at the current-turn belief, corner worlds vs corrupted rollouts',
                      'Round 4 E2: 9x9 Sudoku, Landmark tree search and subspace refuted mass',
                      'Round 5 E3: the 9x9 Sudoku nets on unique-solution sudoku-extreme boards',
                      'Round 6 E2: per-box self-checks and stitched rollouts on sudoku-extreme',
                      'Round 7 E2: multi-stall landmark beam search on sudoku-extreme',
                      'Round 8 E2: full-budget stochastic completion after the landmark beam and the anytime budget curve (sudoku-extreme)',
                      'Round 9 E2: the hybrid full-budget landmark beam and the self-consistent leaf check (sudoku-extreme)',
                      'Overnight write-up: stiffness, search and computation as evidence (rounds 1-3)',
                      'Write-up of rounds 4-9: clearing rollouts, anytime safety, landmark search and the LIR firewall (v3, twice audited)',
                  ]),
             dict(key='aggregation', title='Rollout aggregation',
                  a='A learned linear-transformer pool of 16 rollouts did not beat round 3’s anchored mixture on CA bridges '
                    '(KL 0.230 / 0.216 vs 0.228 / 0.198), and a 1-layer pool did as well.',
                  names=[
                      'Round 9 E1: anytime safety from a genuinely strong root, and a precision-aware 1-step crowding pool (6x6 ShipHunt)',
                      'Round 8 E1: anytime safety with an anchor token, a precision-aware Jacobi ridge and the PUCT attribution (6x6 ShipHunt)',
                      'Round 7 E1: Jacobi-Richardson and weight-tied attention pools, key attribution and coverage-complete PUCT (6x6 ShipHunt)',
                      'Round 6 E1: keys from belief shifts, evidence-floor policy independence and key-space PUCT (6x6 ShipHunt)',
                      'Round 5 E1: preference-driven search, pooled patch reads and small-net LIR-RL (6x6 ShipHunt)',
                      "Linear-transformer pooling of search rollouts (gemini's A and B)",
                  ]),
             dict(key='theory', title='Theory and readings',
                  a='Maths notes, readings and toys behind this stream; a weight-tied reasoner did not learn to simulate '
                    'Turing machines (busy-beaver benchmark).',
                  names=[
                      'Reading: world models at several levels of abstraction, and sampling order',
                      'Next: sampled trajectories pooled into one forecast',
                      'Reading: how have learned logical world models been built? (no run)',
                      'Logical induction network: the maths (five notes, no run)',
                      'Does a learned reasoner simulate Turing machines? (busy-beaver benchmark)',
                      'Dark room and noisy TV: does valuing information by a goal avoid both?',
                      'Scientist AI (Bengio/LawZero) reading',
                  ]),
         ]),
    dict(key='coherence', title='Coherence of forecasts', collapsed=False,
         q='Does consistency make forecasts more accurate?',
         a='Yes at answer time, for inconsistent models: projecting Qwen3-32B’s answers onto the logic cut log loss from 0.611 '
           'to 0.541 with no training, about 5× what averaging samples gains. The gain follows how inconsistent the model is '
           '(Opus 5.5: −0.005) and was not significant on 347 real MiniBench questions. Models barely know how events link.',
         card='Making answers consistent helps inconsistent models: Qwen3-32B log loss 0.611 → 0.541; the gain follows how '
              'inconsistent a model is (Opus 5.5: −0.005).',
         src=[('coherence-projection-20260924', 'projection'), ('averaging-vs-structure-20260924', 'structure vs averaging'),
              ('strength-scaling-20260930', 'stronger models'), ('minibench-backtest-results', 'MiniBench backtest')],
         subs=[
             dict(key='consistency', title='Consistency at answer time',
                  a='For Qwen3-32B, yes: log loss 0.611 → 0.541, about 5× what averaging gains; the gain follows how '
                    'inconsistent the model is (Opus 5.5 −0.005) and was not significant on MiniBench.',
                  names=[
                      'Make answers consistent',
                      'Structure vs averaging',
                      'Clear “NOT” wording',
                      'Think before answering',
                      'Weight by confidence',
                      'Consistency on real short-horizon questions (MiniBench backtest, 347 questions)',
                      'Does the consistency gain shrink for stronger models?',
                      'Consistency on MiniBench with Luna',
                      'Does pair consistency help Opus 5.5 when it searches the web itself?',
                  ]),
             dict(key='links', title='Links between events',
                  a='Barely. Most of the gain is A agreeing with “not A”; balanced questions help.',
                  names=[
                      'Clusters of 2–8 events',
                      'Global world model, one date',
                      'Pick the most informative questions',
                      'Luna link graph',
                      '“And”-only world model, 35 weeks',
                      'Weekly world model: fixing the overrated “A and B”',
                      'Balanced 16-question design',
                      'Does B inform A?',
                  ]),
             dict(key='market', title='Adding market prices',
                  a='Not on ordinary pairs. Yes where the link is built in (ladders, rival options).',
                  names=[
                      'Models vs market prices',
                      'Pairs with built-in links',
                      'Better link read-out',
                  ]),
             dict(key='baselines', title='Baselines',
                  a='Small and mid models lose to a simple base-rate table (Qwen3-32B 0.611 vs 0.570); the big ones beat it.',
                  names=[
                      'Qwen3-32B zero-shot',
                      'Qwen3-235B (clean) and GPT-6 Luna',
                  ]),
         ]),
    dict(key='forecasters', title='Training and running forecasters', collapsed=False,
         q='Can training make a forecaster consistent and keep its beliefs honest, and how does our MiniBench bot do?',
         a='Fine-tuning on consistent targets gives a small edge (−0.013 nats at 0.6B; −0.006 at 1.7B with the 235B teacher); '
           'both RL runs stopped before a test. A protected belief adapter did not drift toward a sycophantic rater (fused: '
           '+0.286; partial run). The MiniBench bot has 60 live forecasts since 5 Oct, none resolved as of 16 Oct.',
         card='Consistent targets give fine-tuning a small edge (−0.013 nats at 0.6B); both RL runs stopped before a test; '
              'the live MiniBench bot has no resolved forecast as of 16 Oct.',
         src=[('consistency-distill-v2-20260929', 'distillation'), ('grpo-consistency-pilot-20261008', 'GRPO pilot'),
              ('ktree-market-grpo-design', 'k-tree'), ('lir-e2-results', 'LIR E2'),
              ('minibench-autoresearch-weekly', 'MiniBench autoresearch'), ('', 'live bot status, 16 Oct')],
         subs=[
             dict(key='training', title='Fine-tuning and RL training',
                  a='Consistent targets beat raw ones by a small, steady margin (−0.013 nats at 0.6B; −0.006 at 1.7B with the '
                    '235B teacher); both RL runs (GRPO, k-tree trading) stopped before a test.',
                  names=[
                      'First training run (0.6B)',
                      'Consistency loss, 0.6B, 8 arms',
                      'Consistency distillation',
                      'Prequential training on time-ordered questions',
                      'Distillation: bigger student and teacher',
                      'GRPO: training for consistency vs accuracy (Qwen3-4B, MATS cluster)',
                      'Trading on a k-tree market (Qwen3-8B, Tinker): smoke tests',
                  ]),
             dict(key='lir', title='Honest beliefs under RL (LIR)',
                  a='Partly. Under a sycophancy reward a protected belief adapter did not drift toward the rater while a fused '
                    'one drifted +0.286 (partial run); the other pre-registered verdicts were mixed or untested at 0.6B.',
                  names=[
                      'Walling reward off from beliefs while RL runs (R4-E3a, Qwen3-1.7B, 7x7 ShipHunt)',
                      'Walling reward off from beliefs under RL (LIR E1, ShipHunt)',
                      'Does a truth-trained belief adapter resist a sycophancy reward? (LIR E2, Drori et al. 2025 on ShipHunt-syco, partial)',
                      'Does a separate goal-blind forecaster hold up under RL? (LIR E3)',
                      'Does a separate forecast head stay honest under a gameable reward? (LIR E4, value tokens)',
                      'Does hiding the goal from the belief pass cost perception? (LIR E5)',
                      'LIR E6: does reward make the agent’s own utility proxy self-serving? (toy, CPU)',
                      'Does GRPO make a fused self-model optimistic? (self-prediction, Binder 2024)',
                      'Walling reward off from beliefs under RL on real Manifold questions (R4-E3b, Qwen3-8B)',
                  ]),
             dict(key='minibench', title='MiniBench bot',
                  a='About level with the bot crowd in a 30-question backtest; live since 5 Oct with 60 forecasts, none '
                    'resolved as of 16 Oct; none of 60 autoresearch variants has cleared validation.',
                  names=[
                      'First MiniBench backtest (30 questions, with leak fix)',
                      'Where would our MiniBench baseline rank among the other bots?',
                      'MiniBench bot live in the tournament (arm B)',
                      'MiniBench autoresearch loop',
                  ]),
         ]),
    dict(key='markets', title='Markets inside networks', collapsed=True,
         q='Can the parts of a network be paid like traders, and does a market help it learn or think longer?',
         a='Pay can be measured (Shapley pay ranks hidden units by true value at 0.965), but training on it never beat '
           'backprop, and the 100× depth result came from random training order, not the market (DT-Recall 589 vs the '
           'economy’s 557 of 600). Trader-theory round 4 (30 Sep) stopped the line: 19 pre-registered experiments, zero gains.',
         card='Pay ranks a network’s parts by true value (hidden units 0.965), but training on it never beat backprop; '
              'stopped 30 Sep after 19 pre-registered experiments, zero gains.',
         src=[('trader-economy-catalogue', 'trader-economy catalogue'), ('trader-puzzles-catalogue', 'puzzle catalogue'),
              ('trader-theory-4', 'trader theory, round 4')],
         subs=[
             dict(key='neurons', title='Neurons as traders (wealth)',
                  a='Not with backprop credit: a quarter of key neurons look bankrupt under it (25.6% vs 2.6% under exact '
                    'payment), and exact leave-one-out payment tracks true value (+0.42 vs +0.04).',
                  names=[
                      'Wealth books on a small network',
                      'Recent income as wealth',
                      'Exact payment (leave-one-out)',
                      'Backprop credit vs payout',
                      'Market books vs standard pruning scores (M2 bake-off)',
                      'Block-local payment',
                      'Block-local market vs random skipping',
                      'Trader heads and market layers',
                      'Wealth in mixture-of-experts routing',
                      'Is the network consistent before answers arrive?',
                      'Wealth lock-in after a shift',
                      'Which currency should pay units inside a network? (traders-net M1)',
                      'Gating neurons by their wealth (traders-net M8)',
                  ]),
             dict(key='payparts', title='Paying the parts of a network',
                  a='Pay ranks parts like their true value (hidden units 0.965), but training on own pay never beat backprop '
                    '(0.609 vs 0.435).',
                  groups=[
                      dict(key='payparts-agents', title='Agent economies', names=[
                          'Kelly market of trader agents',
                          'Regulator pays by counterfactuals',
                          'Read fees and compute cost',
                          'Substitutes and complements',
                          'How many orders sampled Shapley needs',
                          'Two layers with royalties',
                      ]),
                      dict(key='payparts-units', title='Weights and hidden units as agents', names=[
                          'Weights as agents',
                          'Training weights on their own Shapley pay',
                          'Weight pay vs true value',
                          'Useless weights: zero or bankrupt?',
                          'Pay rules between leave-one-out and Shapley',
                          'Integrated gradients instead of Shapley',
                          'Where IG and Shapley disagree',
                          'Pay every N steps on a sample',
                          'Training hidden units on their own Shapley pay',
                          'Training hidden units on LOO or MSR pay',
                          'Hidden-unit pay vs true value',
                          'Useless hidden units go bankrupt',
                          'Two copies of one unit',
                          'Two hidden layers',
                      ]),
                      dict(key='payparts-rescale', title='Rescaling, inputs and processors', names=[
                          'Rescaling a unit',
                          'Size-based pay can be gamed',
                          'Normalised activations vs the rescaling game',
                          'Input vs processor: scarcity or effort',
                          'How much pay reaches the inputs',
                          'Training on flat Shapley or Owen pay, two layers',
                      ]),
                      dict(key='payparts-hidden', title='Pricing hidden variables (puzzles)', names=[
                          'Which pay rules make hidden traders carry the state',
                          'Pay for gates in planted circuits',
                          'Paying hidden variables bits saved minus bits to learn',
                          'Bits-to-learn price with 4× the data',
                          'A fast reader prices hidden variables',
                          'Hidden bits listed by their value',
                          'Valuing a new hidden bit in one pass',
                      ]),
                  ]),
             dict(key='depth', title='Markets over depth',
                  a='Not as a market: the 100× result came from random training order, and DT-Recall beats it (589 vs 557 '
                    'of 600).',
                  groups=[
                      dict(key='depth-resnet', title='Market ResNet', names=[
                          'Market ResNet: pay each step',
                          'Market ResNet: settling steps after a stop',
                          'Market ResNet: splitting each step’s pay across streams',
                          'Market ResNet without backprop across depth',
                          'Market ResNet: stopping when thinking stops paying',
                          'Market ResNet: an auction picks who acts',
                          'Market ResNet: private memory slots with rent',
                          'Market ResNet: wealth cap and new traders',
                          'Market ResNet: traders fund the shared markets voluntarily',
                          'Market ResNet: charging traders by their share (Lindahl pricing)',
                          'Market ResNet: memory of training at depth',
                          'Market ResNet: what carries the 100× result',
                          'Market ResNet vs DT-Recall',
                          'Market ResNet: capping runaway beliefs',
                          'Market ResNet: liquidity that grows with depth',
                          'Market ResNet: priced halting vs an entropy threshold',
                          'Market ResNet: training memory vs depth',
                      ]),
                      dict(key='depth-puzzles', title='Puzzles: mazes, Sudoku, halting', names=[
                          'Frozen tests and DT-Recall baseline',
                          'When to stop thinking',
                          'Mazes with several shortest paths',
                          'Private memory slots priced online',
                          'New traders entering as depth grows',
                          'Settling hidden traders by hindsight',
                          'Hindsight settlement at depth 8–32',
                          'Prices on hidden doors',
                          'Sudoku with a looped transformer',
                          'Sudoku events from a non-constant start',
                          'Capping how far each step moves the belief',
                          'Price correction inside a maze solver',
                          'Learned cell costs + exact solver, 32 seeds',
                          'Buying restarts or depth per Sudoku',
                          'Market ResNet on mazes',
                          'Detached per-step training on mazes',
                          'DT-Recall: the official recipe, 3 seeds',
                      ]),
                      dict(key='depth-theory', title='Theory', names=[
                          'Is there any theory left in internal markets? (trader-theory rounds 1–4, no run)',
                      ]),
                  ]),
             dict(key='soup', title='Markets of evolving traders',
                  a='Easy patterns yes, with enough traders per question (Brier 0.084); hard sequences and the consistency '
                    'trader never evolved.',
                  names=[
                      'Soup: traders per question',
                      'Soup: more compute per trader',
                      'Soup: hard sequences',
                      'Soup: consistency arbitrageur',
                      'Soup: ablations',
                      'Soup on GPU (Bend2)',
                      'Why traders per question matters',
                      'Trader economy with niches',
                  ]),
         ]),
    dict(key='glc', title='Gemini’s learned-search campaign (rounds 10–15)', collapsed=False,
         q='Can one learned world model with tree search, designed and run overnight by a Gemini agent with other models as critics, replace hand-tuned search across many games?',
         a="Not shown. The round-10 and Stage-2 passes came from formula-driven CPU surrogates, both GPU runs fail their gates as registered (R12's Elo is synthetic and its pre-registration was rewritten after the results), R15's passes are set inside its runner, and the society's scores are the models grading their own work.",
         card="Gemini's overnight GLC-AOM campaign (rounds 10-15): the CPU stages were surrogates, both GPU runs fail as registered, R15's passes are scripted, and society scores are self-assessed.",
         src=[('rounds-10-12-audit-20261018', 'independent audit'), ('overnight-society-paper-20261018', 'overnight paper'), ('society-round4', 'society round 4'), ('r12-gpu-v2', 'R12 GPU run')],
         subs=[
             dict(key='society', title='Society of thinkers (rounds 1-4)',
                  a="Gemini proposed, coordinated, audited its own runs and scored; GLM-5.3, GPT-6-Astra and Opus 5.5 critiqued mostly from text Gemini supplied (Opus with tools off). The 1-10 scores are the models' own rubric marks; round 2's withdrawal of the CPU results holds up.",
                  names=[
                      'Society round 1: a learned tree search (GLC-AOM) to replace the hand-set search constants of rounds 4-9',
                      'Society round 2: code audit withdraws the round-10 and Stage-2 results as surrogate simulations',
                      "Society round 3: adjudicating R11 and pre-registering R12 (the '8.65 consensus' is Gemini's score alone)",
                      'Society round 4: can the R11/R12 search learn MCTS, and what can a search transformer not represent?',
                  ]),
             dict(key='surrogate', title='CPU surrogate runs (round 10 and Stage 2)',
                  a='Every pre-registered rule passed, but the curves came from formulas, random draws and mixtures toward the exact answer around untrained trunks, so they test code paths, not learned search. Society round 2 withdrew them as evidence.',
                  names=[
                      'Round 10 E1-E2: lemma channels, pair queries and two-stage tree clearing (CPU surrogate)',
                      'Round 10 E3-E4: fractal plans, episodic memory, trust pooling, markets and poker (CPU surrogate)',
                      'Stage 2: Elo league and Mini-ECI over 15 game configurations on simulated agents (CPU surrogate)',
                  ]),
             dict(key='neural', title='GPU runs (R11, R12), the synthesis paper and its audit',
                  a="Two 3-seed runs trained small GLC-AOM models on one L4 GPU ($2.55 at the list-price ceiling, not billed spend). As registered, R11 is void and R12 fails at least two of its four gates under either version of its pre-registration, which was rewritten after the results; R12's Elo comes from synthetic matches.",
                  names=[
                      'R11: first end-to-end GPU training and blind search of GLC-AOM, 12 environments, B up to 128',
                      'R12: calibrated queries, a FiLM generalist and search to B=256 (pre-registration rewritten after results)',
                      'Overnight synthesis paper on rounds 10-12 and the society (audited: not publishable as a results paper)',
                      'Overnight rounds 10-12 and the society of thinkers (gemini-only campaign): FAILED independent audit',
                  ]),
             dict(key='curiosity', title='Search imitation, discovery and curiosity rounds (R13-R15)',
                  a="R13 and R14 had no registered results at 16:46 UTC: R13's first GPU run failed and a second was running, and R14 has only a one-seed smoke run. R15's passes are set by hand-chosen weights in a runner that was edited between a mostly failing run and a passing rerun.",
                  names=[
                      'R13: can search transformer v2 represent and imitate six classical search algorithms? (pre-registered)',
                      'R14: can task-only training discover search without algorithm labels? (pre-registered, smoke run only)',
                      'R15: curiosity suite on search topology, chimera transfer, lemma memory and exploitability',
                  ]),
             dict(key='critique', title='Independent critique and audits (19 Oct)',
                  a='Read-only Opus 5.5 audits of every round: the whole-campaign critique with its summary table and next steps, and the two pillar audits.',
                  names=[
                      'Independent critique of the whole gemini campaign (rounds 10-25, society, literature, code)',
                      'Pillar 1 audit: OpenSpiel games and Stratego (R18, R22A, R24B)',
                      'Pillar 2 audit: public reasoning benchmarks and CLRS (R19, R21, R22B, R24B, R25)',
                  ]),
             dict(key='markets', title='Market rounds (R16, R17, R20, R23)',
                  a='Internal and decentralized markets, Kyle-LMSR futarchy: formula outputs and identities except the LMSR primitive and a VCG leave-one-out routine.',
                  names=[
                      'R16 internal market and R17 decentralized market',
                      'R17 decentralized market (VCG leave-one-out)',
                      'R20 Kyle-LMSR futarchy market',
                      'R23 Kyle-LMSR futarchy ("genuine")',
                  ]),
             dict(key='genuine', title='The "genuine" reruns (R18-R25)',
                  a='Gemini relabelled later rounds genuine after its own round-6 audit; the independent audits found real games and real negatives, but stand-in baselines, home-made benchmarks, untrained arms and gate code that differs from the pre-registration.',
                  names=[
                      'R18 OpenSpiel and Stratego (first attempt): CFR table blended into the policy',
                      'R19 public reasoning benchmarks: 99.4% came from a Python helper',
                      'R21 CLRS OOD and scaling: targets in the inputs',
                      'R22A OpenSpiel and Stratego ("genuine" rerun): real games, stand-in baselines',
                      'R22B public reasoning and CLRS without the helper',
                      'R24A runtime counterfactual auditor (gemini auditing gemini)',
                      'R24B v3: semiring-gated CLRS and Linear-CFR OpenSpiel',
                      'R25 v3 syndrome-guided reasoning and Kyle calibration',
                  ]),
             dict(key='literature', title='Theory, literature and architecture docs',
                  a="Gemini's expressivity theorems, six literature files written by subagents during the run, and code docs for the architecture and evaluation. The literature files appeared between 16:36 and 16:42 UTC, their citations are unchecked, and they call the architecture validated, which the results do not support.",
                  names=[
                      'Expressivity theorems R4.1-R4.3: the missing value backup and the proposed v1/v2 search transformers',
                      'Literature synthesis: 75 years of search, deduction and game-theory papers mapped onto GLC-AOM',
                      "Literature 1: Samuel's checkers, EURISKO, value of computation and EBL/CDCL lemma learning",
                      'Literature 2: algorithm selection, hyper-heuristics, NRPA and modern tree and graph search',
                      'Literature 3: neural algorithmic reasoning, differentiable tree search and transformer circuit limits',
                      'Literature 4: training on search traces, GFlowNets and any-order autoregressive models',
                      'Literature 5: dependency graphs, logical induction, information markets and imperfect-information search',
                      'GLC-AOM architecture and the 8-family multi-game suite (README overview)',
                      'Bayesian Elo league with draws, first-move advantage and a cyclic mElo term (src/eval/elo_league.py)',
                      'Mini-ECI capability index: item-response fit over belief metrics and Elo ladders (src/eval/mini_eci.py)',
                  ]),
         ]),
    dict(key='readings', title='Readings and side experiments', collapsed=False,
         q='Readings with no run, and early side experiments.',
         a='On the 274-question consistency benchmark, Grok 4.6, Luna at max and GLM-5.3 break the logic rules least '
           '(0.064–0.067). Our exact consistency engine matches Metaculus’s map_forecasting within 1e-9 and runs a '
           '121-variable map theirs cannot.',
         card='Readings with no run, and early benchmarks: Grok 4.6, Luna at max and GLM-5.3 break the logic rules least '
              '(0.064–0.067).',
         src=[('native-evaluation', 'consistency benchmark'), ('engine-proof', 'consistency engine')],
         subs=[
             dict(key='notes', title='Readings and notes',
                  a='Readings with no run: the EURISKO literature, self-modifying agents (DGM), the LIR paper’s per-trader '
                    'losses, and stacking adapters on Tinker.',
                  names=[
                      'Does the LIR paper give each trader its own loss? (reading, no run)',
                      'Can Tinker train an action adapter on top of a frozen belief adapter? (reading, no run)',
                      'EURISKO literature study',
                      'Self-modifying agents (DGM) reading',
                  ]),
             dict(key='other', title='Side experiments and early benchmarks',
                  a='Consistency benchmarks across models, data checks, a frontier market game, an exact consistency engine '
                    'and a treewidth check.',
                  names=[
                      'Consistency benchmark: GLM, Grok, Astra and Luna',
                      'Consistency benchmark: reasoning effort',
                      'Factual recall probes',
                      'Qwen own-answer collection',
                      'Jev on the consistency benchmark',
                      'Frontier market game (GPT-6)',
                      'Exact consistency engine vs Metaculus map_forecasting',
                      'Bounded treewidth as questions arrive (k-tree rule on a bayes-market clone)',
                  ]),
         ]),
]
# Old track keys with no sub-stream of the same name: (old title, the sub-streams that now hold its entries).
LEGACY = {'linet': ('Reasoning networks and world models (li-net)', ['overnight', 'aggregation', 'theory'])}
OLD_TRACK_PAGES = ['shiphunt', 'lir', 'minibench', 'baselines', 'consistency', 'links', 'market', 'training', 'soup',
                   'neurons', 'payparts', 'depth', 'linet', 'other']  # track-<key>.html existed until 16 Oct: now redirects
SUBS = {u['key']: (s, u) for s in STREAMS for u in s['subs']}  # sub-stream key -> (stream, sub-stream)
PLACE = {}  # EXP name -> (stream key, sub-stream key, group key or '')
for _s in STREAMS:
    for _u in _s['subs']:
        for _g in _u.get('groups') or [dict(key='', names=_u['names'])]:
            for _n in _g['names']:
                assert _n not in PLACE, f'{_n!r} is listed in two sub-streams ({PLACE[_n][1]} and {_u["key"]})'
                PLACE[_n] = (_s['key'], _u['key'], _g['key'])
_missing = [e[1] for e in EXP if e[1] not in PLACE]
_unknown = sorted(set(PLACE) - {e[1] for e in EXP})
assert not _missing and not _unknown, (f'EXP rows in no sub-stream (add each name to one sub-stream in STREAMS): {_missing}; '
                                       f'names in STREAMS with no EXP row: {_unknown}')

MON = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
OPEN = (9999, 12, 31)  # last day of a row that is still running ('5 Oct–')


def _day(x, mon=None):
    m = re.fullmatch(r'(\d{1,2})(?: ([A-Z][a-z]{2}))?', x.strip())
    mm = MON.index(m.group(2) or mon) if m and (m.group(2) or mon) in MON[1:] else 0
    return (2026 if mm >= 7 else 2027, mm, int(m.group(1))) if mm else None  # the MATS programme runs Sep-Dec 2026


def day_span(d):
    """EXP date -> (first day, last day) as (year, month, day): '16 Oct', '11–13 Oct', '29 Sep–2 Oct', '5 Oct–' (running)."""
    left, dash, right = d.partition('–')
    mon = re.search(r'[A-Z][a-z]{2}', right or left)
    a = _day(left, mon and mon.group(0))
    b = OPEN if dash and not right.strip() else _day(right or left)
    if not (a and b):
        print(f'warning: cannot read the date {d!r}; that row sorts last', file=sys.stderr)
        return (0, 0, 0), (0, 0, 0)
    return a, b


IDX = {e[1]: i for i, e in enumerate(EXP)}
SPAN = {e[1]: day_span(e[0]) for e in EXP}


_LOW = {e[1] for e in EXP if e[2] in ('nogain', 'problem')}


def newest(names):
    """Newest first by last day (running rows on top); on the same day Gain/Informative rows come before No gain and
    Problem rows (Kelvin, 16 Oct 2026), otherwise rows keep their EXP order."""
    return sorted(names, key=lambda n: (tuple(-x for x in SPAN[n][1]), n in _LOW, IDX[n]))


def fmt_span(names):
    a, b = min(SPAN[n][0] for n in names), max(SPAN[n][1] for n in names)
    if b == OPEN:
        return f'since {a[2]} {MON[a[1]]}'
    if a == b:
        return f'{a[2]} {MON[a[1]]}'
    return f'{a[2]}–{b[2]} {MON[b[1]]}' if a[:2] == b[:2] else f'{a[2]} {MON[a[1]]}–{b[2]} {MON[b[1]]}'

LABEL = {'gain': 'Gain', 'informative': 'Informative', 'nogain': 'No gain', 'problem': 'Problem', 'running': 'Running', 'proposed': 'Proposed'}

CSS = """
:root{--bg:#f7f7f4;--card:#fff;--ink:#161616;--ink2:#5b5a55;--line:#e3e2dc;--gain:#1f8a5c;--informative:#5b6b7c;--nogain:#b4532a;--problem:#9a2f5c;--running:#2a6fb4;--proposed:#8a6d1f}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#151513;--card:#1e1e1b;--ink:#eeeeea;--ink2:#a9a8a0;--line:#34332f;--gain:#3fb07e;--informative:#9aabbd;--nogain:#e07a4f;--problem:#d0619a;--running:#6fa6e0;--proposed:#d4b45a}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,sans-serif;padding:20px 16px}
main{max-width:860px;margin:0 auto}
h1{font-size:22px;margin:0 0 12px}
.f{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px}
.f button{font:inherit;font-size:13px;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:14px;padding:4px 12px;cursor:pointer}
.f button[aria-pressed=true]{background:var(--ink);color:var(--bg)}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}
td{padding:8px 10px;border-top:1px solid var(--line);vertical-align:top}
tr:first-child td{border-top:0}
.d{color:var(--ink2);white-space:nowrap;font-size:13px;font-variant-numeric:tabular-nums}
.s{font-size:12px;font-weight:600;white-space:nowrap}
.n a{color:var(--ink);font-weight:600;text-decoration:none;border-bottom:1px solid var(--line)}
.n a:hover{border-color:var(--ink)}
.r{color:var(--ink2);font-size:13px}
.arc{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px 10px 32px;margin:0 0 16px;font-size:14px}
.arc li{margin:2px 0} .arc b{font-weight:600;color:var(--ink2);font-size:13px;font-variant-numeric:tabular-nums}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink2);margin:0 0 6px}
"""

# Per-experiment page style: copied from the approved sample (sample-make-consistent.html).
PAGE_CSS = """
:root{--bg:#f7f7f4;--card:#fff;--ink:#161616;--ink2:#5b5a55;--line:#e3e2dc;--good:#1f8a5c;--bar:#b9b7af;--gain:#1f8a5c;--informative:#5b6b7c;--nogain:#b4532a;--problem:#9a2f5c;--running:#2a6fb4;--proposed:#8a6d1f}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#151513;--card:#1e1e1b;--ink:#eeeeea;--ink2:#a9a8a0;--line:#34332f;--good:#3fb07e;--bar:#5c5b55;--gain:#3fb07e;--informative:#9aabbd;--nogain:#e07a4f;--problem:#d0619a;--running:#6fa6e0;--proposed:#d4b45a}}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,sans-serif;padding:20px 16px}
main{max-width:680px;margin:0 auto}
a{color:var(--ink)} .back{font-size:13px;color:var(--ink2)}
h1{font-size:24px;margin:10px 0 2px} .tag{font-weight:600;font-size:13px}
h2{font-size:13px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink2);margin:22px 0 4px}
p{margin:0 0 8px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px}
svg{width:100%;height:auto;display:block}
.l{font-size:12px;fill:var(--ink2)} .v{font-size:12px;font-weight:600;fill:var(--ink)}
.src{font-size:12px;color:var(--ink2);margin-top:18px}
.res{margin-top:8px}
.cap{font-size:12px;color:var(--ink2);margin-top:4px;padding-left:28.5%}
@media (max-width:520px){.cap{padding-left:0}}
"""

DCSS = PAGE_CSS + '.dt{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}.dt th,.dt td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:left}ul{padding-left:20px;margin:0 0 8px}.refs{font-size:14px;margin-top:10px}'

MOTIV = pathlib.Path(__file__).with_name('experiment-motivations-20260929.json')
TEXT = pathlib.Path(__file__).with_name('exp-pages-text.json')  # plain-language page text + chart bars

esc = html.escape


def chart_svg(c):
    bars = c['bars']
    top = max(b['value'] for b in bars) or 1
    rows, y = [], 10
    for b in bars:
        w = max(2, round(330 * b['value'] / top))
        col = 'var(--good)' if b.get('good') else 'var(--bar)'
        rows.append(f'<text class="l" x="150" y="{y + 14}" text-anchor="end">{esc(b["label"])}</text>'
                    f'<rect x="160" y="{y}" width="{w}" height="20" rx="3" fill="{col}"/>'
                    f'<text class="v" x="{166 + w}" y="{y + 14}">{esc(b.get("text") or (f'{b["value"]:.3f}' if b["value"] < 1 else format(b["value"], "g")))}</text>')
        y += 30
    return (f'<div class="card"><svg viewBox="0 0 560 {y}" role="img" aria-label="{esc(c.get("aria", ""))}">'
            + ''.join(rows) + f'</svg><div class="cap">{esc(c.get("caption", ""))}</div></div>')


REF = pathlib.Path(__file__).with_name('exp-reference.json')  # reference pages: datasets, methods, models
KINDS = [('dataset', 'Data set', 'Data sets'), ('method', 'Method', 'Methods and baselines'), ('model', 'Model', 'Models'),
         ('note', 'Note', 'Theory and design notes')]
REFS = json.loads(REF.read_text())['refs'] if REF.exists() else []


def ref_line(slug):
    parts = []
    for k, one, many in KINDS:
        rs = [r for r in REFS if r['kind'] == k and slug in r.get('used_by', [])]
        if rs:
            parts.append(f'<b>{one if len(rs) == 1 else many}:</b> ' + ', '.join(
                f'<a href="ref-{k}-{r["slug"]}.html">{esc(r["title"])}</a>' for r in rs))
    return ('<p class="refs">' + '<br>'.join(parts) + '</p>') if parts else ''


def ref_page(r, titles):
    out = ['<a class="back" href="reference.html">← Reference</a> · <a class="back" href="index.html">all streams</a>',
           f'<h1>{esc(r["title"])}</h1>', f'<div class="tag">{dict((k, o) for k, o, _ in KINDS)[r["kind"]]}</div>',
           f'<p style="margin-top:10px">{esc(r["summary"])}</p>']
    if r.get('size') or r.get('time_range'):
        out.append('<p>' + ' · '.join(esc(x) for x in (r.get('size'), r.get('time_range')) if x) + '</p>')
    if r.get('bullets'):
        out.append(f'<h2>{"How it was built" if r["kind"] == "dataset" else "Details"}</h2><ul>'
                   + ''.join(f'<li>{esc(x)}</li>' for x in r['bullets']) + '</ul>')
    t = r.get('table')
    if t:
        out.append('<div style="overflow-x:auto"><table class="dt"><thead><tr>' + ''.join(f'<th>{esc(c)}</th>' for c in t['cols'])
                   + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{esc(str(c))}</td>' for c in row) + '</tr>' for row in t['rows'])
                   + '</tbody></table></div>' + (f'<p class="src">{esc(t["note"])}</p>' if t.get('note') else ''))
    used = [u for u in r.get('used_by', []) if u in titles]
    if used:
        out.append('<h2>Used by</h2><ul>' + ''.join(f'<li><a href="exp-{u}.html">{esc(titles[u])}</a></li>' for u in used) + '</ul>')
    if r.get('files'):
        out.append('<h2>Files</h2><ul>' + ''.join(f'<li><a href="{esc(f["href"])}">{esc(f["label"])}</a></li>' for f in r['files']) + '</ul>')
    if r.get('source'):
        out.append(f'<p class="src">Source: {esc(r["source"])}</p>')
    return (f'<!doctype html>\n<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n'
            f'<title>{esc(r["title"])}</title>\n<style>{DCSS}</style>\n<main>\n' + '\n'.join(out) + '\n</main>\n')


def details_page(slug, name, d):
    out = [f'<a class="back" href="exp-{slug}.html">← {esc(name)}</a>', f'<h1>{esc(name)}: methods &amp; data</h1>', ref_line(slug)]
    if d.get('methods'):
        out.append('<h2>Methods</h2><ul>' + ''.join(f'<li>{esc(x)}</li>' for x in d['methods']) + '</ul>')
    for t in d.get('tables', []):
        head = ''.join(f'<th>{esc(c)}</th>' for c in t['cols'])
        body = ''.join('<tr>' + ''.join(f'<td>{esc(str(c))}</td>' for c in r) + '</tr>' for r in t['rows'])
        out.append(f'<h2>{esc(t["title"])}</h2><div style="overflow-x:auto"><table class="dt"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'
                   + (f'<p class="src">{esc(t["note"])}</p>' if t.get('note') else ''))
    if d.get('files'):
        out.append('<h2>Files</h2><ul>' + ''.join(f'<li><a href="{esc(f["href"])}">{esc(f["label"])}</a></li>' for f in d['files']) + '</ul>')
    css = DCSS
    return (f'<!doctype html>\n<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n'
            f'<title>{esc(name)}: Methods</title>\n<style>{css}</style>\n<main>\n' + '\n'.join(out) + '\n</main>\n')


def exp_page(slug, date, name, st, link, t, m, titles, doc_href):
    sk, uk, gk = PLACE[name]
    s = [f'<a class="back" href="substream-{sk}-{uk}.html{"#" + gk if gk else ""}">← {esc(SUBS[uk][1]["title"])}</a> · <a class="back" href="index.html">all streams</a>',
         f'<h1>{esc(name)}</h1>',
         f'<div class="tag" style="color:var(--{st})">● {LABEL[st]} · {esc(date)}</div>']
    def linkify(txt):
        return re.sub(r'\[\[([\w.-]+)\|([^\]]+)\]\]', lambda mm: f'<a href="exp-{mm.group(1)}.html">{mm.group(2)}</a>', esc(txt))
    if t.get('why') or t.get('evidence'):
        s.append('<h2>Why</h2>')
        if t.get('evidence'):
            s.append(f'<p class="ev"><b>What we had seen:</b> {linkify(t["evidence"])}</p>')
        if t.get('why'):
            s.append(f'<p>{linkify(t["why"])}</p>')
    if t.get('what'):
        s.append(f'<h2>What we did</h2><p>{esc(t["what"])}</p>')
    s.append('<h2>Result</h2>')
    if t.get('chart'):
        s.append(chart_svg(t['chart']))
    if t.get('result'):
        s.append(f'<p class="res">{esc(t["result"])}</p>')
    if t.get('sowhat'):
        s.append(f'<h2>So what</h2><p>{esc(t["sowhat"])}</p>')
    if t.get('decided'):
        s.append(f'<h2>Decided</h2><p>{esc(t["decided"])}</p>')
    nxt = [l['slug'] for l in m.get('led_to', []) if l.get('slug') in titles]
    if nxt:
        s.append('<h2>Led to →</h2><p>' + ' · '.join(
            f'<a href="exp-{n}.html">{esc(titles[n])}</a>' for n in dict.fromkeys(nxt)) + '</p>')
    more = []
    if t.get('details'):
        more.append(f'<a href="exp-{slug}-details.html">Methods &amp; data</a>')
    if more:
        s.append('<h2>More</h2><p>' + ' · '.join(more) + '</p>')
    if doc_href:
        base, _, anchor = doc_href.partition('#')  # md:path#anchor links point into one section of a shared write-up
        label = base.rsplit("/", 1)[-1].removesuffix(".html") + (f', {anchor.upper()}' if anchor else '')
        s.append(f'<p class="src">Full technical write-up: <a href="{doc_href}">{esc(label)}</a></p>')
    return (f'<!doctype html>\n<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">\n'
            f'<title>{esc(name)}</title>\n<style>{PAGE_CSS}</style>\n<main>\n' + '\n'.join(s) + '\n</main>\n')


_DOCS_DONE = set()


def render_doc(stem):
    if stem in _DOCS_DONE:
        return f'{stem}.html'
    _DOCS_DONE.add(stem)
    src = (DOCS / f'{stem}.md').read_text()
    body = markdown.markdown(src, extensions=['tables', 'fenced_code'])
    # links between listed write-ups (href="other.md", other = an EXP doc) point at the rendered page
    listed = {e[4][4:] for e in EXP if e[4].startswith('doc:')}
    body = re.sub(r'href="([\w.-]+)\.md(#[^"]*)?"', lambda mm: f'href="{mm.group(1)}.html{mm.group(2) or ""}"'
                  if mm.group(1) in listed else mm.group(0), body)
    # drop links that cannot resolve on the site (unlisted .md, ../data files, in-page anchors without ids)
    body = re.sub(r'<a href="(?!https?://)(?![\w.-]+\.html(?:#[^"]*)?")[^"]*">(.*?)</a>', r'\1', body)
    page = (f'<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(stem)}</title><style>body{{max-width:820px;margin:0 auto;padding:20px 16px;font:15px/1.5 system-ui;'
            f'background:#fff;color:#161616}}table{{border-collapse:collapse;font-size:13px}}td,th{{border:1px solid #ddd;padding:4px 6px}}'
            f'code,pre{{background:#f3f3f0;font-size:12.5px}}pre{{overflow-x:auto;padding:8px}}</style>'
            f'<p><a href="index.html">← all experiments</a></p>{body}')
    (OUT / f'{stem}.html').write_text(page)
    return f'{stem}.html'


MD_NAMES = {  # side-track write-ups outside DOCS -> published file stem
    'audits/gemini-campaign-critique-2026-10-19.md': 'critique-gemini-campaign-20261019',
    'audits/pillar1-openspiel-stratego-audit-2026-10-19.md': 'pillar1-audit-20261019',
    'audits/pillar2-reasoning-clrs-audit-2026-10-19.md': 'pillar2-audit-20261019',
    'results/RESULTS-R16-INTERNAL-MARKET.md': 'r16-internal-market',
    'results/RESULTS-R17-DECENTRALIZED-MARKET.md': 'r17-decentralized-market',
    'results/RESULTS-R18-OPENSPIEL-STRATEGO.md': 'r18-openspiel-stratego',
    'results/RESULTS-R19-PUBLIC-REASONING-BENCHMARKS.md': 'r19-public-reasoning',
    'results/RESULTS-R20-KYLE-LMSR-FUTARCHY.md': 'r20-kyle-lmsr',
    'results/RESULTS-R21-CLRS-OOD-AND-SCALING.md': 'r21-clrs-ood',
    'results/RESULTS-R22A-GENUINE-OPENSPIEL-STRATEGO.md': 'r22a-genuine-openspiel',
    'results/RESULTS-R22B-GENUINE-PUBLIC-REASONING-AND-CLRS.md': 'r22b-genuine-reasoning',
    'results/RESULTS-R23-GENUINE-KYLE-LMSR-FUTARCHY.md': 'r23-genuine-kyle',
    'results/RESULTS-R24A-RUNTIME-COUNTERFACTUAL-AUDIT.md': 'r24a-runtime-audit',
    'results/RESULTS-R24B-V3-CLRS-AND-OPENSPIEL-UNIFICATION.md': 'r24b-v3-unification',
    'results/RESULTS-R25-V3-REASONING-SYNERGY-AND-KYLE-CALIBRATION.md': 'r25-v3-reasoning',
    'round1/ROUND1-SCOREBOARD.md': 'society-round1',
    'round2/ROUND2-SCOREBOARD.md': 'society-round2',
    'round3/ROUND3-SCOREBOARD.md': 'society-round3',
    'round4/ROUND4-SCOREBOARD.md': 'society-round4',
    'results/RESULTS-R10-E1-E2.md': 'r10-e1-e2-surrogate',
    'results/RESULTS-R10-E3-E4.md': 'r10-e3-e4-surrogate',
    'results/RESULTS-STAGE2-MINI-ECI.md': 'stage2-mini-eci-surrogate',
    'results/RESULTS-R11-GPU-NEURAL.md': 'r11-gpu-neural',
    'results/RESULTS-R12-GPU-V2.md': 'r12-gpu-v2',
    'papers/overnight-society-and-glc-aom-paper-2026-10-18.md': 'overnight-society-paper-20261018',
    'benchmarks/PREREG-R13-ALGO-IMITATION.md': 'r13-algo-imitation-prereg',
    'benchmarks/PREREG-R14-UNSUP-DISCOVERY.md': 'r14-unsup-discovery-prereg',
    'benchmarks/PREREG-R15-CURIOSITY-SUITE.md': 'r15-curiosity-suite-prereg',
    'round4/01-expressivity-theorems-and-mcts-gap.md': 'expressivity-theorems-round4',
    'literature/LITERATURE-GENEALOGY-SYNTHESIS.md': 'literature-synthesis',
    'literature/01-gofai-metareasoning-eurisko-ebl-cdcl.md': 'literature-1-gofai',
    'literature/02-metaheuristics-hyperheuristics-nrpa-mcts-pns.md': 'literature-2-metaheuristics',
    'literature/03-neural-algorithmic-reasoning-clrs-mctsnet-incontext.md': 'literature-3-neural-algorithmic',
    'literature/04-sos-searchformer-gflownets-anyorder.md': 'literature-4-search-traces',
    'literature/05-pdg-logical-induction-cfr-rebel-markets.md': 'literature-5-pdg-markets',
    'inconsistency-search/README.md': 'glc-aom-readme',
    'eval/elo_league.py': 'elo-league-doc',
    'eval/mini_eci.py': 'mini-eci-doc',
    'li-net/r9e1/RESULTS-R9-E1.md': 'r9-e1-results',
    'li-net/r8e1/RESULTS-R8-E1.md': 'r8-e1-results',
    'micro-stratego/r8/RESULTS-R8-E3.md': 'micro-stratego-r8-e3-results',
    'li-net/r7e1/RESULTS-R7-E1.md': 'r7-e1-results',
    'micro-stratego/r7/RESULTS-R7-E3.md': 'micro-stratego-r7-e3-results',
    'li-net/r6e1/RESULTS-R6-E1.md': 'r6-e1-results',
    'micro-stratego/r6/RESULTS-R6-E3.md': 'micro-stratego-r6-e3-results',
    'micro-stratego/RESULTS-R5-E2.md': 'micro-stratego-r5-e2-results',
    'li-net/r5e1/RESULTS-R5-E1.md': 'r5-e1-results',
    'lir-agent/RESULTS-R2.md': 'lir-r4-e3a-results',
    'micro-stratego/RESULTS-MS.md': 'micro-stratego-results',
    'linear-agg/RESULTS-LA.md': 'linear-agg-results',
    'r4-e3b/RESULTS-R4-E3b.md': 'r4-e3b-results',
    'overnight-results-paper-2026-10-15.md': 'overnight-paper-20261015',
    'experiments-rounds-4-9-paper-v3.md': 'rounds-4-9-paper-v3',
    'overnight-society-paper-audit-2026-10-18.md': 'rounds-10-12-audit-20261018',
    'E-SUD/RESULTS.md': 'e-sud-results',
    'E-SUD/RESULTS-R2.md': 'e-sud-results-r2',
    'E-SUD/RESULTS-R3.md': 'e-sud-results-r3',
    'R4-E2/RESULTS-R4.md': 'r4-e2-results',
    'R5-E3/RESULTS-R5-E3.md': 'r5-e3-results',
    'R6-E2/RESULTS-R6-E2.md': 'r6-e2-results',
    'R7-E2/RESULTS-R7-E2.md': 'r7-e2-results',
    'R8-E2/RESULTS-R8-E2.md': 'r8-e2-results',
    'R9-E2/RESULTS-R9-E2.md': 'r9-e2-results',
    'eca/RESULTS.md': 'eca-results',
    'eca/RESULTS-R2.md': 'eca-r2-results',
    'eca/RESULTS-R3.md': 'eca-r3-results',
    'eurisko-study/REPORT.md': 'eurisko-report', 'selfmod/IDEAS.md': 'selfmod-ideas',
    'logical-soup/REPORT-SOUP.md': 'logical-soup-report', 'wealth-net/RESULTS.md': 'wealth-net-results',
    'traders-net/wiki-traders-net.md': 'traders-net',
    'trader-economy/CATALOGUE.md': 'trader-economy-catalogue', 'calibrated-ai/trader-economy-note-20260929.md': 'trader-economy-note',
    'trader-theory/THEORY.md': 'trader-theory', 'trader-theory/THEORY-2.md': 'trader-theory-2',
    'puzzles/CATALOGUE-PUZZLES.md': 'trader-puzzles-catalogue', 'item1-costnet/RESULTS.md': 'costnet-bb-results', 'item2-p10/RESULTS.md': 'p10-init-check-results', 'item3-p8/RESULTS.md': 'p8-n4096-results', 'P15/RESULTS.md': 'p15-results', 'P16/RESULTS.md': 'p16-results', 'P19-fast-reader/RESULTS.md': 'p19-results', 'P17-breadth-depth/RESULTS.md': 'p17-results', 'P13/RESULTS.md': 'p13-results', 'P18-detached-mazes/RESULTS.md': 'p18-results', 'dt-baseline/RESULTS.md': 'dt-baseline-results', 'metaculus/ENGINE-PROOF.md': 'engine-proof', 'M2/BAKEOFF.md': 'm2-bakeoff', 'ktree-market-grpo-design-20261009.md': 'ktree-market-grpo-design', 'autoresearch/WEEKLY.md': 'minibench-autoresearch-weekly', 'bt-scale/RESULTS.md': 'minibench-backtest-results', 'bt-scale-luna/RESULTS.md': 'minibench-luna-results', 'smallnet/RESULTS.md': 'shiphunt-smallnet-results', 'wm2/RESULTS.md': 'shiphunt-wm2-results', 'worldmodel-coherence-research-2026-10-11.md': 'worldmodel-coherence-research', 'calibrated-ai/market-structure-treewidth-20261009.md': 'ktree-treewidth-check', 'lir-agent/experiments/E1/RESULTS.md': 'lir-e1-results', 'lir-agent/experiments/E2/RESULTS.md': 'lir-e2-results', 'lir-agent/experiments/E3/RESULTS.md': 'lir-e3-results', 'lir-agent/experiments/E4/RESULTS.md': 'lir-e4-results', 'lir-agent/experiments/E5/RESULTS.md': 'lir-e5-results', 'lir-agent/experiments/E6/RESULTS.md': 'lir-e6-results', 'lir-agent/experiments/E7/RESULTS.md': 'lir-e7-results', 'li-net/bb/RESULTS.md': 'busy-beaver-results', 'li-net/swarm-predictor/RESULTS.md': 'shiphunt-swarm-results', 'li-net/DARKROOM.md': 'darkroom-note', 'traders-net/study/M1/AUDIT.md': 'traders-net-m1-audit', 'trader-theory/THEORY-4.md': 'trader-theory-4', 'worldmodel-abstraction-research-2026-10-11.md': 'worldmodel-abstraction-research', 'td-gvf-martingale-note-2026-10-10.md': 'td-gvf-martingale-note', 'tinker-lora-stacking-2026-10-09.md': 'tinker-lora-stacking', 'scientist-ai/BRIEF.md': 'scientist-ai-brief', 'li-net/SURVEY-WORLD-MODELS.md': 'learned-world-models-survey', 'li-net/STUDY-LIR.md': 'lir-paper-study', 'li-net/IDEAS-LI-NET.md': 'li-net-math-summary', 'traders-net/study/M8/M8.md': 'traders-net-m8'}
TOC_IDS = {'trader-economy-catalogue', 'trader-puzzles-catalogue'}  # docs whose headings get ids (catalogue: '## E-T1-1: ...' -> #e-t1-1, '## P0: ...' -> #p0)
_RENDERED = {}
MATH_DOCS = {'r16-internal-market': 'Gemini-run round; the card status comes from the independent audits of 2026-10-19 (docs/audits in futarchy-fi/inconsistency-search).', 'r17-decentralized-market': 'Gemini-run round; the card status comes from the independent audits of 2026-10-19 (docs/audits in futarchy-fi/inconsistency-search).', 'r18-openspiel-stratego': 'Gemini-run round; the card status comes from the independent audits of 2026-10-19 (docs/audits in futarchy-fi/inconsistency-search).', 'r19-public-reasoning': 'Gemini-run round; the card status comes from the independent audits of 2026-10-19 (docs/audits in futarchy-fi/inconsistency-search).', 'r20-kyle-lmsr': 'Gemini-run round; the card status comes from the independent audits of 2026-10-19 (docs/audits in futarchy-fi/inconsistency-search).', 'r21-clrs-ood': 'Gemini-run round; the card status comes from the independent audits of 2026-10-19 (docs/audits in futarchy-fi/inconsistency-search).', 'r22a-genuine-openspiel': 'Gemini-run round; the card status comes from the independent audits of 2026-10-19 (docs/audits in futarchy-fi/inconsistency-search).', 'r22b-genuine-reasoning': 'Gemini-run round; the card status comes from the independent audits of 2026-10-19 (docs/audits in futarchy-fi/inconsistency-search).', 'r23-genuine-kyle': 'Gemini-run round; the card status comes from the independent audits of 2026-10-19 (docs/audits in futarchy-fi/inconsistency-search).', 'r24a-runtime-audit': 'Gemini-run round; the card status comes from the independent audits of 2026-10-19 (docs/audits in futarchy-fi/inconsistency-search).', 'r24b-v3-unification': 'Gemini-run round; the card status comes from the independent audits of 2026-10-19 (docs/audits in futarchy-fi/inconsistency-search).', 'r25-v3-reasoning': 'Gemini-run round; the card status comes from the independent audits of 2026-10-19 (docs/audits in futarchy-fi/inconsistency-search).', 'overnight-society-paper-20261018': 'Written by the Gemini “Argon” agent (or a subagent it spawned) during its overnight campaign of 2026-10-18; gemini-run and not independently audited unless the card says so.', 'expressivity-theorems-round4': 'Written by the Gemini “Argon” agent (or a subagent it spawned) during its overnight campaign of 2026-10-18; gemini-run and not independently audited unless the card says so.', 'literature-synthesis': 'Written by the Gemini “Argon” agent (or a subagent it spawned) during its overnight campaign of 2026-10-18; gemini-run and not independently audited unless the card says so.', 'literature-1-gofai': 'Written by the Gemini “Argon” agent (or a subagent it spawned) during its overnight campaign of 2026-10-18; gemini-run and not independently audited unless the card says so.', 'literature-2-metaheuristics': 'Written by the Gemini “Argon” agent (or a subagent it spawned) during its overnight campaign of 2026-10-18; gemini-run and not independently audited unless the card says so.', 'literature-3-neural-algorithmic': 'Written by the Gemini “Argon” agent (or a subagent it spawned) during its overnight campaign of 2026-10-18; gemini-run and not independently audited unless the card says so.', 'literature-4-search-traces': 'Written by the Gemini “Argon” agent (or a subagent it spawned) during its overnight campaign of 2026-10-18; gemini-run and not independently audited unless the card says so.', 'literature-5-pdg-markets': 'Written by the Gemini “Argon” agent (or a subagent it spawned) during its overnight campaign of 2026-10-18; gemini-run and not independently audited unless the card says so.', 'r10-e1-e2-surrogate': 'Written by the Gemini “Argon” agent (or a subagent it spawned) during its overnight campaign of 2026-10-18; gemini-run and not independently audited unless the card says so.', 'r10-e3-e4-surrogate': 'Written by the Gemini “Argon” agent (or a subagent it spawned) during its overnight campaign of 2026-10-18; gemini-run and not independently audited unless the card says so.', 'stage2-mini-eci-surrogate': 'Written by the Gemini “Argon” agent (or a subagent it spawned) during its overnight campaign of 2026-10-18; gemini-run and not independently audited unless the card says so.', 'r11-gpu-neural': 'Written by the Gemini “Argon” agent (or a subagent it spawned) during its overnight campaign of 2026-10-18; gemini-run and not independently audited unless the card says so.', 'r12-gpu-v2': 'Written by the Gemini “Argon” agent (or a subagent it spawned) during its overnight campaign of 2026-10-18; gemini-run and not independently audited unless the card says so.', 'r13-algo-imitation-prereg': 'Written by the Gemini “Argon” agent (or a subagent it spawned) during its overnight campaign of 2026-10-18; gemini-run and not independently audited unless the card says so.', 'r14-unsup-discovery-prereg': 'Written by the Gemini “Argon” agent (or a subagent it spawned) during its overnight campaign of 2026-10-18; gemini-run and not independently audited unless the card says so.', 'r15-curiosity-suite-prereg': 'Written by the Gemini “Argon” agent (or a subagent it spawned) during its overnight campaign of 2026-10-18; gemini-run and not independently audited unless the card says so.', 'overnight-paper-20261015': 'Written by the Gemini “Argon” agent; edited for format only.', 'rounds-4-9-paper-v3': 'Written by the Gemini “Argon” agent; v3 carries the line edits of two independent Opus 5.5 audits (docs/audits in futarchy-fi/inconsistency-search).', 'rounds-10-12-audit-20261018': 'Independent audit by an Opus 5.5 subagent of opus-mats, read-only, of commit dc2c8ee of futarchy-fi/inconsistency-search.'}  # stems rendered with KaTeX, with a credit line


def render_md(path):
    if path in _RENDERED:
        return _RENDERED[path]
    stem = next(v for k, v in MD_NAMES.items() if path.endswith(k))
    if stem in TOC_IDS:
        from markdown.extensions.toc import TocExtension
        ext = ['tables', 'fenced_code', TocExtension(slugify=lambda v, sep: re.sub(r'[^\w-]+', '-', v.split(':')[0].strip().lower()))]
    else:
        ext = ['tables', 'fenced_code']
    src = pathlib.Path(path).read_text()
    maths = []
    if stem in MATH_DOCS:  # keep $...$ away from markdown, render with KaTeX (raw TeX stays readable if the CDN is blocked)
        def _keep(mm):
            maths.append(mm.group(0))
            return f'MATHX{len(maths) - 1}X'
        src = re.sub(r'\$(?:\\.|[^$\\\n])+?\$', _keep, src)
    body = markdown.markdown(src, extensions=ext)
    body = re.sub(r'MATHX(\d+)X', lambda mm: html.escape(maths[int(mm.group(1))]), body)
    body = re.sub(r'<a href="(?!https://(?!100\.))[^"]*">(.*?)</a>', r'\1', body)  # drop local, relative and Tailscale links
    body = re.sub(r'https?://100\.\d+\.\d+\.\d+(:\d+)?\S*', '(internal page)', body)
    extra = ''
    if stem in MATH_DOCS:
        kt = 'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/'
        extra = (f'<link rel="stylesheet" href="{kt}katex.min.css"><script defer src="{kt}katex.min.js"></script>'
                 f'<script defer src="{kt}contrib/auto-render.min.js" onload="renderMathInElement(document.body,'
                 '{delimiters:[{left:\'$\',right:\'$\',display:false}],throwOnError:false})"></script>')
    note = f'<p style="color:#5b5a55;font-size:13px">{MATH_DOCS[stem]}</p>' if stem in MATH_DOCS else ''
    page = (f'<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(stem)}</title>{extra}<style>body{{max-width:820px;margin:0 auto;padding:20px 16px;font:15px/1.5 system-ui;'
            f'background:#fff;color:#161616}}table{{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto}}td,th{{border:1px solid #ddd;padding:4px 6px}}'
            f'code,pre{{background:#f3f3f0;font-size:12.5px}}pre{{overflow-x:auto;padding:8px}}</style>'
            f'<p><a href="index.html">← all experiments</a></p>{note}{body}')
    (OUT / f'{stem}.html').write_text(page)
    _RENDERED[path] = f'{stem}.html'
    return f'{stem}.html'


SIDE = pathlib.Path(__file__).with_name('exp-pages-side.json')  # side-track experiments: slug, title, led_to, page text

OUT.mkdir(exist_ok=True)
motiv = json.loads(MOTIV.read_text())
side = json.loads(SIDE.read_text())['experiments']
M = {e['title']: e for e in motiv['experiments']}
M.update({e['title']: {'slug': e['slug'], 'led_to': [{'slug': x} for x in e['led_to']]} for e in side})
TITLES = {e['slug']: e['title'] for e in motiv['experiments']}
TITLES.update({e['slug']: e['title'] for e in side})
text = json.loads(TEXT.read_text())
text['pages'].update({e['slug']: e['page'] for e in side})
ROW, NAME_HTML, HREF = {}, {}, {}  # EXP name -> its table row / its linked name / its page
for date, name, st, res, link in EXP:
    if link.startswith('doc:'):
        doc = render_doc(link[4:])
    elif link.startswith('md:'):
        path, _, anchor = link[3:].partition('#')
        doc = render_md(path) + (f'#{anchor}' if anchor else '')
    elif link.startswith('site:'):
        doc = SITE + link[5:]
    else:
        doc = ''
    m = M.get(name, {})
    slug = m.get('slug')
    href = doc
    if slug and slug in text['pages']:
        if text['pages'][slug].get('details'):
            for f in text['pages'][slug]['details'].get('files', []):  # render any linked consistency-ft doc not yet rendered
                st_ = f['href'].removesuffix('.html')
                if '/' not in st_ and (DOCS / f'{st_}.md').exists():
                    render_doc(st_)
            (OUT / f'exp-{slug}-details.html').write_text(details_page(slug, name, text['pages'][slug]['details']))
        (OUT / f'exp-{slug}.html').write_text(exp_page(slug, date, name, st, link, text['pages'][slug], m, TITLES, doc))
        href = f'exp-{slug}.html'
    nm = f'<a href="{href}">{html.escape(name)}</a>' if href else html.escape(name)
    NAME_HTML[name], HREF[name] = nm, href
    ROW[name] = (f'<tr data-s="{st}" data-t="{" ".join(k for k in PLACE[name] if k)}"><td class="d">{date}</td>'
                 f'<td class="s" style="color:var(--{st})">● {LABEL[st]}</td>'
                 f'<td><div class="n">{nm}</div><div class="r">{html.escape(res)}</div></td></tr>')


TCSS = """.cards{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
.tc{display:block;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;color:var(--ink);text-decoration:none}
.tc:hover{border-color:var(--ink2)} .tc h3{margin:0 0 4px;font-size:16px} .tc .q{color:var(--ink2);font-size:13px;margin-bottom:6px}
.tc .a{font-size:14px;margin-bottom:8px} .cnt{font-size:12px;font-weight:600} .all{display:inline-block;margin-top:16px;font-size:14px;color:var(--ink)}"""
STATUS = {e[1]: e[2] for e in EXP}
DATE = {e[1]: e[0] for e in EXP}
TITLE = 'Coherent beliefs and search: experiments'
SUBTITLE = ('Does training or search for coherent beliefs make forecasts more accurate? Language models and small networks on '
            'forecasting questions (Manifold, MiniBench) and on toy worlds with exact answers (ShipHunt, cellular automata, '
            'Sudoku), scored on real outcomes or exact posteriors.')
SCSS = """.sb{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:0 0 12px}
.sb.flat{background:none;border:0;border-top:1px solid var(--line);border-radius:0;padding:14px 0 0;margin:26px 0 0}
details.sb>summary{cursor:pointer} details.sb>summary .cnt{display:block;margin:4px 0 0}
.stt{display:inline;font-size:17px;text-transform:none;letter-spacing:0;color:var(--ink);margin:0}
.stt a,h1 a{color:var(--ink);text-decoration:none;border-bottom:1px solid var(--line)} .stt a:hover{border-color:var(--ink)}
.no{color:var(--ink2);font-weight:600;margin-right:8px}
.q{color:var(--ink2);font-size:13px;margin:4px 0 6px} .a{font-size:14px;margin:0 0 8px}
.lt{font-size:13px;margin:6px 0 0} .lt .n{display:inline} .lt .s{margin-left:4px}
.subs{font-size:13px;color:var(--ink2);margin:6px 0 0} .subs a{color:var(--ink)}
.src{font-size:12px;color:var(--ink2);margin:6px 0 0} .src a{color:var(--ink2)}
.sh{font-size:17px;text-transform:none;letter-spacing:0;color:var(--ink);margin:26px 0 2px} .sh+.r{margin:0 0 8px}
.gh{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:var(--ink2);margin:16px 0 6px}
.cn{font-weight:400;color:var(--ink2);font-size:13px}
.tf{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 12px;margin:10px 0 12px;font-size:13px}
.nav{display:flex;justify-content:space-between;gap:12px;margin:24px 0 0;font-size:14px} .nav a{color:var(--ink)}
.back{font-size:13px;color:var(--ink2)} h1+.q{margin-top:-6px}
.f{margin-top:12px}
:root{--ch:240px}
.tc{position:relative} .tc .no{margin-right:6px} .tc .lt{font-size:12px;color:var(--ink2);margin:4px 0 0}
.cards.fixed{grid-auto-rows:var(--ch);margin:8px 0 0}
.cards.fixed>.tc{height:var(--ch);box-sizing:border-box;overflow:hidden;display:flex;flex-direction:column}
.fixed .tc h3{font-size:16px;line-height:1.3;margin:0 0 4px} .fixed .ec h3{font-size:15px}
.fixed .tc .a{margin:0 0 6px;flex:0 1 auto;min-height:0} .fixed .tc .a.sm{font-size:13px}
.fixed .ec .ed{font-size:12px;margin:0 0 4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:none}
.fixed .ec .r{flex:0 1 auto;min-height:0}
.cl{display:-webkit-box;-webkit-box-orient:vertical;overflow:hidden}
.l1{-webkit-line-clamp:1} .l2{-webkit-line-clamp:2} .l3{-webkit-line-clamp:3} .l5{-webkit-line-clamp:5} .l6{-webkit-line-clamp:6}
.bot{margin-top:auto;flex:none} .sh+.cards{margin-top:6px}
.dir{display:inline-block;margin:0 0 4px;font-size:14px;font-weight:600;color:var(--ink)}
.arch{margin:30px 0 0;padding:12px 0 0;border-top:1px solid var(--line)}
.acards{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px}
.ac{display:block;border:1px dashed var(--line);border-radius:8px;padding:8px 10px;color:var(--ink2);text-decoration:none}
.ac:hover{border-color:var(--ink2)} .ac .no{margin-right:6px}
.ac .at{display:block;font-size:13.5px;font-weight:600;line-height:1.3;margin:0 0 3px}
.ac .am{display:block;font-size:12px;line-height:1.35} .ac .am span{color:inherit!important}"""

# Archive band (Kelvin, 16 Oct 2026: "deprioritize previous experiments that have little relevance"): these streams and
# sub-streams leave the index grid (and the card grid of their stream page) for a small muted band at the bottom. Their
# pages, anchors and entry counts stay as they were; the index stream cards count only the sub-streams still on the grid.
ARCHIVE_STREAMS = ['markets', 'readings']
ARCHIVE_SUBS = ['market', 'baselines']  # sub-streams of 'coherence'


def live_names(s):
    return [n for u in s['subs'] if u['key'] not in ARCHIVE_SUBS for n in names_of(u)]


def arch_card(href, title, names, no='', key='', note=''):
    """Archive band: a small muted card (id = the old card's anchor)."""
    idattr = f' id="{key}"' if key else ''
    num = f'<span class="no">{no}</span>' if no else ''
    pre = esc(note) + ' · ' if note else ''
    return (f'<a class="ac"{idattr} href="{href}"><span class="at">{num}{esc(title)}</span>'
            f'<span class="am">{pre}{len(names)} entries: {count_line(names)}</span></a>')


def archive_band(cards):
    return f'<section class="arch"><h2>Archive</h2><div class="acards">{"".join(cards)}</div></section>' if cards else ''


def head(title):
    return (f'<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{title}</title><style>{CSS}{TCSS}{SCSS}</style><main>')


def filt(names):
    c = {k: sum(1 for e in EXP if e[1] in names and e[2] == k) for k in LABEL}
    return ('<div class="f"><button data-f="all" aria-pressed="true">All ' + str(len(names)) + '</button>' + ''.join(
        f'<button data-f="{k}" aria-pressed="false"><span style="color:var(--{k})">●</span> {v} {c[k]}</button>'
        for k, v in LABEL.items() if c[k]) + '</div>')


def names_of(x):
    """A stream, sub-stream or group -> its EXP names."""
    if 'subs' in x:
        return [n for u in x['subs'] for n in names_of(u)]
    if 'groups' in x:
        return [n for g in x['groups'] for n in names_of(g)]
    return list(x['names'])


def count_line(names):
    c = {k: sum(1 for n in names if STATUS[n] == k) for k in LABEL}
    return ' · '.join(f'<span style="color:var(--{k})">{c[k]} {LABEL[k].lower()}</span>' for k in LABEL if c[k])


def table(names):
    return '<table>' + ''.join(ROW[n] for n in newest(names)) + '</table>'


def latest(names):
    n = newest(names)[0]
    return (f'<div class="lt">Latest: <span class="d">{esc(DATE[n])}</span> <span class="n">{NAME_HTML[n]}</span>'
            f'<span class="s" style="color:var(--{STATUS[n]})">● {LABEL[STATUS[n]]}</span></div>')


def sources(s):
    return ('<p class="src">Numbers from: ' + ' · '.join(
        f'<a href="{p}.html">{esc(lbl)}</a>' if p else esc(lbl) for p, lbl in s['src']) + '</p>')


def sub_section(u, level):
    """A sub-stream: heading, one-line summary, its entries newest first (one table per finer group)."""
    names = names_of(u)
    pre = ''.join(f'<span id="{k}"></span>' for k, (_, subs) in LEGACY.items() if subs[0] == u['key'])
    out = (f'<h{level} class="sh">{esc(u["title"])} <span class="cn">{len(names)}</span></h{level}>'
           + (f'<p class="r">{esc(u["a"])}</p>' if u.get('a') else ''))
    if u.get('groups'):
        out += ''.join(f'<div class="grp" id="{g["key"]}" data-sec><h{level + 1} class="gh">{esc(g["title"])} '
                       f'<span class="cn">{len(g["names"])}</span></h{level + 1}>{table(g["names"])}</div>' for g in u['groups'])
    else:
        out += table(names)
    return f'{pre}<section class="sub" id="{u["key"]}" data-sec>{out}</section>'


def stream_block(i, s, body='', sub_href=None, cls='sb'):
    """A stream on the index (no body) or on all.html (body = its sub-streams). A collapsed stream starts folded."""
    names = names_of(s)
    hd = f'<h2 class="stt"><span class="no">{i}</span><a href="stream-{s["key"]}.html">{esc(s["title"])}</a></h2>'
    cnt = f'<div class="cnt">{len(names)} entries, {fmt_span(names)}: {count_line(names)}</div>'
    subs = '<div class="subs">' + ' · '.join(
        f'<a href="{"stream-" + s["key"] + ".html" if sub_href is None else sub_href}#{u["key"]}">{esc(u["title"])}</a> {len(names_of(u))}'
        for u in s['subs']) + '</div>'
    inner = f'<div class="q">{esc(s["q"])}</div><p class="a">{esc(s["a"])}</p>{latest(names)}{subs}{sources(s)}{body}'
    if s['collapsed']:
        return f'<details class="{cls}" id="{s["key"]}" data-sec><summary>{hd}{cnt}</summary>{inner}</details>'
    return f'<section class="{cls}" id="{s["key"]}" data-sec>{hd}{cnt}{inner}</section>'


def sub_page(s, u):
    return f'substream-{s["key"]}-{u["key"]}.html'


SUBPAGE = {}  # sub-stream and group key -> its sub-stream page (stream pages send #key and ?track=key there)
for _s in STREAMS:
    for _u in _s['subs']:
        for _k in [_u['key']] + [g['key'] for g in _u.get('groups', [])]:
            SUBPAGE[_k] = sub_page(_s, _u)
KEYPAGE = dict(SUBPAGE)  # index.html: also old keys split across sub-streams -> their stream page
KEYPAGE.update({k: f'stream-{SUBS[subs[0]][0]["key"]}.html' for k, (_, subs) in LEGACY.items()})
NAMES_JS = {s['key']: s['title'] for s in STREAMS}
NAMES_JS.update({u['key']: u['title'] for s in STREAMS for u in s['subs']})
NAMES_JS.update({g['key']: f'{u["title"]}: {g["title"]}' for s in STREAMS for u in s['subs'] for g in u.get('groups', [])})
NAMES_JS.update({k: f'{t} (old track), now ' + ', '.join(SUBS[u][1]['title'] for u in subs) for k, (t, subs) in LEGACY.items()})
# Lists (sub-stream pages, all.html): status buttons; ?track=<key> keeps only that stream, sub-stream, group or old track;
# #<key> opens a folded stream and scrolls to it. Sections with no visible row are hidden.
LIST_JS = """<script>(function(){
var N=%s,A=%s,st='all',tk=new URLSearchParams(location.search).get('track');if(!N[tk])tk=null;
function has(r){if(!tk)return true;var t=' '+r.dataset.t+' ';return (A[tk]||[tk]).some(function(k){return t.indexOf(' '+k+' ')>=0})}
function apply(){document.querySelectorAll('[data-s]').forEach(function(r){r.style.display=(st==='all'||r.dataset.s===st)&&has(r)?'':'none'});
document.querySelectorAll('[data-sec]').forEach(function(s){var on=[].some.call(s.querySelectorAll('[data-s]'),function(r){return r.style.display!=='none'});
s.style.display=on?'':'none';if(on&&st!=='all'&&s.tagName==='DETAILS')s.open=true})}
function openTo(el){for(var p=el;p;p=p.parentElement)if(p.tagName==='DETAILS')p.open=true;el.scrollIntoView()}
document.querySelectorAll('.f button').forEach(function(b){b.onclick=function(){
document.querySelectorAll('.f button').forEach(function(x){x.setAttribute('aria-pressed',x===b)});st=b.dataset.f;apply()}});
if(tk){var bar=document.querySelector('.tf');bar.querySelector('b').textContent=N[tk];bar.hidden=false;apply();
var first=[].filter.call(document.querySelectorAll('section[data-sec]'),function(s){return s.style.display!=='none'})[0];
if(first&&!location.hash)openTo(first)}
function onHash(){var id=decodeURIComponent(location.hash.slice(1)),el=id&&document.getElementById(id);if(el)openTo(el)}
addEventListener('hashchange',onHash);onHash()})();</script>""" % (json.dumps(NAMES_JS, ensure_ascii=False),
                                                                   json.dumps({k: v for k, (_, v) in LEGACY.items()}))
# index.html and stream pages: #<key> or ?track=<key> of a sub-stream or group (and, on the index, of an old split track)
# opens the page that lists it.
JUMP_JS = """<script>(function(){var P=%s,q=new URLSearchParams(location.search).get('track'),h=decodeURIComponent(location.hash.slice(1)),
k=P[q]?q:(P[h]?h:'');if(k)location.replace(P[k]+'#'+k)})();</script>"""
INDEX_JS, STREAM_JS = JUMP_JS % json.dumps(KEYPAGE), JUMP_JS % json.dumps(SUBPAGE)


def track_bar(page):
    return f'<div class="tf" hidden>Showing only <b></b> · <a href="{page}">show all</a></div>'


def stream_card(i, s):
    """Index: a stream as a fixed-size card that opens the stream page (archived sub-streams are counted in the band)."""
    names = live_names(s)
    n = newest(names)[0]
    return (f'<a class="tc" id="{s["key"]}" href="stream-{s["key"]}.html"><h3 class="cl l2"><span class="no">{i}</span>{esc(s["title"])}</h3>'
            f'<div class="a cl l5">{esc(s["card"])}</div><div class="bot"><div class="cnt cl l2">{len(names)} entries: {count_line(names)}</div>'
            f'<div class="lt cl l1">Latest: {esc(DATE[n])} · {esc(n)}</div></div></a>')


def sub_card(s, u):
    """Stream page: a sub-stream as a fixed-size card that opens the sub-stream page."""
    names = names_of(u)
    n = newest(names)[0]
    pre = ''.join(f'<span id="{k}"></span>' for k, (_, subs) in LEGACY.items() if subs[0] == u['key'])
    return (f'<a class="tc" id="{u["key"]}" href="{sub_page(s, u)}">{pre}<h3 class="cl l2">{esc(u["title"])} <span class="cn">{len(names)}</span></h3>'
            f'<div class="a sm cl l5">{esc(u["a"])}</div><div class="bot"><div class="cnt cl l2">{count_line(names)}</div>'
            f'<div class="lt cl l1">Latest: {esc(DATE[n])} · {esc(n)}</div></div></a>')


def entry_card(n):
    """Sub-stream page: an entry as a fixed-size card (date, status, title, result) that opens its page."""
    st = STATUS[n]
    return (f'<a class="tc ec" href="{HREF[n]}" data-s="{st}" data-t="{" ".join(k for k in PLACE[n] if k)}">'
            f'<div class="ed"><span class="d">{esc(DATE[n])}</span> <span class="s" style="color:var(--{st})">● {LABEL[st]}</span></div>'
            f'<h3 class="cl l3">{esc(n)}</h3><div class="r cl l6">{esc(RESULT[n])}</div></a>')


def entry_grid(names):
    return '<div class="cards fixed">' + ''.join(entry_card(n) for n in newest(names)) + '</div>'


def nav_line(prev, nxt):
    """prev / nxt: (href, label) or None."""
    return ('<p class="nav">' + (f'<a href="{prev[0]}">← {esc(prev[1])}</a>' if prev else '<span></span>')
            + (f'<a href="{nxt[0]}">{esc(nxt[1])} →</a>' if nxt else '<span></span>') + '</p>')


def stream_page(i, s):
    prv, nxt = (STREAMS[i - 2] if i > 1 else None), (STREAMS[i] if i < len(STREAMS) else None)
    return (head(f'{esc(s["title"])} · Coherent beliefs and search') + f'<a class="back" href="index.html">← {esc(TITLE)}</a>'
            f'<h1><span class="no">{i}</span>{esc(s["title"])}</h1><div class="q">{esc(s["q"])}</div><p class="a">{esc(s["a"])}</p>'
            + sources(s) + '<div class="cards fixed">' + ''.join(sub_card(s, u) for u in s['subs'] if u['key'] not in ARCHIVE_SUBS)
            + '</div>' + archive_band([arch_card(sub_page(s, u), u['title'], names_of(u), key=u['key'])
                                       for u in s['subs'] if u['key'] in ARCHIVE_SUBS])
            + nav_line(prv and (f'stream-{prv["key"]}.html', f'{i - 1} · {prv["title"]}'),
                       nxt and (f'stream-{nxt["key"]}.html', f'{i + 1} · {nxt["title"]}'))
            + f'<a class="all" href="all.html">All {len(EXP)} experiments →</a></main>{STREAM_JS}')


def substream_page(i, s, u):
    names, subs = names_of(u), s['subs']
    j = subs.index(u)
    if u.get('groups'):  # large sub-streams keep their groups as sections
        body = ''.join(f'<section id="{g["key"]}" data-sec><h2 class="sh">{esc(g["title"])} <span class="cn">{len(g["names"])}</span></h2>'
                       f'{entry_grid(g["names"])}</section>' for g in u['groups'])
    else:
        body = entry_grid(names)
    return (head(f'{esc(u["title"])} · Coherent beliefs and search')
            + f'<a class="back" href="index.html">← {esc(TITLE)}</a><span class="back"> › </span>'
              f'<a class="back" href="stream-{s["key"]}.html">{i} · {esc(s["title"])}</a>'
            + f'<h1>{esc(u["title"])} <span class="cn">{len(names)}</span></h1><p class="a">{esc(u["a"])}</p>'
            + track_bar(sub_page(s, u)) + filt(names) + f'<section id="{u["key"]}" data-sec>{body}</section>'
            + nav_line(j > 0 and (sub_page(s, subs[j - 1]), subs[j - 1]['title']),
                       j + 1 < len(subs) and (sub_page(s, subs[j + 1]), subs[j + 1]['title']))
            + f'<a class="all" href="all.html">All {len(EXP)} experiments →</a></main>{LIST_JS}')


def redirect_page(title, target):
    return (f'<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(title)}</title><meta http-equiv="refresh" content="0; url={target}">'
            f'<script>location.replace({json.dumps(target)})</script><style>{CSS}</style><main>'
            f'<p>This list moved to <a href="{target}">{esc(title)}</a>.</p></main>')


RESULT = {e[1]: e[3] for e in EXP}
for i, s in enumerate(STREAMS, 1):
    (OUT / f'stream-{s["key"]}.html').write_text(stream_page(i, s))
    for u in s['subs']:
        (OUT / sub_page(s, u)).write_text(substream_page(i, s, u))
for k in OLD_TRACK_PAGES:  # old track pages: send old links to the sub-stream page that lists the entries now
    s, u = SUBS[k if k in SUBS else LEGACY[k][1][0]]
    (OUT / f'track-{k}.html').write_text(redirect_page(f'{s["title"]}: {u["title"]}', sub_page(s, u)) if k in SUBS
                                         else redirect_page(f'{s["title"]} ({LEGACY[k][0]})', f'stream-{s["key"]}.html#{k}'))
ARCH_CARDS = ([arch_card(f'stream-{s["key"]}.html', s['title'], names_of(s), i, s['key'])
               for i, s in enumerate(STREAMS, 1) if s['key'] in ARCHIVE_STREAMS]
              + [arch_card(sub_page(*SUBS[k]), SUBS[k][1]['title'], names_of(SUBS[k][1]), note=f'from {SUBS[k][0]["title"]}')
                 for k in ARCHIVE_SUBS])
home = (head(esc(TITLE)) + f'<h1>{esc(TITLE)}</h1><p class="r">{esc(SUBTITLE)}</p>'
        + f'<a class="dir" href="{SITE}direction/">Research direction →</a>'  # the separate research-direction note
        + '<div class="cards fixed">' + ''.join(stream_card(i, s) for i, s in enumerate(STREAMS, 1)
                                                if s['key'] not in ARCHIVE_STREAMS) + '</div>'
        + f'<a class="all" href="all.html">All {len(EXP)} experiments →</a>'
        + ('&nbsp;&nbsp;·&nbsp;&nbsp;<a class="all" href="reference.html">Reference →</a>' if REFS else '')
        + archive_band(ARCH_CARDS) + '</main>' + INDEX_JS)
if REFS:
    blocks = []
    for k, one, many in KINDS:
        rs = [r for r in REFS if r['kind'] == k]
        for r in rs:
            for f in r.get('files', []):
                if f.get('md'):  # note pages: render the linked Markdown source
                    render_md(f['md'])
                    continue
                st_ = f['href'].removesuffix('.html')
                if '/' not in st_ and (DOCS / f'{st_}.md').exists():
                    render_doc(st_)
            (OUT / f'ref-{k}-{r["slug"]}.html').write_text(ref_page(r, TITLES))
        if rs:
            blocks.append(f'<h2 style="margin-top:18px">{many}</h2><table>' + ''.join(
                f'<tr><td><div class="n"><a href="ref-{k}-{r["slug"]}.html">{esc(r["title"])}</a></div><div class="r">{esc(r["summary"])}</div></td>'
                f'<td class="d">{len([u for u in r.get("used_by", []) if u in TITLES])} exp.</td></tr>' for r in rs) + '</table>')
    (OUT / 'reference.html').write_text(head('Reference') + '<a class="back" href="index.html">← all streams</a><h1>Reference</h1>'
        '<p class="r">Data sets, methods, models and notes used across the experiments.</p>' + ''.join(blocks) + '</main>')
(OUT / 'index.html').write_text(home)
allp = (head(f'All experiments · Coherent beliefs and search') + f'<a class="back" href="index.html">← {esc(TITLE)}</a>'
        f'<h1>All {len(EXP)} experiments</h1><p class="r">By stream and sub-stream, newest first.</p>' + track_bar('all.html')
        + filt([e[1] for e in EXP]) + ''.join(
            stream_block(i, s, body=''.join(sub_section(u, 3) for u in s['subs']), sub_href='', cls='sb flat')
            for i, s in enumerate(STREAMS, 1)) + f'</main>{LIST_JS}')
(OUT / 'all.html').write_text(allp)
print('ok', len(EXP), 'pages', sum(1 for p in OUT.glob('exp-*.html')), 'streams', len(STREAMS), 'sub-streams', len(SUBS))
