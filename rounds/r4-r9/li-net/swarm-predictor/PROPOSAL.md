# Swarm predictor on ShipHunt: proposal

eurisko study, 2026-10-05. Written by an Opus subagent for eurisko from the design Kelvin and eurisko agreed that day. Status: proposal and pre-registration draft; nothing has been launched and no GPU has been used. Files: `li-net/swarm-predictor/`. Each number names the file it comes from.

## 1. TL;DR

- A small Qwen3 model writes many short reasoning chains about a ShipHunt board. Each chain ends with probabilities, each with a stated confidence, for a few statements about the board. A pool turns all claims into one coherent set of prices over legal boards. When the board is revealed, each chain is paid by settlement, and the chain sampler is trained by GRPO on that payment alone.
- Hypothesis: settlement-trained chains pooled this way price the board closer to exact Bayes than the mean or the majority vote of the same chains (section 2). If they do not, the market-flavoured part of the programme fails at its first step.
- Done today on CPU: `pool.py` and `settle.py`, checked on synthetic chains against the harness's exact Bayes oracle (RESULTS-SMOKE.md). The check changed one choice. The pool now pools each statement and then projects onto legal boards, because the one-step pool over boards double counts correlated statements. Where chains carry partly independent information, this pool's Brier against Bayes is 8-57% below the better of mean and vote (12 of 12 cells). It never loses to the mean when all chains read the same evidence (4 of 4). Settlement credit correlates with each chain's true informativeness in 18 of 18 cells (Spearman 0.11-0.41).
- Cost at the L4 ceiling rate: $21.7 for the full plan on Qwen3-0.6B, or $30.4 if decoding is twice as slow as assumed (`cost_estimate.json`). Cap $40.
- Decision needed: approve Stage 0 ($0.58: Gate 0 on 0.6B and 1.7B, plus a timing probe). Its result picks the model and re-prices the rest before any training.

## 2. Hypothesis, and what refutes it

Hypothesis: on ShipHunt, a swarm of short Qwen3 reasoning chains whose sampler is trained by GRPO on settlement alone, with its claims combined by the confidence-weighted log-linear pool projected onto legal boards, prices statements about the board closer to exact Bayes than the mean or the majority vote of the same number of chains of the same length.

Papers modified: self-consistency (Wang et al. 2023, arXiv 2203.11171) samples chains and takes a majority vote. The GFlowNet sampler of Hu et al. (2024, arXiv 2310.04363) trains the sampler to draw chains in proportion to their posterior, and it also aggregates by vote. Modification: replace the vote with a pool that adds evidence under the board's rules, and replace the sampler's objective with settlement paid against that pool.

Primary test. The cell is n = 16 chains per round, chains of at most L = 256 tokens, and R = 2 rounds, on 100 test boards per seed and 3 seeds. The metric is Brier against exact Bayes: the mean over boards of the mean over open statements of (price - Bayes)^2. The swarm is the S1 sampler with the D-fit pool (section 4). The comparison is the best of four baselines (mean and vote, over S0 chains and over S1 chains), chosen after the fact, which favours the baselines. The swarm wins only if all three hold:
1. Its Brier is at least 10% below that baseline.
2. The 95% CI of the paired difference is below 0 (two-level bootstrap: seeds, then boards).
3. The difference is negative in each of the 3 seeds.

Any other outcome refutes the hypothesis at this scale. The next work is then the Predictor without markets (Kelvin, 2026-10-05: "adopt markets only if the predictor exploration finds them", IDEAS-LI-NET.md, last section).

Compute is equal by construction. Mean, vote and every pool read the same chains, so the language-model compute is identical, and the pools cost milliseconds of CPU.

## 3. The protocol

Board. The harness preset `medium`: 5 x 4 cells, one ship of length 3 and one of length 2, and 6 probes that are each wrong with probability 0.15. Each board also gets 4 noiseless shots at random unprobed cells; these are the revealed cells. Exact Bayes is `ShipHunt.posterior()`, imported from the harness unchanged. `pool.Board` splits it over 510 labelled layouts, which makes statements about a named ship exact (RESULTS-SMOKE.md).

Statements. There are 20 of the form "cell X is occupied" and 8 of the form "the ship of length L lies horizontally in row r". Revealed cells are not statements, so each board has 24 open statements. Statements such as "plan k hits >= m in n shots" come with the decision side (section 10).

Chain. One sample from the sampler, with thinking mode off and at most L new tokens. The prompt gives the rules, the probe reports, the shots and the 6 statements chosen by Refocus. It averages 311 tokens (`cost_estimate.json`, which also holds an example). The chain argues briefly and ends with one line per statement, such as `S1: p=0.70 conf=0.8`. A chain with fewer than 6 parseable lines makes no claims.

Pool (`pool.py`). After every round, the pool prices all 28 statements from all claims so far. The house prior p0 knows only the rules (uniform over legal boards), so evidence reaches the pool only through claims. Each claim moves its statement's log-odds by b (logit q - logit p0), with stake b = kappa x conf / m^alpha, where m is the number of claims on that statement. This is the log-linear pool mu ∝ p0 x prod p_i^(beta_i/beta_0) of STUDY-LIR 3c. The prior is subtracted once per claim (the negative-weight prior copy of STUDY-LIR 3f), so the pool can add evidence: alpha = 0 adds, and alpha = 1 averages. kappa and alpha are fit on revealed boards only: by 2-fold cross-fitting over the test boards at evaluation, and by a refit every 10 training steps on the previous 80 boards.

The pooled table is then projected onto legal boards. The projection is the distribution over legal boards that minimises KL to p0 plus 20 x KL to each pooled price (`pool.project`, the fair-price projection of IDEAS-LI-NET section 1). Its marginals are the prices. This pool, D-fit, is arm (d). The one-step version over whole boards, W-fit, is reported but not primary, because it counts correlated statements as separate evidence (RESULTS-SMOKE.md, finding 1).

Settlement (`settle.py`). When the board is revealed, claim j of chain i is paid s_ij = (y_j - mu_j)(logit q_ij - logit p0_j). Here y_j is the truth and mu_j the D-fit price after the chain's round, the chain's own claim included. This is the chain's log-score gain over the prior minus the pool's expected gain, which is s_i = l_i(x) - E_mu[l_i] in STUDY-LIR 3c. A chain is paid s_i, the sum over its claims. A chain that agrees with the pool earns nothing in expectation; it earns by moving away from the price in the direction the board turns out.

GRPO uses the advantage (s_i - group mean) / group sd over the 8 chains of one board and round, as in Shao et al. (2024, arXiv 2402.03300). No environment reward reaches the sampler. One rule is not settlement: a chain with no parseable claim is paid the group minimum minus 0.1. Without it, silence (paid 0) would beat informative chains that were unlucky.

Refocus (`refocus.py`; hard-coded, not learned). Round 1 asks about the open cells next to any reported hit (probe or shot), padded to 12 statements with random open statements; each chain gets a random 6 of the 12. Rounds 2 and 3 ask about the 12 claimed statements with the largest residual: the sum over claims of conf x (logit q - logit mu)^2, divided by the sum of conf. This is the confidence-weighted spread of claims around the current price. Resolve is the pool recomputed from all claims so far, after each round.

Ablation: round-robin over the open statements within 2 hops (Manhattan distance) of a hit, 12 per round in a fixed order. Here "the query" is the set of hit cells.

Compute axes: n chains per round, L tokens per chain, R rounds. Comparisons across (n, L, R) use the token budget n x L x R.

## 4. Arms

| arm | chain sampler | why |
|---|---|---|
| S0 | the base model (LoRA adapter at zero), sampling seeds 0-2 | untrained chains; S0 with vote is self-consistency (Wang et al. 2023) |
| S1 | GRPO on settlement only, KL 0.04 to the base model as in Shao et al. (2024) | the design |
| S2 | the same GRPO, paid the chain's own log-score gain, the sum of log(q_ij(y_j) / p0_j(y_j)), with no pool | control: paid for being right, not for beating the price; the first arm cut if money runs short |
| S3 | trajectory balance toward R(chain) = p_base(chain given prompt) x prod_j q_j(y_j), with log Z estimated per group | GFlowNet sampler (Hu et al. 2024): chains drawn in proportion to their posterior given the revealed board; used with mean and vote |

S1-S3 train for 150 steps of 8 boards x 2 rounds x 8 chains, with L drawn from {64, 256} at each step, LoRA rank 8 on the frozen base (the harness `RoleModel` default) and 3 seeds each. Hu et al. use a modified subtrajectory balance. Here the reward exists only for a finished chain, so S3 uses the full-trajectory case of that loss.

Pools, all computed on the same chains: the prior (floor), mean (a), vote (b, the share of claims with p > 0.5), LL-fit (c), W-fit, D-fit (d, primary) and exact Bayes (ceiling). Also one chain of n x L tokens (n x L = 256, 1024, 4096) for S0 and S1 at R = 1.

Evaluation grid per checkpoint, on 100 test boards per seed. With L = 64: n = 1, 4, 16, 64 and R = 1, 2, 3. With L = 256: n = 1, 4, 16 at R = 1, 2, 3, plus n = 64 at R = 1. Round 1 draws 64 chains, and smaller n are nested subsets of them. Rounds 2 and 3 run separately for each n, because Refocus depends on the pool. Round-robin ablation: S1 at n = 16, L = 64 and 256, R = 3.

## 5. Pre-registered metrics

Primary: section 2.

Secondary predictions, each reported whatever the primary says:
1. Pool: within S0 and within S1, D-fit beats mean and vote at every n >= 4 (paired bootstrap over boards, per seed). The CPU check predicts this when chains carry partly independent information.
2. Training: S1 with D-fit beats S0 with D-fit at the primary cell.
3. Market against own score: S1 has more distinct claim sets and a lower D-fit Brier than S2. Settlement pays for information the pool lacks, while own score pushes every chain toward the same answer.
4. Refocus: R = 2 and 3 beat R = 1 at the same n and L, and the residual rule beats round-robin at R = 3.
5. Martingale: under D-fit, the change in each statement's price from round r to round r + 1 has mean 0 (the 95% CI covers 0), overall and in each of 5 bins of the round-r price. The report gives the mean signed drift and its CI.
6. Stale prices: on statements Refocus never asked about, D-fit's Brier against Bayes is below the prior's. Mean and vote leave those statements at the prior.

Reported without a test:
- log score against the revealed board, and its expected gap to Bayes, KL(Bayes_j || price_j);
- ECE of D-fit and of the mean (10 equal-mass bins, `metrics/calib.py: ece` in the harness);
- per board, the Spearman correlation of settlement with oracle informativeness (`settle.informativeness`), the share of chains paid more than 0, and payment quantiles;
- distinct claim sets per n (claims rounded to 0.1), and the entropy of claims per statement over training;
- kappa and alpha fitted per cell (alpha near 1 means the chains are redundant);
- parse rate, KL to the base, mean payment, and informativeness per claim at steps 0, 50, 100 and 150.

## 6. Stages and stop rules

Stage 0 costs $0.58 at the ceiling rate (`cost_estimate.json: stage0_both_models`). It runs S0 chains on 100 boards at n = 16, L = 64 and 256, R = 1, for Qwen3-0.6B and for Qwen3-1.7B, plus 5 timed GRPO steps each. Gate 0 passes for a model if at least 80% of chains give all 6 claims and the mean informativeness per claim is above 0 (95% CI). That means untrained claims beat the rules-only prior against exact Bayes. We use the smaller model that passes. If neither passes, we stop and report, because there is nothing to pool. (In E3, a 0.6B forecaster gave 0.650 to every plan; `lir-agent/CLAIMS.md`, row E3.)

Re-costing. Stage 0 measures the decoding time per step and the GRPO step time. If the re-costed plan exceeds $35, cuts apply in this order until it fits: drop S2; run the long-chain baseline for S1 only; evaluate 80 boards instead of 100; train 100 steps instead of 150. If it still exceeds $40, we stop and re-plan.

Stage 1 trains S1, S2 and S3 (3 seeds each). A small evaluation (n = 16, L = 128, R = 1, 100 boards) runs at steps 50 and 100. A run stops if its parse rate stays below 50%, or if all 8 chains of a group give identical claims (to 0.1) for 20 steps in a row (collapse). If at step 100 S1's informativeness per claim is not above its step-0 value (95% CI) in 2 of 3 seeds, S1 training stops there and the null is reported.

Stage 2 runs the full grid for every checkpoint and for S0. `analyze.py` is written and committed before Stage 2 starts.

Budget: GPU group `swarm-predictor`, `GPU_GROUP_BUDGET_USD=40`, `GPU_RUN_TIMEOUT=4h`, and at most 2 VMs live. Training and evaluation run in separate rentals when together they would exceed 4 h. The infra counts live reservations against the cap (`lir-agent/CLAIMS.md`, programme checks), so the cap binds before actual spend reaches $40. A run that takes more than 1.5 times its re-costed time stops.

## 7. Compute and cost

Measured, from harness files. For 32 episodes, a 0.6B RL step spends 6.3 s in no-grad passes and 6.5 s in training passes (`harness/runs/gpu-probe/q06-P/steps.jsonl`; the README's 12.74 s per step). The harness's own sequences, tokenized with Qwen3, have 1888 no-grad and 810 training tokens per episode. That gives 9581 tokens/s forward and 4006 tokens/s forward plus backward on one L4 for 0.6B, and 5330 and 2575 for 1.7B (`cost_estimate.json: rates`). These rates include the harness's Python overheads, so they overstate the cost.

Assumed, because the harness never decodes with a KV cache: 0.035 s per decoding step for a batch of up to 128 chains on 0.6B, and 0.045 s on 1.7B. Stage 0 measures this. The second and fifth rows show decoding twice as slow.

| plan (`cost_estimate.json: plans`) | GRPO step | training per arm-seed | evaluation per checkpoint | total L4-h | $ at $0.87/h |
|---|---|---|---|---|---|
| 0.6B, full plan | 36.7 s | 1.59 h | 0.62 h | 24.9 | 21.7 |
| 0.6B, decoding 2x slower | 47.9 s | 2.09 h | 0.93 h | 34.9 | 30.4 |
| 0.6B, decoding 2x slower, no S2, long chains for S1 only | 47.9 s | 2.09 h | 0.93 h | 24.6 | 21.4 |
| 1.7B, no S2, long chains for S1 only | 56.6 s | 2.45 h | 0.95 h | 26.2 | 22.8 |
| 1.7B, decoding 2x slower, no S2, long chains for S1 only, 80 boards | 71.0 s | 3.09 h | 1.08 h | 32.1 | 27.9 |

The full plan contains:
- Stage 0 for both model sizes;
- 3 trained arms x 3 seeds;
- 12 evaluations (9 checkpoints, plus S0 three times);
- 6 long-chain evaluations;
- the round-robin ablation;
- VM setup (pip took 162 s per VM, `harness/runs/gpu-probe/setup-times.txt`).

Every row fits under $40 and leaves $9.6-18.3 for reruns. With 2 VMs in parallel, the 0.6B plans take about 12-18 hours of wall time (24.9 and 34.9 L4-h, halved).

## 8. Code plan

New code lives in `li-net/swarm-predictor/`. The harness is imported through PYTHONPATH, as `harness/scripts/env.sh` sets it, and is never modified.

| file | state | contents |
|---|---|---|
| `pool.py` | written, CPU-checked | legal layouts, statements, rules-only prior, exact Bayes from `ShipHunt.posterior()`; the mean, vote, LL, W and D pools; `project()`; `fit_stakes()` |
| `settle.py` | written, CPU-checked | settlement per claim and per chain, raw-score and stake-weighted variants, exact leave-one-out, oracle informativeness, the stake-gradient check, Spearman |
| `smoke_pool.py` | written, run | the CPU check (RESULTS-SMOKE.md) |
| `estimate_cost.py` | written, run | section 7 |
| `refocus.py` | to write | round-1 cells next to hits, residual targets, round-robin ablation, a random 6 of 12 per chain |
| `run_swarm.py` | to write | loads `harness/model/roles.py: RoleModel` with one adapter; samples chains with `peft.generate` (KV cache on, LoRA merged while sampling); parses claims, pools, settles; GRPO, own-score and trajectory-balance losses from `RoleModel.logits(...)` on chain tokens; boards from `harness/env/shiphunt.py`; CLI `--sampler S0/S1/S2/S3 --model --seed --steps --eval-grid` |
| `analyze.py` | to write before Stage 2 | every metric of section 5 from the logs, with `harness/metrics/calib.py` (ece, boot_ci), and the decision rule of section 2 |

Open questions for Stage 0: whether `RoleModel`'s forward hooks and a LoRA merge work under `generate`, and what the parse rate is at L = 64.

## 9. Seeds and logging

Three seeds per trained arm (training seeds 0, 1, 2); S0 uses sampling seeds 0, 1, 2. Each seed gets 100 test boards from its own seed range, never used in training.

Logs are JSONL in `runs/<arm>-s<seed>/`.
- Per board: the board seed, the evidence, the true layout, and exact Bayes for every statement.
- Per round: the Refocus targets; for every chain, a prompt hash, the full text, the token count, the parsed claims (statement, p, conf) and the parse status; the prices of every pool; the fitted kappa and alpha.
- Per claim and per chain: the settlement, its raw-score and stake-weighted variants, exact leave-one-out, informativeness, and the GRPO advantage.
- Per training step: mean and sd of payments, KL, parse rate, distinct claim sets, wall time per phase, and decoding tokens per second.
- Per run: the model id, hashes of the code files, the harness commit (`git -C harness rev-parse HEAD`), and the GPU-LOG rows.

## 10. Later: the decision side (out of scope)

Plans become shot sequences, priced by the same pool through statements such as "plan k hits >= m in n shots". A softmax chooser over plan prices is the Faith step: decision making with a "faith" soft constraint in LIR (Prop. 7, STUDY-LIR section 2). A proposer is paid on acceptance. It bets that its proposal will be chosen and never trades the conditional p(outcome | plan), which only traders price (Kelvin, 2026-10-05, IDEAS-LI-NET.md). An executor acts as interpreter: it fires the chosen plan's shots and reports the outcomes. Conditional settlement by induction is not part of this experiment.

## 11. PREREG deviations log

Append only, timestamped.

- 2026-10-05T01:47Z (before Stage 0). Approved by Kelvin about 01:40Z with a $40 GPU cap; coordinator rules: at most 2 VMs, run ids `swarm-*`, a launch check at $38. Each deviation below was fixed before any GPU use.
- 2026-10-05T01:47Z. Chain length. Six answer lines take 89 Qwen3 tokens, so a 64-token chain could never finish its claims. L is now the reasoning budget, stated in the prompt as about 0.75 x L words, and generation is capped at L + 16 tokens per requested statement. Compute comparisons use the generated token counts that are logged.
- 2026-10-05T01:47Z. Sampling: temperature 0.7 (Qwen3's non-thinking recommendation) for every arm, with no top-k or top-p truncation. The GRPO, KL and trajectory-balance log-probs use the same tempered distribution. Chains are sampled from a merged bf16 copy of base plus LoRA, refreshed before each sampling phase.
- 2026-10-05T01:47Z. Optimiser (not fixed in section 4): AdamW, learning rate 1e-4, weight decay 0, gradient clip 1.0, 4 sequences per backward chunk; LoRA rank 8, alpha 16 (harness default).
- 2026-10-05T01:47Z. Refocus in rounds 2 and 3 ranks residuals at D-avg prices (no learned stakes), so generation never uses outcomes. Reported prices use cross-fitted D-fit, computed after the runs by `analyze.py`.
- 2026-10-05T01:47Z. Diversity: every chain of a group gets a different 6-of-12 subset, so "distinct claim sets" are always distinct. Measured instead: distinct rounded p values per statement among the chains that claimed it. Collapse means every multi-claimed statement in every group has one rounded p. Both stop rules ("parse rate below 50%", collapse) need 20 consecutive steps.
- 2026-10-05T01:47Z. The S1 step-100 rule is applied per seed: a seed stops at step 100 if the lower 95% bound of its step-100 informativeness per claim is not above its step-0 mean. The null is reported if 2 of 3 seeds stop.
- 2026-10-05T01:47Z. The long-chain baseline asks one chain about all 12 round-1 targets, with reasoning budgets 256, 1024 and 4096 and a 192-token answer allowance. Gate 0 pools both L values.
- 2026-10-05T01:47Z. Storage: the eurisko-study tree is at 4.6 GB, so only compact claim records, payments and timings go to `runs/` on /srv. Chain texts and LoRA checkpoints stay in /tmp/swarm-predictor on General.
- 2026-10-05T01:57Z. Prompt. A CPU check with 8 Qwen3-0.6B chains showed that the prompt's one example line (`S1: p=0.70 conf=0.8`) gets copied: most claims were 0.7, and informativeness per claim was -0.51 (`/tmp/swarm-test/q06-gate0/gate0.json`). A placeholder (`p=P conf=C`) got copied literally (`/tmp/swarm-test/q06-gate0-p2/`). Two prompt variants are now fixed. A keeps the numeric example; B adds an `Answer:` header and two example lines with different values. Both say "Reason briefly, in at most W words" and state the rules' base rate (5 of the 20 cells hold a ship). Stage 0 picks a variant per model on 30 separate dev boards (parse rate >= 80% first, then informativeness per claim). Gate 0 is then judged on its own 100 boards. The parser now accepts a stray letter or bold markup before a number.
- 2026-10-05T02:31Z. Stage 0 split in two. On the first rental, decoding took 0.176 s per step on 0.6B (`runs/gate0-q06/gate0.json`, cell L64), 5 times the assumed 0.035 s. At that speed the 1.7B gate could not finish inside that rental, so its jobs were stopped once the 0.6B gate cells had synced. A second rental (`swarm-stage0b`) first benchmarks generation on 1.7B: dynamic cache, static cache, and static cache with torch.compile, at batch 128 and 32 (`--mode genbench`). It then runs the full 1.7B Gate 0 with the fastest setting (`--gen-impl auto`). All settings sample the same distribution.
- 2026-10-05T03:07Z. Gate 0 outcome on the evidence so far. 0.6B fails: pooled over both lengths, 82% of chains parse and informativeness per claim is -0.384 (`runs/gate0-q06/gate0.json`; `stage0_summary.json`). 1.7B: on the dev boards neither prompt reaches 80% parsing (A 75%, B 75%), so A was chosen by parse rate, as the rule says. Its L = 64 gate cell has informativeness -0.128, 95% CI [-0.135, -0.121] (`runs/gate0-q17/gate0.json`). The L = 256 cell is finishing so that the pooled gate is complete. After it, the Stage 0b jobs are stopped and the 1.7B timing steps are skipped: if neither model passes, there is no main run to cost.
- 2026-10-05T03:07Z. Generation code added during Stage 0 and not used on GPU: a cache option that writes into preallocated buffers (`--gen-impl prealloc`) and a per-job speed check (`auto-bench`). On CPU with Qwen3-0.6B it gives greedy outputs identical to the default cache. It is not timed on GPU.

D-fit replaced W-fit as the primary pool during the CPU check, before this pre-registration; RESULTS-SMOKE.md records why. That change is not a deviation.

## 12. Sources

- Wang, Wei, Schuurmans, Le, Chi, Narang, Chowdhery, Zhou. Self-Consistency Improves Chain of Thought Reasoning in Language Models. ICLR 2023. https://arxiv.org/abs/2203.11171 (fetched 2026-10-05)
- Hu et al. Amortizing intractable inference in large language models. ICLR 2024. https://arxiv.org/abs/2310.04363 and, for the method details, https://arxiv.org/html/2310.04363 (both fetched 2026-10-05)
- Shao et al. DeepSeekMath, which introduces GRPO (group-normalised advantage, KL coefficient 0.04, 64 samples per question). https://arxiv.org/abs/2402.03300 and https://arxiv.org/html/2402.03300 (both fetched 2026-10-05)
- Local files: `li-net/STUDY-LIR.md` (3c, 3f), `li-net/IDEAS-LI-NET.md`, `lir-agent/harness/README.md`, `lir-agent/CLAIMS.md`, and this folder's RESULTS-SMOKE.md and `cost_estimate.json`.
