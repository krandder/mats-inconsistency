# PREREG: ShipHunt small-network data-efficiency loop (smallnet)

2026-10-10, Opus 5.5 subagent for eurisko. Spec: `/home/agent/workspaces/mats/outbox/shiphunt-smallnet-spec-2026-10-10.md`; review and fixes: `SPEC-REVIEW.md`.
Harness: `lir-agent/harness` at commit 47077e771a1045aa6307a6c0bd0ec8efd0c3c7de (clean), env imported unchanged.
Code: `li-net/smallnet/{data,bninfer,netdef,smallnet,analyze,mkjobs}.py`, GPU scripts in `gpu/`. Each rental runs a frozen copy (`/tmp/smallnet/src-<run>-<epoch>`).

## Question and claims under test
1. Spec claim (pre-registered decision rule, verbatim): "Structured training is more data-efficient if k=1 or k=2 reaches k=0's 10k-board single-cell KL with <= 1/3 of the boards (CI over seeds), or beats k=0 at equal boards across the sweep."
2. Loop claim (Kelvin's success criteria): a real-world-valid training regime that is (a) more data-efficient than both plain arms (full map `k0`, cell by cell `k0cell`) at 100-1k boards, and (b) at least as accurate at 10k boards, even at more compute per board.

## Task and data (fixed)
- ShipHunt preset `hard` (5x5, ships 3,2,2, 6 probes, eps 0.2). One episode per board: the true board plus ONE observed probe trajectory. Stages: prefixes after 0, 2, 4, 6 probes.
- Training boards: seed s, board i = env seed 1,000,000 + 100,000 s + i; N in {100, 300, 1000, 3000, 10000} are nested prefixes, so the arms of one seed see the same boards (paired).
- VALIDATION: 3000 boards (env seeds 7,000,000 + i), the same for every arm and N. Used only for early stopping (outcome log-loss, first 1000 boards) and for the loop's selection (single-cell KL, all 3000).
- TEST: 5000 boards (env seeds 9,000,000 + i). Not read by any loop run. Evaluated once, at the end, for the champion vs the plain arms (`--test` runs below).
- Exact posterior marginals (harness enumeration over the 11088 placements): computed only for validation and test boards, and for training boards only for the evaluation-only reference arm `refpost`.

## Network and training (fixed across arms)
- `netdef.Net`: pre-LN transformer over the 25 cell tokens, d = 128, 4 layers, 4 heads, FF 256, dropout 0.1, learned 2-D relative attention bias; 577,433 parameters including every head (all heads are built in every arm, so the init is identical per seed). Inputs per cell: evidence state, reveal state (true value revealed: only arms that condition on true cells), query flag (only `k0cell`).
- AdamW, betas (0.9, 0.98), weight decay 0.05, warmup 100 steps, grad clip 1, batch 64 boards (all stages of each board), random D4 symmetry per board per step (every arm).
- Per-N settings, calibrated on k0 (baseline-favouring; runs in `/tmp/smallnet/runs/{smoke,cal,cal2}`): N=100: lr 3e-4, EMA 0.95, eval every 10 steps, max 2000; N=300: lr 1e-3, EMA 0.95, every 20, max 4000; N=1000: lr 1e-3, EMA 0.99, every 50, max 10000; N=3000: lr 1e-3, EMA 0.99, every 100, max 20000; N=10000: lr 1e-3, EMA 0.99, every 200, max 40000. Evaluation and checkpoints use the EMA weights.
- Early stopping: the per-cell log-loss of the arm's single-cell forecasts on the first 1000 validation boards' TRUE cells (outcome-only; never the posterior), evaluated at the per-N cadence above; keep the best checkpoint; stop after 25 evaluations without improvement, at max steps, or 7-12 min before the rental ends (`SMALLNET_DEADLINE`; recorded as `stop_reason`).

## Arms (code: `smallnet.ARMS`; every config is BASE plus these keys)
| tag | config | what it trains | real-world-valid |
|---|---|---|---|
| E01-k0 | `k0` | full map: 25 cell probabilities in one pass, per-cell log score vs the true board | yes |
| E02-k0cell | `k0cell` | cell by cell: one question per pass (query flag), 25 passes per board-stage | yes |
| E03-k1 | `k1` | random order; each cell picks 1 earlier parent (soft over all earlier cells, expected loss + 0.1 x mean), 2-entry table; loss = log-lik of the true board under the net; readout = exact net marginals averaged over 4 fixed orders | yes |
| E04-k2 | `k2` | the same with 2 parents (pairs among earlier cells), 4-entry tables | yes |
| E05-k2fixed | `k2fixed` | k = 2 with parents fixed to the 2 nearest earlier cells (structure-learning control) | yes |
| E06-masked | `masked` | k0 + auxiliary: reveal each true cell with prob r ~ U(0,1), predict the hidden cells | yes |
| E07-td | `td` (lambda 0) | k0 with stage-t target = stop-grad stage-(t+1) forecast, the true board at stage 6 | yes |
| E08-mart | `mart` (beta 100) | k0 + label-free drift penalty (batch-mean q(t+1) - q(t) per stage and forecast bin) | yes |
| E09-chain | `chain` | full true-value chain (orderless autoregressive objective); readout = no cells revealed | yes |
| E10-refpost | `refpost` | per-cell cross-entropy to the EXACT posterior: capacity reference, never a candidate | NO |
| E11-k2marg | `k2marg` | k0 + the k = 2 net loss on the same network; readout = single-cell head | yes |
| E12-rbtd | `rbtd` (4 levels, lambda 0) | invention, reveal-chain TD: levels of revealed true cells (random order, random cut points); each level's forecast is trained toward the next level's stop-grad forecast, the top level toward the true board; readout = level 0 (nothing revealed) | yes |
| E13-allprefix | `allprefix` | k0 on all 7 prefixes (0-6 probes) of the observed trajectory | yes |
| E14-perm | `perm` | k0 on the 7 prefixes of a random reordering of the 6 observed probes (no new evidence) | yes, if evidence items are exchangeable (true here; not for time-ordered news) |
Later waves add experiments chosen by the selection rule below; each is logged in `LOOP.md` with its hypothesis before it runs.

## Metrics
- Primary: single-cell KL(P* || q), Bernoulli per cell, q clipped to [1e-6, 1 - 1e-6], averaged over 25 cells x 4 stages x boards (also per stage). Single-cell q from each arm's own readout.
- Secondary: per-cell outcome log-loss; ECE (15 equal-mass bins, vs true cells); joint log-likelihood of the true board (k0-type arms: product of single cells; net arms: mixture of the 4 hardened nets; reveal arms: autoregressive in one random order on 500 boards); compute = analytic FLOPs (forward + backward = 3 x forward) to the best checkpoint, and per training board.

## Selection rule of the loop (VALIDATION only)
- Score of an experiment: S_small = mean over N in {100, 300, 1000} of the seed-paired mean log(KL_arm / KL_E01-k0). Requires 3 seeds at each of those N. Negative = more data-efficient than k0. S_all (all 5 N) is the tie-breaker and the "eventually as accurate" check.
- A candidate must be real-world-valid without conditions and have the full sweep (5 N x 3 seeds). refpost is never a candidate. Arms that reorder the observed probes (`stages=perm`) are valid only when evidence items are exchangeable. That holds in ShipHunt but not for time-ordered real-world evidence, so they are reported as a separate track and are NOT eligible as champion (added 16:50Z, before any wave-1 or perm result came home).
- Strict plain baseline: the best plain configuration (k0 or k0cell, any regularisation/lr/EMA tried in the loop) by S_small. A champion must beat it too, so a win cannot be a regularisation effect.
- Champion: the candidate with the lowest S_small whose S_all < 0 (no loss at large N on average). If none has S_all < 0, the lowest S_small, flagged.

## Final TEST protocol (run once, after the loop)
- Champion, E01-k0, E02-k0cell and the strict plain baseline, seeds 0-4, all 5 N, `--test` (retrained; TEST scored once per run).
- Report on TEST: KL per N (mean, 95% t-interval over seeds), paired differences vs each plain arm, and the spec rule for the champion: (a) boards needed to reach k0's 10k KL as a fraction of 10k (log-linear interpolation, bootstrap over seeds, pass if the 97.5% bound <= 1/3), (b) at every N the paired difference < 0 with a 95% interval excluding 0.
- The spec's own rule for k = 1 and k = 2 is evaluated on VALIDATION with the same code (`analyze.py --decision E03-k1,E04-k2`), because TEST is reserved for the champion.

## Budget and stop conditions
- Spend cap: smallnet group $200 (launch refuses at spent + reserve > $190; `guard-smallnet.sh` deletes VMs older than 250 min and the youngest VM when audited + running > $190). Project cap: about $104 headroom (SPEC-REVIEW D.15).
- Pricing (smoke rental `smallnet-smoke`, 0.41 h, $0.36 at the ceiling rate; GPU-LOG.md). Seconds per training step on one L4 shared by 6 runs (`runs/smoke/time-*.json`): k0 0.025-0.035, refpost 0.024, td 0.030, mart 0.031, chain 0.035, perm 0.047, k1 0.054, allprefix 0.058, masked 0.074, rbtd 0.13, k2marg 0.21, k2 0.22, k2fixed 0.25, k0cell 0.89 with 25 questions per step (0.25 with 5). Early stopping lands at about 500 / 1000 / 3000 / 7000 / 13000 steps for N = 100 / 300 / 1k / 3k / 10k.
- Wave 1 = 16 experiments x 5 N x 3 seeds = 240 runs, about 35 process-hours, 6 runs per L4: 8 L4 rentals of about 1-1.5 h each, about $8-10 at the $0.87/h ceiling (reservations $23). The whole loop (3 waves, >= 25 experiments) plus the final 5-seed TEST runs: about $25-40 at the ceiling; spot is lower. Wall-clock is bounded by the slowest single run (k0cell or k2 at N=10k, about 1 h), not by money.
- Stop conditions: no launch if smallnet spend + reserve > $190 (launch.sh) or project spend + reserve > $200 (gpu-up.sh); a run stops at its rental deadline and is reported with `stop_reason`; zero smallnet VMs verified at the end.

## Deviations log (append only, UTC)
- 2026-10-10T15:52Z: guard-smallnet.sh date regex fixed and the guard restarted (SPEC-REVIEW D.15).
- 2026-10-10T16:09Z: added `gpu/audit-failed.sh` + `gpu/launch-retry.sh`: a spot stockout (all 3 L4 zones, 16:08Z) left its $1.42 reservation counted in spend_usd; a COST-AUDIT cost_usd=0 line now releases it after gcloud confirms no instance exists.
- 2026-10-10T16:20Z: cell-by-cell (`k0cell`) asks 5 random questions per board-stage per training step instead of all 25 (`cell_q=5`): with 25 it runs at 0.89 s/step on a shared L4 (smoke `time-k0cell`), which does not fit a 235-min rental at N=10k. Every cell is asked equally often in expectation; evaluation asks all 25.
- 2026-10-10T16:25Z: smoke shows lr 1e-3 with warmup 200 overfits within ~100 steps at N=100 (val KL 0.010 at step 100, 0.75 at step 1300; `runs/smoke/conv-k0-n100-s0.json`) and gives noisy validation curves at N=1000. lr, EMA and evaluation cadence are calibrated on k0 and masked before wave 1 (`jobs/cal.txt`), choosing the setting that is best for k0 (baseline-favouring).
- 2026-10-10T16:36Z: per-N lr / EMA / cadence fixed from the calibration (Network and training section). Evidence: N=100 k0 lr 1e-4..3e-4 6.6-7.0 vs lr 1e-3 7.2-7.8 (x1e-3 KL, 2 seeds, `runs/cal2`); N=300 lr 1e-3 4.5-4.6 vs 5.9-6.2; N=10000 lr 1e-3 + EMA 0.99 0.39 vs no EMA 0.51 vs lr 3e-4 0.67 (`runs/cal`). Patience 25 evaluations. Warmup 200 -> 100.
- 2026-10-10T16:37Z: runs stop training at the rental deadline (SMALLNET_DEADLINE from remote.sh) and evaluate their best checkpoint, so no result is lost to a VM timeout.
- 2026-10-10T16:50Z: champion eligibility excludes reordered-evidence arms (perm), which are reported separately; written before any wave-1 result came home.
- 2026-10-10T16:45Z: wave 2a (E17-E25, rbtd neighbourhood and combinations) launched in parallel with wave 1 on the strength of the 1-seed smoke signal, to save wall-clock (LOOP.md).
- 2026-10-10T17:01Z: two wave-2a spot VMs (w2a-vm1, vm4) were preempted about 10 min into their jobs (ssh timeout, remote rc=255). Their 45 jobs were requeued unchanged (`missing.py`, jobs/W2a-r1-vm*.txt). Preempted VMs leave no DELETE line, so `gpu/audit-preempted.sh` appends a COST-AUDIT at ceiling rate x (create to run-end).
- 2026-10-10T17:08Z: wave-1 validation shows the outcome-based early-stopping signal on 1000 boards is noise-limited at N >= 3000 (LOOP.md, observation 17:08Z). Waves 1-2 keep the protocol; the final TEST runs use es_boards = 3000 and patience 40 at N >= 3000 for every arm (champion and plain arms alike). Decided before any TEST evaluation.
- 2026-10-10T17:20Z: 10 of 14 spot L4 rentals were preempted within 35 min (requeued automatically by `gpu/requeue.py`; at most 2 tries per job). New rentals are on-demand (`GPU_PROVISIONING=STANDARD`, opt-in env added to logical-soup/infra/gpu-up.sh, default unchanged). Cost accounting is unchanged: GPU-LOG already charges the on-demand ceiling rate.
- 2026-10-10T17:50Z: editing gpu/launch.sh at 17:20Z broke the launch.sh processes already running (bash reads scripts incrementally: "unexpected EOF" at the last line), so they wrote no END line. Their VMs were still deleted by gpu-run.sh's trap. The requeue daemon then missed 6 preempted rentals (rq2-rq5, rq8, rq9) for about 25 min. Fixed: the daemon treats a rental as ended when its launch-retry and launch.sh processes are gone. The 6 were requeued on-demand at 17:50Z (rq11-rq16). Lesson: never edit a launcher script while instances of it run; copy, edit, rename.
- 2026-10-10T19:10Z: champion E17-rbtd8 selected by the rule with 14/15 loop runs (the missing N=10k seed cannot change S_small, nor make S_all >= 0). Final part 2 launched (LOOP.md, Selection).
- 2026-10-10T20:35Z: final runs complete; TEST read once (`final_analysis.py`). Zero smallnet VMs (20:33Z, checked twice). GPU total $34.83 at the ceiling rate (`gpu/cost.py`, GPU-LOG.md). RESULTS.md written and rendered.
