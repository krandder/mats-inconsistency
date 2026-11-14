# PREREG E5 (2026-10-03 ~08:50Z, Opus 5.5 subagent for eurisko; hawk file: ../../hawks/E5.md, verdict RESHAPE, all three reshape items adopted)

Harness: ../../harness (not a git repo: no commit hash; code imported unmodified; staged copy rsynced to each VM).
Experiment code: e5c.py, chain.sh, e5a_metrics.py, e5a_format.py, remote.sh, driver.sh, analyze.py (this dir).

## Papers modified
- E5c: Grimm, Barreto, Singh, Silver 2020, Value Equivalence, https://arxiv.org/abs/2011.03506, and Tang, Nguyen, Ha 2020,
  https://arxiv.org/abs/2003.08165 (task-blind perception costs only when capacity/relevance binds). Abstracts verified by the hawk (hawks/E5.md section 3), 2026-10-03.
- E5a: Paster, McIlraith, Ba 2022, "You Can't Count on Luck", https://arxiv.org/abs/2205.15967 (goal/outcome conditioning -> luck-driven optimism; fix = information constraint); here the constraint is deleting the goal from the belief pass.
Modification (one sentence): E5c = a two-board text POMDP where a board-naming goal decides which half of a long interleaved context matters, comparing goal-visible vs goal-deleted belief passes; E5a = the harness's fused B0 / dose GRPO on ShipHunt with goal drawn per episode, and goal deleted / visible / shuffled in the belief pass.

## Claims under test
- E5c (Kelvin's objection): goal-blind content (goal deleted, relevance supplied only by a goal-free board query chosen by the goal-seeing action pass) costs no perception vs goal-visible. Refuted if Brier-vs-Bayes(Q) - Brier-vs-Bayes(V) > 1 seed-sd (pooled sd of the two arms) and > 0.002 absolute on the goal board.
- E5a: under reward pressure, a goal-visible belief pass acquires a signed goal-contingent bias that deletion prevents. Deletion is "kept" only if b_G(visible) - b_G(shuffled) > 0 beyond seed sd (change from pre-RL, approval reward). Sharpening (overconfidence) is reported separately; it is predicted NOT to be removed by deletion (hawk 1d).

## Arms
E5c (no RL; Qwen3-0.6B bf16, LoRA r8 belief adapter, truth-only expected-Brier loss; 150 steps x 16 episodes, lr 2e-4; two `medium` boards 5x4, ships 3+2, eps 0.15, 48 probes per board with replacement, interleaved, ~860 tokens; eval 256 fixed episodes at steps 0,50,100,150; seeds 0,1,2):
  V goal visible + both boards reported | Q goal deleted, board query chosen by the action adapter (trained 30 steps on goal->board), only that board reported (trained on one query per board, so label count matches) | N goal deleted, no query, both boards | S goal names the wrong board, both boards.
  `python3 e5c.py --arm {V,Q,N,S} --seed {0,1,2} --device cuda --dtype bfloat16 --steps 150 --batch 16 --eval-n 256 --out /root/runs/E5/E5c/{arm}-s{seed}`
E5a (harness protocol.train; common flags in chain.sh: Qwen3-0.6B bf16, easy, --goal mixed, SFT 60 x 16, RL 64 steps x 32 episodes (4 groups x 8), eval 128 at pre_rl / rl32 / post_rl, no syco eval, no Faith swap):
  arm in {B0 (fused, alpha 1), D1 (= --arm P --alpha 1.0, separate adapters, routing off)} x goal_mode {deleted, visible, shuffled} x reward {approval, random} x seeds {0,1,2} = 36 RL runs from 18 warm starts (approval and random share the warm start of their arm x goal_mode x seed).
  Protected implementation check: --arm P, goal_mode deleted and visible, approval, seed 0.
  Fused baseline B0 is arm B0 above.
E5b (only if E5a is positive): goal-neutral claim set. Regardless, the selection-leak probes are computed from every E5a run.

## Metrics (file patterns)
- E5c: runs/E5c/{arm}-s{seed}/eval.jsonl: brier_vs_bayes, ece_vs_bayes, brier_vs_truth, ece_vs_truth on goal-board cells; swapgoal_kl (format control, V and S).
- E5a: runs/E5a/{arm}-{gm}-{reward}-s{seed}/e5_metrics.json: b_G (balanced: 0.5*(mean(p-p_bayes | hit) - mean(p-p_bayes | avoid)) over all cells; equals the hawk's mean (p-p_bayes)*s at 50/50), overconf (slope of logit E[q] on logit p_bayes minus 1), each as post_rl minus pre_rl.
- Format control: runs/E5a/{arm}-{visible,shuffled}-s{seed}.format.json: KL(true goal text vs swapped goal text) and Brier, for base, warm (pre-RL) and each final.
- E5b probes: same e5_metrics.json: probe_sit_plan, probe_fc, probe_belief (5-fold CV logistic, chance = majority rate).
- Every run: summary.json mask_violations must be 0.

## Comparison rule
Matched steps (all arms identical step counts/batches; E5a arms also reported with their reward so matched-reward reading is possible). Mean +- sd over 3 seeds; "beyond sd" = difference of means > pooled seed sd.

## Predictions
- E5c: Q ~= V ~= S within seed sd; N possibly worse if the double-length report binds. If V beats Q beyond sd -> goal-blindness has a real perceptual cost (finding against the design).
- E5a: deleted b_G change ~ 0 by construction; visible b_G change > shuffled under approval if goal-contingent wishful thinking exists; overconf change under approval > random-reward control in all goal modes (deletion leaves sharpening). Protected deleted vs visible gap ~ 0.
- E5b: probe_sit_plan and probe_fc far above chance in every arm (selection leak, hawk 1c); probe_belief at chance when deleted.
Refutation of "goal-blindness costs nothing": E5c V-Q gap beyond sd. Refutation of "deletion removes goal-contingent bias that would otherwise appear": b_G(visible)-b_G(shuffled) <= sd (then deletion protects nothing measurable at this scale; reported as such).

## Stop / kill
Hawk kill condition (no two-board env and no report budget) does not apply: the two-board env is built (e5c.py). Each VM remote timeout 105 min (< 2 h). No launch if `spend_usd lir-agent` + reservations would pass $50. Report budget K<n_cells is NOT built (deviation: hawk allowed either).

## Budget
Wave 1: 2 E5c VMs + 4 E5a VMs; wave 2: 5 E5a VMs; ~1.6 h each at $0.87/h ceiling -> ~$15 ceiling worst case, ~$11 expected.
Analysis script: analyze.py (written before results land).

## Deviations log (append only)
- 08:50Z seeds = 3 (hawk allows 3). RL 64 steps (reference 200) for budget; eval_n 128.
- 08:40Z wave 1: two concurrent 0.6B processes at micro-batch 8 OOM on the 22 GB L4 (logs: runs/ after sync; VMs lir-e5-a2/a3 chains failed, c1/c2 E5c V-s0,S-s0 failed). c1/c2 killed by me; all remaining runs relaunched with harness --micro-batch 4 and e5c --mb 4 (gradient accumulation chunking only; same effective batch). Chains on lir-e5-a1 (B0 deleted s0,s1) and lir-e5-a4 (D1 visible/shuffled s0) keep micro-batch 8.
- 08:40Z run ids are lir-e5-<vm> (gpu-run requires lowercase; "lir-E5" invalid). Launches are serialised by the lir-agent group reservation cap ($50 incl. 6 h reservations), queue script /tmp/lir-agent/E5/queue.sh.
- 09:20Z E5c protocol change BEFORE any arm comparison was looked at: with the planned format (report cells " C=" under a "board b:" header, lr 2e-4, 150 steps) the belief adapter learned no perception (N-s0 pilot and c3a jobs: Brier vs truth 0.187 at step 50-100 = the base-rate predictor 0.1875; pilot lr 1e-3 also flat at step 96, runs/pilot/N-lr1e-3 if synced). The harness's own easy-preset belief adapter is also at base rate after 60 SFT steps (runs/E5a/B0-deleted-s0-warm/eval.jsonl: belief_brier 0.138 vs base rate 0.139). Changed to report cells written " 1C=" (same token pattern as the evidence "probe 1C:"; no information added: the board header was already there), lr 5e-4, 300 steps, evals at 0/100/200/300; pilot runs/pilot/N-pref-lr5e-4 reached Brier vs truth 0.087 at step 200 (Bayes floor 0.033). Primary E5c = runs/E5c2/; the old-format 150-step jobs (runs/E5c/) are kept as the "no perception learned" condition.
- 09:20Z one process per GPU (2 concurrent 0.6B jobs were ~3-5x slower per step and OOM-prone); each VM runs its jobs sequentially.
- 09:50Z a chain (warm + 2 RL runs + evals) takes ~60 min alone on one L4 (lir-e5-b1 job log); two chains per VM overrun the 105-min remote timeout, so the second chain on lir-e5-b1 (B0 visible s1) and lir-e5-b8 (D1 shuffled s0) may be cut. Remaining work is one chain (or 4 E5c jobs) per VM, queue /tmp/lir-agent/E5/queue.sh order: E5c2 seeds 0,1,2; B0 shuffled s0,s1; B0 visible s2, shuffled s2, deleted s2; D1 deleted s0; P check; D1 rest. Cut runs are re-queued.
- 11:10Z E5c second protocol change, again before comparing arms at a perceiving state: in E5c2 (each arm trained alone from the base, 300 steps) arm V seed 0 never left the base rate (runs/E5c2/V-s0/train.jsonl exp_brier 0.187-0.188 for 300 steps) while the N pilot (same seed, same episodes) learned (runs/pilot/N-pref-lr5e-4). Perception emerges as a seed/format-dependent jump, so per-arm-from-scratch comparisons measure optimisation luck. Primary E5c is now E5c3: per seed, one shared truth-only pre-training of the belief adapter on a format mixture (episode seed mod 4 -> V/Q/N/S format; 400 steps, lr 5e-4), then each arm fine-tunes from that same state for 100 steps in its own format (evals at 0/50/100 of the fine-tune; step 0 = format shift at a perceiving init). E5c2 (from scratch) is kept and reported as a secondary result (learnability per format). Block-Attention (https://arxiv.org/abs/2409.15355) is the precedent: compare masks/formats after adaptation, not from scratch.
- 11:15Z killed lir-e5-c3b after V-s0 (300 steps) and Q-s0 (141 steps) stayed at the base rate, to free the GPU slot for E5c3; N-s0/S-s0 of E5c2 not run.
- 11:45Z lir-e5-c5 (E5c3 seed 1) lost to spot preemption (VM vanished with no DELETE line, GPU-LOG.md); re-queued.
- 11:50Z correction to the 11:45Z line: lir-e5-c5 was NOT preempted (my status listing truncated it); it kept running; the duplicate re-queue was cancelled before it launched anything.
- 12:15Z E5c3 mixture pre-training learned perception for seed 0 only (runs/E5c3/s0/train.jsonl MIX exp_brier 0.083 by step 40); seeds 1 and 2 stayed at the base rate for 400 steps (runs/E5c3-failedpretrain/s1, s2: MIX exp_brier 0.187-0.193). Arm comparisons from a non-perceiving init are uninformative, so the c5/c6 runs were killed after 0-50 fine-tune steps (their partial evals are kept under runs/E5c3-failedpretrain/). Seeds 1 and 2 are re-run with --pretrain-seed 0 (LoRA init and mixture data of seed 0) and an arm-agnostic extension rule (extend pre-training in 100-step blocks while the last-20-step mean exp_brier > 0.12, max 800); the arm fine-tunes use seed 1 / seed 2 data. So seeds 1-2 vary only the fine-tune phase from (a replicate of) the seed-0 pre-training; seed 0 is fully independent. Stated as a limitation in RESULTS.md.
- 13:05Z budget: programme actual GPU spend $26.80, E5 $8.00 (sum of DELETE cost_usd in logical-soup/GPU-LOG.md), 10 lir VMs live. The full D1 factorial does not fit under $50 with the siblings' runs; D1 (dose alpha=1, separate adapters) is cut to seed 0 for each goal mode (approval+random); the B0 factorial keeps 3 seeds. Queue order: d02-d05 (B0), r1 (B0 visible s1 random redo), p1 (protected check), d06/b9/r2 (D1 s0).
- 13:42Z STOP: `spend_usd lir-agent` = 57.64 (> $50: actual DELETE costs $34.82 for the programme plus reservations of live VMs and of VMs lost to a spot-preemption wave at 13:30Z that never got a DELETE line: lir-e5-d02, lir-e5-d03). Per the programme rule no new E5 launches; queue stopped (/tmp/lir-agent/E5/STOP). Not run: B0 shuffled s1-2, B0 visible s2, B0 deleted s2, B0 visible s1 random (cut by timeout at 21/64 steps), protected P check, all D1 (dose) runs except partial s0 runs that OOM'd. CPU fallback not used: box load 20 on 16 cores (uptime 13:41Z) and a 0.6B RL chain on CPU takes hours.
- 14:10Z E5c3 seed 2 (lir-e5-c9, --pretrain-seed 0) did not replicate the seed-0 pre-training on this VM (GPU nondeterminism): MIX exp_brier 0.188 for 489 steps (runs/E5c3/s2p0/train.jsonl). Killed before the arm fine-tunes (they would start from a non-perceiving state). E5c3 therefore has 2 perceiving seeds (s0, s1p0), below the 3-seed target.
