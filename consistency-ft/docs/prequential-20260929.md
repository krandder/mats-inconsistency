# Prequential consistency training, 2026-09-29

**Status: Informative.** The main distillation result replicates under a stricter protocol. Two orderings flip.

**Question.** The distillation run (`docs/consistency-distill-20260929.md`) trained Qwen3-0.6B for 2 epochs on a train split and scored a test split. The split is not ordered in time, so the students were partly trained on questions that close after the test questions.

What happens if every pair is instead forecast before the student trains on it, in close-date order, with outcomes used only after they resolve? The measure is the prequential code length: total bits over the stream. The questions:

- Which targets make the student learn fastest?
- Do consistent (projected) teacher answers beat raw teacher answers and outcomes?
- Does the ranking of the split protocol survive?

**Answer (Qwen3-0.6B, 3 seeds per trained arm).**

- **C has the shortest code.** Training on the teacher's projected answers (C) gives the shortest code of all students: 37,808 bits for the 43,824 answers, 0.863 bits per answer.
  - Raw teacher answers (R): 38,820.
  - Outcomes (O): 41,620.
  - The mix (CO): 40,798.
  - The untrained model (Z): 75,361.
- **C beats R by 1,012 bits** (95% CI [−1,228, −805]) in every seed.
- **Consistency targets are not faster than raw targets.** C beats R by the same margin in every third of the stream, about 0.024 bits per answer. That is a level shift, not faster learning.
- **Teacher targets are much faster than outcomes early on.** C and R lead O by 0.16–0.18 bits per answer in blocks 1–12, because outcomes arrive only after questions close. The gap shrinks to 0.01–0.06 later.
- **The ranking changes in two places.**
  - With the split protocol, O beats CO. Prequentially, CO beats O by 821 bits, CI [−1,314, −425], in all 3 seeds.
  - After projecting each student's own 12 answers, R beats C prequentially: +143 bits, CI [+55, +228]. In the split run they were tied.
- **C < R holds under both protocols.**
- **Every student is close to the base rate.** C beats a prequential base-rate table only once the zero-shot first block is excluded: −1,194 bits [−2,006, −621]. R, O and CO do not beat it on raw answers.

Numbers: `docs/prequential-20260929.json` (full report) and `docs/prequential-20260929-chart.json` (per-block curves). Code: `scripts/preq_data.py`, `scripts/preq_student.py`, `scripts/preq_check_packing.py`, `scripts/preq_check_precision.py`, `scripts/preq_analyze.py`, `scripts/gpu/preq_gpu.sh`, `scripts/gpu/preq_remote.sh`.

## Design

- **Principle.** Prequential evaluation (Dawid 1984, "Present position and potential developments: some personal views. Statistical theory: the prequential approach", *J. R. Statist. Soc. A* 147(2):278–292, doi:10.2307/2981683): forecast each item before it is used for training, and score the sequence of forecasts. The sum of −log2 p over a time-ordered stream is the prequential code length of the labels (Blier & Ollivier 2018, "The description length of deep learning models", NeurIPS 31, pp. 2216–2226, arXiv:1802.07044). Blier & Ollivier code a dataset in chunks and retrain from scratch on all earlier chunks for each new chunk. Here the model is trained online instead: one continuing LoRA run, each item used once.
- **Stream.** All 3,652 pairs of jev-disjoint-v1-filtered-20260919 (train 2,164, validation 422, test 1,066; 43,824 rows = 12 forms each). They are ordered by first close date (as_of + 7 days, where as_of is the forecast date in the teacher prompt), with ties broken by last close date and then by sha256 of the pair id. The stream is cut into 36 blocks of 101–102 pairs. The block time T_k is the earliest as_of in block k. as_of runs from 2025-04-21 to 2025-12-22.
  - The split is not temporal. Test pairs appear from block 0 onward, but are concentrated late: 8 of the 101 pairs in block 0, 74 of 102 in block 30, and all of blocks 34–35. So the ordinary split protocol trains on questions that close after some test questions.
- **When a target may be used.** A row trained before scoring block k must be from an earlier block and usable by T_k:

  | Arm | Target | Usable at |
  |---|---|---|
  | Z | no training | — |
  | R | 32B teacher raw answer | the pair's as_of |
  | C | 32B teacher kl-projected answer | the pair's as_of |
  | O | outcome label | end of the closing week of the events it needs (P or Q: their close + 7 d; joints: the later close + 7 d) |
  | CO | 0.5 · projected + 0.5 · label | as O |

  Teacher answers are usable at the forecast date because the teacher answered as of that date. Test pairs' teacher answers are used only as targets (R, C) once the pair has been scored. No outcome is used before its week has closed.
  - Trainable rows by the end of the stream: 42,612 teacher rows, 36,860 outcome rows. The rest of the outcome rows resolve after the last block.
  - Outcome delay: median 4 blocks, 90th percentile 19 blocks.
- **Online training.** For each block k: train once on every row that became usable by T_k (each row exactly once; within the block, groups are shuffled with a seeded RNG and length-bucketed into steps), then score all 1,212 or 1,224 rows of block k with the current adapter. Block 0 is scored before any training, so every arm equals Z there (checked: difference 0.000 bits for every arm).
- **Student.** Qwen3-0.6B (revision c1899de2), with the prompt, readout and LoRA of `scripts/distill_student.py`:
  - P(YES) = sigmoid(logit YES − logit NO) at the first assistant token.
  - LoRA r16, α32, dropout 0.05 on all linear layers. BCE against the soft target.
  - AdamW, lr 2e-4, 50 warmup steps, then constant. An online learner has no known horizon, so there is no decay.
  - About 32 rows per optimizer step (whole groups of one pair's usable forms). Grad-norm clip 1.0. Max length 1,152 tokens (left truncation).
  - Same hyperparameters for every arm. No tuning, no validation-based early stopping: the stream has no held-out set.
- **Packing.** The 12 prompts of a pair share long prefixes. Each pair is packed as a token trie in one sequence (4D mask: each token sees its trie ancestors; position ids = depth), 3.9× fewer tokens over the whole stream (6.06 M instead of 23.47 M when scoring).
  - Gate at every VM start, `scripts/preq_check_packing.py` in fp32: max |logit difference| between packed and 48 ordinary passes = 4.5e-5 (base) and 6.3e-5 (random LoRA), within the 2e-3 tolerance.
  - One small difference from separate passes: in training, the forms of a pair share one LoRA-dropout mask on their shared prefix.
- **Precision.** The jobs load fp32 weights and run under bf16 autocast. `scripts/preq_check_precision.py`, 72 prompts, mean |logit error| against fp32: bf16 weights, separate passes (as in the distill run) 0.094; fp32 weights + autocast, packed (this run) 0.071. So this run is slightly closer to fp32 than the distill run.
- **Metrics.** Code lengths in bits, summed over the stream:
  - **bits12:** −log2 p(label) over all 12 forms, with p clipped to [0.01, 0.99]. This is the loss the arms train on. It counts each outcome several times, because the 12 forms restate two events.
  - **bits12 after projection:** the same after kl-projecting each pair's 12 answers onto one joint distribution. This is "one call per form, made consistent afterwards".
  - **bits per pair:** the code length of the two outcomes themselves, −log2 q(true atom), where q is the projected joint over the four atoms mixed with 4% uniform. A know-nothing coder pays 2 bits per pair.
- **References** (not learners, same stream):
  - uniform (p = 0.5);
  - a prequential base rate: per form, a Laplace estimate from the outcomes available at T_k;
  - the 32B teacher, raw and projected: the target that R and C imitate.
- **Intervals.** Paired bootstrap over the 36 blocks, 10,000 resamples, of block-level differences of seed-mean code lengths. Learning speed is reported in thirds (blocks 1–12, 13–24, 25–35). Seeds (1, 2, 3) change the LoRA initialization and dropout. Z is deterministic and has one seed.


## Results (43,824 answers, 3,652 pairs, 36 blocks)

Code lengths in bits, summed over the stream. Trained arms show the mean of seeds 1–3; per-row and per-pair values are in parentheses.

| | seeds | bits12 | saved vs Z [95% CI] | seed range | bits12 after projection | bits per pair |
|---|---|---|---|---|---|---|
| Z (untrained) | 1 | 75,361 (1.720) | — | — | 40,848 (0.932) | 7,814 (2.140) |
| R (teacher raw) | 3 | 38,820 (0.886) | 36,541 [33,731, 38,938] | 38,639–38,968 | **36,389** (0.830) | **6,885** (1.885) |
| **C (teacher projected)** | 3 | **37,808** (0.863) | 37,553 [34,696, 39,930] | 37,748–37,868 | 36,532 (0.834) | 6,917 (1.894) |
| O (outcomes) | 3 | 41,620 (0.950) | 33,741 [30,190, 36,722] | 41,355–41,983 | 37,922 (0.865) | 7,214 (1.975) |
| CO (mix) | 3 | 40,798 (0.931) | 34,562 [31,080, 37,433] | 40,519–41,291 | 37,582 (0.858) | 7,137 (1.954) |
| uniform | | 43,824 (1.000) | | | 38,310 (0.874) | 7,304 (2.000) |
| base rate (prequential) | | 38,049 (0.868) | | | 37,590 (0.858) | 7,147 (1.957) |
| teacher raw (32B) | | 39,811 (0.908) | | | 35,359 (0.807) | 6,659 (1.823) |
| teacher projected (32B) | | 35,359 (0.807) | | | 35,359 (0.807) | 6,659 (1.823) |

Savings vs Z after projection: R 4,459 [3,276, 5,880], C 4,316 [3,140, 5,737], O 2,926 [2,175, 3,749], CO 3,266 [2,436, 4,186]. Savings in bits per pair: R 929, C 897, O 600, CO 677.

Z is worse than uniform (1.72 bits per answer), because the zero-shot answers are badly calibrated. Most of each arm's "saved vs Z" is therefore calibration that any training fixes. The comparisons between arms and against the base rate are the informative ones.

**Contrasts.** Negative means the first arm has the shorter code. Intervals are block-bootstrap intervals of the seed-mean difference. "By seed" gives the same-seed differences of totals for seeds 1, 2 and 3.

| Contrast | bits12 [95% CI] | by seed | after projection | bits per pair |
|---|---|---|---|---|
| **C − R** | **−1,012 [−1,228, −805]** | −1,046, −771, −1,220 | +143 [+55, +228] | +32 [+14, +49] |
| C − O | −3,812 [−6,471, −1,856] | −3,548, −4,115, −3,773 | −1,390 [−2,109, −793] | −297 [−443, −176] |
| R − O | −2,799 [−5,533, −856] | −2,502, −3,344, −2,553 | −1,533 [−2,244, −935] | −329 [−474, −207] |
| CO − O | −821 [−1,314, −425] | −836, −1,398, −229 | −340 [−506, −188] | −77 [−114, −44] |
| CO − C | +2,991 [+1,314, +5,496] | +2,712, +2,717, +3,543 | +1,050 [+518, +1,678] | +220 [+114, +344] |
| C − base rate | −241 [−1,809, +2,125] | | −1,058 [−1,522, −607] | −230 [−332, −135] |
| C − base rate, blocks 1–35 | −1,194 [−2,006, −621] | | −1,146 [−1,561, −790] | −247 [−340, −169] |
| R − base rate, blocks 1–35 | −181 [−1,107, +457] | | −1,290 [−1,688, −944] | −279 [−369, −203] |
| O − base rate, blocks 1–35 | +2,618 [+1,122, +4,513] | | +243 [−242, +747] | +50 [−63, +160] |
| CO − base rate, blocks 1–35 | +1,797 [+573, +3,493] | | −97 [−515, +315] | −27 [−127, +63] |
| C − teacher projected, blocks 1–35 | +1,268 [+858, +1,668] | | +1,009 [+613, +1,402] | +226 [+147, +306] |

"Blocks 1–35" leaves out block 0. There every arm is still the zero-shot model: 1.786 bits per answer, against 1.0 for the base rate, which has no data yet.

**Learning speed.** Bits per answer in each third of the stream (mean of seeds), and differences with 95% CIs:

| | blocks 1–12 | blocks 13–24 | blocks 25–35 |
|---|---|---|---|
| Z | 1.711 | 1.704 | 1.740 |
| base rate (prequential) | 0.896 | 0.861 | 0.834 |
| teacher projected (32B) | 0.817 | 0.827 | 0.774 |
| R | 0.882 | 0.871 | 0.825 |
| C | 0.857 | 0.848 | 0.801 |
| O | 1.037 | 0.879 | 0.856 |
| CO | 1.003 | 0.871 | 0.841 |
| C − R | −0.025 [−0.036, −0.013] | −0.022 [−0.028, −0.018] | −0.024 [−0.030, −0.018] |
| C − O | −0.180 [−0.328, −0.063] | −0.030 [−0.044, −0.019] | −0.056 [−0.104, −0.016] |
| R − O | −0.155 [−0.309, −0.039] | −0.008 [−0.022, +0.003] | −0.032 [−0.080, +0.009] |
| CO − O | −0.034 [−0.061, −0.012] | −0.008 [−0.016, −0.002] | −0.016 [−0.027, −0.006] |

**Split protocol vs prequential.**

- Split: 2 epochs on 2,164 train pairs, with early stopping on validation, scored on the 1,066 test pairs. Values are nats per row from the distillation report, converted to bits.
- Prequential: this run, on the same 12,792 test rows, on all rows, and on the last third.

| Arm | split test | prequential, test rows | prequential, all rows | prequential, blocks 25–35 | split test after projection | prequential, all rows after projection |
|---|---|---|---|---|---|---|
| Z | 1.818 | 1.831 | 1.720 | 1.740 | 0.968 | 0.932 |
| R | 0.819 | 0.832 | 0.886 | 0.825 | 0.802 | 0.830 |
| C | **0.801** | **0.808** | **0.863** | **0.801** | **0.798** | 0.834 |
| O | 0.862 | 0.893 | 0.950 | 0.856 | 0.840 | 0.865 |
| CO | 0.880 | 0.879 | 0.931 | 0.841 | 0.868 | 0.858 |

The rankings (best first):

- On raw answers:
  - The split protocol ranks C < R < O < CO < Z.
  - The prequential protocol ranks C < R < CO < O < Z. The same order holds on test rows only, on all rows and on the last third.
- After projection:
  - Split ranks C < R < O < CO.
  - Prequential ranks R < C < CO < O.
- In the split run, CO − O was +0.0124 nats per row [+0.0040, +0.0212] (CO worse). After projection, C − R was −0.0029 [−0.0067, +0.0010].

**Sanity checks** (in the JSON):

- Every trained arm equals Z on block 0 (difference 0.000 bits).
- Z's stream answers on the test rows match the distillation run's Z answers: correlation 0.993, mean |Δp| 0.013, max 0.108. The differences are the expected ones from fp32 vs bf16 weights and packed vs separate passes (see Precision).
- The packing gate passed at each VM start.
- Trained rows per arm: 42,612 for R and C, 36,860 for O and CO. These are the rows available before the last block.

## Reading

- **Projected targets are the better training signal, prequentially too.**
  - C − R is −1,012 bits in total, about −0.023 bits per answer. The split run gave −0.0185 bits per row. The two agree.
  - The sign is the same in all 3 seeds and in every third of the stream.
- **The gain is not faster learning.**
  - The C − R gap is constant over the stream: −0.025, −0.022 and −0.024 per answer in the three thirds.
  - C leads in 33 of the 34 blocks 2–35; in block 1 (one training block, 1,212 rows) R is ahead by 0.018 bits per answer. C keeps the lead from block 2 on without widening it. That fits a coherence effect, not a sample-efficiency effect.
  - Once each student's own answers are projected, R's student is slightly better than C's: +143 bits [+55, +228], and +32 bits per pair [+14, +49]. The split run found no difference here. So the answer is the same as in the distillation run, now sharper: C's gain on raw answers is coherence that projection at inference also gives. A guess, not tested: the raw targets keep a little more information, which projection then recovers.
- **Timing is what hurts outcome training.**
  - O and CO can use an outcome only after the question(s) it depends on have closed (for joints, both). The delay after the row is scored has a median of 4 blocks and a 90th percentile of 19 blocks.
  - Before scoring block 12, O has trained on 4,066 outcome rows and C on 14,604 teacher rows. In blocks 1–12, O is worse than uniform (1.037 bits per answer) and 0.18 bits per answer behind C.
  - O catches up in the middle third. It falls behind again in the last third (−0.056 vs C). There, long questions resolve in large lumps (6,576 outcome rows before block 33, 4,740 before block 34), while the newest questions' outcomes are still missing. This is the likely reason; it was not tested.
  - This handicap is what outcome supervision actually costs a forecaster that must answer before questions resolve. The split protocol hides it.
- **The CO vs O order flips.** In the split run, mixing teacher answers into outcome targets hurt. Prequentially it helps, especially early: CO − O is −0.034 bits per answer in blocks 1–12. So the teacher half of the target is useful when outcomes are scarce. The split result "CO is worse than O" should not be read as a property of the targets.
- **Students barely beat the base rate.**
  - Leaving out block 0, only C beats the prequential base-rate table on raw answers: by 1,194 bits [−2,006, −621], about 0.028 bits per answer.
  - After projection, C and R beat it by 1,146 and 1,290 bits (blocks 1–35).
  - The projected 32B teacher is still 1,009–1,268 bits better than C (blocks 1–35).
  - At this size, most of what the teacher knows about the questions does not reach the student. The distillation run found the same.
- **What this means for choosing targets.** For a small forecaster trained online, teacher targets beat outcome targets by a wide margin, and consistent teacher targets beat raw ones for single-call answers. If answers are projected at inference anyway, raw and projected targets are about equal. The next useful test is a larger student (1.7B), to see whether the gap to the teacher shrinks.

## What went wrong and what is missing

- **Two out-of-memory failures, one preemption, one resume.**
  - On cdistill-preq-a, C:1 and R:1 ran out of GPU memory at about 13:31Z and 13:37Z. The run had two jobs per 40 GB A100, each at about 19.7 GB.
  - Spot preemption then deleted the VM at 13:41:31Z. Z (deterministic) had finished and was synced. The other seed-1 arms were rerun from scratch on cdistill-preq-b, with training micro-batches of at most 4,096 tokens instead of 6,144. The loss is summed and divided by the step's rows, so the update is unchanged.
  - C:1 and R:1 ran out of memory again at block 28. The real cause was one pair (train-pair-02466, block 27) with 8 left-truncated prompts. Truncation removes the shared prefix, so its trie is 8,654 tokens long, while every other pair is at most 2,725.
  - Fix: training splits a group longer than the cap into sub-packs, in the same step with the same loss. It was checked on the VM with a smoke run.
  - The four seed-1 jobs then resumed from their block checkpoints. Every block reseeds its RNGs, and no step before block 28 packed more than 4,096 tokens, so the resumed runs equal fresh runs.
- **Z has one seed.** Z is deterministic: no training, and dropout is off at scoring. Its stream predictions agree with the distill run's Z on the 12,792 test rows: mean |Δp| 0.013, correlation 0.993. The small differences come from fp32 weights here vs bf16 weights there.
- **Qwen3-1.7B was not run.** It was optional, and it would cost about 3× as much per arm.
- **Adapters are not kept.** Checkpoints are deleted when each job ends. Only per-row predictions, per-block events and results were synced.
- **No hyperparameter search.** lr, warmup and batch size come from the distill run, which was tuned for 2 epochs over 2,164 train pairs, not for one online pass. A different lr could change the gaps between arms.

## Cost and cleanup

- **DashScope:** $0. The teacher answers were reused from the distillation run.
- **GCP:** $12.81 of the $15 cap (ceiling estimate: A100 list rate $3.69/h × VM lifetime, from the prequential section of `/home/agent/data/consistency-distill-20260929/gpu/LEDGER.md`). Two spot a2-highgpu-1g VMs, both with `--max-run-duration=4h` and termination action DELETE, plus a spend watchdog:

  | VM | Zone | Up | Down | Cost | How it ended |
  |---|---|---|---|---|---|
  | cdistill-preq-a | us-central1-a | 12:57Z | 13:41Z | $2.70 | preempted |
  | cdistill-preq-b | us-west4-b | 13:46Z | 16:31Z | $10.11 | deleted by the launcher after the final sync |

  The second VM went to us-west4-b because us-central1-a/b/c/f, europe-west4, us-east1-b and us-west1-b had no spot A100 capacity at 13:44Z.
- **Cleanup:** checked at 16:31Z. `gcloud --project=GCP_PROJECT compute instances list --filter="name~^cdistill-preq"` listed 0 items, and `gcloud --project=GCP_PROJECT compute disks list --filter="name~^cdistill-preq"` also listed 0 items. The watchdog was stopped after the delete.
- **Outputs:** synced every 5 minutes, and sha256-verified on every sync (57 files in the final one). They are in `/home/agent/data/prequential-20260929/gpu/`, 14 MB in total:
  - `cdistill-preq-b/Qwen3-0.6B/{C,R,O,CO}-s{1,2,3}/` for the trained arms;
  - `cdistill-preq-a/Qwen3-0.6B/Z-s1/` for Z.

  The stream bundle is in `/home/agent/data/prequential-20260929/bundle/`.

## Reproduce

```
python3 scripts/preq_data.py                                  # stream, blocks, manifest -> /home/agent/data/prequential-20260929/bundle
python3 scripts/preq_check_packing.py --stream .../stream.jsonl.gz --model .../Qwen3-0.6B   # packed vs ordinary, on CPU
scripts/gpu/preq_gpu.sh watchdog &                           # spend cap / age / idle guard
scripts/gpu/preq_gpu.sh up cdistill-preq-b Qwen3-0.6B         # spot A100, code + bundle upload, model download
scripts/gpu/preq_gpu.sh start cdistill-preq-b Qwen3-0.6B "C:1 R:1 O:1 CO:1 C:2 R:2 O:2 CO:2 C:3 R:3 O:3 CO:3"   # Z:1 likewise
scripts/gpu/preq_gpu.sh down cdistill-preq-b "done"           # sync, verify, delete, ledger line
python3 scripts/preq_analyze.py                               # -> docs/prequential-20260929.json, -chart.json
```

Each of the 12 trained runs took about 25 minutes, two at a time on one A100 (40 GB, peak about 13 GB per job).
