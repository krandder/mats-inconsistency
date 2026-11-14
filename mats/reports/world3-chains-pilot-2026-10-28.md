# W3 chain world: build and C2 pilot (2026-10-28)

Status at 13:20Z: **C2 is not readable on W3 yet. This is a blocker, with a diagnosis and a fix.**
1. XOR world, which is the design: no model learns even one hop. All models read only the query's own report, at
   every depth, with and without weight tying, under six training fixes. The barrier is the parity along the path.
2. Equality variant (no parity), where the C2 pilot was then run: the models learn a shortcut that does not
   depend on depth. They recognise a chain by its shared XOR-reliability value, which my information-matching
   rule made unique per chain. So depth, looping and extra iterations change nothing, and no arm gets near Bayes.
Fix: draw XOR reliabilities per edge from a small common set, then rerun. Decision for opus-mats: whether to
spend about $5 more on that rerun. If it still plateaus, the XOR world stays a C5 finding.

## What was built (work/worlds/)
- `world3_chains.py`: M = 8 chains x 12 nodes; random node ids out of 128. Messages are unary reports and XOR
  relations between neighbours (random token order), each tagged with its known reliability; majority messages
  are implemented but off.
  - Information matching: 1/3 of chains are exact; otherwise the XOR reliability is set so that the path from
    the nearest report has the same reliability at every l. The available information KL(Bayes||prior) is
    0.39 to 0.29 nats for l = 1..8.
  - Every chain, distractor or not, is drawn by the same procedure around its own virtual query.
  - Exact Bayes is pair-state forward-backward. References: prior, unary-only, nearest-report-only, Bayes.
  - Harness adapter: harness N = l, so train with --nmin 0 --nmax 4 and evaluate with --Ns 0..8.
  - Tests (20, all pass): forward-backward agrees with enumeration; the oracle is identical when recomputed from
    the token list alone (relabelled, reordered, orientation flipped), so the policy is ignorable; Bayes is
    calibrated against the generator; information is matched across l.
- `looped.py`: weight-tied LoopedSetPFN built on harness Blocks, with input injection. With one iteration it
  equals SetPFN (tested). It is a CLI wrapper around harness train/eval and takes `--loop-eval-iters` for extra
  iterations at test time and `--amp-bf16`. harness.py itself is not edited.
- `w3_dense.py`: trainer with K extra masked query tokens per episode (dense supervision). Its outputs are in
  harness format.
- Pre-registration: `PREREG-W3.md`, with amendments A1 to A5 (also appended to the shared PREREG.md).

## Result 1: plain pilot on the XOR world (seed 0 complete; seeds 1-2 stopped at 10-18k on the same plateau)
Arms U2/U4/U6 (untied) and LP2x2/LP2x3 (looped, also run at 1-6 iterations), trained for 40k steps on l <= 4.

| | l=0 | l=1 | l=4 | l=6 | l=8 |
|---|---|---|---|---|---|
| available info, KL(Bayes‖prior) | 0.49 | 0.39 | 0.37 | 0.29 | 0.31 |
| every arm, KL(Bayes‖model) | 0.13-0.15 | 0.40 | 0.35 | 0.32 | 0.30 |

At l >= 1 every arm sits exactly at the prior. At l = 0 every arm matches the unary-only reference, i.e. it reads
only the query's own report. Depth, weight tying and extra test-time iterations make no difference
(differences are 0.00x nats). The pre-registered training-validity check fails, so R1-R4 are not read.

## Screens on the XOR world (seed 0, validation gap to Bayes; train-only, disclosed as screens)
Plateau level: gap ≈ 0.30, which corresponds to solving l = 0 only.
- Curriculum on l (l <= k, widening every 3k steps), for U2/U4/U6/LP2x3 and for U4 with exact chains: all stay
  on the plateau. Exact chains fail too, so XOR noise is not the obstacle.
- Correlated-latent anneal, where training chains start nearly constant (flip 0.05) and are annealed to iid over
  5k or 10k steps: all stay on the plateau through 20k steps.
- lr 1e-3, or an extra signed-evidence feature (2r-1)c: both stay on the plateau (0.300 and 0.299 at 20k).
- Fewer distractors (M = 2): still on the plateau (0.32). Local CPU checks with M = 1 or 2 and with dense
  queries agree.
- **Equality world** (flip = 0: same messages and noise, but each chain's latent is constant, so the posterior is
  the evidence from every report reachable from q): learnable. With M = 2 the gap falls to 0.057 by 20k steps.
  With M = 8 it is still falling: 0.46, then 0.35, then 0.26 at 5k, 10k and 20k steps.

Diagnosis: the barrier is the parity. With iid bits, each message on the path is independent of x_q, and only
their XOR is informative. The models never get a gradient foothold for even one hop: they fail at l = 1 with 2
chains. Retrieval through random ids and distractors is learnable. Information matching is not the cause: the
labels are far from chance (0.3-0.4 nats available).

## Result 2: C2 pilot on the equality world (flip = 0; 50k steps; 3 seeds, all done)
KL(Bayes‖model), std policy, mean over 3 seeds. Available information is 0.59, 0.50, 0.44 and 0.38 nats at
l = 1, 4, 6 and 8.

| arm | l=0 | l=1 | l=2 | l=4 | l=5 | l=6 | l=8 |
|---|---|---|---|---|---|---|---|
| U2 | 0.20 | 0.30 | 0.25 | 0.22 | 0.18 | 0.23 | 0.26 |
| U4 | 0.19 | 0.30 | 0.24 | 0.22 | 0.19 | 0.24 | 0.26 |
| U6 | 0.20 | 0.30 | 0.24 | 0.22 | 0.17 | 0.23 | 0.26 |
| LP2x2 | 0.20 | 0.30 | 0.25 | 0.22 | 0.17 | 0.21 | 0.23 |
| LP2x3* | 0.22 | 0.33 | 0.29 | 0.28 | 0.22 | 0.26 | 0.29 |

\* LP2x3 seed 0 never left the plateau; the other two seeds match U6.

- Pre-registered readings:
  - Training validity fails: U6 has KL 0.20-0.30 on l <= 2, against the 0.05 threshold.
  - R1: not refuted (U2 at l = 6 has KL 0.23, se 0.02), but the result is vacuous, because no arm reaches 0.05
    anywhere.
  - R2: no rung structure. U2, U4 and U6 are the same to within 0.01 at every l.
  - R3: LP2x2 matches U4 (to within 0.01). LP2x3 counts as matching U6 only because its standard error is large.
  - R4: extra test-time iterations make things worse. LP2x3 at 4, 5 and 6 iterations is +0.01, +0.04 and +0.10
    nats on l = 5..8.
- Why depth does not matter (U4, 3 seeds; nats captured out of nats available):
  - On noisy chains: 0.36 of 0.60 at l = 1, 0.36 of 0.50 at l = 4, 0.27 of 0.44 at l = 6, and 0.18 of 0.39 at
    l = 8.
  - On exact chains: 0.16 of 0.57, then 0.11, 0.06 and 0.01.
- Cause: noisy chains have one XOR reliability per chain, c_path^(1/l'), shared by all of that chain's XOR
  tokens. It acts as a chain label, so a fixed-depth circuit can gather a chain's reports without tracing the path.
  Exact chains all share the value 1, so the shortcut fails on them. This is a design flaw in my information
  matching, and the same leak exists in the XOR world.

## Next (proposed, not run)
1. Remove the label: draw per-edge XOR reliability i.i.d. from a common set such as {0.85, 0.95, 1}. Match
   information by reporting excess loss relative to KL(Bayes‖prior) per l, instead of tuning reliability per
   chain. Rerun the equality-world C2 pilot (about $5, 3 VMs, 2 h).
2. XOR world: keep it as a C5 case (an optimisation barrier, not an expressivity one). The only fix not yet tried
   on GPU is a two-phase schedule: equality world until retrieval is learned, then a slow anneal to iid.

Spend: $8.49 on 11 L4 VMs (ceiling rate, from GPU-LOG) of the $12 cap. All are deleted: checked at 13:20Z
with `gcloud compute instances list`, which shows no w3c2 or w3eq VMs.
Files: work/worlds/{world3_chains.py, looped.py, w3_dense.py, run_w3*.sh, launch_w3.sh, w3_summarize.py,
PREREG-W3.md}. Results: work/worlds/results-w3/: plain-xor-s0, eq-pilot (combined summary.md, eval summaries, logs),
and screen logs.
