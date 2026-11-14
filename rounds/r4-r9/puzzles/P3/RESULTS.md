# P3: does Shapley-type pay make latent traders pick the output-natural factors?

Status: **No gain** (pre-registered falsifier (a) holds; Stage 4 not run; $0 GPU).

Answer. No. In the pre-registered T1 test (variant S, 20 seeds per arm), AS-IG pay aligned in 3/20 seeds, the same as
the additive-reader baseline (3/20), exact Shapley (3/20), Banzhaf (3/20) and LOO (2/20); every Holm-corrected
p = 1.0. AS-IG mostly collapsed to constant codes (Bayes gap 6.3 nats/example). The oracle's predicted equilibria
(natural pair plus a duplicate for Shapley and AS-IG) appeared in 0 of 160 trained runs.
A diagnostic that is not pre-registered let the maker price unvisited buckets. It raised Shapley to 10/20 on
seeds 0-1, but the seeds 2-3 replication does not beat the additive reader (6/20 vs 5/20, Holm p = 0.50).
The maze oracle M0 settles prediction (b) at $0: at c = 6 the chunks that end before the BFS backtrack starts carry -22% (BFS_S) and -21% (BFS_G) of the BFS reader's total value of information (pre-registered threshold < 1%), while DEF earns > 0 in every live chunk. So prediction (b) is settled without training and Stage 4 was not run.

All numbers: `results/summary.json` (built by `analyze.py`), charts: `results/charts.json`, M0: `results/m0.json`.
Intervals are 95% (Wilson for rates, Newcombe for rate differences, t for means over seeds).

## T1, variant S (toy N), pre-registered grid

3 traders, K=2 straight-through codes, 5k steps, batch 256, Adam 1e-3; 20 seeds per arm (2 processes x 10 replicates).
ALIGNED = each of b1, b2 has a trader with NMI >= 0.8, and no live trader has NMI >= 0.5 with b1 XOR b2.
"both nat" = b1 and b2 are both found (the first half of ALIGNED). Bayes gap on 100k fresh examples, nats/example
summed over 32 outputs (Bayes loss 6.35).

| arm | aligned | 95% CI | both nat | rotated trader | constant traders / run | Bayes gap [95% CI] | gap <= 0.02 | ECE |
|---|---|---|---|---|---|---|---|---|
| 1 LOO / backprop | 2/20 | 0.03-0.30 | 0.10 | 0.75 | 1.05 | 1.09 [-0.07, 2.25] | 0.80 | 0.010 |
| 2 additive reader | 3/20 | 0.05-0.36 | 0.75 | 0.65 | 0.00 | 0.20 [-0.22, 0.61] | 0.95 | 0.006 |
| 3 Moebius L1, lambda 1e-3 (picked) | 2/20 | 0.03-0.30 | 0.10 | 0.40 | 0.65 | 2.37 [1.44, 3.30] | 0.40 | 0.011 |
| 3 Moebius L1, lambda 1e-2 | 1/20 | 0.01-0.24 | 0.05 | 0.15 | 0.30 | 2.57 [1.67, 3.48] | 0.35 | 0.010 |
| 4 exact Shapley | 3/20 | 0.05-0.36 | 0.20 | 0.75 | 0.75 | 0.81 [-0.11, 1.73] | 0.35 | 0.009 |
| 5 Banzhaf | 3/20 | 0.05-0.36 | 0.15 | 0.65 | 0.75 | 0.76 [-0.21, 1.73] | 0.60 | 0.006 |
| 6 AS-IG per-output | 3/20 | 0.05-0.36 | 0.15 | 0.05 | 1.70 | 6.28 [4.75, 7.81] | 0.05 | 0.015 |
| 7 AS-IG joint (neg. control) | 1/20 | 0.01-0.24 | 0.05 | 0.65 | 1.00 | 1.78 [0.84, 2.73] | 0.55 | 0.002 |

Most common final profiles (sorted trader classes): LOO `b1|const|xor` 10/20; additive `b1|b2|xor` 10/20;
Shapley `b1|const|xor` 5/20; AS-IG `b1|const|const` 9/20. The tabular arms mostly settle on the rotated sufficient
pair (b1, b1 XOR b2) or collapse; LOO reaches a sufficient code in 16/20 seeds but a rotated one in 14/20.
The additive reader reaches a sufficient code in 19/20 seeds and finds both natural factors in 15/20 but fails ALIGNED because its third trader tracks
b1 XOR b2 (AND = (b1 + b2 - XOR)/2 is additive in those three features).

**Criterion 1 (Gain) fails** on every clause except the controls: AS-IG 3/20 (needs >= 16/20); differences vs LOO,
additive, Moebius and Banzhaf are +0.05, 0.00, +0.05, 0.00 (95% CI of the difference vs additive -0.23 to +0.23),
one-sided Fisher p = 0.50-0.67, Holm p = 1.0 for all four; Bayes gap 6.28 (needs <= 0.02).

Pre-registered falsifiers:
- (a) known baseline explains it: **holds**. AS-IG and exact Shapley are each within 0.30 of both the additive and the
  Moebius baseline, and p >= 0.05 after Holm. So: No gain, T1 plus oracles published, Stage 4 not run, $0 GPU.
- (b) oracle wrong for trained nets: does not hold as defined (joint AS-IG minus LOO = -0.05; Banzhaf keeps a
  rotated trader in 13/20, avoids it in 7/20 = 35% < 50%). But the per-arm oracle predictions are not what training
  reaches: "natural pair plus a duplicate" (arms 4 and 6) occurs in 0/40 runs, and no arm ever keeps a duplicate trader.
- (c) pay without discovery: 3 of the 3 aligned AS-IG runs and 1 of 3 aligned Shapley runs have a Bayes gap > 0.05
  (0.06-0.79), with low ECE (<= 0.035), so without the over-confidence pattern. Those aligned runs are not sufficient codes.
- (d) noise earns: does not hold (below).

Controls (prequential pay, nats/example, 3 seeds): shuffled labels, total pay of the three traders -0.0086
[-0.0094, -0.0077], largest per-trader upper bound 2e-5: pass. A 4th trader frozen at random init reading an
independent x draw: -0.0011 [-0.0029, 0.0007]: pass. Extra, not pre-registered: the same frozen trader reading the
real x earns 0.17 [-0.55, 0.89]; a random function of the real x carries information about z*, so this is not a noise
control. AS ledger (|sum of AS-IG shares - (v(one-hot) - v(marginal))|, 20 midpoint steps): 0.021-0.025 nats/example on the codes
trained by LOO, Shapley and Banzhaf (the oracle predicted 0.015-0.027 at 20 steps), 0.005 on the mostly constant
AS-IG-trained codes, and 0.0004 for the joint-score AS-IG arm on its own value.

## T1, variants Bn and Bm (bits; 10 seeds each)

Aligned 0/10 in all 8 cells (LOO, additive, Shapley, AS-IG on Bn and Bm). Traders find the easy majority factor b2
(NMI >= 0.8 in 10/10 LOO, Shapley and AS-IG runs on Bn) and never the parity factor b1 (0/10 on Bn). Strengthener (2)
fails: on Bm AS-IG aligns in 0/10. Bayes gaps 2.8-11.0 nats/example.

Sanity (K=4, one trader, 1 seed; not evidence): M-soft reaches the Bayes gap (0.000); M-hard and VQ-sup stop at
3.97 with 3 of 4 codes used (the collapse below). Unit tests (Stage 1): all identities hold to <= 1.5e-15
(`results/unit_tests.json`).

## Why the tabular arms collapse (diagnostic, not pre-registered)

The pre-registered maker quotes the x-only price for buckets not visited recently. A trader that moves an example
into an unvisited bucket is then always paid less than before, so codes can merge into visited buckets but never
split out of them; constant codes are absorbing. The additive reader generalises to unseen code combinations and has
no such barrier. Test: let the maker train on codes re-drawn with probability e (`t1.py --explore e`), so that every
bucket is priced from nearby data. Pay rules and scoring unchanged.

| arm, explore 0.05, seeds 0-1 | aligned | Bayes gap | top profile |
|---|---|---|---|
| exact Shapley | 10/20 (CI 0.30-0.70) | 1.46 | `b1\|b2\|const` 9 |
| Banzhaf | 7/20 | 3.71 | `b1\|b2\|const` 6 |
| LOO | 2/20 | 4.32 | |
| AS-IG | 2/20 | 5.95 | `b1\|const\|const` 5 |

At explore 0.2 (dose check): Shapley 13/20 aligned (CI 0.43-0.82), Bayes gap 0.40, top profile `b1|b2|const` 11, LOO 6/20 (CI 0.15-0.52), Bayes gap 3.04. The explore effect grows with the dose for Shapley and stays small for LOO.

Held-out replication on seeds 2-3, with the test fixed in `gen_queue2.py` before any of these runs existed (one-sided
Fisher, Holm over the two comparisons, alpha 0.05):
- Shapley, explore 0.05: 6/20 (CI 0.15-0.52). Against the additive reader, 5/20: difference +0.05 [-0.22, 0.31],
  one-sided p = 0.50, Holm 0.50. Against LOO, explore 0.05, 0/20: difference +0.30 [0.08, 0.52], p = 0.010, Holm 0.020.
- **Not replicated.** The test needed both comparisons, and it fails against the additive reader. Exploration makes
  Shapley beat LOO under the same maker, but it does not beat the barrier-free baseline. Also on seeds 2-3: Banzhaf
  5/20, AS-IG 4/20 and Moebius 1e-3 0/20, all at explore 0.05. The additive reader finds both natural factors in 10/20.
- Pooled over all 40 seeds (reported only, not a test): Shapley at explore 0.05 16/40 vs additive 8/40, one-sided p = 0.043.

## M0 maze oracle (Stage 3, CPU, no code-net training)

Local cellular-automaton DEF (dead-end filling, protected endpoints, K=2), BFS_S and BFS_G (flood with distance mod 3
plus backtrack mark, K=8) on rasterised maze-dataset 1.4.2 mazes (9x9 cells = 19x19 pixels; rounds are pixel steps).

- Build gate: PASS. CA = global simulators (P7 `bfs` distances and a queue-based dead-end pruning) on 1000 perfect
  and 822 percolated mazes (178 with tied shortest paths dropped), 100% for all three rules; CA = maze-dataset's
  solution 100% for BFS_S/BFS_G on both families and for DEF on perfect mazes.
- (ii) DEF with protected endpoints gives the wrong path on 95.8% of 9x9 percolated non-dead-end test mazes already at
  p = 0.05 (n = 742), so p is fixed at 0.05.
- (i) Rounds to the final path, p50 / p99: DEF 58/116 and BFS+backtrack 102/238 on 9x9 perfect mazes; 20/82 and 78/210 on 9x9 percolated (n = 823); 110/222 and 180/454 on 13x13 perfect; 26/99 and 98/329 on 13x13 percolated. BFS_S and BFS_G are identical here. T = max(30, p99 of BFS+backtrack on 9x9 percolated = 210,
  rounded up to a multiple of 6) = 210.
- (iii) Bounded 3x3 reader fitted on the exact codes (600 fit mazes, 300 test mazes, 9x9 percolated):
  - DEF (K=2): mean S_R goes from -88.2 to -71.4 nats per maze (total VOI 16.8). At c = 6, all 8 live chunks earn > 0
    (the smallest mean gain is 0.28). At c = 1, all 48 live chunks earn > 0.
  - BFS_S (K=8): S_R goes from -93.4 to -9.6 (VOI 83.8). The backtrack starts at round 38 (p50), and the path is final
    at round 78 (p50). At c = 6, the first four chunks lose (-2.2, -2.9, -2.1, -0.5). Chunks ending before the
    backtrack carry -22.4% of the VOI (-22.9% at c = 1).
  - BFS_G (K=8): S_R goes from -93.7 to -9.7 (VOI 84.0). Its pre-backtrack share is -21.4% at c = 6 (-21.9% at c = 1).
  - At c = T, each algorithm earns its whole VOI in one chunk. A bounded reader cannot use the flood codes until the
    backtrack marks the path, so a credit horizon shorter than the backtrack pays the wavefront negatively.

Pre-registered reading: BFS earns < 1% of its VOI before its backtrack (it earns -21% to -22%), and DEF earns > 0 in every chunk. So prediction (b) is
settled at $0 (`results/m0.json` `prereg_reading`: `prediction_b_settled: true`), and Stage 4 is not run.

## Stage 4 (GPU credit-horizon sweep): not run

Gate (i) is closed (AS-IG and Shapley 3/20 = additive 3/20, one-sided p = 0.67), and gate (ii) is also closed (M0 settles prediction (b)).
No GPU VM was created. GPU spend: $0.

## Cost

- CPU VMs (T1 grid): trader-cat-p3-1 ran 0.55 h (spot preemption), and trader-cat-p3-2 ran 0.40 h. The total is 0.95 h
  at the $1.20/h ceiling rate, $1.14 (`vm.sh spend`; spot is billed lower). Both VMs and their disks are deleted,
  checked with `gcloud compute instances list` / `disks list --filter=name~^trader-cat-p3` (0 items) at 14:43Z on 09-29.
- GPU: $0. General CPU: unit tests plus M0 (56 min wall time, niced, 2 threads). No paid LLM APIs.
- Charts (`results/charts.json`): `p3_t1_aligned`, `p3_t1_suffgap`, `p3_t1_nmi_traj`, `p3_t1_diag_explore`,
  `p3_m0_SR`, `p3_m0_c6`.

## Deviations from the spec (stated, not hidden)

1. Location: code and results are in `puzzles/P3/` (as the task asked), not `p3-latent-markets/`; the Stage 0 oracle
   stays in `p3-latent-markets/oracle/`.
2. Compute: General ran at load ~100 on 16 cores, so the T1 grid ran on a spot CPU VM in GCP_PROJECT
   (c2d-highcpu-32, 1 thread per job, `vm.sh`; watchdog with 5-min sha256-verified syncs). trader-cat-p3-1 was
   preempted at 14:10Z; the 3 unsynced replication jobs were rerun on trader-cat-p3-2. Unit tests and M0 ran on
   General, niced, 2 threads.
3. Seeds: each 20-seed arm is 2 processes x 10 replicates trained in one batched model (seed s, replicate r); B cells
   are 1 x 10.
4. Random-trader control: the pre-registered run's frozen 4th trader reads an independent x draw, so its code is pure
   noise; the version reading the real x is reported separately as an extra.
5. Moebius lambda picked by mean validation log loss (model gap on 100k fresh examples) over the 20 seeds: 1e-3.
6. The `--explore` diagnostic, the explore 0.2 dose check and the seeds 2-3 replication are not pre-registered.
7. M0: T is rounded up to a multiple of 6 so c=6 chunks tile it; "DEF earns in every chunk" is read over the c=6 chunks
   in which >= 10% of test mazes are still filling (after that DEF has stopped and gains are 0 by construction).
   BFS "chunks that end before its backtrack starts" = chunks ending at or before round L+1 (L = path length in pixels).
   Packages: maze-dataset 1.4.2 --no-deps with muutils pinned to 0.8.12 and tqdm added --no-deps.
8. Budget: this run used the task's cap ($8, watchdog at $6); the spec's $3-4 caps were for the GPU stage that did
   not run. VM rows are in `../../GPU-LEDGER.md` under "P3 latent-market factor selection".

## Files

`t1.py` (T1 trainer and scoring), `test_identities.py` (Stage 1), `m0.py` (Stage 3), `analyze.py` (tables, tests,
charts), `gen_queue.py`, `gen_queue2.py`, `t1_queue*.txt`, `run_queue.sh`, `vm.sh`, `vm_startup.sh`, `vm/LEDGER.md`;
results in `results/` (`t1/*.json` per job, `summary.json`, `charts.json`, `m0.json`, `unit_tests.json`).
