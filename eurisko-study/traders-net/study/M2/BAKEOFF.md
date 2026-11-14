# M2 validity bake-off: do market books rank units better than Taylor/movement?

Lead merge, 2026-09-28 ~06:40Z, of workers 1-4 (`STATUS.md`). Paths are relative to `study/M2/`.
Every number is mean ± sd over wealth-net seeds 1-3 and is re-read from its file by
`python3 check_bakeoff.py` (210/210 match at 06:40Z). Zero ablation unless stated.
Deliverables named by the plan: `oracle.py` (entry point; implementation `w1/oracle.py`) and this file.

## TL;DR
- **Books do not beat |Taylor|/movement.** Neurons: a narrow Spearman win at the final checkpoint
  only (v2_lmsr_inst 0.199±0.075 vs |Taylor| 0.103±0.084), a tie at 8k/20k/30k, and the same book
  prunes *worse than random* (frontier AUC 0.863±0.645 vs random 0.451±0.218). Heads: tie. Readouts: loss
  to movement (0.303±0.022 vs 0.814±0.027). Stop rule: treated as met; pruning work on books stops.
- **Gate B coalition win: NO.** On random coalitions the best book (0.818±0.027) ties |Taylor|
  (0.800±0.010) and movement (0.804±0.023). The script's `coalition_win: true` is only against
  *signed* Taylor (0.115±0.132), a strawman (Molchanov's criterion is squared/absolute).
- **The one book-like signal that wins is exact output settlement** (the retarget's committed-trader
  rule): neurons 0.464±0.041 final, 0.458±0.043 at 20k, vs |Taylor| 0.103 / 0.220. Heads/readouts untested.
- **Sign problem confirmed with ground truth (tracr):** signed first-order credit and signed books
  put negative credit on 42-67% of units known to be important; exact output settlement puts 0%.

## 1. Oracle (worker 1, `w1/ORACLE.md`, `oracle.py`)
Exact held-out ablation of 512 neurons, 8 heads, 64 readouts on bits T..T+4095 (T=24000), at 8k,
20k, 30k and final (31.1k/31.2k/31.1k), plus 250 coalitions per seed at 20k and final (200 random,
50 structured). Intact BCE equals wealth-net `ablation.csv` to 1e-6 in all 12 runs
(`w1/results/oracle_checks_s*_step*.json`); it reads that file and does not recompute it.
Split-half reliability, neurons: 0.945 (8k), 0.925 (20k), 0.916 (final) (`w1/results/summary.json`);
the oracle stop rule (<0.5) is not triggered. Heads are low-power (n=8; final split-half 0.73).
The neuron ranking moves between 8k and 20k (ρ vs final 0.71±0.10, `w1/ORACLE.md` from `w1/results/summary_step_stability.csv`),
so every predictor is scored against the oracle of its own checkpoint.

## 2. Spearman vs oracle Δloss (`w3/results/spearman_summary.csv`)
Neurons, all four checkpoints (the lead reran `w3/metrics.py` after w1's 8k/20k/30k landed; final
values unchanged from w3's draft):

| predictor | 8k | 20k | 30k | final |
|---|---|---|---|---|
| abl_train (exact ablation, training segment) | 0.560±0.085 | 0.492±0.116 | 0.439±0.113 | 0.412±0.106 |
| settle_loo_train (exact output settlement) | n/a | 0.458±0.043 | n/a | **0.464±0.041** |
| absag_int (accumulated abs(a·g)) | 0.421±0.166 | 0.354±0.090 | 0.322±0.128 | 0.313±0.115 |
| v2_lmsr_inst (book) | 0.277±0.099 | 0.226±0.029 | 0.227±0.052 | 0.199±0.075 |
| v2_lmsr_int (book, integrated) | 0.277±0.099 | 0.149±0.028 | 0.162±0.045 | 0.162±0.044 |
| v1_wealth (wealth-net book) | 0.172±0.009 | 0.065±0.067 | 0.060±0.070 | 0.060±0.063 |
| taylor_abs | 0.386±0.168 | 0.220±0.086 | 0.138±0.104 | 0.103±0.084 |
| taylor (signed) | 0.211±0.174 | 0.171±0.059 | 0.137±0.159 | 0.076±0.053 |
| movement | 0.162±0.035 | −0.061±0.054 | −0.077±0.077 | −0.089±0.082 |
| wout_norm (magnitude) | 0.131±0.039 | 0.020±0.037 | −0.015±0.054 | −0.022±0.066 |

_inst = _int at 8k because wealth-net saved no weights before 8k (`w2/PREDICTORS.md`). settle_* exists
only at 20k and final (`w3/results/pred-s*-settle-step*.csv`).

Heads and readouts, final:

| predictor | heads | readouts |
|---|---|---|
| abl_train | 0.929±0.083 | 0.742±0.020 |
| shap_head_exact | 0.865±0.060 | n/a |
| absag_int | 0.714±0.095 | −0.754±0.014 |
| v2_lmsr_inst | 0.698±0.339 | 0.187±0.033 |
| v2_lmsr_int | 0.667±0.401 | 0.303±0.022 |
| v1_wealth | 0.325±0.387 | n/a |
| taylor_abs | 0.809±0.143 | −0.750±0.017 |
| taylor (signed) | 0.111±0.730 | n/a |
| movement | 0.571±0.227 | 0.814±0.027 |
| wout_norm | n/a | 0.724±0.049 |

At 20k: heads v2_lmsr_inst 0.675±0.387 vs taylor_abs 0.762±0.063; readouts v2_lmsr_int 0.353±0.017 vs
movement 0.824±0.011; the same verdicts as final. Readout training magnitudes anti-correlate because
the late checkpoints memorise (held-out BCE 1.04-1.06 in `w1/ORACLE.md` vs train ~0.10-0.14 in `w2/PREDICTORS.md`): zeroing a readout often lowers held-out loss.

## 3. Stop rule (`w3/results/stoprule.json`, rule in `w3/PREREG.md`)
Rule: tie of the best book with Taylor or movement within seed sd at every granularity, at the final
checkpoint → report and stop the pruning work.
- Literal result at final: neurons **win** (diff 0.0959, margin 0.0839), heads **tie**, readouts **loss**.
- The neuron win holds at final only: at 20k (0.226±0.029 vs 0.220±0.086) and 30k (0.227±0.052 vs
  0.138±0.104, diff 0.089 < 0.104) it is a tie, and at 8k |Taylor| is ahead. It also reverses on the
  frontier (§5) and vanishes on coalitions (§4).
- **Decision (lead): stop the pruning work for the books.** Nowhere is a book better than |Taylor|/movement
  in a way that survives a second checkpoint or a second metric. Carry forward exact output settlement only.

## 4. Coalitions and Gate B (`w3/results/coalition_summary.csv`, `additivity_summary.csv`)
Additivity: Spearman(Σ single Δ, coalition Δ) is 0.974 on random coalitions at final (0.963 at 20k),
0.900 on structured (top-k) sets; median |gap|/|Δ| 0.181; 0.425 of random coalitions superadditive.
The top units are subadditive (redundant): top-32 neurons' ΣΔ 2.9/2.4/4.6 nats vs joint 1.66/1.99/2.51
(`w1/ORACLE.md`, `w1/results/summary_coalitions.csv`). Single-unit Δ already ranks the bulk; coalition
credit can matter only among the top few units. (w1's 0.975±0.002 is random *neuron* sets only; w3's
0.974 pools all random kinds.)

Spearman(Σ predictor score, coalition Δ), random coalitions:

| predictor | final | 20k |
|---|---|---|
| shap_head_exact (exact, heads) | 0.935±0.028 | n/a |
| abl_train | 0.908±0.033 | n/a |
| v2_lmsr_int (best book) | 0.818±0.027 | 0.833±0.043 |
| settle_loo_train | 0.816±0.050 | n/a |
| movement | 0.804±0.023 | n/a |
| taylor_abs | 0.800±0.010 | 0.822±0.024 |
| taylor (signed) | 0.115±0.132 | n/a |

Gate B (plan §1: "M2 shows a coalition win") = a book beats Taylor by more than seed sd in 3/3 seeds.
Per-seed diffs (`lead/gate_b.json`) at final, v2_lmsr_int minus |Taylor|: +0.005/+0.057/−0.007 (sd 0.027); at 20k
−0.004/−0.018/+0.056 (sd 0.043). **Not a win.** `stoprule.json` → `coalition_win.any = true` comes only
from `v2_lmsr_* vs taylor` (signed; diffs 0.52-0.82). At 20k the books also beat movement in 3/3 seeds
(diffs 0.163/0.126/0.094, sd 0.077; `lead/gate_b.json` from `lead/gate_b.py`), but not
|Taylor|. **gate_flags.coalition_win = false.**

## 5. Pruning frontier (`w3/results/frontier_summary.csv`, `frontier-final.png`, `frontier-20000.png`)
Zero-ablate the lowest-ranked k% of neurons, held-out ΔBCE on T+6144..T+8191 (disjoint from the oracle,
from w2's `_val` data and from training); AUC = mean ΔBCE over 0-40%, lower is better.

| predictor | final | 20k |
|---|---|---|
| oracle_zero (ceiling) | 0.003±0.061 | not run |
| abl_train | 0.109±0.030 | n/a |
| absag_acc | 0.155±0.077 | n/a |
| settle_loo_train | 0.170±0.018 | 0.146±0.009 |
| taylor_abs | 0.200±0.044 | 0.159±0.114 |
| movement | 0.295±0.129 | n/a |
| random | 0.451±0.218 | 0.485±0.188 |
| v1_wealth | 0.580±0.372 | n/a |
| v2_lmsr_inst | 0.863±0.645 | 0.723±0.805 |
| taylor (signed) | 0.939±0.824 | n/a |

(n/a = in the CSV but not quoted here.) Signed books and signed Taylor prune worse than random: they
give negative credit to important neurons, which are then removed first. Settlement is within sd of
|Taylor| and abs(a·g) on the frontier. Heads prune almost free for every non-random ranking; pruning
readouts *lowers* held-out BCE (oracle −0.420±0.005, `w3/results/frontier_summary.csv`), and only held-out signals find those readouts.

## 6. Ground truth on compiled networks (worker 4, tracr; `w4/PREREG.md`, `w4/results/`)
Four tracr programs (https://github.com/google-deepmind/tracr, https://arxiv.org/abs/2301.05062) with
planted classes: compiled circuit, padding, redundant copies, distractors, harmful noise units. Whole
input space, so the oracle is exact. Stage 1 (fixed weights, β=5, pooled without dyck, which missed the
100%-accuracy bar; `w4/results/summary_metrics.csv`):
- Used vs unused (AUROC): taylor 0.99 on neurons; magnitude fails (0.81 neurons, 0.32 heads);
  exact output settlement (`loo_direct`) 0.60, because it cannot see units that write only to
  intermediate subspaces.
- Share of truly important units given negative credit: signed taylor 0.42±0.12 (neurons), 0.67±0.29
  (heads); loo_direct 0.
- Spearman with the live oracle, neurons: taylor 0.26±0.06; the mean-ablation oracle 0.87±0.04.
Stage 2 (perturb, then fine-tune 2000 Adam steps, hist/sort_unique/reverse; `w4/results/s2_summary_metrics.csv`):
signed books put negative credit on 0.57±0.02 of important neurons and 0.60±0.20 of heads (taylor 0.54±0.07);
their neuron Spearman is 0.05±0.02, while abs(books) reaches 0.81±0.03. On neurons, signed books and
movement are rank-identical (Spearman 1.0 over all stage-2 neurons, `lead/gate_b.json`), the a·g = movement identity
from the plan's premises.

## 7. Fixes the lead made in the merge
- Reran `w3/metrics.py` over all 4 checkpoints (the draft had final only); final numbers unchanged
  (`stoprule.json` byte-identical).
- `w4/analyze.py` crashed in stage 2 (no `oracle_mean` column); guarded it and reran, which writes
  `w4/results/s2_*` and `w4/results/analyze_out.txt`. Worker 4 wrote no `TRACR.md`; §6 is its summary.
- Added `oracle.py` (entry point the plan names) and `check_bakeoff.py` (re-reads every quoted number).

## 8. Not done / unverified
- Frontier oracle row at 20k and frontier at 8k/30k: not run (cost; w3 amendment 1).
- Settlement for heads and readouts does not exist; the verdict there is "missing".
- Heads: n=8 per seed, verdicts low-power.
- Mean-ablation tables are in the same CSVs (`ablation=mean`); the verdict uses zero ablation as preregistered.

Reproduce (from `study/M2/`, pylib `P=/home/agent/workspaces/eurisko-study/wealth-net/.pylib`):
`PYTHONPATH=$P python3 oracle.py 1 --steps final` · `cd w3 && PYTHONPATH=$P python3 metrics.py ../w2/results/pred-s*-step*.csv results/pred-s*-settle-step*.csv`
· `PYTHONPATH=/tmp/traders-net/M2/w4/pylib:$P python3 w4/analyze.py` · `python3 check_bakeoff.py`.

Prior art: Taylor importance, Molchanov et al. 2019 https://arxiv.org/abs/1906.10771 · movement pruning,
Sanh et al. 2020 https://arxiv.org/abs/2005.07683 · integrated gradients https://arxiv.org/abs/1703.01365 ·
conductance https://arxiv.org/abs/1805.12233.
