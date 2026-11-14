# X1 worker 2 pre-registration: skeptic + reproduction for M1, M2, M3

Written 2026-09-28 12:06Z, before any run. Claim: `../STATUS.md` 12:02:13Z (M1, M2, M3).

## Checks (no new runs)
C1. Every number in the orchestrator's M1/M2/M3 summaries is re-read from the file it names
    (`check_claims.py`, writes `results/claims_check.json`). Match = equal at the quoted rounding.
C2. Each workstream's result is compared with its PREREG (arms, metric, seeds, stop rule honoured).
C3. Skeptic questions: train/eval overlap, seed variance and effective n, shuffled/kill controls,
    and whether the gate/stop-rule reading follows the plan's literal text.

## Reproductions (fresh seed, clean /tmp copy of the committed scripts)
- R1 (M1 a, toy): `M1/lead/toy_sign.py` `simulate()` with fresh seeds 4, 5, 6 (Adam and SGD);
  closed form recomputed. Headline: redundant-unit credit, LOO settlement, ablation dBCE.
- R2 (M2 headline + M1 b + M3 A, one shared fresh oracle): `M2/w1/oracle.py` copied and patched
  only in (i) the held-out segment, bits T+8704..T+12799 (disjoint from the oracle segment T..T+4095,
  w3's frontier T+6144..8191 and w2's _val T+8192..8703, and from training < T), (ii) the coalition
  RNG (seed*1000+step+9001), (iii) zero ablation only. Final checkpoint, wealth-net seeds 1-3
  (no new wealth-net training exists; the fresh seed is a fresh evaluation sample and coalition draw,
  not a fresh training seed; disclosed as such). Then `M2/w3/metrics.py` unchanged with
  M2_W1/M2_OUT pointed at the fresh oracle. Headlines: neuron Spearman of v2_lmsr_inst, taylor_abs,
  settle_loo_train; heads; readouts; Gate B random-coalition diffs. The same fresh oracle re-derives
  M1(b) (top-5% neurons with negative all-time book credit vs LOO settlement) and M3 part A
  (Spearman of all-time wealth, gradient norm, settlement).
- R3 (M3 B): `M3/w2/lockin.py` fresh seeds 4, 5, 6 and `aggregate.py`-equivalent summary.
  Headline: post-shift loss kelly vs fs1e-3, lag, decision rule.
- R4 (determinism): rerun the analysis scripts (M1 sign_census, M3 analyze.py) in /tmp copies and diff.

## Metric and rule
Verified: the fresh-seed value lies within 2 sd of the claimed mean (sd = the claim's seed sd; for a
fresh 3-seed run, the fresh mean within 2 sd) and the qualitative claim (sign, ordering, gate) holds.
Refuted: outside 2 sd and the qualitative claim fails. Unverified: no file, or no fresh-seed rerun
possible within budget. 3 fresh seeds where a seed exists; mean +- sd (ddof 1).

## Stop rule / budget
CPU only (no G1 jobs; X1 GPU budget left to siblings). At most 3 concurrent CPU jobs, 1 thread,
only while study jobs < 12. Stop R2 if one seed exceeds 3 h wall; report partial. Outputs: CSV/JSON
under `study/X1/w2/results/`, bulky files under /tmp/traders-net/X1w2/.
