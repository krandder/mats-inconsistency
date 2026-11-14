# Calibration (Phase 2), JS lane, 2026-09-27 (eurisko)

Runs: K=2^12 F=2^12 seeds 1,2 (1000 epochs, in progress), K=2^14 F=2^12 seed 1
(1000 epochs, in progress; restarted at 10:40Z on the sparse-aggregation build
1aa40da). b=64, free income 0, W0=2^20, 16 questions (4 blocks of A,B,A∧B,A∨B).

| run | epoch | Brier all | Brier periodic | Brier LFSR | Brier A∧B | coh(A∧B) ppm | births/epoch | volume | Gini ppm |
|---|---|---|---|---|---|---|---|---|---|
| k12 s1 | 99  | 0.2498 | 0.2487 | 0.2492 | 0.2494 | 1258 | 5 | 3.7k | 172k |
| k12 s1 | 599 | 0.2489 | 0.2440 | 0.2498 | 0.2490 | 5598 | 6 | 27k | 504k |
| k12 s2 | 249 | 0.2490 | 0.2423 | 0.2517 | 0.2490 | 29604 | 18 | 35k | 423k |
| k12 s2 | 499 | 0.2414 | 0.2156 | 0.2507 | 0.2241 | 82167 | 57 | 144k | 613k |
| k14 s1 | 99  | 0.2492 | 0.2492 | 0.2502 | 0.2474 | 5495 | 8 | 14k | 165k |
| k14 s1 | 299 | 0.2400 | 0.2138 | 0.2549 | 0.2205 | 56829 | 97 | 102k | 406k |

Reading (one seed each, no claims yet):
- Gate 1 (periodic family priced): passes, but takeoff needs 250-500 epochs, not
  100; once births/epoch climb past ~50 the periodic and A∧B Brier fall fast.
  Seed variance is large (k12 s1 is still asleep at epoch 600, s2 took off by 250).
- Gate 2 (shadow profit decays): not yet. Shadow momentum/pattern traders keep
  profiting; the market is still far from priced. Coherence violations on A∧B
  are rising (traders price A and B before anyone arbitrages the compound):
  the arbitrage opportunity is growing, which is what the arbitrageur lineage
  needs to appear.
- LFSR, CA, primality, pi families: untouched at 0.25 (the bitter-lesson curve
  question for Phase 3: does K, F or epochs move them).
- Throughput: JS lane 1.08 M instr/s at K=2^12 F=2^8; ~2.6 s/epoch at K=2^14
  after the sparse-aggregation fix (was 25 s).
- Takeoff is slow because random programs rarely emit orders (demands ~1.5k of
  4096 traders at epoch 0). Phase 3 candidates to speed selection without adding
  knowledge: larger K (more lottery tickets), a modest free income during the
  first 50 epochs, or a split threshold below 2×W0. Each is an ablation, not a
  default.

Decision: continue to 1000 epochs on all three; move Phase 3 to the native lane
(GPU VM smoke in progress, pool-o5fo8).

## 1000-epoch results (11:00Z), one seed each

| run | Brier all | periodic | LFSR | CA30 | prime | A∧B | A∨B | coh(A∧B) | births/ep | volume | genotypes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| k12 s1 | 0.247 | 0.240 | 0.252 | 0.250 | 0.243 | 0.242 | 0.251 | 0.019 | 12 | 93k | 3937 |
| k12 s2 | 0.233 | 0.204 | 0.257 | 0.286 | 0.191 | 0.201 | 0.251 | 0.098 | 3 | 334k | 4093 |
| k14 s1 | 0.222 | 0.198 | 0.258 | 0.309 | 0.129 | 0.167 | 0.253 | 0.094 | 106 | 476k | 16198 |

Reading:
- What the market learned is base rates and slow drift, not patterns: primality
  at 0.129 is about p(1−p) for the prime density at these n (constant "not
  prime" scores ~0.10-0.13); periodic at 0.20 is far from the 0 a period detector
  would give. No period-k lineage has appeared; the shadow pattern traders keep
  profiting linearly.
- CA30 gets worse than chance (0.31): rich lineages emit the same demand on every
  question, so a rule that wins on biased families drags the pseudo-random ones
  the wrong way. Ecological, expected, and a measure of how unspecialised the
  population still is.
- A∧B is priced (0.167) while A∨B is not, and the AND coherence violation sits
  at 0.09 with the shadow arbitrageur's cumulative profit still rising: the
  arbitrage opportunity is large and unexploited. No arbitrageur lineage yet.
- Seed variance is large at K=2^12 (s1 barely took off). K=2^14 is qualitatively
  different (births 100+/epoch from epoch 300). Phase 3 needs ≥3 seeds and K up
  to 2^16 or 2^18, which needs the native/GPU lane.
- Native CPU lane on a spot L4 VM (8 vCPU): 2.71 M instr/s, 2.5x the JS lane;
  the GPU itself was not used (no CUDA dispatch in the current program shape).
  Phase 3 blocker: put the epoch loop into Bend2's GPU fork-join form.

Phase 2 verdict: harness trustworthy (ledger exact, resume exact, shadow
references behave, metrics sane). Gates 1 and 3 pass, gate 2 (shadow profit
decays) does not yet at 1000 epochs. Proceed to Phase 3 with longer runs, more
seeds, larger K, and the GPU lane.

## Correction (17:05Z): the Phase 3 "F" axis is the number of questions, not fuel

`run.sh --F` sets the number of questions (logQ); fuel was 256 in every run so
far, including the calibration runs above (which were K=2^12/2^14 with 4096
questions, fuel 256). So the k14-f8 vs k14-f12 difference (periodic 0.084 vs
0.198, shadow traders losing vs profiting) is a traders-per-question effect:
64 vs 4 traders per question. K=2^16 with 4096 questions (16 per question)
lands in between (periodic 0.125-0.133). The bitter-lesson axis measured so far
is K/Q, not per-trader compute. A real fuel sweep (64, 1024, 4096 at K=2^12,
Q=256, 3 seeds; matrices/fuel.json) was started at 17:05Z. My 16:50Z line
"less fuel is better" was wrong and is withdrawn.
