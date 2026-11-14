# M9(c): K/Q sweep in M6's trader layer (worker 3)

Worker 3, 2026-09-28 09:35Z. Prereg `w3/PREREG.md` (written before any run). Code `w3/run_kq.py` (imports
`../M6/w1/m6layer.py` and `run_w1.py` unchanged), `w3/summarize.py`. Raw runs `w3/results/vm/*.json` (42),
tables `w3/results/{SUMMARY.csv,LOTTERY.csv,EXLFSR.json,VERDICT.json}`, figures `w3/kq.png`, `w3/kq2.png`.
Values are mean ± sample sd over seeds 1-3; "paired" means per-seed differences (same stream, same init seed).
Rerun: `cd w3 && PYTHONPATH=/home/agent/workspaces/eurisko-study/wealth-net/.pylib python3 summarize.py`.

## TL;DR
- **The soup's K/Q law is absent in the net** (stop rule, `w3/results/VERDICT.json`: `trend=false` in all
  4 series, `law_present_B_FT=false`, `law_present_A_FT=false`; no family has lottery R² > 0.5 with a positive
  K slope, `w3/results/LOTTERY.csv`). The share of learnable questions priced (Brier < 0.1) is flat in K:
  periodic, ca110, prime are priced at K=1 already; LFSR16 is priced (Brier < 0.1) only by seed luck (1/3 seeds at B-FS and A-FS K=64; B-FT K=1 has 1/3 seeds
  "learnt" at BCE < 0.35 but not priced, `LOTTERY.csv` priced_lfsr16 = 0; lead fix); prng/ca30 never. With a shared trunk, K acts as layer width, not as independent search draws.
- **At fixed layer parameters K does nothing to learnable BCE** (excl. LFSR16, B-FT paired K64−K1:
  +0.0007/+0.0015/−0.0006 nats). At fixed trader size the gain from K=1 to K=16 (−0.010/−0.010/−0.023, 3/3) is
  the width-8 bottleneck of a single head: B-FT at K=1 (one width-128 head) already matches it
  (0.1905 ± 0.0267 vs 0.1885 ± 0.0282, `w3/results/EXLFSR.json`).
- **The only K-specific effect is negative: 64 traders per question memorise noise.** Noise excess
  max(BCE−ln2) on prng/ca30 rises from ~0.002 at K ≤ 16 to 0.0068-0.0072 at K=64, in all four series and 3/3
  paired seeds (+0.0039 to +0.0068 vs K=16). This holds at fixed layer parameters, so it is the per-(question,
  head) scalars (2·Q·K of them) and the sum of many unshrunk heads, not capacity in the shared part.
- Shared computation is almost free here: the trader layer is a small part of the step (FLOPs/step 1.79e8 at
  FS K=1 → 2.29e8 at FS K=64; 1.91e8 at every FT point, `SUMMARY.csv` `flops_step`); the trunk dominates.

## Design (from the prereg)
S0 streams Q=64 (8 families × 8), W=32, prequential single pass B=4, 6000 steps, scored on positions
[12000, 24000); M6 trunk d=64, H=4, L=2; market l0 = per-question logistic, detached. K trader heads per
question, head width hw. Arms: **B** z = l0 + Σd (M6's anchored layer) and **A** z = Σd. Series **FS** hw=8
(layer params 1,160 → 74,240) and **FT** K·hw = 128 (16,640 → 24,704; the 2QK per-question scalars are not
held fixed). K=16/hw=8 is shared by both series. 7 configs × 2 arms × 3 seeds = 42 runs.

## Results

| arm-series | K=1 | K=4 | K=16 | K=64 | evidence |
|---|---|---|---|---|---|
| learnable BCE, B-FS | 0.2847 ± 0.0239 | 0.2758 ± 0.0231 | 0.2729 ± 0.0235 | 0.2362 ± 0.0706 | `SUMMARY.csv` |
| learnable BCE, B-FT | 0.2520 ± 0.0449 | 0.2557 ± 0.0378 | 0.2729 ± 0.0235 | 0.2631 ± 0.0316 | `SUMMARY.csv` |
| learnable BCE, A-FS | 0.2763 ± 0.0224 | 0.2736 ± 0.0218 | 0.2718 ± 0.0219 | 0.2424 ± 0.0369 | `SUMMARY.csv` |
| learnable BCE, A-FT | 0.2698 ± 0.0227 | 0.2494 ± 0.0059 | 0.2718 ± 0.0219 | 0.2613 ± 0.0353 | `SUMMARY.csv` |
| excl. LFSR16, B-FS | 0.2026 ± 0.0287 | 0.1920 ± 0.0278 | 0.1885 ± 0.0282 | 0.1875 ± 0.0270 | `EXLFSR.json` |
| excl. LFSR16, B-FT | 0.1905 ± 0.0267 | 0.1886 ± 0.0266 | 0.1885 ± 0.0282 | 0.1911 ± 0.0264 | `EXLFSR.json` |
| excl. LFSR16, A-FT | 0.1849 ± 0.0272 | 0.1861 ± 0.0273 | 0.1872 ± 0.0263 | 0.1917 ± 0.0274 | `EXLFSR.json` |
| noise excess, B-FS | 0.0020 ± 0.0002 | 0.0018 ± 0.0001 | 0.0022 ± 0.0003 | 0.0068 ± 0.0006 | `SUMMARY.csv` humility |
| noise excess, B-FT | 0.0023 ± 0.0015 | 0.0019 ± 0.0014 | 0.0022 ± 0.0003 | 0.0072 ± 0.0008 | `SUMMARY.csv` |
| LFSR16 learnt (seeds), B-FS / B-FT | 0/3 / 1/3 | 0/3 / 0/3 | 0/3 / 0/3 | 1/3 / 0/3 | `SUMMARY.csv` lfsr_learnt |
| effective traders per q, B-FS | 1.00 | 3.67 ± 0.09 | 13.45 ± 0.21 | 46.5 ± 4.5 | `SUMMARY.csv` eff_traders_per_q |

- The raw learnable-BCE swings (e.g. B-FS K=64 0.2362 ± 0.0706) are one seed learning LFSR16 (seed 3, LFSR16
  BCE 0.042); M6 found the same seed-luck (`../M6/M6.md`). The stop rule's trend test (|K1−K64| > 2 pooled sd and
  monotone) fails everywhere (`VERDICT.json`).
- A-FT at K=64 (hw=2) is worse excl. LFSR16 by +0.0069/+0.0066/+0.0069 (3/3 paired): very thin traders cost.
- Heads do not collapse: the participation ratio of |d_i| stays at 66-93% of K (`eff_traders_per_q`).
- FLOPs to Brier 0.1: ca110 reaches it at 5-7e10 training FLOPs at every K ≥ 4 in both arms (`SUMMARY.csv`
  `flops_to_b01_ca110`); only B-FS K=1 is slower (2.2e11 ± 1.9e11). Macro learnable rolling Brier < 0.1 is
  reached by seeds 1-2 at positions 1.6k-8.4k, flat in K except B-FS K=1 (4.9k, 14.8k); seed 3 mostly never
  (`w3/results/vm/*.json` `pos_to_brier01`).
- AND/OR violations: 0.050-0.064 everywhere, no trend (`SUMMARY.csv` viol_all).

## Reading for Q7
In the soup, density K/Q mattered because each trader is an independent random program: more draws per
question = more chances one finds the pattern (M9(a) fits R² 0.79-0.94, `M9-a.md`). In the net every head reads
the same trunk residual, so the pattern is found (or not) by the shared trunk, once; adding heads only adds
readouts of the same features. The law is about search, not about markets. What the net's shared computation
buys at equal FLOPs is that K=1 already prices what K=64 prices; what extra traders per question cost is noise
memorisation (a humility loss), which is the M6 kill-control problem (boosting shrinkage), not a pricing gain.

## Deviations from the prereg
- The box's load was 45-55 (above the prereg's gate of 40), so all 42 runs ran CPU-only (CUDA hidden) on a G1
  VM (job `m9-w3-kq-sweep`, `../G1/queue/done/m9-w3-kq-sweep/status.json`), same code. Three local runs started
  earlier were discarded before any result was used (moved to `/tmp/traders-net/M9/w3/old/`).
- Per-family FLOPs to Brier 0.1 (`fam_pos_to_brier01`) was added to `run_kq.py` before the final runs; the
  prereg's macro version is also reported.
- Lottery R² is undefined for families constant across K (priced 0 or 1 everywhere); those are counted as
  "no law".

## Cost
GPU queue: $0.30 audited (`../G1/COST.md`, VM tn-gpu-m9-w3-kq-sweep-09280909-17599, deleted at 09:30Z);
wall 21 min. No local CPU runs in the result.

## Prior art
- Lottery-ticket hypothesis: https://arxiv.org/abs/1803.03635 (width as more "tickets"; here extra heads on a
  shared trunk are not independent tickets).
- Boosting with shrinkage as the fix for many unshrunk additive corrections: Friedman 2001,
  https://doi.org/10.1214/aos/1013203451 (unverified).
