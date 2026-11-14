# E-SUD round 3 results: deep contradictions, and believed mass vs count

Run 2026-10-15 09:24-10:12Z. The spec is gemini's `overnight-followups-round-3.md`, Rank 1, relayed by opus-mats
(am #3643, #3650) after Kelvin asked for new experiments with gemini. The nets are the round-1b 150k-step ones,
with no retraining. The pre-registration, `PREREG-R3.md`, was saved at 09:39:45Z and hashed at 09:40Z. Its header says "~09:50Z";
that is wrong, before any chain ran on the hard split
(`PREREG-R3.sha256`; frozen copy of the code at `results/sud.py.r3-frozen`).

## Verdict

- **R3-D1a (one pass fails on deep contradictions): PASS.** On unsolvable boards whose contradiction needs at least 3
  rounds of singles, P0 log loss is 1.84 nats (1.97 / 1.94 / 1.63 per seed). On solvable boards it is 0.019. The
  one-pass net calls these unsolvable boards solvable.
- **R3-D1b (null search fixes them; tau = 1, M = 8): PASS for all three selectors.**
  - Unsolvable-board log loss falls from 1.84 to 0.006-0.012: a 99.4-99.6% cut against both P0 and witness-only.
    The CI lower bounds are at least 98.6%.
  - Solvable boards are cut 99.99%.
  - No unsolvable board ever got a false witness.
- **R3-D1c (untempered believed mass beats chain count): FAIL, 0 of 6 conditions.**
  - Part 1, ranking: believed mass does rank unresolved boards. AUC(-C^bel) at M = 1 / M = 2 is:
    - 0.87 / 0.93 for P3 at tau 1.5, and 0.87 / 0.89 for P3 at tau 2.0. Both P3 conditions pass part 1.
    - 0.76 / 0.85 for P1 at tau 1.5, and 0.71 / 0.81 for P1 at tau 2.0. These fail at M = 1.
    - The mixed bag's first chain is P1, so it matches P1 at M = 1.
  - Part 2, log loss: the floored mass update never beats the floored count control by 10% with a CI above 0. At
    M = 1 it is 1.5-4.0x worse; at M = 2 and 4 it is mixed and within noise.
- **Why:** almost every theta fit sits at the grid's upper edge (1000, with eta_max 0.9999). Validation asks for
  "any refuted chain means unsolvable", and the one-parameter (1 - C)^theta curve can only approximate that through
  theta. The count control states it directly. Mass carries ranking information, but this likelihood form does not
  turn it into lower log loss.
- **The 20-60% recall regime was not reached.** Even on the hard split, per-chain witness recall on solvable boards
  is high:
  - tau 1: 85% (P1), 97% (P2, P3);
  - tau 1.5: 80% (P1), 95% (P3);
  - tau 2: 73% (P1), 90% (P3).

  P1 figures are 3-seed means as recomputed by the audit (85 / 80 / 73%). An earlier draft quoted seed-1 values.

  By M = 4, 98.6-99.9% of solvable boards have a witness. Few boards stay unresolved at M = 2 and 4: 0-53 solvable
  ones per seed. So part 2 of D1c rests on very few boards. The pre-registered statistic was still applied as written.

## Setup

- **Hard split** (`code/r3hard.py`, spec filters): 1,000 validation and 1,000 test boards, 50/50, with 10-15 givens.
  - Unsolvable boards need `prop_depth >= 3`. Test depths: 3 = 392, 4 = 69, 5 = 8, 999 = 31.
  - Solvable boards need 2-400 completions and a dead-end rate of at least 0.25 over 10 random-MRV rollouts (mean
    0.39).
  - Getting 500 unsolvable boards per split took about 220k candidate corruptions.
  - Clue counts: 11.7 (solvable) vs 12.2 (unsolvable).
- **Chains.** K = 16 per board, val and test, seeds 0-2.
  - tau 1: P1 random, P2 VoI, P3 confidence-first. tau 1.5 and 2: P1 and P3. Mixed bag: P1 and P3 chains
    alternate.
  - Each chain records tempered proposal log-mass (logq) and untempered believed log-mass (logp_bel).
  - C^bel and C^prop are summed over distinct refuted completions.
- **Arms.**
  - P0; witness-only.
  - null_floor: (1 - eta) + eta (1 - C^bel)^theta.
  - null_floor_prop: the same with C^prop (secondary).
  - count_floor: (1 - eta) + eta exp(-kappa M).
  - Fits are on hard validation only.

## Numbers (test; per seed unless marked as a mean)

R3-D1b at M = 8, tau 1 (mean cut, seed-t 95% CI):

| selector | unsolvable vs P0 | unsolvable vs witness-only | solvable vs P0 |
|---|---|---|---|
| P1 random | 99.38% [98.99, 99.77] | same | 99.99% |
| P2 VoI | 99.58% [98.59, 100.56] | same | 99.99% |
| P3 confidence-first | 99.64% [98.86, 100.42] | same | 99.99% |

Total Solvable log loss: null_floor (believed mass) vs count_floor; seeds 0 / 1 / 2.

| condition | M = 1 | M = 2 | M = 4 |
|---|---|---|---|
| tau 1.5, P1 | 0.22 / 0.31 / 0.25 vs 0.11 / 0.12 / 0.16 | 0.064 / 0.093 / 0.085 vs 0.045 / 0.078 / 0.091 | 0.014 / 0.018 / 0.008 vs 0.014 / 0.016 / 0.008 |
| tau 1.5, P3 | 0.105 / 0.131 / 0.053 vs 0.036 / 0.046 / 0.035 | 0.015 / 0.029 / 0.008 vs 0.017 / 0.018 / 0.025 | 0.008 / 0.006 / 0.005 vs 0.000 / 0.006 / 0.000 |
| tau 2, P1 | 0.41 / 0.47 / 0.35 vs 0.13 / 0.14 / 0.18 | 0.15 / 0.21 / 0.16 vs 0.09 / 0.13 / 0.15 | 0.038 / 0.056 / 0.024 vs 0.022 / 0.034 / 0.025 |
| tau 2, P3 | 0.25 / 0.28 / 0.15 vs 0.072 / 0.070 / 0.059 | 0.067 / 0.075 / 0.034 vs 0.040 / 0.028 / 0.016 | 0.009 / 0.009 / 0.005 vs 0.014 / 0.006 / 0.005 |

D1c part 2 (mass vs count cut, mean and seed-t CI) at M = 2 / M = 4:

| condition | M = 2 | M = 4 |
|---|---|---|
| tau 1.5, P1 | -19% [-81, 43] | -3% [-18, 12] |
| tau 1.5, P3 | +5% [-161, 170] | -4100x (one seed: count at 0.000 vs mass at 0.008) |
| tau 1.5, mixed | -27% | +8% [-53, 69] |
| tau 2, P1 | -50% | -44% |
| tau 2, P3 | -122% [-252, 9] | -12% |
| tau 2, mixed | -163% | -150% |

The tempered proposal mass (null_floor_prop) does no better. Its best cut is +38.5% (tau 1.5 P3, M = 2), CI
[-42, 119].

AUC(-C^bel) on boards still without a witness (3-seed mean; unresolved solvable boards per seed in brackets):

| condition | M = 1 | M = 2 |
|---|---|---|
| tau 1.5, P1 | 0.76 [102-123] | 0.85 [26-31] |
| tau 1.5, P3 | 0.87 [20-34] | 0.93 [2-8] |
| tau 2, P1 | 0.71 [138-153] | 0.81 [48-53] |
| tau 2, P3 | 0.87 [46-56] | 0.89 [8-14] |
| mixed | as P1 at M = 1 | 0.86 (both taus) |

## Diagnostic: the first round-3 step (superseded "20-60% recall" filter)

This used a pool of 1,800 boards (k = 6-20, 50/50). At tau 1, per-chain witness recall on solvable boards was
0.78-0.99 for P1 and at least 0.97 for P3 at every clue count. Fewer givens give more completions and do not lower
recall. Recall reached 20-60% only by tempering the proposal:
- tau 3: 0.53 (P1) and 0.60 (P3);
- tau 4: about 0.2.

The own check was 100% accurate on that pool. Files: `data/r3/pool_val.npz`, `results/r3/pool/`.

## Deviations

1. The opus-mats cap note changed from $15 (#3643) to $5 (#3650); the $5 cap was used.
2. The spec leaves the mixed bag's temperatures open ("1.5-2.0 / 1.0-1.5"). The pre-registered choice was P1 and P3
   alternating at the same tau.
3. The spec does not say how R3-D1c combines conditions. The pre-registered choice is strict: all 6 conditions must
   pass. Under any looser reading the result is the same, since no condition passes both parts.
4. The theta fits hit the grid's upper edge (1000) in nearly every condition. The grid was not widened after looking.

## Cost and cleanup

Round 3 used one on-demand L4 (`esud-r3`) for 0.70 h, $0.61 at the ceiling, against a $5 cap. It was deleted at
10:08Z. `gcloud compute instances list --filter=name~esud` and `gcloud compute disks list --filter=name~esud` were
both empty afterwards.

## Audit

`AUDIT-R3.md`: supported, with minor corrections. An independent numpy scorer reproduces every fit and every
verdict. The split was re-checked on all 2,000 boards. The corrections are applied above (M = 1 ratio, the
proposal-mass best case, P1 recall means, the pre-registration timestamp, unresolved counts). Clue counts are not
matched (AUC of -k is 0.61); the pre-registration discloses this.

## Files

- `PREREG-R3.md`, `PREREG-R3.sha256`
- `code/r3hard.py`, `code/r3.py`, `code/sud.py` (logp_bel, `ESUD_SPLIT_PREFIX`)
- `data/r3/hard/`, `results/r3/chains/`, `results/r3/R3-ANALYSIS.json`
- Audit: `AUDIT-R3.md`
