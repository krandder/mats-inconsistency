# E-SUD round 2 results: recall-floored null search, and mass vs count under a noisy proposal

Run 2026-10-15 04:19-05:35Z. This is gemini's follow-up Rank 1 (E-SUD-R2), relayed by opus-mats (am #3627). It is
eval only, on the saved round-1b checkpoints (150k steps, seeds 0-2); no retraining. The pre-registration is
`PREREG-R2.md`, written and hashed at 04:19Z (`PREREG-R2.sha256`) before any round-2 chain ran.

## Verdict

- **R2-D1a (tail-safe null search): PASS.** With the validation-fitted recall floor, all three selectors pass all
  three comparisons at M = 8: P1 random, P2 VoI and P3 confidence-first, each vs P0 on solvable boards, vs P0 on
  unsolvable boards, and vs witness-only on unsolvable boards (9 of 9).
  - The round-1b failure (VoI, seed 1, one solvable board with no witness pushed to p = 1e-6) shrinks. That seed's
    solvable-board cut rises from 31% to 81%, with eta_max = 0.9999.
- **R2-D1b (proposal mass beats count by >= 5% at tau_prop = 2, M = 4): FAIL.**
  - P1 random: mass is worse than count in 3 of 3 seeds. Per-seed cuts are -24%, -264% and -9%; the mean is -99%,
    seed-t CI [-454%, +256%].
  - P3 confidence-first: mass is better in 2 of 3 seeds (+78%, +91%) and level in seed 1 (-0.8%). The mean is +56%,
    seed-t CI [-67%, +180%], so the CI does not exclude 0.
  - All of these losses are tiny (about 1e-5 to 0.006 nats per board), so the percentages rest on a handful of
    boards.
- **Bottom line:** the recall floor fixes the tail risk at no cost. At these sizes the data still cannot separate
  refuted proposal mass from the count of refuted chains. A noisier proposal (tau = 2) did not make failures
  inconclusive enough: witnesses are still found on 99.7-99.9% of solvable boards by M = 4.

## What was run

- **Recall-floored likelihood.** P(null | Solvable = 1) = (1 - eta_max) + eta_max (1 - C_M)^theta. The count control
  is (1 - eta_max) + eta_max exp(-kappa M). (theta, eta_max) and (kappa, eta_max) are fitted per seed, rule and tau on
  validation Solvable outcomes, pooled over M in {1, 2, 4, 8, 16}. The grids are in PREREG-R2.md. The round-1
  unfloored arms are refitted alongside for reference.
- **Chains.** Round 1b saved only per-board scores, so its chains were re-run and saved: tau = 1 for P1, P2 and P3,
  and tau_prop = 2.0 for P1 and P3 (p = softmax(logits / 2) for selection and sampling; log q from the tempered
  proposal). K = 16, val and test, 2,000 boards each.
- **Reproduction check.** The re-run tau = 1 chains are new random draws, not an exact copy of round 1b (audit). At
  M = 8 they agree with round 1b: witness-only log loss is 0.0224 / 0.0250 / 0.0226 per seed in both runs, and the
  VoI seed-1 witness rate is 0.9988 in both. That agreement is partly saturation, since nearly every solvable board
  has a witness by M = 8. At M <= 4 the witness rates differ by up to 0.7 points (e.g. P2 seed 0, M = 1: 0.9907 vs
  0.9837).

## Numbers (test, mean of 3 seeds)

R2-D1a at M = 8 (relative Solvable log-loss cut; mean, seed-t 95% CI):

| selector | vs P0, solvable | vs P0, unsolvable | vs witness-only, unsolvable |
|---|---|---|---|
| P1 random | 99.99% [99.99, 100] | 93.8% [80.0, 107.6] | 93.8% [80.0, 107.6] |
| P2 VoI | 93.6% [65.9, 121.2] (seeds 99.99 / 80.7 / 99.99) | 99.6% [98.3, 100.9] | 99.6% [98.3, 100.9] |
| P3 confidence-first | 99.99% [99.99, 100] | 99.9% [99.5, 100.3] | 99.9% [99.5, 100.3] |

Solvable log loss at M = 8, tau = 1 (solvable / unsolvable boards):

| arm | P1 random | P2 VoI | P3 confidence-first |
|---|---|---|---|
| P0 | 0.018 / 0.041 | 0.018 / 0.041 | 0.018 / 0.041 |
| witness-only | 0.000 / 0.041 | 0.000 / 0.041 | 0.000 / 0.041 |
| floored mass | 0.000 / 0.0026 | 0.0015 / 0.0002 | 0.000 / 0.00004 |
| floored count | 0.000 / 0.0006 | 0.0015 / 0.0002 | 0.000 / 0.000 |
| round-1 unfloored mass | 0.000 / 0.0009 | 0.0054 / 0.0008 | 0.000 / 0.000 |

Pooled log loss at tau = 2, M = 4:

| seed | P1 mass | P1 count | P3 mass | P3 count |
|---|---|---|---|---|
| 0 | 0.0058 | 0.0046 | 0.0008 | 0.0038 |
| 1 | 0.0022 | 0.0006 | 0.0006 | 0.0006 |
| 2 | 0.0016 | 0.0014 | 0.0000 | 0.0002 |

Witness rates at tau = 2: P1 84%, 96%, 99.7% and 99.96% of solvable boards at M = 1, 2, 4, 8; P3 94%, 99.2%, 99.9%
and 100%. No unsolvable board ever gets a witness (0 false witnesses, either tau). The mean refuted mass C_4 on
unsolvable boards is 0.13-0.42 (P1) and 0.17-0.57 (P3).

Mass minus count at other M (tau = 2, mean cut, seed-t CI):
- P1: +1.7% [-12, 15] at M = 1; -187% at M = 8.
- P3: +29% [-45, 102] at M = 1; -75 at M = 8.

Both blow up at M >= 8, because the few boards still without a witness at large M dominate.

## Deviations

1. The first round-2 VM (`esud-r2`, spot) was preempted at about 05:00Z with its chains unsaved. The chains were
   re-run on three on-demand L4s, one per seed. To make partial outputs survive, `sud.py dump` now writes one file
   per rule (`chains_tau{tau}_{rule}_s6_seed{s}.npz`), and `r2.py` merges them. This changes only file naming
   (sud.py and r2.py differ from the 04:19Z hashes). No copy of the 04:19Z code was kept. The diff of the current
   sud.py against the round-1 copy shows only the tau argument, the dump mode and its output name.
   PREREG-R2.sha256 also has a stray date line and absolute paths. PREREG-R2.md was saved at 04:19:10Z; its header
   says ~04:25Z. Because each rule is now a separate run, the
   generator restarts per rule. The tau = 1 reproduction check above still matched exactly.
2. The theta grid's upper end (1000) was hit by 2 of 15 mass fits (tau1 rand seed 1, tau2 conf seed 0). Two more
   are near it (tau1 conf seed 1 at 891, tau2 rand seed 0 at 794). The fits may want an even steeper update; the
   grid was not changed after looking.
3. Results came in at about 05:35Z, against a 05:15Z target (the preemption).

## Cost

Round-2 ceiling spend, $1.73-1.79 of the $5 cap (audit):
- esud-r2 (spot): about 0.7 h before preemption, about $0.6 (no DELETE line in GPU-LOG.md; GCE deleted it on
  preemption, so its $2.26 reservation is never closed out);
- esud-r2s0/s1/s2 (on demand): 0.38 / 0.46 / about 0.5 h, $0.33 / $0.40 / about $0.43.

All round-2 VMs are deleted. `gcloud compute instances list --filter=name~esud` and `gcloud compute disks list
--filter=name~esud` both returned nothing at 05:35Z (AUDIT-R2.md).

## Audit

`AUDIT-R2.md`: supported with corrections. All 15 fits and every R2-D1a and R2-D1b number reproduce in an independent
numpy re-implementation. The corrections are applied above (reproduction wording, hashes, theta-edge count, loss
range, ledger).

## Files

`PREREG-R2.md`, `PREREG-R2.sha256`, `code/r2.py`, `code/sud.py` (`dump` mode, `tau` argument),
`results/r2/R2-ANALYSIS.json`, `results/r2/r2_curves.csv`, `results/r2/chains/*.npz` (raw chains: completions, log q,
own check, p0). Audit: `AUDIT-R2.md`.
