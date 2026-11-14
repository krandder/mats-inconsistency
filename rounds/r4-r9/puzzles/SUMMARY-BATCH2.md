# Trader-economy puzzles, batch 2 (P8-P12): valuing hidden variables

For Kelvin. Sources, read 2026-09-29: `CATALOGUE-PUZZLES.md` entries P8-P12 and each experiment's `RESULTS.md`. Two independent verifiers checked each experiment, and they agreed with every status below. For the P0-P7 batch, see `SUMMARY.md`.

Spend: about $17.15, or $21.09 at the rate ceiling. P11 went over its $8 cap at the ceiling rate.

## Bottom line

Five experiments, zero gains. Across both batches the count is twelve experiments and zero gains.

| Outcome | Experiments |
|---|---|
| Informative | P9 (randomized clamping), P10 (masked inputs), P11 (wealth cap as trust region) |
| No gain | P8 (pay by bits saved), P12 (flow-conservation arbitrageur) |

The one live lead again comes from outside the market: learned local costs plus an exact solver. It solved 27% of wide mazes exactly, against about 3% for DT, but it rests on one seed.

## Importance versus learnability (P8)

- **The principled price ranks worse than importance alone.** The price is output bits saved minus the bits it costs to learn Z (R4). Its Kendall tau against deployed use is 0.33. Importance alone scores 0.70, and plain validation scores 0.83.
- **The learnability charge is far too large.** For deep intermediates, learning Z costs tens of thousands of bits, so R4 ends up ranking by learnability. The best exchange rate was about 0.06 bits of learnability per bit of importance, not 1.
- **A copy of Y.** R4's one real virtue is that it prices an exact copy of Y at zero by construction: the bits saved equal the bits needed to learn it. Importance alone would pay a copy the maximum. Our deployed-truth ruler did not settle this, though. It rewarded a 5%-noisy copy of Y (nz05) above the real intermediate g2. Our ruler failed the very case you asked about.
- **Proposing the variables is the binding problem, not pricing them.** The non-oracle pool proposed nothing worth listing: all 16 first-level prices were negative. The oracle tower missed the BFS ladder, and on parity the tower recovered the structure in 0 of 3 seeds, where brute force did in 3 of 3.
- **The tower question is still untested.** Your question was whether a tower built backwards from the outputs reaches the input. We tested it only through a unit test (U6), which the verifiers found to be a tautology.

## Randomized clamping, or futarchy on z (P8 R6/R6b, P9)

- **The math.** Clamping z at random and voiding the mismatched trades is the same as revealing the true z, plus importance-weighting variance. On the net's own latents it reduces to teacher forcing on itself. There is nothing new to learn from it, and it adds noise.
- **On exogenous variables it measures correctly, but only after a retune that saw the test numbers.** The pre-registered recipe missed every bar.
- **On the net's own latents it measures nothing useful.** Clamp pay found must-pass cells with AUROC 0.45. A free local heuristic scored 0.76.
- **As a training signal it showed nothing.** The result was a null at the accuracy floor. The A/A check also failed (same-seed reruns differed by 4 points), so the training null cannot be trusted either way.
- **The cheap forward-pass version (R6).** It reproduces R4's prices at 2-4% of R4's CPU, but R4's prices are the bad ones. The literal void market (R6b) keeps about 115 of its 4,088 effective samples and cannot rank.
- **Verdict.** Clamping does not find "important and easy to learn" variables. As a measuring tool, it is a noisier way to reveal z.

## Masked inputs and the minimal Bayesian representation (P10)

- **Choosing what to reveal.** A learned head that picks which masked cell to reveal did worse than counting masked neighbours (rank correlation 0.03 vs 0.19). Branching was never built.
- **Learned events.** Every learned event collapsed to a constant, and the planted positive was de-listed in all 5 seeds. This may be a bug: 9 of 15 encoders were already constant at initialisation.
- **The auction's royalty tracks what the decoder leans on, not what helps.** The one bit that survived made the likelihood bound worse in 3 of 4 seeds.
- **The minimal-Bayes frontier is reachable by a known method.** A plain any-order autoregressive model reaches 1.016 nats against an entropy of 1.007 in 48 passes. Loss against passes is already the "minimal representation under compute cost" curve. It needs no market.

## Trust regions and consistency traders (P11, P12)

- **The LMSR wealth cap is a hard stop, not a trust region.** 98-99% of on-path pixels exhaust it.
  - It beats the total-variation cap by 8 points, but only because it hurts less. It ties the uncapped model.
  - Known fixes beat it by 13 points (DT-L) and 58 points (progressive loss, on its trained seeds).
  - There was no overthinking to fix in the first place.
- **The hard-coded flow-conservation trader helps only in distribution** (66% vs 57% exact). Plain post-hoc decoding does better (73%) with no training change.
  - On wide mazes it is null (2.5% vs 3.1%).
  - Its prices can tell when a maze is wrong (AUROC about 0.86 at fixed size), but they cannot repair it.

## Process, harshly

- **Failed gates did not stop the runs.** The P9 A/A check, the P11 sensitivity gate and the P12 9x9 gate (60% against a 90% bar) all failed, and the runs went ahead. Batch 1 had the same fault, and it has not been fixed.
- **Designs were underpowered.**
  - P9's seed spread is 13 points.
  - P12's true detectable effect is 22-25 points, not what its design stated.
  - Several arms sat at the accuracy floor.
  - Most nulls here are weak nulls.
- **Retunes and controls came after seeing the test numbers.** The P9a retune did, and the P10 positive controls sat beside the pre-registered results. P8 reused earlier single-run files without disclosing it.
- **The runners made factual errors, and the verifiers caught them.** In P12 the runner reported the wrong BP variant, channel count and training width. In P11 the runner switched the cost-cap rate mid-run. Keep verification mandatory.
- **The rulers themselves were weak.** P8's truth metric rewards a noisy copy of Y, and P10's G1 target is mostly noise (rho -0.009).

## What next

1. **Stop testing pay rules as training signals.** In P3, P4, P8, P9 and P10, no pay rule beat a cheap selector or a free heuristic. Keep prices only as diagnostics, and always compare them first with the cheap selector.
2. **Run the cheap bug checks before believing two of the nulls.**
   - P10: rerun with a non-degenerate encoder initialisation, or with the rate term off during warm-up. This shows whether events still collapse.
   - P8: rerun with a pre-registered gain-over-base truth at N = 4096, where the baselines are strong. This shows whether learnability should count once there is more data.
3. **Treat learned local costs plus exact clearing (CostNet+BB) as the only live lead.**
   - Run at least 3 seeds at the trained width, with the seed count set by a power calculation.
   - The market reading: traders sell local, checkable costs, and an exact solver clears them.
   - The idea is known (Vlastelica et al. 2020). The open question is only whether the market framing adds anything to it.
4. **Hard rules for any further run.**
   - Baseline gates stop the run automatically.
   - Seed counts come from a power calculation.
   - No retuning after seeing test numbers.
   - Cost caps are checked at the ceiling rate.
5. **The tower-to-input question needs a proposer that can list candidate intermediates without an oracle.** P8 shows ours cannot.

**Decision needed from Kelvin:** fund the CostNet+BB seeds and the two bug checks, or close the hidden-variable pricing line. Our whole P12 run cost $4.88, which gives a sense of scale.
