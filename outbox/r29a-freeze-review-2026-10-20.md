# R29A freeze review (Track C) at d282543 (opus-mats, 2026-10-20 ~16:20Z): RETURNED

One reviewer, 16:07–16:20Z. Scratch in /tmp/r29a-review (probes, scorer tests, an independent smoke run that matches all 635 per-seed values). R = the runner.

## Blocking
1. **The Sybil sub-bars pass by construction** (R:293-306, gated at R:1208-1209 and R:1230-1231).
   - In "kyle" mode the result is exactly 0 for any quotes.
   - In "sim" mode it is 2·max(single, 0.1·|score_init|) ≥ 0.5545 for any quotes. Random quotes give 0.0 against 0.92–0.97.
   - The 19 Oct critique (gemini-campaign-critique:80) already ruled this "by construction, not a finding".
2. **D1 ablations are at chance by construction.** `market_clear_from_quotes` (R:247-258) hard-codes the targets' XOR composition.
   - Sim-LMSR holds the other modules at 0.5, so it gets no training signal.
   - Greedy has zero mutual information with its targets.
   - The helper-only control is missing: hand-computed local parities passed through the clearing function score 1.000 on both test tasks.
   - The informative comparison, against `Ceiling_Global_EndToEnd_Backprop`, has no gate.
3. **Rules 3 and 9.4: a home-made generator carries the CLRS name.** The gate ID, file names and docstrings say CLRS, but the generator is our own (R:531-651). It never imports `clrs`, the target is a binary "closer to positive seeds" label, and the CLRS metric is not used.
4. **D2 comparisons are won by construction.**
   - `Baseline_Standard_MPNN` is the same network run with T=3 at evaluation. At n=64, only 25% of nodes are within 3 hops of a seed. It is also our own training, yet it is gated (rule 9.5).
   - The heuristic control is capped at 2 hops. An unbounded non-learned propagation rule scores 1.000 on all three tasks at n=64, and that control is missing.
   - The energy bar compares an undamped update with a tanh-damped one.

## Advisory (disclosed limitations)
- D0 is mostly constants: the watched label arrays are never indexed, and the flags and episode counts are literals. No leak of evaluation labels was found.
- The oracle checks are weak: D1's "brute force" restates the same formula, and D2 is checked at n=24, not at n=64.
- "Decentralized" only means there are no autograd edges. Every module's target uses the global labels and the global clearing function.
- The title claims training on n∈[8,16]; the code trains at n=16 only.
- Scorer gaps: a smoke JSON with `smoke_cpu` flipped to false and its gates recomputed is accepted. It checks no evaluation sizes, step counts or device. Hashes are checked against disk, not the freeze tag. `--allow-smoke` skips the hash checks.
- Rule 9.9: bar history could not be verified, because the repo has a single commit and the PREREG already held the smoke numbers.

## Fit to the agenda
D1 is three small MLPs joined by a fixed spin product, trained by per-module finite-difference targets. D2 is a weight-tied recurrent attention GNN with a step-difference fixed-point penalty, unrolled longer at test. Neither part has an any-order world model or a search policy.
