# R28B freeze review (Track C) at 92c800a (opus-mats, 2026-10-20 ~16:10Z): RETURNED

One reviewer, 16:00–16:08Z. Scratch in /tmp/r28b-review (repo clone, probes, mutation tests). R = the runner.

## Blocking
1. **The "learned world model" is the exact game tree.**
   - R:594 enumerates the whole tree. R:631-635 expand every action, and R:617-622 cycle chance outcomes in fixed order instead of sampling them.
   - Sampled payoffs equal the exact ones within 3e-8 (Kuhn), 7e-9 (Liar's Dice) and 5e-7 (Sheriff).
   - The network enters only through an 8% blend (R:678), and infosets and legal actions come from the full-tree oracle (R:724-727).
   - The "0 oracle calls" counter counts only `evaluate_q_and_reach`.
   - "B" is the number of mirror-prox iterations (R:877), not a branching search.
2. **Rule 9.1: random-network control.** It does not run through the arm's pipeline. Run that way with an untrained world-model network, it passes D2 (Sheriff r1: v1 0.8719 and SW 1.5641, against the 0.80 and 1.50 bars), which voids D2. On Kuhn D1 at B=16, the random network gives NashConv 0.190 against 0.157 trained.
3. **D2 holds by construction.**
   - The search uses the same pessimistic follower best-response calculation as the evaluator (R:758-847 and R:914-983), on exact payoffs.
   - The Kuhn negative control is the minimax theorem.
   - The "ceiling" is exceeded (1.171 > 1.06).
4. **Rule 9.5.** `External_Sampled_MCCFR` is gated without a published recipe, and its budget is not matched: Leduc 4.75, the same as uniform.
5. **The annealing test uses the wrong control.**
   - Fixed τ uses τ_start with β=0.80. R27C used τ=0.06 and β=2.4, and the PREREG misquotes this.
   - Fixed τ at τ_end beats annealing on Liar's Dice (0.0407 vs 0.0727).
   - The same-stage tabular control requested by the R27C audit is missing; tabular ≈ neural.
6. **`scripts/preflight_check.py` does not do 3 of the 6 checks PROTOCOL 10.2 requires.**
   - There is no bar-history check.
   - The split-overlap check reads an optional key that R28B does not write.
   - The random-network check is claimed but not coded.
   - It requires every smoke gate to pass, which rewards bars set below smoke values.
   - 7 of 7 mutations went undetected: a bar loosened everywhere, the prose bar changed only, the random-net clause dropped, a CI inverted, smoke seeds equal to full seeds, the random-net arm removed, and the oracle counter made a constant.
   - `smoke_full_rng_seed_overlap` is a constant.
   - Until preflight_check.py is fixed and passes these mutation tests, no Track-C PREREG can rely on it.

## Advisory (disclosed limitations if the round proceeds)
- The bars were written after the smoke run on the same games; only the RNG seeds are fresh. Liar's Dice bar 0.12 vs smoke 0.093; Sheriff SW bar 1.50 vs smoke 1.629.
- The outbox hashes for score_r28.py and preflight_check.py match neither the commit nor the working tree.
- Constants are missing from §5: the D2 search constants, the schedules, the myopic arm, the token tying, the grid and the CFR+ counts.
- KNOWN_CFR_CURVES are a self-regression. The scorer writes no headline numbers or controls.
- Leduc infosets that rollouts never reached are treated as roots. Sheriff r2 adds little over r1.

## Fit to the agenda
This round runs proximal MMD on exact payoffs (or a near-exact payoff table) for B iterations and distils the result into an MLP. D2 is a random search over leader policies, scored by an exact follower best response. There is no learned world model, any-order or otherwise, and no search policy, so it does not test Kelvin's agenda.
