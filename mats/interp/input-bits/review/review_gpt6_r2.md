codex
**Verdict: major revision.** The arithmetic reproduces, but §6.7 compares different estimands and overstates the evidence for a ~10% tail.

Round-1 findings, in their original order:

1. **Partly resolved; Major.** [L334](/home/agent/workspaces/mats/outbox/input-bits-2026-10-31.md:334) correctly limits C(t)’s resolution, but L402–407 again assert an identified, width-independent low-order distribution. Qualify these claims.
2. **Partly resolved; Major.** [L306](/home/agent/workspaces/mats/outbox/input-bits-2026-10-31.md:306) properly calls the model a surrogate; L370–373 still assert that RQMC cannot reduce the growing tail. Present that as an assumption.
3. **Resolved.** [L421](/home/agent/workspaces/mats/outbox/input-bits-2026-10-31.md:421) distinguishes finite-budget gains from asymptotics; L448 qualifies the MLD prediction.
4. **Resolved for descriptive reporting.** [L425](/home/agent/workspaces/mats/outbox/input-bits-2026-10-31.md:425) discloses the oracle envelope; fixed-allocation gains reproduce, including 6.776× at 64×128. This does not validate independently selected allocations.
5. **Resolved.** [L231](/home/agent/workspaces/mats/outbox/input-bits-2026-10-31.md:231) explicitly withdraws the ANOVA-dimension identity.
6. **Resolved.** [L460](/home/agent/workspaces/mats/outbox/input-bits-2026-10-31.md:460) distinguishes unaffordability from larger-budget wins; charged gains 2.944× and 2.518× reproduce.
7. **Partly resolved; Minor.** [L466](/home/agent/workspaces/mats/outbox/input-bits-2026-10-31.md:466) discloses omitted matrix work, but `part3_scores.json` still stores unaffordable extrapolations as ordinary `per_run.gain` values. Mark them historical or unavailable.
8. **Resolved in the scorecard.** [L488](/home/agent/workspaces/mats/outbox/input-bits-2026-10-31.md:488) marks partial layer coverage; L493 supplies both CVQ aggregations. Propagate those labels to the summary/body.
9. **Partly resolved; Minor.** [L39](/home/agent/workspaces/mats/outbox/input-bits-2026-10-31.md:39) fixes the annealed identity and L47 the radial coverage, but L53 calls 1.06% a maximum. It is the median; n=128 nets reach **3.955%**.
10. **Resolved.** [L212](/home/agent/workspaces/mats/outbox/input-bits-2026-10-31.md:212) labels ratios of medians; L232 supplies median per-net ratios.

Additional findings:

11. **Major: §6.7’s “same integrand” reconciliation misses a conditioning difference.** At [L386–390](/home/agent/workspaces/mats/outbox/input-bits-2026-10-31.md:386), both experiments use output \(a=f(U,Z)\), but 2B measures the ANOVA distribution of \(g(U)=E_Z[f(U,Z)]\), with older noise independent. [2D’s runner](/home/agent/workspaces/mats/work/input-bits/scratch/amend2d_meandim.py:4) holds older noise shared and measures the variance-weighted conditional dimension \(E_Z[V_ZD_Z]/E_Z[V_Z]\). These coincide when the window includes the entire history, at n=16,32; they need not at n=64,128. The stationary-numerics estimator uses the conditional convention, but a different history/window and net population.

    Moreover, `S1_win` is an unconditional, total-variance-normalized, degree-4-truncated additive estimate. Combining it with conditional `D_coord` does **not** give an exact non-additive mean order. Small older-noise variance does not bound its contribution to an order-weighted moment. **Fix:** measure matching conditional C(t) and additive shares, or measure the dimension of \(g\); correct the estimand labels and qualify A2D.3 meanwhile.

    Recomputed medians from the saved run statistics and per-net `S1_win`:

    | n | D_block | D_coord | Report’s non-additive formula |
    |---|---:|---:|---:|
    | 16 | 2.281820 | 3.754234 | 7.347231 |
    | 32 | 2.193896 | 3.393468 | 7.805580 |
    | 64 | 2.649657 | 5.113680 | 13.330287 |
    | 128 | 3.487953 | 7.319550 | 18.581500 |

    Endpoint ratios **1.52858, 1.94968, 2.52905** reproduce. The JSON saves estimated dimensions, not refresh numerators/batches, so this verifies aggregation rather than independently reconstructing each estimator.

12. **Major: growth confined to a ~10% tail is only consistent with the evidence.** [L402–407](/home/agent/workspaces/mats/outbox/input-bits-2026-10-31.md:402) promotes a fitted explanation to a measurement. Even the supplied sensitivity fits give n=128 tail masses **21.5%, 11.1%, 10.1%** when its representative order is 6,12,30. Neither three C(t) values nor an additional first moment uniquely locates the growth. **Fix the summary to:** “Mean interaction order grows with width, while no shift is detected at the measured C(t) values. Movement within a high-order tail is a possible reconciliation; its mass and effect on RQMC gains are not established.” Remove L273’s contradictory “does not get more interacting.”

13. **Minor: remaining numerical and label inconsistencies.** [L278](/home/agent/workspaces/mats/outbox/input-bits-2026-10-31.md:278) should say S1 changes by **+0.031**, not −0.03; L394’s 7.4 should round to **7.3**; L457’s “18 cells” means **18 net-depths**. L496 should say the predicted decrease in C(0.75) was not detected, rather than an order shift “is not there.”

Checked read-only on 2026-10-31 against the reports, runners, scorer and JSON files; Amendment 2D’s registration and runner hashes match.
