**Verdict: major revision.** The numerical artifacts reproduce, but several conclusions exceed what was measured or registered. Required changes, in priority order:

1. **Identify the B&B estimate actually evaluated.** The summary calls it “deductive,” but §2’s accuracy tables and wins use `point`, which §2.1 correctly identifies as adaptive sampling. The deductive `mid` estimate has **no resolved RQMC wins** in the saved scored checkpoints. At \(n=3,L=32,2^{22}\) passes, median nMSE is **\(6.11\times10^{-15}\) for point versus \(3.40\times10^{-8}\) for mid**. Label the tables explicitly, report midpoint performance, and rewrite “which prong wins” accordingly. The certificate bounds midpoint error, not point error.

2. **Withdraw the claimed conservative 4.1× design ceiling.** At ARC’s \(256\times32\), [kernel results](/home/agent/workspaces/mats/work/unrestricted/runs/keven_n256_L32.json) give estimated \(a_2=-0.00226846\), although harmonic energy is nonnegative. Substituting this into \(b_2\le na_2/(n-1)\) cannot produce a valid upper bound on \(b_2\). The resulting **4.132×** is an uncertain plug-in estimate, not a proven or conservative cap. Three networks, sampled output coordinates, and integration noise require uncertainty control. Retain the population theorem; relabel the numerical application throughout the summary and §3.5.

3. **P3 and P4 are not fully scored by their registered rules.** [The scorer](/home/agent/workspaces/mats/work/unrestricted/scratch/analyze_eval.py:110) silently drops unavailable comparisons.
   - P3 lacks the \(2^{22}\) comparison for all three \(n=8,L=4\) networks.
   - P4 requires the largest scored budget reached. All 18 networks reach \(2^{22}\), but the scorer substitutes \(2^{18}\) or \(2^{20}\).
   
   Report these as incomplete under the registered rule, or explicitly declare the substituted rule. Remove the claim of measured equal-cost RQMC superiority at four million passes for widths 16–32. Their poor B&B performance is clear; that particular comparison was not measured.

4. **Apply the registered truth-resolution rule to displayed ratios.** The main table prints unresolved values as ordinary numbers; P7 also includes unresolved deterministic-QMC errors. At \(n=6,L=64,N=2^{18}\), seeds 5000 and 5002 fail the ten-times-truth threshold. Mark unresolved entries and propagate bounds into aggregates. P7 remains refuted by its resolved \(2^{10}\) and \(2^{14}\) results, so this correction need not change its overall outcome.

5. **Keep the multilevel theorem, but distinguish estimated efficiency from achieved performance.** The telescoping construction is unbiased conditional on the reference path, with fresh independent estimation samples. Its prefix costs and ideal allocation formula are correct under the stated leading-order schoolbook model. However, [the measurement script](/home/agent/workspaces/mats/work/unrestricted/scratch/mlmc_check.py:56) selects the best prefix using the same observations that estimate its efficiency. The reported gains exclude pilot/selection cost, setup, integer allocation, and lower-order work; they are **estimated ideal variance–cost gains**, not measured end-to-end speedups.

   Keep C0’s refutation conditional on the layerwise-cost and doubling assumptions, against **radius-exact MC with allowed batching**. It establishes neither superiority to RQMC nor the stated finite-precision alternative without an error analysis. Also replace “per-net variance stays of order 1” with **ensemble-average variance**.

6. **Correct the multilevel prose against its own table.** “Only where several times deeper than wide” contradicts the **1.53×** estimate at \(n=L=128\). “None at \(L\le n/4\)” contradicts ratios **0.960** and **0.911**; say “approximately break-even, without a demonstrated practical gain.” The 15× result is conditional on 12 networks with positive **empirical** angular variance among 32 seeds. Comparisons with another lane’s RQMC gains use different baselines and populations and do not establish relative performance.

7. **Specify the certificate norm in the new partition theorem.** §3.7 quotes
   \[
   K\gtrsim(n^3/\varepsilon)^{(n-2)/2}.
   \]
   [Astra’s derivation](/home/agent/workspaces/mats/work/unrestricted/coprover/astra_r3_answer.md:222) assigns this to summed-coordinate width. For Euclidean width, corresponding to the experimental certificate metric, it substitutes **\(n^{5/2}\)**. State the norm, expected-width quantifier, and dimension-uniform capture assumptions.

8. **Narrow the scaling and magnitude claims.** No 10M-parameter experiment was performed. Small-width failures do not establish that an entire family cannot reach that parameter count; the enumeration statement is conditional, and \(M^D\) describes a dense polynomial representation rather than every polynomial algorithm. Label these entries “untested; extrapolation” and specify width/depth/accuracy.
   
   Several numerical summaries also overstate their ranges: width-4 losses span roughly **33×–235,416×** across scored networks and budgets, not universally \(10^3\)–\(10^4\). RQMC costs to \(10^{-4}\) reach **8,192 passes** at width 32, contradicting “130–2,000 throughout.”

9. **Complete the scoring and deviation disclosures.** P11 uses `f1_J + f2_J` for the **first 16 output coordinates**, not the full output vector at larger widths. This was present in the registered code, but the report must identify the measured quantity. Add P9 explicitly as descriptive. Record the missing-budget scoring substitutions and the multilevel extension beyond the original configurations/seeds, with times where recoverable. Most listed post-hoc analyses are otherwise clearly declared.

10. **Fix the reusable allocator’s budget guarantee.** [allocate()](/home/agent/workspaces/mats/work/unrestricted/lib/mlmc.py:124) can exceed its budget: \(V=[1,0]\), \(C=[0.5,1]\), budget 10 produces counts \([20,1]\), costing **11 passes before setup**. Reserve minimum allocations before distributing the remainder and reject infeasible budgets. This does not invalidate the telescoping identity or the saved ideal-ratio calculations.

11. **Finish the review-A cleanup.** The main risk distinction, arithmetic-model qualification, threshold-seed withdrawal, polynomial uncertainty, and spin-glass caveats were substantially corrected. Remaining overstatements include “first failure at depth 9” despite only grid checks below 9, and “tiny tails are the low-temperature end.” `FORMAL.md` also retains contradicted claims above its appended errata; prominently mark those sections superseded. The numerical design ceiling in finding 2 repeats the same estimate-versus-bound mistake that review A corrected for polynomials.

Independent reproduction from raw JSON:

| Quantity | Reproduced value |
|---|---:|
| B&B point nMSE, \(3\times4\), \(2^{22}\) checkpoint | \(1.50168\times10^{-12}\) |
| B&B point nMSE, \(4\times8\), same checkpoint | \(2.38830\times10^{-7}\) |
| B&B point nMSE, \(6\times32\), same checkpoint | \(4.01565\times10^{-5}\) |
| B&B point nMSE, \(32\times32\), same checkpoint | 0.558292 |
| Two-level variance–cost ratio, \(128\times128\) | 0.655009 |
| Two-level variance–cost ratio, \(8\times256\) | 0.0653046 |
| P13 region-count slope | −0.201126 |

The factor-2 sensitivity also reproduces: **five winning checkpoints across four runs** become ratios **unavailable, 2.268, 0.359, 1.630, 5.641**. This supports fragility of the point-estimate wins.

P1, P5, P6, P8, P10a–c, P12 and P13 reproduce under their stated calculations; P2’s threshold conclusion survives, subject to resolution handling. P7 and P11 require the qualifications above. P10c’s “hit” remains only a plug-in scoring outcome, as already acknowledged.

Checked read-only on **2026-10-31, approximately 14:19 UTC**: regenerated both score objects in memory exactly, verified registration/amendment and registered-code hashes, and reran both test suites successfully. No files modified.
