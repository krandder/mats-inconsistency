The revised [report](/home/agent/workspaces/mats/outbox/narrow-deep-m1-2026-10-30.md) supports its central conclusion: **the tested implementations fail the registered M1 criteria; their observed limitations are not lower bounds.** Several summary statements, scorecard entries, and generated tables still need correction.

1. **The summary still overstates RQMC’s dominance.** Section 5 correctly preserves net 217’s **3.138× TS advantage at 15,390.9 paths**, but the summary says RQMC wins at every budget above \(2^{13}\). Delete that universal claim. Section 5’s claim that RQMC beats TS on all four non-winning nets is also false: net 203 has a **2.291× TS advantage**, below the registered 3× threshold.

   The winning-net counts themselves are correct:

   | Arm | Nets with an admissible ≥3× advantage | Best ratios |
   |---|---|---|
   | TS | 217, 221 | 54.012×, 10.207× |
   | BO | 220, 221 | 10.285×, 14.462× |

   Both remain **2/6**, so the negative M1 result survives.

2. **P1/P3/P4 still need qualified scoring.**

   - **P1:** The mean-error clause holds on the four studied nets; I reproduce the minimum **0.00431994**. The registered polynomial meets are poor, but “every meet at c ≥ 4 is at least 0.11” is false: KP2 on net 220 at c=8 gives **0.0771800**. The equal-total-compute clause also retains the unresolved downstream-cost qualification. Report diagnostic support rather than an unqualified hit for that clause.
   - **P3:** Every tested D0 configuration misses \(10^{-5}\). That establishes failure of the tested configurations, **not the registered refutation condition requiring >10,000 paths**. Remove “P3’s refutation: true in that sense.”
   - **P4:** The explicit admission that the comparison uses an oracle meet plus a nominal 25-path downstream is good. It also means “miss” is not a completed equal-compute test. Use “diagnostics oppose the bottleneck prediction; formal equal-compute condition not established.”
   - **P9:** The corrected oracle counts support the input threshold clause and the **endpoint** increase from cut 8 to input on 5/6 nets. They do not establish monotonic increase through intermediate cuts. BO’s two-net count is correct under the stated counter.
   - **P10:** The four-net noise-effect count is correct. The second clause is correctly marked missed **under a pointwise interpretation**, because of net 217’s 15,391-path win. State that interpretation explicitly.

   Separately, the summary says the registered polynomial meet beats the nominal small-budget comparator on only one net. Under that same diagnostic convention it does so on **three**: 216, 217 and 220, with ratios approximately **33.2×, 4.09× and 2.07×**.

3. **The corrected per-cut table is numerically right, but its provenance and reproduction remain incomplete.**

   I checked **all 108 threshold entries**: 54 cells, each at \(10^{-5}\) and \(10^{-4}\). Every registered smallest-k value matches `runs/kbycut2_*.json`. All five printed stable-k annotations also match the \(10^{-5}\) criterion.

   Remaining corrections:

   - Specify that the square-bracket annotations apply only to \(10^{-5}\), or provide stable counts at both thresholds. At \(10^{-4}\), there are **12 cells where smallest and stable k differ**. For example, net 220 at input is **3 versus 6**, and at cut 2 is **2 versus 7**.
   - “The input is unchanged” is false for the recorded runs. Net 217 changes from **8 to 7** at \(10^{-5}\). The rerun also reduces nets 203 and 217 from **65,536 to 32,768 paths**; disclose this. The oracle estimates input covariance empirically, so the implemented transformation is not literally unchanged there either.
   - The registered probe-direction results exist in the JSON but are omitted from the revised table.
   - [analyze.py:153](/home/agent/workspaces/mats/work/narrow-deep-m1/scratch/analyze.py:153) still loads **v1** files and computes **stable k under a “smallest k” heading**. Consequently, `results/tables.md` does not reproduce section 2.

4. **“Genuine small-budget advantage” is defensible as an observed comparison, but the K2X explanation remains wrong on net 217.**

   At c=2 on 217, the stored decomposition is:

   \[
   \text{upstream}=+3.199\%\mathrm{SD},\qquad
   \text{downstream}=-4.783\%\mathrm{SD},\qquad
   \text{total}=-1.583\%\mathrm{SD}.
   \]

   That win is also cancellation-assisted. Net 221 at c=4 has same-sign terms and is different **in this decomposition**. Preserve both observed advantages, but remove the categorical separation of K2X wins from cancellations.

   The summary’s “12–145× worse” range also does not reproduce the small-budget K2X comparisons. At c=2, losses on 216/220/223 are approximately **9.82×/15.96×/30.66×**, while 203 wins 2.29×.

5. **The scaling claims remain stronger than their measurements.**

   - The width-8 input-error ranges in section 8 are stale. Corrected S-oracle ranges are **\(5.29\times10^{-5}\)–0.0690 at k=4**, and **\(9.08\times10^{-6}\)–0.001045 at k=6**.
   - The first positive cut where k=4 meets \(10^{-4}\) is **2–5 on all six nets**, not 2–8 on five. Net 220 already passes at input.
   - Width-16’s first tested passing cut is **4**, not “4–8.”
   - K2’s layer-2 error approximately halves **from width 16 to 32**: 0.000924→0.000448. The heterogeneous width-8 results do not establish halving at every doubling.
   - “About 3n/4 directions” combines different error thresholds, and k=24 was never tested at width 32. It cannot supply a measured fixed-error exponential-cost argument.
   - At width 32, cut 8 has **16,295 distinct classes among 16,384 inputs**, not one per input.

   The one-net-per-width caveat helps, but these numerical and logical errors should still be removed.

6. **Some first-round limitations are acknowledged without being consistently carried through.**

   The noise-effect table checks out: its total-effect and P3 columns reproduce the JSON, and its deep-only column reproduces the logs with the disclosed normalization limitation. BO tail medians also reproduce: **0.804, 2.520, 2.973, 2.248, 4.718, 2.345**.

   However, section 3 still calls GHM an “exact mixture over the realised gate patterns”; it should describe the implemented approximation directly. The estimator’s [direction construction](/home/agent/workspaces/mats/work/narrow-deep-m1/lib/bvf.py:174) remains \(C^{1/2}V\): the correction was to the oracle, not the estimator. Finally, `analyze.py` excludes \(2^{20}\) correctly but still prints the unresolved TS ratio without enforcing the \(10/P\) rule. The `ddof=0` issue remains unfixed.

First-round disposition: **#3, #7, #10 and #11 are adequately answered for this scoped negative report. #1, #2, #4, #5, #6, #8, #9, #12, #13 and #14 are partly answered**, for the reasons above. None is wholly ignored.

**Verdict: minor revision.** The remaining corrections do not overturn the narrow M1 conclusion and can be made without new large experiments.

Checked on **2026-10-31, approximately 01:50 UTC**: all 14 original findings; registration and amendments; v1/v2 oracle code and results; TS/BO ratios and truth cutoff; forward meets; P3 and noise diagnostics; bias decompositions; tail medians; scaling results; and table-generation code. Read-only checks used the specified Python under `nice -n 19`, with one-thread limits. **No files changed.**
