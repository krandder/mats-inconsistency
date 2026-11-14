# Part 2 claims review

**Revision status:** the numbered findings below audit the frozen 09:27 report, not the subsequently edited live draft. During handoff the author began applying fixes. A read-only diff inspection at 09:49 UTC confirmed the two cost fixes and substantial revisions to the headline, budget labels, smallest-p comparison, noise discussion and scope of the conclusions. These observations are recorded below; they are not a full re-review of the revised report or its new confirmatory experiment.

Checked 2026-10-29T09:48:25 UTC, using the report snapshot taken at 09:27 UTC. **The numerical tables largely reproduce, but Part 2 needs substantive claim corrections. In particular, the depth-4 1e-12 win depends on an unspecified median convention, several “2^20 paths” results select between two budgets, and the smallest-p comparison changes the registered budgets.** I found no critical estimator failure in this claims audit; the original snapshot has seven major and five minor findings, some now being addressed by the author.

This review covers only Part 2. References to report lines mean [the frozen reviewed snapshot](claims_snapshot/report.md), SHA-256 `63a4083c41b30c3790c2e25417af2d84de01dc22183524a7737b7fde7ed244b2`. The live preregistration includes **amendment 3**, in addition to the two named in the brief. Its hash, `c90b7b55551723cb8c7851a5a3c181db9a8e835105a969affead54ded28a6641`, matches `prereg_hashes.txt`; I applied it to U1. I found no `tables_part2.txt`; the supplied `tables_s*.txt`, `summary_s*.json`, and rendered tables are the numerical record reviewed. The later `PREREG-TAILS-confirm.md` concerns a different experiment and is outside this review.

All writes were inside `/tmp/arcn_review2/`. Checks used the requested Python, CPU only, `OMP_NUM_THREADS=1`, and at most two concurrent processes. No new large-network simulations were run. Raw-output reconstruction, independent bootstrap calculations, and sensitivity checks are in [claims_checks](claims_checks/). The slow first version of the bootstrap check was interrupted and replaced by a vectorized calculation.

## Findings, ranked by severity

1. **Major: the headline 1e-12 win depends on using the lower of the two middle observations as the median.**

   Evidence: report lines 343–347, 493–500, 541; `analyze_tails.py:203–227,310–325`; preregistration §4. The scorer uses `quantile(..., method='inverted_cdf')` for **point medians**, including each net's eight-event median and the median of 20 net ratios, as well as for confidence bounds. Thus four finite costs and four infinities give a finite per-net cost. The usual even-sample median, averaging the middle two observations, gives infinity. Avoiding interpolation between infinities in CI endpoints does not require changing the point-median convention.

   At L=4, k=12, sigma=.001, the lower median gives 15/20 nets finite U3 B50 and reproduces **82.79× [70.19,85.35]**. The midpoint median gives 10/20 finite nets, a ratio of **18.73× [0,79.79]**, and no win. At sigma=.003 the corresponding result changes from **83.30× [71.46,87.48]** to **17.99× [0,86.00]**. These cells have no provisional k=12 events, so truth filtering does not explain the difference. See `claims_checks/check_claims.txt`.

   The preregistration did not resolve this even-sample/infinity convention. The lower median is a legitimate quantile convention; this is not an arithmetic error. It is nevertheless consequential analysis flexibility, absent from the claim that the result follows unambiguously from the registered rule. The depth-4 k=8 and k=10 wins and depth-8 k=6 win survive both conventions. U3r's depth-8 k=8 win also survives, although its midpoint-median CI is much wider: approximately [10.6,51.5], versus [44.5,50.5].

   **Fix:** specify point medians separately from bootstrap endpoint quantiles, show both conventions, and remove the unconditional depth-4 1e-12 win from the headline. Suggested wording: “U3 wins amortised at depth 4 for target probabilities 1e-8 and 1e-10, and at depth 8 for 1e-6. The depth-4 1e-12 result is sensitive to the convention for medians containing unreached events.” Do not retrospectively describe the lower-median choice as preregistered.

2. **Major: the tables and summary label budget-selected results as a fixed 2^20-path method.**

   Evidence: report lines 359, 374, 458–459, 475–477, 501–504, 543, 569; `analyze_tails.py:174–187`. The reported U3 and C-MCx rows use the cheapest successful budget **per event**, selecting between 2^20 and 2^22 paths as amendment 1 requires. They are not fixed-2^20 rows. A median cost equal to the smaller budget does not make the success fraction a fixed-budget fraction.

   At sigma=.001, depth 4:

   | Target k | Fixed 2^20 U3 successes | Budget-selected U3 successes reported |
   |---|---:|---:|
   | 4 | 159/160 | 159/160 |
   | 6 | 149/160 | 149/160 |
   | 8 | 131/160 = 81.875% | 134/160 = 83.75% |
   | 10 | 106/160 = 66.25% | 112/160 = 70% |
   | 12 | 80/160 = 50% | 85/160 = 53.125% |

   At depth 8 the corresponding sequences are **143,112,61,34,22** versus **143,114,64,36,25** successes. U3r at depth 8 has fixed-budget counts **147,129,91,71,48**, versus selected counts **147,132,97,74,54**. C-MCx-Edgeworth is affected too: at depth 4, k=4, its selected 157 successes include one event not successful at 2^20. Evidence: `check_claims.txt`, `check_radial.txt`.

   **Fix:** label these rows “smallest successful tested budget: 2^20 or 2^22 paths; four replicates per budget.” Retain the selected-budget numbers, which implement amendment 1. Alternatively show separate fixed-budget rows with the counts above. For fixed 2^20, the depth-4 k=8/10/12 miss fractions are **18.125%, 33.75%, 50%**, not 16%, 30%, 47%. State that this is a retrospective B50 measurement using reference truth to select a budget, not an implemented stopping rule. For these moment-method rows, “success” means empirical relative RMSE ≤.5 across replicates, not that every estimate is within 50%.

   Also make the denominator visible: these moment methods share work across **256 outputs and five thresholds**, dividing shared cost by 1,280. The existing setup explains this, but the headline's “over the 256 outputs” alone leaves out a material source of the advantage.

3. **Major: the smallest-p table substitutes much larger budgets for U3 and the moment baselines without identifying a protocol deviation.**

   Evidence: preregistration §4 specifies ARC's K3f cost and 3e11 FLOPs for randomized better-than-zero comparisons. `analyze_tails.py:270–299` instead evaluates moment methods and U1 at their tested budgets, while MC and GS are evaluated at the registered cost budgets. Report lines 370–372 and 512–526 present the resulting smallest-p values together.

   A 2^20-path U3 estimate costs **5.704e11, 1.140e12, and 4.554e12 FLOPs** at depths 4, 8, 32, respectively, before amortisation. All exceed 3e11, and all exceed the corresponding single-vector ARC cost. The stated U3 endpoints **1e-12 / 1e-12 / 1e-6** do reproduce at this larger tested budget, including after provisional-event removal. They do not answer the registered equal-budget question. Amendment 1 changes B50 selection, not these better-than-zero budgets. The scorer also omits Kerdock's registered-budget better-than-zero comparison.

   **Fix:** split the table into “registered fixed-cost comparisons” and “supplementary comparisons at tested budgets,” give actual FLOPs for each tested-budget row, and mark U3/C-MCx/U1 at the registered budgets as **not evaluated** unless a justified measured result is available. Do not scale the biased moment readout's error as 1/budget. Suggested summary: “At its measured 2^20-path budget, U3 beats zero to target k=12/12/6 at depths 4/8/32; this budget exceeds both registered fixed-cost budgets.”

4. **Major: the failed candidate tests are generalized to all backward functions and to a uniform twofold bound on noise leverage.**

   Evidence: report lines 346, 367–369, 443, 573–577; `killtests.txt` and `x2.py`. U4 tests Gaussian proposals of ranks 0,3,10 at **L=4, sigma=.001, k=8** on four development nets. U5 measures a conditional-probability leverage diagnostic on **two development nets, seeds 102–103**, at k=8. It does not run every layer-wise SMC algorithm, and it does not test input-directed backward functions generally.

   The range **.455–1.110** is a range of **per-net medians over eight outputs**, not a lower bound on every event's r. The saved U5 output includes a minimum r estimate **.102** at seed 102, L=32, sigma=.01; its reciprocal is about 9.8. Even the smallest median, .455, has reciprocal 2.20. Values above one are noisy estimates of a probability ratio whose true value cannot exceed one. The exploration value .26 quoted in P2.4 comes from seed 100, a different development set. No uncertainty interval for these r estimates establishes a hard upper bound on gain.

   **Fix:** replace “Backward functions have no leverage at low noise” with “The tested Gaussian input proposals failed, and the prior-input noise-only leverage diagnostic failed its registered median-r bar.” Say “Estimated per-net median r ranged from .455 to 1.110 on seeds 102–103; this suggests limited typical variance reduction for estimators retaining independent prior input draws.” State the actual depth/sigma/event coverage and distinguish exploration seed 100. Replace “None … works at sigma <= .01” with “None of the tested backward-function candidates produced a qualifying evaluation win at the three tested noise levels.”

5. **Major: the explanation asserts relative-error control and a causal four-moment limitation that were not established.**

   Evidence: report lines 557–559 and 568–572. “The saddlepoint tilts to the threshold first, so its error is relative” is not a guarantee supplied by `deduce.py`. It applies Lugannani–Rice to a fourth-order truncated CGF, which its own docstring acknowledges need not be a valid CGF; it uses fallbacks when necessary. The experiment supplies neither a relative-error bound nor the assumptions of an asymptotic theorem establishing one for these nets. Likewise, observing failure of the tested readouts does not show that every possible four-moment readout must fail, or isolate higher cumulants as the sole cause rather than moment error, invalid truncated CGFs, and readout approximation.

   **Fix:** “The saddlepoint readout tilts a truncated CGF toward the threshold. It improves empirical relative errors in these cells, but no relative-error guarantee is established here.” Replace “What limits it is …” with “The deterioration with depth is consistent with information lost by the four-moment approximation; this experiment does not isolate that explanation.” Preserve the existing biased/post-hoc labels.

6. **Major: the noise discussion confuses stable verdicts with absence of a measurable effect, and misidentifies an extreme.**

   Evidence: report lines 533–549 and prediction row 599; `check_claims.txt`, `check_details.txt`. The reported central probability shifts approximately reproduce. But “Nothing changes beyond noise from sigma=.001 to .003” is stronger than the analysis supports. At L=32,k=4, the paired log10 probability shift has median **.006835 decades** and a 10,000-net-resample percentile CI **[.004141,.011706]**. This is a reviewer sensitivity calculation, not a preregistered significance test, but it directly undermines the blanket no-change wording. The main win verdicts can remain unchanged while probabilities move.

   For sigma .001→.01 at L=32,k=12, **15.73× is the 95th percentile**, not the maximum. The largest saved reference-probability ratio is **169.39×**, seed 8/output 3. That event passes the registered validation checks at both sigmas. The maximum for .001→.003 is 33.93× and does involve provisional truth. The shallow “at most .006” figure is also rounded: the maximum absolute displayed median is .006404 decades.

   **Fix:** “The primary method-win verdicts are unchanged between .001 and .003 under the stated scoring convention; deep-event probabilities nevertheless increase modestly. At L=32,k=12, the .01/.001 ratio has a 95th percentile of about 16 and an observed maximum about 169.” Call these ratios of estimated reference probabilities, provide the paired validated-event sensitivity, and do not infer that the wins categorically arise independently of noise without a sigma=0 comparison. The rarest-depth-4 significance statement also needs finding 1's median caveat.

7. **Major: “splitting is the only accurate method at depth 32” conflicts with the report's own results and with its two accuracy criteria.**

   Evidence: report lines 346–347, 580–582; depth 32 table. Analytic MC has finite B50 at every level by construction. At k=4 Kerdock reaches every event under the scoring rule, and U3 reaches **90/160** events after budget selection. U3 also passes the separate better-than-zero criterion at k=6. Thus “only accurate method there” and “Monte Carlo moments … fail beyond 1e-4” cannot stand without defining the intended accuracy/cost criterion.

   Further, C-GS's “every event” means **variance-based projected B50 is finite**, not that an independent implementation was run at each predicted B50 and achieved the target. GS uses the truth runs' measured variance and a modeled 500-particle pilot. At deep rare levels, many of those same reference estimates are provisional. Calling projected reach “every event within 50%” conflates projection with direct verification.

   **Fix:** “At depth 32 none of our proposed methods has a qualifying win. Per-event splitting has a finite projected B50 for every tested event, including the rarest targets; direct MC also has finite analytic B50 but becomes far more expensive. Beyond k=4, U3 misses the 50%-RMSE target for most events, although it remains better than zero at k=6 at its tested budget.” Keep the provisional-truth and extrapolation caveats beside this statement.

8. **Minor: depth 32 ARC's single-threshold-vector cost is overstated by about 24%, and that wrong budget is used in the fixed-cost comparison.**

   Evidence: report lines 354, 521; `analyze_tails.py:282`; raw `kp_0_L32_s0.001_k3f.json` and equivalent files. The shared propagation is **196,344,374,101** FLOPs and one heaviside terminal step is **12,810,987,275**, totaling **209,155,361,376 = 2.09e11**. The reported 2.6e11 is the shared propagation plus **all five** terminal steps, **260,399,310,476**. Depth 4 and depth 8 correctly quote one terminal step.

   **Fix:** replace 2.6e11 by 2.1e11 wherever it means one threshold vector, and derive the “ARC cost” budget from the stored counter instead of a hard-coded value. `check_budget_ci.txt` verifies that correcting it leaves the displayed MC/GS better-than-zero endpoints unchanged, with and without provisional events, at all three sigmas. This is why I rank it minor despite the sizeable cost error.

9. **Minor: the promised summary confidence intervals are incomplete, and their scope needs stating.**

   Evidence: preregistration §4 requests median B50, reached fraction and median relative error with net-bootstrap CIs. `analyze_tails.py:261–267` bootstraps B50 but emits only point estimates for reached fractions and relative errors; `summary_s*.json` stores no CI for either. Better-than-zero uses a separate bootstrap internally but does not publish its numeric error interval. Thus “full tables with 95% CIs” does not supply all registered summary uncertainty.

   **Fix:** add net-bootstrap CIs for reached fractions and error summaries, and retain the actual numeric intervals behind better-than-zero decisions. State that the bootstrap resamples 20 nets with eight outputs kept together and treats each supplied truth/replicate-error estimate as fixed. It does not separately propagate finite truth-run or four-replicate moment-estimation uncertainty, and these are pointwise intervals, not a simultaneous guarantee across cells.

10. **Minor: “checks pass at 1e-4 and 1e-6” overstates truth validation and gives an unregistered check registered status.**

    Evidence: progress line 400 and setup lines 419–426; `truthcheck.txt`. At sigma=.001, L=32 there are **5/160** MC discrepancies beyond 3SE at k=4 and **6/200** independent-splitting discrepancies. The scorer correctly marks covered failures provisional. Across other cells there are additional failures. MC at k=6 was not the registered check; §2(b) specified k=4. The k=6 diagnostic also has zero counts in two cells, and `truthcheck.py` drops those observations from its log-ratio summaries, leaving 159 rather than 160 events there.

    **Fix:** “Aggregate cross-checks are broadly consistent; individual failures are flagged provisional. Registered MC validation is at k=4. The additional k=6 MC diagnostic is exploratory, with zero-count cases reported separately.” The existing detailed failure counts should remain. The 48/82,54/83,29/65 deep provisional counts are correct and need no change.

11. **Minor: several prediction/comparison descriptions overclaim what was replicated.**

    Evidence: report lines 528–529, 583, 592, 595. ARC's quoted “beats every baseline …” criterion is not the report's “beats zero” criterion, and the event selection/noise differ. Agreement near 1e-8 is an agreement of scale, not a replication of that comparative claim. The U2 prediction row says the post-hoc gap is +.16 to +.34 at k≥6, but depth 8,k12 is **+.09375**, already reported elsewhere. The “MC cheapest sampler” prediction is marked right while the same row says Kerdock is cheaper; at sigma=.001 all three k=4 tables indeed give Kerdock a lower amortised median cost than MC.

    **Fix:** “Our better-than-zero cutoff is of a similar order to ARC's reported depth 4 range; the criteria and events differ.” Change the U2 gap range to **+.09 to +.34**. Mark the cheapest-MC prediction “close, but Kerdock is cheaper at k=4”; retain the verified within-1.5× statement.

12. **Minor: the claim that the code-review cost correction is complete does not cover the kill-test implementation.**

    Evidence: report lines 398–399,619; `killtests.py` still omits U1's hidden-layer readout recomputation and fit costs and uses twice the selected direction's run cost. The evaluation scorer now includes both directions and the omitted costs. These two scoring paths therefore do not yet use the same corrected cost model.

    **Fix:** use a shared corrected U1 cost calculation in both places and regenerate the kill-test output. This does **not** change the verdict: `check_misc.txt` recomputes development k=8 using the corrected evaluation cost formula and obtains **27/32,17/32,13/32** under the ≤1,<1,≤.81 rules. Qualify “every fix” until both paths are corrected. Remaining small resampling/counter approximations are already disclosed and do not justify withholding the principal results.

## Claims checked and found correct

These checks support the results subject to the qualifications above.

- **Table transcription and raw reconstruction.** Every non-radial method's displayed B50 point values and reached fractions in the three summary JSON files reproduced from the raw truth, kprop, moment, Kerdock and mode outputs, for both all-event and validated-event tables. There were zero point-value mismatches (`check_claims.txt`). I separately recomputed U3r for all nine depth/sigma cells using an algebraically equivalent vectorized radial integration, checked against the original scalar function, with zero B50/fraction mismatches (`check_radial.txt`). The report's P2.3 table numbers agree with the rendered tables; the material issue is their budget labels, not transcription.

- **Kill-test numbers and eligibility.** A fresh `killtests.py` run exactly reproduces the supplied `killtests.txt` (`diff` empty). U2's k=8 success fractions are 21/32 versus 6/32 at depth 4, and 10/32 versus 6/32 at depth 8: improvements .46875 and .125. It fails the +.20 bar at depth 8. U2r also fails there. U3 has 26/32 successes at depth 4 and passes. U4's median relative variances round to 3650,873,656 and all fail 100. U5's displayed per-net medians reproduce and all exceed .1. U1 passes 17/32 under the written <1 rule, with 13/32 under the .9-RMSE margin, including after fully corrected cost accounting. Applying amendment 3 yields no qualifying evaluation U1 win under both reach rules.

- **Post-hoc separation.** U2 and its displayed variants are explicitly labelled post hoc in the summary, P2.3, P2.5 and the text outputs, and are absent from the sampling/hybrid win comparisons. No surviving sample-free candidate is claimed. The late confirmatory preregistration was not used to rehabilitate these evaluation results.

- **Reported win outputs under the implemented convention.** Lower-median/net-bootstrap calculations reproduce the reported 37/55/83 ratios at depth 4 and 9.4 at depth 8 for sigma=.001;38/57/83 and 9.5 at .003;38/56/76 and 9.5 at .01. The last .01 interval includes zero. U3r's reported 47× depth 8,k8 result reproduces. Competitors are selected from the full sampling/hybrid set, including Gaussian/Edgeworth moment baselines and Kerdock where tested. U3 has no per-event win and no depth 32 win. U1's looser-rule-only wins and their disappearance under the stricter rule reproduce. These are empirical pointwise comparisons, with finding 1's decisive median sensitivity.

- **Smallest-p arithmetic at the budgets actually used.** At sigma=.001, raw-error net bootstraps reproduce C-K3f endpoints 8/4/none; K2 6/4/none; augmented K3 8/6/not-run; C-MCx-Edgeworth@m20 8/6/4; U3@m20 12/12/6; U1 at its tested budget has endpoint k=4 only at depth 32; and post-hoc U2 12/10/none. MC at the actual one-vector ARC cost gives none/4/4; MC at 3e11 gives 4/4/4; GS at 3e11 gives 12/12/8. These displayed endpoints persist after excluding provisional events. The moment-method endpoints remain supplementary for the reason in finding 3.

- **Truth references and exclusions.** Probability means, relative sample variances, relative SEs and split-half ratios use the stated probability-scale calculations. Raw output has 20 evaluation seeds, three depths, three sigmas, eight outputs and five thresholds: 7,200 events, 160 per (L,sigma,k). Independent MC files contain 2^23 paths, with distinct tagged streams; truth uses 16 frozen runs at depths 4/8 and 32 at depth 32,2,000 particles each. All nine provisional-count vectors reproduce. At depth 32 the k=10/12 counts are 48/82,54/83,29/65. At depths 4/8 the totals are 0/9,1/10,2/5 across the three sigmas, consistent with the report's maxima. The aggregate MC and independent-splitting log-ratio/z-SD ranges reproduce. Primary U3 win cells do not rely on provisional-event inclusion under the implemented convention; the k=12 median sensitivity exists even with no provisional events. A supplementary U2 depth 32,k4 better-than-zero result at sigma=.01 does depend on including provisional events and should not be promoted to a robust result.

- **ARC construction and accounting.** Source comparison with upstream `arc_src/kprop_by_width.py:550` confirms factor=True, use_avg_metric=False, the SIMPLE K=3 harmonic description, L−1 ReLUs followed by the terminal linear/heaviside readout with bias−t. Adding Gaussian noise to the second cumulant after each ReLU is the stated noisy-network adaptation. It uses ARC's `NamedFlopCounter`; shared propagation and terminal readouts are separated. Four polynomial readouts really cost 5.912e9 FLOPs at depth 4 and are charged to U2. Per-event U2/K3f costs about 2.29× at depth 4 and 1.69× at depth 8; five-threshold amortised ratios about .84 and .88 reproduce. The earlier small-network code-review equivalence check remains applicable; I did not rerun a full kprop simulation. The single-vector depth 32 quotation is the exception in finding 8.

- **Sampling cost conventions.** MC's analytic N50=4(1−p)/p, hidden-noise draw charges, and division by 256 times the number of threshold levels served match amendment 1. Moment accumulation/readouts are included; shared moment work divides by 1280. GS charges a computed 500-particle pilot and the measured frozen-run variance/run cost, with no cross-output amortisation. Kerdock first-layer products are no longer counted twice, and rotation/block costs scale with independent run replication. The evaluation U1 counter charges hidden readout recomputation. The report correctly states that output amortisation is necessary for U3's cost advantage and that its per-event cost loses to GS. These are modeled FLOP comparisons; accuracy was measured on eight fixed outputs per net, not on all 256, and the denominator 256 extrapolates the evaluated workload to all outputs.

- **Other descriptive numbers.** U2's depth 4 post-hoc mean per-net success-fraction gains are .31875,.34375,.23125,.25 for k=6/8/10/12; at depth 8 they are .34375,.23125,.1625,.09375. The quoted bootstrap intervals reproduce to bootstrap Monte Carlo/rounding precision. Augmented-description gaps .375–.41875 reproduce. For fixed 2^20-path U3 misses at shallow depth, replicate-mean estimate/reference ratios have median about 1.9 at k=6 and 8; at k=12 the 10th/90th percentiles are .290/8.073, supporting the stated .3-to 8 range when that threshold is named. U1's single-run target-success fractions at sigma=.001 are 48.75%,13.125%,3.125% at k=4,8,12, matching 49%,13%,3%. Depth 32 K3f estimates are negative for 53.75–67.5% of events across the five levels, supporting “most.”

- **Compute record.** There are 460 evaluation kprop result files; missing augmented depth 8/sigma=.01 coverage is disclosed. `GPU-LOG.md` records 2.76+2.10=4.86 VM-hours and ceiling-cost estimates $2.40+$1.83=$4.23; the lane-total arithmetic is consistent. It records deletion and an empty instance-list check at 08:39 UTC. This is verification of the historical log, not an independent live cloud-state or billing check. The assertion that no evaluation result was inspected before an amendment is an author chronology statement; hashes and launch logs alone cannot prove what was read.

## Suggested replacement summary

“On 20 evaluation nets per cell, Monte Carlo moments with a saddlepoint readout give clear amortised cost wins at depth 4 for target probabilities 1e-8 and 1e-10 and at depth 8 for 1e-6. These wins share work across 256 outputs and five thresholds and select the smallest successful tested moment budget, 2^20 or 2^22 paths. The depth 4 result at 1e-12 depends on the median convention for unreached events. The estimates are biased, fail the 50%-RMSE target on a substantial fraction of events, and lose to splitting when charged per event without sharing. None of the tested backward-function candidates gives a qualifying win at depth 32. The saddlepoint readout of ARC's own description failed its kill test; its evaluation results remain post hoc.”

Source-integrity recheck: `claims_checks/source_manifest.json` records the reviewed hashes and the final comparison with the live source files.


## Concurrent author revisions observed at handoff

At 09:49 UTC I inspected and saved the changed report and scorer files in `claims_checks/concurrent_update/`, including diffs against the reviewed snapshot. I made no edits to those source files.

- The headline now restricts the robust U3 wins to depth 4 at k=8/10 and depth 8 at k=6, describes the k=12 median sensitivity, and names output-plus-threshold amortisation. The table budget labels and supplementary larger-budget better-than-zero comparisons have been separated. These are substantive responses to findings 1–3.
- The revised text narrows the candidate conclusions, distinguishes projected GS reach, and corrects the probability-ratio maximum to about 170. It also corrects the U2 gap and cheapest-sampler prediction. These address substantial portions of findings 4, 6, 7 and 11. This observation does not certify that every occurrence of the old wording has been removed.
- `analyze_tails.py` now uses 2.09e11 for the depth-32 single-vector ARC budget. `killtests.py` now includes U1 hidden readout recomputation, fit costs and both directions. The latter matches the formula independently checked in `check_misc.py`; its registered k=8 verdict remains 17/32. Thus the code corrections requested in findings 8 and 12 are present in this captured revision.
- New fraction CIs and `results/sensitivity.txt` are referenced. I have not audited that newly produced artifact. At the captured 09:49 revision, the P2.5 claim that the saddlepoint's error “is relative” and the progress-log statement “checks pass at 1e-4 and 1e-6” still remain. The revised report also references a new P2.10 confirmatory experiment, which is outside this snapshot audit.

The original findings and reproduction evidence are retained so the author can trace each correction. A later status claim about the revised report should use a new snapshot and check the new results rather than treating this review as approval of those additions.
