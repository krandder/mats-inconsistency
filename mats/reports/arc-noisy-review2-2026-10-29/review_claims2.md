# Part 2 claims re-review

Checked **2026-10-29 10:03 UTC** against the [frozen current report](claims2_snapshot/report.md), SHA-256 `be215fa5ab1a54215227572ee24a5f1fb411014dc4964fd64e34c0ab38d1a33e`. **The numerical results largely reproduce; the confirmatory primary test fails overall. Several original findings remain partially fixed.** No new estimator failure found.

1. **Major — state the joint confirmatory failure explicitly and narrow the conclusion.** P2.10:725 says “The primary test passes at depth 4 and fails at depth 8”; Summary:367 says the test “confirms a positive difference”; P2.10:728 concludes “reliably more accurate.” The registration requires **both** depths to have a mean gain ≥.20 and CI lower endpoint >0. Recomputed primary gains are **.24375** and **.14375**, so the joint primary fails. The secondary fraction tests specify “the same” criterion at k=6/10/12; all three also fail jointly because every depth-8 gain is <.20. The positive pointwise intervals are real evidence for a narrower claim, but do not satisfy these registered effect-size tests.

   **Fix:** lead Summary/P2.10 with: “The registered joint primary test failed: depth 4 met the criterion; depth 8 did not. At k=6–12, U2 increased the fraction within 50%, with positive pointwise 95% intervals, but none of the secondary joint +.20 tests passed.” Replace “reliably more accurate” with the measured fraction claim. Attribute “does not help at depth 32” to the earlier evaluation, since this confirmation tested only depths 4/8.

2. **Major — the unsupported relative-error guarantee remains (original #5).** P2.5:591–592 still says “The saddlepoint tilts to the threshold first, so its error is relative,” despite P2.6's correct caveat and P2.8's claim that this was reworded.

   **Fix:** “The saddlepoint tilts a truncated CGF toward the threshold and improves empirical errors in these cells; no relative-error guarantee is established here.” Remove the corresponding unsupported absolute-versus-relative guarantee for ARC's readout.

3. **Major — the diagnostic remains a claimed bound in two places (original #4).** Progress:405 retains “r >= 0.46: at most ~2x”; P2.2:460 retains “between 0.46 and 1.1 … so at most about 2x.” These are estimated **per-net medians**, not eventwise bounds: the saved minimum is ≈.102, and even 1/.455≈2.20. P2.4:569 also mixes the exploratory .26 from seed 100 with kill-test development results without identifying it. P2.6's assertion that the conditional law “spreads over many input directions” is an explanation, not established by these failed proposals.

   **Fix:** use the Summary's qualified median wording throughout: “Estimated per-net median r=.455–1.110 on seeds 102–103, suggesting limited typical gains; no uniform bound established.” Label seed 100 as exploration and the conditional-law explanation as a hypothesis.

4. **Minor — registered reporting is still incomplete (original #3/#9).** `sensitivity.txt` adds fraction CIs only for U3, fixed-budget U3, C-K3f and U2 at depths 4/8, without validated-event versions. Median-error CIs and numerical intervals underlying better-than-zero decisions remain absent; other methods and depth 32 lack fraction CIs. `sensitivity.py` uses **4,000**, versus the registered **10,000**, resamples. The registered-budget Kerdock comparison remains omitted.

   **Fix:** supply the missing summaries/intervals using 10,000 net resamples, including validated events; explicitly mark Kerdock at registered budgets “not evaluated” unless calculated. Change P2.8's “all taken” to identify these outstanding items. Existing bootstrap-scope caveats are correct.

5. **Minor — noise invariance is still overstated (original #6).** Progress:412 says “the verdicts are the same” at all three sigmas; P2.4:581–582 says “nothing in the wins depends on the noise within 0.001-0.01.” The lower-median depth-4 k=12 verdict changes at .01, as the adjacent table correctly shows. P2.7 still marks the blanket “No change” prediction “right.”

   **Fix:** “The headline U3 wins at depth 4 k=8/10 and depth 8 k=6 persist at all three sigmas; the convention-sensitive k=12 win fails at .01.” Retain the probability-shift distinction and label its new CI as reviewer sensitivity. Add the paired validated check: L=32,k=4,.003/.001 median log10 shift **.00664 [.00410,.00960]**, 152 jointly validated events.

6. **Minor — the ARC replication wording survives (original #11).** P2.6:620 says ARC's method “does what ARC reported … (better than zero to about 1e-8).” ARC's quoted claim concerns beating every baseline, which is a different criterion.

   **Fix:** reuse P2.3's corrected wording: “Our better-than-zero cutoff is of a similar order to ARC's reported range; the criteria and events differ.”

7. **Minor — tighten the remaining numerical qualifications.**

   | Location / wording | Exact correction |
   |---|---|
   | P2.3:519, “9.4x [8.8,9.7], under both” | Lower median **9.414 [8.790,9.713]**; midpoint **9.267 [8.758,9.967]**. Both win, but the numbers differ. |
   | Summary:346, “70–84%” after “every noise level” | **69–86%** across those cells/sigmas; **70–84% at sigma=.001**. |
   | P2.3:528, “at 1e-10 and 1e-12 … 0.3x to 8x” | Replicate-mean estimate/truth 10th–90th percentiles among misses: **.39–5.46 at k=10; .29–8.07 at k=12**. |
   | P2.3:529, “everywhere (0.009–0.065x)” | That range describes **depth 4**; several deeper-cell median ratios are **0** because U3 never reaches the target for the median net. The no-win conclusion holds. |
   | P2.4:564–565, “at most 0.006” / “within 0.16” | Say **about .0064 decades** / **about .162 decades**; current rounding is not a strict bound. |
   | P2.10:724, “the same with provisional events removed” | Say **same verdicts**: depth-8 k=12 changes from **.1125 [.06,.16]** to **.11786 [.06,.17]**. |

**Status of the original 12 findings:** #1 fixed substantively (minor ratio correction above); #2 fixed; #3 budget separation fixed, Kerdock omission remains; #4 partial; #5 partial; #6 partial; #7 fixed; #8 fixed; #9 partial; #10 fixed; #11 partial; #12 fixed. Thus six are substantively fixed and six remain partial.

**What I checked:**

- Reconstructed evaluation B50 points/reached fractions from raw outputs for all nine depth/sigma cells, all/validated events, including independent radial integration: **zero mismatches**. Rechecked both median conventions, wins, fixed/selected budgets, better-than-zero endpoints, costs, noise shifts, bias descriptions and corrected U1 kill counts **27/17/13**. Independently reproduced `sensitivity.txt` **exactly**.
- Independently reconstructed confirmatory truth means/checks, thresholds and saddlepoint readouts for all 20 seeds, then 10,000 net bootstraps. All eight displayed gains/CIs agree to bootstrap precision. Verified **2+8 provisional events**, U2 better-than-zero cutoffs **k=12/8**, ARC **k=8/4**, and cost ratios **2.2867/1.6931** per vector and **.8422/.8800** shared. These cutoffs and primary verdict survive truth filtering.
- Both preregistration hashes match `prereg_hashes.txt`. The confirmatory header says 09:30 while the hash log says 09:27; both precede the logged 09:32 threshold job. This supports chronology, not proof of what the author inspected. GPU log supports **.37 h/$0.32** and the historical deletion check; no live cloud/billing check performed.
- Evidence/scripts: [claims2_checks](claims2_checks/), especially `check_confirm.txt`, `check_claims.txt`, `check_radial.txt`, `check_sensitivity.txt`, and `manifest.json`. The manifest recheck found all reviewed sources and 1,112 raw JSON files unchanged. Requested Python, CPU only, OMP=1, ≤2 processes, per-check CPU limits <10 minutes; all writes stayed here. No new network simulations.
