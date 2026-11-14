codex
**Verdict: minor revision.** The conditioning fix works and the 2E arithmetic reproduces. Several Summary claims still need narrower labels.

1. **Round-2 finding 1: partly resolved.** The unidentified-tail caveat is restored, but Summary §5 and §6.7 still describe a width-independent low-order distribution rather than “no detected change at the measured C(t).”

2. **Finding 2: partly resolved.** §6.6 now labels unchanged high-order efficiency as an assumption; the Summary’s categorical “RQMC removes pairwise variance well only when its points far outnumber the coordinates” still exceeds the calibration evidence.

3. **Finding 7: resolved.** All **132** unaffordable records in `part3_scores.json` have `gain: null`, with extrapolations confined to `gain_extrapolated_historical`; omitted matrix costs are disclosed.

4. **Finding 9: resolved.** The Summary now distinguishes the **1.059% median** older-noise floor from its **3.955% maximum**, preserves radial coverage, and correctly states the annealed Hermite identity.

5. **Finding 11: partly resolved.** 2E correctly shares older noise and matches 2D’s conditioning. However, its finite-t slope is an additive-share **upper proxy**, so the reported non-additive orders remain proxies, not direct measurements.

6. **Finding 12: resolved for the tail claim.** The Summary now calls high-order-tail movement a possible reconciliation and explicitly leaves its mass and effect on RQMC gains unestablished.

7. **Finding 13: partly resolved.** The sign, rounding and net-depth labels are fixed, but [L512](/home/agent/workspaces/mats/outbox/input-bits-2026-10-31.md:512) still says “an order shift that is not there.” Replace with “the predicted decrease in C(0.75) was not detected.”

8. **Independent 2E recomputation matches the saved scores.** Each entry below is the median of per-net quantities, calculated from run JSON:

   | n | Cc(.25) | Cc(.5) | Cc(.75) | C(0) | Additive proxy | Non-additive-order proxy |
   |---|---:|---:|---:|---:|---:|---:|
   | 16 | 0.158164 | 0.351039 | 0.594993 | 0.003918 | 0.632656 | 8.033406 |
   | 64 | 0.173189 | 0.369604 | 0.616582 | −0.002587 | 0.692756 | 14.983737 |
   | 128 | 0.170047 | 0.361762 | 0.602107 | 0.008792 | 0.680188 | 20.268204 |

   Endpoint proxy ratio: **2.522990**. All three registered 2E rules pass. JSON stores estimated correlations/dimensions, not underlying batches, so this verifies aggregation rather than reconstructing path-level estimates.

9. **The proxy qualification must reach the Summary and table headings.** Writing conditional order shares as \(s_k\),
   \[
   A=C_c(0.25)/0.25=s_1+0.25s_2+0.25^2s_3+\cdots .
   \]
   Thus \(A\ge s_1\), and \((D-A)/(1-A)\) overstates the true non-additive mean when \(D>1\). “Additive share 0.63–0.69; mean order 8→20” is not supported literally. Growth itself survives conservative algebraic bounds from these point estimates: median non-additive orders lie within **6.28–8.03**, **11.49–14.98**, **15.45–20.27**, respectively; these are not confidence intervals.

10. **Two §6.7 numerical/interpretive corrections remain.** The largest median 2E–2B difference is **0.0101403** at \(n=64,t=.25\), so “within 0.01” should be “within 0.011” or “about 0.01.” “Older-noise main effect below 0.01” describes estimated medians: individual estimates reach **0.014375**. Agreement at three t values does not establish that 2B and 2E are the same function or distribution.

11. **Not every Summary sentence is supported at its stated label.** Beyond findings 1–2 and 9:
   - “Amplitude freezes at 10.65” needs **annealed expected amplitude variance in the joint limit**.
   - “Within 10% at n=256” needs the evaluated depth/layer restriction; P1.17–18 remain partially evaluated.
   - “A learned rotation fails at n≥128 because pilots are too noisy” is too broad: at **128×64**, rotation gains **4.287×** versus **2.942×** without rotation. Pilot noise as the explanation remains **HEURISTIC**, with proxy mismatch another possibility.
   - “Only dyadic multilevel gain grows with width” needs its Part 3/depth scope; the Summary itself reports two-level growth **1.3→1.9 at L=n**.
   - Replace “all findings were applied” with the actual remaining qualifications.

Checked read-only **2026-10-31 20:33 UTC** against the report, runners, scorer and JSON; Amendment 2E registration and runner hashes match.
