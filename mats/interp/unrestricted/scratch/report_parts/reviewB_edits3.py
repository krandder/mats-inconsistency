p = '/home/agent/workspaces/mats/outbox/unrestricted-estimators-2026-10-31.md'
s = open(p).read()
rep = {
"| P3. n in {6, 8}: >= 100x worse than RQMC at every budget in [2^14, 2^22] on all 30 nets | **Hit**: 30 of 30 (10^4-10^6 worse). |":
"| P3. n in {6, 8}: >= 100x worse than RQMC at every budget in [2^14, 2^22] on all 30 nets | **Hit on the available comparisons**: 30 of 30 (10^4-10^6 worse); the 2^22 comparison is missing for the three 8 x 4 nets (their 2^22 record overshoots RQMC's largest N by more than 1%). |",
"| P4. n in {16, 32}: >= 1000x worse than RQMC at the largest budget reached, certificate nMSE >= 1 throughout, all 18 nets | **Hit**: 18 of 18 (10^5-10^6 worse; certificate nMSE 6e5 to 2e34). |":
"| P4. n in {16, 32}: >= 1000x worse than RQMC at the largest budget reached, certificate nMSE >= 1 throughout, all 18 nets | **Hit under a substituted rule**: all 18 nets reached 2^22 passes, but RQMC was run only to 2^20 there, so the comparison uses the largest checkpoint with an RQMC value (2^18-2^20): 10^5-10^6 worse; certificate nMSE 6e5 to 2e34 throughout. Under the registered rule it cannot be scored. |",
"| P8. n in {6, 8}: B&B loses to RB-MC at every budget on >= 2/3 of nets | **Hit**: 30 of 30. |":
"| P8. n in {6, 8}: B&B loses to RB-MC at every budget on >= 2/3 of nets | **Hit**: 30 of 30. |\n| P9. secondary arm against primary (descriptive) | Within a factor of a few everywhere; better at width 3 and depth 16-32. |",
"| P11. f1 + f2 in [0.3, 0.95] on every net, and the median 2-design ceiling changes by < 2x from n = 16 to n = 256 (L = n/4) | **Refuted** on the second clause: the first holds (0.30-0.93), but the median ceiling fell from 7.0 to 2.2, a factor 3.2. Designs have even less room at width 256 than predicted. |":
"| P11. f1 + f2 in [0.3, 0.95] on every net, and the median 2-design ceiling changes by < 2x from n = 16 to n = 256 (L = n/4) | **Refuted** on the second clause: the first holds (0.30-0.93), but the median ceiling fell from 7.0 to 2.2, a factor 3.2. The measured quantity is f1 + f2 over the first 16 output coordinates (the registered code computes degree 2 on at most 16 outputs), which is all outputs only at width 16. |",
"""- Analysis choices fixed before any B&B evaluation result: comparisons use each B&B record's own cost (slightly above
  2^k because of batch granularity) for every baseline, and RQMC interpolation clamps a 1% overshoot of its largest N.""":
"""- Analysis choices fixed before any B&B evaluation result: comparisons use each B&B record's own cost (slightly above
  2^k because of batch granularity) for every baseline, and RQMC interpolation clamps a 1% overshoot of its largest N.
  Comparisons beyond RQMC's largest N are dropped rather than scored; this affects P3 (the 8 x 4 nets at 2^22) and P4
  (scored at 2^18-2^20), as stated in the table.
- The registered resolution rule (an nMSE counts only if at least ten times the truth's own error) is applied in
  scoring P1 and marked with * in the results table; P7 includes a few unresolved deterministic-QMC cells at 2^18
  (for example two of the 6 x 64 nets) and stays refuted on its resolved 2^10 and 2^14 values.""",
"""  the cost-to-precision table; the factor-2 FLOP sensitivity of the width-3 wins; the dev-only kink-split arm; the
  multilevel-sampler measurements of 3.7 (`scratch/mlmc_check.py`, declared in PREREG.md at 12:40Z, fresh seeds
  9100-9131).""":
"""  the cost-to-precision table; the factor-2 FLOP sensitivity of the width-3 wins; the dev-only kink-split arm; the
  midpoint-estimate statistics of 2.2; the multilevel-sampler measurements of 3.7 (`scratch/mlmc_check.py`, declared
  in PREREG.md at 12:40Z; run 12:40-12:47Z on 13 configurations with seeds 9100-9103, then at 12:48-12:50Z on 256 x 32,
  8 x 32 and 8 x 256, the last with seeds 9100-9131 because most width-8 nets are dead at depth 256).""",
"""- GPT-6 review B (fresh session) of the experiments and the summary: PENDING.""":
"""- GPT-6 review B (fresh session, high effort, 14:15-14:21Z; `review/review_gpt6_B.md`), on the whole report:
  verdict "major revision", 11 findings, all applied. The main ones: the tables showed the centre-point estimate,
  which samples, while the summary called the method deductive; the deductive midpoint estimate never beats RQMC, and
  the summary and verdict now say so. The 4.1x design cap at 256 x 32 was a plug-in estimate (its a_2 estimate was
  negative), not a bound. P3 and P4 were scored with dropped or substituted budgets, now declared. Unresolved table
  entries are marked. The multilevel figures are estimated ideal gains with prefix selection on the same data, and
  the prose now matches the table. The partition bound's norm is stated. Several ranges were corrected (width-4
  losses 33x-2e5x; RQMC to 1e-4 up to 8,200 passes). P9 and P11's measured quantity are stated. The reusable
  allocator could exceed its budget; fixed and tested (`tests/test_mlmc.py`). The spin-glass and tail wording was
  narrowed. The reviewer reproduced seven headline numbers from the raw files and the factor-2 sensitivity.""",
}
for a, b in rep.items():
    assert a in s, a[:70]
    s = s.replace(a, b)
s = s.replace("Lane: unrestricted estimators (theory and experiment), target M1. Bead pool-uwmcvo. Status: final draft, before the\nlast GPT-6 review.",
              "Lane: unrestricted estimators (theory and experiment), target M1. Bead pool-uwmcvo. Status: final, after two GPT-6\nreviews (both \"major revision\"; every finding applied).")
open(p, 'w').write(s)
print('ok')
