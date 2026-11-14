Checked 2026-10-29 against the supplied files. Regenerated tables match `results/tables.md` exactly; writes were intercepted. E3, available E4 results, E5’s main table, blind-comparison values and a11’s **0.005066** agree.

Review1 dispositions, one line each:

1. **Partly:** controls strengthen the two-pass result; standardisation remains confounded.
2. **Yes:** the numerical errors identified in review1 are corrected.
3. **Partly:** direct prediction comparisons support “near history-blind,” not identical predictions or zero history use.
4. **Partly:** budget qualifiers improve matters; the conclusion incorrectly says **one** N=12 seed escaped, versus **two**.
5. **Yes:** W2 conclusions now concern the tested architectures.
6. **Partly:** definitions and correlations are clearer; “carries variance” remains stronger than demonstrated.
7. **Partly:** probability averaging is disclosed, but the registered logit intervention remains untested.
8. **Partly:** clause-level scoring improves; untested interventions still receive overly broad “held” verdicts.
9. **Partly:** approximation and reference uncertainty are disclosed; timing still excludes quadrature setup.
10. **Partly:** recipes and headings are corrected; historical version authentication remains incomplete.

Ranked remaining findings:

- **High — E4 claims lack supplied evidence.** `flat1_s1`, both `told` seeds and both `learned` seeds have no evaluation files in the supplied directories. Their five quoted fraction/gap pairs, and P12/P13 verdicts, are **unverified**, not demonstrated mismatches. Available results support “no recovered history value,” not “no arm used history.” The round-routing explanation is appropriately labelled untested in E4; retain that qualification in the Summary.

- **High — E2/M4 summaries overgeneralise.** E2’s N=3 `labout k=1` costs **0.0949 and 0.0541**, not uniformly +0.095/+0.10 (also P9). N=12 s1 is partly escaped (**0.0938**), not history-blind at 0.20. M4 N=3 rank-1 distortion is **0.0160–0.0396**, not just 0.016; rank 4–8 reaches **0.0038**, exceeding “within 0.003.” Rank 2 is **0.0051**, so say “about 0.005.” M4 does not uniformly rescue stuck nets: N=24 s0 reaches **0.2279/0.1481/0.2026/0.1987**, contradicting the blanket 0.06–0.15 description.

- **Medium — Controls support the architecture, not an isolated causal mechanism.** One-pass controls have **22,195–22,967** parameters versus **19,923–20,697** for two-pass, not “the same … about 23k.” This favours the substantive comparison, but requires accurate wording. No-auxiliary means match the report; label these as **arm means**, not seed ranges. P21’s literal “within 25%” fails for N=24,m2=4: improvement is **29.3%**. “No worse” holds.

- **Medium — Learning-speed conclusion exceeds a11.** One seed at 4× beating the dense net at 1× establishes improvement, not that the architecture gap is “mostly learning speed.” Budget and cooldown changed together; equal-budget comparators remain pending. “m=4–32 within noise” also overstates: N=12,m4 and m32 seed ranges do not overlap.

- **Medium — E5 narrative arithmetic.** Original-rule L=16 is **10.5×** the reference, outside “2–3× too high.” P14’s corrected/refined qualifying-tail errors span approximately **−6% to +6%**, not −6% to +4%. P16 should remain an extrapolated comparison.

- **Low — Summary precision.** Label R² spans **0.75–0.78**, not 0.76–0.78. “Every net” should read “measured nets.” The introduction lists amendments 1–10 although a11 is discussed.
