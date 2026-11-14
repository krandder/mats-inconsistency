Model: GPT-6 (Codex)

Checked 2026-10-29 14:45 UTC; read-only inspection, no Python or recomputation.

1. **High — Binding and “full size” are overclaimed.** Draft:7–8,20,91–94 substitute a post-hoc slope threshold for failed P106. Coincident threshold crossings within 250-step sampling do not identify the instant or mechanism of binding. Onset rom is only 0.307/0.255 versus peaks 0.433/0.401 (`e16_post.json`, `rom_large_at_onset_peak_final`). Fix: “A substantial departure appears in the same sampled interval as a reliability-slope jump”; label this exploratory in the plain-words version too. Describe rom as a matched-pair weight difference, not an absolute error against Bayes. P107's band even includes t_dep (`e16.json`, `band_steps`), so “before binding” also needs qualification.

2. **High — Permanent sign transition is false.** Draft:35 says s6 stays negative from 26k, but `e16.json:lineages.6.last_cross_to_negative_middle` is 50k. `e16_post.py:18` tests five consecutive negatives, not all remaining checkpoints. Fix: distinguish the first five-checkpoint negative run from the last observed crossing; acknowledge reversals during constant learning rate. P109 still passes its narrower registered bar.

3. **High — Undisclosed endpoint substitution.** Registration (`PREREG-B.md:1306`) and `e16_summ.py:57` define final s5 as 89k: rom large 0.157520, middle 0.064354 (`e16.json:lineages.5.final`). Draft:30,36,64 instead uses 90k's 0.144/0.053. Fix all “final” references and P108's table; retain 90k as explicitly last-step descriptive data. P108 survives, but peak/final ratios become 2.75 and 5.15; clarify the baseline for “3 to 5.”

4. **Medium — A/B mechanism exceeds the evidence.** Draft:10–12,97–99 treats a sign signature as group identity and refinement as explanation. Update 25 explicitly says the defining wiring tests were not run on bases. A scalar surrogate's R² does not establish the whole reliability code or why the sign changes. Fix: “These two selected lineages support a later A-like middle-range signature”; keep refinement/stalling hypotheses provisional.

5. **Medium — “Stays flat” misdescribes s6.** Draft:41 reports 0.06→0.30; `e16_tables.md` shows 0.47 at 15k. Fix: “rises then declines”; retain P110's failed correlation bar and describe behavioural decline as a fluctuating trend.

Checks: three code hashes match; measures, three-checkpoint events and P106–P111 bars match registration. Validation curves match via jq; model/phase-1 storage bytes match via unzip/cmp, supporting saved determinism evidence. Log births 14:12:11 and earliest NPZ 14:12:20 follow both hashes; timestamps support chronology, not immutable proof.

Survive: replay evidence, P107–P109/P111 passes, P106/P110 failures. Reword: onset, decline, A/B interpretation. Fail: permanent negativity from 26k, registered-final numbers, full-size-at-binding certainty.
