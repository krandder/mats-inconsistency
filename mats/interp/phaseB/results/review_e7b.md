Model: GPT-6

Checked 2026-10-29 09:12 UTC by reading Amendment 14, code, result JSONs, metadata, and stat timestamps; no Python or recomputation.

1. “Group B is not one type” needs an explicit **post hoc** label and weaker wording: heterogeneous correlations suggest substructure, but do not establish distinct mechanistic types. E7.2 confirms B solely by mean correlation with A ≤0.55; no B cohesion or subgroup criterion was registered. Four successes support this operational classifier on these four nets, not a general taxonomy.

2. Correct the fresh A–B range: JSON gives **0.40–0.47**, not 0.44–0.47 (nb6–nb7 = 0.402118). “At most 0.43 with A” needs “approximately 0.43 on average”: nb7 averages 0.433814 and individual correlations reach 0.463808. Other table entries and reported pair values match rounding.

3. Timing supports the intended order, without immutable provenance. For mx2/nb5/nb6/nb7, metadata precedes prediction, which precedes share_test.npz mtime: 07:27:58→07:34:35; 08:25:17→08:31:33; 09:04:43→09:09:48; 09:05:55→09:10:34 UTC. Code asserts absence before predicting and schedules prediction first. Mutable timestamps cannot exclude earlier deleted data. Prediction also reuses confirmation’s seed-1 noise on the first 2,000 episodes, beyond the disclosed episode overlap.

4. Disclose fine-tuning seeds **9/15/16/17**, distinct from base seeds **2/5/6/7**. New bases ran 90,000 steps; existing mix4_s2 ran 50,000, allowed by the registration. All fine-tunes match 20,000+8,000 steps/ramp 1,000. Manual base exclusion checking is disclosed; metadata confirms every base/net gap <0.03. No outcome-changing recipe deviation found.

Claims surviving: qgroup.py implements the registered thresholds and E6 cross-replicate estimator. A means 0.6654/0.6568 exceed B means by 0.2120/0.2304; B-to-A means 0.4276/0.4338 pass. **P66 survives, 4/4**, with descriptive evidence suggesting B heterogeneity.
