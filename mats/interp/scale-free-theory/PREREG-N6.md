# N6 pre-registration (scale-free-theory, written before running)

Measurement M1 (interaction profile, same code path as n5_interaction.py) on the 12 noisy phase-B nets
(work/cm-interp-B/runs/*_s0.05; group A: L2_s0.05, L2seed1_s0.05, L2mx0_s0.05, L2mx1_s0.05, L2mx2_s0.05,
L2nb6_s0.05; group B: L2seed2_s0.05, L2ws0_s0.05, L2ws3_s0.05, L2mx4_s0.05, L2nb5_s0.05, L2nb7_s0.05).
Predictive = noise-marginalised mixture over 8 noise draws (nnoise.predict_mix logic), mixture head (K = 4).

Predictions:
P-N6a: every net's cross-source interaction share is below its within-source share (nets learned the task).
P-N6b: group A's mean cross-source share is lower than group B's (A = "each source processed on its own before the
       sources are combined"), and at least 5 of 6 A nets lie below the B median.
P-N6c: across the 12 nets, cross-source share correlates positively with the net's KL to Bayes (as in the clean cohort).
Refuted as stated if the inequalities fail. Read-only use of work/cm-interp-B (nnoise.py) and work/tiny-pfn.
