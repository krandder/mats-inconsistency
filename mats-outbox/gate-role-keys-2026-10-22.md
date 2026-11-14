# Gate: role-keys learned aggregator (2026-10-22, adversarial check of role-keys-learned-2026-10-22.md)

**Verdict: PASS with fixes.** The core result holds: fresh-trained learner within 0.0044 nats of the joint ceiling (seed means). Three claims are overstated, and one "Y labels only" caveat is missing.

Checks (code read; numbers recomputed from role_learn.json; /tmp/gate_chk.py re-derived the dev sets and ceilings):
1. **Disjoint, val-selected: yes.** Train stream default_rng([seed,99]), val 8000001, dev 8000002. Best checkpoint chosen on val; dev is used only for scoring. Pool tests use [DEV_SEED,1..3] and the same Pool(M,seed) for both models, so the comparisons are paired.
2. **Same episodes, same regret: yes.** ceilings() runs on each dev set. It reproduces role_curves.json exactly (p=2,4; sigma 0.25,2; 6-digit match). true_bayes_logit equals the enumerated p_true (max diff 4e-16). Both use KL(true-roles Bayes || method).
3. **Leakage:** inputs are [key, readings] only. No W, roles or Y reach the forward pass. **But checkpoint selection uses val KL to true-roles Bayes, which needs the true roles.** So "Y labels only" is not strictly true: the gradients see only Y, but model selection uses privileged information.
4. **Recomputed:** max mean gap 0.0044 (p4 s0.5; per-seed max 0.0050 at p4 s0.25). p4 s0.25 content captured 0.0068/0.0110 = 62%, s0.5 83%. Blind 0.1211/0.2230 vs content-only Bayes 0.1209/0.2230. FSQ b2 p4 s0 0.0078. Persistent s2 (ii) 0.568 vs prior 0.255. All match the note.
5. **"Beats keys-only where content helps": not at p=2 s=0.25.** Learned minus keys-only per seed: -0.0001, +0.0001, -0.0007 (mean -0.0003, t=-1.0 with n=3). The episode-level SE is about 0.0004 to 0.001, so the difference is not significant. It is clear at p4 s0.25 (-0.0068, t=-15) and at every sigma of 0.5 or more.
6. **Shortcut test:** the inputs are right: the pool model sees fixed eps and fixed roles. The "worse than prior" claim fails at s=0.5 (ii): 0.259 vs prior 0.255, and seed 2 (0.240) is better than the prior. The prior differs by test set (0.255 for (ii), 0.266 for (iii), about 0.30 for (i)), not "about 0.26". Column (i) labels the fresh model's bound as 0. Its real bound there is the joint ceiling, 0.181 (s0.5) and 0.266 (s2), and the fresh model's 0.186 and 0.269 sit at that ceiling.

Must-fix in the note:
- M1: Restrict "beats keys-only" to p4 s≥0.25 and p2 s≥0.5, and say p2 s0.25 is not distinguishable.
- M2: For persistent at s0.5 (ii), say "near the prior (0.259 vs 0.255; 1 of 3 seeds below)". Give the prior for each test set.
- M3: In shortcut column (i), show the joint ceiling as the fresh model's bound. Also note that the iid-role joint ceiling is not exact Bayes for 6 distinct pool agents with balanced roles.
- M4: State that checkpoint selection uses true-roles KL, or reselect on val Y log-loss. That needs a rerun, because only val_kl is logged.
Minor: say "within 0.0045" holds for seed means only. Record the exact command, since the docstring defaults are 3000/12000 but the runs used 2000/10000. The "best fixed 2-bit code 0.0030" has no source in these files. The 0.0045 headline sits next to p4 s5 best_step 500/750: selection there is early and noisy.
Note: this gate reran role_learn_summary.py, which rewrote role_learn_plot.json and role-learned.html. The script is deterministic and the inputs are the same.
