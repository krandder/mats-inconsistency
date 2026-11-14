# Gate: gemini FSQ joint remedies (4ec7dfc), 2026-10-22

Opus 5.5 gate subagent for opus-mats. Read-only worktree /tmp/gategem at 4ec7dfc; CPU only, nothing retrained. S = benchmarks/explore_joint_remedies_fsq.py, J = joint-remedies summary.json.
Checked at 2026-10-22: the README re-rendered from J with S's render_readme is byte-identical; per-seed means recomputed from J match every claimed number; aggbench sha256 at f0f686b = cba12a72..., matching J.

**Verdict.** The numbers are real. But none of them tests Kelvin's challenge: the keyed arms are structure-known, and the one blind arm is far behind (+0.08 at b=6).

1. **Curriculum: Y labels only, but it reaches into the simulator.** biased_bit_curriculum (parity_learn.py:128) sets the prior on all 6 latent bits to 0.8, then 0.65, then 0.5. No latent bit is ever a target. The l0 offset in the biased stages (prior_logit, pl:114) uses k. The builder tags this "biased-bit curriculum", not "Y labels only"; tag it **structure-informed (simulator-side)**.
   - The anchor gets the identical curriculum: same stages (S:1276), best-val in the final stage only. E2's curriculum filters on an observed token feature (S:890), so it is genuinely label-only.
   - Matching against the curriculum anchor is favourable: it is worse (0.0087) than the plain transformer (0.0068) and the ISAB at the same 8k steps (0.0020). The b=2 arm (0.0085) is +0.0017 against the plain transformer and +0.0065 against the ISAB.
2. **Keys: the decoder-side keys are outside b, but the b=2 parity is trivial.** With dec_keys the decoder gets var one-hots plus accuracy (S:276-279). At b=2 the identity code (2 readings -> 2 bits) is lossless, so 0.0085 only shows that FSQ learns the identity map; it tests no compression.
   - "keyed b=4 and b=6 worse than b=2" (0.0126, 0.0267) is optimisation noise, not a bit effect.
   - **keys0 is truly blind:** decoder input = m only (S:280-281). The encoder sees its own token (readings plus its own key), the same as builder's blind arms.
3. **Bits: PASS, one cosmetic nit.** Per-sample deterministic FSQ with b binary dims; no batch statistics, dropout 0. Evaluated at alpha=1 (S:446), and symbols are checked by Channel(b).send. Distinct symbols per agent are <= 2^b in every run (max 4/16/64).
   - Nit: the eval message is z_tilde + (z_hard - z_tilde).detach(), so it equals z_hard only up to float32 rounding (~1e-7). It is not exploitable in practice, but evaluating on z_hard would make it exactly b bits.
4. **Fresh / val / steps.**
   - Z9 trains on fresh GPU episodes; val (VAL_SEED) and dev (DEV_SEED) are disjoint. E2 is NOT fresh: a 100k pool, 4000x512 steps (~20 epochs), with val at dev offset 50k. It is the same data as its anchor.
   - The (d) and (e) arms get +400 decoder-only lock-in steps on fresh episodes, so 8400 vs 8000 for the anchor.
   - Anchor best_step is not stored in J, so anchor convergence is unknown. Builder's batch 7 found the anchors still improving at 10-11k, so all gaps are lower bounds.
   - best_uncompressed has a hard-coded 0.0010 floor (S:1332).
5. **Provenance: mostly OK.** The README is generated, with a 4-decimal check (S:1540).
   - J records f0f686b, clean, but the clean flag covers aggbench/ only. The benchmark script differs from f0f686b (adds --reuse-z9-from and per-bit merging), and that diff is in main() I/O only, not in training.
   - The merged J keeps only the first part's harness block, so the other per-bit parts' provenance is not recorded.
   - The fsq-vq and decoupling JSONs also record f0f686b; I did not re-render those READMEs.
6. **Blind cross-check: they agree on a gap, not on parity.** Gemini keys0 b=6: 0.0813 (0.084/0.075/0.086). Builder b7_recon_power_k0 + fine-tune: 0.081 (0.120/0.071/0.053, 11k steps). At b=4: 0.219 vs 0.166.
   - Two unrelated methods (FSQ + curriculum vs Gaussian + self-reconstruction + power) give a robust estimate: blind learned b=6 is about +0.07 to +0.08 behind uncompressed (ISAB 0.002). A blind hand code reaches 0.006 at 5.49 bits, so the gap is learning, not bits.
   - The matching means are partly coincidence: builder seed 2 reaches 0.053, so 0.08 is not a ceiling.
   - Gemini's b=4 blind (0.219) only ties the searched 4-bit blind code (0.216).
   - Kelvin's challenge is NOT met blind.
7. **Other claims.**
   - Warm-start 0.0099 at b=4 and 0.0090 at b=6 is keyed (structure-known): +0.003 and +0.002 against the plain 8k transformer, +0.008 and +0.007 against ISAB.
   - E2 b=6 0.0459 vs 0.0415: the 95% CI [0.042, 0.050] lies at or above the anchor, so the arm does not beat it. The anchor itself is weak: the blind extremized_q is 0.0403 and the oracle-structure llr_q is 0.0162. "Level with uncompressed" overstates it.
   - The decoupling (iii) 0/3 is on the original Z9-p2 (prior 0.458) with gauss/gumbel, not FSQ on lf. It does not isolate "FSQ fixes joint learning" (spec and channel are confounded).

**Must-fix before reporting:** tag the biased curriculum structure-informed; label every keyed arm STRUCTURE-KNOWN and say b=2 keyed is the identity code; report gaps against the plain transformer and the ISAB too; record anchor best_step; state that E2 is not fresh; headline the blind result as +0.08 at b=6.
