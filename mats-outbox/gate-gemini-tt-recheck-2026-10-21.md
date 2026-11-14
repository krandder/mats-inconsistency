# Narrow re-check: gemini trader-transformer must-fixes (31d7320, fa016d9, 414b2ab), 2026-10-21
Checked at about 09:00Z on origin/main 414b2ab (after git fetch). Source: JSON/summary.json in tt-gate-e2z8syb, tt-gate-xor-fresh, tt-abc-*, tt-grid-*, plus runner code at that commit.

1. Labels: **partly fixed.** The abc and grid READMEs are clean: "EXPLORATORY baseline", no Gain/Confirmed/Decided/"completely solves", and distill is graded "Oracle Ceiling". The two new READMEs (tt-gate-*) have no EXPLORATORY label and add new overclaims: "Key Findings", "Converge Well Below 0.01", "every single non-none architecture below 0.01" (false, see 7), "completely eliminates the every_prefix tax", "strictly helps". The Z6 prose leads with B_loss_distill (0.0003) but does not say it is an oracle arm. I did not read the mail.
2. Matched bar: **fixed in code and JSON, wrong in the README text.** bar_fct_piece_bidir is built from the same piece-cell tokens as the grid arms (in_dim 4) and trained the same way: w64, L4, the same per-seed 20k pool, 2000 steps, lr, wd, and early stopping with patience 800. Two small differences: the bar uses dropout 0.1 against 0 for the trader arms, and it has no mask for absent cells. bar_fct_deepsets_bidir uses the same deepsets tokens as the trader arms. The 3000x512 headline bar is also run. Result: the bidirectional bar beats the matched causal arm in 6 of 6 cases.
   **New error:** the README labels the linear trader arms (A_attn_*, B_*, C_*) "Flat 24-d (no report/4)". In the JSON they have in_dim 49, the same as full_context_transformer_none, so they DO get report/4 (bar_fct_no_report has 48). The §2 diagnosis 1 therefore compares mismatched inputs. With the input matched, FCT_none scores 0.0609 and A_attn_all 0.0843, so the trader arm is worse.
3. E3-biased dropped and the Z8 claim withdrawn: **not fixed in the new README.** E3-biased is dropped from the grid claims, and the abc/grid READMEs say ext_log_pool beats all arms on Z8. But the tt-gate-e2z8syb §1 baseline table does not match its own JSON:
   - Z8: ext_log_pool shows 0.0445 (JSON 0.0344) and oracle_clearing 0.0446 (JSON 0.0236).
   - E3-sybil: ext_log_pool shows 0.1325 (JSON 0.0777).
   - Effect: on Z8, ext_log_pool (0.0344) still beats every trader and grid arm. Only bar_fct_piece_bidir is lower (0.0331), and so is the oracle distill. The wrong table hides this.
   - The XOR README has the same problem on Z5: it says oracle_clearing is 0.1973 and "fail[s] completely", but the JSON has 0.0000. Z5 ext_log_pool shows 0.1578 (JSON 0.1341).
4. Z8 prefix Bayes: **fixed.** Run commit 204e78a contains 2546bb2, and the runner asserts final == bayes (< 1e-8) on every env it runs. In the JSON, prefix_bayes_final_maxabs_err is 3.6e-15 on E2, Z8 and E3-sybil, 7.1e-15 on Z5 and 3.6e-15 on Z6. For Z8 distill, final_prefix_kl equals kl (0.0271 = 0.0271). Z8 B_loss_distill_deepsets is 0.0318±0.0019 (oracle). The unit test covers only E2, Z1 and Z8; the runtime assert covers the rest.
5. Provenance: **fixed.**
   - The new runs record git_commit 204e78a with git_dirty False. Train pools are resampled per seed (start = s*n_train). E3-sybil has 3 seeds. The E3 and fresh102k runs are committed (e5039a6), and all README links resolve.
   - Residue: the old abc-e2z8 and grid JSONs still say "unknown" (the READMEs name 605faec and f6ba763). The grid summary still says partial: true. Each result.json holds only one seed: seed 0, except E3-sybil, which holds seed 1.
6. Rank-free factorial: **fixed.** The full 2x2 (mask x readout) is run with both losses, and rank-free is lower in 12 of 12 cells (verified). Caveat: rank-free also drops the slot embeddings, so the mask effect and the embedding effect are not separated.
7. XOR envs: **fixed in substance.** Z5 and Z6 were run with 3 seeds at 3000x512 fresh episodes; Z6-parity3 is held out, and the README says so.
   - Spot-check, Z6: every non-none arm is at or below 0.0069, and A_attn_none is 0.2466. Matches the claim.
   - Z5: A_attn_none is 0.1981 (not 0.2466). A_attn_first (0.0120), B_loss_every_prefix (0.0135) and B_..._reperm (0.0147) are above 0.01, so the README's "every single non-none architecture below 0.01" is false.
   - The bars also reach about 0.003 on Z5 and Z6, so these envs do not separate the architectures.

Spot-check E2: bar_fct_piece_bidir 0.0309±0.0025 vs grid_rankfree_cellT1 0.0320±0.0006, both match the JSON. The 0.0011 gap is smaller than the bar's SD. The bar was still improving at step 2000; the grid early-stopped with its best at step 600. The "2.3-nat gap" is not in either README, and the README's "within 0.001-0.004 nats" checks out.

## Verdict
The numbers in the JSONs are sound and usable as an EXPLORATORY baseline: commits and seeds are recorded, the matched bars are valid, and Z8 prefix Bayes is exact. The new READMEs are **NOT usable as written.** Remaining must-fixes for gemini:
- (a) Regenerate both §1 baseline tables from the JSONs (Z8, E3-sybil, Z5). State that ext_log_pool (0.0344) beats every causal arm on Z8.
- (b) Relabel the linear trader arms as 49-d with report/4, and rewrite §2 diagnosis 1.
- (c) Add the EXPLORATORY status to both tt-gate READMEs, and remove "every single…below 0.01", "Key Findings", "completely eliminates" and "strictly helps". Mark distill as oracle in the prose.
