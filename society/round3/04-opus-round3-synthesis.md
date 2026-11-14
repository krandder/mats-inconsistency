# Round 3 Synthesis & Critique: Adjudication of `R11-GPU-NEURAL` and Pre-Registration of `R12-GPU-NEURAL-V2`

- **Document ID:** `R3-4c-SYNTHESIS-CRITIQUE`
- **Author:** Member 4c (Claude Opus 5.5 [1m], via `claude-any`), Round-3 Synthesis & Critique
- **Date:** 2026-10-18
- **Repository:** `futarchy-fi/inconsistency-search`
- **Evidence base:** I used only the inline excerpts of `RESULTS-R11-GPU-NEURAL.md`, `results/r11_gpu_neural_results.json` (as quoted), `GPU-LOG.md` lines 2177–2189 (as quoted), and Member 14's `01-round3-r11-neural-audit.md`. I ran no tools and read no files.
- **Labeling convention:**
  - **[REPORTED]** means a number is quoted verbatim from the inline artifacts.
  - **[DERIVED-4c]** means I computed the number by hand from reported numbers. These derivations should be re-checked mechanically.
  - **[PRIOR-4c]** means a forecast I am committing to before Round 12.
  - **[NOT ATTESTED]** means the number is not present in the inline context. I do not reconstruct such numbers from memory.

---

## 1. Header & Executive Adjudication of `R11-GPU-NEURAL`

### 1.1 Bottom line

`R11-GPU-NEURAL` is the first round in this program where I can certify the following:

1. A randomly initialized neural world model was trained by SGD on a real `NVIDIA L4`.
2. It was evaluated on blind test batches whose oracle fields were replaced by trapping sentinels.
3. It produced a positive, log-linear test-time compute-scaling signal for the GLC tree-search family. Every flat baseline degraded.

That changes the epistemic status of the project. In Round 2 my Blind-Empirical score averaged **1.83**, because the surrogate runners contained oracle interpolation (e.g. `0.55 * p_true + 0.45 * p_anc`) and seed-conditional branches. Member 14 reports that every one of those constructs has been removed. The runtime canaries are bitwise invariant.

However, the claim that R11 is now near the top of the scale (Member 14's 8.23 mean) is not supported by the artifacts. Section 2.5 lists twelve specific problems I found in the reported numbers. Four of them materially weaken headline claims:

| # | Problem | Headline claim affected |
| :-- | :-- | :-- |
| 1 | The pairwise Elo graph is partially built from belief metrics. | `R11-D3` alignment and `R11-D2` general factor |
| 2 | `AUROC(m_ql) = 1.0000` is tautological. | Integrity check `F5.1a` |
| 3 | The `Unified-Generalist` fails the `R²_red ≥ 0.92` bar (`0.9155`). | `R11-D1` |
| 4 | The aggregate single-agent KL is **U-shaped** in `B`, with its minimum at `B = 8`. Compute scaling is monotone in Mini-ECI but not in belief quality. | `R11-D1` and anytime monotonicity |

My realized score is **6.33 / 10**. That is a real lift of **+4.50** over my Round-2 Blind-Emp score and **+0.83** over my Round-2 Proj-R11 projection. It falls well short of the auditor's 8.23.

### 1.2 Adjudication of the `R11-D0` Zero-Oracle-Leakage Gate

| Sub-gate | Evidence [REPORTED] | 4c verdict | 4c caveat |
| :-- | :-- | :-: | :-- |
| **S1–S5** (static AST rules) | `0` violations across `glc_aom.py`, `kv_tree_search.py`, `run_r11_gpu_neural.py`. Exemption list contains only training, calibration, stripping and post-search scoring functions. None of the search-execution functions are exempt. | **PASS** | See S6 for the canary mapping. |
| **S6** (synthetic canaries) | `3/3` caught | **PASS, with a documentation defect** | The two documents disagree on which rule caught each canary. The Results report says canary 2 was caught by S2 (`FORBIDDEN_META_KEY`) and canary 3 by S3 (reflection). Member 14 says canary 2 (`mechanistic_redundancy_delta_c`) was caught by S3 and canary 3 (`seed == 102 and b_budget == 2`) by S4. The S6 table therefore cannot be cited until the canary→rule mapping is reconciled. |
| **R1** (`PoisonTensor` traps) | `3/3` `OracleLeakError` per seed | **PASS** | Sound. A read raises an error rather than returning a plausible value. |
| **R2** (label-permutation and zeroing invariance) | `global_max_diff = 0.000000000` (12 envs × seeds 101, 102, 103, model plus `TreePagedKVSearch`) | **PASS** | This shows test-time outputs do not depend on test labels. It does **not** test train/test instance overlap. R12 must add split-hash disjointness (§3.5, H1). |
| **R3** (training actually happened) | `‖Δθ‖₂ = 17.9362`; loss `15.0813 → 0.6579` (−95.6%); B=0 KL `+14.85%` better than uniform | **PASS on the pre-registered bar, but weaker than it looks** | (a) The gate ledger says "94.9%" while §3.2 says "−95.6%". (b) The Results report says "Step 1 → Step 384", while the audit says "Step 0 → Step 179" with `generalist_steps = 180`. The 15 steps/env × 12 envs arithmetic supports **180**, so "384" (also used in the D4 diagnosis) is wrong. (c) **B=0 KL versus θ_init is −1.18% on average** (seed 101: −18.94%). On the unweighted seed mean, the trained generalist's root beliefs are no better than its random initialization. The "+14.85% vs uniform" is mostly carried by seed 103 (+32.67%). |
| **R4** (KV-cache equivalence) | `r4_kv_equiv_max_diff = 0.000000000` | **PASS** | Sound to B=128. It must be re-proven at B=256 (§3.4). |
| **Cost** | `$0.08` (attempt 1, `soup-gpu-r11gpu-1018040509-13412`) + `$0.58` (attempt 2, `soup-gpu-r11gpu-1018041347-24711`) = **`$0.66`** against a `$3.50` cap. Both VMs deleted. Ledger `320.32 → 320.98`. | **PASS** | `0.67 h × $0.87/h = $0.583` [DERIVED-4c]. Consistent. |

**Gate verdict: `R11-D0` PASS.** The defects above are documentation and interpretation defects, not leakage. I find no evidence that test oracle fields reached any model forward pass or search call. That is the single most important result of Round 11.

### 1.3 Tier-1 Analytic Operator Ceiling (`R10`/`Stage-2`) vs. Tier-1 Real End-to-End Neural Run (`R11-GPU-NEURAL`)

| Dimension | `R10`/`Stage-2` analytic operator ceiling | `R11-GPU-NEURAL` real neural (L4, 3 seeds) | Reality / ceiling | 4c interpretation |
| :-- | :-: | :-: | :-: | :-- |
| Generalist Mini-ECI, B=0→128 | `100.00 → 130.73` (+30.73) | `98.34 → 105.30` (**+6.95**) | 0.23 | About a quarter of the operator ceiling survives finite-step SGD. The ceiling measured what a perfect lemma emitter would buy, not what this emitter buys. |
| Pre-registered D1 bar | +18.0 | +6.95 | 0.39 | Honest miss. |
| Redundancy fit R²_red | 0.996 (circular: data generated by the fitted law) | 0.9443 (KV-Tree) / **0.9155** (Generalist) | — | Non-circular now, which is a real improvement. Only one of the two GLC cohorts clears 0.92. |
| Witness discovery | `14.2% → 88.5%` | `0.52% → 43.06%` | 0.49 at B=128 | Large real effect: 82× relative increase. Heavily concentrated in `equiv_sat` (100%) and ShipHunt (58–81%). Sudoku is nearly zero (5.6% / 0.0%). |
| Flat baselines | Mild gains (surrogate) | **All degrade**: Δ ∈ [−36.82, −10.72] | sign flip | The most diagnostic qualitative finding of the round (§2.1). |
| Leakage status | Oracle interpolation present (Round-2 finding) | Zero leakage (`R11-D0`) | — | The ceiling's numbers were never evidence. R11's are. |
| Spend | — | `$0.66` | — | Exceptional evidence-per-dollar. |

The correct framing is not "R11 fell short of R10." R10 was an upper bound under oracle-quality components. R11 is the first lower bound under real components. The gap between them (+6.95 vs +30.73) is the quantity Round 12 should close, and the three partial sub-bars (§2.3) are the map for closing it.

---

## 2. Scientific Analysis: Breakthroughs vs. Honest Negative/Partial Results

### 2.1 Why `GLC_Tree_Active_PUCT` scales while every flat baseline degrades

**Reported facts.**

| Cohort | ΔMini-ECI (128−0) | κ per doubling | R²_log | R²_red |
| :-- | :-: | :-: | :-: | :-: |
| `Unified-Generalist` | +6.95 | +0.7637 | 0.9241 | 0.9155 |
| `GLC-AOM-KV-Tree` | +5.43 | +0.7389 | 0.9110 | 0.9443 |

- Witness discovery rises from 0.52% to 43.06%.
- Mean Elo rises from 927.4 to 1195.6 (+268.1).
- Flat baselines all have ΔMini-ECI ∈ [−36.82, −10.72].

**Mechanism (consistent with the artifacts).** Under oracle-quality root marginals, naive pooling of sibling evidence is nearly harmless, because each branch's log-odds update is close to correct. Under real root marginals with correlated errors, flat pooling double-counts shared evidence. The conjunctive log-odds accumulate the same error once per branch. Two pieces of evidence support this:

- `Flat-Econ-Prec-K3` KL blows up from `0.5759 → 1.5017` [REPORTED, audit §4.1].
- `P3-Uniform-MC` Mini-ECI swings non-monotonically between 31.91 and 89.86 across budgets.

These are the signatures of an unpreconditioned, non-contractive update. `TwoStageTreeClearer` preserves a Gershgorin bound of `0.998281 < 1` (F7.1). It therefore cannot amplify root error geometrically with depth. This lowers the effective redundancy (ρ_IRT = 0.04 vs 0.52 for the Specialist) and keeps Gram crowding at 0.1536, versus 0.2085 for flat cohorts.

**What this result is.** It is the cleanest real-world confirmation so far of the operator-level thesis: preconditioned, contractive tree clearing is necessary for search to help when the world model is imperfect. The sign flip relative to the surrogate is exactly what a real test should reveal.

**What it is not.**

1. **It is not monotone belief improvement.** The Generalist's mean KL is minimized at **B = 8 (0.5225)** and rises to **0.5319** at B = 128. Brier behaves the same way (minimum 0.2479 at B = 8, 0.2548 at B = 128). Mini-ECI keeps rising because it is dominated by witness discovery and Elo items. Search keeps finding witnesses after it has stopped improving marginals. F7.2's stepwise bar (max rise 0.0045 ≤ 0.015) passes, but the cumulative rise from B=8 to B=128 is +0.0094 [DERIVED-4c]. A stepwise bar cannot detect slow drift. R12-D3 adds a cumulative bar.
2. **It is not "monotonic."** Mini-ECI is non-decreasing on 7/8 transitions; it dips from 103.25 to 102.76 at B=64. Mean Elo dips from 1123.6 to 1086.0 at B=64. "Near-monotonic" is the accurate word.
3. **κ alone does not discriminate scaling.** `Flat-Ataraxos-MMD` has **κ = +0.8256**, which is higher than the Generalist's +0.7637, despite Δ = −10.72 and R²_log = 0.0381. The report's statement "all flat baselines degrade (κ ∈ [−4.39, −0.65])" is therefore **factually wrong** for that cohort. The slope is meaningless without a fit-quality co-requirement. Conversely, `P8-Lead-Hybrid` has R²_log = 0.9191 for a *degrading* curve, so log-linearity alone is not evidence of beneficial scaling either. R12 bars must jointly require κ > 0, R²_log ≥ threshold, and Δ > 0.
4. **The significance claim is unverified.** "p < 0.01" for +6.95 is asserted, but no per-seed Mini-ECI deltas or t-interval are given. With df = 2 (t₀.₉₇₅ = 4.3027), significance at p < 0.01 requires a very tight seed spread. The per-seed values must be published.

### 2.2 The three full PASSes

#### `R11-D2` — Bifactor IRT (5/5 on the pre-registered bars)

| Statistic | Value [REPORTED] | Bar |
| :-- | :-: | :-: |
| PC1 | 64.75% (λ₁ = 5.828 vs λ₂ = 1.369) | ≥ 58% |
| ω_h | 0.8961 | ≥ 0.72 |
| ECV | 0.7350 | ≥ 0.68 |
| LOTO r | 0.7122 | ≥ 0.60 |
| Minimum λ_g | 0.4891 (Sudoku) | ≥ 0.45, all 9 items |

On the pre-registered bars this is a clean pass, and I score it as such for gate purposes. As psychometric evidence of a single general reasoning factor, it is weaker than the numbers suggest. There are three reasons:

- **Non-independent observations.** N = 108 "agents" are 12 model×search cohorts × 9 budgets. Budget replicates of one cohort share almost all their variance. This inflates PC1 and ω_h in the same way that repeated measures inflate a naive correlation. The effective N is closer to 12 than to 108.
- **Shared measurement channel.** The five strategic items are Elo ratings. Member 14 found (§5.1 of the audit) that the Elo graph is supplemented with pairwise matches scored as `payoff = −1.8·KL − 0.6·Brier + 0.45·witness_rate`. Part of the strategic items is therefore a linear function of the same belief metrics that define the single-agent items. Some of the "general factor" is a shared construction, not a shared ability.
- **Possible Heywood clamping.** Three items (Sudoku, Galanis, Poker) have communality exactly **0.9500**. Identical values at a round ceiling almost certainly indicate an upper clamp on h², applied to suppress Heywood cases (h² ≥ 1). Clamped items inflate ω_t (0.9803) and should be disclosed.

**Verdict:** PASS as pre-registered. The "general factor" claim stays provisional until it is re-estimated on interactive-only Elo with a cluster-robust specification (§3.5, H3–H4).

#### `R11-D3` — Elo alignment (3/3)

| Statistic | Value [REPORTED] | Bar |
| :-- | :-: | :-: |
| Pearson r(Mini-ECI, Elo) | 0.8862 | ≥ 0.85 |
| Spearman ρ(Mini-ECI, Elo) | 0.9652 | ≥ 0.85 |
| GLC@B=32 minus Flat-Mix@B=32 | +751.3 ALL Elo | ≥ +120 |

Arithmetic check: +751.3 = 1080.3 − 329.0, +928.4 = 1123.6 − 195.2, and +1654.8 = 1080.3 + 574.5 [DERIVED-4c, all exact]. The same circularity caveat applies with more force here. Mini-ECI and Elo share the KL, Brier and witness inputs through the payoff-scored matches, so r = 0.886 is partly built in.

There is also a labeling ambiguity. The Results report calls `1080.3 / 1123.6` "GLC-AOM@B=32". In the §6 table, however, 1080.3 is the **`Unified-Generalist@B=32`** row; `GLC-AOM-KV-Tree@B=32` is 1081.6. Both comfortably clear +120, so the verdict is unaffected, but the cohort must be named precisely.

Interactive-only evidence does exist. On Hex-5x5 and Connect-4 (genuine turn-by-turn play), GLC@B=32 sits at roughly 954–1010, while Flat-Mix sits at 735.6 and 526.8. The direction of the effect survives without payoff-scored matches; its magnitude is not established. **Verdict:** PASS as pre-registered, with the magnitude unproven.

#### `R11-D5` — LIR Causal Firewall (5/5 on 3-seed means)

| Sub-bar | Value [REPORTED] |
| :-- | :-- |
| Gradient blockade | 0.000000 |
| Post-RL belief drift | 0.000000 |
| D_bias(B=0) | 0.000000 |
| Action flip rate | 85.42% (seeds: 75.0%, 100.0%, 81.25%) |
| Skewed belief KL | 0.2031 (LIR) vs 0.2612 (Fused), −22.25% (DERIVED-4c: −22.24%, rounding-consistent) |

The exact zeros are what an architectural `stop_gradient` firewall should produce. They are correct, but they hold by construction, so they verify implementation rather than discover anything. The scientifically informative sub-bars are the flip rate (beliefs fixed, actions respond to utility) and the skewed-KL comparison.

Three caveats I could not leave out:

1. **Per-seed robustness is 2/3, not 3/3.** On seed 101, LIR *loses* both comparative sub-bars:
   - F4.4 skewed KL: 0.2059 (LIR) vs 0.2004 (Fused).
   - F4.5 Elo: 1070.8 (LIR) vs 1080.1 (Fused).

   The 3-seed-mean PASS is legitimate under the pre-registration. "5/5 PASS" without this disclosure overstates robustness.
2. **Arithmetic error.** The reported "+526.2 mean Elo" should be 1123.6 − 593.4 = **+530.2** [DERIVED-4c]. The difference is linear, so seed-averaging cannot explain a 4.0-point discrepancy. It is minor in magnitude, but it **directly contradicts the auditor's "zero transcription discrepancies"** certification.
3. **The firewall may carry a belief-quality cost the report does not discuss.** `Fused` has the **lowest aggregate mean KL of any GLC cohort** (0.4647 at B=128, versus 0.5319 for the LIR-masked Generalist). The gap is driven largely by `sudoku_9x9_extreme` (1.1153 vs 2.5862). Fused also has the worst Brier (0.3321) and the worst Mini-ECI of the GLC family. So the firewall is not free: on deductive single-agent tasks, policy-side gradients appear to *help* belief sharpness, at a cost in calibration and strategic play.

   This is a genuine trade-off, not a refutation. It should be reported as a fourth honest partial finding. The tension is also visible in the alignment statistics: r(Mini-ECI, −KL) = 0.74 but ρ = 0.50.

### 2.3 The three honest partial/negative sub-bars and their root causes

#### (a) `R11-D1`: +6.95 real neural Mini-ECI vs the +18.0 analytic ceiling

**Root cause.** A tree search can only redistribute and verify what the world model's root and branch marginals encode. The evidence for an under-trained model:

- The generalist receives **15 gradient steps per environment**, about 120 samples per env.
- Its B=0 beliefs are, on the seed mean, no better than its initialization (−1.18% vs θ_init).

With weak branch marginals, the search gain is bounded mainly by witness verification, which is a discrete, model-independent signal. Belief refinement contributes little. That is exactly the pattern observed: witnesses rise to 43.06% while KL bottoms out at B=8.

The redundancy fit supports the same reading. α = 1.893 with ρ_IRT = 0.04 means low redundancy, so the gain is not saturated by repeated evaluation of the same states. The tree is limited by node quality, not by node diversity. The non-saturating jump from B=64 to B=128 (+2.54 Mini-ECI) also suggests that a larger budget would yield more. That motivates R12-D3.

**Prediction from this diagnosis:** Iso-Update training (R12-D2) should raise ΔMini-ECI more than raising B alone will.

#### (b) `R11-D4`: query steering wins on variance and accuracy, fails on raw `query_kl`

Reported facts for `A4_GLC_BilinearAux_QuerySteered`:

| Metric | Equiv-SAT (A4 vs A3_Random) | Sudoku-6x6 (A4 vs A3_Random) |
| :-- | :-: | :-: |
| Var KL at B=16 | 0.0079 vs 1.6535 (209×; DERIVED-4c: 209.3×) | 0.8518 vs 3.5453 (4.2×; DERIVED-4c: 4.16×) |
| Peak `query_acc` (B=16) | 58.1% (highest of 4 arms) | 78.5% (highest of 4 arms) |
| `query_kl` at B=32 | 0.8735 | 1.0219 |
| `query_kl` bar | ≤ 0.15 | ≤ 0.15 |

**Root cause** (Member 14, audit §4.4, code-level):

- `query_head` is trained only on 1-step sibling consensus, `h_cons_sib = h_cons + 0.5·(v_exp_true + v_exp_alt)`.
- At test time, `h_cons_curr` accumulates up to 32 cleared updates, which inflates its norm in the Z^W_aux subspace.
- With no depth normalization and no temperature, logits become overconfident. The same argmax yields higher accuracy, but miscalibrated magnitudes yield higher Bernoulli KL.

This is a textbook case of accuracy and calibration coming apart, and the diagnosis is convincing.

The report's framing of A4 also needs correcting:

1. **A3 is not the right comparator for the variance claim.** On Equiv-SAT, A4's Var KL (0.0079) is **identical to `A1_NoAux` (0.0078–0.0080)**. On Sudoku-6x6 at B=16 it is also identical (A4 0.8518 vs A1 0.8519). The "209×" is A4 *avoiding damage that A3 does*, not A4 improving on having no auxiliary channel. The correct claim is "query steering makes the bilinear channel safe." The stronger claim, "query steering makes it beneficial for variable marginals," is unsupported.
2. **`A1_NoAux` beats A4 on `query_kl` at every B > 0** (Equiv-SAT 0.5189 vs 0.8735; Sudoku 0.4743 vs 1.0219 at B=32).
3. **The accuracy advantage over A1 is small.** At peak: Equiv-SAT 58.1% vs 56.0%; Sudoku 78.5% vs 77.1%. Equiv-SAT witness rates move in steps of 2.1% (2.1, 18.8, 95.8), which implies about 48 instances, so 2 percentage points is about one instance. Three seeds cannot establish this.
4. **Two other D4 sub-bars do not discriminate A4.** Equiv-SAT witness reaches 100% at B ≥ 16 for **all four** arms. PR(A4) = 0.081 is below PR(A3) = 0.084 and PR(A2) = 0.046–0.056 is not far off.

**The important implication for Round 12:** `query_kl` at **B = 0** is already **0.5224** (Equiv-SAT) and **0.5157** (Sudoku), and it is identical across all four arms. A post-hoc 2-parameter temperature/bias fit cannot reduce KL below what the model's ranking supports. If B=0 sits at 0.52, the ≤ 0.15 bar at B=32 requires the search itself to add substantial information. Calibration alone removes the *depth-induced excess* (≈ 0.35 nats on Equiv-SAT, ≈ 0.51 on Sudoku) and nothing more. My [PRIOR-4c] is therefore that **R12-D1 will fail the absolute 0.15 bar** and pass the relative bars I specify in §3.2. Saying so in advance is what keeps the bar honest.

#### (c) `R11-D6`: Generalist vs Specialist

Reported facts:

| Metric | Generalist | Specialist |
| :-- | :-: | :-: |
| Parameters | 2.04M | 24.43M |
| Steps per env | 15 | 30–60 |
| Mean SA KL (B=128) | 0.5319 | 0.7896 (Generalist −32.6%; DERIVED-4c −32.64%) |
| κ | +0.7637 | +0.1881 (ratio 4.06×; DERIVED-4c 4.060×) |
| Witness rate | 43.06% | 39.41% |
| Mean MP Elo at B=32 | 1123.6 | 1438.0 |

**Root cause of the Elo gap** (audit §4.6): a 2–4× per-environment gradient asymmetry, plus multi-game interference in a shared trunk with no game conditioning. I accept this as the leading hypothesis. R12-D2 is designed to falsify it.

What the report under-states:

1. **The Specialist dominates Mini-ECI at every budget.** The Specialist at **B = 0 (106.26)** already exceeds the Generalist at **B = 128 (105.30)**, and the Specialist at B=128 reaches 108.37. The Generalist wins on *slope*, not on *level*. Part of the 4.06× slope ratio is a ceiling effect: the Specialist starts near its plateau (R²_log = 0.4396).
2. **The KL win is narrow by environment.** Per-env in §9.1 at B=128, the Generalist wins clearly on `equiv_sat`, `sudoku_6x6` and `sudoku_9x9_extreme`. It is roughly tied on `shiphunt_6x6` and `minesweeper`. **The Specialist wins on 7 of 12 environments**: `shiphunt_8x8`, `hex`, `connect4`, both `micro_stratego` variants, `galanis`, and `poker`.
3. **The headline KL means cannot be reproduced from the per-env table.** I averaged the §9.1 per-env KLs [DERIVED-4c]:

   | Averaging | Generalist B=128 | Specialist B=128 | Generalist reduction |
   | :-- | :-: | :-: | :-: |
   | Unweighted, 12 envs | 0.4491 | 0.5673 | −20.8% |
   | Unweighted, 6 single-agent envs | 0.6573 | 0.9928 | −33.8% |
   | Reported (§4.2) | 0.5319 | 0.7896 | −32.6% |

   The Generalist's B=0 12-env mean is 0.4784, against a reported 0.5759. The aggregation weights (per-variable? per-instance?) are undocumented. Depending on the averaging rule, the headline −32.6% becomes anything from about −21% to −34%. It must be re-derived with explicit weights.
4. **The parameter ledger is internally inconsistent.** "12 × 2.04M = 24.43M" is arithmetically right (12 × 2,035,868 = 24,430,416). But the audit says each Specialist trains on "a single family (1 or 2 environments)". Sudoku, ShipHunt and Micro-Stratego each pair two environments, which implies **9 specialists (≈ 18.3M)**, not 12. One of the two descriptions is wrong.
5. **The root-identity anchor is ambiguous.** `Unified-Generalist@B=0` scores Mini-ECI 98.34 and Elo 988.9, while the anchor `1-pass::B0_root@B=0` scores 100.0 and 1000.0. F1.1 asserts Search(B=0) ≡ Forward() exactly. If `B0_root` is the same model's root pass, the two rows should coincide. If it is a different model, the report must say which.

### 2.4 Integrity panel notes (`F1`–`F10`)

- **F5.1a `AUROC(m_ql) = 1.0000` is void.** Its label is `m_ql ≥ median(m_ql)`, i.e. the metric is scored against itself (audit §5.2). F5.1b (0.8038) is a test of m_qs against m_ql, not against ground truth.
- **F6.2a plan flip is 97.92% on all three seeds.** That plausibly equals 47/48 deterministically. It is fine, but it reflects instance granularity rather than seed variance.
- **F6.2b is strong:** ShipHunt-8x8 KL `1.5435 → 0.0715` with the active plan.
- **F7.1 Gershgorin bound is 0.998281.** This is only 0.0017 from the instability boundary at B=128. That margin is the main technical risk for R12-D3.

### 2.5 Consolidated discrepancy ledger (for the R12 pre-registration)

| ID | Finding | Source | Severity |
| :-- | :-- | :-- | :-: |
| X1 | Loss drop reported as 94.9% (gate ledger) and 95.6% (§3.2) | Results report | Low |
| X2 | Training step count 384 (report) vs 180 (audit; 15 × 12) | Results report | Medium (feeds the D4 diagnosis text) |
| X3 | S6 canary→rule mapping differs between report and audit | Both | Medium |
| X4 | "+526.2" mean Elo should be +530.2 | Results report; audit's zero-discrepancy claim | Low in size, high for audit credibility |
| X5 | `Flat-Ataraxos` κ = +0.8256 contradicts "κ ∈ [−4.39, −0.65]" | Results report | Medium |
| X6 | Generalist R²_red = 0.9155 < 0.92; the pass is shown via a different cohort | Results report | Medium |
| X7 | Mean KL not reproducible from per-env table | Results report | High |
| X8 | Specialist count: 12 vs about 9 families | Report and audit | Medium |
| X9 | `AUROC(m_ql)` tautological | Audit-identified | High (void sub-bar) |
| X10 | Elo partially belief-payoff scored, so Mini-ECI↔Elo is partly circular | Audit-identified; amplified by 4c | High |
| X11 | Three communalities clamped at 0.9500; N=108 not independent | 4c | High (for D2 interpretation) |
| X12 | B=0 KL vs θ_init = −1.18% (seed 101: −18.94%) | Audit footnote | Medium |

None of these is leakage. All of them are fixable. Several of them (X7, X9, X10, X11) affect what the passes *mean*, which is why my realized score is well below the auditor's.

---

## 3. Pre-Registration Specification for Round 12 (`R12-GPU-NEURAL-V2` on `NVIDIA L4`)

### 3.1 Realized R11 numbers vs my Round-2 `[PRIOR]` table, and why the partials matter

| Quantity | My Round-2 [PRIOR] | R11 realized | Assessment |
| :-- | :-: | :-: | :-- |
| Generalist ΔMini-ECI (B=0→128) | **[+3.5, +11.0]** (attributed to me by Member 14, audit §4.1) | **+6.95** | Inside the band, near its geometric centre. Calibrated forecast. |
| SA KL, Generalist vs Specialist | [NOT ATTESTED] | −32.6% (reported; −21% to −34% depending on weights, X7) | Cannot be honestly scored post hoc. |
| LIR action flip rate | [NOT ATTESTED] | 85.42% | Cannot be honestly scored post hoc. Bar was ≥ 50%. |
| PC1 | [NOT ATTESTED] | 64.75% | Bar was ≥ 58%. |
| ω_h | [NOT ATTESTED] | 0.8961 | Bar was ≥ 0.72. |

I will not reconstruct the unattested prior entries from memory, because a prior recalled after seeing the outcome is not a prior. The one attested entry was confirmed. My Round-2 Proj-R11 mean of **5.50** was **conservative by +0.83** relative to my realized score (6.33), and it was correct in predicting that the run would *not* reach the analytic ceiling.

**Why reporting the three partial sub-bars is a triumph of scientific integrity.** Round 10 produced a +30.73 Mini-ECI gain, R²_red = 0.996 and witness rates of 88.5%. Those numbers came from runners whose outputs were partly *interpolated toward the answer*. Round 11 removed the interpolation, and three bars that the surrogate had "passed" turned into honest partials:

1. **+6.95 instead of +18.0 is the measured value of the architecture under real weights.** A surrogate that smoothed it up to 18 would have hidden the main open problem, branch-marginal quality, behind a constant.
2. **The `query_kl` blowup exposes a specific mechanism:** depth-accumulated logit scale. Smoothing would have erased it, and with it the most actionable engineering finding of the round. A surrogate cannot have a calibration bug, because it has no logits to miscalibrate.
3. **The 15 vs 30–60 steps/env Elo gap exposes a confound in the experimental design itself.** That is exactly the kind of finding that only appears when the experiment is real enough for its design to matter.

Each partial converts directly into a falsifiable R12 hypothesis. A program that reports misses this specifically has earned the right to have its passes believed, after the caveats in §2.5 are cleared.

### 3.2 `R12-D1`: Calibrated Bilinear Query Readout on `split="cal"`

#### Hypothesis

The excess of `query_kl(A4, B=32)` over `query_kl(A4, B=0)` (≈ +0.35 nats on Equiv-SAT, +0.51 on Sudoku-6x6) is caused by depth-accumulated norm inflation in the Z^W_aux subspace. It is removable by:

- depth RMS normalization, plus
- a 2-parameter post-depth affine calibration fitted only on `split="cal"`,

without loss of `query_acc`.

#### Equations

Let `h_b ∈ ℝ^{d}` be the accumulated consensus state after `b` cleared tree steps, and let `P_W` project onto the Z^W_aux coordinates (d_aux = 64). Define:

$$
r_b = \sqrt{\tfrac{1}{d_{\text{aux}}}\,\|P_W h_b\|_2^2 + \epsilon}, \qquad
\tilde h_b = h_b - P_W h_b + \frac{\rho_1}{r_b}\,P_W h_b,
$$

where `ρ₁` is a non-trainable buffer equal to the EMA of `r₁` over the 1-step sibling-consensus states seen in training. Only the auxiliary subspace is rescaled, so it matches the training-time scale. The query head is unchanged:

$$
\ell_q(b) = w_q^\top \tilde h_b + c_q .
$$

The calibrated logit and probability are:

$$
\hat\ell_q(b) = \frac{\ell_q(b)}{\tau_q} + b_q, \qquad \tau_q = \operatorname{softplus}(s_q) + 10^{-3}, \qquad \hat p_q(b) = \sigma(\hat\ell_q(b)).
$$

`(s_q, b_q)` minimize the soft-target Bernoulli cross-entropy on the calibration split, pooled across `B ∈ {0, 2, 4, 8, 16, 32}`, with all other parameters frozen:

$$
(s_q^\star, b_q^\star) = \arg\min_{s,b} \sum_{(x,b)\in\mathcal{D}_{\text{cal}}} -\big[y\log\hat p_q + (1-y)\log(1-\hat p_q)\big].
$$

The fit is **shared across depths**. A depth-indexed (τ_q^{(b)}) variant is an *exploratory* arm only. It is not used for the bar, because 12 parameters fitted on ~48 instances per environment would invite overfitting.

#### PyTorch specification

```python
class DepthRMSNorm(nn.Module):
    def __init__(self, aux_slice: slice, eps: float = 1e-6, ema: float = 0.99):
        super().__init__()
        self.aux, self.eps, self.ema = aux_slice, eps, ema
        self.register_buffer("rho1", torch.ones(()))      # set during training only

    @torch.no_grad()
    def update_rho1(self, h1: torch.Tensor):              # called on 1-step h_cons_sib in train
        r1 = h1[..., self.aux].pow(2).mean(-1).add(self.eps).sqrt().mean()
        self.rho1.mul_(self.ema).add_((1 - self.ema) * r1)

    def forward(self, hb: torch.Tensor) -> torch.Tensor:
        z = hb[..., self.aux]
        rb = z.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        out = hb.clone()
        out[..., self.aux] = z * (self.rho1 / rb)
        return out

class QueryCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.s_q = nn.Parameter(torch.tensor(0.5413))      # softplus(0.5413) ≈ 1.0
        self.b_q = nn.Parameter(torch.zeros(()))
    def forward(self, logit_q):
        tau = F.softplus(self.s_q) + 1e-3
        return logit_q / tau + self.b_q
```

#### Fitting protocol

- `calibrate_query_temperature(model, cal_batches)` is the only function added to `EXEMPT_SCORING_OR_TRAINING_FUNCTIONS`. Its exemption is justified by an AST rule that **forbids it from receiving any object derived from `split="test"`**. This is checked by a provenance tag on `EnvBatch.split`, asserted at function entry.
- Optimizer: `torch.optim.LBFGS([s_q, b_q], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")`, with model weights under `torch.no_grad()` and `requires_grad_(False)`.
- After fitting, `(τ_q, b_q)` are frozen, serialized, and SHA-256-hashed **before** any test batch is loaded. The hash goes into the results JSON.
- Test batches pass through `strip_oracle_fields`, and the R1 `PoisonTensor` traps remain active. `query_targets` on test are consumed only by `evaluate_blind_predictions` after `search()` returns.
- **Fairness control:** `A1_NoAux`, `A2_LinearAux` and `A3` receive the identical DepthRMSNorm + calibration treatment. R11 showed that A1 beats uncalibrated A4 on `query_kl`, so the R12 claim must be that calibrated A4 beats calibrated A1.

#### Pass/fail bars (3-seed mean; R12 also reports 5-seed results if budget allows, §3.5)

| Bar | Criterion | [PRIOR-4c] |
| :-- | :-- | :-- |
| **D1.a (primary, absolute, retained from R11)** | `query_kl(A4_cal, B=32) ≤ 0.15` on **both** Equiv-SAT and Sudoku-6x6 | **FAIL** (p ≈ 0.85). Predicted: Equiv-SAT ∈ [0.40, 0.55], Sudoku ∈ [0.38, 0.52]. The B=0 floor is ≈ 0.52. |
| **D1.b (depth-excess removal)** | `query_kl(A4_cal, B=32) − query_kl(A4_cal, B=0) ≤ +0.02` on both envs | PASS (p ≈ 0.75) |
| **D1.c (search adds information)** | `query_kl(A4_cal, B=32) < query_kl(A1_cal, B=32)`, one-sided paired t over seeds, p < 0.05 | Uncertain (p ≈ 0.45) |
| **D1.d (no accuracy cost)** | `query_acc(A4_cal, B=16) ≥ query_acc(A4_uncal, B=16) − 1 instance` | PASS (p ≈ 0.90). The affine map with τ > 0 preserves sign, except where b_q shifts the threshold. |
| **D1.e (calibration quality)** | 15-bin ECE of `p̂_q` on test ≤ 0.05 at B=32 | PASS (p ≈ 0.65) |
| **D1.f (leakage)** | R11-D0 S1–S6 + R1–R4 re-pass, including a **new canary**: a calibrator that reads a test batch must be caught | Required gate |

The absolute bar D1.a is **not relaxed**. If D1.b passes and D1.a fails, the correct conclusion, pre-registered here, is: "the depth bug is fixed; the remaining gap is world-model quality, not calibration."

### 3.3 `R12-D2`: Iso-Update, Game-Modulated `Unified-Generalist` vs `Single-Game-Specialist`

#### Hypothesis

The R11 multiplayer Elo deficit (1123.6 vs 1438.0) is caused by:

- **H2a:** per-environment update asymmetry (15 vs 30–60 steps/env), and
- **H2b:** gradient interference in an unconditioned shared trunk.

Under equalized updates plus lightweight game conditioning, the Generalist reaches non-inferiority on *pure interactive* Elo and keeps its single-agent KL and κ advantages.

#### Arms (all trained from random initialization; the same seeds per arm)

| Arm | Params | Updates/env | Total updates | Purpose |
| :-- | :-: | :-: | :-: | :-- |
| `Spec-Iso` | 2,035,868 per family specialist (count published explicitly; resolves X8) | 60 | 60 per specialist | Reference |
| `Gen-R11` | 2,035,868 | 15 | 180 | R11 replication |
| `Gen-Iso` | 2,035,868 | 60 | 720 (round-robin) | Tests H2a |
| `Gen-Iso-FiLM` | 2,035,868 + 12,288 | 60 | 720 | Tests H2a + H2b |
| `Gen-Iso-FLOP` (control) | 2,035,868 | 60 at matched batch | matched to total Spec FLOPs | Separates "more updates" from "more total compute" |
| `Gen-Iso-LoRA2` (exploratory) | + 49,152 | 60 | 720 | Exploratory only; exceeds the 2% overhead cap |

#### FiLM conditioning

The environment identity `e` comes from the **public header token**, which the AST auditor must verify is not on the oracle list. The header id is not a private `meta` key. For each layer `ℓ ∈ {1..4}`:

$$
\mathrm{FiLM}_\ell(h; e) = (1 + \gamma_{\ell,e}) \odot \mathrm{LN}(h) + \beta_{\ell,e}, \qquad \gamma_{\ell,e}, \beta_{\ell,e} \in \mathbb{R}^{128},\ \text{zero-initialized}.
$$

Parameter overhead is `4 × 2 × 128 × 12 = 12,288` (0.60% of 2.04M, within the 2% cap). For comparison, a LoRA of rank r on W_Q and W_V costs `4 × 2 × (2 × 128 × r) × 12 = 24,576·r`. Rank 1 gives 1.2% and rank 2 gives 2.4%; that is why LoRA is exploratory.

```python
class EnvFiLM(nn.Module):
    def __init__(self, n_env=12, d=128, n_layers=4):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(n_layers, n_env, d))
        self.beta  = nn.Parameter(torch.zeros(n_layers, n_env, d))
    def forward(self, h, layer: int, env_id: torch.LongTensor):  # env_id from public header
        g = self.gamma[layer, env_id].unsqueeze(1)                 # [B,1,d]
        b = self.beta[layer, env_id].unsqueeze(1)
        return (1 + g) * h + b                                     # applied post-LN
```

FiLM parameters belong to the world-model trunk (V-side) and the policy trunk separately. The LIR firewall test (F1.3 gradient blockade = 0) must re-pass with FiLM inserted.

#### Interactive-only Elo protocol (fixes X10)

- **Only genuine alternating-move play** via `t_env.play_match(pol_a, pol_b, ...)` on Hex-5x5, Connect-4, Micro-Stratego-4x4/6x6, and the sequential protocols of Galanis-Market and Proposal-Poker.
- **No belief-payoff pairwise matches** enter the primary Elo. Belief-payoff ratings may be reported in a separate, clearly labeled table.
- At least **64 games per ordered pairing per env per seed**, with seat swap.
- Bradley–Terry MLE with a weak Gaussian prior (σ = 400). Ratings get 1,000-resample bootstrap 95% CIs over games, and the seed mean gets a t-interval (df = n_seeds − 1).
- The anchor is fixed and named. `1-pass::B0_root` must be explicitly documented as `Gen-Iso@B=0` (or not); this resolves the anchor question in §2.3(c).

#### Pass/fail bars

| Bar | Criterion | [PRIOR-4c] |
| :-- | :-- | :-- |
| **D2.a (MP non-inferiority)** | Interactive ALL-league Elo(`Gen-Iso-FiLM@B=32`) − Elo(`Spec-Iso@B=32`) ≥ −50, with the lower 95% CI bound ≥ −120 | PASS (p ≈ 0.55) |
| **D2.b (H2a isolation)** | Elo(`Gen-Iso`) − Elo(`Gen-R11`) ≥ +100 | PASS (p ≈ 0.70) |
| **D2.c (H2b isolation)** | Elo(`Gen-Iso-FiLM`) − Elo(`Gen-Iso`) ≥ +40 | Uncertain (p ≈ 0.45) |
| **D2.d (SA KL retained)** | Mean SA KL(`Gen-Iso-FiLM`, B=128) < Spec-Iso, using **pre-declared weights**: unweighted mean over the 6 single-agent envs, with per-variable-weighted reported as secondary (fixes X7) | PASS (p ≈ 0.70). The Specialist also gains from Iso, so the margin will shrink below −32.6%. |
| **D2.e (scaling retained)** | κ(`Gen-Iso-FiLM`) > κ(`Spec-Iso`) **and** R²_log(Gen) ≥ 0.85 **and** Δ(128−0) > 0 | PASS (p ≈ 0.65) |
| **D2.f (level, newly reported)** | Mini-ECI(`Gen-Iso-FiLM`, B=128) ≥ Mini-ECI(`Spec-Iso`, B=128) − 1.0 | Uncertain (p ≈ 0.40). R11 showed the Specialist higher at every B. |
| **D2.g (root learning)** | B=0 KL vs θ_init improves on **every** seed (fixes X12) | PASS (p ≈ 0.80) |

**Falsification:** If D2.b fails, H2a (update asymmetry) is refuted as the main cause of the Elo gap. The program must then look for representational causes, such as capacity or the absence of per-game value heads.

### 3.4 `R12-D3`: Ultra-Deep KV-Tree Search up to `B = 256`

#### Budget grid

`B ∈ {0, 1, 2, 4, 8, 16, 32, 64, 128, 256}` (10 points, 9 doublings).

#### Spectral contraction

Let `T_b` be the preconditioned clearing operator applied at tree step `b`, and let `𝒯_B = T_B ⋯ T_1` be the composite. R11 reached ‖T_b‖_∞ ≤ 0.998281, a margin of only 1.7 × 10⁻³. Per-step contraction does not guarantee bounded *accumulated perturbation* at depth 256. Error terms injected at each step sum geometrically with ratio q = max_b ‖T_b‖_∞, so the bound is ε_total ≤ ε/(1−q) ≈ 580·ε at q = 0.99828 [DERIVED-4c].

R12 therefore introduces an explicit margin:

$$
T_b \leftarrow \frac{1-\delta}{\max\!\big(1-\delta,\ \|T_b\|_\infty\big)}\,T_b, \qquad \|T_b\|_\infty = \max_i \sum_j |(T_b)_{ij}|, \qquad \delta = 5\times10^{-3},
$$

computed in `float64` on each clearing call. This guarantees q ≤ 0.995 and an amplification bound ≤ 200. The rescale is logged whenever it activates, together with its activation frequency.

```python
def enforce_contraction(T: torch.Tensor, delta: float = 5e-3):
    n_inf = T.double().abs().sum(-1).amax(-1)                      # per-batch ‖T‖_∞
    scale = (1 - delta) / torch.clamp(n_inf, min=1 - delta)
    return (T.double() * scale[..., None, None]).to(T.dtype), n_inf
```

#### Paged KV memory budget

The configuration is d_model = 128, L = 4, H = 4, head dim 32, with fp16 K and V.

- Per token: `2 × 4 × 128 × 2 B = 2,048 B`.
- With a shared root prefix of P tokens and a tree of 256 nodes × Δ tokens per node, memory ≈ `2 KB × (P + 256·Δ)` per instance.
- For P = 512 and Δ = 32 this is ≈ 17.4 MB per instance [DERIVED-4c], which is negligible on 22.03 GB even at batch 256.

The binding constraint is wall-clock time, not VRAM. Page size is 16 tokens, with copy-on-write at fork points.

#### Pass/fail bars

| Bar | Criterion | [PRIOR-4c] |
| :-- | :-- | :-- |
| **D3.a (spectral)** | `max_{seed, env, b ≤ 256} ‖T_b‖_∞ < 1.0` before rescaling is logged; ≤ 0.995 after. **Also** the empirical composite gain `‖𝒯_256 x‖_∞ / ‖x‖_∞ < 1.0` on 1,000 random probes | PASS (p ≈ 0.90) |
| **D3.b (KV equivalence)** | R4 cached vs uncached root/leaf max diff ≤ 10⁻⁵ at B = 256 (fp32 reference) | PASS (p ≈ 0.85) |
| **D3.c (stepwise anytime)** | Max stepwise KL rise ≤ 0.015, with ≥ 7/9 non-increasing steps | Uncertain (p ≈ 0.50) |
| **D3.d (cumulative anytime, new)** | `KL(B=256) ≤ min_{b} KL(b) + 0.010` (targets the R11 U-shape) | **FAIL** (p ≈ 0.65). Witness-chasing will continue to degrade marginals past B ≈ 8–16 unless the R12-D2 root improves. |
| **D3.e (scaling law)** | Generalist: κ > 0, R²_log ≥ 0.85 over all 10 budgets, and ΔMini-ECI(256−0) ≥ +9.0. **Saturation test:** compare log-linear vs `a − c·B^{−α}` by AIC; report the winner, with no bar on which one wins. | Δ ≥ +9.0: PASS (p ≈ 0.55). Predicted Δ(256−0) ∈ [+7.5, +12.0] (with Iso-FiLM trunk) |
| **D3.f (witness)** | Witness rate non-decreasing in B, and rate(256) ≥ 50% | PASS (p ≈ 0.55) |
| **D3.g (runtime)** | Per-seed wall-clock ≤ 40 min on L4 | PASS (p ≈ 0.70) |

### 3.5 Additional locks for `PREREG-R12-GPU-V2.md`

| ID | Hardening item |
| :-- | :-- |
| H1 | **Split disjointness:** SHA-256 of canonicalized instances, with `train ∩ cal ∩ test = ∅` asserted and logged. This is a new R2b sub-gate. |
| H2 | **AUROC ground truth:** `m_ql` and `m_qs` are scored against an independent label, namely whether the leaf's unmasked assignment is consistent with test `target_tokens`, computed post-search inside an exempt evaluator. The tautological median label is removed (fixes X9). |
| H3 | **Interactive-only bifactor:** D2 psychometrics are recomputed with strategic items taken from interactive Elo only (fixes X10). |
| H4 | **Cluster-robust psychometrics:** primary PC1, ω_h and ECV are computed on cohort-level aggregates at a fixed B (one row per model × search cohort, B = 32). A secondary mixed-effects fit with a cohort random effect is reported. Clamped communalities are disclosed (fixes X11). |
| H5 | **Unified Gram crowding:** computed identically on both `edge_keys_bank` and `root_keys_bank` for all cohorts with B ≥ 2. |
| H6 | **Aggregation transparency:** every aggregate KL states its weights, and the per-env table must reproduce the aggregate within 10⁻⁴ (fixes X7). |
| H7 | **Per-seed reporting:** every comparative sub-bar reports its per-seed pass count (e.g. "2/3 seeds"), with seed-mean t-intervals and explicit p-values (fixes the unverified "p < 0.01"). |
| H8 | **Seeds:** 5 seeds `{101, 102, 103, 104, 105}` if the spend projection allows, otherwise 3. The choice is fixed **before** launch. |
| H9 | **Spend cap: `$5.00`.** Estimated at ≈ $0.29/seed for R11 load × about 4× training × about 1.6× (B=256 and the extra arms) ≈ $1.85/seed, so 5 seeds ≈ $9 would exceed the cap. **Therefore 3 seeds ≈ $5.5 at the upper estimate.** The pre-registered resolution is 3 seeds with a hard stop at $5.00. If the projection exceeds the cap after seed 1, the exploratory arms (`Gen-Iso-LoRA2`, `Gen-Iso-FLOP`) are dropped first. |
| H10 | **Transcription audit:** an automated script regenerates every number in `RESULTS-R12` from the JSON. Discrepancies like X1, X2 and X4 become CI failures. |

---

## 4. 3-Round Trajectory Rubric Table (`D1`–`D6`, 1.0–10.0)

**Dimension key:**

- D1 — Universal KV-Cache Tree Machine
- D2 — Active Fractal Prospectus Plans
- D3 — Bidirectional Latent Lemma Space & Queries
- D4 — Present-Time Episodic Memory & LIR Firewall
- D5 — Anytime Monotonicity & Self-Consistency
- D6 — Multi-Game Elo League & Mini-ECI Bifactor IRT

| Dimension | R1 Proj | R2 Math/Unit | R2 Blind-Emp | R2 Proj-R11 | **R3 Realized-R11** | **R3 Proj-R12** |
| :-- | :-: | :-: | :-: | :-: | :-: | :-: |
| **D1** KV-Tree Machine | 7.50 | 7.50 | 2.50 | 6.00 | **7.00** | **7.80** |
| **D2** Fractal Plans | 6.50 | 7.00 | 2.50 | 5.50 | **7.00** | **7.50** |
| **D3** Lemmas & Queries | 7.50 | 7.50 | 1.50 | 5.00 | **5.50** | **7.00** |
| **D4** Memory & LIR | 6.50 | 6.50 | 1.50 | 5.00 | **7.00** | **7.60** |
| **D5** Anytime Monotonicity | 8.00 | 7.50 | 2.00 | 5.50 | **6.00** | **7.20** |
| **D6** Elo & Mini-ECI IRT | 7.00 | 8.00 | 1.00 | 6.00 | **5.50** | **7.00** |
| **Unweighted Mean** | **7.17** | **7.33** | **1.83** | **5.50** | **6.33** | **7.35** |

### Score justifications (R3 Realized-R11)

- **D1 = 7.00.** Real, leak-free tree search with exact KV equivalence (`0.000000000`) and exact B=0 root identity. Positive scaling in the GLC family against universal flat degradation. Deductions:
  - The generalist fails R²_red ≥ 0.92 (0.9155).
  - The "κ ∈ [−4.39, −0.65]" claim is false (X5).
  - Under-trained roots (−1.18% vs θ_init).
- **D2 = 7.00.** The plan head causally steers search (97.92% flip) and cuts ShipHunt-8x8 KL by 95.4%. This is the strongest mechanistic result of the round, and it is not built in by construction. Deductions: flip-rate granularity, and only one environment carries the plan claim.
- **D3 = 5.50.** The even-parity bilinear channel is safe (A4 ≈ A1 on Var KL; A2 cancellation is confirmed). However, A4 does not beat no-auxiliary on Var KL, loses to A1 on `query_kl` at every B > 0, has a ~1-instance accuracy advantage, and fails its absolute bar.
- **D4 = 7.00.** The exact firewall zeros and the 85.42% action flip are strong. Deductions:
  - Seed 101 loses both comparative sub-bars.
  - The +526.2 arithmetic error (X4).
  - The unexamined belief-KL cost of the firewall (Fused has the best aggregate KL).
- **D5 = 6.00.** The Gershgorin bound and the stepwise KL bar hold. But aggregate KL and Brier are U-shaped (minimum at B=8), Mini-ECI and Elo dip at B=64, and the stepwise bar is structurally blind to cumulative drift.
- **D6 = 5.50.** All pre-registered D2 and D3 bars pass. Deductions:
  - Mini-ECI↔Elo alignment is partly circular (X10).
  - N=108 is not independent and communalities are clamped (X11).
  - `AUROC(m_ql)` is void (X9).
  - The mean-KL aggregation is irreproducible (X7).
  - The Specialist leads Mini-ECI at every budget.

### Trajectory reading

| Transition | Change in mean | Meaning |
| :-- | :-: | :-- |
| R2 Blind-Emp → R3 Realized | **+4.50** | The largest single-round evidentiary gain in the program. It is entirely attributable to removing oracle leakage and running real SGD. |
| R2 Proj-R11 → R3 Realized | **+0.83** | My Round-2 projection was modestly pessimistic. |
| R3 Realized → Member 14 | **−1.90** | Reflects ledger items X4–X11, which the audit did not surface. |

The **R3 Proj-R12 mean of 7.35** is conditional on four things:

1. H1–H10 adopted verbatim.
2. The interactive-only re-estimation of D2 and D3 still clearing its bars.
3. R12-D1.b passing.
4. R12-D2.b passing.

My own priors predict that R12-D1.a (absolute `query_kl ≤ 0.15`) and R12-D3.d (cumulative anytime KL) will **fail**. The projected scores already price those failures in. If they pass, the projection is conservative. If the interactive-only Elo erases the D3 alignment (r < 0.85), D6 should be re-scored to ≤ 5.0. That is the most important single risk to this projection, and I record it here so that Round 4 can hold me to it.
