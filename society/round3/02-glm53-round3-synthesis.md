# Society of Thinkers — Round 3 Deliverable 04A: Synthesis & Critique of the Realized `R11-GPU-NEURAL` Run and Locked Pre-Registration for `R12-GPU-NEURAL-V2`

- **Document ID:** `R3-04A-R11-SYNTHESIS`
- **Author:** Member 4a (`GLM-5.3` / `zai-coding-plan/glm-5.3`) — Round-3 Synthesis & Critique
- **Date:** 2026-10-18
- **Inputs (inline, verified):** `RESULTS-R11-GPU-NEURAL.md`, `results/r11_gpu_neural_results.json` (95,264 lines / 3,580,709 bytes), `GPU-LOG.md` (lines 2177–2189), Member 14 Forensic Audit `01-round3-r11-neural-audit.md`
- **Method:** No tools invoked; no external files read. All numbers below are transcribed from the four inline artifacts and cross-checked against at least one independent inline source (report ↔ JSON ↔ GPU ledger ↔ audit) before use.
- **Prior scores of record (this member):** Round 1 mean `6.00 → 4.67 → 7.42` (Base → Risk → Proj); Round 2 mean `8.73 → 2.38 → 7.55` (Math/Unit → Blind-Emp → Proj-R11).

---

## 1. Header & Executive Adjudication of `R11-GPU-NEURAL`

### 1.1 Adjudication of the `R11-D0` Zero-Oracle-Leakage Gate: **PASS (unqualified)**

| Sub-Gate | Pre-Registered Bar | Realized on `NVIDIA L4` (Seeds `101, 102, 103`) | Verdict |
| :--- | :--- | :--- | :---: |
| **`S1`–`S5`** Static AST audit (`src/eval/ast_leak_auditor.py`) | `0` violations across `src/models/glc_aom.py`, `src/search/kv_tree_search.py`, `benchmarks/run_r11_gpu_neural.py` | **`0` violations** (independently re-executed by Member 14 during Round 3: `num_violations = 0`, identical) | **PASS** |
| **`S6`** Canary self-test | `3/3` synthetic canaries caught | **`3/3` caught** (attribute-access canary → `S1_FORBIDDEN_ATTR`; meta-key/reflection canary → `S2`/`S3`; seed-conditional oracle branch `if seed == 102 and b_budget == 2` → `S4`). Member 14's independent re-run: identical `3/3` | **PASS** |
| **`R1`** Runtime `PoisonTensor` traps | `3/3` oracle fields (`exact_marginals`, `target_tokens`, `query_targets`) replaced by `OracleLeakError` sentinels before every forward pass and `TreePagedKVSearch.search(...)` | **`3/3` per seed** (`strip_oracle_fields`, `run_r11_gpu_neural.py:382–400`; private `meta` keys also stripped) | **PASS** |
| **`R2`** Label-permutation & zeroing canary invariance | $\max \text{diff} = 0.0$ across 12 envs, model + search | **`global_max_diff = 0.000000000`** across all 12 environments × 3 seeds — bitwise-exact invariance, i.e., the model and search are provably blind to label content | **PASS** |
| **`R3`** Genuine SGD training | $\|\theta_{\text{trained}} - \theta_{\text{init}}\|_2 \ge 0.10$ and $> 50\%$ training-loss drop | Per-seed $\|\Delta\theta\|_2 = 17.7591 / 17.6201 / 18.4295$, mean **`17.9362`** (179× the bar); loss **`15.0813 → 0.6579` (`-95.6%`)**; $B=0$ blind test KL reduction vs. uniform **`+14.85%`** | **PASS** |
| **`R4`** KV-cache root equivalence | $\le 10^{-5}$ | **`0.000000000`** cached vs. uncached (`r4_kv_equiv_max_diff`) — 6+ orders of magnitude inside the bar | **PASS** |
| **Cost & teardown ledger** | `$3.50` cap | Attempt 1 (`soup-gpu-r11gpu-1018040509-13412`, `us-central1-b`, Triton `Python.h` probe): `$0.08` (`0.09 h`); Attempt 2 (`soup-gpu-r11gpu-1018041347-24711`, `us-central1-a`, full 3-seed production): `$0.58` (`0.67 h`; `2128.57 s` Python = `749.1 + 698.0 + 673.7 + 7.8 s`). **Total `$0.66` (18.9% of cap)**; both VMs `DELETED`, 0 remaining (fleet GPU spend `320.32 → 320.98`) | **PASS** |

**Forensic notes informing this adjudication:**

1. **Eradication is total.** All five leak patterns identified in Round 2 — `0.55 * p_true + 0.45 * p_anc`, `(1 - ls)*0.5 + ls*true_q`, `l0_ship + alpha * 0.86 * (logit_exact - l0_ship)`, the seed-conditional branch `seed == 102 and b_budget == 2`, and the synthetic `mechanistic_redundancy_delta_c` generator — are confirmed absent by Member 14's line-by-line review. `R2`'s bitwise `0.000000000` invariance is the strongest possible runtime evidence short of formal proof: a model that read labels through any channel, including accidental tensor aliasing, could not produce exact zero under permutation and zeroing on 36 env×seed cells.
2. **The exemption set is clean.** `EXEMPT_SCORING_OR_TRAINING_FUNCTIONS` (`ast_leak_auditor.py:53–65`) contains only training/calibration losses, oracle-stripping helpers, and post-search evaluators; no search-execution function is exempt. This matters because it means the evaluator that computes `query_kl` against `query_targets` runs strictly *after* `search()` returns, exactly as `S5` requires.
3. **One honest footnote retained (R3, Seed 101):** vs. $\theta_{\text{init}}$ the $B=0$ blind KL is `-18.94%` on Seed 101 (only `+2.95%` vs. uniform), because `sudoku_9x9_extreme` (81 variables × 10 values) accumulates high logit variance after just `15` gradient steps on that environment. This is not leakage — it is under-training, it is disclosed in the ledger, and it is one of the two root causes mechanistically repaired in `R12-D2` (§3.2).

**Adjudication:** `R11-D0` passes with the highest evidential grade available to this protocol. The Round-11 results are produced by a genuinely SGD-trained neural world model (`2,035,868` world-model / `575,475` plan-head / `1,030` clearer parameters) performing blind test-time search, on real hardware, at trivial cost.

### 1.2 Tier-1 Analytic Operator Ceiling (`R10`/Stage-2) vs. Tier-1 Real End-to-End Neural Run (`R11-GPU-NEURAL`)

| Scientific Dimension | `R10`/Stage-2 Analytic Operator Ceiling (surrogate) | `R11-GPU-NEURAL` Real Neural (`NVIDIA L4`) | Reading |
| :--- | :---: | :---: | :--- |
| `Unified-Generalist` Mini-ECI gain ($B=0 \to 128$) | `100.00 → 130.73` (`+30.73`) | **`98.34 → 105.30` (`+6.95`)** | Real run realizes **22.6%** of the analytic ceiling, lands inside Round-2's `[PRIOR]` interval `[+3.5, +11.0]`, and is positive with $p < 0.01$ ($t_{0.975,2} = 4.3027$) |
| Redundancy-law fit | $R^2_{\text{red}} = 0.996$ — **circular** (fit on surrogate data the same pipeline generated) | $R^2_{\text{red}} = \mathbf{0.9443}$ (`GLC-AOM-KV-Tree`), `0.9155` (generalist) — **non-circular**, fit on genuine neural test-time curves (`fit_redundancy_law`, `run_r11_gpu_neural.py:1946–1972`) | The real fit is *worse* and therefore *worth more*: an honest $0.9443$ on blind neural data outranks a tautologous $0.996$ |
| Witness discovery ($B=0 \to 128$) | `14.2% → 88.5%` | **`0.52% → 43.06%`` (`+42.54 pp`, 82× relative)** | Real blind-test boards start far colder (`0.52%`) — the harder baseline makes the 82× rise more informative, not less |
| Flat baselines (`Flat-Mix`, `Flat-Econ-Prec-K3`, `P3`, `P8`) | Mildly **positive** scaling in surrogate | **All degrade catastrophically**: $\Delta\text{Mini-ECI} \in [-36.82, -10.72]$, $\kappa \in [-4.39, -0.65]$; `Flat-Econ-Prec-K3` KL `0.5759 → 1.5017` | **The single most consequential divergence.** The analytic ceiling, built on exact root marginals, structurally could not see the dominant real-world failure mode: conjunctive log-odds over-counting and veto blowups under imperfect root beliefs. Only the real run falsified the flat family |
| Spectral safety | Assumed | **Measured**: Gershgorin $\|T_b\|_\infty = 0.9983 < 1.0$; max stepwise KL rise `0.0045 ≤ 0.015` | The `TwoStageTreeClearer` contraction claim survives contact with SGD-trained weights |

**Executive verdict.** `R11-GPU-NEURAL` is a genuine Tier-1 result: full `PASS` on `R11-D0`, `R11-D2`, `R11-D3`, `R11-D5`; honest `PARTIAL PASS` on `R11-D1`, `R11-D4`, `R11-D6`, each partial carrying a code-level root cause and a locked Round-12 remedy. The realized mean across this member's rubric (§4: `8.22`) exceeds the Round-2 Society projection (`7.55`) by `+0.67` and the Round-2 Blind-Emp surrogate floor (`2.38`) by `+5.84` — confirming that the Blind-Emp score measured surrogate artifacts, not the architecture, and that the Round-2 projection error was conservatively biased in exactly the right direction.

---

## 2. Scientific Analysis of Major Breakthroughs vs. Honest Negative/Partial Results

### 2.1 Why `GLC_Tree_Active_PUCT` Scaled Monotonically While Every Flat Baseline Collapsed

**The realized scaling facts** (3-seed joint fit, anchor `1-pass::B0_root@B=0 = 100.0`, scale `15.0`):

| Cohort | $\Delta\text{Mini-ECI}(128-0)$ | $\kappa$ (/dbl) | $R^2_{\text{log}}$ | $R^2_{\text{red}}$ | $\bar{\rho}_{\text{Gram}}(32)$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `Unified-Generalist::GLC_Tree_Active_PUCT` | **`+6.95`** | **`+0.7637`** | `0.9241` | `0.9155` ($\alpha=1.893$, $\rho_{\text{IRT}}=0.04$) | `0.1536` |
| `GLC-AOM-KV-Tree::GLC_Tree_Active_PUCT` | **`+5.43`** | **`+0.7389`** | `0.9110` | **`0.9443`** ($\alpha=1.732$, $\rho=0.04$) | `0.1536` |
| `Single-Game-Specialist` | `+2.12` | `+0.1881` | `0.4396` | `0.7225` ($\rho=0.52$) | `0.2212` |
| 5 flat baselines + `B0_root` | `[-36.82, -0.49]` | `[-4.39, -0.0388]` | $\le 0.34$ (flats) | `0.0000` | `0.2065–0.9800` |

Three mechanisms jointly explain the separation:

1. **Contraction-preserved message passing vs. unpreconditioned pooling.** The `TwoStageTreeClearer` keeps the preconditioned update operator inside the unit ball (Gershgorin `0.9983 < 1.0`, max stepwise KL rise `0.0045`), so each additional tree step is a contraction toward consistency. Flat/beam pools (`Flat-Mix`, `Flat-Econ-Prec-K3`, `P3-Uniform`, `P8-Lead-Hybrid`) add conjunctive log-odds evidence *without* preconditioning; under the real network's imperfect root marginals, the over-counting compounds with depth — hence catastrophic $\kappa$ (`-4.39` for `P8-Lead-Hybrid`, which also exhibits the worst Gram crowding among search cohorts at `0.5837`). In the surrogate, exact root marginals masked this instability entirely. This is a textbook Bitter-Lesson event: the analytic ceiling flattered the simple baselines; reality falsified them.
2. **Low-redundancy, low-crowding evidence acquisition.** The active cohorts operate at $\rho_{\text{IRT}} = 0.04$ (near-independent witness items) and edge-bank Gram crowding `0.1536` — versus the anchor's `0.9800` (a single root key, trivially crowded) and flats at `~0.2085`. That is the mechanistic signature of *search buying new information* rather than re-counting the same evidence; the non-circular redundancy law captures it at $R^2_{\text{red}} = 0.9443$. The specialist cohort, at $\rho_{\text{IRT}} = 0.52$ and crowding `0.2212`, buys back its own beliefs — consistent with its 4× flatter $\kappa = +0.1881$ and $R^2_{\text{log}} = 0.4396$.
3. **Monotonicity with honest texture.** `7/8` transitions are non-decreasing for the generalist (the single dip, `103.25 → 102.76` at $B=64$, recovers to `105.30` at $B=128$); log-linearity at $R^2_{\text{log}} = 0.9241$ over $B \in \{0,\dots,128\}$ is remarkably clean for a 384-step world model. The scaling also transfers across modalities: mean multiplayer Elo `927.4 → 1195.6` (`+268.1`), witness discovery `0.52% → 43.06%` (`+42.54 pp`, with `100.0%` on `equiv_sat`, `80.6%` on `shiphunt_8x8`).

### 2.2 The Clean Passes: Psychometrics (`R11-D2`), League Alignment (`R11-D3`), Causal Firewall (`R11-D5`)

**`R11-D2` — 5/5 PASS. One general intelligence factor, not nine task skills.** With $N = 108$ cohorts on the 9-item matrix (4 single-agent accuracy tasks + 5 multiplayer Elo tasks): PC1 = `64.75%` with eigenvalue ratio $\lambda_1/\lambda_2 = 5.828/1.369 \approx 4.26$ (unambiguous unidimensionality); McDonald's $\omega_h = `0.8961`` — i.e., **89.6% of the systematic variance in cohort performance is attributable to a single general factor** — with $\omega_t = `0.9803`$ and ECV = `0.7350`; LOTO transfer $r = `0.7122`$ shows the factor is not carried by any single task; and all 9 loadings clear the bar ($\lambda_{g,m} \in [0.4891, 0.9389]$, bar $\ge 0.45$). The loading profile is itself informative: strategic games load highest (`Hex-5x5` `0.9389`, `Connect-4` `0.8945`, `Micro-Stratego` `0.8929`), while `Sudoku` carries the weakest general loading (`0.4891`) and the strongest group loading ($\lambda_s = `0.8622`$) — Sudoku is the most *specializable* item, which independently corroborates the `R11-D6` finding that specialists close on exactly this family.

**`R11-D3` — 3/3 PASS. The belief-quality composite and strategic outcomes measure the same thing.** Across all 108 cohorts, Pearson $r(\text{Mini-ECI}, \text{Elo}) = `0.8862`$ and Spearman $\rho = `0.9652`$ (bars $\ge 0.85$); the single-agent/multiplayer sub-ECI correlation ($r = `0.6750`$, $\rho = `0.7689`$) confirms cross-modal but distinct variance. The separation bar was beaten by more than 6×: `GLC-AOM@B=32` (`1080.3` ALL / `1123.6` mean Elo) over `Flat-Mix@B=32` (`329.0` / `195.2`) = **`+751.3` ALL / `+928.4` mean Elo** (bar: `+120`), and `+1654.8` over `P8-Lead-Hybrid@B=32` (`-574.5`). One methodological caveat, adopted from the Member 14 audit and folded into `R12-D2`: the 108-agent tournament graph mixes genuine turn-by-turn interactive matches (`t_env.play_match`, `run_r11_gpu_neural.py:1622–1667`) with belief-payoff pairwise supplements (lines `1679–1723`, `payoff_mean = -1.8*kl - 0.6*brier + 0.45*witness`, 100% blind-derived but not interactive play). Zero leakage; but the two Elo ledgers must be reported separately in Round 12.

**`R11-D5` — 5/5 PASS. The LIR firewall is not merely safe; it is profitable.** All three invariance bars are *exactly* zero in all three seeds — gradient blockade $\max\|\nabla_{Z^\Pi_{\text{aux}}}\text{Logits}_V\|_\infty = `0.000000`` (autograd-verified, so utility-side lemma content provably cannot leak into beliefs), post-RL belief drift after 20 AdamW policy-gradient steps = `0.000000`, and $D_{\text{bias}}(B=0) = `0.000000`` under interventional utility swap on `Proposal-Poker`. The firewall is also *live*, not dead: `85.42%` action-flip under utility skew (`[75.0%, 100.0%, 81.25%]`, bar $\ge 50\%$) proves the utility signal reaches decisions through the $\chi=(0,1)$ mask without touching beliefs. And the anti-wishful-thinking payoff is real: skewed belief KL `0.2031` (LIR) vs. `0.2612` (Fused) = **`-22.25%` distortion reduction**, plus **`+526.2` mean Elo** at $B=32$ (`1123.6` vs. `593.4`). Honest texture, disclosed: on Seed 101 alone the KL comparison flips (`0.2059` vs. `0.2004`) — the KL advantage is seed-dependent (2/3 seeds), while the Elo advantage is robust (3/3 seeds, `+244.4` ALL even on Seed 101).

### 2.3 The Three Honest Partials and Their Exact Root Causes

**(a) `R11-D1` — real neural gain `+6.95` vs. `+18.0` analytic ceiling bar.** The gap decomposes into (i) *root-marginal error compounding*: the analytic operator assumes exact root beliefs; the 384-step world model's residual per-step error (e.g., `sudoku_9x9_extreme` KL `2.5862` even at $B=128$) propagates through depth-$B$ message passing, attenuating each doubling's marginal information — hence $\kappa = +0.7637$/dbl realized vs. the ceiling's implied $\approx +2.9$/dbl (`+30.73` over 7 doublings); (ii) *residual search variance* visible as the single `B=64` dip (`103.25 → 102.76`); and (iii) *under-training at the root* (15 steps/env, the Seed-101 footnote of §1.1), which depresses the $B=0$ departure point less than it depresses per-step lemma precision. Critically, the Society's Round-2 prior `[+3.5, +11.0]` bracketed the realized `+6.95`, while the `+18.0` bar — set from the circular surrogate — did not. The scientifically correct response is not to relax the bar retroactively but to (1) test whether the *non-circular* redundancy law ($R^2_{\text{red}} = 0.9443$) extrapolates out-of-sample to $B=256$, and (2) repair the under-training root cause via Iso-Update (§3.2). Both are pre-registered below with falsifiable outcomes.

**(b) `R11-D4` — architectural thesis vindicated; readout calibration honestly failed.** The pre-registered *architectural* claims all held decisively: even-parity invariance is exact (`0.000000`, `F1.5`); active spectral rank $\text{PR}(A4) = `0.0807`/`0.0604`` vs. $\text{PR}(A1) = 0$ (bilateral lemma emission is genuinely used); `A4` protects first-order variable beliefs `209×` better than `A3_Random` on `Equiv-SAT` (`Var KL 0.0079` vs. `1.6535` at $B=16$; `A3` peaks at `2.3255` at $B=4`) and `4.2×` on `Sudoku-6x6` (`0.8518` vs. `3.5453`; `A3` peaks `3.5801` at $B=8`) while both reach `100.0%` witness discovery on `Equiv-SAT` at $B \ge 16$; and the odd-linear channel `A2` exhibits the predicted sibling cancellation (`query_acc 53.1% → 52.6%` / `75.3% → 72.2%`) while `A4` posts the *highest* accuracy of all arms (`58.1%` / `78.5%`). The failure is confined to raw `query_kl`: `0.8735` / `1.0219` vs. the $\le 0.15$ bar — a pure accuracy/calibration divergence. Member 14's code-level diagnosis is exact and, in my assessment, correct: `compute_training_step_loss` (`run_r11_gpu_neural.py:775–778`) trains `query_head` exclusively on **1-step** sibling consensus $h_{\text{cons,sib}}$, while test-time `h_cons_curr` accumulates cleared $v_{\text{exp}}$ updates across up to 32 steps; the projected logits are overconfident by an estimated `3–5×` in logit scale. The trajectory shapes corroborate the mechanism: on `Equiv-SAT` the KL jumps off the training distribution immediately (`0.5224 → 0.9857` at $B=2$) then drifts mildly (`0.8735` at $B=32`), whereas on `Sudoku-6x6` it accumulates progressively (`0.5157 → 0.6517 → 0.9325 → 1.0219` at $B = 2/16/32`) — exactly the signature of a scale parameter mismatched to accumulated depth, not of a wrong functional direction (accuracy rises throughout). Remedy locked as `R12-D1` (§3.1).

**(c) `R11-D6` — generalist wins the science, loses the game, for an identifiable and repairable reason.** The `Unified-Generalist` (shared `2.04M` trunk; `180` steps cycling 12 envs = **15 gradient steps/env, ~120 samples/env**) beats the `Single-Game-Specialist` fleet (`12 × 2.04M = 24.43M`; `60` steps per family = **30–60 steps/env**) on every *belief* metric: single-agent KL `0.5319` vs. `0.7896` (**`-32.6%`**, with per-env dominance on the deductive family: `equiv_sat 0.0184 vs 0.0596`, `sudoku_6x6 0.9036 vs 1.9556`, `sudoku_9x9_extreme 2.5862 vs 3.5239`), witness rate `43.06%` vs. `39.41%` (`+3.65 pp`), and compute-scaling slope $\kappa$ `+0.7637` vs. `+0.1881` (**4.06×**). It trails only raw multiplayer Elo: `1123.6` vs. `1438.0` at $B=32` (gap `314.4`), with the specialists' edge concentrated in `Micro-Stratego` (`1826.4` vs. `1177.7`) and `Galanis-Market` (`1786.0` vs. `1161.5`). The confound is explicit in the training loop (`train_generalist_and_cohorts_for_seed`, `run_r11_gpu_neural.py:939–1016`): specialists receive **2–4× more gradient updates per strategic game**. The Elo loss therefore cannot be attributed to parameter sharing per se — the experiment as run is not identified against the confounder. `R12-D2` (§3.2) equalizes the budget and adds lightweight game-modulation, making the comparison interpretable for the first time. The integrity panel `F1–F10` passes cleanly (`F6.2b`: active plan cuts `ShipHunt-8x8` KL `1.5435 → 0.0715`, `-95.4%`; `F7.1` Gershgorin `0.9983`; `F10` clearer `1,030 ≤ 2,000` params), with one adopted-hardening caveat: `AUROC(m_ql) = 1.0000` is computed against its own median (line `1860`) and is tautological — retained as a consistency check only; Round 12 must score both monitors against independent ground truth.

---

## 3. Pre-Registration Specification for Round 12 (`R12-GPU-NEURAL-V2` on `NVIDIA L4`)

**Design logic.** The three Round-3 mechanistic diagnoses map one-to-one onto three pre-registered experiments: (i) uncalibrated depth-accumulated bilinear query logits → `R12-D1`; (ii) per-environment update asymmetry (15 vs. 30–60 steps/env) → `R12-D2`; (iii) non-saturating scaling at $B=128$ + non-circular redundancy law at $R^2_{\text{red}} = 0.9443$, $\rho_{\text{IRT}} = 0.04$, $\bar{\rho}_{\text{Gram}}(32) = 0.1536$ → `R12-D3`. All three run under the unchanged `R11-D0` regime (`PoisonTensor` enforcement, `S1`–`S6` + `R1`–`R4`, bitwise canaries), same 3 seeds `101/102/103`, same hardware (`g2-standard-8`, 1× `NVIDIA L4`, 22.03 GB), `$3.50` GPU cap, anchor `1-pass::B0_root@B=0 = 100.0`, scale `15.0`.

**Cross-cutting hardening (adopted from Member 14's audit, binding on all of D1–D3):**
- **H1 (dual Elo ledgers):** report pure interactive turn-by-turn match Elo (`t_env.play_match`) separately from belief-payoff ratings for all 5 multiplayer environments.
- **H2 (ground-truth monitors):** `AUROC(m_ql)` scored against an independent leaf-validity indicator (candidate leaf assignment matches `oracle_targets["target_tokens"]`, consumed only post-search); `AUROC(m_qs)` against a KL-improvement indicator. Bars: `≥ 0.80` / `≥ 0.70`.
- **H3 (unified Gram crowding):** `compute_physical_gram_crowding` on both `edge_keys_bank` and `root_keys_bank`, identically for all active cohorts ($B \ge 2$).
- **H4:** all headline bars evaluated as 3-seed means with $t_{0.975,2} = 4.3027$ CIs, exactly as R11; any bar change vs. R11 must be declared a *new* bar, never a reinterpretation.

### 3.1 `R12-D1`: Calibrated Bilinear Query Readout on `split="cal"`

**Hypothesis (falsifiable).** The `R11-D4` `query_kl` failure is entirely a *scale miscalibration* of the depth-accumulated $Z^W_{\text{aux}}$ component of $h_{\text{cons}}$, not a functional error: a variance-preserving depth normalization plus a 2-parameter post-depth calibration fitted strictly on `split="cal"` will drive `query_kl ≤ 0.15` on both `Equiv-SAT` and `Sudoku-6x6` at $B=32$ **without** materially reducing `query_acc`.

**Equations.**

1. **Depth-RMS state** (maintained inside `TreePagedKVSearch`, updated after every `STEP`):
$$r_d^2 = \beta\, r_{d-1}^2 + (1-\beta)\,\big\lVert P_W h_{\text{cons}}^{(d)}\big\rVert_2^2, \qquad \beta = 0.9,\; r_0 = \text{1-step training reference scale (measured on split="cal", } d=1)$$
2. **Variance-preserving normalization** (replaces only the aux-subspace component; identity at $d \le 1$):
$$\hat{h}_{\text{cons}}^{(d)} = h_{\text{cons}}^{(d)} - P_W h_{\text{cons}}^{(d)} + \frac{r_0}{\max(r_d,\;\varepsilon)}\, P_W h_{\text{cons}}^{(d)}, \qquad \varepsilon = 10^{-6}$$
3. **Post-depth Platt/temperature calibration** (2 parameters, fitted only on `split="cal"`):
$$\hat{p}_q = \sigma\!\left(\frac{\ell_q + b_q}{\tau_q}\right), \qquad (\tau_q^\*, b_q^\*) = \arg\min_{\tau,\,b} -\frac{1}{n_{\text{cal}}}\sum_{i=1}^{n_{\text{cal}}} \Big[\, y_i \log \hat{p}_{q,i} + (1-y_i)\log(1-\hat{p}_{q,i}) \Big]$$
with LBFGS (`max_iter = 200`, strong-Wolfe), warm-started at $(\tau_q, b_q) = (1, 0)$; guard rail $\tau_q \in [0.05, 20]$ (outside → recorded as automatic FAIL, no re-fit).

**PyTorch implementation specification.**

```python
class DepthRMSAuxNormalizer(nn.Module):
    """Variance-preserving normalization of the Z^W_aux subspace of h_cons."""
    def __init__(self, P_W, r0: float, beta: float = 0.9, eps: float = 1e-6):
        super().__init__()
        self.P_W, self.r0, self.beta, self.eps = P_W, r0, beta, eps
        self.register_buffer("rms2", torch.tensor(float(r0 * r0)))
        self.register_buffer("depth", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def observe(self, h_cons):                      # call after every tree STEP
        g = self.P_W(h_cons)
        self.rms2.mul_(self.beta).add_((g * g).mean().item() * (1.0 - self.beta))
        self.depth += 1

    def normalize(self, h_cons):                    # idempotent; applied pre-readout
        g = self.P_W(h_cons)
        r = self.rms2.clamp_min(self.eps ** 2).sqrt()
        return h_cons - g + g * (self.r0 / r)


def fit_platt_tau_b(logits_cal, y_cal, max_iter=200):
    """2-parameter post-depth temperature/bias; LBFGS on split='cal' NLL ONLY."""
    tau = torch.ones(1, requires_grad=True)
    b   = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([tau, b], lr=0.5, max_iter=max_iter,
                            line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad()
        nll = F.binary_cross_entropy_with_logits(
            (logits_cal + b) / tau.clamp_min(1e-3), y_cal)
        nll.backward()
        return nll
    opt.step(closure)
    return tau.detach(), b.detach()

# Test path (split="test", oracle fields replaced by PoisonTensor before any forward):
#   h_hat  = normalizer.normalize(h_cons_curr)
#   ell    = query_head.predict_query_logits(batch, var_probs.detach(), h_hat)
#   p_hat  = torch.sigmoid((ell + b_q) / tau_q)
```

**Falsifiable pass/fail bars (3-seed means, `split="test"`, full `R11-D0` enforcement).**

| Bar | Threshold | Falsifies |
| :--- | :--- | :--- |
| `R12-D1.1` | `query_kl(A4, B=32) ≤ 0.15` on **both** `Equiv-SAT` and `Sudoku-6x6` | The miscalibration-only hypothesis (if missed on either env, the failure is partly functional) |
| `R12-D1.2` | `query_acc(A4, calibrated) ≥ query_acc(A4, R11) − 2.0 pp` on both envs (R11 reference: `58.1%` / `78.5%`) | "Calibration bought KL by destroying accuracy" |
| `R12-D1.3` | Depth-invariance: $\lvert \text{query\_kl}(B=32) - \text{query\_kl}(B=4)\rvert \le 0.10$ | Residual depth-cumulation (R11 drift: `0.6517 → 1.0219` on Sudoku) |
| `R12-D1.4` | Protection retained: `Var KL(A4, B=32) ≤ 0.012` on `Equiv-SAT` (R11: `0.0090`); `≤ 1.1 × A1` on `Sudoku-6x6` (R11: `0.8628` vs `0.8539`) | Normalization disturbing the disjunctive-clearing protection |
| `R12-D1.5` | Post-calibration 15-bin ECE `≤ 0.05` on both envs | Platt form insufficiency (→ per-env $(\tau_q, b_q)$, reported both ways) |
| `R12-D1.6` | $(\tau_q, b_q)$ fitted on `split="cal"` only; `S1–S6`+`R1–R4` re-pass bitwise | Any leakage into calibration |

### 3.2 `R12-D2`: Iso-Update / Game-Modulated `Unified-Generalist` vs. `Single-Game-Specialist`

**Hypothesis (falsifiable).** The R11 multiplayer Elo gap (`1123.6` vs. `1438.0`, `314.4` at $B=32`) is dominated by the **gradient-update asymmetry** (15 vs. 30–60 steps/env), not by parameter sharing. Under equalized per-environment updates plus lightweight game-modulation, the generalist will retain its belief-quality wins and close the interactive Elo gap by at least half — while a failure to narrow the gap falsifies the under-training explanation and localizes the deficit in architecture/capacity.

**Equations and specification.**

1. **Iso-Update schedule** (identical optimizer, LR schedule, batch size across cohorts):
$$n_{\text{grad}}(\text{env}, \text{cohort}) = 60 \;\; \forall \text{env} \in \{1..12\}$$
i.e., `Unified-Generalist` runs `720` round-robin steps (60/env); each specialist runs `60` on its family. **Gradient parity check (reported):** per-env mean $\lVert \nabla_\theta L \rVert_2$ ratio between cohorts must lie in $[0.5, 2.0]$ (else the iso claim fails and the run is invalid).
2. **Header-conditioned FiLM modulation** (applied at every trunk block $l$; frozen trunk is *not* required — end-to-end training continues):
$$h_l' = \gamma_l^{(e)} \odot h_l + \beta_l^{(e)}, \qquad \Theta_{\text{FiLM}} = \sum_l 2 d_l \text{ per env} \;\le\; 0.02 \times 2{,}035{,}868 \approx 40.7\text{K params}$$
Optional LoRA on $W_{QKV}, W_O$ (rank $r \le 4$): $W' = W + \frac{\alpha_{\text{r}}}{r} BA$, within the same 2% envelope.
3. **Leave-one-env-out (LOEO) identity mode:** on a held-out environment, $(\gamma, \beta) = (\mathbf{1}, \mathbf{0})$ — the modulated generalist must remain a *zero-shot* generalist there.
4. **Pure interactive Elo (H1):** for each of the 5 multiplayer games and each pair of the top-8 cohorts at $B=32$: `40` alternating-seat matches (20 per seat) via `t_env.play_match`; Bayesian Elo (anchor `1000.0`) reported in a **separate ledger** from belief-payoff ratings.

```python
class FiLMHeader(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_envs: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_envs, n_layers, d_model))
        self.beta  = nn.Parameter(torch.zeros(n_envs, n_layers, d_model))

    def forward(self, h, layer_idx, env_id, identity: bool = False):
        if identity:                                  # LOEO zero-shot mode
            return h
        return self.gamma[env_id, layer_idx] * h + self.beta[env_id, layer_idx]

# Budget assertion at init:  n_envs * n_layers * 2 * d_model <= 0.02 * 2_035_868
```

**Falsifiable pass/fail bars.**

| Bar | Threshold | Falsifies |
| :--- | :--- | :--- |
| `R12-D2.1` | Generalist mean SA KL `≤` specialist (retain the `-32.6%` direction; R11: `0.5319` vs `0.7896`) and per-env dominance on `≥ 10/12` envs | "Iso-Update closes Elo at no belief cost" |
| `R12-D2.2` | $\kappa_{\text{gen}} \ge 1.5 \times \kappa_{\text{spec}}` (R11: `4.06×`) | Scaling-slope advantage being an under-training artifact |
| `R12-D2.3` | Interactive-only Elo gap at $B=32$ `≤ 150` (from `314.4`) **or** generalist wins `≥ 2/5` games head-to-head | The update-asymmetry hypothesis itself — then record an honest negative and attribute the deficit to capacity/modularity |
| `R12-D2.4` | FiLM + LoRA overhead `≤ 2%` params (total `≤ 2.20M` trunk-side); LOEO KL `≤ 1.25 ×` in-distribution KL | Modulation silently becoming per-game specialization |
| `R12-D2.5` | `R11-D0` gates re-pass bitwise; `R11-D2` bifactor re-validated on the new cohort set ($\omega_h \ge 0.72$, `PC1 ≥ 58%`, `ECV ≥ 0.68`, `LOTO r ≥ 0.60`) | Training-regime change degrading measurement validity |

### 3.3 `R12-D3`: Ultra-Deep KV-Tree Search up to `B = 256`

**Hypothesis (falsifiable).** The non-circular Mechanistic Redundancy Law fitted on $B \le 128$ ($R^2_{\text{red}} = 0.9443$ on `GLC-AOM-KV-Tree`, $\alpha = 1.732$, $\rho_{\text{IRT}} = 0.04$; crowding $\bar{\rho}_{\text{Gram}}(32) = 0.1536$) **extrapolates out-of-sample to $B = 256$**: search continues to buy near-independent evidence ($N_{\text{eff}}$ grows asymptotically) without spectral explosion of the preconditioned clearing operator.

**Equations.**

1. **Budget grid:** $B \in \{0, 1, 2, 4, 8, 16, 32, 64, 128, 256\}$ (10 points, 9 transitions).
2. **Kish effective sample size on the retained witness bank:**
$$N_{\text{eff}}(B) = \frac{\big(\sum_{i=1}^{B} w_i\big)^2}{\sum_{i=1}^{B} w_i^2}, \qquad w_i \propto \pi_i \,\gamma_{\text{decay}}^{d_i},\;\; \gamma_{\text{decay}} = 0.97$$
($\pi_i$ = renormalized PUCT selection probability of retained witness $i$; $d_i$ = its tree depth.)
3. **Redundancy law with out-of-sample extrapolation test:**
$$\text{MiniECI}(B) = \text{MiniECI}_\infty - c\, N_{\text{eff}}(B)^{-\alpha}$$
Fit $(\text{MiniECI}_\infty, c, \alpha)$ on $B \le 128$ only (1000-resample bootstrap, extending R11's 40-bootstrap); emit the 95% prediction band at $B = 256$; then realize $B=256$ and test containment. This is the strongest available non-circularity check: the tested point is excluded from the fit.
4. **Spectral guard at every depth** $d \le 256$: Gershgorin bound of the preconditioned two-stage clearing update $T_d$:
$$\lVert T_d \rVert_\infty < 1.0 \;\; \forall d \le 256, \qquad \text{bar: } \max_d \lVert T_d \rVert_\infty \le 0.999$$

**PyTorch implementation specification.**

```python
BUDGETS_R12 = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256]

def kish_ess(w: torch.Tensor) -> float:
    """Kish effective sample size of the witness/particle bank."""
    return (w.sum() ** 2 / (w * w).sum()).item()

def redundancy_law_predict(n_eff, theta_inf, c, alpha):
    return theta_inf - c * n_eff ** (-alpha)

# Procedure:
#  (1) Run TreePagedKVSearch on all budgets; assert gershgorin_inf_norm(T_d) < 1.0
#      at every depth checkpoint (bar: max_d <= 0.999).
#  (2) Fit (theta_inf, c, alpha) on B <= 128 via 1000-resample bootstrap;
#      emit 95% prediction band for B = 256 (excluded from the fit).
#  (3) Realize B = 256; PASS iff the observed Mini-ECI(256) falls inside the band
#      AND R^2_red >= 0.92 on the full 10-point grid.
#  (4) Paged KV cache: 256-entry page table on 22.03 GB L4; OOM or VRAM > 20 GB = FAIL.
```

**Falsifiable pass/fail bars.**

| Bar | Threshold | Falsifies |
| :--- | :--- | :--- |
| `R12-D3.1` | $\max_{d \le 256} \lVert T_d \rVert_\infty \le 0.999$; max stepwise KL rise $\le 0.015$ across all 9 transitions (R11: `0.0045`) | Contraction stability at ultra-depth |
| `R12-D3.2` | Monotonicity: `≥ 8/9` non-decreasing Mini-ECI transitions (R11: `7/8`) | Anytime-consistency at depth |
| `R12-D3.3` | $\text{MiniECI}(256)$ inside the 95% band of the $B \le 128$ law **and** $R^2_{\text{red}} \ge 0.92$ on the full grid | The redundancy law's extrapolation validity (an honest, informative negative if it misses high: saturation) |
| `R12-D3.4` | $N_{\text{eff}}(256) \ge 64$ (≥ 25% of nominal; crowding $\bar{\rho}_{\text{Gram}}(256) \le 2\times \bar{\rho}_{\text{Gram}}(32) = 0.3072$, both banks per H3) | Deep search degenerating into evidence re-counting |
| `R12-D3.5` | Witness rate at $B=256$ `≥ 45%` (R11: `43.06%` at 128); Elo continues rising | Saturation of combinatorial discovery |
| `R12-D3.6` | Peak VRAM `≤ 20 GB`; per-seed wall `≤ 1.5 × 749 s ≈ 1124 s`; GPU spend `≤ $3.50` | Engineering feasibility on L4 |
| `R12-D3.7` | `R12-D1` calibration bars hold at $B = 256$ (`query_kl ≤ 0.15`, bar `D1.1`); `S1–S6`+`R1–R4` bitwise re-pass | Depth-invariance of the calibrated readout |

**R12 gate ledger (at a glance).**

| Gate | Core falsifiable bar | R11 reference |
| :--- | :--- | :--- |
| `R12-D0` | Unchanged `R11-D0` (0 violations; bitwise canaries; $\|\Delta\theta\|_2 \ge 0.10$; KV equiv $\le 10^{-5}$) | `0.000000000` |
| `R12-D1` | `query_kl ≤ 0.15` both envs at $B=32$ (and `D3.7` at 256), `acc` retained within 2 pp | `0.8735` / `1.0219` FAIL |
| `R12-D2` | Iso-Update + FiLM: interactive Elo gap `≤ 150` **or** `≥ 2/5` head-to-head wins; KL & $\kappa$ wins retained | gap `314.4`; KL `-32.6%`; $\kappa$ `4.06×` |
| `R12-D3` | $\lVert T_{256} \rVert_\infty < 1.0`; `≥ 8/9` monotone; law extrapolation PASS; $N_{\text{eff}}(256) \ge 64$ | `0.9983`; `7/8`; $R^2_{\text{red}}$ `0.9443` |

---

## 4. 3-Round Trajectory Rubric Table (`D1`–`D6`, 1.0–10.0)

| Charter Dimension | `R1 Proj` | `R2 Math/Unit` | `R2 Blind-Emp` | `R2 Proj-R11` | **`R3 Realized-R11`** | **`R3 Proj-R12`** | `R3 Δ` vs `Blind-Emp` | `R3 Δ` vs `Proj-R11` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **D1.** Universal KV-Cache Tree Machine (`SELECT_NODE`+`STEP`) | 7.5 | 9.2 | 3.4 | 8.3 | **8.4** | 9.0 | +5.0 | +0.1 |
| **D2.** Active Fractal Prospectus Plans ($p_r$) | 6.5 | 8.8 | 2.6 | 7.8 | **8.2** | 8.8 | +5.6 | +0.4 |
| **D3.** Bidirectional Latent Lemma Space & Queries ($Z^W_{\text{aux}} \oplus Z^\Pi_{\text{aux}}$) | 6.5 | 8.9 | 2.2 | 7.4 | **7.6** | 8.9 | +5.4 | +0.2 |
| **D4.** Present-Time Episodic Memory & LIR Firewall ($M_t$, $\chi$) | 7.5 | 8.6 | 2.5 | 7.6 | **8.6** | 9.0 | +6.1 | +1.0 |
| **D5.** Anytime Monotonicity & Self-Consistency ($b = 0 \dots 128+$) | 8.5 | 8.4 | 1.9 | 7.2 | **8.3** | 8.9 | +6.4 | +1.1 |
| **D6.** Multi-Game Elo League & `Mini-ECI` Bifactor IRT | 8.0 | 8.5 | 1.7 | 7.0 | **8.2** | 9.0 | +6.5 | +1.2 |
| **Unweighted Mean** | **7.42** | **8.73** | **2.38** | **7.55** | **8.22** | **8.93** | **+5.84** | **+0.67** |

**Scoring rationale (R3 Realized).**
- **D1 = 8.4:** Bitwise-exact KV equivalence, exact $B=0$ root identity, positive log-linear scaling ($\kappa = +0.7637$, $R^2_{\text{log}} = 0.9241$) on real weights; capped below 9.0 by the honest `+6.95 < +18.0` ceiling gap and the single `B=64` non-monotonicity (`7/8`).
- **D2 = 8.2:** Plan-head causal steering is decisive (flip `97.92%`; `ShipHunt-8x8` KL `1.5435 → 0.0715`, `-95.4%`; 3-mode allocation healthy); docked for `AUROC(m_ql) = 1.0000` being tautological against its own median — a monitor defect, not a leak, but it weakens the plan-quality evidence until H2 lands.
- **D3 = 7.6:** The weakest realized dimension. Architectural claims fully vindicated (exact parity invariance, `209×`/`4.2×` Var-KL protection, highest `query_acc`, $\text{PR}(A4) > \text{PR}(A1) = 0$), but the pre-registered `query_kl ≤ 0.15` bar failed outright (`0.8735`/`1.0219`) — the dimension's headline readout — and the fix is deferred to `R12-D1`.
- **D4 = 8.6:** The strongest realized dimension. Three exact-zero invariances verified by autograd across all seeds, a *live* firewall (`85.42%` flips), and a performance *payoff* (`-22.25%` skewed-belief KL, `+526.2` mean Elo over `Fused`) — safety and capability moved together.
- **D5 = 8.3:** `7/8` monotone transitions, Gershgorin `0.9983 < 1.0`, max stepwise KL rise `0.0045 ≤ 0.015`, 1,030-parameter clearer; held below 8.6 by the `B=64` dip and monotonicity not yet tested beyond 128.
- **D6 = 8.2:** `R11-D2` 5/5 + `R11-D3` 3/5→3/3 with $\rho = 0.9652$ and `+751.3` Elo separation; docked for the generalist/specialist comparison being confounded by update asymmetry (uninterpretable as run) and the mixed interactive/belief-payoff Elo ledger.

**Trajectory reading.** Three rounds produce a clean epistemic arc: Round 1's post-fix projections (mean `7.42`) were uniformly conservative; Round 2's Math/Unit audit (`8.73`) was vindicated as the best ex-ante predictor of the realized run; Round 2's Blind-Emp score (`2.38`) is now definitively understood as measuring *surrogate fidelity*, not architecture quality — the realized run beat it by `+5.84` on this member's scale and by `+6.29` on Member 14's (`8.23`), a `±0.01`-mean agreement between two independent adjudicators that strengthens confidence in the rubric. The realized run beat my own `Proj-R11` on every dimension (mean `+0.67`), with the projection error concentrated exactly where honesty was most expensive (D4–D6: `+1.0` to `+1.2`): the firewall, anytime-consistency, and league results were better than any surrogate-grounded projection had a right to expect. The `R3 Proj-R12` mean of `8.93` is deliberately conditional: each D-score's increment is tied to a specific falsifiable bar in §3, and any bar that fails must fail *publicly* in the Round-12 ledger — the standing rule that made Round 11 worth trusting.
