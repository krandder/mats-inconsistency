> **PARKED 2026-10-19 (Kelvin): not frozen, not run**

# Pre-Registration Master Index: Round 26 (`R26-A` & `R26-B`) — External Public `OpenSpiel` & `Ataraxos` (`arXiv:2511.07312`) Baselines

- **Round ID:** `R26` (Split into **`R26-A`** and **`R26-B`** per [`outbox/r26-freeze-review-2026-10-19.md`](file:///home/agent/workspaces/inconsistency-search/outbox/r26-freeze-review-2026-10-19.md))
- **Status:** `PARKED 2026-10-19 (Kelvin): not frozen, not run`
- **Date Revised:** `2026-10-19`
- **Protocol Governance:** [`docs/PROTOCOL.md`](file:///home/agent/workspaces/inconsistency-search/docs/PROTOCOL.md) (Sections 1–8) & [`outbox/r26-freeze-review-2026-10-19.md`](file:///home/agent/workspaces/inconsistency-search/outbox/r26-freeze-review-2026-10-19.md)
- **Proposer & Coder:** `gemini` (`agy`)
- **Freezer:** `opus-mats` (commits `PREREG + runner + PREREG.sha256` with tags `freeze-R26A` and `freeze-R26B`)
- **Authorized Runner:** `eurisko` or `calibrated-ai` (never `gemini`)
- **Authorized Auditor:** `board-opus` subagent of `opus-mats`

---

## 1. Split Structure (`R26-A` and `R26-B`)

Following `opus-mats`'s freeze review recommendation (`outbox/r26-freeze-review-2026-10-19.md`), `R26` is split into two independent, self-contained pre-registrations and runners so that `R26-A` (`OpenSpiel`, CPU/L4-light, `5` seeds) and `R26-B` (`Ataraxos` `10x10` Stratego, CUDA `pystratego`, `3` seeds) can be frozen, executed, scored, and audited independently:

1. **Round `R26-A` (`OpenSpiel` Public Baselines with Reproduced Learning Curves, `5` Seeds `101..105`):**
   - **Pre-Registration:** [`benchmarks/PREREG-R26A-OPENSPIEL-PUBLIC-BASELINES.md`](file:///home/agent/workspaces/inconsistency-search/benchmarks/PREREG-R26A-OPENSPIEL-PUBLIC-BASELINES.md)
   - **Runner Script:** [`benchmarks/run_r26a_openspiel_public_baselines.py`](file:///home/agent/workspaces/inconsistency-search/benchmarks/run_r26a_openspiel_public_baselines.py)
   - **Upstream Patch & Unit Test:** [`external/patches/openspiel_jax_jnp_clip_shim.patch`](file:///home/agent/workspaces/inconsistency-search/external/patches/openspiel_jax_jnp_clip_shim.patch) & [`tests/test_external_patches.py`](file:///home/agent/workspaces/inconsistency-search/tests/test_external_patches.py)
   - **Target Output JSON:** `results/r26a/r26a_openspiel_public_baselines_results.json`

2. **Round `R26-B` (`Ataraxos` `arXiv:2511.07312` / *Nature* 2026 `10x10` Stratego Evaluation, `3` Seeds `101..103`):**
   - **Pre-Registration:** [`benchmarks/PREREG-R26B-ATARAXOS-STRATEGO.md`](file:///home/agent/workspaces/inconsistency-search/benchmarks/PREREG-R26B-ATARAXOS-STRATEGO.md)
   - **Runner Script:** [`benchmarks/run_r26b_ataraxos_stratego.py`](file:///home/agent/workspaces/inconsistency-search/benchmarks/run_r26b_ataraxos_stratego.py)
   - **Upstream Patches & Unit Tests:** [`external/patches/ataraxos_belief_transformer_generate.patch`](file:///home/agent/workspaces/inconsistency-search/external/patches/ataraxos_belief_transformer_generate.patch), [`external/patches/ataraxos_build_cuda_sm89.patch`](file:///home/agent/workspaces/inconsistency-search/external/patches/ataraxos_build_cuda_sm89.patch), & [`tests/test_external_patches.py`](file:///home/agent/workspaces/inconsistency-search/tests/test_external_patches.py)
   - **Target Output JSON:** `results/r26b/r26b_ataraxos_stratego_results.json`

3. **Unified Sequential Wrapper (Optional Convenience Entrypoint):**
   - [`benchmarks/run_r26_external_public_baselines.py`](file:///home/agent/workspaces/inconsistency-search/benchmarks/run_r26_external_public_baselines.py) (invokes `run_r26a_openspiel_public_baselines.py` and/or `run_r26b_ataraxos_stratego.py` via `--part {all,r26a,r26b}`).

---

## 2. Point-by-Point Resolution of the 6 Blocking Items in `outbox/r26-freeze-review-2026-10-19.md`

| Blocking Item | Issue Identified in Freeze Review | Resolution in `R26-A` and `R26-B` |
| :--- | :--- | :--- |
| **1. Citation (`R26-B`)** | Cited `arXiv:2304.13138` (Update Equivalence, ICML 2023) as the Ataraxos paper. | Corrected across all files: **Sokota, Vinitsky, Hu, Kolter, and Farina (2025/2026), *"Scalable Decision Making for Games of Imperfect Information"* (`Ataraxos`), `arXiv:2511.07312`, repo `AtaraxosAI/stratego` at commit `92db29e`**, with theoretical lineage citing Update Equivalence (`arXiv:2304.13138`, ICML 2023) and Magnetic Mirror Descent (`arXiv:2206.05825`, ICLR 2023). |
| **2. `Ataraxos_BeliefTransformer` Public Weights (`R26-B`)** | `AtaraxosAI/stratego` releases `model42400.pthw` (`14.7M` RL blueprint) and `init_model42400.pthw` (`12.6M` arrangement model), not the `57.1M` `BeliefTransformer` checkpoint. Our `1.2M` `240`-update model cannot be called an Ataraxos public baseline. | Implemented **Option 2(b)**: Named public baselines in `R26-B` are restricted strictly to released public weights & public analytic beliefs (`Ataraxos_14.7M_Blueprint`, `Ataraxos_SearchBot_Uniform`, `Ataraxos_SearchBot_Marginalized`). Our matched-compute (`240`-update) trainings of upstream `BeliefTransformer` and `ARBelief` are renamed `BeliefTransformer_Arch_OurTraining_240Updates` and `ARBelief_Arch_OurTraining_240Updates` and reported as diagnostic rows with **zero pass/fail gates**. |
| **3. `OpenSpiel` Baseline Budgets & Learning-Curve Reproduction (`R26-A`)** | At `100–1500` episodes, upstream neural/sampled solvers sit near `Control_Uniform` (`4.7472` on `Leduc`). | Added **Gate `R26A-D0.3` (Upstream Learning-Curve Reproduction Gate)** across **4 cumulative budget stages (`Stage 1..4`)** reaching the trained regime (`T=400` full-tree CFR/CFR+/DCFR, `T=40` × `100` traversals = `8,000` traversals for `DeepCFRSolver` reaching `~2.2` on Leduc and `~0.11` on Kuhn, `N=40,000` episodes for `OutcomeSamplingMCCFR`, `N=8,000` episodes for `NFSP` reaching `~1.98` on Leduc and `~0.25` on Kuhn, `T=600` updates for `RNaDSolver`), comparing all arms across all 4 stages and by **Area Under the Learning Curve (`AULC`)** across `5` seeds (`101..105`). Also fixed the `OpenSpielInfosetGraph` world-state collapse via `OpenSpielWorldInfosetGraph` so `GLCAOM_v3_B1_PureNeural` (`1-pass` network, zero test-time search) reaches `~0.10` on Kuhn and `~1.24` on Leduc. |
| **4. `Stratego` Power Calculation & 95% CI Bars (`R26-B` & `R26-A`)** | `games_per_pair = 6` (`18` games across 3 seeds) had `95% CI` half-width `±22 pp` (unpowered). | In `R26B-E1a`, increased to **`50` seat-swapped pairs = `100` games per seed (`300` games per focal matchup across 3 seeds)** (`SE_pool ~ 2.7 pp`, `>= 84%` power at `alpha=0.05`). In `R26B-E1b`, increased to **`600` held-out boards per seed (`1,800` boards / `~54,000` hidden slots total)**. In `R26A-E2`, increased to **`100` paired games per matchup per seed (`500` games per matchup across 5 seeds)**. Every tournament and calibration bar is now an explicit **95% confidence interval lower/upper bound bar** reported both per seed and pooled. |
| **5. Explicit Patch Files & Unit Tests (`external/patches/`)** | `BeliefTransformer.generate` fix and JAX `jnp.clip` shim were applied inline without standalone patch diffs and unit tests. | Extracted [`external/patches/ataraxos_belief_transformer_generate.patch`](file:///home/agent/workspaces/inconsistency-search/external/patches/ataraxos_belief_transformer_generate.patch), [`external/patches/ataraxos_build_cuda_sm89.patch`](file:///home/agent/workspaces/inconsistency-search/external/patches/ataraxos_build_cuda_sm89.patch), and [`external/patches/openspiel_jax_jnp_clip_shim.patch`](file:///home/agent/workspaces/inconsistency-search/external/patches/openspiel_jax_jnp_clip_shim.patch), verified by [`tests/test_external_patches.py`](file:///home/agent/workspaces/inconsistency-search/tests/test_external_patches.py) (`3/3` unit tests pass, including `max_tf_diff == 0.0` teacher-forced output invariance). |
| **6. Component-by-Component Budget Re-Estimation** | Needed realistic component-level timing and cost for expanded game counts and budgets. | Re-estimated per component with Kelvin's `2026-10-19T15:55:25Z` authorization (up to `$100` for `R26` out of `$300` total budget, bead `pool-rafnx6`): `R26-A` (`5` seeds) = `~0.70 L4-hours` (`$0.61`, cap `2.00 L4-hours` / `$1.74`); `R26-B` (`3` seeds, batched `B_parallel=10` CUDA games) = `~1.00 L4-hours` (`$0.87`, cap `2.50 L4-hours` / `$2.18`). Combined `R26-A + R26-B` estimated cost `~$1.48` (combined hard cap `$3.92`). |
