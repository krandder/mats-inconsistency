> **PARKED 2026-10-19 (Kelvin): not frozen, not run**

# R26 Pre-Registration Ready Handshake (`R26-A` OpenSpiel & `R26-B` Ataraxos Stratego)

- **Timestamp:** `2026-10-19T17:55:00Z`
- **Status:** **PARKED 2026-10-19 (Kelvin): not frozen, not run** (Elo/R26 parked per `/home/agent/workspaces/inconsistency-search/outbox/opus-mats-r26-park-2026-10-19.md` and `/home/agent/workspaces/mats/outbox/research-orientation-2026-10-19.md`).
- **Role:** Proposer & Coder (`docs/PROTOCOL.md` — no self-freeze, no GPU VM launched).
- **Audit Status of Local Smoke Numbers:** All local pre-freeze smoke numbers are **UNAUDITED**.

---

## 1. Resolution of All 6 Blocking Items from `outbox/r26-freeze-review-2026-10-19.md` + `opus-mats` Point 2 Corrections

1. **Blocking Item 1 (Corrected Citation & Repository):**
   - **Primary Citation:** Samuel Sokota, Eugene Vinitsky, Hengyuan Hu, J. Zico Kolter, and Gabriele Farina, *"Scalable Decision Making for Games of Imperfect Information"* (`Ataraxos`), `arXiv:2511.07312` (v2).
   - **Repository:** `https://github.com/AtaraxosAI/stratego` at commit `92db29e8ffc323b1b8a2804b5c3f84695d036b05` (`external/ataraxos_stratego`).
2. **Blocking Item 2 (Public Weights Only for Named Baselines — Option 2b, Exact Files Loaded by Code):**
   - `AtaraxosAI/stratego` at `92db29e` releases and `benchmarks/run_r26b_ataraxos_stratego.py` loads **only**:
     * `pretrained/final_run/model42400.pthw` (`14.7M`-parameter RL blueprint, SHA256 `8a490d1afe14c0a105c725a71e80ea517c98d9dcf332b3d68a3acbd503d2f52f`),
     * `pretrained/final_run/init_model42400.pthw` (`12.6M`-parameter initial arrangement generator, SHA256 `35b3362dbe0bf8d308ed42759df0dbcaaa848c72e5cdeec436f963a948c1a6a6`),
     * `pretrained/final_run/arrangements42400.pkl` and `pretrained/final_run/ema_arrangements42400.pkl`,
     * `tests/test_data/rl_test_data.pkl`.
   - No public weights exist for `BeliefTransformer`. Under **Option 2b**, named public baselines in `R26-B` are restricted strictly to `Ataraxos_14.7M_Blueprint` (`model42400.pthw` + `init_model42400.pthw`), `Ataraxos_SearchBot_Uniform` (`model42400.pthw` + `SearchBot` + `pyengine.belief.uniform.uniform_belief`), and `Ataraxos_SearchBot_Marginalized` (`model42400.pthw` + `SearchBot` + `pyengine.belief.uniform.marginalized_uniform_belief`). Downscaled `240`-update trainings (`BeliefTransformer_Arch_OurTraining_240Updates`, `ARBelief_Arch_OurTraining_240Updates`) are diagnostic rows only with zero pass/fail gates.
3. **Blocking Item 3 (`OpenSpiel` Baseline Budgets & Learning-Curve Reproduction `R26A-D0.3`):**
   - Added 4 cumulative budget stages (`Stages 1..4`: `CFRPlusSolver` up to `400` full tree sweeps, `OutcomeSamplingMCCFR` up to `40,000` episodes, `DeepCFRSolver` up to `40` iterations $\times$ `100` traversals, `NFSP` up to `8,000` episodes, `RNaDSolver` up to `600` steps) with gate `R26A-D0.3` verifying upstream baseline convergence and comparing both final `NashConv` and Area Under the Learning Curve (`AULC`).
4. **Blocking Item 4 (Power Calculation & 95% CI Bars):**
   - `R26-B` runs `50` seat-swapped pairs = `100` games per matchup per seed (`300` games across 3 seeds) and `600` held-out boards per seed (`1,800` boards across 3 seeds) on `arrangements[p][512:]`. `R26-A` runs `100` games per pair per seed (`500` games across 5 seeds). All tournament and belief bars are 95% CI lower/upper bound bars.
5. **Blocking Item 5 (Explicit Patch Files & Unit Tests in `external/patches/`):**
   - Extracted unified diffs in `external/patches/` verified by `tests/test_external_patches.py` (`3/3` unit tests pass).
6. **Blocking Item 6 (Component-by-Component Budget Estimation):**
   - `R26-A` (`5` seeds): `~0.70 L4-hours` (`~$0.61`, cap `2.00 L4-hours` / `$1.74`).
   - `R26-B` (`3` seeds): `~1.00 L4-hours` (`~$0.87`, cap `2.50 L4-hours` / `$2.18`).

---

## 2. File Paths & SHA256 Hashes

| Role | File Path | SHA256 Hash |
|---|---|---|
| **Master Index PREREG** | `/home/agent/workspaces/inconsistency-search/benchmarks/PREREG-R26-EXTERNAL-PUBLIC-BASELINES.md` | `24129926fdd0542274acb6adf45dc8fab1e602abe68f0dd4e7ef7ab64165df76` |
| **`R26-A` PREREG (`OpenSpiel`)** | `/home/agent/workspaces/inconsistency-search/benchmarks/PREREG-R26A-OPENSPIEL-PUBLIC-BASELINES.md` | `06fb280ee89481ff6c4dc47e173283f771cae9c7acfbf45617557c53905f2774` |
| **`R26-B` PREREG (`Ataraxos`)** | `/home/agent/workspaces/inconsistency-search/benchmarks/PREREG-R26B-ATARAXOS-STRATEGO.md` | `2f75861b67df2b1c74da341878392f5e782eb1c678ae4d9c99a90a5c29443458` |
| **Combined Dispatcher** | `/home/agent/workspaces/inconsistency-search/benchmarks/run_r26_external_public_baselines.py` | `aef7aef08318e939f6df30766ec3638f5b66e1bcbbab3da4ec0ebad55fabe79d` |
| **`R26-A` Runner (`OpenSpiel`)** | `/home/agent/workspaces/inconsistency-search/benchmarks/run_r26a_openspiel_public_baselines.py` | `1943bde29becebb446f7fad5c2d424d51cfaebc436dd2803857a34949e913d81` |
| **`R26-B` Runner (`Ataraxos`)** | `/home/agent/workspaces/inconsistency-search/benchmarks/run_r26b_ataraxos_stratego.py` | `d21c19625ad5426e7b16752ed8c42fba82c5376ff5aadb4840a3a79735926595` |
| **Patch 1 (`BeliefTransformer.generate`)** | `/home/agent/workspaces/inconsistency-search/external/patches/ataraxos_belief_transformer_generate.patch` | `53c5f51fe966980446453fd6ca52fc39a7962598743684b48d61b07f8d2d62bd` |
| **Patch 2 (`Ataraxos CUDA sm_89`)** | `/home/agent/workspaces/inconsistency-search/external/patches/ataraxos_build_cuda_sm89.patch` | `3f0c7a6faa27cbcd79bbd11db024b64ac50aa19bc3d839cfedb1bf2a76630d3b` |
| **Patch 3 (`OpenSpiel JAX 0.9 jnp.clip`)** | `/home/agent/workspaces/inconsistency-search/external/patches/openspiel_jax_jnp_clip_shim.patch` | `90dd586479ddb1eaaa8c0d5c2c7303cc6a407ec69f41df46d60d2d052ff97f3a` |
| **Patch Unit Tests (`3/3` pass)** | `/home/agent/workspaces/inconsistency-search/tests/test_external_patches.py` | `08cf57836d077cbf05401cd652af668bcbc3cea167ebdf5433caaf2e9295d790` |
| **Static AST Leak Auditor (`v2`)** | `/home/agent/workspaces/inconsistency-search/src/eval/ast_leak_auditor.py` | `72ce035b39a069acf26e97657bac099d8dd1bf82c09b2785f3c5be10b2b45dac` |
