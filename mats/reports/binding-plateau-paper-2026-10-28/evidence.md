# Local evidence index

Checked 2026-10-28T14:02:13Z by reading and hashing these artifacts. No training, checkpoint evaluation or publication is part of this index. The file table below gives full SHA256 values; rerun `sha256sum PATH` before reuse. JSON rows should be selected by their width, condition, landmark or run fields, not assumed array positions.

## E1 — New cohorts and the two AUC definitions

`P/evidence/baseline-analysis.json`, `groups[width=16 or 32,condition=baseline]`: `observed_horizon=80000`; `plateau_incidence.events/denominator` is 12/50 and 4/50; `acquisition["0.1"]` gives 39/50 and 47/50; `escape_among_plateau["0.1"]` gives 1/12 and 1/4. Each has `wilson95`. `terminal_fresh_below_threshold["0.1"].events` independently gives 39 and 47. `per_seed` retains event/detection dates.

**Selected summaries of prespecified retrospective analyses:** `early_predictors[width=16,condition=baseline].landmarks[landmark=1000].metrics.query_reading_mass` gives `fixed_direction_auc=0.9067599`, `stratified_seed_bootstrap95=[0.8135198,0.9720280]`. At 5000, `eligible=48` and `metrics.own_source_contrast` gives 0.9852580 [0.9484029,1]. This analysis excludes retrospectively confirmed first-point onsets at/before each landmark, rather than only confirmations already observable then. Implementation: `A/analyze.py::early_predictors`; registered metrics/directions/landmarks: `A/PREREG.md`.

All 20 width-16 baseline metric/landmark combinations, with fixed directions, eligible counts and intervals, are in [P/evidence/auc-all-landmarks.md](evidence/auc-all-landmarks.md). Selection of the two highlighted AUCs was not preregistered. The intervals do not adjust for selecting results from this set.

**Post-review operational sensitivity:** `P/evidence/confirmation-risk-set.json`, `groups[width=16].landmarks[landmark=5000]`, retains all 50 unconfirmed seeds; `metrics.own_source_contrast.confirmation_risk_set_auc=0.9860140` and `stratified_seed_bootstrap95=[0.9533800,1]`. `retained_pending_confirmation` identifies seeds 100 and 110. The same four metrics, five landmarks and 2,000 stratified seed-bootstrap draws are used; the 1k result is unchanged. This is a secondary sensitivity, not a replacement of the registered analysis. Its `input_sha256` records the script, analyzer and raw cohort inputs. Its original `/tmp/astra-pfn-baselines.json` input differs from the paper's baseline copy only in `checked_at`, verified for this index.

The baseline JSON's `sources` names both raw directories: `/tmp/l4runs/astra-pfn28-w16b/runs/astra-pfn28-w16b/baseline` and `/tmp/l4runs/astra-pfn28-w32/runs/astra-pfn28-w32/baseline`. `curve.json` contains `step` and per-seed `gap`; `meta.json` contains `seeds`, `final_step`, and `test.gap`. Hashes of all four files are in the sensitivity's `input_sha256`. These are local retrieval paths, not packaged public data.

**GPU conformance:** exact synced `conformance.json` records are retained as `P/evidence/gpu-conformance-w16.json`, `P/evidence/gpu-conformance-w32.json`, and `P/evidence/gpu-conformance-batch.json`. Their original paths are:

- Width-16 job: `/tmp/l4runs/astra-pfn28-w16b/runs/astra-pfn28-w16b/conformance.json`.
- Width-32 job: `/tmp/l4runs/astra-pfn28-w32/runs/astra-pfn28-w32/conformance.json`.
- Batch job: `/tmp/l4runs/astra-pfn28-batch/runs/astra-pfn28-batch/conformance.json`.

All three records have identical bytes and report `device="cuda"`, `passed=true`, and PyTorch `2.14.1+cu130`. Each runs the same deterministic suite: widths 16/32, epsilon 1e-8/1e-12, three paired models, batch 7 and five optimizer updates. `A/study.py::conformance_case` compares batched versus individual predictions, clipped gradients and AdamW updates. The maximum saved prediction error is 7.748603820800781e-7; the maximum non-key-bias parameter error is 4.6193599700927734e-7. These engine checks cover the stated small batches and update horizon. The CPU record is `A/cpu-conformance.json`.

The parameter comparison excludes the key-projection bias slice: adding the same key bias adds a constant to each attention-logit row, canceled by softmax in exact arithmetic. FP32 residual gradients can yield different Adam updates in that direction. Those differences are retained in `max_key_bias_gauge_errors` (maximum 0.008247045800089836); they are not silently zeroed or covered by the non-key-bias tolerance.

## E2 — Historical predictions

`O/tiny-pfn-plateau-historical-audit-2026-10-28.json`: `cases[config,seed].model_minus_exact_query_only_nll` contains `mean`, `sample_se`, and `normal95`; `exact_references.query_only_minus_bayes` gives the query-only reference gap/SE. Each case's `files` records exact prediction, metadata and test-set paths/hashes. The accompanying note explains the heterogeneous pooled 21/95 historical count.

The six stalled runs are `w16_h1_L2_mix4_s3`, `w16_h1_L2_mix4_s4`, `w16_h2_L2_ds4_mix4_s1`, `w32_h2_L2_mix4_s3`, `w16_h2_L2_s3`, and `w16_h2_L2_s4`. Their predictions are `/home/agent/workspaces/mats/work/tiny-pfn/runs/RUN/test_preds.npz` (fields `nll`, `mu`, `kl`). These use the **retained best-validation `model.pt` checkpoints**, not necessarily the terminal weights, and are distinct from E1's new cohorts. `A/reference-input-sha256.json.files` covers all six prediction files, both saved test sets and reference code/results; every listed hash was checked for this index.

## E3 — Restricted Bayes references

`A/binding-reference.json.references[query_mixture,binding_blind_mixture,binding_blind_gaussian,full_mixture,full_gaussian].gap_to_full` contains means and paired episode SEs. `information_decomposition.unordered_reliability_gain` and `.source_binding_gain` give the two information terms. `comparisons[run].model_minus_blind_mixture`, `.model_minus_blind_gaussian`, and `.model_kl_minus_blind_gaussian_kl` separate sampled and target-integrated comparisons.

`A/pooled-reference.json.references[pooled_mixture,pooled_gaussian].gap_to_full` gives the scalar-total-residual references. `comparisons[run].model_minus_pooled_mixture`, `.model_minus_pooled_gaussian`, and `.model_kl_minus_pooled_gaussian_kl` support the unrestricted-density and Gaussian-head-specific comparisons. `A/REFERENCE_REVIEW.md` records the Gaussian projection identity and interpretation limits. Episode SEs are not seed uncertainty; these references were developed after historical inspection.

The input manifest also hashes `world.py` and both reference scripts. The pooled result's recorded calculation-source hash is `78b7e0710c61a330d9db85389812db33af270cc95203495d60be4aa7b46558d9`; its current source includes a later prose-only range correction, recorded separately in that manifest.

## E4 — Fresh source-association intervention

`A/binding-intervention.json`: `seed=40000`, `n=4096`, `selection.selected_n=1988` at the fixed half-sum symmetric-KL threshold 0.3. Select each of eleven `models` rows by `run`/`historical_status`: `selected.mean_response_rms_ratio`, `.signed_slope_through_origin`, `.pearson_correlation`, `.symmetric_kl_ratio_of_means`, and `.symmetric_kl_ratio_paired_se` give the reported responses; `all` gives full-set counterparts. `attention_before.pooled.own_other_ratio` is the per-token attention ratio pooled across heads/sources. `quadrature` and top-level `validation` retain numerical checks.

All eleven inputs are historically selected **best-validation `model.pt` checkpoints**. `A/binding-intervention-inputs.json.files` gives exact original checkpoint/metadata paths and hashes; `selection_rows` records best steps and historical gaps. Controls are the lowest successful sibling seed in each of four families, plus `w16_h2_L2_s2`, the documented plateau-then-escape case. These are selected contrasts, not random seed replications. The manifest's `fresh_outcomes_computed=false` records its pre-evaluation timing, not the status of the later completed result. All checkpoint/metadata hashes were rechecked for this index.

`A/binding-intervention-arrays.npz` retains `bayes_mean_change`, `bayes_symmetric_kl`, `selected_bayes_skl_ge_point3`, original/intervened inputs, and `RUN__mean_change`, `RUN__symmetric_kl`, `RUN__directional_kl`. `binding-intervention-sha256.txt` and the result's `source_sha256`, `inputs_sha256`, and `arrays_sha256` bind the protocol, code, input manifest and arrays.

Distribution-shift caveat for E4: the fixed cycle leaves inputs on support but
changes their joint distribution. Before subset selection, conditional independence
gives Cov(S_s, Q_s²) = Cov(4v_s, 1 + v_s) = 4 Var(v_s) > 0. Attaching a different
source's residual group makes that covariance zero because source variances are
independent. The identities need not survive conditioning on the selected
high-change subset. The intervention measures functional sensitivity under that
shift; it is not a distribution-preserving randomization.

## E5 — Earlier acquired-network mechanism evidence

`O/tiny-pfn-interp-step1-2026-10-27.md` (frozen here as `P/evidence/prior-mechanism-snapshot.md`, same SHA256) is an **internal reproducibility report**, not a publication. Relevant locations are the source-pooling attention table (lines65–84) and residual-change patches (lines135–158). Main acquired models there are earlier warm-started mixture runs, not E1's new cohorts.

For tags `main_s0`–`main_s4`, `small_s2`, `small_s4`, and `plateau_s3`, `I/out/TAG.json.attn_trained.L1H0/1.QQ.ratio_lab` measures source-selective pooling; `probes_trained["logSS|QQ|mlp1"]` and `["E_lab|QQ|mlp1"]` locate reliability information. `I/out/steps_TAG.json.step2_trained.a_QQ_mlp1.mean_vs_full` measures post-MLP1 state transfer; `.full_swap.mean`, `.a_QQ_mlp1.mean`, and `.R_random_QQ_mlp1.mean` retain Bayes-shift and control comparisons. `step2_untrained` retains the untrained control. Mixed preregistered patch outcomes and the random-direction confound remain in the report; the paper uses the narrower qualitative relationship.

## Figure

`P/figures/binding-summary-data.json.panel_a` and `.panel_b` contain exact plotted reference/checkpoint values; `inputs` and `outputs` record numerical-input and PDF/PNG hashes. Each checkpoint is one point, without a pooled model-level confidence interval.

## Exact locations and SHA256

The aliases below expand only filesystem prefixes (no implicit filename changes):

- `P = /home/agent/workspaces/mats/outbox/binding-plateau-paper-2026-10-28`
- `A = /home/agent/workspaces/mats/work/astra-tiny-pfn-plateau-20261028` (symlink to the experiment checkout)
- `O = /home/agent/workspaces/mats/outbox`
- `I = /home/agent/workspaces/mats/work/tiny-pfn-interp`

| Evidence | File under the exact prefix above | SHA256 |
|---|---|---|
| E1 | [P/evidence/auc-all-landmarks.md](/home/agent/workspaces/mats/outbox/binding-plateau-paper-2026-10-28/evidence/auc-all-landmarks.md) | `d61614f6e720f6a9acbae7d6ac28c1dd8ce848a2384173ae61757a76f0f2eee5` |
| E1 | [P/evidence/gpu-conformance-w16.json](/home/agent/workspaces/mats/outbox/binding-plateau-paper-2026-10-28/evidence/gpu-conformance-w16.json) | `bf38d14966ffe782483cb601bc50dffa2753b01ba73df29b52e4b439da9cbcec` |
| E1 | [P/evidence/gpu-conformance-w32.json](/home/agent/workspaces/mats/outbox/binding-plateau-paper-2026-10-28/evidence/gpu-conformance-w32.json) | `bf38d14966ffe782483cb601bc50dffa2753b01ba73df29b52e4b439da9cbcec` |
| E1 | [P/evidence/gpu-conformance-batch.json](/home/agent/workspaces/mats/outbox/binding-plateau-paper-2026-10-28/evidence/gpu-conformance-batch.json) | `bf38d14966ffe782483cb601bc50dffa2753b01ba73df29b52e4b439da9cbcec` |
| E1 | [A/cpu-conformance.json](/home/agent/workspaces/mats/work/astra-tiny-pfn-plateau-20261028/cpu-conformance.json) | `ea9992c4cb6d9ad2e869095865a16491462d72e30aa3597ca6f06a813731faff` |
| E1 | [P/evidence/baseline-analysis.json](/home/agent/workspaces/mats/outbox/binding-plateau-paper-2026-10-28/evidence/baseline-analysis.json) | `1a54193b8ef9ff6697db081c620bc1aa9a0807dbe8bc7662b820ddcdb9b44298` |
| E1 | [P/evidence/baseline-analysis.md](/home/agent/workspaces/mats/outbox/binding-plateau-paper-2026-10-28/evidence/baseline-analysis.md) | `bd389e057ccafbd740127ee884f2094df766338f0dc634a5a94c6e2179036393` |
| E1 | [P/evidence/confirmation-risk-set.json](/home/agent/workspaces/mats/outbox/binding-plateau-paper-2026-10-28/evidence/confirmation-risk-set.json) | `6d4d11a238f57dfe435f6e6aea766b51a099170cae2ae22be45deca083db0b59` |
| E1 | [P/evidence/confirmation_risk_set.py](/home/agent/workspaces/mats/outbox/binding-plateau-paper-2026-10-28/evidence/confirmation_risk_set.py) | `6bfef83cc7590bc771074960d3903f91f3895e2d91f2143bbae81b2e9156a688` |
| E1 | [A/PREREG.md](/home/agent/workspaces/mats/work/astra-tiny-pfn-plateau-20261028/PREREG.md) | `f1d0def7ef647f6202c385baf3b4e9d8d6509094a06e88f220205104845a3ee8` |
| E1 | [A/study.py](/home/agent/workspaces/mats/work/astra-tiny-pfn-plateau-20261028/study.py) | `48e6c615db3c0a8b09d4220adffac82563bc022f17e31ddaff5631f206437e8d` |
| E1 | [A/analyze.py](/home/agent/workspaces/mats/work/astra-tiny-pfn-plateau-20261028/analyze.py) | `00c29d80b1ea8713c2064673e43adae6d28214465abeaf545c3dceaade15f5ec` |
| E2 | [O/tiny-pfn-plateau-historical-audit-2026-10-28.json](/home/agent/workspaces/mats/outbox/tiny-pfn-plateau-historical-audit-2026-10-28.json) | `1a1d6fa794b139b179fdd19af81628ca396d7e48d5c21aa8a502f2d0522b75f2` |
| E2 | [O/tiny-pfn-plateau-historical-audit-2026-10-28.md](/home/agent/workspaces/mats/outbox/tiny-pfn-plateau-historical-audit-2026-10-28.md) | `e8ac1f15eaf9157f791c0aa90d5bdeefcc09195c789d3cab7663b7e19ef930f4` |
| E2/E3 | [A/reference-input-sha256.json](/home/agent/workspaces/mats/work/astra-tiny-pfn-plateau-20261028/reference-input-sha256.json) | `84ab42a2565b170e78e35dab5954a63b51ac1881e75e6882086cfee964d95c06` |
| E3 | [A/binding-reference.json](/home/agent/workspaces/mats/work/astra-tiny-pfn-plateau-20261028/binding-reference.json) | `fdc07e0e81d80fec58e9fa432d202dfce5739c4f87e21f17b131b014361fb4d3` |
| E3 | [A/pooled-reference.json](/home/agent/workspaces/mats/work/astra-tiny-pfn-plateau-20261028/pooled-reference.json) | `98dade2a710899ab38a0c527bcbff50c2149e003366d3708a18f02eb6161f573` |
| E3 | [A/REFERENCE_REVIEW.md](/home/agent/workspaces/mats/work/astra-tiny-pfn-plateau-20261028/REFERENCE_REVIEW.md) | `60fc214a96497201baa26dc0fae6320eceb7412c66ed910ee394f034e7ef1ba5` |
| E4 | [A/BINDING_INTERVENTION_PREREG.md](/home/agent/workspaces/mats/work/astra-tiny-pfn-plateau-20261028/BINDING_INTERVENTION_PREREG.md) | `49d5522988b2045a3fae7c82a6011bb8cd1dd8a971ac403d8a91e711cfc483ac` |
| E4 | [A/binding-intervention-inputs.json](/home/agent/workspaces/mats/work/astra-tiny-pfn-plateau-20261028/binding-intervention-inputs.json) | `594706ca58dd8999f0bee31b79e38516f7a4b6214ad082626bd46cb1c585cd17` |
| E4 | [A/binding-intervention.json](/home/agent/workspaces/mats/work/astra-tiny-pfn-plateau-20261028/binding-intervention.json) | `b805971c141b6dea38756bf8ed7b0993ab8a716807ecbbd0357635fc8ed96ebd` |
| E4 | [A/binding-intervention-arrays.npz](/home/agent/workspaces/mats/work/astra-tiny-pfn-plateau-20261028/binding-intervention-arrays.npz) | `2a44c49296bdcfa103e87b9c139109b88eb57a54bc5cefee90fb89f9195b9ae9` |
| E4 | [A/binding-intervention-validation.json](/home/agent/workspaces/mats/work/astra-tiny-pfn-plateau-20261028/binding-intervention-validation.json) | `030dfb9fffb51842212341087958f257f2c0945f4ee6136734f80ab33df2b1bb` |
| E4 | [A/binding-intervention-sha256.txt](/home/agent/workspaces/mats/work/astra-tiny-pfn-plateau-20261028/binding-intervention-sha256.txt) | `0db4c010b5ded59690429c96f18ed4d2b451349075a17e484263b52eef45ea04` |
| E5 | [P/evidence/prior-mechanism-snapshot.md](/home/agent/workspaces/mats/outbox/binding-plateau-paper-2026-10-28/evidence/prior-mechanism-snapshot.md) | `db66ed8ef1dcf25b452ed8c4d8af7f16cf08a9730ab736e534d897ac40e57b67` |
| E5 | [I/PREREG.md](/home/agent/workspaces/mats/work/tiny-pfn-interp/PREREG.md) | `d087fb352bd1b13b173c8522aeea111f778b1ff2fe3b61b69243985457545428` |
| E5 | [I/locate.py](/home/agent/workspaces/mats/work/tiny-pfn-interp/locate.py) | `4b4c50a42d92fcb2277c700d7dff5e61d724df68c75cc3c9240c67170c5ab812` |
| E5 | [I/steps.py](/home/agent/workspaces/mats/work/tiny-pfn-interp/steps.py) | `9367a13eb2b780e6696092e656ec9c5d70dc2fedc68aa448f4ca39c5edb363a8` |
| Figure | [P/make_binding_figure.py](/home/agent/workspaces/mats/outbox/binding-plateau-paper-2026-10-28/make_binding_figure.py) | `cd73b329433c40273a17dbac927ab725b530651d97e9b1b97cd86e0ddae18f35` |
| Figure | [P/figures/binding-summary-data.json](/home/agent/workspaces/mats/outbox/binding-plateau-paper-2026-10-28/figures/binding-summary-data.json) | `96f04e4e2312653930e388ef635b53e0813e91dc02572808824468317f8074b5` |
| Figure | [P/figures/binding-summary.pdf](/home/agent/workspaces/mats/outbox/binding-plateau-paper-2026-10-28/figures/binding-summary.pdf) | `73f6a337ac649c585fd270b9cd5d6b4220afd83e3c017c51efa9f66cd643c8af` |
| E5 | [I/out/main_s0.json](/home/agent/workspaces/mats/work/tiny-pfn-interp/out/main_s0.json) | `0c4904336b8eead1df107a1c86948d68d5432a7260e8b9f3b3f60b5e2792564a` |
| E5 | [I/out/steps_main_s0.json](/home/agent/workspaces/mats/work/tiny-pfn-interp/out/steps_main_s0.json) | `73c16308aa6b8fa8e0dfc4bfea3b32a53490670c8f4cf23825143b74d5595d64` |
| E5 | [I/out/main_s1.json](/home/agent/workspaces/mats/work/tiny-pfn-interp/out/main_s1.json) | `56e0f626c1dc5f27a3a4258b92bbee4a3d9c5a187e8a1b204a7aad6e3f2378ac` |
| E5 | [I/out/steps_main_s1.json](/home/agent/workspaces/mats/work/tiny-pfn-interp/out/steps_main_s1.json) | `80f37cc8f3a6e499ab182a64ea8bbdcadf2a42b2cceae54643a97d9119436650` |
| E5 | [I/out/main_s2.json](/home/agent/workspaces/mats/work/tiny-pfn-interp/out/main_s2.json) | `f38d6a10a04525ad14c3f7cc319e31a56a21c3f9f2e71340f8e11e9537c10827` |
| E5 | [I/out/steps_main_s2.json](/home/agent/workspaces/mats/work/tiny-pfn-interp/out/steps_main_s2.json) | `fe8eb03bc85a5aa873f62628ef1ea08ef01735c280e71f681bf1a5a3a2740872` |
| E5 | [I/out/main_s3.json](/home/agent/workspaces/mats/work/tiny-pfn-interp/out/main_s3.json) | `72a351c3d4ea9fc33903d233ce83f57310b45d6541fb5997e38e96fd13dff725` |
| E5 | [I/out/steps_main_s3.json](/home/agent/workspaces/mats/work/tiny-pfn-interp/out/steps_main_s3.json) | `c477164cad7bb3f72c27bbbb11ec91cf6579b101d8e0be76aa5286c286c20c6a` |
| E5 | [I/out/main_s4.json](/home/agent/workspaces/mats/work/tiny-pfn-interp/out/main_s4.json) | `261e15770eafe761d6a82d2507d6c65be378271d7f11da639daf17799b3999dc` |
| E5 | [I/out/steps_main_s4.json](/home/agent/workspaces/mats/work/tiny-pfn-interp/out/steps_main_s4.json) | `64d1a106ad0ab9c2c5f9d1461eb87fd80032853e65bd3ee056c4b98bff2fd02f` |
| E5 | [I/out/small_s2.json](/home/agent/workspaces/mats/work/tiny-pfn-interp/out/small_s2.json) | `44f606ef18f0b0fbcb66f34af5388bdd1b175112df00b9123750ec0e37532644` |
| E5 | [I/out/steps_small_s2.json](/home/agent/workspaces/mats/work/tiny-pfn-interp/out/steps_small_s2.json) | `9a48dabd58e5a69f2307d3fc130d82b49b6403987c05016c813127e9be0a8036` |
| E5 | [I/out/small_s4.json](/home/agent/workspaces/mats/work/tiny-pfn-interp/out/small_s4.json) | `730fc2c375d43a3005979468038150eae3ad233069d43c00533bb4384a27cfde` |
| E5 | [I/out/steps_small_s4.json](/home/agent/workspaces/mats/work/tiny-pfn-interp/out/steps_small_s4.json) | `0342f744feeaf8e219660a9905367539292f4d643d45a6c322435487c238d589` |
| E5 | [I/out/plateau_s3.json](/home/agent/workspaces/mats/work/tiny-pfn-interp/out/plateau_s3.json) | `93a02794d25879b40effd9f9aee5ad377fc6d8aecfb532198f2255b57dd314bd` |
| E5 | [I/out/steps_plateau_s3.json](/home/agent/workspaces/mats/work/tiny-pfn-interp/out/steps_plateau_s3.json) | `e75bc112ac968ba900837f3f6f27eb2a054f42f6f7e86435b85cdccc0cd911f1` |

The analyzer later received a prose-only clarification of retrospective landmark eligibility (commit `f0cbaca1`); sensitivity input hashes retain its calculation-time version. Numerical analysis code is unchanged.
