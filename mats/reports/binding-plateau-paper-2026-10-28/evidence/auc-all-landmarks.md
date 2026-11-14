# All 20 width-16 baseline AUC analyses

Checked 2026-10-28T14:02:13Z against the retained baseline-analysis JSON. This table transcribes existing results without rerunning inference or bootstrap sampling.

Source: [baseline-analysis.json](baseline-analysis.json), SHA256 `1a54193b8ef9ff6697db081c620bc1aa9a0807dbe8bc7662b820ddcdb9b44298`. Select `early_predictors[width=16,condition=baseline].landmarks` and the four metrics below. The source retains full floating-point values.

The four metric directions and five landmarks were prespecified in [PREREG.md](/home/agent/workspaces/mats/work/astra-tiny-pfn-plateau-20261028/PREREG.md). The paper's highlighted 1k query-reading and 5k own-source AUCs are **selected summaries of prespecified analyses**; selection of that particular pair was not preregistered. All 20 results are shown below.

Eligibility uses retrospectively identified first-point acquisition onset strictly after the landmark, or no acquisition by the 80k horizon. Acquisition requires three consecutive gaps below 0.10. Each eligible outcome class must have at least five seeds. AUC uses the fixed direction shown, without reversing it when the observed AUC is below 0.5.

Intervals are 95% stratified seed-bootstrap intervals from 2,000 draws, unadjusted for the 20 comparisons and conditional on the fixed evaluation episodes. These cohort associations do not establish independent forecasting validity or causal mechanisms. The [confirmation-based sensitivity](confirmation-risk-set.json) changes risk-set eligibility; it does not address selection optimism.

| Landmark (updates) | Metric | Fixed direction | Eligible (acquire / not by 80k) | AUC | 95% interval |
|---:|---|---|---:|---:|---|
| 0 | own_source_contrast | higher | 50 (39 / 11) | 0.5594405594 | [0.3402680653, 0.7669580420] |
| 0 | binding_disruption_nll | higher | 50 (39 / 11) | 0.7878787879 | [0.6526223776, 0.8974358974] |
| 0 | query_reading_mass | higher | 50 (39 / 11) | 0.7738927739 | [0.5850815851, 0.9277389277] |
| 0 | gap | lower | 50 (39 / 11) | 0.5011655012 | [0.3030303030, 0.7133449883] |
| 1000 | own_source_contrast | higher | 50 (39 / 11) | 0.4079254079 | [0.2307109557, 0.6037878788] |
| 1000 | binding_disruption_nll | higher | 50 (39 / 11) | 0.4592074592 | [0.2843240093, 0.6387529138] |
| 1000 | query_reading_mass | higher | 50 (39 / 11) | 0.9067599068 | [0.8135198135, 0.9720279720] |
| 1000 | gap | lower | 50 (39 / 11) | 0.7808857809 | [0.6340326340, 0.8974941725] |
| 3000 | own_source_contrast | higher | 50 (39 / 11) | 0.8508158508 | [0.7319347319, 0.9463869464] |
| 3000 | binding_disruption_nll | higher | 50 (39 / 11) | 0.6456876457 | [0.4941724942, 0.7878787879] |
| 3000 | query_reading_mass | higher | 50 (39 / 11) | 0.9580419580 | [0.8834498834, 1.0000000000] |
| 3000 | gap | lower | 50 (39 / 11) | 0.7179487179 | [0.5547202797, 0.8624708625] |
| 5000 | own_source_contrast | higher | 48 (37 / 11) | 0.9852579853 | [0.9484029484, 1.0000000000] |
| 5000 | binding_disruption_nll | higher | 48 (37 / 11) | 0.8206388206 | [0.6904176904, 0.9262899263] |
| 5000 | query_reading_mass | higher | 48 (37 / 11) | 0.9582309582 | [0.8771498771, 1.0000000000] |
| 5000 | gap | lower | 48 (37 / 11) | 0.7911547912 | [0.6560196560, 0.9066339066] |
| 10000 | own_source_contrast | higher | 28 (17 / 11) | 0.9893048128 | [0.9518716578, 1.0000000000] |
| 10000 | binding_disruption_nll | higher | 28 (17 / 11) | 0.9518716578 | [0.8502673797, 1.0000000000] |
| 10000 | query_reading_mass | higher | 28 (17 / 11) | 0.9572192513 | [0.8556149733, 1.0000000000] |
| 10000 | gap | lower | 28 (17 / 11) | 0.9411764706 | [0.8342245989, 1.0000000000] |
