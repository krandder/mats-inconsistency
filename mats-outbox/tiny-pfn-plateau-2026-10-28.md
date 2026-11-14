# Tiny-PFN plateau study

**Final analysis, checked 2026-10-28 13:48 UTC against all six saved arms, independent raw-curve checks and live GCP instance listing. Five arms reached their requested horizons; width-32 batch was administratively stopped with a verified 19k checkpoint.**
Report due 15:00 UTC; compute cap $10; launcher cost estimate $6.09.
All three owned VMs are deleted. Searchformer remains paused.

The clearest result is a source-binding failure in six retained stalled models.
On fresh episodes, cyclically swapping the source identities attached to the
labelled readings barely changes their predictions. Successful controls change
in approximately the direction and magnitude required by Bayes. This directly
tests use of the labelled-to-query source association, beyond comparing losses.

On 1,988 of 4,096 episodes where Bayes changes substantially, stalled models'
predictive-mean response is only **0.24–0.47% of Bayes's response** in RMS.
Five successful controls give **82–99%**, with signed slopes 0.76–0.98 and
correlations 0.92–0.99. Per-token own-source/other-source first-layer attention
is 0.997–1.240 in stalled models and 14.6–233 in the controls.

Calling the stalled solution “query-only” is nevertheless inaccurate: it uses
more context than exact query-only Bayes or a single pooled residual statistic.
The evidence supports extracting some reliability information while making
little effective use of source association under this intervention. It does
not identify the exact algorithm or explain why optimization misses binding.
The paired training study found no acquisition rescue from lowering Adam epsilon.
A larger batch improved width-16 acquisition from 39/50 to 44/50 at equal
episode exposure, but with substantial uncertainty (exact paired p=0.0625).
At width 32 the corresponding difference was one seed, 47/50 to 48/50.
The study supports learning source binding as a next experimental target;
it has not established the optimization mechanism or a training remedy.

## What the model must infer

Each episode has three sources whose noise variances are independently drawn
from {0.1, 0.5, 2}. Each source supplies noisy readings of four questions with
known answers and one query with an unknown answer. The model predicts the
query answer's distribution. Source and question identity vectors are freshly
sampled each episode. A good predictor uses the labelled residuals to estimate
which sources are reliable, then weights the query readings accordingly.
The 27 variance assignments can be marginalized exactly, giving a Bayes reference.
All gaps below are natural-log-loss differences in nats; smaller is better.

## The new experiment

Widths 16 and 32, two heads, two layers, Gaussian output head; 50 initializations
per condition, seeds 100–149. Baseline: batch 256, Adam epsilon 1e-8, 80k updates.
The epsilon intervention changes only epsilon to 1e-12. The batch intervention
requests batch 1024 for 20k updates. Baseline 80k and batch 20k each expose a model to
20,480,000 episodes. Width-16 batch reached 20k; width-32 batch's last saved
checkpoint is 19k after the fixed launcher timeout, giving 19,456,000 episodes.
The batch arm was added prospectively from throughput
measurements. All runs use AdamW, learning rate 1e-3, decay 0.01, 1k warmup,
per-model gradient clipping at 1, FP32, and no cooldown or convergence stopping.

Initial weights and random streams are paired within each width. At a given
update, the small batch is a prefix of the large batch; equal total exposure
does not give the two conditions identical episode sequences. The batch change
also alters the number of optimizer updates, warmup exposure, Adam's memory in
episodes, and cumulative weight decay. It is a practical batch intervention,
not an isolated estimate of gradient-noise effects.

A plateau requires ten consecutive 1k evaluations with validation gap in
[0.14, 0.18], spanning 9k updates. Acquisition requires three consecutive gaps
below 0.10; escape additionally requires a preceding plateau. The first point
is the event time and the third confirms it. Unobserved events remain censored.
Sensitivities use 0.12 and 0.05. Every terminal checkpoint also receives a fresh
4,096-episode evaluation. Confirmation spans 2k updates, hence different numbers
of examples at different batch sizes.

| Width / condition | Observed / requested updates | Plateau entrants | Acquisition <0.10 | Escapes / entrants | Fresh terminal gap <0.10 |
|---|---:|---:|---:|---:|---:|
| 16 baseline | 80k / 80k | 12/50 | 39/50 | 1/12 | 39/50 |
| 16 epsilon | 80k / 80k | 12/50 | 39/50 | 1/12 | 39/50 |
| 16 batch 1024 | 20k / 20k | 7/50 | 44/50 | 1/7 | 45/50 |
| 32 baseline | 80k / 80k | 4/50 | 47/50 | 1/4 | 47/50 |
| 32 epsilon | 80k / 80k | 4/50 | 46/50 | 0/4 | 46/50 |
| 32 batch 1024 | 19k / 20k | 2/50 | 48/50 | 0/2 | 48/50 |

Every row retains all 50 seeds. The last row has administrative censoring for
all seeds; it is not a 20k result. The last confirmable acquisition onset is
78k for the 80k arms, 18k for batch-16, and 17k for batch-32.
“Acquired” means a qualifying three-point interval occurred; it does not require
the gap to remain below threshold forever. The terminal column is a separate
single-checkpoint evaluation. Full Wilson intervals are in ANALYSIS.md.

The width-16 baseline acquired in 39/50 seeds (78%;
Wilson 95% interval 64.8–87.2%). Twelve seeds entered a plateau; one escaped
by 80k (8.3%; interval 1.5–35.4%). The fresh terminal set also places 39/50
below 0.10. Width 32 acquired in 47/50 seeds (94%; Wilson interval 83.8–97.9%),
with four plateau entrants and one escape (25%; interval 4.6–69.9%). Both
baselines reached 80k updates.

The completed width-16 epsilon intervention has the same 39 acquiring seeds,
12 plateau entrants and one escape as baseline at 80k. Its mean fresh-test
gap difference is -0.00009 nats (paired seed-bootstrap 95% interval -0.00038
to +0.00015). This epsilon reduction did not rescue a seed at the primary
threshold. At width 32, epsilon acquired in 46/50 versus baseline's 47/50:
46 paired seeds acquired in both, one only with baseline, none only with
epsilon, and three in neither (exact McNemar p=1). Both arms have four plateau
entrants; baseline's one late escape is absent under epsilon. The fresh-test
gap difference is +0.00269 nats (paired seed-bootstrap 95% interval +0.00016
to +0.00722). Neither width shows a rescued seed from reducing epsilon;
this does not establish population equivalence or identical trajectories.

The completed width-16 batch-1024 arm acquired in 44/50 seeds by 20k updates,
versus 39/50 for baseline at equal episode exposure. Five paired seeds acquired
only with the larger batch and none only with baseline (unadjusted exact
McNemar p=0.0625). Its mean fresh-test gap difference is -0.00761 nats
(paired seed-bootstrap 95% interval -0.01761 to +0.00118). This is suggestive
of improvement, with substantial uncertainty. The advantage is clearer at equal
updates, where the larger batch has seen four times as many examples.

The primary event definitions use update intervals: a plateau spans 2.304
million examples at batch 256 but 9.216 million at batch 1024. A post-hoc
sensitivity evaluates both arms on their common exposure grid, keeping every
fourth baseline evaluation. Acquisition at 0.10 remains 39/50 versus 44/50;
plateau entry becomes 11/50 versus 7/50, with zero versus one escape. Thus
the primary acquisition contrast survives this change, while plateau and
escape counts depend on the observation window. Conditional escape comparisons
also involve different, treatment-dependent entrant sets.
The larger batch is slower to acquire for the early part of the cohort when
measured in examples: on the common grid, confirming acquisition in half the
seeds takes 5.120 million episodes for baseline and 8.192 million for batch
1024. Higher eventual incidence and early data efficiency are different outcomes.

At width 32, compare batch at 19k with baseline at 76k for the same 19.456
million episodes: acquisition is 48/50 versus 47/50, one batch-only seed and
none baseline-only (exact McNemar p=1). The matched-exposure-window sensitivity
gives the same counts. The paired validation-gap difference is -0.00292 nats
(95% bootstrap interval -0.00880 to +0.00112). The baseline terminal checkpoint
is at 80k, so no fresh-test difference at matched exposure is available.
The evidence does not establish a batch benefit at width 32.

Threshold sensitivity also limits the batch interpretation. At the tighter
0.05 threshold and equal exposure, width-16 acquisition is 34/50 baseline
versus 29/50 batch on the native update grids, and 32/50 versus 29/50 on the
common exposure grid. At width 32 it is 47/50 versus 46/50 under either grid.
Reported paired p-values are exploratory and unadjusted for multiple comparisons.

Late escapes remain rare: width-16 baseline and epsilon seed 106 first acquire
at 22k, confirmed at 24k; width-32 baseline seed 115 acquires at 33k, confirmed
at 35k; width-16 batch seed 131 acquires at 17k, confirmed at 19k.
These finite horizons do not prove permanent trapping, and conditional escape
rates have few entrants and wide intervals.

The early Adam diagnostics are inconsistent with epsilon-dominated denominators
in the measured weights. At 1k, 3k, 5k and 10k, every recorded fraction with
sqrt(vhat) <= epsilon is zero for first-layer Q/K weights and the input
source-ID columns. Across all six arms, the largest epsilon divided by a
per-seed coordinate p10 is 4.89245e-5 (0.00489245%), at width-32 batch, 10k,
Q/K seed 111. This p10 is 0.000204397, not the smallest coordinate.
It is a fixed-state denominator diagnostic, not a causal trajectory estimate;
the first 999 updates, unsampled times, biases and other parameter slices
remain unmeasured. Together with the paired outcomes, this gives no support
for simply transferring the small-epsilon remedy to this setting.

Early probes measure first-layer own-source attention, second-layer attention
to query readings, and the loss effect of disrupting labelled-source binding.
Validation gap is the simple comparator. Predictor directions were fixed before
results; primary analyses exclude seeds whose retrospectively identified
acquisition onset is at or before the landmark. AUC requires
at least five seeds in each outcome class. Per-head scans and acquirers-only
speed correlations are exploratory; neither establishes causality.

In the width-16 baseline, query-reading attention at 1k has AUC 0.907
(stratified seed-bootstrap 95% interval 0.814–0.972) for acquisition by 80k.
Own-source contrast at 5k has AUC 0.985 (0.948–1.000), among the 48 seeds
whose retrospective onset is after 5k. All 50 seeds were still unconfirmed at
5k: two excluded seeds only confirmed at 6k and 7k. A post-hoc sensitivity
using this operational risk set gives own-source AUC 0.986 (0.953–1.000),
query-reading AUC 0.960, binding-disruption AUC 0.830 and lower-gap AUC 0.802.
The 1k result is unchanged. The primary associations use prespecified metrics and
landmarks, but intervals are unadjusted for multiple comparisons and condition
on the fixed evaluation episodes. They are not a causal diagnosis or a
separately validated predictor.

![Confirmed acquisition by episode exposure](../work/astra-tiny-pfn-plateau-20261028/figures/final/acquisition_by_episodes.png)

The plot counts acquisition at confirmation and stops each arm at its observed
horizon. Larger-batch confirmation spans four times as many episodes; the
common-grid sensitivity above removes that difference.

## What the historical audit changes

The old 21/95 count pools 19 heterogeneous two-layer configurations, including
micro models, mixture heads, warm starts and different training horizons.
Fourteen plateau entrants are micro cases; their failures do not establish a
capacity limit. Seven entrants occur in configurations with successful sibling
seeds. The old one-point gap <0.14 rule counts five escapes; only one reaches
three consecutive gaps <0.10. That width-16 seed first crosses at 49k and is
confirmed at 51k, motivating the longer new horizon.

On the retained 16,000-episode test, exact query-only Bayes has gap 0.184982.
Four stalled mixture models beat it by 0.03861–0.04109 nats, paired episode SE
0.00215–0.00220. Two stalled Gaussian models beat it by about 0.023 nats, SE
about 0.0028. This is evidence for some nonquery-context use; it does not tell
us which field or computation matters.

## Locating the missing information

Let Q be the three query readings, S the three source-indexed labelled residual
sums of squares, U their unordered multiset, and T their total. These form nested
information sets: Q; (Q,T); (Q,U); (Q,S). We computed exact Bayes distributions
for each. U forgets which labelled group belongs to which query source; T also
forgets how the total divides among groups.

| Available information | Exact mixture gap | Moment Gaussian gap |
|---|---:|---:|
| Query readings Q | 0.184982 | 0.210571 |
| Query readings plus total residual T | 0.155333 | 0.173080 |
| Query readings plus unordered source residuals U | 0.125523 | 0.145922 |
| Full source-associated residuals S | 0 | 0.007887 |

These are sampled test NLL gaps. Detailed reports retain paired uncertainty.
The standard mutual-information chain rule gives

`I(Y;S|Q) = I(Y;T|Q) + I(Y;U|Q,T) + I(Y;S|Q,U)`.

The finite-test estimates are 0.029649 nats from total residuals, 0.029811 from
retaining their source-group partition without association, and 0.125523 from
recovering association. Thus source association supplies roughly two thirds
of the available context value in this world. This is an application of a
standard identity, not a new theorem.

The four stalled mixture models beat exact total-residual Bayes by
0.009–0.011 nats (SE about 0.0014), excluding that literal restricted predictor
class up to finite-test uncertainty. The two Gaussian models beat its optimal
moment Gaussian by about 0.011 nats. A conditional moment Gaussian minimizes
expected log loss among Gaussian predictors using those same inputs, so this
is the relevant head-matched comparison.

The unordered-reliability Gaussian is a closer descriptive reference: its
sampled NLL differs from the four stalled mixtures by only -0.00203 to +0.00045
nats, SE about 0.0023. However, the models remain 0.0184–0.0209 nats worse than
the exact unordered mixture. Target-integrated scores also distinguish them
from the unordered Gaussian. Matching a score does not identify a computation;
these findings allow imperfect unordered inference or partially successful
binding. Since the models have mixture heads, the remaining gap cannot simply
be attributed to a Gaussian-head floor.

The historical references were developed after inspecting the plateau data.
They are exploratory. Permutation invariance, an independent 162-state posterior
enumeration, Gamma special cases and a matrix-exponential density check support
the calculations. They do not identify the neural mechanism.

The fresh saved-checkpoint intervention fixes the cycle (1,2,0), seed 40000,
n=4096, model-selection rule and metrics before inference. Only labelled-source
IDs change; readings and labels remain fixed. Full Bayes is recomputed using
the inverse source assignment. The unordered reference is invariant to numerical
precision. A token-order equivalence check verifies that the neural intervention
and recomputed Bayes refer to the same changed input.

Distribution change is the half-sum of forward and reverse KL. The prespecified
subset requires Bayes change ≥0.3 nats, retaining 1,988 episodes. On that subset,
the ratio of mean model change to mean Bayes change is 8.7e-6–3.3e-5 for the six
stalled models and 0.802–0.986 for the five successful controls. The full-set
results agree qualitatively. Numerical integration was checked by grid and tail
expansion, analytic Gaussian KL and an independent adaptive-mixture calculation.

These are historically selected model phenotypes evaluated on fresh episodes,
not eleven random training replications. One control is the genuine late-escape
Gaussian seed; four are the lowest successful sibling seeds in the relevant
families. Sensitivity to this one 3-cycle neither proves invariance under all
source permutations nor identifies the unordered-reliability computation.
The cycled inputs remain on support but do not preserve the observational joint
distribution. In the unselected population, E[S_s | v_s] = 4v_s and
E[Q_s² | v_s] = 1 + v_s, giving Cov(S_s, Q_s²) = 4 Var(v_s) > 0 by conditional
independence. After the fixed cycle, the residual group attached to a query
source comes from an independent variance, giving covariance zero. These
identities need not hold after selecting the high-change subset. The test
therefore measures functional sensitivity under this distribution shift.
The intervention establishes a functional binding-use difference; it does not
establish that attention is the unique causal mediator or identify a training fix.

## Literature and implications

Seven primary papers were checked; the linked literature map gives exact
versions, sections and figures.

| Work | Relevant finding | Limit on transfer |
|---|---|---|
| Kirsch et al. (2022), 2212.04458 | Smaller Adam epsilon shortened plateaus; larger batches helped in updates but hurt in examples. | Relevant optimizer moments must be comparable; neither effect is guaranteed here. |
| Nguyen & Reddy (2024), 2412.00104 | Initialization-sensitive, long-tailed acquisition in a reduced classification model. | Its exponential escape equation and SGD dynamics do not predict this PFN's Adam times. |
| Reddy (2023), 2312.03002 | Removing an easy intermediate strategy made 4/6 reduced-model seeds fail. | Destroying the intermediate solution is not an established remedy. |
| Chan et al. (2022), 2205.05055 | Diversity, burstiness and changing label mappings influence ICL. | Does not establish a particular reliability curriculum for this world. |
| Raventós et al. (2023), 2306.15063 | Task-pool diversity changes regression behavior; tested batches can agree at equal examples. | A finite task-pool Bayes predictor still uses context; finite support and failure to bind fresh source IDs are distinct hypotheses. |
| Singh et al. (2023), 2311.08360 | Acquired ICL can disappear; some regularizers help persistence. | Their loss-added L2 coefficients are not portable to AdamW decay. |
| Zucchet et al. (2025), 2505.17863 | Repetition can shorten sparse-attention plateaus. | The fitted repetition variable is within-context copies, not minibatch size. |

## A next test, proposed rather than run

The most direct follow-up would test
whether brief source-matching supervision improves subsequent acquisition.
Keep the model, paired seeds, episode exposure and final likelihood objective
fixed; compare ordinary training, an early auxiliary loss on correct source
matches, and a shuffled-match control with comparable loss scale. Remove the
auxiliary loss before the main evaluation. Avoid prescribing uniform attention
within a source: earlier mechanism work shows that residual-dependent weights
can carry the required second moment. Measure sustained acquisition and the
same identity-only intervention on fresh episodes. Improvement would support
a routing-learning bottleneck; it would not establish a universal remedy or
explain every stalled seed. No follow-up training has been launched.

## Evidence and reproduction

- [Historical audit](tiny-pfn-plateau-historical-audit-2026-10-28.md).
- [Primary-source literature map](tiny-pfn-plateau-literature-2026-10-28.md).
- [Preregistration](../work/astra-tiny-pfn-plateau-20261028/PREREG.md).
- [Unordered-reliability reference](../work/astra-tiny-pfn-plateau-20261028/BINDING_REFERENCE.md).
- [Total-residual reference](../work/astra-tiny-pfn-plateau-20261028/POOLED_REFERENCE.md).
- [Fresh binding intervention](../work/astra-tiny-pfn-plateau-20261028/BINDING_INTERVENTION.md).
- [All six arms, endpoints and paired analyses](../work/astra-tiny-pfn-plateau-20261028/ANALYSIS.md).
- [Adam moment diagnostics](../work/astra-tiny-pfn-plateau-20261028/OPTIMIZER_SUMMARY.md).
- [Common exposure-window sensitivity](../work/astra-tiny-pfn-plateau-20261028/MATCHED_EXPOSURE.md).
- [Result hashes, recovered-checkpoint verification and live deletion check](../work/astra-tiny-pfn-plateau-20261028/RESULTS_VERIFICATION.json).
- [Learning curves](../work/astra-tiny-pfn-plateau-20261028/figures/final/learning_curves.pdf); [figure inputs and values](../work/astra-tiny-pfn-plateau-20261028/figures/final/plot_provenance.json).

Training source commit `718fb40d39babfdd8677387cfca817db2ee904ed`; batch amendment
`1f3307c17ceef58e7b64cfd928990161312dcb05`. CPU and GPU multi-step
forward/gradient/update conformance passed; maximum prediction difference was
below 8e-7. Key-projection bias is a softmax-invariant direction whose tiny
floating-point gradients produce different Adam updates; those differences
are recorded separately. The sampler's distribution, token layout and prefix
pairing were independently checked. Original tiny-pfn files are unchanged.

The final analysis retains all 300 seed-condition records. Independent scans
of the raw curves reproduced the plateau, acquisition and escape counts at all
three thresholds and the actual censoring horizons. Curves, metadata and
terminal checkpoint copies match the synced originals by SHA-256; gaps are
finite and seed identities/endpoints agree. The endpoint analyzer's 12 tests
passed. Raw JSON, lifecycle logs, derived tables and figure provenance are
retained under the owned experiment directory. Terminal weight files are
retained locally and hashed; they are excluded from Git.

The batch launcher exited 124 at its fixed administrative timeout and synced
before deletion. Width-32 batch had no final metadata, so a one-thread CPU
evaluation loaded the saved 19k weights without training. It reproduced the
last validation-gap vector with maximum absolute difference 5.96e-8, then ran
the same fresh seed-30000 evaluation used by the completed arms. Recovery
metadata records this check and hashes the weights, curve, prospective decision
and recovery script. Any later unsaved updates are unobserved.

The three DELETE records estimate $1.65, $2.25 and $2.19, totaling **$6.09**.
These are the launcher's unverified $0.87/hour list-price ceiling times VM age,
not an invoice; the Billing API is disabled. At 13:47:55 UTC, the recorded
read-only command
`gcloud compute instances list --project=GCP_PROJECT --filter='name~soup-gpu-astra-pfn28' --format=json`
returned `[]`, corroborating all three deletion records. An initial width-16
provisioning attempt failed from regional stockouts before training and is
retained as an infrastructure failure, not a model outcome.

From the experiment directory, reproduce derived outputs without training:

```sh
python3 analyze.py results --json-out analysis.json --markdown-out ANALYSIS.md
python3 summarize_optimizer.py results --json-out optimizer-summary.json --markdown-out OPTIMIZER_SUMMARY.md
python3 matched_exposure_sensitivity.py results --json-out matched-exposure.json --markdown-out MATCHED_EXPOSURE.md
/home/agent/.venvs/aggext/bin/python -I -B plot.py results --out-dir figures/final
```
