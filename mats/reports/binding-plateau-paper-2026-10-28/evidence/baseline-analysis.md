# Tiny-PFN plateau study analysis

Checked at 2026-10-28T12:25:24.235908+00:00 from curve and metadata files only.

Primary acquisition and escape require three consecutive validation gaps below 0.10. Plateau entry requires ten consecutive 1,000-update evaluations in [0.14, 0.18]. For the first-point event-time estimand, non-events are right-censored at the latest evaluation that begins a fully observed three-point confirmation window.

## Outcomes

| width | condition | last eval / event-time censor / requested | plateau | acquisition | escape among plateau | weak <.14 escape | admin censored |
|---:|---|---:|---:|---:|---:|---:|---:|
| 16 | baseline | 80000/78000/80000 | 12/50 (24.0%; 14.3-37.4%) | 39/50 (78.0%; 64.8-87.2%) | 1/12 (8.3%; 1.5-35.4%) | 2/12 (16.7%; 4.7-44.8%) | 0 |
| 32 | baseline | 80000/78000/80000 | 4/50 (8.0%; 3.2-18.8%) | 47/50 (94.0%; 83.8-97.9%) | 1/4 (25.0%; 4.6-69.9%) | 1/4 (25.0%; 4.6-69.9%) | 0 |

## Threshold sensitivity

| width | condition | threshold | acquired | escaped among plateau | terminal fresh set below threshold |
|---:|---|---:|---:|---:|---:|
| 16 | baseline | 0.10 | 39/50 (78.0%; 64.8-87.2%) | 1/12 (8.3%; 1.5-35.4%) | 39/50 (78.0%; 64.8-87.2%) |
| 16 | baseline | 0.12 | 39/50 (78.0%; 64.8-87.2%) | 1/12 (8.3%; 1.5-35.4%) | 40/50 (80.0%; 67.0-88.8%) |
| 16 | baseline | 0.05 | 34/50 (68.0%; 54.2-79.2%) | 0/12 (0.0%; 0.0-24.2%) | 27/50 (54.0%; 40.4-67.0%) |
| 32 | baseline | 0.10 | 47/50 (94.0%; 83.8-97.9%) | 1/4 (25.0%; 4.6-69.9%) | 47/50 (94.0%; 83.8-97.9%) |
| 32 | baseline | 0.12 | 47/50 (94.0%; 83.8-97.9%) | 1/4 (25.0%; 4.6-69.9%) | 47/50 (94.0%; 83.8-97.9%) |
| 32 | baseline | 0.05 | 47/50 (94.0%; 83.8-97.9%) | 1/4 (25.0%; 4.6-69.9%) | 47/50 (94.0%; 83.8-97.9%) |

## Episode-matched timing semantics

The event step is the first point of the three-point sequence; the detection step is the third. An event is counted only when its detection step is observed by the comparison horizon. Episode matching uses the largest episode exposure with an evaluation observed in both runs. Because evaluation is every 1,000 updates, event-to-detection confirmation spans 2,000 updates; the number of examples in that interval therefore depends on batch size.

| width | pair (batch) | common episode exposure | matched steps | event-to-detection examples |
|---:|---|---:|---:|---:|

## Paired acquisition discordance

| width | pair | match | threshold | both | first only | second only | neither | exact McNemar p |
|---:|---|---|---:|---:|---:|---:|---:|---:|

## Descriptive paired gap differences

Values are the second condition's validation gap minus the first condition's gap; negative values favor the second condition. Intervals are paired seed-bootstrap 95% intervals. These continuous comparisons are exploratory supplements to the preregistered event endpoints.

| width | pair | match | horizons | n | mean gap difference (95% interval) |
|---:|---|---|---:|---:|---:|

## Early predictors

AUC is reported only when at least five landmark-eligible seeds later acquire and five do not acquire by the observed horizon. Directions were fixed before results: higher own-source contrast, binding-disruption NLL and query-reading mass, and lower validation gap predict acquisition. Intervals are stratified seed-bootstrap 95% intervals. Per-head/source component scans remain exploratory in JSON.

| width | condition | landmark | eligible (yes/no) | metric | fixed direction | AUC (bootstrap 95%) |
|---:|---|---:|---:|---|---|---:|
| 16 | baseline | 0 | 50 (39/11) | `own_source_contrast` | higher | 0.559 (0.340-0.767) |
| 16 | baseline | 0 | 50 (39/11) | `binding_disruption_nll` | higher | 0.788 (0.653-0.897) |
| 16 | baseline | 0 | 50 (39/11) | `query_reading_mass` | higher | 0.774 (0.585-0.928) |
| 16 | baseline | 0 | 50 (39/11) | `gap` | lower | 0.501 (0.303-0.713) |
| 16 | baseline | 1000 | 50 (39/11) | `own_source_contrast` | higher | 0.408 (0.231-0.604) |
| 16 | baseline | 1000 | 50 (39/11) | `binding_disruption_nll` | higher | 0.459 (0.284-0.639) |
| 16 | baseline | 1000 | 50 (39/11) | `query_reading_mass` | higher | 0.907 (0.814-0.972) |
| 16 | baseline | 1000 | 50 (39/11) | `gap` | lower | 0.781 (0.634-0.897) |
| 16 | baseline | 3000 | 50 (39/11) | `own_source_contrast` | higher | 0.851 (0.732-0.946) |
| 16 | baseline | 3000 | 50 (39/11) | `binding_disruption_nll` | higher | 0.646 (0.494-0.788) |
| 16 | baseline | 3000 | 50 (39/11) | `query_reading_mass` | higher | 0.958 (0.883-1.000) |
| 16 | baseline | 3000 | 50 (39/11) | `gap` | lower | 0.718 (0.555-0.862) |
| 16 | baseline | 5000 | 48 (37/11) | `own_source_contrast` | higher | 0.985 (0.948-1.000) |
| 16 | baseline | 5000 | 48 (37/11) | `binding_disruption_nll` | higher | 0.821 (0.690-0.926) |
| 16 | baseline | 5000 | 48 (37/11) | `query_reading_mass` | higher | 0.958 (0.877-1.000) |
| 16 | baseline | 5000 | 48 (37/11) | `gap` | lower | 0.791 (0.656-0.907) |
| 16 | baseline | 10000 | 28 (17/11) | `own_source_contrast` | higher | 0.989 (0.952-1.000) |
| 16 | baseline | 10000 | 28 (17/11) | `binding_disruption_nll` | higher | 0.952 (0.850-1.000) |
| 16 | baseline | 10000 | 28 (17/11) | `query_reading_mass` | higher | 0.957 (0.856-1.000) |
| 16 | baseline | 10000 | 28 (17/11) | `gap` | lower | 0.941 (0.834-1.000) |
| 32 | baseline | 0 | 50 (47/3) | insufficient outcome classes |  |  |
| 32 | baseline | 1000 | 50 (47/3) | insufficient outcome classes |  |  |
| 32 | baseline | 3000 | 44 (41/3) | insufficient outcome classes |  |  |
| 32 | baseline | 5000 | 16 (13/3) | insufficient outcome classes |  |  |
| 32 | baseline | 10000 | 6 (3/3) | insufficient outcome classes |  |  |

## Exploratory acquisition-speed associations

Acquirers-only, selection-biased when censored. Spearman rho compares each metric with the first-point event step among landmark-eligible seeds that eventually acquire. These descriptive associations make no causal claim.

| width | condition | landmark | eventual acquirers | metric | Spearman rho | status |
|---:|---|---:|---:|---|---:|---|
| 16 | baseline | 0 | 39 | `own_source_contrast` | -0.120 | ok |
| 16 | baseline | 0 | 39 | `binding_disruption_nll` | -0.236 | ok |
| 16 | baseline | 0 | 39 | `query_reading_mass` | -0.071 | ok |
| 16 | baseline | 0 | 39 | `gap` | -0.040 | ok |
| 16 | baseline | 1000 | 39 | `own_source_contrast` | -0.182 | ok |
| 16 | baseline | 1000 | 39 | `binding_disruption_nll` | -0.069 | ok |
| 16 | baseline | 1000 | 39 | `query_reading_mass` | -0.295 | ok |
| 16 | baseline | 1000 | 39 | `gap` | -0.040 | ok |
| 16 | baseline | 3000 | 39 | `own_source_contrast` | -0.518 | ok |
| 16 | baseline | 3000 | 39 | `binding_disruption_nll` | -0.320 | ok |
| 16 | baseline | 3000 | 39 | `query_reading_mass` | -0.433 | ok |
| 16 | baseline | 3000 | 39 | `gap` | 0.254 | ok |
| 16 | baseline | 5000 | 37 | `own_source_contrast` | -0.838 | ok |
| 16 | baseline | 5000 | 37 | `binding_disruption_nll` | -0.722 | ok |
| 16 | baseline | 5000 | 37 | `query_reading_mass` | -0.423 | ok |
| 16 | baseline | 5000 | 37 | `gap` | 0.706 | ok |
| 16 | baseline | 10000 | 17 | `own_source_contrast` | -0.662 | ok |
| 16 | baseline | 10000 | 17 | `binding_disruption_nll` | -0.761 | ok |
| 16 | baseline | 10000 | 17 | `query_reading_mass` | -0.417 | ok |
| 16 | baseline | 10000 | 17 | `gap` | 0.758 | ok |
| 32 | baseline | 0 | 47 | `own_source_contrast` | -0.172 | ok |
| 32 | baseline | 0 | 47 | `binding_disruption_nll` | 0.026 | ok |
| 32 | baseline | 0 | 47 | `query_reading_mass` | -0.294 | ok |
| 32 | baseline | 0 | 47 | `gap` | 0.007 | ok |
| 32 | baseline | 1000 | 47 | `own_source_contrast` | -0.263 | ok |
| 32 | baseline | 1000 | 47 | `binding_disruption_nll` | -0.218 | ok |
| 32 | baseline | 1000 | 47 | `query_reading_mass` | -0.413 | ok |
| 32 | baseline | 1000 | 47 | `gap` | 0.283 | ok |
| 32 | baseline | 3000 | 41 | `own_source_contrast` | -0.752 | ok |
| 32 | baseline | 3000 | 41 | `binding_disruption_nll` | -0.664 | ok |
| 32 | baseline | 3000 | 41 | `query_reading_mass` | -0.319 | ok |
| 32 | baseline | 3000 | 41 | `gap` | 0.608 | ok |
| 32 | baseline | 5000 | 13 | `own_source_contrast` | -0.657 | ok |
| 32 | baseline | 5000 | 13 | `binding_disruption_nll` | -0.404 | ok |
| 32 | baseline | 5000 | 13 | `query_reading_mass` | 0.104 | ok |
| 32 | baseline | 5000 | 13 | `gap` | 0.519 | ok |
| 32 | baseline | 10000 | 3 | `own_source_contrast` | missing | insufficient_eventual_acquirers |
| 32 | baseline | 10000 | 3 | `binding_disruption_nll` | missing | insufficient_eventual_acquirers |
| 32 | baseline | 10000 | 3 | `query_reading_mass` | missing | insufficient_eventual_acquirers |
| 32 | baseline | 10000 | 3 | `gap` | missing | insufficient_eventual_acquirers |

Full event times, censoring, quantiles, predictor components and paired sensitivity analyses are in the JSON output.
