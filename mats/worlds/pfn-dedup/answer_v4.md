## Answer (v4, 2026-10-27: prior-level vs in-context, the split Kelvin asked for)

Two different things lower a model's error compared with counting every reading:
- **prior-level discounting**: shrinkage, and down-weighting extra rows, learned from the training distribution. A
  readout can do this without comparing tokens within the episode.
- **in-context aggregation**: using this episode's evidence that two rows are one source, or share an error.

The cleanest test adds one row to an episode in one of two forms: a true copy of a source, or an independent new
source with the same query reading (table A). A reader without within-episode comparison treats the two identically.

- **Depth 1 without keys is prior-level only.** A copy pulls the prediction 0.049, an independent source 0.054: it
  barely tells them apart (9%). On shared-error and singleton episodes its error sits at the level of R2, Bayes on
  the query values alone with no identities or labels (0.029 vs 0.030; 0.027 vs 0.023–0.029). Prediction confirmed.
- **Depths 2–8 without keys deduplicate in context, about halfway.** A copy pulls 0.046–0.048, an independent source
  0.087–0.093, so the models discount about half of a copy beyond the prior (44–49%). They use value equality across
  the labelled questions, not the source identity vector: a copy with a new identity behaves exactly like a copy.
  Depth beyond 2 adds nothing.
- **With keys, in-context deduplication is strong at every depth** (69% at depth 1, 80–83% at depths 2–8). The copy
  shares its source's group key, which is direct in-context evidence.
- **Shared errors are mostly prior-level.** Of each model's MSE gain over count-all on shared-error episodes, 60–79%
  is reproduced by R1, the best additive pool with no comparisons, and 75–99% by R2. What is left in context:
  - without keys: depth 1 0.01, depths 2–8 0.11–0.14 beyond R2;
  - with keys: depth 1 0.19, depths 2–8 0.24–0.25 beyond R2.
- **No model deduplicates fully.** Bayes ignores a copy (pull 0); the best models still give it a pull of 0.026.
- **Attention is necessarily the carrier,** since MLPs act per token. The in-context part rides on the group keys or
  on comparing labelled values across rows. At depth 1 without keys, readout attention does only prior-level work.

Caveat for table A: the independent-source variant gives an episode 7 distinct sources, which never occurs in
training. The duplicate-excess version (table C) is confounded: R1, being additive, cannot express softmax
normalisation, which is itself a prior-level discount. Table C credits depth 1 with 0.68 "in-context" on duplicates;
table A shows that this is prior-level row discounting. Table A is the test to trust.

<details><summary>Superseded headlines (v3, 2026-10-27 06:30Z; before the prior-level/in-context split)</summary>

