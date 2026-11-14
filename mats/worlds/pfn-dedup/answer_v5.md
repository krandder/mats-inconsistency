## What survives (2026-10-28, after review 2)

- **Every trained model deduplicates partly from within-episode evidence. None does it fully.** In distribution it
  removes 0.25–0.70 of an exact copy; Bayes removes all of it. A reader with no within-episode comparison removes
  0.01 (section 0, table A).
- **Without keys, the cue is mostly the exact tie of the copy's query reading.** Removing the tie halves the effect;
  identity vectors and labelled-value equality barely matter (table B). Depth 1 does this as well as or better than
  depths 2–8 (0.36 vs 0.25–0.28), so depth adds nothing here.
- **With keys, the cue is the shared group key, and depth helps.** A copy with a new key is barely discounted
  (0.02–0.12). Depth 1 removes 0.49 of a copy, depths 2–8 0.62–0.70.
- **On shared errors, deeper models with keys remove 85–88% of the error left by a reader that compares query values
  only (R2); without keys, 36–44%; depth 1 without keys, 3%.** Most of the improvement over count-all is plain
  shrinkage (60–79%), which any reader learns, so that normalisation understates the in-context part (table C).
- **Withdrawn:** the v4 claims "depth 1 without keys is prior-level only (prediction confirmed)", "depths 2–8 dedup
  about halfway by labelled value equality", "69% at depth 1 with keys" and "shared errors mostly prior-level". The
  v4 copy-vs-independent test kept an exact query tie in the "independent" row, and a tie is copy evidence. The
  preregistered prediction "depth 1 ≈ prior level" **failed**: depth 1 beats R1 on every split and removes 0.36 of a
  copy (R1: 0.01).

## Answer (v5, 2026-10-28)

The question was whether the deduplication is prior-level (shrinkage and row discounting learned from the training
distribution) or in-context (this episode's evidence that two rows are one source, or share an error). Measured
in distribution:

| | depth 1 | depths 2–8 |
|---|---|---|
| copy removed in context, no keys | 0.36 [0.24, 0.47] | 0.25–0.28 |
| copy removed in context, keys | 0.49 [0.45, 0.52] | 0.62–0.70 |
| main cue, no keys | exact query tie | exact query tie |
| main cue, keys | group key | group key |
| R2's shared-error loss removed, no keys | 0.03 | 0.36–0.44 |
| R2's shared-error loss removed, keys | 0.60 | 0.85–0.88 |

- **Without keys, depth helps shared errors but not copies.** Depth 1 handles shared errors only at the level of a
  query-value comparison (R2). Depths 2–8 use labelled residuals for another 36–44%. On copies, depth 1 does as well
  as or better than depths 2–8, all relying on the exact query tie.
- **Keys make both problems easier, and depth then helps both.**
- **Where it happens.** MLPs act per token, but the query's MLP acts on the attention-pooled vector. So depth 1 is a
  "pool then nonlinearity" reader, and it can detect a tie among pooled readings. Saying that depth 1 does only
  prior-level work in its readout was wrong.
- **Convergence:** 15/24 runs converged by the prereg rule; depth 2 and depth 8 without keys, and depth 8 with keys,
  are 1/3 each. Converged and unconverged seeds of the same arm agree (seed ranges in table A).
- **Statistics:** CIs are from a bootstrap over episodes with the seeds fixed; seed ranges are shown separately. The
  three-seed spread is small next to the gaps between arms that the headlines rely on.

