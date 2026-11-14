# PREREG-LA amendments and errata (each with its time; PREREG-LA.md itself is never edited after hashing)

- 2026-10-16 12:49Z, erratum (no content change): PREREG-LA.md's header says "Written ~13:25Z"; that time is wrong. The
  file was saved at 12:48:03Z (stat mtime) and hashed at 12:48:10Z (sha256 9abce49f...534ef4, PREREG-LA.sha256), before
  any training or evaluation run. A copy was kept at /tmp/linear-agg/prereg-copy/PREREG-LA-prelaunch.md.

- 2026-10-16 12:56Z, infrastructure (no design change): on the A VM the first six Diag-L6 and GD++-L6 jobs crashed within
  5 s, before training: torch 2.14 routes a backward batched outer product to a Triton kernel whose launcher needs Python.h,
  absent from the VM image. python3.12-dev was installed on the running VM (12:56Z) and the six jobs re-queued unchanged.
- 2026-10-16 13:07Z, amendment A-1 (training recipe, Experiment A; made before any test-set evaluation; only training and
  validation logs had been read): **no restarts.** A batch with a non-finite loss is skipped (as before), a validation
  evaluation whose MSE is non-finite or not the best never becomes the selected checkpoint, and the model is the best
  finite-validation checkpoint (the pre-registered selection rule). Dropped: the "validation MSE > 10 or non-finite ->
  restart at lr / 3" and "50 skipped batches -> restart" rules. Reason: all three Full-L6 runs hit the "validation MSE > 10"
  trigger at every lr tried (3e-4, 1e-4, 3.3e-5), each time around step 2,000, while the training loss stayed near 0.9: a
  few validation episodes make the degree-3^L polynomial explode (validation MSE 9e5 and 1.6e9), so lowering lr did not
  help and the rule only cut training short (best validation MSE 0.92 after 3 truncated runs, vs 0.84 for a 1,000-step
  CPU timing run).
- 2026-10-16 13:07Z, amendment A-2 (Experiment A): training in **float32** instead of float64 (evaluation stays float64,
  models loaded in float64). Reason: float64 on the L4 ran at 46-87 ms per step with 6 concurrent jobs, about 1.5-3 h for
  the 36 models; float32 is several times faster. The pre-registration's 1,000-step timing runs were float32.
- 2026-10-16 13:07Z, consequence: every A job that had started under the old recipe (Full-L6 x 3, Full-L4 x 3, Diag-L4 x 3;
  three of them had finished) was killed or discarded and re-queued, so all 36 models share the amended recipe; their old
  outputs are overwritten by the re-runs and are not used.
- 2026-10-16 13:09Z, amendment A-3 (Experiment B aggregator training, before any B training or evaluation): the same
  change as A-1: no restarts (dropped "validation log loss > 10 or non-finite -> restart at lr / 3" and "50 skipped batches
  -> restart"); non-finite batches skipped; an epoch whose validation log loss is non-finite or not the best is never
  selected; early stopping (patience 30 epochs) unchanged. Reason: the same exploding-polynomial failure seen in A.
- 2026-10-16 13:49Z, amendment A-4 (Experiment A; after reading training and validation logs only, before any test
  evaluation): **reinstate the pre-registered "more than 50 skipped (non-finite) batches -> restart at lr / 3, at most 2
  restarts" rule**, which A-1 removed together with the "validation MSE > 10" trigger. A-1 meant to remove only the
  latter. Why it matters: in float32 a finite but huge loss can overflow the gradient, clipping then turns the parameters
  into NaN, and every later batch is skipped. Diag-L6 seeds 1 and 2 never had a finite validation checkpoint (19,254 and
  19,640 of 20,000 batches skipped); Diag-L6 seed 0 and Full-L6 seeds 0 and 2 skipped 13,110, 18,463 and 13,730 batches
  after early checkpoints. The rule binds only in runs with more than 50 skipped batches: those five runs are retrained
  (same seeds, on-demand L4, run la-a4); the other 31 runs skipped at most 3 batches, so the rule would not have changed
  them and they are kept. Their superseded logs are kept in runs/a-superseded/.
- 2026-10-16 13:52Z, infrastructure: the on-demand L4 for the five A-4 retrains (run la-a4) could not be created (stockout
  in all 11 zones tried, 13:49-13:51Z; no VM existed, no cost). The five retrains run on this box's CPU instead (float32,
  1 thread each, 4 at a time, same seeds and recipe). torch's CPU and CUDA generators give different random streams, so
  these five models saw different (equally distributed) training episodes than a GPU run would have.
