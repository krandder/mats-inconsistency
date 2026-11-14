# GPU-LOG (work/kstar-frames)

Approved by the coordinator 2026-10-31 ~00:15Z: one on-demand L4 via work/run_on_l4.sh, hard cap $3, delete after
the runs and verify in the instance list. g2-standard-8 + 1x L4, list-price ceiling $0.87/h; GPU_RUN_TIMEOUT 9000 s
(worst case ~(0.25 h setup + 2.5 h) x 0.87 = $2.4).

| run id | launched (UTC) | purpose | deleted / verified | cost |
|---|---|---|---|---|
launch 2026-10-31T00:16:34Z rid=kstar1
| kstar1 | 2026-10-31T00:16Z (VM soup-gpu-kstar1-1031001637-6566, us-central1-a, READY 00:18Z) | runs A-E at n = 128, 64. Run E (states) finished for all 6 nets and came back; runs A-D were stopped at 00:36Z because batched cusolver SVD of the masked Jacobians was ~170x slower than eigh (7.9 s vs 46 ms per batch of 64, measured on the VM) | deleted 00:37Z by gpu-run.sh (rc 143 after the stop); `gcloud compute instances list --filter=name~kstar` at 00:38Z: 0 items | $0.28 (0.32 h x $0.87/h ceiling) |
launch 2026-10-31T00:38:12Z rid=kstar2 (eigh on CUDA; runs A-D at n = 128, 64)
| kstar2 | 2026-10-31T00:38Z (VM soup-gpu-kstar2-1031003818-28592, us-central1-b after a us-central1-a stockout; READY 00:40Z) | runs A-D at n = 128, 64 with eigh on CUDA. All width-64 jobs finished; at n = 128, every run-B analysis and the L = 16 pipelines of nets 401, 402 failed with CUDA out-of-memory (batched cusolver eigh workspace ~10 GB, three jobs sharing the GPU) | deleted 01:33Z by gpu-run.sh (rc 0); instance list at 01:34Z: 0 items | $0.79 (0.91 h x $0.87/h ceiling) |
launch 2026-10-31T01:34:48Z rid=kstar3 (sequential reruns: n=128 run-B analyses with chunked eigh, n=128 L=16 nets 401/402, run E at n=32)
| kstar3 | 2026-10-31T01:34Z | sequential reruns | never created: L4 stockout in us-central1-a/b/c | $0 |
launch 2026-10-31T01:36:04Z rid=kstar4 (same as kstar3, more zones)
| kstar4 | 2026-10-31T01:36Z (VM soup-gpu-kstar4-1031013608-22681, us-central1-a, READY 01:38Z) | the kstar2 jobs that had failed, one at a time (n = 128 run-B analyses with chunked eigh; n = 128, L = 16 nets 401 and 402; run E at n = 32). All 21 jobs rc 0; regenerated run-B data bit-identical to kstar2's | deleted 02:19Z by gpu-run.sh (rc 0); instance list at 02:20Z: 0 items | $0.62 (0.71 h x $0.87/h ceiling) |

launch 2026-10-31T02:46:00Z rid=kstar5 (float64 covering recompute at n = 64, 128 after the GPT-6 review; B data regenerated, deterministic)
| kstar5 | 2026-10-31T02:46Z (VM soup-gpu-kstar5-1031024604-1471, us-central1-a) | float64 covering recompute at n = 64, 128 after the GPT-6 review (run-B data regenerated, bit-identical); 24 jobs rc 0 | deleted 03:14Z by gpu-run.sh (rc 0); instance list at 03:14Z: 0 items | $0.40 (0.46 h x $0.87/h ceiling) |

Total: $2.09 of the $3 cap (kstar1 0.28 + kstar2 0.79 + kstar3 0, never created + kstar4 0.62 + kstar5 0.40), list-price ceilings. Instance list empty at 03:14Z.
