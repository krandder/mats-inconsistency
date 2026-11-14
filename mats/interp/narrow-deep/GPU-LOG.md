# GPU log (work/narrow-deep)

Cap $5 (brief). On-demand L4 via work/run_on_l4.sh (VM auto-deletes; ceiling $0.87/h).

| run id | launched (UTC) | type | purpose | timeout | deleted / verified | cost estimate |
|---|---|---|---|---|---|---|
| nd-train-1 | 2026-10-29T21:58:45Z | on-demand L4 (GPU_PROVISIONING=STANDARD) via gpu/launch_gpu.sh | RNN 2nd-argmax T=64/32 width 16; sorting MLPs width 16 (PREREG Part 4) | 2.5 h | never created: L4 stockout in us-central1-a/b/c (CREATE-FAILED ~22:05Z) | $0 |
| nd-train-2 | 2026-10-29T21:59:51Z | on-demand L4, GPU_ZONES asia-southeast1-b/c, europe-west4-a/b/c, europe-west1-b/c, us-east1-c/d, us-east4-a/c, us-west1-a/b, us-west4-a | same as nd-train-1 | 2.5 h | created as soup-gpu-nd-train-2-1029215956-25353 in asia-southeast1-b ~22:00Z after us-central1 stockouts; ran to RUN-GPU-DONE (rc 0); MLP L=128 job stopped by hand at 22:14Z; deleted 23:30Z by gpu-run.sh ("Deleted ..." in gpu-run.out); instance list checked 23:31Z: 0 nd-train VMs | ~$1.31 (1.5 h x $0.87/h ceiling) |

Total GPU cost: about $1.31 of the $5 cap (nd-train-1 never created a VM).
