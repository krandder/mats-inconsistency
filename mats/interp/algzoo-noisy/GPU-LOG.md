# GPU log (work/algzoo-noisy)

| run id | launched (UTC) | type | purpose | timeout | deleted / verified | cost estimate |
|---|---|---|---|---|---|---|
| azn-wide-1 | 2026-10-29T01:12:46Z | on-demand L4 (GPU_PROVISIONING=STANDARD) via work/run_on_l4.sh | from-scratch noisy RNNs: widths 16/32/64 | 3 h | never created: L4 stockout in us-central1-a/b/c (CREATE-FAILED 01:13Z) | $0 |
| azn-wide-2 | 2026-10-29T01:16:39Z | on-demand L4 (GPU_PROVISIONING=STANDARD), GPU_ZONES us-east1/us-east4/us-west1/us-west4 | same as azn-wide-1 | 3 h | never created: L4 stockout in 9 US zones (CREATE-FAILED 01:18Z) | $0 |
| azn-wide-3 | 2026-10-29T01:18:50Z | on-demand L4 (GPU_PROVISIONING=STANDARD); created as soup-gpu-azn-wide-3-1029011852-25379 in asia-southeast1-b at about 01:22Z after stockouts in europe-west4-a/b/c and europe-west1-b/c | same as azn-wide-1 | 3 h | ran 01:22-03:46Z (rc 0); deleted 03:47Z by gpu-run.sh; instance list checked 03:48Z: no azn VM remains (the three soup-gpu-arcn-* VMs listed are another run) | $2.12 (2.44 h x $0.87/h ceiling) |

Total GPU cost: about $2.12 of the $5 cap. Two earlier launches (azn-wide-1, azn-wide-2) never created a VM (L4 stockouts).
- 2026-10-29T05:01Z: final re-check, `gcloud compute instances list` shows no azn-* or algzoo VM (0 matches). Total GPU spend stays $2.12.
