# R6-E3 GPU ledger (budget group r6, group cap $12 shared with R6-E1; own ceiling $1.80)

Rebuilt 2026-10-17T06:55Z by r6/gpu/ledger.py from logical-soup/GPU-LOG.md. Rates are
the UNVERIFIED on-demand list-price ceiling ($0.87/h for g2-standard-8 + 1 L4; Cloud Billing API disabled). Spot VMs
(provisioning SPOT) are normally 60-70% cheaper; STANDARD is on-demand. True spend is at most this total.

**Total (ceiling): $0.26** of the $1.80 R6-E3 ceiling, over 1 VMs.

| VM | run | zone | provisioning | created | deleted | cost USD | basis |
|---|---|---|---|---|---|---|---|
| soup-gpu-r6e3-main-1017063551-4510 | r6e3-main | us-central1-a | STANDARD | 2026-10-17T06:37Z | 2026-10-17T06:54Z | 0.26 | DELETE line, age 0.3 h x $0.87/h ceiling |

Live r6e3 VMs at rebuild time (gcloud instances list, name~soup-gpu-r6e3-): none

Left-over r6e3 disks (gcloud disks list, name~soup-gpu-r6e3-): none
