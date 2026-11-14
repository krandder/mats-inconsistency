# R7-E3 GPU ledger (budget group r7, group cap $6.50 shared with R7-E1 and R7-E2; own ceiling $1.50)

Rebuilt 2026-10-17T10:01Z by r7/gpu/ledger.py from logical-soup/GPU-LOG.md. Rates are
the UNVERIFIED on-demand list-price ceiling ($0.87/h for g2-standard-8 + 1 L4; Cloud Billing API disabled). Spot VMs
(provisioning SPOT) are normally 60-70% cheaper; STANDARD is on-demand. True spend is at most this total.

**Total (ceiling): $0.27** of the $1.50 R7-E3 ceiling, over 1 VMs.

| VM | run | zone | provisioning | created | deleted | cost USD | basis |
|---|---|---|---|---|---|---|---|
| soup-gpu-r7e3-main-1017094026-8653 | r7e3-main | us-central1-a | STANDARD | 2026-10-17T09:40Z | 2026-10-17T09:59Z | 0.27 | DELETE line, age 0.31 h x $0.87/h ceiling |

Live r7e3 VMs at rebuild time (gcloud instances list, name~soup-gpu-r7e3-): none

Left-over r7e3 disks (gcloud disks list, name~soup-gpu-r7e3-): none
