# R8-E3 GPU ledger (budget group r8, group cap $2.20 shared with R8-E1; own ceiling $0.50)

Rebuilt 2026-10-17T16:36Z by r8/gpu/ledger.py from logical-soup/GPU-LOG.md. Rates are
the UNVERIFIED on-demand list-price ceiling ($0.87/h for g2-standard-8 + 1 L4; Cloud Billing API disabled). Spot VMs
(provisioning SPOT) are normally 60-70% cheaper; STANDARD is on-demand. True spend is at most this total.

**Total (ceiling): $0.46** of the $0.50 R8-E3 ceiling, over 3 VMs.

| VM | run | zone | provisioning | created | deleted | cost USD | basis |
|---|---|---|---|---|---|---|---|
| soup-gpu-r8e3-main-1017155844-4062 | r8e3-main | us-central1-a | STANDARD | 2026-10-17T15:59Z | 2026-10-17T16:12Z | 0.18 | DELETE line, age 0.21 h x $0.87/h ceiling |
| soup-gpu-r8e3-es-1017161527-19250 | r8e3-es | us-central1-a | STANDARD | 2026-10-17T16:16Z | 2026-10-17T16:25Z | 0.13 | DELETE line, age 0.15 h x $0.87/h ceiling |
| soup-gpu-r8e3-es2-1017162557-241 | r8e3-es2 | us-central1-a | STANDARD | 2026-10-17T16:26Z | 2026-10-17T16:36Z | 0.15 | DELETE line, age 0.17 h x $0.87/h ceiling |

Live r8e3 VMs at rebuild time (gcloud instances list, name~soup-gpu-r8e3-): none

Left-over r8e3 disks (gcloud disks list, name~soup-gpu-r8e3-): none
