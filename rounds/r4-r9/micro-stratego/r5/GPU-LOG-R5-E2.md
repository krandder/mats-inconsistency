# R5-E2 GPU ledger (budget group r5, group cap $22 shared with R5-E1; own ceiling $9)

Rebuilt 2026-10-17T05:57Z by r5/gpu/ledger.py from logical-soup/GPU-LOG.md. Rates are
the UNVERIFIED on-demand list-price ceiling ($0.87/h for g2-standard-8 + 1 L4; Cloud Billing API disabled). Spot VMs
(provisioning SPOT) are normally 60-70% cheaper; STANDARD is on-demand. True spend is at most this total.

**Total (ceiling): $2.73** of the $9.00 R5-E2 ceiling, over 5 VMs.

| VM | run | zone | provisioning | created | deleted | cost USD | basis |
|---|---|---|---|---|---|---|---|
| soup-gpu-r5e2-s999-smoke-1017032612-20773 | r5e2-s999-smoke | us-central1-a | SPOT | 2026-10-17T03:26Z | 2026-10-17T03:34Z | 0.10 | DELETE line, age 0.12 h x $0.87/h ceiling |
| soup-gpu-r5e2-s101-1017033621-16895 | r5e2-s101 | us-central1-a | SPOT | 2026-10-17T03:36Z | 2026-10-17T04:30Z | 0.77 | DELETE line, age 0.88 h x $0.87/h ceiling |
| soup-gpu-r5e2-s102-1017033621-14520 | r5e2-s102 | us-central1-a | SPOT | 2026-10-17T03:37Z | 2026-10-17T04:30Z | 0.77 | DELETE line, age 0.88 h x $0.87/h ceiling |
| soup-gpu-r5e2-s103-1017033621-9474 | r5e2-s103 | us-central1-a | SPOT | 2026-10-17T03:37Z | GCE preemption | 0.37 | COST-AUDIT line (spot preempted by GCE, auto-deleted), age 0.42 h x $0.87/h ceiling |
| soup-gpu-r5e2-s103-b-1017041104-24544 | r5e2-s103-b | us-central1-a | STANDARD | 2026-10-17T04:11Z | 2026-10-17T05:02Z | 0.72 | DELETE line, age 0.83 h x $0.87/h ceiling |

Live r5e2 VMs at rebuild time (gcloud instances list, name~soup-gpu-r5e2-): none

Left-over r5e2 disks (gcloud disks list, name~soup-gpu-r5e2-): none
