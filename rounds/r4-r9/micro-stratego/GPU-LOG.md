# R4-E1 micro-Stratego GPU ledger (budget group r4-e1, cap $15)

Rebuilt 2026-10-16T20:00Z by gpu/ledger.py from logical-soup/GPU-LOG.md. Rates are the
UNVERIFIED on-demand list-price ceiling ($0.87/h for g2-standard-8 + 1 L4; Cloud Billing API disabled). Spot VMs
(provisioning SPOT) are normally 60-70% cheaper; STANDARD is on-demand. True spend is at most this total.

**Total (ceiling): $1.68** over 7 VMs.

| VM | run | zone | provisioning | created | deleted | cost USD | basis |
|---|---|---|---|---|---|---|---|
| soup-gpu-e1-s101-1016185459-566 | e1-s101 | us-east4-c | SPOT | 2026-10-16T18:55Z | 2026-10-16T19:29Z | 0.47 | DELETE line, age 0.54 h x $0.87/h ceiling |
| soup-gpu-e1-s102-1016190113-31126 | e1-s102 | us-central1-a | SPOT | 2026-10-16T19:01Z | 2026-10-16T19:30Z | 0.41 | DELETE line, age 0.47 h x $0.87/h ceiling |
| soup-gpu-e1-s103-1016190137-16686 | e1-s103 | us-east1-b | SPOT | 2026-10-16T19:02Z | 2026-10-16T19:31Z | 0.41 | DELETE line, age 0.47 h x $0.87/h ceiling |
| soup-gpu-e1-x102-1016193834-5172 | e1-x102 | us-east1-b | SPOT | 2026-10-16T19:39Z | 2026-10-16T19:45Z | 0.09 | DELETE line, age 0.1 h x $0.87/h ceiling |
| soup-gpu-e1-x103-1016193938-8741 | e1-x103 | us-central1-a | SPOT | 2026-10-16T19:39Z | 2026-10-16T19:48Z | 0.10 | DELETE line, age 0.12 h x $0.87/h ceiling |
| soup-gpu-e1-x101-1016193958-22825 | e1-x101 | us-central1-a | SPOT | 2026-10-16T19:40Z | 2026-10-16T19:47Z | 0.10 | DELETE line, age 0.11 h x $0.87/h ceiling |
| soup-gpu-e1-y103-1016195216-7823 | e1-y103 | us-east1-b | SPOT | 2026-10-16T19:52Z | 2026-10-16T20:00Z | 0.10 | DELETE line, age 0.11 h x $0.87/h ceiling |
| soup-gpu-e1-y1-1016194802-26320 | e1-y1 |  |  |  | 2026-10-16T19:50Z | 0.00 | create failed (stockout), no instance |
| soup-gpu-e1-y101-1016195015-20127 | e1-y101 |  |  |  | 2026-10-16T19:52Z | 0.00 | create failed (stockout), no instance |

Live r4-e1 VMs at rebuild time (gcloud, name~soup-gpu-e1-): none
