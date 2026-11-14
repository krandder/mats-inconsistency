# linear-agg GPU ledger (budget group linear-agg, cap $10)

Rebuilt 2026-10-16T14:13Z by gpu/ledger.py from logical-soup/GPU-LOG.md. Rates are the
UNVERIFIED on-demand list-price ceiling ($0.87/h for g2-standard-8 + 1 L4; Cloud Billing API disabled). Spot VMs
(provisioning SPOT) are normally 60-70% cheaper; STANDARD is on-demand. True spend is at most this total.

**Total (ceiling): $1.40** over 4 VMs.

| VM | run | zone | provisioning | created | deleted | cost USD | basis |
|---|---|---|---|---|---|---|---|
| soup-gpu-la-a-1016125104-32073 | la-a | us-central1-a | SPOT | 2026-10-16T12:51Z | 2026-10-16T13:40Z | 0.70 | DELETE line, age 0.8 h x $0.87/h ceiling |
| soup-gpu-la-b-1016125124-3108 | la-b | us-east1-b | SPOT | 2026-10-16T12:52Z | 2026-10-16T13:21Z | 0.41 | DELETE line, age 0.47 h x $0.87/h ceiling |
| soup-gpu-la-a2-1016131857-19513 | la-a2 | us-central1-a | SPOT | 2026-10-16T13:19Z | 2026-10-16T13:26Z | 0.01 | DELETE line, age 0.0 h x $0.87/h ceiling |
| soup-gpu-la-a3-1016132648-7580 | la-a3 | us-east4-c | STANDARD | 2026-10-16T13:27Z | 2026-10-16T13:48Z | 0.28 | DELETE line, age 0.32 h x $0.87/h ceiling |
| soup-gpu-la-a4-1016134931-24167 | la-a4 |  |  |  |  | 0.00 | create failed (stockout), no instance |

Live linear-agg VMs at rebuild time (gcloud, name~soup-gpu-la-): none
