# E-SUD spend ledger (budget group overnight-sud, hard cap $50)

The authoritative per-VM lines (CREATE / DELETE with cost_usd = rate ceiling x age) are in the shared ledger
`/home/agent/workspaces/eurisko-study/logical-soup/GPU-LOG.md` (grep `esud`). Check the group's total:
`bash -c 'source /home/agent/workspaces/eurisko-study/logical-soup/infra/lib.sh; spend_usd overnight-sud'`.
Rate: $0.87/h ceiling for g2-standard-8 + 1x L4 (on-demand list price; these are SPOT VMs, which bill less; the
Billing API is disabled, so real spend is unverified and lower). Every VM: spot, max-run 6h, DELETE on termination.

| time (UTC) | event | VM | note |
|---|---|---|---|
| 00:54 | create | soup-gpu-esud-a-1015005430-29922 | smoke (4x4, 2 min + 10k steps) then 6x6 seed 0; reserve $4.00 |
| 01:11 | CREATE-FAILED | soup-gpu-esud-b-1015011059-19026 | spot L4 stockout in us-central1-a/b/c; no VM, $0 (ledger keeps its $3.13 reservation) |
| 01:12 | CREATE-FAILED | soup-gpu-esud-c-1015011138-15967 | same; seeds 1 and 2 moved onto esud-a |
| 02:04 | create | soup-gpu-esud-d-1015020350-23015 | round 1b seed 0; preempted by GCE within ~5 min (DELETE on termination), never used; < $0.10 |
| 02:05 | CREATE-FAILED | soup-gpu-esud-e-* | spot and on-demand stockout |
| 02:07 | create | soup-gpu-esud-f-1015020709-24003 | round 1b seeds 1, 2 (150k steps); ON-DEMAND (spot stocked out; gpu-up fallback); reserve $4.00 |
| 02:15 | (reuse) | esud-a | round 1b seed 0 runs on esud-a next to round 1's evals |

Smoke price (01:15Z): 4x4 smoke trained 10k steps in 3.6 min and evaluated 600 boards x 3 rules x 16 chains in
about 1 min on one L4. 6x6 seed 0 trains at about 26 s per 1000 steps on a shared GPU (30k steps about 13 min). Plan:
3 VMs x about 1.2 h = about $3.2 at the ceiling; with margin < $8. Cap $50.
| 04:08 | DELETE | soup-gpu-esud-f-1015020709-24003 | age 2.00 h, $1.74 at ceiling |
| 04:15 | DELETE | soup-gpu-esud-a-1015005430-29922 | age 3.33 h, $2.90 at ceiling |

Final (04:15Z): ceiling spend about $4.74 (esud-a $2.90 + esud-f $1.74 + esud-d ~$0.10). `spend_usd overnight-sud`
shows $26.90 because it also counts $22.26 of reservations from 3 failed creates and the preempted esud-d, which
were never billed. Check at 04:15Z: `gcloud compute instances list --filter=name~esud` and `gcloud compute disks list
--filter=name~esud` both empty.
