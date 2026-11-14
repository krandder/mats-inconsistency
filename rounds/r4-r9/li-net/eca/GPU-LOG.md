# E-CA GPU ledger (budget group overnight-ca, cap $110)

Rebuilt 2026-10-15T11:59Z by gpu/ledger.py from logical-soup/GPU-LOG.md. Rates are the
UNVERIFIED on-demand list-price ceiling ($0.87/h for g2-standard-8 + 1 L4; Cloud Billing API disabled), so true spend
is at most this. Create attempts that failed on stockout created no instance and cost nothing.

**Total (ceiling): $33.56** over 31 VMs.

| VM | run | zone | provisioning | created | deleted | cost USD | basis |
|---|---|---|---|---|---|---|---|
| soup-gpu-eca-smoke-1015010900-6656 | eca-smoke | us-central1-a | STANDARD | 2026-10-15T01:09Z | 2026-10-15T01:35Z | 0.37 | DELETE line, age 0.42 h x $0.87/h ceiling |
| soup-gpu-eca-r110-s0-1015013505-109 | eca-r110-s0 | us-central1-c | STANDARD | 2026-10-15T01:35Z | 2026-10-15T02:42Z | 0.95 | DELETE line, age 1.09 h x $0.87/h ceiling |
| soup-gpu-eca-r90-s0-1015013514-18759 | eca-r90-s0 | us-central1-b | STANDARD | 2026-10-15T01:39Z | 2026-10-15T02:50Z | 1.01 | DELETE line, age 1.16 h x $0.87/h ceiling |
| soup-gpu-eca-r90-s1-1015013517-748 | eca-r90-s1 | us-central1-c | STANDARD | 2026-10-15T01:40Z | 2026-10-15T02:50Z | 0.99 | DELETE line, age 1.14 h x $0.87/h ceiling |
| soup-gpu-eca-r90-s2-1015013520-8425 | eca-r90-s2 | us-central1-a | STANDARD | 2026-10-15T01:42Z | 2026-10-15T02:49Z | 0.98 | DELETE line, age 1.13 h x $0.87/h ceiling |
| soup-gpu-eca-t8-s0-1015013650-13120 | eca-t8-s0 | us-central1-a | STANDARD | 2026-10-15T01:43Z | 2026-10-15T03:21Z | 1.42 | DELETE line, age 1.63 h x $0.87/h ceiling |
| soup-gpu-eca-t8-s1-1015013653-5825 | eca-t8-s1 | us-central1-a | STANDARD | 2026-10-15T01:43Z | 2026-10-15T03:16Z | 1.34 | DELETE line, age 1.54 h x $0.87/h ceiling |
| soup-gpu-eca-t8-s2-1015013656-28168 | eca-t8-s2 | us-central1-a | STANDARD | 2026-10-15T01:44Z | 2026-10-15T03:36Z | 1.63 | DELETE line, age 1.87 h x $0.87/h ceiling |
| soup-gpu-eca-r110-s1-1015014038-19283 | eca-r110-s1 | us-central1-a | STANDARD | 2026-10-15T01:44Z | 2026-10-15T02:54Z | 1.00 | DELETE line, age 1.15 h x $0.87/h ceiling |
| soup-gpu-eca-r110-s2-1015014121-1832 | eca-r110-s2 | us-central1-a | STANDARD | 2026-10-15T01:44Z | 2026-10-15T02:53Z | 0.98 | DELETE line, age 1.13 h x $0.87/h ceiling |
| soup-gpu-eca-b-r110-s0-1015023338-30583 | eca-b-r110-s0 | us-central1-a | STANDARD | 2026-10-15T02:33Z | 2026-10-15T03:49Z | 1.10 | DELETE line, age 1.26 h x $0.87/h ceiling |
| soup-gpu-eca-b-r110-s1-1015023340-17723 | eca-b-r110-s1 | us-central1-a | STANDARD | 2026-10-15T02:37Z | 2026-10-15T03:42Z | 0.97 | DELETE line, age 1.12 h x $0.87/h ceiling |
| soup-gpu-eca-b-r110-s2-1015023342-4193 | eca-b-r110-s2 | us-central1-b | STANDARD | 2026-10-15T02:37Z | 2026-10-15T04:08Z | 1.30 | DELETE line, age 1.5 h x $0.87/h ceiling |
| soup-gpu-eca-b-r90-s0-1015023344-12498 | eca-b-r90-s0 | us-central1-a | STANDARD | 2026-10-15T02:38Z | 2026-10-15T03:54Z | 1.10 | DELETE line, age 1.26 h x $0.87/h ceiling |
| soup-gpu-eca-b-r90-s1-1015023346-29072 | eca-b-r90-s1 | us-central1-a | STANDARD | 2026-10-15T02:38Z | 2026-10-15T03:48Z | 1.01 | DELETE line, age 1.16 h x $0.87/h ceiling |
| soup-gpu-eca-b-r90-s2-1015023348-45 | eca-b-r90-s2 | us-central1-a | STANDARD | 2026-10-15T02:38Z | 2026-10-15T03:39Z | 0.87 | DELETE line, age 1.0 h x $0.87/h ceiling |
| soup-gpu-eca-r2a-r110-1015043013-24309 | eca-r2a-r110 | us-central1-b | STANDARD | 2026-10-15T04:30Z | 2026-10-15T05:06Z | 0.51 | DELETE line, age 0.59 h x $0.87/h ceiling |
| soup-gpu-eca-r2a-r90-1015043015-9713 | eca-r2a-r90 | us-west1-a | STANDARD | 2026-10-15T04:32Z | 2026-10-15T04:59Z | 0.37 | DELETE line, age 0.42 h x $0.87/h ceiling |
| soup-gpu-eca-r2b-s0-1015043017-13341 | eca-r2b-s0 | us-central1-a | STANDARD | 2026-10-15T04:33Z | 2026-10-15T05:19Z | 0.67 | DELETE line, age 0.77 h x $0.87/h ceiling |
| soup-gpu-eca-r2b-s1-1015043019-9397 | eca-r2b-s1 | us-central1-a | STANDARD | 2026-10-15T04:36Z | 2026-10-15T05:24Z | 0.73 | DELETE line, age 0.84 h x $0.87/h ceiling |
| soup-gpu-eca-r2b-s2-1015043021-10618 | eca-r2b-s2 | us-central1-b | STANDARD | 2026-10-15T04:36Z | 2026-10-15T05:19Z | 0.60 | DELETE line, age 0.69 h x $0.87/h ceiling |
| soup-gpu-eca3-smoke-1015094703-908 | eca3-smoke | us-central1-b | STANDARD | 2026-10-15T09:47Z | 2026-10-15T10:31Z | 0.63 | DELETE line, age 0.72 h x $0.87/h ceiling |
| soup-gpu-eca3-b-s0-1015100519-14864 | eca3-b-s0 | us-central1-a | STANDARD | 2026-10-15T10:08Z | 2026-10-15T11:34Z | 1.24 | DELETE line, age 1.43 h x $0.87/h ceiling |
| soup-gpu-eca3-c-s0-1015100521-14787 | eca3-c-s0 | us-central1-a | STANDARD | 2026-10-15T10:09Z | 2026-10-15T11:46Z | 1.40 | DELETE line, age 1.61 h x $0.87/h ceiling |
| soup-gpu-eca3-a-s1-1015100523-15929 | eca3-a-s1 | us-central1-a | STANDARD | 2026-10-15T10:09Z | 2026-10-15T11:55Z | 1.52 | DELETE line, age 1.75 h x $0.87/h ceiling |
| soup-gpu-eca3-b-s1-1015100525-10004 | eca3-b-s1 | us-central1-a | STANDARD | 2026-10-15T10:10Z | 2026-10-15T11:55Z | 1.51 | DELETE line, age 1.74 h x $0.87/h ceiling |
| soup-gpu-eca3-c-s1-1015100527-24673 | eca3-c-s1 | us-central1-a | STANDARD | 2026-10-15T10:10Z | 2026-10-15T11:51Z | 1.45 | DELETE line, age 1.67 h x $0.87/h ceiling |
| soup-gpu-eca3-a-s2-1015100529-16202 | eca3-a-s2 | us-central1-a | STANDARD | 2026-10-15T10:11Z | 2026-10-15T11:57Z | 1.53 | DELETE line, age 1.76 h x $0.87/h ceiling |
| soup-gpu-eca3-b-s2-1015100531-13011 | eca3-b-s2 | us-central1-b | STANDARD | 2026-10-15T10:12Z | 2026-10-15T11:58Z | 1.53 | DELETE line, age 1.76 h x $0.87/h ceiling |
| soup-gpu-eca3-c-s2-1015100533-16931 | eca3-c-s2 | us-central1-b | STANDARD | 2026-10-15T10:13Z | 2026-10-15T11:47Z | 1.35 | DELETE line, age 1.55 h x $0.87/h ceiling |
| soup-gpu-eca3-a-s0-1015101007-2717 | eca3-a-s0 | us-central1-a | STANDARD | 2026-10-15T10:14Z | 2026-10-15T11:57Z | 1.50 | DELETE line, age 1.72 h x $0.87/h ceiling |
| soup-gpu-eca-r110-s1-1015013508-10751 | eca-r110-s1 |  |  |  | 2026-10-15T04:09Z | 0.00 | create failed (stockout), no instance |
| soup-gpu-eca-r110-s2-1015013511-29803 | eca-r110-s2 |  |  |  | 2026-10-15T04:09Z | 0.00 | create failed (stockout), no instance |
| soup-gpu-eca3-a-s0-1015100517-10835 | eca3-a-s0 |  |  |  |  | 0.00 | create failed (stockout), no instance |

Live eca VMs at rebuild time (gcloud): none

MATS cluster: not used (opus-mats 01:0xZ: 8/8 L40s allocated, next free slot 11-19); zero jobs submitted.
