p = '/home/agent/workspaces/mats/outbox/unrestricted-estimators-2026-10-31.md'
s = open(p).read()
rep = {
"| P1. n = 3, L in {4, 8}: B&B beats RQMC by >= 3x at some budget in [2^16, 2^22] on >= 4 of 6 nets | PENDING |":
"| P1. n = 3, L in {4, 8}: B&B beats RQMC by >= 3x at some budget in [2^16, 2^22] on >= 4 of 6 nets | **Refuted**: 0 of 6 (parity at best at 1e6 passes; the larger apparent wins at 4e6 passes are below the truth's resolution). Resolved wins appear only at depth 16-32 (4 of 24 runs, 3.8-18x), outside the registered range. |",
"| P2. n = 4: >= 10x worse than RQMC at every budget in [2^12, 2^22] on >= 9 of 12 nets | PENDING |":
"| P2. n = 4: >= 10x worse than RQMC at every budget in [2^12, 2^22] on >= 9 of 12 nets | **Hit**: 12 of 12 (10^3-10^4 worse). |",
"| P3. n in {6, 8}: >= 100x worse than RQMC at every budget in [2^14, 2^22] on all 30 nets | PENDING |":
"| P3. n in {6, 8}: >= 100x worse than RQMC at every budget in [2^14, 2^22] on all 30 nets | **Hit**: 30 of 30 (10^4-10^6 worse). |",
"| P5. certificate nMSE <= 1e-6 on >= 4 of 6 nets at n = 3, L <= 8; on none at n >= 6 | PENDING |":
"| P5. certificate nMSE <= 1e-6 on >= 4 of 6 nets at n = 3, L <= 8; on none at n >= 6 | **Hit**: 6 of 6 at width 3 (certified nMSE 5e-11 to 9e-11), none of 30 at widths 6-8. |",
"| P6. median B&B nMSE at 2^20 increases with n (3, 4, 6, 8) at L = 8 and 16, by >= 100x | PENDING |":
"| P6. median B&B nMSE at 2^20 increases with n (3, 4, 6, 8) at L = 8 and 16, by >= 100x | **Hit**: 6e-11, 2e-6, 9e-4, 6e-3 (L = 8); 3e-10, 1e-6, 4e-4, 4e-3 (L = 16). |",
"| P7. geometric mean of QMC-det / RQMC nMSE in [1/3, 3] at N = 2^10, 2^14, 2^18 (its stated rationale was wrong [A]; scored as an empirical prediction) | PENDING |":
"| P7. geometric mean of QMC-det / RQMC nMSE in [1/3, 3] at N = 2^10, 2^14, 2^18 (its stated rationale was wrong [A]; scored as an empirical prediction) | **Refuted**: 0.30, 0.22, 0.37; the centred deterministic Sobol set was better than scrambled Sobol by 2.7-4.5x. |",
"| P8. n in {6, 8}: B&B loses to RB-MC at every budget on >= 2/3 of nets | PENDING |":
"| P8. n in {6, 8}: B&B loses to RB-MC at every budget on >= 2/3 of nets | **Hit**: 30 of 30. |",
"| P11. f1 + f2 in [0.3, 0.95] on every net, and the median 2-design ceiling changes by < 2x from n = 16 to n = 256 (L = n/4) | PENDING |":
"| P11. f1 + f2 in [0.3, 0.95] on every net, and the median 2-design ceiling changes by < 2x from n = 16 to n = 256 (L = n/4) | **Refuted** on the second clause: the first holds (0.30-0.93), but the median ceiling fell from 7.0 to 2.2, a factor 3.2. Designs have even less room at width 256 than predicted. |",
"| P13. region-count log-log slope in L at n = 3 in [1, 3] | PENDING |":
"| P13. region-count log-log slope in L at n = 3 in [1, 3] | **Refuted**: the counts do not grow (medians 22, 23, 18, 15 at depth 4-32); healthy narrow nets collapse. |",
}
for a, b in rep.items():
    assert a in s, a[:50]
    s = s.replace(a, b)
open(p, 'w').write(s)
print('ok')
