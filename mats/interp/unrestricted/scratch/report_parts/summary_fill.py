p = '/home/agent/workspaces/mats/outbox/unrestricted-estimators-2026-10-31.md'
s = open(p).read()
rep = {
"| P4. n in {16, 32}: >= 1000x worse than RQMC at the largest budget reached, certificate nMSE >= 1 throughout, all 18 nets | PENDING |":
"| P4. n in {16, 32}: >= 1000x worse than RQMC at the largest budget reached, certificate nMSE >= 1 throughout, all 18 nets | **Hit**: 18 of 18 (10^5-10^6 worse; certificate nMSE 6e5 to 2e34). |",
"""- Against radius-exact scrambled-Sobol RQMC at equal FLOPs: [width 3]; [width 4]; [widths 6-8]; [widths 16-32]. The
  certificate is [result].""":
"""- Against radius-exact scrambled-Sobol RQMC at equal FLOPs: at width 3 it reaches parity at 1e6-4e6 passes (both
  near nMSE 1e-12) and wins 4-18x in 4 of 24 runs, all at depth 16-32, wins that shrink below 3x in most cases if
  its cost were doubled; at width 4 it loses 10^3-10^4x; at widths 6-8, 10^4-10^6x; at widths 16-32 it stays at the
  accuracy of 2-30 Monte Carlo samples after 4e6 passes, 10^5-10^6x behind. The certificate is tight only at width 3
  (nMSE 5e-13 to 6e-8) and is vacuous from width 6 on, growing to 1e34 at depth 64.""",
"""- Cost: [passes to a fixed precision against width and depth]. A cell costs 46-170 forward passes. Nothing in this
  family can reach 10M+ parameters.""":
"""- Cost: the passes it needs for nMSE 1e-4 grow about 10x per unit of width (1e4-4e4 at width 3, 2e6-8e6 at width 6,
  out of reach at 8 and above), against 130-2,000 for RQMC, and a cell costs 46-170 forward passes. Nothing in this
  family can reach 10M+ parameters.""",
"""- Unscrambled Sobol points do [about as well as] RQMC, but they are sampling at fixed points, and the gain of any""":
"""- Centred unscrambled Sobol points did 2.7-4.5x better than scrambled Sobol at widths 3-8 (geometric means), but
  they are sampling at fixed points, and the gain of any""",
"""   precision. ARC's mean is the additive, infinite-temperature end, which sampling already reaches.""":
"""   precision. ARC's mean is the additive, infinite-temperature end, which sampling already reaches.""",
}
for a, b in rep.items():
    assert a in s, a[:60]
    s = s.replace(a, b)
s = s.replace("DRAFT; numbers in [brackets] are filled after the queue.\n\n", "")
open(p, 'w').write(s)
print('ok')
