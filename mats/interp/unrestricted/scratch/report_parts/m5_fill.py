p = '/home/agent/workspaces/mats/outbox/unrestricted-estimators-2026-10-31.md'
s = open(p).read()
a = "| M5. noise effect at sigma = 1e-3 (descriptive) | PENDING |"
b = ("| M5. noise effect at sigma = 1e-3 (descriptive) | nMSE of (noisy mean - noiseless mean), paired MC with 2^20 inputs: "
     "widths 6-8: 6e-12 to 7e-7 at depth 4-8, 7e-8 to 1e-4 at depth 16-64 (one net with tiny variance: 3e-3); widths 16-32: "
     "below 4e-8 at every depth. So at widths 6-8 and depth >= 16 any noiseless deterministic estimate is biased at the "
     "level RQMC reaches with 10^4-10^6 passes; at widths 16-32 and depth <= 32 the noise does not matter at any budget "
     "we ran. |")
assert a in s
s = s.replace(a, b)
open(p, 'w').write(s)
print('ok')
