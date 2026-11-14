p = '/home/agent/workspaces/mats/outbox/unrestricted-estimators-2026-10-31.md'
s = open(p).read()
rep = {
"| MEASURED for our B&B: PENDING. | A polynomial cell count":
"| MEASURED for our B&B: passes to nMSE 1e-4 grow about 10x per unit of width from 3 to 6 and exceed 8.4e6 from width 8 (section 2.3). | A polynomial cell count",
"| Extra hypothesis (sampled pattern counts are lower bounds only). MEASURED at n = 3: PENDING. |":
"| Extra hypothesis (sampled pattern counts are lower bounds only). MEASURED at n = 3: 14-32 patterns, flat in depth 4-32, far below the bound (healthy nets only). |",
}
for a, b in rep.items():
    assert a in s, a[:50]
    s = s.replace(a, b)
# the process section had a stale "Deviations. PENDING" (the deviations list is in section 6)
s = s.replace("**Deviations.** PENDING.", "**Deviations.** Listed with the scorecard in section 6.")
open(p, 'w').write(s)
print('ok')
