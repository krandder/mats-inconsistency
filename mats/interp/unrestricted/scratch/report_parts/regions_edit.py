p = '/home/agent/workspaces/mats/outbox/unrestricted-estimators-2026-10-31.md'
s = open(p).read()
old = """  random biases (abstract verified), not this bias-free spherical version. At L = alpha n it is exp(O(n log n)).
  MEASURED region counts at n = 3: PENDING."""
new = """  random biases (abstract verified), not this bias-free spherical version. At L = alpha n it is exp(O(n log n)).
  MEASURED (registered M4, `scratch/regions.py`): healthy width-3 nets show only 14-32 distinct gate patterns on 2^22
  Fibonacci-sphere directions, with medians 22, 23, 18, 15 at depth 4, 8, 16, 32: no growth with depth (P13, slope
  in [1, 3], refuted). Narrow critical nets that stay alive collapse onto few gate histories, so the count saturates
  far below the hypothesised bound; these are lower bounds and describe the healthy subpopulation (3-60% of seeds)."""
assert old in s, 'marker not found'
s = s.replace(old, new)
open(p, 'w').write(s)
print('ok')
