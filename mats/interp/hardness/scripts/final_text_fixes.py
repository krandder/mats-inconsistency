"""Final wording fixes after the numerics were filled in (K = 2 advantage shrinks slowly with width)."""
base = '/home/agent/workspaces/mats/work/hardness/'
p = base + 'draft.md'
d = open(p).read()
reps = [
("""- At fixed width, K = 2's error grows roughly like depth^2 at shallow depth (ARC's observation), then levels off: from
  depth 32-64 on, 1/gain stays between about 0.2 and 1.3 at every width tested, up to depth 512. At fixed depth it
  improves with width, as ARC proved for fixed depth.
- Along L = alpha n (alpha = 1/8, 1/4, 1/2), K = 2 at its own (single, low) budget stays within a factor of about 1-3 of
  plain Monte Carlo up to n = 1024. Its advantage does not grow with n (fitted slopes of nMSE against n are -0.3 to
  -0.9, against -1 for Monte Carlo at equal cost); at alpha = 1/8 it falls from 8 at n = 32 toward the level of the
  other ratios.""",
"""- At fixed width, K = 2's loss ratio 1/gain grows about 4-5x per doubling of depth at shallow depth (error growing like
  depth^2, ARC's observation) and only about 1.2-2x per doubling beyond depth 64 (0.6-0.9 at depth 64-128, 1.3-2.3 at
  depth 256-512 for widths 512-1024). At fixed depth it improves with width, as ARC proved for fixed depth.
- Along L = alpha n (alpha = 1/8, 1/4, 1/2), K = 2 at its own (single, low) budget stays within a factor of about 3 of
  plain Monte Carlo, either way, up to n = 1024, but its advantage shrinks slowly with width: geometric means of 2-8 at
  widths 32-128 fall to 0.65-1.1 at width 1024 (4 seeds; intervals include 1). Fitted slopes of log nMSE against log n
  are -0.4 to -0.8, shallower than Monte Carlo's -1 at equal cost, so on these data K = 2 loses ground slowly."""),
("""stays bounded as n grows, and for K >= 3 tends to 0. Evidence: 6.3 (K = 2 holds at about 1-3 without growth;
factorized K = 3 falls with width at alpha = 1/8 and loses 25-250x at alpha = 1/4); the mechanisms of 6.6-6.11 and 7.2.""",
"""stays bounded as n grows, and in fact tends to 0 (fast for factorized K = 3, slowly if at all for K = 2). Evidence: 6.3
(K = 2's advantage falls from 2-8 to about 0.7-1.1 between widths 32 and 1024; factorized K = 3 falls with width at
alpha = 1/8 and loses 27-247x at alpha = 1/4); the mechanisms of 6.6-6.11 and 7.2."""),
("""   budget it stays within a factor of about 1-3 of plain Monte Carlo at every depth and width tested, with no trend
   toward a growing advantage; the higher-budget points of the hierarchy are where the hierarchy falls behind (6.3).""",
"""   budget it stays within a factor of about 3 of plain Monte Carlo, either way, at every depth and width tested, but
   its advantage shrinks slowly with width; the higher-budget point of the hierarchy (factorized K = 3) falls behind
   fast (6.3)."""),
]
for a, b in reps:
    assert a in d, 'MISSING draft: ' + a[:70]
    d = d.replace(a, b)
open(p, 'w').write(d)

h = open(base + 'notes/head.txt').read()
hreps = [
("""- New measurement: ARC's cheapest closure (K = 2) stays within a factor of about 1-3 of plain Monte Carlo, at its own
  budget, along depth = width/8, /4 and /2 up to width 1024; its advantage does not grow with width.""",
"""- New measurement: ARC's cheapest closure (K = 2) stays within a factor of about 3 of plain Monte Carlo, either way, at
  its own budget, along depth = width/8, /4 and /2 up to width 1024, but its advantage shrinks slowly with width (from
  2-8 at widths 32-128 to about 0.7-1.1 at width 1024)."""),
("""K = 2 at its own budget gains a factor of about 1-3 over plain Monte Carlo at equal cost for depth/width 1/8 to 1/2 and widths 64-1024 (8 at width 32), with no growth in width; its error grows like depth^2 at shallow depth and then levels off (tables in 6.3).""",
"""K = 2's gain over plain Monte Carlo at equal cost, for depth/width 1/8 to 1/2, falls from 2-8 at widths 32-128 to about 0.7-1.1 at width 1024 (fitted slopes -0.4 to -0.8 against Monte Carlo's -1); its error grows like depth^2 at shallow depth and much more slowly beyond depth 64 (tables in 6.3)."""),
("""- Empirically the cheapest closure already roughly matches plain Monte Carlo at its own low budget along L proportional
  to n.""",
"""- Empirically the cheapest closure roughly matches plain Monte Carlo at its own low budget along L proportional to n
  (within about 3x either way up to width 1024), while losing ground slowly as width grows."""),
]
for a, b in hreps:
    assert a in h, 'MISSING head: ' + a[:70]
    h = h.replace(a, b)
open(base + 'notes/head.txt', 'w').write(h)
print('ok')
