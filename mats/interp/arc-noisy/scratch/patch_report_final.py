"""One-off: final H2 numbers (regenerated with dev nets 100-107 complete) into the outbox report."""
p = '/home/agent/workspaces/mats/outbox/arc-noisy-deep-2026-10-29.md'
s = open(p).read()
def rep(a, b):
    global s
    if a not in s:
        raise SystemExit('MISSING: ' + a[:90])
    s = s.replace(a, b)
rep("and the better of the two entries by 1.04x to 1.75x; 0.99x at 3e11 |",
    "and the better of the two entries by 1.04x to 1.76x; 0.99x at 3e11 |")
rep("nothing at sigma = 0.001 or 0.003: H2 at 0.84-1.06x of the Kerdock sampler with no significant gain;",
    "nothing at sigma = 0.001 or 0.003: H2 at 0.84-1.06x of the Kerdock sampler (its 1-2% point gains at 6.8e10-1e11 pass the median CI test but not the mean one);")
rep("""| 1e10 | K=2 augmented | 1.75 [1.70, 1.82] | 1.75 [1.70, 1.82] |""", """| 1e10 | K=2 augmented | 1.76 [1.71, 1.82] | 1.76 [1.71, 1.82] |""")
rep("""| 2e10 | factorized K=3 | 3.79 [3.39, 4.04] | 1.09 [1.07, 1.09] |""", """| 2e10 | factorized K=3 | 3.78 [3.39, 4.03] | 1.09 [1.07, 1.09] |""")
rep("""| 6.8e10 | factorized K=3 | 1.52 [1.41, 1.58] | 1.52 [1.41, 1.58] |""", """| 6.8e10 | factorized K=3 | 1.52 [1.40, 1.58] | 1.52 [1.40, 1.58] |""")
rep("""| 1e11 | factorized K=3 | 1.39 [1.33, 1.47] | 1.39 [1.33, 1.47] |""", """| 1e11 | factorized K=3 | 1.39 [1.32, 1.47] | 1.39 [1.32, 1.47] |""")
rep("""| 2e11 | factorized K=3 augmented | 1.24 [1.20, 1.37] | 1.24 [1.20, 1.37] |""", """| 2e11 | factorized K=3 | 1.28 [1.24, 1.30] | 1.28 [1.25, 1.29] |""")
rep("""| 9.89 [9.11, 10.69] |""", """| 9.89 [9.11, 10.71] |""")
rep("""| 3e11 | factorized K=3 | 0.99 [0.93, 1.00] | 0.99 [0.95, 1.00] |""", """| 3e11 | factorized K=3 | 0.99 [0.93, 1.00] | 0.99 [0.94, 1.00] |""")
rep("""(2e9 and 5e10: 5.09x / 1.15x and 2.20x / 1.58x.) At sigma = 0.003: 1.18-9.6x over the sampler and 1.04-1.79x over the better
of the two, 0.98x at 3e11; sigma = 0.01 is the same.""", """(2e9 and 5e10: 5.09x / 1.15x and 2.19x / 1.58x.) At sigma = 0.003: 1.18-9.6x over the sampler and 1.04-1.80x over the better
of the two, 0.97x at 3e11; at sigma = 0.01: 1.19-10.1x and 1.05-1.75x, 0.96x at 3e11. (Final tables use development nets
100-107; an earlier run with fewer development kprop files gave the same picture within 0.04.)""")
rep("""- Against the better of the two entry types at the same budget, H2 still gains 1.04-1.75x, most where they are comparable""",
    """- Against the better of the two entry types at the same budget, H2 still gains 1.04-1.76x, most where they are comparable""")
rep("""two at the same budget.""", """two at the same budget.""")
rep("""shallow depth, combining ARC's K=2/K=3 with a well-weighted sampled estimate is worth 1.04-1.75x over the better of the""",
    """shallow depth, combining ARC's K=2/K=3 with a well-weighted sampled estimate is worth 1.04-1.76x over the better of the""")
rep("""| H2: kprop estimate combined with the Kerdock mean, weights from development nets | 24 evaluation nets | depth 8: win at sigma <= 0.01 (section 6); depth 32: no significant gain at sigma 0.001-0.003, +1-6% over [6.8e10, 2.72e11] at sigma = 0.03 |""",
    """| H2: kprop estimate combined with the Kerdock mean, weights from development nets | 24 evaluation nets | depth 8: win at sigma <= 0.01 (section 6); depth 32: no win at sigma 0.001-0.003 (1-2% point gains, mean CIs include 1); +2% at 6.8e10 at sigma = 0.01; +1-6% over [6.8e10, 2.72e11] at sigma = 0.03 |""")
open(p, 'w').write(s)
print('patched')
