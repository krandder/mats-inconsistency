"""Apply the claims-review fixes to the outbox report (one-off)."""
p = '/home/agent/workspaces/mats/outbox/arc-noisy-deep-2026-10-29.md'
s = open(p).read()
R = []
def rep(a, b):
    global s
    if a not in s:
        raise SystemExit('MISSING: ' + a[:90])
    s = s.replace(a, b)

# --- summary headline and first paragraph
rep("""**Depth 32: no win for either entry type at any noise level up to 0.03, and the obstacle is the depth problem ARC calls
open. Depth 8: one modest win for the hybrid entry, built on ARC's own method.** At sigma <= 0.003 the noise is under 0.5%
of the depth-32 variance and moves the depth-32 means by nMSE 4e-7, so the primary results are about ARC's problem.""",
"""**Depth 32: no win for either entry type at the primary noise levels (sigma = 0.001, 0.003), and the obstacle is the
depth problem ARC calls open. Depth 8: one modest win for the hybrid entry, built on ARC's own method.** At sigma <= 0.003
the noise is under 1% of the depth-32 variance and moves the depth-32 means by nMSE 3e-7, so the primary results are about
ARC's problem.""")
rep("""| ours | H2, ARC's kprop estimate averaged with the Kerdock mean under error models fitted on development nets: beats the best sampler at every budget from 1e9 to 2.72e11 (median gain 1.19x to 9.9x, all CIs above 1) and the better of the two entries by 1.04x to 1.75x; 0.99x at 3e11 | nothing: H2 0.92-1.02x of the Kerdock sampler; linear controls with ARC's closures 0.03-0.9x; none of our sample-free families beats kprop |""",
"""| ours | H2, ARC's kprop estimate averaged with the Kerdock mean under error models fitted on development nets: beats the best sampler at every budget from 1e9 to 2.72e11 (median gain 1.19x to 9.9x, all CIs above 1) and the better of the two entries by 1.04x to 1.75x; 0.99x at 3e11 | nothing at sigma = 0.001 or 0.003: H2 within about 15% of the Kerdock sampler with no significant gain; linear controls with ARC's closures 0.03-0.9x; none of our sample-free families beats kprop |""")
rep("""The depth-8 win holds at sigma = 0.001, 0.003 and 0.01, so the lowest noise level at which anything of ours wins is 0.001
(the lowest tested). It is a hybrid-entry win whose sample-free half is ARC's K=2/K=3: the "new" part is only combining a
description and a sampler by their error models. At depth 32 the only win is at sigma = 0.1: antithetic noise on the
Kerdock design beats every plain sampler at every budget (1.4-2.2x), because noise is then 83% of the output variance. At
sigma <= 0.03 that trick loses (0.5-0.9x).""",
"""The depth-8 win holds at sigma = 0.001, 0.003 and 0.01, so the lowest noise level at which anything of ours wins is 0.001
(the lowest tested). It is a hybrid-entry win whose sample-free half is ARC's K=2/K=3: the "new" part is only combining a
description and a sampler by their error models. At depth 32 the wins start above the primary levels and are small until
sigma = 0.1: H2 beats the Kerdock sampler by 1-6% over [6.8e10, 2.72e11] at sigma = 0.03 (and by 2% at 6.8e10 only at
sigma = 0.01), found by the claims review; at sigma = 0.1, antithetic noise on the Kerdock design beats every plain sampler at
every budget (1.4-2.2x), because noise is then 83% of the output variance. At sigma <= 0.03 that trick loses (0.5-0.9x).""")
_i = s.index("Why depth 32 is blocked (section 5, measured):"); _j = s.index("## Progress log")
s = s[:_i] + """Why depth 32 is blocked (section 5, measured; the claims review narrowed the wording):
1. The depth-32 means depend on non-Gaussian joint structure at intermediate layers. Replacing layer 8, 16 or 24 by a
   Gaussian with the (estimated) mean and covariance and then running the real remaining layers leaves nMSE 8e-5 to 5e-4 at
   depth 32 (3 nets); keeping the empirical per-neuron marginals (2^18 paths) with Gaussian dependence (our
   copula/neuron code) does not help (2.3e-4). That is 12-75x the Kerdock sampler's error at the Phase-1 budget. This rules
   out these replacements, not every possible description.
2. One layer at a time, the missing information is per-neuron and fourth-order in the next layer's pre-activations:
   their cumulants of order 3-4 cut the Gaussian one-step error 19-120x, the same cumulants rebuilt from the top 8 collapse
   modes only 1.9-4.1x. Carrying them by ARC's K = 4 costs about 1e14 FLOPs at 256 x 32 (not a lower bound for other ways).
3. Hybrids need what our descriptions cannot supply. With the exact layer-m mean, a linear control on the Kerdock design
   would gain 1.1-1.4x (m = 12), 1.6-2.0x (m = 16) and 4.1-5.5x (m = 24) at 5e10-3e11 FLOPs; with ARC's factorized K=3 mean,
   its bias turns every layer into a loss (0.03-0.9x). When the coarse level has to be sampled itself, the three two-level
   splits we modelled are worse than one level.
4. Noise does not help at the primary levels: an error made at layer l survives to depth 32 about the same way (within 16%)
   up to sigma = 0.03 (one direction per depth, one net). Only around sigma = 0.1 does noise change the picture (kprop's
   depth-32 error falls 13-19x, the Kerdock design's efficiency halves, and sample-free and sampling meet).

So, as a control mean, a description must be right to about 1e-6 at depth 16 or more before it helps at depth 32. That is
ARC's open problem (matching sampling at large depth); none of our families gets there. The ARC note's prior of about 5%
for beating ARC at depth 16-32 looks right.

""" + s[_j:]
rep("""- 04:33Z all truths in; every lane VM deleted, instance list empty. 04:37Z final tables; claims review (section 8).""",
"""- 04:33Z all truths in; every lane VM deleted, instance list empty. 04:37Z final tables.
- 04:39-04:50Z claims review (GPT-6, high effort): confirmed the depth-8 H2 and sigma = 0.1 antithetic wins; found that H2
  also wins narrowly at depth 32 at sigma = 0.03 (headline narrowed to the primary levels), refuted my arithmetic rejection
  of noise extrapolation (its exploratory check: +4% at depth 8), and corrected several numbers and wordings (section 8).""")
# --- section 2 table: 24-net shifts
rep("""| shift of the true means vs sigma = 0.001, depth 8 (nMSE units, median of 18-19 nets) | 0 (within truth noise) | 3.7e-7 | 3.1e-5 | 3.8e-3 |
| same, depth 32 | 4.0e-7 | 6.1e-5 | 4.8e-3 | 0.45 |""",
"""| shift of the true means vs sigma = 0.001, depth 8 (nMSE units with the sigma = 0.001 variance, median of 24 nets) | 0 (within truth noise) | 3.9e-7 | 3.3e-5 | 4.1e-3 |
| same, depth 32 | 3.1e-7 | 5.1e-5 | 3.9e-3 | 0.39 |""")
# --- sampler description: exactness
rep("""stratified chi radii so every point is N(0, I) and the
  estimator is unbiased under noise.""", """stratified chi radii so every point is N(0, I) and the
  estimator is unbiased under noise. (The full direction set is a spherical design of strength 5; with randomized radii and
  partial blocks our version is unbiased but not exact for degree-5 polynomials.)""")
# --- candidates table
rep("""| as above | 10% better than K=2; closure errors at depth are 69-84% orthogonal to the mean direction |""",
    """| as above | 9% better than our Gaussian closure but 11% worse than K=2 at depth 32; closure errors at depth are 69-84% orthogonal to the mean direction |""")
rep("""| Noise as a resource, sample-free: run the closure on a noisier copy and extrapolate in sigma^2 | arithmetic on the measured shifts | the shift (6e-5 at 0.01 to 0.45 at 0.1) dwarfs the closure gain (2x at 0.03); extrapolation weights amplify closure errors about 2x |""",
    """| Noise as a resource, sample-free: run the closure on noisier copies and extrapolate in sigma^2 | I rejected it on arithmetic; the reviewer showed that argument was wrong | untested by me. The reviewer's exploratory check (not preregistered, on evaluation nets): K=3f-augmented at sigma = 0.03 and 0.1 extrapolated to 0.001 gives 1.83e-6 vs 1.92e-6 at depth 8 (+4%, CI above 1, cost 2.1e11); no tested depth-32 pair passed. Open |""")
rep("""| H2: kprop estimate combined with the Kerdock mean, weights from development nets | 24 evaluation nets | depth 8: win (section 6); depth 32: 0.92-1.02x |""",
    """| H2: kprop estimate combined with the Kerdock mean, weights from development nets | 24 evaluation nets | depth 8: win at sigma <= 0.01 (section 6); depth 32: no significant gain at sigma 0.001-0.003, +1-6% over [6.8e10, 2.72e11] at sigma = 0.03 |""")
# --- section 3: Kerdock exactness sentence in 5.3
rep("""3. **Hybrids inherit the same limit.** The Kerdock design integrates input polynomials up to degree 5 exactly, which is
   where the linear predictability of shallow layers lives; what it leaves is predictable only from deep layers. On the""",
"""3. **Hybrids inherit the same limit.** The Kerdock design removes most of the low-order input dependence, which is where
   the linear predictability of shallow layers lives; the oracle controls below show that what it leaves is predictable
   mainly from deep layers. On the""")
rep("""bias wins: 0.6-0.9x at m = 4, 0.4-0.7x at 8, 0.05-0.5x at 12, 0.04-0.13x at 16, 0.03-0.07x at 24. Two-level sampling over depth does not
   help either: with the oracle gains G(m) (the deep level keeps 1/G of the Kerdock variance, the coarse level at most the
   rest), cost x variance is (sqrt(32/G) + sqrt(m (1 - 1/G)))^2 against 32 for one level: 39.7 at m = 8 (G = 1.15), 46.2
   at 16 (G = 2.1), 47.3 at 24 (G = 5.2), worse in every case even with a perfect coarse estimator (this assumes the design
   reduces both levels' variance alike).""",
"""bias wins: 0.6-0.9x at m = 4, 0.4-0.7x at 8, 0.05-0.5x at 12, 0.04-0.13x at 16, 0.03-0.07x at 24 (the oracle numbers use the
   truth's 1/N correction although oracle and target share truth paths, which makes them slightly optimistic; the effect
   is about 1-2% at these levels). Two-level sampling over depth, with the coarse expectation itself sampled: if the deep
   level keeps 1/G(m) of the Kerdock variance and the coarse level carries the rest at m/32 of the path cost, cost x variance
   is (sqrt(32/G) + sqrt(m (1 - 1/G)))^2 against 32 for one level: 39.6 at m = 8 (G = 1.15), 46.2 at 16 (G = 2.1), 47.4 at 24
   (G = 5.2). These three modelled splits are worse than one level; with the coarse expectation given for free the gain would
   be G, which is the oracle control above, i.e. ARC's open problem again.""")
rep("""4. **Noise does not help at the primary levels.** A 1% mean shift made at layer l keeps 0.009 / 0.10 / 0.11 / 0.25 / 0.74
   of its energy at depth 32 for l = 4 / 8 / 16 / 24 / 28 at sigma = 0.001 (the participants' law prod(1 - arccos(c)/pi)
   predicts 0.012 / 0.032 / 0.12 / 0.37 / 0.62), unchanged within 15% up to sigma = 0.03 and 2-4x smaller at sigma = 0.1
   (one direction per depth, one net).""",
"""4. **Noise does not help at the primary levels.** A 1% mean shift made at layer l keeps 0.009 / 0.10 / 0.11 / 0.25 / 0.74
   of its energy at depth 32 for l = 4 / 8 / 16 / 24 / 28 at sigma = 0.001 (the participants' law prod(1 - arccos(c)/pi)
   predicts 0.012 / 0.032 / 0.12 / 0.37 / 0.62), unchanged within 16% up to sigma = 0.03 and 1.4-3.9x smaller at sigma = 0.1
   (one direction per depth, one net).""")
rep("""So a description has to be right to about 1e-6 at depth 16 or more before it helps either entry type, which is the open
problem ARC states (matching sampling at large depth). Our families do not get there.""",
"""So a description used this way has to be right to about 1e-6 at depth 16 or more before it helps at depth 32, which is the
open problem ARC states (matching sampling at large depth). Our families do not get there.""")
# --- section 6 / 7 prediction status
rep("""- PREREG amendment 2 predicted >= 1.2x with CIs above 1 from 1e9 to 2e11 and 1.0-1.2x at 2.72e11-3e11 at depth 8 (held), and a
  gain above 1 at depth 32 for 1e9-5e9 (did not appear: 0.94-1.00x).""",
"""- PREREG amendment 2 predicted >= 1.2x with CIs above 1 from 1e9 to 2e11 at depth 8 (held) and 1.0-1.2x at 2.72e11-3e11
  (held at 2.72e11, 1.18-1.19x; just missed at 3e11, 0.97-0.99x), and a gain above 1 at depth 32 for 1e9-5e9 (not
  significant: point estimates 0.94-1.00x at sigma = 0.001 and up to 1.06x at sigma = 0.003, CIs including 1).""")
rep("""| H2 combination | depth 8: >= 1.2x from 1e9 to 2e11 with CIs above 1, 1.0-1.2x at 2.72e11-3e11; depth 32: > 1 only at 1e9-5e9 | depth 8 held; depth 32 did not (0.94-1.00x at 1e9-5e9) |""",
"""| H2 combination | depth 8: >= 1.2x from 1e9 to 2e11 with CIs above 1, 1.0-1.2x at 2.72e11-3e11; depth 32: > 1 only at 1e9-5e9 | depth 8 held except 3e11 (0.97-0.99x); depth 32 not significant at 1e9-5e9; unpredicted small wins at sigma = 0.03 (+1-6%, 6.8e10-2.72e11) |""")
rep("""| C5 noise | kprop at depth 32 within 1.5x of sigma = 0.001 for sigma <= 0.01, >= 2x lower at 0.1; last-noise integration gains < 2% at sigma <= 0.01 | held: within 1.1x at 0.01, 13-19x lower at 0.1; integration is 1-3% worse after its cost |""",
"""| C5 noise | kprop at depth 32 within 1.5x of sigma = 0.001 for sigma <= 0.01, >= 2x lower at 0.1; last-noise integration gains < 2% at sigma <= 0.01 | held: within 1.1x at 0.01, 13-19x lower at 0.1; integration is 1-3% worse after its cost |""")
open(p, 'w').write(s)
print('patched')
