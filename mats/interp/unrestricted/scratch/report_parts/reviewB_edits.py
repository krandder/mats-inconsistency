p = '/home/agent/workspaces/mats/outbox/unrestricted-estimators-2026-10-31.md'
s = open(p).read()
D = '/home/agent/workspaces/mats/work/unrestricted/scratch/report_parts/'
old_table = open(D + 'table_all.txt').read().rstrip('\n')
new_table = open(D + 'table_all_v2.txt').read().rstrip('\n')
assert old_table in s
s = s.replace(old_table, new_table)
rep = {
# (1) summary: deductive vs point
"""**Short answer.** The adaptive deductive estimator I built (a certified branch-and-bound) catches up with RQMC only
at width 3, and its cost grows exponentially with width: that part of the progress prong failed.""":
"""**Short answer.** The adaptive whole-network estimator I built (a certified branch-and-bound) fails as a deductive
method: its bounds-only estimate never beats RQMC, and its centre-point estimate, which evaluates the net's gates at
chosen points and so is structured sampling, catches up with RQMC only at width 3. Its cost grows exponentially with
width.""",
"""- Against radius-exact scrambled-Sobol RQMC at equal FLOPs: at width 3 it reaches parity at 1e6-4e6 passes (both""":
"""- The deductive estimate (the midpoint of the certified bounds) never beats RQMC: its best ratio over all 144
  net-arm runs and budgets is 0.18 (width 3). The centre-point estimate, against radius-exact scrambled-Sobol RQMC at
  equal FLOPs: at width 3 it reaches parity at 1e6-4e6 passes (both""",
"""  to 1e-4); at width 4 it loses 10^3-10^4x; at widths 6-8, 10^4-10^6x; at widths 16-32 it stays at the
  accuracy of 2-30 Monte Carlo samples after 4e6 passes, 10^5-10^6x behind.""":
"""  to 1e-4); at width 4 it loses 33x-2e5x; at widths 6-8, 10^4-10^6x; at widths 16-32 it stays at the accuracy of 2-30
  Monte Carlo samples after 4e6 passes, and 10^5-10^6x behind RQMC at 2.6e5-1e6 passes, the largest budgets where RQMC
  was run at those widths.""",
"""  out of reach at 8 and above), against 130-2,000 for RQMC, and a cell costs 46-170 forward passes. Extrapolating
  that growth (HEURISTIC), nothing in this family can reach 10M+ parameters.""":
"""  out of reach at 8 and above), against 130-8,200 for RQMC, and a cell costs 46-170 forward passes. Extrapolating
  that growth (HEURISTIC; no large model was run), this family cannot reach 10M+ parameters.""",
# (2) design ceiling relabel in summary
"""  2-design ceiling at L = n/4 falls from about 7 (width 16) to about 2 (widths 64-256), and astra's proven bound caps
  positive-weight designs of up to n(n+2)/6 points at 4.1x at ARC's 256 x 32 (kernel values measured post hoc) and at
  about 4x in the infinite-width limit.""":
"""  2-design ceiling at L = n/4 falls from about 7 (width 16) to about 2 (widths 64-256); astra's proven bound caps
  positive-weight designs at about 4x in the infinite-width limit, and a plug-in estimate of the same bound at ARC's
  256 x 32 is also about 4x (post hoc, 3 nets, noisy: not itself a bound).""",
# (5)(6) multilevel in summary
"""   inputs through fast matrix multiplication beats serial MC by n^{0.19}, and the multilevel sampler beats
   radius-exact MC by Omega(n / log^2 n) at depth proportional to width (PROVEN UNDER schoolbook-type costs; measured
   two-level gains 1.5x at L = n = 128, 1.7-3x at L = 4n, 7x at L = 16n, 15x at width 8 and depth 256; none at L <= n/4).""":
"""   inputs through fast matrix multiplication beats serial MC by n^{0.19}, and the multilevel sampler beats
   radius-exact MC by Omega(n / log^2 n) at depth proportional to width (PROVEN UNDER astra's layerwise cost
   hypotheses, which schoolbook costs satisfy; estimated ideal two-level gains, setup and pilot excluded: 1.5x at
   L = n = 128, 1.7-3x at L = 4n, 7x at L = 16n, 15x at width 8 and depth 256; about break-even at L <= n/4).""",
"""late layers through one reference trajectory and beats radius-exact MC by a factor that grows with depth (proven
asymptotically; measured 15x at width 8, depth 256; break-even at ARC's 256 x 32). So deterministic estimators can""":
"""late layers through one reference trajectory and beats radius-exact MC by a factor that grows with depth (proven
asymptotically under stated cost hypotheses; estimated ideal gain 15x at width 8, depth 256; about break-even at ARC's
256 x 32). So deterministic estimators can""",
"""**Which prong wins.** For deductive estimators the difficulties prong wins: none of ours or of the analysed classes
matches sampling at scale, and no impossibility proof is in reach. The one positive result of the lane came from
red-teaming the difficulties prong: plain MC is the wrong baseline at depth.""":
"""**Which prong wins.** For deductive estimators the difficulties prong wins: our deductive estimate never beat RQMC,
none of the analysed classes matches sampling at scale, and no impossibility proof is in reach. The one positive
result of the lane came from red-teaming the difficulties prong, and it is about sampling: plain MC is the wrong
baseline at depth.""",
# (1) section 2.2 add mid paragraph and big-width fix (3)
"""- **Widths 16 and 32** (P4: hit, 18 of 18 nets). At 4e6 passes the B&B's nMSE is 0.03-0.09 at width 16 and
  0.39-0.56 at width 32, about what 2-30 Monte Carlo samples give; RQMC at the same cost is 10^5-10^6 times better
  (several RQMC columns are empty because RQMC was run only to 2^20 at these widths). No cell is ever certified, and
  the certificate's nMSE is 6e5 to 2e34.""":
"""- **Widths 16 and 32** (P4: hit under a substituted rule, 18 of 18 nets; see section 6). At 4e6 passes the B&B's nMSE
  is 0.03-0.09 at width 16 and 0.39-0.56 at width 32, about what 2-30 Monte Carlo samples give. RQMC was run only to
  2^20 at these widths; at the largest B&B checkpoints with an RQMC value (2.6e5-1e6 passes) RQMC is 10^5-10^6 times
  better. No cell is ever certified, and the certificate's nMSE is 6e5 to 2e34.
- **The deductive estimate.** The table shows the centre-point estimate. The midpoint of the certified bounds, the
  part that never evaluates the net at a point, never beats RQMC: its best resolved RQMC ratio over all runs and
  budgets is 0.18 (width 3, depth 8). Its median nMSE at 4e6 passes is 8e-12 to 3e-8 at width 3, 2e-5 to 2e7 at width
  4, and 0.5 to 4e34 at widths 6-32 (the midpoint of vacuous bounds). Entries marked * include a net whose B&B error is
  below ten times the truth's own error (registered resolution rule): those ratios are not claimed.""",
"""- **Width 4.** 10^3-10^4 times worse than RQMC at every budget (P2: hit, 12 of 12 nets); at 4e6 passes it is between
  slightly worse than and 7x better than radius-exact MC.""":
"""- **Width 4.** 33x to 2e5x worse than RQMC at every scored budget (P2: hit, 12 of 12 nets); at 4e6 passes it is
  between slightly worse than and 7x better than radius-exact MC.""",
# (8) 2.3 text
"""  4e4-3e5 at width 4, 2e6-8e6 at width 6) and is out of reach (> 8.4e6) at width 8 and above, while RQMC needs 130-2,000
  passes throughout.""":
"""  4e4-3e5 at width 4, 2e6-8e6 at width 6) and is out of reach (> 8.4e6) at width 8 and above, while RQMC needs
  130-8,200 passes throughout.""",
"| only at width 3, 4 of 24 runs, fragile under a 2x cost change | no |":
"| centre-point estimate only at width 3, 4 of 24 runs, fragile under a 2x cost change; the deductive estimate never | no (extrapolated from widths 3-32; untested at scale) |",
"| exact, so it wins as eps -> 0 at fixed n | no |":
"| exact, so it wins as eps -> 0 at fixed n | no (conditional on the count; untested at scale) |",
"| yes in cost; the gain needs depth >> width or very large width |":
"| cost scales like MC; the estimated gain needs depth about n or more at the widths tested, or very large width (untested at scale) |",
"| bounded: 2-design ceiling against RB-MC at L = n/4 falls from 7 (width 16) to about 2 (widths 64-256), 1.7 at 256 x 32 (heuristic ceiling, measured inputs); about 4x for positive designs in the mean-field limit [A]; Kerdock 2.4x over plain MC at 256 x 32 (arc-noisy lane) | yes, with a constant gain only |":
"| bounded: 2-design ceiling against RB-MC at L = n/4 falls from 7 (width 16) to about 2 (widths 64-256), 1.7 at 256 x 32 (heuristic ceiling, measured inputs); about 4x for positive designs in the mean-field limit [A]; Kerdock 2.4x over plain MC at 256 x 32 (arc-noisy lane) | yes in cost, with a constant gain only (extrapolation) |",
"| M^D coefficients | degree 4-8 worth a few MC samples at depth 16-32 (estimates) | no |":
"| M^D coefficients in a dense representation | degree 4-8 worth a few MC samples at depth 16-32 (plug-in estimates) | no (extrapolation) |",
}
for a, b in rep.items():
    assert a in s, a[:70]
    s = s.replace(a, b)
open(p, 'w').write(s)
print('ok')
