p = '/home/agent/workspaces/mats/outbox/unrestricted-estimators-2026-10-31.md'
s = open(p).read()
rep = {
"""    MC). The low-degree share, and with it any design's room, falls as width and depth grow together. MEASURED,
    post hoc kernel test of M-design: PENDING.""":
"""    MC). The low-degree share, and with it any design's room, falls as width and depth grow together.
  - MEASURED, post hoc (`scratch/kernel_even.py`, 3 nets per size, ratio of averages): kernel values k(-1), k(0) and
    the degree-2 energy give the even power-series mass above degree 2 as a fraction of the angular variance: 0.03,
    0.11, 0.20, 0.13, 0.24 at (16, 4), (32, 8), (64, 16), (128, 32), (256, 64), and 0.48 at ARC's 256 x 32. Through
    astra's bound this caps the gain of any positive-weight design with at most n(n+2)/6 points (about 11,000 at width
    256) over RB-MC at 64, 19, 10, 15, 8.4 and 4.1. The fraction drifts toward the mean-field 1/2 (noisily), so the
    cap at ARC's setting is already near its limiting value of about 4. b_2 is bounded via the degree-2 harmonic
    energy (b_2 <= n a_2/(n-1) [A]), which makes the even-tail estimate a lower estimate and the cap conservative.""",
"| PROVEN in mean field (fraction 1/2, gain bound about 4) [A]; CONJECTURE at finite width. MEASURED: PENDING. |":
"| PROVEN in mean field (fraction 1/2, gain bound about 4) [A]; CONJECTURE at finite width. MEASURED (post hoc, 3 nets per size): the fraction is 0.03-0.24 at widths 16-256 with L = n/4 and 0.48 at 256 x 32, drifting toward 1/2. |",
"""  2.7e5-4.2e6 passes). At depth 4-8, none (P1, registered for depth <= 8: refuted). Deep width-3 nets are easier because
  they collapse onto few gate patterns (14-32 per net at every depth, section 3.5).""":
"""  2.7e5-4.2e6 passes). At depth 4-8, none (P1, registered for depth <= 8: refuted). Deep width-3 nets are easier because
  they collapse onto few gate patterns (14-32 per net at every depth, section 3.5). All of this is for the noiseless
  mean. At sigma = 1e-3 the noise moves the mean of these width-3 nets by nMSE 9e-10 to 1e-4 (post hoc, paired MC),
  far more than the 1e-12 where the B&B catches up with RQMC, so in ARC's noisy setting the width-3 results say
  nothing about the target there.""",
}
for a, b in rep.items():
    assert a in s, a[:60]
    s = s.replace(a, b)
a = "we ran. |"
b = ("we ran. Post hoc at widths 3-4: 6e-10 to 1e-4 (up to 0.4 on one width-4 net whose output variance is tiny). |")
assert a in s
s = s.replace(a, b, 1)
s = s.replace("Lane: unrestricted estimators (theory and experiment), target M1. Bead pool-uwmcvo. Status: DRAFT; sections marked\nPENDING wait for the evaluation queue.",
              "Lane: unrestricted estimators (theory and experiment), target M1. Bead pool-uwmcvo. Status: final draft, before the\nlast GPT-6 review.")
open(p, 'w').write(s)
print('ok')
