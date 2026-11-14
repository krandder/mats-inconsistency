"""identity.py: checker's test of the note's section-3 'PROVEN' item on random 2x2 joints.
(a) the identity P(X=1|C)/P(X=0|C) = odds(X | C, y) * P(y | C, X=0) / P(y | C, X=1), for y = 0 and 1;
(b) the four larger-context numbers fix the joint (reconstruct it from them alone);
(c) the reverse: given only the two smaller-context numbers, how many free numbers remain among the four larger ones."""
import numpy as np
rng = np.random.default_rng(0)
worst_a = worst_b = 0.0
for _ in range(20000):
    J = rng.dirichlet(np.ones(4)).reshape(2, 2)                 # J[x, y]
    px, py = J.sum(1), J.sum(0)
    y_x = J / px[:, None]                                       # P(y | x)
    x_y = J / py[None, :]                                       # P(x | y)
    for y in (0, 1):
        worst_a = max(worst_a, abs(px[1] / px[0] - (x_y[1, y] / x_y[0, y]) * (y_x[0, y] / y_x[1, y])) / (px[1] / px[0]))
    odds_x = (x_y[1, 0] / x_y[0, 0]) * (y_x[0, 0] / y_x[1, 0])  # from larger-context numbers only
    px1 = odds_x / (1 + odds_x)
    Jr = np.stack([(1 - px1) * y_x[0], px1 * y_x[1]])
    worst_b = max(worst_b, np.abs(Jr - J).max())
print('(a) identity, worst relative error over 20,000 joints x 2 values of y: %.2e' % worst_a)
print('(b) joint rebuilt from the four larger-context numbers, worst abs error: %.2e' % worst_b)
# (c) fix the marginals; the joint is J = [[1-px-py+t, py-t],[px-t, t]] for t in an interval: ONE free number.
px, py = 0.3, 0.6
ts = np.linspace(max(0, px + py - 1) + 1e-3, min(px, py) - 1e-3, 5)
print('(c) marginals P(X=1)=0.3, P(Y=1)=0.6 fixed; larger-context numbers [P(Y=1|X=0), P(Y=1|X=1), P(X=1|Y=0), P(X=1|Y=1)] as the one free number t moves:')
for t in ts:
    print('    t=%.3f ' % t, np.round([(py - t) / (1 - px), t / px, (px - t) / (1 - py), t / py], 3).tolist())
print('    so the smaller-context numbers cut the four larger-context numbers from 3 free numbers (a compatible set) to 1: the reverse direction is partial, not nil.')
