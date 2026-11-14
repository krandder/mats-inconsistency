"""Second-order (1/n^2) Kramers-Moyal correction of the radius law (continues km_first_order.py).
P = P0 + P1/n + P2/n^2 for X = r/n; Q/n = (S/n) X with S ~ chi2_n independent is then mixed exactly to order n^-2."""
import sympy as sp

x, nn, t, s, y = sp.symbols('x n t s y', positive=True)
EY = [sp.Integer(1)] + [sp.Rational(1, 2) * sp.factorial2(2 * k - 1) for k in range(1, 10)]
KY = sp.series(sp.log(sum(EY[k] * t ** k / sp.factorial(k) for k in range(10))), t, 0, 10).removeO()
kapY = [sp.factorial(k) * KY.coeff(t, k) for k in range(10)]
kapA = {k: (2 / nn) ** k * nn * kapY[k] for k in range(2, 10)}
Ka = sum(kapA[k] * t ** k / sp.factorial(k) for k in range(2, 10))
Ma = sp.series(sp.exp(Ka), t, 0, 10).removeO()
mom_a = [sp.Integer(1)] + [sp.factorial(k) * Ma.coeff(t, k) for k in range(1, 10)]
KMAX = 9


def Dk(k):
    return sp.expand(nn * sum(sp.binomial(k, j) * mom_a[j] * x ** j * (1 / nn) ** (k - j) for j in range(k + 1)))


def coef(expr, p):
    ex = sp.expand(expr.subs(nn, 1 / s))
    return sp.simplify(ex.coeff(s, p))


Ds = {k: Dk(k) for k in range(1, KMAX + 1)}
Dc = {p: {k: coef(Ds[k], p) for k in range(1, KMAX + 1)} for p in range(0, 3)}
print('order-2 parts of D_k:', Dc[2])
P0 = sp.Rational(2, 5) * x ** -2 * sp.exp(-sp.Rational(2, 5) / x)
P1 = 13 * (25 * x ** 2 - 2) * sp.exp(-sp.Rational(2, 5) / x) / (1875 * x ** 4)


def S(Dp, P):
    return sum(((-1) ** k / sp.factorial(k)) * sp.diff(Dp[k] * P, x, k - 1) for k in range(1, KMAX + 1))


# check order 1 reproduces
chk1 = sp.simplify(S(Dc[0], P1) + S(Dc[1], P0))
print('order-1 residual:', chk1)
rhs = -sp.simplify(S(Dc[1], P1) + S(Dc[2], P0))
fprime = sp.simplify(rhs / (sp.Rational(5, 2) * x ** 2 * P0))
print("f2' =", sp.factor(fprime))
f = sp.simplify(sp.integrate(fprime, x))
norm = sp.simplify(sp.integrate(sp.simplify(P0 * f), (x, 0, sp.oo)))
P2 = sp.simplify(P0 * (f - norm))
print('P2 =', sp.factor(P2))
print('order-2 residual:', sp.simplify(S(Dc[0], P2) + S(Dc[1], P1) + S(Dc[2], P0)))
F2 = sp.simplify(sp.integrate(P2, (x, 0, x)))
print('F2_r =', sp.factor(F2))
# Q/n = (1+e) X, e = S/n - 1:  F_Q(y) = E F_X(y/(1+e)); expand to n^-2 using the moments of e
Fx = sp.Function('F')
# E[g(y/(1+e))] = sum_k g^(k)(y) E[(y/(1+e) - y)^k]/k!;  moments of e: chi2_n/n - 1
e = sp.symbols('e')
Se = sp.series(1 / (1 + e) - 1, e, 0, 6).removeO()
# raw moments of e: cumulants kappa_k(chi2_n/n) = 2^{k-1}(k-1)! n / n^k
kap_e = {k: 2 ** (k - 1) * sp.factorial(k - 1) * nn / nn ** k for k in range(2, 7)}
Ke = sum(kap_e[k] * t ** k / sp.factorial(k) for k in range(2, 7))
Me = sp.series(sp.exp(Ke), t, 0, 7).removeO()
mom_e = [sp.Integer(1), sp.Integer(0)] + [sp.factorial(k) * Me.coeff(t, k) for k in range(2, 7)]


def Emom(poly):
    poly = sp.expand(poly)
    return sum(poly.coeff(e, k) * mom_e[k] for k in range(0, 7))


Fr = sp.Lambda(x, sp.exp(-sp.Rational(2, 5) / x) + sp.integrate(P1, (x, 0, x)) / nn + F2 / nn ** 2)
Fry = Fr(y)
tot = 0
for k in range(0, 5):
    dk = sp.diff(Fry, y, k)
    tot += dk * Emom((y * Se) ** k) / sp.factorial(k)
tot = sp.expand(tot)
FQ = {p: sp.simplify(coef(tot.subs(y, x), p)) for p in range(0, 3)}
print('F_Q coefficients:')
for p in range(3):
    print(p, sp.factor(FQ[p]))
with open('/home/agent/workspaces/mats/work/stationary-numerics/results/km_second_order.txt', 'w') as fh:
    fh.write('P2(x) [X = r/n] = %s\nF2_r(x) = %s\nF_Q: F0 = %s\nF1 = %s\nF2 = %s\n' % (sp.factor(P2), sp.factor(F2), FQ[0], sp.factor(FQ[1]), sp.factor(FQ[2])))
