"""First-order (1/n) Kramers-Moyal correction to the inverse-gamma limit of the radius law (sympy).
r-chain: X = r/n, X' = A X + 1/n, kappa_k(A) = (2/n)^k n kappa_k(Y), Y = Bern(1/2) Z^2.
D_k = n E[(X'-X)^k | X = x];  stationary KM:  -D1 P + (1/2)(D2 P)' - (1/6)(D3 P)'' + (1/24)(D4 P)''' - ... = 0.
Q/n = (S/n) X with S ~ chi2_n independent adds (2/5) P0 to the 1/n coefficient of the CDF."""
import sympy as sp

x, nn, t, s = sp.symbols('x n t s', positive=True)
# raw moments of Y = B Z^2 (B fair coin): E Y^k = (1/2) (2k-1)!!
EY = [sp.Integer(1)] + [sp.Rational(1, 2) * sp.factorial2(2 * k - 1) for k in range(1, 8)]
KY = sp.series(sp.log(sum(EY[k] * t ** k / sp.factorial(k) for k in range(8))), t, 0, 8).removeO()
kapY = [sp.factorial(k) * KY.coeff(t, k) for k in range(8)]
print('kappa_k(Y):', kapY[1:])
kapA = {k: (2 / nn) ** k * nn * kapY[k] for k in range(2, 8)}
Ka = sum(kapA[k] * t ** k / sp.factorial(k) for k in range(2, 8))
Ma = sp.series(sp.exp(Ka), t, 0, 8).removeO()
mom_a = [sp.Integer(1)] + [sp.factorial(k) * Ma.coeff(t, k) for k in range(1, 8)]


def Dk(k):
    return sp.expand(nn * sum(sp.binomial(k, j) * mom_a[j] * x ** j * (1 / nn) ** (k - j) for j in range(k + 1)))


def coef(expr, p):
    ex = sp.expand(expr.subs(nn, 1 / s))
    return sp.simplify(ex.coeff(s, p))


Ds = {k: Dk(k) for k in range(1, 8)}
D0 = {k: coef(Ds[k], 0) for k in range(1, 8)}
D1 = {k: coef(Ds[k], 1) for k in range(1, 8)}
print('order-0 parts', D0)
print('order-1 parts', D1)
P0 = sp.Rational(2, 5) * x ** -2 * sp.exp(-sp.Rational(2, 5) / x)
flux0 = -D0[1] * P0 + sp.Rational(1, 2) * sp.diff(D0[2] * P0, x)
print('zeroth-order flux check:', sp.simplify(flux0))
# first order:  -P1 + (1/2)(D0_2 P1)' + [ -D1_1 P0 + sum_k (-1)^k/k! d^{k-1}(D1_k P0) ] = 0
src = -D1[1] * P0 + sum(((-1) ** k / sp.factorial(k)) * sp.diff(D1[k] * P0, x, k - 1) for k in range(2, 8))
rhs = -src
fprime = sp.simplify(rhs / (sp.Rational(5, 2) * x ** 2 * P0))
print("f' =", sp.factor(fprime))
f = sp.simplify(sp.integrate(fprime, x))
print('f (particular) =', f)
norm = sp.simplify(sp.integrate(sp.simplify(P0 * f), (x, 0, sp.oo)))
print('int P0 f =', norm)
P1 = sp.simplify(P0 * (f - norm))
F1 = sp.simplify(sp.integrate(P1, (x, 0, x)))
print('P1 =', P1)
print('F1 (r-chain, X = r/n) =', F1)
print('large-x: F1 ~', sp.series(F1, x, sp.oo, 3))
FQ1 = sp.simplify(F1 + sp.Rational(2, 5) * P0)
print('F1 for Q/n:', FQ1)
print('large-x FQ1 ~', sp.series(FQ1, x, sp.oo, 3))
med0 = sp.Rational(2, 5) / sp.log(2)
m1 = sp.simplify(-FQ1.subs(x, med0) / P0.subs(x, med0))
m1r = sp.simplify(-F1.subs(x, med0) / P0.subs(x, med0))
print('median of Q/n = 2/(5 ln 2) + m1/n, m1 =', m1, '=', sp.N(m1, 12))
print('median of r/n: m1 =', m1r, '=', sp.N(m1r, 12))
with open('/home/agent/workspaces/mats/work/stationary-numerics/results/km_first_order.txt', 'w') as fh:
    fh.write('P1(x) [X = r/n] = %s\nF1_r(x) = %s\nF1_Q(x) = %s\nmedian coefficient m1 (Q) = %s = %s\n'
             'median coefficient m1 (r) = %s = %s\n' % (P1, F1, FQ1, m1, sp.N(m1, 15), m1r, sp.N(m1r, 15)))
