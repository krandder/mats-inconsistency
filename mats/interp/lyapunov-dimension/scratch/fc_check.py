import math
from scipy.integrate import quad
def x_of(p, N):
    return math.exp((N + 1) * math.log(math.sin((N + 1) * p)) - math.log(math.sin(p)) - N * math.log(math.sin(N * p)))
def rhoA(p, N):  # without 1/x
    return math.exp(2*math.log(math.sin(p)) + (N-1)*math.log(math.sin(N*p)) - N*math.log(math.sin((N+1)*p))) / math.pi
def dxdp(p, N, h=1e-8):
    return (x_of(p+h, N) - x_of(p-h, N)) / (2*h)
for N in (1, 2, 4, 32):
    pm = math.pi/(N+1)
    for name, f in (("A", lambda p: rhoA(p, N)), ("B", lambda p: rhoA(p, N)/x_of(p, N))):
        m0 = quad(lambda p: f(p)*abs(dxdp(p, N)), 1e-9, pm-1e-9, limit=500)[0]
        m1 = quad(lambda p: x_of(p, N)*f(p)*abs(dxdp(p, N)), 1e-9, pm-1e-9, limit=500)[0]
        m2 = quad(lambda p: x_of(p, N)**2*f(p)*abs(dxdp(p, N)), 1e-9, pm-1e-9, limit=500)[0]
        print(N, name, round(m0, 4), round(m1, 4), round(m2, 4), "target m1=1 m2=", N+1)
