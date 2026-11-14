"""Free-probability (n -> inf at fixed N) counts: n * FC_N mass above delta^2 * edge, and the top profile
log(s_k / s_1) read from FC_N quantiles at (k - 1/2)/n."""
import math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import fc

for N in (1, 2, 4):
    print(N, 'mass', round(fc.tail_mass(1e-300, N), 4))
DELTAS = [math.exp(-1), math.exp(-2), 1e-2, 1e-3]
pairs = [tuple(int(v) for v in s.split('x')) for s in (sys.argv[1] if len(sys.argv) > 1 else
         '128x16,128x32,128x64,256x32,256x64,256x128,512x64,512x128').split(',')]
for n, N in pairs:
    out = [n * fc.tail_mass(d * d * fc.edge(N), N) for d in DELTAS]
    top = fc.quantile_from_top(0.5 / n, N)
    prof = []
    for k in (2, 4, 8, 16, 32, 64):
        if k < n:
            prof.append(0.5 * math.log(fc.quantile_from_top((k - 0.5) / n, N) / top))
    print(f"n {n} N {N}: FC count above delta^2*edge (delta = e^-1, e^-2, 1e-2, 1e-3): "
          + " ".join(f"{c:.1f}" for c in out) + f" | 0.5 log(top quantile / edge) {0.5 * math.log(top / fc.edge(N)):+.2f}"
          + " | FC log(s_k/s_1), k = 2,4,8,16,32,64: " + " ".join(f"{v:+.2f}" for v in prof), flush=True)
