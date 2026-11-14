"""T0d (theory): predicted per-path k* = N_surv(J; 0.01) from Fuss-Catalan at half width, (n/2) FC_{L'} mass above
delta^2 x_+ (lyapunov lane Corollary C2), for the widths and remaining depths of this lane."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
import fc
d = 0.01
for Lp in (2, 4, 8, 12, 16, 20, 24, 28):
    row = []
    for n in (32, 64, 128):
        row.append(f"n={n}: {(n/2)*fc.tail_mass(d*d*fc.edge(Lp), Lp):5.1f}")
    print(f"L'={Lp:2d}  " + "  ".join(row))
