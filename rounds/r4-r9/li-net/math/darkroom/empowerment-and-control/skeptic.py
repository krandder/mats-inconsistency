"""Skeptic checks for empowerment-and-control (2026-10-03). Reuses toy.py. numpy only, seconds.
Run: python3 skeptic.py
S1  The brief's own toy (dark, iid noise, PASSIVE learnable stream, each with a cost): who wins?
S2  Burda et al.'s literal noisy TV ("a TV that randomly changes channels when an action is taken"):
    action 'press' redraws the image uniformly from C channels, other actions keep the image.
S3  Remote with band buttons: each button redraws fresh noise from its own band of the screen.
S4  Same external-cost flip as part B, but for the white-noise room alone (noise vs lit, no dark room).
S5  Sensor boundary for the white-noise room: noise + proprioception.
"""
import numpy as np
from toy import blahut_arimoto, channel, make_rooms, ring_step, K

rng = np.random.default_rng(0)
rooms = make_rooms(rng)
E = {nm: [blahut_arimoto(channel(r, n))[0] for n in (1, 2, 3)] for nm, r in rooms.items()}
D, N, P, L = ("dark (sensor constant)", "white noise (iid, action-independent)",
              "passive periodic stream", "lit ring (controllable)")

print("S1. Brief's toy: dark / iid noise / passive learnable stream, J = E_3 - lam*cost")
cost = {D: 0.0, N: 0.5, P: 0.5}
for lam in (0.0, 1e-9, 1e-3, 1.0):
    J = {k: E[k][2] - lam * c for k, c in cost.items()}
    b = max(J.values())
    print(f"  lam={lam:g}: E_3 = {[round(E[k][2], 3) for k in cost]}, winners = {[k for k, v in J.items() if abs(v - b) < 1e-12]}")
print("  -> any lam > 0 sends the agent to the dark room; lam = 0 is a 3-way tie. No value of lam avoids it.")


def tv_channel(C, n, bands=None):
    """Open-loop n-step channel for a TV. Actions (0,1,2). bands=None: 0 = keep image (starts on channel 0),
    1,2 = press (uniform redraw over C). bands=k: action a in {1,2} redraws from band a (C//2 symbols each),
    action 0 keeps the current image. Sensor = image after n steps."""
    import itertools
    rows = []
    for seq in itertools.product((0, 1, 2), repeat=n):
        last = [a for a in seq if a != 0]
        dist = np.zeros(C)
        if not last:
            dist[0] = 1.0
        elif bands is None:
            dist[:] = 1.0 / C
        else:
            h = C // 2
            lo = 0 if last[-1] == 1 else h
            dist[lo:lo + h] = 1.0 / h
        rows.append(dist)
    return np.array(rows)


print("\nS2. Burda noisy TV (press -> fresh uniform image over C channels; keep -> same image)")
for C in (2, 4, 16, 256, 4096):
    vals = [blahut_arimoto(tv_channel(C, n))[0] for n in (1, 2, 3)]
    print(f"  C={C:5d}: E_1..3 = {[round(v, 3) for v in vals]}   (passive learnable stream: 0, dark: 0)")
print("  -> E > 0 for every C >= 2 and INCREASES with the noise entropy log2 C (limit 1 bit).")
print("     The canonical noisy TV beats darkness and passive learnable structure with no parameter.")
print(f"     vs lit+noise eps=0.25 (E_1={E['lit ring + noise eps=0.25'][0]:.3f}): TV wins once C >= 16.")

print("\nS3. Band remote: 'keep', 'button 1' (fresh noise, lower half of screen), 'button 2' (upper half)")
for C in (4, 16, 256, 4096):
    vals = [blahut_arimoto(tv_channel(C, n, bands=True))[0] for n in (1, 2, 3)]
    print(f"  C={C:5d}: E_1..3 = {[round(v, 3) for v in vals]}   lit ring E_1..3 = {[round(v, 3) for v in E[L]]}")
print("  -> at horizon n=1 the pure-noise remote approaches the lit ring (log2 3); the horizon decides")
print("     whether the structured controllable room is preferred (n>=2) or nearly tied (n=1).")

print("\nS4. White-noise room vs lit ring with costs (noise 0.5, lit 1.0), no dark room on offer, n=3")
lam_star = E[L][2] / 0.5
for lam in (1.0, 5.0, 5.5, 5.7, 8.0):
    J = {N: E[N][2] - lam * 0.5, L: E[L][2] - lam * 1.0}
    print(f"  lam={lam}: winner = {max(J, key=J.get)}")
print(f"  -> switch at lam* = {lam_star:.3f}: the white-noise room is chosen above it, exactly like part B's dark room.")

print("\nS5. Sensor boundary: white noise + proprioception (sensor = (position, noise symbol))")
unif = np.full(K, 1.0 / K)
noise_prop = (0, ring_step, lambda s: np.kron(np.eye(K)[s], unif))
print("  E_1..3 =", [round(blahut_arimoto(channel(noise_prop, n))[0], 3) for n in (1, 2, 3)],
      " (= lit ring; noise room alone = 0)")
