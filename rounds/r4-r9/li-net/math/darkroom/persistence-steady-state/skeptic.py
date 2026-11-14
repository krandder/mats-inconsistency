"""Skeptic check for persistence-steady-state.

Hidden parameter: the sensory symbols' energy gap  b = (E1 - E0)/kT.
check.py uses the Boyd-Mandal-Crutchfield ratchet with E0 = E1 = 0 (BMC 2016, Sec. III:
"taken to be zero for symmetry and simplicity"), so the equilibrium tape is uniform and the
fuel in a stream is 1 - h_mu.  With a gap b, the equilibrium symbol law is
p_eq(1) = e^-b / (1 + e^-b), and the most work an ideal ratchet tuned to the agent's
predictor q can take per symbol is (Esposito & Van den Broeck 2011, work <= kT * relative
entropy to equilibrium):
    W(x) = log2 q(x | past) - log2 p_eq(x)       [units kT ln 2]
with mean = relative-entropy rate D(stream || p_eq iid) = -h_mu - E[log2 p_eq(X)].
b = 0 reproduces check.py exactly.  Rooms: D dark constant 000..., N iid noise,
S Markov flip 0.95, A alternating 0101..., B bright constant 111... (control).
Run: python3 skeptic.py   (about 30 s)
"""
import numpy as np
rng = np.random.default_rng(1)
log2 = np.log2
LOG2E = 1 / np.log(2)

def Hb(p):
    p = np.clip(p, 1e-300, 1); q = np.clip(1 - p, 1e-300, 1)
    return float(-(p * log2(p) + q * log2(q)))

# room: (P(x0=1), P(x_t=1 | x_{t-1}=0), P(x_t=1 | x_{t-1}=1))
ROOMS = {"D": (0.0, 0.0, 0.0), "N": (0.5, 0.5, 0.5), "S": (0.5, 0.95, 0.05),
         "A": (0.5, 1.0, 0.0), "B": (1.0, 1.0, 1.0)}
def h(room, k):
    p0, a, b = ROOMS[room]
    if room == "D" or room == "B": return 0.0
    return 1.0 if k == 0 else 0.5 * Hb(a) + 0.5 * Hb(b)   # stationary marginal 1/2 for N, S, A
def m1(room):  # stationary fraction of 1s
    return {"D": 0.0, "B": 1.0}.get(room, 0.5)
def peq1(gap): return np.exp(-gap) / (1 + np.exp(-gap))
def fuel(room, k, gap):
    p1 = peq1(gap); f = m1(room)
    return -h(room, k) - (f * log2(p1) + (1 - f) * log2(1 - p1))

def W_samples(room, k, gap, n, T):
    p0, a, b = ROOMS[room]
    x = np.zeros((n, T), dtype=int)
    x[:, 0] = rng.random(n) < p0
    for t in range(1, T):
        pr = np.where(x[:, t-1] == 1, b, a)
        x[:, t] = rng.random(n) < pr
    m = m1(room)
    if k == 0:
        q = np.where(x == 1, m, 1 - m)
    else:
        q = np.empty(x.shape)
        q[:, 0] = np.where(x[:, 0] == 1, p0, 1 - p0)
        pr = np.where(x[:, :-1] == 1, b, a)
        q[:, 1:] = np.where(x[:, 1:] == 1, pr, 1 - pr)
    p1 = peq1(gap)
    with np.errstate(divide="ignore"):
        return log2(q) - np.where(x == 1, log2(p1), log2(1 - p1))

def survival(room, k, gap, c, eps, roomcost=0.0, n=3000, T=1000, E0=10.0, Emax=20.0):
    W = W_samples(room, k, gap, n, T)
    E = np.full(n, E0); alive = np.ones(n, bool)
    for t in range(T):
        E = np.minimum(Emax, E + W[:, t] - c - eps * k - roomcost)
        alive &= E >= 0
    return alive.mean()

def choose(rooms, gap, c, eps=0.05, roomcost=None, E0=10.0, Emax=20.0):
    roomcost = roomcost or {}
    rows = []
    for r in rooms:
        best = max((survival(r, k, gap, c, eps, roomcost.get(r, 0.0), E0=E0, Emax=Emax),
                    fuel(r, k, gap) - c - eps * k - roomcost.get(r, 0.0), -k) for k in (0, 1))
        rows.append((r, -best[2], best[0], best[1]))
    top = max(x[2] for x in rows)
    if top == 0: return rows, "none survive"
    surv = [x for x in rows if x[2] >= top - 1e-9]
    mm = max(x[3] for x in surv)
    return rows, "".join(x[0] for x in surv if x[3] >= mm - 1e-9)

print("=" * 78)
print("1. Check: gap b = 0 reproduces check.py's fuel 1 - h_mu")
print("=" * 78)
for r in "DNSA":
    print(f"   {r}: fuel(k=1, b=0) = {fuel(r, 1, 0.0):.3f}   check.py 1 - h_mu = {1 - h(r, 1):.3f}")

print("\n" + "=" * 78)
print("2. Mean fuel per symbol (bits of kT ln2, best model order) as the gap b varies")
print("=" * 78)
print(f"   {'b=gap/kT':>9s} " + " ".join(f"{r:>7s}" for r in "DNSAB") + "   fuel ranking (D,N,S only)")
for gap in (0.0, 0.2, 0.396, 0.6, 1.0, 1.386, 2.0, 5.0, 20.0, 80.0):
    f = {r: max(fuel(r, 0, gap), fuel(r, 1, gap)) for r in "DNSAB"}
    rk = " > ".join(sorted("DNS", key=lambda r: -f[r]))
    print(f"   {gap:9.3f} " + " ".join(f"{f[r]:7.3f}" for r in "DNSAB") + f"   {rk}")
print(f"""   closed form: fuel(S)-fuel(N) = 1 - h_S = {1-h('S',1):.3f} for every b  (parameter-free part)
                fuel(S)-fuel(D) = -h_S + b*log2(e)/2  > 0  iff b > 2 h_S ln2 = {2*h('S',1)*np.log(2):.3f}
                fuel(N)-fuel(D) = -1   + b*log2(e)/2  > 0  iff b > 2 ln2     = {2*np.log(2):.3f}
                fuel(B)-fuel(N) = 1 for every b: at equal brightness, lower entropy rate always wins
   b -> infinity: constant dark 000... is the equilibrium tape (fuel -> 0), the same as 'no input'.
   Visible photons at body temperature: b = h nu / kT ~ 2.3 eV / 0.027 eV ~ 85.""")

print("\n" + "=" * 78)
print("3. R3 survival choice (same rule as check.py: max survival, then margin), eps = 0.05")
print("   c set to half of S's fuel at each b so that rooms are viable; T=1000, E0=10, Emax=20")
print("=" * 78)
for gap in (0.0, 1.0, 5.0, 20.0):
    c = 0.5 * fuel("S", 1, gap)
    rows, win = choose("DNS", gap, c)
    det = "  ".join(f"{r}:k={k},surv={s:.2f},net={m:+.2f}" for r, k, s, m in rows)
    print(f"   b={gap:5.1f} c={c:6.2f}  winner {win:3s}  | {det}")

print("\n   3b. Same, with a per-step cost on the structured room (the brief: 'each with a cost'):")
for gap, rc in ((0.0, 0.8), (5.0, 0.8), (20.0, 0.8), (20.0, 0.5)):
    c = 0.5 * fuel("S", 1, gap)
    rows, win = choose("DNS", gap, c, roomcost={"S": rc})
    det = "  ".join(f"{r}:surv={s:.2f},net={m:+.2f}" for r, k, s, m in rows)
    print(f"   b={gap:5.1f} c={c:6.2f} cost_S={rc}  winner {win:3s}  | {det}")

print("\n   3c. Researcher's C1 setting (c=0.5) with only the gap changed:")
for gap in (0.0, 0.5, 1.0, 2.0):
    rows, win = choose("DNS", gap, 0.5)
    print(f"   b={gap:4.1f}  winner {win}   " + "  ".join(f"{r}:surv={s:.2f},net={m:+.2f}" for r, k, s, m in rows))

print("\n" + "=" * 78)
print("4. Second hidden parameter: the energy buffer (E0, Emax) against the fluctuation scale b")
print("   b = 5, c = half of S's fuel, cost_S = 0.8 per step (S's info surplus is 0.714)")
print("=" * 78)
for E0, Emax in ((10, 20), (30, 60), (100, 200), (300, 600)):
    rows, win = choose("DNS", 5.0, 0.5 * fuel("S", 1, 5.0), roomcost={"S": 0.8}, E0=E0, Emax=Emax)
    print(f"   E0={E0:4d} Emax={Emax:4d}  winner {win:3s}  | " + "  ".join(f"{r}:surv={s:.2f},net={m:+.2f}" for r, k, s, m in rows))
print("   b = 1.0, c = half of S's fuel, no room cost (D vs S):")
for E0, Emax in ((2, 4), (10, 20), (30, 60)):
    rows, win = choose("DNS", 1.0, 0.5 * fuel("S", 1, 1.0), E0=E0, Emax=Emax)
    print(f"   E0={E0:4d} Emax={Emax:4d}  winner {win:3s}  | " + "  ".join(f"{r}:surv={s:.2f},net={m:+.2f}" for r, k, s, m in rows))

print("""\nSummary
  The fuel ranking of D, N, S is decided by one dimensionless physical number, the sensor's
  energy gap b = (E1-E0)/kT, which check.py fixes at 0 by copying BMC's 'E0 = E1 = 0'.
    b < 0.396 : D > S > N   (dark room preferred; researcher's verdict)
    0.396 < b < 1.386 : S > D > N   (dark avoided, noise avoided)
    b > 1.386 : S > N > D   (dark avoided; noise preferred to darkness; with a cost on S above
                1 - h_S and a buffer large enough for the noise room's fluctuations, N wins)
  Survival (not mean fuel) adds a second decisive knob: the buffer (E0, Emax) against the
  fluctuation scale b decides D vs S at b = 1 and S vs N at b = 5.
  Only the comparative statement is parameter-free: at equal mean sensory energy, fuel falls by
  exactly the entropy-rate difference (S beats N by 1 - h_S, B beats N by 1 bit, for every b).
""")
