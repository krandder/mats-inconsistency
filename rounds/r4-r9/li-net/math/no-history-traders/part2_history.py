"""Part 2a: a non-converging price sequence on one independent sentence phi that no history-free trader
(among a test set) exploits, while a history-using trader exploits it.  Prices in {0.3, 0.7}.
Worth in world W: sum_i g_i (W(phi) - x_i).  Exploit = bounded below and unbounded above.
Market rule (uses the market's own history, oracle knowledge of traders): low phase until every HF trader
has worth(phi=0) < -j ; then high phase until every HF trader has worth(phi=1) < -j ; j = phase index."""
import numpy as np
LO, HI = 0.3, 0.7
hf = {  # history-free traders: shares bought on day n at price x
 'const   (0.5-x)':        lambda n, x: 0.5 - x,
 'n^-1/2  (0.5-x)':        lambda n, x: (0.5 - x)/np.sqrt(n),
 'asym    2(.5-x)|.5(.5-x)': lambda n, x: 2*(0.5-x) if x < 0.5 else 0.5*(0.5-x),
 'n^-1/4  (0.6-x)':        lambda n, x: (0.6 - x)/n**0.25,
}

# vectorised traders: rate(n) * (c - x)
hf = {'const   (0.5-x)': (lambda n: np.ones_like(n), 0.5),
      'n^-1/2  (0.5-x)': (lambda n: n**-0.5, 0.5),
      'n^-1/4  (0.6-x)': (lambda n: n**-0.25, 0.6),
      'n^-1/3  (0.4-x)': (lambda n: n**-(1/3), 0.4)}
K = len(hf); w0 = np.zeros(K); w1 = np.zeros(K); mn = np.zeros(K)
h = 0.0; cash = 0.0; hmin = 0.0; hmax = 0.0
n = 0; B = 1 << 16



for j in range(1, 4):
    for x, world in ((LO, 0), (HI, 1)):
        while True:
            days = np.arange(n+1, n+B+1, dtype=float)
            G = np.stack([r(days)*(c - x) for r, c in hf.values()])        # K x B shares
            W = (w0 if world == 0 else w1)[:, None] + np.cumsum(G*(world - x), axis=1)
            ok = np.all(W < -j, axis=0); stop = int(np.argmax(ok)) if ok.any() else B-1
            G = G[:, :stop+1]
            c0 = w0[:, None] + np.cumsum(G*(0-x), axis=1); c1 = w1[:, None] + np.cumsum(G*(1-x), axis=1)
            mn = np.minimum(mn, np.minimum(c0.min(1), c1.min(1)))
            w0, w1 = c0[:, -1], c1[:, -1]; n += stop+1
            if ok.any(): break
        tgt = 1.0 if x == LO else 0.0; d = tgt - h; h += d; cash -= d*x   # history trader, once per phase
        hmin = min(hmin, cash, cash + h); hmax = max(hmax, cash, cash + h)
    print(f'phase {j}: day {n:>11}, HF min plausible worth so far = {mn.round(2)}, max = {np.maximum(w0,w1).round(1)}; history trader worth in [{hmin:.2f},{hmax:.2f}]', flush=True)
print('HF traders:', list(hf))
print('Prices alternate 0.3/0.7 forever (no convergence). Every HF trader is driven below -j in phase j')
print('(not bounded below => not an exploit). The history trader (target position 1 at 0.3, 0 at 0.7) stays >= -0.3')
print('and gains 0.4 per cycle (bounded below, unbounded above => exploit).')

# Single-trader version (market tailored to one HF trader), 10 phases: min worth -> -inf, history trader exploits.
r, c = (lambda n: np.ones_like(n)), 0.5
w0 = w1 = 0.0; mn = 0.0; n = 0; h = cash = 0.0; hmax = 0.0; hmin = 0.0
for j in range(1, 11):
    for x, world in ((LO, 0), (HI, 1)):
        g = c - x; step = g*(world - x)          # constant per-day change of the watched worth
        cur = w0 if world == 0 else w1
        L = int(np.floor((-j - cur)/step)) + 1 if cur >= -j else 1
        w0 += L*g*(0-x); w1 += L*g*(1-x); n += L; mn = min(mn, w0, w1)
        d = (1.0 if x == LO else 0.0) - h; h += d; cash -= d*x; hmin = min(hmin, cash, cash+h); hmax = max(hmax, cash, cash+h)
    print(f'single const trader, phase {j:>2}: day {n:>10}, min worth {mn:12.1f}; history trader in [{hmin:.2f},{hmax:.2f}]')
