"""N1: singular values of products of N iid Ginibre n x n matrices (entries N(0, 1/n)) against
(a) Newman's Lyapunov exponents, lambda_k = 0.5 [log(2/n) + psi((n-k+1)/2)], and
(b) free probability: squared singular values ~ Fuss-Catalan FC_N (n -> infinity at fixed N).
Reported: log(s_k/s_1) at selected k; N_surv(delta) = #{k: s_k >= delta s_1} for delta = e^-1, e^-2, 1e-2, 1e-3;
k*_F(delta) = min k with sum_{j>k} s_j^2 <= delta^2 sum s_j^2.
python n1_gauss.py --ns 128,256,512 --taus 0.125,0.25,0.5 --reps 4"""
import argparse, json, math, os
import numpy as np
from scipy.special import digamma
from scipy.integrate import quad

ap = argparse.ArgumentParser()
ap.add_argument('--ns', default='128,256,512'); ap.add_argument('--taus', default='0.125,0.25,0.5')
ap.add_argument('--reps', type=int, default=4); ap.add_argument('--out', default='runs/n1_gauss.json')
A = ap.parse_args()
DELTAS = [math.exp(-1), math.exp(-2), 1e-2, 1e-3]


def newman_gaps(n, N, half_width=False):
    """N (lambda_k - lambda_1), k = 1..n, for iid Ginibre n x n."""
    k = np.arange(1, n + 1)
    lam = 0.5 * digamma((n - k + 1) / 2.0)
    return N * (lam - lam[0])


def fc_tail_fraction(N, y):
    """Fuss-Catalan(N) mass above y (squared singular values of a product of N Ginibre, var 1/n, n -> inf).
    Parametrisation (Haagerup-Moller / Neuschel): x(p) = sin((N+1)p)^(N+1) / (sin p sin(Np)^N), p in (0, pi/(N+1)),
    density rho(x) = sin(p)^2 sin(Np)^(N-1) / (pi x sin((N+1)p)^N)."""
    def x_of(p):
        return math.exp((N + 1) * math.log(math.sin((N + 1) * p)) - math.log(math.sin(p)) - N * math.log(math.sin(N * p)))
    def dens(p):
        x = x_of(p)
        return math.exp(2 * math.log(math.sin(p)) + (N - 1) * math.log(math.sin(N * p))
                        - N * math.log(math.sin((N + 1) * p))) / (math.pi * x)
    pmax = math.pi / (N + 1)
    # mass above y = integral over p in (0, p_y) of rho(x(p)) |dx/dp| dp; do it numerically in x via p
    # find p_y with x(p_y) = y (x decreasing in p)
    lo, hi = 1e-12, pmax - 1e-12
    if y >= x_of(lo):
        return 0.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if x_of(mid) > y:
            lo = mid
        else:
            hi = mid
    py = 0.5 * (lo + hi)
    h = 1e-7
    def integrand(p):
        dxdp = (x_of(min(p + h, pmax - 1e-12)) - x_of(max(p - h, 1e-12))) / (2 * h)
        return dens(p) * abs(dxdp)
    val, _ = quad(integrand, 1e-10, py, limit=400)
    return val


def fc_edge(N):
    return (N + 1) ** (N + 1) / N ** N


def counts(logs):
    """logs: log s_k descending. Returns N_surv(delta) and k*_F(delta)."""
    r = logs - logs[0]
    out = {}
    s2 = np.exp(2 * r)
    tail = np.cumsum(s2[::-1])[::-1]          # tail[k] = sum_{j>=k} s_j^2 (0-based)
    tot = tail[0]
    for d in DELTAS:
        nsurv = int((r >= math.log(d)).sum())
        kF = int(np.argmax(np.append(tail[1:], 0.0) <= d * d * tot)) + 1
        out[f'{d:.3g}'] = dict(nsurv=nsurv, kF=kF)
    return out


rng = np.random.default_rng(20261029)
res = []
for n in [int(v) for v in A.ns.split(',')]:
    for tau in [float(v) for v in A.taus.split(',')]:
        N = max(1, int(round(tau * n)))
        reps = []
        for r in range(A.reps):
            J = np.eye(n); logscale = 0.0
            for l in range(N):
                J = (rng.standard_normal((n, n)) / math.sqrt(n)) @ J
                s = np.linalg.norm(J); J /= s; logscale += math.log(s)
            sv = np.linalg.svd(J, compute_uv=False)
            logs = np.log(np.maximum(sv, 1e-300)) + logscale
            reps.append(logs)
        L = np.array(reps)
        mean_gap = (L - L[:, :1]).mean(0)
        newg = newman_gaps(n, N)
        cnt = [counts(l) for l in L]
        # Newman-predicted counts (top exponent as reference)
        newc = counts(newg)
        # free probability: count of FC_N mass above delta^2 * x_edge, times n
        fcc = {}
        for d in DELTAS:
            fcc[f'{d:.3g}'] = n * fc_tail_fraction(N, d * d * fc_edge(N))
        ks = [k for k in (2, 4, 8, 16, 32, 64, 128) if k <= n]
        rec = dict(n=n, N=N, tau=N / n, reps=A.reps,
                   log_s1=float(L[:, 0].mean()), log_s1_sd=float(L[:, 0].std()),
                   log_s1_newman=float(N * 0.5 * (math.log(2.0 / n) + digamma(n / 2.0))),
                   log_s1_fc_edge=0.5 * math.log(fc_edge(N)),
                   gap={k: float(mean_gap[k - 1]) for k in ks}, gap_newman={k: float(newg[k - 1]) for k in ks},
                   counts_mean={d: dict(nsurv=float(np.mean([c[d]['nsurv'] for c in cnt])),
                                         kF=float(np.mean([c[d]['kF'] for c in cnt]))) for d in cnt[0]},
                   counts_newman=newc, nsurv_fc_edge=fcc,
                   coordinator={f'{d:.3g}': 1 + 2 * n / N * math.log(1 / d) for d in DELTAS})
        res.append(rec)
        print(f"n {n} N {N} (tau {N/n:.3f}): log s1 {rec['log_s1']:+.2f}+-{rec['log_s1_sd']:.2f} "
              f"(Newman {rec['log_s1_newman']:+.2f}, FC edge {rec['log_s1_fc_edge']:+.2f})", flush=True)
        print("   log(s_k/s_1) k=" + ",".join(map(str, ks)) + ": meas "
              + " ".join(f"{rec['gap'][k]:+.2f}" for k in ks) + " | Newman "
              + " ".join(f"{rec['gap_newman'][k]:+.2f}" for k in ks), flush=True)
        for d in cnt[0]:
            print(f"   delta {d}: N_surv meas {rec['counts_mean'][d]['nsurv']:.1f}  Newman {newc[d]['nsurv']}  "
                  f"FC(edge) {fcc[d]:.1f}  coordinator 1+(2n/N)ln(1/d) {rec['coordinator'][d]:.1f} | "
                  f"k*_F meas {rec['counts_mean'][d]['kF']:.1f} Newman {newc[d]['kF']}", flush=True)
os.makedirs(os.path.dirname(A.out), exist_ok=True)
json.dump(res, open(A.out, 'w'), indent=1)
