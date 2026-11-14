"""Seed-count power calculation for item 2 (written before any run; numbers go into PREREG.md).
Primary: per-seed fraction of the 3 learned slots that are non-constant at the end (cvae mechanism), paired
difference arm - original init, seed-level one-sample t-test, Holm over 3 treatment arms (worst case alpha/3)."""
import numpy as np
from scipy import stats


def power_t(delta, sd, n, alpha):
    df = n - 1
    tc = stats.t.ppf(1 - alpha / 2, df)
    ncp = delta / (sd / np.sqrt(n))
    return float(1 - stats.nct.cdf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp))


def sim_power(p_slot, sd_seed, n, alpha, reps=20000, rng=np.random.default_rng(0)):
    """Monte Carlo: seed-level p_s ~ clip(N(p_slot, sd_seed)), 3 slots ~ Bin(3, p_s); original arm = 0 (P10: 0/15)."""
    ok = 0
    for _ in range(reps):
        ps = np.clip(rng.normal(p_slot, sd_seed, n), 0, 1)
        f = rng.binomial(3, ps) / 3
        if f.std(ddof=1) == 0:
            ok += f.mean() > 0
            continue
        t = f.mean() / (f.std(ddof=1) / np.sqrt(n))
        ok += t > stats.t.ppf(1 - alpha / 2, n - 1)
    return ok / reps


a3 = 0.05 / 3
print("P10 inputs: orig non-constant fraction 0/15 in auction and 0/15 in cvae (SD 0); binomial SD of a 3-slot fraction at p=0.5 is",
      round(np.sqrt(0.25 / 3), 3))
for delta, sd in ((0.5, 0.35), (0.33, 0.35), (0.5, 0.45)):
    row = [f"n={n}: {power_t(delta, sd, n, a3):.2f}" for n in (3, 5, 6, 8, 10, 12)]
    print(f"analytic t, delta={delta}, sd={sd}, alpha={a3:.4f}:", ", ".join(row))
for p, sds in ((0.5, 0.15), (0.33, 0.15)):
    row = [f"n={n}: {sim_power(p, sds, n, a3, reps=4000):.2f}" for n in (5, 8, 10)]
    print(f"simulated, slot p={p}, seed sd={sds}:", ", ".join(row))
# NLL secondary (end-to-end test NLL vs reg2): P10 auction - reg2 = +0.104, seed-t 95% CI [-0.104, 0.313], n=5
sd_nll = (0.313 - (-0.104)) / 2 * np.sqrt(5) / stats.t.ppf(0.975, 4)
print(f"NLL secondary: P10 seed SD of (auction - reg2) = {sd_nll:.3f} nats/cell; power at n=10 for a 0.05 gap:",
      round(power_t(0.05, sd_nll, 10, 0.05), 2), "; n for 80% power at 0.05:",
      next(n for n in range(3, 400) if power_t(0.05, sd_nll, n, 0.05) >= 0.8))
# 'still collapses' reading: if 0 of 3n slots are non-constant, one-sided 95% upper bound on the slot rate
for n in (5, 8, 10):
    print(f"n={n}: 0/{3*n} non-constant -> 95% upper bound on the per-slot rate {1 - 0.05 ** (1 / (3 * n)):.3f}")
