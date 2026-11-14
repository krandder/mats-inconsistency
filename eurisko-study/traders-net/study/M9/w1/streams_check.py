"""M9(a) w1: what each soup family offers a trader that reads k ages of its own 32-bit history.
Re-implements logical-soup/src/questions.bend streams (periodic, LFSR16 Galois 0xB400, CA30,
CA110 on a 32-cell ring scanned column by column, primality of n, pi-digit parity) and the
history word the VM sees (bit k = outcome at age k, age 0 = newest, width 32).
For k = 0..4 it finds the best lookup-table predictor over all k-subsets of ages 0..31:
tables fitted on streams from train seeds, Brier measured on test seeds.
Run: python3 streams_check.py  (numpy only; ~2-4 min on one core)
"""
import itertools, json, os, csv
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T = 1200          # steps per stream after warm-up
WARM = 40
PI_DIGITS = "3141592653589793238462643383279502884197169399375105820974944592"


def s_periodic(seed, param):
    p = max(2, min(param, 16)); ph = seed % p; out = []
    for _ in range(T + WARM):
        out.append(1 if ph == 0 else 0); ph = (ph + 1) % p
    return out


def s_lfsr(seed):
    st = max(1, seed & 65535); out = []
    for _ in range(T + WARM):
        out.append(st & 1); st = (st >> 1) ^ ((st & 1) * 46080)
    return out


def ca_row(rule, row):
    nxt = 0
    for c in range(32):
        l = (row >> ((c + 31) & 31)) & 1; m = (row >> c) & 1; r = (row >> ((c + 1) & 31)) & 1
        nxt |= ((rule >> (4 * l + 2 * m + r)) & 1) << c
    return nxt


def s_ca(seed, rule):
    row = seed & 0xFFFFFFFF; col = 0; out = []
    for _ in range(T + WARM):
        out.append((row >> col) & 1)
        if col == 31:
            row = ca_row(rule, row); col = 0
        else:
            col += 1
    return out


def is_prime(v):
    if v < 2: return 0
    d = 2
    while d * d <= v:
        if v % d == 0: return 0
        d += 1
    return 1


def s_prime(seed):
    n = seed & 65535; out = []
    for _ in range(T + WARM):
        out.append(is_prime(n & 65535)); n = (n + 1) & 65535
    return out


def s_pi(seed):
    i = seed & 63; out = []
    for _ in range(T + WARM):
        out.append(int(PI_DIGITS[i]) & 1); i = (i + 1) & 63
    return out


def gen(fam, seed, j):
    if fam == 'periodic': return s_periodic(seed, 2 + j % 15)
    if fam == 'lfsr': return s_lfsr(seed)
    if fam == 'ca30': return s_ca(seed, 30)
    if fam == 'ca110': return s_ca(seed, 110)
    if fam == 'prime': return s_prime(seed)
    if fam == 'pi': return s_pi(seed)


def design(fam, seeds):
    """Return H (N,32) bits of history ages 0..31 and y (N,) next outcome."""
    Hs, ys = [], []
    for j, sd in enumerate(seeds):
        b = np.array(gen(fam, int(sd), j), dtype=np.int8)
        t = np.arange(WARM, T + WARM)
        H = np.stack([b[t - 1 - a] for a in range(32)], 1)
        Hs.append(H); ys.append(b[t])
    return np.concatenate(Hs), np.concatenate(ys).astype(float)


def best_k(Htr, ytr, Hte, yte, k):
    base = ytr.mean()
    if k == 0:
        return (), float(((yte - base) ** 2).mean())
    best = (None, 9.0)
    w = (1 << np.arange(k))
    for ages in itertools.combinations(range(32), k):
        ctr = Htr[:, ages] @ w; cte = Hte[:, ages] @ w
        n = np.bincount(ctr, minlength=1 << k); s = np.bincount(ctr, weights=ytr, minlength=1 << k)
        p = np.where(n > 0, (s + base) / (n + 1), base)
        b = float(((yte - p[cte]) ** 2).mean())
        if b < best[1]:
            best = (ages, b)
    return best


def main():
    rng = np.random.default_rng(7)
    res, rows = {}, []
    for fam in ['periodic', 'lfsr', 'ca30', 'ca110', 'prime', 'pi']:
        tr = rng.integers(1, 2 ** 32, 45); te = rng.integers(1, 2 ** 32, 45)
        Htr, ytr = design(fam, tr); Hte, yte = design(fam, te)
        res[fam] = {}
        for k in range(0, 5):
            if fam == 'periodic' and k > 1:
                # per-question periods differ; a single shared table is not the right trader. Use per-period best.
                pass
            ages, b = best_k(Htr, ytr, Hte, yte, k)
            res[fam][k] = dict(ages=list(ages), brier=b, gain_vs_prior=0.25 - b)
            rows.append(dict(family=fam, k_reads=k, best_ages=' '.join(map(str, ages)), brier_test=round(b, 5),
                             gain_vs_0p25=round(0.25 - b, 5)))
            print(fam, k, ages, round(b, 4), flush=True)
    # periodic per question: one trader per question reads age p-1
    per = []
    for p in range(2, 17):
        b = np.array(s_periodic(3, p)); per.append(dict(p=p, base_brier=(1 / p) * (1 - 1 / p)))
    res['periodic_per_question'] = dict(note='per-question trader reading age p-1 predicts exactly (Brier 0); '
                                        'base-rate Brier averaged over p=2..16', base_brier_mean=float(np.mean([d['base_brier'] for d in per])))
    json.dump(res, open(os.path.join(HERE, 'streams_check.json'), 'w'), indent=1)
    with open(os.path.join(HERE, 'streams_check.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
