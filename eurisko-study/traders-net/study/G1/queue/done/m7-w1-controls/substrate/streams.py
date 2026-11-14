"""Synthetic binary question streams, re-implemented from logical-soup/src/questions.bend.

Families (codes as in the Bend2 version):
  0 periodic, 1 LFSR-16, 2 CA rule 30, 3 CA rule 110, 4 primality of n,
  5 pi-digit parity, 6 A AND B, 7 A OR B.
Everything is deterministic given the seed. Differences from Bend2: pi uses real
decimal digits of pi (not a 64-digit loop), each question gets its own seed-derived
parameters, and compounds pair two base questions of different families.
"""
import sys
import numpy as np

sys.set_int_max_str_digits(0)

FAMILIES = ["periodic", "lfsr16", "ca30", "ca110", "prime", "pi", "and", "or"]
N_FAM = len(FAMILIES)


def periodic(period, phase, T):
    return np.array([(phase + t) % period == 0 for t in range(T)], dtype=np.uint8)


def lfsr16(state, T):
    # x^16+x^14+x^13+x^11+1, right-shifting Galois form (tap 0xB400 = 46080).
    state = max(1, state & 0xFFFF)
    out = np.empty(T, dtype=np.uint8)
    for t in range(T):
        out[t] = state & 1
        state = (state >> 1) ^ ((state & 1) * 46080)
    return out


def ca(rule, row, T, width=32):
    # Emit a 32-cell periodic-boundary row one cell at a time, then step the automaton.
    cells = np.array([(row >> i) & 1 for i in range(width)], dtype=np.uint8)
    if cells.sum() == 0:
        cells[0] = 1
    out = np.empty(T, dtype=np.uint8)
    rule_bits = np.array([(rule >> i) & 1 for i in range(8)], dtype=np.uint8)
    t = 0
    while t < T:
        n = min(width, T - t)
        out[t:t + n] = cells[:n]
        t += n
        idx = 4 * np.roll(cells, 1) + 2 * cells + np.roll(cells, -1)
        cells = rule_bits[idx]
    return out


def _sieve(n):
    s = np.ones(n + 1, dtype=bool)
    s[:2] = False
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]:
            s[i * i::i] = False
    return s


_PRIME = _sieve(65535)


def prime(n0, T):
    n = (n0 + np.arange(T)) & 0xFFFF
    return _PRIME[n].astype(np.uint8)


_PI = None


def pi_digits(n):
    """First n decimal digits of pi (leading 3 included), Gibbons-style Machin via integers."""
    global _PI
    if _PI is not None and len(_PI) >= n:
        return _PI[:n]
    # Machin: pi = 16 atan(1/5) - 4 atan(1/239), fixed point with guard digits.
    guard = 20
    scale = 10 ** (n + guard)

    def atan_inv(x):
        total, term, k, x2, sign = 0, scale // x, 1, x * x, 1
        while term:
            total += sign * (term // k)
            term //= x2
            k += 2
            sign = -sign
        return total

    p = 16 * atan_inv(5) - 4 * atan_inv(239)
    s = str(p)[:n]
    _PI = np.frombuffer(s.encode(), dtype=np.uint8) - ord("0")
    return _PI


def pi_parity(offset, T):
    d = pi_digits(offset + T)
    return (d[offset:offset + T] % 2).astype(np.uint8)


def make_questions(Q=64, T=24000, seed=0):
    """Return bits [Q, T] (uint8), family id per question [Q], and a description list."""
    assert Q % N_FAM == 0, "Q must be a multiple of 8"
    k = Q // N_FAM
    rng = np.random.default_rng(seed)
    bits, fam, desc = [], [], []
    base = []  # indices of base questions, grouped by family
    for f in range(6):
        idx = []
        for j in range(k):
            if f == 0:
                period = int(2 + (j * 5 + rng.integers(0, 3)) % 15)  # 2..16
                phase = int(rng.integers(0, period))
                b = periodic(period, phase, T); d = f"periodic p={period} phase={phase}"
            elif f == 1:
                st = int(rng.integers(1, 65536)); b = lfsr16(st, T); d = f"lfsr16 state={st}"
            elif f == 2:
                r = int(rng.integers(1, 2 ** 32)); b = ca(30, r, T); d = f"ca30 row={r:#x}"
            elif f == 3:
                r = int(rng.integers(1, 2 ** 32)); b = ca(110, r, T); d = f"ca110 row={r:#x}"
            elif f == 4:
                n0 = int(rng.integers(2, 65536)); b = prime(n0, T); d = f"prime n0={n0}"
            else:
                off = int(rng.integers(0, 2000)); b = pi_parity(off, T); d = f"pi-parity offset={off}"
            idx.append(len(bits)); bits.append(b); fam.append(f); desc.append(d)
        base.append(idx)
    for f, name in ((6, "and"), (7, "or")):
        for j in range(k):
            fa, fb = rng.choice(6, size=2, replace=False)
            a = base[fa][int(rng.integers(0, k))]; b_ = base[fb][int(rng.integers(0, k))]
            b = (bits[a] & bits[b_]) if f == 6 else (bits[a] | bits[b_])
            bits.append(b.astype(np.uint8)); fam.append(f)
            desc.append(f"{name}(q{a} {FAMILIES[fam[a]]}, q{b_} {FAMILIES[fam[b_]]})")
    return np.stack(bits), np.array(fam), desc


if __name__ == "__main__":
    b, f, d = make_questions(64, 4000, 1)
    for q in range(0, 64, 4):
        print(q, FAMILIES[f[q]], d[q], "mean=%.3f" % b[q].mean())
