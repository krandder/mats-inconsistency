"""Stream changes for traders-net (S0 worker 2).

Three changes to the wealth-net streams (substrate/streams.py, a verbatim copy):
1. Seeded PRNG-parity replaces pi-parity (family 5). Each question gets its own PCG64 bit
   generator from numpy SeedSequence.spawn, so no two questions share a sequence up to a shift
   (pi questions were windows of one digit string at offsets 0..2000: shifted copies).
   Bit t = top bit of the t-th raw 64-bit PCG64 output (random_raw is stream-stable in numpy).
2. A held-out periodic family (id 8, "periodic_ho"): periods 17..31, disjoint from the training
   periodic periods 2..16, all of them < W=32 so a window can see a whole cycle. Its questions
   are inactive (never sampled, trained or scored) before `release`, active from `release` on.
3. Regime shift: at every multiple b of `regime_every` (b < T), round(frac * n_eligible)
   eligible questions, drawn at random, switch to a different base family (fresh parameters,
   generator started at b). Eligible = every non-held-out question. Targets = base families
   0..5 other than the current one; compound questions (AND/OR) that are chosen become base
   questions; compounds that are not chosen keep AND/OR of their operands' *current* bits, so
   they change whenever an operand switches (logged as indirect).

Independent SeedSequence children are used for base questions, the held-out family and the
regime draws, so switching the regime shift on or off leaves the base and held-out bits before
the first boundary identical (arms are paired).

API
  s = make_streams(Q=64, T=60000, seed=1, heldout=8, release=None (=T//2),
                   regime_every=10000, regime_frac=0.25)   # regime_every=0 disables
  s.bits [Qt,T] uint8, s.fam_t [Qt,T] int8 (family per step), s.fam = s.fam0, s.active [Qt,T] bool,
  s.fam0 [Qt], s.desc [Qt] (initial), s.events (list of dicts), s.spec (dict, JSON-able)
  to_array_stream(s) -> prequential.ArrayStream (worker1 interface; needs torch)
  make_questions(Q, T, seed) -> (bits, fam0, desc): drop-in for streams.make_questions with
  prng-parity instead of pi, no held-out, no shift.
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from streams import periodic, lfsr16, ca, prime  # noqa: E402  (verbatim wealth-net generators)

FAMILIES = ["periodic", "lfsr16", "ca30", "ca110", "prime", "prng", "and", "or", "periodic_ho"]
N_FAM = len(FAMILIES)
BASE = list(range(6))
HO = 8
HO_PERIODS = (17, 31)   # inclusive range, disjoint from training periods 2..16


def prng_parity(ss, T):
    """T bits from an independent PCG64 stream seeded by SeedSequence `ss` (or an int)."""
    bg = np.random.PCG64(ss)
    return (bg.random_raw(T) >> np.uint64(63)).astype(np.uint8)


def _base_bits(f, rng, T, j=0):
    """One base-family question of family f with parameters drawn from rng."""
    if f == 0:
        period = int(2 + (j * 5 + rng.integers(0, 3)) % 15)  # 2..16, as wealth-net
        phase = int(rng.integers(0, period))
        return periodic(period, phase, T), f"periodic p={period} phase={phase}"
    if f == 1:
        st = int(rng.integers(1, 65536)); return lfsr16(st, T), f"lfsr16 state={st}"
    if f == 2:
        r = int(rng.integers(1, 2 ** 32)); return ca(30, r, T), f"ca30 row={r:#x}"
    if f == 3:
        r = int(rng.integers(1, 2 ** 32)); return ca(110, r, T), f"ca110 row={r:#x}"
    if f == 4:
        n0 = int(rng.integers(2, 65536)); return prime(n0, T), f"prime n0={n0}"
    if f == 5:
        key = int(rng.integers(0, 2 ** 63)); return prng_parity(key, T), f"prng key={key}"
    if f == HO:
        period = int(rng.integers(HO_PERIODS[0], HO_PERIODS[1] + 1))
        phase = int(rng.integers(0, period))
        return periodic(period, phase, T), f"periodic_ho p={period} phase={phase}"
    raise ValueError(f)


class Streams:
    """Arrays: bits [Qt,T], fam_t [Qt,T], active [Qt,T], fam0, desc, events, spec.
    Use to_array_stream(s) to get worker1's prequential Stream (position convention)."""
    pass


def to_array_stream(s):
    """prequential.ArrayStream over s: position i predicts bits[:, i+W] from bits[:, i:i+W];
    live[i, q] = active[q, i+W]; held-out window entries at times < release read 0.5, so a
    held-out question is never context for any question before it is released. n_fam = 9."""
    import torch
    from prequential import ArrayStream

    class _HOStream(ArrayStream):
        def window(self, idx):
            w = super().window(idx)
            if s.n_main < s.Q:
                idx_t = torch.as_tensor(idx, dtype=torch.long)
                times = idx_t[:, None] + torch.arange(self.W)[None, :]            # [n,W]
                hid = (times < s.release)[:, None, :].expand(-1, s.Q - s.n_main, -1)
                w[:, s.n_main:, :] = torch.where(hid, torch.full_like(w[:, s.n_main:, :], 0.5),
                                                 w[:, s.n_main:, :])
            return w

    return _HOStream(s.bits, s.fam0, s.W, live=s.active[:, s.W:].T.copy(), desc=s.desc, n_fam=N_FAM)


def make_streams(Q=64, T=60000, seed=1, heldout=8, release=None, regime_every=10000,
                 regime_frac=0.25, W=32):
    assert Q % 8 == 0, "Q must be a multiple of 8"
    k = Q // 8
    release = T // 2 if release is None else int(release)
    ss_base, ss_ho, ss_reg, ss_prng = np.random.SeedSequence(seed).spawn(4)
    rng = np.random.default_rng(ss_base)
    n_bound = len(range(regime_every, T, regime_every)) if regime_every and regime_every > 0 else 0
    prng_keys = iter(ss_prng.spawn(k + n_bound * Q))
    bits, fam0, desc, base = [], [], [], []
    for f in BASE:
        idx = []
        for j in range(k):
            if f == 5:
                b, d = prng_parity(next(prng_keys), T), "prng (spawned PCG64)"
            else:
                b, d = _base_bits(f, rng, T, j)
            idx.append(len(bits)); bits.append(b); fam0.append(f); desc.append(d)
        base.append(idx)
    ops = {}
    for f, name in ((6, "and"), (7, "or")):
        for j in range(k):
            fa, fb = rng.choice(6, size=2, replace=False)
            a = base[fa][int(rng.integers(0, k))]; b_ = base[fb][int(rng.integers(0, k))]
            ops[len(bits)] = (f, a, b_)
            bits.append(np.zeros(T, np.uint8)); fam0.append(f)
            desc.append(f"{name}(q{a} {FAMILIES[fam0[a]]}, q{b_} {FAMILIES[fam0[b_]]})")
    rho = np.random.default_rng(ss_ho)
    n_main = len(bits)
    for j in range(heldout):
        b, d = _base_bits(HO, rho, T)
        bits.append(b); fam0.append(HO); desc.append(d)
    bits = np.stack(bits); Qt = len(bits)
    fam = np.repeat(np.array(fam0, np.int8)[:, None], T, axis=1)
    active = np.ones((Qt, T), bool)
    active[n_main:, :release] = False
    events = [{"t": release, "kind": "release", "q": q, "fam": HO, "desc": desc[q]}
              for q in range(n_main, Qt)] if heldout else []
    # Regime shift: iterate boundaries in time order.
    rr = np.random.default_rng(ss_reg)
    comp = dict(ops)          # q -> (op, a, b) while q is still a compound
    if regime_every and regime_every > 0:
        n_sw = int(round(regime_frac * n_main))
        for b0 in range(regime_every, T, regime_every):
            chosen = np.sort(rr.choice(n_main, size=n_sw, replace=False))
            for q in chosen:
                q = int(q)
                cur = int(fam[q, b0])
                tgt = int(rr.choice([f for f in BASE if f != cur]))
                if tgt == 5:
                    nb, d = prng_parity(next(prng_keys), T - b0), "prng (spawned PCG64)"
                else:
                    nb, d = _base_bits(tgt, rr, T - b0, j=int(rr.integers(0, 15)))
                if q in comp:     # compound becomes a base question from b0 on
                    op, a, bb = comp.pop(q)
                    _fill_comp(bits, q, op, a, bb, 0, b0)
                bits[q, b0:] = nb; fam[q, b0:] = tgt
                events.append({"t": b0, "kind": "switch", "q": q, "from": cur, "to": tgt, "desc": d})
    # Fill compounds that stayed compounds until T.
    for q, (op, a, bb) in comp.items():
        _fill_comp(bits, q, op, a, bb, 0, T)
    for q, (op, a, bb) in ops.items():   # log indirect changes (operand switched)
        end = T if q in comp else min([e["t"] for e in events if e.get("q") == q and e["kind"] == "switch"] + [T])
        for e in events:
            if e["kind"] == "switch" and e["q"] in (a, bb) and e["t"] < end:
                events.append({"t": e["t"], "kind": "indirect", "q": q, "via": e["q"]})
    events.sort(key=lambda e: (e["t"], e["q"], e["kind"]))
    s = Streams()
    s.bits, s.fam_t, s.active, s.fam0, s.desc, s.events = bits, fam, active, np.array(fam0), desc, events
    s.fam = s.fam0
    s.n_main, s.release, s.T, s.Q, s.W = n_main, release, T, Qt, W
    s.spec = {"Q": Q, "Qt": Qt, "T": T, "seed": seed, "heldout": heldout, "release": release,
              "regime_every": regime_every, "regime_frac": regime_frac, "families": FAMILIES,
              "ho_periods": list(HO_PERIODS), "n_switch_events": sum(e["kind"] == "switch" for e in events)}
    return s


def _fill_comp(bits, q, op, a, b, t0, t1):
    x, y = bits[a, t0:t1], bits[b, t0:t1]
    bits[q, t0:t1] = (x & y) if op == 6 else (x | y)


def make_questions(Q=64, T=24000, seed=0):
    """Drop-in for streams.make_questions: prng-parity instead of pi, no held-out, no shift."""
    s = make_streams(Q, T, seed, heldout=0, regime_every=0)
    return s.bits, s.fam0.copy(), s.desc


if __name__ == "__main__":
    s = make_streams(64, 60000, 1)
    print(s.spec)
    for q in range(0, s.Q, 4):
        print(q, FAMILIES[s.fam0[q]], s.desc[q], "mean=%.3f" % s.bits[q].mean())
    print(sum(e["kind"] == "switch" for e in s.events), "switches;",
          sum(e["kind"] == "indirect" for e in s.events), "indirect")
