"""Post hoc (not pre-registered; written after the test numbers): collapse per spawn epoch in the auction arms, the
bank atoms matched by the surviving learned events, the encoder learning rate at each spawn, and exact bounds.
Descriptive only.  python posthoc.py > results/posthoc.txt"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = f"{HERE}/vmout/results"
bank = json.load(open(f"{RES}/s6_bank_T1_seed0.json"))["atoms"]


def const(p, thr=0.01):
    return min(p, 1 - p) <= thr


def lr(step, total=20000, base=2e-3, warm=200):  # train_s6.cosine_lr
    return base * min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total)))


def cp_upper(x, n, a=0.05):
    """one-sided exact (Clopper-Pearson) upper bound on a binomial rate"""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        m = (lo + hi) / 2
        cdf = sum(math.comb(n, k) * m ** k * (1 - m) ** (n - k) for k in range(x + 1))
        lo, hi = (m, hi) if cdf > a else (lo, m)
    return hi


print("encoder LR at spawn: step 0 -> %.2e (after 200-step warm-up 2.0e-03), 7000 -> %.2e, 14000 -> %.2e, 18000 -> %.2e"
      % (lr(0), lr(7000), lr(14000), lr(18000)))
print()
for arm in ("auction", "auction_both"):
    print(f"== {arm}: per learned slot, (spawn step, steps until first constant on the val trace (None = never),"
          f" range of val p1 from 500 to 4000 steps after the spawn)")
    ep = {}
    for s in range(5):
        r = json.load(open(f"{RES}/m2_{arm}_T1_seed{s}.json"))
        tr = {t[0]: t[1:] for t in r["p1_trace_val"]}
        resp = [(x["slot"], x["step"]) for x in r["log"]["respawns"]]
        for k in range(3):
            births = [0] + [st for sl, st in resp if sl == f"learned{k}"]
            row = []
            for i, b in enumerate(births):
                e = births[i + 1] if i + 1 < len(births) else 20001
                # the trace at a respawn step is taken before that step's settlement, so it belongs to the old encoder
                first = next((t for t in sorted(tr) if (t > b or b == 0) and t <= e and const(tr[t][k])), None)
                late = [tr[t][k] for t in sorted(tr) if b + 500 <= t <= min(e, b + 4000)]  # before any rate gradient
                row.append((b, None if first is None else first - b,
                            None if not late else round(max(late) - min(late), 3)))
                ep.setdefault(min(i, 2), []).append(first is None)
            lv = r["nmi"]["live"][k]
            if arm == "auction_both":
                a_c, a_n = bank[lv["argmax_cell"]]["spec"], bank[lv["argmax_noncell"]]["spec"]
                print(f"  s{s} slot{k} {row}  end: test p1 {lv['p1']:.3f} G {lv['G']:+.4f} NMI cell {lv['max_nmi_cell']:.2f} {a_c}"
                      f" / non-cell {lv['max_nmi_noncell']:.2f} {a_n}")
            else:
                print(f"  s{s} slot{k} {len(births)} spawns, last at {births[-1]}; end test p1 {lv['p1']:.3f}")
    if arm == "auction_both":
        for i, v in sorted(ep.items()):
            print(f"  spawn epoch {i}: never constant on val in {sum(v)}/{len(v)} slots")
    print()

print("== cvae arms: learned slots non-constant at the end (test, 0.01)")
for arm in ("cvae", "cvae_init", "cvae_warm", "cvae_both"):
    n = 0
    for s in range(10):
        r = json.load(open(f"{RES}/m2_{arm}_T1_seed{s}.json"))
        for k, lv in enumerate(r["nmi"]["live"]):
            if not const(lv["p1"]):
                n += 1
                a_n = bank[lv["argmax_noncell"]]["spec"]
                tr = [(t[0], round(t[k + 1], 2)) for t in r["p1_trace_val"] if t[0] in (0, 50, 100, 500, 1000, 2000, 4000, 19500)]
                print(f"  {arm} s{s} slot{k}: test p1 {lv['p1']:.3f} G {lv['G']:+.4f} NMI non-cell {lv['max_nmi_noncell']:.2f} {a_n}; val trace {tr}")
    print(f"  {arm}: {n}/30 non-constant; one-sided 95% exact upper bound on the per-slot rate {cp_upper(n, 30):.3f}")
