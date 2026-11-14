"""Multiply-accumulates (MACs; FLOPs = 2 x MACs) per pixel for P13's nets, the equal-test-FLOP step t_eq,
and training FLOPs per step. Elementwise ops (tanh, ReLU, adds, scores) are ignored for every arm.

  python flops.py            prints the table for unit test (g)
"""
import json, math, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SIDES = {9: 24, 13: 32, 19: 44, 33: 72, 59: 124, 99: 204, 201: 408}
J_TRAIN = 5


def nlev(side):
    return max(1, math.ceil(math.log2(max(side, 2))))


def dtr_macs(w=128, c=3):
    """(init, step, head) MACs per pixel for DT-recall 2D"""
    return 3 * w * 9, (w + c) * w * 9 + 4 * w * w * 9, (w * 32 + 32 * 8 + 8 * 2) * 9


def trader_macs(J, w=224, we=32, insq=0):
    """per-step MACs per pixel for Trader2D at J levels (the emb conv is recomputed every step)"""
    nb = 2 if insq else 1
    m = 3 * we * 9 + (we + nb + 3) * w * 9
    for j in range(J):
        m += (we + nb + 2) * w * 9 + (2 * w * 9 if j else 0) + w * w + 6 * w + w * 32 + 32
    return m


def jlevels(side, jopt):
    J = nlev(side)
    return (min(J, J_TRAIN) if jopt == "cap" else J), (J_TRAIN / J if jopt == "scale" else 1.0)


def mref_test_macs(n, T=None):
    """cumulative MACs per pixel of M-ref (DT-recall w=128) to T = 12n steps plus one readout"""
    T = T or 12 * n
    i, s, h = dtr_macs()
    return i + s * T + h


def t_eq(n, jopt="none", w=224, insq=0):
    """Trader2D step at which its cumulative test MACs equal M-ref's at 12n"""
    J, _ = jlevels(SIDES[n], jopt)
    return int(mref_test_macs(n) // trader_macs(J, w, insq=insq))


def train_macs_per_step(arm, D=30, w=224, insq=0, bs=50, side=24, alpha=0.01, prog=False, gradckpt=False):
    """expected training MACs per optimizer step over the whole batch (fwd = 1, bwd = 2 x fwd)"""
    px = bs * side * side
    if arm == "dtr":
        i, s, h = dtr_macs()
        T = 30
        full = 3 * (i + T * s + h)                  # full T=30 loss with grad
        # progressive: n ~ U[0,T-1] no-grad steps, k ~ U[1,T-n] grad steps (expectations)
        En = (T - 1) / 2
        Ek = sum((T - n + 1) / 2 for n in range(T)) / T
        progm = i + En * s + 3 * (i + Ek * s + h)
        return px * (full + progm) if alpha != 1 else px * progm
    f = trader_macs(nlev(side), w, insq=insq)
    if arm == "econ":
        return px * f * D * (1 + 3)                  # no-grad rollout, then settlement fwd + bwd on every step
    # bptt: full rollout with grad (+1 recompute with gradient checkpointing), plus the progressive part
    per = 3 + (1 if gradckpt else 0)
    m = D * per
    if prog:
        T = D
        En = (T - 1) / 2
        Ek = sum((T - n + 1) / 2 for n in range(T)) / T
        m = (D * per if alpha < 1 else 0) + En + Ek * per
    return px * f * m


def table():
    out = {"dtr_w128_macs_per_pix": dict(zip(["init", "step", "head"], dtr_macs())), "sizes": {}}
    for n, side in SIDES.items():
        row = {"pixels": side, "J": nlev(side), "mref_12n_macs_per_pix": mref_test_macs(n)}
        for jopt in ["none", "scale", "cap"]:
            J, sc = jlevels(side, jopt)
            row[jopt] = {"J_used": J, "trader_step_macs": trader_macs(J), "ratio_to_dtr_step": round(trader_macs(J) / dtr_macs()[1], 3),
                         "t_eq": t_eq(n, jopt), "12n": 12 * n}
        out["sizes"][n] = row
    out["train_macs_per_step"] = {
        "dtr": train_macs_per_step("dtr"), "econ": train_macs_per_step("econ"),
        "bptt_dscov": train_macs_per_step("bptt", gradckpt=True),
        "bptt_progcov": train_macs_per_step("bptt", prog=True, gradckpt=True)}
    return out


if __name__ == "__main__":
    print(json.dumps(table(), indent=1))
