#!/usr/bin/env python3
"""R9-E1 audit, check 6: parameter counts by hand (from the class definitions) and by instantiating the registered classes
from a hash-checked copy of the staged source (audit/tmpsrc, deleted after the audit); econ_prec_K3's closed form on identical
reads with each seed's fitted (u, a, b), my own implementation of the formula in PREREG 1.3 (no e1v6 import for that part).
Run: PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 PYTHONPATH=/tmp/r4e3a/pylib:audit/tmpsrc R8E1_DIR=... python3 check6_params.py"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "tmpsrc")
for k in ("R5E1_DIR", "R6E1_DIR", "R7E1_DIR", "R8E1_DIR"):
    os.environ[k] = SRC
os.environ["SMALLNET_DIR"] = os.path.join(SRC, "smallnet")
sys.path.insert(0, SRC)

import torch  # noqa: E402

torch.set_num_threads(2)


# ---- hand counts from the class definitions (Linear(i, o) = i*o + o; MultiheadAttention(d) = 4 d^2 + 4 d; LayerNorm(d) = 2 d)
def lin(i, o):
    return i * o + o


def mha(d):
    return 3 * d * d + 3 * d + d * d + d


def ln(d):
    return 2 * d


NC, KD, VD, D = 36, 16, 16, 64
key = {"K1": lin(128, KD), "K3": lin(NC, KD), "K4": lin(128 + NC, KD)}
block = mha(D) + ln(D) + (ln(D) + lin(D, 2 * D) + lin(2 * D, D))         # attention + pre-LN + FFN(LN, 64->128->64)
attn_body = lin(NC, VD) + lin(KD + VD + 2, D) + lin(D, NC)               # Wv, inp, out
hand = {
    "looped_attn": attn_body + block + 3 * D + key["K3"],                  # one shared block, step_emb [3, 64]
    "attn_K3": attn_body + 3 * block + key["K3"],                          # 3 blocks
    "attn_K1": attn_body + 3 * block + key["K1"],
    "looped_lin": 4 + 4 * NC + 1 + key["K4"],                              # eta [4], P [4, 36], log_s
    "jacobi": NC + 4 + 4 * NC + NC + key["K3"],                            # lam_raw [36], eta [4], P [4, 36], s [36]
    "jacobi_prec": NC + 4 + 4 * NC + NC + 2 + key["K3"],                   # + a, b
    "econ_prec": 3,                                                        # u, a, b (uses attn_L3-K3's K3 key head, not counted)
}

# ---- instantiate the registered classes (staged copy, hashes checked against MANIFEST.sha256 before this run)
import e1v3 as E3  # noqa: E402
import e1v4 as E4  # noqa: E402
import e1v5 as E5  # noqa: E402
import wm  # noqa: E402

n = lambda *ms: int(sum(p.numel() for m in ms for p in m.parameters()))
inst = {
    "looped_attn": n(E3.KeyNet("K3", 128), E5.LoopedAttnT3P(3.4, T=3)),
    "attn_K3": n(E3.KeyNet("K3", 128), E5.AttnL3P(3.4)),
    "attn_K1": n(E3.KeyNet("K1", 128), wm.AttnL3()),
    "looped_lin": n(E3.KeyNet("K4", 128), E3.LoopedLinT4(4, 2.0, 1.0 / 24)),
    "jacobi": n(E3.KeyNet("K3", 128), E4.JacobiLinT4(4, 2.0)),
    "jacobi_prec": n(E3.KeyNet("K3", 128), E5.JacobiPrecT4(4, 2.0)),
}
claimed = {"looped_attn": 39428, "attn_K3": 106180, "jacobi_prec": 814, "econ_prec": 3}
RES = "/home/agent/workspaces/eurisko-study/li-net/r9e1/results/main"
summ = {s: json.load(open(f"{RES}/s{s}/summary.json"))["fit"] for s in (101, 102, 103)}
print("param counts: hand", hand)
print("param counts: instantiated", inst, "| K3 key head alone", n(E3.KeyNet("K3", 128)))
print("param counts: summary.json (all seeds equal):", all(summ[s]["params"] == summ[101]["params"] for s in summ), summ[101]["params"])
print("claims (PREREG / spec):", {k: (v, hand[k], inst.get(k)) for k, v in claimed.items()})


# ---- econ_prec_K3 closed form on N identical reads (unit keys, no background): my implementation of PREREG 1.3
def econ_prec_W(N, beta, u, a, b, bg=0.0):
    lam = math.log1p(math.exp(a - b * beta))                               # softplus(a - b beta)
    c = N * beta * 1.0 + bg                                                # relu(<k, k>)^2 = 1 for identical unit keys
    return N * u * beta / (lam + c)


for s, f in summ.items():
    u, a, b = f["econ_prec"]["u"], f["econ_prec"]["a"], f["econ_prec"]["b"]
    bt = f["beta_by_tau"]
    row = {}
    for t in ("0.3", "1.5", "2.5"):
        W = {N: econ_prec_W(N, bt[t], u, a, b) for N in (1, 2, 4, 8, 16)}
        row[t] = {"lambda": round(math.log1p(math.exp(a - b * bt[t])), 4), "W1": round(W[1], 4), "W16/W1": round(W[16] / W[1], 4),
                  "W8/W1": round(W[8] / W[1], 4)}
    # cross-check with e1v6's own function on a tiny synthetic bag (unit-norm identical keys)
    import e1v6 as E6
    k = torch.nn.functional.normalize(torch.randn(1, 1, 16), dim=-1).expand(1, 16, 16).contiguous()
    w = E6.econ_prec_weights(torch.full((1, 16), bt["0.3"]), k, torch.ones(1, 16, dtype=torch.bool), u, a, b)
    print(f"s{s} econ_prec closed form at calibration beta (no background reads):", row,
          "| e1v6.econ_prec_weights W(16) at tau 0.3", round(float(w.sum()), 5), "vs mine", round(econ_prec_W(16, bt["0.3"], u, a, b), 5),
          "| summary lambda_at_beta", {t: round(v, 4) for t, v in f["econ_prec"]["lambda_at_beta"].items()})
