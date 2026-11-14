"""Cost sharing for a shared trunk (T1 lemma 3). Importable by M7 and K1.

Rule (THEORY lemma 3): every FLOP is charged to someone, and the sum of charges equals the FLOPs
actually spent (budget balance).
  * A trader's own FLOPs (expert MLP, trader head, update) are separable: it pays them in full.
  * A shared FLOP (trunk, window features) is split among the traders that consume it on that
    token/question, equally per consumer (Shapley value of the per-token "any user" game, which
    equals Aumann-Shapley over usage when each consumer uses one unit).
  * A per-trader FLOP that runs whether or not the trader is used (its router column) is rent:
    charged on every token seen, routed or not.
Marginal-cost pricing is provided for comparison; it leaves the trunk unfunded.

FLOP convention: one multiply-add = 2 FLOPs; forward only. Training ~ 3x forward (fwd + bwd).
"""
import numpy as np


# ----------------------------------------------------------------------------- generic rules
def aumann_shapley(grad, x, n=256, m=4):
    """AS charges x_i * int_0^1 dC/dx_i(t x) dt. grad(x) -> dC/dx.
    Midpoint rule after t = u^m (m=4), which removes the integrable t^(r-1) singularity at t=0
    of concave (economies-of-scale) costs; plain midpoint converges slowly there."""
    x = np.asarray(x, float)
    u = (np.arange(n) + 0.5) / n
    t = u ** m
    w = m * u ** (m - 1)
    return x * np.mean([wi * grad(ti * x) for ti, wi in zip(t, w)], axis=0)


def marginal(grad, x):
    x = np.asarray(x, float)
    return x * grad(x)


def shapley(cost_of_set, K):
    """Exact Shapley value of a cost game over K players; cost_of_set(frozenset) -> float."""
    from math import factorial
    v = {}
    for m in range(1 << K):
        v[m] = cost_of_set(frozenset(i for i in range(K) if m >> i & 1))
    phi = np.zeros(K)
    for i in range(K):
        for m in range(1 << K):
            if m >> i & 1:
                continue
            s = bin(m).count("1")
            phi[i] += factorial(s) * factorial(K - s - 1) / factorial(K) * (v[m | 1 << i] - v[m])
    return phi


def union_shapley(users, F):
    """Public-good components: component f costs F[f] if any of users[f] is active.
    Shapley = equal split of each component among its users (closed form)."""
    K = 1 + max(max(u) for u in users if u)
    phi = np.zeros(K)
    for u, c in zip(users, F):
        for i in u:
            phi[i] += c / len(u)
    return phi


# ----------------------------------------------------------------------------- M7: MoE FLOPs
def moe_token_flops(d=64, L=2, H=4, T=64, W=32, E=8, k=2, h_e=None, mlp=4):
    """Per-token forward FLOPs of an S0/wealth-net-shaped transformer whose MLPs are MoE layers.
    T = tokens attended over (questions per example). h_e = expert hidden width
    (default: fine-grained, k experts together = one dense MLP, h_e = mlp*d/k)."""
    if h_e is None:
        h_e = mlp * d // k
    embed = 2 * W * d + d                                # window linear + qid add
    attn = (2 * d * 3 * d) + (2 * T * d) + (2 * T * d) + (2 * d * d) + 5 * H * T + 10 * d  # qkv, QK, AV, o, softmax, LN
    readout = 10 * d + 2 * d                             # final LN + readout dot
    trunk = embed + L * attn + readout
    router_col = 2 * d                                   # one expert's router column, per token, per layer
    expert = 2 * (2 * d * h_e) + h_e                     # fc1 + fc2 + relu, per routed token
    dense_mlp = 2 * (2 * d * mlp * d) + mlp * d
    return {"trunk": trunk, "attn_per_layer": attn, "embed": embed, "readout": readout,
            "router_col": router_col, "expert": expert, "dense_mlp": dense_mlp,
            "h_e": h_e, "d": d, "L": L, "E": E, "k": k, "T": T}


def moe_charges(counts, n_tokens, fl, rule="shapley"):
    """Charges per (layer, expert), from routing counts.
    counts[l, e] = tokens routed to expert e of layer l; n_tokens = tokens processed.
    rule: 'shapley' (budget balanced: own FLOPs + trunk/(L*k) per routed token + router-column
    rent on every token seen) or 'marginal' (own FLOPs + rent; the trunk is left unfunded).
    Returns (charges[L, E], total_flops_spent)."""
    counts = np.asarray(counts, float)
    L, E = counts.shape
    k = fl["k"]
    own = counts * fl["expert"]
    rent = np.full((L, E), n_tokens * fl["router_col"], float)
    if rule == "shapley":
        trunk = counts * fl["trunk"] / (L * k)             # L*k expert activations per token
    elif rule == "marginal":
        trunk = np.zeros((L, E))
    else:
        raise ValueError(rule)
    total = n_tokens * (fl["trunk"] + L * E * fl["router_col"]) + counts.sum() * fl["expert"]
    return own + rent + trunk, total


# ----------------------------------------------------------------------------- K1: trader FLOPs
def k1_trader_flops(kind, k=None, W=None):
    """Own and shared FLOPs per question traded by one trader.
    pattern-k: shared = read k bits into an index (k shift-or ops = 2k); own = table lookup,
      Kelly stake from belief and price (~10), count update (~4).
    logistic-W: shared = +-1 encoding of the window (W); own = dot (2W) + sigmoid (~4) +
      Kelly stake (~10) + SGD update (2W + 2)."""
    if kind == "pattern":
        return {"shared": 2 * k, "own": 14, "shared_key": f"pattern{k}"}
    if kind == "logistic":
        return {"shared": W, "own": 4 * W + 16, "shared_key": f"window{W}"}
    raise ValueError(kind)


def k1_charges(trader_kinds, trades):
    """trader_kinds: list of (kind, param) per trader; trades: bool [N, Q] (who trades q this step).
    Equal split of each (shared_key, question) feature among the traders that consume it.
    Returns (charge per trader, FLOPs spent)."""
    trades = np.asarray(trades, bool)
    N, Q = trades.shape
    specs = [k1_trader_flops(kd, k=p, W=p) if kd == "logistic" else k1_trader_flops(kd, k=p) for kd, p in trader_kinds]
    keys = sorted({s["shared_key"] for s in specs})
    charge = np.zeros(N)
    spent = 0.0
    for key in keys:
        members = np.array([s["shared_key"] == key for s in specs])
        cost = next(s["shared"] for s in specs if s["shared_key"] == key)
        users = trades & members[:, None]               # [N,Q]
        n_users = users.sum(0)                          # per question
        active = n_users > 0
        spent += cost * active.sum()
        share = np.where(active, cost / np.maximum(n_users, 1), 0.0)
        charge += (users * share[None, :]).sum(1)
    own = np.array([s["own"] for s in specs], float)
    charge += own * trades.sum(1)
    spent += (own * trades.sum(1)).sum()
    return charge, spent


def breakeven_price(info_nats_per_trade, flops_per_trade):
    """c* (nats per FLOP) at which a trader earning info_nats_per_trade breaks even."""
    return info_nats_per_trade / flops_per_trade
