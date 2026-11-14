"""Shared analysis helpers (used by the on-VM headroom gate and by analyze.py)."""
import json
import numpy as np
from halting import Obs, stops_for, rule_stat, knob_grid, SIMPLE_RULES

SIMPLE_FIELDS = ("kl1", "kl2", "conf_sum", "conf_min", "ent")


def load_pool(dirn, name):
    O = dict(np.load(f"{dirn}/obs_{name}.npz", allow_pickle=False))
    info = json.loads(str(O.pop("info")))
    M = {k[5:]: O.pop(k) for k in list(O) if k.startswith("meta_")}
    try:
        E = dict(np.load(f"{dirn}/eval_{name}.npz"))
    except FileNotFoundError:
        E = {}
    nc = len(info["c_grid"])
    for k in list(O):
        if k.endswith("_bits"):
            O[k] = np.unpackbits(O[k], axis=2)[:, :, :nc].astype(bool)
    return O, E, M, info


def obs_view(O, info, hseed=None):
    d = {k: O[k] for k in SIMPLE_FIELDS}
    if hseed is not None:
        pre = f"h{hseed}_"
        for k, v in O.items():
            if k.startswith(pre):
                d["h_" + k[len(pre):]] = v
    return Obs(d, info["Tc"], info["head_times"])


def rule_knobs(dev_views, rule, Tc=None, info=None):
    if rule == "R0":
        return np.arange(1, Tc + 1)
    if rule == "R13":
        return None
    stats = []
    for ob in dev_views:
        s = rule_stat(ob, rule)
        stats.append(s[:, 1:ob.Tc][np.isfinite(s[:, 1:ob.Tc])])
    return knob_grid(stats)


def at_stops(A, stops):
    """A (N, T+1), stops (N, K) -> (N, K)."""
    return np.take_along_axis(A, stops, axis=1)


def oracle_cost_at(first_ok, cost_of_t, a_star_frac, N):
    """Cheapest subset of solvable examples (each at its earliest exact step) reaching a* accuracy.
    first_ok: (N,) earliest exact t or -1. Returns mean cost over N (unsolved/unused cost 0)."""
    c = np.sort(np.array([cost_of_t(t) for t in first_ok if t >= 0], float))
    need = int(np.ceil(a_star_frac * N - 1e-9))
    if need > len(c):
        return np.nan
    return c[:need].sum() / N


def cost_at(acc, cost, a_star):
    ok = acc >= a_star - 1e-12
    return cost[ok].min() if ok.any() else np.nan


def maze_headroom(O, E, info, dev_views=None):
    """Headroom gate on one dev pool: 1 - oracle/envelope in steps (= pixel-steps at one size)
    at a* = 0.98 x best fixed-T exact match; envelope over the simple rules."""
    ob = obs_view(O, info)
    Tc = info["Tc"]
    exact = E["exact"].astype(bool)
    N = exact.shape[0]
    fixed_acc = exact[:, 1:].mean(0)
    best = fixed_acc.max()
    a_star = 0.98 * best
    rows = {}
    for r in SIMPLE_RULES:
        kn = rule_knobs(dev_views or [ob], r, Tc)
        st = stops_for(ob, r, kn)
        acc = at_stops(exact, st).mean(0)
        cost = st.mean(0).astype(float)
        rows[r] = cost_at(acc, cost, a_star)
    env_rule = min(rows, key=lambda k: rows[k] if np.isfinite(rows[k]) else np.inf)
    env = rows[env_rule]
    first = np.where(exact[:, 1:].any(1), exact[:, 1:].argmax(1) + 1, -1)
    orc = oracle_cost_at(first, lambda t: t, a_star, N)
    return {"a_star": a_star, "best_fixed_acc": float(best), "best_fixed_T": int(fixed_acc.argmax() + 1),
            "rule_cost_at_a_star": {k: float(v) for k, v in rows.items()}, "envelope_rule": env_rule,
            "envelope_steps": float(env), "oracle_steps": float(orc), "headroom": float(1 - orc / env),
            "n": int(N), "frac_never_solved": float((first < 0).mean())}
