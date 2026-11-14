"""Causal halting rules applied post hoc to stored observations.

Every rule is "stop at the first decision time t >= 1 where s(t) <= knob", forced stop at
T_cap, where s(t) is a statistic computed from whitelisted obs fields at times <= t.
stop(knob) = #{t : running_min(s)(t) > knob}  (s[0] = +inf, so stops are >= 1).
"""
import numpy as np

INF = np.inf

# whitelist of obs fields each rule family may read (enforced by rule_stat)
RULE_FIELDS = {
    "R2": {"kl1", "kl2"}, "R3sum": {"conf_sum"}, "R3min": {"conf_min"}, "R3ent": {"ent"},
    "R4": {"h_R4"}, "R4c": {"h_R7"}, "R4e": {"h_R4", "ent"}, "R7": {"h_R7"}, "R7abl": {"h_R7abl"},
    "R8": {"h_R8"}, "R9": {"h_R9"}, "R9s": {"h_R9s"}, "R7TT": {"h_R7TT"}, "R11": {"h_R11"},
    "R12": {"h_R12"}, "R12TT": {"h_R12TT"}, "R5_30": {"h_R5_30"}, "R5_100": {"h_R5_100"},
    "R13": {"h_R13_bits"},
}
H_LIST = [1, 2, 4, 8, 16, 32, 64, "cap"]
HK_LIST = [2, 4, 8, 16, 32, 64, "cap"]


class Obs:
    """Read-only view of one pool's obs; records which fields a rule touched."""
    def __init__(self, d, Tc, head_times):
        self.d, self.Tc, self.head_times = d, int(Tc), np.asarray(head_times)
        self.touched = set()

    def __getitem__(self, k):
        self.touched.add(k)
        return self.d[k]

    def __contains__(self, k):
        return k in self.d


def _full(obs, grid_vals):
    """Place grid values (N, G) into a full (N, Tc+1) array, +inf off the grid."""
    N = grid_vals.shape[0]
    out = np.full((N, obs.Tc + 1), INF)
    out[:, obs.head_times] = grid_vals
    out[:, 0] = INF
    return out


def heff(obs, hlist):
    t = obs.head_times.astype(float)
    cols = [np.minimum(float(H), obs.Tc - t) if H != "cap" else (obs.Tc - t) for H in hlist]
    return np.stack(cols, 1)  # (G, nH)


def voc_stat(obs, V, hlist):
    he = heff(obs, hlist)[None]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(he > 0, V / np.maximum(he, 1e-9), -INF)
    return _full(obs, r.max(2))


def rule_stat(obs, rule):
    """Returns s (N, Tc+1) for knob rules, or ('bits', stops) handled in stops_for."""
    fam = rule.split(":")[0]
    before = set(obs.touched)
    if fam == "R2":
        _, lag, pat = rule.split(":")
        kl = obs[f"kl{lag}"].astype(float).copy()
        kl[:, 0] = INF
        pat = int(pat)
        N, T1 = kl.shape
        s = np.full_like(kl, INF)
        for t in range(pat, T1):
            s[:, t] = kl[:, t - pat + 1:t + 1].max(1)
    elif fam == "R3sum":
        s = -obs["conf_sum"].astype(float)
    elif fam == "R3min":
        s = 1 - obs["conf_min"].astype(float)
    elif fam == "R3ent":
        s = obs["ent"].astype(float).copy()
    elif fam == "R4":
        s = _full(obs, obs["h_R4"][:, :, 0].astype(float))
    elif fam == "R4c":
        V = obs["h_R7"].astype(float)[:, :, [0, 7]]
        s = voc_stat(obs, V, [1, "cap"])
    elif fam == "R4e":
        v1 = obs["h_R4"][:, :, 0].astype(float)
        ent = obs["ent"].astype(float)[:, obs.head_times]
        rem = (obs.Tc - obs.head_times).astype(float)[None]
        with np.errstate(divide="ignore", invalid="ignore"):
            e = np.where(rem > 0, ent / np.maximum(rem, 1e-9), -INF)
        s = _full(obs, np.maximum(v1, e))
    elif fam in ("R7", "R7abl", "R9s", "R7TT"):
        s = voc_stat(obs, obs[f"h_{fam}"].astype(float), H_LIST)
    elif fam in ("R8", "R9"):
        s = voc_stat(obs, obs[f"h_{fam}"].astype(float), HK_LIST)
    elif fam in ("R11", "R12", "R12TT"):
        s = _full(obs, -obs[f"h_{fam}"][:, :, 0].astype(float))
    elif fam in ("R5_30", "R5_100"):
        lam = 1 / (1 + np.exp(-obs[f"h_{fam}"][:, :, 0].astype(float)))
        surv = np.cumprod(1 - lam, axis=1)       # grid = every step on T2m
        s = _full(obs, surv)
    else:
        raise ValueError(rule)
    s = s.copy()
    s[:, 0] = INF
    s[:, obs.Tc] = -INF  # forced stop at T_cap
    new = obs.touched - before
    allowed = RULE_FIELDS[fam]
    assert new <= allowed, (rule, new, allowed)
    return s


def stops_from_stat(s, knobs):
    """s (N, T+1) with s[:,0]=inf; knobs sorted ascending. Returns (N, K) stop times."""
    m = np.minimum.accumulate(s, axis=1)
    N = s.shape[0]
    out = np.empty((N, len(knobs)), dtype=np.int32)
    negk = -np.asarray(knobs, dtype=float)
    for i in range(N):
        out[i] = np.searchsorted(-m[i], negk, side="left")
    return np.minimum(out, s.shape[1] - 1)


def stops_qact(obs):
    bits = obs["h_R13_bits"]              # (N, G, K) bool
    N, G, K = bits.shape
    any_ = bits.any(1)
    first = bits.argmax(1)                # (N, K) grid index
    t = obs.head_times[first]
    return np.where(any_, t, obs.Tc).astype(np.int32)


def stops_for(obs, rule, knobs):
    if rule == "R13":
        return stops_qact(obs)
    if rule == "R0":
        return np.broadcast_to(np.minimum(np.asarray(knobs, dtype=np.int32), obs.Tc),
                               (obs.d["kl1"].shape[0], len(knobs))).copy()
    return stops_from_stat(rule_stat(obs, rule), knobs)


def knob_grid(dev_stats, n=200):
    """200 knob values from quantiles of a rule's finite statistic over dev decision points."""
    v = np.concatenate([x[np.isfinite(x)] for x in dev_stats])
    lv = np.concatenate([np.geomspace(1e-4, 0.5, n // 2), np.linspace(0.5, 1, n - n // 2 + 1)[1:]])
    k = np.quantile(v, lv) if len(v) else np.zeros(1)
    k = np.unique(np.concatenate([k, [v.min() - 1e-9 if len(v) else 0]]))
    return k


SIMPLE_RULES = ["R0", "R2:1:1", "R2:1:8", "R2:2:1", "R2:2:8", "R3sum", "R3min", "R3ent"]
HEAD_RULES_T2M = ["R4", "R4c", "R4e", "R5_30", "R5_100", "R11", "R12", "R13", "R7", "R7abl", "R8"]
HEAD_RULES_MAZE = ["R4", "R4c", "R4e", "R11", "R12", "R12TT", "R13", "R7", "R7abl", "R8", "R9", "R9s", "R7TT"]


def head_decisions(rule, t, Tc, maze_grid=4):
    """Number of head evaluations spent by a head rule that stops at t (for cost)."""
    if rule in SIMPLE_RULES or rule.startswith("R2") or rule.startswith("R3") or rule == "R0":
        return 0
    return t // maze_grid if maze_grid > 1 else t
