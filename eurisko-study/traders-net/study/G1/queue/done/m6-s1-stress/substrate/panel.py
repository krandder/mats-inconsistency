"""Metric panel shared by every traders-net run (S0, worker 4). numpy only.

Every run reports the same panel:
  - prequential BCE, Brier and ECE, overall and per family      -> scores()
  - AND/OR/D-prove coherence violations before resolution        -> violations()
  - E_C(t) exploitation slopes (E_C itself is defined by M5)     -> ec_slopes()
  - wealth Gini against a shuffled-label null                    -> gini(), gini_null()
  - dead / bankrupt fraction                                     -> dead_bankrupt()
  - cumulative FLOPs                                             -> FlopCounter
  - conservation residual of the books                           -> conservation_residual()

Conventions
  p: predicted probability that the label is 1, recorded BEFORE training on that label
     (prequential: test, then train). y: labels in {0,1}. fam: integer family id per row.
  Everything takes flat arrays; callers flatten [T,Q] with fam broadcast to the same shape.
"""
import math
import numpy as np

EPS = 1e-7


# ---------------------------------------------------------------- scores
def bce(p, y):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).mean()) if p.size else float("nan")


def brier(p, y):
    p = np.asarray(p, float); y = np.asarray(y, float)
    return float(((p - y) ** 2).mean()) if p.size else float("nan")


def ece(p, y, n_bins=10):
    """Expected calibration error, equal-width bins on [0,1], weighted by bin count.
    A prediction of exactly 1.0 goes to the last bin."""
    p = np.asarray(p, float); y = np.asarray(y, float)
    if p.size == 0:
        return float("nan")
    b = np.minimum((p * n_bins).astype(int), n_bins - 1)
    cnt = np.bincount(b, minlength=n_bins)
    sp = np.bincount(b, weights=p, minlength=n_bins)
    sy = np.bincount(b, weights=y, minlength=n_bins)
    return float(np.abs(sp - sy).sum() / p.size)


def scores(p, y, fam=None, names=None, n_bins=10):
    """{'all': {n,bce,brier,ece}, '<family>': {...}} over the given prequential rows."""
    p = np.ravel(np.asarray(p, float)); y = np.ravel(np.asarray(y, float))
    out = {"all": dict(n=int(p.size), bce=bce(p, y), brier=brier(p, y), ece=ece(p, y, n_bins))}
    if fam is not None:
        fam = np.ravel(np.broadcast_to(fam, np.shape(p)) if np.ndim(fam) else fam)
        for f in np.unique(fam):
            m = fam == f
            key = names[int(f)] if names is not None else str(int(f))
            out[key] = dict(n=int(m.sum()), bce=bce(p[m], y[m]), brier=brier(p[m], y[m]),
                            ece=ece(p[m], y[m], n_bins))
    return out


# ---------------------------------------------------------------- coherence
# A constraint is a tuple (kind, *question_indices). Kinds and violation magnitude v >= 0,
# computed from prices P (probabilities) at one time:
#   ("and", c, a, b)   c = a AND b : v = relu(Pc - min(Pa,Pb)) + relu(Pa + Pb - 1 - Pc)
#   ("or",  c, a, b)   c = a OR  b : v = relu(max(Pa,Pb) - Pc) + relu(Pc - Pa - Pb)
#   ("not", c, a)      c = NOT a   : v = |Pc - (1 - Pa)|
#   ("equal", c, a)    c has the same outcome as a (twin): v = |Pc - Pa|
#   ("taut", c)        c is a tautology (A or not A, (A and B) -> A): v = 1 - Pc
#   ("contra", c)      c is a contradiction (A and not A):             v = Pc
#   ("exhaustive", a, b, c, ...) exactly one is true:                  v = |sum P - 1|
# The AND/OR bounds are the Frechet bounds, so v = 0 iff some joint distribution fits.
KINDS = ("and", "or", "not", "equal", "taut", "contra", "exhaustive")


def _viol(kind, P, idx):
    if kind == "and":
        c, a, b = idx
        return np.maximum(P[..., c] - np.minimum(P[..., a], P[..., b]), 0) + np.maximum(P[..., a] + P[..., b] - 1 - P[..., c], 0)
    if kind == "or":
        c, a, b = idx
        return np.maximum(np.maximum(P[..., a], P[..., b]) - P[..., c], 0) + np.maximum(P[..., c] - P[..., a] - P[..., b], 0)
    if kind == "not":
        c, a = idx
        return np.abs(P[..., c] - (1 - P[..., a]))
    if kind == "equal":
        c, a = idx
        return np.abs(P[..., c] - P[..., a])
    if kind == "taut":
        return 1 - P[..., idx[0]]
    if kind == "contra":
        return P[..., idx[0]]
    if kind == "exhaustive":
        return np.abs(P[..., list(idx)].sum(-1) - 1)
    raise ValueError(kind)


def violations(P, constraints, unresolved=None, tol=0.01):
    """Coherence violations before resolution.

    P: [T,Q] prices (or [Q]). unresolved: bool [T,Q], True where the question's label has
    NOT been revealed at that time (default: all unresolved). A (time, constraint) pair is
    counted only while every question in it is still unresolved.
    Returns {kind: {n, mean, max, frac_gt_tol}} plus 'all'.
    """
    P = np.atleast_2d(np.asarray(P, float))
    U = np.ones_like(P, bool) if unresolved is None else np.atleast_2d(np.asarray(unresolved, bool))
    acc = {k: [] for k in KINDS}
    for con in constraints:
        kind, idx = con[0], tuple(int(i) for i in con[1:])
        live = U[:, list(idx)].all(1)
        acc[kind].append(_viol(kind, P, idx)[live])
    out, allv = {}, []
    for k, vs in acc.items():
        if not vs:
            continue
        v = np.concatenate(vs); allv.append(v)
        out[k] = _vsum(v, tol)
    out["all"] = _vsum(np.concatenate(allv) if allv else np.zeros(0), tol)
    return out


def _vsum(v, tol):
    if v.size == 0:
        return dict(n=0, mean=float("nan"), max=float("nan"), frac_gt_tol=float("nan"))
    return dict(n=int(v.size), mean=float(v.mean()), max=float(v.max()), frac_gt_tol=float((v > tol).mean()))


# ---------------------------------------------------------------- exploitation slopes
def ec_slopes(t, E):
    """Slopes of an exploitation curve E_C(t) (cumulative profit of trader class C against the
    market; the exact E_C is M5's). Returns:
      loglog: exponent b in E ~ a t^b fitted on points with t>0 and E>0 (b<1: sublinear, the
              logical-induction criterion is that E stays bounded, so b -> 0);
      lin_last_half: least-squares slope of E vs t over the last half of the points;
      final: E at the last time.
    """
    t = np.asarray(t, float); E = np.asarray(E, float)
    m = (t > 0) & (E > 0)
    b = float(np.polyfit(np.log(t[m]), np.log(E[m]), 1)[0]) if m.sum() >= 2 else float("nan")
    h = len(t) // 2
    lin = float(np.polyfit(t[h:], E[h:], 1)[0]) if len(t) - h >= 2 else float("nan")
    return dict(loglog=b, lin_last_half=lin, final=float(E[-1]) if len(E) else float("nan"))


# ---------------------------------------------------------------- wealth
def gini(w):
    """Gini for possibly signed wealth: sum_ij |wi-wj| / (2 n sum_i |wi|).
    Equals the usual Gini when w >= 0; stays in [0,1] for signed w (|wi-wj| <= |wi|+|wj|).
    0 for all-equal wealth; 0 for an all-zero vector by convention."""
    w = np.ravel(np.asarray(w, float))
    n = w.size
    den = 2 * n * np.abs(w).sum()
    if n == 0 or den == 0:
        return 0.0
    ws = np.sort(w)
    # sum_ij |wi-wj| = 2 sum_i (2i - n + 1) ws[i]  (i from 0, sorted ascending)
    num = 2 * ((2 * np.arange(n) - n + 1) * ws).sum()
    return float(num / den)


def gini_null(wealth_fn, y, n_null=20, seed=0, groups=None):
    """Gini of wealth vs a shuffled-label null.

    wealth_fn(y) -> wealth vector computed from labels y (the caller re-runs its books on y).
    Null: labels permuted (within each group if `groups` is given, e.g. per family, so base
    rates are kept). Returns observed gini, null mean/sd, z, and a one-sided p-value
    (fraction of null >= observed, add-one smoothed).
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    g_obs = gini(wealth_fn(y))
    null = []
    for _ in range(n_null):
        ys = y.copy()
        if groups is None:
            ys = rng.permutation(ys.ravel()).reshape(y.shape)
        else:
            gr = np.broadcast_to(groups, y.shape)
            flat, gf = ys.ravel(), gr.ravel()
            for g in np.unique(gf):
                m = np.flatnonzero(gf == g)
                flat[m] = flat[rng.permutation(m)]
            ys = flat.reshape(y.shape)
        null.append(gini(wealth_fn(ys)))
    null = np.array(null)
    sd = float(null.std(ddof=1)) if n_null > 1 else float("nan")
    z = float((g_obs - null.mean()) / sd) if sd and sd > 0 else float("nan")
    p = float((1 + (null >= g_obs).sum()) / (1 + n_null))
    return dict(gini=g_obs, null_mean=float(null.mean()), null_sd=sd, z=z, p=p, n_null=n_null)


def dead_bankrupt(act_count=None, wealth=None, floor=0.0):
    """dead: fraction of units that never fired (act_count == 0) in the window;
    bankrupt: fraction with wealth <= floor. Either input may be None."""
    out = {}
    if act_count is not None:
        a = np.ravel(np.asarray(act_count, float))
        out["dead_frac"] = float((a == 0).mean()) if a.size else float("nan")
    if wealth is not None:
        w = np.ravel(np.asarray(wealth, float))
        out["bankrupt_frac"] = float((w <= floor).mean()) if w.size else float("nan")
    return out


def conservation_residual(R, parts, bias=0.0):
    """Relative books residual |R - sum(parts) - bias| / max(|R|, 1e-12)."""
    R = float(np.sum(R))
    r = R - float(np.sum(parts)) - float(np.sum(bias))
    return dict(abs=abs(r), rel=abs(r) / max(abs(R), 1e-12))


# ---------------------------------------------------------------- compute
class FlopCounter:
    """Cumulative FLOPs by kind ('fwd', 'bwd', 'books', ...)."""

    def __init__(self):
        self.by_kind = {}

    def add(self, n, kind="fwd"):
        self.by_kind[kind] = self.by_kind.get(kind, 0) + int(n)
        return self

    @property
    def total(self):
        return sum(self.by_kind.values())


def net_forward_flops(B, Q, W, d=64, H=4, L=2, mlp=4):
    """Multiply-adds x2 for one forward of wealth-net's Net (tokens = questions).
    Embedding W->d, per block qkv (3d^2), scores+mix (2 Q d per token), out (d^2),
    MLP (2 mlp d^2), readout d. LayerNorm/softmax/elementwise ignored (<2%)."""
    per_tok = W * d + L * (3 * d * d + 2 * Q * d + d * d + 2 * mlp * d * d) + d
    return 2 * B * Q * per_tok


# ---------------------------------------------------------------- one-call row
def panel_row(p, y, fam=None, names=None, P=None, constraints=None, unresolved=None,
              wealth=None, act_count=None, floor=0.0, flops=None, cons=None, ec=None):
    """Flatten the panel into {metric: value} or {metric: {family: value}} for runlog.
    Only the parts whose inputs are given are computed. `ec` = (t, E); `cons` = (R, parts, bias)."""
    row = {}
    sc = scores(p, y, fam, names)
    for m in ("bce", "brier", "ece", "n"):
        row[m] = {k: v[m] for k, v in sc.items()}
    if P is not None and constraints:
        vi = violations(P, constraints, unresolved)
        row["viol_mean"] = {k: v["mean"] for k, v in vi.items()}
        row["viol_frac"] = {k: v["frac_gt_tol"] for k, v in vi.items()}
    if wealth is not None:
        row["gini"] = gini(wealth)
    db = dead_bankrupt(act_count, wealth, floor)
    row.update(db)
    if flops is not None:
        row["flops_cum"] = flops.total if isinstance(flops, FlopCounter) else float(flops)
    if cons is not None:
        row["cons_resid_rel"] = conservation_residual(*cons)["rel"]
    if ec is not None:
        s = ec_slopes(*ec)
        row["ec_loglog"] = s["loglog"]; row["ec_lin"] = s["lin_last_half"]
    return row


# ---------------------------------------------------------------- adapters
def constraints_from_li(kind, ops=None, group=None):
    """Constraint list from worker 3's LI stream arrays (substrate/li_families.py):
    kind[Q] strings in {base,twin,solo,taut_or,contra,impl,ex3,and,or,exp}, ops[Q,2] operand
    indices (-1 = none), group[Q] exclusive-exhaustive triple id (-1 = none).
    taut_or (A or not A) and impl ((A and B) -> A) are tautologies; contra (A and not A) a
    contradiction; twin equals its base ops[q,0]; and/or use ops; ex3 members sharing a group id form one 'exhaustive'."""
    cons, groups = [], {}
    for q, k in enumerate(kind):
        k = str(k)
        if k in ("taut_or", "impl"):
            cons.append(("taut", q))
        elif k == "contra":
            cons.append(("contra", q))
        elif k == "twin" and ops is not None and ops[q][0] >= 0:
            cons.append(("equal", q, int(ops[q][0])))
        elif k in ("and", "or") and ops is not None and ops[q][0] >= 0 and ops[q][1] >= 0:
            cons.append((k, q, int(ops[q][0]), int(ops[q][1])))
        elif k == "ex3" and group is not None and group[q] >= 0:
            groups.setdefault(int(group[q]), []).append(q)
    cons += [("exhaustive", *m) for _, m in sorted(groups.items())]
    return cons


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def panel_from_prequential(res, fam, names=None, stream=None):
    """Panel scores from worker 1's run_prequential output (dict with z, y, mask [n_pos,Q]).
    If `stream` carries kind/ops/group (LI stream), coherence violations are added, counted
    only where the question is unresolved: position i is unresolved for q while the label
    has not been revealed, i.e. always at trade time for delay>0 (the price is recorded
    before reveal), so every masked row counts for questions with delay>0."""
    m = res["mask"].astype(bool)
    p = sigmoid(res["z"])
    F = np.broadcast_to(np.asarray(fam)[None, :], m.shape)
    row = panel_row(p[m], res["y"][m], fam=F[m], names=names)
    if stream is not None and hasattr(stream, "kind"):
        cons = constraints_from_li(stream.kind, getattr(stream, "ops", None), getattr(stream, "group", None))
        if cons:
            unres = m & (np.asarray(stream.delay)[None, :] > 0)
            vi = violations(p, cons, unres)
            row["viol_mean"] = {k: v["mean"] for k, v in vi.items()}
            row["viol_frac"] = {k: v["frac_gt_tol"] for k, v in vi.items()}
    return row
