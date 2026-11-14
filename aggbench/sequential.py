"""Sequential aggregation: agents report to a CEO one at a time (Kelvin's design, 2026-10-21).

Kelvin: "Each of the agent observes, reports to the CEO sequentially. The CEO (aggregation) updates and gives a
"summary of the consensus" to the next agent. ... The only challenge is the info-bottleneck there the messages
may be smaller than the full observations maybe." On agent-specific views: "I like the idea of the agents doing
the queries, and the consensus answering the queries."

Protocol (one episode; every array below carries a leading batch axis n):

    state = init(n)                                   consensus memory starts empty
    for t, i in enumerate(order):                     order: a seeded random permutation, recorded
        q_t = query(inputs, i, t)                     agent side, from agent i's OWN inputs (None = broadcast)
        v_t = view(state, q_t, t)        (n, w_view)  consensus side: attention readout for q_t, or a summary
        m_t = encode(inputs, i, v_t, t)  (n, w_msg)   agent side, from agent i's own inputs and v_t only
        state = update(state, m_t, t)                 the CEO updates its memory
    forecast = readout(state)                         {"logit": (n,)} or {"mean": (n, d), "prec": (n, d, d)}

The modules are plain callables (torch nn.Modules work: they are callables). `inputs` is a dict name -> array
(n, N, ...) of every agent's inputs; agent-side modules get the whole dict plus the per-episode agent index
i (n,) and must read row i only. They get the whole dict so that `check_sequential_locality` can perturb the
other agents and prove the rule. CEO-side modules (init, view, update, readout) never receive inputs.

The default mode is QUERIED: the agent builds a query from its own inputs and the consensus memory answers by
attention (consensus memory = KV cache, agent query = cross-attention read, message = new KV entry). BROADCAST
(query=None: everyone gets the same summary of width w_view) is the comparison.

References (exact or labelled; none of them is learned):
  exact_sequential_bayes   unbounded messages: each agent sends its whole likelihood function over a latent world
                           that makes agents conditionally independent; the CEO keeps every message. Equals the
                           global Bayes posterior in every order (E1-E3 and the zoo; tested).
  coarse_action_cascade    Bikhchandani-Hirshleifer-Welch: each agent sends one coarse action, the sign of its
                           belief (public consensus + own evidence); the CEO updates rationally given the marginal
                           action likelihoods. Cascade-prone by design: shows the cascade metric works.
  scalar_report_sequential SCALAR-REPORT BASELINE (labelled): each agent sends its local log-odds, the CEO adds them.
                           Not the design (rich messages are); kept only for comparison.

Metrics: `regret` (to the global Bayes posterior), `cascade_curve` (how much of the private information of the
agent at position t reaches the final posterior and its message, measured by resampling that agent's private
input), `order_sensitivity` (variance of the final posterior across orders). `full_context_bar` calls the
full-context transformer reference (aggbench.transformer, Kelvin's bar) in the same setting.
"""
from dataclasses import dataclass, field, replace

import numpy as np

from . import make_env
from .metrics import gaussian_kl, logsig, per_episode, summarize, summarize_values

ORDER_SALT = 0x5E9
RESAMPLE_SALT = 0x5EA


# ----------------------------------------------------------------------------- array helpers (numpy or torch)

def _is_torch(x):
    return type(x).__module__.split(".")[0] == "torch"


def _np(x):
    if _is_torch(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def gather(arr, i):
    """Row i[e] of arr[e] for every episode e: (n, N, ...) -> (n, ...). Works for numpy and torch arrays."""
    if _is_torch(arr):
        import torch
        idx = torch.as_tensor(np.asarray(i), device=arr.device)
        return arr[torch.arange(arr.shape[0], device=arr.device), idx]
    return arr[np.arange(arr.shape[0]), np.asarray(i)]


def own_vector(inputs, i, names=None):
    """Agent i's own inputs flattened to one vector per episode, NaN -> 0 plus a finite-mask. (n, F) numpy."""
    parts = []
    for k in names or sorted(inputs):
        x = _np(gather(inputs[k], i)).reshape(len(i), -1).astype(float)
        fin = np.isfinite(x)
        parts += [np.where(fin, x, 0.0), fin.astype(float)]
    return np.concatenate(parts, axis=1) if parts else np.zeros((len(i), 0))


def _cat(xs, axis):
    if _is_torch(xs[0]):
        import torch
        return torch.cat(xs, dim=axis)
    return np.concatenate(xs, axis=axis)


def _zeros_like_rows(ref, n, width):
    if _is_torch(ref):
        import torch
        return torch.zeros(n, width, dtype=ref.dtype, device=ref.device)
    return np.zeros((n, width))


def _softmax(x, axis=-1):
    if _is_torch(x):
        return x.softmax(dim=axis)
    z = x - x.max(axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis, keepdims=True)


# ----------------------------------------------------------------------------- protocol and runner

@dataclass
class SequentialProtocol:
    """Pluggable modules of the sequential protocol (see the module docstring for the call order).

    encode(inputs, i, view, t) -> (n, w_msg)   agent side: own inputs (row i) + view only
    query(inputs, i, t) -> (n, w_q) or None    agent side: own inputs only; None module = broadcast mode
    view(state, query, t) -> (n, w_view)       consensus side
    update(state, message, t) -> state         consensus side
    readout(state) -> forecast                 consensus side
    init(n) -> state                           consensus side; the empty consensus
    w_msg, w_view: declared widths (the information bottleneck); the runner enforces them when set.
    label: e.g. "scalar-report baseline" or "reference (not learned)"; recorded with every result."""
    encode: object
    view: object
    update: object
    readout: object
    init: object
    query: object = None
    w_msg: int = None
    w_view: int = None
    name: str = "protocol"
    label: str = ""

    @property
    def mode(self):
        return "broadcast" if self.query is None else "queried"

    def describe(self):
        return {"name": self.name, "label": self.label, "mode": self.mode, "w_msg": self.w_msg,
                "w_view": self.w_view}


@dataclass
class Trace:
    order: np.ndarray                 # (n, N) agent index at each position
    queries: list                     # per position: (n, w_q) or None
    views: list                       # per position: (n, w_view)
    messages: list                    # per position: (n, w_msg)
    forecast: dict
    meta: dict = field(default_factory=dict)


def sample_orders(seeds, n_agents, order_seed=0):
    """One random arrival order per episode, (n, N), drawn from the episode seed and order_seed (deterministic)."""
    return np.stack([np.random.default_rng([int(s), ORDER_SALT, int(order_seed)]).permutation(n_agents)
                     for s in seeds])


def _check_width(x, w, what, name):
    if w is not None and (x.ndim != 2 or x.shape[1] != w):
        raise ValueError(f"{name}: {what} has shape {tuple(x.shape)}, declared width {w}")


def run_protocol(protocol, inputs, order):
    """Run the protocol on every episode with the given arrival order (n, N). Returns a Trace."""
    order = np.asarray(order)
    n, N = order.shape
    state = protocol.init(n)
    queries, views, messages = [], [], []
    for t in range(N):
        i = order[:, t]
        q = protocol.query(inputs, i, t) if protocol.query is not None else None
        v = protocol.view(state, q, t)
        _check_width(v, protocol.w_view, "view", protocol.name)
        m = protocol.encode(inputs, i, v, t)
        _check_width(m, protocol.w_msg, "message", protocol.name)
        state = protocol.update(state, m, t)
        queries.append(q)
        views.append(v)
        messages.append(m)
    return Trace(order, queries, views, messages, protocol.readout(state), {"protocol": protocol.describe()})


# ----------------------------------------------------------------------------- generic consensus modules

@dataclass
class KVState:
    """Consensus memory: one key and one value per message so far. k (n, T, d_k), v (n, T, d_v); None = empty."""
    n: int
    k: object = None
    v: object = None

    def __len__(self):
        return 0 if self.v is None else self.v.shape[1]


def kv_init(n):
    """The empty consensus memory of n episodes."""
    return KVState(n)


class AppendKV:
    """update: the message is a new KV entry; its first d_k columns are the key, the rest the value."""

    def __init__(self, d_k):
        self.d_k = d_k

    def __call__(self, state, m, t):
        k, v = m[:, None, :self.d_k], m[:, None, self.d_k:]
        if state.v is None:
            return KVState(state.n, k, v)
        return KVState(state.n, _cat([state.k, k], 1), _cat([state.v, v], 1))


class AttentionView:
    """Queried view: softmax(q K^T / sqrt(d_k)) V over the consensus memory, then an optional projection
    (d_v, w_view). Empty memory -> zeros. `proj` may be a numpy array, a torch tensor or a callable."""

    def __init__(self, w_view, proj=None, temperature=None):
        self.w_view, self.proj, self.temperature = w_view, proj, temperature

    def _project(self, x):
        if self.proj is None:
            return x
        return self.proj(x) if callable(self.proj) else x @ self.proj

    def __call__(self, state, q, t):
        if q is None:
            raise ValueError("AttentionView needs a query (queried mode); use MeanView for broadcast")
        if len(state) == 0:
            return _zeros_like_rows(q, q.shape[0], self.w_view)
        tau = self.temperature or np.sqrt(q.shape[-1])
        if _is_torch(q):
            import torch
            a = torch.einsum("nd,ntd->nt", q, state.k) / tau
            out = torch.einsum("nt,ntv->nv", _softmax(a), state.v)
        else:
            a = np.einsum("nd,ntd->nt", q, state.k) / tau
            out = np.einsum("nt,ntv->nv", _softmax(a), state.v)
        return self._project(out)


class MeanView(AttentionView):
    """Broadcast view: the same summary for every agent (mean of the memory's values, optional projection).
    Ignores the query. Empty memory -> zeros."""

    def __call__(self, state, q, t):
        if len(state) == 0:
            return np.zeros((state.n, self.w_view))
        return self._project(state.v.mean(1))


def linear_protocol(inputs, w_msg, w_view, mode="queried", d_k=4, seed=0, prior_logit=0.0):
    """A minimal UNTRAINED example of the module interfaces (random linear maps + tanh), bernoulli readout.
    For tests, smoke runs and as a template: experiments replace these modules with learned ones.
    Agent: query = tanh(W_q x_i); message = tanh(W_m [x_i, view]) with its first d_k entries the key.
    Consensus: AppendKV; AttentionView (queried) or mean of values (broadcast); readout = prior + w . mean(V)."""
    if mode not in ("queried", "broadcast"):
        raise ValueError("mode must be 'queried' or 'broadcast'")
    n = next(iter(inputs.values())).shape[0]
    F = own_vector(inputs, np.zeros(n, int)).shape[1]
    rng = np.random.default_rng(seed)
    d_v = w_msg - d_k
    if d_v < 1:
        raise ValueError("need w_msg > d_k")
    Wq = rng.normal(size=(F, d_k)) / np.sqrt(max(F, 1))
    Wm = rng.normal(size=(F + w_view, w_msg)) / np.sqrt(F + w_view)
    P = rng.normal(size=(d_v, w_view)) / np.sqrt(d_v)
    w_out = rng.normal(size=d_v)

    def query(inp, i, t):
        return np.tanh(own_vector(inp, i) @ Wq)

    def encode(inp, i, view, t):
        return np.tanh(np.concatenate([own_vector(inp, i), _np(view)], 1) @ Wm)

    def readout(state):
        return {"logit": prior_logit + state.v.mean(1) @ w_out}

    view = AttentionView(w_view, P) if mode == "queried" else MeanView(w_view, P)
    return SequentialProtocol(encode, view, AppendKV(d_k), readout, kv_init,
                              query=query if mode == "queried" else None, w_msg=w_msg, w_view=w_view,
                              name=f"linear-{mode}", label="untrained example (not a result)")


def agent_inputs_dict(batch, visibility="none", alignment="unaligned"):
    """batch.agent_inputs as the inputs dict of the protocol: {"obs", "keys"} (headline: unaligned, none/noisy)."""
    obs, keys, _ = batch.agent_inputs(visibility, alignment=alignment)
    return {"obs": obs.reshape(*obs.shape[:2], -1), "keys": keys}


# ----------------------------------------------------------------------------- exact local likelihoods (references)

def _family(batch):
    if batch.kind == "gaussian":
        return "gaussian"
    if "W" in batch.oracle:
        return "rollouts"
    if "cov" in batch.oracle:
        from .zoo import FAMILIES
        return "private_reads" if FAMILIES.get(batch.env) == "hidden-provenance" else "pieces"
    return "tabular"


def _tables(batch):
    """Zoo tables of every episode: prior (n, W), target (n, W), lik (n, N, W, S), values (n, N, S). Cached."""
    cache = batch.__dict__.setdefault("_seq_tables", {})
    if "t" not in cache:
        env = make_env(batch.env, allow_heldout=True, **batch.config)
        rows = [env.tables_for(batch, e) for e in range(len(batch))]
        cache["t"] = tuple(np.stack([r[j] for r in rows]) for j in range(4))
    return cache["t"]


def exact_inputs(batch):
    """Each agent's exact own inputs for the reference protocols: its observation turned into its likelihood over a
    latent world that makes agents conditionally independent, using its OWN observation model only (the agent
    knows its own model; that is local knowledge). Plus "llr" = its report log-odds minus the prior (bernoulli).

      E1 (gaussian): latent (theta, eta); "h" = H_i^T s_i / sig_i^2, "J" = H_i^T H_i / sig_i^2 with H_i = [A_i, B_i]
      E2/E3 (pieces): latent (Y, pieces); "seen" (K, by piece identity) and "lam" = 2 mu_k x_k on seen pieces
      zoo (tabular): latent w; "loglik" = log P(o_i | w) for every world state w
    The reference assumes each agent knows the identity of its evidence pieces (E2/E3) and its own model; it is a
    ceiling, not a learned design. E4 is not supported (rollouts are not conditionally independent sources)."""
    fam = _family(batch)
    out = {}
    if fam == "gaussian":
        A, Bf, sig = batch.oracle["A"], batch.oracle["Bf"], batch.oracle["sig"]
        H = np.concatenate([A, Bf], -1)                                  # (n, N, m, D)
        w = 1.0 / sig ** 2                                                # (n, N)
        out["h"] = np.einsum("enmd,enm->end", H, batch.obs) * w[..., None]
        J = np.einsum("enmd,enmf->endf", H, H) * w[..., None, None]
        out["J"] = J.reshape(*J.shape[:2], -1)
        return out
    if fam == "pieces":
        cov, mu = batch.oracle["cov"], batch.oracle["mu"]
        out["seen"] = cov.astype(float)
        out["lam"] = np.where(cov, 2 * mu[:, None, :] * np.nan_to_num(batch.obs), 0.0)
    elif fam == "private_reads":                      # Z8: agent i reads piece k as r_ik = x_k + tau eps_ik
        cov, mu = batch.oracle["cov"], batch.oracle["mu"]
        out["seen"] = cov.astype(float)
        out["reads"] = np.where(cov, np.nan_to_num(batch.obs), 0.0)
        out["mu"] = cov * mu[:, None, :]
    elif fam == "tabular":
        _, _, lik, _ = _tables(batch)
        sym = batch.oracle["sym"]
        n, N = sym.shape
        with np.errstate(divide="ignore"):
            out["loglik"] = np.log(np.take_along_axis(lik, sym[:, :, None, None], -1)[..., 0])
    else:
        raise NotImplementedError("sequential references support E1, E2/E3, Z8 and the zoo, not E4 rollouts")
    out["llr"] = (batch.reports["logit"] - batch.public["prior_logit"])[..., None]
    return out


def _sum_v(state):
    return state.v.sum(1)


def exact_sequential_bayes(batch):
    """Reference: exact sequential Bayes with UNBOUNDED messages (inputs = exact_inputs(batch)).

    Each agent sends its whole likelihood (w_msg = its full width); the CEO appends it to the consensus memory
    (KV cache with every message) and the readout combines them: E1 prior + sum of natural parameters, marginal
    of theta; E2/E3 prior + LLR of each piece in the union, counted once; zoo prior over w + sum of log-likelihoods.
    Agents ignore the view (w_view = 0): with unbounded messages the view is not needed. Exact in every order."""
    fam = _family(batch)
    n = len(batch)
    if fam == "gaussian":
        D = batch.oracle["A"].shape[-1] + batch.oracle["Bf"].shape[-1]
        d = batch.oracle["A"].shape[-1]
        names = ("h", "J")

        def readout(state):
            s = _sum_v(state)
            h, P = s[:, :D], s[:, D:].reshape(n, D, D) + np.eye(D)
            cov = np.linalg.inv(P)
            mean = np.einsum("nij,nj->ni", cov, h)[:, :d]
            return {"mean": mean, "prec": np.linalg.inv(cov[:, :d, :d])}
        w = D + D * D
    elif fam == "pieces":
        K = batch.oracle["cov"].shape[-1]
        l0 = batch.public["prior_logit"]
        names = ("seen", "lam")

        def readout(state):
            seen, lam = state.v[..., :K], state.v[..., K:]
            cnt = seen.sum(1)
            per_piece = lam.sum(1) / np.maximum(cnt, 1)   # duplicates of a piece carry the same LLR
            return {"logit": l0 + (per_piece * (cnt > 0)).sum(1)}
        w = 2 * K
    elif fam == "private_reads":
        from .zoo import PrivateReadConfig
        K = batch.oracle["cov"].shape[-1]
        l0 = batch.public["prior_logit"]
        tau2 = PrivateReadConfig(**batch.config).read_noise ** 2
        names = ("seen", "reads", "mu")

        def readout(state):
            # Bayes: l0 + sum_k 2 mu_k (sum of the n_k reads of piece k) / (n_k + tau^2), over pieces read at all
            seen, reads, mus = state.v[..., :K], state.v[..., K:2 * K], state.v[..., 2 * K:]
            cnt, sr, mu = seen.sum(1), reads.sum(1), mus.max(1)
            return {"logit": l0 + np.where(cnt > 0, 2 * mu * sr / (cnt + tau2), 0.0).sum(1)}
        w = 3 * K
    elif fam == "tabular":
        prior, target, lik, _ = _tables(batch)
        names = ("loglik",)
        with np.errstate(divide="ignore"):
            lp = np.log(prior)

        def readout(state):
            s = lp + state.v.sum(1)
            with np.errstate(invalid="ignore"):
                a = np.logaddexp.reduce(np.where(target == 1, s, -np.inf), 1)
                b = np.logaddexp.reduce(np.where(target == 0, s, -np.inf), 1)
            return {"logit": a - b}
        w = lik.shape[2]
    else:
        raise NotImplementedError("sequential references support E1, E2/E3 and the zoo, not E4 rollouts")

    def encode(inputs, i, view, t):
        return np.concatenate([gather(inputs[k], i).reshape(n, -1) for k in names], 1)

    def view(state, q, t):
        return np.zeros((n, 0))

    return SequentialProtocol(encode, view, AppendKV(0), readout, kv_init, query=None, w_msg=w, w_view=0,
                              name="exact_sequential_bayes", label="reference: unbounded messages (not learned)")


def scalar_report_sequential(batch):
    """SCALAR-REPORT BASELINE (labelled; not the design): each agent sends its report log-odds minus the prior
    (w_msg = 1), the CEO adds them (the sequential form of sum_logodds). Order-invariant; over-counts shared evidence."""
    if batch.kind != "bernoulli":
        raise NotImplementedError("bernoulli environments only")
    n, l0 = len(batch), batch.public["prior_logit"]
    return SequentialProtocol(
        encode=lambda inp, i, v, t: gather(inp["llr"], i).reshape(n, 1),
        view=lambda s, q, t: np.zeros((n, 0)), update=AppendKV(0),
        readout=lambda s: {"logit": l0 + s.v.sum((1, 2))}, init=kv_init, w_msg=1, w_view=0,
        name="scalar_report_sequential", label="scalar-report baseline")


@dataclass
class ActionModel:
    """Empirical distribution of an agent's own log-odds shift given Y (from a train batch), used by the
    coarse-action CEO to update rationally: P(action = +1 | Y, public p) = P(llr > -p | Y)."""
    llr1: np.ndarray
    llr0: np.ndarray

    @classmethod
    def fit(cls, train_batch):
        llr = train_batch.reports["logit"] - train_batch.public["prior_logit"]
        y = train_batch.truth.astype(bool)
        return cls(np.sort(llr[y].ravel()), np.sort(llr[~y].ravel()))

    def update(self, p, a):
        """Log-likelihood ratio of action a (n,) in {-1, 0, +1} given public log-odds p (n,) (offset by prior)."""
        def tail(s, x):  # P(llr > x), smoothed
            return (len(s) - np.searchsorted(s, x, "right") + 0.5) / (len(s) + 1)
        up = np.log(tail(self.llr1, -p)) - np.log(tail(self.llr0, -p))
        down = np.log(1 - tail(self.llr1, -p)) - np.log(1 - tail(self.llr0, -p))
        return np.where(a > 0, up, np.where(a < 0, down, 0.0))


def coarse_action_cascade(batch, action_model):
    """Cascade-prone reference (Bikhchandani-Hirshleifer-Welch). Broadcast view = the public consensus log-odds
    (w_view = 1). Agent i's belief = view + its own llr; it sends only the sign of that belief (w_msg = 1, one
    action). The CEO updates the public log-odds by the action's likelihood ratio under `action_model` (agents'
    llr distribution given Y, fitted on train; it ignores overlap between agents). Once the public belief outweighs
    any one agent's evidence, actions stop depending on private information: an information cascade."""
    if batch.kind != "bernoulli":
        raise NotImplementedError("bernoulli environments only")
    n, l0 = len(batch), batch.public["prior_logit"]

    def encode(inp, i, view, t):
        return np.sign(view[:, :1] + gather(inp["llr"], i).reshape(n, 1))

    def update(state, m, t):
        return state + action_model.update(state - l0, m[:, 0])

    return SequentialProtocol(encode, view=lambda s, q, t: s[:, None].copy(), update=update,
                              readout=lambda s: {"logit": s.copy()}, init=lambda n_: np.full(n_, float(l0)),
                              w_msg=1, w_view=1, name="coarse_action_cascade",
                              label="reference: cascade-prone coarse actions (BHW), not learned")


# ----------------------------------------------------------------------------- locality

def _perturb(inputs, mask, rng):
    """Copy of inputs with finite float entries of agents where mask (n, N) is True shifted by N(0, 1) noise."""
    out = {}
    for k, x in inputs.items():
        x = _np(x).copy()
        if np.issubdtype(x.dtype, np.floating):
            m = mask.reshape(mask.shape + (1,) * (x.ndim - 2))
            noise = rng.normal(size=x.shape)
            x = np.where(m & np.isfinite(x), x + noise, x)
        out[k] = x
    return out


def _same(a, b, atol):
    if a is None or b is None:
        return a is None and b is None
    return np.allclose(_np(a), _np(b), atol=atol, equal_nan=True)


def check_sequential_locality(protocol, inputs, order, trials=3, seed=0, atol=1e-9):
    """Sequential extension of messages.check_locality. Raises AssertionError on a violation; returns a record.

    1. Message and query locality: at a position t, perturb every other agent's inputs and recompute agent t's
       query and message from the perturbed inputs and the SAME recorded view: they must not change.
    2. Causality: perturb the agents at positions > t and rerun: every query, view and message at positions <= t
       must be unchanged (no module may look at agents that have not reported yet).
    CEO-side modules never receive inputs (by construction of run_protocol)."""
    rng = np.random.default_rng(seed)
    order = np.asarray(order)
    n, N = order.shape
    base = run_protocol(protocol, inputs, order)
    positions = sorted(set(int(p) for p in rng.integers(0, N, trials)) | {0, N - 1})
    for t in positions:
        i = order[:, t]
        others = np.ones((n, N), bool)
        others[np.arange(n), i] = False
        pert = _perturb(inputs, others, rng)
        if protocol.query is not None and not _same(protocol.query(pert, i, t), base.queries[t], atol):
            raise AssertionError(f"{protocol.name}: the query at position {t} depends on other agents' inputs")
        if not _same(protocol.encode(pert, i, base.views[t], t), base.messages[t], atol):
            raise AssertionError(f"{protocol.name}: the message at position {t} depends on other agents' inputs "
                                 "(beyond its view)")
        if t < N - 1:
            later = np.zeros((n, N), bool)
            later[np.arange(n)[:, None], order[:, t + 1:]] = True
            tr = run_protocol(protocol, _perturb(inputs, later, rng), order)
            for s in range(t + 1):
                for what, a, b in (("query", tr.queries[s], base.queries[s]), ("view", tr.views[s], base.views[s]),
                                   ("message", tr.messages[s], base.messages[s])):
                    if not _same(a, b, atol):
                        raise AssertionError(f"{protocol.name}: the {what} at position {s} depends on agents "
                                             f"after position {t}")
    return {"passed": True, "positions": positions, "trials": trials, "seed": seed, "mode": protocol.mode}


# ----------------------------------------------------------------------------- private resampling

def resample_private(batch, agent, seed=0):
    """A copy of batch in which agent[e]'s PRIVATE input is redrawn given the latent world, everything else fixed.
    E1: its private noise eps_i (theta and the shared factors eta kept). E2/E3: the pieces that only it sees (shared
    pieces kept, so the data stay consistent; an agent with no private pieces is unchanged). Zoo: its symbol, from
    P(o_i | w) at the true w. local, report, bayes (and the zoo reports ceiling) are recomputed."""
    fam = _family(batch)
    agent = np.asarray(agent)
    n = len(batch)
    rng = np.random.default_rng([RESAMPLE_SALT, int(seed), int(batch.seeds[0])])
    obs = batch.obs.copy()
    local = {k: v.copy() for k, v in batch.local.items()}
    report = {k: v.copy() for k, v in batch.reports.items()}
    bayes = {k: np.array(v, copy=True) for k, v in batch.bayes.items()}
    oracle = dict(batch.oracle)
    if fam == "gaussian":
        from .envs import gaussian_global, gaussian_local
        A, Bf, sig = batch.oracle["A"], batch.oracle["Bf"], batch.oracle["sig"]
        theta, eta = batch.truth, batch.oracle["eta"]
        for e in range(n):
            i = agent[e]
            obs[e, i] = A[e, i] @ theta[e] + Bf[e, i] @ eta[e] + sig[e, i] * rng.normal(size=obs.shape[-1])
            lm, lp = gaussian_local(A[e, i:i + 1], Bf[e, i:i + 1], sig[e, i:i + 1], obs[e, i:i + 1])
            for d_ in (local, report):
                d_["mean"][e, i], d_["prec"][e, i] = lm[0], lp[0]
            bayes["mean"][e], bayes["prec"][e] = gaussian_global(A[e], Bf[e], sig[e], obs[e])
    elif fam == "pieces":
        cov, mu = batch.oracle["cov"], batch.oracle["mu"]
        x = batch.oracle["x"].copy()
        l0 = batch.public["prior_logit"]
        a, b = batch.oracle["a"], batch.oracle["b"]
        for e in range(n):
            i = agent[e]
            others = np.delete(cov[e], i, 0).any(0)
            priv = cov[e, i] & ~others
            y = 2 * batch.truth[e] - 1
            x[e, priv] = y * mu[e, priv] + rng.normal(size=priv.sum())
        lam = 2 * mu * x
        obs = np.where(cov, x[:, None, :], np.nan)
        local["logit"] = l0 + np.einsum("enk,ek->en", cov.astype(float), lam)
        report["logit"] = l0 + a * (local["logit"] - l0) + b
        bayes["logit"] = l0 + (lam * cov.any(1)).sum(1)
        oracle["x"] = x
    elif fam == "private_reads":
        raise NotImplementedError("resample_private does not support Z8 (private read noise) yet")
    elif fam == "tabular":
        from .zoo import tabular_ceilings
        prior, target, lik, values = _tables(batch)
        sym = batch.oracle["sym"].copy()
        ceil = np.array(batch.oracle["report_logit"], float, copy=True)
        for e in range(n):
            i, w = agent[e], batch.oracle["w"][e]
            p = lik[e, i, w]
            sym[e, i] = rng.choice(len(p), p=p / p.sum())
            obs[e, i] = values[e, i, sym[e, i]]
            r, bayes["logit"][e], ceil[e] = tabular_ceilings(prior[e], target[e], lik[e], sym[e])
            local["logit"][e], report["logit"][e] = r, r
        oracle["sym"], oracle["report_logit"] = sym, ceil
    else:
        raise NotImplementedError("resampling supports E1, E2/E3 and the zoo")
    out = replace(batch, obs=obs, local=local, reports=report, bayes=bayes, oracle=oracle)
    if "_seq_tables" in batch.__dict__:
        out.__dict__["_seq_tables"] = batch.__dict__["_seq_tables"]  # structure unchanged
    return out


# ----------------------------------------------------------------------------- metrics

def forecast_kl(f, g, kind):
    """KL(f || g) per episode."""
    if kind == "bernoulli":
        a, b = f["logit"], g["logit"]
        pa = np.exp(logsig(a))
        return pa * (logsig(a) - logsig(b)) + (1 - pa) * (logsig(-a) - logsig(-b))
    return gaussian_kl(f["mean"], f["prec"], g["mean"], g["prec"])


def regret(forecast, batch, n_boot=2000):
    """Mean regret to the global Bayes posterior with a 95% bootstrap CI (metrics.summarize)."""
    return summarize(forecast, batch, n_boot=n_boot)["regret"]


def cascade_curve(protocols, make_inputs, batch, order=None, positions=None, seed=0, n_boot=1000):
    """How much of the private information of the agent at position t survives, for t = 0..N-1.

    For each position t, the private input of the agent arriving at t is resampled given the latent world
    (resample_private) and every protocol is rerun in the same order. Per position and protocol:
      influence        how far the final posterior moves: E |delta final log-odds| (bernoulli) or
                       E sqrt(2 KL) (gaussian; the Mahalanobis length of a pure mean shift). 0 = none of that
                       agent's private information reached the final answer
      kl               E KL(final | original || final | resampled)
      regret_increase  E logloss(resampled) - E logloss(original), scored on the original truth
      message_response fraction of episodes in which the agent's own message changed
      retention        influence / influence of the first protocol (pass the exact reference first)
      message_retention message_response / message_response of the first protocol
    An information cascade shows as retention and message_retention falling towards 0 with position (late agents'
    messages stop depending on what they saw), usually with early agents over-weighted (retention > 1).
    protocols: dict name -> protocol (built on `batch`; the structure is unchanged by resampling).
    make_inputs: dict name -> function(batch) -> inputs dict (or one function for all)."""
    n, N = len(batch), batch.obs.shape[1]
    order = sample_orders(batch.seeds, N) if order is None else np.asarray(order)
    positions = list(range(N)) if positions is None else list(positions)
    mk = make_inputs if isinstance(make_inputs, dict) else {k: make_inputs for k in protocols}
    base = {k: run_protocol(p, mk[k](batch), order) for k, p in protocols.items()}
    out = {k: {"position": positions, "influence": [], "kl": [], "regret_increase": [], "message_response": []}
           for k in protocols}
    for t in positions:
        rb = resample_private(batch, order[:, t], seed=seed * 1000 + t)
        for k, p in protocols.items():
            tr = run_protocol(p, mk[k](rb), order)
            kl = forecast_kl(base[k].forecast, tr.forecast, batch.kind)
            if batch.kind == "bernoulli":
                infl = np.abs(_np(tr.forecast["logit"]) - _np(base[k].forecast["logit"]))
            else:
                infl = np.sqrt(2 * np.maximum(kl, 0))
            dreg = per_episode(tr.forecast, batch)["logloss"] - per_episode(base[k].forecast, batch)["logloss"]
            dm = np.abs(_np(tr.messages[t]) - _np(base[k].messages[t])).reshape(n, -1).max(1) > 1e-9
            out[k]["influence"].append(summarize_values(infl, n_boot))
            out[k]["kl"].append(summarize_values(kl, n_boot))
            out[k]["regret_increase"].append(summarize_values(dreg, n_boot))
            out[k]["message_response"].append(float(dm.mean()))
    ref = next(iter(protocols))
    for k in protocols:
        out[k]["retention"] = [a["mean"] / max(b["mean"], 1e-12)
                               for a, b in zip(out[k]["influence"], out[ref]["influence"])]
        out[k]["message_retention"] = [a / max(b, 1e-12)
                                       for a, b in zip(out[k]["message_response"], out[ref]["message_response"])]
    return out


def order_sensitivity(protocol, inputs, batch, n_orders=8, n_boot=1000):
    """Variance of the final posterior across n_orders random arrival orders (order_seed 0..n_orders-1), per
    episode, then averaged: bernoulli = variance of the final log-odds; gaussian = summed variance of the mean.
    Also the mean regret over orders. Exact Bayes gives 0."""
    N = batch.obs.shape[1]
    fc = [run_protocol(protocol, inputs, sample_orders(batch.seeds, N, s)).forecast for s in range(n_orders)]
    key = "logit" if batch.kind == "bernoulli" else "mean"
    stack = np.stack([_np(f[key]) for f in fc])                       # (R, n) or (R, n, d)
    var = stack.var(0)
    var = var if var.ndim == 1 else var.sum(-1)
    reg = np.mean([per_episode(f, batch)["regret"].mean() for f in fc])
    return {"variance": summarize_values(var, n_boot), "mean_regret_over_orders": float(reg), "n_orders": n_orders}


def full_context_bar(env, **kwargs):
    """Kelvin's bar: the full-context transformer that reads every agent's raw inputs. Calls
    aggbench.transformer.evaluate_full_context_transformer(env, **kwargs) (on main since c918cc6; it trains, so
    pass the same visibility, alignment and n_train as the protocol run). Returns {"available": False} if the
    module cannot be imported."""
    try:
        from .transformer import evaluate_full_context_transformer
    except ImportError as exc:
        return {"available": False, "note": f"aggbench.transformer not importable ({exc}); compare when it lands"}
    return {"available": True, "result": evaluate_full_context_transformer(env, **kwargs)}


# ----------------------------------------------------------------------------- reference table

def reference_table(env_name, n_train=2000, n_eval=2000, n_orders=8, n_boot=1000, **overrides):
    """Reference protocols on one environment (dev split): regret to Bayes, order sensitivity, cascade retention."""
    from .baselines import oracle_clearing
    env = make_env(env_name, allow_heldout=True, **overrides)
    dv = env.sample("dev", n_eval)
    protos = {"exact_sequential_bayes": exact_sequential_bayes(dv)}
    if dv.kind == "bernoulli":
        protos["coarse_action_cascade"] = coarse_action_cascade(dv, ActionModel.fit(env.sample("train", n_train)))
        protos["scalar_report_sequential"] = scalar_report_sequential(dv)
    inputs = exact_inputs(dv)
    casc = cascade_curve(protos, exact_inputs, dv, n_boot=n_boot)
    rows = []
    for k, p in protos.items():
        tr = run_protocol(p, inputs, sample_orders(dv.seeds, dv.obs.shape[1]))
        rows.append({"protocol": k, "label": p.label, "w_msg": p.w_msg, "w_view": p.w_view,
                     "regret": regret(tr.forecast, dv, n_boot),
                     "order_variance": order_sensitivity(p, inputs, dv, n_orders, n_boot)["variance"],
                     "retention": casc[k]["retention"], "message_retention": casc[k]["message_retention"],
                     "influence": [c["mean"] for c in casc[k]["influence"]]})
    if dv.kind == "bernoulli":
        rows.append({"protocol": "oracle_clearing", "label": "scalar-report ceiling (not sequential)",
                     "w_msg": 1, "w_view": None, "regret": regret(oracle_clearing(dv), dv, n_boot),
                     "order_variance": None, "retention": None, "message_retention": None, "influence": None})
    return {"env": env_name, "n_eval": n_eval, "rows": rows}


def format_table(tab):
    lines = [f"{tab['env']} (dev, n={tab['n_eval']})",
             "| protocol | w_msg | w_view | regret [95% CI] | order var | retention by position | message retention |",
             "|---|---|---|---|---|---|---|"]
    for r in tab["rows"]:
        g = r["regret"]
        ov = "-" if r["order_variance"] is None else f"{r['order_variance']['mean']:.3g}"
        ret = "-" if r["retention"] is None else " ".join(f"{x:.2f}" for x in r["retention"])
        resp = "-" if r["message_retention"] is None else " ".join(f"{x:.2f}" for x in r["message_retention"])
        name = r["protocol"] + (f" ({r['label']})" if "baseline" in r["label"] or "ceiling" in r["label"] else "")
        lines.append(f"| {name} | {r['w_msg']} | {r['w_view'] if r['w_view'] is not None else '-'} | "
                     f"{g['mean']:.3f} [{g['ci_low']:.3f}, {g['ci_high']:.3f}] | {ov} | {ret} | {resp} |")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Reference protocols of sequential aggregation")
    ap.add_argument("--env", nargs="+", default=["E2", "Z1"])
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    for name in a.env:
        print(format_table(reference_table(name, n_train=a.n, n_eval=a.n)))
        print()
