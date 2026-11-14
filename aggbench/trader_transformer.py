"""trader_transformer: Kelvin's trader-transformer design (added 2026-10-21).

Kelvin: "(1) first, imagine that each information each 'trader' see is just one token. (2) second, imagine that what
it does is just the same as any transformer does when looking at a token. It also has self-attention over all
previous 'traders' (runs of the same network) over the previous tokens. (3) at the final layer of the last run, it
predicts something". His empirical questions: "(A) how much we need the self-attention at each layer analogy for the
traders. (B) can we train this correctly without supervision on each 'token', and only on 'final token'? (C) how much
of the 'residual stream' logic is kind-of-transversed in this setup?"

Model: a plain pre-norm decoder-only transformer. One token per trader (aggbench.transformer.full_context: obs,
present mask, own sensor model, provenance key at `visibility`, report; unaligned by default), traders in a random
order per episode (sequential.sample_orders, seeded and recorded), a causal mask over traders, the prediction read
from the final layer at the LAST trader's position. No positional encoding by default (pos="learned" adds one).
Bernoulli environments only (E2-E4 and the zoo).

Knobs:
  attn_layers  (A) layers where traders read earlier traders: "all", "last", "first", "every_other" (alternate
               layers, ending with the last), "none", or an explicit tuple of layer indices. Other layers are
               MLP-only on each trader's own residual stream.
  loss         (B) "final": outcome loss at the last position only. "every_prefix": every position predicts the
               outcome (as in next-token training; the target is still the outcome, never Bayes). "distill": every
               position fitted to the PREFIX BAYES posterior; it uses the oracle and is labelled so (uses_oracle).
  token_state  (C) how information crosses traders in the reading layers:
               "kv"        standard causal self-attention over earlier traders' layer states;
               "recurrent" no cross-trader attention: a GRU running summary per reading layer passes in the trader
                           direction (trader t reads the summary of traders < t, then updates it): Kelvin's earlier
                           CEO-summary design;
               "feedback"  Feedback Transformer (Fan et al. 2020, arXiv:2002.09402): trader t is computed through all
                           layers before trader t+1; each reading layer of trader t attends over ONE memory per earlier
                           trader, a learned softmax mix of all that trader's layer states (plus its own current state).
  trader_encoder  layout "trader" only: "linear" (default; the trader's whole token through one Linear) or
               "deepsets" (E2/E3: a shared per-piece encoder over the trader's piece cells, summed over its present
               pieces; the cells of piece_tokens, without the report).
  device       None = cuda when available, else cpu (threads apply on cpu only).
  pos          "none" (default) or "learned" (a learned embedding of the arrival position; grid layouts: of the
               token index and of the trader index).
  layout       "trader" (default): one token per trader. Kelvin's 3-D version (depth, token index, trader index), with
               his rule "each info path goes through one specific layer only once": each trader contributes one token
               per evidence piece it sees (E2/E3 only; pieces in the trader's own random slot order, padded to
               max_pieces with empty cells), cell (t, r) = token t of the trader at position r. The standard transformer
               computation is kept; only the attention mask changes:
               "grid"      (default grid) axial by head inside every attention layer: `token_heads` heads read along the
                           token axis (same trader, t' <= t), the other heads along the trader axis (same token index,
                           r' <= r). Kelvin: "it senough to see either the t' <= t; or the r' <= r, even with different
                           heads etc." A diagonal path takes two layers; every path passes each layer once.
               "axial"     comparison: alternate attention layers read along the token axis (even layer index) or the
                           trader axis (odd).
               "grid_full" comparison: every cell (t', r') with t' <= t and r' <= r.
               Readout at the last cell (last token slot of the last trader); with every_prefix/distill the prefix
               predictions are read at the last token slot of each trader, whose cone holds all tokens of traders <= r.
               Grid layouts use token_state="kv".

    from aggbench.trader_transformer import evaluate_trader_transformer
    rec = evaluate_trader_transformer(env, attn_layers="last", loss="every_prefix", token_state="kv", threads=2)

Smoke runs give no verdicts; this module makes no regret claims. Needs torch (imported lazily).
"""
from dataclasses import dataclass

import numpy as np

from .sequential import exact_inputs, exact_sequential_bayes, sample_orders
from .transformer import full_context

DEFAULTS = {"width": 64, "layers": 3, "heads": 4, "steps": 3000, "batch": 512, "lr": 2e-3, "attn_layers": "all",
            "token_state": "kv", "loss": "final", "pos": "none", "layout": "trader", "token_heads": None}
TOKEN_STATES = ("kv", "recurrent", "feedback")
LAYOUTS = ("trader", "grid", "axial", "grid_full")
GRID_LAYOUTS = ("grid", "axial", "grid_full")
LOSSES = ("final", "every_prefix", "distill")


def resolve_attn_layers(spec, layers):
    """Layer indices (0..layers-1) where traders read earlier traders."""
    if isinstance(spec, str):
        named = {"all": range(layers), "last": [layers - 1], "first": [0], "none": [],
                 "every_other": [l for l in range(layers) if l % 2 == (layers - 1) % 2]}
        if spec not in named:
            raise ValueError(f"attn_layers must be one of {sorted(named)} or a tuple of layer indices")
        return tuple(named[spec])
    out = tuple(sorted({int(l) for l in spec}))
    if any(l < 0 or l >= layers for l in out):
        raise ValueError(f"attn_layers {out} outside 0..{layers - 1}")
    return out


@dataclass
class TraderContext:
    """Tokens in arrival order: (n, N, F) for layout "trader", (n, N, T, F) cells for the grid layouts; the order
    (n, N); public constants; and (training only, for loss="distill") the prefix Bayes log-odds (n, N): Bayes given
    the first t+1 traders. predict never reads prefix_bayes."""
    kind: str
    tokens: np.ndarray
    order: np.ndarray
    public: dict
    prefix_bayes: np.ndarray = None
    layout: str = "trader"


def prefix_bayes_logits(batch, order):
    """Exact Bayes log-odds given the traders at positions 0..t, for every t: (n, N). Uses the oracle (exact
    sequential Bayes, aggbench.sequential); for distillation and diagnostics only. Not for E4."""
    p, inp = exact_sequential_bayes(batch), exact_inputs(batch)
    n, N = order.shape
    state, out = p.init(n), np.zeros((n, N))
    for t in range(N):
        i = order[:, t]
        state = p.update(state, p.encode(inp, i, p.view(state, None, t), t), t)
        out[:, t] = p.readout(state)["logit"]
    return out


def piece_tokens(batch, visibility="none", alignment="unaligned", max_pieces=None):
    """Grid cells (n, N, T, F): one token per evidence piece a trader sees (E2/E3), in the trader's own slot order
    (random under alignment="unaligned"), padded with empty cells to T = max_pieces (default n_pieces). A cell is
    [present, x_k, the trader's own model of piece k (coverage bit, mu_k), the provenance key of slot k at
    `visibility`]; empty cells are all zero."""
    if "cov" not in batch.oracle:
        raise ValueError("grid layouts need piece-structured inputs (E2/E3); other environments have no natural pieces")
    obs, keys, _, own = batch.agent_inputs(visibility, alignment=alignment, own_model=True)
    n, N, K = obs.shape
    T = max_pieces or K
    feat = [obs[..., None], own.reshape(n, N, 2, K).transpose(0, 1, 3, 2)]
    if keys.shape[-1]:
        feat.append(keys.reshape(n, N, 2, K).transpose(0, 1, 3, 2))
    feat = np.concatenate(feat, -1)                                   # (n, N, K, F-1), slot-indexed
    present = np.isfinite(obs)
    if present.sum(-1).max() > T:
        raise ValueError(f"a trader sees {present.sum(-1).max()} pieces > max_pieces={T}")
    cells = np.zeros((n, N, T, feat.shape[-1] + 1), np.float32)
    rank = np.cumsum(present, -1) - 1                                  # position of each seen piece in its trader
    e, i, k = np.nonzero(present)
    cells[e, i, rank[e, i, k], 0] = 1.0
    cells[e, i, rank[e, i, k], 1:] = feat[e, i, k]
    return cells


def trader_context(batch, visibility="none", alignment="unaligned", order_seed=0, with_prefix_bayes=False,
                   layout="trader", max_pieces=None, trader_encoder="linear"):
    if batch.kind != "bernoulli":
        raise ValueError("trader_transformer supports bernoulli environments only")
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}")
    if layout == "trader" and trader_encoder == "linear":
        tok = full_context(batch, visibility, alignment, own_model=True).tokens
    else:
        tok = piece_tokens(batch, visibility, alignment, max_pieces)
    n, N = tok.shape[:2]
    order = sample_orders(batch.seeds, N, order_seed)
    tokens = tok[np.arange(n)[:, None], order]
    pb = prefix_bayes_logits(batch, order) if with_prefix_bayes else None
    return TraderContext(batch.kind, tokens, order, dict(batch.public), pb, layout)


def grid_masks(N, T, layout, heads, token_heads, layers):
    """Disallowed-attention masks per layer over cells c = r * T + t: a list of (C, C) or (heads, C, C) bool arrays
    (True = may not attend)."""
    r, t = np.divmod(np.arange(N * T), T)
    tok_axis = ~((r[:, None] == r[None]) & (t[None] <= t[:, None]))
    trd_axis = ~((t[:, None] == t[None]) & (r[None] <= r[:, None]))
    if layout == "grid_full":
        return [~((t[None] <= t[:, None]) & (r[None] <= r[:, None]))] * layers
    if layout == "axial":
        return [tok_axis if l % 2 == 0 else trd_axis for l in range(layers)]
    th = heads // 2 if token_heads is None else token_heads
    if not 0 <= th <= heads:
        raise ValueError("token_heads must be in 0..heads")
    return [np.stack([tok_axis] * th + [trd_axis] * (heads - th))] * layers


def _net(in_dim, width, layers, heads, attn_layers, token_state, pos, max_traders, layout="trader", token_heads=None,
         trader_encoder="linear"):
    import torch
    nn = torch.nn
    A = set(attn_layers)

    class TraderNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("x_mean", torch.zeros(in_dim))
            self.register_buffer("x_sd", torch.ones(in_dim))
            self.inp = nn.Linear(in_dim, width)
            if trader_encoder == "deepsets":       # per-piece shared encoder phi, summed over present pieces, then rho
                self.inp = nn.Sequential(nn.Linear(in_dim, width), nn.GELU(), nn.Linear(width, width))
                self.rho = nn.Sequential(nn.GELU(), nn.Linear(width, width))
            self.pos = nn.Embedding(max_traders, width) if pos == "learned" else None
            self.tpos = nn.Embedding(max_traders, width) if pos == "learned" and layout != "trader" else None
            self.readout_tok = nn.Parameter(0.02 * torch.randn(1, 1, 1, width)) if layout != "trader" else None
            self.ln1 = nn.ModuleList(nn.LayerNorm(width) for _ in range(layers))
            self.ln2 = nn.ModuleList(nn.LayerNorm(width) for _ in range(layers))
            self.mlp = nn.ModuleList(nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
                                     for _ in range(layers))
            if token_state in ("kv", "feedback"):
                self.attn = nn.ModuleDict({str(l): nn.MultiheadAttention(width, heads, batch_first=True)
                                           for l in A})
            if token_state == "recurrent":
                self.gru = nn.ModuleDict({str(l): nn.GRUCell(width, width) for l in A})
                self.read = nn.ModuleDict({str(l): nn.Linear(width, width) for l in A})
                self.s0 = nn.ParameterDict({str(l): nn.Parameter(torch.zeros(width)) for l in A})
            if token_state == "feedback":
                self.mem_w = nn.Parameter(torch.zeros(layers + 1))
            self.norm = nn.LayerNorm(width)
            self.head = nn.Linear(width, 1)

        def embed(self, x):
            z = self.inp((x - self.x_mean) / self.x_sd)
            if trader_encoder == "deepsets":
                z = self.rho((z * x[..., :1]).sum(2))                # x[..., 0] = present flag of each piece cell
            if self.pos is not None:
                pe = self.pos(torch.arange(z.shape[1], device=z.device))
                z = z + (pe[:, None] if z.ndim == 4 else pe)
            if self.tpos is not None:
                z = z + self.tpos(torch.arange(z.shape[2], device=z.device))
            return z

        def cells(self, x):
            """Grid layouts: x (n, N, T, F) -> log-odds offset at every cell plus a dedicated readout token per trader
            (n, N, T + 1), fixing B13 so readout is never an empty padding cell."""
            n, N, T, _ = x.shape
            z_pieces = self.embed(x)
            r_tok = self.readout_tok.expand(n, N, 1, width)
            if self.pos is not None:
                r_tok = r_tok + self.pos(torch.arange(N, device=x.device))[None, :, None, :]
            T_ext = T + 1
            z = torch.cat([z_pieces, r_tok], dim=2).reshape(n, N * T_ext, width)
            masks = [torch.as_tensor(m, device=z.device) for m in grid_masks(N, T_ext, layout, heads, token_heads, layers)]
            for l in range(layers):
                if l in A:
                    a = self.ln1[l](z)
                    m = masks[l] if masks[l].ndim == 2 else masks[l].repeat(n, 1, 1)
                    z = z + self.attn[str(l)](a, a, a, attn_mask=m, need_weights=False)[0]
                z = z + self.mlp[l](self.ln2[l](z))
            return self.head(self.norm(z))[..., 0].reshape(n, N, T_ext)

        def forward(self, x):
            """x (n, N, F) in arrival order -> per-position log-odds offsets (n, N); grid layouts read the last token
            slot of each trader."""
            if layout != "trader":
                return self.cells(x)[:, :, -1]
            z = self.embed(x)
            if token_state == "feedback":
                z = self._feedback(z)
            else:
                N = z.shape[1]
                causal = torch.triu(torch.ones(N, N, dtype=torch.bool, device=z.device), 1)
                for l in range(layers):
                    if l in A:
                        a = self.ln1[l](z)
                        if token_state == "kv":
                            z = z + self.attn[str(l)](a, a, a, attn_mask=causal, need_weights=False)[0]
                        else:
                            z = self._recurrent(z, a, l)
                    z = z + self.mlp[l](self.ln2[l](z))
            return self.head(self.norm(z))[..., 0]

        def _recurrent(self, z, a, l):
            s = self.s0[str(l)].expand(len(z), -1)
            out = []
            for t in range(z.shape[1]):
                out.append(z[:, t] + self.read[str(l)](s))         # read the summary of traders < t
                s = self.gru[str(l)](a[:, t], s)                     # then add own state to it
            return torch.stack(out, 1)

        def _feedback(self, z):
            w = self.mem_w.softmax(0)
            mems, outs = [], []
            for t in range(z.shape[1]):
                h = z[:, t:t + 1]                                    # (n, 1, width)
                states = [h]
                for l in range(layers):
                    if l in A:
                        a = self.ln1[l](h)
                        kv = torch.cat(mems + [a], 1)                # one memory per earlier trader + own state
                        h = h + self.attn[str(l)](a, kv, kv, need_weights=False)[0]
                    h = h + self.mlp[l](self.ln2[l](h))
                    states.append(h)
                mems.append(sum(wi * s for wi, s in zip(w, states)))
                outs.append(h)
            return torch.cat(outs, 1)

    return TraderNet()


class TraderTransformer:
    """aggbench learned-aggregator interface: .module, .fit(TraderContext, truth), .predict(TraderContext) -> the
    LAST trader's forecast {"logit"}; .predict_prefix(ctx) -> (n, N) log-odds at every position."""

    def __init__(self, seed, in_dim, width=64, layers=3, heads=4, steps=3000, batch=512, lr=2e-3, attn_layers="all",
                 token_state="kv", loss="final", pos="none", max_traders=64, threads=2, layout="trader",
                 token_heads=None, device=None, trader_encoder="linear"):
        import torch
        if token_state not in TOKEN_STATES:
            raise ValueError(f"token_state must be one of {TOKEN_STATES}")
        if loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}")
        if pos not in ("none", "learned"):
            raise ValueError("pos must be 'none' or 'learned'")
        if layout not in LAYOUTS:
            raise ValueError(f"layout must be one of {LAYOUTS}")
        if layout != "trader" and token_state != "kv":
            raise ValueError("grid layouts use token_state='kv'")
        if trader_encoder not in ("linear", "deepsets"):
            raise ValueError("trader_encoder must be 'linear' or 'deepsets'")
        if trader_encoder == "deepsets" and layout != "trader":
            raise ValueError("trader_encoder='deepsets' applies to layout='trader'")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if threads and self.device == "cpu":
            torch.set_num_threads(threads)
        torch.manual_seed(seed)
        self.seed, self.steps, self.batch, self.lr, self.loss = seed, steps, batch, lr, loss
        self.attn_layers = resolve_attn_layers(attn_layers, layers)
        self.layout, self.trader_encoder = layout, trader_encoder
        self.module = _net(in_dim, width, layers, heads, self.attn_layers, token_state, pos, max_traders, layout,
                           token_heads, trader_encoder)
        self.uses_oracle = loss == "distill"
        self.log = []

    def loss_fn(self, logits, y, prefix_bayes=None):
        """logits (b, N) including the prior, y (b,) outcomes; prefix_bayes (b, N) for loss="distill"."""
        import torch
        bce = torch.nn.functional.binary_cross_entropy_with_logits
        if self.loss == "final":
            return bce(logits[:, -1], y)
        if self.loss == "every_prefix":
            return bce(logits, y[:, None].expand_as(logits))
        if prefix_bayes is None:
            raise ValueError('loss="distill" needs ctx.prefix_bayes (trader_context(..., with_prefix_bayes=True))')
        return bce(logits, torch.sigmoid(prefix_bayes))

    def _check(self, ctx):
        if ctx.layout != self.layout:
            raise ValueError(f"context layout {ctx.layout!r} != model layout {self.layout!r}")
        if (ctx.tokens.ndim == 4) != (self.layout != "trader" or self.trader_encoder == "deepsets"):
            raise ValueError("trader_encoder='deepsets' needs trader_context(..., trader_encoder='deepsets') and "
                             "the linear encoder needs one token per trader")

    def fit(self, ctx, truth):
        """Trains on self.device (cuda when available); the module is back on the CPU afterwards
        (controls.random_weights draws CPU tensors)."""
        import torch
        self._check(ctx)
        dev = self.device
        self.module.to(dev)
        x = torch.as_tensor(ctx.tokens, dtype=torch.float32, device=dev)
        flat = x.reshape(-1, x.shape[-1])
        with torch.no_grad():
            self.module.x_mean.copy_(flat.mean(0))
            self.module.x_sd.copy_(flat.std(0).clamp_min(1e-3))
        y = torch.as_tensor(truth, dtype=torch.float32, device=dev)
        pb = None if ctx.prefix_bayes is None else torch.as_tensor(ctx.prefix_bayes, dtype=torch.float32, device=dev)
        l0 = float(ctx.public.get("prior_logit", 0.0))
        opt = torch.optim.AdamW(self.module.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr, total_steps=self.steps, pct_start=0.05)
        gen = np.random.default_rng(self.seed)
        self.module.train()
        ema = None
        for step in range(self.steps):
            idx = torch.as_tensor(gen.integers(0, len(x), self.batch), device=dev)
            loss = self.loss_fn(l0 + self.module(x[idx]), y[idx], None if pb is None else pb[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)
            opt.step()
            sched.step()
            ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
            if (step + 1) % 500 == 0 or step + 1 == self.steps:
                self.log.append({"step": step + 1, "loss_ema": ema})
        self.module.cpu()
        return self

    def predict_prefix(self, ctx):
        import torch
        self._check(ctx)
        dev = getattr(self, "device", "cpu")
        self.module.to(dev).eval()
        with torch.no_grad():
            x = torch.as_tensor(ctx.tokens, dtype=torch.float32)
            out = torch.cat([self.module(x[s:s + 2048].to(dev)).double().cpu() for s in range(0, len(x), 2048)])
        self.module.cpu()
        return float(ctx.public.get("prior_logit", 0.0)) + out.numpy()

    def predict(self, ctx):
        return {"logit": self.predict_prefix(ctx)[:, -1]}


def evaluate_trader_transformer(env, n_train=20000, n_eval=2000, visibility="none", alignment="unaligned", seed=0,
                                n_boot=2000, threads=2, device=None, **hp):
    """Train on n_train train episodes, evaluate on n_eval dev episodes with the harness controls. Returns
    {"methods": {name: metrics.summarize}, "prefix_kl": mean KL(prefix Bayes || prediction) per position (trained
    arm), "config" (with uses_oracle), "log", "forecasts"}."""
    from .controls import run_with_controls
    from .metrics import summarize
    from .results import split_record
    from .sequential import forecast_kl
    cfg = dict(DEFAULTS, **hp)
    tr, dv = env.sample("train", n_train), env.sample("dev", n_eval)
    lay, enc = cfg["layout"], hp.get("trader_encoder", "linear")
    ctr = trader_context(tr, visibility, alignment, with_prefix_bayes=cfg["loss"] == "distill", layout=lay,
                         trader_encoder=enc)
    try:
        cdv = trader_context(dv, visibility, alignment, with_prefix_bayes=True, layout=lay, trader_encoder=enc)
    except NotImplementedError:          # E4: no prefix Bayes
        cdv = trader_context(dv, visibility, alignment, layout=lay, trader_encoder=enc)
    made = []

    def make(s):
        m = TraderTransformer(s, ctr.tokens.shape[-1], threads=threads, device=device, **cfg)
        made.append(m)
        return m

    pb_eval, cdv.prefix_bayes = cdv.prefix_bayes, None      # predict never sees the oracle
    arms = run_with_controls(make, ctr, tr.truth, cdv, seed=seed)
    name = "trader_transformer"
    methods = {name if k == "trained" else f"{name}:{k.split(':')[1]}": summarize(f, dv, n_boot=n_boot)
               for k, f in arms.items()}
    prefix_kl = None
    if pb_eval is not None:
        pred = made[0].predict_prefix(cdv)
        prefix_kl = [float(forecast_kl({"logit": pb_eval[:, t]}, {"logit": pred[:, t]}, "bernoulli").mean())
                     for t in range(pred.shape[1])]
    return {"methods": methods, "learned_arms": [name], "prefix_kl": prefix_kl,
            "controls": {name: {"random_network": f"{name}:random_network", "untrained_init": f"{name}:untrained_init"}},
            "splits": {"train": split_record(tr, "train"), "eval": split_record(dv, "dev")},
            "config": {"visibility": visibility, "alignment": alignment, "n_train": n_train, "seed": seed, **cfg,
                       "device": made[0].device,
                       "attn_layers_resolved": list(made[0].attn_layers), "uses_oracle": made[0].uses_oracle},
            "log": made[0].log, "forecasts": arms}
