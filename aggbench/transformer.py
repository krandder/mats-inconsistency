"""full_context_transformer: the learned reference every aggregator is compared with (added 2026-10-21).

Kelvin: "Our aggregation mechanism cannot be worse than a transformer with self-attention that can read all the
info sequentially." and "all kinds of simple things we can imagine need to be learnable if they would be
learnable by a transformer that sees all the info sequentially and has to make a decision after."

A plain transformer encoder (pre-norm self-attention, CLS readout, no positional encoding, no hand structure)
over one token per agent. A token is everything batch.agent_inputs gives about that agent plus its report:
[raw observation (NaN -> 0), observation-present mask, its own sensor model (always), provenance key at the
chosen visibility, report]. It is trained
from outcomes (truth labels, no distillation) with the same budget as the lanes' aggregators (default 20k train
episodes, up to 3000 AdamW steps of 512, OneCycle lr 1e-3, weight decay 0.01, dropout 0.1; 10% of the train
episodes held out, scored every 100 steps, best weights kept, stop after 1000 steps without improvement) and run
through controls.run_with_controls. Without early stopping it memorizes 20k episodes (E2: dev regret 1.05).
Output: bernoulli envs log-odds = prior log-odds + head; E1 a Gaussian (mean, Cholesky factor of precision).

    from aggbench.transformer import evaluate_full_context_transformer
    rec = evaluate_full_context_transformer(env, visibility="none", alignment="unaligned")   # regime="fresh"

E3-biased: the biased agent's raw observation is honest, so this transformer sees through the bias; its E3-biased
number is not an adversarial test. Needs torch (imported lazily; the rest of aggbench stays numpy-only).
"""
from dataclasses import dataclass

import numpy as np

DEFAULTS = {"width": 64, "layers": 3, "heads": 4, "steps": 3000, "batch": 512, "lr": 1e-3, "val_frac": 0.1,
            "eval_every": 100, "patience": 1000, "dropout": 0.1, "weight_decay": 0.01}


@dataclass
class FullContext:
    """What the transformer reads for a batch: per-agent tokens (n, N, F), plus kind and public constants."""
    kind: str
    tokens: np.ndarray
    public: dict


def full_context(batch, visibility="none", alignment="unaligned", mixing=False, own_model=True):
    """One token per agent: [obs (NaN -> 0), obs-present mask, own sensor model, provenance key at `visibility`,
    report]. Reports are log-odds / 4 (bernoulli) or mean and flattened precision (E1). own_model=False drops
    the agent's own model (then Bayes is out of reach at visibility none; diagnostics only)."""
    obs, keys, public, own = batch.agent_inputs(visibility, alignment=alignment, mixing=mixing, own_model=True)
    if own_model:
        keys = np.concatenate([own, keys], -1)
    n, N = obs.shape[:2]
    o = obs.reshape(n, N, -1)
    present = np.isfinite(o)
    if batch.kind == "bernoulli":
        rep = batch.reports["logit"][..., None] / 4
    else:
        rep = np.concatenate([batch.reports["mean"], batch.reports["prec"].reshape(n, N, -1)], -1)
    sparse = "cov" in batch.oracle or "W" in batch.oracle      # E2-E4: unseen pieces / unused steps are NaN
    parts = [np.nan_to_num(o), present.astype(float)] if sparse else [o]
    tokens = np.concatenate(parts + [keys, rep], -1).astype(np.float32)
    return FullContext(batch.kind, tokens, dict(public))


FRESH_STATS_START = 9_000_000   # train-split seeds used only for input standardization in the fresh regime


_SAMPLER = {}


def _sampler_init(name, config, visibility, alignment):
    from .envs import make_env
    _SAMPLER.update(env=make_env(name, allow_heldout=True, **config), vis=visibility, al=alignment)


def _sampler_batch(args):
    start, n = args
    b = _SAMPLER["env"].sample("train", n, start=start)
    return full_context(b, _SAMPLER["vis"], _SAMPLER["al"]).tokens, np.asarray(b.truth, np.float32)


class FreshStream:
    """Training regime "fresh" (PFN-style; the headline regime): every step draws a NEW minibatch of simulator
    episodes (train-split seeds start + step * batch, never reused). Every episode has its own seed, so unaligned
    slot permutations are resampled with every minibatch. `samplers` worker processes build batches ahead
    (prefetch window 2 x samplers). Input standardization uses 4096 train-split episodes from FRESH_STATS_START,
    which are never trained on."""

    def __init__(self, env, visibility, alignment, batch, steps, start=0, samplers=2):
        from dataclasses import asdict
        self.name, self.config = env.name, asdict(env.config)
        self.visibility, self.alignment = visibility, alignment
        self.batch, self.steps, self.start, self.samplers = batch, steps, start, samplers
        self.kind, self.public = env.kind, dict(env.public())
        stats = env.sample("train", 4096, start=FRESH_STATS_START)
        self.stats_tokens = full_context(stats, visibility, alignment).tokens
        self.d = stats.public.get("d")

    @property
    def n_episodes(self):
        return self.batch * self.steps

    def __iter__(self):
        from collections import deque
        jobs = iter([(self.start + s * self.batch, self.batch) for s in range(self.steps)])
        args = (self.name, self.config, self.visibility, self.alignment)
        if not self.samplers:
            _sampler_init(*args)
            for j in jobs:
                yield _sampler_batch(j)
            return
        import multiprocessing
        with multiprocessing.get_context("fork").Pool(self.samplers, initializer=_sampler_init, initargs=args) as pool:
            pending = deque()
            for j in jobs:
                pending.append(pool.apply_async(_sampler_batch, (j,)))
                if len(pending) >= 2 * self.samplers:
                    yield pending.popleft().get()
            while pending:
                yield pending.popleft().get()


def _torch():
    import torch
    return torch


def _net(in_dim, out_dim, width, layers, heads, dropout=0.0):
    torch = _torch()
    nn = torch.nn

    class FCTNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("x_mean", torch.zeros(in_dim))
            self.register_buffer("x_sd", torch.ones(in_dim))
            self.inp = nn.Linear(in_dim, width)
            self.cls = nn.Parameter(0.02 * torch.randn(1, 1, width))
            layer = nn.TransformerEncoderLayer(width, heads, dim_feedforward=4 * width, dropout=dropout,
                                               batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(width)
            self.out = nn.Linear(width, out_dim)

        def forward(self, x):
            z = self.inp((x - self.x_mean) / self.x_sd)
            z = torch.cat([self.cls.expand(len(z), 1, -1), z], 1)
            return self.out(self.norm(self.enc(z)[:, 0]))

    return FCTNet()


def _gauss_from_head(torch, h, d):
    mean = h[:, :d]
    L = torch.zeros(len(h), d, d, dtype=h.dtype, device=h.device)
    rows, cols = torch.tril_indices(d, d)
    L[:, rows, cols] = h[:, d:]
    diag = torch.nn.functional.softplus(torch.diagonal(L, dim1=-2, dim2=-1) + 0.5413) + 1e-3
    L = torch.tril(L, -1) + torch.diag_embed(diag)
    return mean, L


class FullContextTransformer:
    """aggbench learned-aggregator interface: .module, .fit(FullContext, truth), .predict(FullContext)."""

    def __init__(self, seed, in_dim, kind, d=None, width=64, layers=3, heads=4, steps=3000, batch=512, lr=2e-3,
                 threads=None, device="cpu", val_frac=0.1, eval_every=100, weight_decay=1e-4, dropout=0.0,
                 patience=None):
        torch = _torch()
        if threads:
            torch.set_num_threads(threads)
        torch.manual_seed(seed)
        self.seed, self.kind, self.d, self.device = seed, kind, d, device
        self.steps, self.batch, self.lr = steps, batch, lr
        self.val_frac, self.eval_every, self.weight_decay, self.patience = val_frac, eval_every, weight_decay, patience
        self.best_step = None
        out_dim = 1 if kind == "bernoulli" else d + d * (d + 1) // 2
        self.module = _net(in_dim, out_dim, width, layers, heads, dropout)
        self.log = []

    def _loss(self, torch, h, y, l0):
        if self.kind == "bernoulli":
            return torch.nn.functional.binary_cross_entropy_with_logits(l0 + h[:, 0], y)
        mean, L = _gauss_from_head(torch, h, self.d)
        r = torch.einsum("nji,nj->ni", L, y - mean)                    # L^T (theta - mean)
        return (0.5 * (r ** 2).sum(1) - torch.log(torch.diagonal(L, dim1=-2, dim2=-1)).sum(1)).mean()

    def fit(self, ctx, truth):
        """Trains on `self.device` (cuda when available, see evaluate_full_context_transformer); the module is back
        on the CPU afterwards (controls.random_weights draws CPU tensors). Early stopping: a held-out val_frac of
        the TRAIN episodes is scored every eval_every steps and the best weights are kept (without it the
        transformer memorizes 20k continuous-input episodes: train regret < 0, dev regret > the prior's)."""
        import copy
        torch = _torch()
        dev = getattr(self, "device", "cpu")
        if isinstance(ctx, FreshStream):
            return self._fit_stream(torch, dev, ctx)
        self.module.to(dev)
        gen = np.random.default_rng(self.seed)
        perm = gen.permutation(len(ctx.tokens))
        n_val = int(round(self.val_frac * len(perm))) if self.val_frac else 0
        val, fit_idx = perm[:n_val], perm[n_val:]
        x = torch.tensor(ctx.tokens[fit_idx], device=dev)
        y = torch.tensor(np.asarray(truth)[fit_idx], dtype=torch.float32, device=dev)
        xv = torch.tensor(ctx.tokens[val], device=dev)
        yv = torch.tensor(np.asarray(truth)[val], dtype=torch.float32, device=dev)
        flat = x.reshape(-1, x.shape[-1])
        with torch.no_grad():
            self.module.x_mean.copy_(flat.mean(0))
            self.module.x_sd.copy_(flat.std(0).clamp_min(1e-3))
        l0 = float(ctx.public.get("prior_logit", 0.0))
        opt = torch.optim.AdamW(self.module.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr, total_steps=self.steps, pct_start=0.05)
        best, best_state, best_step = np.inf, None, 0
        ema = None
        for step in range(self.steps):
            self.module.train()
            idx = torch.tensor(gen.integers(0, len(x), self.batch), device=dev)
            loss = self._loss(torch, self.module(x[idx]), y[idx], l0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)
            opt.step()
            sched.step()
            ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
            if n_val and ((step + 1) % self.eval_every == 0 or step + 1 == self.steps):
                self.module.eval()
                with torch.no_grad():
                    vl = float(sum(self._loss(torch, self.module(xv[s:s + 4096]), yv[s:s + 4096], l0).item()
                                   * len(xv[s:s + 4096]) for s in range(0, len(xv), 4096)) / len(xv))
                if vl < best:
                    best, best_step = vl, step + 1
                    best_state = copy.deepcopy(self.module.state_dict())
            if (step + 1) % 500 == 0:
                self.log.append({"step": step + 1, "loss_ema": ema, "best_val": best, "best_step": best_step})
            if n_val and self.patience and step + 1 - best_step >= self.patience:
                self.log.append({"step": step + 1, "stopped_early": True, "best_val": best, "best_step": best_step})
                break
        if best_state is not None:
            self.module.load_state_dict(best_state)
        self.best_step = best_step
        self.module.cpu()
        return self

    def _fit_stream(self, torch, dev, stream):
        """Fresh regime: one gradient step per fresh minibatch, no reuse, no early stopping (nothing to overfit)."""
        self.module.to(dev)
        flat = torch.tensor(stream.stats_tokens.reshape(-1, stream.stats_tokens.shape[-1]), device=dev)
        with torch.no_grad():
            self.module.x_mean.copy_(flat.mean(0))
            self.module.x_sd.copy_(flat.std(0).clamp_min(1e-3))
        l0 = float(stream.public.get("prior_logit", 0.0))
        opt = torch.optim.AdamW(self.module.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, self.lr, total_steps=stream.steps, pct_start=0.05)
        self.module.train()
        ema = None
        for step, (xb, yb) in enumerate(stream):
            loss = self._loss(torch, self.module(torch.tensor(xb, device=dev)), torch.tensor(yb, device=dev), l0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)
            opt.step()
            sched.step()
            ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
            if (step + 1) % 500 == 0:
                self.log.append({"step": step + 1, "loss_ema": ema})
        self.best_step = stream.steps
        self.module.cpu()
        return self

    def predict(self, ctx):
        torch = _torch()
        self.module.eval()
        outs = []
        with torch.no_grad():
            x = torch.tensor(ctx.tokens)
            for s in range(0, len(x), 2048):
                outs.append(self.module(x[s:s + 2048]).double())
        h = torch.cat(outs)
        if self.kind == "bernoulli":
            return {"logit": float(ctx.public.get("prior_logit", 0.0)) + h[:, 0].numpy()}
        mean, L = _gauss_from_head(torch, h, self.d)
        return {"mean": mean.numpy(), "prec": (L @ L.transpose(1, 2)).numpy()}


def evaluate_full_context_transformer(env, n_train=20000, n_eval=2000, visibility="none", alignment="unaligned",
                                      seed=0, n_boot=2000, threads=None, device=None, regime="fresh", samplers=2,
                                      **hp):
    """Train, then evaluate on n_eval dev episodes with the harness controls.

    regime="fresh" (default, the headline regime): steps x batch fresh simulator episodes, one minibatch per step,
    no reuse, slot permutations resampled per minibatch (FreshStream); n_train is ignored.
    regime="pool": a fixed pool of n_train train episodes with early stopping on 10% of them (never dev).
    Returns {"methods": {name: metrics.summarize}, "learned_arms", "controls", "splits", "config", "log"}."""
    from .controls import run_with_controls
    from .metrics import summarize
    from .results import split_record
    from .splits import episode_seeds
    if regime not in ("fresh", "pool"):
        raise ValueError("regime must be 'fresh' or 'pool'")
    cfg = dict(DEFAULTS, **hp)
    dv = env.sample("dev", n_eval)
    cdv = full_context(dv, visibility, alignment)
    if regime == "pool":
        tr = env.sample("train", n_train)
        ctr, truth = full_context(tr, visibility, alignment), tr.truth
        train_split = split_record(tr, "train")
    else:
        ctr = FreshStream(env, visibility, alignment, cfg["batch"], cfg["steps"], samplers=samplers)
        truth = None
        train_split = {"split": "train", "first_seed": int(episode_seeds("train", 1)[0]), "n": ctr.n_episodes}
    in_dim = (ctr.stats_tokens if regime == "fresh" else ctr.tokens).shape[-1]
    d = dv.public.get("d")
    made = []

    if device is None:
        device = "cuda" if _torch().cuda.is_available() else "cpu"

    def make(s):
        m = FullContextTransformer(s, in_dim, dv.kind, d=d, threads=threads, device=device, **cfg)
        made.append(m)
        return m

    arms = run_with_controls(make, ctr, truth, cdv, seed=seed)
    name = "full_context_transformer"
    methods = {name if k == "trained" else f"{name}:{k.split(':')[1]}": summarize(f, dv, n_boot=n_boot)
               for k, f in arms.items()}
    methods["bayes"] = summarize(dict(dv.bayes), dv, n_boot=n_boot)
    return {"methods": methods, "learned_arms": [name],
            "controls": {name: {"random_network": f"{name}:random_network", "untrained_init": f"{name}:untrained_init"}},
            "splits": {"train": train_split, "eval": split_record(dv, "dev")},
            "config": {"visibility": visibility, "alignment": alignment, "regime": regime,
                       "n_train": n_train if regime == "pool" else train_split["n"], "seed": seed,
                       "device": device, **cfg},
            "log": made[0].log, "forecasts": arms}


def decompose_regret(aggregator_forecast, messages_train, messages_eval, train_batch, eval_batch, distill=True,
                     n_boot=2000, seed=0, threads=None, **hp):
    """Split an aggregator's regret to Bayes into information lost by its messages and aggregation error.

    messages_*: (n, N, m) per-agent messages (whatever the agents send; each agent's own inputs only). A
    high-capacity predictor (FullContextTransformer over one token per message, default 2x width and 2x steps)
    is trained on the train messages to estimate the best-from-messages forecast: to the outcomes, or with
    distill=True (bernoulli only, declared in the output) to the soft Bayes posterior of the train episodes,
    which has much lower variance. Then on eval_batch:

        total             = regret(aggregator)
        information_loss  = regret(best_from_messages)              (messages versus Bayes)
        aggregation_error = regret(aggregator) - regret(best_from_messages), paired

    The predictor is an estimate of the best-from-messages ceiling from below (a weak predictor inflates
    information_loss and can make aggregation_error negative); report it with its CI and training budget."""
    from .metrics import paired_difference, sigmoid, summarize
    kind = eval_batch.kind
    if distill and kind != "bernoulli":
        raise NotImplementedError("distill=True is implemented for bernoulli environments; pass distill=False")
    cfg = dict(DEFAULTS, width=2 * DEFAULTS["width"], steps=2 * DEFAULTS["steps"])
    cfg.update(hp)
    ctr = FullContext(kind, np.asarray(messages_train, np.float32), dict(train_batch.public))
    cev = FullContext(kind, np.asarray(messages_eval, np.float32), dict(eval_batch.public))
    target = sigmoid(train_batch.bayes["logit"]) if distill else train_batch.truth
    model = FullContextTransformer(seed, ctr.tokens.shape[-1], kind, d=train_batch.public.get("d"),
                                   threads=threads, **cfg).fit(ctr, target)
    best = model.predict(cev)
    return {
        "total": summarize(aggregator_forecast, eval_batch, n_boot=n_boot)["regret"],
        "information_loss": summarize(best, eval_batch, n_boot=n_boot)["regret"],
        "aggregation_error": paired_difference(aggregator_forecast, best, eval_batch, n_boot=n_boot),
        "predictor": {"labels": "bayes posterior (distillation)" if distill else "truth", **cfg},
        "best_from_messages": best, "log": model.log,
    }
