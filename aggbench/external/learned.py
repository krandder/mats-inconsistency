"""Learned set aggregators built from official or third-party code, with thin aggbench adapters.

Architectures (vendored, see aggbench/external/vendor and SOURCES in __init__):
  deepsets          DeepSet class of the Set Transformer authors' repository (juho-lee/set_transformer, models.py;
                    Zaheer et al. 2017 form with mean pooling), official default width 128. Its per-agent encoder
                    is a message function (locality is checked).
  set_transformer   SetTransformer class of the same repository (Lee et al. 2019): ISAB x2 encoder, PMA + SAB x2
                    decoder; official defaults (dim_hidden 128, 4 heads, 32 inducing points, no layer norm). It reads
                    every agent's inputs jointly.
  tarmac_r{R}       TarMAC (Das et al. 2019, arXiv:1810.11187) with R communication rounds. The authors released no
                    code; the attention step is MATE's MessageAggregator (XuehaiPan/mate, MIT): each agent sends a
                    message [signature k_j, value v_j] = tanh(W h_j), forms a query q_i = W_q h_i and receives
                    c_i = sum_j softmax_j(q_i . k_j / sqrt(d_k)) v_j; its GRU then updates h_i <- GRU([x_i, c_i], h_i),
                    as in the TarMAC paper. h_i starts as an encoder of agent i's own token only (locality checked).
                    After R rounds each agent outputs a forecast and the aggregate is their mean. The agent loop
                    (encoder, GRU, readout) is ours and follows the paper's description.

Inputs (no tokenization of ours for the rich case):
  rich (default)    aggbench.transformer.full_context(batch, visibility, alignment): one token per agent with its raw
                    observation, present mask, own sensor model, provenance key and report, exactly what
                    full_context_transformer reads. Headline: alignment "unaligned"; "aligned" is a giveaway control.
  scalar baseline   a View (reports + keys at visibility none): token = [own report, key].
Training regimes (PROTOCOL, 2026-10-21; the optimizer of aggbench.transformer.DEFAULTS: AdamW lr 1e-3, weight decay
0.01, OneCycle with 5% warm-up, 3000 steps of 512, gradient clip 1.0):
  fresh (headline)  aggbench.transformer.FreshStream: every step a NEW minibatch of simulator episodes, never reused
                    (scalar inputs: ScalarFreshStream, the same seeds); standardization from its 4096 stats episodes.
  pool              the 20k train pool, early stopping on a held-out 10% of TRAIN (scored every 100 steps, patience
                    1000), never on dev.
The official architectures are used as published (no dropout added). Each object follows aggbench.controls:
.module, .fit(inputs, truth), .predict(inputs).
"""
import copy

import numpy as np
import torch
from torch import nn

from ..transformer import FRESH_STATS_START, FreshStream, full_context
from .vendor.mate_tarmac import MessageAggregator
from .vendor.set_transformer_models import DeepSet, SetTransformer

HP = {"steps": 3000, "batch": 512, "lr": 1e-3, "weight_decay": 0.01, "val_frac": 0.1, "eval_every": 100,
      "patience": 1000}


def rich_inputs(batch, visibility="noisy", alignment="unaligned"):
    """The full-context tokens of aggbench.transformer (own model on)."""
    return full_context(batch, visibility, alignment)


def _report_features(kind, reports, public):
    if kind == "bernoulli":
        return (reports["logit"] - public["prior_logit"])[..., None]
    m, P = reports["mean"], reports["prec"]
    iu = np.triu_indices(m.shape[-1])
    return np.concatenate([m, np.einsum("naij,naj->nai", P, m), P[..., iu[0], iu[1]]], -1)


def tokens(x):
    """(n, N, F) per-agent features: FullContext tokens (rich) or [own report, key] of a View (scalar baseline)."""
    if hasattr(x, "tokens"):
        return np.asarray(x.tokens)
    return np.concatenate([_report_features(x.kind, x.reports, x.public), x.keys], -1)


class ScalarFreshStream(FreshStream):
    """FreshStream for the scalar-report baseline: the same fresh train-split episodes, tokens = [report] (keys none)."""

    def __init__(self, env, batch, steps, start=0):
        super().__init__(env, "none", "aligned", batch, steps, start=start, samplers=0)
        self.env = env
        self.stats_tokens = tokens(env.sample("train", 4096, start=FRESH_STATS_START).view("none"))

    def __iter__(self):
        for s in range(self.steps):
            b = self.env.sample("train", self.batch, start=self.start + s * self.batch)
            yield tokens(b.view("none")).astype(np.float32), np.asarray(b.truth, np.float32)


def out_dim(kind, d=None):
    return 1 if kind == "bernoulli" else d + d * (d + 1) // 2


class TarMAC(nn.Module):
    def __init__(self, dim_input, dim_output, rounds=1, dim_hidden=128, key_dim=16, value_dim=32):
        super().__init__()
        self.rounds, self.key_dim, self.value_dim = rounds, key_dim, value_dim
        self.enc = nn.Sequential(nn.Linear(dim_input, dim_hidden), nn.ReLU(), nn.Linear(dim_hidden, dim_hidden))
        self.msg = nn.Linear(dim_hidden, key_dim + value_dim)
        self.agg = MessageAggregator(key_dim, value_dim, dim_hidden)
        self.gru = nn.GRUCell(dim_hidden + value_dim, dim_hidden)
        self.out = nn.Linear(dim_hidden, dim_output)

    def first_messages(self, X):                            # agent i's first message depends on its own token only
        return torch.tanh(self.msg(self.enc(X)))

    def forward(self, X):                                     # X (B, N, F)
        B, N, _ = X.shape
        x = self.enc(X)
        h = x
        for _ in range(self.rounds):
            messages = torch.tanh(self.msg(h))                # (B, N, Dk + Dv): signature and value
            # MessageAggregator takes (B', T, Na * Dm) messages and (B', T, Dh) hidden states; one row per receiver.
            msgs = messages.reshape(B, 1, 1, N * messages.shape[-1]).expand(B, N, 1, -1).reshape(B * N, 1, -1)
            c = self.agg(msgs, h.reshape(B * N, 1, -1)).reshape(B, N, -1)
            h = self.gru(torch.cat([x, c], -1).reshape(B * N, -1), h.reshape(B * N, -1)).reshape(B, N, -1)
        return self.out(h).mean(1, keepdim=True)              # (B, 1, O): mean of the agents' forecasts


def build(arch, dim_input, dim_output):
    if arch == "deepsets":
        return DeepSet(dim_input, 1, dim_output)
    if arch == "set_transformer":
        return SetTransformer(dim_input, 1, dim_output)
    if arch.startswith("tarmac_r"):
        return TarMAC(dim_input, dim_output, rounds=int(arch[len("tarmac_r"):]))
    raise ValueError(arch)


class SetAggregator:
    """aggbench learned aggregator around one of the architectures above."""

    def __init__(self, arch, kind, dim_input, d=None, seed=0, device=None, **hp):
        torch.manual_seed(seed)
        self.arch, self.kind, self.d, self.seed = arch, kind, d, seed
        self.module = build(arch, dim_input, out_dim(kind, d))
        self.hp = dict(HP, **hp)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.stats = None
        self.best_step = None
        self.log = []

    def _std(self, x):
        mu, sd = self.stats if self.stats is not None else (x.mean((0, 1)), np.maximum(x.std((0, 1)), 1e-3))
        return torch.tensor((x - mu) / sd, dtype=torch.float32)

    def _head(self, out):
        out = out[:, 0, :]
        if self.kind == "bernoulli":
            return {"logit": out[:, 0]}
        d = self.d
        L = torch.zeros(out.shape[0], d, d, dtype=out.dtype, device=out.device)
        il = torch.tril_indices(d, d)
        L[:, il[0], il[1]] = out[:, d:]
        diag = torch.nn.functional.softplus(torch.diagonal(L, dim1=1, dim2=2)) + 1e-3
        return {"mean": out[:, :d], "L": torch.tril(L, -1) + torch.diag_embed(diag)}

    def _loss(self, X, y, l0):
        f = self._head(self.module(X))
        if self.kind == "bernoulli":
            return nn.functional.binary_cross_entropy_with_logits(f["logit"] + l0, y)
        diff = torch.einsum("nji,nj->ni", f["L"], y - f["mean"])      # L^T (theta - mean), prec = L L^T
        logdet = 2 * torch.log(torch.diagonal(f["L"], dim1=1, dim2=2)).sum(1)
        return (0.5 * (diff ** 2).sum(1) - 0.5 * logdet).mean()

    def _step(self, opt, sched, loss):
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)
        opt.step()
        sched.step()

    def fit(self, inputs, truth):
        """FreshStream -> fresh regime; anything else -> pool regime with early stopping (see module docstring)."""
        hp, dev = self.hp, self.device
        stream = isinstance(inputs, FreshStream)
        x0 = inputs.stats_tokens if stream else tokens(inputs)
        self.stats = (x0.mean((0, 1)), np.maximum(x0.std((0, 1)), 1e-3))
        l0 = float(inputs.public.get("prior_logit", 0.0))
        self.module.to(dev)
        steps = inputs.steps if stream else hp["steps"]
        opt = torch.optim.AdamW(self.module.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, hp["lr"], total_steps=max(steps, 1), pct_start=0.05)
        if stream:
            self.module.train()
            for xb, yb in inputs:
                self._step(opt, sched, self._loss(self._std(xb).to(dev), torch.tensor(yb, device=dev), l0))
            self.best_step = steps
        else:
            gen = np.random.default_rng(self.seed)
            perm = gen.permutation(len(x0))
            n_val = int(round(hp["val_frac"] * len(perm)))
            X = self._std(x0).to(dev)
            Y = torch.tensor(np.asarray(truth), dtype=torch.float32, device=dev)
            val, fit_idx = torch.tensor(perm[:n_val], device=dev), perm[n_val:]
            best, best_state, best_step, step = np.inf, None, 0, 0
            for step in range(steps):
                self.module.train()
                idx = torch.tensor(fit_idx[gen.integers(0, len(fit_idx), hp["batch"])], device=dev)
                self._step(opt, sched, self._loss(X[idx], Y[idx], l0))
                if n_val and ((step + 1) % hp["eval_every"] == 0 or step + 1 == steps):
                    self.module.eval()
                    with torch.no_grad():
                        vl = sum(self._loss(X[val[s:s + 4096]], Y[val[s:s + 4096]], l0).item() * len(val[s:s + 4096])
                                 for s in range(0, n_val, 4096)) / n_val
                    if vl < best:
                        best, best_step, best_state = vl, step + 1, copy.deepcopy(self.module.state_dict())
                if n_val and hp["patience"] and step + 1 - best_step >= hp["patience"]:
                    break
            if best_state is not None:
                self.module.load_state_dict(best_state)
            self.best_step = best_step
            self.log.append({"best_step": best_step, "best_val_loss": best, "stopped_at": step + 1})
        self.module.cpu()
        return self

    def predict(self, inputs):
        self.module.eval()
        with torch.no_grad():
            X = self._std(tokens(inputs))
            outs = [self._head(self.module(X[i:i + 4096])) for i in range(0, len(X), 4096)]
        if self.kind == "bernoulli":
            logit = torch.cat([o["logit"] for o in outs]).double().numpy()
            return {"logit": float(inputs.public.get("prior_logit", 0.0)) + logit}
        mean = torch.cat([o["mean"] for o in outs]).double().numpy()
        L = torch.cat([o["L"] for o in outs]).double().numpy()
        return {"mean": mean, "prec": L @ np.swapaxes(L, 1, 2)}

    def messages(self, tok):
        """Per-agent messages before any mixing (deepsets: encoder output; tarmac: first-round message)."""
        with torch.no_grad():
            X = self._std(np.asarray(tok))
            if self.arch == "deepsets":
                return self.module.enc(X).double().numpy()
            if self.arch.startswith("tarmac"):
                return self.module.first_messages(X).double().numpy()
        raise ValueError(f"{self.arch} has no per-agent message (it reads all agents' inputs jointly)")


def locality(agg, inputs):
    """aggbench.messages.check_locality at the token level (each token holds one agent's own inputs only): True, or
    the error text."""
    from ..messages import check_locality
    tok = tokens(inputs).astype(float)
    try:
        return check_locality(lambda o, k: agg.messages(o), tok, np.zeros(tok.shape[:2] + (0,)), atol=1e-5)
    except AssertionError as err:
        return str(err)


def maker(arch, inputs, **hp):
    """Factory `make(seed)` for aggbench.controls.run_with_controls; `inputs` is a FullContext, View or stream."""
    x = inputs.stats_tokens if isinstance(inputs, FreshStream) else tokens(inputs)
    d = inputs.public.get("d")
    return lambda seed: SetAggregator(arch, inputs.kind, x.shape[-1], d=d, seed=seed, **hp)
