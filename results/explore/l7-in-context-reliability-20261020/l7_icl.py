"""L7 (EXPLORATORY): learning who to trust in context.

A *task* is one aggbench episode's source configuration (coverage, piece strengths, adversary roles and
distortions, agent order), taken unmodified from env.episode(seed). Within a task the configuration stays
fixed; the latent Y and the evidence x are redrawn for every past episode with the environment's own law
(y ~ Bern(prior), x = (2y-1) mu + N(0, 1), report = l0 + a (cov @ 2 mu x) + b). The aggregator sees the
current reports plus the last H episodes' reports and outcomes of the same sources (no keys, visibility
"none"), and must learn reliability and overlap from that track record without weight updates.

The query episode of every evaluated task IS the harness dev episode (truth, reports, bayes unchanged), so
metrics.summarize runs on an unmodified harness Batch. Histories are drawn from per-task RNGs seeded by the
(dev or train) episode seed. The test split is never touched.

Arms (per env, per H in {0, 4, 16, 64}):
  icl_attn@H        one table transformer (attention across sources within an episode, and across episodes
                    within a source) trained on train tasks of all four Bernoulli envs; outputs per-source
                    trust weights w_i and an offset c: logit = l0 + sum_i w_i (r_i - l0) + c. At H=0 it is a
                    fixed learned aggregator of the current reports only.
  conj_bayes@H      online Bayesian learner: r | Y ~ N(c + s delta, Sigma) (s = +-1), conjugate
                    matrix-normal-inverse-Wishart prior with empirical-Bayes hyperparameters from train
                    tasks; predictive Student-t ratio. With known (c, delta, Sigma) this is exactly
                    oracle_clearing, so it converges to the oracle as H grows.
  wealth_kelly@H    market wealth update: each source bets its belief (Kelly), wealth_i *= p_i(y_t); price =
                    wealth-weighted linear pool (= Bayesian model averaging over sources).
  hedge_logpool@H   Hedge weights w_i ~ exp(-eta * cumulative log loss_i), logit = l0 + alpha sum_i w_i (r_i -
                    l0); (eta, alpha) fitted on train tasks for each H.
  fixed: harness baselines (extremized_log_pool fitted on train, log_pool, ...) and oracle_clearing / bayes.
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np

from aggbench import make_env
from aggbench.baselines import all_baselines
from aggbench.controls import run_with_controls
from aggbench.envs import View
from aggbench.metrics import bernoulli_logloss, logsig, paired_difference, summarize
from aggbench.results import make_result, split_record, write_result
from aggbench.verify import oracle_check

ENVS = ["E2", "E3-sybil", "E3-biased", "E3-collude"]
HS = [0, 4, 16, 64]
HMAX = 64
NMAX = 8
TRAIN_ROWS = 72  # rows per train task: up to 64 context + 8 queries


# ----------------------------------------------------------------------------- task sequences

def resample_task(batch, e, n_rows, rng):
    """n_rows fresh episodes of task e (same structure as batch episode e). Returns (reports (R,N), y (R,))."""
    o = batch.oracle
    mu, cov, a, b = o["mu"][e], o["cov"][e].astype(float), o["a"][e], o["b"][e]
    prior, l0 = batch.config["prior"], batch.public["prior_logit"]
    y = (rng.random(n_rows) < prior).astype(np.int64)
    x = (2 * y - 1)[:, None] * mu[None, :] + rng.standard_normal((n_rows, len(mu)))
    return report_from_x(x, mu, cov, a, b, l0), y


def report_from_x(x, mu, cov, a, b, l0):
    local = l0 + (2 * mu * x) @ cov.T
    return l0 + a * (local - l0) + b


def check_resampler(batch):
    """Max abs difference between the harness reports and report_from_x on the harness's own x."""
    o, l0 = batch.oracle, batch.public["prior_logit"]
    worst = 0.0
    for e in range(len(batch)):
        r = report_from_x(o["x"][e][None], o["mu"][e], o["cov"][e].astype(float), o["a"][e], o["b"][e], l0)[0]
        worst = max(worst, float(np.abs(r - batch.reports["logit"][e]).max()))
    return worst


def histories(batch, n_rows, salt):
    """Per-task histories, each from an RNG seeded by (episode seed, salt)."""
    R, Y = [], []
    for e in range(len(batch)):
        rng = np.random.default_rng([int(batch.seeds[e]), salt])
        r, y = resample_task(batch, e, n_rows, rng)
        R.append(r)
        Y.append(y)
    return np.stack(R), np.stack(Y)


def pad(x, axis_len=NMAX):
    w = [(0, 0)] * x.ndim
    w[-1] = (0, axis_len - x.shape[-1])
    return np.pad(x, w)


# ----------------------------------------------------------------------------- non-learned history baselines

def wealth_kelly(hist, hy, q, H, l0):
    """Wealth-weighted linear pool; wealth_i = prod_{t<H} p_i(y_t) (uniform initial wealth)."""
    s = (2 * hy[:, :H] - 1)[..., None]
    logw = logsig(s * hist[:, :H]).sum(1)  # (M, N)
    logw = logw - logw.max(1, keepdims=True)
    w = np.exp(logw)
    w /= w.sum(1, keepdims=True)
    lp = np.log((w * np.exp(logsig(q))).sum(1))
    lq = np.log((w * np.exp(logsig(-q))).sum(1))
    return lp - lq


def hedge_logpool(hist, hy, q, H, l0, eta, alpha):
    s = (2 * hy[:, :H] - 1)[..., None]
    loss = -logsig(s * hist[:, :H]).sum(1)
    z = -eta * loss
    z = z - z.max(1, keepdims=True)
    w = np.exp(z)
    w /= w.sum(1, keepdims=True)
    return l0 + alpha * (w * (q - l0)).sum(1)


def fit_hedge(hist, hy, q, y, H, l0):
    best = None
    for eta in [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
        for alpha in np.exp(np.linspace(np.log(0.2), np.log(12), 41)):
            ll = bernoulli_logloss(hedge_logpool(hist, hy, q, H, l0, eta, alpha), y).mean()
            if best is None or ll < best[0]:
                best = (ll, eta, float(alpha))
    return {"eta": best[1], "alpha": best[2], "train_logloss": float(best[0])}


class ConjBayes:
    """r_t = B^T [1, s_t] + eps, eps ~ N(0, Sigma); B | Sigma ~ MN(B0, V0, Sigma), Sigma ~ IW(Psi0, nu0)."""

    def __init__(self, N, l0, c_bar, d_bar, sig_diag, sig_off, tau, vc, vd):
        self.N, self.l0 = N, l0
        self.B0 = np.stack([np.full(N, c_bar), np.full(N, d_bar)])  # in units of r - l0
        self.V0inv = np.diag([1 / vc, 1 / vd])
        S = np.full((N, N), sig_off) + np.eye(N) * (sig_diag - sig_off)
        self.nu0 = N + 1 + tau
        self.Psi0 = tau * S  # prior mean of Sigma = S
        self.hp = {"tau": tau, "vc": vc, "vd": vd, "c_bar": c_bar, "d_bar": d_bar,
                   "sig_diag": sig_diag, "sig_off": sig_off}

    def predict(self, hist, hy, q, H):
        N, l0 = self.N, self.l0
        R = hist[:, :H] - l0
        s = 2.0 * hy[:, :H] - 1
        X = np.stack([np.ones_like(s), s], -1)  # (M, H, 2)
        Pn = self.V0inv[None] + np.einsum("mha,mhb->mab", X, X)
        Vn = np.linalg.inv(Pn)
        Bn = Vn @ (self.V0inv @ self.B0 + np.einsum("mha,mhn->man", X, R))
        Psin = (self.Psi0[None] + np.einsum("mhi,mhj->mij", R, R) + (self.B0.T @ self.V0inv @ self.B0)[None]
                - np.einsum("mai,mab,mbj->mij", Bn, Pn, Bn))
        Psin = 0.5 * (Psin + np.swapaxes(Psin, 1, 2))
        df = self.nu0 + H - N + 1
        Pinv = np.linalg.inv(Psin)
        out = []
        for sg in (1.0, -1.0):
            x = np.array([1.0, sg])
            m = np.einsum("a,man->mn", x, Bn)
            g = 1 + np.einsum("a,mab,b->m", x, Vn, x)
            d = (q - l0) - m
            quad = np.einsum("mi,mij,mj->m", d, Pinv, d) * df / g
            out.append(-(df + N) / 2 * np.log1p(quad / df) - N / 2 * np.log(g))
        return l0 + out[0] - out[1]


def train_moments(rows, ys, l0):
    """Pooled per-source class-conditional moments over train tasks (reports only, no oracle)."""
    s = 2.0 * ys - 1
    pos, neg = s > 0, s < 0
    ok = (pos.sum(1) >= 2) & (neg.sum(1) >= 2)
    rows, s, pos, neg = rows[ok] - l0, s[ok], pos[ok], neg[ok]
    mp = (rows * pos[..., None]).sum(1) / pos.sum(1, keepdims=True)
    mn = (rows * neg[..., None]).sum(1) / neg.sum(1, keepdims=True)
    c, d = (mp + mn) / 2, (mp - mn) / 2
    resid = rows - np.where(pos[..., None], mp[:, None], mn[:, None])
    cov = np.einsum("mhi,mhj->mij", resid, resid) / (rows.shape[1] - 2)
    N = rows.shape[2]
    off = ~np.eye(N, dtype=bool)
    return dict(c_bar=float(c.mean()), d_bar=float(d.mean()), sig_diag=float(np.einsum("mii->m", cov).mean() / N),
                sig_off=float(cov[:, off].mean()))


def fit_conj(rows, ys, l0, N):
    mom = train_moments(rows, ys, l0)
    hist, hy, q, y = rows[:, :HMAX], ys[:, :HMAX], rows[:, HMAX], ys[:, HMAX]
    best = None
    for tau in [0.5, 1, 2, 4, 8, 16]:
        for vc in [0.01, 0.1, 1.0]:
            for vd in [0.1, 0.3, 1.0, 3.0]:
                m = ConjBayes(N, l0, tau=tau, vc=vc, vd=vd, **mom)
                ll = np.mean([bernoulli_logloss(m.predict(hist, hy, q, H), y).mean() for H in (4, 16, 64)])
                if best is None or ll < best[0]:
                    best = (ll, m)
    best[1].hp["train_logloss_H4_16_64"] = float(best[0])
    return best[1]


# ----------------------------------------------------------------------------- in-context attention aggregator

def _torch():
    import torch
    return torch


def build_net(d=64, heads=4, layers=4):
    torch = _torch()
    nn, F = torch.nn, torch.nn.functional

    class MHA(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)

        def forward(self, x, mask):  # x (B, L, d); mask bool broadcastable to (B, 1, L, L), True = attend
            B, L, _ = x.shape
            q, k, v = self.qkv(x).reshape(B, L, 3, heads, d // heads).permute(2, 0, 3, 1, 4)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            return self.o(y.transpose(1, 2).reshape(B, L, d))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1, self.ln2, self.ln3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
            self.src, self.row = MHA(), MHA()
            self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

        def forward(self, z, src_mask, row_mask):
            B, T, N, _ = z.shape
            u = self.ln1(z).reshape(B * T, N, d)
            sm = src_mask[:, None, None, None, :].expand(B, T, 1, 1, N).reshape(B * T, 1, 1, N)
            z = z + self.src(u, sm).reshape(B, T, N, d)
            u = self.ln2(z).transpose(1, 2).reshape(B * N, T, d)
            rm = row_mask[:, None, None].expand(B, N, 1, T, T).reshape(B * N, 1, T, T)
            z = z + self.row(u, rm).reshape(B, N, T, d).transpose(1, 2)
            return z + self.ff(self.ln3(z))

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
            self.blocks = nn.ModuleList([Block() for _ in range(layers)])
            self.ln = nn.LayerNorm(d)
            self.head_w, self.head_c = nn.Linear(d, 1), nn.Linear(d, 1)

        def forward(self, r, ys, H, src_mask, Q):
            """r (B, T, N) reports minus l0; ys (B, T) in {-1, +1}; rows [0, T-Q) are context, of which the
            first H[b] are valid; the last Q rows are queries. Returns (B, Q) logits minus l0."""
            B, T, N = r.shape
            Tc = T - Q
            ar = torch.arange(T, device=r.device)
            valid = (ar[None] < H[:, None]) & (ar[None] < Tc)
            isq = (ar >= Tc).float()[None].expand(B, T)
            yfeat = torch.where(valid, ys, torch.zeros_like(ys))
            feats = torch.stack([r / 4, yfeat[..., None].expand(B, T, N), isq[..., None].expand(B, T, N)], -1)
            feats = feats * (valid | (ar[None] >= Tc))[..., None, None].float()
            z = self.embed(feats)
            row_mask = valid[:, None, :] | torch.eye(T, dtype=torch.bool, device=r.device)[None]
            for blk in self.blocks:
                z = blk(z, src_mask, row_mask)
            zq = self.ln(z[:, Tc:])  # (B, Q, N, d)
            m = src_mask[:, None, :].float()
            w = self.head_w(zq)[..., 0] * m
            pooled = (zq * m[..., None]).sum(2) / m.sum(2, keepdim=True)
            return (w * r[:, Tc:]).sum(-1) + self.head_c(pooled)[..., 0]

    return Net()


class ICLAggregator:
    """Learned aggregator for aggbench.controls: .module, .fit(view, truth), .predict(view)."""

    def __init__(self, seed, steps, device, batch=128, lr=1e-3, log=None, max_minutes=None):
        torch = _torch()
        torch.manual_seed(seed)
        self.seed, self.steps, self.device, self.batch, self.lr = seed, steps, device, batch, lr
        self.module = build_net()  # kept on CPU outside fit/predict (aggbench.controls.random_weights is CPU-only)
        self.max_minutes = max_minutes
        self.log = log if log is not None else []

    def fit(self, view, truth):
        torch = _torch()
        dev = self.device
        rows = torch.tensor(view.reports["rows_logit"] - view.public["prior_logit"], dtype=torch.float32, device=dev)
        ys = torch.tensor(2.0 * truth - 1, dtype=torch.float32, device=dev)
        mask = torch.tensor(view.reports["mask"], device=dev)
        M, T, N = rows.shape
        Q = T - HMAX
        hgrid = torch.tensor([0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64], device=dev)
        gen = torch.Generator(device=dev).manual_seed(self.seed + 17)
        self.module.to(dev)
        opt = torch.optim.AdamW(self.module.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=self.steps, pct_start=0.05)
        self.module.train()
        t0, run = time.time(), []
        for step in range(self.steps):
            idx = torch.randint(0, M, (self.batch,), device=dev, generator=gen)
            perm = torch.rand(self.batch, T, device=dev, generator=gen).argsort(1)
            r = rows[idx].gather(1, perm[..., None].expand(-1, -1, N))
            y = ys[idx].gather(1, perm)
            H = hgrid[torch.randint(0, len(hgrid), (self.batch,), device=dev, generator=gen)]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=dev.startswith("cuda")):
                out = self.module(r, y, H, mask[idx], Q)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out.float(), (y[:, -Q:] > 0).float())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)
            opt.step()
            sched.step()
            run.append(loss.detach())
            if (step + 1) % max(1, self.steps // 20) == 0:
                self.log.append({"step": step + 1, "train_bce": float(torch.stack(run).mean()), "sec": round(time.time() - t0, 1)})
                print(self.log[-1], flush=True)
                run = []
                if self.max_minutes and time.time() - t0 > 60 * self.max_minutes:
                    self.log.append({"stopped_early_at_step": step + 1, "reason": f"time cap {self.max_minutes} min"})
                    print(self.log[-1], flush=True)
                    break
        self.module.eval()
        self.module.cpu()

    def predict(self, view):
        torch = _torch()
        rp = view.reports
        l0 = view.public["prior_logit"]
        out = []
        net = copy.deepcopy(self.module).to(self.device).eval()
        with torch.no_grad():
            for s in range(0, len(rp["q_logit"]), 1024):
                sl = slice(s, s + 1024)
                r = np.concatenate([rp["hist_logit"][sl], rp["q_logit"][sl][:, None]], 1) - l0
                y = np.concatenate([2.0 * rp["hist_y"][sl] - 1, np.zeros((r.shape[0], 1))], 1)
                tt = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=self.device)
                o = net(tt(r), tt(y), tt(rp["hist_len"][sl], torch.long), tt(rp["mask"][sl], torch.bool), 1)
                out.append(o[:, 0].double().cpu().numpy())
        return {"logit": l0 + np.concatenate(out)}


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=30000, help="train tasks per env")
    ap.add_argument("--n-dev", type=int, default=2000, help="dev tasks per env")
    ap.add_argument("--n-tune", type=int, default=2000, help="train tasks per env for baseline hyperparameters")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-minutes", type=float, default=None, help="training time cap")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--track", default="EXPLORATORY", choices=["EXPLORATORY", "SMOKE"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-id", default="l7-in-context-reliability-20261020")
    args = ap.parse_args()
    torch = _torch()
    torch.set_num_threads(2 if args.device == "cpu" else 8)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    envs = {n: make_env(n) for n in ENVS}
    data = {}
    for name, env in envs.items():
        tr, dv = env.sample("train", args.n_train), env.sample("dev", args.n_dev)
        N = tr.reports["logit"].shape[1]
        tr_rows, tr_y = histories(tr, TRAIN_ROWS, salt=1)
        dv_hist, dv_hy = histories(dv, HMAX, salt=2)
        data[name] = dict(env=env, tr=tr, dv=dv, N=N, tr_rows=tr_rows, tr_y=tr_y, dv_hist=dv_hist, dv_hy=dv_hy,
                          resampler_check=check_resampler(dv))
        print(name, "data", round(time.time() - t0, 1), "s; resampler max diff", data[name]["resampler_check"], flush=True)

    # training view: all envs, padded to NMAX sources
    l0 = envs["E2"].public()["prior_logit"]
    assert all(e.public()["prior_logit"] == l0 for e in envs.values())
    tr_rows = np.concatenate([pad(data[n]["tr_rows"]) for n in ENVS])
    tr_y = np.concatenate([data[n]["tr_y"] for n in ENVS])
    tr_mask = np.concatenate([pad(np.ones((args.n_train, data[n]["N"]), bool)) for n in ENVS])
    train_view = View("bernoulli", {"rows_logit": tr_rows, "mask": tr_mask}, np.zeros(tr_rows.shape[:1] + (NMAX, 0)),
                      {"prior_logit": l0})

    # eval view: env x H blocks, each the env's dev tasks with the first H history rows
    blocks, ev = [], {k: [] for k in ("q_logit", "hist_logit", "hist_y", "hist_len", "mask")}
    for n in ENVS:
        d = data[n]
        for H in HS:
            m = args.n_dev
            ev["q_logit"].append(pad(d["dv"].reports["logit"]))
            ev["hist_logit"].append(pad(d["dv_hist"]))
            ev["hist_y"].append(d["dv_hy"])
            ev["hist_len"].append(np.full(m, H))
            ev["mask"].append(pad(np.ones((m, d["N"]), bool)))
            blocks.append((n, H))
    ev = {k: np.concatenate(v) for k, v in ev.items()}
    eval_view = View("bernoulli", ev, np.zeros(ev["q_logit"].shape + (0,)), {"prior_logit": l0})

    train_log = []
    made = []

    def make(seed):
        made.append(ICLAggregator(seed, args.steps, args.device, batch=args.batch, log=train_log,
                                  max_minutes=args.max_minutes))
        return made[-1]

    arms = run_with_controls(make, train_view, tr_y, eval_view, seed=0)
    torch.save(made[0].module.state_dict(), out / "icl_attn_trained.pt")
    np.savez_compressed(out / "forecasts.npz", blocks=np.array([f"{n}@H{h}" for n, h in blocks]),
                        **{k.replace(":", "_"): v["logit"] for k, v in arms.items()})
    print("trained", round(time.time() - t0, 1), "s", flush=True)

    def block(f, n, H):
        i = blocks.index((n, H))
        return {"logit": f["logit"][i * args.n_dev:(i + 1) * args.n_dev]}

    summary = {}
    for n in ENVS:
        d, env = data[n], data[n]["env"]
        dv, tr, N = d["dv"], d["tr"], d["N"]
        tune = slice(0, args.n_tune)
        rows_t, y_t = d["tr_rows"][tune], d["tr_y"][tune]
        forecasts, alpha = all_baselines(tr, dv, "none")
        conj = fit_conj(rows_t, y_t, l0, N)
        hedge_hp = {}
        learned, controls = [], {}
        for H in HS:
            forecasts[f"icl_attn@H{H}"] = block(arms["trained"], n, H)
            forecasts[f"icl_attn@H{H}:control:random_network"] = block(arms["control:random_network"], n, H)
            forecasts[f"icl_attn@H{H}:control:untrained_init"] = block(arms["control:untrained_init"], n, H)
            learned.append(f"icl_attn@H{H}")
            controls[f"icl_attn@H{H}"] = {"random_network": f"icl_attn@H{H}:control:random_network",
                                          "untrained_init": f"icl_attn@H{H}:control:untrained_init"}
            hist, hy, q = d["dv_hist"], d["dv_hy"], dv.reports["logit"]
            forecasts[f"conj_bayes@H{H}"] = {"logit": conj.predict(hist, hy, q, H)}
            forecasts[f"wealth_kelly@H{H}"] = {"logit": wealth_kelly(hist, hy, q, H, l0)}
            hp = fit_hedge(rows_t[:, :HMAX], y_t[:, :HMAX], rows_t[:, HMAX], y_t[:, HMAX], H, l0)
            hedge_hp[H] = hp
            forecasts[f"hedge_logpool@H{H}"] = {"logit": hedge_logpool(hist, hy, q, H, l0, hp["eta"], hp["alpha"])}
        methods = {k: summarize(f, dv) for k, f in forecasts.items()}
        paired = {k: {"vs_extremized_log_pool": paired_difference(f, forecasts["extremized_log_pool"], dv),
                      "vs_oracle_clearing": paired_difference(f, forecasts["oracle_clearing"], dv)}
                  for k, f in forecasts.items() if "@H" in k and "control" not in k}
        result = make_result(
            run_id=args.run_id, track=args.track, smoke=args.track == "SMOKE", env=env, visibility="none",
            splits={"train": split_record(tr, "train"), "eval": split_record(dv, "dev")},
            oracle_check=oracle_check(env), methods=methods, learned_arms=learned, controls=controls,
            extra={
                "question": "L7: can an aggregator attending over each source's past reports and outcomes (in "
                            "context) learn reliability and overlap, approaching oracle_clearing as H grows?",
                "history_lengths": HS,
                "history_construction": "task = harness episode's structure (oracle cov, mu, a, b, agent order); "
                                        "H past episodes redrawn with fresh Y, x from the env's law, RNG seeded by "
                                        "[episode seed, salt]; salt 1 train, 2 dev; query = the harness episode",
                "resampler_check_max_abs_diff": d["resampler_check"],
                "visibility_note": "no keys: reliability and overlap must come from the track record",
                "paired_regret_differences": paired,
                "icl_minus_conj": {str(H): paired_difference(forecasts[f"icl_attn@H{H}"],
                                                             forecasts[f"conj_bayes@H{H}"], dv) for H in HS},
                "extremize_alpha": alpha,
                "conj_bayes_hyperparameters": conj.hp,
                "hedge_hyperparameters": {str(k): v for k, v in hedge_hp.items()},
                "icl_train": {"steps": args.steps, "batch_tasks": args.batch, "queries_per_task": TRAIN_ROWS - HMAX,
                              "train_tasks_per_env": args.n_train, "envs_mixed": ENVS, "loss": "BCE on truth",
                              "log": train_log},
                "runner_sha256": __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest(),
                "baseline_tuning": f"first {args.n_tune} train tasks per env (rows 0-63 history, row 64 query)",
            })
        write_result(out, result, name=f"result-{n}.json")
        summary[n] = {k: methods[k]["regret"] for k in methods}
        print(n, {k: round(v["mean"], 4) for k, v in summary[n].items() if "control" not in k}, flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("done", round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
