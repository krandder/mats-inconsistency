"""ForecastWorld: a high-dimensional forecasting world with news-like sources, forecasts plus source keys,
and exact Bayes ceilings (added 2026-10-22; stage 2 of the Deger plan, no LLMs).

The plan (2026-10-20 addendum): language models that output forecasts plus key-like descriptions of the
information they have, with aggregation on top. This module is that
setting with the LLMs replaced by exact Bayesian forecasters, so every number has an exact ceiling.

World (one fixed INSTANCE per instance_seed; episodes are fresh draws inside it):
  latent   theta ~ N(0, I_D), D = 50
  question y_q = 1[a_q . theta + e_q > 0], e_q ~ N(0, s_q^2), q = 1..Q (Q = 20); a_q touches q_factors factors
  source   s_j = B_j theta + n_j, n_j ~ N(0, sigma_j^2 I_rows); B_j (rows x D) touches src_factors factors,
           sigma_j log-uniform (varied reliability). M = 40 training sources plus M_heldout sources never
           used in training (their embeddings E_j are never seen by a learned aggregator during training).
  popularity  source with popularity rank r has weight r^(-alpha) (Zipf). alpha = 0: uniform reading;
           large alpha: most forecasters read the same few sources (shared errors, double counting).
  forecaster i reads k distinct sources S_i, drawn without replacement in proportion to popularity
           (Plackett-Luce via Gumbel top-k), and outputs
           (i)  forecasts on all Q questions, the exact local posterior:
                P_i(y_q = 1) = Phi(a_q . mu_i / sqrt(a_q' Sigma_i a_q + s_q^2)),
                (mu_i, Sigma_i) = Gaussian posterior of theta given its sources;
           (ii) a key  key_i = sum_{j in S_i} E_j + key_sigma * eps,  E_j ~ N(0, I_16) fixed per source.

Ceilings (both exact):
  (A) bayes_full: posterior given every source read by anyone (union U), then the exact probit per question.
  (B) bayes_oracle_keys: posterior given the forecasts and the exact source sets. The probit is inverted:
      z_iq = Phi^-1(p_iq) * sqrt(a_q' Sigma_i a_q + s_q^2) = a_q . mu_i, and mu_i = Sigma_i sum_{j in S_i}
      B_j' s_j / sigma_j^2 is linear in the source readings s. So z = H s with H known from the sets, and the
      posterior of theta given z is linear-Gaussian (GLS). We whiten s = L w (L L' = Cov(s)), take the row
      space V_r of H L (SVD, relative tolerance rank_tol), and condition on V_r' w:
          mean = F V_r S_r^-1 U_r' z,   cov = I - F V_r V_r' F',   F = B' L^-T.
      (B) is a coarsening of (A), so its expected log loss is never below (A)'s. It equals (A) for N = 1, and
      when sets are disjoint and Q >= D. With Q < D the forecasts reveal only Q projections of each mu_i, and
      with overlapping sets the shared part cannot be subtracted, so (B) is below (A).

Metrics are per question: KL(A || method) averaged over the Q questions (the regret: its expectation equals
the expected log-loss excess over A), log loss and Brier on the outcomes. Bootstrap CIs are over episodes.

Episodes are generated in chunks with torch (float64): chunk c of split S uses generator seed
splits.episode_seeds(S, 1, start=c), so train / dev / test chunks are disjoint by construction.
"""
from dataclasses import asdict, dataclass

import numpy as np

from .splits import episode_seeds

NAME = "forecast_world"


@dataclass
class ForecastWorldConfig:
    D: int = 50
    Q: int = 20
    M: int = 40
    M_heldout: int = 10
    rows: int = 8               # scalar readings per source
    src_factors: int = 8        # factors a source touches
    q_factors: int = 5          # factors a question touches
    src_noise: tuple = (0.3, 3.0)   # sigma_j log-uniform in this range
    q_noise: tuple = (0.3, 0.8)     # s_q uniform in this range (a_q has unit norm)
    key_dim: int = 16
    N: int = 10
    k: int = 3
    alpha: float = 1.0
    key_sigma: float = 0.5
    instance_seed: int = 0
    # raw observations for LEARNED forecasters (added 2026-10-22 redirect): o_j = R [s_j; nu_j], R a fixed random
    # orthogonal (obs_dim x obs_dim) map shared by all sources, nu_j ~ N(0, nuis_sd^2) nuisance features
    obs_dim: int = 32
    nuis_sd: float = 3.0
    # source descriptions: desc_j = P z_j + desc_noise * eps_j (fixed per source), z_j = [per-factor loading norm
    # of B_j (D values, zero off the factors it touches), log sigma_j]: WHICH factors a source observes and how
    # reliably. desc_random_j is an unrelated random vector of the same size (labelled control).
    desc_dim: int = 32
    desc_noise: float = 0.1
    # no-overlap control (gate 2026-10-23): the N*k sources read in an episode are all distinct (drawn without
    # replacement by popularity, then dealt to forecasters at random)
    disjoint: bool = False


def _torch():
    import torch
    return torch


class ForecastWorld:
    kind = "bernoulli"

    def __init__(self, **overrides):
        torch = _torch()
        self.config = ForecastWorldConfig(**overrides)
        self.name = NAME
        c = self.config
        g = torch.Generator().manual_seed(1_000_003 * (c.instance_seed + 1))
        f64 = torch.float64
        Mt = c.M + c.M_heldout
        A = torch.zeros(c.Q, c.D, dtype=f64)
        for q in range(c.Q):
            idx = torch.randperm(c.D, generator=g)[:c.q_factors]
            A[q, idx] = torch.randn(c.q_factors, generator=g, dtype=f64)
        self.A = A / A.norm(dim=1, keepdim=True)
        lo, hi = c.q_noise
        self.s_q = lo + (hi - lo) * torch.rand(c.Q, generator=g, dtype=f64)
        B = torch.zeros(Mt, c.rows, c.D, dtype=f64)
        for j in range(Mt):
            idx = torch.randperm(c.D, generator=g)[:c.src_factors]
            B[j][:, idx] = torch.randn(c.rows, c.src_factors, generator=g, dtype=f64) / np.sqrt(c.src_factors)
        self.B = B
        lo, hi = np.log(c.src_noise[0]), np.log(c.src_noise[1])
        self.src_sd = torch.exp(lo + (hi - lo) * torch.rand(Mt, generator=g, dtype=f64))
        self.E = torch.randn(Mt, c.key_dim, generator=g, dtype=f64)
        # popularity ranks over all Mt sources (random, independent of reliability); training sources keep
        # their relative order when held-out sources are excluded
        self.rank_all = torch.randperm(Mt, generator=g).to(f64) + 1.0
        self.J = torch.einsum("jrd,jre->jde", B, B) / self.src_sd[:, None, None] ** 2   # (Mt, D, D)
        # whitening for ceiling B: s (all sources stacked, Mt*rows) = Bs theta + n
        Bs = B.reshape(Mt * c.rows, c.D)
        R = (self.src_sd[:, None] ** 2).expand(Mt, c.rows).reshape(-1)
        Cs = Bs @ Bs.T + torch.diag(R)
        self.L = torch.linalg.cholesky(Cs)
        self.F = torch.linalg.solve_triangular(self.L, Bs, upper=False).T     # B' L^-T  (D, S)
        self.G = (Bs / R[:, None]).T                                          # B' R^-1  (D, S)
        # learned-forecaster extras come from a separate stream, so every earlier instance quantity is unchanged
        g2 = torch.Generator().manual_seed(7_000_003 * (c.instance_seed + 1))
        self.R_obs = torch.linalg.qr(torch.randn(c.obs_dim, c.obs_dim, generator=g2, dtype=f64))[0]
        z = torch.cat([B.norm(dim=1), torch.log(self.src_sd)[:, None]], 1)              # (Mt, D + 1)
        P = torch.randn(c.desc_dim, c.D + 1, generator=g2, dtype=f64) / np.sqrt(c.D + 1)
        desc = z @ P.T
        desc = (desc - desc.mean(0)) / desc.std(0)
        self.desc = desc + c.desc_noise * torch.randn(Mt, c.desc_dim, generator=g2, dtype=f64)
        self.desc_random = torch.randn(Mt, c.desc_dim, generator=g2, dtype=f64)
        self.device = "cpu"

    def to(self, device):
        """Move the instance tensors to `device` (randomness is always drawn on the CPU, so episodes are identical
        on every device up to floating-point rounding)."""
        self.device = device
        for name in ("A", "s_q", "B", "src_sd", "E", "J", "L", "F", "G", "R_obs", "desc", "desc_random"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    # ------------------------------------------------------------------ public facts
    def public(self):
        c = self.config
        return {"Q": c.Q, "D": c.D, "M": c.M, "M_heldout": c.M_heldout, "key_dim": c.key_dim, "N": c.N, "k": c.k}

    def popularity(self, heldout=False):
        """Reading probabilities' weights over the Mt sources (held-out sources weight 0 unless heldout)."""
        torch = _torch()
        c = self.config
        avail = torch.ones(c.M + c.M_heldout, dtype=torch.bool)
        rank_all = self.rank_all
        if not heldout:
            avail[c.M:] = False
        r = torch.zeros_like(rank_all)
        r[avail] = torch.argsort(torch.argsort(rank_all[avail])).to(r.dtype) + 1.0   # re-rank 1..m
        w = torch.where(avail, r ** (-c.alpha), torch.zeros_like(r))
        return w / w.sum()

    def top_share(self, top=3, heldout=False, n=20000, seed=123):
        """Share of forecasters that read at least one of the `top` most popular sources (Monte Carlo)."""
        torch = _torch()
        w = self.popularity(heldout)
        g = torch.Generator().manual_seed(seed)
        sets = self._draw_sets(g, n, 1, w)[:, 0]
        topidx = torch.argsort(w, descending=True)[:top]
        return float(sets[:, topidx].any(-1).double().mean())

    def _draw_disjoint(self, g, n, N, w):
        torch = _torch()
        k, Mt = self.config.k, w.shape[0]
        if N * k > int((w > 0).sum()):
            raise ValueError("disjoint sets need N*k <= number of available sources")
        u = torch.rand(n, Mt, generator=g, dtype=torch.float64).clamp_min(1e-300)
        score = torch.where(w > 0, torch.log(w.clamp_min(1e-300)) - torch.log(-torch.log(u)),
                            torch.full_like(u, -np.inf))
        idx = score.topk(N * k, dim=-1).indices                                             # (n, N*k)
        shuffle = torch.argsort(torch.rand(n, N * k, generator=g, dtype=torch.float64), -1)
        idx = torch.gather(idx, 1, shuffle).view(n, N, k)
        mask = torch.zeros(n, N, Mt, dtype=torch.float64)
        mask.scatter_(-1, idx, 1.0)
        return mask

    def _draw_sets(self, g, n, N, w):
        torch = _torch()
        Mt = w.shape[0]
        u = torch.rand(n, N, Mt, generator=g, dtype=torch.float64).clamp_min(1e-300)
        gumbel = -torch.log(-torch.log(u))
        score = torch.where(w > 0, torch.log(w.clamp_min(1e-300)) + gumbel, torch.full_like(gumbel, -np.inf))
        idx = score.topk(self.config.k, dim=-1).indices
        mask = torch.zeros(n, N, Mt, dtype=torch.float64)
        mask.scatter_(-1, idx, 1.0)
        return mask

    # ------------------------------------------------------------------ sampling
    def sample(self, split, n, chunk=0, heldout=False, N=None, key_sigma=None, ceiling_b=False, obs=False):
        """n fresh episodes from chunk `chunk` of `split` (generator seed = splits.episode_seeds(split, 1, chunk)).
        heldout=True lets forecasters read the held-out sources too. Returns a dict of float64 torch tensors."""
        torch = _torch()
        c = self.config
        N = c.N if N is None else N
        key_sigma = c.key_sigma if key_sigma is None else key_sigma
        seed = int(episode_seeds(split, 1, start=chunk)[0]) * 2 + int(heldout)
        g = torch.Generator().manual_seed(seed)
        f64 = torch.float64
        Mt = c.M + c.M_heldout
        dev = self.device
        theta = torch.randn(n, c.D, generator=g, dtype=f64).to(dev)
        e = torch.randn(n, c.Q, generator=g, dtype=f64).to(dev) * self.s_q
        lat = theta @ self.A.T
        y = (lat + e > 0).to(f64)
        s = torch.einsum("jrd,nd->njr", self.B, theta) + \
            torch.randn(n, Mt, c.rows, generator=g, dtype=f64).to(dev) * self.src_sd[None, :, None]
        h = torch.einsum("jrd,njr->njd", self.B, s / self.src_sd[None, :, None] ** 2)      # (n, Mt, D)
        if c.disjoint:
            mask = self._draw_disjoint(g, n, N, self.popularity(heldout)).to(dev)
        else:
            mask = self._draw_sets(g, n, N, self.popularity(heldout)).to(dev)              # (n, N, Mt)
        eps = torch.randn(n, N, c.key_dim, generator=g, dtype=f64).to(dev)
        out = {"theta": theta, "y": y, "s": s, "mask": mask}
        out.update(self.local_posteriors(mask, h))
        out["key_clean"], out["key_eps"] = mask @ self.E, eps
        out["key"] = out["key_clean"] + key_sigma * eps
        union = (mask.sum(1) > 0).to(f64)                                                  # (n, Mt)
        pu = self.local_posteriors(union[:, None], h)
        out["bayes_probit"] = pu["probit"][:, 0]
        out["union"] = union
        if ceiling_b:
            out["oracle_probit"] = self.ceiling_b(out)
        if obs:     # raw observations from a separate stream: the episode above is identical with or without them
            go = torch.Generator().manual_seed(seed + 500_000_000)
            nu = torch.randn(n, Mt, c.obs_dim - c.rows, generator=go, dtype=f64).to(dev) * c.nuis_sd
            out["obs"] = torch.cat([s, nu], -1) @ self.R_obs.T                                # (n, Mt, obs_dim)
            out["idx"] = mask.topk(c.k, dim=-1).indices.sort(-1).values                      # (n, N, k)
        return out

    def local_posteriors(self, mask, h):
        """Exact local forecasts for reading sets `mask` (n, N, Mt) given per-source information vectors h."""
        torch = _torch()
        c = self.config
        n, N, _ = mask.shape
        P = torch.eye(c.D, dtype=h.dtype, device=h.device) + torch.einsum("nim,mde->nide", mask, self.J)
        hi = torch.einsum("nim,nmd->nid", mask, h)
        Lc = torch.linalg.cholesky(P)
        mu = torch.cholesky_solve(hi[..., None], Lc)[..., 0]                               # (n, N, D)
        X = torch.cholesky_solve(self.A.T.expand(n, N, c.D, c.Q), Lc)                       # Sigma_i A'
        var = (self.A.T * X).sum(-2)                                                       # a_q' Sigma_i a_q
        z = mu @ self.A.T                                                                  # a_q . mu_i
        denom = torch.sqrt(var + self.s_q ** 2)
        return {"probit": z / denom, "denom": denom, "mu": mu, "X": X}

    def ceiling_b(self, ep, rank_tol=1e-9, chunk=None):
        """Exact posterior probit per question given the forecasts and the exact source sets (GLS)."""
        torch = _torch()
        c = self.config
        dev = ep["mask"].device
        if not hasattr(self, "_cpu"):     # batched float64 SVD is run on the CPU
            self._cpu = {k: getattr(self, k).cpu() for k in ("A", "s_q", "G", "L", "F")}
        A, s_q, G, L, F = (self._cpu[k] for k in ("A", "s_q", "G", "L", "F"))
        mask, X = ep["mask"].cpu(), ep["X"].cpu()
        n, N, Mt = mask.shape
        chunk = chunk or max(8, 2048 // N)
        z = (ep["probit"] * ep["denom"]).cpu().reshape(n, N * c.Q)          # invert the probit: z_iq = a_q . mu_i
        colmask = mask.repeat_interleave(c.rows, dim=-1)              # (n, N, S)
        out = []
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            # H[i, q, :] = a_q' Sigma_i B_j' R_j^-1 on the columns of j in S_i
            H = torch.einsum("nidq,ds->niqs", X[a:b], G) * colmask[a:b, :, None, :]
            H = H.reshape(b - a, N * c.Q, -1)
            Hw = H @ L                                           # z = H s = (H L) w
            U, S, Vh = torch.linalg.svd(Hw, full_matrices=False)
            keep = S > rank_tol * S[:, :1]
            Sinv = torch.where(keep, 1.0 / S.clamp_min(1e-300), torch.zeros_like(S))
            wproj = (Vh.transpose(-1, -2) * Sinv[:, None, :]) @ (U.transpose(-1, -2) @ z[a:b, :, None])  # V_r V_r' w
            mean = (F @ wproj)[..., 0]                            # (b-a, D)
            FV = F @ (Vh.transpose(-1, -2) * keep[:, None, :].to(S.dtype))   # F V_r
            AFV = A @ FV                                          # (b-a, Q, S)
            var = 1.0 - (AFV ** 2).sum(-1)                             # a_q'(I - F V_r V_r' F')a_q, ||a_q|| = 1
            out.append(mean @ A.T / torch.sqrt(var + s_q ** 2))
        return torch.cat(out).to(dev)

    def describe(self):
        return {"name": self.name, "config": asdict(self.config)}


# ---------------------------------------------------------------------- probability helpers (torch)
def probit_to_logit(u):
    torch = _torch()
    return torch.special.log_ndtr(u) - torch.special.log_ndtr(-u)


def logit_metrics(logit, bayes_probit, y):
    """Per-episode metrics (mean over questions): kl = KL(A || q), logloss, brier, regret (logloss - A's)."""
    torch = _torch()
    logit = logit.to(torch.float64)
    lsig = torch.nn.functional.logsigmoid
    lp, lm = lsig(logit), lsig(-logit)
    bp, bm = torch.special.log_ndtr(bayes_probit), torch.special.log_ndtr(-bayes_probit)
    pb = bp.exp()
    kl = pb * (bp - lp) + (1 - pb) * (bm - lm)
    ll = -(y * lp + (1 - y) * lm)
    llb = -(y * bp + (1 - y) * bm)
    brier = (torch.sigmoid(logit) - y) ** 2
    return {k: v.mean(-1).cpu().numpy() for k, v in
            {"kl": kl, "logloss": ll, "brier": brier, "regret": ll - llb}.items()}


# ---------------------------------------------------------------------- non-learned pools
def pool_mean_prob(logit):
    torch = _torch()
    p = torch.sigmoid(logit).mean(1).clamp(1e-12, 1 - 1e-12)
    return torch.log(p) - torch.log1p(-p)


def pool_mean_logodds(logit):
    return logit.mean(1)


def pool_median(logit):
    return logit.median(1).values if logit.shape[1] % 2 else logit.sort(1).values[:, [logit.shape[1] // 2 - 1,
                                                                                       logit.shape[1] // 2]].mean(1)


def pool_sum_logodds(logit):
    """Naive independent-evidence product (prior log-odds is 0 here): double counts shared sources."""
    return logit.sum(1)


def fit_extremize(logit, y, grid=None):
    """Extremizing factor for the mean log-odds, fit by log loss on training episodes (grid then refine)."""
    torch = _torch()
    m = logit.mean(1)
    lsig = torch.nn.functional.logsigmoid

    def loss(a):
        return float(-(y * lsig(a * m) + (1 - y) * lsig(-a * m)).mean())
    grid = np.linspace(0.25, 12.0, 95) if grid is None else grid
    best = min(grid, key=loss)
    fine = np.linspace(max(0.05, best - 0.25), best + 0.25, 51)
    return float(min(fine, key=loss))


# ---------------------------------------------------------------------- learned aggregator (set attention)
ARMS = ("nokey", "key", "oracle")


def token_features(ep, arm, key_sigma=None, M_total=None, standardize=False):
    """Per-forecaster token: [forecast log-odds on all Q questions / 4, then nothing (nokey), the noisy key
    (key), or the exact multi-hot source set (oracle)]. Returns (tokens float32, log-odds float32)."""
    torch = _torch()
    lo = probit_to_logit(ep["probit"]).clamp(-20, 20)
    parts = [lo / 4]
    if arm == "key":
        key = ep["key_clean"] + key_sigma * ep["key_eps"]
        if standardize:     # key coordinates have sd sqrt(k + sigma^2); bring them to unit scale
            key = key / torch.sqrt(ep["mask"].sum(-1, keepdim=True) + key_sigma ** 2)
        parts.append(key)
    elif arm == "oracle":
        parts.append(ep["mask"])
    elif arm != "nokey":
        raise ValueError(f"unknown arm {arm!r}")
    return torch.cat(parts, -1).float(), lo.float()


def make_aggregator(in_dim, Q, width=128, layers=3, heads=4, dropout=0.0, seed=0, readout="diag", split=None):
    """Permutation-invariant set transformer over forecaster tokens (no positional encoding) with a CLS token.
    Output log-odds per question: out_q = sum_i beta_iq * l_iq + head(CLS)_q, where l_iq are the forecasters'
    log-odds and beta_iq is read from each forecaster's final token (a context-dependent log-linear pool that
    can discount duplicates or go negative), plus a free correction from the CLS token. readout="full" reads a
    Q x Q matrix per forecaster instead: out_q = sum_i sum_q' beta_iqq' l_iq' (the form of the GLS posterior mean,
    which uses every forecast of every forecaster for every question). split=Q gives the forecasts and the rest of
    the token (key) separate input projections (gate 2026-10-23: keep a noisy key from swamping the forecasts)."""
    torch = _torch()
    nn = torch.nn
    torch.manual_seed(seed)

    class SplitEmbed(nn.Module):
        def __init__(self):
            super().__init__()
            self.ef, self.ek = nn.Linear(split, width), nn.Linear(in_dim - split, width)
            self.post = nn.Sequential(nn.GELU(), nn.Linear(width, width))

        def forward(self, x):
            return self.post(self.ef(x[..., :split]) + self.ek(x[..., split:]))

    class Agg(nn.Module):
        def __init__(self):
            super().__init__()
            if split is None:
                self.embed = nn.Sequential(nn.Linear(in_dim, width), nn.GELU(), nn.Linear(width, width))
            else:
                self.embed = SplitEmbed()
            self.cls = nn.Parameter(torch.zeros(1, 1, width))
            layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout, batch_first=True,
                                               norm_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(width)
            self.beta = nn.Linear(width, Q * Q if readout == "full" else Q)
            self.head = nn.Linear(width, Q)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(self, tokens, lo):
            n, N, _ = tokens.shape
            x = torch.cat([self.cls.expand(n, 1, -1), self.embed(tokens)], 1)
            x = self.norm(self.enc(x))
            if readout == "full":
                beta = self.beta(x[:, 1:]).view(n, N, Q, Q) / Q + torch.eye(Q, device=lo.device) / N
                return torch.einsum("niqp,nip->nq", beta, lo) + self.head(x[:, 0])
            beta = self.beta(x[:, 1:]) + 1.0 / N          # starts near the mean log-odds
            return (beta * lo).sum(1) + self.head(x[:, 0])

    return Agg()


# ---------------------------------------------------------------------- learned forecasters (2026-10-22 redirect)
LEARNED_ARMS = ("fc", "fckey", "fckeyv", "wide", "bar", "fairbar")


def make_learned_system(env, arm, desc="meaning", width=128, layers=3, heads=4, key_dim=16, msg_dim=64, seed=0):
    """Forecasters are one shared network over the raw observations of the sources each read; the aggregator is a
    set transformer over their messages; trained jointly on outcomes only.

    arm  "fc"    message = the forecaster's own forecasts on all Q questions (trained by its own log loss on the
                 outcomes; the aggregator's gradient does not reach them, so they stay forecasts)
         "fckey" message = those forecasts + a learned key (key_dim, trained end to end through the aggregator)
         "wide"  message = a free-form vector (msg_dim), trained end to end only
         "bar"   no forecasters: one transformer reads every DISTINCT source read by anyone (obs + description)
                 plus one token per forecaster (its multi-hot reading set). It is handed deduplication: a learned (A).
         "fairbar" the fair full-context bar (gate 2026-10-23): one transformer over all N*k (forecaster, source)
                 tokens, duplicates included, same descriptions, no union, no multi-hot; a random per-episode slot
                 tag marks each forecaster's tokens
         "fckeyv" fckey with the readout matched to "wide" (learned value map instead of the raw forecasts)
    desc "meaning" (desc_j describes which factors source j loads on and its reliability) or "random" (control).
    Forecaster: per-source token [o_j / nuis_sd, desc_j] with an additive and a bilinear (o x desc) layer, summed
    over its k sources (DeepSets), then an MLP. Aggregator readout: out_q = sum_i beta_iq l_iq + head(CLS)_q, with
    l_i the forecasts (fc, fckey) or a learned projection of the message (wide, bar)."""
    torch = _torch()
    nn = torch.nn
    torch.manual_seed(seed)
    c = env.config
    Q = c.Q
    Mt = c.M + c.M_heldout

    class SrcEmbed(nn.Module):
        def __init__(self, out):
            super().__init__()
            self.o, self.d = nn.Linear(c.obs_dim, out), nn.Linear(c.desc_dim, out)
            self.o2, self.d2 = nn.Linear(c.obs_dim, out), nn.Linear(c.desc_dim, out)

        def forward(self, o, d):
            return self.o(o) + self.d(d) + self.o2(o) * self.d2(d)

    class Forecaster(nn.Module):
        def __init__(self, extra):
            super().__init__()
            self.src = SrcEmbed(width)
            self.phi = nn.Sequential(nn.GELU(), nn.Linear(width, width), nn.GELU(), nn.Linear(width, width))
            self.rho = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
            self.fc = nn.Linear(width, Q)
            self.extra = nn.Linear(width, extra) if extra else None

        def forward(self, o, d):
            h = self.rho(self.phi(self.src(o, d)).sum(-2))
            return self.fc(h), (self.extra(h) if self.extra is not None else None)

    class Agg(nn.Module):
        def __init__(self, in_dim, learned_values):
            super().__init__()
            self.embed = nn.Sequential(nn.Linear(in_dim, width), nn.GELU(), nn.Linear(width, width))
            self.cls = nn.Parameter(torch.zeros(1, 1, width))
            layer = nn.TransformerEncoderLayer(width, heads, 4 * width, 0.0, batch_first=True, norm_first=True,
                                               activation="gelu")
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(width)
            self.beta = nn.Linear(width, Q)
            self.val = nn.Linear(in_dim, Q) if learned_values else None
            self.head = nn.Linear(width, Q)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(self, tokens, lo=None, pad=None, emb=None):
            n, T, _ = tokens.shape
            e = self.embed(tokens) if emb is None else emb
            x = torch.cat([self.cls.expand(n, 1, -1), e], 1)
            kpm = None if pad is None else torch.cat([torch.zeros(n, 1, dtype=torch.bool, device=pad.device), pad], 1)
            x = self.norm(self.enc(x, src_key_padding_mask=kpm))
            if lo is None:
                lo = self.val(tokens)
            beta = self.beta(x[:, 1:]) + 1.0 / T
            if pad is not None:
                beta = beta.masked_fill(pad[..., None], 0.0)
            return (beta * lo).sum(1) + self.head(x[:, 0])

    class System(nn.Module):
        def __init__(self):
            super().__init__()
            self.arm, self.desc_mode = arm, desc
            if arm == "fairbar":
                self.src = SrcEmbed(width)
                self.src_mlp = nn.Sequential(nn.GELU(), nn.Linear(width, width))
                self.slot = nn.Parameter(0.5 * torch.randn(64, width))
                self.agg = Agg(width, learned_values=True)
            elif arm == "bar":
                self.src = SrcEmbed(width)
                self.src_mlp = nn.Sequential(nn.GELU(), nn.Linear(width, width))
                self.reader = nn.Linear(Mt, width)
                self.type_emb = nn.Parameter(torch.zeros(2, width))
                self.agg = Agg(width, learned_values=True)
            else:
                extra = {"fc": 0, "fckey": key_dim, "fckeyv": key_dim, "wide": msg_dim}[arm]
                self.fore = Forecaster(extra)
                in_dim = {"fc": Q, "fckey": Q + key_dim, "fckeyv": Q + key_dim, "wide": msg_dim}[arm]
                self.agg = Agg(in_dim, learned_values=(arm in ("wide", "fckeyv")))

        def descs(self, ep):
            return (env.desc if self.desc_mode == "meaning" else env.desc_random).float()

        def forward(self, ep):
            """Returns (aggregate log-odds (n, Q), local forecaster log-odds (n, N, Q) or None)."""
            D_ = self.descs(ep)
            obs = ep["obs"].float() / c.nuis_sd
            n = obs.shape[0]
            if self.arm == "bar":
                U = int(min(Mt, c.k * ep["mask"].shape[1]))
                uval, uidx = ep["union"].topk(U, dim=-1)                                    # read sources first
                pad = uval < 0.5
                o = torch.gather(obs, 1, uidx[..., None].expand(-1, -1, c.obs_dim))
                srct = self.src_mlp(self.src(o, D_[uidx])) + self.type_emb[1]
                fct = self.reader(ep["mask"].float()) + self.type_emb[0]
                emb = torch.cat([srct, fct], 1)
                pad = torch.cat([pad, torch.zeros(fct.shape[:2], dtype=torch.bool, device=pad.device)], 1)
                return self.agg(emb, None, pad=pad, emb=emb), None
            idx = ep["idx"]                                                                 # (n, N, k)
            N = idx.shape[1]
            o = torch.gather(obs[:, None].expand(-1, N, -1, -1), 2, idx[..., None].expand(-1, -1, -1, c.obs_dim))
            if self.arm == "fairbar":
                # every (forecaster, source) reading is a token, duplicates included; a random slot tag per episode
                # marks which tokens belong to the same forecaster (equivariant: slots are reshuffled each call)
                slots = self.slot[torch.argsort(torch.rand(n, self.slot.shape[0], device=obs.device), -1)[:, :N]]
                emb = self.src_mlp(self.src(o, D_[idx])) + slots[:, :, None, :]
                emb = emb.reshape(n, N * idx.shape[2], -1)
                return self.agg(emb, None, emb=emb), None
            fc, extra = self.fore(o, D_[idx])
            if self.arm == "fc":
                return self.agg(fc.detach() / 4, fc.detach()), fc
            if self.arm == "fckey":
                return self.agg(torch.cat([fc.detach() / 4, extra], -1), fc.detach()), fc
            if self.arm == "fckeyv":   # same message, readout matched to "wide" (learned value map)
                return self.agg(torch.cat([fc.detach() / 4, extra], -1)), fc
            return self.agg(extra), None

    return System()
