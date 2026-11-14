"""Learnability experiments on grouped k-parity (Z9-p2), after the learnability review of 2026-10-22
(outbox/learnability-review-2026-10-22.md, section 5, steps 1, 3, 4, 5 and bug B4/B6/B7/B8/B10/B12).

Everything here trains on FRESH episodes: a vectorized sampler of grouped Z9 (`Z9Spec`, `sample`) draws a new minibatch
every step, so nothing is memorized (B4); validation KL on fixed fresh episodes picks the best-val checkpoint; train
and dev KL are logged. The sampler reproduces aggbench's Z9-p2 tokens and Bayes exactly (tests/aggbench/
test_parity_learn.py rebuilds them from env.sample episodes), minus the constant local-logit feature (B6).

Arms (`train_arm`): an agent-local encoder over its own token -> channel -> set decoder.
  channel  "gauss"  b dims, tanh + N(0, 1/3) in training, sign at eval: exactly b bits (as compression v2)
           "float"  the encoder's 16-dim output, unbounded (b = inf; a diagnostic, not a budget)
           "none"   no encoder: the decoder reads every raw token (the uncompressed anchor)
  decoder  "transformer" (CLS, 2 layers, width 64, as compression v2), "deepsets" and "isab" (the vendored
           juho-lee DeepSet and SetTransformer, ISAB x2 + PMA, official defaults)
  dec_keys the decoder also receives each agent's own model keys['full'] OUTSIDE the channel (review step 1: makes
           "match the closed form" well posed; a learned decoder otherwise has to infer which bit a symbol is about)
Curricula: `fixed_identity_curriculum` (k = 2 -> 3 -> 4 with parity bits keeping their var ids, absent bits unread,
distractors fixed at vars 4, 5: no role flip, B7) and `biased_bit_curriculum` (P(bit = 1) = 0.8 -> 0.65 -> 0.5, step
5). Normalization uses samples from every stage (B6). The auxiliary prefix-parity loss uses the LATENT bits (tagged
"latent-label supervision") and is only used at b >= 4 (B12). Evaluation draws no global RNG (B10). For the gauss
channel, after training the encoder is frozen and the decoder fine-tuned 1000 steps on the sign messages it is scored
on (B8); both scores are reported. Symbol entropy per agent is logged (B9).

    python3 -m aggbench.parity_learn --exp step1 --out-dir runs/x      # torch; GPU
"""
from dataclasses import dataclass, field, replace
import json
import math
import sys

import numpy as np

SIGMA2 = 1.0 / 3.0
NV = 6           # var ids: parity bits 0..k-1 (k <= 4), distractors 4, 5
DIST_IDS = (4, 5)
VAL_SEED, DEV_SEED, STATS_SEED = 7_000_001, 7_000_002, 7_000_003


@dataclass
class Z9Spec:
    k: int = 4
    readers_per_bit: int = 3
    n_dist_agents: int = 2
    p: int = 2                          # readings per agent
    acc: tuple = (0.85, 0.97)
    readers_random: bool = False
    bit_prob: float = 0.5
    readers_alpha: float = 0.0          # readers_random only: > 0 draws per-episode bit weights ~ Dirichlet(alpha), so
    #                                     readers per bit are more uneven (smaller count leak, more unread bits)

    @property
    def n_readers(self):
        return self.k * self.readers_per_bit + self.n_dist_agents

    @property
    def n_agents(self):
        assert self.n_readers % self.p == 0, "readers must split into agents of p readings"
        return self.n_readers // self.p


def tokens_from(var, acc, read, p):
    """Per-agent tokens [readings (p), (one-hot var (NV), accuracy) per reading] and the key part, from per-reader
    arrays (n, R). Layout = aggbench agent_tokens for Z9-p2 without the constant local-logit feature."""
    n, R = var.shape
    N = R // p
    onehot = np.eye(NV, dtype=np.float32)[var]                                  # (n, R, NV)
    keys = np.concatenate([onehot, acc[..., None].astype(np.float32)], -1).reshape(n, N, p * (NV + 1))
    obs = read.astype(np.float32).reshape(n, N, p)
    return np.concatenate([obs, keys], -1), keys


def bayes_logit(var, acc, read, bits_prior, k):
    """Exact log-odds of Y = XOR of bits 0..k-1 given every reading: per bit L_j = logit(q) + sum of reading LLRs,
    P(Y = 1) = (1 - prod_j (1 - 2 sigma(L_j))) / 2, and 1 - 2 sigma(L) = -tanh(L / 2)."""
    q = bits_prior
    llr = np.where(read == 1, 1.0, -1.0) * np.log(acc / (1 - acc))
    t = np.ones(var.shape[0])
    for j in range(k):
        L = math.log(q / (1 - q)) + (llr * (var == j)).sum(1)
        t = t * -np.tanh(L / 2)
    t = np.clip(t, -1 + 1e-15, 1 - 1e-15)
    return np.log1p(-t) - np.log1p(t)


def sample(spec, n, rng):
    """n fresh episodes: dict tokens, keys, y, bayes, aux (prefix parities, 3 slots), aux_mask."""
    R = spec.n_readers
    if spec.readers_random and spec.readers_alpha > 0:
        w = rng.dirichlet(np.full(spec.k, spec.readers_alpha), n)                  # (n, k)
        u = rng.random((n, spec.k * spec.readers_per_bit, 1))
        par = (u > np.cumsum(w, 1)[:, None, :]).sum(-1).clip(0, spec.k - 1)
    elif spec.readers_random:
        par = rng.integers(0, spec.k, (n, spec.k * spec.readers_per_bit))
    else:
        par = np.broadcast_to(np.repeat(np.arange(spec.k), spec.readers_per_bit), (n, spec.k * spec.readers_per_bit))
    dist = np.asarray(DIST_IDS)[rng.integers(0, len(DIST_IDS), (n, spec.n_dist_agents))]
    var = np.concatenate([par, dist], 1)
    var = np.take_along_axis(var, np.argsort(rng.random((n, R)), 1), 1)        # random positions
    acc = rng.uniform(*spec.acc, (n, R))
    bits = (rng.random((n, NV)) < spec.bit_prob).astype(np.int64)
    truth = np.take_along_axis(bits, var, 1)
    read = np.where(rng.random((n, R)) < acc, truth, 1 - truth)
    y = bits[:, :spec.k].sum(1) % 2
    tok, keys = tokens_from(var, acc, read, spec.p)
    aux = np.zeros((n, 3), np.float32)
    pref = np.cumsum(bits[:, :spec.k], 1) % 2
    aux[:, :spec.k - 1] = pref[:, :spec.k - 1]
    mask = np.zeros(3, np.float32)
    mask[:spec.k - 1] = 1
    return {"tokens": tok, "keys": keys, "y": y.astype(np.float32), "aux": aux, "aux_mask": mask,
            "bayes": bayes_logit(var, acc, read, spec.bit_prob, spec.k), "prior_logit": prior_logit(spec)}


def prior_logit(spec):
    q, k = spec.bit_prob, spec.k
    p1 = (1 - (1 - 2 * q) ** k) / 2
    return float(math.log(p1 / (1 - p1)))


def fixed_identity_curriculum(final=None, steps=(2000, 2000, 4000)):
    """k = 2 -> 3 -> 4: parity bits keep var ids 0..k-1, bits >= k are unread, distractors stay vars 4, 5 (B7)."""
    final = final or Z9Spec()
    nd3 = 1 if (3 * final.readers_per_bit) % 2 else 2                 # keep the reader count divisible by p = 2
    return [(replace(final, k=2, n_dist_agents=2), steps[0]), (replace(final, k=3, n_dist_agents=nd3), steps[1]),
            (final, steps[2])]


def biased_bit_curriculum(final=None, steps=(2000, 2000, 4000), probs=(0.8, 0.65)):
    final = final or Z9Spec()
    return [(replace(final, bit_prob=probs[0]), steps[0]), (replace(final, bit_prob=probs[1]), steps[1]),
            (final, steps[2])]


# ----------------------------------------------------------------------------- models

def build(F, Fk, b, channel, decoder, dec_keys, n_aux=3, width=64, layers=2, heads=4, float_dim=16, in_norm=False,
          key_scale=1.0, power_norm=False, p=2, id_bits=0, queried=False, fsq=False):
    """in_norm (2026-10-22 key fix): the message and the keys get separate learned projections, each followed by a
    LayerNorm, summed. Without it the decoder input is Linear(concat(message, normalized keys)): at initialization
    a float message has rms 0.08 against unit-scale keys, so the keys swamp it (energy ratio ~13:1)."""
    import torch
    from torch import nn
    from .external.vendor.set_transformer_models import DeepSet, SetTransformer
    D = {"gauss": b, "float": float_dim, "none": F}[channel]
    if fsq:     # FSQ channel (Mentzer et al. 2309.15505, gemini's 1-D 4-level variant): b / 2 dims x 4 levels = b bits
        assert channel == "gauss" and b % 2 == 0, "fsq replaces the Gaussian channel and needs an even b"
        D = b // 2
    din = D + (Fk if dec_keys else 0)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("x_mean", torch.zeros(F))
            self.register_buffer("x_sd", torch.ones(F))
            self.register_buffer("k_mean", torch.zeros(Fk))
            self.register_buffer("k_sd", torch.ones(Fk))
            self.enc = (nn.Sequential(nn.Linear(F, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, D))
                        if channel != "none" else None)
            self.own_head = nn.Sequential(nn.Linear(D, 64), nn.GELU(), nn.Linear(64, 1))
            # QUERIED encoder (2026-10-22, Kelvin: "it is essential for the consensus to query the agent or vice-versa
            # before the compression happens"): agents speak in index order (positions are random per episode); agent
            # t's encoder reads its own token and, by attention, the consensus = every earlier agent's b-bit message
            # as received (unbounded, read-only), then writes its own b-bit message
            self.queried = queried and channel == "gauss"
            if self.queried:
                self.q_trunk = nn.Sequential(nn.Linear(F, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU())
                self.q_q = nn.Linear(128, 64)
                self.q_kv = nn.Linear(D, 64)
                self.q_attn = nn.MultiheadAttention(64, 4, batch_first=True)
                self.q_head = nn.Sequential(nn.Linear(128 + 64, 128), nn.GELU(), nn.Linear(128, D))
            # fixed-power channel (2026-10-22): the pre-channel message is standardized per dimension over the batch
            # (BatchNorm, no affine), so E[s^2] = 1 = the power constraint, enforced instead of "<= 1"
            self.pnorm = nn.BatchNorm1d(D, affine=False) if (power_norm and channel == "gauss") else None
            # masked-agent pretraining head: a query token from the held-out agent's own key, readings predicted
            self.qproj = nn.Linear(Fk, width)
            self.pre_out = nn.Linear(width, p)
            # blind self-reconstruction head (pretrain_mode "recon"): each agent's readings and the var id of each
            # reading, predicted at its own slot from all agents' messages; no key reaches the decoder
            self.rec_out = nn.Linear(width, p * (1 + NV))
            # factorised message (id_bits > 0): message dims [:id_bits] = identity channel, [id_bits:] = value
            # channel; the identity part is pretrained by an AGENT-LOCAL head to reconstruct the agent's own var ids
            self.id_bits = id_bits
            self.id_head = (nn.Sequential(nn.Linear(id_bits, 64), nn.GELU(), nn.Linear(64, p * NV)) if id_bits
                            else None)
            if in_norm:
                self.p_m = nn.Sequential(nn.Linear(D, width), nn.LayerNorm(width))
                self.p_k = nn.Sequential(nn.Linear(Fk, width), nn.LayerNorm(width)) if dec_keys else None
            if decoder == "transformer":
                self.inp = nn.Linear(width if in_norm else din, width)
                self.cls = nn.Parameter(0.02 * torch.randn(1, 1, width))
                layer = nn.TransformerEncoderLayer(width, heads, 4 * width, 0.0, batch_first=True, norm_first=True)
                self.dec = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
                self.norm = nn.LayerNorm(width)
                self.out = nn.Linear(width, 1 + n_aux)
            elif decoder == "deepsets":
                self.dec = DeepSet(width if in_norm else din, 1, 1 + n_aux)
            elif decoder == "isab":
                self.dec = SetTransformer(width if in_norm else din, 1, 1 + n_aux)
            else:
                raise ValueError(decoder)

        def _fsq(self, u):
            """4 levels per dim, straight-through rounding (deterministic, no noise); returns (levels, message) and
            stores the usage term (0.05 x (log 4 - entropy of the batch-mean soft level assignment), per dim)."""
            zt = 1.5 * (1.0 + torch.tanh(u / 1.5))
            zh = torch.round(zt).clamp(0.0, 3.0)
            m = (zt + (zh - zt).detach() - 1.5) / 1.5
            centers = torch.arange(4.0, device=u.device, dtype=u.dtype)
            ps = torch.softmax(-2.0 * (zt[..., None] - centers) ** 2, -1)
            pbar = ps.reshape(-1, u.shape[-1], 4).mean(0).clamp_min(1e-8)
            self.last_usage = 0.05 * (math.log(4.0) + (pbar * pbar.log()).sum(-1)).mean()
            return zh, m

        def _channel(self, u, mode, gen=None):
            if fsq:
                zh, m = self._fsq(u)
                return zh - 1.5, m
            if self.pnorm is not None:
                sh = u.shape
                s = self.pnorm(u.reshape(-1, sh[-1])).reshape(sh)
            else:
                s = torch.tanh(u)
            if mode == "quant":
                return s, torch.where(s >= 0, 1.0, -1.0)
            noise = torch.randn(s.shape, generator=gen, device=s.device) if gen is not None else torch.randn_like(s)
            return s, s + math.sqrt(self.sigma2) * noise

        def _queried(self, z, mode, gen=None):
            h = self.q_trunk(z)                                   # (B, N, 128), each agent's own token only
            B, N = h.shape[:2]
            outs, pre, usage = [], [], []
            for t in range(N):
                if t == 0:
                    ctx = torch.zeros(B, 64, device=h.device, dtype=h.dtype)
                else:
                    kv = self.q_kv(torch.stack(outs, 1))          # the consensus as received: earlier messages only
                    ctx = self.q_attn(self.q_q(h[:, t:t + 1]), kv, kv, need_weights=False)[0][:, 0]
                u_t = self.q_head(torch.cat([h[:, t], ctx], -1))
                s_t, m_t = self._channel(u_t, mode, gen)
                pre.append(s_t)
                outs.append(m_t)
                if fsq:
                    usage.append(self.last_usage)
            self.last_s = torch.stack(pre, 1)
            if fsq:     # gate on batch 9: the usage term is the mean over ALL agents (it used to be the last agent's)
                self.last_usage = torch.stack(usage).mean()
            return torch.stack(outs, 1)

        def message(self, x, mode, gen=None):
            z = (x - self.x_mean) / self.x_sd
            if channel == "none":
                return z
            if self.queried:
                m = self._queried(z, mode, gen)
                self.last_power = float((self.last_s.detach() ** 2).mean())
                return m
            u = self.enc(z)
            if channel == "float":
                return u
            if fsq:
                zh, m = self._fsq(u)
                self.last_levels = zh
                self.last_power = float((m.detach() ** 2).mean())
                return m
            if self.pnorm is not None:
                sh = u.shape
                s = self.pnorm(u.reshape(-1, sh[-1])).reshape(sh)
            else:
                s = torch.tanh(u)
            self.last_power = float((s.detach() ** 2).mean())
            if mode == "quant":
                return torch.where(s >= 0, 1.0, -1.0)
            noise = torch.randn(s.shape, generator=gen, device=s.device) if gen is not None else torch.randn_like(s)
            return s + math.sqrt(self.sigma2) * noise

        def symbols(self, x):
            if fsq:
                self.message(x, "quant")
                lv = (self.last_s + 1.5) if self.queried else self.last_levels
                lv = torch.round(lv).long()
                return (lv * (4 ** torch.arange(lv.shape[-1], device=lv.device))).sum(-1)
            if self.queried:
                self.message(x, "quant")
                u = self.last_s
                return ((u >= 0).long() * (2 ** torch.arange(u.shape[-1], device=u.device))).sum(-1)
            u = self.enc((x - self.x_mean) / self.x_sd)
            if self.pnorm is not None:
                sh = u.shape
                u = self.pnorm(u.reshape(-1, sh[-1])).reshape(sh)
            return ((u >= 0).long() * (2 ** torch.arange(u.shape[-1], device=u.device))).sum(-1)

        use_keys = True                       # set False during a key warm-up (keys zeroed out of the decoder)
        sigma2 = SIGMA2                       # training channel noise; a schedule may lower it early on

        def decode(self, m, keys):
            if in_norm:
                h = self.p_m(m)
                if dec_keys and self.use_keys:
                    h = h + self.p_k(key_scale * (keys - self.k_mean) / self.k_sd)
            else:
                kk = key_scale * (keys - self.k_mean) / self.k_sd
                h = m if not dec_keys else torch.cat([m, kk if self.use_keys else torch.zeros_like(kk)], -1)
            if decoder == "transformer":
                h = self.inp(h)
                h = torch.cat([self.cls.expand(len(h), -1, -1), h], 1)
                o = self.out(self.norm(self.dec(h)[:, 0]))
            else:
                o = self.dec(h)[:, 0]
            return o[:, 0], o[:, 1:]

        def decode_masked(self, m, keys, idx):
            """Masked-agent prediction (no Y): agent idx's message is replaced by a query built from its own key;
            the decoder predicts its p readings from the OTHER agents' messages (transformer decoder only)."""
            h = self._embed(m, keys)
            q = self.qproj((keys - self.k_mean) / self.k_sd)
            sel = torch.nn.functional.one_hot(idx, h.shape[1]).to(h.dtype)[..., None]
            h = h * (1 - sel) + q * sel
            h = torch.cat([self.cls.expand(len(h), -1, -1), h], 1)
            z = self.norm(self.dec(h)[:, 1:])
            return self.pre_out((z * sel).sum(1))

        def _embed(self, m, keys):
            """Decoder input tokens (transformer path), honouring the key warm-up (use_keys)."""
            if in_norm:
                h = self.p_m(m)
                if dec_keys and self.use_keys:
                    h = h + self.p_k(key_scale * (keys - self.k_mean) / self.k_sd)
                return self.inp(h)
            kk = key_scale * (keys - self.k_mean) / self.k_sd
            return self.inp(m if not dec_keys else torch.cat([m, kk if self.use_keys else torch.zeros_like(kk)], -1))

        def decode_factor_masked(self, m, keys, idx):
            """Blind masked prediction for the factorised message: agent idx keeps its IDENTITY bits (they are part of
            its message) and its value bits are zeroed; the decoder predicts its p readings at its slot."""
            sel = torch.nn.functional.one_hot(idx, m.shape[1]).to(m.dtype)[..., None]
            vmask = torch.ones_like(m)
            vmask[..., self.id_bits:] = 0
            m2 = m * (1 - sel) + m * vmask * sel
            h = self._embed(m2, keys)
            h = torch.cat([self.cls.expand(len(h), -1, -1), h], 1)
            z = self.norm(self.dec(h)[:, 1:])
            return self.pre_out((z * sel).sum(1))

        def decode_recon(self, m, keys):
            """Blind self-reconstruction: per-agent logits for its p readings and p var ids (NV classes each)."""
            h = self._embed(m, keys)
            h = torch.cat([self.cls.expand(len(h), -1, -1), h], 1)
            z = self.norm(self.dec(h)[:, 1:])
            return self.rec_out(z)

    return Net()


def _kl(q, b):
    from .metrics import logsig
    pb = 1 / (1 + np.exp(-b))
    return pb * (logsig(b) - logsig(q)) + (1 - pb) * (logsig(-b) - logsig(-q))


def _entropy_bits(sym):
    _, c = np.unique(sym, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def train_arm(name, stages, b=None, channel="gauss", decoder="transformer", dec_keys=False, arm="aware", seed=0,
              in_norm=False, key_warmup=0, tag=None, key_scale=1.0, noise_sched=None, power_norm=False,
              pretrain=0, pretrain_mode="masked", id_bits=0, pretrain_scale=None, queried=False, fsq=False,
              aux=0.0, aux_warmup=4000, batch=512, lr=1e-3, eval_every=200, n_val=2000, n_dev=4000, finetune=1000,
              device=None, n_boot=2000):
    """One arm on fresh episodes. stages: [(Z9Spec, steps)]; the last spec is the task scored. arm "aware": end to
    end on Y (+ aux * latent prefix parities, annealed to 0 by aux_warmup, only if b >= 4 or b is None). arm "self":
    encoder on own-evidence Y prediction, frozen, then the decoder (each for the total steps). Returns a record."""
    import copy
    import torch
    from .compression import Channel
    from .metrics import summarize_values
    torch.manual_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    final = stages[-1][0]
    val = sample(final, n_val, np.random.default_rng(VAL_SEED))
    dev = sample(final, n_dev, np.random.default_rng(DEV_SEED))
    st = [sample(s, 4096, np.random.default_rng([STATS_SEED, i])) for i, (s, _) in enumerate(stages)]   # all stages (B6)
    pad = lambda a, N: np.concatenate([a, np.zeros((len(a), N - a.shape[1], a.shape[2]), a.dtype)], 1)
    T = np.concatenate([x["tokens"].reshape(-1, x["tokens"].shape[-1]) for x in st])
    K = np.concatenate([x["keys"].reshape(-1, x["keys"].shape[-1]) for x in st])
    F, Fk = T.shape[1], K.shape[1]
    use_aux = aux > 0 and (b is None or b >= 4)
    net = build(F, Fk, b, channel, decoder, dec_keys, in_norm=in_norm, key_scale=key_scale,
                power_norm=power_norm, p=final.p, id_bits=id_bits, queried=queried, fsq=fsq).to(device)
    with torch.no_grad():
        net.x_mean.copy_(torch.tensor(T.mean(0))), net.x_sd.copy_(torch.tensor(T.std(0)).clamp_min(1e-2))
        net.k_mean.copy_(torch.tensor(K.mean(0))), net.k_sd.copy_(torch.tensor(K.std(0)).clamp_min(1e-2))
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    tt = lambda a: torch.tensor(a, device=device)
    rng = np.random.default_rng([seed, 99])
    eval_gen = torch.Generator(device=device).manual_seed(12345)
    rec = {"name": name, "b": b, "channel": channel, "decoder": decoder, "dec_keys": dec_keys, "arm": arm,
           "in_norm": in_norm, "key_warmup": key_warmup, "curriculum_tag": tag, "key_scale": key_scale,
           "noise_sched": noise_sched, "power_norm": power_norm, "pretrain": pretrain,
           "pretrain_mode": pretrain_mode if pretrain else None,
           "pretrain_query": ("held-out agent's own key (training only)" if pretrain and pretrain_mode == "masked"
                              else ({"recon": "none: blind self-reconstruction",
                                     "factor": "none: blind (id bits local key reconstruction; value bits masked "
                                               "prediction queried by the agent's own id bits)"}[pretrain_mode]
                                    if pretrain else None)),
           "id_bits": id_bits, "queried": queried, "fsq": fsq,
           "channel_detail": "FSQ, b/2 dims x 4 levels, usage term 0.05" if fsq else None,
           "query": "unbounded read-only (earlier agents' b-bit messages)" if queried else "none (one-shot)",
           "seed": seed, "aux": aux if use_aux else 0.0, "stages": [(s.__dict__, n) for s, n in stages],
           "supervision": "latent-label supervision" if use_aux else "Y labels only", "curve": []}

    def logits(d, mode):
        net.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(d["y"]), 2048):
                x, k = tt(d["tokens"][s:s + 2048]), tt(d["keys"][s:s + 2048])
                out.append(net.decode(net.message(x, mode, eval_gen), k)[0].cpu().numpy())
        net.train()
        return d["prior_logit"] + np.concatenate(out).astype(float)

    eval_mode = "quant" if channel == "gauss" else "train"

    def fit(params, loss_fn, phase, total_stages, select=True):
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        total = sum(n for _, n in total_stages)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total)
        best = (math.inf, None, -1)
        t = 0
        for spec, n_steps in total_stages:
            l0 = prior_logit(spec)
            for _ in range(n_steps):
                d = sample(spec, batch, rng)
                x, k, y = tt(d["tokens"]), tt(d["keys"]), tt(d["y"])
                net.use_keys = t >= key_warmup or phase == "finetune_sign"
                if noise_sched and phase != "finetune_sign":   # (sigma2_start, steps): geometric anneal to SIGMA2
                    s0, n_s = noise_sched
                    net.sigma2 = SIGMA2 if t >= n_s else s0 * (SIGMA2 / s0) ** (t / n_s)
                else:
                    net.sigma2 = SIGMA2
                loss, lg = loss_fn(x, k, y, d, t, l0)
                opt.zero_grad()
                loss.backward()
                diag = t % eval_every == 0 or t == total - 1
                if diag and net.enc is not None:
                    enc_mods = ([net.q_trunk, net.q_q, net.q_kv, net.q_attn, net.q_head] if getattr(net, "queried", False)
                                else [net.enc])          # the queried encoder lives in q_*; net.enc is unused there
                    gn = [p_.grad.norm() ** 2 for mod in enc_mods for p_ in mod.parameters() if p_.grad is not None]
                    enc_grad = float(sum(gn) ** 0.5) if gn else 0.0
                    with torch.no_grad():
                        msg_sd = float(net.message(x[:256], "train").std())
                        msg_power = getattr(net, "last_power", None)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                if t % eval_every == 0 or t == total - 1:
                    v = float(_kl(logits(val, eval_mode), val["bayes"]).mean())
                    tr = float(_kl(lg.detach().cpu().numpy().astype(float), d["bayes"]).mean()) if lg is not None \
                        else None
                    rec["curve"].append({"phase": phase, "step": t, "val_kl": v, "train_kl": tr, "sigma2": net.sigma2,
                                         "k": spec.k, "bit_prob": spec.bit_prob, "keys_on": net.use_keys,
                                         "enc_grad": enc_grad if net.enc is not None else None,
                                         "msg_sd": msg_sd if net.enc is not None else None,
                                         "msg_power": msg_power if net.enc is not None else None,
                                         "pre_loss": loss.item() if phase == "pretrain" else None})
                    if select and spec == final and v < best[0] and net.use_keys and net.sigma2 == SIGMA2:
                        best = (v, copy.deepcopy(net.state_dict()), t)
                t += 1
        if best[1] is not None:
            net.load_state_dict(best[1])
        return best

    def aware_loss(x, k, y, d, t, l0):
        lg, za = net.decode(net.message(x, "train"), k)
        lg = lg + l0
        loss = bce(lg, y) + (net.last_usage if fsq else 0.0)
        if use_aux and t < aux_warmup:
            m = tt(d["aux_mask"])
            loss = loss + aux * (1 - t / aux_warmup) * (bce(za, tt(d["aux"]), reduction="none") * m).sum(-1).mean() \
                / m.sum()
        return loss, lg

    if pretrain:            # masked-agent prediction, no Y: encoder + decoder learn a code from the overlap signal
        P = final.p

        def pre_loss(x, k, y, d, t, l0):
            m = net.message(x, "train")
            if pretrain_mode == "factor":     # blind factorised: local id reconstruction + masked value prediction
                n, N = x.shape[:2]
                vars_ = k.reshape(n, N, P, NV + 1)[..., :NV].argmax(-1)
                lid = torch.nn.functional.cross_entropy(net.id_head(m[..., :id_bits]).reshape(-1, NV),
                                                        vars_.reshape(-1))
                idx = torch.randint(0, N, (n,), device=x.device)
                lval = bce(net.decode_factor_masked(m, k, idx), x[torch.arange(n, device=x.device), idx][:, :P])
                return lid + lval, None
            if pretrain_mode == "recon":      # blind: every agent's readings and var ids from all agents' messages
                o = net.decode_recon(m, k)
                n, N = x.shape[:2]
                lr_ = bce(o[..., :P], x[..., :P])
                vars_ = k.reshape(n, N, P, NV + 1)[..., :NV].argmax(-1)
                lv = torch.nn.functional.cross_entropy(o[..., P:].reshape(-1, NV), vars_.reshape(-1))
                return lr_ + lv + (net.last_usage if fsq else 0.0), None
            idx = torch.randint(0, x.shape[1], (x.shape[0],), device=x.device)
            logits_p = net.decode_masked(m, k, idx)
            target = x[torch.arange(len(x), device=x.device), idx][:, :P]
            return bce(logits_p, target), None
        fit(list(net.parameters()), pre_loss, "pretrain", [(final, pretrain)], select=False)
    if arm == "aware":
        best = fit(list(net.parameters()), aware_loss, "aware", stages)
    else:
        enc = list(net.enc.parameters()) + list(net.own_head.parameters())
        if arm == "self":       # "random_enc": the encoder stays at its random init (non-aware control)
            fit(enc, lambda x, k, y, d, t, l0: (bce(net.own_head(net.message(x, "train"))[..., 0] + l0,
                                                   y[:, None].expand(-1, x.shape[1])), None), "self_stage1", stages)
        for p_ in enc:
            p_.requires_grad_(False)
        dec = [p_ for n_, p_ in net.named_parameters() if not (n_.startswith("enc") or n_.startswith("own_head"))]

        def dec_loss(x, k, y, d, t, l0):
            with torch.no_grad():
                m = net.message(x, "train")
            lg = net.decode(m, k)[0] + l0
            return bce(lg, y), lg
        best = fit(dec, dec_loss, "self_stage2", stages)
    net.use_keys = True
    rec["best_val_kl"], rec["best_step"] = best[0], best[2]
    fq = logits(dev, eval_mode)
    kl = _kl(fq, dev["bayes"])
    rec["regret"] = summarize_values(kl, n_boot)
    rec["prior_regret"] = float(_kl(np.full(len(kl), dev["prior_logit"]), dev["bayes"]).mean())
    if channel == "gauss":
        net.eval()                                # running BatchNorm stats, no update from dev data
        with torch.no_grad():
            sym = net.symbols(tt(dev["tokens"])).cpu().numpy().astype(np.int64)
        net.train()
        Channel(b).send(sym)
        rec["symbol_entropy_bits"] = _entropy_bits(sym.ravel())
        rec["distinct_symbols"] = int(len(np.unique(sym)))
        rec["regret_noisy"] = summarize_values(_kl(logits(dev, "train"), dev["bayes"]), n_boot)
        if finetune:     # B8: encoder frozen, decoder fine-tuned on the sign messages it is scored on
            for p_ in net.parameters():
                p_.requires_grad_(False)
            dec = [p_ for n_, p_ in net.named_parameters() if not (n_.startswith("enc") or n_.startswith("own_head"))]
            for p_ in dec:
                p_.requires_grad_(True)

            def ft_loss(x, k, y, d, t, l0):
                with torch.no_grad():
                    m = net.message(x, "quant")
                lg = net.decode(m, k)[0] + l0
                return bce(lg, y), lg
            fit(dec, ft_loss, "finetune_sign", [(final, finetune)])
            rec["regret_after_sign_finetune"] = summarize_values(_kl(logits(dev, "quant"), dev["bayes"]), n_boot)
    print(f"[{name}] b={b} {channel}/{decoder} keys={dec_keys} {arm} seed={seed}: dev KL {rec['regret']['mean']:.4f} "
          f"(best val {best[0]:.4f} @ {best[2]})" + (f" H={rec['symbol_entropy_bits']:.2f}b" if channel == "gauss"
                                                     else ""), file=sys.stderr, flush=True)
    return rec


# ----------------------------------------------------------------------------- experiments

def experiments(exp, steps=8000):
    S = Z9Spec()
    one = [(S, steps)]
    rr = Z9Spec(readers_random=True)
    E = []
    if exp in ("step1", "all"):
        for keys in (True, False):
            for b, ch in ((2, "gauss"), (4, "gauss"), (None, "float")):
                E.append(dict(name=f"step1_keys{int(keys)}", stages=one, b=b, channel=ch, dec_keys=keys))
    if exp in ("anchor", "step1", "all"):
        E.append(dict(name="anchor_fresh", stages=one, channel="none"))
    if exp in ("step3", "all"):
        cur = fixed_identity_curriculum(S, (steps // 4, steps // 4, steps // 2))
        for b in (2, 4):
            E.append(dict(name="step3_curr_labelonly", stages=cur, b=b, channel="gauss", dec_keys=True))
        E.append(dict(name="step3_curr_aux", stages=cur, b=4, channel="gauss", dec_keys=True, aux=1.0,
                      aux_warmup=steps // 2))
    if exp in ("step4", "all"):
        for spec, tag in ((rr, "rr"), (S, "fixed")):
            for dec in ("deepsets", "isab"):
                E.append(dict(name=f"step4_{tag}_{dec}", stages=[(spec, steps)], channel="none", decoder=dec))
        E.append(dict(name="step4_rr_anchor", stages=[(rr, steps)], channel="none"))
        E.append(dict(name="step4_rr_dvib_inf", stages=[(rr, steps)], b=None, channel="float"))
    lf = Z9Spec(readers_random=True, readers_alpha=1.0, readers_per_bit=6)    # leak-reduced primary (13 agents)
    if exp == "batch10_queried_b8_fix_4x":   # gate on batch 9: queried b = 8 with the per-agent usage fix, 4x steps
        E.append(dict(name="b10_queried_blind_recon_fsq_4x_usagefix", stages=[(lf, 4 * steps)], b=8, channel="gauss",
                      fsq=True, pretrain=4 * 2000, pretrain_mode="recon", queried=True))
    if exp in ("batch10_oneshot_b8", "batch10_queried_b6"):   # 8x steps: 16000 pretraining + 64000 Y steps
        b = 8 if exp.endswith("b8") else 6
        q = "queried" in exp
        E.append(dict(name=f"b10_{'queried' if q else 'oneshot'}_blind_recon_fsq_8x", stages=[(lf, 8 * steps)], b=b,
                      channel="gauss", fsq=True, pretrain=8 * 2000, pretrain_mode="recon", queried=q))
    if exp in ("batch9_oneshot", "batch9_queried"):    # blind self-reconstruction + FSQ channel, 4x steps
        for b in (6, 8):
            E.append(dict(name=f"b9_{exp.split('_')[1]}_blind_recon_fsq_4x", stages=[(lf, 4 * steps)], b=b,
                          channel="gauss", fsq=True, pretrain=4 * 2000, pretrain_mode="recon",
                          queried=exp == "batch9_queried"))
    if exp in ("batch8_query_blind", "batch8_query_known"):     # queried vs one-shot at equal b, same harness
        for b in (2, 4, 6):
            for q in (True, False):
                if exp == "batch8_query_blind":
                    E.append(dict(name=f"b8_{'queried' if q else 'oneshot'}_blind_recon_power", stages=[(lf, steps)],
                                  b=b, channel="gauss", pretrain=2000, pretrain_mode="recon", power_norm=True,
                                  queried=q))
                else:
                    E.append(dict(name=f"b8_{'queried' if q else 'oneshot'}_known_masked_power", stages=[(lf, steps)],
                                  b=b, channel="gauss", pretrain=2000, power_norm=True, dec_keys=True, key_scale=0.1,
                                  key_warmup=1000, queried=q))
    if exp in ("batch8_scale", "batch8_scale_anchor", "batch8_factor", "batch8_anchor_long"):
        if exp == "batch8_scale":     # 4x steps: fully blind self-reconstruction + fixed power, b = 6, 8
            for b in (6, 8):
                E.append(dict(name="b8_recon_power_4x", stages=[(lf, 4 * steps)], b=b, channel="gauss",
                              pretrain=4 * 2000, pretrain_mode="recon", power_norm=True))
        if exp == "batch8_scale_anchor":
            E.append(dict(name="b8_anchor_transformer_4x", stages=[(lf, 4 * (steps + 2000))], channel="none"))
            E.append(dict(name="b8_anchor_isab_4x", stages=[(lf, 4 * (steps + 2000))], channel="none",
                          decoder="isab"))
        if exp == "batch8_anchor_long":   # anchors were still improving at their last step: train 2x longer than 4x
            E.append(dict(name="b8_anchor_transformer_8x", stages=[(lf, 8 * (steps + 2000))], channel="none"))
        if exp == "batch8_factor":    # factorised message: id bits + value bits, blind, fixed power
            for b, ib in ((6, 4), (8, 6)):
                E.append(dict(name="b8_factor_power", stages=[(lf, steps)], b=b, channel="gauss", pretrain=2000,
                              pretrain_mode="factor", id_bits=ib, power_norm=True))
                E.append(dict(name="b8_factor", stages=[(lf, steps)], b=b, channel="gauss", pretrain=2000,
                              pretrain_mode="factor", id_bits=ib))
    if exp in ("batch7_ctl", "batch7_anchor", "batch7_blind"):   # gate on batch 6
        if exp == "batch7_ctl":       # same-harness plain controls: tanh, no pretraining, concat keys
            for keys in (False, True):
                kw = dict(dec_keys=True, key_scale=0.1, key_warmup=1000) if keys else {}
                for n_steps in (steps, steps + 2000):
                    for b in (2, 4, 6):
                        E.append(dict(name=f"b7_plain_k{int(keys)}_{n_steps}", stages=[(lf, n_steps)], b=b,
                                      channel="gauss", **kw))
        if exp == "batch7_anchor":    # step-matched anchors: 10000 (= 2000 + 8000) and 11000 (+ fine-tune) steps
            for n_steps in (steps + 2000, steps + 3000):
                E.append(dict(name=f"b7_anchor_isab_{n_steps}", stages=[(lf, n_steps)], channel="none",
                              decoder="isab"))
                E.append(dict(name=f"b7_anchor_transformer_{n_steps}", stages=[(lf, n_steps)], channel="none"))
            E.append(dict(name="b7_anchor_transformer_pretrain", stages=[(lf, steps)], channel="none", pretrain=2000))
        if exp == "batch7_blind":     # fully blind pretraining: self-reconstruction, no key outside the encoder
            for b in (2, 4, 6):
                E.append(dict(name="b7_recon_k0", stages=[(lf, steps)], b=b, channel="gauss", pretrain=2000,
                              pretrain_mode="recon"))
                E.append(dict(name="b7_recon_power_k0", stages=[(lf, steps)], b=b, channel="gauss", pretrain=2000,
                              pretrain_mode="recon", power_norm=True))
    if exp in ("batch6_k0", "batch6_k1"):   # fixed-power channel and masked-agent pretraining, Y labels only
        keys = exp.endswith("1")
        kw = dict(dec_keys=True, key_scale=0.1, key_warmup=1000) if keys else {}
        for b in (2, 4, 6):
            E.append(dict(name=f"b6_power_k{int(keys)}", stages=[(lf, steps)], b=b, channel="gauss",
                          power_norm=True, **kw))
            E.append(dict(name=f"b6_pretrain_k{int(keys)}", stages=[(lf, steps)], b=b, channel="gauss",
                          pretrain=2000, **kw))
            E.append(dict(name=f"b6_both_k{int(keys)}", stages=[(lf, steps)], b=b, channel="gauss",
                          power_norm=True, pretrain=2000, **kw))
    if exp == "batch5":      # gate on b481850 (part B): matched key warm-up, both biased arms, same-data anchors
        lfc = fixed_identity_curriculum(lf, (steps // 4, steps // 4, steps // 2))
        lfb = biased_bit_curriculum(lf, (steps // 4, steps // 4, steps // 2))
        for b, ch in ((2, "gauss"), (4, "gauss"), (None, "float")):     # x0.1 keys with the same warm-up as in_norm
            E.append(dict(name="b5_step1_ks01_w1000", stages=one, b=b, channel=ch, dec_keys=True, key_scale=0.1,
                          key_warmup=1000))
        for b in (2, 4, 6):
            E.append(dict(name="b5_curr_ks01_w1000", stages=lfc, b=b, channel="gauss", dec_keys=True, key_scale=0.1,
                          key_warmup=1000, tag="structure-informed"))
        for b in (2, 4, 6, 8):
            E.append(dict(name="b5_bias_keys0", stages=lfb, b=b, channel="gauss", tag="biased-bit curriculum"))
        E.append(dict(name="b5_bias_ks01_w1000", stages=lfb, b=4, channel="gauss", dec_keys=True, key_scale=0.1,
                      key_warmup=1000, tag="biased-bit curriculum"))
        E.append(dict(name="b5_lf_deepsets", stages=[(lf, steps)], channel="none", decoder="deepsets"))
        E.append(dict(name="b5_lf_isab", stages=[(lf, steps)], channel="none", decoder="isab"))
        E.append(dict(name="b5_lf_transformer", stages=[(lf, steps)], channel="none"))
        E.append(dict(name="b5_lf_isab_curr", stages=lfc, channel="none", decoder="isab", tag="structure-informed"))
        E.append(dict(name="b5_lf_isab_bias", stages=lfb, channel="none", decoder="isab", tag="biased-bit curriculum"))
    if exp == "batch4":      # 2026-10-22: x0.1 keys on the leak-reduced spec; biased-bit curve; noise schedule
        lfc = fixed_identity_curriculum(lf, (steps // 4, steps // 4, steps // 2))
        lfb = biased_bit_curriculum(lf, (steps // 4, steps // 4, steps // 2))
        for b in (2, 4, 6):
            E.append(dict(name="b4_curr_ks01", stages=lfc, b=b, channel="gauss", dec_keys=True, key_scale=0.1,
                          tag="structure-informed"))
        for b in (2, 4, 6, 8):
            E.append(dict(name="b4_bias_ks01", stages=lfb, b=b, channel="gauss", dec_keys=True, key_scale=0.1,
                          tag="biased-bit curriculum"))
        E.append(dict(name="b4_anchor_bias", stages=lfb, channel="none", tag="biased-bit curriculum"))
        for b in (2, 4, 6):
            E.append(dict(name="b4_noise_sched", stages=[(lf, steps)], b=b, channel="gauss",
                          noise_sched=(0.01, steps // 2)))
    if exp in ("lf_main_keys0", "lf_main_keys1", "lf_anchor"):
        if exp == "lf_anchor":
            E.append(dict(name="lf_anchor", stages=[(lf, 2 * steps)], channel="none"))
            E.append(dict(name="lf_isab", stages=[(lf, 2 * steps)], channel="none", decoder="isab"))
            lfc = fixed_identity_curriculum(lf, (steps // 4, steps // 4, steps // 2))
            E.append(dict(name="lf_anchor_curr", stages=lfc, channel="none", tag="structure-informed"))
        else:
            keys = exp.endswith("1")
            lfc = fixed_identity_curriculum(lf, (steps // 4, steps // 4, steps // 2))
            for b in (2, 4, 6):
                for arm in ("aware", "self", "random_enc"):
                    E.append(dict(name=f"lf_{arm}_keys{int(keys)}", stages=[(lf, steps)], b=b, channel="gauss",
                                  dec_keys=keys, arm=arm, in_norm=True, key_warmup=1000 if keys else 0))
                E.append(dict(name=f"lf_curr_keys{int(keys)}", stages=lfc, b=b, channel="gauss", dec_keys=keys,
                              in_norm=True, key_warmup=1000 if keys else 0, tag="structure-informed"))
    if exp == "keyscale":     # the gate's CPU ablation variant: concatenated keys x 0.1 (no LayerNorm path)
        for b, ch in ((2, "gauss"), (4, "gauss"), (None, "float")):
            E.append(dict(name="keyscale01_keys1", stages=one, b=b, channel=ch, dec_keys=True, key_scale=0.1))
    if exp == "gatefix":      # gate 2026-10-22 MF1/MF2: in_norm + keys off for 1000 steps; step 3 with keys on/off;
        #                      the anchor under each curriculum and for twice the steps
        for keys in (True, False):
            for b, ch in ((2, "gauss"), (4, "gauss"), (None, "float")):
                E.append(dict(name=f"gate_step1_keys{int(keys)}", stages=one, b=b, channel=ch, dec_keys=keys,
                              in_norm=True, key_warmup=1000 if keys else 0))
        cur = fixed_identity_curriculum(S, (steps // 4, steps // 4, steps // 2))
        for keys in (True, False):
            for b in (2, 4):
                E.append(dict(name=f"gate_step3_curr_keys{int(keys)}", stages=cur, b=b, channel="gauss",
                              dec_keys=keys, in_norm=True, key_warmup=1000 if keys else 0, tag="structure-informed"))
        E.append(dict(name="gate_anchor_curr", stages=cur, channel="none", tag="structure-informed"))
        E.append(dict(name="gate_anchor_long", stages=[(S, 2 * steps)], channel="none"))
        rrb = biased_bit_curriculum(rr, (steps // 4, steps // 4, steps // 2))
        E.append(dict(name="gate_rr_anchor_bias", stages=rrb, channel="none", tag="biased-bit curriculum"))
        rrc = fixed_identity_curriculum(rr, (steps // 4, steps // 4, steps // 2))
        E.append(dict(name="gate_rr_anchor_curr", stages=rrc, channel="none", tag="structure-informed"))
        E.append(dict(name="gate_rr_isab_long", stages=[(rr, 2 * steps)], channel="none", decoder="isab"))
    if exp == "keyfix":         # item 1 (2026-10-22): step 1 with the in_norm key path, keys on and off
        for keys in (True, False):
            for b, ch in ((2, "gauss"), (4, "gauss"), (None, "float")):
                E.append(dict(name=f"keyfix_keys{int(keys)}", stages=one, b=b, channel=ch, dec_keys=keys, in_norm=True))
    if exp in ("rr_main", "rr_main_keys0", "rr_main_keys1"):   # item 2: readers random (leak-REDUCED, not leak-free)
        E.append(dict(name="rr_anchor", stages=[(rr, steps)], channel="none"))
        cur = fixed_identity_curriculum(rr, (steps // 4, steps // 4, steps // 2))
        ks = {"rr_main": (False, True), "rr_main_keys0": (False,), "rr_main_keys1": (True,)}[exp]
        for keys in ks:
            for b in (2, 4, 6):
                for arm in ("aware", "self", "random_enc"):
                    E.append(dict(name=f"rr_{arm}_keys{int(keys)}", stages=[(rr, steps)], b=b, channel="gauss",
                                  dec_keys=keys, arm=arm, in_norm=True))
                E.append(dict(name=f"rr_curr_keys{int(keys)}", stages=cur, b=b, channel="gauss", dec_keys=keys,
                              in_norm=True))
    if exp == "rr_bias":        # item 3: biased-bit curriculum on the leak-reduced variant, keys matched across b
        cur = biased_bit_curriculum(rr, (steps // 4, steps // 4, steps // 2))
        for keys in (False, True):
            E.append(dict(name=f"rr_bias_keys{int(keys)}", stages=cur, b=4, channel="gauss", dec_keys=keys, in_norm=True))
            E.append(dict(name=f"rr_bias_keys{int(keys)}", stages=cur, b=None, channel="float", dec_keys=keys,
                          in_norm=True))
    if exp in ("step5", "all"):
        cur = biased_bit_curriculum(S, (steps // 4, steps // 4, steps // 2))
        E.append(dict(name="step5_bias_curr", stages=cur, b=4, channel="gauss", dec_keys=True))
        E.append(dict(name="step5_bias_curr", stages=cur, b=None, channel="float"))
    return E


def main(argv=None):
    import argparse
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["step1"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    out = {"schema": "aggbench.parity-learn/v1", "results": []}
    try:
        from .results import harness_info
        out["harness"] = harness_info()
    except Exception:   # noqa: BLE001
        pass
    seen = set()
    for exp in a.exp:
        for e in experiments(exp, a.steps):
            for seed in a.seeds:
                key = json.dumps({**{k: v for k, v in e.items() if k != "stages"}, "seed": seed}, sort_keys=True,
                                 default=str) + str([(s.__dict__, n) for s, n in e["stages"]])
                if key in seen:
                    continue
                seen.add(key)
                out["results"].append(train_arm(seed=seed, **e))
                with open(os.path.join(a.out_dir, "parity_learn.json"), "w") as f:
                    json.dump(out, f, default=lambda x: None if x is None else (x.__dict__ if hasattr(x, "__dict__")
                                                                               else float(x)))


if __name__ == "__main__":
    main()


# ----------------------------------------------------------------------------- blind closed-form references (step 0)

def _local_configs(tokens, p=2):
    """Per-agent local configuration index (v1, v2, o1, o2) -> v1 * NV * 4 + v2 * 4 + o1 * 2 + o2, from tokens."""
    n, N, _ = tokens.shape
    obs = np.rint(tokens[..., :p]).astype(np.int64)
    keys = tokens[..., p:].reshape(n, N, p, NV + 1)
    var = keys[..., :NV].argmax(-1)
    return var[..., 0] * NV * 4 + var[..., 1] * 4 + obs[..., 0] * 2 + obs[..., 1]


def config_likelihoods(spec, n_prior=20000, seed=5):
    """P(config, w) for one agent: (NV * NV * 4, 2^NV), the vars from the per-agent structure prior (tallied from
    fresh structures), accuracy marginalized (mean of its prior), readings given the world bits."""
    d = sample(spec, n_prior, np.random.default_rng(seed))
    keys = d["keys"].reshape(n_prior, spec.n_agents, spec.p, NV + 1)
    var = keys[..., :NV].argmax(-1).reshape(-1, spec.p)
    pv = np.zeros((NV, NV))
    np.add.at(pv, (var[:, 0], var[:, 1]), 1)
    pv /= pv.sum()
    a = float(np.mean(spec.acc))
    W = 2 ** NV
    bits = (np.arange(W)[:, None] >> np.arange(NV)) & 1
    prior_w = np.prod(np.where(bits == 1, spec.bit_prob, 1 - spec.bit_prob), 1)
    P = np.zeros((NV * NV * 4, W))
    for v1 in range(NV):
        for v2 in range(NV):
            for o1 in (0, 1):
                for o2 in (0, 1):
                    P[v1 * NV * 4 + v2 * 4 + o1 * 2 + o2] = pv[v1, v2] * np.where(bits[:, v1] == o1, a, 1 - a) * \
                        np.where(bits[:, v2] == o2, a, 1 - a)
    y = bits[:, :spec.k].sum(1) % 2
    return P, prior_w, y


def blind_forecast(code, P, prior_w, y, cfg, L):
    """Exact posterior given each agent's symbol code[cfg] (agents independent draws from the per-agent prior;
    exact when the symbol reveals the vars)."""
    T = np.zeros((L, P.shape[1]))
    np.add.at(T, code, P)
    with np.errstate(divide="ignore"):
        logT = np.log(T)
    lp = np.log(prior_w) + logT[code[cfg]].sum(1)
    num = np.logaddexp.reduce(np.where(y == 1, lp, -np.inf), -1)
    den = np.logaddexp.reduce(np.where(y == 0, lp, -np.inf), -1)
    return np.clip(np.nan_to_num(num - den), -60, 60)


def named_codes():
    """Hand codes over configs (v1, v2, o1, o2): raw readings (no identity), count of parity 1-readings, and the
    bit-identity pair (which parity bit or 'distractor', reading; unordered)."""
    c = np.arange(NV * NV * 4)
    v1, v2, o1, o2 = c // (NV * 4), (c // 4) % NV, (c // 2) % 2, c % 2
    raw = o1 * 2 + o2
    count = o1 * (v1 < 4) + o2 * (v2 < 4)
    t1 = np.where(v1 < 4, 1 + v1 * 2 + o1, 0)
    t2 = np.where(v2 < 4, 1 + v2 * 2 + o2, 0)
    a, b = np.minimum(t1, t2), np.maximum(t1, t2)
    pair = a * 9 - a * (a - 1) // 2 + (b - a)
    _, pair = np.unique(pair, return_inverse=True)
    return {"raw (2 bits, no identity)": (raw, 4), "count of parity 1-readings (log2 3 bits)": (count, 3),
            "bit identity + reading, unordered pair": (pair, int(pair.max()) + 1)}


def searched_code(b, P, prior_w, y, cfg, truth_bayes, init=None, sweeps=3, seed=0):
    """Coordinate search over maps config -> 2^b symbols minimizing train KL (a blind upper bound on the b-bit
    optimum, not the optimum)."""
    L = 2 ** b
    rng = np.random.default_rng(seed)
    code = (init % L) if init is not None else rng.integers(0, L, P.shape[0])
    loss = lambda c: float(_kl(blind_forecast(c, P, prior_w, y, cfg, L), truth_bayes).mean())
    best = loss(code)
    used = np.unique(cfg)
    for _ in range(sweeps):
        improved = False
        for c in rng.permutation(used):
            for s in range(L):
                if s == code[c]:
                    continue
                old = code[c]
                code[c] = s
                v = loss(code)
                if v < best - 1e-9:
                    best, improved = v, True
                else:
                    code[c] = old
        if not improved:
            break
    return code, best


def blind_references(spec=None, n_train=3000, n_dev=4000, bits=(1, 2, 3, 4), seed=0):
    """Blind (structure-free) closed-form references on grouped Z9 with per-agent symbol entropy."""
    from .metrics import summarize_values
    spec = spec or Z9Spec()
    P, prior_w, y = config_likelihoods(spec)
    tr = sample(spec, n_train, np.random.default_rng([seed, 1]))
    dv = sample(spec, n_dev, np.random.default_rng(DEV_SEED))
    ctr, cdv = _local_configs(tr["tokens"]), _local_configs(dv["tokens"])
    rows = []

    def score(name, code, L, bits_):
        f = blind_forecast(code, P, prior_w, y, cdv, L)
        kl = _kl(f, dv["bayes"])
        rows.append({"code": name, "bits": bits_, "regret": summarize_values(kl, 1000),
                     "symbol_entropy_bits": _entropy_bits(code[cdv].ravel())})

    named = named_codes()
    for name, (code, L) in named.items():
        score(name, code, L, round(math.log2(L), 2))
    for b in bits:
        init = named["count of parity 1-readings (log2 3 bits)"][0] if b >= 2 else None
        code, _ = searched_code(b, P, prior_w, y, ctr, tr["bayes"], init=init, seed=seed)
        score(f"searched {b}-bit code (coordinate search on train)", code, 2 ** b, b)
    return rows


def _multiset_keys(sym, L):
    """Exchangeable decoder input: the sorted symbol multiset of an episode as one integer."""
    s = np.sort(sym, 1)
    return (s * (L ** np.arange(s.shape[1]))).sum(1)


def tabulated_forecast(code, L, spec, cfg_eval, n_tab=2_000_000, seed=11, chunk=250_000):
    """Blind decoder that uses the JOINT structure prior: P(Y | multiset of symbols) tabulated from n_tab fresh
    simulator episodes (add-1/2 smoothing; unseen multisets get the prior). Exact up to Monte Carlo error; it sees
    the coupling between agents (e.g. exactly 3 readers per parity bit) that per-agent decoders miss."""
    rng = np.random.default_rng(seed)
    tab = {}
    keys_all, y_all = [], []
    for s0 in range(0, n_tab, chunk):
        d = sample(spec, min(chunk, n_tab - s0), rng)
        keys_all.append(_multiset_keys(code[_local_configs(d["tokens"])], L))
        y_all.append(d["y"])
    k, y = np.concatenate(keys_all), np.concatenate(y_all)
    uk, inv = np.unique(k, return_inverse=True)
    n1 = np.bincount(inv, weights=y)
    n = np.bincount(inv)
    p = (n1 + 0.5) / (n + 1.0)
    l0 = prior_logit(spec)
    ke = _multiset_keys(code[cfg_eval], L)
    idx = np.searchsorted(uk, ke)
    ok = (idx < len(uk)) & (uk[np.minimum(idx, len(uk) - 1)] == ke)
    pe = np.where(ok, p[np.minimum(idx, len(uk) - 1)], 1 / (1 + math.exp(-l0)))
    return np.log(pe) - np.log1p(-pe), float(ok.mean())


def blind_references_tabulated(spec=None, n_dev=4000, n_tab=2_000_000, search_bits=(1, 2, 3), n_search=200_000,
                               seed=0):
    """Blind references with the joint-prior tabulated decoder (codes with <= 8 symbols) and the per-agent decoder
    for the bit-identity code (exact: its symbols reveal the vars). Per-agent symbol entropy logged."""
    from .metrics import summarize_values
    spec = spec or Z9Spec()
    dv = sample(spec, n_dev, np.random.default_rng(DEV_SEED))
    cdv = _local_configs(dv["tokens"])
    rows = []

    def add(name, code, L, bits_, f, cover=None):
        kl = _kl(f, dv["bayes"])
        rows.append({"code": name, "bits": bits_, "regret": summarize_values(kl, 1000),
                     "symbol_entropy_bits": _entropy_bits(code[cdv].ravel()), "tab_coverage": cover})

    named = named_codes()
    for name, (code, L) in named.items():
        if L <= 8:
            f, cov = tabulated_forecast(code, L, spec, cdv, n_tab)
        else:
            P, prior_w, y = config_likelihoods(spec)
            f, cov = blind_forecast(code, P, prior_w, y, cdv, L), None
        add(name, code, L, round(math.log2(L), 2), f, cov)
    # coordinate search with the tabulated decoder on n_search train episodes (scored on fresh dev, n_tab table)
    srch = sample(spec, n_search, np.random.default_rng([seed, 7]))
    cs, ys = _local_configs(srch["tokens"]), srch["y"]
    used = np.unique(cs)
    for b in search_bits:
        L = 2 ** b
        code = named["count of parity 1-readings (log2 3 bits)"][0].copy() % L if b >= 2 else \
            named["raw (2 bits, no identity)"][0] % 2

        def loss(c):                      # train log loss of the tabulated decoder (in-sample, smoothed)
            k = _multiset_keys(c[cs], L)
            _, inv = np.unique(k, return_inverse=True)
            n1, n = np.bincount(inv, weights=ys), np.bincount(inv)
            p = (n1 + 0.5) / (n + 1.0)
            return float(-(n1 * np.log(p) + (n - n1) * np.log1p(-p)).sum() / len(ys))
        best = loss(code)
        rng = np.random.default_rng(seed)
        for _ in range(3):
            improved = False
            for c in rng.permutation(used):
                for s in range(L):
                    if s == code[c]:
                        continue
                    old, code[c] = code[c], s
                    v = loss(code)
                    if v < best - 1e-7:
                        best, improved = v, True
                    else:
                        code[c] = old
            if not improved:
                break
        f, cov = tabulated_forecast(code, L, spec, cdv, n_tab, seed=12)
        add(f"searched {b}-bit code (coordinate search, tabulated decoder)", code, L, b, f, cov)
    return rows
