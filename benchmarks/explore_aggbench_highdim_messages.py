#!/usr/bin/env python3
"""Track-E Aggbench High-Dimensional Local Messages & Clearing Experiment.

Tests whether per-agent strictly local encoders emitting high-dimensional key-value
messages (d_v in {1, 2, 4, 8, 12}) coupled with slot-equivariant Gram/Woodbury
overlap clearing and peer-consistency re-calibration can beat the 1D scalar
report ceiling (`oracle_clearing`, ~0.017 regret) and approach the global
Bayes posterior (`bayes`, 0.000 regret) on E2, E3-sybil, E3-collude, and E3-biased,
trained strictly on binary truth labels Y in {0, 1}.

Rules enforced:
- `aggbench` is imported unmodified (`harness.git_dirty == False`).
- Strict per-agent message locality verified via `aggbench.messages.check_locality`
  for every learned encoder before aggregation.
- Fair E3-biased distortion applied to the biased agent's seen observations via
  `biased_agent_obs(batch)` prior to encoding (unobserved slots remain NaN, and
  no oracle role/a/b fields are ever passed on AgentInputView).
- Every learned arm is evaluated with `aggbench.controls.run_with_controls`
  (`trained`, `control:untrained_init`, `control:random_network`).
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
from dataclasses import dataclass
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)

from aggbench import make_env
from aggbench.baselines import all_baselines
from aggbench.controls import run_with_controls
from aggbench.messages import check_locality
from aggbench.metrics import paired_difference, summarize
from aggbench.results import make_result, split_record, write_result
from aggbench.verify import oracle_check


ENVS = ("E2", "E3-sybil", "E3-collude", "E3-biased")
VISIBILITIES = ("full", "noisy")
DV_DIMS = (1, 2, 4, 8, 12)


def env_slug(env_name: str) -> str:
    return env_name.lower().replace("-", "_")


def biased_agent_obs(batch) -> np.ndarray:
    """Apply E3-biased distortion to the biased agent's seen observations before encoding.

    Matches the scalar report distortion `a * sum_{k in seen} lam_k + b` across the seen
    pieces while keeping unseen pieces as `np.nan` (no empty-bucket leak, no oracle `a/b`
    passed to the encoder or aggregator).
    """
    obs = batch.obs.copy()
    a, b = batch.oracle["a"], batch.oracle["b"]
    mu = batch.oracle["mu"]
    for e, i in zip(*np.nonzero(batch.oracle["role"] == 2)):
        seen = np.isfinite(obs[e, i])
        lam = 2 * mu[e] * obs[e, i]
        lam_new = a[e, i] * lam + b[e, i] / max(1, seen.sum())
        obs[e, i] = np.where(seen, lam_new / (2 * mu[e]), np.nan)
    return obs


@dataclass
class AgentInputView:
    """Per-agent local inputs for learned message encoders (no oracle fields)."""
    obs: np.ndarray
    keys: np.ndarray
    public: dict


def make_agent_input_view(batch, visibility: str = "full") -> AgentInputView:
    _, keys, public = batch.agent_inputs(visibility)
    obs = biased_agent_obs(batch)
    return AgentInputView(obs=obs, keys=keys, public=public)


def make_bucket_matrix(K: int, d_v: int) -> torch.Tensor:
    """Partition K slots into d_v disjoint buckets: shape (d_v, K)."""
    P = torch.zeros(d_v, K, dtype=torch.float32)
    for k in range(K):
        b = (k * d_v) // K
        P[b, k] = 1.0
    return P


class NaiveMHAModule(nn.Module):
    """Naive non-slot-equivariant MLP local encoder + standard MultiheadAttention aggregator."""

    def __init__(self, K: int, key_dim: int, d_v: int, hidden: int = 24):
        super().__init__()
        self.K = K
        self.d_v = d_v
        self.in_dim = 2 * K + key_dim
        self.enc_fc1 = nn.Linear(self.in_dim, hidden)
        self.enc_fc2 = nn.Linear(hidden, d_v)
        self.msg_embed = nn.Linear(d_v + key_dim, hidden)
        self.attn = nn.MultiheadAttention(hidden, num_heads=2, batch_first=True)
        self.out_fc1 = nn.Linear(hidden, hidden)
        self.out_fc2 = nn.Linear(hidden, 1)

    def encode_tensors(self, obs_t: torch.Tensor, keys_t: torch.Tensor) -> torch.Tensor:
        c_obs = (~torch.isnan(obs_t)).float()
        x = torch.nan_to_num(obs_t, nan=0.0)
        inp = torch.cat([x, c_obs, keys_t], dim=-1)
        v_msg = self.enc_fc2(F.silu(self.enc_fc1(inp)))
        return torch.cat([v_msg, keys_t], dim=-1)

    def aggregate_messages(self, msg: torch.Tensor) -> torch.Tensor:
        z = F.silu(self.msg_embed(msg))
        z = z + self.attn(z, z, z, need_weights=False)[0]
        h = F.silu(self.out_fc1(z.mean(dim=1)))
        return self.out_fc2(h)[:, 0]


class SlotKVModule(nn.Module):
    """Slot-Equivariant Key-Value Local Encoder + Exact Clearing & Confiability Aggregator.

    1. Local encoder (`encode_tensors`): maps each agent i's local slot features
       `(x_{i,k}, c_{i,k}^{obs}, c_{i,k}^{key}, m_{i,k}^{key})` independently across agents
       into a d_v-dimensional bucketed value vector `v_i in R^{d_v}` plus slot key weights.
    2. Aggregator (`aggregate_messages`): operates strictly on the emitted messages:
       - Forms consensus slot strength across agents reporting key strengths on slot k.
       - Computes robust Huber peer-disagreement and key-calibrated magnitude residuals
         per agent to estimate reliability `agent_trust` and affine de-biasing `(scale, shift)`.
       - Solves the regularized per-bucket clearing system
         `(C_b^T B C_b + diag(lam / m_b)) w_b = C_b^T B v_b` for each bucket b in 1..d_v.
    """

    def __init__(self, K: int, d_v: int, hidden: int = 24):
        super().__init__()
        self.K = K
        self.d_v = d_v
        P = make_bucket_matrix(K, d_v)
        self.register_buffer("P", P)  # (d_v, K)
        self.bucket_indices = [
            torch.nonzero(P[b] > 0.5, as_tuple=False).squeeze(-1) for b in range(d_v)
        ]
        self.equal_buckets = (K % d_v == 0)

        # Per-slot local value encoder: 6 features -> 1 scalar per slot
        self.val_fc1 = nn.Linear(6, hidden)
        self.val_fc2 = nn.Linear(hidden, 1)

        # Per-slot local key encoder: 4 features -> [c_hat, m_self, m_obs_key]
        self.key_fc1 = nn.Linear(4, hidden)
        self.key_fc2 = nn.Linear(hidden, 3)

        # Per-agent reliability & affine calibration head from message-level consistency features
        self.trust_fc1 = nn.Linear(10, hidden)
        self.trust_fc2 = nn.Linear(hidden, 3)

        self.raw_log_lam = nn.Parameter(torch.tensor([-3.5]))

    def encode_tensors(self, obs_t: torch.Tensor, keys_t: torch.Tensor) -> torch.Tensor:
        K = self.K
        c_obs = (~torch.isnan(obs_t)).float()
        x = torch.nan_to_num(obs_t, nan=0.0)
        if keys_t.shape[-1] >= 2 * K:
            c_key = keys_t[..., :K]
            m_key = keys_t[..., K:2 * K]
        else:
            c_key = torch.zeros_like(x)
            m_key = torch.zeros_like(x)

        key_in = torch.stack([c_key, m_key, m_key * m_key, c_obs], dim=-1)
        k_raw = self.key_fc2(F.silu(self.key_fc1(key_in)))  # (n, N, K, 3)
        c_hat = torch.sigmoid(k_raw[..., 0] + 2.0) * c_obs
        m_self = (F.softplus(k_raw[..., 1]) + 0.05) * c_obs
        m_obs_key = (F.softplus(k_raw[..., 2]) + 0.05) * (c_key > 0.5).float()

        val_in = torch.stack([
            x,
            x * m_self,
            x * m_key,
            c_obs,
            c_key,
            m_key,
        ], dim=-1)
        slot_v = self.val_fc2(F.silu(self.val_fc1(val_in)))[..., 0] * c_obs  # (n, N, K)
        v_msg = torch.einsum("bk,nik->nib", self.P, slot_v)                  # (n, N, d_v)
        return torch.cat([v_msg, c_hat, m_self, m_obs_key], dim=-1)          # (n, N, d_v + 3*K)

    def _clear_buckets(
        self,
        v_in: torch.Tensor,
        c_in: torch.Tensor,
        m_in: torch.Tensor,
        lam: torch.Tensor,
    ) -> torch.Tensor:
        """Solve (C_b^T B C_b + diag(lam / (m_b + eps))) w_b = C_b^T B v_b per bucket b."""
        n, N, d_v = v_in.shape
        K = self.K
        m_slot = m_in.max(dim=1).values + 1e-2  # (n, K)
        lam_slot = lam / m_slot                 # (n, K)

        if d_v == K:
            num_k = (c_in * v_in).sum(dim=1)              # (n, K)
            den_k = (c_in ** 2).sum(dim=1) + lam_slot     # (n, K)
            return (num_k / den_k).sum(dim=-1)
        elif self.equal_buckets:
            s_b = K // d_v
            C_b = c_in.view(n, N, d_v, s_b).permute(0, 2, 1, 3)  # (n, d_v, N, s_b)
            v_b = v_in.permute(0, 2, 1).unsqueeze(-1)            # (n, d_v, N, 1)
            lam_b = lam_slot.view(n, d_v, s_b)                   # (n, d_v, s_b)
            Gram_s = torch.matmul(C_b.transpose(-1, -2), C_b) + torch.diag_embed(lam_b)
            rhs = torch.matmul(C_b.transpose(-1, -2), v_b)
            w_b = torch.linalg.solve(Gram_s, rhs).squeeze(-1)
            return w_b.sum(dim=(-1, -2))
        else:
            cleared_parts = []
            for b, idx in enumerate(self.bucket_indices):
                s_b = len(idx)
                C_b = c_in[:, :, idx]        # (n, N, s_b)
                v_b = v_in[:, :, b:b + 1]    # (n, N, 1)
                lam_b = lam_slot[:, idx]     # (n, s_b)
                if s_b == 1:
                    num = (C_b[..., 0] * v_b[..., 0]).sum(dim=1)
                    den = (C_b[..., 0] ** 2).sum(dim=1) + lam_b[..., 0]
                    cleared_parts.append(num / den)
                else:
                    Gram_s = torch.matmul(C_b.transpose(-1, -2), C_b) + torch.diag_embed(lam_b)
                    rhs = torch.matmul(C_b.transpose(-1, -2), v_b)
                    w_b = torch.linalg.solve(Gram_s, rhs).squeeze(-1)
                    cleared_parts.append(w_b.sum(dim=-1))
            return torch.stack(cleared_parts, dim=-1).sum(dim=-1)

    def aggregate_messages(self, msg: torch.Tensor) -> torch.Tensor:
        n, N, _ = msg.shape
        d_v, K = self.d_v, self.K
        v_msg_raw = msg[..., :d_v]
        c_msg = msg[..., d_v:d_v + K]
        m_self = msg[..., d_v + K:d_v + 2 * K]
        m_obs_key = msg[..., d_v + 2 * K:d_v + 3 * K]

        # Consensus slot strength across agents who received a key observation on slot k
        has_k = (m_obs_key > 1e-4).float()
        cnt_k = has_k.sum(dim=1, keepdim=True)  # (n, 1, K)
        mean_k_global = m_obs_key.sum(dim=(1, 2), keepdim=True) / has_k.sum(
            dim=(1, 2), keepdim=True
        ).clamp(min=1.0)
        m_cons_slot = torch.where(
            cnt_k > 0.5,
            m_obs_key.sum(dim=1, keepdim=True) / cnt_k.clamp(min=1.0),
            mean_k_global,
        )  # (n, 1, K)

        # Rescale each agent's bucket value from its local m_self to consensus m_cons_slot
        m_self_bucket = torch.einsum("bk,nik->nib", self.P, m_self * c_msg)
        m_cons_bucket = torch.einsum("bk,nik->nib", self.P, m_cons_slot * c_msg)
        c_bucket = torch.einsum("bk,nik->nib", self.P, c_msg)  # (n, N, d_v)
        active_b = (c_bucket > 1e-4).float()                   # (n, N, d_v)
        ratio_m = torch.where(
            active_b > 0.5,
            (m_cons_bucket + 1e-3) / (m_self_bucket + 1e-3),
            torch.ones_like(m_self_bucket),
        )
        v_msg = v_msg_raw * ratio_m.clamp(0.2, 5.0)
        m_msg = m_cons_slot * c_msg
        m_bucket = m_cons_bucket

        n_active_i = active_b.sum(dim=-1).clamp(min=1.0)       # (n, N)
        sig_b = torch.sqrt(m_bucket + 1e-4) + 0.1              # (n, N, d_v)
        z_b = (v_msg / sig_b) * active_b                       # (n, N, d_v)

        # Robust Huber-weighted peer consensus per bucket
        huber_w = active_b / (1.0 + 0.25 * (z_b ** 2))
        sum_wv_b = (huber_w * v_msg).sum(dim=1, keepdim=True) - (huber_w * v_msg)
        sum_w_b = huber_w.sum(dim=1, keepdim=True) - huber_w
        has_peer_b = ((active_b.sum(dim=1, keepdim=True) - active_b) > 0.5).float() * active_b
        peer_mean_b = sum_wv_b / sum_w_b.clamp(min=1e-4)
        diff_peer = (v_msg - peer_mean_b) * has_peer_b
        n_shared_i = has_peer_b.sum(dim=-1).clamp(min=1.0)
        abs_diff_peer = diff_peer.abs().sum(dim=-1) / n_shared_i
        signed_diff_peer = diff_peer.sum(dim=-1) / n_shared_i
        frac_disagree = ((diff_peer.abs() > 0.25).float() * has_peer_b).sum(dim=-1) / n_shared_i
        frac_agree = ((diff_peer.abs() < 0.08).float() * has_peer_b).sum(dim=-1) / n_shared_i

        norm_ratio_i = z_b.abs().sum(dim=-1) / n_active_i
        norm_ratio_peer = norm_ratio_i - norm_ratio_i.mean(dim=-1, keepdim=True)
        signed_z_i = z_b.sum(dim=-1) / n_active_i
        signed_z_peer = signed_z_i - signed_z_i.mean(dim=-1, keepdim=True)

        v_tot_signed = v_msg.sum(dim=-1)
        m_tot = m_bucket.sum(dim=-1)

        trust_in = torch.stack([
            abs_diff_peer,
            signed_diff_peer,
            frac_disagree,
            frac_agree,
            norm_ratio_i,
            norm_ratio_peer,
            signed_z_i,
            signed_z_peer,
            v_tot_signed,
            m_tot,
        ], dim=-1)  # (n, N, 10)
        trust_out = self.trust_fc2(F.silu(self.trust_fc1(trust_in)))  # (n, N, 3)
        agent_trust = torch.sigmoid(trust_out[..., 0] + 1.0)          # (n, N)
        agent_scale = F.softplus(trust_out[..., 1] + 0.55) + 0.1      # (n, N)
        agent_shift = trust_out[..., 2]                               # (n, N)

        c_frac = c_bucket / c_bucket.sum(dim=-1, keepdim=True).clamp(min=1e-4)
        v_recal = ((v_msg - agent_shift.unsqueeze(-1) * c_frac) / agent_scale.unsqueeze(-1)) * active_b

        sqrt_trust = torch.sqrt(agent_trust + 1e-6).unsqueeze(-1)
        c_weighted = c_msg * sqrt_trust
        v_weighted = v_recal * sqrt_trust

        lam = F.softplus(self.raw_log_lam[0]) + 1e-4
        return self._clear_buckets(v_weighted, c_weighted, m_msg, lam)


class HighDimAggregator:
    """Learned local encoder + message aggregator compatible with `aggbench.controls`."""

    def __init__(
        self,
        seed: int,
        module_fn,
        epochs: int = 30,
        batch_size: int = 400,
        lr: float = 8e-3,
    ):
        torch.manual_seed(seed)
        self.module = module_fn()
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    def encode(self, obs: np.ndarray, keys: np.ndarray) -> np.ndarray:
        """Strictly local per-agent message encoder checked by `aggbench.messages.check_locality`."""
        self.module.eval()
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32)
            keys_t = torch.tensor(keys, dtype=torch.float32)
            msg = self.module.encode_tensors(obs_t, keys_t)
            return msg.double().numpy()

    def fit(self, view: AgentInputView, truth: np.ndarray):
        """End-to-end training strictly on binary truth labels `truth`."""
        self.module.train()
        obs_t = torch.tensor(view.obs, dtype=torch.float32)
        keys_t = torch.tensor(view.keys, dtype=torch.float32)
        y_t = torch.tensor(truth, dtype=torch.float32)
        l0 = float(view.public["prior_logit"])
        n = len(obs_t)

        opt = torch.optim.AdamW(self.module.parameters(), lr=self.lr, weight_decay=1e-4)
        total_steps = self.epochs * max(1, (n + self.batch_size - 1) // self.batch_size)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
        gen = torch.Generator().manual_seed(42)

        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=gen)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                opt.zero_grad()
                msg = self.module.encode_tensors(obs_t[idx], keys_t[idx])
                logit = l0 + self.module.aggregate_messages(msg)
                loss = F.binary_cross_entropy_with_logits(logit, y_t[idx])
                loss.backward()
                opt.step()
                sched.step()
        self.module.eval()

    def predict(self, view: AgentInputView) -> dict:
        """Inference pipeline: local encode -> message-only aggregation -> logit forecast."""
        self.module.eval()
        with torch.no_grad():
            obs_t = torch.tensor(view.obs, dtype=torch.float32)
            keys_t = torch.tensor(view.keys, dtype=torch.float32)
            l0 = float(view.public["prior_logit"])
            msg = self.module.encode_tensors(obs_t, keys_t)
            logit = l0 + self.module.aggregate_messages(msg)
            return {"logit": logit.double().numpy()}


def run_single_cell(
    env_name: str,
    visibility: str,
    n_train: int = 2000,
    n_dev: int = 2000,
    n_boot: int = 2000,
    run_scaling: bool = False,
) -> dict:
    """Run all baselines, naive MHA arms, and SlotKV d_v in {1, 2, 4, 8, 12} arms on (env, visibility)."""
    t0 = time.time()
    env = make_env(env_name)
    tr = env.sample("train", n_train)
    dv = env.sample("dev", n_dev)
    K = env.config.n_pieces
    key_dim = tr.keys[visibility].shape[-1]

    forecasts, fitted_alpha = all_baselines(tr, dv, visibility)
    tr_view = make_agent_input_view(tr, visibility)
    dv_view = make_agent_input_view(dv, visibility)

    arm_specs = [
        ("naive_mha_dv1", lambda s: HighDimAggregator(s, lambda: NaiveMHAModule(K, key_dim, 1))),
        ("naive_mha_dv12", lambda s: HighDimAggregator(s, lambda: NaiveMHAModule(K, key_dim, 12))),
    ]
    for d_v in DV_DIMS:
        arm_specs.append(
            (f"slot_kv_dv{d_v}", lambda s, d=d_v: HighDimAggregator(s, lambda: SlotKVModule(K, d)))
        )

    learned_arms = []
    controls = {}
    locality_checks = {}
    paired_vs_oracle_clearing = {}
    paired_vs_extremized = {}

    for arm_prefix, make_fn in arm_specs:
        # 1. Verify locality on untrained encoder
        probe = make_fn(0)
        assert check_locality(probe.encode, dv_view.obs[:32], dv_view.keys[:32])

        # 2. Train and evaluate with controls
        arms_out = run_with_controls(make_fn, tr_view, tr.truth, dv_view, seed=0)

        # 3. Also verify locality on trained encoder
        trained_obj = make_fn(0)
        trained_obj.fit(tr_view, tr.truth)
        assert check_locality(trained_obj.encode, dv_view.obs[:32], dv_view.keys[:32])
        locality_checks[arm_prefix] = True

        trained_name = f"{arm_prefix}:trained"
        ui_name = f"{arm_prefix}:control:untrained_init"
        rn_name = f"{arm_prefix}:control:random_network"

        forecasts[trained_name] = arms_out["trained"]
        forecasts[ui_name] = arms_out["control:untrained_init"]
        forecasts[rn_name] = arms_out["control:random_network"]

        learned_arms.append(trained_name)
        controls[trained_name] = {
            "untrained_init": ui_name,
            "random_network": rn_name,
        }
        paired_vs_oracle_clearing[trained_name] = paired_difference(
            arms_out["trained"], forecasts["oracle_clearing"], dv, metric="regret", n_boot=n_boot, seed=0
        )
        paired_vs_extremized[trained_name] = paired_difference(
            arms_out["trained"], forecasts["extremized_log_pool"], dv, metric="regret", n_boot=n_boot, seed=0
        )

    # Optional sample-size scaling ablation on E2 (full) for N_train in {500, 2000, 8000}
    sample_scaling = {}
    if run_scaling:
        for n_tr_scale in (500, 2000, 8000):
            tr_s = env.sample("train", n_tr_scale)
            tr_s_view = make_agent_input_view(tr_s, visibility)
            epochs_s = 35 if n_tr_scale == 500 else (30 if n_tr_scale == 2000 else 20)
            make_scale = lambda s, ep=epochs_s: HighDimAggregator(
                s, lambda: SlotKVModule(K, 12), epochs=ep
            )
            arms_s = run_with_controls(make_scale, tr_s_view, tr_s.truth, dv_view, seed=0)
            sum_s = summarize(arms_s["trained"], dv, n_boot=n_boot, seed=0)["regret"]
            diff_oc_s = paired_difference(
                arms_s["trained"], forecasts["oracle_clearing"], dv, metric="regret", n_boot=n_boot, seed=0
            )
            sample_scaling[str(n_tr_scale)] = {
                "n_train": n_tr_scale,
                "regret": sum_s,
                "diff_vs_oracle_clearing": diff_oc_s,
            }

    methods = {k: summarize(f, dv, n_boot=n_boot, seed=0) for k, f in forecasts.items()}
    oc_check = oracle_check(env, n=24)

    extra = {
        "locality_verified": all(locality_checks.values()),
        "locality_checks": locality_checks,
        "fitted_extremize_alpha": float(fitted_alpha),
        "paired_difference_regret_vs_oracle_clearing": paired_vs_oracle_clearing,
        "paired_difference_regret_vs_extremized_log_pool": paired_vs_extremized,
        "biased_distortion_applied_to_seen_obs": bool(env_name == "E3-biased"),
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    if sample_scaling:
        extra["sample_scaling_dv12"] = sample_scaling

    run_id = f"agg_highdim_{env_slug(env_name)}_{visibility}"
    result = make_result(
        run_id=run_id,
        track="EXPLORATORY",
        smoke=False,
        env=env,
        visibility=visibility,
        splits={"train": split_record(tr, "train"), "eval": split_record(dv, "dev")},
        oracle_check=oc_check,
        methods=methods,
        learned_arms=learned_arms,
        controls=controls,
        extra=extra,
    )
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", default="results/explore")
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-dev", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    all_results = {}

    for env_name in ENVS:
        for vis in VISIBILITIES:
            run_scaling = (env_name == "E2" and vis == "full")
            print(f"\n>>> Running {env_name} | visibility={vis} (scaling={run_scaling}) ...", flush=True)
            res = run_single_cell(
                env_name=env_name,
                visibility=vis,
                n_train=args.n_train,
                n_dev=args.n_dev,
                n_boot=args.n_boot,
                run_scaling=run_scaling,
            )
            run_dir = out_root / res["run_id"]
            out_path = write_result(run_dir, res)
            all_results[(env_name, vis)] = res

            oc = res["methods"]["oracle_clearing"]["regret"]
            ex = res["methods"]["extremized_log_pool"]["regret"]
            print(
                f"  oracle_clearing     : {oc['mean']:.4f} [{oc['ci_low']:.4f}, {oc['ci_high']:.4f}]\n"
                f"  extremized_log_pool : {ex['mean']:.4f} [{ex['ci_low']:.4f}, {ex['ci_high']:.4f}]",
                flush=True,
            )
            for arm in res["learned_arms"]:
                r = res["methods"][arm]["regret"]
                d_oc = res["extra"]["paired_difference_regret_vs_oracle_clearing"][arm]
                rn_name = res["controls"][arm]["random_network"]
                r_rn = res["methods"][rn_name]["regret"]
                print(
                    f"  {arm:24s}: {r['mean']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}] "
                    f"| vs_oc {d_oc['mean']:+.4f} [{d_oc['ci_low']:+.4f}, {d_oc['ci_high']:.4f}] "
                    f"| rnd {r_rn['mean']:.4f}",
                    flush=True,
                )
            if "sample_scaling_dv12" in res["extra"]:
                for n_s, sc in res["extra"]["sample_scaling_dv12"].items():
                    sr = sc["regret"]
                    sd = sc["diff_vs_oracle_clearing"]
                    print(
                        f"  scaling N_train={n_s:4s}   : {sr['mean']:.4f} [{sr['ci_low']:.4f}, {sr['ci_high']:.4f}] "
                        f"| vs_oc {sd['mean']:+.4f} [{sd['ci_low']:+.4f}, {sd['ci_high']:.4f}]",
                        flush=True,
                    )
            print(f"  wrote {out_path}", flush=True)

    # Also write primary result into results/explore/expagghighdim/result.json
    primary_res = dict(all_results[("E2", "full")])
    primary_res["run_id"] = "expagghighdim"
    primary_res["extra"] = dict(primary_res["extra"])
    primary_res["extra"]["all_cells_summary"] = {
        f"{e}_{v}": {
            "oracle_clearing": all_results[(e, v)]["methods"]["oracle_clearing"]["regret"],
            "extremized_log_pool": all_results[(e, v)]["methods"]["extremized_log_pool"]["regret"],
            "naive_mha_dv12": all_results[(e, v)]["methods"]["naive_mha_dv12:trained"]["regret"],
            "slot_kv_dv1": all_results[(e, v)]["methods"]["slot_kv_dv1:trained"]["regret"],
            "slot_kv_dv2": all_results[(e, v)]["methods"]["slot_kv_dv2:trained"]["regret"],
            "slot_kv_dv4": all_results[(e, v)]["methods"]["slot_kv_dv4:trained"]["regret"],
            "slot_kv_dv8": all_results[(e, v)]["methods"]["slot_kv_dv8:trained"]["regret"],
            "slot_kv_dv12": all_results[(e, v)]["methods"]["slot_kv_dv12:trained"]["regret"],
            "slot_kv_dv12_vs_oracle_clearing": all_results[(e, v)]["extra"][
                "paired_difference_regret_vs_oracle_clearing"
            ]["slot_kv_dv12:trained"],
            "slot_kv_dv12_vs_extremized": all_results[(e, v)]["extra"][
                "paired_difference_regret_vs_extremized_log_pool"
            ]["slot_kv_dv12:trained"],
        }
        for (e, v) in all_results
    }
    write_result(out_root / "expagghighdim", primary_res)
    print(f"\nAll 8 cells + expagghighdim written successfully.", flush=True)


if __name__ == "__main__":
    main()
