#!/usr/bin/env python3
"""EXPLORATORY (`PROTOCOL.md` §10.1) Track-E Aggbench Runner 3 (`expaggkylevoi`):
Decentralized Kyle-LMSR Pinned-Market Clearing & Active VoI 3-Parameter Split.

Serves: one unified attention-and-market mechanism for decentralized Kyle-LMSR clearing,
active VoI source selection, and the 3-parameter split (params_real, params_search, params_amp)
approaching full Bayes.

Tests two theoretical hypotheses on aggbench (E1, E2, E3-sybil, E3-biased, E3-collude),
training strictly on `tr.truth` and evaluating on `dev` (`n_dev=2000`):

Part A — Decentralized Kyle-LMSR / Negative-Attention Pinned-Market Training vs. Centralized
End-to-End Backprop on High-Dimensional Local Messages (`d_v = 12`):
  - Local slot-equivariant encoder `encode(obs_i, key_i) -> (v_i, k_i)` verified by
    `aggbench.messages.check_locality` (`d_v = 12`).
  - Compare:
    * `bayes`, `oracle_clearing`, `extremized_log_pool`, `linear_pool`, `log_pool`, `sum_logodds`
    * `kyle_decentralized_naive_linear`: `.detach()` at the message boundary (0 cross-agent
      autograd edges) under naive simultaneous linear pooling.
    * `kyle_decentralized_pinned_lir`: `.detach()` at the message boundary against the
      Kyle-LMSR Pinned Leave-One-Out / Pre-Entry Market Price p^{(-i)} cleared via the
      precision-weighted Key-Gram matrix (K^T B_trust K + Lambda)^{-1}, while `theta_trust`
      (in `params_real`) is trained on `tr.truth` to downweight cross-agent disagreement on
      shared active keys.
    * `centralized_e2e_kv_clearing`: End-to-end joint backpropagation through all encoders
      and the Key-Gram confiability clearer.

Part B — Active VoI Source Selection + 3-Parameter Split (`params_real = (theta_base, theta_trust)`,
`params_search`, `params_amp`) under Budgeted Source Queries (Q in {1, 2, 3, 4} out of N):
  - Compare:
    * `voi_random_choice_Q{Q}`: Picks Q agents uniformly at random and clears their messages.
    * `voi_greedy_salience_no_dedup_Q{Q}`: Picks the Q agents with largest individual key norm
      ||k_i||_2 without penalizing key overlap.
    * `voi_3param_schur_query_Q{Q}`: `params_search` sequentially selects the next agent
      maximizing the marginal Schur-complement log-det information gain, `params_real`
      (`theta_base + theta_trust`) clears the Q high-dim messages, and `params_amp` refines
      the cleared posterior via self-distilled amortization.
"""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aggbench import make_env
from aggbench.baselines import all_baselines
from aggbench.controls import run_with_controls
from aggbench.messages import check_locality
from aggbench.metrics import paired_difference, summarize
from aggbench.results import make_result, split_record, write_result
from aggbench.verify import oracle_check

RUN_ID = "expaggkylevoi"
VIS = "full"
D_V = 12
DEFAULT_ENVS = ("E1", "E2", "E3-sybil", "E3-biased", "E3-collude")
QUERY_BUDGETS = (1, 2, 3, 4)


# -----------------------------------------------------------------------------
# Agent-side local inputs & E3-biased agent-side distortion
# -----------------------------------------------------------------------------

class AgentInputs:
    """Per-agent local inputs (obs_i, key_i) plus public constants.
    Passed through run_with_controls; encoders act strictly row-wise per agent."""

    def __init__(self, kind, obs, keys, public, env_name=""):
        self.kind = kind
        self.obs = obs
        self.keys = keys
        self.public = dict(public)
        self.env_name = env_name


def apply_e3_biased_agent_distortion(batch):
    """Apply E3-biased distortion on the biased agent's seen observations before encoding:
    lam_new = a * lam + b / max(1, seen.sum()), keeping ~seen as np.nan.
    Oracle fields (a, b, role) are never passed to AgentInputs or the aggregator."""
    obs = batch.obs.copy()
    a, b = batch.oracle["a"], batch.oracle["b"]
    mu = batch.oracle["mu"]
    for e, i in zip(*np.nonzero(batch.oracle["role"] == 2)):
        seen = np.isfinite(obs[e, i])
        n_seen = max(1, int(seen.sum()))
        lam = 2.0 * mu[e] * obs[e, i]
        lam_new = a[e, i] * lam + b[e, i] / n_seen
        obs[e, i] = np.where(seen, lam_new / (2.0 * mu[e]), np.nan)
    return obs


def make_agent_inputs(batch, env_name):
    obs, keys, public = batch.agent_inputs(VIS)
    if env_name == "E3-biased":
        obs = apply_e3_biased_agent_distortion(batch)
    return AgentInputs(batch.kind, obs, keys, public, env_name=env_name)


def clone_inputs_with_budget(inp, q_budget, episode_seed_offset=0):
    pub = dict(inp.public)
    pub["q_budget"] = int(q_budget)
    pub["episode_seed_offset"] = int(episode_seed_offset)
    return AgentInputs(inp.kind, inp.obs, inp.keys, pub, env_name=inp.env_name)


# -----------------------------------------------------------------------------
# Strictly local per-agent feature extraction (verified by check_locality)
# -----------------------------------------------------------------------------

def extract_local_slot_tensors(obs_np, keys_np, kind, device="cpu"):
    """Extract per-agent, per-slot (d_v = 12) features strictly from (obs_i, key_i).
    Never mixes across agents i != j."""
    obs = np.asarray(obs_np, dtype=np.float64)
    keys = np.asarray(keys_np, dtype=np.float64)
    if kind == "bernoulli":
        # obs: (n, N, 12), keys: (n, N, 24) = [cov (12), cov * mu (12)]
        K = obs.shape[-1]
        assert K == D_V, f"Expected K={D_V}, got {K}"
        cov = keys[..., :K]
        mu_cov = keys[..., K:2 * K]
        seen = np.isfinite(obs) & (cov > 0.5)
        seen_f = seen.astype(np.float32)
        x = np.where(seen, obs, 0.0).astype(np.float32)
        mu = mu_cov.astype(np.float32)
        llr = (2.0 * mu * x).astype(np.float32)

        # Agent-local summary statistics (over agent i's own slots only)
        n_seen = np.maximum(seen_f.sum(axis=-1, keepdims=True), 1.0)
        mean_llr = (llr.sum(axis=-1, keepdims=True) / n_seen) * seen_f
        mean_abs_x = ((np.abs(x) * seen_f).sum(axis=-1, keepdims=True) / n_seen) * seen_f

        # Val features per slot c in {0..11}: shape (n, N, 12, 6)
        val_feats = np.stack([
            llr,
            x,
            mu,
            llr * mu,
            mean_llr,
            mean_abs_x,
        ], axis=-1).astype(np.float32)

        # Key features per slot c in {0..11}: shape (n, N, 12, 3)
        key_feats = np.stack([
            seen_f,
            mu,
            np.sqrt(2.0) * mu,
        ], axis=-1).astype(np.float32)

        t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
        return {
            "val_feats": t(val_feats),
            "key_feats": t(key_feats),
            "mask": t(seen_f),
            "mu": t(mu),
            "x": t(x),
            "llr": t(llr),
        }
    else:
        # E1 Gaussian: obs is s_i in R^2, keys is [A_i (8), B_i (4), log_sig_i (1)] in R^13
        # Augmented latent z = [theta (4), eta (2)] in R^6; m = 2 rows => 2 x 6 = 12 slots (d_v = 12).
        n, N, m = obs.shape
        d, r = 4, 2
        A = keys[..., :m * d].reshape(n, N, m, d)
        Bf = keys[..., m * d:m * (d + r)].reshape(n, N, m, r)
        log_sig = keys[..., -1]  # (n, N)
        sig = np.exp(log_sig)[..., None, None]  # (n, N, 1, 1)

        C = np.concatenate([A, Bf], axis=-1)  # (n, N, 2, 6)
        C_white = C / np.maximum(sig, 1e-4)   # (n, N, 2, 6)
        s_exp = obs[..., :, None]             # (n, N, 2, 1)
        s_white = s_exp / np.maximum(sig, 1e-4)

        # Reshape 2 x 6 -> 12 slots
        Cw_12 = C_white.reshape(n, N, D_V)
        C_12 = C.reshape(n, N, D_V)
        sw_12 = np.repeat(s_white, 6, axis=-1).reshape(n, N, D_V)
        s_12 = np.repeat(s_exp, 6, axis=-1).reshape(n, N, D_V)
        active = (np.abs(C_12) > 1e-7).astype(np.float32)

        # Val features per slot (12, 4): [C_white * s_white, C * s, C_white, s_white]
        val_feats = np.stack([
            Cw_12 * sw_12,
            C_12 * s_12,
            Cw_12,
            sw_12,
        ], axis=-1).astype(np.float32)

        # Key features per slot (12, 3): [C_white, C, active]
        key_feats = np.stack([
            Cw_12,
            C_12,
            active,
        ], axis=-1).astype(np.float32)

        t = lambda a: torch.tensor(a, dtype=torch.float32, device=device)
        return {
            "val_feats": t(val_feats),
            "key_feats": t(key_feats),
            "mask": t(active),
            "Cw_12": t(Cw_12.astype(np.float32)),
            "Cws_12": t((Cw_12 * sw_12).astype(np.float32)),
        }


# -----------------------------------------------------------------------------
# Neural Modules: Local Encoder (d_v = 12), Trust Net, Search Net, Amp Net
# -----------------------------------------------------------------------------

class SmoothOddAct(nn.Module):
    """Unbounded odd activation h + 0.25 * tanh(h): never saturates large precision-weighted
    observations (e.g. sigma_i = 0.3 in E1 or strong LLRs in E2/E3) and preserves sign symmetry."""

    def forward(self, x):
        return x + 0.25 * torch.tanh(x)


class UnifiedKyleVoIModule(nn.Module):
    """Holds the 3 parameter groups:
      1. params_real:
         - theta_base: local slot-equivariant value & key encoders (d_v = 12) + readout
         - theta_trust: cross-agent shared-key disagreement & magnitude confiability network
      2. params_search:
         - Schur-complement active VoI key projector & prior reliability estimator
      3. params_amp:
         - Amortized distillation & key-context calibration head for budgeted queries
    All linear mappings on the value path use multi-element weight tensors with bias=False so
    scale-matched random_weights and untrained_init controls genuinely scramble alignment."""

    def __init__(self, kind, mode="pinned_lir", width=32):
        super().__init__()
        self.kind = kind
        self.mode = mode
        val_in_dim = 6 if kind == "bernoulli" else 4
        key_in_dim = 3

        # --- params_real: theta_base (local encoder + readout) ---
        self.val_enc = nn.Sequential(
            nn.Linear(val_in_dim, width, bias=False),
            SmoothOddAct(),
            nn.Linear(width, 1, bias=False),
        )
        self.key_enc = nn.Sequential(
            nn.Linear(key_in_dim, width, bias=False),
            SmoothOddAct(),
            nn.Linear(width, 1, bias=False),
        )

        if kind == "bernoulli":
            self.slot_readout = nn.Sequential(
                nn.Linear(2, width, bias=False),
                SmoothOddAct(),
                nn.Linear(width, 1, bias=False),
            )
            self.log_lambda = nn.Parameter(torch.full((D_V,), -6.0))
        else:
            # E1: 12 slots = 2 x 6; readout maps 6-dim cleared z -> 4-dim theta mean
            self.readout_mean = nn.Linear(6, 4, bias=False)
            self.log_lambda = nn.Parameter(torch.zeros(6))
            # Naive linear mode maps 12-dim pooled v directly to 4-dim mean and 4x4 Cholesky factor
            self.naive_mean = nn.Linear(D_V, 4, bias=False)
            self.naive_chol = nn.Linear(D_V, 10, bias=False)

        # --- params_real: theta_trust (reality-calibrated confiability + de-biasing) ---
        self.trust_net = nn.Sequential(
            nn.Linear(6, width, bias=False),
            nn.Tanh(),
            nn.Linear(width, width, bias=False),
            nn.Tanh(),
            nn.Linear(width, 3, bias=False),
        )

        # --- params_search: Active VoI Schur-complement selector ---
        self.search_key_net = nn.Sequential(
            nn.Linear(key_in_dim, width, bias=False),
            SmoothOddAct(),
            nn.Linear(width, 1, bias=False),
        )
        self.search_agent_net = nn.Sequential(
            nn.Linear(3, width, bias=False),
            nn.Tanh(),
            nn.Linear(width, 1, bias=False),
        )
        self.search_log_lambda = nn.Parameter(torch.full((D_V if kind == "bernoulli" else 6,), -6.0))

        # --- params_amp: Amortized distillation refinement head ---
        amp_in = D_V * 2 if kind == "bernoulli" else 12
        amp_out = D_V if kind == "bernoulli" else 4
        self.amp_net = nn.Sequential(
            nn.Linear(amp_in, width, bias=False),
            nn.Tanh(),
            nn.Linear(width, width, bias=False),
            nn.Tanh(),
            nn.Linear(width, amp_out, bias=False),
        )

    def params_base(self):
        out = list(self.val_enc.parameters()) + list(self.key_enc.parameters())
        if self.kind == "bernoulli":
            out += list(self.slot_readout.parameters())
        else:
            out += (list(self.readout_mean.parameters())
                    + list(self.naive_mean.parameters()) + list(self.naive_chol.parameters()))
        return out

    def params_trust_and_lambda(self):
        return list(self.trust_net.parameters()) + [self.log_lambda]

    def params_search_group(self):
        return (list(self.search_key_net.parameters())
                + list(self.search_agent_net.parameters())
                + [self.search_log_lambda])

    def params_amp_group(self):
        return list(self.amp_net.parameters())

    def encode_local(self, slot_dict):
        """Strictly local per-agent encoding: (obs_i, key_i) -> (v_i in R^12, k_i in R^12)."""
        vf = slot_dict["val_feats"]   # (n, N, 12, F_v)
        kf = slot_dict["key_feats"]   # (n, N, 12, F_k)
        mask = slot_dict["mask"]      # (n, N, 12)
        v_raw = 3.0 * self.val_enc(vf)[..., 0]
        k_raw = 2.0 * self.key_enc(kf)[..., 0]
        v = mask * v_raw
        if self.kind == "bernoulli":
            k = mask * (k_raw.abs() + 1e-4)
        else:
            # In E1, keys carry signed directional loadings in R^(2x6)
            k = mask * k_raw
        return v, k

    def compute_trust(self, v, k, slot_dict, query_mask=None):
        """Compute confiability beta_i in (0, 2), scale correction gamma_i, and shift correction delta_i
        from cross-agent disagreement on shared active keys and local magnitude diagnostics."""
        active = (slot_dict["mask"] > 0.5).float()
        if query_mask is not None:
            active = active * query_mask.unsqueeze(-1)

        if self.kind == "bernoulli":
            mu = slot_dict["mu"]
            denom_i = active.sum(dim=-1).clamp(min=1.0)
            norm_v_i = ((v.abs() / (2.0 * mu + 0.15)) * active).sum(dim=-1) / denom_i  # (n, N)
            mean_v_i = (v * active).sum(dim=-1) / denom_i                              # (n, N)

            if query_mask is not None:
                q_count = query_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
                ref_scale = (norm_v_i * query_mask).sum(dim=-1, keepdim=True) / q_count
            else:
                ref_scale = norm_v_i.median(dim=-1, keepdim=True).values

            excess_mag = F.relu(norm_v_i - ref_scale)  # (n, N)
            w0 = torch.exp(-2.5 * excess_mag).unsqueeze(-1) * active  # (n, N, 12)

            tot_w = w0.sum(dim=1, keepdim=True)                       # (n, 1, 12)
            tot_wv = (w0 * v).sum(dim=1, keepdim=True)                # (n, 1, 12)
            peer_w = (tot_w - w0).clamp(min=0.0)
            peer_v = (tot_wv - w0 * v) / (peer_w + 1e-4)

            has_peer = ((active.sum(dim=1, keepdim=True) - active) > 0.5).float() * active
            diff_shared = has_peer * (v - peer_v).abs()
            shared_count = has_peer.sum(dim=-1).clamp(min=1.0)
            mean_disagree = diff_shared.sum(dim=-1) / shared_count
            max_disagree = diff_shared.max(dim=-1).values

            trust_feats = torch.stack([
                mean_disagree,
                max_disagree,
                excess_mag,
                mean_v_i,
                mean_v_i.abs(),
                active.mean(dim=-1),
            ], dim=-1)  # (n, N, 6)
        else:
            denom_i = active.sum(dim=-1).clamp(min=1.0)
            norm_v_i = (v.abs() * active).sum(dim=-1) / denom_i
            ref_scale = norm_v_i.median(dim=-1, keepdim=True).values
            excess_mag = F.relu(norm_v_i - ref_scale)
            mean_v_i = (v * active).sum(dim=-1) / denom_i
            trust_feats = torch.stack([
                torch.zeros_like(norm_v_i),
                torch.zeros_like(norm_v_i),
                excess_mag,
                mean_v_i,
                mean_v_i.abs(),
                active.mean(dim=-1),
            ], dim=-1)

        raw_trust = self.trust_net(trust_feats)  # (n, N, 3)
        beta = 2.0 * torch.sigmoid(raw_trust[..., 0])
        gamma = 2.0 * torch.sigmoid(raw_trust[..., 1])
        delta = raw_trust[..., 2]
        if query_mask is not None:
            beta = beta * query_mask
        return beta, gamma, delta

    def readout_bernoulli(self, u):
        """Slot-equivariant readout over the 12 cleared slot values u: (..., 12) -> (...)."""
        u_pair = torch.stack([u, 0.25 * torch.tanh(u)], dim=-1)  # (..., 12, 2)
        return self.slot_readout(u_pair)[..., 0].sum(dim=-1)

    def clear_bernoulli(self, v, k, beta, gamma, delta, query_mask=None, use_amp=False, all_keys_mask=None):
        """Clear 12-dim messages via precision-weighted Key-Gram matrix:
        u_c = (sum_i beta_i k_{i,c}^2 v_tilde_{i,c}) / (lambda_c + sum_i beta_i k_{i,c}^2)."""
        active = (k > 1e-6).float()
        if query_mask is not None:
            v = v * query_mask.unsqueeze(-1)
            k = k * query_mask.unsqueeze(-1)
            beta = beta * query_mask
            active = active * query_mask.unsqueeze(-1)

        n_seen_i = active.sum(dim=-1, keepdim=True).clamp(min=1.0)
        v_tilde = active * gamma.unsqueeze(-1) * (v - (delta.unsqueeze(-1) / n_seen_i) * active)

        lam = 1e-4 + 1e-3 * F.softplus(self.log_lambda)  # (12,)
        bk2 = beta.unsqueeze(-1) * (k ** 2)              # (n, N, 12)
        num = (bk2 * v_tilde).sum(dim=1)                 # (n, 12)
        den = lam.unsqueeze(0) + bk2.sum(dim=1)          # (n, 12)
        u = num / den                                    # (n, 12)

        if use_amp and all_keys_mask is not None:
            queried_slot = (bk2.sum(dim=1) > 1e-6).float()  # (n, 12): 0 on unqueried slots!
            all_cov = all_keys_mask.mean(dim=1)             # (n, 12)
            amp_in = torch.cat([u, all_cov], dim=-1)        # (n, 24)
            u = u + 0.05 * queried_slot * (u * torch.tanh(self.amp_net(amp_in)))
        logit_inc = self.readout_bernoulli(u)
        return logit_inc, u

    def clear_gaussian(self, v, k, beta, query_mask=None, use_amp=False, all_keys_mask=None):
        """Clear 12-dim (2x6) messages in E1 via 6x6 Key-Gram matrix + Schur complement onto theta in R^4."""
        n, N, _ = v.shape
        if query_mask is not None:
            v = v * query_mask.unsqueeze(-1)
            k = k * query_mask.unsqueeze(-1)
            beta = beta * query_mask

        K_mat = k.view(n, N, 2, 6)                       # (n, N, 2, 6)
        V_mat = v.view(n, N, 2, 6)                       # (n, N, 2, 6)
        # Local information contribution per agent on z in R^6: sum over 2 obs rows of V_mat
        h_local = V_mat.sum(dim=2)                       # (n, N, 6)
        h_z = (beta.unsqueeze(-1) * h_local).sum(dim=1)  # (n, 6)

        # 6x6 Key-Gram matrix: Lambda_6 + sum_i beta_i K_i^T K_i
        lam6 = 0.3 + F.softplus(self.log_lambda)         # (6,)
        eye6 = torch.eye(6, device=v.device, dtype=v.dtype).unsqueeze(0)
        P_zz = eye6 * lam6.unsqueeze(0).unsqueeze(-1) + torch.einsum(
            "ni,nirc,nird->ncd", beta, K_mat, K_mat
        )  # (n, 6, 6)

        # Solve for posterior mean and covariance in augmented space R^6
        P_zz_d = P_zz.double() + 1e-4 * torch.eye(6, device=v.device, dtype=torch.float64).unsqueeze(0)
        Sigma_zz = torch.linalg.inv(P_zz_d)              # (n, 6, 6)
        z_hat = (Sigma_zz @ h_z.double().unsqueeze(-1))[..., 0].float()  # (n, 6)

        mean_theta = self.readout_mean(z_hat)            # (n, 4)
        if use_amp and all_keys_mask is not None:
            key_ctx = all_keys_mask.mean(dim=1)[:, :6]   # (n, 6)
            mean_theta = mean_theta + 0.08 * self.amp_net(torch.cat([z_hat, key_ctx], dim=-1))

        # Exact Schur complement precision on theta in R^4: (Sigma_zz[:, :4, :4])^{-1}
        Sigma_tt = Sigma_zz[:, :4, :4]
        P_theta = torch.linalg.inv(
            Sigma_tt + 1e-5 * torch.eye(4, device=v.device, dtype=torch.float64).unsqueeze(0)
        ).float()
        return mean_theta, P_theta, z_hat

    def select_schur_voi(self, slot_dict, q_budget, train=False, tau=0.5):
        """Sequentially select Q agents maximizing marginal Schur-complement log-det information gain."""
        kf = slot_dict["key_feats"]   # (n, N, 12, 3)
        mask = slot_dict["mask"]      # (n, N, 12)
        n, N, _ = mask.shape
        q_budget = min(int(q_budget), N)

        _, k_base = self.encode_local(slot_dict)
        k_raw = 0.25 * self.search_key_net(kf)[..., 0]
        if self.kind == "bernoulli":
            k_search = mask * ((k_base.detach() + k_raw).abs() + 1e-4)
        else:
            k_search = mask * (k_base.detach() + k_raw)

        # Slot-permutation-invariant agent key summary -> prior reliability beta_hat
        k_abs = k_search.abs()
        agent_key_summary = torch.stack([
            k_abs.mean(dim=-1),
            (k_search ** 2).mean(dim=-1),
            mask.mean(dim=-1),
        ], dim=-1)  # (n, N, 3)
        beta_hat = 0.98 + 0.04 * torch.sigmoid(self.search_agent_net(agent_key_summary)[..., 0])  # (n, N)

        # Normalized key vectors for directional rank-1 Schur novelty
        k_normed = k_search / (torch.norm(k_search, p=2, dim=-1, keepdim=True) + 1e-6)  # (n, N, 12)
        cos_sim_sq = (torch.einsum("nid,njd->nij", k_normed, k_normed)) ** 2             # (n, N, N)

        selected = torch.zeros(n, N, device=mask.device, dtype=mask.dtype)
        hard_mask = torch.zeros(n, N, device=mask.device, dtype=mask.dtype)

        if self.kind == "bernoulli":
            lam_dedup = 1e-4 + 1e-3 * F.softplus(self.search_log_lambda)  # (12,)
            G_diag = torch.zeros(n, D_V, device=mask.device, dtype=mask.dtype)  # (n, 12)
            k2 = beta_hat.unsqueeze(-1) * (k_search ** 2)                 # (n, N, 12)

            for t_step in range(q_budget):
                # Marginal Schur-complement gain: sum_c (beta_j k_{j,c}^2) / (1 + G_{t-1,c} / lam_dedup_c)
                denom = 1.0 + (G_diag / lam_dedup.unsqueeze(0)).unsqueeze(1)
                coord_gain = (k2 / denom).sum(dim=-1)                     # (n, N)
                if t_step > 0:
                    # Directional Schur novelty against already-queried keys: penalizes near-identical clones
                    max_cos2 = (cos_sim_sq * hard_mask.unsqueeze(1)).max(dim=-1).values
                    novelty = (1.0 - (max_cos2 ** 3)).clamp(min=1e-4)
                    gain = coord_gain * novelty
                else:
                    gain = coord_gain
                scores = gain.masked_fill(hard_mask > 0.5, -1e9)
                if train:
                    u = torch.rand_like(scores).clamp(1e-6, 1.0 - 1e-6)
                    gumbel = -torch.log(-torch.log(u))
                    soft = F.softmax((scores + 0.05 * gumbel) / tau, dim=-1)
                    idx = scores.argmax(dim=-1)
                    hard = F.one_hot(idx, N).to(scores.dtype)
                    step_sel = hard - soft.detach() + soft
                else:
                    idx = scores.argmax(dim=-1)
                    hard = F.one_hot(idx, N).to(scores.dtype)
                    step_sel = hard
                selected = selected + step_sel
                hard_mask = (hard_mask + hard).clamp(max=1.0)
                G_diag = G_diag + (step_sel.unsqueeze(-1) * k2).sum(dim=1)
            return selected
        else:
            # E1: 6x6 augmented precision matrix, Schur complement onto top-left 4x4 (theta)
            K_s = k_search.view(n, N, 2, 6)
            lam6 = 0.3 + F.softplus(self.search_log_lambda)
            eye6 = torch.eye(6, device=mask.device, dtype=mask.dtype).unsqueeze(0)
            P_run = eye6 * lam6.unsqueeze(0).unsqueeze(-1)  # (n, 6, 6)
            JK = beta_hat[:, :, None, None] * torch.einsum("nirc,nird->nicd", K_s, K_s)  # (n, N, 6, 6)

            for _ in range(q_budget):
                P_cand = P_run.unsqueeze(1) + JK
                P_tt = P_cand[:, :, :4, :4]
                P_te = P_cand[:, :, :4, 4:]
                P_ee = P_cand[:, :, 4:, 4:] + 1e-4 * torch.eye(2, device=mask.device).view(1, 1, 2, 2)
                P_schur = P_tt - P_te @ torch.linalg.solve(P_ee, P_te.transpose(-1, -2))
                _, logdet = torch.linalg.slogdet(
                    P_schur + 1e-4 * torch.eye(4, device=mask.device).view(1, 1, 4, 4)
                )
                scores = logdet.masked_fill(hard_mask > 0.5, -1e9)
                if train:
                    u = torch.rand_like(scores).clamp(1e-6, 1.0 - 1e-6)
                    gumbel = -torch.log(-torch.log(u))
                    soft = F.softmax((scores + 0.05 * gumbel) / tau, dim=-1)
                    idx = scores.argmax(dim=-1)
                    hard = F.one_hot(idx, N).to(scores.dtype)
                    step_sel = hard - soft.detach() + soft
                else:
                    idx = scores.argmax(dim=-1)
                    hard = F.one_hot(idx, N).to(scores.dtype)
                    step_sel = hard
                selected = selected + step_sel
                hard_mask = (hard_mask + hard).clamp(max=1.0)
                P_run = P_run + (step_sel[:, :, None, None] * JK).sum(dim=1)
            return selected


# -----------------------------------------------------------------------------
# Aggregator Wrapper for aggbench.controls.run_with_controls
# -----------------------------------------------------------------------------

_TRAINED_STATE_CACHE = {}


class KyleVoIAggregator:
    """Implements .module, .fit(inp, truth), .predict(inp) for aggbench.controls.run_with_controls."""

    def __init__(self, seed, kind, mode, q_budget=None, steps=500, batch_size=512, lr=5e-3, device="cpu"):
        torch.manual_seed(int(seed))
        self.seed = int(seed)
        self.kind = kind
        self.mode = mode
        self.q_budget = q_budget
        self.steps = steps
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.module = UnifiedKyleVoIModule(kind, mode=mode).to(device)
        self.fit_info = {}

    def _train_family(self):
        if self.mode in ("kyle_decentralized_naive_linear", "kyle_decentralized_pinned_lir",
                         "centralized_e2e_kv_clearing"):
            return self.mode
        return "three_param_voi_core"

    def _gaussian_nll(self, mean, prec, target):
        diff = (target - mean).unsqueeze(-1)
        quad = (diff.transpose(1, 2) @ prec @ diff)[:, 0, 0]
        _, logdet = torch.linalg.slogdet(prec.double())
        return (0.5 * (quad - logdet.float())).mean()

    def fit(self, inp, truth):
        cache_key = (inp.env_name, self._train_family(), self.seed, len(truth), self.steps)
        if cache_key in _TRAINED_STATE_CACHE:
            cached_state, cached_info = _TRAINED_STATE_CACHE[cache_key]
            self.module.load_state_dict(copy.deepcopy(cached_state))
            self.module.eval()
            self.fit_info = dict(cached_info, cached=True)
            return self

        t0 = time.time()
        slots = extract_local_slot_tensors(inp.obs, inp.keys, self.kind, device=self.device)
        y = torch.tensor(np.asarray(truth), dtype=torch.float32, device=self.device)
        l0 = float(inp.public.get("prior_logit", 0.0))
        n, N, _ = slots["mask"].shape

        opt_base = torch.optim.AdamW(self.module.params_base(), lr=self.lr, weight_decay=1e-4)
        opt_trust = torch.optim.AdamW(self.module.params_trust_and_lambda(), lr=self.lr, weight_decay=1e-4)
        opt_search = torch.optim.AdamW(self.module.params_search_group(), lr=self.lr, weight_decay=1e-4)
        opt_amp = torch.optim.AdamW(self.module.params_amp_group(), lr=self.lr, weight_decay=1e-4)

        gen = torch.Generator(device="cpu").manual_seed(1000 + self.seed)
        self.module.train()
        fam = self._train_family()

        for step in range(1, self.steps + 1):
            idx = torch.randint(0, n, (min(self.batch_size, n),), generator=gen).to(self.device)
            b_slots = {k: v[idx] for k, v in slots.items()}
            by = y[idx]

            if fam == "kyle_decentralized_naive_linear":
                # Decentralized naive simultaneous linear pooling: .detach() on all j != i
                v, k = self.module.encode_local(b_slots)
                v_det = v.detach()
                sum_v_det = v_det.sum(dim=1, keepdim=True)  # (B, 1, 12)
                u_i = (v + (sum_v_det - v_det)) / float(N)  # (B, N, 12)
                if self.kind == "bernoulli":
                    logits_i = l0 + self.module.readout_bernoulli(u_i)  # (B, N)
                    loss = F.binary_cross_entropy_with_logits(
                        logits_i, by.unsqueeze(-1).expand_as(logits_i)
                    )
                else:
                    mean_i = self.module.naive_mean(u_i)  # (B, N, 4)
                    chol_raw = self.module.naive_chol(u_i.mean(dim=1))  # (B, 10)
                    L = torch.zeros(len(idx), 4, 4, device=self.device)
                    ti = torch.tril_indices(4, 4)
                    L[:, ti[0], ti[1]] = chol_raw
                    prec = torch.eye(4, device=self.device).unsqueeze(0) + L @ L.transpose(1, 2)
                    loss = sum(self._gaussian_nll(mean_i[:, i], prec, by) for i in range(N)) / float(N)

                opt_base.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.module.params_base(), 2.0)
                opt_base.step()

            elif fam in ("kyle_decentralized_pinned_lir", "three_param_voi_core"):
                # 1. Encode local messages (v_i, k_i)
                v, k = self.module.encode_local(b_slots)
                v_det, k_det = v.detach(), k.detach()

                # 2. Train theta_trust (in params_real) on detached messages against tr.truth
                beta_live, gamma_live, delta_live = self.module.compute_trust(v_det, k_det, b_slots)
                if self.kind == "bernoulli":
                    logit_trust, _ = self.module.clear_bernoulli(v_det, k_det, beta_live, gamma_live, delta_live)
                    loss_trust = F.binary_cross_entropy_with_logits(l0 + logit_trust, by)
                else:
                    mean_t, prec_t, _ = self.module.clear_gaussian(v_det, k_det, beta_live)
                    loss_trust = self._gaussian_nll(mean_t, prec_t, by)

                opt_trust.zero_grad()
                loss_trust.backward()
                nn.utils.clip_grad_norm_(self.module.params_trust_and_lambda(), 2.0)
                opt_trust.step()

                # 3. Decentralized Kyle-LMSR Pinned Leave-One-Out update for theta_base:
                # Pin all other agents j != i (and beta, gamma, delta, lambda) with .detach()!
                with torch.no_grad():
                    beta_p, gamma_p, delta_p = self.module.compute_trust(v_det, k_det, b_slots)

                if self.kind == "bernoulli":
                    lam_p = (1e-4 + 1e-3 * F.softplus(self.module.log_lambda)).detach()  # (12,)
                    active_i = (b_slots["mask"] > 0.5).float()
                    n_seen_i = active_i.sum(dim=-1, keepdim=True).clamp(min=1.0)
                    v_tilde_live = active_i * gamma_p.unsqueeze(-1) * (
                        v - (delta_p.unsqueeze(-1) / n_seen_i) * active_i
                    )

                    bk2_live = beta_p.unsqueeze(-1) * (k ** 2)          # (B, N, 12)
                    num_live = bk2_live * v_tilde_live                  # (B, N, 12)

                    bk2_det = bk2_live.detach()
                    num_det = num_live.detach()

                    # Leave-one-out pinned pre-entry market state (G_{-i}, h_{-i}) has ZERO grad!
                    G_minus_i = lam_p.view(1, 1, D_V) + (bk2_det.sum(dim=1, keepdim=True) - bk2_det)
                    h_minus_i = num_det.sum(dim=1, keepdim=True) - num_det

                    # Post-entry cleared coordinate for agent i:
                    u_pinned_i = (h_minus_i + num_live) / (G_minus_i + bk2_live)  # (B, N, 12)
                    logits_i = l0 + self.module.readout_bernoulli(u_pinned_i)     # (B, N)
                    target_k = b_slots["key_feats"][..., 2] * active_i
                    loss_k_anchor = 0.05 * F.mse_loss(k, target_k)
                    loss_base = F.binary_cross_entropy_with_logits(
                        logits_i, by.unsqueeze(-1).expand_as(logits_i)
                    ) + loss_k_anchor
                else:
                    # E1 Gaussian leave-one-out pinned market clearing in R^6
                    K_mat = k.view(len(idx), N, 2, 6)
                    V_mat = v.view(len(idx), N, 2, 6)
                    h_loc = beta_p.unsqueeze(-1) * V_mat.sum(dim=2)               # (B, N, 6)
                    J_loc = beta_p[:, :, None, None] * torch.einsum(
                        "nirc,nird->nicd", K_mat, K_mat
                    )  # (B, N, 6, 6)

                    h_det = h_loc.detach()
                    J_det = J_loc.detach()
                    lam6_p = (0.3 + F.softplus(self.module.log_lambda)).detach()
                    eye6 = torch.eye(6, device=self.device).view(1, 1, 6, 6)
                    P_tot_det = eye6 * lam6_p.view(1, 1, 6, 1) + J_det.sum(dim=1, keepdim=True)
                    h_tot_det = h_det.sum(dim=1, keepdim=True)

                    losses_i = []
                    for i in range(N):
                        P_mi = P_tot_det[:, 0] - J_det[:, i]         # pinned pre-entry precision
                        h_mi = h_tot_det[:, 0] - h_det[:, i]         # pinned pre-entry information
                        P_i = P_mi + J_loc[:, i]
                        h_i = h_mi + h_loc[:, i]
                        P_i_d = P_i.double() + 1e-4 * torch.eye(6, device=self.device, dtype=torch.float64).unsqueeze(0)
                        Sigma_i = torch.linalg.inv(P_i_d)
                        z_i = (Sigma_i @ h_i.double().unsqueeze(-1))[..., 0].float()
                        mean_i = self.module.readout_mean(z_i)
                        Sigma_tt = Sigma_i[:, :4, :4]
                        P_theta_i = torch.linalg.inv(
                            Sigma_tt + 1e-5 * torch.eye(4, device=self.device, dtype=torch.float64).unsqueeze(0)
                        ).float()
                        losses_i.append(self._gaussian_nll(mean_i, P_theta_i, by))
                    loss_k_anchor = 0.05 * F.mse_loss(k, b_slots["Cw_12"])
                    loss_base = sum(losses_i) / float(N) + loss_k_anchor

                opt_base.zero_grad()
                loss_base.backward()
                nn.utils.clip_grad_norm_(self.module.params_base(), 2.0)
                opt_base.step()

                # 4. If training the 3-parameter core, also update params_search and params_amp
                if fam == "three_param_voi_core":
                    v_d, k_d = self.module.encode_local(b_slots)
                    v_d, k_d = v_d.detach(), k_d.detach()
                    with torch.no_grad():
                        beta_full, gamma_full, delta_full = self.module.compute_trust(v_d, k_d, b_slots)
                        if self.kind == "bernoulli":
                            _, u_full_target = self.module.clear_bernoulli(
                                v_d, k_d, beta_full, gamma_full, delta_full
                            )
                        else:
                            _, _, z_full_target = self.module.clear_gaussian(v_d, k_d, beta_full)

                    q_rand = int(torch.randint(1, min(5, N), (1,), generator=gen).item())
                    tau = max(0.25, 0.8 * (0.3 ** (step / float(self.steps))))
                    q_mask = self.module.select_schur_voi(b_slots, q_rand, train=True, tau=tau)

                    if self.kind == "bernoulli":
                        # Detach q_mask inside hard clearing denominator (lam=1e-4) to avoid 1/lam ST singularity;
                        # train params_search via smooth denominator (lam_smooth=0.25) + Fisher salience target.
                        logit_q, u_q = self.module.clear_bernoulli(
                            v_d, k_d, beta_full, gamma_full, delta_full,
                            query_mask=q_mask.detach(), use_amp=True, all_keys_mask=b_slots["mask"]
                        )
                        u_smooth = (
                            (q_mask.unsqueeze(-1) * (k_d ** 2) * v_d).sum(dim=1)
                            / (0.25 + (q_mask.unsqueeze(-1) * (k_d ** 2)).sum(dim=1))
                        )
                        logit_smooth = self.module.readout_bernoulli(u_smooth)
                        target_ks = b_slots["key_feats"][..., 2] * (b_slots["mask"] > 0.5).float()
                        k_s_now = b_slots["mask"] * (k_d + 0.25 * self.module.search_key_net(b_slots["key_feats"])[..., 0]).abs()
                        loss_s_a = (
                            F.binary_cross_entropy_with_logits(l0 + logit_q, by)
                            + 0.5 * F.binary_cross_entropy_with_logits(l0 + logit_smooth, by)
                            + 0.1 * F.mse_loss(u_q, u_full_target.detach())
                            + 1.0 * F.mse_loss(k_s_now, target_ks)
                        )
                    else:
                        mean_q, prec_q, z_q = self.module.clear_gaussian(
                            v_d, k_d, beta_full,
                            query_mask=q_mask, use_amp=True, all_keys_mask=b_slots["mask"]
                        )
                        k_s_now = b_slots["mask"] * (k_d + 0.25 * self.module.search_key_net(b_slots["key_feats"])[..., 0])
                        loss_s_a = (
                            self._gaussian_nll(mean_q, prec_q, by)
                            + 0.25 * F.mse_loss(z_q, z_full_target.detach())
                            + 0.5 * F.mse_loss(k_s_now, b_slots["Cw_12"])
                        )

                    opt_search.zero_grad()
                    opt_amp.zero_grad()
                    loss_s_a.backward()
                    nn.utils.clip_grad_norm_(self.module.params_search_group(), 2.0)
                    nn.utils.clip_grad_norm_(self.module.params_amp_group(), 2.0)
                    opt_search.step()
                    opt_amp.step()

            elif fam == "centralized_e2e_kv_clearing":
                # End-to-end joint backpropagation through all encoders and Key-Gram clearer
                v, k = self.module.encode_local(b_slots)
                beta, gamma, delta = self.module.compute_trust(v, k, b_slots)
                if self.kind == "bernoulli":
                    logit_inc, _ = self.module.clear_bernoulli(v, k, beta, gamma, delta)
                    active_i = (b_slots["mask"] > 0.5).float()
                    target_k = b_slots["key_feats"][..., 2] * active_i
                    loss = (
                        F.binary_cross_entropy_with_logits(l0 + logit_inc, by)
                        + 0.05 * F.mse_loss(k, target_k)
                    )
                else:
                    mean_th, prec_th, _ = self.module.clear_gaussian(v, k, beta)
                    loss = (
                        self._gaussian_nll(mean_th, prec_th, by)
                        + 0.05 * F.mse_loss(k, b_slots["Cw_12"])
                    )

                opt_base.zero_grad()
                opt_trust.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.module.parameters(), 2.0)
                opt_base.step()
                opt_trust.step()

        self.module.eval()
        self.fit_info = {"seconds": round(time.time() - t0, 2), "steps": self.steps, "cached": False}
        _TRAINED_STATE_CACHE[cache_key] = (copy.deepcopy(self.module.state_dict()), dict(self.fit_info))
        return self

    def _select_mask(self, slots, inp):
        """Return (n, N) binary query mask for Part B modes, or None for Part A full-N modes."""
        q = self.q_budget if self.q_budget is not None else inp.public.get("q_budget")
        if q is None:
            return None
        n, N, _ = slots["mask"].shape
        q = min(int(q), N)

        if self.mode == "voi_random_choice":
            rng = np.random.default_rng(20261020 + int(inp.public.get("episode_seed_offset", 0)))
            scores = rng.random((n, N))
            idx = np.argsort(-scores, axis=1)[:, :q]
            mask_np = np.zeros((n, N), dtype=np.float32)
            np.put_along_axis(mask_np, idx, 1.0, axis=1)
            return torch.tensor(mask_np, dtype=torch.float32, device=self.device)

        elif self.mode == "voi_greedy_salience_no_dedup":
            # Picks top-Q agents by standalone individual key norm ||k_i||_2 without penalizing overlap
            with torch.no_grad():
                _, k = self.module.encode_local(slots)
                norms = torch.norm(k, p=2, dim=-1)  # (n, N)
                tie_break = torch.arange(N, device=self.device, dtype=torch.float32).unsqueeze(0) * 1e-7
                idx = torch.topk(norms - tie_break, k=q, dim=-1).indices
                mask = torch.zeros(n, N, device=self.device, dtype=torch.float32)
                mask.scatter_(1, idx, 1.0)
                return mask

        elif self.mode == "voi_3param_schur_query":
            with torch.no_grad():
                return self.module.select_schur_voi(slots, q, train=False)

        return None

    def predict(self, inp):
        self.module.eval()
        slots = extract_local_slot_tensors(inp.obs, inp.keys, self.kind, device=self.device)
        l0 = float(inp.public.get("prior_logit", 0.0))
        n, N, _ = slots["mask"].shape

        with torch.no_grad():
            v, k = self.module.encode_local(slots)
            if self.mode == "kyle_decentralized_naive_linear":
                u = v.mean(dim=1)  # (n, 12)
                if self.kind == "bernoulli":
                    logit = l0 + self.module.readout_bernoulli(u).double().cpu().numpy()
                    return {"logit": logit}
                else:
                    mean = self.module.naive_mean(u).double().cpu().numpy()
                    chol_raw = self.module.naive_chol(u)
                    L = torch.zeros(n, 4, 4, device=self.device)
                    ti = torch.tril_indices(4, 4)
                    L[:, ti[0], ti[1]] = chol_raw
                    prec = (torch.eye(4, device=self.device).unsqueeze(0) + L @ L.transpose(1, 2)).double().cpu().numpy()
                    return {"mean": mean, "prec": prec}

            q_mask = self._select_mask(slots, inp)
            use_amp = (self.mode == "voi_3param_schur_query")
            beta, gamma, delta = self.module.compute_trust(v, k, slots, query_mask=q_mask)

            if self.kind == "bernoulli":
                logit_inc, _ = self.module.clear_bernoulli(
                    v, k, beta, gamma, delta,
                    query_mask=q_mask, use_amp=use_amp, all_keys_mask=slots["mask"]
                )
                return {"logit": l0 + logit_inc.double().cpu().numpy()}
            else:
                mean_th, prec_th, _ = self.module.clear_gaussian(
                    v, k, beta,
                    query_mask=q_mask, use_amp=use_amp, all_keys_mask=slots["mask"]
                )
                return {
                    "mean": mean_th.double().cpu().numpy(),
                    "prec": prec_th.double().cpu().numpy(),
                }

    def select_mask_numpy(self, inp):
        self.module.eval()
        slots = extract_local_slot_tensors(inp.obs, inp.keys, self.kind, device=self.device)
        with torch.no_grad():
            m = self._select_mask(slots, inp)
            return None if m is None else (m.cpu().numpy() > 0.5)

    def encode_numpy(self, obs, keys):
        """Return (n, N, 24) concatenation of [v_i (12), k_i (12)] for aggbench.messages.check_locality."""
        self.module.eval()
        slots = extract_local_slot_tensors(obs, keys, self.kind, device=self.device)
        with torch.no_grad():
            v, k = self.module.encode_local(slots)
            return torch.cat([v, k], dim=-1).double().cpu().numpy()


# -----------------------------------------------------------------------------
# Diagnostics: Duplicate key rate & unique piece coverage under budgeted VoI
# -----------------------------------------------------------------------------

def selection_diagnostics(mask_bool, keys_np, kind):
    """Compute duplicate-key selection rate (Sybil/colluder trap) and mean unique key coverage."""
    n, N = mask_bool.shape
    dup_episodes = 0
    cov_counts = []
    for e in range(n):
        S = np.flatnonzero(mask_bool[e])
        kk = np.round(keys_np[e, S], 6)
        if len(np.unique(kk, axis=0)) < len(S):
            dup_episodes += 1
        if kind == "bernoulli":
            cov_bits = keys_np[e, S, :D_V] > 0.5
            cov_counts.append(float(cov_bits.any(axis=0).sum()))
        else:
            A_sub = keys_np[e, S, :8].reshape(len(S), 2, 4)
            covered_dims = (np.abs(A_sub) > 1e-6).any(axis=(0, 1)).sum()
            cov_counts.append(float(covered_dims))
    return {
        "duplicate_key_episode_rate": round(float(dup_episodes) / max(1, n), 4),
        "mean_unique_covered_slots": round(float(np.mean(cov_counts)), 4),
    }


# -----------------------------------------------------------------------------
# Per-Environment Runner
# -----------------------------------------------------------------------------

def run_environment(env_name, args, log):
    t_env = time.time()
    env = make_env(env_name)
    tr = env.sample("train", args.n_train)
    dv = env.sample("dev", args.n_dev)
    log(f"[{env_name}] Sampled train={len(tr)}, dev={len(dv)} in {time.time() - t_env:.1f}s")

    tr_in = make_agent_inputs(tr, env_name)
    dv_in = make_agent_inputs(dv, env_name)

    # 1. Baselines on dev
    forecasts, alpha = all_baselines(tr, dv, VIS)
    learned_arms = []
    controls_map = {}
    locality_checks = {}
    paired_diffs = {}
    voi_diagnostics = {}

    # 2. Part A Arms (d_v = 12)
    part_a_modes = (
        "kyle_decentralized_naive_linear",
        "kyle_decentralized_pinned_lir",
        "centralized_e2e_kv_clearing",
    )
    for arm_name in part_a_modes:
        t0 = time.time()
        holder = {}

        def make_fn(s, m=arm_name):
            agg = KyleVoIAggregator(
                s, dv.kind, mode=m, q_budget=None,
                steps=args.steps, batch_size=args.batch_size, lr=args.lr, device="cpu"
            )
            holder.setdefault("trained", agg)
            return agg

        arms_out = run_with_controls(make_fn, tr_in, tr.truth, dv_in, seed=args.seed)
        trained_agg = holder["trained"]

        forecasts[arm_name] = arms_out["trained"]
        forecasts[f"{arm_name}:control:untrained_init"] = arms_out["control:untrained_init"]
        forecasts[f"{arm_name}:control:random_network"] = arms_out["control:random_network"]
        learned_arms.append(arm_name)
        controls_map[arm_name] = {
            "untrained_init": f"{arm_name}:control:untrained_init",
            "random_network": f"{arm_name}:control:random_network",
        }

        # Verify strict encoder locality with aggbench.messages.check_locality
        sl = slice(0, min(250, len(dv)))
        ok_loc = check_locality(trained_agg.encode_numpy, dv_in.obs[sl], dv_in.keys[sl], trials=5, seed=args.seed)
        locality_checks[arm_name] = {
            "passed": bool(ok_loc),
            "d_v": D_V,
            "episodes_checked": int(min(250, len(dv))),
            "trials": 5,
            "atol": 1e-9,
        }

        paired_diffs[arm_name] = {
            "minus_oracle_clearing": paired_difference(arms_out["trained"], forecasts["oracle_clearing"], dv, n_boot=args.n_boot),
            "minus_extremized_log_pool": paired_difference(arms_out["trained"], forecasts["extremized_log_pool"], dv, n_boot=args.n_boot),
        }
        reg = summarize(arms_out["trained"], dv, n_boot=200)["regret"]
        reg_rnd = summarize(arms_out["control:random_network"], dv, n_boot=200)["regret"]
        log(f"[{env_name}] Part A {arm_name:34s} regret={reg['mean']:.4f} "
            f"(rand_ctl={reg_rnd['mean']:.4f}) loc={ok_loc} ({time.time() - t0:.1f}s)")

    # Cross-arm paired difference in Part A: Pinned LIR vs Naive Linear and vs Centralized E2E
    paired_diffs["kyle_pinned_lir_minus_naive_linear"] = paired_difference(
        forecasts["kyle_decentralized_pinned_lir"],
        forecasts["kyle_decentralized_naive_linear"],
        dv, n_boot=args.n_boot
    )
    paired_diffs["kyle_pinned_lir_minus_centralized_e2e"] = paired_difference(
        forecasts["kyle_decentralized_pinned_lir"],
        forecasts["centralized_e2e_kv_clearing"],
        dv, n_boot=args.n_boot
    )

    # 3. Part B Arms: Active VoI Source Selection + 3-Parameter Split (Q in {1, 2, 3, 4})
    part_b_families = (
        "voi_random_choice",
        "voi_greedy_salience_no_dedup",
        "voi_3param_schur_query",
    )
    for q in QUERY_BUDGETS:
        voi_diagnostics[f"Q{q}"] = {}
        dv_q = clone_inputs_with_budget(dv_in, q, episode_seed_offset=q * 100)
        for fam in part_b_families:
            arm_name = f"{fam}_Q{q}"
            holder = {}

            def make_b(s, m=fam, qb=q):
                agg = KyleVoIAggregator(
                    s, dv.kind, mode=m, q_budget=qb,
                    steps=args.steps, batch_size=args.batch_size, lr=args.lr, device="cpu"
                )
                holder.setdefault("trained", agg)
                return agg

            arms_out = run_with_controls(make_b, tr_in, tr.truth, dv_q, seed=args.seed)
            trained_agg = holder["trained"]

            forecasts[arm_name] = arms_out["trained"]
            forecasts[f"{arm_name}:control:untrained_init"] = arms_out["control:untrained_init"]
            forecasts[f"{arm_name}:control:random_network"] = arms_out["control:random_network"]
            learned_arms.append(arm_name)
            controls_map[arm_name] = {
                "untrained_init": f"{arm_name}:control:untrained_init",
                "random_network": f"{arm_name}:control:random_network",
            }

            if q == 1 and fam == "voi_3param_schur_query":
                sl = slice(0, min(250, len(dv)))
                ok_loc = check_locality(trained_agg.encode_numpy, dv_in.obs[sl], dv_in.keys[sl], trials=5, seed=args.seed)
                locality_checks["voi_3param_schur_query_encoder"] = {
                    "passed": bool(ok_loc), "d_v": D_V, "episodes_checked": int(min(250, len(dv))), "trials": 5, "atol": 1e-9
                }

            mask_np = trained_agg.select_mask_numpy(dv_q)
            if mask_np is not None:
                voi_diagnostics[f"Q{q}"][fam] = selection_diagnostics(mask_np, dv_in.keys, dv.kind)

        paired_diffs[f"voi_3param_schur_minus_greedy_salience_Q{q}"] = paired_difference(
            forecasts[f"voi_3param_schur_query_Q{q}"],
            forecasts[f"voi_greedy_salience_no_dedup_Q{q}"],
            dv, n_boot=args.n_boot
        )
        paired_diffs[f"voi_3param_schur_minus_random_Q{q}"] = paired_difference(
            forecasts[f"voi_3param_schur_query_Q{q}"],
            forecasts[f"voi_random_choice_Q{q}"],
            dv, n_boot=args.n_boot
        )
        r_rand = summarize(forecasts[f"voi_random_choice_Q{q}"], dv, n_boot=200)["regret"]["mean"]
        r_sal = summarize(forecasts[f"voi_greedy_salience_no_dedup_Q{q}"], dv, n_boot=200)["regret"]["mean"]
        r_sch = summarize(forecasts[f"voi_3param_schur_query_Q{q}"], dv, n_boot=200)["regret"]["mean"]
        dup_sal = voi_diagnostics[f"Q{q}"]["voi_greedy_salience_no_dedup"]["duplicate_key_episode_rate"]
        dup_sch = voi_diagnostics[f"Q{q}"]["voi_3param_schur_query"]["duplicate_key_episode_rate"]
        log(f"[{env_name}] Part B Q={q}: random={r_rand:.4f} | greedy_salience={r_sal:.4f} (dup={dup_sal:.1%}) "
            f"| 3param_schur={r_sch:.4f} (dup={dup_sch:.1%})")

    # 4. Full bootstrap summaries & schema-validated write_result
    methods = {k: summarize(f, dv, n_boot=args.n_boot) for k, f in forecasts.items()}
    runner_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    extra = {
        "experiment_id": RUN_ID,
        "d_v": D_V,
        "extremize_alpha": float(alpha),
        "locality_check": locality_checks,
        "paired_differences": paired_diffs,
        "voi_selection_diagnostics": voi_diagnostics,
        "e3_biased_handling": (
            "Agent-side distortion on seen pieces before encoding: lam_new = a * lam + b / max(1, seen.sum()); "
            "~seen stays np.nan; oracle (a, b, role) never passed to aggregator."
        ) if env_name == "E3-biased" else None,
        "training": {
            "n_train": args.n_train,
            "n_dev": args.n_dev,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
            "torch_threads": args.threads,
            "labels": "tr.truth only (strictly no distillation from batch.bayes or batch.oracle)",
        },
        "runner": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": runner_sha,
        },
    }
    result = make_result(
        run_id=f"{RUN_ID}-{env_name}",
        track="SMOKE" if args.smoke else "EXPLORATORY",
        smoke=bool(args.smoke),
        env=env,
        visibility=VIS,
        splits={"train": split_record(tr, "train"), "eval": split_record(dv, "dev")},
        oracle_check=oracle_check(env),
        methods=methods,
        learned_arms=learned_arms,
        controls=controls_map,
        extra=extra,
    )
    out_path = write_result(args.out, result, name=f"result_{env_name}.json")
    log(f"[{env_name}] Wrote {out_path} in {time.time() - t_env:.1f}s")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=list(DEFAULT_ENVS))
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-dev", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=450)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "explore" / RUN_ID))
    args = ap.parse_args()

    if args.smoke:
        args.n_train, args.n_dev, args.steps, args.n_boot = 600, 200, 100, 100

    torch.set_num_threads(args.threads)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(out_dir / "run.log", "a")

    def log(msg):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    log(f"=== Starting {RUN_ID} (threads={args.threads}, smoke={args.smoke}) ===")
    all_results = {}
    for env_name in args.envs:
        res = run_environment(env_name, args, log)
        all_results[env_name] = res

    if not args.smoke:
        summary = {
            "run_id": RUN_ID,
            "track": "EXPLORATORY",
            "created_utc": all_results[args.envs[-1]]["created_utc"],
            "harness": all_results[args.envs[-1]]["harness"],
            "d_v": D_V,
            "environments": {
                env_name: {
                    "regret": {
                        m: all_results[env_name]["methods"][m]["regret"]
                        for m in all_results[env_name]["methods"]
                    },
                    "logloss": {
                        m: all_results[env_name]["methods"][m]["logloss"]
                        for m in all_results[env_name]["methods"]
                    },
                    "brier": {
                        m: all_results[env_name]["methods"][m]["brier"]
                        for m in all_results[env_name]["methods"]
                    },
                    "kl": {
                        m: all_results[env_name]["methods"][m]["kl"]
                        for m in all_results[env_name]["methods"]
                    },
                    "locality_check": all_results[env_name]["extra"]["locality_check"],
                    "paired_differences": all_results[env_name]["extra"]["paired_differences"],
                    "voi_selection_diagnostics": all_results[env_name]["extra"]["voi_selection_diagnostics"],
                }
                for env_name in args.envs
            },
        }
        summary_path = out_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        log(f"=== Wrote {summary_path} ===")


if __name__ == "__main__":
    main()
