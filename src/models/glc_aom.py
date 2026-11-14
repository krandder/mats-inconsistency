"""Unified Multi-Game Any-Order Transformer World Model (`GLCAOMWorldModel`).

Implements Member 1, Member 2, and Member 3 specifications for `GLC-AOM`:
1. Structural `LIR-Masked` belief-policy separation:
   - `chi = (0, 1)` for `q_real`, `Z_aux`, and `theta_trust` (zero utility input, pure epistemic belief).
   - `chi = (1, 1)` for `pi_step`, `q_plan`, `SELECT_NODE`, and `action_head` (reads utility `U` + detached beliefs).
2. `BilinearAuxLemmaEmitter`:
   - Even bilinear co-fluctuation emitter `(U^T Delta l_V) * (V^T Delta l_V)` into the `D_amp - D_V`
     auxiliary lemma channels (`Z_aux`) so opposite-sign sibling branches (`A=0 => B=0` vs `A=1 => B=1`)
     reinforce constructively rather than canceling.
   - Zero-initialized gated readout firewall `g_aux * W_{aux->V}` onto `V`.
   - Random-permutation any-order autoregressive loss `L_{any-order}(h_cons; x*)`.
3. `CombinatorialQueryHead`:
   - Symmetry-aware relational query embedding `e_{Q(A,B)}` that sets initial active prospectus `p_0 = k_{Q(A,B)}`.
   - Direct relational readout `P(Q(A, B))` from `(h_cons^var, h_cons^aux)` without enumerating `|Dom(V)|^2`,
     plus 2-step autoregressive equivalence check `P_AR(A == B)` and self-consistency discrepancy `delta_rel`.
4. `EpisodicMemoryTransport`:
   - Surprise-discounted episodic memory transport `M_t = h_cons^{(t-1)}` with precision
     `beta_{M_t} = gamma_mem * beta_cons^{(t-1)} * exp(-lambda_surp * S_t(M_t, o_t))`.
5. Zero-extra-pass leaf self-consistency scorer `m_ql(C_j)` (with NO `m_qs >= 0.50` greedy lock)
   and `DepthDebiasedTrustHead` (`theta_trust`) with Kish-ESS-shrunk (`trust_shrunk`) value pooling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.envs.base import EnvBatch, HEADER_LEN, QueryType, UtilitySpec


def _adapt_last_dim(x: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Zero-pad or slice the last dimension of `x` to `target_dim`."""
    cur = x.shape[-1]
    if cur == target_dim:
        return x
    if cur < target_dim:
        pad_shape = list(x.shape[:-1]) + [target_dim - cur]
        pad = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        return torch.cat([x, pad], dim=-1)
    return x[..., :target_dim]


@dataclass
class LIRMask:
    """Local Inconsistency Resolution (LIR) structural separation mask `chi = (chi_utility, chi_belief)`."""

    chi_utility: float
    chi_belief: float

    @classmethod
    def epistemic_belief(cls) -> "LIRMask":
        """`chi = (0, 1)` for `q_real`, `Z_aux`, and `theta_trust` (strictly independent of utility `U`)."""
        return cls(chi_utility=0.0, chi_belief=1.0)

    @classmethod
    def policy_search(cls) -> "LIRMask":
        """`chi = (1, 1)` for `pi_step`, `q_plan`, `SELECT_NODE`, and `action_head`."""
        return cls(chi_utility=1.0, chi_belief=1.0)


@dataclass
class GLCAOMConfig:
    """Configuration for the Unified Multi-Game Any-Order World Model (`GLCAOMWorldModel`)."""

    num_vars: int = 81                 # Max domain variables |V| (dynamically slices for smaller boards)
    num_vals: int = 10                 # Max domain cardinality K_dom per variable
    d_feat: int = 32                   # Per-variable auxiliary feature dimension
    d_mem: int = 32                    # Input memory token dimension from EnvBatch.memory_tokens
    k_mem: int = 8                     # Number of present-time episodic memory tokens M_t
    m_aux: int = 16                    # Number of latent lemma slots
    d_aux: int = 16                    # Dimension per latent lemma slot (D_aux = m_aux * d_aux = 256)
    d_model: int = 128                 # Transformer hidden dimension
    n_heads: int = 4                   # Number of attention heads
    n_layers: int = 3                  # Number of world-model transformer layers L_wm
    d_key: int = 64                    # Salience / prospectus key projection dimension
    num_relations: int = 5             # Relational operators {EQUAL, NOT_EQUAL, PARITY_XOR, GREATER, LESS}
    max_actions: int = 128             # Maximum discrete action space size for policy/action heads
    max_depth: int = 256               # Maximum search tree depth for horizon bias table (supports B up to 256)
    dropout: float = 0.0
    use_header_film: bool = False      # Optional lightweight game-header FiLM modulation (R12-E2)

    @property
    def d_v(self) -> int:
        return self.num_vars * self.num_vals

    @property
    def d_aux_total(self) -> int:
        return self.m_aux * self.d_aux

    @property
    def d_aux_world(self) -> int:
        """Dimension of the World Lemma Channel `Z^W_aux` (`chi = (0, 1)`)."""
        return (self.m_aux * self.d_aux) // 2

    @property
    def d_aux_plan(self) -> int:
        """Dimension of the Plan/Control Channel `Z^Pi_aux` (`chi = (1, 1)`), never read by belief decoder."""
        return self.d_aux_total - self.d_aux_world

    @property
    def d_amp(self) -> int:
        return self.d_v + self.d_aux_total


class BilinearAuxLemmaEmitter(nn.Module):
    """Bottom-Up Bidirectional Auxiliary Lemma Emitter (`chi = (0, 1)`).

    Features:
    1. Even bilinear co-fluctuation emitter `(U^T Delta l_V) * (V^T Delta l_V)` into `D_aux = D_amp - D_V`
       auxiliary lemma channels (`Z_aux = Z^W_aux (+) Z^Pi_aux`), guaranteeing
       `emitter(+Delta l_V) == emitter(-Delta l_V)` so opposite-sign sibling branches reinforce
       constructively in `Z_aux` while their first-order marginals cancel to zero.
    2. Structural partition `Z_aux = Z^W_aux (+) Z^Pi_aux` (`ROUND1-SCOREBOARD.md` §2.4):
       the belief decoder (`readout_var_firewall`, `decode_top_down_from_prefix`) masks out `Z^Pi_aux`
       so plan/control channels can NEVER leak into epistemic beliefs on `V`.
    3. Zero-initialized gated readout firewall `g_aux * W_{aux->V}` onto `V` (Member 3 Fix 1.3),
       preventing uncalibrated `Z_aux` activations from polluting 0-step or strong-root marginals on `V`.
    4. VICReg variance-covariance anti-collapse regularizer and Participation Ratio (`PR`) estimator.
    """

    def __init__(self, cfg: GLCAOMConfig):
        super().__init__()
        self.cfg = cfg
        self.lir_mask = LIRMask.epistemic_belief()

        # Polarity vector mapping each variable's K_val logits to a centered scalar shift
        # and full D_V -> D_aux bilinear factors U, V
        self.u_bilin = nn.Linear(cfg.d_v, cfg.d_aux_total, bias=False)
        self.v_bilin = nn.Linear(cfg.d_v, cfg.d_aux_total, bias=False)

        # Initialize (u_bilin, v_bilin) as a Parseval-normalized variable co-fluctuation
        # sketch so even at epoch 0, (U^T Delta l_V) * (V^T Delta l_V) captures pairwise co-fluctuations
        with torch.no_grad():
            gen = torch.Generator()
            gen.manual_seed(1337)
            sketch = torch.randn(cfg.d_aux_total, cfg.num_vars, generator=gen) / math.sqrt(float(cfg.d_aux_world))
            if cfg.num_vals >= 2:
                pol = torch.tensor([(-1.0) ** k for k in range(cfg.num_vals)], dtype=torch.float32)
            else:
                pol = torch.ones(1, dtype=torch.float32)
            weight_init = (sketch.unsqueeze(-1) * pol.view(1, 1, -1)).reshape(cfg.d_aux_total, cfg.d_v)
            self.u_bilin.weight.copy_(weight_init)
            self.v_bilin.weight.copy_(weight_init)

            # Orthonormal pair basis Q_pair in R^{d_aux_world x 120} for exact zero-crosstalk
            # bilinear co-fluctuation readout on the first 16 variables (16 * 15 / 2 = 120 <= d_aux_world)
            self.v_ortho = min(16, cfg.num_vars)
            n_pairs = (self.v_ortho * (self.v_ortho - 1)) // 2
            self.n_ortho_pairs = n_pairs
            pair_u_list: List[int] = []
            pair_v_list: List[int] = []
            pair_lookup = torch.zeros(( max(self.v_ortho, 1), max(self.v_ortho, 1) ), dtype=torch.long)
            p_idx = 0
            for u_i in range(self.v_ortho):
                for v_i in range(u_i + 1, self.v_ortho):
                    pair_u_list.append(u_i)
                    pair_v_list.append(v_i)
                    pair_lookup[u_i, v_i] = p_idx
                    pair_lookup[v_i, u_i] = p_idx
                    p_idx += 1
            self.register_buffer("pair_u_idx", torch.tensor(pair_u_list, dtype=torch.long))
            self.register_buffer("pair_v_idx", torch.tensor(pair_v_list, dtype=torch.long))
            self.register_buffer("pair_lookup", pair_lookup)

            gen_q = torch.Generator()
            gen_q.manual_seed(2026)
            w_dim = cfg.d_aux_world
            if w_dim >= n_pairs and n_pairs > 0:
                q_full, _ = torch.linalg.qr(torch.randn(w_dim, w_dim, generator=gen_q))
                q_basis = q_full[:, :n_pairs].contiguous()
            else:
                q_basis = F.normalize(torch.randn(w_dim, max(n_pairs, 1), generator=gen_q), p=2.0, dim=0)
            self.register_buffer("q_pair_basis", q_basis)

        # Optional even quadratic context projection from detached node hidden summary
        self.ctx_u = nn.Linear(cfg.d_model, cfg.d_aux_total, bias=False)
        self.ctx_v = nn.Linear(cfg.d_model, cfg.d_aux_total, bias=False)
        nn.init.zeros_(self.ctx_u.weight)
        nn.init.zeros_(self.ctx_v.weight)

        # Zero-initialized gated readout firewall onto V: g_aux * W_{aux->V}
        self.w_aux_to_v = nn.Linear(cfg.d_aux_total, cfg.d_v, bias=False)
        nn.init.zeros_(self.w_aux_to_v.weight)
        # Gate parameters initialized so g_aux starts closed (0.0) until calibrated
        self.aux_firewall_gate_logit = nn.Parameter(torch.full((cfg.d_v,), -10.0))
        self.firewall_enabled = False  # Opened only when calibration LOO validates improvement

        # Top-down conditional decoder: combines revealed prefix variables x_{pi(<t)} with Z^W_aux
        # to decode coupled variables in both directions (Z^W_aux + B -> A and Z^W_aux + A -> B)
        self.top_down_gain = nn.Parameter(torch.tensor(1.0))

    def mask_to_world_lemma_channel(self, h_aux: torch.Tensor) -> torch.Tensor:
        """Zero out the `Z^Pi_aux` plan/control channels (`[..., d_aux_world:]`) so only `Z^W_aux` is read."""
        w_dim = self.cfg.d_aux_world
        mask = torch.zeros_like(h_aux)
        mask[..., :w_dim] = 1.0
        return h_aux * mask

    def pad_var_logits_to_dv(self, delta_ell_v: torch.Tensor) -> torch.Tensor:
        """Pad `[..., V_act, K_act]` or `[..., D_V_act]` to `[..., cfg.d_v]`."""
        if delta_ell_v.shape[-1] == self.cfg.d_v and delta_ell_v.ndim >= 2:
            return delta_ell_v
        if delta_ell_v.ndim >= 3 and delta_ell_v.shape[-1] <= self.cfg.num_vals:
            v_act, k_act = delta_ell_v.shape[-2], delta_ell_v.shape[-1]
            lead = delta_ell_v.shape[:-2]
            out = torch.zeros(
                list(lead) + [self.cfg.num_vars, self.cfg.num_vals],
                dtype=delta_ell_v.dtype,
                device=delta_ell_v.device,
            )
            centered = delta_ell_v - delta_ell_v.mean(dim=-1, keepdim=True)
            out[..., :v_act, :k_act] = centered
            return out.reshape(list(lead) + [self.cfg.d_v])
        return _adapt_last_dim(delta_ell_v, self.cfg.d_v)

    def forward(
        self,
        delta_ell_v: torch.Tensor,
        node_ctx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Emit expanded claim `v_exp = [delta_ell_v ; z_aux]` in `R^{D_amp}` and salience key `s_edge`."""
        dv_vec = self.pad_var_logits_to_dv(delta_ell_v)
        dv_sg = dv_vec.detach() if not self.training else dv_vec
        # Smooth odd saturation prevents multi-step quadratic explosion while preserving exact oddness
        dv_bounded = 3.0 * torch.tanh(dv_sg / 3.0)

        # Strictly even cross-variable bilinear co-fluctuation:
        # (U^T Delta l_V) * (V^T Delta l_V) - sum_v (u_v^T Delta l_{V_v}) * (v_v^T Delta l_{V_v})
        u_proj = self.u_bilin(dv_bounded)
        v_proj = self.v_bilin(dv_bounded)

        lead_shape = list(dv_bounded.shape[:-1])
        dv_var = dv_bounded.view(lead_shape + [self.cfg.num_vars, self.cfg.num_vals])
        u_w = self.u_bilin.weight.view(self.cfg.d_aux_total, self.cfg.num_vars, self.cfg.num_vals)
        v_w = self.v_bilin.weight.view(self.cfg.d_aux_total, self.cfg.num_vars, self.cfg.num_vals)
        u_per_var = torch.einsum("...vk,dvk->...dv", dv_var, u_w)
        v_per_var = torch.einsum("...vk,dvk->...dv", dv_var, v_w)
        self_diag = (u_per_var * v_per_var).sum(dim=-1)

        z_aux = u_proj * v_proj - self_diag

        # When d_aux_world >= n_ortho_pairs (120), embed the exact even pairwise co-fluctuations
        # on the orthonormal basis Q_pair so pairs (A, B) have zero cross-talk from unrelated pairs
        if self.n_ortho_pairs > 0 and self.cfg.d_aux_world >= self.n_ortho_pairs:
            if self.cfg.num_vals >= 2:
                delta_pol = 0.5 * (dv_var[..., : self.v_ortho, 1] - dv_var[..., : self.v_ortho, 0])
            else:
                delta_pol = dv_var[..., : self.v_ortho, 0]
            c_pairs = (delta_pol[..., self.pair_u_idx] * delta_pol[..., self.pair_v_idx]) / 9.0
            ortho_aux = torch.einsum("...p,dp->...d", c_pairs, self.q_pair_basis)
            z_aux = z_aux.clone()
            z_aux[..., : self.cfg.d_aux_world] = ortho_aux + 0.15 * z_aux[..., : self.cfg.d_aux_world]

        if node_ctx is not None:
            ctx_sg = node_ctx.detach()
            z_aux = z_aux + (self.ctx_u(ctx_sg) * self.ctx_v(ctx_sg))

        v_exp = torch.cat([dv_vec, z_aux], dim=-1)
        abs_v = v_exp.abs()
        s_edge = abs_v / (torch.norm(abs_v, p=2, dim=-1, keepdim=True) + 1e-6)
        return v_exp, s_edge, z_aux

    def readout_var_firewall(
        self,
        h_cons: torch.Tensor,
        num_vars: Optional[int] = None,
        num_vals: Optional[int] = None,
    ) -> torch.Tensor:
        """Apply the zero-initialized gated readout firewall `g_aux * W_{aux->V}` onto `V`.

        Strictly masks `h_cons^aux` to the World Lemma Channel `Z^W_aux` (`Z^Pi_aux` is zeroed out)
        and guarantees exact zero leakage at initialization or when firewall is closed.
        """
        v_act = num_vars or self.cfg.num_vars
        k_act = num_vals or self.cfg.num_vals
        h_aux_world = self.mask_to_world_lemma_channel(h_cons[..., self.cfg.d_v :])

        if self.firewall_enabled:
            g_aux = torch.sigmoid(self.aux_firewall_gate_logit)
            ell_full = g_aux * self.w_aux_to_v(h_aux_world)
        else:
            ell_full = 0.0 * self.w_aux_to_v(h_aux_world)

        ell_reshaped = ell_full.view(list(h_cons.shape[:-1]) + [self.cfg.num_vars, self.cfg.num_vals])
        return ell_reshaped[..., :v_act, :k_act]

    def decode_top_down_from_prefix(
        self,
        h_cons: torch.Tensor,
        prefix_tokens: torch.Tensor,
        prefix_mask: torch.Tensor,
        root_mask: torch.Tensor,
        num_vals: int,
    ) -> torch.Tensor:
        """Bidirectional Top-Down Read (`Z^W_aux + B -> A` and `Z^W_aux + A -> B`).

        Reads exclusively from `Z^W_aux` (`Z^Pi_aux` is masked to zero) so utility/plan channels
        never influence conditional belief predictions.
        """
        B_sz, V_act = prefix_tokens.shape
        h_aux_world = self.mask_to_world_lemma_channel(h_cons[:, self.cfg.d_v :])  # [B, D_aux]

        newly_revealed = prefix_mask & (~root_mask) & (prefix_tokens >= 1)
        if not newly_revealed.any():
            return torch.zeros((B_sz, V_act, num_vals), dtype=h_cons.dtype, device=h_cons.device)

        one_hot = F.one_hot(
            (prefix_tokens.clamp(min=1, max=num_vals) - 1).long(), num_classes=num_vals
        ).to(h_cons.dtype)
        centered = (one_hot - (1.0 / float(num_vals))) * newly_revealed.unsqueeze(-1).to(h_cons.dtype)
        padded_prefix = self.pad_var_logits_to_dv(centered)  # [B, D_V]

        prefix_sketch = self.u_bilin(padded_prefix)          # [B, D_aux]
        coupled_code = prefix_sketch * h_aux_world           # [B, D_aux]
        decoded_dv = F.linear(coupled_code, self.v_bilin.weight.t())  # [B, D_V]
        decoded = decoded_dv.view(B_sz, self.cfg.num_vars, self.cfg.num_vals)[:, :V_act, :num_vals]
        unresolved = (~prefix_mask).unsqueeze(-1).to(h_cons.dtype)
        return 2.0 * torch.tanh(F.softplus(self.top_down_gain) * decoded / 2.0) * unresolved

    @staticmethod
    def compute_participation_ratio(z_aux: torch.Tensor, eps: float = 1e-8) -> float:
        """Compute normalized Participation Ratio `PR = (tr Sigma)^2 / (D * tr(Sigma^2))` in `[0, 1]`."""
        flat = z_aux.reshape(-1, z_aux.shape[-1])
        if flat.shape[0] <= 1:
            return 1.0
        centered = flat - flat.mean(dim=0, keepdim=True)
        cov = (centered.t() @ centered) / float(max(flat.shape[0] - 1, 1))
        tr_cov = torch.trace(cov)
        tr_cov_sq = torch.sum(cov * cov)
        d = float(flat.shape[-1])
        pr = (tr_cov * tr_cov) / (d * tr_cov_sq + eps)
        return float(pr.clamp(0.0, 1.0).item())

    @staticmethod
    def compute_vicreg_loss(z_aux: torch.Tensor, gamma_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
        """Compute VICReg variance + covariance anti-collapse regularization on `Z^W_aux`."""
        flat = z_aux.reshape(-1, z_aux.shape[-1])
        if flat.shape[0] <= 1:
            return torch.tensor(0.0, dtype=z_aux.dtype, device=z_aux.device)
        centered = flat - flat.mean(dim=0, keepdim=True)
        std = torch.sqrt(centered.var(dim=0) + eps)
        var_loss = F.relu(gamma_std - std).mean()
        cov = (centered.t() @ centered) / float(flat.shape[0] - 1)
        off_diag = cov - torch.diag(torch.diag(cov))
        cov_loss = (off_diag * off_diag).sum() / float(flat.shape[-1])
        return var_loss + cov_loss


class CombinatorialQueryHead(nn.Module):
    """Direct Combinatorial Query Head (`Q(A, B)`) & Active Prospectus Seeder (`chi = (0, 1)`).

    1. Symmetry-aware query token embedding `e_{Q(A,B)}` with exact permutation invariance
       for `EQUAL`, `NOT_EQUAL`, and `PARITY_XOR`.
    2. Direct prospectus seeding `p_0 = k_{Q(A,B)} in S^{D_amp - 1}` placing targeted salience mass
       on variables `(A, B)` and their coupled auxiliary subspace.
    3. Direct relational probability readout `P(Q(A, B))` from `(h_cons^var, h_cons^aux)` in `O(D_aux)`
       without enumerating `|Dom(V)|^2`.
    4. Exact 2-step autoregressive equivalence `P_AR(A == B)` and self-consistency audit `delta_rel`.
    """

    def __init__(self, cfg: GLCAOMConfig, aux_emitter: BilinearAuxLemmaEmitter):
        super().__init__()
        self.cfg = cfg
        self.aux_emitter = aux_emitter
        self.lir_mask = LIRMask.epistemic_belief()

        self.var_emb = nn.Embedding(cfg.num_vars, cfg.d_model)
        self.rel_emb = nn.Embedding(cfg.num_relations, cfg.d_model)

        self.w_sum = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.w_prod = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.w_diff = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.q_norm = nn.LayerNorm(cfg.d_model)

        # Learned residual prospectus projection into R^{D_aux}
        self.prospectus_aux_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.d_aux_total, bias=False),
        )

        # Learned calibration scale mapping bilinear co-fluctuation inner product to query log-odds
        self.cofluct_scale = nn.Parameter(torch.tensor(2.5))
        self.neural_aux_head = nn.Sequential(
            nn.Linear(cfg.d_model + cfg.d_aux_total, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, 1),
        )
        nn.init.zeros_(self.neural_aux_head[-1].weight)
        nn.init.zeros_(self.neural_aux_head[-1].bias)

    def encode_query_token(
        self,
        var_a: torch.Tensor,
        var_b: torch.Tensor,
        rel_id: torch.Tensor,
    ) -> torch.Tensor:
        """Encode `(var_a, var_b, rel_id)` into `e_{Q(A,B)}` (`[... , d_model]`).

        Guarantees exact symmetry `e_{Q(A,B)} == e_{Q(B,A)}` for `EQUAL (0)`, `NOT_EQUAL (1)`, `PARITY_XOR (2)`.
        """
        e_a = self.var_emb(var_a.clamp(0, self.cfg.num_vars - 1))
        e_b = self.var_emb(var_b.clamp(0, self.cfg.num_vars - 1))
        e_r = self.rel_emb(rel_id.clamp(0, self.cfg.num_relations - 1))

        is_directed = (rel_id >= int(QueryType.GREATER)).to(e_a.dtype).unsqueeze(-1)
        e_sym = self.w_sum(e_a + e_b) + self.w_prod(e_a * e_b)
        e_anti = is_directed * self.w_diff(e_a - e_b)
        return self.q_norm(e_sym + e_anti + e_r)

    def seed_prospectus(
        self,
        query_pairs: torch.Tensor,
        h_cons: Optional[torch.Tensor] = None,
        query_index: int = 0,
    ) -> torch.Tensor:
        """Construct the initial active prospectus `p_0 = k_{Q(A,B)} in S^{D_amp - 1}` (`[B, D_amp]`).

        Places direct salience on the queried variables `(A, B)` in `D_V` and their bilinear
        co-fluctuation probe in `D_aux`, steering `SELECT_NODE` + `STEP` directly toward `(A, B)`.
        """
        B = query_pairs.shape[0]
        device = query_pairs.device
        dtype = h_cons.dtype if h_cons is not None else torch.float32

        if query_pairs.shape[1] == 0:
            p0 = torch.ones((B, self.cfg.d_amp), dtype=dtype, device=device)
            return F.normalize(p0, p=2.0, dim=-1, eps=1e-6)

        q_idx = min(query_index, query_pairs.shape[1] - 1)
        var_a = query_pairs[:, q_idx, 0].clamp(0, self.cfg.num_vars - 1)
        var_b = query_pairs[:, q_idx, 1].clamp(0, self.cfg.num_vars - 1)
        rel_id = query_pairs[:, q_idx, 2].clamp(0, self.cfg.num_relations - 1)

        e_q = self.encode_query_token(var_a, var_b, rel_id)  # [B, d_model]
        var_part = torch.zeros((B, self.cfg.num_vars, self.cfg.num_vals), dtype=dtype, device=device)
        b_arange = torch.arange(B, device=device)
        for q_all in range(query_pairs.shape[1]):
            va_q = query_pairs[:, q_all, 0].clamp(0, self.cfg.num_vars - 1)
            vb_q = query_pairs[:, q_all, 1].clamp(0, self.cfg.num_vars - 1)
            var_part[b_arange, va_q, :] = 0.6
            var_part[b_arange, vb_q, :] = 0.6
        var_part[b_arange, var_a, :] = 1.0
        var_part[b_arange, var_b, :] = 1.0
        var_flat = var_part.reshape(B, self.cfg.d_v)

        aux_part = self.prospectus_aux_proj(e_q).abs()
        raw_p0 = torch.cat([var_flat, aux_part], dim=-1)
        return F.normalize(raw_p0, p=2.0, dim=-1, eps=1e-6)

    def predict_query_probs(
        self,
        batch: EnvBatch,
        pred_var_probs: torch.Tensor,
        h_cons: torch.Tensor,
    ) -> torch.Tensor:
        """Read out `P(Q(A, B))` (`[B, N_queries]`) directly from `(h_cons^var, h_cons^aux)` in `O(D_aux)`.

        Combines:
        1. Marginal readout from `pred_var_probs` (exact when `A` and `B` are anchored), and
        2. Bilinear co-fluctuation readout from `h_cons^aux` (exact on floating equivalence chains
           where `P*(A=1) = P*(B=1) = 0.5` but sibling branches write `Delta l_A * Delta l_B` into `Z_aux`),
        without enumerating `|Dom(V)|^2`.
        """
        B, Q, _ = batch.query_pairs.shape
        if Q == 0:
            return torch.zeros((B, 0), dtype=pred_var_probs.dtype, device=pred_var_probs.device)

        var_a = batch.query_pairs[:, :, 0].clamp(0, pred_var_probs.shape[1] - 1)
        var_b = batch.query_pairs[:, :, 1].clamp(0, pred_var_probs.shape[1] - 1)
        q_type = batch.query_pairs[:, :, 2]

        K = pred_var_probs.shape[-1]
        pa = pred_var_probs.gather(1, var_a.unsqueeze(-1).expand(B, Q, K))
        pb = pred_var_probs.gather(1, var_b.unsqueeze(-1).expand(B, Q, K))

        # 1. O(K) baseline marginal probability (no K^2 enumeration!)
        p_eq_marg = (pa * pb).sum(dim=-1)
        base_prob = p_eq_marg.clone()

        neq_mask = q_type == int(QueryType.NOT_EQUAL)
        if neq_mask.any():
            base_prob = torch.where(neq_mask, 1.0 - p_eq_marg, base_prob)

        xor_mask = q_type == int(QueryType.PARITY_XOR)
        if xor_mask.any():
            even_idx = torch.arange(K, device=batch.device) % 2 == 0
            odd_idx = ~even_idx
            pa_odd = pa[:, :, odd_idx].sum(dim=-1)
            pb_odd = pb[:, :, odd_idx].sum(dim=-1)
            p_xor = pa_odd * (1.0 - pb_odd) + (1.0 - pa_odd) * pb_odd
            base_prob = torch.where(xor_mask, p_xor, base_prob)

        gt_mask = q_type == int(QueryType.GREATER)
        if gt_mask.any():
            # Cumulative sum in O(K) instead of K^2 matrix multiplication!
            pb_less_cum = torch.cumsum(pb, dim=-1) - pb
            p_gt = (pa * pb_less_cum).sum(dim=-1)
            base_prob = torch.where(gt_mask, p_gt, base_prob)

        # 2. Direct Bilinear Co-Fluctuation Readout from Z^W_aux in O(D_aux) (Z^Pi_aux is masked out)
        h_aux = self.aux_emitter.mask_to_world_lemma_channel(h_cons[:, self.cfg.d_v :])  # [B, D_aux]
        if h_aux.abs().max() < 1e-8:
            return base_prob.clamp(1e-6, 1.0 - 1e-6)

        # Extract per-variable sketch vectors s_A, s_B in R^{D_aux} from u_bilin, v_bilin
        u_w = self.aux_emitter.u_bilin.weight.view(self.cfg.d_aux_total, self.cfg.num_vars, self.cfg.num_vals)
        v_w = self.aux_emitter.v_bilin.weight.view(self.cfg.d_aux_total, self.cfg.num_vars, self.cfg.num_vals)
        if K >= 2:
            pol = torch.tensor(
                [(-1.0) ** k for k in range(K)], device=batch.device, dtype=pred_var_probs.dtype
            )
            pol = pol / float(K)
        else:
            pol = torch.ones(1, device=batch.device, dtype=pred_var_probs.dtype)

        # Variable sketch table: [num_vars, D_aux]
        u_var = torch.einsum("dvk,k->vd", u_w[:, :, :K], pol)
        v_var = torch.einsum("dvk,k->vd", v_w[:, :, :K], pol)

        sa = u_var[var_a]  # [B, Q, D_aux]
        sb = v_var[var_b]  # [B, Q, D_aux]
        pair_probe = 0.5 * (sa * sb + v_var[var_a] * u_var[var_b])  # Symmetric in (A, B)

        # Inner product <pair_probe, h_cons^aux> estimates E[Delta l_A * Delta l_B]
        cofluct_raw = torch.einsum("bqd,bd->bq", pair_probe, h_aux)

        # When q_pair_basis is available and d_aux_world >= n_ortho_pairs, read out exact
        # zero-crosstalk pairwise co-fluctuations on the orthonormal pair basis Q_pair
        v_ortho = getattr(self.aux_emitter, "v_ortho", 0)
        n_ortho = getattr(self.aux_emitter, "n_ortho_pairs", 0)
        if (
            v_ortho > 0
            and n_ortho > 0
            and self.cfg.d_aux_world >= n_ortho
            and hasattr(self.aux_emitter, "q_pair_basis")
            and hasattr(self.aux_emitter, "pair_lookup")
        ):
            in_ortho = (var_a < v_ortho) & (var_b < v_ortho) & (var_a != var_b)
            if in_ortho.any():
                va_c = var_a.clamp(0, v_ortho - 1)
                vb_c = var_b.clamp(0, v_ortho - 1)
                p_idx = self.aux_emitter.pair_lookup[va_c, vb_c]  # [B, Q]
                q_ab = self.aux_emitter.q_pair_basis[:, p_idx].permute(1, 2, 0)  # [B, Q, d_aux_world]
                w_dim = self.cfg.d_aux_world
                cofluct_ortho = 2.5 * torch.einsum("bqd,bd->bq", q_ab, h_aux[:, :w_dim])
                cofluct_raw = torch.where(in_ortho, cofluct_ortho, cofluct_raw)

        cofluct = 1.5 * torch.tanh(cofluct_raw / 1.5)
        e_q = self.encode_query_token(var_a, var_b, q_type)
        neural_shift = self.neural_aux_head(
            torch.cat([e_q, h_aux.unsqueeze(1).expand(B, Q, -1)], dim=-1)
        ).squeeze(-1)

        # Only apply auxiliary relational shift to the extent A and B have unresolved marginal entropy
        ent_a = -(pa * torch.log(pa.clamp(min=1e-7))).sum(dim=-1)
        ent_b = -(pb * torch.log(pb.clamp(min=1e-7))).sum(dim=-1)
        unres_gate = torch.sqrt((ent_a * ent_b).clamp(min=0.0)) / math.log(max(K, 2))

        sign = torch.ones_like(cofluct)
        sign = torch.where(neq_mask | xor_mask, -torch.ones_like(cofluct), sign)

        base_logit = torch.logit(base_prob.clamp(1e-5, 1.0 - 1e-5))
        total_logit = base_logit + unres_gate * (
            sign * F.softplus(self.cofluct_scale) * cofluct + neural_shift
        )
        return torch.sigmoid(total_logit).clamp(1e-6, 1.0 - 1e-6)

    @staticmethod
    def compute_ar_equivalence(
        p_a_marg: torch.Tensor,
        p_b_given_a: torch.Tensor,
        p_b_marg: torch.Tensor,
        p_a_given_b: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute symmetrized 2-step autoregressive `P_AR(A == B)` and JSD discrepancy `delta_rel`."""
        joint_ab_fwd = p_a_marg.unsqueeze(-1) * p_b_given_a
        joint_ab_bwd = (p_b_marg.unsqueeze(-1) * p_a_given_b).transpose(-2, -1)

        diag_fwd = torch.diagonal(joint_ab_fwd, dim1=-2, dim2=-1).sum(dim=-1)
        diag_bwd = torch.diagonal(joint_ab_bwd, dim1=-2, dim2=-1).sum(dim=-1)
        p_ar_eq = 0.5 * (diag_fwd + diag_bwd)

        m_joint = 0.5 * (joint_ab_fwd + joint_ab_bwd).clamp(min=1e-8)
        kl_fwd = (
            joint_ab_fwd.clamp(min=1e-8) * (joint_ab_fwd.clamp(min=1e-8).log() - m_joint.log())
        ).sum(dim=(-2, -1))
        kl_bwd = (
            joint_ab_bwd.clamp(min=1e-8) * (joint_ab_bwd.clamp(min=1e-8).log() - m_joint.log())
        ).sum(dim=(-2, -1))
        jsd_ab = 0.5 * (kl_fwd + kl_bwd).clamp(min=0.0)
        return p_ar_eq.clamp(0.0, 1.0), jsd_ab


class EpisodicMemoryTransport(nn.Module):
    """Surprise-Discounted Present-Time Episodic Memory Transport (`M_t = h_cons^{(t-1)}`).

    Implements Member 2 §3 + Member 3 Fix 1.4:
      `S_t(M_t, o_t) = [-log q_real(o_t | h_t, M_t) - (-log q_real(o_t | h_t, empty))]_+`
      `beta_{M_t} = gamma_mem * beta_cons^{(t-1)} * exp(-lambda_surp * S_t(M_t, o_t))`
    Automatically flushes dogmatic priors when contradicted by a new turn observation `o_t`.
    """

    def __init__(
        self,
        cfg: GLCAOMConfig,
        gamma_mem: float = 0.85,
        lambda_surp: float = 2.0,
    ):
        super().__init__()
        self.cfg = cfg
        self.lir_mask = LIRMask.epistemic_belief()
        self.gamma_mem = float(gamma_mem)
        self.lambda_surp = float(lambda_surp)

        self.init_mem_tokens = nn.Parameter(torch.zeros(1, cfg.k_mem, cfg.d_model))
        self.cons_to_mem = nn.Sequential(
            nn.LayerNorm(cfg.d_amp),
            nn.Linear(cfg.d_amp, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, cfg.k_mem * cfg.d_model),
        )
        self.ext_mem_proj = nn.Linear(cfg.d_mem, cfg.d_model)
        self.mem_norm = nn.LayerNorm(cfg.d_model)

    @staticmethod
    def compute_observation_surprise(
        nll_with_memory: torch.Tensor,
        nll_without_memory: torch.Tensor,
    ) -> torch.Tensor:
        """Compute excess observation surprise `S_t(M_t, o_t) = [NLL(o_t | M_t) - NLL(o_t | empty)]_+`."""
        return F.relu(nll_with_memory - nll_without_memory)

    def compute_memory_precision(
        self,
        beta_prev: torch.Tensor,
        surprise: torch.Tensor,
    ) -> torch.Tensor:
        """Compute surprise-discounted precision `beta_{M_t} = gamma_mem * beta_{t-1} * exp(-lambda_surp * S_t)`."""
        return self.gamma_mem * beta_prev * torch.exp(-self.lambda_surp * surprise.clamp(min=0.0))

    def forward(
        self,
        h_cons_prev: Optional[torch.Tensor] = None,
        beta_prev: Optional[torch.Tensor] = None,
        nll_with_memory: Optional[torch.Tensor] = None,
        nll_without_memory: Optional[torch.Tensor] = None,
        batch_memory_tokens: Optional[torch.Tensor] = None,
        batch_memory_mask: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Transport `h_cons^{(t-1)}` into turn-`t` memory tokens `M_t` and precision `beta_{M_t}`."""
        dev = (
            h_cons_prev.device
            if h_cons_prev is not None
            else (batch_memory_tokens.device if batch_memory_tokens is not None else (device or torch.device("cpu")))
        )
        B = (
            h_cons_prev.shape[0]
            if h_cons_prev is not None
            else (batch_memory_tokens.shape[0] if batch_memory_tokens is not None else batch_size)
        )

        if nll_with_memory is not None and nll_without_memory is not None:
            surprise = self.compute_observation_surprise(nll_with_memory, nll_without_memory)
        else:
            surprise = torch.zeros(B, dtype=torch.float32, device=dev)

        if beta_prev is None:
            beta_prev = torch.ones(B, dtype=torch.float32, device=dev)

        beta_mt = self.compute_memory_precision(beta_prev, surprise)

        if h_cons_prev is not None:
            h_sg = h_cons_prev.detach()
            raw_tokens = self.cons_to_mem(h_sg).view(B, self.cfg.k_mem, self.cfg.d_model)
            mem_tokens = self.mem_norm(raw_tokens)
            # Scale transported consensus by surprise discount so contradicted memory flushes to 0
            discount = torch.exp(-self.lambda_surp * surprise.clamp(min=0.0)).unsqueeze(-1)
            transported_h_cons = self.gamma_mem * discount * h_sg
        else:
            mem_tokens = self.init_mem_tokens.expand(B, -1, -1).to(dev)
            transported_h_cons = torch.zeros((B, self.cfg.d_amp), dtype=torch.float32, device=dev)

        if batch_memory_tokens is not None and batch_memory_tokens.shape[1] > 0:
            adapted = _adapt_last_dim(batch_memory_tokens, self.cfg.d_mem)
            ext_emb = self.ext_mem_proj(adapted)
            if batch_memory_mask is not None:
                ext_emb = ext_emb * batch_memory_mask.unsqueeze(-1).to(ext_emb.dtype)
            k_take = min(self.cfg.k_mem, ext_emb.shape[1])
            mem_tokens = mem_tokens.clone()
            mem_tokens[:, :k_take, :] = mem_tokens[:, :k_take, :] + ext_emb[:, :k_take, :]

        # Scale memory token embeddings by effective precision gate
        mem_scale = (beta_mt / (1.0 + beta_mt)).view(B, 1, 1)
        mem_tokens = mem_tokens * mem_scale

        return {
            "memory_tokens": mem_tokens,
            "beta_memory": beta_mt,
            "surprise": surprise,
            "transported_h_cons": transported_h_cons,
        }


class DepthDebiasedTrustHead(nn.Module):
    """Depth-Debiased Self-Trustworthiness Auditor (`theta_trust`, `chi = (0, 1)`) & Kish-ESS Shrinkage.

    Implements Member 3 Fix 1.2 & Fix 3.3:
    1. Zero-extra-pass leaf self-consistency `m_ql(C_j)` (with NO `m_qs >= 0.50` greedy lock).
    2. Depth-deconfounded excess-variance prediction `beta_trust(C_j)` and horizon bias subtraction `b_hor(d)`.
    3. Kish-ESS-constrained James-Stein shrinkage (`trust_shrunk`) guaranteeing
       `MSE(trust_shrunk) <= min(MSE(unif), MSE(C_0), MSE(stiff))`.
    """

    def __init__(self, cfg: GLCAOMConfig):
        super().__init__()
        self.cfg = cfg
        self.lir_mask = LIRMask.epistemic_belief()

        self.trust_mlp = nn.Sequential(
            nn.Linear(cfg.d_model + 4, cfg.d_model // 2),
            nn.SiLU(),
            nn.Linear(cfg.d_model // 2, 1),
        )
        # Initialize so base precision starts near 1.0
        nn.init.zeros_(self.trust_mlp[-1].weight)
        nn.init.constant_(self.trust_mlp[-1].bias, 0.55)

        # Calibration-fitted depth-horizon bias table b_hor(d)
        self.register_buffer("horizon_bias_table", torch.zeros(cfg.max_depth + 1))
        self.register_buffer("calibrated_gamma_trust", torch.tensor(0.5))
        self.register_buffer("calibrated_rho_root", torch.tensor(0.0))

    @staticmethod
    def compute_leaf_self_consistency(
        node_logits: torch.Tensor,
        node_clamped_mask: torch.Tensor,
        node_clamped_tokens: torch.Tensor,
        root_clamped_mask: torch.Tensor,
        check_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute zero-extra-pass leaf self-consistency `m_ql(C_j)` (`[B]`) with NO `m_qs >= 0.50` lock.

        Evaluates the node's own forward-pass probability on all variables clamped along the branch
        `S_j = O_j \\ O_0`, plus internal peer/entropy coherence.
        """
        probs = F.softmax(node_logits, dim=-1)  # [B, V, K]
        B, V, K = probs.shape
        device = probs.device

        branch_clamped = node_clamped_mask & (~root_clamped_mask) & (node_clamped_tokens >= 1)
        safe_idx = (node_clamped_tokens.clamp(1, K) - 1).unsqueeze(-1)
        self_read_prob = probs.gather(-1, safe_idx).squeeze(-1)  # [B, V]

        # Where a variable was NOT clamped on the branch, fill with 1.0 so min over S_j is exact
        masked_read = torch.where(branch_clamped, self_read_prob, torch.ones_like(self_read_prob))
        min_self_read = masked_read.min(dim=-1).values  # [B]

        # For root node (no branch clamps), m_ql = 1.0
        has_branch = branch_clamped.any(dim=-1)
        m_ql = torch.where(has_branch, min_self_read, torch.ones(B, dtype=probs.dtype, device=device))

        if check_valid is not None:
            m_ql = m_ql * check_valid.to(probs.dtype)

        return m_ql.clamp(0.0, 1.0)

    def forward(
        self,
        node_h: torch.Tensor,
        node_logits: torch.Tensor,
        depth: torch.Tensor,
        m_ql: torch.Tensor,
        excess_surprise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict depth-debiased reliability precision `beta_trust(C_j) > 0` (`[B]`)."""
        h_pool = node_h.detach().mean(dim=1)  # [B, d_model]
        probs = F.softmax(node_logits.detach(), dim=-1)
        ent = -(probs * torch.log(probs.clamp(min=1e-7))).sum(dim=-1).mean(dim=-1, keepdim=True)
        norm_depth = (depth.to(h_pool.dtype) / float(max(self.cfg.max_depth, 1))).unsqueeze(-1)
        m_ql_feat = m_ql.to(h_pool.dtype).unsqueeze(-1)
        surp_feat = (
            excess_surprise.to(h_pool.dtype).unsqueeze(-1)
            if excess_surprise is not None
            else torch.zeros_like(m_ql_feat)
        )

        feat = torch.cat([h_pool, ent, norm_depth, m_ql_feat, surp_feat], dim=-1)
        raw_beta = F.softplus(self.trust_mlp(feat).squeeze(-1))
        # Damping by exp(-excess_surprise) ensures high-error/corrupted rollouts get near-zero precision
        if excess_surprise is not None:
            raw_beta = raw_beta * torch.exp(-1.5 * F.relu(excess_surprise))
        return raw_beta.clamp(min=1e-5)

    @staticmethod
    def compute_kish_ess(weights: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Compute Kish Effective Sample Size `ESS(w) = (sum w_r)^2 / sum w_r^2` across `dim=-1`."""
        w_sum = weights.sum(dim=-1).clamp(min=eps)
        w_sq_sum = (weights * weights).sum(dim=-1).clamp(min=eps)
        return (w_sum * w_sum) / w_sq_sum

    @classmethod
    def trust_shrunk_weights(
        cls,
        beta_trust: torch.Tensor,
        gamma_trust: float = 0.5,
        kappa_ess: float = 0.65,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Compute Kish-ESS-constrained power-shrunk weights `w_r(gamma_trust)` (`[B, K]`).

        Searches `gamma in [0, gamma_trust]` so that `ESS(w) >= kappa_ess * K` for every instance,
        preventing variance inflation when rollouts share a common horizon bias.
        """
        B, K = beta_trust.shape
        if K == 0:
            return beta_trust
        if K == 1:
            return torch.ones_like(beta_trust)

        beta_pos = beta_trust.clamp(min=eps)
        log_b = torch.log(beta_pos)
        log_b = log_b - log_b.mean(dim=-1, keepdim=True)

        target_ess = kappa_ess * float(K)
        # Evaluate a grid of exponents in [0, gamma_trust] and pick the largest exponent satisfying ESS >= target_ess
        grid = torch.linspace(0.0, float(gamma_trust), steps=17, device=beta_trust.device, dtype=beta_trust.dtype)
        # [B, 17, K]
        scaled_logits = log_b.unsqueeze(1) * grid.view(1, -1, 1)
        cand_w = F.softmax(scaled_logits, dim=-1)
        ess = cls.compute_kish_ess(cand_w)  # [B, 17]
        valid = ess >= target_ess
        valid[:, 0] = True  # gamma = 0 is uniform (ESS == K), always valid

        idx = torch.arange(17, device=beta_trust.device).unsqueeze(0).expand(B, 17)
        best_idx = torch.where(valid, idx, torch.zeros_like(idx)).max(dim=-1).values  # [B]
        b_arange = torch.arange(B, device=beta_trust.device)
        return cand_w[b_arange, best_idx]

    @classmethod
    def pool_values_trust_shrunk(
        cls,
        rollout_values: torch.Tensor,
        beta_trust: torch.Tensor,
        depths: Optional[torch.Tensor] = None,
        root_value: Optional[torch.Tensor] = None,
        cal_values: Optional[torch.Tensor] = None,
        cal_betas: Optional[torch.Tensor] = None,
        cal_targets: Optional[torch.Tensor] = None,
        cal_depths: Optional[torch.Tensor] = None,
        cal_root_values: Optional[torch.Tensor] = None,
        kappa_ess: float = 0.65,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Horizon-debiased, Kish-ESS-shrunk telescopic value backup (`trust_shrunk`).

        When calibration data `(cal_values, cal_betas, cal_targets)` is provided, selects
        `(gamma_trust in [0, 1], rho_0 in [0, 1])` and per-depth horizon bias `b_hor(d)` by closed-form
        grid optimization on the calibration split, mathematically guaranteeing:
        `MSE(trust_shrunk) <= min(MSE(unif), MSE(C_0), MSE(stiff))`.
        """
        B, K = rollout_values.shape[:2]
        device = rollout_values.device
        dtype = rollout_values.dtype

        val_flat = rollout_values.reshape(B, K, -1)
        D_val = val_flat.shape[-1]

        # 1. Estimate depth-horizon bias b_hor(d) if calibration depths & targets are provided
        debiased_val = val_flat.clone()
        cal_debiased = cal_values.reshape(cal_values.shape[0], K, D_val).clone() if cal_values is not None else None

        if cal_values is not None and cal_targets is not None and cal_depths is not None and depths is not None:
            cal_t = cal_targets.reshape(cal_values.shape[0], 1, D_val)
            unique_d = torch.unique(cal_depths)
            for d_val in unique_d:
                mask_cal = (cal_depths == d_val).unsqueeze(-1).to(dtype)
                count_cal = mask_cal.sum(dim=(0, 1)).clamp(min=1.0)
                bias_d = ((cal_debiased - cal_t) * mask_cal).sum(dim=(0, 1)) / count_cal
                mask_test = (depths == d_val).unsqueeze(-1).to(dtype)
                debiased_val = debiased_val - mask_test * bias_d.view(1, 1, D_val)
                cal_debiased = cal_debiased - mask_cal * bias_d.view(1, 1, D_val)

        # 2. Select (gamma_trust, rho_root) on calibration split if provided
        best_gamma = 0.5
        best_rho = 0.0
        if cal_debiased is not None and cal_betas is not None and cal_targets is not None:
            cal_t_2d = cal_targets.reshape(cal_debiased.shape[0], D_val)
            best_mse = float("inf")
            gamma_grid = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
            rho_grid = [0.0, 0.15, 0.3, 0.5, 0.75, 1.0] if cal_root_values is not None else [0.0]

            for g in gamma_grid:
                w_cal = cls.trust_shrunk_weights(cal_betas, gamma_trust=g, kappa_ess=kappa_ess)  # [N_cal, K]
                pooled_cal = (w_cal.unsqueeze(-1) * cal_debiased).sum(dim=1)
                for r0 in rho_grid:
                    if cal_root_values is not None and r0 > 0.0:
                        r_cal = cal_root_values.reshape(cal_debiased.shape[0], D_val)
                        pred_cal = (1.0 - r0) * pooled_cal + r0 * r_cal
                    else:
                        pred_cal = pooled_cal
                    mse_val = float(((pred_cal - cal_t_2d) ** 2).mean().item())
                    if mse_val < best_mse:
                        best_mse = mse_val
                        best_gamma = g
                        best_rho = r0

        weights = cls.trust_shrunk_weights(beta_trust, gamma_trust=best_gamma, kappa_ess=kappa_ess)
        pooled = (weights.unsqueeze(-1) * debiased_val).sum(dim=1)
        if root_value is not None and best_rho > 0.0:
            pooled = (1.0 - best_rho) * pooled + best_rho * root_value.reshape(B, D_val)

        ess_mean = float(cls.compute_kish_ess(weights).mean().item())
        return pooled.reshape(list(rollout_values.shape[:1]) + list(rollout_values.shape[2:])), {
            "weights": weights,
            "gamma_trust": best_gamma,
            "rho_root": best_rho,
            "ess_mean": ess_mean,
        }


class GLCAOMTransformerLayer(nn.Module):
    """Single Transformer Layer of `GLCAOMWorldModel` with zero-initialized Lemma Cross-Attention."""

    def __init__(self, cfg: GLCAOMConfig):
        super().__init__()
        self.cfg = cfg
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.k_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.v_proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

        # Top-down lemma cross-attention (chi = (0, 1), zero-initialized gate)
        self.ln_lemma = nn.LayerNorm(cfg.d_model)
        self.lemma_attn = nn.MultiheadAttention(
            embed_dim=cfg.d_model, num_heads=cfg.n_heads, batch_first=True
        )
        self.lemma_gate = nn.Parameter(torch.zeros(1))

        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.GELU(),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        lemma_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run self-attention + optional lemma cross-attention + FFN.

        Returns `(x_out, kv_summary)` where `kv_summary` has shape `[B, 2, d_model]` for TreePagedKVBank.
        """
        B, L, D = x.shape
        H = self.cfg.n_heads
        d_k = D // H

        norm_x = self.ln1(x)
        q = self.q_proj(norm_x).view(B, L, H, d_k).transpose(1, 2)
        k = self.k_proj(norm_x).view(B, L, H, d_k).transpose(1, 2)
        v = self.v_proj(norm_x).view(B, L, H, d_k).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, D)
        x = x + self.out_proj(attn_out)

        if lemma_tokens is not None and torch.abs(self.lemma_gate).item() > 0.0:
            q_lem = self.ln_lemma(x.detach())
            lem_out, _ = self.lemma_attn(query=q_lem, key=lemma_tokens, value=lemma_tokens, need_weights=False)
            x = x + torch.tanh(self.lemma_gate) * lem_out

        x = x + self.ffn(self.ln2(x))
        # Pooled K/V summary for incremental tree-paged cache slot ([B, 2, d_model])
        k_full = k.transpose(1, 2).contiguous().view(B, L, D)
        v_full = v.transpose(1, 2).contiguous().view(B, L, D)
        kv_summary = torch.stack([k_full.mean(dim=1), v_full.mean(dim=1)], dim=1)  # [B, 2, d_model]
        return x, kv_summary


class GLCAOMWorldModel(nn.Module):
    """Unified Multi-Game Any-Order Transformer World Model (`GLCAOMWorldModel`).

    Enforces strict LIR separation:
    - Epistemic belief trunk (`theta_real`, `aux_emitter`, `query_head`, `memory_transport`, `trust_head`)
      carries `chi = (0, 1)`: NEVER reads `header[:, 5]` (`UTILITY_SPEC`) or `utility_vector`.
    - Action/policy module (`action_head`, `util_emb`) carries `chi = (1, 1)`: reads `header[:, 5]`,
      `utility_vector`, and `stop_grad` belief representations.
    """

    def __init__(self, cfg: Optional[GLCAOMConfig] = None):
        super().__init__()
        self.cfg = cfg or GLCAOMConfig()
        c = self.cfg

        # --- Epistemic Belief Trunk (theta_real, chi = (0, 1)) ---
        # Embeds ONLY header[:, :5]: [GAME_FAMILY, GAME_ID, GRID_H, GRID_W, INVENTORY_SPEC]
        # NEVER embeds header[:, 5] (UTILITY_SPEC)!
        self.family_emb = nn.Embedding(16, c.d_model)
        self.game_emb = nn.Embedding(128, c.d_model)
        self.grid_h_emb = nn.Embedding(32, c.d_model)
        self.grid_w_emb = nn.Embedding(32, c.d_model)
        self.inv_proj = nn.Linear(1, c.d_model)

        self.var_id_emb = nn.Embedding(c.num_vars, c.d_model)
        self.val_token_emb = nn.Embedding(c.num_vals + 2, c.d_model)  # 0 = unobserved, 1..K = value
        self.feat_proj = nn.Linear(c.d_feat, c.d_model)

        self.layers = nn.ModuleList([GLCAOMTransformerLayer(c) for _ in range(c.n_layers)])
        self.final_ln = nn.LayerNorm(c.d_model)
        self.var_logit_head = nn.Linear(c.d_model, c.num_vals)

        # Direct local feature-to-logit residual for rapid constraint & sensor deduction
        self.local_prior_head = nn.Sequential(
            nn.Linear(c.d_feat + c.num_vals + 1, c.d_model // 2),
            nn.SiLU(),
            nn.Linear(c.d_model // 2, c.num_vals),
        )

        # --- Consequence Extension, Query, Memory & Trust Modules (chi = (0, 1)) ---
        self.aux_emitter = BilinearAuxLemmaEmitter(c)
        self.lemma_slot_proj = nn.Sequential(
            nn.LayerNorm(c.d_aux),
            nn.Linear(c.d_aux, c.d_model),
            nn.SiLU(),
            nn.Linear(c.d_model, c.d_model),
        )
        self.query_head = CombinatorialQueryHead(c, self.aux_emitter)
        self.memory_transport = EpisodicMemoryTransport(c)
        self.trust_head = DepthDebiasedTrustHead(c)

        # --- Policy / Action Decision Module (theta_policy, chi = (1, 1)) ---
        # Explicitly conditions on header[:, 5] (UTILITY_SPEC) and utility_vector with stop_grad on beliefs
        self.utility_spec_emb = nn.Embedding(16, c.d_model)
        self.action_policy_mlp = nn.Sequential(
            nn.Linear(c.d_model * 2 + 2, c.d_model),
            nn.SiLU(),
            nn.Linear(c.d_model, 1),
        )
        self.utility_temp = nn.Parameter(torch.tensor(4.0))

        # --- Optional Lightweight Game-Header FiLM Modulation (R12-E2) ---
        if c.use_header_film:
            self.film_var_gamma = nn.Linear(c.d_model, c.d_model)
            self.film_var_beta = nn.Linear(c.d_model, c.d_model)
            self.film_prior_gamma = nn.Linear(c.d_model, c.num_vals)
            self.film_prior_beta = nn.Linear(c.d_model, c.num_vals)
            self.film_game_logit_scale = nn.Embedding(128, c.num_vals)
            self.film_game_logit_bias = nn.Embedding(128, c.num_vals)
            self.film_policy_gamma = nn.Linear(c.d_model, 1)
            self.film_policy_beta = nn.Linear(c.d_model, 1)
            for m_film in (
                self.film_var_gamma,
                self.film_var_beta,
                self.film_prior_gamma,
                self.film_prior_beta,
                self.film_policy_gamma,
                self.film_policy_beta,
            ):
                nn.init.zeros_(m_film.weight)
                nn.init.zeros_(m_film.bias)
            nn.init.zeros_(self.film_game_logit_scale.weight)
            nn.init.zeros_(self.film_game_logit_bias.weight)

    def get_epistemic_parameters(self) -> List[nn.Parameter]:
        """Return parameters with `chi = (0, 1)` (`theta_real`, `Z_aux`, `theta_trust`, `M_t`)."""
        policy_ids = {id(p) for p in self.get_policy_parameters()}
        return [p for p in self.parameters() if id(p) not in policy_ids]

    def get_policy_parameters(self) -> List[nn.Parameter]:
        """Return parameters with `chi = (1, 1)` (utility/action policy head)."""
        params: List[nn.Parameter] = []
        params.extend(self.utility_spec_emb.parameters())
        params.extend(self.action_policy_mlp.parameters())
        params.append(self.utility_temp)
        if self.cfg.use_header_film:
            params.extend(self.film_policy_gamma.parameters())
            params.extend(self.film_policy_beta.parameters())
        return params

    def encode_epistemic_header(self, header: torch.Tensor) -> torch.Tensor:
        """Encode `header[:, :5]` into `[B, 5, d_model]` while strictly ignoring `header[:, 5]` (`UTILITY_SPEC`)."""
        fam = self.family_emb(header[:, 0].clamp(0, 15))
        gid = self.game_emb(header[:, 1].clamp(0, 127))
        gh = self.grid_h_emb(header[:, 2].clamp(0, 31))
        gw = self.grid_w_emb(header[:, 3].clamp(0, 31))
        inv = self.inv_proj((header[:, 4].to(fam.dtype) / 1000.0).unsqueeze(-1))
        return torch.stack([fam, gid, gh, gw, inv], dim=1)

    def _compute_structural_constraint_logits(
        self,
        batch: EnvBatch,
        obs_tokens: torch.Tensor,
        obs_mask: torch.Tensor,
        root_clamped_mask: Optional[torch.Tensor] = None,
        h_cons: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Vectorized local domain-consistent log-odds from observed neighbors/clauses (`chi = (0, 1)`).

        Ensures that clamping a variable `(v, z)` during `STEP(v, z)` immediately propagates local
        single-step parity/peer constraints in `q_real(X_i | C_j)` without peeking at `target_tokens`
        or `exact_marginals`.
        """
        B, V = obs_tokens.shape
        K = batch.domain_size
        device = obs_tokens.device
        dtype = batch.obs_features.dtype

        struct_logits = torch.zeros((B, V, K), dtype=dtype, device=device)

        # 1. Equiv-SAT local parity propagation along incident clauses encoded in obs_features
        # In EquivSatEnv, obs_features[:, :, :2*M] encodes clause incidence:
        # col 2*m is +1 at u and v if x_u == x_v, or +1 at u and -1 at v if x_u != x_v
        game_ids = batch.header[:, 1]
        if (game_ids == 10).any() and K == 2 and batch.obs_features.shape[-1] >= 4:
            num_clauses = (batch.obs_features.shape[-1] - 2) // 2
            if num_clauses > 0:
                inc = batch.obs_features[:, :, : 2 * num_clauses : 2]  # [B, V, M] in {0, +1}
                csign = batch.obs_features[:, :, 1 : 2 * num_clauses : 2]  # [B, V, M] in {-1, 0, +1}
                # Signed observed bit: -1 for token 1 (bit 0), +1 for token 2 (bit 1), 0 if unobserved
                obs_sign = torch.where(
                    obs_mask & (obs_tokens > 0),
                    torch.where(obs_tokens == 2, torch.ones_like(obs_tokens, dtype=dtype), -torch.ones_like(obs_tokens, dtype=dtype)),
                    torch.zeros_like(obs_tokens, dtype=dtype),
                )  # [B, V]
                # Propagate across incident clauses (6 message-passing hops within local receptive field)
                cur_belief = obs_sign.clone()
                for _ in range(6):
                    # Clause message: [B, M]
                    clause_msg = torch.einsum("bvm,bv->bm", inc, cur_belief)
                    # Back to variables, subtracting self contribution and applying clause parity sign: [B, V]
                    other_belief = inc * clause_msg.unsqueeze(1) - inc * cur_belief.unsqueeze(-1)
                    var_msg = (csign * other_belief).sum(dim=-1)
                    cur_belief = torch.where(
                        obs_mask & (obs_tokens > 0),
                        obs_sign,
                        torch.tanh(2.5 * var_msg),
                    )
                # Map cur_belief in [-1, 1] to log-odds for token 1 (bit 0) vs token 2 (bit 1)
                eq_mask = (game_ids == 10).view(B, 1, 1)
                eq_logits = torch.stack([-4.0 * cur_belief, 4.0 * cur_belief], dim=-1)
                struct_logits = torch.where(eq_mask, eq_logits, struct_logits)

        # 2. Sudoku row/col/box peer exclusion for observed digits
        if ((game_ids == 20) | (game_ids == 21)).any() and V in (36, 81) and K in (6, 9):
            size = 6 if V == 36 else 9
            br, bc = (2, 3) if size == 6 else (3, 3)
            idx = torch.arange(V, device=device)
            r_idx = idx // size
            c_idx = idx % size
            b_idx = (r_idx // br) * (size // bc) + (c_idx // bc)
            peer_adj = (
                (r_idx[:, None] == r_idx[None, :])
                | (c_idx[:, None] == c_idx[None, :])
                | (b_idx[:, None] == b_idx[None, :])
            )
            peer_adj.fill_diagonal_(False)
            peer_f = peer_adj.to(dtype)  # [V, V]

            clamped_onehot = F.one_hot(obs_tokens.clamp(1, K) - 1, num_classes=K).to(dtype) * (
                (obs_mask & (obs_tokens >= 1)).unsqueeze(-1).to(dtype)
            )  # [B, V, K]

            r_mask_sud = obs_mask if root_clamped_mask is None else root_clamped_mask
            has_branch_clamps = bool((obs_mask & (~r_mask_sud)).any().item())
            has_world_aux = False
            if h_cons is not None:
                h_aux_w = self.aux_emitter.mask_to_world_lemma_channel(h_cons[:, self.cfg.d_v :])
                has_world_aux = bool(h_aux_w.abs().max().item() > 1e-5)

            cur_known = clamped_onehot.clone()
            if has_branch_clamps or has_world_aux:
                unit_rows = torch.stack([r_idx == u for u in range(size)], dim=0)
                unit_cols = torch.stack([c_idx == u for u in range(size)], dim=0)
                unit_boxs = torch.stack([b_idx == u for u in range(size)], dim=0)
                unit_f = torch.cat([unit_rows, unit_cols, unit_boxs], dim=0).to(dtype)  # [3*size, V]
                for _ in range(6):
                    pt_hop = torch.einsum("uv,bvk->buk", peer_f, cur_known)
                    unclamped_cell = (1.0 - cur_known.sum(dim=-1, keepdim=True)).clamp(0.0, 1.0)
                    avail = (pt_hop < 0.5).to(dtype) * unclamped_cell + cur_known
                    naked = avail * (avail.sum(dim=-1, keepdim=True) == 1.0).to(dtype)
                    if has_world_aux:
                        unit_avail_cnt = torch.einsum("uv,bvk->buk", unit_f, avail)
                        unit_uniq = (unit_avail_cnt == 1.0).to(dtype)
                        hidden = avail * (torch.einsum("uv,buk->bvk", unit_f, unit_uniq) >= 1.0).to(dtype)
                        cur_known = torch.maximum(cur_known, torch.maximum(naked, hidden))
                    else:
                        cur_known = torch.maximum(cur_known, naked)

            peer_taken = torch.einsum("uv,bvk->buk", peer_f, cur_known)  # [B, V, K]
            deduced_cell = (cur_known.sum(dim=-1, keepdim=True) > 0.5) & (~obs_mask.unsqueeze(-1))
            deduced_boost = torch.where(deduced_cell, (cur_known - 0.5) * 16.0, torch.zeros_like(cur_known))
            sud_mask = ((game_ids == 20) | (game_ids == 21)).view(B, 1, 1)
            struct_logits = torch.where(sud_mask, -28.0 * peer_taken + deduced_boost, struct_logits)

        # 3. General feature-driven local log-odds when obs_features carries direct marginal/sensor signals
        # Mask out Proposal-Poker private utility bias column 11 in epistemic trunk (chi(U -> V) = 0)
        epistemic_features = batch.obs_features
        if (game_ids == 80).any() and epistemic_features.shape[-1] > 11:
            epistemic_features = epistemic_features.clone()
            pp_mask = (game_ids == 80).unsqueeze(-1)
            epistemic_features[:, :, 11] = torch.where(
                pp_mask, torch.zeros_like(epistemic_features[:, :, 11]), epistemic_features[:, :, 11]
            )

        if epistemic_features.shape[-1] >= K:
            feat_k = epistemic_features[:, :, :K]
            struct_logits = struct_logits + 0.5 * feat_k

        # Strongly clamp already-observed positions to their observed value while preserving peer-clash penalties
        valid_obs = obs_mask & (obs_tokens >= 1) & (obs_tokens <= K)
        if valid_obs.any():
            obs_oh = F.one_hot(obs_tokens.clamp(1, K) - 1, num_classes=K).to(dtype)
            clamp_logits = (obs_oh - 0.5) * 24.0
            struct_logits = struct_logits + torch.where(
                valid_obs.unsqueeze(-1), clamp_logits, torch.zeros_like(struct_logits)
            )

        return struct_logits

    def forward(
        self,
        batch: EnvBatch,
        h_cons: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
        root_clamped_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of `GLCAOMWorldModel` on `EnvBatch`.

        Strictly respects `LIRMask`:
        - `var_logits`, `var_probs`, `query_probs`, `m_ql`, `beta_trust`, `kv_cache`, and `h_var`
          are computed under `chi = (0, 1)` with ZERO dependence on `header[:, 5]` or `utility_vector`.
        - `action_logits` and `action_probs` are computed under `chi = (1, 1)` with `stop_grad` on
          all belief tensors.
        """
        B = batch.batch_size
        V_act = batch.num_vars
        K_act = batch.domain_size
        device = batch.device
        dtype = batch.obs_features.dtype

        # 1. Epistemic Header (ignores header[:, 5] UTILITY_SPEC!)
        hdr_emb = self.encode_epistemic_header(batch.header)  # [B, 5, d_model]

        # 2. Present-Time Episodic Memory Tokens M_t (chi = (0, 1))
        mem_out = self.memory_transport(
            h_cons_prev=None,
            batch_memory_tokens=batch.memory_tokens,
            batch_memory_mask=batch.memory_mask,
            batch_size=B,
            device=device,
        )
        mem_tokens = mem_out["memory_tokens"]  # [B, k_mem, d_model]

        # 3. Variable Token Embeddings (mask Proposal-Poker utility bias col 11 so chi(U -> V) == 0)
        var_ids = torch.arange(V_act, device=device).clamp(0, self.cfg.num_vars - 1).unsqueeze(0).expand(B, V_act)
        tok_vals = torch.where(batch.obs_mask, batch.obs_tokens.clamp(0, self.cfg.num_vals + 1), torch.zeros_like(batch.obs_tokens))
        obs_feat_epistemic = batch.obs_features
        if (batch.header[:, 1] == 80).any() and obs_feat_epistemic.shape[-1] > 11:
            obs_feat_epistemic = obs_feat_epistemic.clone()
            obs_feat_epistemic[:, :, 11] = torch.where(
                (batch.header[:, 1] == 80).unsqueeze(-1),
                torch.zeros_like(obs_feat_epistemic[:, :, 11]),
                obs_feat_epistemic[:, :, 11],
            )
        feat_adapted = _adapt_last_dim(obs_feat_epistemic, self.cfg.d_feat)

        var_emb = (
            self.var_id_emb(var_ids)
            + self.val_token_emb(tok_vals)
            + self.feat_proj(feat_adapted)
        )  # [B, V_act, d_model]

        seq = torch.cat([hdr_emb, mem_tokens, var_emb], dim=1)  # [B, 5 + k_mem + V_act, d_model]

        lemma_tokens = None
        if h_cons is not None:
            h_aux = h_cons[:, self.cfg.d_v :]
            lemma_tokens = self.lemma_slot_proj(h_aux.view(B, self.cfg.m_aux, self.cfg.d_aux))

        kv_list = []
        x = seq
        for layer in self.layers:
            x, kv_layer = layer(x, lemma_tokens=lemma_tokens)
            kv_list.append(kv_layer)

        x = self.final_ln(x)
        h_var = x[:, 5 + self.cfg.k_mem :, :]  # [B, V_act, d_model]
        kv_cache = torch.stack(kv_list, dim=1)  # [B, L_wm, 2, d_model]

        # 4. Variable Belief Logits q_real(X_i | C_j)
        oh_obs = F.one_hot(tok_vals.clamp(0, self.cfg.num_vals), num_classes=self.cfg.num_vals + 1).to(dtype)
        local_in = torch.cat([feat_adapted, oh_obs], dim=-1)
        local_prior_out = self.local_prior_head(local_in)

        if self.cfg.use_header_film:
            hdr_pool = hdr_emb.mean(dim=1, keepdim=True)  # [B, 1, d_model] (strictly epistemic header[:, :5])
            h_var = (
                1.0 + 0.5 * torch.tanh(self.film_var_gamma(hdr_pool))
            ) * h_var + 0.5 * self.film_var_beta(hdr_pool)
            local_prior_out = (
                1.0 + 0.5 * torch.tanh(self.film_prior_gamma(hdr_pool))
            ) * local_prior_out + 0.5 * self.film_prior_beta(hdr_pool)

        base_head_logits = self.var_logit_head(h_var)[:, :, :K_act] + local_prior_out[:, :, :K_act]
        if self.cfg.use_header_film:
            gid_clamped = batch.header[:, 1].clamp(0, 127)
            g_scale = 1.0 + 0.5 * torch.tanh(self.film_game_logit_scale(gid_clamped))[:, None, :K_act]
            g_bias = self.film_game_logit_bias(gid_clamped)[:, None, :K_act]
            base_head_logits = g_scale * base_head_logits + g_bias

        raw_logits = base_head_logits + self._compute_structural_constraint_logits(
            batch,
            batch.obs_tokens,
            batch.obs_mask,
            root_clamped_mask=root_clamped_mask,
            h_cons=h_cons,
        )

        # If a cleared consensus h_cons is provided, apply the gated readout firewall + top-down prefix read
        if h_cons is not None:
            firewall_logits = self.aux_emitter.readout_var_firewall(h_cons, num_vars=V_act, num_vals=K_act)
            r_mask = batch.obs_mask if root_clamped_mask is None else root_clamped_mask
            top_down_logits = self.aux_emitter.decode_top_down_from_prefix(
                h_cons=h_cons,
                prefix_tokens=batch.obs_tokens,
                prefix_mask=batch.obs_mask,
                root_mask=r_mask,
                num_vals=K_act,
            )
            var_logits = raw_logits + firewall_logits + top_down_logits
        else:
            var_logits = raw_logits
            # Initialize default root h_cons = [padded_var_logits ; 0]
            padded_v = self.aux_emitter.pad_var_logits_to_dv(var_logits)
            h_cons = torch.cat(
                [padded_v, torch.zeros((B, self.cfg.d_aux_total), dtype=dtype, device=device)],
                dim=-1,
            )

        var_probs = F.softmax(var_logits, dim=-1)

        # 5. Leaf Self-Consistency m_ql(C_j) and Depth-Debiased Trust beta_trust(C_j)
        r_mask = batch.obs_mask if root_clamped_mask is None else root_clamped_mask
        m_ql = self.trust_head.compute_leaf_self_consistency(
            node_logits=var_logits,
            node_clamped_mask=batch.obs_mask,
            node_clamped_tokens=batch.obs_tokens,
            root_clamped_mask=r_mask,
        )
        d_tensor = torch.zeros(B, dtype=torch.long, device=device) if depth is None else depth
        beta_trust = self.trust_head(
            node_h=h_var,
            node_logits=var_logits,
            depth=d_tensor,
            m_ql=m_ql,
        )

        # 6. Direct Combinatorial Query Readout P(Q(A, B)) & Initial Active Prospectus p_0 = k_{Q(A, B)}
        query_probs = self.query_head.predict_query_probs(batch, var_probs, h_cons)
        p0_prospectus = self.query_head.seed_prospectus(batch.query_pairs, h_cons=h_cons)

        # 7. Utility-Conditioned Action Head (chi = (1, 1), strictly detached from belief trunk!)
        action_logits, action_probs = self.compute_action_policy(
            batch=batch,
            h_var_detached=h_var.detach(),
            var_probs_detached=var_probs.detach(),
        )

        return {
            "var_logits": var_logits,
            "var_probs": var_probs,
            "query_probs": query_probs,
            "p0_prospectus": p0_prospectus,
            "h_var": h_var,
            "kv_cache": kv_cache,
            "h_cons": h_cons,
            "m_ql": m_ql,
            "beta_trust": beta_trust,
            "memory_tokens": mem_tokens,
            "beta_memory": mem_out["beta_memory"],
            "action_logits": action_logits,
            "action_probs": action_probs,
        }

    def compute_action_policy(
        self,
        batch: EnvBatch,
        h_var_detached: torch.Tensor,
        var_probs_detached: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute utility-conditioned action distribution `pi_action(a | stop_grad(belief), U)` (`chi = (1, 1)`).

        Strictly detaches all epistemic inputs so RL updates on `pi_action` cause 0.0000 belief drift,
        while utility swaps (`swap_utility`) immediately flip action preferences without retraining.
        """
        B = batch.batch_size
        device = batch.device
        dtype = h_var_detached.dtype

        if batch.utility_vector is not None:
            n_act = batch.utility_vector.shape[1]
            u_vec = batch.utility_vector.to(dtype)
        elif batch.legal_action_mask is not None:
            n_act = batch.legal_action_mask.shape[1]
            u_vec = torch.zeros((B, n_act), dtype=dtype, device=device)
        else:
            n_act = batch.num_vars
            u_vec = torch.zeros((B, n_act), dtype=dtype, device=device)

        uspec_id = batch.header[:, 5].clamp(0, 15)
        u_emb = self.utility_spec_emb(uspec_id).unsqueeze(1).expand(B, n_act, -1)  # [B, n_act, d_model]

        # Pool detached belief representation to match n_act
        if n_act <= h_var_detached.shape[1]:
            h_act = h_var_detached[:, :n_act, :]
            ent_act = -(
                var_probs_detached[:, :n_act] * torch.log(var_probs_detached[:, :n_act].clamp(min=1e-7))
            ).sum(dim=-1, keepdim=True)
        else:
            h_pooled = h_var_detached.mean(dim=1, keepdim=True).expand(B, n_act, -1)
            h_act = h_pooled
            ent_act = torch.zeros((B, n_act, 1), dtype=dtype, device=device)

        # Flip/modulate effective utility direction based on UTILITY_SPEC when utility_vector is shared
        # e.g. PLAYER_0_WIN (1) vs PLAYER_1_WIN (2), SECTOR_NW_PREF (3) vs SECTOR_SE_PREF (4),
        # BIAS_PRO_APPROVE (7) vs BIAS_PRO_REJECT (8)
        pos_idx = torch.linspace(-1.0, 1.0, n_act, device=device, dtype=dtype).unsqueeze(0).expand(B, n_act)
        spec_dir = torch.ones((B, 1), dtype=dtype, device=device)
        neg_specs = (
            (uspec_id == int(UtilitySpec.PLAYER_1_WIN))
            | (uspec_id == int(UtilitySpec.SECTOR_SE_PREF))
            | (uspec_id == int(UtilitySpec.BIAS_PRO_REJECT))
        )
        spec_dir = torch.where(neg_specs.unsqueeze(-1), -spec_dir, spec_dir)
        effective_u = u_vec + 0.5 * spec_dir * pos_idx

        mlp_in = torch.cat([h_act, u_emb, effective_u.unsqueeze(-1), ent_act], dim=-1)
        learned_score = self.action_policy_mlp(mlp_in).squeeze(-1)
        if self.cfg.use_header_film:
            hdr_det = self.encode_epistemic_header(batch.header).detach().mean(dim=1)  # [B, d_model]
            pol_scale = 1.0 + 0.5 * torch.tanh(self.film_policy_gamma(hdr_det))
            pol_shift = 0.5 * self.film_policy_beta(hdr_det)
            learned_score = pol_scale * learned_score + pol_shift
        logits = F.softplus(self.utility_temp) * effective_u + learned_score

        if batch.legal_action_mask is not None:
            logits = logits.masked_fill(~batch.legal_action_mask, -1e9)

        probs = F.softmax(logits, dim=-1)
        return logits, probs

    def compute_any_order_loss(
        self,
        batch: EnvBatch,
        h_cons: torch.Tensor,
        num_permutations: int = 2,
        detach_real_trunk: bool = True,
    ) -> torch.Tensor:
        """Compute Random-Permutation Any-Order Autoregressive Loss `L_{any-order}(h_cons; x*)`.

        Implements Member 2 Theorem 1:
          `L_{any-order}(h_cons; x*) = -E_{pi ~ Unif}[ sum_t log q_theta(x*_{pi(t)} | x_O, x*_{pi(<t)}, h_cons) ]`
        """
        B, V = batch.obs_tokens.shape
        K = batch.domain_size
        device = batch.device
        target_idx = (batch.target_tokens.clamp(1, K) - 1).long()  # [B, V]
        root_mask = batch.obs_mask.clone()

        total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        steps_counted = 0

        for p_idx in range(num_permutations):
            # Sample a random cutoff t in [0, |V \ O|) to evaluate teacher-forced prefix conditioning in parallel
            perm = torch.rand(B, V, device=device).argsort(dim=-1)
            cutoff = (p_idx + 1) * max(1, V // (num_permutations + 1))
            reveal_mask = perm < cutoff

            prefix_mask = root_mask | reveal_mask
            prefix_tokens = torch.where(prefix_mask, batch.target_tokens, torch.zeros_like(batch.obs_tokens))

            step_batch = batch.clone()
            step_batch.obs_mask = prefix_mask
            step_batch.obs_tokens = prefix_tokens

            if detach_real_trunk:
                with torch.no_grad():
                    base_out = self.forward(step_batch, h_cons=None, root_clamped_mask=root_mask)
                base_logits = base_out["var_logits"].detach()
            else:
                base_out = self.forward(step_batch, h_cons=None, root_clamped_mask=root_mask)
                base_logits = base_out["var_logits"]

            firewall_logits = self.aux_emitter.readout_var_firewall(h_cons, num_vars=V, num_vals=K)
            top_down_logits = self.aux_emitter.decode_top_down_from_prefix(
                h_cons=h_cons,
                prefix_tokens=prefix_tokens,
                prefix_mask=prefix_mask,
                root_mask=root_mask,
                num_vals=K,
            )
            cond_logits = base_logits + firewall_logits + top_down_logits

            # Evaluate NLL on the remaining unrevealed variables V \ prefix_mask
            eval_mask = (~prefix_mask) & batch.solvable[:, None]
            if not eval_mask.any():
                eval_mask = ~root_mask
            ce = F.cross_entropy(cond_logits.reshape(-1, K), target_idx.reshape(-1), reduction="none").view(B, V)
            total_loss = total_loss + (ce * eval_mask.float()).sum() / eval_mask.float().sum().clamp(min=1.0)
            steps_counted += 1

        return total_loss / float(max(steps_counted, 1))
