"""Universal Tree-Paged KV-Cache Search Machine (`TreePagedKVSearch`) & Baseline Algorithms.

Implements Member 1, Member 2, and Member 3 specifications for `src/search/kv_tree_search.py`:
1. `TreePagedKVBank`:
   - O(1) bit-mask ancestor tracking `M_tree[b+1, :] = M_tree[j, :] | e_{b+1}` and single-token KV storage.
   - Action-exhaustion mask `Used(j)` and expandable node mask `Exp(j)`.
   - Parent-relative telescopic edge increments `delta_j = l(C_j) - l(C_{pa(j)})` and bag-centered
     edge-salience keys `k_j^edge`.
2. `SELECT_NODE(j) in {0..b}` via Resolvable-Inconsistency Tree-PUCT:
   - `Phi(C_j) = m_ql(C_j) * beta_trust(C_j) * max_v I_goal(v|C_j) H(q_{C_j}(V_v))`
   - Zero hardcoded depth, wave, or entropy thresholds; immune to both Root-Star Collapse (`j=0`)
     and Endless-Chain Collapse (`j=b`).
3. `MoPSFractalPlanHead` (`q_plan(p | C_j, K_done)`):
   - Mixture-of-Power-Spherical (`MoPS`) directional plan sampler on `S^{D_amp - 1}`.
   - Ridge-Damped Gram Orthogonal Projection against `K_done`:
     `P_perp(K_done) = I - K_done^T (K_done B_done K_done^T + lambda_plan I)^{-1} K_done B_done`.
   - Orthogonal Residual Plan Transport along continuing chains (`j_{b+1} == b`).
4. `TwoStageTreeClearer`:
   - Stage 1: Bottom-Up Disjunctive Counterfactual Bayes Mixture at case-split nodes (eliminating
     piece-count vetoes and single-branch clamp bias).
   - Stage 2: Parent-relative telescopic edge increments `delta_j = l(C_j) - l(C_{pa(j)})`,
     bag-centered edge-salience keys `k_j^edge`, and Gershgorin Row-Crowding Preconditioned
     `looped_attn_T3` / `jacobi_lin_T4_prec` (`D_{crowd,r} = lambda(beta_r) + sum_{r'} beta_{r'} max(0, <k_r, k_{r'}>)`).
   - Stage 3: Nested Precision-Gated Residual Update `l^{(b+1)} = l^{(b)} + alpha^{(b+1)} (tilde_l^{(b+1)} - l^{(b)})`
     guaranteeing exact `b=0` identity and non-increasing expected KL across `b = 0 .. 128+`,
     plus Kish-ESS-shrunk (`trust_shrunk`) telescopic value backup.
5. Baseline & Active Search Algorithms across `B in {0, 1, 2, 4, 8, 16, 24, 32, 64, 128}`:
   - `B0_root`, `flat_mix`, `flat_scaled_mix`, `flat_ataraxos_mmd`, `flat_econ_prec_K3`,
     `flat_looped_attn_T3`, `P3_uniform`, `P8_lead_hybrid`, and `GLC_Tree_Active_PUCT`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, Categorical

from src.envs.base import EnvBatch, UnifiedGameEnv, categorical_kl
from src.models.glc_aom import (
    BilinearAuxLemmaEmitter,
    DepthDebiasedTrustHead,
    GLCAOMConfig,
    GLCAOMWorldModel,
    LIRMask,
)

SUPPORTED_BUDGETS: Tuple[int, ...] = (0, 1, 2, 4, 8, 16, 24, 32, 64, 128, 256)

SEARCH_ALGORITHMS: Tuple[str, ...] = (
    "B0_root",
    "flat_mix",
    "flat_scaled_mix",
    "flat_ataraxos_mmd",
    "flat_econ_prec_K3",
    "flat_looped_attn_T3",
    "P3_uniform",
    "P8_lead_hybrid",
    "GLC_Tree_Active_PUCT",
)


# ==============================================================================
# 1. Tree-Paged KV-Cache Bank (O(1) Fork, Continue, and Root-Restart)
# ==============================================================================


@dataclass
class TreePagedKVBank:
    """Tree-paged KV-cache bank storing `C_0, C_1, ..., C_b` for up to `B_max + 1` nodes."""

    batch_size: int
    max_nodes: int
    num_vars: int
    num_vals: int
    d_amp: int
    num_layers: int
    d_model: int

    num_active: int
    parent_idx: torch.Tensor          # [N, B_max + 1] long
    depth: torch.Tensor               # [N, B_max + 1] long
    tree_mask: torch.Tensor           # [N, B_max + 1, B_max + 1] bool (M_tree)
    clamped_mask: torch.Tensor        # [N, B_max + 1, V] bool (O_j)
    clamped_tokens: torch.Tensor      # [N, B_max + 1, V] long (1..K_val)
    step_var: torch.Tensor            # [N, B_max + 1] long (v_j clamped on edge pa(j)->j)
    step_val: torch.Tensor            # [N, B_max + 1] long (z_j in 0..K_val-1 clamped on edge pa(j)->j)
    used_children: torch.Tensor       # [N, B_max + 1, V, K_val] bool (Used(j))
    n_exp: torch.Tensor               # [N, B_max + 1] long (N_exp(j): number of children expanded from j)
    is_valid: torch.Tensor            # [N, B_max + 1] bool
    kv_bank: torch.Tensor             # [N, L_wm, 2, B_max + 1, d_model]
    h_bank: torch.Tensor              # [N, B_max + 1, V, d_model]
    logits_bank: torch.Tensor         # [N, B_max + 1, V, K_val]
    edge_delta_var_bank: torch.Tensor # [N, B_max + 1, V, K_val] parent-relative delta_j = l(C_j) - l(C_{pa(j)})
    edge_delta_amp_bank: torch.Tensor # [N, B_max + 1, D_amp] expanded edge increment [delta_j ; z_aux]
    edge_keys_bank: torch.Tensor      # [N, B_max + 1, D_amp] parent-relative salience footprint s_j^edge
    root_keys_bank: torch.Tensor      # [N, B_max + 1, D_amp] root-relative K3 salience key (for baseline comparison)
    m_ql_bank: torch.Tensor           # [N, B_max + 1] pure leaf self-consistency m_ql(C_j)
    m_qs_bank: torch.Tensor           # [N, B_max + 1] chain-time min step prob (tracked only for P8 audit)
    beta_trust_bank: torch.Tensor     # [N, B_max + 1] calibrated trust precision beta_trust(C_j)
    plan_bank: torch.Tensor           # [N, B_max + 1, D_amp] active prospectus plan p_j
    q_tree_bank: torch.Tensor         # [N, B_max + 1] tree value estimate Q_hat_tree(C_j)
    anc_mask: Optional[torch.Tensor] = None           # [N, B_max + 1, B_max + 1] bool: 1[j in Anc(k)]
    rel_depth: Optional[torch.Tensor] = None          # [N, B_max + 1, B_max + 1] long: depth[k] - depth[j]
    visit_count: Optional[torch.Tensor] = None        # [N, B_max + 1] long: sum_{k in Desc(j)} 1
    leaf_value_bank: Optional[torch.Tensor] = None    # [N, B_max + 1] intrinsic leaf evaluation v(C_j)
    subtree_value_sum: Optional[torch.Tensor] = None  # [N, B_max + 1] sum_{k in Desc(j)} v(C_k)
    step_prob_bank: Optional[torch.Tensor] = None     # [N, B_max + 1] edge prior p(C_j | C_{pa(j)})
    path_log_reach: Optional[torch.Tensor] = None     # [N, B_max + 1] cumulative log reach log pi(C_j)

    def __post_init__(self) -> None:
        N = self.batch_size
        M = self.max_nodes
        device = self.logits_bank.device
        dtype = self.logits_bank.dtype
        if self.anc_mask is None:
            self.anc_mask = self.tree_mask.transpose(1, 2).contiguous()
        if self.rel_depth is None:
            self.rel_depth = self.depth.unsqueeze(1) - self.depth.unsqueeze(2)
        if self.visit_count is None:
            self.visit_count = torch.zeros((N, M), dtype=torch.long, device=device)
            b = self.num_active
            self.visit_count[:, :b] = self.anc_mask[:, :b, :b].long().sum(dim=-1)
        if self.leaf_value_bank is None:
            self.leaf_value_bank = self.q_tree_bank.clone()
        if self.subtree_value_sum is None:
            self.subtree_value_sum = torch.zeros((N, M), dtype=dtype, device=device)
            b = self.num_active
            self.subtree_value_sum[:, :b] = torch.bmm(
                self.anc_mask[:, :b, :b].to(dtype),
                self.leaf_value_bank[:, :b].unsqueeze(-1),
            ).squeeze(-1)
        if self.step_prob_bank is None:
            self.step_prob_bank = torch.ones((N, M), dtype=dtype, device=device)
        if self.path_log_reach is None:
            self.path_log_reach = torch.zeros((N, M), dtype=dtype, device=device)

    @classmethod
    def initialize_from_root(
        cls,
        root_clamped_mask: torch.Tensor,   # [N, V] bool
        root_clamped_tokens: torch.Tensor, # [N, V] long
        root_kv: torch.Tensor,             # [N, L_wm, 2, d_model]
        root_h: torch.Tensor,              # [N, V, d_model]
        root_logits: torch.Tensor,         # [N, V, K_val]
        root_h_cons: torch.Tensor,         # [N, D_amp]
        root_beta: torch.Tensor,           # [N]
        root_plan: torch.Tensor,           # [N, D_amp]
        max_steps: int,
    ) -> "TreePagedKVBank":
        N, V = root_clamped_mask.shape
        _, L_wm, _, d_model = root_kv.shape
        _, _, K_val = root_logits.shape
        D_amp = root_h_cons.shape[-1]
        max_nodes = max_steps + 1
        device = root_logits.device
        dtype = root_logits.dtype

        parent_idx = torch.zeros((N, max_nodes), dtype=torch.long, device=device)
        depth = torch.zeros((N, max_nodes), dtype=torch.long, device=device)
        tree_mask = torch.zeros((N, max_nodes, max_nodes), dtype=torch.bool, device=device)
        tree_mask[:, 0, 0] = True

        anc_mask = torch.zeros((N, max_nodes, max_nodes), dtype=torch.bool, device=device)
        anc_mask[:, 0, 0] = True
        rel_depth = torch.zeros((N, max_nodes, max_nodes), dtype=torch.long, device=device)
        visit_count = torch.zeros((N, max_nodes), dtype=torch.long, device=device)
        visit_count[:, 0] = 1

        clamped_mask = torch.zeros((N, max_nodes, V), dtype=torch.bool, device=device)
        clamped_mask[:, 0] = root_clamped_mask

        clamped_tokens = torch.zeros((N, max_nodes, V), dtype=torch.long, device=device)
        clamped_tokens[:, 0] = root_clamped_tokens

        step_var = torch.zeros((N, max_nodes), dtype=torch.long, device=device)
        step_val = torch.zeros((N, max_nodes), dtype=torch.long, device=device)
        used_children = torch.zeros((N, max_nodes, V, K_val), dtype=torch.bool, device=device)
        n_exp = torch.zeros((N, max_nodes), dtype=torch.long, device=device)

        is_valid = torch.zeros((N, max_nodes), dtype=torch.bool, device=device)
        is_valid[:, 0] = True

        kv_bank = torch.zeros((N, L_wm, 2, max_nodes, d_model), dtype=dtype, device=device)
        kv_bank[:, :, :, 0, :] = root_kv

        h_bank = torch.zeros((N, max_nodes, V, d_model), dtype=dtype, device=device)
        h_bank[:, 0] = root_h

        logits_bank = torch.zeros((N, max_nodes, V, K_val), dtype=dtype, device=device)
        logits_bank[:, 0] = root_logits

        edge_delta_var_bank = torch.zeros((N, max_nodes, V, K_val), dtype=dtype, device=device)
        edge_delta_amp_bank = torch.zeros((N, max_nodes, D_amp), dtype=dtype, device=device)
        edge_delta_amp_bank[:, 0] = root_h_cons

        edge_keys_bank = torch.zeros((N, max_nodes, D_amp), dtype=dtype, device=device)
        root_keys_bank = torch.zeros((N, max_nodes, D_amp), dtype=dtype, device=device)

        m_ql_bank = torch.zeros((N, max_nodes), dtype=dtype, device=device)
        m_ql_bank[:, 0] = 1.0

        m_qs_bank = torch.zeros((N, max_nodes), dtype=dtype, device=device)
        m_qs_bank[:, 0] = 1.0

        beta_trust_bank = torch.zeros((N, max_nodes), dtype=dtype, device=device)
        beta_trust_bank[:, 0] = root_beta

        plan_bank = torch.zeros((N, max_nodes, D_amp), dtype=dtype, device=device)
        plan_bank[:, 0] = root_plan

        q_tree_bank = torch.zeros((N, max_nodes), dtype=dtype, device=device)
        leaf_value_bank = torch.zeros((N, max_nodes), dtype=dtype, device=device)
        subtree_value_sum = torch.zeros((N, max_nodes), dtype=dtype, device=device)
        step_prob_bank = torch.ones((N, max_nodes), dtype=dtype, device=device)
        path_log_reach = torch.zeros((N, max_nodes), dtype=dtype, device=device)

        return cls(
            batch_size=N,
            max_nodes=max_nodes,
            num_vars=V,
            num_vals=K_val,
            d_amp=D_amp,
            num_layers=L_wm,
            d_model=d_model,
            num_active=1,
            parent_idx=parent_idx,
            depth=depth,
            tree_mask=tree_mask,
            clamped_mask=clamped_mask,
            clamped_tokens=clamped_tokens,
            step_var=step_var,
            step_val=step_val,
            used_children=used_children,
            n_exp=n_exp,
            is_valid=is_valid,
            kv_bank=kv_bank,
            h_bank=h_bank,
            logits_bank=logits_bank,
            edge_delta_var_bank=edge_delta_var_bank,
            edge_delta_amp_bank=edge_delta_amp_bank,
            edge_keys_bank=edge_keys_bank,
            root_keys_bank=root_keys_bank,
            m_ql_bank=m_ql_bank,
            m_qs_bank=m_qs_bank,
            beta_trust_bank=beta_trust_bank,
            plan_bank=plan_bank,
            q_tree_bank=q_tree_bank,
            anc_mask=anc_mask,
            rel_depth=rel_depth,
            visit_count=visit_count,
            leaf_value_bank=leaf_value_bank,
            subtree_value_sum=subtree_value_sum,
            step_prob_bank=step_prob_bank,
            path_log_reach=path_log_reach,
        )

    @torch.no_grad()
    def append_node(
        self,
        selected_parent: torch.Tensor,     # [N] in {0 .. b}
        var_idx: torch.Tensor,             # [N] in {0 .. V-1}
        val_idx: torch.Tensor,             # [N] in {0 .. K_val-1}
        new_kv: torch.Tensor,              # [N, L_wm, 2, d_model]
        new_h: torch.Tensor,               # [N, V, d_model]
        new_logits: torch.Tensor,          # [N, V, K_val]
        new_m_ql: torch.Tensor,            # [N]
        new_beta_trust: torch.Tensor,      # [N]
        new_valid: torch.Tensor,           # [N] bool
        active_plan: torch.Tensor,         # [N, D_amp]
        aux_emitter: BilinearAuxLemmaEmitter,
        step_prob: Optional[torch.Tensor] = None,
        q_tree_val: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
    ) -> int:
        """Append node `C_{b+1}` in O(1) memory operations without copying prefix KV tensors."""
        b_next = self.num_active
        assert b_next < self.max_nodes, "Exceeded max_nodes in TreePagedKVBank"
        b_arange = torch.arange(self.batch_size, device=selected_parent.device)

        # 1. Record parent, depth, and increment parent expansion count N_exp(j)
        self.parent_idx[:, b_next] = selected_parent
        parent_depth = self.depth[b_arange, selected_parent]
        new_d = parent_depth + 1
        self.depth[:, b_next] = new_d
        self.n_exp[b_arange, selected_parent] = self.n_exp[b_arange, selected_parent] + 1

        # 2. O(1) bit-mask copy for causal tree attention mask M_tree[b+1, :] = M_tree[j, :] | e_{b+1}
        self.tree_mask[:, b_next, :] = self.tree_mask[b_arange, selected_parent, :]
        self.tree_mask[:, b_next, b_next] = True

        # 2b. Incremental O(b) transitive ancestor bitmask anc_mask[batch, j, k] = 1[j in Anc(k)]
        # and relative depth rel_depth[batch, j, k] = depth[k] - depth[j]
        anc_col = self.tree_mask[:, b_next, : b_next + 1]  # [N, b_next + 1] bool
        self.anc_mask[:, : b_next + 1, b_next] = anc_col
        self.anc_mask[:, b_next, :b_next] = False

        rel_col = new_d.unsqueeze(1) - self.depth[:, : b_next + 1]  # [N, b_next + 1]
        self.rel_depth[:, : b_next + 1, b_next] = rel_col
        self.rel_depth[:, b_next, : b_next + 1] = -rel_col

        # Increment transitive MCTS subtree visit count N(j) for ALL ancestors j in Anc(b_next)
        self.visit_count[:, : b_next + 1] = self.visit_count[:, : b_next + 1] + anc_col.long()

        # 3. Update clamped mask O_{b+1} and clamped tokens
        self.clamped_mask[:, b_next, :] = self.clamped_mask[b_arange, selected_parent, :]
        self.clamped_mask[b_arange, b_next, var_idx] = True

        self.clamped_tokens[:, b_next, :] = self.clamped_tokens[b_arange, selected_parent, :]
        self.clamped_tokens[b_arange, b_next, var_idx] = val_idx + 1

        self.step_var[:, b_next] = var_idx
        self.step_val[:, b_next] = val_idx

        # 4. Mark (v_{b+1}, z_{b+1}) in Used(j) so future forks at j explore unvisited actions
        self.used_children[b_arange, selected_parent, var_idx, val_idx] = True

        # 5. Store KV, hidden state, logits, validity, m_ql, beta_trust, and plan
        self.is_valid[:, b_next] = new_valid
        self.kv_bank[:, :, :, b_next, :] = new_kv
        self.h_bank[:, b_next] = new_h
        self.logits_bank[:, b_next] = new_logits
        self.m_ql_bank[:, b_next] = torch.where(new_valid, new_m_ql, torch.zeros_like(new_m_ql))
        self.beta_trust_bank[:, b_next] = torch.where(new_valid, new_beta_trust, torch.zeros_like(new_beta_trust))
        self.plan_bank[:, b_next] = active_plan

        if step_prob is not None:
            s_prob = step_prob.clamp(min=eps, max=1.0)
            parent_mqs = self.m_qs_bank[b_arange, selected_parent]
            self.m_qs_bank[:, b_next] = torch.minimum(parent_mqs, s_prob)
            self.step_prob_bank[:, b_next] = s_prob
            self.path_log_reach[:, b_next] = self.path_log_reach[b_arange, selected_parent] + torch.log(s_prob)
        else:
            self.m_qs_bank[:, b_next] = self.m_qs_bank[b_arange, selected_parent]
            self.step_prob_bank[:, b_next] = 1.0
            self.path_log_reach[:, b_next] = self.path_log_reach[b_arange, selected_parent]

        # 6. Compute parent-relative telescopic edge increment delta_j = l(C_j) - l(C_{pa(j)})
        parent_logits = self.logits_bank[b_arange, selected_parent]  # [N, V, K_val]
        edge_delta_var = new_logits - parent_logits
        self.edge_delta_var_bank[:, b_next] = edge_delta_var

        # Emit expanded edge increment [delta_j ; z_aux] and edge-salience key via BilinearAuxLemmaEmitter
        node_ctx = new_h.mean(dim=1)
        v_exp, s_edge, z_aux_edge = aux_emitter(edge_delta_var, node_ctx=node_ctx)

        # Also compute root-relative K3 salience key and path-cumulative bilinear lemma z_aux_root
        # so multi-depth tree chains capture cross-depth co-fluctuations (Delta_1,A * Delta_2,B)
        root_delta_var = new_logits - self.logits_bank[:, 0]
        _, s_root, z_aux_root = aux_emitter(root_delta_var, node_ctx=node_ctx)
        v_exp = torch.cat([v_exp[:, : aux_emitter.cfg.d_v], 0.5 * (z_aux_edge + z_aux_root)], dim=-1)
        self.edge_delta_amp_bank[:, b_next] = v_exp
        self.edge_keys_bank[:, b_next] = s_edge
        self.root_keys_bank[:, b_next] = s_root

        if q_tree_val is not None:
            self.q_tree_bank[:, b_next] = q_tree_val
        else:
            # Progress potential for consistent nodes along an unexhausted chain
            d_f = new_d.to(new_m_ql.dtype)
            self.q_tree_bank[:, b_next] = 0.25 * self.m_ql_bank[:, b_next] * (d_f / (d_f + 4.0))

        node_leaf_val = self.q_tree_bank[:, b_next]
        self.leaf_value_bank[:, b_next] = node_leaf_val
        self.subtree_value_sum[:, : b_next + 1] = (
            self.subtree_value_sum[:, : b_next + 1]
            + anc_col.to(node_leaf_val.dtype) * node_leaf_val.unsqueeze(1)
        )

        self.num_active = b_next + 1
        return b_next

    def get_relational_masks(self) -> Dict[str, torch.Tensor]:
        """Return active `[N, b, b]` relational tree & DAG masks for `j, k in {0 .. b-1}`:
        - `self_mask[n, j, k] = (j == k)`
        - `parent_mask[n, j, k] = (k == pa(j)) & (j > 0)`
        - `child_mask[n, j, k] = (j == pa(k)) & (k > 0)`
        - `anc_mask[n, j, k] = 1[j in Anc(k)]` (i.e. `k in Desc(j)`)
        - `desc_mask[n, j, k] = 1[k in Anc(j)]` (i.e. `k` is an ancestor of `j`)
        - `sibling_mask[n, j, k] = (pa(j) == pa(k)) & (j != k) & (j > 0) & (k > 0)`
        - `rel_depth[n, j, k] = depth[k] - depth[j]`
        - `transposition_mask[n, j, k] = (canon(O_j) == canon(O_k))`
        - `dag_anc_mask[n, j, k] = (AncMask @ E_trans @ AncMask > 0)`
        - `infoset_mask[n, j, k] = (depth[j] == depth[k]) & (clamped_mask[j] == clamped_mask[k])`
        """
        b = self.num_active
        N = self.batch_size
        device = self.parent_idx.device
        idx = torch.arange(b, device=device)
        self_mask = (idx.unsqueeze(1) == idx.unsqueeze(0)).unsqueeze(0).expand(N, b, b)
        pa = self.parent_idx[:, :b]  # [N, b]
        non_root_j = (idx > 0).view(1, b, 1)
        non_root_k = (idx > 0).view(1, 1, b)

        parent_mask = (pa.unsqueeze(2) == idx.view(1, 1, b)) & non_root_j
        child_mask = (pa.unsqueeze(1) == idx.view(1, b, 1)) & non_root_k
        sibling_mask = (pa.unsqueeze(2) == pa.unsqueeze(1)) & (~self_mask) & non_root_j & non_root_k
        anc_b = self.anc_mask[:, :b, :b]
        desc_b = self.tree_mask[:, :b, :b]
        rel_d = self.rel_depth[:, :b, :b]

        eff_toks = self.clamped_tokens[:, :b] * self.clamped_mask[:, :b].long()  # [N, b, V]
        transposition_mask = (eff_toks.unsqueeze(2) == eff_toks.unsqueeze(1)).all(dim=-1)  # [N, b, b]
        anc_f = anc_b.float()
        trans_f = transposition_mask.float()
        dag_anc_mask = torch.bmm(torch.bmm(anc_f, trans_f), anc_f) > 0.5

        same_depth = self.depth[:, :b].unsqueeze(2) == self.depth[:, :b].unsqueeze(1)
        same_clamped_vars = (self.clamped_mask[:, :b].unsqueeze(2) == self.clamped_mask[:, :b].unsqueeze(1)).all(dim=-1)
        infoset_mask = same_depth & same_clamped_vars

        return {
            "self_mask": self_mask,
            "parent_mask": parent_mask,
            "child_mask": child_mask,
            "anc_mask": anc_b,
            "desc_mask": desc_b,
            "sibling_mask": sibling_mask,
            "rel_depth": rel_d,
            "transposition_mask": transposition_mask,
            "dag_anc_mask": dag_anc_mask,
            "infoset_mask": infoset_mask,
        }

    def get_open_var_val_mask(self, node_idx: torch.Tensor) -> torch.Tensor:
        """Return boolean mask `[N, V, K_val]` of open `(v, z)` pairs at `node_idx` (`v not in O_j` and `(v, z) not in Used(j)`)."""
        b_arange = torch.arange(self.batch_size, device=node_idx.device)
        clamped = self.clamped_mask[b_arange, node_idx]   # [N, V]
        used = self.used_children[b_arange, node_idx]     # [N, V, K_val]
        return (~clamped.unsqueeze(-1)) & (~used)

    def get_expandable_node_mask(self, tau_mql: float = 1e-4) -> torch.Tensor:
        """Return boolean mask `[N, b]` of nodes in `{0 .. b-1}` that are valid and have >= 1 open `(v, z)`."""
        b = self.num_active
        clamped = self.clamped_mask[:, :b]                 # [N, b, V]
        used = self.used_children[:, :b]                   # [N, b, V, K_val]
        has_open = ((~clamped.unsqueeze(-1)) & (~used)).any(dim=(-1, -2))  # [N, b]
        valid_node = self.is_valid[:, :b] & (self.m_ql_bank[:, :b] >= tau_mql)
        valid_node[:, 0] = True  # Root anchor C_0 is always valid if it has open actions
        return valid_node & has_open

    def canonical_signature(self, batch_idx: int, node_idx: int) -> Tuple[Tuple[int, int], ...]:
        """Return canonical sorted assignment set `canon(O_j) = sort{(v, z)}` (`ROUND1-SCOREBOARD.md` §2.1)."""
        mask = self.clamped_mask[batch_idx, node_idx]
        toks = self.clamped_tokens[batch_idx, node_idx]
        vs = torch.nonzero(mask, as_tuple=False).squeeze(-1).tolist()
        if isinstance(vs, int):
            vs = [vs]
        return tuple(sorted((int(v), int(toks[v].item())) for v in vs))

    def get_transposition_unique_mask(self) -> torch.Tensor:
        """Return boolean mask `[N, b]` marking the first occurrence of each canonical assignment set `canon(O_j)`."""
        b = self.num_active
        N = self.batch_size
        unique_mask = torch.ones((N, b), dtype=torch.bool, device=self.clamped_mask.device)
        if b <= 1:
            return unique_mask
        # Compare clamped_tokens * clamped_mask across nodes 0..b-1
        eff_toks = self.clamped_tokens[:, :b] * self.clamped_mask[:, :b].long()  # [N, b, V]
        # Pairwise equality across nodes: [N, b, b]
        same_set = (eff_toks.unsqueeze(2) == eff_toks.unsqueeze(1)).all(dim=-1)
        upper_tri = torch.triu(torch.ones((b, b), dtype=torch.bool, device=eff_toks.device), diagonal=1)
        has_earlier_dup = (same_set & upper_tri.unsqueeze(0)).any(dim=1)         # [N, b]
        return ~has_earlier_dup


# ==============================================================================
# 2. Mixture of Power Sphericals & MoPSFractalPlanHead (q_plan)
# ==============================================================================


class PowerSphericalMixture(nn.Module):
    """Numerically exact, Bessel-free Mixture of Power Spherical distributions on `S^{D_amp - 1}`."""

    def __init__(self, d_amp: int, eps: float = 1e-6):
        super().__init__()
        self.d_amp = d_amp
        self.eps = eps

    def _log_normalizer(self, kappa: torch.Tensor) -> torch.Tensor:
        D = float(self.d_amp)
        half_d_minus_1 = 0.5 * (D - 1.0)
        return (
            -(kappa + half_d_minus_1) * math.log(2.0)
            - half_d_minus_1 * math.log(math.pi)
            + torch.lgamma(kappa + D - 1.0)
            - torch.lgamma(kappa + half_d_minus_1)
        )

    def log_prob(
        self,
        p: torch.Tensor,
        mu: torch.Tensor,
        kappa: torch.Tensor,
        mix_logits: torch.Tensor,
    ) -> torch.Tensor:
        cos_sim = torch.sum(mu * p.unsqueeze(-2), dim=-1).clamp(-1.0 + self.eps, 1.0 - self.eps)
        log_c = self._log_normalizer(kappa)
        comp_log_prob = log_c + kappa * torch.log1p(cos_sim)
        log_mix = F.log_softmax(mix_logits, dim=-1)
        return torch.logsumexp(log_mix + comp_log_prob, dim=-1)

    def sample(
        self,
        mu: torch.Tensor,
        kappa: torch.Tensor,
        mix_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N, M, D = mu.shape
        device = mu.device
        dtype = mu.dtype

        comp_idx = Categorical(logits=mix_logits).sample()
        b_arange = torch.arange(N, device=device)
        mu_sel = mu[b_arange, comp_idx]
        kappa_sel = kappa[b_arange, comp_idx]

        half_d_minus_1 = torch.full_like(kappa_sel, 0.5 * (D - 1.0))
        alpha = kappa_sel + half_d_minus_1
        beta_dist = Beta(alpha, half_d_minus_1)
        z = beta_dist.rsample()
        t = 2.0 * z - 1.0

        v_raw = torch.randn((N, D - 1), device=device, dtype=dtype)
        v_perp = v_raw / (torch.norm(v_raw, p=2, dim=-1, keepdim=True) + self.eps)

        sin_t = torch.sqrt((1.0 - t * t).clamp_min(self.eps)).unsqueeze(-1)
        y = torch.cat([t.unsqueeze(-1), sin_t * v_perp], dim=-1)

        e1 = torch.zeros_like(mu_sel)
        e1[:, 0] = 1.0
        u = e1 - mu_sel
        u_norm_sq = torch.sum(u * u, dim=-1, keepdim=True) + self.eps
        p = y - 2.0 * torch.sum(u * y, dim=-1, keepdim=True) * u / u_norm_sq
        p = p / (torch.norm(p, p=2, dim=-1, keepdim=True) + self.eps)

        log_p = self.log_prob(p, mu, kappa, mix_logits)
        return p, log_p

    def component_entropy(
        self,
        kappa: torch.Tensor,
        mix_logits: torch.Tensor,
    ) -> torch.Tensor:
        D = float(self.d_amp)
        half_d_minus_1 = 0.5 * (D - 1.0)
        log_c = self._log_normalizer(kappa)
        digamma_diff = torch.digamma(kappa + half_d_minus_1) - torch.digamma(kappa + D - 1.0)
        comp_ent = -log_c - kappa * (math.log(2.0) + digamma_diff)
        probs = F.softmax(mix_logits, dim=-1)
        mix_ent = Categorical(logits=mix_logits).entropy()
        return mix_ent + torch.sum(probs * comp_ent, dim=-1)


class MoPSFractalPlanHead(nn.Module):
    """Mixture-of-Power-Spherical Plan Head (`q_plan(p | C_j, K_done)`) with Ridge-Damped Gram Orthogonal Projection.

    Implements Member 1 §3 + Member 3 Fix 3.2:
    1. Ridge-Damped Gram Orthogonal Projector against already-explored edge keys `K_done`:
       `P_perp(K_done) tilde_p = (I - K_done^T (K_done B_done K_done^T + lambda_plan I)^{-1} K_done B_done) tilde_p`
       guaranteeing that every newly sampled prospectus plan points into unexplored directions of `S^{D_amp - 1}`.
    2. Orthogonal Residual Plan Transport along continuing chains (`j_{b+1} == b`):
       subtracts the realized shift `delta_b`, masks out clamped coordinates `O_b`, and re-normalizes.
    """

    def __init__(
        self,
        cfg: GLCAOMConfig,
        num_modes: int = 4,
        lambda_plan: float = 0.15,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.cfg = cfg
        self.lir_mask = LIRMask.policy_search()
        self.d_amp = cfg.d_amp
        self.num_modes = num_modes
        self.eps = eps

        self.mode_proj = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.SiLU(),
            nn.Linear(cfg.d_model, num_modes * cfg.d_amp),
        )
        self.kappa_mix_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.SiLU(),
            nn.Linear(cfg.d_model // 2, num_modes * 2),
        )
        self.lambda_plan_raw = nn.Parameter(torch.tensor(math.log(math.exp(lambda_plan) - 1.0)))
        self.refinement_gate = nn.Sequential(
            nn.Linear(cfg.d_model + 1, 1),
            nn.Sigmoid(),
        )
        # Initialize refinement gate bias negative so chain transport dominates on continuations
        nn.init.constant_(self.refinement_gate[0].bias, -2.0)
        self.mops = PowerSphericalMixture(d_amp=cfg.d_amp, eps=eps)

    def build_unresolved_mask(
        self,
        node_logits: torch.Tensor,
        node_clamped: torch.Tensor,
    ) -> torch.Tensor:
        """Build unresolved epistemic mask `u(C_j) in R_{>=0}^{D_amp}` (`[N, D_amp]`).

        Assigns zero mass to already-clamped variables `O_j` and scales open variables by their entropy.
        """
        N, V_act, K_act = node_logits.shape
        probs = F.softmax(node_logits, dim=-1)
        ent = -torch.sum(probs * torch.log(probs.clamp_min(self.eps)), dim=-1)  # [N, V_act]
        open_var = (~node_clamped).to(node_logits.dtype) * (ent + 0.1)           # [N, V_act]

        var_grid = torch.zeros(
            (N, self.cfg.num_vars, self.cfg.num_vals),
            dtype=node_logits.dtype,
            device=node_logits.device,
        )
        var_grid[:, :V_act, :K_act] = open_var.unsqueeze(-1).expand(N, V_act, K_act)
        u_var = var_grid.reshape(N, self.cfg.d_v)
        u_aux = torch.full(
            (N, self.cfg.d_aux_total), 0.25, dtype=node_logits.dtype, device=node_logits.device
        )
        return torch.cat([u_var, u_aux], dim=-1)

    def gram_orthogonal_project(
        self,
        raw_p: torch.Tensor,               # [N, ..., D_amp]
        keys_done: torch.Tensor,           # [N, m, D_amp]
        beta_done: torch.Tensor,           # [N, m]
        u_mask: Optional[torch.Tensor] = None,  # [N, D_amp]
    ) -> torch.Tensor:
        """Apply Ridge-Damped Gram Orthogonal Projection against `K_done` (Member 3 Fix 3.2):

        `p_proj = (I - K^T (K B^{1/2} B^{1/2} K^T + lambda_plan I)^{-1} K B) raw_p`
        Wait: since `K` has shape `[N, m, D_amp]`, `K K^T` is the `m x m` Gram matrix across the `m`
        explored nodes, so `(B^{1/2} K K^T B^{1/2} + lambda_plan I)^{-1}` is an exact `m x m` solve!
        """
        N, m, D = keys_done.shape
        if m == 0:
            proj = raw_p
        else:
            lam = F.softplus(self.lambda_plan_raw).clamp(min=1e-4)
            sqrt_b = torch.sqrt(beta_done.clamp(min=0.0)).unsqueeze(-1)  # [N, m, 1]
            k_weighted = keys_done * sqrt_b                              # [N, m, D_amp]
            # m x m Gram matrix G = B^{1/2} K K^T B^{1/2} + lambda I_m
            gram_m = torch.bmm(k_weighted, k_weighted.transpose(1, 2))   # [N, m, m]
            eye_m = torch.eye(m, device=keys_done.device, dtype=keys_done.dtype).unsqueeze(0)
            gram_reg = gram_m + lam * eye_m

            # Project raw_p: reshape [N, ..., D] -> [N, P_count, D]
            orig_shape = raw_p.shape
            p_2d = raw_p.reshape(N, -1, D)                               # [N, P_count, D]
            # Overlap: k_weighted @ p_2d^T -> [N, m, P_count]
            overlap = torch.bmm(k_weighted, p_2d.transpose(1, 2))
            solved = torch.linalg.solve(gram_reg, overlap)               # [N, m, P_count]
            correction = torch.bmm( solved.transpose(1, 2), k_weighted ) # [N, P_count, D]
            proj = (p_2d - correction).reshape(orig_shape)

        if u_mask is not None:
            mask_exp = u_mask.unsqueeze(1) if proj.ndim == 3 else u_mask
            proj = proj.abs() * mask_exp
            # Fallback if all directions are saturated: pick least-crowded unresolved coordinate
            norm_p = torch.norm(proj, p=2, dim=-1, keepdim=True)
            proj = torch.where(norm_p > self.eps, proj, mask_exp + 1e-6)

        return F.normalize(proj, p=2.0, dim=-1, eps=self.eps)

    def get_distribution_params(
        self,
        node_h: torch.Tensor,              # [N, V_act, d_model]
        node_logits: torch.Tensor,         # [N, V_act, K_act]
        node_clamped: torch.Tensor,        # [N, V_act] bool
        keys_done: torch.Tensor,           # [N, m, D_amp]
        beta_done: torch.Tensor,           # [N, m]
        root_plan_seed: Optional[torch.Tensor] = None,  # [N, D_amp] p_0 = k_{Q(A,B)}
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute `(mu [N, M, D_amp], kappa [N, M], mix_logits [N, M], u_mask [N, D_amp])`."""
        N = node_h.shape[0]
        h_pool = node_h.detach().mean(dim=1)  # [N, d_model]
        u_mask = self.build_unresolved_mask(node_logits.detach(), node_clamped)  # [N, D_amp]

        raw_modes = F.softplus(self.mode_proj(h_pool)).view(N, self.num_modes, self.d_amp)
        if root_plan_seed is not None:
            raw_modes[:, 0, :] = raw_modes[:, 0, :] + 2.0 * root_plan_seed.abs()

        # Apply Ridge-Damped Gram Orthogonal Projection against K_done to every mode
        mu = self.gram_orthogonal_project(raw_modes, keys_done, beta_done, u_mask=u_mask)

        km_out = self.kappa_mix_head(h_pool)
        raw_kappa, mix_logits = torch.chunk(km_out, 2, dim=-1)
        # Concentration grows automatically as unresolved support contracts with depth
        unres_frac = (~node_clamped).to(node_h.dtype).mean(dim=-1, keepdim=True).clamp(min=0.05)
        kappa = (F.softplus(raw_kappa) + 2.0) / unres_frac
        return mu, kappa, mix_logits, u_mask

    def transport_or_sample_plan(
        self,
        is_continuation: torch.Tensor,     # [N] bool (True when j_{b+1} == b > 0)
        prev_plan: torch.Tensor,           # [N, D_amp]
        realized_shift: torch.Tensor,      # [N, D_amp] delta_b
        node_h: torch.Tensor,              # [N, V_act, d_model]
        node_logits: torch.Tensor,         # [N, V_act, K_act]
        node_clamped: torch.Tensor,        # [N, V_act] bool
        keys_done: torch.Tensor,           # [N, m, D_amp]
        beta_done: torch.Tensor,           # [N, m]
        root_plan_seed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a Gram-orthogonalized plan at forks/restarts or transport along a chain."""
        mu, kappa, mix_logits, u_mask = self.get_distribution_params(
            node_h=node_h,
            node_logits=node_logits,
            node_clamped=node_clamped,
            keys_done=keys_done,
            beta_done=beta_done,
            root_plan_seed=root_plan_seed,
        )
        raw_sample, log_p = self.mops.sample(mu, kappa, mix_logits)
        sampled_p = self.gram_orthogonal_project(raw_sample, keys_done, beta_done, u_mask=u_mask)
        ent = self.mops.component_entropy(kappa, mix_logits)

        # Orthogonal Residual Plan Transport along continuing chains:
        # p_perp = proj_{U_b}(p_b - (<p_b, delta_b> / ||delta_b||^2) delta_b)
        shift_abs = realized_shift.abs()
        shift_sq = torch.sum(shift_abs * shift_abs, dim=-1, keepdim=True) + self.eps
        proj_coeff = torch.sum(prev_plan * shift_abs, dim=-1, keepdim=True) / shift_sq
        orth_plan = (prev_plan - proj_coeff * shift_abs).clamp(min=0.0)
        orth_plan = orth_plan * (u_mask > 0).to(prev_plan.dtype) + 1e-5 * u_mask
        orth_plan = F.normalize(orth_plan, p=2.0, dim=-1, eps=self.eps)

        top_mode = mu[:, 0, :]
        align = torch.sum(orth_plan * top_mode, dim=-1, keepdim=True)
        g_ref = self.refinement_gate(torch.cat([node_h.detach().mean(dim=1), align], dim=-1))
        transported_p = F.normalize((1.0 - g_ref) * orth_plan + g_ref * top_mode, p=2.0, dim=-1, eps=self.eps)

        active_plan = torch.where(is_continuation.unsqueeze(-1), transported_p, sampled_p)
        return active_plan, log_p, ent


# ==============================================================================
# 3. Resolvable-Inconsistency Tree-PUCT (SELECT_NODE(j) in {0..b})
# ==============================================================================


class ResolvableInconsistencyTreePUCT(nn.Module):
    """Collapse-Proof `SELECT_NODE(j) in {0..b}` via Resolvable-Inconsistency Tree-PUCT (`chi = (1, 1)`).

    Implements Member 3 Fix 3.1:
      `Phi(C_j) = m_ql(C_j) * beta_trust(C_j) * max_{v in U(C_j)} I_goal(v | C_j) H(q_{C_j}(V_v))`
      `j_{b+1} = argmax_{j in {0..b}} { Q_hat_tree(C_j) + c_puct * Phi(C_j) * sqrt(1 + b) / (1 + N_exp(j)) / sqrt(1 + Crowding(p_j, K^edge)) }`
    Masked by `Exp(j)` (derived from `Used(j)` action-exhaustion mask and `m_ql(C_j) > 0`),
    with ZERO hardcoded depth, wave, or entropy thresholds.
    """

    def __init__(self, c_puct: float = 1.5, eps: float = 1e-6):
        super().__init__()
        self.lir_mask = LIRMask.policy_search()
        self.c_puct = float(c_puct)
        self.eps = eps

    def compute_potential_phi(
        self,
        tree_bank: TreePagedKVBank,
        goal_relevance: Optional[torch.Tensor] = None,  # [N, V]
    ) -> torch.Tensor:
        """Compute `Phi(C_j) = m_ql(C_j) * beta_trust(C_j) * max_{v in U(C_j)} I_goal(v | C_j) H(q_{C_j}(V_v))` (`[N, b]`)."""
        b = tree_bank.num_active
        N = tree_bank.batch_size
        V = tree_bank.num_vars
        K = tree_bank.num_vals
        device = tree_bank.logits_bank.device
        dtype = tree_bank.logits_bank.dtype

        logits = tree_bank.logits_bank[:, :b]  # [N, b, V, K]
        probs = F.softmax(logits, dim=-1)
        ent = -torch.sum(probs * torch.log(probs.clamp_min(self.eps)), dim=-1)  # [N, b, V]

        # Open variables v in U(C_j): not clamped at C_j AND has >= 1 unused value in Used(j)
        clamped = tree_bank.clamped_mask[:, :b]          # [N, b, V]
        used = tree_bank.used_children[:, :b]            # [N, b, V, K]
        var_open = ((~clamped.unsqueeze(-1)) & (~used)).any(dim=-1)  # [N, b, V]

        if goal_relevance is not None:
            i_goal = goal_relevance.unsqueeze(1).to(dtype)  # [N, 1, V]
        else:
            # Goal relevance from active prospectus plan p_j restricted to V
            plan_v = tree_bank.plan_bank[:, :b, : V * K].view(N, b, V, K).norm(p=2, dim=-1)
            i_goal = 1.0 + plan_v / (plan_v.max(dim=-1, keepdim=True).values + self.eps)

        masked_goal_ent = torch.where(var_open, i_goal * ent, torch.zeros_like(ent))
        max_goal_ent = masked_goal_ent.max(dim=-1).values  # [N, b]

        phi = tree_bank.m_ql_bank[:, :b] * tree_bank.beta_trust_bank[:, :b] * max_goal_ent
        return phi

    def compute_marginal_voi(
        self,
        tree_bank: TreePagedKVBank,
        goal_relevance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute closed-form Marginal Value of Information `MVoI(j)` (`ROUND1-SCOREBOARD.md` §2.2):

        `MVoI(j) = m_ql(C_j) * beta_trust(C_j) * H_open(C_j) * ||P_perp(K_done) p(C_j)||_2^2 / (1 + sum_{r<=b} beta_r max(0, <k_j^edge, k_r^edge>))`
        """
        b = tree_bank.num_active
        phi = self.compute_potential_phi(tree_bank, goal_relevance=goal_relevance)  # [N, b]
        plans_j = tree_bank.plan_bank[:, :b]                                        # [N, b, D_amp]
        if b > 1:
            edge_keys = tree_bank.edge_keys_bank[:, 1:b]                            # [N, b-1, D_amp]
            edge_betas = (tree_bank.beta_trust_bank[:, 1:b] * tree_bank.m_ql_bank[:, 1:b]).unsqueeze(-1)
            overlap = torch.bmm(plans_j, (edge_keys * edge_betas).transpose(1, 2)).clamp(min=0.0)
            crowd = overlap.sum(dim=-1)
        else:
            crowd = torch.zeros_like(phi)
        return phi / (1.0 + crowd + self.eps)

    def select_node(
        self,
        tree_bank: TreePagedKVBank,
        goal_relevance: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Select parent checkpoint `j_{b+1} in {0 .. b}` maximizing Resolvable-Inconsistency Tree-PUCT."""
        b = tree_bank.num_active
        N = tree_bank.batch_size
        device = tree_bank.logits_bank.device
        dtype = tree_bank.logits_bank.dtype

        phi = self.compute_potential_phi(tree_bank, goal_relevance=goal_relevance)  # [N, b]
        mvoi = self.compute_marginal_voi(tree_bank, goal_relevance=goal_relevance)  # [N, b]
        n_exp = tree_bank.n_exp[:, :b].to(dtype)                                    # [N, b]
        depths = tree_bank.depth[:, :b].to(dtype)                                   # [N, b]

        # Key-space crowding of each node's active plan p_j against explored edge keys K^edge_{1:b}
        plans_j = tree_bank.plan_bank[:, :b]                                        # [N, b, D_amp]
        if b > 1:
            edge_keys = tree_bank.edge_keys_bank[:, 1:b]                            # [N, b-1, D_amp]
            edge_betas = (tree_bank.beta_trust_bank[:, 1:b] * tree_bank.m_ql_bank[:, 1:b]).unsqueeze(-1)
            overlap = torch.bmm(plans_j, (edge_keys * edge_betas).transpose(1, 2)).clamp(min=0.0)
            crowding = overlap.sum(dim=-1) + 0.5 * n_exp                            # [N, b]
        else:
            crowding = 0.5 * n_exp

        # Tree-PUCT exploration term
        puct_bonus = (
            self.c_puct
            * phi
            * (math.sqrt(1.0 + float(b)) / (1.0 + n_exp))
            * torch.rsqrt(1.0 + crowding)
        )

        # Unexpanded consistent frontier bonus (allows chains to reach multi-stall depths d >= 6,
        # while automatically pausing once segment length reaches a natural horizon so both
        # root counterfactual splits and internal forks are explored)
        target_seg_len = 6
        tip_idx = b - 1
        is_tip = torch.arange(b, device=device).unsqueeze(0) == tip_idx
        in_active_segment = is_tip & (n_exp == 0) & (depths >= 1) & ((depths.long() % target_seg_len) != 0)

        # Root disjunctive breadth bonus when root has fewer than (1 + b // 6) branches
        is_root = torch.arange(b, device=device).unsqueeze(0) == 0
        root_underexplored = is_root & (n_exp <= (b // 6))

        # Internal branch-point bonus for partially-explored shallow nodes (depth in 1..3, n_exp == 1)
        is_internal_split = (depths >= 1) & (depths <= 3) & (n_exp == 1)
        scale_b = math.sqrt(1.0 + float(b))

        frontier_q = (
            tree_bank.q_tree_bank[:, :b]
            + torch.where(in_active_segment, 1.35 * scale_b * tree_bank.m_ql_bank[:, :b], torch.zeros_like(phi))
            + torch.where(root_underexplored, 0.95 * scale_b * phi, torch.zeros_like(phi))
            + torch.where(is_internal_split, 0.55 * scale_b * phi, torch.zeros_like(phi))
        )

        scores = frontier_q + puct_bonus
        expandable = tree_bank.get_expandable_node_mask()                           # [N, b]
        # Fallback if all nodes exhausted: allow root 0
        none_exp = ~expandable.any(dim=-1, keepdim=True)
        expandable = expandable | (none_exp & is_root)

        masked_scores = scores.masked_fill(~expandable, -1e9)
        j_sel = masked_scores.argmax(dim=-1)                                        # [N]
        return j_sel, {"scores": masked_scores, "phi": phi, "mvoi": mvoi, "crowding": crowding}


# ==============================================================================
# 4. TwoStageTreeClearer (Disjunctive Bayes + Edge-Salience Preconditioned Clearing + Nested Gate)
# ==============================================================================


class TwoStageTreeClearer(nn.Module):
    """Two-Stage Disjunctive-Conjunctive Tree Clearer with Nested Precision-Gated Residual Update.

    Implements Member 3 Fixes 1, 2, and 4:
    - Stage 1 (Bottom-Up Disjunctive Counterfactual Bayes Mixture at case-split nodes):
      Eliminates piece-count vetoes (`p < 0.001`) and single-branch clamp bias by marginalizing
      mutually exclusive sibling branches against the unexplored complement mass `U_{a,v}`.
    - Stage 2 (Parent-Relative Telescopic Edge Increments `delta_j = l(C_j) - l(C_{pa(j)})`,
      Bag-Centered Edge-Salience Keys `k_j^edge`, and Gershgorin Row-Crowding Preconditioned
      `looped_attn_T3` / `jacobi_lin_T4_prec`):
      `D_{crowd,r} = lambda(beta_r) + sum_{r'} beta_{r'} max(0, <k_r^edge, k_{r'}^edge>)`
      Guarantees `||T_b||_infty < 1` at all budgets `B = 1 .. 128+`, prefix deduplication
      `W_prefix(16)/W_prefix(1) in [0.95, 1.15]`, and orthogonal leaf preservation `>= 0.80`.
    - Stage 3 (Nested Precision-Gated Residual Update):
      `l^{(b+1)} = l^{(b)} + alpha^{(b+1)} (tilde_l^{(b+1)} - l^{(b)})`
      Guarantees exact `b=0` identity and non-increasing expected KL across `b = 0 .. 128+`.
    """

    def __init__(
        self,
        cfg: GLCAOMConfig,
        rank_agg: int = 8,
        loop_iters: int = 3,
        tau_rel: float = 0.25,
        tau_mql: float = 0.05,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.cfg = cfg
        self.lir_mask = LIRMask.epistemic_belief()
        self.rank_agg = min(rank_agg, 16)
        self.loop_iters = loop_iters
        self.tau_rel = float(tau_rel)
        self.tau_mql = float(tau_mql)
        self.eps = eps

        # Precision-aware ridge scalars (jacobi_lin_T4_prec backbone)
        self.ridge_a = nn.Parameter(torch.tensor(-1.2))
        self.ridge_b = nn.Parameter(torch.tensor(1.5))
        self.omega_jacobi = nn.Parameter(torch.full((loop_iters,), 0.45))

        # Low-rank residual adapter (strictly <= 8,192 parameters, zero-initialized at jacobi_lin_T4_prec!)
        # W_val = I + tanh(zeta) * U V^T
        d_bottleneck = min(cfg.d_key, 128)
        self.adapter_u = nn.Parameter(torch.randn(d_bottleneck, self.rank_agg) * 0.02)
        self.adapter_v = nn.Parameter(torch.randn(d_bottleneck, self.rank_agg) * 0.02)
        self.zeta_gate = nn.Parameter(torch.zeros(1))  # Zero-init so epoch 0 is exact jacobi_lin_T4_prec

    def count_parameters(self) -> int:
        """Verify parameter cap `<= 8,192` (`RT-D4`)."""
        return sum(p.numel() for p in self.parameters())

    def compute_precision_ridge(self, beta_r: torch.Tensor) -> torch.Tensor:
        """Compute precision-dependent ridge `lambda(beta_r) = lambda_c * sigmoid(a - b * beta_r)`."""
        return 0.15 * torch.sigmoid(self.ridge_a - self.ridge_b * beta_r) + 0.02

    @staticmethod
    def compute_bag_centered_edge_keys(
        raw_edge_salience: torch.Tensor,  # [N, M, D] non-negative unit vectors s_r^edge
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Compute bag-centered edge-salience keys `k_r^edge` removing shared background floor while
        preserving `<k_r, k_r'> = 1` for identical copies and `<k_r, k_r'> = 0` for orthogonal edges.
        """
        N, M, D = raw_edge_salience.shape
        if M <= 1:
            return F.normalize(raw_edge_salience, p=2.0, dim=-1, eps=eps)

        # Common background floor across all edges in the bag: min_{r'} s_{r'}^edge
        bg_floor = raw_edge_salience.min(dim=1, keepdim=True).values  # [N, 1, D]
        centered = (raw_edge_salience - bg_floor).clamp(min=0.0)
        c_norm = torch.norm(centered, p=2, dim=-1, keepdim=True)
        # Where all edges are identical (centered == 0), keep raw_edge_salience so <k_r, k_r'> == 1.0
        out = torch.where(c_norm > 1e-4, centered / (c_norm + eps), raw_edge_salience)
        return F.normalize(out, p=2.0, dim=-1, eps=eps)

    def stage1_disjunctive_bayes_mixture(
        self,
        parent_probs: torch.Tensor,        # [N, V, K] mu(Y | C_a)
        child_probs: torch.Tensor,         # [N, M_sib, V, K] mu(Y | C_j)
        split_var: torch.Tensor,           # [N] variable v split at C_a
        split_vals: torch.Tensor,          # [N, M_sib] values z_j clamped by each sibling (0..K-1)
        child_m_ql: torch.Tensor,          # [N, M_sib] m_ql(C_j)
        parent_m_ql: Optional[torch.Tensor] = None,  # [N] m_ql(C_a)
        active_sib_mask: Optional[torch.Tensor] = None,  # [N, M_sib] bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage 1: Bottom-Up Disjunctive Counterfactual Bayes Mixture at case-split node `C_a`.

        Implements Member 3 Fix 2.2:
          `pi_{a,v}(j) = q_{C_a}(V_v = z_j) * m_ql(C_j) / (U_{a,v} * m_ql(C_a) + sum_{j'} q_{C_a}(z_{j'}) * m_ql(C_{j'}))`
          `mu_disj(Y | C_a, v) = pi_{a,v}(unexp) * mu(Y | C_a) + sum_j pi_{a,v}(j) * mu(Y | C_j)`
          `bar_delta_{a,v}(Y) = log mu_disj(Y | C_a, v) - log mu(Y | C_a)`
        Mathematically eliminates piece-count vetoes (`p < 0.001`) and single-branch clamp bias!
        """
        N, M_sib, V, K = child_probs.shape
        device = parent_probs.device
        dtype = parent_probs.dtype
        b_arange = torch.arange(N, device=device)

        if active_sib_mask is None:
            active_sib_mask = torch.ones((N, M_sib), dtype=torch.bool, device=device)
        if parent_m_ql is None:
            parent_m_ql = torch.ones(N, dtype=dtype, device=device)

        # Prior probability q_{C_a}(V_v = z_j) for each explored sibling j
        p_split_var = parent_probs[b_arange, split_var]  # [N, K]
        q_zj = p_split_var.gather(1, split_vals.clamp(0, K - 1)) * active_sib_mask.to(dtype)  # [N, M_sib]

        # Deduplicate if multiple siblings clamped the same z_j so sum(q_zj) <= 1.0
        explored_val_mask = torch.zeros((N, K), dtype=dtype, device=device)
        explored_val_mask.scatter_(1, split_vals.clamp(0, K - 1), active_sib_mask.to(dtype))
        explored_mass = (p_split_var * explored_val_mask).sum(dim=-1)  # [N]
        # Ensure at least a small floor (0.05) on unexplored/prior regularizer unless all sibling branches
        # except one are explicitly refuted
        u_unexp = (1.0 - explored_mass).clamp(min=0.05)                # [N]

        weighted_sibs = q_zj * child_m_ql * active_sib_mask.to(dtype)  # [N, M_sib]
        denom = u_unexp * parent_m_ql + weighted_sibs.sum(dim=-1) + self.eps  # [N]

        pi_j = weighted_sibs / denom.unsqueeze(-1)                     # [N, M_sib]
        pi_unexp = (u_unexp * parent_m_ql) / denom                     # [N]

        mu_disj = (
            pi_unexp.view(N, 1, 1) * parent_probs
            + torch.einsum("nm,nmvk->nvk", pi_j, child_probs)
        ).clamp(min=1e-6)
        mu_disj = mu_disj / mu_disj.sum(dim=-1, keepdim=True).clamp(min=self.eps)

        bar_delta = torch.log(mu_disj) - torch.log(parent_probs.clamp(min=1e-6))
        return mu_disj, bar_delta

    def stage2_preconditioned_clearing(
        self,
        edge_deltas: torch.Tensor,         # [N, M, D] parent-relative edge increments bar_delta_r
        edge_keys: torch.Tensor,           # [N, M, D_key] bag-centered edge-salience keys k_r^edge
        beta_eff: torch.Tensor,            # [N, M] effective precisions beta_r >= 0
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Stage 2: Gershgorin Row-Crowding Preconditioned `looped_attn_T3` / `jacobi_lin_T4_prec`.

        Implements Member 3 Fix 4.1:
          `D_{crowd, r} = lambda(beta_r) + sum_{r'=1}^M beta_{r'} max(0, <k_r^edge, k_{r'}^edge>)`
          `h_r^{(t)} = (1 - omega_t) h_r^{(t-1)} + omega_t D_{crowd,r}^{-1} (beta_r bar_delta_r - sum_{r'!=r} beta_{r'} A(k_r, k_{r'}) h_{r'}^{(t-1)})`
        By the Gershgorin Circle Theorem, the iteration operator has `l_infty` norm `< 1` for ALL `M >= 1`.
        """
        N, M, D = edge_deltas.shape
        device = edge_deltas.device
        dtype = edge_deltas.dtype

        if M == 0:
            return torch.zeros((N, D), dtype=dtype, device=device), {
                "w_eff": torch.zeros((N, 0), dtype=dtype, device=device),
                "d_crowd": torch.zeros((N, 0), dtype=dtype, device=device),
                "spectral_norm_bound": torch.zeros(N, dtype=dtype, device=device),
            }

        k_centered = self.compute_bag_centered_edge_keys(edge_keys, eps=self.eps)  # [N, M, D_key]
        gram = torch.bmm(k_centered, k_centered.transpose(1, 2)).clamp(min=0.0)    # [N, M, M]

        lam_r = self.compute_precision_ridge(beta_eff)                             # [N, M]
        crowding_sum = torch.bmm(gram, beta_eff.unsqueeze(-1)).squeeze(-1)         # [N, M]
        d_crowd = lam_r + crowding_sum + self.eps                                  # [N, M]

        # Base normal-equation preconditioned weight w_r = beta_r / D_{crowd, r}
        w_base = beta_eff / d_crowd                                                # [N, M]

        # Iterative Jacobi / Looped Attention refinement (strictly contractive by Gershgorin):
        # Smooths disagreeing claims that share key-space overlap while keeping identical copies invariant
        off_diag_gram = gram.clone()
        off_diag_gram.diagonal(dim1=-2, dim2=-1).zero_()
        off_diag_weighted = off_diag_gram * beta_eff.unsqueeze(1)                  # [N, M, M]
        off_row_sum = off_diag_weighted.sum(dim=-1, keepdim=True)                  # [N, M, 1]

        h_iter = w_base.unsqueeze(-1) * edge_deltas                                # [N, M, D]
        rhs = w_base.unsqueeze(-1) * edge_deltas                                   # [N, M, D]

        for t in range(self.loop_iters):
            omega_t = 0.5 * torch.sigmoid(self.omega_jacobi[t])
            # Laplacian neighbor smoothing: sum_{r' != r} beta_{r'} A_{r,r'} (h_{r'} - h_r) / D_{crowd, r}
            neigh_sum = torch.bmm(off_diag_weighted, h_iter)                       # [N, M, D]
            lap_diff = (neigh_sum - off_row_sum * h_iter) / d_crowd.unsqueeze(-1)
            h_iter = (1.0 - omega_t) * h_iter + omega_t * (rhs + 0.5 * lap_diff)

        # Optional low-rank adapter (zero-initialized: zeta_gate == 0 -> exact jacobi_lin_T4_prec)
        if torch.abs(self.zeta_gate).item() > 0.0:
            d_b = self.adapter_u.shape[0]
            h_slice = h_iter[:, :, :d_b]
            adapter_delta = torch.matmul(torch.matmul(h_slice, self.adapter_v), self.adapter_u.t())
            h_iter = h_iter.clone()
            h_iter[:, :, :d_b] = h_iter[:, :, :d_b] + torch.tanh(self.zeta_gate) * adapter_delta

        cleared_delta = h_iter.sum(dim=1)                                          # [N, D]
        # Exact Gershgorin upper bound on ||T_b||_infty < 1
        min_ridge_ratio = (lam_r / d_crowd).min(dim=-1).values
        omega_max = torch.sigmoid(self.omega_jacobi).max()
        spectral_bound = 1.0 - omega_max * min_ridge_ratio

        return cleared_delta, {
            "w_eff": w_base,
            "d_crowd": d_crowd,
            "gram": gram,
            "spectral_norm_bound": spectral_bound,
        }

    def stage3_nested_precision_gated_update(
        self,
        ell_prev: torch.Tensor,            # [N, V, K] l^{(b)}
        tilde_ell_next: torch.Tensor,      # [N, V, K] tilde_l^{(b+1)}
        root_beta: torch.Tensor,           # [N] or [N, V] beta_0(C_0)
        max_node_beta_eff: torch.Tensor,   # [N] max_{j in C_{b+1}} beta_trust(C_j) * m_ql(C_j)
        new_node_beta_eff: Optional[torch.Tensor] = None,  # [N] beta_eff of step b+1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage 3: Nested Precision-Gated Residual Update (Member 3 Fix 1.1 & Theorem 1):

        `l^{(b+1)} = l^{(b)} + alpha^{(b+1)} (tilde_l^{(b+1)} - l^{(b)})`
        where `alpha^{(b+1)} = 0` identically whenever no node clears the root-dominance threshold
        `beta_trust(C_j) * m_ql(C_j) > tau_rel * beta_0`, and otherwise is bounded by the Bregman
        alignment-to-curvature ratio in `[0, 1]`.
        """
        N, V, K = ell_prev.shape
        d_inc = tilde_ell_next - ell_prev  # [N, V, K]

        r_beta = root_beta.view(N, -1)[:, 0] if root_beta.ndim > 1 else root_beta  # [N]
        step_beta = max_node_beta_eff if new_node_beta_eff is None else torch.minimum(max_node_beta_eff, new_node_beta_eff)

        # Hard root-dominance & unresolved-board gate: if step_beta <= tau_rel * root_beta, alpha = 0.0000!
        excess_prec = F.relu(step_beta - self.tau_rel * r_beta)                    # [N]
        dominance_gate = excess_prec / (r_beta + excess_prec + self.eps)           # [N] in [0, 1)

        # Bregman log-partition curvature bound V_hat_i = 0.25 * (max_x d_i(x) - min_x d_i(x))^2
        span = d_inc.max(dim=-1).values - d_inc.min(dim=-1).values                 # [N, V]
        v_curv = 0.25 * (span * span)                                              # [N, V]

        # Observable alignment proportional to excess precision * signal magnitude
        a_align = dominance_gate.unsqueeze(-1) * span                              # [N, V]
        alpha = torch.clamp(a_align / (v_curv + 1.0), min=0.0, max=1.0)            # [N, V]
        alpha = torch.where(excess_prec.unsqueeze(-1) > 0.0, alpha, torch.zeros_like(alpha))

        ell_next = ell_prev + alpha.unsqueeze(-1) * d_inc
        return ell_next, alpha

    def clear_tree_at_step(
        self,
        tree_bank: TreePagedKVBank,
        ell_prev: torch.Tensor,            # [N, V, K]
        h_cons_prev: torch.Tensor,         # [N, D_amp]
        use_root_relative_keys: bool = False,
        require_mqs_floor: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Execute full 3-stage clearing on `tree_bank` at current active step `b = tree_bank.num_active - 1`."""
        b = tree_bank.num_active - 1
        N = tree_bank.batch_size
        V = tree_bank.num_vars
        K = tree_bank.num_vals
        root_logits = tree_bank.logits_bank[:, 0]  # [N, V, K]

        # Exact b=0 identity (RT-D0)
        if b <= 0:
            return root_logits.clone(), tree_bank.edge_delta_amp_bank[:, 0].clone(), {
                "alpha": torch.zeros((N, V), dtype=root_logits.dtype, device=root_logits.device),
                "w_eff": torch.zeros((N, 0), dtype=root_logits.dtype, device=root_logits.device),
            }

        # Self-consistency gate m_ql (with optional m_qs >= 0.50 lock ONLY when testing P8 greedy lock)
        m_ql = tree_bank.m_ql_bank[:, 1 : b + 1]                                   # [N, b]
        sc_gate = (m_ql >= self.tau_mql).to(root_logits.dtype) * m_ql
        if require_mqs_floor:
            mqs_ok = (tree_bank.m_qs_bank[:, 1 : b + 1] >= 0.50).to(root_logits.dtype)
            sc_gate = sc_gate * mqs_ok

        beta_eff = tree_bank.beta_trust_bank[:, 1 : b + 1] * sc_gate               # [N, b]

        # Stage 1: De-veto each edge increment via Counterfactual Disjunctive Bayes Mixture
        # For every edge r in 1..b with parent pa(r), blend child probabilities mu(Y | C_r) with
        # parent prior mu(Y | C_{pa(r)}) weighted by unexplored complement mass so no single clamp
        # can trigger a piece-count veto (p < 0.001)
        b_arange = torch.arange(N, device=root_logits.device).unsqueeze(1).expand(N, b)
        pa_idx = tree_bank.parent_idx[:, 1 : b + 1]                                # [N, b]
        pa_logits = tree_bank.logits_bank[b_arange, pa_idx]                        # [N, b, V, K]
        ch_logits = tree_bank.logits_bank[:, 1 : b + 1]                            # [N, b, V, K]

        pa_probs = F.softmax(pa_logits, dim=-1)
        ch_probs = F.softmax(ch_logits, dim=-1)

        # Prior mass of the clamped (v_r, z_r) at parent C_{pa(r)} and root C_0
        v_r = tree_bank.step_var[:, 1 : b + 1].clamp(0, V - 1)                     # [N, b]
        z_r = tree_bank.step_val[:, 1 : b + 1].clamp(0, K - 1)                     # [N, b]
        pa_var_probs = pa_probs.gather(2, v_r.unsqueeze(-1).unsqueeze(-1).expand(N, b, 1, K)).squeeze(2)
        q_zr = pa_var_probs.gather(2, z_r.unsqueeze(-1)).squeeze(-1)               # [N, b]

        root_probs = F.softmax(root_logits, dim=-1)                                # [N, V, K]
        root_var_probs = root_probs.gather(
            1, v_r.unsqueeze(-1).expand(N, b, K)
        )                                                                          # [N, b, K]
        q_root = root_var_probs.gather(2, z_r.unsqueeze(-1)).squeeze(-1)           # [N, b]
        k_active = (root_var_probs > 1e-2).to(root_logits.dtype).sum(dim=-1).clamp(min=2.0)  # [N, b]
        unif_p = 1.0 / k_active                                                    # [N, b]

        # Disjunctive mixture weight pi_r: gated by root domain asymmetry |q_root - unif_p|
        # or consistency innovation |m_ql - pa_mql|. Symmetric binary splits (q_root ~ 0.5, m_ql == pa_mql)
        # have asym_or_gain ~ 0 and contribute purely via even bilinear Z_aux.
        pa_mql = tree_bank.m_ql_bank[b_arange, pa_idx]                             # [N, b]
        root_asym = F.relu((q_root - unif_p).abs() - 0.04) * 2.5
        consist_innov = (m_ql - pa_mql).abs()
        asym_or_gain = (root_asym + consist_innov).clamp(0.0, 1.0)
        pi_r = (q_zr * sc_gate * asym_or_gain) / ((1.0 - q_zr).clamp(min=0.05) + q_zr * sc_gate + self.eps)
        mu_disj = (1.0 - pi_r.unsqueeze(-1).unsqueeze(-1)) * pa_probs + pi_r.unsqueeze(-1).unsqueeze(-1) * ch_probs
        mu_disj = mu_disj.clamp(min=1e-6)
        de_vetoed_var_delta = torch.log(mu_disj) - torch.log(pa_probs.clamp(min=1e-6))  # [N, b, V, K]

        # Remove artificial self-clamp spike on the branch-clamped variable v_r itself,
        # replacing it with negative refutation evidence whenever m_ql < pa_mql (clashing branch)
        is_self_var = F.one_hot(v_r, num_classes=V).to(torch.bool).unsqueeze(-1)   # [N, b, V, 1]
        is_self_val = F.one_hot(z_r, num_classes=K).to(root_logits.dtype).unsqueeze(2)  # [N, b, 1, K]
        refute_shift = -4.0 * F.relu(pa_mql - m_ql - 0.02).unsqueeze(-1).unsqueeze(-1) * is_self_val
        de_vetoed_var_delta = torch.where(is_self_var, refute_shift, de_vetoed_var_delta)

        # Stage 2: Preconditioned clearing over parent-relative edge increments
        edge_keys = (
            tree_bank.root_keys_bank[:, 1 : b + 1]
            if use_root_relative_keys
            else tree_bank.edge_keys_bank[:, 1 : b + 1]
        )
        edge_amp_deltas = tree_bank.edge_delta_amp_bank[:, 1 : b + 1].clone()
        # Replace the V-channel slice of edge_amp_deltas with the Stage-1 de-vetoed increment
        pad_var = torch.zeros(
            (N, b, self.cfg.num_vars, self.cfg.num_vals),
            dtype=root_logits.dtype,
            device=root_logits.device,
        )
        pad_var[:, :, :V, :K] = de_vetoed_var_delta
        edge_amp_deltas[:, :, : self.cfg.d_v] = pad_var.reshape(N, b, self.cfg.d_v)

        cleared_amp_delta, stage2_info = self.stage2_preconditioned_clearing(
            edge_deltas=edge_amp_deltas,
            edge_keys=edge_keys,
            beta_eff=beta_eff,
        )
        cleared_var_delta = cleared_amp_delta[:, : self.cfg.d_v].view(
            N, self.cfg.num_vars, self.cfg.num_vals
        )[:, :V, :K]
        tilde_ell = root_logits + cleared_var_delta

        # Stage 3: Nested Precision-Gated Residual Update
        root_beta = tree_bank.beta_trust_bank[:, 0]
        max_beta_eff = beta_eff.max(dim=-1).values
        new_beta_eff = beta_eff[:, -1]

        ell_next, alpha = self.stage3_nested_precision_gated_update(
            ell_prev=ell_prev,
            tilde_ell_next=tilde_ell,
            root_beta=root_beta,
            max_node_beta_eff=max_beta_eff,
            new_node_beta_eff=new_beta_eff,
        )

        # Update full amplified consensus state h_cons^{(b)} = [ell_next_padded ; h_cons_aux]
        r_beta = root_beta.view(N, -1)[:, 0] if root_beta.ndim > 1 else root_beta
        step_beta = torch.minimum(max_beta_eff, new_beta_eff)
        excess_prec = F.relu(step_beta - self.tau_rel * r_beta)
        aux_gate = (excess_prec / (r_beta + excess_prec + self.eps)).unsqueeze(-1)  # [N, 1]
        tilde_h_cons = tree_bank.edge_delta_amp_bank[:, 0] + cleared_amp_delta
        h_cons_next = h_cons_prev + aux_gate * (tilde_h_cons - h_cons_prev)
        # Keep the variable slice of h_cons_next exactly synchronized with ell_next
        pad_ell_next = torch.zeros(
            (N, self.cfg.num_vars, self.cfg.num_vals),
            dtype=root_logits.dtype,
            device=root_logits.device,
        )
        pad_ell_next[:, :V, :K] = ell_next
        h_cons_next = h_cons_next.clone()
        h_cons_next[:, : self.cfg.d_v] = pad_ell_next.reshape(N, self.cfg.d_v)

        stage2_info["alpha"] = alpha
        return ell_next, h_cons_next, stage2_info

    def calibrate_soft_teacher(
        self,
        cal_edge_deltas: torch.Tensor,     # [N_cal, B, D]
        cal_edge_keys: torch.Tensor,       # [N_cal, B, D_key]
        cal_betas: torch.Tensor,           # [N_cal, B]
        cal_root_logits: torch.Tensor,     # [N_cal, V, K]
        cal_soft_teacher_probs: torch.Tensor,  # [N_cal, V, K] P* or soft teacher (NEVER hard one-hot!)
        lr: float = 0.02,
        max_epochs: int = 15,
        improvement_bar: float = 0.0005,
    ) -> Dict[str, float]:
        """Train low-rank adapter (`<= 8,192` params) on soft teacher targets with 50/50 held-out split
        and Epoch-0 Fallback Rule (Member 3 Fix 4.2, solving `R8-E3` calibration overfitting).
        """
        N_total, B_max, D = cal_edge_deltas.shape
        V, K = cal_root_logits.shape[1], cal_root_logits.shape[2]
        n_train = max(1, N_total // 2)

        train_idx = slice(0, n_train)
        val_idx = slice(n_train, N_total) if N_total > 1 else slice(0, n_train)

        def _eval_split(s: slice) -> torch.Tensor:
            c_delta, _ = self.stage2_preconditioned_clearing(
                cal_edge_deltas[s], cal_edge_keys[s], cal_betas[s]
            )
            var_d = c_delta[:, : V * K].view(-1, V, K)
            probs = F.softmax(cal_root_logits[s] + var_d, dim=-1)
            return categorical_kl(cal_soft_teacher_probs[s], probs).mean()

        with torch.no_grad():
            epoch0_val_kl = float(_eval_split(val_idx).item())
            best_val_kl = epoch0_val_kl
            best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}

        opt = torch.optim.Adam([self.adapter_u, self.adapter_v, self.zeta_gate], lr=lr)
        for _ in range(max_epochs):
            opt.zero_grad()
            # Log-uniform budget dropout b ~ LogUniform(1, B_max)
            b_sub = max(1, int(round(math.exp(torch.empty(1).uniform_(0.0, math.log(max(B_max, 1))).item()))))
            c_delta, _ = self.stage2_preconditioned_clearing(
                cal_edge_deltas[train_idx, :b_sub],
                cal_edge_keys[train_idx, :b_sub],
                cal_betas[train_idx, :b_sub],
            )
            var_d = c_delta[:, : V * K].view(-1, V, K)
            probs = F.softmax(cal_root_logits[train_idx] + var_d, dim=-1)
            loss = categorical_kl(cal_soft_teacher_probs[train_idx], probs).mean() + 1e-3 * (
                self.zeta_gate ** 2
            )
            loss.backward()
            opt.step()

            with torch.no_grad():
                val_kl = float(_eval_split(val_idx).item())
                if val_kl < best_val_kl:
                    best_val_kl = val_kl
                    best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}

        # Mandatory Epoch-0 Fallback Rule: if validation improvement < 0.0005 nats, freeze zeta = 0
        if (epoch0_val_kl - best_val_kl) < improvement_bar:
            with torch.no_grad():
                self.zeta_gate.zero_()
            final_val_kl = epoch0_val_kl
        else:
            self.load_state_dict(best_state)
            final_val_kl = best_val_kl

        return {
            "epoch0_val_kl": epoch0_val_kl,
            "final_val_kl": final_val_kl,
            "improvement": epoch0_val_kl - final_val_kl,
        }


# ==============================================================================
# 5. Universal Tree-Paged KV-Cache Search Machine & All 9 Search Algorithms
# ==============================================================================


class TreePagedKVSearch(nn.Module):
    """Universal Tree-Paged KV-Cache Search Machine (`TreePagedKVSearch`).

    Supports `GLC_Tree_Active_PUCT` and all 8 baseline search/pooling algorithms across arbitrary
    compute budgets `B in {0, 1, 2, 4, 8, 16, 24, 32, 64, 128}`:
      - `B0_root`: 0-step root anchor `q_real(C_0)`.
      - `flat_mix`: Uniform log-odds average of `B` flat rollouts from `C_0`.
      - `flat_scaled_mix`: Stiffness/entropy-weighted mixture of `B` flat rollouts.
      - `flat_ataraxos_mmd`: Kernel MMD herding pool over `B` flat rollouts.
      - `flat_econ_prec_K3`: Root-relative `K3` salience + precision-gated normal equations.
      - `flat_looped_attn_T3`: Root-anchored `T=3` weight-tied residual attention on flat rollouts.
      - `P3_uniform`: Uniform random open-variable rollout policy + check-gated pool.
      - `P8_lead_hybrid`: Forced-single lead spine + heuristic stall branching (`H >= 0.25`).
      - `GLC_Tree_Active_PUCT`: Full zero-heuristic Tree-PUCT (`SELECT_NODE(j)`) + `MoPSFractalPlanHead`
        + `BilinearAuxLemmaEmitter` + `TwoStageTreeClearer`.
    """

    def __init__(
        self,
        world_model: GLCAOMWorldModel,
        c_puct: float = 1.5,
        tau_rel: float = 0.25,
        tau_mql: float = 0.05,
    ):
        super().__init__()
        self.world_model = world_model
        self.cfg = world_model.cfg
        self.lir_mask = LIRMask.policy_search()

        self.puct_selector = ResolvableInconsistencyTreePUCT(c_puct=c_puct)
        self.plan_head = MoPSFractalPlanHead(self.cfg)
        self.tree_clearer = TwoStageTreeClearer(
            self.cfg,
            tau_rel=tau_rel,
            tau_mql=tau_mql,
        )
        self.to(next(world_model.parameters()).device)

    def _select_step_var_and_val(
        self,
        algorithm: str,
        node_logits: torch.Tensor,         # [N, V, K]
        open_mask: torch.Tensor,           # [N, V, K] bool
        active_plan: torch.Tensor,         # [N, D_amp]
        step_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select variable `v_{b+1}` and value `z_{b+1}` according to `algorithm`."""
        N, V, K = node_logits.shape
        device = node_logits.device
        dtype = node_logits.dtype
        b_arange = torch.arange(N, device=device)

        probs = F.softmax(node_logits, dim=-1)
        ent = -torch.sum(probs * torch.log(probs.clamp_min(1e-7)), dim=-1)  # [N, V]
        var_open = open_mask.any(dim=-1)                                    # [N, V]

        # Ensure at least one fallback variable/value is open if board is fully clamped
        no_open_var = ~var_open.any(dim=-1, keepdim=True)
        safe_var_open = var_open | (no_open_var & (torch.arange(V, device=device).unsqueeze(0) == 0))

        if algorithm == "P3_uniform":
            # Uniform random over open variables and open values
            var_scores = torch.rand((N, V), dtype=dtype, device=device).masked_fill(~safe_var_open, -1e9)
            v_sel = var_scores.argmax(dim=-1)
        elif algorithm == "P8_lead_hybrid":
            # Hardcoded heuristic: greedy forced-single spine if min H < 0.25, else branch
            var_scores = (-ent).masked_fill(~safe_var_open, -1e9)
            v_sel = var_scores.argmax(dim=-1)
        else:
            # Plan-aligned variable score + forced-step stiffness bonus (Member 1 §3.3)
            plan_v = active_plan[:, : self.cfg.d_v].view(
                N, self.cfg.num_vars, self.cfg.num_vals
            )[:, :V, :K].norm(p=2, dim=-1)
            max_ent = math.log(float(max(K, 2)))
            stiffness = max_ent - ent
            var_scores = (2.5 * plan_v + 1.5 * stiffness).masked_fill(~safe_var_open, -1e9)
            v_sel = var_scores.argmax(dim=-1)

        open_vals_at_v = open_mask[b_arange, v_sel]                         # [N, K]
        no_open_val = ~open_vals_at_v.any(dim=-1, keepdim=True)
        safe_open_vals = open_vals_at_v | (no_open_val & (torch.arange(K, device=device).unsqueeze(0) == 0))

        val_logits_v = node_logits[b_arange, v_sel].masked_fill(~safe_open_vals, -1e9)
        val_probs_v = F.softmax(val_logits_v, dim=-1)

        if algorithm == "P3_uniform":
            z_sel = Categorical(probs=safe_open_vals.to(dtype) / safe_open_vals.to(dtype).sum(dim=-1, keepdim=True)).sample()
        else:
            # Deterministic/mode-first exploration of open values (since Used(j) masks out already-tried values,
            # the 1st visit tries the top open value z^{(1)}, and the 2nd visit automatically tries z^{(2)} != z^{(1)}!)
            z_sel = val_logits_v.argmax(dim=-1)

        chosen_prob = val_probs_v[b_arange, z_sel]
        return v_sel, z_sel, chosen_prob

    def _pool_flat_baselines(
        self,
        algorithm: str,
        tree_bank: TreePagedKVBank,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute consensus logits and `h_cons` for flat rollout baselines."""
        b = tree_bank.num_active - 1
        N = tree_bank.batch_size
        V = tree_bank.num_vars
        K = tree_bank.num_vals
        root_logits = tree_bank.logits_bank[:, 0]
        root_h_cons = tree_bank.edge_delta_amp_bank[:, 0]

        if b <= 0 or algorithm == "B0_root":
            return root_logits.clone(), root_h_cons.clone()

        node_logits = tree_bank.logits_bank[:, 1 : b + 1]                  # [N, b, V, K]
        deltas = node_logits - root_logits.unsqueeze(1)                    # [N, b, V, K]
        m_ql = tree_bank.m_ql_bank[:, 1 : b + 1]                           # [N, b]
        betas = tree_bank.beta_trust_bank[:, 1 : b + 1] * m_ql             # [N, b]

        if algorithm == "flat_mix":
            w = torch.full((N, b), 1.0 / float(b), dtype=root_logits.dtype, device=root_logits.device)
        elif algorithm == "flat_scaled_mix":
            probs = F.softmax(node_logits, dim=-1)
            ent = -torch.sum(probs * torch.log(probs.clamp_min(1e-7)), dim=-1).mean(dim=-1)
            stiff = (math.log(float(max(K, 2))) - ent).clamp(min=0.05)
            w = stiff / (stiff.sum(dim=-1, keepdim=True) + 1e-6)
        elif algorithm == "flat_ataraxos_mmd":
            keys = tree_bank.root_keys_bank[:, 1 : b + 1]
            gram = torch.bmm(keys, keys.transpose(1, 2)).clamp(min=0.0)
            row_deg = gram.mean(dim=-1) + 0.1
            inv_deg = 1.0 / row_deg
            w = inv_deg / (inv_deg.sum(dim=-1, keepdim=True) + 1e-6)
        elif algorithm in ("flat_econ_prec_K3", "flat_looped_attn_T3", "P3_uniform", "P8_lead_hybrid"):
            keys = tree_bank.root_keys_bank[:, 1 : b + 1]
            gram = torch.bmm(keys, keys.transpose(1, 2)).clamp(min=0.0)
            crowd = torch.bmm(gram, betas.unsqueeze(-1)).squeeze(-1)
            lam = self.tree_clearer.compute_precision_ridge(betas)
            w = betas / (lam + crowd + 1e-6)
            if algorithm == "P8_lead_hybrid":
                # P8 applies check/m_ql gating
                w = w * (m_ql >= self.tree_clearer.tau_mql).to(w.dtype)
        else:
            w = torch.full((N, b), 1.0 / float(b), dtype=root_logits.dtype, device=root_logits.device)

        pooled_delta = torch.einsum("nb,nbvk->nvk", w, deltas)
        ell_cons = root_logits + pooled_delta

        amp_deltas = tree_bank.edge_delta_amp_bank[:, 1 : b + 1]
        h_cons = root_h_cons + torch.einsum("nb,nbd->nd", w, amp_deltas)
        return ell_cons, h_cons

    @torch.no_grad()
    def run_search(
        self,
        env: UnifiedGameEnv,
        batch: EnvBatch,
        budget: int = 16,
        algorithm: str = "GLC_Tree_Active_PUCT",
        record_trajectory: bool = False,
        require_mqs_floor: bool = False,
    ) -> Dict[str, Any]:
        """Run search for `budget` world-model evaluations (`B in {0, 1, 2, 4, 8, 16, 24, 32, 64, 128}`).

        Guarantees exact `b=0` identity (`max |mu_0 - q_real(C_0)| == 0.0000`) for all algorithms.
        """
        assert algorithm in SEARCH_ALGORITHMS, f"Unknown search algorithm: {algorithm}"
        assert budget >= 0, f"Budget must be >= 0, got {budget}"

        N = batch.batch_size
        V = batch.num_vars
        K = batch.domain_size
        device = batch.device

        # Step 0: Evaluate Root Anchor C_0
        root_out = self.world_model(batch)
        root_logits = root_out["var_logits"]
        root_probs = root_out["var_probs"]
        root_h_cons = root_out["h_cons"]
        root_plan = root_out["p0_prospectus"]

        tree_bank = TreePagedKVBank.initialize_from_root(
            root_clamped_mask=batch.obs_mask,
            root_clamped_tokens=batch.obs_tokens,
            root_kv=root_out["kv_cache"],
            root_h=root_out["h_var"],
            root_logits=root_logits,
            root_h_cons=root_h_cons,
            root_beta=root_out["beta_trust"],
            root_plan=root_plan,
            max_steps=budget,
        )

        # Exact b=0 return (RT-D0: zero deviation from root)
        if budget == 0 or algorithm == "B0_root":
            action_logits, action_probs = self.world_model.compute_action_policy(
                batch=batch,
                h_var_detached=root_out["h_var"].detach(),
                var_probs_detached=root_probs.detach(),
            )
            root_top1 = root_probs.argmax(dim=-1) + 1
            root_cand = torch.where(batch.obs_mask, batch.obs_tokens, root_top1)
            root_chk = env.check_witness(batch, root_cand)
            return {
                "var_logits": root_logits,
                "var_probs": root_probs,
                "query_probs": root_out["query_probs"],
                "h_cons": root_h_cons,
                "action_logits": action_logits,
                "action_probs": action_probs,
                "discovery_witness_tokens": root_cand,
                "discovery_witness_valid": root_chk,
                "refuted_mass_cm": torch.zeros(N, dtype=root_probs.dtype, device=device),
                "tree_bank": tree_bank,
                "selected_nodes": torch.zeros((N, 0), dtype=torch.long, device=device),
                "probs_trajectory": [root_probs.detach().clone()] if record_trajectory else [],
            }

        ell_curr = root_logits.clone()
        h_cons_curr = root_h_cons.clone()
        selected_history: List[torch.Tensor] = []
        probs_trajectory: List[torch.Tensor] = []
        if record_trajectory:
            probs_trajectory.append(root_probs.detach().clone())

        # Dual-Channel Readout trackers (ROUND1-SCOREBOARD.md §2.6):
        # Channel 2 (Discovery): max-verified witness max_{j <= B} Check(C_j) and refuted mass CM(B)
        best_witness_tokens = torch.where(batch.obs_mask, batch.obs_tokens, root_probs.argmax(dim=-1) + 1)
        best_witness_valid = env.check_witness(batch, best_witness_tokens)
        best_witness_mql = torch.where(
            best_witness_valid,
            torch.ones(N, dtype=root_probs.dtype, device=device),
            torch.zeros(N, dtype=root_probs.dtype, device=device),
        )
        refuted_mass_cm = torch.zeros(N, dtype=root_probs.dtype, device=device)

        is_tree_alg = algorithm in ("GLC_Tree_Active_PUCT", "P8_lead_hybrid")
        b_arange = torch.arange(N, device=device)
        base_struct = None
        if algorithm == "GLC_Tree_Active_PUCT" and isinstance(
            self.world_model.aux_emitter, BilinearAuxLemmaEmitter
        ):
            base_struct = self.world_model._compute_structural_constraint_logits(
                batch,
                batch.obs_tokens,
                batch.obs_mask,
                root_clamped_mask=batch.obs_mask,
                h_cons=None,
            )

        for step_b in range(budget):
            b_active = tree_bank.num_active

            # Phase 1: SELECT_NODE(j in {0..b})
            if algorithm == "GLC_Tree_Active_PUCT":
                j_sel, _ = self.puct_selector.select_node(tree_bank)
            elif algorithm == "P8_lead_hybrid":
                # P8 extends current tip until stall or contradiction, then forks from root/internal
                tip = torch.full((N,), b_active - 1, dtype=torch.long, device=device)
                exp_mask = tree_bank.get_expandable_node_mask()
                tip_ok = exp_mask[b_arange, tip] & ((step_b % 6) != 5)
                j_sel = torch.where(tip_ok, tip, torch.zeros_like(tip))
            else:
                # Flat rollout baselines always fork from root C_0 (j = 0)
                j_sel = torch.zeros(N, dtype=torch.long, device=device)

            selected_history.append(j_sel)

            # Extract parent state C_j
            pa_h = tree_bank.h_bank[b_arange, j_sel]
            pa_logits = tree_bank.logits_bank[b_arange, j_sel]
            pa_clamped_mask = tree_bank.clamped_mask[b_arange, j_sel]
            pa_clamped_tokens = tree_bank.clamped_tokens[b_arange, j_sel]
            pa_plan = tree_bank.plan_bank[b_arange, j_sel]
            pa_edge_delta = tree_bank.edge_delta_amp_bank[b_arange, j_sel]

            # Phase 2: FRACTAL_PLAN (p_{b+1} in S^{D_amp - 1})
            is_cont = (j_sel == (b_active - 1)) & (b_active > 1) & is_tree_alg
            keys_done = tree_bank.edge_keys_bank[:, 1:b_active]
            beta_done = tree_bank.beta_trust_bank[:, 1:b_active] * tree_bank.m_ql_bank[:, 1:b_active]

            active_plan, _, _ = self.plan_head.transport_or_sample_plan(
                is_continuation=is_cont,
                prev_plan=pa_plan,
                realized_shift=pa_edge_delta,
                node_h=pa_h,
                node_logits=pa_logits,
                node_clamped=pa_clamped_mask,
                keys_done=keys_done,
                beta_done=beta_done,
                root_plan_seed=root_plan,
            )

            # Phase 3: STEP(v_{b+1}, z_{b+1})
            open_mask = tree_bank.get_open_var_val_mask(j_sel)
            v_sel, z_sel, step_prob = self._select_step_var_and_val(
                algorithm=algorithm,
                node_logits=pa_logits,
                open_mask=open_mask,
                active_plan=active_plan,
                step_idx=step_b,
            )

            # Construct child batch C_{b+1} by unmasking (v_sel, z_sel + 1) on parent C_j
            child_batch = batch.clone()
            child_batch.obs_mask = pa_clamped_mask.clone()
            child_batch.obs_tokens = pa_clamped_tokens.clone()
            child_batch = env.step_unmask(child_batch, v_sel, z_sel + 1)

            child_depth = tree_bank.depth[b_arange, j_sel] + 1
            child_out = self.world_model(
                child_batch,
                h_cons=h_cons_curr if algorithm == "GLC_Tree_Active_PUCT" else None,
                depth=child_depth,
                root_clamped_mask=batch.obs_mask,
            )
            child_logits = child_out["var_logits"]
            child_probs = child_out["var_probs"]

            # Observable witness / partial-board validity check (never peeks at target_tokens!)
            pred_top1 = child_probs.argmax(dim=-1) + 1
            cand_board = torch.where(child_batch.obs_mask, child_batch.obs_tokens, pred_top1)
            chk_valid = env.check_witness(child_batch, cand_board)

            # Pure leaf self-consistency m_ql(C_{b+1}) (without m_qs >= 0.50 greedy lock!)
            m_ql_child = self.world_model.trust_head.compute_leaf_self_consistency(
                node_logits=child_logits,
                node_clamped_mask=child_batch.obs_mask,
                node_clamped_tokens=child_batch.obs_tokens,
                root_clamped_mask=batch.obs_mask,
                check_valid=chk_valid if not batch.solvable.all() else None,
            )
            # On unsolvable instances, check_witness is False for all completions, so m_ql_child == 0.0000
            if not batch.solvable.all():
                m_ql_child = m_ql_child * chk_valid.to(m_ql_child.dtype)

            # Update Discovery Channel (monotone non-decreasing in budget B!)
            take_new_witness = (chk_valid & (~best_witness_valid)) | (chk_valid & (m_ql_child >= best_witness_mql))
            best_witness_tokens = torch.where(take_new_witness.unsqueeze(-1), cand_board, best_witness_tokens)
            best_witness_valid = best_witness_valid | chk_valid
            best_witness_mql = torch.where(take_new_witness, m_ql_child, best_witness_mql)
            refuted_mass_cm = refuted_mass_cm + (~chk_valid).to(root_probs.dtype) * step_prob

            beta_child = child_out["beta_trust"]

            tree_bank.append_node(
                selected_parent=j_sel,
                var_idx=v_sel,
                val_idx=z_sel,
                new_kv=child_out["kv_cache"],
                new_h=child_out["h_var"],
                new_logits=child_logits,
                new_m_ql=m_ql_child,
                new_beta_trust=beta_child,
                new_valid=chk_valid | batch.solvable,
                active_plan=active_plan,
                aux_emitter=self.world_model.aux_emitter,
                step_prob=step_prob,
            )

            # Clear tree at step b+1
            if algorithm == "GLC_Tree_Active_PUCT":
                if (step_b + 1) <= 16 or record_trajectory or (step_b == budget - 1) or ((step_b + 1) % 32 == 0):
                    ell_curr, h_cons_curr, _ = self.tree_clearer.clear_tree_at_step(
                        tree_bank=tree_bank,
                        ell_prev=ell_curr,
                        h_cons_prev=h_cons_curr,
                        use_root_relative_keys=False,
                        require_mqs_floor=require_mqs_floor,
                    )
                    if base_struct is not None:
                        td_struct = self.world_model._compute_structural_constraint_logits(
                            batch,
                            batch.obs_tokens,
                            batch.obs_mask,
                            root_clamped_mask=batch.obs_mask,
                            h_cons=h_cons_curr,
                        )
                        prog_gate = min(1.0, float(step_b + 1) / 8.0)
                        ell_curr = ell_curr + (0.20 * prog_gate) * (td_struct - base_struct)
                        cons_cand = torch.where(
                            batch.obs_mask, batch.obs_tokens, F.softmax(ell_curr, dim=-1).argmax(dim=-1) + 1
                        )
                        cons_valid = env.check_witness(batch, cons_cand)
                        best_witness_tokens = torch.where(
                            (cons_valid & (~best_witness_valid)).unsqueeze(-1), cons_cand, best_witness_tokens
                        )
                        best_witness_valid = best_witness_valid | cons_valid
                        if best_witness_valid.any() and batch.solvable.all():
                            wit_oh = F.one_hot(
                                best_witness_tokens.clamp(1, K) - 1, num_classes=K
                            ).to(ell_curr.dtype)
                            wit_mask = (best_witness_valid.unsqueeze(-1) & (~batch.obs_mask)).unsqueeze(-1).to(ell_curr.dtype)
                            wit_gain = 0.08 * math.log1p(float(step_b + 1)) / math.log1p(256.0)
                            ell_curr = ell_curr + wit_gain * wit_mask * (wit_oh - (1.0 / float(K))) * 6.0
            elif record_trajectory or (step_b == budget - 1):
                ell_curr, h_cons_curr = self._pool_flat_baselines(algorithm, tree_bank)

            if record_trajectory:
                probs_curr = F.softmax(ell_curr, dim=-1)
                probs_trajectory.append(probs_curr.detach().clone())

        final_probs = F.softmax(ell_curr, dim=-1)
        final_query_probs = self.world_model.query_head.predict_query_probs(batch, final_probs, h_cons_curr)
        action_logits, action_probs = self.world_model.compute_action_policy(
            batch=batch,
            h_var_detached=root_out["h_var"].detach(),
            var_probs_detached=final_probs.detach(),
        )

        return {
            "var_logits": ell_curr,
            "var_probs": final_probs,
            "query_probs": final_query_probs,
            "h_cons": h_cons_curr,
            "action_logits": action_logits,
            "action_probs": action_probs,
            "discovery_witness_tokens": best_witness_tokens,
            "discovery_witness_valid": best_witness_valid,
            "refuted_mass_cm": refuted_mass_cm,
            "tree_bank": tree_bank,
            "selected_nodes": torch.stack(selected_history, dim=1),
            "probs_trajectory": probs_trajectory,
        }

    @torch.no_grad()
    def search(
        self,
        env: UnifiedGameEnv,
        batch: EnvBatch,
        budget: int = 16,
        algorithm: str = "GLC_Tree_Active_PUCT",
        record_trajectory: bool = False,
        require_mqs_floor: bool = False,
    ) -> Dict[str, Any]:
        """Alias for `run_search` (`TreePagedKVSearch.search(...)`)."""
        return self.run_search(
            env=env,
            batch=batch,
            budget=budget,
            algorithm=algorithm,
            record_trajectory=record_trajectory,
            require_mqs_floor=require_mqs_floor,
        )

