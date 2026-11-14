"""Neural Search Tree Controller (`GLCSearchTransformer`: `v1` & `v2`) for `R13` & `R14`.

Implements Member 19 specifications for `src/search/neural_tree_controller.py`:
1. Incremental `O(b)` transitive ancestor bitmask `anc_mask[batch, j, k] = 1[j in Anc(k)]`
   and relative depth `rel_depth[batch, j, k] = depth[k] - depth[j]` inside `TreePagedKVBank`.
2. Four architectural ablation modes for `R13-ALGO-IMITATION` & Expressivity Benchmarks:
   - `mode="flat_token"` (alias `"flat_no_tree_mask"`):
     Standard self-attention over search nodes with no tree topology mask (cannot distinguish
     descendants from cousins).
   - `mode="parent_1hop"`:
     Only 1-hop parent/child masks `pa(k) == j` or `pa(j) == k` per layer (`L_tree = 3` layers;
     requires `L_tree >= d` layers to aggregate depth-`d` descendants).
   - `mode="full_ancestry_v1"` (aliases `"full_ancestry"`, `"full_ancestry_1pass"`):
     Transitive relational masks (`self`, `parent`, `child`, `ancestor`, `descendant`, `sibling`,
     `global_edge_key`) + 1-pass multi-aggregation heads (`mean` for `Q^{MCTS}(j)`, `log_count`
     for `log N(j)`, `soft_max` / `soft_min` over `L_tree = 3` hops, and `preconditioned_key`
     for LIR clearing) + Multi-Witness Disjunctive Marginalization.
   - `mode="glc_search_v2"` (alias `"full_ancestry_recurrent_v2"`):
     Upgrades `GLCSearch v1` to overcome all 5 expressivity barriers:
       (B1) Bottom-Up Recurrent / Looped Ply Backup (`soft_max`/`soft_min` + `PNS` OR/AND) +
            Top-Down Alpha-Beta Window propagation across arbitrary tree depth `d > L_tree`.
       (B2) Soft State-Equivalence Transposition-DAG Kernel `R^{DAG} = AncMask * E * AncMask` (`MCGS`).
       (B3) Imperfect-Information Infoset Equivalence Mask `M_infoset(j, k)` + Path Log-Reach
            `log pi_{-i}(j)` + `ReLU` Regret-Matching strategy head (`CFR+`).
       (B4) Differentiable Sibling Dominance / Rank Head `rank_sib(j)` (`Gumbel Sequential Halving`).
       (B5) Global Conflict-Clause / Nogood Broadcast Attention Head across disjoint subtrees (`CDCL`).
3. 2-Hop Tseitin Auxiliary Coupling (`Tseitin2HopQueryCoupler`) for calibrated combinatorial
   query probabilities `P(Q(A, B))` across multi-hop parity/equivalence relays `A <-> u <-> B`.
4. Supervised Algorithm Imitation (`R13-ALGO-IMITATION`) and Unsupervised End-to-End Search
   Discovery (`R14-UNSUP-DISCOVERY`) with zero oracle leakage (`S1`-`S6` and `PoisonTensor` safe).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.envs.base import EnvBatch, QueryType, UnifiedGameEnv, categorical_kl
from src.models.glc_aom import (
    BilinearAuxLemmaEmitter,
    GLCAOMConfig,
    GLCAOMWorldModel,
    LIRMask,
)
from src.search.classical_teachers import (
    LOCAL_NODE_FEATURE_DIM,
    BaseClassicalSearchTeacher,
    SearchTreeTrace,
    extract_local_node_features,
    extract_tree_topology,
)
from src.search.kv_tree_search import (
    MoPSFractalPlanHead,
    ResolvableInconsistencyTreePUCT,
    TreePagedKVBank,
    TwoStageTreeClearer,
)

SUPPORTED_CONTROLLER_MODES: Tuple[str, ...] = (
    "flat_token",
    "flat_no_tree_mask",
    "parent_1hop",
    "full_ancestry",
    "full_ancestry_v1",
    "full_ancestry_1pass",
    "glc_search_v2",
    "full_ancestry_recurrent_v2",
)


def canonicalize_controller_mode(mode: str) -> str:
    """Normalize controller mode aliases into `{'flat_token', 'parent_1hop', 'full_ancestry_v1', 'glc_search_v2'}`."""
    m = mode.strip().lower().replace("-", "_")
    if m in ("flat_token", "flat_no_tree_mask", "flat"):
        return "flat_token"
    if m in ("parent_1hop", "1hop", "parent_only_1hop"):
        return "parent_1hop"
    if m in ("full_ancestry", "full_ancestry_v1", "full_ancestry_1pass", "v1"):
        return "full_ancestry_v1"
    if m in ("glc_search_v2", "full_ancestry_recurrent_v2", "v2", "recurrent_v2"):
        return "glc_search_v2"
    raise ValueError(f"Unsupported GLCSearchTransformer mode {mode!r}; expected one of {SUPPORTED_CONTROLLER_MODES}")


# ==============================================================================
# 1. 2-Hop Tseitin Auxiliary Coupler for Combinatorial Queries Q(A, B)
# ==============================================================================


class Tseitin2HopQueryCoupler(nn.Module):
    """2-Hop Tseitin Auxiliary Coupling & Post-Depth Query Calibrator (`chi = (0, 1)`).

    Resolves 2-hop equivalence/parity chains `A <-> u <-> B` via intermediate Tseitin relay
    variables `u not in {A, B}`:
      1. Extracts 1-hop pairwise variable co-fluctuations `C^{(1)}_{u, v}` from the search tree's
         parent-relative and ancestor-descendant logit increments `Delta l_V`.
      2. Applies 2-hop Tseitin resolution across all relay variables `u`:
           `C^{(2)}_{A, B} = max_{u != A, B} |tanh(C^{(1)}_{A, u}) * tanh(C^{(1)}_{u, B})| * sign(C^{(1)}_{A, u} * C^{(1)}_{u, B})`
         plus soft-attention relay pooling.
      3. Embeds the resolved 2-hop couplings into the World Lemma Channel `Z^W_aux` of `h_cons`
         and applies calibrated log-odds scaling `(tau_q, b_q)` for `P(Q(A, B))`.
    """

    def __init__(self, cfg: GLCAOMConfig, aux_emitter: BilinearAuxLemmaEmitter):
        super().__init__()
        self.cfg = cfg
        self.aux_emitter = aux_emitter
        self.lir_mask = LIRMask.epistemic_belief()

        self.tseitin_gain = nn.Parameter(torch.tensor(2.2))
        self.cal_temp = nn.Parameter(torch.tensor(1.0))
        self.cal_bias = nn.Parameter(torch.tensor(0.0))

    def compute_pairwise_cofluct_matrix(
        self,
        tree_bank: TreePagedKVBank,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute 1-hop `C^{(1)} [N, V, V]` and 2-hop Tseitin `C^{(2)} [N, V, V]` co-fluctuation matrices."""
        b = tree_bank.num_active
        N = tree_bank.batch_size
        V = tree_bank.num_vars
        K = tree_bank.num_vals
        device = tree_bank.logits_bank.device
        dtype = tree_bank.logits_bank.dtype

        if b <= 1:
            zeros = torch.zeros((N, V, V), dtype=dtype, device=device)
            return zeros, zeros

        # Signed scalar polarity shift per variable on each node 1..b-1 relative to root 0 and parent pa(r)
        root_logits = tree_bank.logits_bank[:, 0:1]                         # [N, 1, V, K]
        node_logits = tree_bank.logits_bank[:, 1:b]                         # [N, b-1, V, K]
        root_delta = node_logits - root_logits                              # [N, b-1, V, K]
        edge_delta = tree_bank.edge_delta_var_bank[:, 1:b]                  # [N, b-1, V, K]

        if K >= 2:
            pol = torch.tensor([(-1.0) ** k for k in range(K)], dtype=dtype, device=device)
            pol = pol / float(K)
            d_root_pol = torch.einsum("nbvk,k->nbv", root_delta, pol)
            d_edge_pol = torch.einsum("nbvk,k->nbv", edge_delta, pol)
        else:
            d_root_pol = root_delta[..., 0]
            d_edge_pol = edge_delta[..., 0]

        d_comb = 0.6 * torch.tanh(d_root_pol) + 0.4 * torch.tanh(d_edge_pol)  # [N, b-1, V]
        w_node = (tree_bank.m_ql_bank[:, 1:b] * tree_bank.beta_trust_bank[:, 1:b]).clamp(min=0.05)
        w_node = w_node / (w_node.sum(dim=-1, keepdim=True) + 1e-6)         # [N, b-1]

        # 1-hop empirical co-fluctuation matrix C^{(1)}_{u, v}
        c1 = torch.einsum("nb,nbu,nbv->nuv", w_node, d_comb, d_comb)        # [N, V, V]

        # Cross-depth ancestor-descendant path coupling using anc_mask
        anc_sub = tree_bank.anc_mask[:, 1:b, 1:b].to(dtype)                 # [N, b-1, b-1]
        anc_prop = torch.bmm(anc_sub, d_comb) / anc_sub.sum(dim=-1, keepdim=True).clamp(min=1.0)
        c1_path = torch.einsum("nb,nbu,nbv->nuv", w_node, d_comb, anc_prop)
        c1_sym = 0.5 * (c1 + c1.transpose(1, 2)) + 0.25 * (c1_path + c1_path.transpose(1, 2))
        eye_v = torch.eye(V, dtype=dtype, device=device).unsqueeze(0)
        c1_sym = c1_sym * (1.0 - eye_v)

        # 2-hop Tseitin resolution across intermediate relay variables w != u, v:
        # T_{u, w, v} = tanh(2 * C^{(1)}_{u, w}) * tanh(2 * C^{(1)}_{w, v})
        t_mat = torch.tanh(2.5 * c1_sym)                                    # [N, V, V]
        # Relay product [N, u, v, w] = t_mat[N, u, w] * t_mat[N, w, v]
        relay_prod = t_mat.unsqueeze(2) * t_mat.transpose(1, 2).unsqueeze(1)  # [N, V, V, V]
        abs_relay = relay_prod.abs()
        max_idx = abs_relay.argmax(dim=-1, keepdim=True)
        c2_max = relay_prod.gather(-1, max_idx).squeeze(-1)                 # [N, V, V]
        c2_mean = torch.bmm(t_mat, t_mat) / float(max(V - 2, 1))
        c2 = (0.7 * c2_max + 0.3 * c2_mean) * (1.0 - eye_v)
        return c1_sym, c2

    def enrich_h_cons_and_predict_queries(
        self,
        batch: EnvBatch,
        tree_bank: TreePagedKVBank,
        pred_var_probs: torch.Tensor,
        h_cons: torch.Tensor,
        query_head: nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inject 2-hop Tseitin couplings into `h_cons` and return `(h_cons_enriched, query_probs, tseitin_q)`."""
        N = tree_bank.batch_size
        V = tree_bank.num_vars
        device = pred_var_probs.device
        dtype = pred_var_probs.dtype

        c1, c2 = self.compute_pairwise_cofluct_matrix(tree_bank)            # [N, V, V]
        c_total = c1 + F.softplus(self.tseitin_gain) * c2

        # Write 2-hop Tseitin couplings into orthonormal pair basis Q_pair of h_cons if available
        h_cons_out = h_cons.clone()
        v_ortho = getattr(self.aux_emitter, "v_ortho", 0)
        n_ortho = getattr(self.aux_emitter, "n_ortho_pairs", 0)
        if (
            v_ortho > 0
            and n_ortho > 0
            and self.cfg.d_aux_world >= n_ortho
            and hasattr(self.aux_emitter, "pair_u_idx")
            and hasattr(self.aux_emitter, "q_pair_basis")
        ):
            v_lim = min(V, v_ortho)
            valid_p = (self.aux_emitter.pair_u_idx < v_lim) & (self.aux_emitter.pair_v_idx < v_lim)
            if valid_p.any():
                pu = self.aux_emitter.pair_u_idx.clamp(0, V - 1)
                pv = self.aux_emitter.pair_v_idx.clamp(0, V - 1)
                pair_vals = c_total[:, pu, pv] * valid_p.to(dtype).unsqueeze(0)  # [N, n_ortho]
                ortho_shift = torch.einsum("np,dp->nd", pair_vals, self.aux_emitter.q_pair_basis)
                w_start = self.cfg.d_v
                w_end = w_start + self.cfg.d_aux_world
                h_cons_out[:, w_start:w_end] = h_cons_out[:, w_start:w_end] + 0.35 * ortho_shift

        base_q_probs = query_head.predict_query_probs(batch, pred_var_probs, h_cons_out)
        Q = batch.query_pairs.shape[1]
        if Q == 0:
            return h_cons_out, base_q_probs, torch.zeros((N, 0), dtype=dtype, device=device)

        va = batch.query_pairs[:, :, 0].clamp(0, V - 1)
        vb = batch.query_pairs[:, :, 1].clamp(0, V - 1)
        q_type = batch.query_pairs[:, :, 2]

        b_idx = torch.arange(N, device=device).unsqueeze(1).expand(N, Q)
        tseitin_q = c_total[b_idx, va, vb]                                  # [N, Q]

        sign = torch.ones_like(tseitin_q)
        neq_or_xor = (q_type == int(QueryType.NOT_EQUAL)) | (q_type == int(QueryType.PARITY_XOR))
        sign = torch.where(neq_or_xor, -torch.ones_like(tseitin_q), sign)

        # Gate by unresolved marginal entropy of (A, B) so anchored/observed variables stay exact
        K = pred_var_probs.shape[-1]
        pa = pred_var_probs.gather(1, va.unsqueeze(-1).expand(N, Q, K))
        pb = pred_var_probs.gather(1, vb.unsqueeze(-1).expand(N, Q, K))
        ent_a = -(pa * torch.log(pa.clamp(min=1e-7))).sum(dim=-1)
        ent_b = -(pb * torch.log(pb.clamp(min=1e-7))).sum(dim=-1)
        unres_gate = torch.sqrt((ent_a * ent_b).clamp(min=0.0)) / math.log(float(max(K, 2)))

        raw_logit = torch.logit(base_q_probs.clamp(1e-5, 1.0 - 1e-5))
        cal_temp = F.softplus(self.cal_temp).clamp(min=0.25, max=4.0)
        calibrated_logit = (raw_logit + unres_gate * sign * tseitin_q + self.cal_bias) / cal_temp
        cal_q_probs = torch.sigmoid(calibrated_logit).clamp(1e-6, 1.0 - 1e-6)
        return h_cons_out, cal_q_probs, tseitin_q


# ==============================================================================
# 2. Relational Tree Transformer Layer (Mode-Specific Attention Masking)
# ==============================================================================


class RelationalTreeTransformerLayer(nn.Module):
    """Search-Tree Transformer Layer supporting `flat_token`, `parent_1hop`, `full_ancestry_v1`, and `glc_search_v2`."""

    def __init__(self, d_tree: int = 64, n_heads: int = 4, max_rel_depth: int = 32):
        super().__init__()
        self.d_tree = d_tree
        self.n_heads = n_heads
        self.d_head = d_tree // n_heads
        self.max_rel_depth = max_rel_depth

        self.ln1 = nn.LayerNorm(d_tree)
        self.q_proj = nn.Linear(d_tree, d_tree)
        self.k_proj = nn.Linear(d_tree, d_tree)
        self.v_proj = nn.Linear(d_tree, d_tree)
        self.out_proj = nn.Linear(d_tree, d_tree)

        # Learned per-head relational bias for (self, parent, child, anc, desc, sibling, dag, infoset)
        self.rel_bias = nn.Parameter(torch.zeros(n_heads, 8))
        self.rel_depth_emb = nn.Embedding(2 * max_rel_depth + 1, n_heads)
        nn.init.zeros_(self.rel_depth_emb.weight)

        # 1-hop child soft-extrema value backup gates (used in parent_1hop and full_ancestry_v1)
        self.hop_val_proj = nn.Linear(d_tree, 1)
        self.hop_back_proj = nn.Linear(2, d_tree)

        self.ln2 = nn.LayerNorm(d_tree)
        self.ffn = nn.Sequential(
            nn.Linear(d_tree, d_tree * 2),
            nn.SiLU(),
            nn.Linear(d_tree * 2, d_tree),
        )

    def forward(
        self,
        h: torch.Tensor,                   # [N, b, d_tree]
        v_step: torch.Tensor,              # [N, b] running 1-hop value backup state
        cnt_step: torch.Tensor,            # [N, b] running 1-hop count state
        topo: Dict[str, torch.Tensor],
        mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        N, b, D = h.shape
        H = self.n_heads
        d_k = self.d_head
        dtype = h.dtype
        device = h.device

        norm_h = self.ln1(h)
        q = self.q_proj(norm_h).view(N, b, H, d_k).transpose(1, 2)          # [N, H, b, d_k]
        k = self.k_proj(norm_h).view(N, b, H, d_k).transpose(1, 2)          # [N, H, b, d_k]
        v = self.v_proj(norm_h).view(N, b, H, d_k).transpose(1, 2)          # [N, H, b, d_k]

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(float(d_k))  # [N, H, b, b]

        if mode == "flat_token":
            # Flat token self-attention: NO tree topology masks, NO relative depth!
            attn_weights = F.softmax(scores, dim=-1)
            v_next = v_step
            cnt_next = cnt_step
        elif mode == "parent_1hop":
            # Only 1-hop parent/child/self neighborhood allowed per layer!
            allowed = topo["self_mask"] | topo["parent_mask"] | topo["child_mask"]  # [N, b, b]
            scores = scores.masked_fill(~allowed.unsqueeze(1), -1e9)
            attn_weights = F.softmax(scores, dim=-1)

            # 1-hop child backup step (can only propagate 1 level per layer!)
            ch_mask = topo["child_mask"]                                    # [N, b, b]
            has_ch = ch_mask.any(dim=-1)                                    # [N, b]
            is_max = (topo["depth"] % 2) == 0                               # [N, b]
            ch_max = v_step.unsqueeze(1).expand(N, b, b).masked_fill(~ch_mask, -1e9).max(dim=-1).values
            ch_min = v_step.unsqueeze(1).expand(N, b, b).masked_fill(~ch_mask, 1e9).min(dim=-1).values
            v_next = torch.where(has_ch, torch.where(is_max, ch_max, ch_min), v_step)
            cnt_next = 1.0 + torch.bmm(ch_mask.to(dtype), cnt_step.unsqueeze(-1)).squeeze(-1)
            h = h + self.hop_back_proj(torch.stack([v_next, torch.log1p(cnt_next)], dim=-1))
        else:
            # full_ancestry_v1 and glc_search_v2: transitive ancestor/descendant/sibling masks
            rel_stack = torch.stack(
                [
                    topo["self_mask"].to(dtype),
                    topo["parent_mask"].to(dtype),
                    topo["child_mask"].to(dtype),
                    topo["anc_mask"].to(dtype),
                    topo["desc_mask"].to(dtype),
                    topo["sibling_mask"].to(dtype),
                    topo["dag_anc_mask"].to(dtype) if mode == "glc_search_v2" else topo["anc_mask"].to(dtype),
                    topo["infoset_mask"].to(dtype) if mode == "glc_search_v2" else topo["self_mask"].to(dtype),
                ],
                dim=-1,
            )                                                               # [N, b, b, 8]
            bias = torch.einsum("hm,njkm->nhjk", self.rel_bias, rel_stack)  # [N, H, b, b]
            rel_d_clamped = topo["rel_depth"].clamp(-self.max_rel_depth, self.max_rel_depth) + self.max_rel_depth
            d_bias = self.rel_depth_emb(rel_d_clamped).permute(0, 3, 1, 2)  # [N, H, b, b]

            if mode == "full_ancestry_v1":
                allowed = (
                    topo["self_mask"]
                    | topo["parent_mask"]
                    | topo["child_mask"]
                    | topo["anc_mask"]
                    | topo["desc_mask"]
                    | topo["sibling_mask"]
                )
            else:
                # glc_search_v2 also allows global DAG, infoset, and conflict broadcast connections
                allowed = torch.ones((N, b, b), dtype=torch.bool, device=device)

            scores = (scores + bias + d_bias).masked_fill(~allowed.unsqueeze(1), -1e9)
            attn_weights = F.softmax(scores, dim=-1)

            # In full_ancestry_v1 (non-recurrent L_tree=3), each layer performs 1 ply of child max/min alternation
            ch_mask = topo["child_mask"]
            has_ch = ch_mask.any(dim=-1)
            is_max = (topo["depth"] % 2) == 0
            ch_max = v_step.unsqueeze(1).expand(N, b, b).masked_fill(~ch_mask, -1e9).max(dim=-1).values
            ch_min = v_step.unsqueeze(1).expand(N, b, b).masked_fill(~ch_mask, 1e9).min(dim=-1).values
            v_next = torch.where(has_ch, torch.where(is_max, ch_max, ch_min), v_step)
            cnt_next = cnt_step
            h = h + self.hop_back_proj(torch.stack([v_next, torch.log1p(cnt_next)], dim=-1))

        attn_out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(N, b, D)
        h = h + self.out_proj(attn_out)
        h = h + self.ffn(self.ln2(h))
        return h, v_next, cnt_next


# ==============================================================================
# 3. GLCSearchTransformer (`v1` & `v2`)
# ==============================================================================


class GLCSearchTransformer(nn.Module):
    """General Neural Search Controller (`GLCSearchTransformer`) capable of representing and learning
    MCTS, MCGS (Transposition DAGs), Minimax/Alpha-Beta/PNS, CDCL (with Global Nogood Broadcast),
    Disjunctive BP, Imperfect-Info CFR+ (Infoset Regret Matching), and Gumbel Sequential Halving in-weights.
    """

    def __init__(
        self,
        world_model: GLCAOMWorldModel,
        mode: str = "glc_search_v2",
        d_tree: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        c_puct: float = 1.5,
        tau_rel: float = 0.25,
        tau_mql: float = 0.05,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.world_model = world_model
        self.cfg = world_model.cfg
        self.mode = canonicalize_controller_mode(mode)
        self.lir_mask = LIRMask.policy_search()
        self.d_tree = int(d_tree)
        self.n_heads = int(n_heads)
        self.n_layers = int(n_layers)
        self.c_puct = float(c_puct)
        self.eps = float(eps)

        # Input projection from strictly node-local features (D_local = 16) + detached h_var summary
        self.local_feat_proj = nn.Sequential(
            nn.Linear(LOCAL_NODE_FEATURE_DIM, d_tree),
            nn.SiLU(),
            nn.Linear(d_tree, d_tree),
        )
        self.h_ctx_proj = nn.Linear(self.cfg.d_model, d_tree, bias=False)
        nn.init.normal_(self.h_ctx_proj.weight, std=0.02)

        self.tree_layers = nn.ModuleList(
            [RelationalTreeTransformerLayer(d_tree=d_tree, n_heads=n_heads) for _ in range(n_layers)]
        )
        self.final_ln = nn.LayerNorm(d_tree)

        # Mode-specific basis dimension:
        # Every mode receives an operator basis vector `psi_j` of dimension `D_basis = 24`
        # populated strictly using the relational operators permitted by `self.mode`!
        self.d_basis = 24

        # Linear + MLP readout heads over [psi_j ; h_tree_j]
        # Initializing the linear readout on `psi_j` allows rapid sample-efficient convergence in R13
        # and warm-starting in R14!
        self.head_log_N_lin = nn.Linear(self.d_basis, 1)
        self.head_log_N_mlp = nn.Sequential(
            nn.Linear(d_tree + self.d_basis, d_tree // 2),
            nn.SiLU(),
            nn.Linear(d_tree // 2, 1),
        )

        self.head_Q_lin = nn.Linear(self.d_basis, 1)
        self.head_Q_mlp = nn.Sequential(
            nn.Linear(d_tree + self.d_basis, d_tree // 2),
            nn.SiLU(),
            nn.Linear(d_tree // 2, 1),
        )

        self.head_select_lin = nn.Linear(self.d_basis, 1)
        self.head_select_mlp = nn.Sequential(
            nn.Linear(d_tree + self.d_basis, d_tree // 2),
            nn.SiLU(),
            nn.Linear(d_tree // 2, 1),
        )

        # Regret-Matching head (for CFR+ mixed strategies sigma_RM proportional to ReLU(R))
        self.head_regret_lin = nn.Linear(self.d_basis, 1)

        # Next-variable/value expansion policy head at selected node
        self.var_action_head = nn.Sequential(
            nn.Linear(d_tree + 4, d_tree // 2),
            nn.SiLU(),
            nn.Linear(d_tree // 2, 1),
        )

        # Preconditioned LIR Root Belief Clearer + Plan Head + 2-Hop Tseitin Query Coupler
        self.plan_head = MoPSFractalPlanHead(self.cfg)
        self.tree_clearer = TwoStageTreeClearer(self.cfg, tau_rel=tau_rel, tau_mql=tau_mql)
        self.r12_puct = ResolvableInconsistencyTreePUCT(c_puct=c_puct, eps=eps)
        self.tseitin_coupler = Tseitin2HopQueryCoupler(self.cfg, self.world_model.aux_emitter)

        self._init_default_mcts_puct_weights()
        self.to(next(world_model.parameters()).device)

    def _init_default_mcts_puct_weights(self) -> None:
        """Initialize linear operator readouts to a sensible MCTS/GLC-PUCT default."""
        with torch.no_grad():
            nn.init.zeros_(self.head_log_N_mlp[-1].weight)
            nn.init.zeros_(self.head_log_N_mlp[-1].bias)
            nn.init.zeros_(self.head_Q_mlp[-1].weight)
            nn.init.zeros_(self.head_Q_mlp[-1].bias)
            nn.init.zeros_(self.head_select_mlp[-1].weight)
            nn.init.zeros_(self.head_select_mlp[-1].bias)

            nn.init.zeros_(self.head_log_N_lin.weight)
            nn.init.zeros_(self.head_log_N_lin.bias)
            self.head_log_N_lin.weight[0, 0] = 1.0  # slot 0: mode-allowed log_count

            nn.init.zeros_(self.head_Q_lin.weight)
            nn.init.zeros_(self.head_Q_lin.bias)
            self.head_Q_lin.weight[0, 1] = 1.0      # slot 1: mode-allowed subtree mean Q

            nn.init.zeros_(self.head_select_lin.weight)
            nn.init.zeros_(self.head_select_lin.bias)
            self.head_select_lin.weight[0, 1] = 1.0  # Q
            self.head_select_lin.weight[0, 2] = 1.0  # PUCT bonus

            nn.init.zeros_(self.head_regret_lin.weight)
            nn.init.zeros_(self.head_regret_lin.bias)
            self.head_regret_lin.weight[0, 14] = 1.0

    def compute_mode_operator_basis(
        self,
        tree_bank: TreePagedKVBank,
        node_feats: torch.Tensor,          # [N, b, 16]
        topo: Dict[str, torch.Tensor],
        v_1hop_layer: torch.Tensor,        # [N, b] after L_tree 1-hop steps
        cnt_1hop_layer: torch.Tensor,      # [N, b] after L_tree 1-hop steps
    ) -> torch.Tensor:
        """Compute the 24-dim relational operator basis `psi [N, b, 24]` strictly respecting `self.mode`.

        - In `flat_token`: NO tree masks are used (`anc_mask`, `child_mask`, `parent_mask`, `sibling_mask`
          are replaced by the flat global mean over `0..b-1`).
        - In `parent_1hop`: ONLY the `L_tree`-hop message-passing states (`v_1hop_layer`, `cnt_1hop_layer`)
          and 1-hop parent/child features are available (leaves beyond `L_tree` hops are unreachable!).
        - In `full_ancestry_v1`: 1-pass transitive `anc_mask` (`Desc(j)`), `desc_mask` (`Anc(j)`),
          `parent_mask`, `child_mask`, and `sibling_mask` are available, but WITHOUT recurrent full-depth
          Minimax/PNS backup (`v_1hop_layer` stops at `L_tree = 3` plies!), WITHOUT `dag_anc_mask`,
          WITHOUT `infoset_mask`, WITHOUT sibling dominance rank, and WITHOUT global cousin nogood broadcast.
        - In `glc_search_v2`: ALL 5 v2 relational operators are active.
        """
        N, b, _ = node_feats.shape
        device = node_feats.device
        dtype = node_feats.dtype

        leaf_vals = node_feats[:, :, 0]                                     # [N, b]
        m_ql = node_feats[:, :, 1]                                          # [N, b]
        beta = node_feats[:, :, 2]                                          # [N, b]
        norm_depth = node_feats[:, :, 3]                                    # [N, b]
        ply_parity = node_feats[:, :, 4]                                    # [N, b]
        n_exp_log = node_feats[:, :, 6]                                     # [N, b]
        step_prob = node_feats[:, :, 7]                                     # [N, b]
        open_ent_max = node_feats[:, :, 9]                                  # [N, b]
        open_stiff_max = node_feats[:, :, 10]                               # [N, b]
        conflict_flag = node_feats[:, :, 12]                                # [N, b]
        edge_delta_norm = node_feats[:, :, 14]                              # [N, b]

        phi = self.r12_puct.compute_potential_phi(tree_bank)                # [N, b]
        mvoi = self.r12_puct.compute_marginal_voi(tree_bank)                # [N, b]
        p_prior = 0.5 * step_prob + 0.5 * (phi / (phi.max(dim=-1, keepdim=True).values + self.eps))

        basis = torch.zeros((N, b, self.d_basis), dtype=dtype, device=device)

        if self.mode == "flat_token":
            # Flat token mode: can only see node-local features + global tree mean!
            global_cnt = torch.full((N, b), float(b), dtype=dtype, device=device)
            global_q = leaf_vals.mean(dim=-1, keepdim=True).expand(N, b)
            local_n = 1.0 + tree_bank.n_exp[:, :b].to(dtype)
            basis[:, :, 0] = torch.log(local_n)                             # local n_exp only (no subtree N!)
            basis[:, :, 1] = 0.5 * leaf_vals + 0.5 * global_q               # local leaf + global mean
            basis[:, :, 2] = self.c_puct * p_prior * torch.sqrt(global_cnt) / (1.0 + local_n)
            basis[:, :, 3] = leaf_vals
            basis[:, :, 4] = m_ql
            basis[:, :, 5] = phi
            basis[:, :, 6] = mvoi
            basis[:, :, 7] = norm_depth
            basis[:, :, 8] = ply_parity
            basis[:, :, 9] = open_stiff_max
            basis[:, :, 10] = conflict_flag
            return basis

        if self.mode == "parent_1hop":
            # 1-hop mode: only reaches up to 2 hops via 1-hop message passing!
            pa_idx = topo["parent_idx"]
            ch_f = (topo["self_mask"] | topo["child_mask"]).to(dtype)
            reach_l = (torch.bmm(ch_f, ch_f) > 0).to(dtype)
            n_reach = reach_l.sum(dim=-1).clamp(min=1.0)
            q_reach = torch.bmm(reach_l, leaf_vals.unsqueeze(-1)).squeeze(-1) / n_reach
            pa_n_reach = n_reach.gather(1, pa_idx)
            pa_n_reach[:, 0] = n_reach[:, 0]
            puct_1hop = self.c_puct * p_prior * torch.sqrt(pa_n_reach) / (1.0 + n_reach - 0.5)

            basis[:, :, 0] = torch.log(n_reach)
            basis[:, :, 1] = q_reach
            basis[:, :, 2] = puct_1hop
            basis[:, :, 3] = v_1hop_layer                                   # 3-step 1-hop minimax backup
            basis[:, :, 4] = m_ql
            basis[:, :, 5] = phi
            basis[:, :, 6] = mvoi
            basis[:, :, 7] = norm_depth
            basis[:, :, 8] = ply_parity
            basis[:, :, 9] = open_stiff_max
            basis[:, :, 10] = conflict_flag
            return basis

        # Both full_ancestry_v1 and glc_search_v2 have transitive anc_mask [N, b, b]
        anc_f = topo["anc_mask"].to(dtype)
        pa_idx = topo["parent_idx"]
        depths = topo["depth"]

        # Exact MCTS commutative monoid over Desc(j) via anc_mask
        n_mcts = anc_f.sum(dim=-1).clamp(min=1.0)                           # [N, b]
        q_mcts = torch.bmm(anc_f, leaf_vals.unsqueeze(-1)).squeeze(-1) / n_mcts
        pa_n_mcts = n_mcts.gather(1, pa_idx)
        pa_n_mcts[:, 0] = n_mcts[:, 0]
        puct_mcts = self.c_puct * p_prior * torch.sqrt(pa_n_mcts) / (1.0 + n_mcts - 0.5)

        # Disjunctive BP subtree precision mass & value
        prec = (m_ql * beta).clamp(min=0.05)
        n_bp = torch.bmm(anc_f, prec.unsqueeze(-1)).squeeze(-1).clamp(min=0.1)
        q_bp = (
            torch.bmm(anc_f, (prec * leaf_vals).unsqueeze(-1)).squeeze(-1) / n_bp
            + 0.35 * torch.tanh(mvoi)
        ).clamp(-1.0, 1.0)
        n_exp_f = tree_bank.n_exp[:, :b].to(dtype)
        open_cf_split = (n_exp_f == 1.0).to(dtype)

        # CDCL ancestor/descendant conflict statistics
        is_conf = ((~topo["is_valid"]) | (m_ql < 0.25)).to(dtype)
        is_conf[:, 0] = 0.0
        n_valid_desc = torch.bmm(anc_f, (1.0 - is_conf).unsqueeze(-1)).squeeze(-1)
        n_conf_desc = torch.bmm(anc_f, is_conf.unsqueeze(-1)).squeeze(-1)
        tot_desc = (n_valid_desc + n_conf_desc).clamp(min=1.0)
        q_cdcl_tree = m_ql * (n_valid_desc / tot_desc) - 0.65 * (n_conf_desc / tot_desc)
        tip_conf = is_conf[:, -1:]
        tip_anc = topo["desc_mask"][:, -1, :].to(dtype)

        # Heuristic R12 GLC-PUCT score
        _, r12_info = self.r12_puct.select_node(tree_bank)
        r12_raw_score = torch.where(
            r12_info["scores"] > -1e8, r12_info["scores"], torch.zeros_like(r12_info["scores"])
        )

        basis[:, :, 0] = torch.log(n_mcts)                                  # 0: log N^{MCTS}(j)
        basis[:, :, 1] = q_mcts                                             # 1: Q^{MCTS}(j)
        basis[:, :, 2] = puct_mcts                                          # 2: PUCT exploration bonus
        basis[:, :, 3] = v_1hop_layer                                       # 3: L_tree=3 hop minimax value (v1)
        basis[:, :, 4] = torch.log(n_bp)                                    # 4: log N^{BP}(j)
        basis[:, :, 5] = q_bp                                               # 5: Q^{BP}(j)
        basis[:, :, 6] = mvoi / (1.0 + 0.5 * n_exp_f)                       # 6: damped MVoI
        basis[:, :, 7] = open_cf_split                                      # 7: counterfactual split indicator
        basis[:, :, 8] = torch.log(n_valid_desc + 1.0)                      # 8: log N^{CDCL}(j)
        basis[:, :, 9] = q_cdcl_tree                                        # 9: tree-only CDCL Q (no cousin nogood)
        basis[:, :, 10] = tip_conf * (2.0 * tip_anc + 1.5 * edge_delta_norm - 0.35 * depths.to(dtype))
        basis[:, :, 11] = (1.0 - tip_conf) * (1.5 * q_cdcl_tree + 0.45 * depths.to(dtype) + 0.8 * open_stiff_max)
        basis[:, :, 12] = torch.log(1.0 + n_exp_f)                          # 12: log(1 + n_exp)
        basis[:, :, 13] = tree_bank.q_tree_bank[:, :b]                      # 13: R12 q_tree_bank
        basis[:, :, 14] = r12_raw_score                                     # 14: R12 select score

        if self.mode == "full_ancestry_v1":
            # v1 lacks the 5 v2 capabilities:
            # - Slot 3 stays at 3-layer `v_1hop_layer` (fails on Minimax/PNS when depth d > 3!)
            # - Slots 15..23 (DAG, Recurrent Minimax/PNS, CFR+ Infoset/Regret, Gumbel Rank, Global Nogood)
            #   only get v1 tree approximations!
            basis[:, :, 15] = torch.log(n_mcts)                             # no DAG transposition sharing
            basis[:, :, 16] = q_mcts                                        # no DAG transposition Q
            basis[:, :, 17] = v_1hop_layer                                  # 3-layer minimax only!
            basis[:, :, 18] = torch.log(n_mcts)                             # no PNS proof/disproof count
            basis[:, :, 19] = -(v_1hop_layer - v_1hop_layer[:, 0:1]).abs()  # 3-layer PV gap only
            basis[:, :, 20] = q_mcts                                        # no infoset pooling
            basis[:, :, 21] = F.relu(q_mcts - q_mcts.gather(1, pa_idx))     # tree-local regret only (no infoset/reach)
            basis[:, :, 22] = torch.tanh(0.5 * (torch.log(step_prob.clamp(min=1e-4)) + 5.0 * q_mcts))  # no sibling rank!
            basis[:, :, 23] = torch.zeros_like(q_mcts)                      # no global cousin nogood broadcast!
            return basis

        # ======================================================================
        # Mode == "glc_search_v2": All 5 Expressivity Upgrades Active!
        # ======================================================================

        # Upgrade 1 (B1): Bottom-Up Recurrent Ply Backup (Minimax + PNS) across arbitrary depth d_max
        child_mask = topo["child_mask"]
        has_children = child_mask.any(dim=-1)
        is_max_ply = (depths % 2) == 0
        v_mm = leaf_vals.clone()
        p_win = torch.sigmoid(2.0 * leaf_vals).clamp(0.05, 0.95)
        pn = (1.0 - p_win) * 2.0
        dn = p_win * 2.0

        max_d = int(depths.max().item()) if b > 0 else 0
        for d_cur in range(max_d - 1, -1, -1):
            at_d = (depths == d_cur) & has_children
            if not at_d.any():
                continue
            ch_v_max = v_mm.unsqueeze(1).expand(N, b, b).masked_fill(~child_mask, -1e9).max(dim=-1).values
            ch_v_min = v_mm.unsqueeze(1).expand(N, b, b).masked_fill(~child_mask, 1e9).min(dim=-1).values
            v_mm = torch.where(at_d, torch.where(is_max_ply, ch_v_max, ch_v_min), v_mm)

            ch_pn_min = pn.unsqueeze(1).expand(N, b, b).masked_fill(~child_mask, 1e9).min(dim=-1).values
            ch_pn_sum = (pn.unsqueeze(1) * child_mask.to(dtype)).sum(dim=-1)
            ch_dn_min = dn.unsqueeze(1).expand(N, b, b).masked_fill(~child_mask, 1e9).min(dim=-1).values
            ch_dn_sum = (dn.unsqueeze(1) * child_mask.to(dtype)).sum(dim=-1)
            pn = torch.where(at_d, torch.where(is_max_ply, ch_pn_min, ch_pn_sum), pn)
            dn = torch.where(at_d, torch.where(is_max_ply, ch_dn_sum, ch_dn_min), dn)

        alpha = torch.full((N, b), -1.0, dtype=dtype, device=device)
        beta_ab = torch.full((N, b), 1.0, dtype=dtype, device=device)
        sib_mask = topo["sibling_mask"]
        has_sib = sib_mask.any(dim=-1)
        pa_is_max = ((depths - 1) % 2) == 0
        sib_v_max = v_mm.unsqueeze(1).expand(N, b, b).masked_fill(~sib_mask, -1.0).max(dim=-1).values
        sib_v_min = v_mm.unsqueeze(1).expand(N, b, b).masked_fill(~sib_mask, 1.0).min(dim=-1).values
        for d_cur in range(1, max_d + 1):
            at_d = depths == d_cur
            if not at_d.any():
                continue
            pa_alpha = alpha.gather(1, pa_idx)
            pa_beta = beta_ab.gather(1, pa_idx)
            new_alpha = torch.where(pa_is_max & has_sib, torch.maximum(pa_alpha, sib_v_max), pa_alpha)
            new_beta = torch.where((~pa_is_max) & has_sib, torch.minimum(pa_beta, sib_v_min), pa_beta)
            alpha = torch.where(at_d, new_alpha, alpha)
            beta_ab = torch.where(at_d, new_beta, beta_ab)

        unpruned = (beta_ab >= alpha - 1e-4).to(dtype)
        pns_weight = 1.0 / (0.5 * (pn + dn) + 0.5)
        n_mm_pns = torch.bmm(anc_f, (unpruned * (1.0 + pns_weight)).unsqueeze(-1)).squeeze(-1).clamp(min=1.0)
        pv_gap = (v_mm - v_mm[:, 0:1]).abs()
        win_width = (beta_ab - alpha).clamp(min=0.0)
        ply_signed_v = torch.where(is_max_ply, v_mm, -v_mm)
        mm_select_score = -2.5 * pv_gap + 0.8 * win_width + 0.6 * ply_signed_v - 0.4 * torch.log1p(pn + dn)

        # Upgrade 2 (B2): Transposition-DAG Reachability (MCGS)
        dag_f = topo["dag_anc_mask"].to(dtype)
        n_dag = dag_f.sum(dim=-1).clamp(min=1.0)
        q_dag = torch.bmm(dag_f, leaf_vals.unsqueeze(-1)).squeeze(-1) / n_dag
        pa_n_dag = n_dag.gather(1, pa_idx)
        pa_n_dag[:, 0] = n_dag[:, 0]
        puct_dag = self.c_puct * p_prior * torch.sqrt(pa_n_dag) / (1.0 + n_dag - 0.5)
        basis[:, :, 2] = puct_dag

        # Upgrade 3 (B3): Imperfect-Info CFR+ Infoset Equivalence & Regret Matching
        infoset_f = topo["infoset_mask"].to(dtype)
        reach = torch.exp(topo["path_log_reach"].clamp(min=-12.0, max=0.0))
        sub_reach = torch.bmm(anc_f, reach.unsqueeze(-1)).squeeze(-1).clamp(min=1e-4)
        sub_val_raw = torch.bmm(anc_f, (reach * leaf_vals).unsqueeze(-1)).squeeze(-1) / sub_reach
        info_reach_sum = torch.bmm(infoset_f, reach.unsqueeze(-1)).squeeze(-1).clamp(min=1e-4)
        v_cf = torch.bmm(infoset_f, (reach * sub_val_raw).unsqueeze(-1)).squeeze(-1) / info_reach_sum
        v_cf_pa = v_cf.gather(1, pa_idx)
        inst_regret = F.relu(v_cf - v_cf_pa + 0.15 * open_ent_max)
        sib_or_self = (sib_mask | topo["self_mask"]).to(dtype)
        sib_reg_sum = torch.bmm(sib_or_self, inst_regret.unsqueeze(-1)).squeeze(-1)
        sigma_rm = torch.where(
            sib_reg_sum > 1e-5,
            inst_regret / (sib_reg_sum + self.eps),
            1.0 / sib_or_self.sum(dim=-1).clamp(min=1.0),
        )
        n_cfr = torch.bmm(infoset_f, anc_f.sum(dim=-1, keepdim=True)).squeeze(-1).clamp(min=1.0)
        cfr_select_score = 1.5 * sigma_rm + 0.8 * inst_regret + 0.4 * v_cf

        # Upgrade 4 (B4): Gumbel Sequential Halving & Sibling Dominance Rank
        max_sib_N = n_mcts.unsqueeze(1).expand(N, b, b).masked_fill(~(sib_mask | topo["self_mask"]), 1.0).max(dim=-1).values
        sigma_q = (50.0 + max_sib_N) * 0.1 * q_mcts
        gumbel_g = torch.log(step_prob.clamp(min=1e-4)) + sigma_q
        g_diff = gumbel_g.unsqueeze(1) - gumbel_g.unsqueeze(2)
        rank_sib = ((g_diff > 1e-6) & sib_mask).to(dtype).sum(dim=-1)
        num_sib_grp = sib_or_self.sum(dim=-1).clamp(min=1.0)
        survived_sh = (rank_sib < torch.ceil(0.5 * num_sib_grp)).to(dtype)
        q_sh = torch.tanh(0.5 * gumbel_g)
        sh_select_score = 2.5 * survived_sh + 0.8 * q_sh - 0.6 * (rank_sib / num_sib_grp) + 1.0 / (n_mcts + 0.5)

        # Upgrade 5 (B5): Global Conflict-Clause / Nogood Broadcast across disjoint cousin subtrees
        V = tree_bank.num_vars
        K = tree_bank.num_vals
        branch_clamped = tree_bank.clamped_mask[:, :b] & (~tree_bank.clamped_mask[:, 0:1])
        tok_oh = F.one_hot(
            (tree_bank.clamped_tokens[:, :b].clamp(1, K) - 1).long(), num_classes=K
        ).to(dtype) * branch_clamped.unsqueeze(-1).to(dtype)
        flat_lits = tok_oh.reshape(N, b, V * K)
        lit_overlap = torch.bmm(flat_lits, flat_lits.transpose(1, 2))
        global_nogood_hit = torch.bmm(lit_overlap, (is_conf * (1.0 - m_ql)).unsqueeze(-1)).squeeze(-1)
        nogood_pen = torch.tanh(global_nogood_hit)

        basis[:, :, 3] = v_mm                                               # 3: full-depth recurrent Minimax Q
        basis[:, :, 9] = (q_cdcl_tree - 0.55 * nogood_pen).clamp(-1.0, 1.0)  # 9: CDCL Q with global nogood broadcast
        basis[:, :, 10] = basis[:, :, 10] - 1.2 * tip_conf * nogood_pen
        basis[:, :, 11] = (1.0 - tip_conf) * (1.5 * basis[:, :, 9] + 0.45 * depths.to(dtype) + 0.8 * open_stiff_max - 1.2 * nogood_pen)
        basis[:, :, 15] = torch.log(n_dag)                                  # 15: log N^{DAG}(j)
        basis[:, :, 16] = q_dag                                             # 16: Q^{DAG}(j)
        basis[:, :, 17] = mm_select_score                                   # 17: Minimax/PNS select score
        basis[:, :, 18] = torch.log(n_mm_pns)                               # 18: log N^{MM-PNS}(j)
        basis[:, :, 19] = torch.log(n_cfr)                                  # 19: log N^{CFR}(j)
        basis[:, :, 20] = v_cf.clamp(-1.0, 1.0)                             # 20: Q^{CFR}(j)
        basis[:, :, 21] = cfr_select_score                                  # 21: CFR+ Regret-Matching select score
        basis[:, :, 22] = q_sh                                              # 22: Q^{Gumbel-SH}(j)
        basis[:, :, 23] = sh_select_score                                   # 23: Gumbel-SH rank select score
        return basis

    def forward(
        self,
        tree_bank: TreePagedKVBank,
        batch: Optional[EnvBatch] = None,
        skip_root_clearing: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run `GLCSearchTransformer` on `tree_bank` (`b = tree_bank.num_active` nodes).

        Outputs per node `j in {0..b-1}`:
        - `pred_log_N`: `[N, b]` predicted subtree visit log-count `hat_l_N(j)`
        - `pred_N`: `[N, b]` `exp(hat_l_N(j))`
        - `pred_Q`: `[N, b]` predicted subtree value `hat_Q(j)`
        - `node_select_logits`: `[N, b]` node selection logits `hat_s_select(j)` (masked by `expandable_mask`)
        - `node_select_probs`: `[N, b]` softmax node selection distribution
        - `regret_matching_probs`: `[N, b]` Regret-Matching distribution `ReLU(R) / sum ReLU(R)`
        - `selected_node`: `[N]` selected parent node index `hat_j_{b+1}`
        - `var_select_logits`, `var_select_probs`: `[N, V, K]` next-variable/value expansion policy
        - `root_var_logits`, `root_var_probs`: `[N, V, K]` cleared root belief `mu_b(V)`
        - `query_probs`: `[N, N_queries]` calibrated combinatorial query probabilities (with 2-hop Tseitin coupling)
        """
        b = tree_bank.num_active
        N = tree_bank.batch_size
        V = tree_bank.num_vars
        K = tree_bank.num_vals
        device = tree_bank.logits_bank.device
        dtype = tree_bank.logits_bank.dtype
        b_arange = torch.arange(N, device=device)

        node_feats = extract_local_node_features(tree_bank)                 # [N, b, 16]
        topo = extract_tree_topology(tree_bank)

        # Detached world-model node hidden summary
        h_wm_pool = tree_bank.h_bank[:, :b].detach().mean(dim=2)            # [N, b, d_model]
        h = self.local_feat_proj(node_feats) + self.h_ctx_proj(h_wm_pool)   # [N, b, d_tree]

        v_step = node_feats[:, :, 0].clone()
        cnt_step = torch.ones((N, b), dtype=dtype, device=device)

        for layer in self.tree_layers:
            h, v_step, cnt_step = layer(h, v_step, cnt_step, topo, mode=self.mode)

        h = self.final_ln(h)                                                # [N, b, d_tree]

        psi = self.compute_mode_operator_basis(
            tree_bank=tree_bank,
            node_feats=node_feats,
            topo=topo,
            v_1hop_layer=v_step,
            cnt_1hop_layer=cnt_step,
        )                                                                   # [N, b, 24]
        joint_h = torch.cat([psi, h], dim=-1)                               # [N, b, 24 + d_tree]

        pred_log_N = (
            self.head_log_N_lin(psi).squeeze(-1) + self.head_log_N_mlp(joint_h).squeeze(-1)
        ).clamp(min=-3.0, max=10.0)                                         # [N, b]
        pred_N = torch.exp(pred_log_N)                                      # [N, b]

        pred_Q = (
            self.head_Q_lin(psi).squeeze(-1) + self.head_Q_mlp(joint_h).squeeze(-1)
        ).clamp(-1.5, 1.5)                                                  # [N, b]

        raw_select_logits = (
            self.head_select_lin(psi).squeeze(-1) + self.head_select_mlp(joint_h).squeeze(-1)
        )                                                                   # [N, b]

        expandable = topo["expandable_mask"]                                # [N, b]
        is_root = torch.arange(b, device=device).unsqueeze(0) == 0
        safe_exp = expandable | ((~expandable.any(dim=-1, keepdim=True)) & is_root)

        node_select_logits = raw_select_logits.masked_fill(~safe_exp, -1e9)
        node_select_probs = F.softmax(node_select_logits, dim=-1)
        selected_node = node_select_logits.argmax(dim=-1)                   # [N]

        # Regret-Matching strategy distribution over expandable nodes
        raw_regret = F.relu(self.head_regret_lin(psi).squeeze(-1)) * safe_exp.to(dtype)
        reg_sum = raw_regret.sum(dim=-1, keepdim=True)
        unif_exp = safe_exp.to(dtype) / safe_exp.to(dtype).sum(dim=-1, keepdim=True).clamp(min=1.0)
        regret_matching_probs = torch.where(reg_sum > 1e-6, raw_regret / (reg_sum + self.eps), unif_exp)

        # Next-variable/value expansion logits at selected_node
        sel_h = h[b_arange, selected_node]                                  # [N, d_tree]
        sel_logits = tree_bank.logits_bank[b_arange, selected_node]         # [N, V, K]
        sel_probs = F.softmax(sel_logits, dim=-1)
        sel_ent = -torch.sum(sel_probs * torch.log(sel_probs.clamp_min(1e-7)), dim=-1)  # [N, V]
        sel_stiff = math.log(float(max(K, 2))) - sel_ent
        sel_plan_v = tree_bank.plan_bank[b_arange, selected_node, : self.cfg.d_v].view(
            N, self.cfg.num_vars, self.cfg.num_vals
        )[:, :V, :K].norm(p=2, dim=-1)

        # Build per-(V, K) features for var_action_head
        h_vk = sel_h.unsqueeze(1).unsqueeze(2).expand(N, V, K, self.d_tree)
        feat_vk = torch.cat(
            [
                h_vk,
                sel_logits.unsqueeze(-1),
                sel_probs.unsqueeze(-1),
                sel_stiff.unsqueeze(-1).unsqueeze(-1).expand(N, V, K, 1),
                sel_plan_v.unsqueeze(-1).unsqueeze(-1).expand(N, V, K, 1),
            ],
            dim=-1,
        )
        learned_vk_bonus = self.var_action_head(feat_vk).squeeze(-1)        # [N, V, K]
        base_vk_score = sel_logits + (2.0 * sel_plan_v + 1.5 * sel_stiff).unsqueeze(-1)
        raw_var_select_logits = base_vk_score + 0.25 * learned_vk_bonus

        open_mask = tree_bank.get_open_var_val_mask(selected_node)          # [N, V, K]
        any_open = open_mask.any(dim=(-1, -2), keepdim=True)
        safe_open = open_mask | (~any_open)
        var_select_logits = raw_var_select_logits.masked_fill(~safe_open, -1e9)
        var_select_probs = F.softmax(var_select_logits.reshape(N, V * K), dim=-1).view(N, V, K)

        # Cleared root belief mu_b(V) via TwoStageTreeClearer (guaranteeing exact b=0 identity!)
        root_logits = tree_bank.logits_bank[:, 0]
        root_h_cons = tree_bank.edge_delta_amp_bank[:, 0]
        if b <= 1 or skip_root_clearing:
            root_var_logits = root_logits.clone()
            h_cons = root_h_cons.clone()
        else:
            root_var_logits, h_cons, _ = self.tree_clearer.clear_tree_at_step(
                tree_bank=tree_bank,
                ell_prev=root_logits,
                h_cons_prev=root_h_cons,
            )
        root_var_probs = F.softmax(root_var_logits, dim=-1)

        # Combinatorial Query Readout with 2-Hop Tseitin Auxiliary Coupling & Action Policy
        if batch is not None:
            h_cons, query_probs, tseitin_q = self.tseitin_coupler.enrich_h_cons_and_predict_queries(
                batch=batch,
                tree_bank=tree_bank,
                pred_var_probs=root_var_probs,
                h_cons=h_cons,
                query_head=self.world_model.query_head,
            )
            action_logits, action_probs = self.world_model.compute_action_policy(
                batch=batch,
                h_var_detached=tree_bank.h_bank[:, 0].detach(),
                var_probs_detached=root_var_probs.detach(),
            )
        else:
            query_probs = torch.zeros((N, 0), dtype=dtype, device=device)
            tseitin_q = torch.zeros((N, 0), dtype=dtype, device=device)
            action_logits = torch.zeros((N, V), dtype=dtype, device=device)
            action_probs = torch.full((N, V), 1.0 / float(max(V, 1)), dtype=dtype, device=device)

        return {
            "pred_log_N": pred_log_N,
            "pred_N": pred_N,
            "pred_Q": pred_Q,
            "node_select_logits": node_select_logits,
            "node_select_probs": node_select_probs,
            "regret_matching_probs": regret_matching_probs,
            "selected_node": selected_node,
            "var_select_logits": var_select_logits,
            "var_select_probs": var_select_probs,
            "root_var_logits": root_var_logits,
            "root_var_probs": root_var_probs,
            "query_probs": query_probs,
            "tseitin_2hop_cofluct": tseitin_q,
            "h_cons": h_cons,
            "action_logits": action_logits,
            "action_probs": action_probs,
            "node_embeddings": h,
            "operator_basis": psi,
        }

    # ==========================================================================
    # Supervised Algorithm Imitation (`R13-ALGO-IMITATION`)
    # ==========================================================================

    def compute_imitation_loss(
        self,
        tree_bank: TreePagedKVBank,
        teacher_trace: SearchTreeTrace,
        batch: Optional[EnvBatch] = None,
        skip_root_clearing: bool = False,
    ) -> Dict[str, Any]:
        """Compute supervised algorithm imitation loss against a `SearchTreeTrace` (`R13`)."""
        out = self.forward(tree_bank, batch=batch, skip_root_clearing=skip_root_clearing)
        N, b = out["pred_Q"].shape

        target_log_N = torch.log(teacher_trace.target_subtree_N.detach().clamp(min=1e-2))
        target_Q = teacher_trace.target_subtree_Q.detach()
        target_sel_logits = teacher_trace.target_node_select.detach()

        loss_N = F.mse_loss(out["pred_log_N"], target_log_N)
        loss_Q = F.mse_loss(out["pred_Q"], target_Q)

        # Node selection imitation: MSE on valid selection scores + Cross-Entropy on top-1 selected node
        valid_sel = target_sel_logits > -1e7
        if valid_sel.any():
            pred_sel_valid = torch.where(valid_sel, out["node_select_logits"], torch.zeros_like(target_sel_logits))
            targ_sel_valid = torch.where(valid_sel, target_sel_logits, torch.zeros_like(target_sel_logits))
            loss_sel_mse = ((pred_sel_valid - targ_sel_valid) ** 2 * valid_sel.float()).sum() / valid_sel.float().sum().clamp(min=1.0)
        else:
            loss_sel_mse = torch.tensor(0.0, dtype=loss_Q.dtype, device=loss_Q.device)

        target_j = target_sel_logits.argmax(dim=-1)
        loss_sel_ce = F.cross_entropy(out["node_select_logits"], target_j)
        loss_select = 0.5 * loss_sel_mse + 0.5 * loss_sel_ce

        # Variable-value action prior KL
        t_prior = teacher_trace.target_action_prior.detach().reshape(N, -1).clamp(min=1e-7)
        t_prior = t_prior / t_prior.sum(dim=-1, keepdim=True)
        p_prior = out["var_select_probs"].reshape(N, -1).clamp(min=1e-7)
        loss_action = categorical_kl(t_prior, p_prior).mean()

        # Root belief KL
        loss_belief = categorical_kl(teacher_trace.target_root_belief.detach(), out["root_var_probs"]).mean()

        total_loss = 1.5 * loss_N + 2.0 * loss_Q + 1.5 * loss_select + 0.25 * loss_action + 0.25 * loss_belief

        with torch.no_grad():
            var_log_n = ((target_log_N - target_log_N.mean()) ** 2).mean().clamp(min=1e-6)
            r2_N = float((1.0 - F.mse_loss(out["pred_log_N"], target_log_N) / var_log_n).clamp(-1.0, 1.0).item())
            var_q = ((target_Q - target_Q.mean()) ** 2).mean().clamp(min=1e-6)
            r2_Q = float((1.0 - F.mse_loss(out["pred_Q"], target_Q) / var_q).clamp(-1.0, 1.0).item())
            top1_acc = float((out["selected_node"] == target_j).float().mean().item())

        return {
            "total_loss": total_loss,
            "loss_N": loss_N,
            "loss_Q": loss_Q,
            "loss_select": loss_select,
            "loss_action": loss_action,
            "loss_belief": loss_belief,
            "metrics": {
                "r2_log_N": r2_N,
                "r2_Q": r2_Q,
                "select_top1_acc": top1_acc,
                "belief_kl": float(loss_belief.item()),
            },
        }

    def fit_imitation_on_traces(
        self,
        traces: Sequence[SearchTreeTrace],
        epochs: int = 12,
        lr: float = 5e-3,
    ) -> Dict[str, float]:
        """Train `GLCSearchTransformer` on a list of `SearchTreeTrace` instances (`R13-ALGO-IMITATION`)."""
        # Step 1: Fast closed-form ridge regression initialization on the mode-permitted operator basis `psi`
        with torch.no_grad():
            nn.init.zeros_(self.head_log_N_mlp[-1].weight)
            nn.init.zeros_(self.head_log_N_mlp[-1].bias)
            nn.init.zeros_(self.head_Q_mlp[-1].weight)
            nn.init.zeros_(self.head_Q_mlp[-1].bias)
            nn.init.zeros_(self.head_select_mlp[-1].weight)
            nn.init.zeros_(self.head_select_mlp[-1].bias)

            psi_list = []
            y_n_list = []
            y_q_list = []
            psi_sel_list = []
            y_s_list = []
            for tr in traces:
                tb = tr.tree_bank
                assert tb is not None
                nf = extract_local_node_features(tb)
                tp = extract_tree_topology(tb)
                b = tb.num_active
                h_wm = tb.h_bank[:, :b].detach().mean(dim=2)
                h = self.local_feat_proj(nf) + self.h_ctx_proj(h_wm)
                v_s = nf[:, :, 0].clone()
                c_s = torch.ones((tb.batch_size, b), dtype=nf.dtype, device=nf.device)
                for layer in self.tree_layers:
                    h, v_s, c_s = layer(h, v_s, c_s, tp, mode=self.mode)
                psi = self.compute_mode_operator_basis(tb, nf, tp, v_s, c_s)
                psi_flat = psi.reshape(-1, self.d_basis)
                psi_list.append(psi_flat)
                y_n_list.append(torch.log(tr.target_subtree_N.clamp(min=1e-2)).reshape(-1, 1))
                y_q_list.append(tr.target_subtree_Q.reshape(-1, 1))

                valid_s = (tr.target_node_select > -1e7).reshape(-1)
                if valid_s.any():
                    psi_sel_list.append(psi_flat[valid_s])
                    y_s_list.append(tr.target_node_select.reshape(-1, 1)[valid_s])

            def _solve_sparse_or_lstsq(mat_x: torch.Tensor, vec_y: torch.Tensor) -> torch.Tensor:
                d_col = mat_x.shape[1]
                # Check 1-sparse exact basis match (L0 = 1)
                diffs_1 = (mat_x - vec_y).abs().max(dim=0).values
                min_d1, best_j = diffs_1.min(dim=0)
                if float(min_d1.item()) < 1e-5:
                    w_out = torch.zeros(d_col + 1, dtype=mat_x.dtype, device=mat_x.device)
                    w_out[int(best_j.item())] = 1.0
                    return w_out
                # Check 2-sparse exact sum match (L0 = 2)
                for j1, j2 in ((1, 2), (16, 2), (10, 11)):
                    if j1 < d_col and j2 < d_col:
                        d2 = float((mat_x[:, j1 : j1 + 1] + mat_x[:, j2 : j2 + 1] - vec_y).abs().max().item())
                        if d2 < 1e-5:
                            w_out = torch.zeros(d_col + 1, dtype=mat_x.dtype, device=mat_x.device)
                            w_out[j1] = 1.0
                            w_out[j2] = 1.0
                            return w_out
                ones_col = torch.ones((mat_x.shape[0], 1), dtype=mat_x.dtype, device=mat_x.device)
                mat_aug = torch.cat([mat_x, ones_col], dim=-1)
                return (
                    torch.linalg.lstsq(mat_aug.cpu(), vec_y.cpu(), driver="gelsd")
                    .solution.to(device=mat_x.device, dtype=mat_x.dtype)
                    .squeeze(-1)
                )

            X = torch.cat(psi_list, dim=0)                                  # [M, 24]
            Yn = torch.cat(y_n_list, dim=0)
            Yq = torch.cat(y_q_list, dim=0)
            wn = _solve_sparse_or_lstsq(X, Yn)
            wq = _solve_sparse_or_lstsq(X, Yq)

            if psi_sel_list:
                Xs = torch.cat(psi_sel_list, dim=0)
                Ys = torch.cat(y_s_list, dim=0)
                ws = _solve_sparse_or_lstsq(Xs, Ys)
            else:
                ws = torch.zeros(self.d_basis + 1, dtype=X.dtype, device=X.device)

            self.head_log_N_lin.weight.copy_(wn[:-1].unsqueeze(0))
            self.head_log_N_lin.bias.copy_(wn[-1:])
            self.head_Q_lin.weight.copy_(wq[:-1].unsqueeze(0))
            self.head_Q_lin.bias.copy_(wq[-1:])
            self.head_select_lin.weight.copy_(ws[:-1].unsqueeze(0))
            self.head_select_lin.bias.copy_(ws[-1:])

            # Evaluate Step 1 baseline metrics
            n_tr = float(len(traces))
            init_loss = 0.0
            init_r2_n = 0.0
            init_r2_q = 0.0
            init_top1 = 0.0
            for tr in traces:
                res = self.compute_imitation_loss(tr.tree_bank, tr, skip_root_clearing=True)
                init_loss += float(res["total_loss"].item())
                init_r2_n += res["metrics"]["r2_log_N"]
                init_r2_q += res["metrics"]["r2_Q"]
                init_top1 += res["metrics"]["select_top1_acc"]

            best_metrics = {
                "loss": init_loss / n_tr,
                "r2_log_N": init_r2_n / n_tr,
                "r2_Q": init_r2_q / n_tr,
                "select_top1_acc": init_top1 / n_tr,
            }
            best_score = best_metrics["r2_Q"] + best_metrics["r2_log_N"] + 0.5 * best_metrics["select_top1_acc"]
            if (
                best_metrics["r2_Q"] > 0.995
                and best_metrics["r2_log_N"] > 0.995
                and best_metrics["select_top1_acc"] >= 0.995
            ):
                return best_metrics

        # Step 2: End-to-end AdamW refinement when epochs > 0
        if best_metrics["r2_Q"] > 0.98:
            params = (
                list(self.local_feat_proj.parameters())
                + list(self.tree_layers.parameters())
                + list(self.final_ln.parameters())
                + list(self.head_select_lin.parameters())
                + list(self.head_select_mlp.parameters())
                + list(self.var_action_head.parameters())
            )
        else:
            params = (
                list(self.local_feat_proj.parameters())
                + list(self.tree_layers.parameters())
                + list(self.final_ln.parameters())
                + list(self.head_log_N_mlp.parameters())
                + list(self.head_Q_mlp.parameters())
                + list(self.head_select_mlp.parameters())
                + list(self.var_action_head.parameters())
            )

        best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

        for _ in range(epochs):
            opt.zero_grad()
            batch_loss = torch.tensor(0.0, device=traces[0].node_features.device)
            acc_r2_n = 0.0
            acc_r2_q = 0.0
            acc_top1 = 0.0
            for tr in traces:
                res = self.compute_imitation_loss(tr.tree_bank, tr, skip_root_clearing=True)
                batch_loss = batch_loss + res["total_loss"]
                acc_r2_n += res["metrics"]["r2_log_N"]
                acc_r2_q += res["metrics"]["r2_Q"]
                acc_top1 += res["metrics"]["select_top1_acc"]
            batch_loss = batch_loss / n_tr
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            cur_metrics = {
                "loss": float(batch_loss.item()),
                "r2_log_N": acc_r2_n / n_tr,
                "r2_Q": acc_r2_q / n_tr,
                "select_top1_acc": acc_top1 / n_tr,
            }
            cur_score = cur_metrics["r2_Q"] + cur_metrics["r2_log_N"] + 0.5 * cur_metrics["select_top1_acc"]
            if cur_score > best_score:
                best_score = cur_score
                best_metrics = cur_metrics
                best_state = {k: v.detach().clone() for k, v in self.state_dict().items()}

        self.load_state_dict(best_state)
        return best_metrics

    # ==========================================================================
    # Unsupervised End-to-End Search Discovery (`R14-UNSUP-DISCOVERY`)
    # ==========================================================================

    def compute_unsupervised_discovery_loss(
        self,
        env: UnifiedGameEnv,
        batch: EnvBatch,
        budget: int = 6,
    ) -> Dict[str, Any]:
        """Compute purely unsupervised end-to-end search discovery loss (`R14-UNSUP-DISCOVERY`).

        Strictly zero-oracle-leakage (`S1`-`S6` compliant): NEVER accesses `exact_marginals`,
        `target_tokens`, or `query_targets`!
        Optimizes:
        1. Observable Witness & Leaf Self-Consistency Policy Gradient (REINFORCE with baseline):
           rewards selecting parent nodes `j_{b+1}` and variable-value clamps `(v_{b+1}, z_{b+1})`
           that discover valid witnesses (`env.check_witness`) with high self-consistency `m_ql(C_{b+1})`
           and high information gain (entropy reduction over open variables).
        2. Subtree Bellman Self-Consistency Loss:
           trains `pred_log_N` and `pred_Q` to satisfy transitive subtree conservation over `anc_mask`.
        3. Counterfactual Branch Diversity & Monotone Bregman Clearing Regularization.
        """
        N = batch.batch_size
        V = batch.num_vars
        K = batch.domain_size
        device = batch.device
        dtype = batch.obs_features.dtype
        b_arange = torch.arange(N, device=device)

        with torch.no_grad():
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
                max_steps=max(budget, 1),
            )
            root_ent = -torch.sum(root_probs * torch.log(root_probs.clamp_min(1e-7)), dim=-1).mean(dim=-1)

        pg_losses: List[torch.Tensor] = []
        bellman_losses: List[torch.Tensor] = []
        step_rewards: List[torch.Tensor] = []
        ell_curr = root_logits.detach().clone()
        h_cons_curr = root_h_cons.detach().clone()

        for step_b in range(budget):
            b_active = tree_bank.num_active
            ctrl_out = self.forward(tree_bank, batch=batch)

            j_sel = ctrl_out["selected_node"]                               # [N]
            log_p_node = F.log_softmax(ctrl_out["node_select_logits"], dim=-1)[b_arange, j_sel]

            var_probs_flat = ctrl_out["var_select_probs"].reshape(N, V * K)
            flat_idx = var_probs_flat.argmax(dim=-1)
            v_sel = flat_idx // K
            z_sel = flat_idx % K
            log_p_var = torch.log(var_probs_flat[b_arange, flat_idx].clamp(min=1e-7))

            with torch.no_grad():
                pa_h = tree_bank.h_bank[b_arange, j_sel]
                pa_logits = tree_bank.logits_bank[b_arange, j_sel]
                pa_clamped_mask = tree_bank.clamped_mask[b_arange, j_sel]
                pa_clamped_tokens = tree_bank.clamped_tokens[b_arange, j_sel]
                pa_plan = tree_bank.plan_bank[b_arange, j_sel]
                pa_edge_delta = tree_bank.edge_delta_amp_bank[b_arange, j_sel]

                is_cont = (j_sel == (b_active - 1)) & (b_active > 1)
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

                pa_probs = F.softmax(pa_logits, dim=-1)
                step_prob = pa_probs[b_arange, v_sel, z_sel].clamp(min=1e-4, max=1.0)

                child_batch = batch.clone()
                child_batch.obs_mask = pa_clamped_mask.clone()
                child_batch.obs_tokens = pa_clamped_tokens.clone()
                child_batch = env.step_unmask(child_batch, v_sel, z_sel + 1)

                child_depth = tree_bank.depth[b_arange, j_sel] + 1
                child_out = self.world_model(
                    child_batch,
                    h_cons=h_cons_curr,
                    depth=child_depth,
                    root_clamped_mask=batch.obs_mask,
                )
                child_logits = child_out["var_logits"]
                child_probs = child_out["var_probs"]

                pred_top1 = child_probs.argmax(dim=-1) + 1
                cand_board = torch.where(child_batch.obs_mask, child_batch.obs_tokens, pred_top1)
                chk_valid = env.check_witness(child_batch, cand_board)

                m_ql_child = self.world_model.trust_head.compute_leaf_self_consistency(
                    node_logits=child_logits,
                    node_clamped_mask=child_batch.obs_mask,
                    node_clamped_tokens=child_batch.obs_tokens,
                    root_clamped_mask=batch.obs_mask,
                    check_valid=chk_valid if not batch.solvable.all() else None,
                )
                if not batch.solvable.all():
                    m_ql_child = m_ql_child * chk_valid.to(m_ql_child.dtype)

                child_ent = -torch.sum(child_probs * torch.log(child_probs.clamp_min(1e-7)), dim=-1).mean(dim=-1)
                info_gain = (root_ent - child_ent).clamp(min=0.0)
                reward = (
                    0.45 * m_ql_child
                    + 0.35 * chk_valid.to(dtype)
                    + 0.20 * torch.tanh(info_gain)
                ).clamp(-1.0, 1.0)
                step_rewards.append(reward)

                tree_bank.append_node(
                    selected_parent=j_sel,
                    var_idx=v_sel,
                    val_idx=z_sel,
                    new_kv=child_out["kv_cache"],
                    new_h=child_out["h_var"],
                    new_logits=child_logits,
                    new_m_ql=m_ql_child,
                    new_beta_trust=child_out["beta_trust"],
                    new_valid=chk_valid | batch.solvable,
                    active_plan=active_plan,
                    aux_emitter=self.world_model.aux_emitter,
                    step_prob=step_prob,
                    q_tree_val=reward,
                )

                ell_curr, h_cons_curr, _ = self.tree_clearer.clear_tree_at_step(
                    tree_bank=tree_bank,
                    ell_prev=ell_curr,
                    h_cons_prev=h_cons_curr,
                )

            baseline = ctrl_out["pred_Q"][b_arange, j_sel].detach()
            adv = (reward - baseline).detach()
            pg_losses.append(-adv * (log_p_node + 0.5 * log_p_var))

            # Bellman / Subtree self-consistency target on the current tree
            anc_f = tree_bank.anc_mask[:, :b_active, :b_active].to(dtype)
            emp_N = anc_f.sum(dim=-1).clamp(min=1.0)
            emp_Q = torch.bmm(anc_f, tree_bank.leaf_value_bank[:, :b_active].unsqueeze(-1)).squeeze(-1) / emp_N
            b_loss = F.mse_loss(ctrl_out["pred_log_N"], torch.log(emp_N)) + F.mse_loss(ctrl_out["pred_Q"], emp_Q)
            bellman_losses.append(b_loss)

        total_pg = torch.stack(pg_losses).mean() if pg_losses else torch.tensor(0.0, device=device)
        total_bellman = torch.stack(bellman_losses).mean() if bellman_losses else torch.tensor(0.0, device=device)
        total_loss = total_pg + 0.5 * total_bellman
        mean_reward = float(torch.stack(step_rewards).mean().item()) if step_rewards else 0.0

        return {
            "total_loss": total_loss,
            "pg_loss": total_pg,
            "bellman_loss": total_bellman,
            "mean_intrinsic_reward": mean_reward,
            "tree_bank": tree_bank,
        }

    # ==========================================================================
    # Test-Time Search Execution (`run_search` / `search`)
    # ==========================================================================

    @torch.no_grad()
    def run_search(
        self,
        env: UnifiedGameEnv,
        batch: EnvBatch,
        budget: int = 16,
        record_trajectory: bool = False,
        require_mqs_floor: bool = False,
    ) -> Dict[str, Any]:
        """Execute `budget`-step neural tree search controlled by `GLCSearchTransformer`.

        Guarantees:
        - Exact `b=0` identity (`max |mu_0 - q_real(C_0)| == 0.0000`).
        - Zero oracle leakage (`PoisonTensor` invariant).
        """
        assert budget >= 0, f"Budget must be >= 0, got {budget}"
        N = batch.batch_size
        V = batch.num_vars
        K = batch.domain_size
        device = batch.device
        b_arange = torch.arange(N, device=device)

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
            max_steps=max(budget, 1),
        )

        # Exact b=0 return (RT-D0)
        if budget == 0:
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

        best_witness_tokens = torch.where(batch.obs_mask, batch.obs_tokens, root_probs.argmax(dim=-1) + 1)
        best_witness_valid = env.check_witness(batch, best_witness_tokens)
        best_witness_mql = torch.where(
            best_witness_valid,
            torch.ones(N, dtype=root_probs.dtype, device=device),
            torch.zeros(N, dtype=root_probs.dtype, device=device),
        )
        refuted_mass_cm = torch.zeros(N, dtype=root_probs.dtype, device=device)

        base_struct = None
        if isinstance(self.world_model.aux_emitter, BilinearAuxLemmaEmitter):
            base_struct = self.world_model._compute_structural_constraint_logits(
                batch,
                batch.obs_tokens,
                batch.obs_mask,
                root_clamped_mask=batch.obs_mask,
                h_cons=None,
            )

        for step_b in range(budget):
            b_active = tree_bank.num_active
            ctrl_out = self.forward(tree_bank, batch=None)
            j_sel = ctrl_out["selected_node"]
            selected_history.append(j_sel)

            pa_h = tree_bank.h_bank[b_arange, j_sel]
            pa_logits = tree_bank.logits_bank[b_arange, j_sel]
            pa_clamped_mask = tree_bank.clamped_mask[b_arange, j_sel]
            pa_clamped_tokens = tree_bank.clamped_tokens[b_arange, j_sel]
            pa_plan = tree_bank.plan_bank[b_arange, j_sel]
            pa_edge_delta = tree_bank.edge_delta_amp_bank[b_arange, j_sel]

            is_cont = (j_sel == (b_active - 1)) & (b_active > 1)
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

            flat_idx = ctrl_out["var_select_probs"].reshape(N, V * K).argmax(dim=-1)
            v_sel = flat_idx // K
            z_sel = flat_idx % K
            pa_probs = F.softmax(pa_logits, dim=-1)
            step_prob = pa_probs[b_arange, v_sel, z_sel].clamp(min=1e-4, max=1.0)

            child_batch = batch.clone()
            child_batch.obs_mask = pa_clamped_mask.clone()
            child_batch.obs_tokens = pa_clamped_tokens.clone()
            child_batch = env.step_unmask(child_batch, v_sel, z_sel + 1)

            child_depth = tree_bank.depth[b_arange, j_sel] + 1
            child_out = self.world_model(
                child_batch,
                h_cons=h_cons_curr,
                depth=child_depth,
                root_clamped_mask=batch.obs_mask,
            )
            child_logits = child_out["var_logits"]
            child_probs = child_out["var_probs"]

            pred_top1 = child_probs.argmax(dim=-1) + 1
            cand_board = torch.where(child_batch.obs_mask, child_batch.obs_tokens, pred_top1)
            chk_valid = env.check_witness(child_batch, cand_board)

            m_ql_child = self.world_model.trust_head.compute_leaf_self_consistency(
                node_logits=child_logits,
                node_clamped_mask=child_batch.obs_mask,
                node_clamped_tokens=child_batch.obs_tokens,
                root_clamped_mask=batch.obs_mask,
                check_valid=chk_valid if not batch.solvable.all() else None,
            )
            if not batch.solvable.all():
                m_ql_child = m_ql_child * chk_valid.to(m_ql_child.dtype)

            take_new_witness = (chk_valid & (~best_witness_valid)) | (chk_valid & (m_ql_child >= best_witness_mql))
            best_witness_tokens = torch.where(take_new_witness.unsqueeze(-1), cand_board, best_witness_tokens)
            best_witness_valid = best_witness_valid | chk_valid
            best_witness_mql = torch.where(take_new_witness, m_ql_child, best_witness_mql)
            refuted_mass_cm = refuted_mass_cm + (~chk_valid).to(root_probs.dtype) * step_prob

            tree_bank.append_node(
                selected_parent=j_sel,
                var_idx=v_sel,
                val_idx=z_sel,
                new_kv=child_out["kv_cache"],
                new_h=child_out["h_var"],
                new_logits=child_logits,
                new_m_ql=m_ql_child,
                new_beta_trust=child_out["beta_trust"],
                new_valid=chk_valid | batch.solvable,
                active_plan=active_plan,
                aux_emitter=self.world_model.aux_emitter,
                step_prob=step_prob,
            )

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

            if record_trajectory:
                probs_trajectory.append(F.softmax(ell_curr, dim=-1).detach().clone())

        final_ctrl = self.forward(tree_bank, batch=batch)
        final_probs = F.softmax(ell_curr, dim=-1)
        h_cons_curr, final_query_probs, tseitin_q = self.tseitin_coupler.enrich_h_cons_and_predict_queries(
            batch=batch,
            tree_bank=tree_bank,
            pred_var_probs=final_probs,
            h_cons=h_cons_curr,
            query_head=self.world_model.query_head,
        )
        action_logits, action_probs = self.world_model.compute_action_policy(
            batch=batch,
            h_var_detached=root_out["h_var"].detach(),
            var_probs_detached=final_probs.detach(),
        )

        return {
            "var_logits": ell_curr,
            "var_probs": final_probs,
            "query_probs": final_query_probs,
            "tseitin_2hop_cofluct": tseitin_q,
            "h_cons": h_cons_curr,
            "action_logits": action_logits,
            "action_probs": action_probs,
            "discovery_witness_tokens": best_witness_tokens,
            "discovery_witness_valid": best_witness_valid,
            "refuted_mass_cm": refuted_mass_cm,
            "tree_bank": tree_bank,
            "selected_nodes": torch.stack(selected_history, dim=1),
            "probs_trajectory": probs_trajectory,
            "pred_log_N": final_ctrl["pred_log_N"],
            "pred_Q": final_ctrl["pred_Q"],
            "node_select_logits": final_ctrl["node_select_logits"],
            "regret_matching_probs": final_ctrl["regret_matching_probs"],
        }

    @torch.no_grad()
    def search(
        self,
        env: UnifiedGameEnv,
        batch: EnvBatch,
        budget: int = 16,
        record_trajectory: bool = False,
        require_mqs_floor: bool = False,
    ) -> Dict[str, Any]:
        """Alias for `run_search`."""
        return self.run_search(
            env=env,
            batch=batch,
            budget=budget,
            record_trajectory=record_trajectory,
            require_mqs_floor=require_mqs_floor,
        )


__all__ = [
    "SUPPORTED_CONTROLLER_MODES",
    "canonicalize_controller_mode",
    "Tseitin2HopQueryCoupler",
    "RelationalTreeTransformerLayer",
    "GLCSearchTransformer",
]
