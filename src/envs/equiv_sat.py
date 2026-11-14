"""Latent Equivalence / Parity & Direct Combinatorial Query (`P(A == B)`) Benchmark (`Equiv-SAT`).

Tests two core capabilities from the Society of Thinkers Charter:
1. Multi-step deduction over parity / equivalence chains when an anchor variable is observed.
2. Direct relational query rollouts `Q(A, B)` (`P(X_A == X_B)`) and bidirectional latent lemmas
   (`D_amp >> |V|`) on unanchored ("floating") equivalence components where exact single-variable
   marginals are strictly 50/50 (`P*(X_A = 1) = P*(X_B = 1) = 0.5`), so any architecture limited to
   independent node marginals is blind (`P_marg(A == B) = 0.5`), whereas conditioning on `X_A = z`
   or carrying relational lemmas in `D_amp` recovers `P*(X_A == X_B) in {0, 1}` exactly.

Variable & Token Encoding:
- Header: `[CONSTRAINT_CSP (1), EQUIV_SAT (10), 1, N_vars, num_clauses, EPISTEMIC_TRUTH (0)]`
- `obs_tokens`: `[B, N_vars]` in `{0 (unobserved), 1 (bit=0), 2 (bit=1)}`
- `obs_features`: `[B, N_vars, D_feat]` encoding incident clause structure (clause-incidence features
  so a single-world model can read local parity constraints without seeing global transitive closures).
- `query_pairs`: `[B, N_queries, 3]` `(var_a, var_b, QueryType.EQUAL)` testing both anchored and
  floating multi-hop equivalence pairs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch

from .base import (
    EnvBatch,
    GameFamily,
    GameID,
    QueryType,
    UnifiedGameEnv,
    UtilitySpec,
    bernoulli_kl,
)


class EquivSatEnv(UnifiedGameEnv):
    """Vectorized Latent Equivalence / Parity (`Equiv-SAT`) environment with exact GF(2) / Bayes oracle."""

    game_family = GameFamily.CONSTRAINT_CSP
    game_id = GameID.EQUIV_SAT

    def __init__(
        self,
        num_vars: int = 16,
        num_clauses: int = 12,
        num_queries: int = 8,
        num_anchored: int = 2,
        clause_noise: float = 0.0,
        unsolvable_frac: float = 0.0,
    ):
        self.num_vars = int(num_vars)
        self.num_clauses = int(num_clauses)
        self.num_queries = int(num_queries)
        self.num_anchored = int(num_anchored)
        self.clause_noise = float(clause_noise)
        self.unsolvable_frac = float(unsolvable_frac)

        self.grid_h = 1
        self.grid_w = self.num_vars
        self.inventory_spec = self.num_clauses * 100 + self.num_queries
        self.default_utility_spec = UtilitySpec.EPISTEMIC_TRUTH
        self.domain_size = 2  # token 1 -> bit 0, token 2 -> bit 1

        # Feature dimension per variable:
        # For each of `num_clauses` binary parity clauses (x_u XOR x_v = c_m):
        # +1 if var is u and c_m=0 (same), -1 if var is u and c_m=1 (diff),
        # plus normalized variable index and observed status -> 2 * num_clauses + 2 features.
        self.feat_dim = 2 * self.num_clauses + 2

    def sample_batch(
        self,
        batch_size: int,
        split: str = "test",
        seed: int = 0,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> EnvBatch:
        """Generate `batch_size` Equiv-SAT instances with exact marginals and `P(A == B)` targets.

        Each instance partitions the `N_vars` variables into:
        - Component 0 (anchored): a chain of length `N_vars // 2` where `num_anchored` roots are observed
          (so multi-step propagation determines `P*(X_i) in {0, 1}`).
        - Component 1 (floating / unanchored): a chain of length `N_vars - N_vars // 2` with ZERO observed
          variables at the root! Every variable in Component 1 has exact marginal `P*(X_i = 1) = 0.5`,
          yet every pair `(a, b)` inside Component 1 has a deterministic relation `P*(X_a == X_b) in {0, 1}`
          determined by the XOR sum along the chain!
        """
        dev = torch.device(device)
        split_offset = {"train": 0, "val": 100_000, "cal": 200_000, "test": 300_000}.get(split, 400_000)
        rng = np.random.default_rng([seed, split_offset, self.num_vars, self.num_clauses])

        B = batch_size
        N = self.num_vars
        M = self.num_clauses
        Q = self.num_queries
        half = N // 2

        obs_tokens = np.zeros((B, N), dtype=np.int64)
        obs_mask = np.zeros((B, N), dtype=bool)
        target_tokens = np.zeros((B, N), dtype=np.int64)
        exact_marginals = np.full((B, N, 2), 0.5, dtype=np.float32)
        obs_features = np.zeros((B, N, self.feat_dim), dtype=np.float32)
        query_pairs = np.zeros((B, Q, 3), dtype=np.int64)
        query_targets = np.zeros((B, Q), dtype=np.float32)
        solvable = np.ones(B, dtype=bool)

        clause_edges = np.zeros((B, M, 3), dtype=np.int64)  # (u, v, rhs_xor)

        for b in range(B):
            perm = rng.permutation(N)
            comp0 = perm[:half]
            comp1 = perm[half:]

            # Draw ground-truth bit assignment x* in {0, 1}^N
            bits = rng.integers(0, 2, size=N, dtype=np.int64)
            target_tokens[b] = bits + 1  # 1 -> bit 0, 2 -> bit 1

            # Build spanning chains on comp0 and comp1
            edges = []
            for idx in range(len(comp0) - 1):
                u, v = int(comp0[idx]), int(comp0[idx + 1])
                rhs = int(bits[u] ^ bits[v])
                edges.append((u, v, rhs))
            for idx in range(len(comp1) - 1):
                u, v = int(comp1[idx]), int(comp1[idx + 1])
                rhs = int(bits[u] ^ bits[v])
                edges.append((u, v, rhs))

            # Pad or trim to M clauses using redundant consistent chords within components
            while len(edges) < M:
                comp = comp0 if rng.random() < 0.5 else comp1
                u, v = rng.choice(comp, size=2, replace=False)
                u, v = int(u), int(v)
                edges.append((u, v, int(bits[u] ^ bits[v])))
            if len(edges) > M:
                # Keep chain edges from both components so both have multi-hop paths
                n0 = min(half - 1, M // 2)
                n1 = min(len(comp1) - 1, M - n0)
                edges = edges[:n0] + edges[half - 1 : half - 1 + n1]
                while len(edges) < M:
                    u, v = int(comp0[0]), int(comp0[-1])
                    edges.append((u, v, int(bits[u] ^ bits[v])))

            rng.shuffle(edges)

            # Optionally inject an unsolvable parity contradiction on a cycle
            is_unsolv = rng.random() < self.unsolvable_frac
            if is_unsolv and len(comp0) >= 3 and M >= 3:
                u0, u1, u2 = int(comp0[0]), int(comp0[1]), int(comp0[2])
                edges[0] = (u0, u1, int(bits[u0] ^ bits[u1]))
                edges[1] = (u1, u2, int(bits[u1] ^ bits[u2]))
                edges[2] = (u0, u2, int(1 ^ bits[u0] ^ bits[u2]))  # contradictory chord!
                solvable[b] = False

            for m_idx, (u, v, rhs) in enumerate(edges):
                clause_edges[b, m_idx] = (u, v, rhs)
                # Incident clause encoding on endpoints u and v:
                obs_features[b, u, 2 * m_idx] = 1.0
                obs_features[b, v, 2 * m_idx] = 1.0
                sign = -1.0 if rhs == 1 else 1.0
                obs_features[b, u, 2 * m_idx + 1] = sign
                obs_features[b, v, 2 * m_idx + 1] = sign

            # Observe `num_anchored` variables in comp0 ONLY; comp1 remains 100% floating!
            n_anc = min(self.num_anchored, len(comp0))
            for a_idx in range(n_anc):
                v_anc = int(comp0[a_idx])
                obs_mask[b, v_anc] = True
                obs_tokens[b, v_anc] = int(bits[v_anc] + 1)

            obs_features[b, :, -2] = np.linspace(0.0, 1.0, N, dtype=np.float32)
            obs_features[b, :, -1] = obs_mask[b].astype(np.float32)

            # Exact posterior via connected-component parity propagation
            marg_b, pair_eq_b, is_consistent = self._solve_exact_gf2(
                num_vars=N,
                edges=clause_edges[b],
                obs_mask=obs_mask[b],
                obs_tokens=obs_tokens[b],
            )
            solvable[b] = is_consistent
            exact_marginals[b] = marg_b

            # Construct queries Q(A, B):
            # Half of queries are floating multi-hop pairs inside comp1 (where node marginals are 0.5/0.5
            # so only conditional rollouts or D_amp relational lemmas can solve them!);
            # Half of queries are anchored multi-hop pairs inside comp0 or cross-component pairs (0.5).
            for q_idx in range(Q):
                if q_idx < (Q * 3) // 4 and len(comp1) >= 2:
                    # Floating equivalence query inside comp1 (exact P*(A == B) in {0, 1}, marginal = 0.5!)
                    d_max = min(len(comp1) - 1, max(1, M - (half - 1)))
                    i1 = int(rng.integers(0, max(1, d_max)))
                    i2 = int(rng.integers(i1 + 1, d_max + 1))
                    va, vb = int(comp1[i1]), int(comp1[i2])
                elif q_idx < Q - 1 and len(comp0) >= 2:
                    # Anchored equivalence query inside comp0
                    i1, i2 = rng.choice(min(len(comp0), max(2, M // 2 + 1)), size=2, replace=False)
                    va, vb = int(comp0[i1]), int(comp0[i2])
                else:
                    # Cross-component query (independent -> P*(A == B) = 0.5)
                    va, vb = int(comp0[0]), int(comp1[0])
                query_pairs[b, q_idx] = (va, vb, int(QueryType.EQUAL))
                query_targets[b, q_idx] = float(pair_eq_b[va, vb])

        header = self.build_batch_header(B, device=dev)
        return EnvBatch(
            header=header,
            obs_tokens=torch.as_tensor(obs_tokens, dtype=torch.long, device=dev),
            obs_mask=torch.as_tensor(obs_mask, dtype=torch.bool, device=dev),
            obs_features=torch.as_tensor(obs_features, dtype=torch.float32, device=dev),
            memory_tokens=torch.zeros((B, 1, 8), dtype=torch.float32, device=dev),
            memory_mask=torch.zeros((B, 1), dtype=torch.bool, device=dev),
            query_pairs=torch.as_tensor(query_pairs, dtype=torch.long, device=dev),
            query_targets=torch.as_tensor(query_targets, dtype=torch.float32, device=dev),
            target_tokens=torch.as_tensor(target_tokens, dtype=torch.long, device=dev),
            exact_marginals=torch.as_tensor(exact_marginals, dtype=torch.float32, device=dev),
            solvable=torch.as_tensor(solvable, dtype=torch.bool, device=dev),
            meta={"clause_edges": torch.as_tensor(clause_edges, dtype=torch.long, device=dev)},
        )

    @staticmethod
    def _solve_exact_gf2(
        num_vars: int,
        edges: np.ndarray,
        obs_mask: np.ndarray,
        obs_tokens: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Exact linear-time BFS/DFS solver over binary XOR clauses `x_u ^ x_v = rhs`.

        Returns:
            marg: `[N, 2]` exact marginal `P*(X_i = bit | clauses, obs)`
            pair_eq: `[N, N]` exact pairwise probability `P*(X_i == X_j | clauses, obs)`
            consistent: `bool` whether all cycles and observations are mutually consistent.
        """
        adj = [[] for _ in range(num_vars)]
        for u, v, rhs in edges:
            u, v, rhs = int(u), int(v), int(rhs)
            adj[u].append((v, rhs))
            adj[v].append((u, rhs))

        comp_id = np.full(num_vars, -1, dtype=np.int64)
        rel_parity = np.zeros(num_vars, dtype=np.int64)  # parity relative to component root
        marg = np.full((num_vars, 2), 0.5, dtype=np.float32)
        pair_eq = np.full((num_vars, num_vars), 0.5, dtype=np.float32)
        np.fill_diagonal(pair_eq, 1.0)
        consistent = True

        cid = 0
        for root in range(num_vars):
            if comp_id[root] != -1:
                continue
            comp_id[root] = cid
            rel_parity[root] = 0
            queue = [root]
            nodes = [root]
            fixed_root_val: Optional[int] = None
            if obs_mask[root]:
                fixed_root_val = int(obs_tokens[root] - 1)

            while queue:
                curr = queue.pop(0)
                for nxt, rhs in adj[curr]:
                    expected_rel = int(rel_parity[curr] ^ rhs)
                    if comp_id[nxt] == -1:
                        comp_id[nxt] = cid
                        rel_parity[nxt] = expected_rel
                        nodes.append(nxt)
                        queue.append(nxt)
                        if obs_mask[nxt]:
                            implied_root = int((obs_tokens[nxt] - 1) ^ expected_rel)
                            if fixed_root_val is None:
                                fixed_root_val = implied_root
                            elif fixed_root_val != implied_root:
                                consistent = False
                    else:
                        if rel_parity[nxt] != expected_rel:
                            consistent = False

            if not consistent:
                return np.zeros((num_vars, 2), dtype=np.float32), pair_eq, False

            if fixed_root_val is not None:
                for u in nodes:
                    bit_u = int(fixed_root_val ^ rel_parity[u])
                    marg[u, 0] = 1.0 if bit_u == 0 else 0.0
                    marg[u, 1] = 1.0 if bit_u == 1 else 0.0

            for u in nodes:
                for v in nodes:
                    pair_eq[u, v] = 1.0 if (rel_parity[u] == rel_parity[v]) else 0.0
            cid += 1

        # Also fill pair_eq between two different anchored components where both have fixed values
        for u in range(num_vars):
            for v in range(num_vars):
                if comp_id[u] != comp_id[v]:
                    # Independent components: P(X_u == X_v) = p_u(0)p_v(0) + p_u(1)p_v(1)
                    pair_eq[u, v] = float(marg[u, 0] * marg[v, 0] + marg[u, 1] * marg[v, 1])

        return marg, pair_eq, True

    def step_unmask(
        self,
        batch: EnvBatch,
        var_idx: torch.Tensor,
        val_idx: torch.Tensor,
    ) -> EnvBatch:
        """Clamp `var_idx` to `val_idx` (`1` or `2`) and update observation mask & feature column."""
        nxt = super().step_unmask(batch, var_idx, val_idx)
        nxt.obs_features[:, :, -1] = nxt.obs_mask.float()
        return nxt

    def exact_posterior_for_batch(self, batch: EnvBatch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recompute exact marginals `[B, N, 2]` and query probabilities `[B, Q]` for any clamped state `batch`.

        Useful for oracle unit tests and verifying that clamping one end of a floating component
        immediately collapses the conditional marginal on the other end (`P(X_B | X_A = z)`).
        """
        B = batch.batch_size
        N = batch.num_vars
        Q = batch.query_pairs.shape[1]
        edges_np = batch.meta["clause_edges"].detach().cpu().numpy()
        mask_np = batch.obs_mask.detach().cpu().numpy()
        tok_np = batch.obs_tokens.detach().cpu().numpy()
        qpairs_np = batch.query_pairs.detach().cpu().numpy()

        margs = np.zeros((B, N, 2), dtype=np.float32)
        qprobs = np.zeros((B, Q), dtype=np.float32)
        solv = np.zeros(B, dtype=bool)
        for b in range(B):
            m_b, peq_b, ok_b = self._solve_exact_gf2(N, edges_np[b], mask_np[b], tok_np[b])
            margs[b] = m_b
            solv[b] = ok_b
            for q in range(Q):
                va, vb, qt = int(qpairs_np[b, q, 0]), int(qpairs_np[b, q, 1]), int(qpairs_np[b, q, 2])
                p_eq = peq_b[va, vb]
                qprobs[b, q] = (1.0 - p_eq) if qt == int(QueryType.NOT_EQUAL) else p_eq
        dev = batch.device
        return (
            torch.as_tensor(margs, dtype=torch.float32, device=dev),
            torch.as_tensor(qprobs, dtype=torch.float32, device=dev),
            torch.as_tensor(solv, dtype=torch.bool, device=dev),
        )

    def check_witness(
        self,
        batch: EnvBatch,
        candidate_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized check of whether `candidate_tokens` (`[B, N]` in `{1, 2}`) satisfies all clauses & observations."""
        matches_obs = (~batch.obs_mask) | (candidate_tokens == batch.obs_tokens)
        in_range = (candidate_tokens == 1) | (candidate_tokens == 2)
        bits = (candidate_tokens - 1).clamp(0, 1)  # [B, N]

        edges = batch.meta["clause_edges"]  # [B, M, 3]
        u = edges[:, :, 0]
        v = edges[:, :, 1]
        rhs = edges[:, :, 2]
        bu = bits.gather(1, u)
        bv = bits.gather(1, v)
        clauses_ok = ((bu ^ bv) == rhs).all(dim=-1)
        return matches_obs.all(dim=-1) & in_range.all(dim=-1) & clauses_ok
