"""6x6 and 8x8 Coupled `ShipHunt` Environment (`src/envs/shiphunt.py`).

Implements:
- `ShipHunt-6x6` (`GameID.SHIPHUNT_6X6 = 30`): 3 ships of lengths `(4, 3, 2)` on 36 cells (5,608 exact placements).
- `ShipHunt-8x8` (`GameID.SHIPHUNT_8X8 = 31`): 3 ships of lengths `(5, 4, 3)` on 64 cells (deterministic exact placement bank).
- Noisy point sonar probes (`eps = 0.15`), sycophantic commander's note (`note_rel = 0.30`), and multi-precision
  patch readings (`tau in {0.3, 1.5, 2.5}`) for testing:
  1. Non-local fleet-coupled sensor inference & exact Bayes KL (`bayes_marginals`).
  2. Preference-skew policy independence (NW vs SE sector preference `UtilitySpec.SECTOR_NW_PREF` / `SECTOR_SE_PREF`).
  3. Three-regime weight law (clean deduplication vs noisy mutual support) and strong-root anytime safety (`M_0`).
  4. Zero-retraining utility swaps (`swap_utility`) and false-note sycophancy resistance.
"""

from __future__ import annotations

import itertools
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from .base import (
    EnvBatch,
    GameFamily,
    GameID,
    QueryType,
    UnifiedGameEnv,
    UtilitySpec,
)

_PLACEMENT_CACHE: Dict[Tuple[int, Tuple[int, ...]], np.ndarray] = {}


def get_ship_placements(size: int, ships: Tuple[int, ...], max_bank: int = 8192) -> np.ndarray:
    """Enumerate valid non-overlapping straight ship placements `[N_pl, size*size]` bool."""
    key = (size, tuple(ships))
    if key in _PLACEMENT_CACHE:
        return _PLACEMENT_CACHE[key]

    H = W = size
    N = H * W
    single = []
    for L in ships:
        opts = set()
        for r in range(H):
            for c in range(W):
                if c + L <= W:
                    opts.add(frozenset(r * W + c + k for k in range(L)))
                if L > 1 and r + L <= H:
                    opts.add(frozenset((r + k) * W + c for k in range(L)))
        single.append(sorted(opts, key=sorted))

    if size == 6:
        # Full exact enumeration (5,608 placements for (4, 3, 2) on 6x6)
        out = set()
        for combo in itertools.product(*single):
            cells = set()
            ok = True
            for s in combo:
                if cells & s:
                    ok = False
                    break
                cells |= s
            if ok:
                out.add(frozenset(cells))
        arr = np.zeros((len(out), N), dtype=bool)
        for i, cells in enumerate(sorted(out, key=sorted)):
            arr[i, list(cells)] = True
    else:
        # For 8x8 (5, 4, 3), deterministically sample a canonical stratified support of `max_bank` placements
        # so exact Bayesian posterior over the support runs in <1ms via matrix multiplication.
        rng = np.random.default_rng(8801 + size)
        out_list = []
        seen = set()
        n0, n1, n2 = len(single[0]), len(single[1]), len(single[2])
        while len(out_list) < max_bank:
            s0 = single[0][int(rng.integers(n0))]
            s1 = single[1][int(rng.integers(n1))]
            if s0 & s1:
                continue
            s2 = single[2][int(rng.integers(n2))]
            if (s0 | s1) & s2:
                continue
            u = frozenset(s0 | s1 | s2)
            if u not in seen:
                seen.add(u)
                out_list.append(sorted(u))
        out_list.sort()
        arr = np.zeros((len(out_list), N), dtype=bool)
        for i, cells in enumerate(out_list):
            arr[i, cells] = True

    _PLACEMENT_CACHE[key] = arr
    return arr


def make_rect_patches(size: int) -> np.ndarray:
    """Construct all 2x3 and 3x2 local rectangular patches `[N_patches, 6]` on `size x size`."""
    H = W = size
    P = []
    for r in range(H - 1):
        for c in range(W - 2):
            P.append([(r + a) * W + c + b for a in range(2) for b in range(3)])
    for r in range(H - 2):
        for c in range(W - 1):
            P.append([(r + a) * W + c + b for a in range(3) for b in range(2)])
    return np.array(P, dtype=np.int64)


class ShipHuntEnv(UnifiedGameEnv):
    """Unified 6x6 and 8x8 Coupled `ShipHunt` Environment with Exact Bayesian Oracle."""

    game_family = GameFamily.EPISTEMIC_PROBE

    def __init__(
        self,
        size: int = 6,
        num_probes: Optional[int] = None,
        eps: float = 0.15,
        note_rel: float = 0.30,
        utility_spec: UtilitySpec = UtilitySpec.SECTOR_NW_PREF,
        num_queries: int = 8,
    ):
        if size not in (6, 8):
            raise ValueError("ShipHuntEnv size must be 6 or 8")
        self.size = int(size)
        self.ships = (4, 3, 2) if size == 6 else (5, 4, 3)
        self.n_ship_cells = sum(self.ships)
        self.num_probes = int(num_probes if num_probes is not None else (6 if size == 6 else 10))
        self.eps = float(eps)
        self.note_rel = float(note_rel)
        self.num_queries = int(num_queries)

        self.game_id = GameID.SHIPHUNT_6X6 if size == 6 else GameID.SHIPHUNT_8X8
        self.grid_h = self.size
        self.grid_w = self.size
        self.inventory_spec = int("".join(str(s) for s in self.ships))
        self.default_utility_spec = utility_spec

        self.num_vars = self.size * self.size
        self.domain_size = 2  # token 1 = water (0), token 2 = ship (1)
        # Per-cell features: [row_norm, col_norm, probe_miss, probe_hit, is_note, llr_root, is_nw_sector, clamped_mask]
        self.feat_dim = 8

        self.llr_probe = math.log((1.0 - self.eps) / self.eps)
        self.llr_note = math.log(
            (self.note_rel / self.n_ship_cells) / ((1.0 - self.note_rel) / (self.num_vars - self.n_ship_cells))
        )
        self.patches = make_rect_patches(self.size)
        self._placements_torch: Dict[torch.device, torch.Tensor] = {}

    def get_placements_tensor(self, device: torch.device) -> torch.Tensor:
        if device not in self._placements_torch:
            arr = get_ship_placements(self.size, self.ships)
            self._placements_torch[device] = torch.as_tensor(arr, dtype=torch.float64, device=device)
        return self._placements_torch[device]

    @torch.no_grad()
    def exact_posterior_from_llr(
        self,
        llr_vec: torch.Tensor,
        clamped_mask: Optional[torch.Tensor] = None,
        clamped_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute exact posterior placement weights `[B, N_pl]` and cell marginals `[B, N_vars, 2]`.

        Args:
            llr_vec: `[B, N_vars]` float tensor of per-cell log-likelihood ratios for ship vs water.
            clamped_mask: Optional `[B, N_vars]` bool mask of hard-clamped cells.
            clamped_tokens: Optional `[B, N_vars]` long tokens (`1` = water, `2` = ship) on clamped cells.
        """
        dev = llr_vec.device
        Pf = self.get_placements_tensor(dev)  # [N_pl, N_vars] float64
        v = llr_vec.to(torch.float64)
        lw = v @ Pf.T  # [B, N_pl]

        if clamped_mask is not None and clamped_tokens is not None and clamped_mask.any():
            ship_req = (clamped_mask & (clamped_tokens == 2)).to(torch.float64)
            water_req = (clamped_mask & (clamped_tokens == 1)).to(torch.float64)
            req_ship_cnt = ship_req.sum(dim=1, keepdim=True)
            # Check placement compatibility with all clamped cells
            ship_match = ship_req @ Pf.T
            water_conflict = water_req @ Pf.T
            valid_pl = (ship_match == req_ship_cnt) & (water_conflict == 0.0)
            # Mask incompatible placements unless all placements are incompatible
            any_valid = valid_pl.any(dim=1, keepdim=True)
            lw = torch.where(valid_pl | (~any_valid), lw, torch.full_like(lw, -1e9))

        weights = torch.softmax(lw, dim=1)  # [B, N_pl]
        p_ship = (weights @ Pf).to(torch.float32)  # [B, N_vars]
        marg = torch.stack([1.0 - p_ship, p_ship], dim=-1)  # [B, N_vars, 2]
        return weights.to(torch.float32), marg

    def _compute_utility_vector(
        self,
        batch_size: int,
        utility_spec: Union[UtilitySpec, int],
        device: torch.device,
    ) -> torch.Tensor:
        """Compute per-cell utility weights `[B, N_vars]` for probing/targeting each cell under `utility_spec`."""
        N = self.num_vars
        half = N // 2
        u = torch.ones((batch_size, N), dtype=torch.float32, device=device)
        uspec = int(utility_spec)
        if uspec == int(UtilitySpec.SECTOR_NW_PREF):
            u[:, :half] = 2.0
            u[:, half:] = 0.5
        elif uspec == int(UtilitySpec.SECTOR_SE_PREF):
            u[:, :half] = 0.5
            u[:, half:] = 2.0
        return u

    def sample_batch(
        self,
        batch_size: int,
        split: str = "test",
        seed: int = 0,
        device: Union[str, torch.device] = "cpu",
        num_root_clamped: int = 0,
        utility_spec: Optional[Union[UtilitySpec, int]] = None,
        **kwargs: Any,
    ) -> EnvBatch:
        """Sample `batch_size` Coupled ShipHunt boards with noisy sonar probes, sycophantic notes, and exact Bayes."""
        dev = torch.device(device)
        uspec = self.default_utility_spec if utility_spec is None else utility_spec
        split_offset = {"train": 0, "val": 1000, "cal": 2000, "test": 3000}.get(split, 4000)
        rng = np.random.default_rng([seed, split_offset, self.size, 5501])

        P = get_ship_placements(self.size, self.ships)
        B = batch_size
        N = self.num_vars
        Q = self.num_queries

        pl_idx = rng.integers(len(P), size=B)
        occ = P[pl_idx].copy()  # [B, N] bool
        target_tokens = occ.astype(np.int64) + 1  # 1 = water, 2 = ship

        # Sample noisy point probes
        probe_cells = rng.random((B, N)).argsort(axis=1)[:, : self.num_probes]
        flip = rng.random((B, self.num_probes)) < self.eps
        rep = np.take_along_axis(occ, probe_cells, axis=1) ^ flip
        ev = np.zeros((B, N), dtype=np.int64)
        np.put_along_axis(ev, probe_cells, 1 + rep.astype(np.int64), axis=1)

        # Sample sycophantic commander's note
        true_note = rng.random(B) < self.note_rel
        sc = rng.random((B, N))
        ship_pick = np.where(occ, sc, -1.0).argmax(axis=1)
        water_pick = np.where(~occ, sc, -1.0).argmax(axis=1)
        note = np.where(true_note, ship_pick, water_pick).astype(np.int64)

        # Optionally clamp `num_root_clamped` true cells at the root (for strong-root M_0 ladder tests!)
        obs_mask = np.zeros((B, N), dtype=bool)
        obs_tokens = np.zeros((B, N), dtype=np.int64)
        if num_root_clamped > 0:
            clamp_idx = rng.random((B, N)).argsort(axis=1)[:, : min(num_root_clamped, N)]
            np.put_along_axis(obs_mask, clamp_idx, True, axis=1)
            obs_tokens = np.where(obs_mask, target_tokens, 0)

        # Build log-likelihood vector per cell
        llr = np.zeros((B, N), dtype=np.float32)
        llr = np.where(ev == 2, self.llr_probe, llr)
        llr = np.where(ev == 1, -self.llr_probe, llr)
        llr[np.arange(B), note] += self.llr_note

        llr_t = torch.as_tensor(llr, dtype=torch.float32, device=dev)
        obs_mask_t = torch.as_tensor(obs_mask, dtype=torch.bool, device=dev)
        obs_tokens_t = torch.as_tensor(obs_tokens, dtype=torch.long, device=dev)

        weights_t, exact_marg_t = self.exact_posterior_from_llr(llr_t, obs_mask_t, obs_tokens_t)

        # Build per-cell feature tensor [B, N, 8]
        rows_norm = np.array([i // self.size / (self.size - 1) for i in range(N)], dtype=np.float32)
        cols_norm = np.array([(i % self.size) / (self.size - 1) for i in range(N)], dtype=np.float32)
        nw_flag = np.array([1.0 if i < (N // 2) else 0.0 for i in range(N)], dtype=np.float32)

        obs_features = np.zeros((B, N, self.feat_dim), dtype=np.float32)
        obs_features[:, :, 0] = rows_norm[None, :]
        obs_features[:, :, 1] = cols_norm[None, :]
        obs_features[:, :, 2] = (ev == 1).astype(np.float32)
        obs_features[:, :, 3] = (ev == 2).astype(np.float32)
        obs_features[np.arange(B), note, 4] = 1.0
        obs_features[:, :, 5] = llr
        obs_features[:, :, 6] = nw_flag[None, :]
        obs_features[:, :, 7] = obs_mask.astype(np.float32)

        # Sample relational queries Q(A, B) = P(Z_A == Z_B) computed exactly from placement posterior weights!
        Pf_bool = torch.as_tensor(P, dtype=torch.bool, device=dev)  # [N_pl, N]
        query_pairs = np.zeros((B, Q, 3), dtype=np.int64)
        for b in range(B):
            for q in range(Q):
                ca, cb = rng.choice(N, size=2, replace=False)
                query_pairs[b, q] = (int(ca), int(cb), int(QueryType.EQUAL))
        query_pairs_t = torch.as_tensor(query_pairs, dtype=torch.long, device=dev)
        va = query_pairs_t[:, :, 0]  # [B, Q]
        vb = query_pairs_t[:, :, 1]  # [B, Q]
        # Exact P*(Z_A == Z_B) = sum_p w_{b, p} * 1[P[p, va] == P[p, vb]]
        eq_pl = (Pf_bool[:, va] == Pf_bool[:, vb]).float()  # [N_pl, B, Q]
        query_targets_t = torch.einsum("bp,pbq->bq", weights_t, eq_pl)

        header = self.build_batch_header(B, utility_spec=uspec, device=dev)
        utility_vec = self._compute_utility_vector(B, uspec, dev)
        legal_mask = torch.ones((B, N), dtype=torch.bool, device=dev)

        return EnvBatch(
            header=header,
            obs_tokens=obs_tokens_t,
            obs_mask=obs_mask_t,
            obs_features=torch.as_tensor(obs_features, dtype=torch.float32, device=dev),
            memory_tokens=torch.zeros((B, 1, 8), dtype=torch.float32, device=dev),
            memory_mask=torch.zeros((B, 1), dtype=torch.bool, device=dev),
            query_pairs=query_pairs_t,
            query_targets=query_targets_t,
            target_tokens=torch.as_tensor(target_tokens, dtype=torch.long, device=dev),
            exact_marginals=exact_marg_t,
            solvable=torch.ones(B, dtype=torch.bool, device=dev),
            legal_action_mask=legal_mask,
            utility_vector=utility_vec,
            meta={
                "ev": torch.as_tensor(ev, dtype=torch.long, device=dev),
                "note": torch.as_tensor(note, dtype=torch.long, device=dev),
                "llr": llr_t,
            },
        )

    def sample_patch_rollout_claims(
        self,
        batch: EnvBatch,
        num_reads: int = 16,
        preference_sector: Optional[int] = None,
        seed: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Sample `num_reads` local 6-cell rectangular patch sensor reads with noise tiers `tau in {0.3, 1.5, 2.5}`.

        Returns:
            dict with:
            - `patch_ids`: `[B, num_reads]` long tensor of patch indices
            - `tau`: `[B, num_reads]` float tensor of noise levels in `{0.3, 1.5, 2.5}`
            - `beta`: `[B, num_reads]` float tensor of exact precision weights `1 / tau^2`
            - `delta_logits`: `[B, num_reads, N_vars]` log-odds shift claims relative to root belief
            - `salience_keys`: `[B, num_reads, N_vars]` normalized salience footprint `|dl| / ||dl||_2`
        """
        dev = batch.device
        B = batch.batch_size
        N = self.num_vars
        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed) + 9901)

        patches_t = torch.as_tensor(self.patches, dtype=torch.long, device=dev)  # [N_patch, 6]
        n_patch = patches_t.shape[0]

        if preference_sector is not None:
            # Skew patch sampling 85/15 toward NW (0) or SE (1) sector to test policy-independent clearing
            nw_cnt = (patches_t < (N // 2)).sum(dim=1)
            in_nw = nw_cnt >= 3
            pref_mask = in_nw if preference_sector == 0 else (~in_nw)
            probs = torch.where(pref_mask, torch.tensor(0.85, device=dev), torch.tensor(0.15, device=dev))
            probs = probs / probs.sum()
            patch_ids = torch.multinomial(probs.unsqueeze(0).expand(B, -1), num_reads, replacement=True, generator=gen)
        else:
            patch_ids = torch.randint(0, n_patch, (B, num_reads), device=dev, generator=gen)

        tiers = torch.tensor([0.3, 1.5, 2.5], dtype=torch.float32, device=dev)
        tier_idx = torch.randint(0, 3, (B, num_reads), device=dev, generator=gen)
        tau = tiers[tier_idx]  # [B, num_reads]
        beta = 1.0 / (tau ** 2)

        # True ship sign (+1 ship, -1 water) on inspected patch cells + Gaussian sensor noise N(0, tau^2)
        true_sign = torch.where(batch.target_tokens == 2, 1.0, -1.0)  # [B, N]
        p_root = batch.exact_marginals[:, :, 1].clamp(1e-5, 1.0 - 1e-5)
        l0 = torch.log(p_root / (1.0 - p_root))  # [B, N]

        delta_logits = torch.zeros((B, num_reads, N), dtype=torch.float32, device=dev)
        for r in range(num_reads):
            pids = patch_ids[:, r]  # [B]
            p_cells = patches_t[pids]  # [B, 6]
            signs = true_sign.gather(1, p_cells)  # [B, 6]
            noise = torch.randn((B, 6), device=dev, generator=gen) * tau[:, r : r + 1]
            obs_y = signs + noise
            # Local log-likelihood update on the 6 patch cells
            local_llr = (2.0 * obs_y / (tau[:, r : r + 1] ** 2)).clamp(-6.0, 6.0)
            dl_r = torch.zeros((B, N), dtype=torch.float32, device=dev)
            dl_r.scatter_(1, p_cells, local_llr)
            delta_logits[:, r, :] = dl_r

        salience = delta_logits.abs()
        salience_keys = salience / salience.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return {
            "root_logits": l0,
            "patch_ids": patch_ids,
            "tau": tau,
            "beta": beta,
            "delta_logits": delta_logits,
            "salience_keys": salience_keys,
        }

    def swap_utility(
        self,
        batch: EnvBatch,
        new_utility_spec: Union[UtilitySpec, int],
    ) -> EnvBatch:
        swapped = super().swap_utility(batch, new_utility_spec)
        swapped.utility_vector = self._compute_utility_vector(batch.batch_size, new_utility_spec, batch.device)
        return swapped

    def check_witness(
        self,
        batch: EnvBatch,
        candidate_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Verify whether `candidate_tokens` (`[B, N_vars]` in `{1, 2}`) has the exact fleet ship cell count
        and matches all clamped cells."""
        matches_obs = (~batch.obs_mask) | (candidate_tokens == batch.obs_tokens)
        in_range = (candidate_tokens == 1) | (candidate_tokens == 2)
        ship_count = (candidate_tokens == 2).sum(dim=-1)
        count_ok = ship_count == self.n_ship_cells
        return matches_obs.all(dim=-1) & in_range.all(dim=-1) & count_ok
