"""Unified Multi-Game Environment & Token Interface (`GLC-AOM`).

Provides:
1. Unambiguous 6-token hierarchical header:
   `[GAME_FAMILY, GAME_ID, GRID_H, GRID_W, INVENTORY_SPEC, UTILITY_SPEC]`
2. Standardized `EnvBatch` tensor container supporting:
   - Discrete state tokens `obs_tokens` `[B, N_vars]` & observation mask `obs_mask` `[B, N_vars]`
   - Continuous per-variable features `obs_features` `[B, N_vars, D_feat]`
   - Present-time Episodic Memory tokens `memory_tokens` (`M_t`) `[B, N_mem, D_mem]`
   - Direct Combinatorial / Relational Query tokens `query_pairs` (`Q(A, B)`) `[B, N_queries, 3]`
   - Modular Utility specification `utility_vector` (`U`) with zero-retraining `swap_utility`
   - Exact Bayesian posterior marginals `exact_marginals` (`P^*`) for KL / Brier evaluation
3. Base `UnifiedGameEnv` interface for any-order unmasking (`STEP(v, z)`), witness checking,
   belief/action evaluation, and multi-agent Elo match play.
"""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

if "/tmp/r4e3a/pylib" not in sys.path and os.path.isdir("/tmp/r4e3a/pylib"):
    sys.path.insert(0, "/tmp/r4e3a/pylib")

import numpy as np
import torch
import torch.nn.functional as F

HEADER_LEN = 6

SPLIT_OFFSETS: Dict[str, int] = {
    "train": 0,
    "cal": 100_000,
    "cal_fit": 100_000,
    "val": 200_000,
    "cal_val": 200_000,
    "test": 300_000,
}

PUBLIC_META_KEYS = frozenset(
    {
        "clause_edges",
        "clue_counts",
        "to_move",
        "utility_bias_y",
        "yes_count",
        "no_count",
        "lobby_note",
        "ev",
        "note",
        "p_from",
        "round_idx",
    }
)


class OracleLeakError(RuntimeError):
    """Raised when any code attempts to read or compute on a stripped oracle tensor (`R11-D0` R1)."""


class PoisonTensor:
    """Zero-oracle-leakage sentinel replacing `exact_marginals`, `target_tokens`, and `query_targets` (`R11-D0` R1).

    Stores zero underlying data (only `.shape`, `.dtype`, `.device`) and raises `OracleLeakError`
    on any `__torch_function__`, indexing, iteration, arithmetic, or host conversion (`.item()`,
    `.cpu()`, `.numpy()`, `.tolist()`, `.detach()`).
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device: Union[str, torch.device],
        field_name: str = "oracle",
    ) -> None:
        self._shape = tuple(int(s) for s in shape)
        self._dtype = dtype
        self._device = torch.device(device)
        self._field_name = field_name

    @classmethod
    def from_tensor(cls, t: Union[torch.Tensor, "PoisonTensor"], field_name: str = "oracle") -> "PoisonTensor":
        return cls(tuple(int(s) for s in t.shape), t.dtype, t.device, field_name=field_name)

    @property
    def shape(self) -> torch.Size:
        return torch.Size(self._shape)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def ndim(self) -> int:
        return len(self._shape)

    def size(self, dim: Optional[int] = None) -> Union[torch.Size, int]:
        if dim is None:
            return torch.Size(self._shape)
        return self._shape[dim]

    def clone(self) -> "PoisonTensor":
        return PoisonTensor(self._shape, self._dtype, self._device, self._field_name)

    def to(self, *args: Any, **kwargs: Any) -> "PoisonTensor":
        dev = self._device
        for a in args:
            if isinstance(a, (str, torch.device)):
                dev = torch.device(a)
        if "device" in kwargs and kwargs["device"] is not None:
            dev = torch.device(kwargs["device"])
        return PoisonTensor(self._shape, self._dtype, dev, self._field_name)

    def _raise_leak(self, op_name: str) -> Any:
        raise OracleLeakError(
            f"R11-D0 OracleLeakError: forbidden access ({op_name}) on poisoned oracle field '{self._field_name}'"
        )

    @classmethod
    def __torch_function__(cls, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        raise OracleLeakError(
            f"R11-D0 OracleLeakError: forbidden torch function {getattr(func, '__name__', str(func))} on PoisonTensor"
        )

    def __getitem__(self, idx: Any) -> Any:
        return self._raise_leak("__getitem__")

    def __setitem__(self, idx: Any, val: Any) -> None:
        self._raise_leak("__setitem__")

    def __iter__(self) -> Any:
        return self._raise_leak("__iter__")

    def item(self) -> Any:
        return self._raise_leak("item")

    def cpu(self) -> Any:
        return self._raise_leak("cpu")

    def cuda(self, *args: Any, **kwargs: Any) -> Any:
        return self._raise_leak("cuda")

    def numpy(self) -> Any:
        return self._raise_leak("numpy")

    def tolist(self) -> Any:
        return self._raise_leak("tolist")

    def detach(self) -> Any:
        return self._raise_leak("detach")

    def float(self) -> Any:
        return self._raise_leak("float")

    def long(self) -> Any:
        return self._raise_leak("long")

    def bool(self) -> Any:
        return self._raise_leak("bool")

    def __add__(self, other: Any) -> Any:
        return self._raise_leak("__add__")

    def __radd__(self, other: Any) -> Any:
        return self._raise_leak("__radd__")

    def __sub__(self, other: Any) -> Any:
        return self._raise_leak("__sub__")

    def __rsub__(self, other: Any) -> Any:
        return self._raise_leak("__rsub__")

    def __mul__(self, other: Any) -> Any:
        return self._raise_leak("__mul__")

    def __rmul__(self, other: Any) -> Any:
        return self._raise_leak("__rmul__")

    def __truediv__(self, other: Any) -> Any:
        return self._raise_leak("__truediv__")

    def __gt__(self, other: Any) -> Any:
        return self._raise_leak("__gt__")

    def __lt__(self, other: Any) -> Any:
        return self._raise_leak("__lt__")

    def __ge__(self, other: Any) -> Any:
        return self._raise_leak("__ge__")

    def __le__(self, other: Any) -> Any:
        return self._raise_leak("__le__")


class GameFamily(IntEnum):
    """Top-level game family token (Header index 0)."""
    CONSTRAINT_CSP = 1
    EPISTEMIC_PROBE = 2
    TACTICAL_PERFECT_INFO = 3
    TACTICAL_IMPERFECT_INFO = 4
    PREDICTION_MARKET = 5


class GameID(IntEnum):
    """Specific game identifier token (Header index 1)."""
    EQUIV_SAT = 10
    SUDOKU_6X6 = 20
    SUDOKU_9X9_EXTREME = 21
    SHIPHUNT_6X6 = 30
    SHIPHUNT_8X8 = 31
    MINESWEEPER_6X6 = 40
    HEX_5X5 = 50
    CONNECT4_6X7 = 51
    MICRO_STRATEGO_4X4 = 60
    MICRO_STRATEGO_6X6 = 61
    GALANIS_MARKET_T3S111Y2 = 70
    GALANIS_MARKET_T3S110 = 71
    GALANIS_MARKET_T3S111 = 72
    GALANIS_MARKET_T3S111O2YE2 = 73
    PROPOSAL_POKER = 80


class UtilitySpec(IntEnum):
    """Modular utility / goal mandate token (Header index 5)."""
    EPISTEMIC_TRUTH = 0          # Pure belief calibration / KL minimization
    PLAYER_0_WIN = 1             # Zero-sum win value for Player 0 (Red / First)
    PLAYER_1_WIN = 2             # Zero-sum win value for Player 1 (Blue / Second)
    SECTOR_NW_PREF = 3           # ShipHunt: prioritize NW sector ships
    SECTOR_SE_PREF = 4           # ShipHunt: prioritize SE sector ships
    SAFE_SURVIVAL = 5            # Minesweeper: penalize mine detonation
    MARKET_HONEST_PROFIT = 6     # Prediction market / Proposal Poker: unbiased LMSR / staking profit
    BIAS_PRO_APPROVE = 7         # Proposal Poker / Manipulation: positive private bias (+y_i) for approve/up
    BIAS_PRO_REJECT = 8          # Proposal Poker / Manipulation: negative private bias (-y_i) for reject/down


class QueryType(IntEnum):
    """Relational query operator Q(A, B) (3rd column of `query_pairs`)."""
    EQUAL = 0        # P(X_A == X_B)
    NOT_EQUAL = 1    # P(X_A != X_B)
    PARITY_XOR = 2   # P((X_A + X_B) mod 2 == 1)
    GREATER = 3      # P(X_A > X_B)


def build_header(
    batch_size: int,
    game_family: Union[GameFamily, int],
    game_id: Union[GameID, int],
    grid_h: int,
    grid_w: int,
    inventory_spec: int,
    utility_spec: Union[UtilitySpec, int] = UtilitySpec.EPISTEMIC_TRUTH,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Construct the unambiguous 6-token hierarchical header tensor `[B, 6]`."""
    row = torch.tensor(
        [
            int(game_family),
            int(game_id),
            int(grid_h),
            int(grid_w),
            int(inventory_spec),
            int(utility_spec),
        ],
        dtype=torch.long,
        device=device,
    )
    return row.unsqueeze(0).expand(batch_size, HEADER_LEN).clone()


@dataclass
class EnvBatch:
    """Unified vectorized batch across all 8 game families.

    Attributes:
        header: `[B, 6]` long tensor `[GAME_FAMILY, GAME_ID, GRID_H, GRID_W, INVENTORY_SPEC, UTILITY_SPEC]`.
        obs_tokens: `[B, N_vars]` long tensor (`0` = unobserved/masked, `1..K_dom` = observed value).
        obs_mask: `[B, N_vars]` bool tensor (`True` where variable is observed/clamped).
        obs_features: `[B, N_vars, D_feat]` float32 tensor of auxiliary per-variable features.
        memory_tokens: `[B, N_mem, D_mem]` float32 tensor representing present-time episodic memory `M_t`.
        memory_mask: `[B, N_mem]` bool tensor (`True` for active memory slots).
        query_pairs: `[B, N_queries, 3]` long tensor `(var_a, var_b, query_type)` for direct `Q(A, B)` rollouts.
        query_targets: `[B, N_queries]` float32 tensor of ground-truth query probabilities `P*(Q(A, B))`.
        target_tokens: `[B, N_vars]` long tensor of realized ground-truth values `1..K_dom` (training/eval only).
        exact_marginals: `[B, N_vars, K_dom]` float32 tensor of exact Bayesian marginals `P*(X_i = d | x_O)`.
        solvable: `[B]` bool tensor indicating whether the instance has >= 1 valid solution.
        legal_action_mask: `[B, N_actions]` bool tensor of legal actions at the current decision point.
        utility_vector: `[B, N_actions]` float32 tensor of modular action utilities under active `UTILITY_SPEC`.
        meta: dictionary of environment-specific auxiliary tensors/objects.
    """
    header: torch.Tensor
    obs_tokens: torch.Tensor
    obs_mask: torch.Tensor
    obs_features: torch.Tensor
    memory_tokens: torch.Tensor
    memory_mask: torch.Tensor
    query_pairs: torch.Tensor
    query_targets: Union[torch.Tensor, PoisonTensor]
    target_tokens: Union[torch.Tensor, PoisonTensor]
    exact_marginals: Union[torch.Tensor, PoisonTensor]
    solvable: torch.Tensor
    legal_action_mask: Optional[torch.Tensor] = None
    utility_vector: Optional[torch.Tensor] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def batch_size(self) -> int:
        return int(self.header.shape[0])

    @property
    def num_vars(self) -> int:
        return int(self.obs_tokens.shape[1])

    @property
    def domain_size(self) -> int:
        return int(self.exact_marginals.shape[2])

    @property
    def device(self) -> torch.device:
        return self.header.device

    def to(self, device: Union[str, torch.device]) -> "EnvBatch":
        """Move all tensor fields in the batch to `device`."""
        dev = torch.device(device)
        new_meta = {}
        for k, v in self.meta.items():
            if isinstance(v, (torch.Tensor, PoisonTensor)):
                new_meta[k] = v.to(dev)
            else:
                new_meta[k] = v
        return EnvBatch(
            header=self.header.to(dev),
            obs_tokens=self.obs_tokens.to(dev),
            obs_mask=self.obs_mask.to(dev),
            obs_features=self.obs_features.to(dev),
            memory_tokens=self.memory_tokens.to(dev),
            memory_mask=self.memory_mask.to(dev),
            query_pairs=self.query_pairs.to(dev),
            query_targets=self.query_targets.to(dev),
            target_tokens=self.target_tokens.to(dev),
            exact_marginals=self.exact_marginals.to(dev),
            solvable=self.solvable.to(dev),
            legal_action_mask=self.legal_action_mask.to(dev) if self.legal_action_mask is not None else None,
            utility_vector=self.utility_vector.to(dev) if self.utility_vector is not None else None,
            meta=new_meta,
        )

    def clone(self) -> "EnvBatch":
        """Deep-copy tensor fields for tree branching (`SELECT_NODE(j)` + `STEP(v, z)`)."""
        new_meta = {}
        for k, v in self.meta.items():
            if isinstance(v, (torch.Tensor, PoisonTensor)):
                new_meta[k] = v.clone()
            elif isinstance(v, np.ndarray):
                new_meta[k] = v.copy()
            else:
                new_meta[k] = copy.copy(v)
        return EnvBatch(
            header=self.header.clone(),
            obs_tokens=self.obs_tokens.clone(),
            obs_mask=self.obs_mask.clone(),
            obs_features=self.obs_features.clone(),
            memory_tokens=self.memory_tokens.clone(),
            memory_mask=self.memory_mask.clone(),
            query_pairs=self.query_pairs.clone(),
            query_targets=self.query_targets.clone(),
            target_tokens=self.target_tokens.clone(),
            exact_marginals=self.exact_marginals.clone(),
            solvable=self.solvable.clone(),
            legal_action_mask=self.legal_action_mask.clone() if self.legal_action_mask is not None else None,
            utility_vector=self.utility_vector.clone() if self.utility_vector is not None else None,
            meta=new_meta,
        )

    def strip_oracle_labels(self) -> "EnvBatch":
        """Replace oracle fields (`exact_marginals`, `target_tokens`, `query_targets`) with `PoisonTensor`
        and remove all non-public keys from `meta` (`PREREG-R11-GPU-NEURAL.md` §1.2 Rule `R1`).
        """
        public_meta: Dict[str, Any] = {}
        for k, v in self.meta.items():
            if k in PUBLIC_META_KEYS:
                if isinstance(v, (torch.Tensor, PoisonTensor)):
                    public_meta[k] = v.clone()
                elif isinstance(v, np.ndarray):
                    public_meta[k] = v.copy()
                else:
                    public_meta[k] = copy.copy(v)
        return EnvBatch(
            header=self.header.clone(),
            obs_tokens=self.obs_tokens.clone(),
            obs_mask=self.obs_mask.clone(),
            obs_features=self.obs_features.clone(),
            memory_tokens=self.memory_tokens.clone(),
            memory_mask=self.memory_mask.clone(),
            query_pairs=self.query_pairs.clone(),
            query_targets=PoisonTensor.from_tensor(self.query_targets, field_name="query_targets"),
            target_tokens=PoisonTensor.from_tensor(self.target_tokens, field_name="target_tokens"),
            exact_marginals=PoisonTensor.from_tensor(self.exact_marginals, field_name="exact_marginals"),
            solvable=self.solvable.clone(),
            legal_action_mask=self.legal_action_mask.clone() if self.legal_action_mask is not None else None,
            utility_vector=self.utility_vector.clone() if self.utility_vector is not None else None,
            meta=public_meta,
        )

    def split_and_poison_oracle(self) -> Tuple["EnvBatch", "EnvBatch"]:
        """Return `(batch_blind, oracle_labels_batch)` for zero-leakage test-time evaluation (`R11-D0`)."""
        return self.strip_oracle_labels(), self.clone()


def categorical_kl(p_true: torch.Tensor, q_pred: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute elementwise categorical KL(p_true || q_pred) over the last dimension `[..., K]`."""
    p = p_true.clamp(min=0.0)
    p = p / p.sum(dim=-1, keepdim=True).clamp(min=eps)
    q = q_pred.clamp(min=eps)
    q = q / q.sum(dim=-1, keepdim=True).clamp(min=eps)
    kl = (p * (torch.log(p.clamp(min=eps)) - torch.log(q))).sum(dim=-1)
    return kl.clamp(min=0.0)


def bernoulli_kl(p_true: torch.Tensor, q_pred: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Compute elementwise binary KL(p_true || q_pred)."""
    p = p_true.clamp(eps, 1.0 - eps)
    q = q_pred.clamp(eps, 1.0 - eps)
    kl = p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))
    return kl.clamp(min=0.0)


class UnifiedGameEnv:
    """Abstract base class for all 8 unified game environments."""

    game_family: GameFamily
    game_id: GameID
    grid_h: int
    grid_w: int
    inventory_spec: int
    default_utility_spec: UtilitySpec = UtilitySpec.EPISTEMIC_TRUTH
    num_vars: int
    domain_size: int

    def build_batch_header(
        self,
        batch_size: int,
        utility_spec: Optional[Union[UtilitySpec, int]] = None,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        uspec = self.default_utility_spec if utility_spec is None else utility_spec
        return build_header(
            batch_size=batch_size,
            game_family=self.game_family,
            game_id=self.game_id,
            grid_h=self.grid_h,
            grid_w=self.grid_w,
            inventory_spec=self.inventory_spec,
            utility_spec=uspec,
            device=device,
        )

    def sample_batch(
        self,
        batch_size: int,
        split: str = "test",
        seed: int = 0,
        device: Union[str, torch.device] = "cpu",
        **kwargs: Any,
    ) -> EnvBatch:
        """Sample a batch of environment instances on `device`."""
        raise NotImplementedError

    def step_unmask(
        self,
        batch: EnvBatch,
        var_idx: torch.Tensor,
        val_idx: torch.Tensor,
    ) -> EnvBatch:
        """Apply a 1-step variable assignment `(v, z)` (`STEP(v, z)`) to a cloned batch.

        Args:
            batch: Current `EnvBatch` of size `B`.
            var_idx: `[B]` long tensor of variable indices in `0 .. N_vars - 1`.
            val_idx: `[B]` long tensor of assigned domain values in `1 .. K_dom`.
        """
        nxt = batch.clone()
        b_arange = torch.arange(nxt.batch_size, device=nxt.device)
        nxt.obs_tokens[b_arange, var_idx] = val_idx
        nxt.obs_mask[b_arange, var_idx] = True
        return nxt

    def check_witness(
        self,
        batch: EnvBatch,
        candidate_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Verify whether `candidate_tokens` (`[B, N_vars]` in `1..K_dom`) is a valid witness.

        Must only check observable rules/constraints (never peeking at `target_tokens` on hidden variables).
        Returns `[B]` bool tensor.
        """
        # Default fallback checks consistency with clamped observations
        matches_obs = (~batch.obs_mask) | (candidate_tokens == batch.obs_tokens)
        in_range = (candidate_tokens >= 1) & (candidate_tokens <= batch.domain_size)
        return matches_obs.all(dim=-1) & in_range.all(dim=-1)

    def swap_utility(
        self,
        batch: EnvBatch,
        new_utility_spec: Union[UtilitySpec, int],
    ) -> EnvBatch:
        """Return a shallow-cloned batch with updated `UTILITY_SPEC` header and `utility_vector`.

        Used to test zero-retraining utility swaps and belief-action separation (`LIR-Masked`).
        """
        swapped = batch.clone()
        swapped.header[:, 5] = int(new_utility_spec)
        return swapped

    def evaluate_beliefs(
        self,
        batch: EnvBatch,
        pred_probs: torch.Tensor,
        pred_query_probs: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Compute standardized belief metrics over unresolved variables and relational queries.

        Args:
            batch: `EnvBatch` with `exact_marginals` `[B, N_vars, K_dom]`.
            pred_probs: `[B, N_vars, K_dom]` predicted probability distribution per variable.
            pred_query_probs: Optional `[B, N_queries]` predicted probability for each query `Q(A, B)`.
        """
        unresolved = ~batch.obs_mask
        solv_mask = batch.solvable[:, None] & unresolved
        if not solv_mask.any():
            solv_mask = torch.ones_like(unresolved, dtype=torch.bool)

        kl_mat = categorical_kl(batch.exact_marginals, pred_probs)
        brier_mat = ((pred_probs - batch.exact_marginals) ** 2).sum(dim=-1)
        pred_top1 = pred_probs.argmax(dim=-1) + 1
        true_top1 = batch.exact_marginals.argmax(dim=-1) + 1
        acc_mat = (pred_top1 == true_top1).float()

        # Fill observed positions with their clamped values before checking full witness
        full_cand = torch.where(batch.obs_mask, batch.obs_tokens, pred_top1)
        is_witness = self.check_witness(batch, full_cand)
        solv_b = batch.solvable
        witness_rate = float(is_witness[solv_b].float().mean().item()) if solv_b.any() else 0.0

        metrics = {
            "kl": float(kl_mat[solv_mask].mean().item()),
            "brier": float(brier_mat[solv_mask].mean().item()),
            "acc": float(acc_mat[solv_mask].mean().item()),
            "witness_rate": witness_rate,
        }

        if batch.query_pairs.shape[1] > 0:
            if pred_query_probs is None:
                pred_query_probs = self.derive_query_probs_from_marginals(batch, pred_probs)
            q_kl = bernoulli_kl(batch.query_targets, pred_query_probs)
            q_brier = (pred_query_probs - batch.query_targets) ** 2
            q_acc = ((pred_query_probs >= 0.5) == (batch.query_targets >= 0.5)).float()
            metrics["query_kl"] = float(q_kl.mean().item())
            metrics["query_brier"] = float(q_brier.mean().item())
            metrics["query_acc"] = float(q_acc.mean().item())

        return metrics

    @staticmethod
    def derive_query_probs_from_marginals(
        batch: EnvBatch,
        pred_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Independent-marginal fallback for `Q(A, B)` when no direct query rollout is supplied."""
        B, Q, _ = batch.query_pairs.shape
        if Q == 0:
            return torch.zeros((B, 0), dtype=torch.float32, device=batch.device)
        var_a = batch.query_pairs[:, :, 0]
        var_b = batch.query_pairs[:, :, 1]
        q_type = batch.query_pairs[:, :, 2]

        K = pred_probs.shape[-1]
        pa = pred_probs.gather(1, var_a.unsqueeze(-1).expand(B, Q, K))
        pb = pred_probs.gather(1, var_b.unsqueeze(-1).expand(B, Q, K))

        p_eq = (pa * pb).sum(dim=-1)
        out = p_eq.clone()
        neq_mask = q_type == int(QueryType.NOT_EQUAL)
        if neq_mask.any():
            out = torch.where(neq_mask, 1.0 - p_eq, out)
        xor_mask = q_type == int(QueryType.PARITY_XOR)
        if xor_mask.any():
            even_idx = torch.arange(K, device=batch.device) % 2 == 0
            odd_idx = ~even_idx
            pa_odd = pa[:, :, odd_idx].sum(dim=-1)
            pb_odd = pb[:, :, odd_idx].sum(dim=-1)
            p_xor = pa_odd * (1.0 - pb_odd) + (1.0 - pa_odd) * pb_odd
            out = torch.where(xor_mask, p_xor, out)
        gt_mask = q_type == int(QueryType.GREATER)
        if gt_mask.any():
            tril = torch.tril(torch.ones((K, K), device=batch.device), diagonal=-1)
            p_gt = torch.einsum("bqk,kl,bql->bq", pa, tril, pb)
            out = torch.where(gt_mask, p_gt, out)
        return out.clamp(0.0, 1.0)
