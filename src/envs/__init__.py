"""Unified Multi-Game Environment Suite (`src/envs`).

Exports all 8 benchmark environment families under a single hierarchical token interface:
`[GAME_FAMILY, GAME_ID, GRID_H, GRID_W, INVENTORY_SPEC, UTILITY_SPEC]`.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .base import (
    HEADER_LEN,
    PUBLIC_META_KEYS,
    SPLIT_OFFSETS,
    EnvBatch,
    GameFamily,
    GameID,
    OracleLeakError,
    PoisonTensor,
    QueryType,
    UnifiedGameEnv,
    UtilitySpec,
    bernoulli_kl,
    build_header,
    categorical_kl,
)
from .equiv_sat import EquivSatEnv
from .galanis_market import GalanisMarketEnv
from .hex_c4 import Connect4Env, HexEnv, TacticalGameEnv
from .micro_stratego import MicroStrategoEnv
from .minesweeper import MinesweeperEnv
from .proposal_poker import ProposalPokerEnv
from .shiphunt import ShipHuntEnv
from .sudoku import SudokuEnv


ENV_REGISTRY = {
    "equiv_sat": lambda **kw: EquivSatEnv(**kw),
    "sudoku_6x6": lambda **kw: SudokuEnv(size=6, **kw),
    "sudoku_9x9_extreme": lambda **kw: SudokuEnv(size=9, extreme=True, **kw),
    "shiphunt_6x6": lambda **kw: ShipHuntEnv(size=6, **kw),
    "shiphunt_8x8": lambda **kw: ShipHuntEnv(size=8, **kw),
    "minesweeper_6x6": lambda **kw: MinesweeperEnv(size=6, **kw),
    "hex_5x5": lambda **kw: HexEnv(**kw),
    "connect4_6x7": lambda **kw: Connect4Env(**kw),
    "micro_stratego_4x4": lambda **kw: MicroStrategoEnv(size=4, **kw),
    "micro_stratego_6x6": lambda **kw: MicroStrategoEnv(size=6, **kw),
    "galanis_t3s111y2": lambda **kw: GalanisMarketEnv(structure_name="t3s111y2", **kw),
    "galanis_t3s110": lambda **kw: GalanisMarketEnv(structure_name="t3s110", **kw),
    "galanis_t3s111": lambda **kw: GalanisMarketEnv(structure_name="t3s111", **kw),
    "galanis_t3s111o2ye2": lambda **kw: GalanisMarketEnv(structure_name="t3s111o2ye2", **kw),
    "proposal_poker": lambda **kw: ProposalPokerEnv(**kw),
}


def make_env(name: str, **kwargs: Any) -> UnifiedGameEnv:
    """Instantiate any registered game environment by name."""
    key = name.lower().strip()
    if key.startswith("galanis_market_"):
        key = key.replace("galanis_market_", "galanis_", 1)
    if key not in ENV_REGISTRY:
        raise ValueError(f"Unknown environment {name!r}; available: {sorted(ENV_REGISTRY)}")
    return ENV_REGISTRY[key](**kwargs)


def list_registered_envs() -> List[str]:
    return list(ENV_REGISTRY.keys())


__all__ = [
    "HEADER_LEN",
    "EnvBatch",
    "GameFamily",
    "GameID",
    "QueryType",
    "UnifiedGameEnv",
    "UtilitySpec",
    "bernoulli_kl",
    "build_header",
    "categorical_kl",
    "EquivSatEnv",
    "SudokuEnv",
    "ShipHuntEnv",
    "MinesweeperEnv",
    "TacticalGameEnv",
    "HexEnv",
    "Connect4Env",
    "MicroStrategoEnv",
    "GalanisMarketEnv",
    "ProposalPokerEnv",
    "ENV_REGISTRY",
    "make_env",
    "list_registered_envs",
]
