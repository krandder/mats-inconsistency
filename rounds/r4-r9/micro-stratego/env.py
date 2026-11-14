#!/usr/bin/env python3
"""
4x4 Diagonal-Touch Micro-Stratego Environment & Exact 720-Setup Bayesian Oracle.

Board geometry (4x4, rows 0..3, cols 0..3):
  Row 3 (Blue back):   o   o   o   o   (Blue setup slots 2, 3, 4, 5)
  Row 2 (Blue front):  .   .   o   o   (Blue setup slots 0, 1 at cols 2, 3)
  Row 1 (Red front):   x   x   .   .   (Red setup slots 0, 1 at cols 0, 1)
  Row 0 (Red back):    x   x   x   x   (Red setup slots 2, 3, 4, 5)

Notice:
- Red's 6 pieces start on [(1,0), (1,1), (0,0), (0,1), (0,2), (0,3)].
- Blue's 6 pieces start on [(2,2), (2,3), (3,0), (3,1), (3,2), (3,3)].
- Armies touch ONLY diagonally at (1,1)-(2,2) at the start!
- Left Flank (cols 0..1) has empty corridor (2,0),(2,1) in front of Red.
- Right Flank (cols 2..3) has empty corridor (1,2),(1,3) in front of Blue.

6 Piece Types per Player (indices 1..6):
  1: Flag (F)    - immobile, rank 0 (capturing wins immediately)
  2: Bomb (B)    - immobile, defeats any attacker except Miner (M); attacker dies, Bomb stays & is revealed
  3: Scout (S)   - moves 1..3 orthogonal empty/enemy squares, rank 2 (moving >1 reveals it is a Scout!)
  4: Miner (M)   - moves 1 orthogonal square, rank 3, defuses/captures Bomb (B)
  5: General (G) - moves 1 orthogonal square, rank 4
  6: Marshal (X) - moves 1 orthogonal square, rank 5 (highest combat rank, loses only to Bomb)
  (Same rank in battle => both pieces are removed.)
"""

from __future__ import annotations
import itertools
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np

# Piece constants
EMPTY = 0
FLAG = 1
BOMB = 2
SCOUT = 3
MINER = 4
GENERAL = 5
MARSHAL = 6

PIECE_NAMES = {
    EMPTY: ".",
    FLAG: "F",
    BOMB: "B",
    SCOUT: "S",
    MINER: "M",
    GENERAL: "G",
    MARSHAL: "X",
}

RED_SLOTS: List[Tuple[int, int]] = [(1, 0), (1, 1), (0, 0), (0, 1), (0, 2), (0, 3)]
BLUE_SLOTS: List[Tuple[int, int]] = [(2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]

# Precompute all 6! = 720 permutations of (FLAG, BOMB, SCOUT, MINER, GENERAL, MARSHAL)
ALL_PERMS: np.ndarray = np.array(
    list(itertools.permutations([FLAG, BOMB, SCOUT, MINER, GENERAL, MARSHAL])),
    dtype=np.int8,
)  # shape (720, 6)


def resolve_battle(attacker: int, defender: int) -> Tuple[int, bool]:
    """
    Returns (outcome, flag_captured) where outcome is:
      +1 if attacker wins and occupies square,
      -1 if defender wins (Bomb stays on square),
       0 if both die (equal rank).
    """
    if defender == FLAG:
        return +1, True
    if defender == BOMB:
        if attacker == MINER:
            return +1, False
        else:
            return -1, False  # Bomb survives and stays revealed
    if attacker > defender:
        return +1, False
    elif attacker < defender:
        return -1, False
    else:
        return 0, False


@dataclass
class PieceTracker:
    """Tracks the 6 pieces of each player by their original setup slot id (0..5)."""
    # piece_type[player, slot] in 1..6
    piece_type: np.ndarray          # (2, 6) int8
    # pos[player, slot] = (r, c) or (-1, -1) if dead
    pos: np.ndarray                 # (2, 6, 2) int8
    # alive[player, slot] bool
    alive: np.ndarray               # (2, 6) bool
    # has_moved[player, slot] bool (publicly observable: if True, cannot be F or B!)
    has_moved: np.ndarray           # (2, 6) bool
    # moved_long[player, slot] bool (publicly observable: if True, MUST be Scout S!)
    moved_long: np.ndarray          # (2, 6) bool
    # revealed[player, slot] bool (publicly observable: exact piece_type is known!)
    revealed: np.ndarray            # (2, 6) bool
    # min_rank_beaten[player, slot] int8 (public constraints from battles)
    turn: int = 0
    winner: Optional[int] = None    # 0, 1, or -1 for draw

    def clone(self) -> "PieceTracker":
        return PieceTracker(
            piece_type=self.piece_type.copy(),
            pos=self.pos.copy(),
            alive=self.alive.copy(),
            has_moved=self.has_moved.copy(),
            moved_long=self.moved_long.copy(),
            revealed=self.revealed.copy(),
            turn=self.turn,
            winner=self.winner,
        )


class MicroStrategoEnv:
    MAX_TURNS = 24

    @staticmethod
    def new_game(red_perm_idx: int, blue_perm_idx: int) -> PieceTracker:
        piece_type = np.zeros((2, 6), dtype=np.int8)
        piece_type[0] = ALL_PERMS[red_perm_idx]
        piece_type[1] = ALL_PERMS[blue_perm_idx]

        pos = np.zeros((2, 6, 2), dtype=np.int8)
        for s, (r, c) in enumerate(RED_SLOTS):
            pos[0, s] = (r, c)
        for s, (r, c) in enumerate(BLUE_SLOTS):
            pos[1, s] = (r, c)

        alive = np.ones((2, 6), dtype=bool)
        has_moved = np.zeros((2, 6), dtype=bool)
        moved_long = np.zeros((2, 6), dtype=bool)
        revealed = np.zeros((2, 6), dtype=bool)
        return PieceTracker(piece_type, pos, alive, has_moved, moved_long, revealed, turn=0, winner=None)

    @staticmethod
    def board_maps(state: PieceTracker) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (owner_grid, slot_grid) each (4, 4) with -1 for empty."""
        owner = np.full((4, 4), -1, dtype=np.int8)
        slot = np.full((4, 4), -1, dtype=np.int8)
        for p in range(2):
            for s in range(6):
                if state.alive[p, s]:
                    r, c = state.pos[p, s]
                    owner[r, c] = p
                    slot[r, c] = s
        return owner, slot

    @staticmethod
    def legal_actions(state: PieceTracker, player: Optional[int] = None) -> List[Tuple[int, int, int]]:
        """
        Returns list of legal moves (slot_id, target_r, target_c) for `player`.
        """
        if state.winner is not None:
            return []
        if player is None:
            player = state.turn % 2
        owner, _ = MicroStrategoEnv.board_maps(state)
        actions: List[Tuple[int, int, int]] = []
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for s in range(6):
            if not state.alive[player, s]:
                continue
            ptype = state.piece_type[player, s]
            if ptype in (FLAG, BOMB):
                continue
            r0, c0 = int(state.pos[player, s, 0]), int(state.pos[player, s, 1])
            max_steps = 3 if ptype == SCOUT else 1
            for dr, dc in dirs:
                for step in range(1, max_steps + 1):
                    nr, nc = r0 + dr * step, c0 + dc * step
                    if not (0 <= nr < 4 and 0 <= nc < 4):
                        break
                    occ = owner[nr, nc]
                    if occ == player:
                        break  # blocked by own piece
                    actions.append((s, nr, nc))
                    if occ == 1 - player:
                        break  # cannot jump over enemy piece
        return actions

    @staticmethod
    def step(state: PieceTracker, action: Tuple[int, int, int]) -> PieceTracker:
        """Applies `action = (slot_id, target_r, target_c)` and returns new state."""
        nxt = state.clone()
        player = nxt.turn % 2
        opp = 1 - player
        s, nr, nc = action
        r0, c0 = int(nxt.pos[player, s, 0]), int(nxt.pos[player, s, 1])
        dist = abs(nr - r0) + abs(nc - c0)

        nxt.has_moved[player, s] = True
        if dist > 1:
            nxt.moved_long[player, s] = True
            nxt.revealed[player, s] = True  # moving >1 square reveals Scout

        owner, slot_grid = MicroStrategoEnv.board_maps(nxt)
        if owner[nr, nc] == -1:
            # Move into empty square
            nxt.pos[player, s] = (nr, nc)
        else:
            # Battle with opponent piece at (nr, nc)
            opp_s = int(slot_grid[nr, nc])
            att_type = int(nxt.piece_type[player, s])
            def_type = int(nxt.piece_type[opp, opp_s])
            # Both pieces are revealed in battle!
            nxt.revealed[player, s] = True
            nxt.revealed[opp, opp_s] = True
            outcome, flag_cap = resolve_battle(att_type, def_type)
            if flag_cap:
                nxt.alive[opp, opp_s] = False
                nxt.pos[opp, opp_s] = (-1, -1)
                nxt.pos[player, s] = (nr, nc)
                nxt.winner = player
                nxt.turn += 1
                return nxt
            if outcome == +1:
                nxt.alive[opp, opp_s] = False
                nxt.pos[opp, opp_s] = (-1, -1)
                nxt.pos[player, s] = (nr, nc)
            elif outcome == -1:
                nxt.alive[player, s] = False
                nxt.pos[player, s] = (-1, -1)
            else:
                nxt.alive[player, s] = False
                nxt.pos[player, s] = (-1, -1)
                nxt.alive[opp, opp_s] = False
                nxt.pos[opp, opp_s] = (-1, -1)

        nxt.turn += 1
        # Check if next player has any legal moves
        opp_moves = MicroStrategoEnv.legal_actions(nxt, opp)
        if len(opp_moves) == 0:
            my_moves = MicroStrategoEnv.legal_actions(nxt, player)
            nxt.winner = player if len(my_moves) > 0 else -1
        elif nxt.turn >= MicroStrategoEnv.MAX_TURNS:
            nxt.winner = -1  # draw at turn limit
        return nxt

    @staticmethod
    def exact_consistent_setup_mask(state: PieceTracker, target_player: int) -> np.ndarray:
        """
        Computes the exact boolean mask over all 720 initial setups for `target_player`
        that are logically consistent with all public observations in `state`!
        - If revealed[target_player, s]: perm[s] == piece_type[target_player, s]
        - If moved_long[target_player, s]: perm[s] == SCOUT
        - If has_moved[target_player, s]: perm[s] not in (FLAG, BOMB)
        """
        mask = np.ones(720, dtype=bool)
        for s in range(6):
            if state.revealed[target_player, s]:
                mask &= (ALL_PERMS[:, s] == state.piece_type[target_player, s])
            elif state.moved_long[target_player, s]:
                mask &= (ALL_PERMS[:, s] == SCOUT)
            elif state.has_moved[target_player, s]:
                mask &= (ALL_PERMS[:, s] != FLAG) & (ALL_PERMS[:, s] != BOMB)
        return mask

    @staticmethod
    def exact_slot_marginals(state: PieceTracker, target_player: int, setup_prior: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Returns the exact Bayesian posterior marginal table P*(Z_s = ptype | public history)
        of shape (6, 6) for slots s=0..5 and piece types 1..6.
        """
        mask = MicroStrategoEnv.exact_consistent_setup_mask(state, target_player)
        if setup_prior is None:
            weights = mask.astype(np.float64)
        else:
            weights = setup_prior * mask
        total = weights.sum()
        if total <= 0:
            weights = mask.astype(np.float64)
            total = weights.sum()
        weights /= total
        marginals = np.zeros((6, 6), dtype=np.float64)
        for ptype in range(1, 7):
            # ALL_PERMS[:, s] == ptype has shape (720, 6)
            marginals[:, ptype - 1] = weights @ (ALL_PERMS == ptype)
        return marginals


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    win_counts = {0: 0, 1: 0, -1: 0}
    lengths = []
    branching = []
    consistent_counts_mid = []
    for ep in range(500):
        r_idx = int(rng.integers(0, 720))
        b_idx = int(rng.integers(0, 720))
        st = MicroStrategoEnv.new_game(r_idx, b_idx)
        while st.winner is None:
            acts = MicroStrategoEnv.legal_actions(st)
            branching.append(len(acts))
            if st.turn == 8:
                consistent_counts_mid.append(int(MicroStrategoEnv.exact_consistent_setup_mask(st, 1).sum()))
            act = acts[int(rng.integers(0, len(acts)))]
            st = MicroStrategoEnv.step(st, act)
        win_counts[st.winner] += 1
        lengths.append(st.turn)
    print(f"500 random games: wins={win_counts}, mean_len={np.mean(lengths):.1f}")
    print(f"Mean legal moves/turn: {np.mean(branching):.2f} (max {np.max(branching)})")
    print(f"Mean consistent opponent setups at turn 8: {np.mean(consistent_counts_mid):.1f} / 720")
