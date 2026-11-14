#!/usr/bin/env python3
"""R5-E2 rung-2 board: 6x6 8-piece Barrage-style micro-Stratego (extends ../env.py; the 4x4 code is untouched).

Board (rows 0..5, cols 0..5), lakes (2,2) and (3,3) are impassable:
  row 5  .  o  o  o  o  .      Blue back slots 7 6 5 4   (Blue setup squares are the 180-degree rotation of Red's)
  row 4  o  o  .  .  o  o      Blue front slots 3 2 . . 1 0
  row 3  .  .  .  ~  .  .      ~ = lake
  row 2  .  .  ~  .  .  .
  row 1  x  x  .  .  x  x      Red front slots 0 1 . . 2 3
  row 0  .  x  x  x  x  .      Red back slots . 4 5 6 7 .
Lanes: left flank cols 0..1, centre weave cols 2..3 (the lakes force a sidestep), right flank cols 4..5.

8 pieces per side: Flag(1) x1, Bomb(2) x2, Scout(3) x2, Miner(4) x1, General(5) x1, Marshal(6) x1 on 8 fixed public
squares; the hidden setup is the assignment of the multiset to the 8 slots: 8!/(2!2!) = 10,080 setups per side.
Rules as ../env.py: Flag and Bombs never move; Scouts move any distance through empty squares in a straight line (a
move longer than 1 publicly reveals a Scout) and may attack the first piece they reach; others move 1 square
orthogonally. Battle: Flag is captured by anything (game over); Bomb beats every attacker except the Miner (attacker
removed, Bomb stays); otherwise higher rank wins (Scout 3 < Miner 4 < General 5 < Marshal 6), equal ranks remove both.
Both pieces of a battle are publicly revealed. A player with no legal move loses; MAX_TURNS plies -> draw.
Slots are indexed so that the canonical frame (observer at the bottom; Blue sees the board rotated 180 degrees) maps
own slot s to RED_SQ[s] and opponent slot s to ROT(RED_SQ[s]) for both colours.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

N = 6
LAKES = ((2, 2), (3, 3))
LAKE_SQ = (2 * N + 2, 3 * N + 3)
EMPTY, FLAG, BOMB, SCOUT, MINER, GENERAL, MARSHAL = 0, 1, 2, 3, 4, 5, 6
NAMES = {0: ".", 1: "F", 2: "B", 3: "S", 4: "M", 5: "G", 6: "X"}
MULTISET = (FLAG, BOMB, BOMB, SCOUT, SCOUT, MINER, GENERAL, MARSHAL)
COUNTS = np.array([0, 1, 2, 2, 1, 1, 1])          # per type 0..6
NS = 8
RED_SQ: List[Tuple[int, int]] = [(1, 0), (1, 1), (1, 4), (1, 5), (0, 1), (0, 2), (0, 3), (0, 4)]
BLUE_SQ: List[Tuple[int, int]] = [(N - 1 - r, N - 1 - c) for r, c in RED_SQ]
SQ0 = (RED_SQ, BLUE_SQ)
MAX_TURNS = 60
ALL_SETUPS: np.ndarray = np.array(sorted(set(itertools.permutations(MULTISET))), dtype=np.int8)   # (10080, 8)
assert ALL_SETUPS.shape == (10080, 8)
DIRS = ((1, 0), (-1, 0), (0, -1), (0, 1))
IS_LAKE = np.zeros((N, N), bool)
for _r, _c in LAKES:
    IS_LAKE[_r, _c] = True


def resolve_battle(att: int, de: int) -> Tuple[int, bool]:
    """(+1 attacker wins and moves in, -1 defender wins, 0 both removed; flag captured?)"""
    if de == FLAG:
        return 1, True
    if de == BOMB:
        return (1, False) if att == MINER else (-1, False)
    if att > de:
        return 1, False
    if att < de:
        return -1, False
    return 0, False


@dataclass
class State:
    piece_type: np.ndarray   # (2, 8) int8 true types
    pos: np.ndarray          # (2, 8, 2) int8, (-1,-1) if dead
    alive: np.ndarray        # (2, 8) bool
    has_moved: np.ndarray    # (2, 8) bool (public)
    moved_long: np.ndarray   # (2, 8) bool (public: a Scout)
    revealed: np.ndarray     # (2, 8) bool (public: exact type)
    mcount: np.ndarray       # (2, 8) int moves made (public)
    turn: int = 0
    winner: Optional[int] = None   # 0, 1, or -1 draw

    def clone(self) -> "State":
        return State(self.piece_type.copy(), self.pos.copy(), self.alive.copy(), self.has_moved.copy(),
                     self.moved_long.copy(), self.revealed.copy(), self.mcount.copy(), self.turn, self.winner)


def new_game(red_idx: int, blue_idx: int) -> State:
    pt = np.zeros((2, NS), np.int8)
    pt[0], pt[1] = ALL_SETUPS[red_idx], ALL_SETUPS[blue_idx]
    pos = np.zeros((2, NS, 2), np.int8)
    for p in range(2):
        for s, (r, c) in enumerate(SQ0[p]):
            pos[p, s] = (r, c)
    z = lambda: np.zeros((2, NS), bool)
    return State(pt, pos, np.ones((2, NS), bool), z(), z(), z(), np.zeros((2, NS), np.int16), 0, None)


def board_maps(st: State) -> Tuple[np.ndarray, np.ndarray]:
    owner = np.full((N, N), -1, np.int8)
    slot = np.full((N, N), -1, np.int8)
    for p in range(2):
        for s in range(NS):
            if st.alive[p, s]:
                r, c = st.pos[p, s]
                owner[r, c] = p
                slot[r, c] = s
    return owner, slot


def piece_moves(st: State, X: int, s: int, k: int, owner: np.ndarray) -> List[Tuple[int, int, int]]:
    """Moves (s, r, c) slot s of player X would have if its type were k."""
    if k in (FLAG, BOMB) or not st.alive[X, s]:
        return []
    r0, c0 = int(st.pos[X, s, 0]), int(st.pos[X, s, 1])
    mx = N if k == SCOUT else 1
    out = []
    for dr, dc in DIRS:
        for step in range(1, mx + 1):
            r, c = r0 + dr * step, c0 + dc * step
            if not (0 <= r < N and 0 <= c < N) or IS_LAKE[r, c]:
                break
            o = owner[r, c]
            if o == X:
                break
            out.append((s, r, c))
            if o == 1 - X:
                break
    return out


def legal_actions(st: State, player: Optional[int] = None) -> List[Tuple[int, int, int]]:
    if st.winner is not None:
        return []
    X = st.turn % 2 if player is None else player
    owner, _ = board_maps(st)
    out = []
    for s in range(NS):
        out += piece_moves(st, X, s, int(st.piece_type[X, s]), owner)
    return out


def step(st: State, action: Tuple[int, int, int]) -> State:
    nx = st.clone()
    X = nx.turn % 2
    Y = 1 - X
    s, r, c = action
    r0, c0 = int(nx.pos[X, s, 0]), int(nx.pos[X, s, 1])
    nx.has_moved[X, s] = True
    nx.mcount[X, s] += 1
    if abs(r - r0) + abs(c - c0) > 1:
        nx.moved_long[X, s] = True
        nx.revealed[X, s] = True
    owner, slot = board_maps(nx)
    if owner[r, c] == -1:
        nx.pos[X, s] = (r, c)
    else:
        j = int(slot[r, c])
        a, d = int(nx.piece_type[X, s]), int(nx.piece_type[Y, j])
        nx.revealed[X, s] = True
        nx.revealed[Y, j] = True
        o, fl = resolve_battle(a, d)
        if fl:
            nx.alive[Y, j] = False
            nx.pos[Y, j] = (-1, -1)
            nx.pos[X, s] = (r, c)
            nx.winner = X
            nx.turn += 1
            return nx
        if o == 1:
            nx.alive[Y, j] = False
            nx.pos[Y, j] = (-1, -1)
            nx.pos[X, s] = (r, c)
        elif o == -1:
            nx.alive[X, s] = False
            nx.pos[X, s] = (-1, -1)
        else:
            nx.alive[X, s] = False
            nx.pos[X, s] = (-1, -1)
            nx.alive[Y, j] = False
            nx.pos[Y, j] = (-1, -1)
    nx.turn += 1
    if not legal_actions(nx, Y):
        nx.winner = X if legal_actions(nx, X) else -1
    elif nx.turn >= MAX_TURNS:
        nx.winner = -1
    return nx


def constraint_mask(st: State, Y: int) -> np.ndarray:
    """Setups of player Y (10080) consistent with every public observation about Y."""
    m = np.ones(len(ALL_SETUPS), bool)
    for s in range(NS):
        col = ALL_SETUPS[:, s]
        if st.revealed[Y, s]:
            m &= col == st.piece_type[Y, s]
        elif st.moved_long[Y, s]:
            m &= col == SCOUT
        elif st.has_moved[Y, s]:
            m &= (col != FLAG) & (col != BOMB)
    return m


ONEHOT = np.stack([(ALL_SETUPS == k) for k in range(1, 7)], -1).astype(np.float64).reshape(len(ALL_SETUPS), NS * 6)


def marginals(w: np.ndarray) -> np.ndarray:
    """Weights over the 10080 setups -> per-slot type marginals (8, 6) [slot, type-1]."""
    w = w / w.sum()
    return (w @ ONEHOT).reshape(NS, 6)


def constraint_marginals(st: State, Y: int) -> np.ndarray:
    return marginals(constraint_mask(st, Y).astype(np.float64))


def show(st: State) -> str:
    owner, slot = board_maps(st)
    rows = []
    for r in range(N - 1, -1, -1):
        cells = []
        for c in range(N):
            if IS_LAKE[r, c]:
                cells.append(" ~ ")
            elif owner[r, c] == -1:
                cells.append(" . ")
            else:
                p, s = owner[r, c], slot[r, c]
                t = NAMES[int(st.piece_type[p, s])]
                cells.append(("r" if p == 0 else "b") + t + ("*" if st.revealed[p, s] else " "))
        rows.append(f"{r} " + "".join(cells))
    return "\n".join(rows)


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    lens, wins, br = [], {0: 0, 1: 0, -1: 0}, []
    for g in range(300):
        st = new_game(int(rng.integers(10080)), int(rng.integers(10080)))
        while st.winner is None:
            a = legal_actions(st)
            br.append(len(a))
            st = step(st, a[int(rng.integers(len(a)))])
        wins[st.winner] += 1
        lens.append(st.turn)
    print("random play: wins", wins, "mean len", np.mean(lens), "branching", np.mean(br), max(br))
