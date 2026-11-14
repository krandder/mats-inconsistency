#!/usr/bin/env python3
"""E-MS game layer on top of env.py (unchanged): canonical observer encoding, the softmax-heuristic behaviour
policy family (temperatures 0.5/1/2), its exact action likelihood under every hypothetical opponent setup, the exact
Bayes posterior P*_full over the 720 setups (public-constraint mask x behaviour likelihood, tau unknown -> mixture),
the constraint-only posterior P*_c (env.exact_slot_marginals), a brute-force re-derivation used to verify both, and
simulator Monte-Carlo action values Q*(s, a) for scoring.  Oracle and simulator are used for data generation and
scoring only, never by an arm.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from env import (ALL_PERMS, BLUE_SLOTS, BOMB, FLAG, MARSHAL, MINER, RED_SLOTS, SCOUT, MicroStrategoEnv as E,
                 PieceTracker, resolve_battle)

TAUS = (0.5, 1.0, 2.0)
LOG_TAU_PRIOR = math.log(1.0 / 3.0)
VAL = np.array([0.0, 10.0, 1.0, 1.0, 2.0, 3.0, 4.0])   # piece value by type (index 0 unused)
PROGRESS = 0.3
CMAP = np.array([1, 0, 5, 4, 3, 2])                     # slot -> canonical slot when the observer is Blue (involution)
OPP_INIT_SQ = [(2, 2), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3)]   # canonical initial squares of opponent slots 0..5
ROW_MAJOR = [5, 4, 3, 2, 1, 0]                          # Ataraxos fixed order: back row then front row (canonical)
N_OUT = 5                                               # 0 move, 1 attacker wins, 2 defender wins, 3 tie, 4 flag taken
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# battle value table bv[k, d] for attacker type k vs defender type d
BV = np.zeros((7, 7))
for _k in range(1, 7):
    for _d in range(1, 7):
        o, fl = resolve_battle(_k, _d)
        BV[_k, _d] = 10.0 if fl else (VAL[_d] if o == 1 else (-VAL[_k] if o == -1 else VAL[_d] - VAL[_k]))


def canon_sq(r: int, c: int, p: int) -> int:
    if p == 1:
        r, c = 3 - r, 3 - c
    return 4 * r + c


def canon_slot(s: int, p: int) -> int:
    return s if p == 0 else int(CMAP[s])


# ----------------------------------------------------------------------------------------------- behaviour policy
def slot_moves(state: PieceTracker, X: int, s: int, k: int, owner: np.ndarray) -> List[Tuple[int, int, int]]:
    """Moves slot s of player X would have if its type were k (same rules as env.legal_actions)."""
    if k in (FLAG, BOMB) or not state.alive[X, s]:
        return []
    r0, c0 = int(state.pos[X, s, 0]), int(state.pos[X, s, 1])
    mx = 3 if k == SCOUT else 1
    out = []
    for dr, dc in DIRS:
        for st in range(1, mx + 1):
            nr, nc = r0 + dr * st, c0 + dc * st
            if not (0 <= nr < 4 and 0 <= nc < 4):
                break
            occ = owner[nr, nc]
            if occ == X:
                break
            out.append((s, nr, nc))
            if occ == 1 - X:
                break
    return out


def move_score(state: PieceTracker, X: int, mv, k: int, pubM_Y: np.ndarray, slot_grid: np.ndarray, owner) -> float:
    s, nr, nc = mv
    r0 = int(state.pos[X, s, 0])
    if owner[nr, nc] == 1 - X:
        j = int(slot_grid[nr, nc])
        return float(pubM_Y[j] @ BV[k, 1:])
    adv = (nr - r0) if X == 0 else (r0 - nr)
    return PROGRESS * adv


def policy_probs(state: PieceTracker, tau: float) -> Tuple[List[Tuple[int, int, int]], np.ndarray]:
    """Behaviour policy of the player to move (true own types, public info about the opponent)."""
    X = state.turn % 2
    owner, slot_grid = E.board_maps(state)
    pubM_Y = E.exact_slot_marginals(state, 1 - X)
    acts, sc = [], []
    for s in range(6):
        k = int(state.piece_type[X, s])
        for mv in slot_moves(state, X, s, k, owner):
            acts.append(mv)
            sc.append(move_score(state, X, mv, k, pubM_Y, slot_grid, owner))
    if not acts:
        return acts, np.zeros(0)
    z = np.array(sc) / tau
    z -= z.max()
    p = np.exp(z)
    return acts, p / p.sum()


def move_loglik_table(state: PieceTracker, action) -> np.ndarray:
    """For the player X to move in `state` and its actual `action`: log pi_tau(action | state, setup) for every
    hypothetical setup of X (720, in X's slot order) and tau (3). -inf where the action is impossible."""
    X = state.turn % 2
    owner, slot_grid = E.board_maps(state)
    pubM_Y = E.exact_slot_marginals(state, 1 - X)
    sa, ar, ac = action
    out = np.full((720, len(TAUS)), -np.inf)
    # per-slot per-type score lists
    S = np.zeros((len(TAUS), 6, 7))          # sum exp(score / tau) over moves of slot j as type k
    num = np.full((len(TAUS), 7), 0.0)        # exp(score(action) / tau) if legal for type k
    for j in range(6):
        if not state.alive[X, j]:
            continue
        for k in range(1, 7):
            mvs = slot_moves(state, X, j, k, owner)
            if not mvs:
                continue
            sc = np.array([move_score(state, X, mv, k, pubM_Y, slot_grid, owner) for mv in mvs])
            for ti, tau in enumerate(TAUS):
                S[ti, j, k] = np.exp(sc / tau).sum()
            if j == sa:
                for mi, mv in enumerate(mvs):
                    if mv[1] == ar and mv[2] == ac:
                        for ti, tau in enumerate(TAUS):
                            num[ti, k] = math.exp(sc[mi] / tau)
    P = ALL_PERMS.astype(np.int64)
    for ti in range(len(TAUS)):
        den = np.zeros(720)
        for j in range(6):
            den += S[ti, j][P[:, j]]
        nu = num[ti][P[:, sa]]
        ok = (nu > 0) & (den > 0)
        out[ok, ti] = np.log(nu[ok]) - np.log(den[ok])
    return out


# ----------------------------------------------------------------------------------------------- game play / records
def outcome_code(prev: PieceTracker, nxt: PieceTracker, action) -> Tuple[int, int, int]:
    """(outcome class, revealed attacker type or 0, revealed defender type or 0) of applying action to prev."""
    X = prev.turn % 2
    s, nr, nc = action
    owner, slot_grid = E.board_maps(prev)
    att = int(prev.piece_type[X, s])
    r0, c0 = int(prev.pos[X, s, 0]), int(prev.pos[X, s, 1])
    long_move = abs(nr - r0) + abs(nc - c0) > 1
    if owner[nr, nc] == -1:
        return 0, (att if long_move else 0), 0
    d = int(prev.piece_type[1 - X, int(slot_grid[nr, nc])])
    o, fl = resolve_battle(att, d)
    if fl:
        return 4, att, d
    return {1: 1, -1: 2, 0: 3}[o], att, d


def play_game(red_idx: int, blue_idx: int, taus: Tuple[float, float], rng: np.random.Generator):
    """Self-play with the behaviour policy. Returns list of (state_before, action) and the final state."""
    st = E.new_game(red_idx, blue_idx)
    hist = []
    while st.winner is None:
        acts, p = policy_probs(st, taus[st.turn % 2])
        a = acts[int(rng.choice(len(acts), p=p))]
        hist.append((st, a))
        st = E.step(st, a)
    return hist, st


def result_for(final: PieceTracker, P: int) -> int:
    """0 win, 1 draw, 2 loss for player P."""
    if final.winner == -1:
        return 1
    return 0 if final.winner == P else 2


def encode_root(st: PieceTracker, mcount: np.ndarray) -> Dict[str, np.ndarray]:
    """Canonical encoding from the view of the player to move P. mcount (2,6) = moves made per slot so far."""
    P = st.turn % 2
    O = 1 - P
    sq_content = np.zeros(16, np.int8)
    sq_ownflag = np.zeros(16, np.int8)
    sq_slot = np.zeros(16, np.int8)
    slot_loc = np.full(6, 16, np.int8)
    slot_flags = np.zeros(6, np.int8)
    slot_mcount = np.zeros(6, np.int8)
    slot_pubrev = np.zeros(6, np.int8)
    slot_true = np.zeros(6, np.int8)
    own_true = np.zeros(6, np.int8)
    for s in range(6):
        cs = canon_slot(s, P)
        own_true[cs] = st.piece_type[P, s]
        if st.alive[P, s]:
            q = canon_sq(int(st.pos[P, s, 0]), int(st.pos[P, s, 1]), P)
            sq_content[q] = st.piece_type[P, s]
            sq_ownflag[q] = 1 + int(st.has_moved[P, s]) + 2 * int(st.revealed[P, s])
    for s in range(6):
        cs = canon_slot(s, P)
        slot_true[cs] = st.piece_type[O, s]
        slot_pubrev[cs] = int(st.revealed[O, s])
        slot_flags[cs] = int(st.has_moved[O, s]) + int(st.moved_long[O, s])
        slot_mcount[cs] = min(int(mcount[O, s]), 3)
        if st.alive[O, s]:
            q = canon_sq(int(st.pos[O, s, 0]), int(st.pos[O, s, 1]), P)
            slot_loc[cs] = q
            sq_slot[q] = cs + 1
            sq_content[q] = 7 + (int(st.piece_type[O, s]) if st.revealed[O, s] else 0)
    return dict(sq_content=sq_content, sq_ownflag=sq_ownflag, sq_slot=sq_slot, slot_loc=slot_loc, slot_flags=slot_flags,
                slot_mcount=slot_mcount, slot_pubrev=slot_pubrev, slot_true=slot_true, own_true=own_true,
                turn=np.int8(st.turn))


def encode_ply(prev: PieceTracker, action, nxt: PieceTracker, P: int) -> Tuple[int, int, int, int, int, int]:
    """(actor 0=observer 1=opponent, from canon sq, to canon sq, outcome, att type, def type) in observer P's frame."""
    X = prev.turn % 2
    s, nr, nc = action
    fr = canon_sq(int(prev.pos[X, s, 0]), int(prev.pos[X, s, 1]), P)
    to = canon_sq(nr, nc, P)
    oc, at, dt = outcome_code(prev, nxt, action)
    return (0 if X == P else 1), fr, to, oc, at, dt


# ----------------------------------------------------------------------------------------------- exact posteriors
def posterior_full(hist, t: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact Bayes posterior of the opponent's setup at time t (observer = player to move at t), from the full public
    history: uniform prior over 720 x uniform tau prior x consistency x behaviour likelihood of every opponent move.
    Returns (marginals over canonical slots (6,6) [slot, type-1], joint weights (720,3) over (setup, tau_opp),
    constraint-only marginals (6,6) canonical)."""
    st = hist[t][0]
    P = st.turn % 2
    O = 1 - P
    mask = E.exact_consistent_setup_mask(st, O)
    ll = np.zeros((720, len(TAUS)))
    for tp in range(t):
        s_tp, a_tp = hist[tp]
        if s_tp.turn % 2 == O:
            ll += move_loglik_table(s_tp, a_tp)
    ll[~mask] = -np.inf
    m = ll.max()
    w = np.exp(ll - m)
    w /= w.sum()
    ws = w.sum(1)
    marg = np.zeros((6, 6))
    for k in range(1, 7):
        marg[:, k - 1] = ws @ (ALL_PERMS == k)
    mc = E.exact_slot_marginals(st, O)
    cm = np.zeros((6, 6))
    cf = np.zeros((6, 6))
    for s in range(6):
        cm[canon_slot(s, P)] = marg[s]
        cf[canon_slot(s, P)] = mc[s]
    return cm, w, cf


def posterior_bruteforce(hist, t: int, red_idx: int, blue_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """Independent re-derivation for verification: for every hypothetical opponent setup, replay the whole public
    history with that setup substituted (env.step), require identical public observations at every ply, and multiply
    the behaviour probabilities (policy_probs) of the opponent's moves. Returns canonical (P*_full, P*_c)."""
    st = hist[t][0]
    P = st.turn % 2
    O = 1 - P
    w = np.zeros((720, len(TAUS)))
    wc = np.zeros(720)
    for z in range(720):
        idx = [red_idx, blue_idx]
        idx[O] = z
        g = E.new_game(idx[0], idx[1])
        ok = True
        lik = np.zeros(len(TAUS))
        for tp in range(t):
            s_real, a = hist[tp]
            X = g.turn % 2
            if X == O:
                for ti, tau in enumerate(TAUS):
                    acts, p = policy_probs(g, tau)
                    if a not in acts:
                        ok = False
                        break
                    lik[ti] += math.log(p[acts.index(a)])
                if not ok:
                    break
            else:
                if a not in E.legal_actions(g):
                    ok = False
                    break
            g2 = E.step(g, a)
            nxt_real = hist[tp + 1][0] if tp + 1 < len(hist) else None
            if nxt_real is None or not (np.array_equal(g2.alive, nxt_real.alive) and np.array_equal(g2.pos, nxt_real.pos)
                                        and np.array_equal(g2.revealed, nxt_real.revealed)
                                        and np.array_equal(g2.piece_type[P], nxt_real.piece_type[P])
                                        and np.array_equal(np.where(g2.revealed[O], g2.piece_type[O], 0),
                                                           np.where(nxt_real.revealed[O], nxt_real.piece_type[O], 0))):
                ok = False
                break
            g = g2
        if ok:
            w[z] = np.exp(lik)
            wc[z] = 1.0
    w /= w.sum()
    wc /= wc.sum()
    ws = w.sum(1)
    out_f, out_c = np.zeros((6, 6)), np.zeros((6, 6))
    for k in range(1, 7):
        mf = ws @ (ALL_PERMS == k)
        mcc = wc @ (ALL_PERMS == k)
        for s in range(6):
            out_f[canon_slot(s, P), k - 1] = mf[s]
            out_c[canon_slot(s, P), k - 1] = mcc[s]
    return out_f, out_c


# ----------------------------------------------------------------------------------------------- simulator Q*
def q_star(hist, t: int, red_idx: int, blue_idx: int, actions, n_mc: int, rng: np.random.Generator,
           joint_w: Optional[np.ndarray] = None) -> np.ndarray:
    """Simulator Monte-Carlo value of each root action for the player to move at t: (Z, tau_opp) ~ exact posterior,
    tau_self ~ uniform, behaviour policy to the end. Returns (n_actions, 3) frequencies of win/draw/loss for P."""
    st = hist[t][0]
    P = st.turn % 2
    O = 1 - P
    if joint_w is None:
        _, joint_w, _ = posterior_full(hist, t)
    flat = joint_w.ravel()
    out = np.zeros((len(actions), 3))
    for ai, a in enumerate(actions):
        draws = rng.choice(flat.size, size=n_mc, p=flat)
        for d in draws:
            z, ti = divmod(int(d), len(TAUS))
            g = st.clone()
            g.piece_type = g.piece_type.copy()
            g.piece_type[O] = ALL_PERMS[z]
            taus = [0.0, 0.0]
            taus[O] = TAUS[ti]
            taus[P] = TAUS[int(rng.integers(len(TAUS)))]
            g = E.step(g, a)
            while g.winner is None:
                acts, p = policy_probs(g, taus[g.turn % 2])
                g = E.step(g, acts[int(rng.choice(len(acts), p=p))])
            out[ai, result_for(g, P)] += 1
    return out / n_mc


def flank_of(st: PieceTracker, action) -> int:
    """0 = Left flank (canonical from-column 0..1), 1 = Right (2..3), in the mover's canonical frame."""
    P = st.turn % 2
    s = action[0]
    q = canon_sq(int(st.pos[P, s, 0]), int(st.pos[P, s, 1]), P)
    return 0 if q % 4 < 2 else 1
