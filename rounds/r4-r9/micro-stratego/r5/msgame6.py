#!/usr/bin/env python3
"""R5-E2 game layer for env6 (6x6 Barrage micro-Stratego): the coupled 2-ply behaviour policy, its exact likelihood
under every hypothetical setup (vectorised over the 10,080 setups), the exact Bayes posterior P*_full, the
constraint-only posterior P*_c, canonical encodings, and simulator Monte-Carlo action values Q* (win/draw/loss and
the 4-ply Miner utility) for scoring. The oracle and the simulator are used for data generation, scoring and gates
only, never by an arm or a training loss.

Behaviour policy of player X with setup Z (types of its 8 slots): softmax(score / tau) over legal moves, tau per player
per game in {0.5, 1, 2}. score(move s: fr -> to; Z), with k = Z[s] and pubM = the public constraint-only marginals of
the opponent's pieces (what X can know):
  attack (to holds opponent piece j): E_pubM[j][BV(k, d)]  (battle value; Flag 10, Bomb 1, S 1, M 2, G 3, X 4)
       + reply term after a won battle: -0.7 * P(k wins and stays) * threat(to, k) * (0.5 if defended(Z) else 1)
  move to an empty square: 0.3 * (rows advanced)
       + 0.6 if k is a Scout advancing onto an open lane (the square ahead of `to` is free)
       + 0.8 if k is the Miner, the move closes in on an unmoved opponent piece in the opponent's back two rows, and a
             friendly General or Marshal is within Chebyshev distance 2 of `to` (escort; COUPLED: depends on Z[s'])
       - 0.7 * threat(to, k) * (0.5 if defended(Z) else 1)   (2-ply reply against the public marginals)
  every move: -0.8 if the mover leaves the orthogonal neighbourhood of its own Flag, unless another friendly Bomb or
       Marshal stays next to the Flag (COUPLED: depends on which slot holds the Flag and on the neighbours' types)
  threat(to, k) = max over opponent pieces j next to `to` of P_pubM(j is mobile and beats k; tie 0.5) * VAL[k];
  defended(Z) = a friendly General or Marshal (not the mover) next to `to`.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

import env6 as E

TAUS = (0.5, 1.0, 2.0)
VAL = np.array([0.0, 10.0, 1.0, 1.0, 2.0, 3.0, 4.0])
PROGRESS, SCOUT_BONUS, MINER_BONUS, FLAG_PEN, REPLY_W = 0.3, 0.6, 0.8, 0.8, 0.7
N_OUT = 5               # outcome classes: 0 move, 1 attacker wins, 2 defender wins, 3 tie, 4 flag taken
N = E.N
NS = E.NS
LMAX = 32               # max legal moves stored per position

BV = np.zeros((7, 7))
for _k in range(1, 7):
    for _d in range(1, 7):
        _o, _fl = E.resolve_battle(_k, _d)
        BV[_k, _d] = 10.0 if _fl else (VAL[_d] if _o == 1 else (-VAL[_k] if _o == -1 else VAL[_d] - VAL[_k]))
# BEATS[d, k]: probability-weight that an attacking d beats a defending k (tie 0.5); immobile attackers (F, B) never attack
BEATS = np.zeros((7, 7))
for _d in (E.SCOUT, E.MINER, E.GENERAL, E.MARSHAL):
    for _k in range(1, 7):
        _o, _ = E.resolve_battle(_d, _k)
        BEATS[_d, _k] = 1.0 if _o == 1 else (0.5 if _o == 0 else 0.0)
# SURV[k, d]: attacker k wins against d and stays on the square (not a flag capture, which ends the game)
SURV = np.zeros((7, 7))
for _k in range(1, 7):
    for _d in range(1, 7):
        _o, _fl = E.resolve_battle(_k, _d)
        SURV[_k, _d] = 1.0 if (_o == 1 and not _fl) else 0.0


def fwd(X: int) -> int:
    return 1 if X == 0 else -1


def canon_sq(r: int, c: int, P: int) -> int:
    if P == 1:
        r, c = N - 1 - r, N - 1 - c
    return N * r + c


def flank_of_sq(q: int) -> int:
    """0 = Left (canonical cols 0..2), 1 = Right (cols 3..5)."""
    return 0 if q % N <= 2 else 1


# ------------------------------------------------------------------------------------------------ policy (vectorised)
def candidate_moves(st: E.State, X: int, owner: np.ndarray) -> List[Tuple[int, int, int, bool]]:
    """Every move of X's alive slots that is legal for some type: (s, r, c, long)."""
    out = []
    for s in range(NS):
        if not st.alive[X, s] or (st.revealed[X, s] and st.piece_type[X, s] in (E.FLAG, E.BOMB)):
            continue
        r0, c0 = int(st.pos[X, s, 0]), int(st.pos[X, s, 1])
        for dr, dc in E.DIRS:
            for k in range(1, N + 1):
                r, c = r0 + dr * k, c0 + dc * k
                if not (0 <= r < N and 0 <= c < N) or E.IS_LAKE[r, c]:
                    break
                o = owner[r, c]
                if o == X:
                    break
                out.append((s, r, c, k > 1))
                if o == 1 - X:
                    break
    return out


def _adj4(a, b) -> bool:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1])) == 1


class PolicyCtx:
    """Per-state precomputation for scoring X's candidate moves under many hypothetical setups."""

    def __init__(self, st: E.State, X: int, pubM_Y: Optional[np.ndarray] = None):
        self.st, self.X, self.Y = st, X, 1 - X
        self.owner, self.slot = E.board_maps(st)
        self.pubM = E.constraint_marginals(st, 1 - X) if pubM_Y is None else pubM_Y     # (8, 6)
        self.moves = candidate_moves(st, X, self.owner)
        alive = st.alive[X]
        pos = st.pos[X]
        self.alive = alive
        A = np.zeros((NS, NS), bool)
        for a in range(NS):
            for b in range(NS):
                if alive[a] and alive[b] and a != b:
                    A[a, b] = _adj4(pos[a], pos[b])
        self.A = A
        Y = self.Y
        home = (0, 1) if Y == 0 else (N - 2, N - 1)
        self.targets = [(int(st.pos[Y, j, 0]), int(st.pos[Y, j, 1])) for j in range(NS)
                        if st.alive[Y, j] and not st.has_moved[Y, j] and int(st.pos[Y, j, 0]) in home]
        # per opponent slot: P(mobile and beats k) for k = 1..6
        self.beat = np.zeros((NS, 7))
        for j in range(NS):
            if st.alive[Y, j]:
                self.beat[j] = self.pubM[j] @ BEATS[1:, :]
        # static (setup-independent) per-move quantities
        posX = st.pos[X]
        pl = [(int(posX[b, 0]), int(posX[b, 1])) for b in range(NS)]
        self.static = []
        for (s, r, c, lng) in self.moves:
            r0, c0 = pl[s]
            oth = [b for b in range(NS) if alive[b] and b != s]
            adj_to = np.zeros(NS, bool)
            near2 = np.zeros(NS, bool)
            for b in oth:
                adj_to[b] = abs(r - pl[b][0]) + abs(c - pl[b][1]) == 1
                near2[b] = max(abs(r - pl[b][0]), abs(c - pl[b][1])) <= 2
            if self.owner[r, c] == Y:
                j = int(self.slot[r, c])
                info = dict(att=True, base=BV[1:, 1:] @ self.pubM[j], sv=SURV[1:, 1:] @ self.pubM[j],
                            thr=self.threat(r, c, exclude=j))
            else:
                adv = (r - r0) * fwd(X)
                ar_ = r + fwd(X)
                free = (0 <= ar_ < N) and not E.IS_LAKE[ar_, c] and self.owner[ar_, c] == -1
                miner_ok = bool(self.targets) and self.tdist(r, c) < self.tdist(r0, c0)
                info = dict(att=False, prog=PROGRESS * adv, scout=bool(adv > 0 and free), miner=miner_ok,
                            thr=self.threat(r, c))
            info.update(s=s, r=r, c=c, lng=lng, adj_to=adj_to, near2=near2,
                        ota=np.array([s != b for b in range(NS)]))
            self.static.append(info)

    def tdist(self, r, c) -> int:
        return min((abs(r - a) + abs(c - b) for a, b in self.targets), default=99)

    def threat(self, r, c, exclude=-1) -> np.ndarray:
        """(7,) threat by type k of the piece standing on (r, c)."""
        t = np.zeros(7)
        for dr, dc in E.DIRS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < N and 0 <= cc < N and self.owner[rr, cc] == self.Y:
                j = int(self.slot[rr, cc])
                if j != exclude:
                    t = np.maximum(t, self.beat[j])
        return t * VAL

    def scores(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Z (n, 8) hypothetical types of X's slots -> (score (n, M), legal (n, M)) over self.moves."""
        st, X = self.st, self.X
        n = Z.shape[0]
        Mv = len(self.moves)
        S = np.zeros((n, Mv))
        L = np.zeros((n, Mv), bool)
        Zi = Z.astype(np.int64)
        isGX = (Zi == E.GENERAL) | (Zi == E.MARSHAL)
        isBX = (Zi == E.BOMB) | (Zi == E.MARSHAL)
        fslot = np.argmax(Zi == E.FLAG, 1)                        # flag slot per row
        alive_f = self.alive[fslot]
        fsq = st.pos[X][fslot].astype(np.int64)                   # (n, 2) flag square per row
        nbA = self.A[:, fslot].T                                  # (n, 8) slots next to the flag
        for m, inf in enumerate(self.static):
            s, r, c = inf["s"], inf["r"], inf["c"]
            k = Zi[:, s]
            L[:, m] = (k == E.SCOUT) if inf["lng"] else ((k != E.FLAG) & (k != E.BOMB))
            dfac = np.where((isGX & inf["adj_to"][None]).any(1), 0.5, 1.0)
            if inf["att"]:
                sc = inf["base"][k - 1] - REPLY_W * inf["sv"][k - 1] * inf["thr"][k] * dfac
            else:
                sc = inf["prog"] - REPLY_W * inf["thr"][k] * dfac
                if inf["scout"]:
                    sc = sc + SCOUT_BONUS * (k == E.SCOUT)
                if inf["miner"]:
                    sc = sc + MINER_BONUS * ((k == E.MINER) & (isGX & inf["near2"][None]).any(1))
            fa = self.A[s, fslot]                                          # mover next to the flag now
            ta = (np.abs(fsq[:, 0] - r) + np.abs(fsq[:, 1] - c)) == 1      # `to` next to the flag
            backed = (isBX & nbA & inf["ota"][None]).any(1)
            S[:, m] = sc - FLAG_PEN * (fa & ~ta & ~backed & alive_f)
        return S, L

    def probs_true(self, tau: float) -> Tuple[List[Tuple[int, int, int]], np.ndarray]:
        Z = self.st.piece_type[self.X][None].astype(np.int64)
        S, L = self.scores(Z)
        acts = [(s, r, c) for (s, r, c, _), ok in zip(self.moves, L[0]) if ok]
        z = S[0][L[0]] / tau
        z = z - z.max()
        p = np.exp(z)
        return acts, p / p.sum()

    def loglik_table(self, action, Zs: np.ndarray) -> np.ndarray:
        """log pi_tau(action | state, Z) for hypothetical setups Zs (n, 8) and every tau -> (n, 3); -inf if illegal."""
        s_, r_, c_ = action
        try:
            m = [i for i, (s, r, c, _) in enumerate(self.moves) if (s, r, c) == (s_, r_, c_)][0]
        except IndexError:
            return np.full((Zs.shape[0], len(TAUS)), -np.inf)
        S, L = self.scores(Zs)
        out = np.full((Zs.shape[0], len(TAUS)), -np.inf)
        ok = L[:, m]
        for ti, tau in enumerate(TAUS):
            z = np.where(L, S / tau, -np.inf)
            mx = z.max(1, keepdims=True)
            lse = mx[:, 0] + np.log(np.exp(z - mx).sum(1))
            out[ok, ti] = z[ok, m] - lse[ok]
        return out


def policy_probs(st: E.State, tau: float):
    return PolicyCtx(st, st.turn % 2).probs_true(tau)


# ------------------------------------------------------------------------------------------------ independent scalar policy
def scalar_scores(st: E.State, X: int, Zrow: np.ndarray) -> Dict[Tuple[int, int, int], float]:
    """Plain-loop re-implementation of the score (for verification only): legal moves under Zrow -> score."""
    g = st.clone()
    g.piece_type = g.piece_type.copy()
    g.piece_type[X] = Zrow
    Y = 1 - X
    owner, slot = E.board_maps(g)
    pubM = E.constraint_marginals(st, Y)
    out = {}
    posX = g.pos[X]
    flag = [s for s in range(NS) if g.alive[X, s] and Zrow[s] == E.FLAG]
    home = (0, 1) if Y == 0 else (N - 2, N - 1)
    targets = [(int(g.pos[Y, j, 0]), int(g.pos[Y, j, 1])) for j in range(NS)
               if g.alive[Y, j] and not g.has_moved[Y, j] and int(g.pos[Y, j, 0]) in home]

    def td(r, c):
        return min((abs(r - a) + abs(c - b) for a, b in targets), default=99)

    def beat(j, k):
        return sum(pubM[j, d - 1] * BEATS[d, k] for d in range(1, 7))

    def threat(r, c, k, exclude=-1):
        t = 0.0
        for dr, dc in E.DIRS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < N and 0 <= cc < N and owner[rr, cc] == Y and slot[rr, cc] != exclude:
                t = max(t, beat(int(slot[rr, cc]), k))
        return t * VAL[k]

    for s in range(NS):
        k = int(Zrow[s])
        for (_, r, c) in E.piece_moves(g, X, s, k, owner):
            r0, c0 = int(posX[s, 0]), int(posX[s, 1])
            defended = any(g.alive[X, b] and b != s and abs(r - posX[b, 0]) + abs(c - posX[b, 1]) == 1 and
                           Zrow[b] in (E.GENERAL, E.MARSHAL) for b in range(NS))
            dfac = 0.5 if defended else 1.0
            if owner[r, c] == Y:
                j = int(slot[r, c])
                sc = sum(pubM[j, d - 1] * BV[k, d] for d in range(1, 7))
                sv = sum(pubM[j, d - 1] * SURV[k, d] for d in range(1, 7))
                sc -= REPLY_W * sv * threat(r, c, k, exclude=j) * dfac
            else:
                adv = (r - r0) * fwd(X)
                sc = PROGRESS * adv
                ar_ = r + fwd(X)
                if k == E.SCOUT and adv > 0 and 0 <= ar_ < N and not E.IS_LAKE[ar_, c] and owner[ar_, c] == -1:
                    sc += SCOUT_BONUS
                if k == E.MINER and targets and td(r, c) < td(r0, c0):
                    if any(g.alive[X, b] and b != s and max(abs(r - posX[b, 0]), abs(c - posX[b, 1])) <= 2 and
                           Zrow[b] in (E.GENERAL, E.MARSHAL) for b in range(NS)):
                        sc += MINER_BONUS
                sc -= REPLY_W * threat(r, c, k) * dfac
            if flag:
                f = flag[0]
                fr_, fc_ = int(posX[f, 0]), int(posX[f, 1])
                if abs(r0 - fr_) + abs(c0 - fc_) == 1 and abs(r - fr_) + abs(c - fc_) != 1:
                    backed = any(g.alive[X, b] and b != s and abs(int(posX[b, 0]) - fr_) + abs(int(posX[b, 1]) - fc_) == 1
                                 and Zrow[b] in (E.BOMB, E.MARSHAL) for b in range(NS))
                    if not backed:
                        sc -= FLAG_PEN
            out[(s, r, c)] = sc
    return out


# ------------------------------------------------------------------------------------------------ play and records
def outcome_code(prev: E.State, action) -> Tuple[int, int, int]:
    X = prev.turn % 2
    s, r, c = action
    owner, slot = E.board_maps(prev)
    att = int(prev.piece_type[X, s])
    r0, c0 = int(prev.pos[X, s, 0]), int(prev.pos[X, s, 1])
    lng = abs(r - r0) + abs(c - c0) > 1
    if owner[r, c] == -1:
        return 0, (att if lng else 0), 0
    d = int(prev.piece_type[1 - X, int(slot[r, c])])
    o, fl = E.resolve_battle(att, d)
    if fl:
        return 4, att, d
    return {1: 1, -1: 2, 0: 3}[o], att, d


def play_game(red: int, blue: int, taus, rng):
    st = E.new_game(red, blue)
    hist = []
    while st.winner is None:
        acts, p = policy_probs(st, taus[st.turn % 2])
        a = acts[int(rng.choice(len(acts), p=p))]
        hist.append((st, a))
        st = E.step(st, a)
    return hist, st


def replay(red: int, blue: int, actions):
    st = E.new_game(int(red), int(blue))
    hist = []
    for a in actions:
        a = tuple(int(x) for x in a)
        hist.append((st, a))
        st = E.step(st, a)
    return hist, st


def result_for(final: E.State, P: int) -> int:
    """0 win, 1 draw, 2 loss for player P."""
    if final.winner == -1:
        return 1
    return 0 if final.winner == P else 2


def encode_root(st: E.State) -> Dict[str, np.ndarray]:
    P = st.turn % 2
    O = 1 - P
    sq_content = np.zeros(N * N, np.int8)
    for q in E.LAKE_SQ:
        sq_content[q] = 14                      # lakes are invariant under the 180-degree rotation
    sq_ownflag = np.zeros(N * N, np.int8)
    sq_slot = np.zeros(N * N, np.int8)
    slot_loc = np.full(NS, N * N, np.int8)
    slot_flags = np.zeros(NS, np.int8)
    slot_mcount = np.zeros(NS, np.int8)
    slot_pubrev = np.zeros(NS, np.int8)
    slot_true = np.zeros(NS, np.int8)
    own_true = np.zeros(NS, np.int8)
    for s in range(NS):
        own_true[s] = st.piece_type[P, s]
        if st.alive[P, s]:
            q = canon_sq(int(st.pos[P, s, 0]), int(st.pos[P, s, 1]), P)
            sq_content[q] = st.piece_type[P, s]
            sq_ownflag[q] = 1 + int(st.has_moved[P, s]) + 2 * int(st.revealed[P, s])
    for s in range(NS):
        slot_true[s] = st.piece_type[O, s]
        slot_pubrev[s] = int(st.revealed[O, s])
        slot_flags[s] = int(st.has_moved[O, s]) + int(st.moved_long[O, s])
        slot_mcount[s] = min(int(st.mcount[O, s]), 3)
        if st.alive[O, s]:
            q = canon_sq(int(st.pos[O, s, 0]), int(st.pos[O, s, 1]), P)
            slot_loc[s] = q
            sq_slot[q] = s + 1
            sq_content[q] = 7 + (int(st.piece_type[O, s]) if st.revealed[O, s] else 0)
    return dict(sq_content=sq_content, sq_ownflag=sq_ownflag, sq_slot=sq_slot, slot_loc=slot_loc, slot_flags=slot_flags,
                slot_mcount=slot_mcount, slot_pubrev=slot_pubrev, slot_true=slot_true, own_true=own_true,
                turn=np.int8(st.turn))


def encode_ply(prev: E.State, action, P: int) -> Tuple[int, int, int, int, int, int]:
    X = prev.turn % 2
    s, r, c = action
    fr = canon_sq(int(prev.pos[X, s, 0]), int(prev.pos[X, s, 1]), P)
    to = canon_sq(r, c, P)
    oc, at, dt = outcome_code(prev, action)
    return (0 if X == P else 1), fr, to, oc, at, dt


def legal_canon(st: E.State) -> np.ndarray:
    P = st.turn % 2
    lg = np.full((LMAX, 2), -1, np.int8)
    for i, (s, r, c) in enumerate(E.legal_actions(st)[:LMAX]):
        lg[i] = (canon_sq(int(st.pos[P, s, 0]), int(st.pos[P, s, 1]), P), canon_sq(r, c, P))
    return lg


# ------------------------------------------------------------------------------------------------ exact posteriors
def opponent_loglik_cum(hist, P: int) -> List[np.ndarray]:
    """For observer P: cumulative log-likelihood (10080, 3) of the opponent's moves before each time t (list over t =
    0..T), log sum_j log pi_tau(a_j | s_j, Z) over opponent plies j < t (all 10080 hypothetical setups)."""
    O = 1 - P
    T = len(hist)
    cur = np.zeros((len(E.ALL_SETUPS), len(TAUS)))
    out = [cur.copy()]
    for j in range(T):
        st, a = hist[j]
        if st.turn % 2 == O:
            ctx = PolicyCtx(st, O)
            cur = cur + ctx.loglik_table(a, E.ALL_SETUPS)
        out.append(cur.copy())
    return out


def canon_marg(m: np.ndarray, P: int) -> np.ndarray:
    return m          # opponent slot s is canonical slot s for both colours (BLUE_SQ = rotation of RED_SQ)


def posterior_at(st_t: E.State, ll_t: np.ndarray):
    """Exact posterior over the opponent's setup at state st_t (observer = player to move) from the cumulative
    log-likelihood ll_t (10080, 3). Returns (P*_full marginals (8,6), joint weights (10080,3), P*_c marginals (8,6))."""
    P = st_t.turn % 2
    O = 1 - P
    mask = E.constraint_mask(st_t, O)
    ll = ll_t.copy()
    ll[~mask] = -np.inf
    m = ll.max()
    w = np.exp(ll - m)
    w /= w.sum()
    marg = E.marginals(w.sum(1))
    mc = E.marginals(mask.astype(np.float64))
    return marg, w, mc


def posterior_bruteforce(hist, t: int, red: int, blue: int):
    """Independent check: replay the public history with every hypothetical opponent setup substituted (env6.step), keep
    setups whose public observations match ply by ply, multiply the opponent's move probabilities computed with the
    plain-loop scalar policy. Returns (P*_full, P*_c) marginals (8, 6)."""
    st = hist[t][0]
    P = st.turn % 2
    O = 1 - P
    w = np.zeros((len(E.ALL_SETUPS), len(TAUS)))
    wc = np.zeros(len(E.ALL_SETUPS))
    pub = lambda g: (g.alive.tobytes(), g.pos.tobytes(), g.revealed.tobytes(), g.has_moved.tobytes(),
                     np.where(g.revealed, g.piece_type, 0).tobytes())
    real_pub = [pub(h[0]) for h in hist[:t + 1]]
    for z in range(len(E.ALL_SETUPS)):
        idx = [red, blue]
        idx[O] = z
        g = E.new_game(idx[0], idx[1])
        g.piece_type[P] = hist[0][0].piece_type[P]
        ok = True
        lik = np.zeros(len(TAUS))
        for tp in range(t):
            a = hist[tp][1]
            X = g.turn % 2
            if X == O:
                sc = scalar_scores(g, O, g.piece_type[O])
                if a not in sc:
                    ok = False
                    break
                keys = list(sc.keys())
                v = np.array([sc[kk] for kk in keys])
                ia = keys.index(a)
                for ti, tau in enumerate(TAUS):
                    zz = v / tau
                    lik[ti] += zz[ia] - (zz.max() + math.log(np.exp(zz - zz.max()).sum()))
            else:
                if a not in E.legal_actions(g):
                    ok = False
                    break
            g = E.step(g, a)
            if pub(g) != real_pub[tp + 1]:
                ok = False
                break
        if ok:
            w[z] = np.exp(lik)
            wc[z] = 1.0
    return E.marginals(w.sum(1)), E.marginals(wc)


# ------------------------------------------------------------------------------------------------ simulator Q*
_pm_cache: Dict[bytes, np.ndarray] = {}


def _pub_key(st: E.State, Y: int) -> bytes:
    return st.revealed[Y].tobytes() + st.moved_long[Y].tobytes() + st.has_moved[Y].tobytes() + \
        np.where(st.revealed[Y], st.piece_type[Y], 0).tobytes()


def fast_policy(st: E.State, tau: float):
    X = st.turn % 2
    key = _pub_key(st, 1 - X)
    pm = _pm_cache.get(key)
    if pm is None:
        pm = E.constraint_marginals(st, 1 - X)
        if len(_pm_cache) > 20000:
            _pm_cache.clear()
        _pm_cache[key] = pm
    return PolicyCtx(st, X, pm).probs_true(tau)


def q_star(st: E.State, joint_w: np.ndarray, actions, n_mc: int, rng: np.random.Generator, horizon_miner: int = 4):
    """Simulator MC values of each root action for the player P to move at st: (Z, tau_opp) ~ exact joint posterior,
    tau_self ~ U{0.5,1,2}, behaviour policy to the end. Returns (n_actions, 3) win/draw/loss frequencies and
    (n_actions,) the mean Miner utility over the first `horizon_miner` plies (+1 enemy Miner captured, -1 own Miner
    lost; a tie of two Miners counts both)."""
    P = st.turn % 2
    O = 1 - P
    flat = joint_w.ravel()
    q = np.zeros((len(actions), 3))
    mu = np.zeros(len(actions))
    for ai, a in enumerate(actions):
        draws = rng.choice(flat.size, size=n_mc, p=flat)
        for d in draws:
            z, ti = divmod(int(d), len(TAUS))
            g = st.clone()
            g.piece_type = g.piece_type.copy()
            g.piece_type[O] = E.ALL_SETUPS[z]
            taus = [0.0, 0.0]
            taus[O] = TAUS[ti]
            taus[P] = TAUS[int(rng.integers(len(TAUS)))]
            ply = 0
            act = a
            while True:
                if ply < horizon_miner:
                    oc, at, de = outcome_code(g, act)
                    X = g.turn % 2
                    if oc in (1, 3, 4) and de == E.MINER:
                        mu[ai] += 1.0 if X == P else -1.0
                    if oc in (2, 3) and at == E.MINER:
                        mu[ai] += -1.0 if X == P else 1.0
                g = E.step(g, act)
                ply += 1
                if g.winner is not None:
                    break
                acts, p = fast_policy(g, taus[g.turn % 2])
                act = acts[int(rng.choice(len(acts), p=p))]
            q[ai, result_for(g, P)] += 1
    return q / n_mc, mu / n_mc
