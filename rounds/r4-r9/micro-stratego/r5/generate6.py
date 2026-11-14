#!/usr/bin/env python3
"""R5-E2 data generation (CPU, <= 6 worker processes, OMP_NUM_THREADS=1).
Disjoint setup split 8064 / 1008 / 1008 of the 10,080 setups (one split for both colours; seed 20261017). Self-play
with the coupled behaviour policy (msgame6), tau ~ U{0.5, 1, 2} per player per game: 60k train games on train setups,
5k calibration games on val setups, 3k test games on test setups. Records: one per position (observer = player to
move) with the canonical root encoding, the next <= 6 real plies, the result, the legal list. Scoring sets (oracle /
simulator; for evaluation and gates only):
  oracle_d0_test.npz   2000 test positions, t in [6, 36]: P*_full (exact Bayes incl. behaviour likelihood), P*_c
  oracle_d0_cal.npz    1000 calibration positions, t in [6, 36]: same (gate 3b and the pre-registered capacity rule)
  oracle_q_test.npz    240 test positions (120 general, t in [4, 36]; 120 Miner-contact): Q* (win/draw/loss) and the
                       4-ply Miner utility per legal action (64 simulator playouts each) + P*_full
  oracle_q_cal.npz     200 calibration positions t in [4, 36]: Q* per legal action (64 playouts) (gate 3b)
  d1_pos.npz           500 two-flank test positions t in [4, 36] (observer has legal moves on both flanks)
  verify.json          brute-force replay re-derivation of P*_full and P*_c on 30 test positions (max abs error)
Bulky outputs go to $R5E2_DATA (default /tmp/r5e2/data).
   OMP_NUM_THREADS=1 python3 generate6.py [--workers 6] [--scale 1.0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import env6 as E  # noqa: E402
import msgame6 as G  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
DATA = os.environ.get("R5E2_DATA", "/tmp/r5e2/data")
SPLIT_SEED = 20261017
MAXP = 6
ROOT_FIELDS = ("sq_content", "sq_ownflag", "sq_slot", "slot_loc", "slot_flags", "slot_mcount", "slot_pubrev",
               "slot_true", "own_true")
T_D0 = (6, 36)
T_Q = (4, 36)


def setup_split():
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(len(E.ALL_SETUPS))
    return {"train": np.sort(perm[:8064]), "val": np.sort(perm[8064:9072]), "test": np.sort(perm[9072:])}


def game_records(hist, final):
    T = len(hist)
    states = [h[0] for h in hist] + [final]
    recs = []
    for t in range(T):
        st, a = hist[t]
        P = st.turn % 2
        r = G.encode_root(st)
        ply = np.zeros((MAXP, 6), np.int8)
        n = 0
        for j in range(MAXP):
            if t + j >= T:
                break
            ply[j] = G.encode_ply(states[t + j], hist[t + j][1], P)
            n += 1
        r["ply"] = ply
        r["n_ply"] = np.int8(n)
        r["result"] = np.int8(G.result_for(final, P))
        r["t"] = np.int8(t)
        r["legal"] = G.legal_canon(st)
        r["n_legal"] = np.int8(min(len(E.legal_actions(st)), G.LMAX))
        recs.append(r)
    return recs


def work(args):
    split, setups, g0, g1, base_seed = args
    keys = ROOT_FIELDS + ("turn", "ply", "n_ply", "result", "t", "legal", "n_legal", "game")
    out = {k: [] for k in keys}
    games = []
    for g in range(g0, g1):
        rng = np.random.default_rng([base_seed, g])
        red, blue = int(rng.choice(setups)), int(rng.choice(setups))
        taus = (G.TAUS[int(rng.integers(3))], G.TAUS[int(rng.integers(3))])
        hist, final = G.play_game(red, blue, taus, rng)
        for r in game_records(hist, final):
            for k in keys[:-1]:
                out[k].append(r[k])
            out["game"].append(g)
        acts = np.full((E.MAX_TURNS, 3), -1, np.int8)
        for i, (_, a) in enumerate(hist):
            acts[i] = a
        games.append((g, red, blue, taus[0], taus[1], len(hist), final.winner, acts))
    return out, games


def gen_split(split, setups, n_games, base_seed, workers):
    step = 500
    chunks = [(split, setups, i, min(n_games, i + step), base_seed) for i in range(0, n_games, step)]
    with Pool(workers) as pool:
        res = pool.map(work, chunks)
    rec = {}
    for k in res[0][0]:
        vals = [x for r in res for x in r[0][k]]
        rec[k] = np.stack(vals) if isinstance(vals[0], np.ndarray) and np.ndim(vals[0]) > 0 else np.asarray(vals)
    gm = [x for r in res for x in r[1]]
    np.savez_compressed(os.path.join(DATA, f"{split}.npz"), **rec)
    np.savez_compressed(os.path.join(DATA, f"games_{split}.npz"), game=np.array([x[0] for x in gm]),
                        red=np.array([x[1] for x in gm]), blue=np.array([x[2] for x in gm]),
                        tau_red=np.array([x[3] for x in gm]), tau_blue=np.array([x[4] for x in gm]),
                        length=np.array([x[5] for x in gm]), winner=np.array([x[6] for x in gm]),
                        actions=np.stack([x[7] for x in gm]))
    return rec, gm


def _hist(Gm, gi):
    acts = [a for a in Gm["actions"][gi] if a[0] >= 0]
    return G.replay(int(Gm["red"][gi]), int(Gm["blue"][gi]), acts)


def post_job(args):
    """Exact posterior at (game, t) using only setups consistent at t (the posterior is zero elsewhere)."""
    Gm_path, gi, t = args
    Gm = np.load(Gm_path)
    hist, _ = _hist(Gm, gi)
    st = hist[t][0]
    P = st.turn % 2
    O = 1 - P
    mask = E.constraint_mask(st, O)
    idx = np.nonzero(mask)[0]
    Zs = E.ALL_SETUPS[idx]
    ll = np.zeros((len(idx), len(G.TAUS)))
    for j in range(t):
        s_j, a_j = hist[j]
        if s_j.turn % 2 == O:
            ll += G.PolicyCtx(s_j, O).loglik_table(a_j, Zs)
    m = ll.max()
    w = np.exp(ll - m)
    w /= w.sum()
    full = np.zeros((len(E.ALL_SETUPS), len(G.TAUS)))
    full[idx] = w
    pf = E.marginals(full.sum(1))
    pc = E.marginals(mask.astype(np.float64))
    return pf, pc, full


def q_job(args):
    Gm_path, gi, t, seed, n_mc = args
    Gm = np.load(Gm_path)
    hist, _ = _hist(Gm, gi)
    st = hist[t][0]
    pf, pc, w = post_job((Gm_path, gi, t))
    legal = E.legal_actions(st)[:G.LMAX]
    q, mu = G.q_star(st, w, legal, n_mc, np.random.default_rng(seed))
    P = st.turn % 2
    la = np.full((G.LMAX, 3), -1, np.int8)
    canon = np.full((G.LMAX, 2), -1, np.int8)
    qq = np.full((G.LMAX, 3), np.nan)
    mm = np.full(G.LMAX, np.nan)
    for i, a in enumerate(legal):
        la[i] = a
        canon[i] = (G.canon_sq(int(st.pos[P, a[0], 0]), int(st.pos[P, a[0], 1]), P), G.canon_sq(a[1], a[2], P))
        qq[i] = q[i]
        mm[i] = mu[i]
    return pf, pc, la, canon, qq, mm, len(legal)


def verify_job(args):
    Gm_path, gi, t = args
    Gm = np.load(Gm_path)
    hist, _ = _hist(Gm, gi)
    pf, pc, _ = post_job((Gm_path, gi, t))
    bf, bc = G.posterior_bruteforce(hist, t, int(Gm["red"][gi]), int(Gm["blue"][gi]))
    return float(np.abs(pf - bf).max()), float(np.abs(pc - bc).max())


def miner_contact(st):
    """Public-information filter: an own Miner (known to the mover) alive within Chebyshev distance 2 of an opponent
    piece, or an unrevealed / revealed opponent piece next to the own Miner's possible targets."""
    P = st.turn % 2
    O = 1 - P
    for s in range(E.NS):
        if st.alive[P, s] and st.piece_type[P, s] == E.MINER:
            r0, c0 = st.pos[P, s]
            for j in range(E.NS):
                if st.alive[O, j] and max(abs(int(st.pos[O, j, 0]) - r0), abs(int(st.pos[O, j, 1]) - c0)) <= 2:
                    return True
    return False


def pick_positions(Gm, rng, n, trange, filt=None, one_per_game=True):
    order = rng.permutation(len(Gm["game"]))
    out = []
    for gi in order:
        L = int(Gm["length"][gi])
        ts = [t for t in range(trange[0], min(trange[1] + 1, L))]
        if not ts:
            continue
        if filt is not None:
            hist, _ = _hist(Gm, gi)
            ts = [t for t in ts if filt(hist[t][0])]
            if not ts:
                continue
        out.append((int(gi), int(rng.choice(ts))))
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--skip-games", action="store_true")
    ap.add_argument("--n-mc", type=int, default=64)
    a = ap.parse_args()
    assert a.workers <= 6
    os.makedirs(DATA, exist_ok=True)
    t0 = time.time()
    sp = setup_split()
    np.savez(os.path.join(DATA, "split.npz"), **sp)
    assert not (set(sp["train"]) & set(sp["val"])) and not (set(sp["train"]) & set(sp["test"])) and not (set(sp["val"]) & set(sp["test"]))
    n = {"train": int(60000 * a.scale), "cal": int(5000 * a.scale), "test": int(3000 * a.scale)}
    setups = {"train": sp["train"], "cal": sp["val"], "test": sp["test"]}
    seeds = {"train": 2001, "cal": 2002, "test": 2003}
    meta = {"n_games": n, "split_seed": SPLIT_SEED, "max_turns": E.MAX_TURNS, "taus": G.TAUS}
    if not a.skip_games:
        for s in ("train", "cal", "test"):
            rec, gm = gen_split(s, setups[s], n[s], seeds[s], a.workers)
            w = np.array([x[6] for x in gm])
            meta[s] = {"positions": int(len(rec["t"])), "games": len(gm), "mean_len": float(np.mean([x[5] for x in gm])),
                       "red_win": float((w == 0).mean()), "blue_win": float((w == 1).mean()), "draw": float((w == -1).mean())}
            print(s, meta[s], f"{time.time()-t0:.0f}s", flush=True)
    rng = np.random.default_rng(77)
    gp = {s: os.path.join(DATA, f"games_{s}.npz") for s in ("test", "cal")}
    Gt, Gc = np.load(gp["test"]), np.load(gp["cal"])
    with Pool(a.workers) as pool:
        for tag, Gm, path, nn in (("test", Gt, gp["test"], int(2000 * min(1, a.scale * 4))),
                                  ("cal", Gc, gp["cal"], int(1000 * min(1, a.scale * 4)))):
            pos = pick_positions(Gm, rng, nn, T_D0)
            res = pool.map(post_job, [(path, g, t) for g, t in pos], chunksize=8)
            np.savez_compressed(os.path.join(DATA, f"oracle_d0_{tag}.npz"), game=np.array([g for g, _ in pos]),
                                t=np.array([t for _, t in pos]), pstar_full=np.stack([r[0] for r in res]),
                                pstar_c=np.stack([r[1] for r in res]))
            print("oracle_d0", tag, len(pos), f"{time.time()-t0:.0f}s", flush=True)
        # Q* sets
        nq = max(8, int(120 * min(1, a.scale * 4)))
        gen_pos = pick_positions(Gt, rng, nq, T_Q)
        used = {g for g, _ in gen_pos}
        mc_pos = [x for x in pick_positions(Gt, rng, 3 * nq, T_Q, filt=miner_contact) if x[0] not in used][:nq]
        qpos = gen_pos + mc_pos
        kind = np.array([0] * len(gen_pos) + [1] * len(mc_pos))
        res = pool.map(q_job, [(gp["test"], g, t, 9000 + i, a.n_mc) for i, (g, t) in enumerate(qpos)], chunksize=1)
        np.savez_compressed(os.path.join(DATA, "oracle_q_test.npz"), game=np.array([g for g, _ in qpos]),
                            t=np.array([t for _, t in qpos]), kind=kind, pstar_full=np.stack([r[0] for r in res]),
                            pstar_c=np.stack([r[1] for r in res]), legal=np.stack([r[2] for r in res]),
                            canon=np.stack([r[3] for r in res]), qstar=np.stack([r[4] for r in res]),
                            miner=np.stack([r[5] for r in res]), n_legal=np.array([r[6] for r in res]))
        print("oracle_q test", len(qpos), f"{time.time()-t0:.0f}s", flush=True)
        cpos = pick_positions(Gc, rng, max(8, int(200 * min(1, a.scale * 4))), T_Q)
        res = pool.map(q_job, [(gp["cal"], g, t, 19000 + i, a.n_mc) for i, (g, t) in enumerate(cpos)], chunksize=1)
        np.savez_compressed(os.path.join(DATA, "oracle_q_cal.npz"), game=np.array([g for g, _ in cpos]),
                            t=np.array([t for _, t in cpos]), pstar_full=np.stack([r[0] for r in res]),
                            legal=np.stack([r[2] for r in res]), canon=np.stack([r[3] for r in res]),
                            qstar=np.stack([r[4] for r in res]), miner=np.stack([r[5] for r in res]),
                            n_legal=np.array([r[6] for r in res]))
        print("oracle_q cal", len(cpos), f"{time.time()-t0:.0f}s", flush=True)
        # D1 two-flank positions
        cand = []
        for gi in range(len(Gt["game"])):
            hist, _ = _hist(Gt, gi)
            for t in range(T_Q[0], min(T_Q[1] + 1, len(hist))):
                st = hist[t][0]
                P = st.turn % 2
                fl = {G.flank_of_sq(G.canon_sq(int(st.pos[P, s, 0]), int(st.pos[P, s, 1]), P)) for (s, _, _) in E.legal_actions(st)}
                if fl == {0, 1}:
                    cand.append((gi, t))
        pick = rng.choice(len(cand), size=min(500, len(cand)), replace=False)
        d1 = [cand[i] for i in pick]
        np.savez(os.path.join(DATA, "d1_pos.npz"), game=np.array([g for g, _ in d1]), t=np.array([t for _, t in d1]),
                 n_candidates=len(cand))
        print("d1", len(d1), "of", len(cand), f"{time.time()-t0:.0f}s", flush=True)
        vpos = pick_positions(Gt, rng, max(4, int(30 * min(1, a.scale * 4))), (6, 16))
        vr = pool.map(verify_job, [(gp["test"], g, t) for g, t in vpos], chunksize=1)
    meta["verify"] = {"n": len(vr), "positions": vpos, "max_abs_err_full": max(x[0] for x in vr),
                      "max_abs_err_constraint": max(x[1] for x in vr)}
    meta["sec"] = time.time() - t0
    json.dump(meta, open(os.path.join(DATA, "meta.json"), "w"), indent=1)
    print(json.dumps(meta["verify"]), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
