#!/usr/bin/env python3
"""E-MS data generation (CPU, <= 4 worker processes).
Disjoint setup split 540/90/90 of the 720 permutations (one split for both colours); 60k train games on train setups,
5k calibration games on val setups, 3k evaluation games on test setups; softmax-heuristic self-play with a per-player
per-game temperature tau ~ U{0.5, 1, 2} (no MCTS games: their action likelihood is not computable, so the exact
posterior would not be). Writes data/{train,cal,test}.npz (one record per position, observer = player to move, with the
next <= 6 real plies) and data/games_{split}.npz (setups, taus, actions for replay), then the scoring sets:
  data/oracle_d1.npz   2000 test positions t in [4,12]: P*_full (exact Bayes incl. behaviour likelihood) and P*_c
  data/oracle_q.npz    300 test positions t in [2,14]: Q* (simulator MC, 64 per legal action) + P*_full
  data/d2_pos.npz      500 two-flank test positions t in [2,14] (observer has legal moves on both flanks)
  data/verify.json     brute-force re-derivation of P*_full and P*_c on 40 test positions (max abs error)
   python3 generate_data.py [--workers 4] [--scale 1.0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import msgame as G  # noqa: E402
from env import MicroStrategoEnv as E  # noqa: E402

DATA = os.path.join(HERE, "data")
SPLIT_SEED = 20261016
MAXP = 6
FIELDS_ROOT = ("sq_content", "sq_ownflag", "sq_slot", "slot_loc", "slot_flags", "slot_mcount", "slot_pubrev",
               "slot_true", "own_true", "legal")


def setup_split():
    rng = np.random.default_rng(SPLIT_SEED)
    perm = rng.permutation(720)
    return {"train": np.sort(perm[:540]), "val": np.sort(perm[540:630]), "test": np.sort(perm[630:])}


def replay(red, blue, actions):
    st = E.new_game(int(red), int(blue))
    hist = []
    for a in actions:
        hist.append((st, tuple(int(x) for x in a)))
        st = E.step(st, tuple(int(x) for x in a))
    return hist, st


def game_records(hist, final):
    """Per position t: canonical root encoding + next <= 6 plies + result for the player to move."""
    T = len(hist)
    mcount = np.zeros((2, 6), np.int64)
    recs = []
    states = [h[0] for h in hist] + [final]
    for t in range(T):
        st, a = hist[t]
        P = st.turn % 2
        r = G.encode_root(st, mcount)
        ply = np.zeros((MAXP, 6), np.int8)
        n = 0
        for j in range(MAXP):
            if t + j >= T:
                break
            ply[j] = G.encode_ply(states[t + j], hist[t + j][1], states[t + j + 1], P)
            n += 1
        r["ply"] = ply
        r["n_ply"] = np.int8(n)
        r["result"] = np.int8(G.result_for(final, P))
        r["t"] = np.int8(t)
        lg = np.full((14, 2), -1, np.int8)
        for i, (s_, nr, nc) in enumerate(E.legal_actions(st)):
            lg[i] = (G.canon_sq(int(st.pos[P, s_, 0]), int(st.pos[P, s_, 1]), P), G.canon_sq(nr, nc, P))
        r["legal"] = lg
        recs.append(r)
        X = st.turn % 2
        mcount[X, a[0]] += 1
    return recs


def work(args):
    split, setups, g0, g1, base_seed = args
    out = {k: [] for k in FIELDS_ROOT + ("turn", "ply", "n_ply", "result", "t", "game")}
    games = []
    for g in range(g0, g1):
        rng = np.random.default_rng([base_seed, g])
        red, blue = int(rng.choice(setups)), int(rng.choice(setups))
        taus = (G.TAUS[int(rng.integers(3))], G.TAUS[int(rng.integers(3))])
        hist, final = G.play_game(red, blue, taus, rng)
        for r in game_records(hist, final):
            for k in FIELDS_ROOT + ("turn", "ply", "n_ply", "result", "t"):
                out[k].append(r[k])
            out["game"].append(g)
        acts = np.full((24, 3), -1, np.int8)
        for i, (_, a) in enumerate(hist):
            acts[i] = a
        games.append((g, red, blue, taus[0], taus[1], len(hist), final.winner, acts))
    return out, games


def gen_split(split, setups, n_games, base_seed, workers):
    chunks = [(split, setups, i, min(n_games, i + 2000), base_seed) for i in range(0, n_games, 2000)]
    with Pool(workers) as pool:
        res = pool.map(work, chunks)
    rec = {k: np.asarray([x for r in res for x in r[0][k]]) if k not in FIELDS_ROOT + ("ply",) else
           np.stack([x for r in res for x in r[0][k]]) for k in res[0][0]}
    gm = [x for r in res for x in r[1]]
    np.savez_compressed(os.path.join(DATA, f"{split}.npz"), **rec)
    np.savez_compressed(os.path.join(DATA, f"games_{split}.npz"), game=np.array([x[0] for x in gm]),
                        red=np.array([x[1] for x in gm]), blue=np.array([x[2] for x in gm]),
                        tau_red=np.array([x[3] for x in gm]), tau_blue=np.array([x[4] for x in gm]),
                        length=np.array([x[5] for x in gm]), winner=np.array([x[6] for x in gm]),
                        actions=np.stack([x[7] for x in gm]))
    return rec, gm


def oracle_d1_job(args):
    red, blue, acts, t = args
    hist, _ = replay(red, blue, [a for a in acts if a[0] >= 0])
    pf, _, pc = G.posterior_full(hist, t)
    return pf, pc


def oracle_q_job(args):
    red, blue, acts, t, seed = args
    hist, _ = replay(red, blue, [a for a in acts if a[0] >= 0])
    st = hist[t][0]
    legal = E.legal_actions(st)
    pf, w, pc = G.posterior_full(hist, t)
    q = G.q_star(hist, t, red, blue, legal, 64, np.random.default_rng(seed), joint_w=w)
    la = np.full((14, 3), -1, np.int8)
    qq = np.full((14, 3), np.nan)
    flank = np.full(14, -1, np.int8)
    canon = np.full((14, 2), -1, np.int8)
    P = st.turn % 2
    for i, a in enumerate(legal):
        la[i] = a
        qq[i] = q[i]
        flank[i] = G.flank_of(st, a)
        canon[i] = (G.canon_sq(int(st.pos[P, a[0], 0]), int(st.pos[P, a[0], 1]), P), G.canon_sq(a[1], a[2], P))
    return pf, pc, la, qq, flank, canon, len(legal)


def verify_job(args):
    red, blue, acts, t = args
    hist, _ = replay(red, blue, [a for a in acts if a[0] >= 0])
    pf, _, pc = G.posterior_full(hist, t)
    bf, bc = G.posterior_bruteforce(hist, t, red, blue)
    return float(np.abs(pf - bf).max()), float(np.abs(pc - bc).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--skip-games", action="store_true")
    a = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)
    t0 = time.time()
    sp = setup_split()
    np.savez(os.path.join(DATA, "split.npz"), **sp)
    assert not (set(sp["train"]) & set(sp["val"])) and not (set(sp["train"]) & set(sp["test"])) and not (set(sp["val"]) & set(sp["test"]))
    n = {"train": int(60000 * a.scale), "cal": int(5000 * a.scale), "test": int(3000 * a.scale)}
    setups = {"train": sp["train"], "cal": sp["val"], "test": sp["test"]}
    seeds = {"train": 1001, "cal": 1002, "test": 1003}
    meta = {"n_games": n, "split_seed": SPLIT_SEED}
    if not a.skip_games:
        for s in ("train", "cal", "test"):
            rec, gm = gen_split(s, setups[s], n[s], seeds[s], a.workers)
            w = np.array([x[6] for x in gm])
            meta[s] = {"positions": int(len(rec["t"])), "games": len(gm), "mean_len": float(np.mean([x[5] for x in gm])),
                       "red_win": float((w == 0).mean()), "blue_win": float((w == 1).mean()), "draw": float((w == -1).mean())}
            print(s, meta[s], f"{time.time()-t0:.0f}s", flush=True)
    G_t = np.load(os.path.join(DATA, "games_test.npz"))
    rng = np.random.default_rng(77)
    # D0/D1 positions: test games, t in [4,12]
    cand = [(gi, t) for gi in range(len(G_t["game"])) for t in range(4, min(13, int(G_t["length"][gi])))]
    pick = rng.choice(len(cand), size=min(2000, len(cand)), replace=False)
    d1 = [cand[i] for i in pick]
    jobs = [(int(G_t["red"][g]), int(G_t["blue"][g]), G_t["actions"][g], t) for g, t in d1]
    with Pool(a.workers) as pool:
        res = pool.map(oracle_d1_job, jobs, chunksize=20)
    np.savez_compressed(os.path.join(DATA, "oracle_d1.npz"), game=np.array([g for g, _ in d1]), t=np.array([t for _, t in d1]),
                        pstar_full=np.stack([r[0] for r in res]), pstar_c=np.stack([r[1] for r in res]))
    print("oracle_d1", len(d1), f"{time.time()-t0:.0f}s", flush=True)
    # D3/D4 positions with Q*: test games, t in [2,14]
    cand = [(gi, t) for gi in range(len(G_t["game"])) for t in range(2, min(15, int(G_t["length"][gi])))]
    pick = rng.choice(len(cand), size=min(300, len(cand)), replace=False)
    dq = [cand[i] for i in pick]
    jobs = [(int(G_t["red"][g]), int(G_t["blue"][g]), G_t["actions"][g], t, 9000 + i) for i, (g, t) in enumerate(dq)]
    with Pool(a.workers) as pool:
        res = pool.map(oracle_q_job, jobs, chunksize=5)
    np.savez_compressed(os.path.join(DATA, "oracle_q.npz"), game=np.array([g for g, _ in dq]), t=np.array([t for _, t in dq]),
                        pstar_full=np.stack([r[0] for r in res]), pstar_c=np.stack([r[1] for r in res]),
                        legal=np.stack([r[2] for r in res]), qstar=np.stack([r[3] for r in res]),
                        flank=np.stack([r[4] for r in res]), canon=np.stack([r[5] for r in res]),
                        n_legal=np.array([r[6] for r in res]))
    print("oracle_q", len(dq), f"{time.time()-t0:.0f}s", flush=True)
    # D2 two-flank positions
    cand = []
    for gi in range(len(G_t["game"])):
        hist, _ = replay(int(G_t["red"][gi]), int(G_t["blue"][gi]), [x for x in G_t["actions"][gi] if x[0] >= 0])
        for t in range(2, min(15, len(hist))):
            st = hist[t][0]
            fl = {G.flank_of(st, x) for x in E.legal_actions(st)}
            if fl == {0, 1}:
                cand.append((gi, t))
    pick = rng.choice(len(cand), size=min(500, len(cand)), replace=False)
    d2 = [cand[i] for i in pick]
    np.savez(os.path.join(DATA, "d2_pos.npz"), game=np.array([g for g, _ in d2]), t=np.array([t for _, t in d2]),
             n_candidates=len(cand))
    print("d2", len(d2), "of", len(cand), f"{time.time()-t0:.0f}s", flush=True)
    # brute-force verification of the oracle on 40 test positions
    vj = jobs_v = [(int(G_t["red"][g]), int(G_t["blue"][g]), G_t["actions"][g], t) for g, t in d1[:40]]
    with Pool(a.workers) as pool:
        vr = pool.map(verify_job, vj, chunksize=2)
    meta["verify"] = {"n": len(vr), "max_abs_err_full": max(x[0] for x in vr), "max_abs_err_constraint": max(x[1] for x in vr)}
    meta["sec"] = time.time() - t0
    json.dump(meta, open(os.path.join(DATA, "meta.json"), "w"), indent=1)
    print(json.dumps(meta["verify"]), f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
