#!/usr/bin/env python3
"""R6-E1 audit: the scoring oracle and the board streams, re-implemented by the auditor.

1. Enumerate the distinct occupancy patterns of ships 4/3/2 on 6x6 (own bitmask enumeration; expect 49,504).
2. Regenerate the calibration stream (21, 4,000 boards) and the test stream (22, 500 boards) with the registered generative
   process (numpy default_rng([stream, 7702]); the call order is the one documented for partA2.make_boards2) and check that the
   test boards' occupancy and anchors equal raw.json's for every seed (this validates the regeneration).
3. Exact Bayes on all 36 readings + the note, own implementation (explicit likelihoods, float64), compared with raw.json pstar.
4. Same boards as R5-E1 v2? Compare with li-net/r5e1/results/v2/s101/partA_raw.json (occ, anchors, note, pstar).
5. Calibration vs test: identical boards and shared occupancy patterns (chance level printed).
Run: OMP_NUM_THREADS=2 PYTHONPATH=/tmp/r4e3a/pylib python3 streams_bayes.py
"""
import json
import os

import numpy as np

R6 = "/home/agent/workspaces/eurisko-study/li-net/r6e1"
R5V2 = "/home/agent/workspaces/eurisko-study/li-net/r5e1/results/v2/s101/partA_raw.json"
EPS, NOTE_REL = 0.15, 0.30


def placements():
    def ship_masks(L):
        out = set()
        for r in range(6):
            for c in range(6):
                if c + L <= 6:
                    out.add(sum(1 << (6 * r + c + k) for k in range(L)))
                if r + L <= 6:
                    out.add(sum(1 << (6 * (r + k) + c) for k in range(L)))
        return sorted(out)
    pats = set()
    for a in ship_masks(4):
        for b in ship_masks(3):
            if a & b:
                continue
            for c in ship_masks(2):
                if (a | b) & c:
                    continue
                pats.add(a | b | c)
    cells = sorted(tuple(i for i in range(36) if m >> i & 1) for m in pats)   # sorted by sorted cell list
    P = np.zeros((len(cells), 36), dtype=bool)
    for i, cs in enumerate(cells):
        P[i, list(cs)] = True
    return P


def boards(P, stream, n):
    rng = np.random.default_rng([stream, 7702])
    occ = P[rng.integers(len(P), size=n)].copy()
    s = occ ^ (rng.random((n, 36)) < EPS)
    true_note = rng.random(n) < NOTE_REL
    sc = rng.random((n, 36))
    note = np.where(true_note, np.where(occ, sc, -1.0).argmax(1), np.where(~occ, sc, -1.0).argmax(1))
    anchors = rng.random((n, 36)).argsort(1)[:, :2]
    std6 = rng.random((n, 36)).argsort(1)[:, :6]
    return {"occ": occ, "s": s, "note": note, "anchors": anchors, "std6": std6}


def exact_bayes_full(P, s, note, chunk=50):
    """P*(ship at cell | all 36 readings, note), uniform prior over the distinct patterns, float64."""
    Pf = P.astype(np.float64)
    nship = P.sum(1).astype(np.float64)                                   # 9 for every pattern
    out = []
    for a in range(0, s.shape[0], chunk):
        sb = s[a:a + chunk].astype(np.float64)                             # [b, 36]
        agree = sb @ Pf.T + (1 - sb) @ (1 - Pf).T                          # cells where reading == occupancy
        ll = agree * np.log(1 - EPS) + (36 - agree) * np.log(EPS)
        on = Pf[:, note[a:a + chunk]].T                                    # [b, Npl] 1 if the note cell is a ship cell
        ll = ll + np.where(on > 0, np.log(NOTE_REL / nship)[None], np.log((1 - NOTE_REL) / (36 - nship))[None])
        ll -= ll.max(1, keepdims=True)
        w = np.exp(ll)
        w /= w.sum(1, keepdims=True)
        out.append(w @ Pf)
    return np.concatenate(out)


def main():
    P = placements()
    print(f"distinct occupancy patterns: {len(P)} (expected 49,504); ship cells per pattern {set(P.sum(1).tolist())}")
    T = boards(P, 22, 500)
    C = boards(P, 21, 4000)
    for s in (101, 102, 103):
        raw = json.load(open(os.path.join(R6, "results", "main", f"s{s}", "raw.json")))
        occ_ok = np.array_equal(np.asarray(raw["occ"]).astype(bool), T["occ"])
        anc_ok = np.array_equal(np.asarray(raw["anchors"]), T["anchors"])
        if s == 101:
            pb = exact_bayes_full(P, T["s"], T["note"])
            pstar101 = np.asarray(raw["pstar"])
        d = np.abs(np.asarray(raw["pstar"]) - pb)
        print(f"s{s}: test occ == regenerated stream 22: {occ_ok}; anchors equal: {anc_ok}; "
              f"max |raw pstar - auditor exact Bayes| = {d.max():.2e} (mean {d.mean():.2e}); "
              f"pstar identical to s101's: {np.array_equal(np.asarray(raw['pstar']), pstar101)}")
    r5 = json.load(open(R5V2))
    print(f"R5-E1 v2 partA_raw.json: occ equal {np.array_equal(np.asarray(r5['occ']).astype(bool), T['occ'])}; anchors equal "
          f"{np.array_equal(np.asarray(r5['anchors']), T['anchors'])}; note equal {np.array_equal(np.asarray(r5['note']), T['note'])}; "
          f"max |pstar_R5v2 - pstar_R6| = {np.abs(np.asarray(r5['pstar']) - pstar101).max():.2e}")
    key = lambda b, i: (b["occ"][i].tobytes(), b["s"][i].tobytes(), int(b["note"][i]))
    cal = {key(C, i) for i in range(4000)}
    same_board = sum(key(T, i) in cal for i in range(500))
    calpat = {C["occ"][i].tobytes() for i in range(4000)}
    shared_pat = sum(T["occ"][i].tobytes() in calpat for i in range(500))
    exp_shared = 500 * (1 - (1 - 1 / len(P)) ** 4000)
    print(f"calibration (21) vs test (22): identical boards (occupancy + 36 readings + note) {same_board} of 500; test boards whose "
          f"occupancy pattern also occurs among the 4,000 calibration boards {shared_pat} (chance expectation {exp_shared:.1f})")
    same_s = sum(any(np.array_equal(T['s'][i], C['s'][j]) for j in np.nonzero((C['occ'] == T['occ'][i]).all(1))[0]) for i in range(500))
    print(f"test boards with the same occupancy AND the same 36-reading sonar field as some calibration board: {same_s}")


if __name__ == "__main__":
    main()
