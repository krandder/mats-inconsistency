#!/usr/bin/env python3
"""R7-E1 audit, check 2: my own leak test of e1v4.puct_search7 with the REAL theta_real (ckpt/s101, CPU) on 8 test-stream
boards. The trained pools and theta_trust were not saved, so the K3 key head, attn_L3 and theta_trust are random (fixed seed);
the test concerns information flow, not quality.
For each selection mode (power, greedy), read setting (mixed, clean) and wave w = 1..5: replace the real readings of every cell
that waves 1..w-1 did not read (anchors excluded) by (a) their flips and (b) fresh random bits; waves 1..w must be unchanged.
Positive control: flipping the cells wave 1 read changes some later wave.
Also: the stream-21 (calibration) vs stream-22 (test) board overlap.
Usage: PYTHONPATH=/tmp/r4e3a/pylib OMP_NUM_THREADS=2 python3 flip_test.py > flip-output.txt"""
import json
import sys
import time

import numpy as np
import torch

torch.set_num_threads(2)
B = "/home/agent/workspaces/eurisko-study"
R7 = B + "/li-net/r7e1"
for p in (B + "/li-net/r5e1", B + "/li-net/r6e1", R7):
    sys.path.insert(0, p)
import partA2 as A2  # noqa: E402
import wm  # noqa: E402
import e1v3 as E3  # noqa: E402
import e1v4 as E  # noqa: E402
from types import SimpleNamespace  # noqa: E402

print("run at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
model, meta = E.load_wm_npz(f"{R7}/ckpt/s101/theta_real_v2.npz", "cpu")
for p_ in model.parameters():
    p_.requires_grad_(False)
full = A2.boards2_t(A2.STREAM_TEST2, 500, "cpu")
nb = 8
Bd = {k: v[:nb].clone() for k, v in full.items()}
anchors = A2.cells_mask(Bd["anchors"], "cpu").bool()
ev_root = A2.ev_from(Bd["s"], anchors)
l0 = E.RO.forward_chunks(model, ev_root, Bd["note"])
torch.manual_seed(1234)
keyn = E3.KeyNet("K3", 128).eval()
attn = wm.AttnL3().eval()
with torch.no_grad():
    attn.out.weight.normal_(0, 0.1)
trust = wm.TrustNet(128 + 3).eval()
P = SimpleNamespace(keys={"K3": keyn}, attn={"K3": attn})
cfg = E.Cfg7(device="cpu", chunk=4096)
print("cfg lam_U", cfg.lam_U, "alpha_nov", cfg.alpha_nov, "power", cfg.power, "wave_reads", cfg.wave_reads, "waves", cfg.waves)


def run(Bx, mode, st, waves, seed=11):
    evx = A2.ev_from(Bx["s"], A2.cells_mask(Bx["anchors"], "cpu"))
    assert torch.equal(evx, ev_root)          # the root evidence (anchors + note) is unchanged
    _, picks = E.puct_search7(model, trust, P, Bx, evx, l0, True, torch.Generator().manual_seed(seed), cfg, mode, st, waves)
    return picks


out = {"checks": 0, "failures": [], "positive_control_changed": {}, "distinct_within_wave": True}
for mode in ("power", "greedy"):
    for st in ("mixed", "clean"):
        base = run(Bd, mode, st, 5)
        out["distinct_within_wave"] &= all(len(set(r.tolist())) == 6 for p_ in base for r in p_)
        for w in range(1, 6):
            read = E3.coverage(torch.cat(base[:w - 1], 1), "cpu").bool() if w > 1 else torch.zeros(nb, 36, dtype=torch.bool)
            unread = ~read & ~anchors
            g = torch.Generator().manual_seed(100 + w)
            for kind, newS in (("flip", Bd["s"] ^ unread),
                               ("random", torch.where(unread, torch.rand(nb, 36, generator=g) < 0.5, Bd["s"]))):
                B2 = dict(Bd)
                B2["s"] = newS
                p2 = run(B2, mode, st, w)
                same = all(torch.equal(base[i], p2[i]) for i in range(w))
                out["checks"] += 1
                if not same:
                    out["failures"].append((mode, st, w, kind))
        # positive control: change the readings wave 1 actually read
        r1 = E3.coverage(base[0], "cpu").bool() & ~anchors
        B3 = dict(Bd)
        B3["s"] = Bd["s"] ^ r1
        p3 = run(B3, mode, st, 5)
        out["positive_control_changed"][f"{mode}/{st}"] = bool(torch.equal(base[0], p3[0]) and any(not torch.equal(base[i], p3[i]) for i in range(1, 5)))
print(json.dumps(out))

# ---- calibration (stream 21) vs test (stream 22) boards
C = A2.make_boards2(A2.STREAM_CAL2, 4000)
T = A2.make_boards2(A2.STREAM_TEST2, 500)
key = lambda b, i: (b["occ"][i].tobytes(), b["s"][i].tobytes(), int(b["note"][i]))
ck = {key(C, i) for i in range(4000)}
dup = sum(key(T, i) in ck for i in range(500))
occ_c = {C["occ"][i].tobytes() for i in range(4000)}
shared_occ = sum(T["occ"][i].tobytes() in occ_c for i in range(500))
ev_c = {(C["s"][i].tobytes(), int(C["note"][i])) for i in range(4000)}
shared_ev = sum((T["s"][i].tobytes(), int(T["note"][i])) in ev_c for i in range(500))
print(json.dumps({"stream_ids": [A2.STREAM_CAL2, A2.STREAM_TEST2, A2.STREAM_VAL2, A2.STREAM_MON2],
                  "test_boards_identical_to_a_calibration_board(occ,s,note)": dup,
                  "test_boards_whose_36_readings+note_appear_in_calibration": shared_ev,
                  "test_boards_whose_ship_placement_appears_in_calibration (expected by chance ~ 500*(1-exp(-4000/49504)))": shared_occ,
                  "expected_shared_placements": round(500 * (1 - float(np.exp(-4000 / 49504))), 1)}))
