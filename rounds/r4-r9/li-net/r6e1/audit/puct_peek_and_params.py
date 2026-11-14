#!/usr/bin/env python3
"""R6-E1 audit: (a) parameter counts at the run's sizes, analytic and by instantiation; (b) behavioural no-peeking test of
e1v3.puct_search on CPU with a tiny random world model.

(b) For each wave w = 1..4: flip the real sonar readings of every cell that no read of waves < w covered (anchors excluded,
since they are root evidence) and rerun with the same generator seed. If PUCT never reads a board's real readings before
choosing a read, waves 1..w are identical. Positive control: flipping the readings INSIDE wave-1's patches must change later
waves on some boards (the test can see a dependence when one exists). Also: replacing the true occupancy must change nothing.
Uses a hash-verified copy of the code (scratch/r6e1-audit/codecopy, see codecopy/copy.sha256); writes nothing elsewhere.
Run: OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/tmp/r4e3a/pylib python3 puct_peek_and_params.py
"""
import os
import sys

C = "/home/agent/workspaces/fable-strategy/scratch/r6e1-audit/codecopy"
os.environ["R5E1_DIR"] = C
os.environ["SMALLNET_DIR"] = C + "/smallnet"
sys.path.insert(0, C)
import torch  # noqa: E402

torch.set_num_threads(2)
import e1v3 as E  # noqa: E402
import partA2 as A2  # noqa: E402
import rollouts as RO  # noqa: E402
import wm  # noqa: E402


def lin(i, o, bias=True):
    return i * o + (o if bias else 0)


def analytic():
    d, k = 128, 16
    key = {"K1": lin(d, k), "K2": lin(d, k), "K3": lin(36, k), "K4": lin(d + 36, k)}
    mha = 3 * 64 * 64 + 3 * 64 + lin(64, 64)
    attn = lin(36, 16) + lin(16 + 16 + 2, 64) + 3 * mha + 3 * 128 + 3 * (128 + lin(64, 128) + lin(128, 64)) + lin(64, 36)
    looped = 4 + 4 * 36 + 1
    block = 2 * 2 * d + lin(d, 3 * d) + lin(d, d) + lin(d, 384) + lin(384, d)
    theta = 36 * d + 3 * d + 3 * d + 2 * d + d + 4 * 4 * (121 + 3) + 4 * block + 2 * d + lin(d, 1)
    return {**{f"attn_{v}": attn + key[v] for v in key}, "looped": looped + key["K4"], "theta_real": theta}


def instantiated():
    out = {f"attn_{v}": E.nparams(E.KeyNet(v, 128), wm.AttnL3()) for v in E.VARIANTS}
    out["looped"] = E.nparams(E.KeyNet("K4", 128), E.LoopedLinT4(4, 2.0, 1.0 / 24))
    out["theta_real"] = wm.WorldModel(128, 4, 4, 384).n_params()
    return out


def peek_test(nw, seed=123, n=24):
    torch.manual_seed(0)
    d = 32
    model = wm.WorldModel(d=d, layers=2, heads=2, ff=64).eval()
    trust = wm.TrustNet(d + 3).eval()
    keys = {v: E.KeyNet(v, d).eval() for v in E.VARIANTS}
    attn = {v: wm.AttnL3().eval() for v in E.VARIANTS}
    P = E.Pools6(keys, attn, E.KeyNet("K4", d).eval(), E.LoopedLinT4().eval(), 0.1, 0.5, 0.2)
    for m in [model, trust, *keys.values(), *attn.values()]:
        for p in m.parameters():
            p.requires_grad_(False)
    cfg = E.Cfg6(d=d, device="cpu")
    B = A2.boards2_t(41, n, "cpu")
    anchors = A2.cells_mask(B["anchors"], "cpu")
    ev_root = A2.ev_from(B["s"], anchors)
    l0 = RO.forward_chunks(model, ev_root, B["note"])
    pm = RO.geo("cpu")["pmask"].bool()

    def run(Bx):
        gen = torch.Generator().manual_seed(seed)
        evr = A2.ev_from(Bx["s"], anchors)
        assert torch.equal(evr, ev_root)
        _, waves = E.puct_search(model, trust, P, Bx, evr, l0, nw, gen, cfg)
        return torch.stack(waves, 1)                                    # [n, 4, 6]

    base = run(B)
    res = {}
    for w in range(4):                                                  # flip everything unread before wave w+1
        read = pm[base[:, :w].reshape(n, -1)].any(1) if w > 0 else torch.zeros(n, 36, dtype=torch.bool)
        flip = ~read & ~anchors
        B2 = dict(B)
        B2["s"] = B["s"] ^ flip
        alt = run(B2)
        res[f"wave{w + 1}_identical_after_flipping_unread_cells"] = bool(torch.equal(alt[:, :w + 1], base[:, :w + 1]))
        res[f"cells_flipped_before_wave{w + 1}"] = int(flip.sum())
    B3 = dict(B)
    B3["occ"] = ~B["occ"]
    res["all_waves_identical_after_replacing_occupancy"] = bool(torch.equal(run(B3), base))
    inside = pm[base[:, 0]].any(1) & ~anchors                           # positive control
    B4 = dict(B)
    B4["s"] = B["s"] ^ inside
    alt = run(B4)
    res["positive_control_wave1_identical"] = bool(torch.equal(alt[:, 0], base[:, 0]))
    res["positive_control_boards_with_changed_waves2to4"] = int((alt[:, 1:] != base[:, 1:]).flatten(1).any(1).sum())
    res["boards"] = n
    return res


if __name__ == "__main__":
    a, i = analytic(), instantiated()
    print("parameters analytic   :", a)
    print("parameters instantiated:", i)
    print("looped / attn_K4 =", round(i["looped"] / i["attn_K4"], 5), "(1/%.1f); < 1/5:" % (i["attn_K4"] / i["looped"]),
          i["looped"] < i["attn_K4"] / 5)
    for nw in (True, False):
        print(f"peek test U_{'NW' if nw else 'SE'}:", peek_test(nw))
