"""P0 unit tests (CPU, toy size). Writes results/unit_tests.json.

  1. param counts asserted against frozen/manifest.json
  2. dt_net.pth loads strict=True into dtnet.DTRecall and reproduces Knutson's DTNetOriginal
     (mines-opt-ml/maze-extrapolation) outputs to 1e-5 at every one of 30 iterations
  3. check.py and Knutson's checker accept every true solution; perturbed solutions fail check.py
  4. declared FLOPs per position match torch.utils.flop_counter within 10% (DT 1D/2D, DT-L, MarketResNet)
  5. DT-L has at most 160k parameters
"""
import json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
torch.set_num_threads(int(os.environ.get("P0_THREADS", "1")))
from dtnet import DTRecall, n_params, load_mref  # noqa: E402
from dtl import make_dtl  # noqa: E402
from check import check, knutson, expand  # noqa: E402
from mr_adapter import DTAdapter, MRAdapter, mresnet  # noqa: E402
import p0eval  # noqa: E402

R = {}
man = json.load(open(os.path.join(HERE, "frozen/manifest.json")))
P = man["params"]
got = {"dt2d_w64": n_params(DTRecall(64, 2)), "dt2d_w128": n_params(DTRecall(128, 2)),
       "dt1d_w90_in3": n_params(DTRecall(90, 1)),
       # market-resnet's 158,209 counts the backbone + belief head; the price (ghead) and PonderNet (phead, pout)
       # heads exist in every MarketResNet but are used only by arms C/D
       "mresnet_w128": sum(p.numel() for n, p in mresnet.MarketResNet(w=128).named_parameters()
                           if not n.startswith(("ghead", "phead", "pout")))}
R["params"] = {k: {"got": v, "manifest": P[k], "ok": v == P[k]} for k, v in got.items()}
R["params"]["mresnet_w128_all_heads"] = {"got": n_params(mresnet.MarketResNet(w=128)), "manifest": None, "ok": True}
assert all(r["ok"] for r in R["params"].values()), R["params"]
dl = make_dtl()
R["dtl_params"] = {"width": dl.width, "params": n_params(dl), "ok": n_params(dl) <= 160_000}
assert R["dtl_params"]["ok"]

# ---- 2. DTNet reproduction
sys.path.insert(0, os.environ.get("P0_KNUTSON", "/tmp/p0src/maze-extrapolation"))
from src.models.dt_net_original import DTNetOriginal  # noqa: E402
sets, _ = p0eval.load_frozen(os.path.join(HERE, "frozen"))
bad = p0eval.verify_frozen(sets)
R["frozen_sha256_ok"] = not bad
assert not bad, bad
mref_path = os.path.join(HERE, "src/dt_net.pth")
mine = load_mref(mref_path, "cpu").eval()
sd = torch.load(mref_path, map_location="cpu", weights_only=False)
sd = sd.get("net", sd)
sd = {k.replace("module.", ""): v for k, v in sd.items()}
ref = DTNetOriginal(width=128, in_channels=3, recall=True).eval()
ref.load_state_dict(sd, strict=True)
b = p0eval.make_batch(sets["m9"], np.arange(4), "cpu")
with torch.no_grad():
    allout = ref(b["x"], 30)
    h = mine.init(b["x"])
    diffs = []
    for t in range(30):
        h = mine.step(h, b["x"])
        diffs.append((mine.readout(h) - allout[:, t]).abs().max().item())
R["dtnet_repro"] = {"max_abs_diff": max(diffs), "iters": 30, "ok": max(diffs) <= 1e-5}
assert R["dtnet_repro"]["ok"], diffs

# ---- 3. checkers on true and perturbed solutions
res = {}
for name in ["m9", "m13", "m33", "m59", "v9"] + [k for k in sets if k.startswith("sh_") and "_p0_" in k]:
    d = sets[name]
    n = min(d["meta"]["n"], 200)
    bb = p0eval.make_batch(d, np.arange(n), "cpu")
    sol = bb["sol_px"].long()
    ok_c = check(sol, bb["code"], bb["sp_len"])
    ok_k = knutson(bb["x"], sol, bb["sol_px"])
    # perturbation: switch on one extra open 2x2 block next to the path, and switch off one path block
    code = bb["code"].long()
    nn_ = code.shape[-1]
    lat = sol[:, 3:3 + 2 * nn_:2, 3:3 + 2 * nn_:2]
    extra = sol.clone()
    drop = sol.clone()
    # a maze whose path covers every open cell has no cell to add: those rows are left out of the extra test
    has_x = torch.zeros(n, dtype=torch.bool)
    has_d = torch.zeros(n, dtype=torch.bool)
    for i in range(n):
        free = ((code[i] > 0) & (lat[i] == 0)).nonzero()
        if len(free):
            r, c = free[0].tolist()
            extra[i, 3 + 2 * r:5 + 2 * r, 3 + 2 * c:5 + 2 * c] = 1
            has_x[i] = True
        on = ((lat[i] == 1) & (code[i] == 1)).nonzero()
        if len(on):
            r, c = on[len(on) // 2].tolist()
            drop[i, 3 + 2 * r:5 + 2 * r, 3 + 2 * c:5 + 2 * c] = 0
            has_d[i] = True
    res[name] = {"n": n, "check_true": int(ok_c.sum()), "knutson_true": int(ok_k.sum()),
                 "n_extra": int(has_x.sum()), "n_drop": int(has_d.sum()),
                 "check_extra": int((check(extra, bb["code"], bb["sp_len"]) & has_x).sum()),
                 "check_drop": int((check(drop, bb["code"], bb["sp_len"]) & has_d).sum())}
    assert ok_c.all() and ok_k.all(), (name, res[name])
    assert res[name]["check_extra"] == 0 and res[name]["check_drop"] == 0, (name, res[name])
R["checker"] = res

# ---- 4. FLOP declarations
fl = {}
for nm_, ad, d in [("dt2d_w8", DTAdapter(DTRecall(8, 2).eval(), "maze"), sets["m9"]),
                   ("dt1d_w90", DTAdapter(DTRecall(90, 1).eval(), "prefix"), sets["v_id"]),
                   ("dtl", DTAdapter(dl.eval(), "prefix"), sets["v_id"]),
                   ("mr_w128", MRAdapter(mresnet.MarketResNet(w=128).eval()), sets["v_id"])]:
    fl[nm_] = p0eval.flop_check(ad, d, "cpu")
    assert fl[nm_]["ok"], (nm_, fl[nm_])
R["flops"] = fl
R["all_ok"] = True
outp = os.environ.get("P0_TESTS_OUT", os.path.join(HERE, "results/unit_tests.json"))
os.makedirs(os.path.dirname(outp), exist_ok=True)
json.dump(R, open(outp, "w"), indent=1, default=float)
print(json.dumps(R, default=float))
