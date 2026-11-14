"""Driver for PREREG-w1.md. usage: run_w1.py ARM SEED"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "substrate"))
import numpy as np
import torch
from streams import make_questions, FAMILIES
from model import Net, Logistic
from prequential import ArrayStream, run_prequential, run_multiepoch, per_family, overlap_audit

arm, seed = sys.argv[1], int(sys.argv[2])
torch.set_num_threads(1)
torch.manual_seed(seed); np.random.seed(seed)
Q, W, T = 64, 32, 64032
BULK = f"/tmp/traders-net/S0/w1"
os.makedirs(BULK, exist_ok=True)
out = os.path.join(HERE, "results")
t0 = time.time()
b, f, desc = make_questions(Q, T, seed)
s = ArrayStream(b, f, W)
split = int(s.n_pos * 0.8)
m, lm = Net(Q, W), Logistic(Q, W)
o, ol = torch.optim.Adam(m.parameters(), 1e-3), torch.optim.Adam(lm.parameters(), 1e-2)
res = {"arm": arm, "seed": seed, "Q": Q, "W": W, "T": T, "split": split, "families": FAMILIES}
if arm.startswith("P"):
    B = int(arm[1:])
    r = run_prequential(m, s, o, B=B, baseline=lm, opt_b=ol)
    tail = slice(split, s.n_pos)
    res["bce"], res["brier"] = [x.tolist() for x in per_family(r["z"][tail], r["y"][tail], r["mask"][tail], f)]
    res["bce_log"], res["brier_log"] = [x.tolist() for x in per_family(r["zb"][tail], r["y"][tail], r["mask"][tail], f)]
    res["steps"] = r["steps"]
    # learning curve: per-family prequential BCE per 1000-position block
    with open(os.path.join(out, f"{arm}-s{seed}-curve.csv"), "w") as fh:
        fh.write("pos_lo," + ",".join(f"bce_{n}" for n in FAMILIES) + "," + ",".join(f"bcelog_{n}" for n in FAMILIES) + "\n")
        for lo in range(0, s.n_pos, 1000):
            sl = slice(lo, lo + 1000)
            a1, _ = per_family(r["z"][sl], r["y"][sl], r["mask"][sl], f)
            a2, _ = per_family(r["zb"][sl], r["y"][sl], r["mask"][sl], f)
            fh.write(f"{lo}," + ",".join(f"{x:.5f}" for x in a1) + "," + ",".join(f"{x:.5f}" for x in a2) + "\n")
    np.savez_compressed(os.path.join(BULK, f"{arm}-s{seed}.npz"), z=r["z"], zb=r["zb"], y=r["y"], fam=f)
else:
    rows, info = run_multiepoch(m, s, o, epochs=int(os.environ.get("ME_EPOCHS", 15)), B=128, seed=seed, baseline=lm, opt_b=ol)
    last = rows[-1]
    res.update({"bce": last["ho_bce"].tolist(), "brier": last["ho_brier"].tolist(),
                "in_bce": last["in_bce"].tolist(), "in_brier": last["in_brier"].tolist(),
                "bce_log": last["ho_bce_log"].tolist(), "in_bce_log": last["in_bce_log"].tolist()})
    res.update(info)
    with open(os.path.join(out, f"{arm}-s{seed}-curve.csv"), "w") as fh:
        fh.write("epoch,step," + ",".join(f"in_{n}" for n in FAMILIES) + "," + ",".join(f"ho_{n}" for n in FAMILIES) + "\n")
        for rr in rows:
            fh.write(f"{rr['epoch']},{rr['step']}," + ",".join(f"{x:.5f}" for x in rr["in_bce"]) + "," + ",".join(f"{x:.5f}" for x in rr["ho_bce"]) + "\n")
    res["overlap"] = {FAMILIES[k]: v for k, v in overlap_audit(b, f, split, W).items()}
res["wall_s"] = time.time() - t0
json.dump(res, open(os.path.join(out, f"{arm}-s{seed}.json"), "w"), indent=1)
print(arm, seed, "done", res["wall_s"])
