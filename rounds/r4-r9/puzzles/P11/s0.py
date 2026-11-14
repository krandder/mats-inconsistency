"""P11 S0: generator round-trip vs upstream, cap numerics, 30,000-iteration random-init rollouts with the cap
asserted at every step. Writes OUT/s0.json; exits non-zero if anything fails."""
import json
import math
import os
import sys
import time

import torch

import caps
import gen
from model import build, MState


@torch.no_grad()
def cap_rollout(arm, hp, n=16, iters=30_000, u_scale=1.0, seed=0, dev="cpu"):
    torch.manual_seed(seed)
    net = build(arm).to(dev).eval()
    S = gen.eval_set("val", 17, n)
    x = torch.tensor(S["x"], device=dev).float()
    ms = MState(arm, hp, x[:, 0].shape, dev)
    h = net.init_state(x)
    worst = torch.full((), -float("inf"), dtype=torch.float64, device=dev)
    maxb = torch.zeros((), device=dev)
    nonfin = torch.zeros((), dtype=torch.bool, device=dev)
    remneg = torch.zeros((), dtype=torch.bool, device=dev)
    for t in range(1, iters + 1):
        h = net.step(h, x, ms.feat())
        u = net.out(h) * u_scale
        b_old = ms.b.detach().clone()
        rem_old = None if ms.rem is None else ms.rem.clone()
        b = ms.advance(u)
        nonfin |= ~(torch.isfinite(b).all() & torch.isfinite(h).all())
        d = (b.double() - b_old.double())
        tol = 1e-6 + 8 * caps.ulp32(b_old)
        if arm == "M-cap":
            ex = caps.wl(b_old.double(), d) - (float(hp["eps"]) + tol)
        elif arm == "M-budget":
            ex = caps.wl(b_old.double(), d) - (rem_old + tol)
        else:
            ex = d.abs() - (rem_old + tol)
        worst = torch.maximum(worst, ex.max())
        maxb = torch.maximum(maxb, b.abs().max())
        if ms.rem is not None:
            remneg |= ms.rem.min() < 0
    worst, maxb = float(worst), float(maxb)
    ok = bool(worst <= 0 and not bool(nonfin) and not bool(remneg))
    first_bad = None
    return {"arm": arm, "hp": hp, "u_scale": u_scale, "iters": t, "pass": ok, "first_bad": first_bad,
            "max_cap_excess": worst, "max_abs_b": maxb,
            "remaining_frac_mean": None if ms.rem is None else float((ms.rem / ms.init).mean())}


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "."
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    iters = int(os.environ.get("P11_S0_ITERS", "30000"))
    res = {"device": torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", "torch": torch.__version__}
    t0 = time.time()
    res["roundtrip"] = gen.roundtrip_test(200)
    res["own_generator"] = gen.own_generator_checks(200)
    res["caps"] = caps.s0_tests()
    res["rollouts"] = []
    for arm, hp in (("M-budget", {"W": 6}), ("M-cap", {"eps": 0.3}), ("M-TV", {"V": 8})):
        for us in (1.0, 20.0):
            r = cap_rollout(arm, hp, iters=iters, u_scale=us, dev=dev)
            print(json.dumps(r), flush=True)
            res["rollouts"].append(r)
    res["sec"] = round(time.time() - t0, 1)
    res["pass"] = bool(res["roundtrip"]["pass"] and res["own_generator"]["pass"] and res["caps"]["pass"]
                       and all(r["pass"] for r in res["rollouts"]))
    os.makedirs(out, exist_ok=True)
    json.dump(res, open(os.path.join(out, "s0.json"), "w"), indent=1)
    print("S0", "PASS" if res["pass"] else "FAIL", res["sec"], flush=True)
    sys.exit(0 if res["pass"] else 1)


if __name__ == "__main__":
    main()
