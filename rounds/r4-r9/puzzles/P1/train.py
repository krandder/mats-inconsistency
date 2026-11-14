"""Train the T2m backbone: MarketResNet w=64, arm Bds (final + deep supervision),
L in {64, 128} with T = L, 10k iterations (market-resnet/train.py recipe).
  python train.py --task t2m --seed 0 --width 64 --iters 10000 --out results/t2m_bb_s0.pt
Then the validity gate:  python train.py --task t2m --gate results/t2m_bb_s0.pt
"""
import argparse
import json
import os
import time
import torch
from data import t2m_train_batch, t2m_batch, t2m_curr_batch, t2m_curr2_batch, v2_lr_factor, V2_STAGES
from models import T2mNet
from mresnet import score

HOLD_EMA, MAX_HOLD = 0.5, 3000   # recipe v2 stage holds (deviation 9b)


def bds_loss(model, x, y, mask, T):
    st = model.init_state(x)
    nbits = mask.sum(1)
    S = []
    for _ in range(T):
        st, b, _ = model.step_once(st)
        S.append(score(b, y, mask))
    final = (-S[-1] / nbits).mean()
    ds = torch.stack([(-s / nbits).mean() for s in S]).mean()
    return final + ds, final.item()


@torch.no_grad()
def gate(path, device, n=500, seed=12345):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = T2mNet(w=ck["args"]["width"]).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    g = torch.Generator().manual_seed(seed)
    res = {}
    for L in (64, 128):
        x, y, mask, meta = t2m_batch(n, L, g, device)
        st = m.init_state(x)
        for _ in range(L):
            st, b, _ = m.step_once(st)
        p = torch.sigmoid(b)
        exact = ((p > 0.5).float() == y).all(1)
        r = meta["resolvable"].bool()
        res[L] = {"resolvable_exact": exact[r].float().mean().item(),
                  "blocked_mean_abs_p_minus_half": (p[~r] - 0.5).abs().mean().item()}
    ok = (res[64]["resolvable_exact"] >= 0.95 and res[128]["resolvable_exact"] >= 0.90
          and res[64]["blocked_mean_abs_p_minus_half"] <= 0.1 and res[128]["blocked_mean_abs_p_minus_half"] <= 0.1)
    return {"path": path, "width": ck["args"]["width"], "per_L": res, "pass": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="t2m")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--recipe", default="v1", help="v2 = gradual-L curriculum + lower lr past L=32 (deviation 9b)")
    ap.add_argument("--curr_iters", type=int, default=0, help="warm-up curriculum iterations (deviation 9)")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="results/t2m_bb_s0.pt")
    ap.add_argument("--gate", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    assert a.task == "t2m"
    f = float(os.environ.get("P1_GPU_FRAC", "0"))
    if f > 0 and a.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(f)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if a.gate:
        r = gate(a.gate, a.device)
        print(json.dumps(r))
        with open(a.gate.replace(".pt", ".gate.json"), "w") as f:
            json.dump(r, f, indent=1)
        return
    torch.manual_seed(a.seed)
    g = torch.Generator().manual_seed(a.seed)
    m = T2mNet(w=a.width).to(a.device)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=2e-4)
    if a.recipe == "v2":
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: v2_lr_factor(i, a.iters))
    else:
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.05)
    start, log = 0, []
    delay, ema = 0, None   # v2: stage holds (an unlearned stage is extended, at most MAX_HOLD iterations in total)
    part = a.out + ".partial"
    if os.path.exists(part):
        ck = torch.load(part, map_location="cpu", weights_only=False)
        m.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        start, log = ck["it"] + 1, ck["log"]
        delay, ema = ck.get("delay", 0), ck.get("ema")
        g.set_state(ck["gen"])
    t0 = time.time()
    bounds = {end for end, _ in V2_STAGES}
    it = start
    while it - delay < a.iters:
        ie = it - delay   # position in the recipe (== it unless a stage was held)
        if a.recipe == "v2":
            for pg in opt.param_groups:
                pg["lr"] = a.lr * v2_lr_factor(ie, a.iters)
            L, x, y, mask, _ = t2m_curr2_batch(a.bs, ie, g, a.device)
        elif it < a.curr_iters:
            L, x, y, mask, _ = t2m_curr_batch(a.bs, it, a.curr_iters, g, a.device)
        else:
            L, x, y, mask, _ = t2m_train_batch(a.bs, g, a.device)
        loss, final = bds_loss(m, x, y, mask, L)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        if a.recipe != "v2":
            sched.step()
        ema = final if ema is None else 0.98 * ema + 0.02 * final
        if it % 200 == 0 or ie == a.iters - 1:
            log.append({"it": it, "ie": ie, "L": L, "loss": loss.item(), "final": final, "ema": round(ema, 4),
                        "delay": delay, "sec": round(time.time() - t0, 1)})
            print(json.dumps(log[-1]), flush=True)
        if (it + 1) % 500 == 0:   # partial every 500 it (spot preemption), written atomically
            torch.save({"model": m.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                        "it": it, "log": log, "gen": g.get_state(), "delay": delay, "ema": ema}, part + ".tmp")
            os.replace(part + ".tmp", part)
        if a.recipe == "v2" and (ie + 1) in bounds and ema > HOLD_EMA and delay < MAX_HOLD:
            delay += 1   # stay on this stage's last recipe step until the loss shows it is learned
        it += 1
    torch.save({"model": m.state_dict(), "args": vars(a), "log": log, "delay": delay}, a.out)
    if os.path.exists(part):
        os.remove(part)


if __name__ == "__main__":
    main()
