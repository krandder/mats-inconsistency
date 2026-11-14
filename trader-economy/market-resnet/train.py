"""Train one arm. Usage: python train.py --arm C --seed 0 --out results/ckpt_C_s0.pt"""
import argparse, json, time, os
import torch
from mresnet import MarketResNet, prefix_batch, loss_fn

p = argparse.ArgumentParser()
p.add_argument("--arm", required=True)            # A B Bds C D
p.add_argument("--seed", type=int, default=0)
p.add_argument("--width", type=int, default=128)
p.add_argument("--T", type=int, default=24)       # training steps (iterations)
p.add_argument("--iters", type=int, default=20000)
p.add_argument("--bs", type=int, default=128)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--lmin", type=int, default=8)
p.add_argument("--lmax", type=int, default=32)
p.add_argument("--beta", type=float, default=0.01)  # PonderNet KL weight
p.add_argument("--lam_prior", type=float, default=0.0)
p.add_argument("--halt_cost", type=float, default=0.05)  # Ctt train-time halting cost
p.add_argument("--eps", type=float, default=0.1)
p.add_argument("--out", required=True)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = p.parse_args()

torch.manual_seed(a.seed)
gen = torch.Generator().manual_seed(1000 + a.seed)
model = MarketResNet(w=a.width).to(a.device)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=2e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.iters, pct_start=0.05)
log = []
t0 = time.time()
start = 0
ck = a.out + ".partial"
if os.path.exists(ck):  # resume after spot preemption
    s = torch.load(ck, map_location=a.device)
    model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); sched.load_state_dict(s["sched"])
    start, log = s["it"], s["log"]; gen.set_state(s["gen"].cpu())
for it in range(start, a.iters):
    x, y, m = prefix_batch(a.bs, a.lmin, a.lmax, gen, a.device)
    loss, lg = loss_fn(a.arm, model, x, y, m, a.T, beta_ponder=a.beta,
                       lam_prior=a.lam_prior or None,
                       halt_cost=a.halt_cost if it > a.iters // 5 else None, eps=a.eps)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step()
    if it % 200 == 0 or it == a.iters - 1:
        lg.update(it=it, sec=round(time.time() - t0, 1))
        log.append(lg)
        print(json.dumps(lg), flush=True)
    if it % 2000 == 1999:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "it": it + 1, "log": log, "gen": gen.get_state()}, ck)
torch.save({"model": model.state_dict(), "args": vars(a), "log": log}, a.out)
if os.path.exists(ck):
    os.remove(ck)
