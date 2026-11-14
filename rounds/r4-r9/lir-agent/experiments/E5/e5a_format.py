"""E5a format control (hawks/E5.md 2): for a checkpoint, belief-pass outputs with the TRUE goal text vs the
SWAPPED goal text (hit<->avoid) on the 128 fixed eval envs: KL nats/slot and Brier vs Bayes / truth each.
'base' = untrained LoRA (B=0, i.e. the frozen base). Writes JSON."""
import argparse, json, os, sys
from dataclasses import replace
import numpy as np, torch
sys.path.insert(0, os.environ.get("LIR_HARNESS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "harness")))
from env.shiphunt import ShipHunt, PRESETS
from model.roles import RoleModel, ModelConfig
from protocol.render import Vocab, belief_seq
from protocol.rollout import belief_pass, self_span

DEC = (np.arange(10) + 0.5) / 10
ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="*", default=[])          # name=path
ap.add_argument("--fused", type=lambda x: x == "1", default=False)
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--device", default="cpu"); ap.add_argument("--dtype", default="float32")
ap.add_argument("--n", type=int, default=128); ap.add_argument("--mb", type=int, default=16)
ap.add_argument("--out", required=True)
a = ap.parse_args()
m = RoleModel(ModelConfig(name=a.model, arm_adapters=("fused",) if a.fused else ("belief", "forecast", "action"),
                          dtype=a.dtype, device=a.device))
V = Vocab(m.tok)
cfg = replace(PRESETS["easy"], goal="mixed")
envs = [ShipHunt(cfg, 50_000_000 + i) for i in range(a.n)]
other = {"hit": "avoid", "avoid": "hit"}
gt_true = [e.goal_text() for e in envs]
gt_swap = [ShipHunt(replace(cfg, goal=other[e.goal]), e.seed).goal_text() for e in envs]
pb = np.stack([e.bayes_cell_probs() for e in envs]); occ = np.stack([e.occ.astype(float) for e in envs])
sgn = np.array([1.0 if e.goal == "hit" else -1.0 for e in envs])


@torch.no_grad()
def run(gts):
    P = []
    for i in range(0, len(envs), a.mb):
        ss = [belief_seq(V, e, "visible", gts[i + j]) for j, e in enumerate(envs[i:i + a.mb])]
        lp, _ = belief_pass(m, V, ss, cfg.n_cells, [self_span(s) for s in ss])
        P.append(lp.exp().float().cpu().numpy())
    return np.concatenate(P)


res = {}
for spec in ["base=none"] + a.ckpts:
    name, path = spec.split("=", 1)
    if path != "none":
        if not os.path.exists(path):
            res[name] = {"missing": path}; continue
        m.load(path)
    Pt, Ps = run(gt_true), run(gt_swap)
    qt, qs = Pt @ DEC, Ps @ DEC
    kl = float(np.mean((Pt * (np.log(Pt + 1e-12) - np.log(Ps + 1e-12))).sum(-1)))
    res[name] = {"kl_true_vs_swapped_nats_per_slot": kl,
                 "brier_vs_bayes_true": float(np.mean((qt - pb) ** 2)), "brier_vs_bayes_swapped": float(np.mean((qs - pb) ** 2)),
                 "brier_vs_truth_true": float(np.mean((qt - occ) ** 2)), "brier_vs_truth_swapped": float(np.mean((qs - occ) ** 2)),
                 "b_G_true": float(np.mean((qt - pb) * sgn[:, None])), "b_G_swapped_text": float(np.mean((qs - pb) * sgn[:, None]))}
json.dump(res, open(a.out, "w"), indent=1)
print(json.dumps(res, indent=1))
