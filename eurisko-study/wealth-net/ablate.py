"""Does deleting the poorest traders hurt held-out loss less than deleting other units?

For each seed and saved checkpoint, rank MLP neurons and attention heads by
  wealth      cumulative wealth (most negative first)
  recent      revenue over the last 2000 steps (most negative first)
  random      random order (mean of 5 draws; the control)
  gradnorm    mean loss-gradient norm so far (smallest first)
  actmag      mean |activation| on training data (smallest first)
zero the first k units (fraction 0..40%) and measure BCE and Brier on a fresh stream
segment: the same questions (same seed, since the question-id embedding is tied to them),
positions after everything used in training and in-training evaluation.
Writes results/ablation.csv. Usage: python3 ablate.py [results] [--device cpu]
"""
import csv, glob, json, os, re, sys
import numpy as np
import torch
import torch.nn.functional as F
from streams import make_questions, FAMILIES
from model import Net

FRACS = [0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40]
CRIT = ["wealth", "recent", "random", "gradnorm", "actmag"]


def snap_at(res, tag, step):
    f = os.path.join(res, tag, "snapshots", f"step-{step:05d}.json")
    return json.load(open(f)) if os.path.exists(f) else None


def main(res="results", device="cpu"):
    torch.set_num_threads(8)
    dev = torch.device(device)
    rows = []
    for meta_f in sorted(glob.glob(os.path.join(res, "train-s*.json"))):
        meta = json.load(open(meta_f)); tag = meta["tag"]; seed = meta["seed"]
        Q, W, T = meta["Q"], meta["W"], meta["T"]
        E = 4096
        bits, fam, _ = make_questions(Q, T + E, seed)
        b = torch.tensor(bits, dtype=torch.float32, device=dev)
        wins = b.unfold(1, W, 1)
        ev = torch.arange(T - W, T + E - W)                  # predicts bits[T .. T+E-1]: never seen
        ev_win, ev_y = wins[:, ev].permute(1, 0, 2).contiguous(), b[:, ev + W].T.contiguous()
        tr = torch.randint(0, int((T - W) * 0.8), (1024,), generator=torch.Generator().manual_seed(seed))
        tr_win = wins[:, tr].permute(1, 0, 2).contiguous()
        famt = torch.tensor(fam, device=dev)
        ckpts = []
        for f in glob.glob(os.path.join(res, "weights", f"{tag}-step*.pt")):
            ckpts.append((int(re.search(r"step(\d+)", f).group(1)), f))
        ckpts.append((meta["steps_done"], os.path.join(res, f"{tag}.pt")))
        ckpts = sorted(set(c for c in ckpts if os.path.exists(c[1])))
        # final + up to 3 earlier (spread out)
        if len(ckpts) > 4:
            pick = np.linspace(0, len(ckpts) - 2, 3).round().astype(int).tolist()
            ckpts = [ckpts[i] for i in sorted(set(pick))] + [ckpts[-1]]
        for step, wf in ckpts:
            sn = snap_at(res, tag, step); old = snap_at(res, tag, step - 2000)
            if sn is None:
                print("no snapshot for", tag, step); continue
            m = Net(Q, W, meta["d"], meta["H"], meta["L"]).to(dev)
            m.load_state_dict(torch.load(wf, map_location=dev)); m.eval()
            L, H, N = m.L, m.H, 4 * m.d
            with torch.no_grad():
                rec = {}
                m(tr_win, rec=rec)
                actmag = {"neuron": torch.stack([a.abs().mean((0, 1)) for a in rec["act"]]).cpu().numpy().ravel(),
                          "head": torch.stack([h.norm(dim=-1).mean((0, 2)) for h in rec["hv"]]).cpu().numpy().ravel()}

            def evaluate(mask):
                with torch.no_grad():
                    z = torch.cat([m(ev_win[i:i + 1024], mask=mask) for i in range(0, E, 1024)])
                    bce = F.binary_cross_entropy_with_logits(z, ev_y, reduction="none").mean(0)
                    br = ((torch.sigmoid(z) - ev_y) ** 2).mean(0)
                    pf = [(bce[famt == k].mean().item(), br[famt == k].mean().item()) for k in range(len(FAMILIES))]
                return bce.mean().item(), br.mean().item(), pf

            for g, n, shape in (("neuron", L * N, (L, N)), ("head", L * H, (L, H))):
                w = np.array(sn["wealth"][g]); gn = np.array(sn["gradnorm_cum"][g]).ravel()
                rec_rev = w - np.array(old["wealth"][g]) if old else w
                scores = {"wealth": w, "recent": rec_rev, "gradnorm": gn, "actmag": actmag[g]}
                rng = np.random.default_rng(seed * 1000 + step)
                n_neg = int((w < 0).sum())
                for crit in CRIT:
                    orders = [rng.permutation(n) for _ in range(5)] if crit == "random" else [np.argsort(scores[crit], kind="stable")]
                    for fr in FRACS:
                        k = int(round(fr * n))
                        res_ = []
                        for od in orders:
                            mask = {"neuron": torch.ones(L, N, device=dev), "head": torch.ones(L, H, device=dev)}
                            flat = mask[g].view(-1); flat[torch.tensor(od[:k], dtype=torch.long, device=dev)] = 0
                            res_.append(evaluate(mask))
                        bce = np.mean([r[0] for r in res_]); br = np.mean([r[1] for r in res_])
                        pf = np.mean([[p for p in r[2]] for r in res_], axis=0)
                        row = {"seed": seed, "step": step, "final": int(step == meta["steps_done"]), "granularity": g,
                               "criterion": crit, "frac": fr, "k_removed": k, "n_units": n, "n_negative_wealth": n_neg,
                               "bce": round(bce, 6), "brier": round(br, 6)}
                        for fi, fn in enumerate(FAMILIES):
                            row[f"bce_{fn}"] = round(pf[fi][0], 6); row[f"brier_{fn}"] = round(pf[fi][1], 6)
                        rows.append(row)
                print(tag, step, g, "done", flush=True)
    with open(os.path.join(res, "ablation.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print("wrote", len(rows), "rows")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(args[0] if args else "results", "cuda" if "--cuda" in sys.argv else "cpu")
