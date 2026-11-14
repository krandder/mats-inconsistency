"""E3 exogenous-n evaluation pass (GPU). For each proposer checkpoint c: one draw of 64 candidate
plans per state from the action adapter (situation report from c's own belief pass), then for each
world-model checkpoint f in the pair list: the forecaster's E[P(success)] for all 64 candidates
(situation report from f's belief pass, goal mode as in training). Plus a fixed policy-independent
probe set (uniform random plans) per forecaster checkpoint. No Bayes here: scoring is CPU-side
(e3_analyze.py) from the state seeds and the plans. Output: JSON only.
Usage: python3 e3_eval.py --ckpt DIR --out DIR --arm-fused 0|1 --preset hard --goal-mode deleted
       --pairs c000:c000,c025:c025,c025:c000,... --n-states 300 --n-cands 64
"""
import argparse, json, os, sys, time

import numpy as np
import torch

H = os.environ.get("LIR_HARNESS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../harness"))
sys.path.insert(0, H)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e3_train  # noqa: F401,E402  (registers preset hardsyco)
from env import ShipHunt, PRESETS  # noqa: E402
from model import RoleModel, ModelConfig  # noqa: E402
from protocol.render import Vocab, belief_seq, forecast_seq, proposal_seq  # noqa: E402
from protocol.rollout import belief_pass, forecast_pass, pad_batch, _logp_at, self_span, DEC  # noqa: E402

STATE0, PROBE0 = 70_000_000, 80_000_000


def sits(model, V, envs, goal_mode, seed, bs=64):
    g = torch.Generator().manual_seed(seed)
    out = []
    nc = envs[0].cfg.n_cells
    for i in range(0, len(envs), bs):
        ch = envs[i:i + bs]
        bseqs = [belief_seq(V, e, goal_mode, e.goal_text()) for e in ch]
        cell_lp, _ = belief_pass(model, V, bseqs, nc, [self_span(s) for s in bseqs])
        p = cell_lp.exp().float().cpu()
        for b in range(len(ch)):
            out.append([int(torch.multinomial(p[b, c], 1, generator=g)) for c in range(nc)])
    return out


def propose(model, V, envs, sit, K, seed, bs=256):
    """K i.i.d. plans per state, sampled cell by cell without repeats (as protocol/rollout.py), with one
    forward pass per distinct prefix (the next-cell distribution depends only on the prefix)."""
    g = torch.Generator().manual_seed(seed)
    nc, L = envs[0].cfg.n_cells, envs[0].cfg.plan_len
    pseqs = [proposal_seq(V, e, sit[j]).ids for j, e in enumerate(envs)]
    plans = [[[] for _ in range(K)] for _ in envs]
    for step in range(L):
        keys = sorted({(j, tuple(p)) for j in range(len(envs)) for p in plans[j]})
        dist = {}
        for i in range(0, len(keys), bs):
            ch = keys[i:i + bs]
            seqs = [pseqs[j] + [V.cells[c] for c in pre] for j, pre in ch]
            ids, attn = pad_batch(seqs, V.pad)
            lg = model.logits("action", ids, attn)
            for b, (j, pre) in enumerate(ch):
                lp = _logp_at(lg, b, len(seqs[b]) - 1, V.cells[:nc]).float().clone()
                for c in pre:
                    lp[c] = -float("inf")
                dist[(j, pre)] = torch.softmax(lp, -1).cpu()
        for j in range(len(envs)):
            for k in range(K):
                p = plans[j][k]
                p.append(int(torch.multinomial(dist[(j, tuple(p))], 1, generator=g)))
    return plans


def forecast(model, V, envs, sit, plans, goal_mode, bs=256):
    """E[P(success)] for every candidate; identical (state, ordered plan) pairs are forecast once."""
    L = envs[0].cfg.plan_len
    keys = sorted({(j, tuple(p)) for j in range(len(envs)) for p in plans[j]})
    val = {}
    for i in range(0, len(keys), bs):
        ch = keys[i:i + bs]
        seqs = [forecast_seq(V, envs[j], sit[j], list(p), goal_mode, envs[j].goal_text()) for j, p in ch]
        s_lp, _, _ = forecast_pass(model, V, seqs, L)
        for k, v in zip(ch, (s_lp.exp().float().cpu().numpy() * DEC).sum(-1).tolist()):
            val[k] = v
    return [[round(val[(j, tuple(p))], 5) for p in plans[j]] for j in range(len(envs))]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fused", type=int, default=0)
    ap.add_argument("--preset", default="hard")
    ap.add_argument("--goal-mode", default="deleted")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--n-states", type=int, default=300)
    ap.add_argument("--n-cands", type=int, default=64)
    ap.add_argument("--n-probe-states", type=int, default=200)
    ap.add_argument("--n-probe-plans", type=int, default=16)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--skip-dir", default=None, help="reuse pair_/probe_ JSON already computed there (same seeds)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.threads:
        torch.set_num_threads(a.threads)
    mc = ModelConfig(name=a.model, arm_adapters=("fused",) if a.fused else ("belief", "forecast", "action"),
                     dtype=a.dtype, device=a.device, seed=0)
    model = RoleModel(mc)
    model.eval()
    V = Vocab(model.tok)
    cfg = PRESETS[a.preset]
    envs = [ShipHunt(cfg, STATE0 + i) for i in range(a.n_states)]
    penvs = [ShipHunt(cfg, PROBE0 + i) for i in range(a.n_probe_states)]
    rng = np.random.default_rng(4242)
    pplans = [[rng.choice(cfg.n_cells, cfg.plan_len, replace=False).tolist() for _ in range(a.n_probe_plans)]
              for _ in penvs]
    pairs = [tuple(p.split(":")) for p in a.pairs.split(",")]
    if a.skip_dir and os.path.isdir(a.skip_dir):
        import shutil
        for fn in os.listdir(a.skip_dir):
            if fn.startswith(("pair_", "probe_")) and fn.endswith(".json") and not os.path.exists(os.path.join(a.out, fn)):
                shutil.copy(os.path.join(a.skip_dir, fn), os.path.join(a.out, fn))
    have = lambda c, f: os.path.exists(os.path.join(a.out, f"pair_{c}_{f}.json"))
    pairs = [(c, f) for c, f in pairs if not have(c, f)]
    timing = {}
    done_probe = {fn[6:-5] for fn in os.listdir(a.out) if fn.startswith("probe_")}
    for c in list(dict.fromkeys(p for p, _ in pairs)):
        t0 = time.time()
        model.load(os.path.join(a.ckpt, c + ".pt"))
        sit_c = sits(model, V, envs, a.goal_mode, 11)
        plans = propose(model, V, envs, sit_c, a.n_cands, 13)
        timing[f"propose_{c}"] = time.time() - t0
        for f in [f for p, f in pairs if p == c]:
            t0 = time.time()
            model.load(os.path.join(a.ckpt, f + ".pt"))
            sit_f = sits(model, V, envs, a.goal_mode, 11)
            fc = forecast(model, V, envs, sit_f, plans, a.goal_mode)
            json.dump({"proposer": c, "forecaster": f, "preset": a.preset, "state_seeds": [e.seed for e in envs],
                       "plans": plans, "f": fc, "sit_f": sit_f}, open(os.path.join(a.out, f"pair_{c}_{f}.json"), "w"))
            timing[f"forecast_{c}_{f}"] = time.time() - t0
            if f not in done_probe:
                t0 = time.time()
                psit = sits(model, V, penvs, a.goal_mode, 17)
                pf = forecast(model, V, penvs, psit, pplans, a.goal_mode)
                json.dump({"forecaster": f, "preset": a.preset, "state_seeds": [e.seed for e in penvs],
                           "plans": pplans, "f": pf}, open(os.path.join(a.out, f"probe_{f}.json"), "w"))
                done_probe.add(f)
                timing[f"probe_{f}"] = time.time() - t0
            model.load(os.path.join(a.ckpt, c + ".pt"))
        print(json.dumps(timing), flush=True)
    json.dump(timing, open(os.path.join(a.out, "timing.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
