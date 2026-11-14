"""E4 structural leak check on the CURRENT harness (role passes, R2). Replaces hawks/E4-leak.py, whose
single-pass API (T, gating, relevance router) no longer exists in model/roles.py. For each arm
config, perturb each reward-trained parameter group and measure, at the claim positions of the
forecast pass (success/steps/hit) and of the belief pass (cell claims), the max |delta logit| and the
grad norm of a claim-position loss w.r.t. that group. 0.0 exactly = no parametric path.
Router: not present in the current harness (README 'Not built yet' item 3), recorded as such."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import e4lib as E
from env import ShipHunt, PRESETS
from model.roles import RoleModel, ModelConfig
from protocol.render import Vocab, forecast_seq, belief_seq
from protocol.rollout import pad_batch

CONFIGS = {  # name: (adapters, model class, reward-trained groups)
    "P (M2: role passes, goal deleted)": (("belief", "forecast", "action"), RoleModel, ["action"]),
    "twin (P, action lr 0)": (("belief", "forecast", "action"), RoleModel, ["action"]),
    "M1 (three adapters, gating all)": (("belief", "forecast", "action"), E.AllGatesRoleModel, ["action"]),
    "M0 (fused, forecast tokens out of GRPO)": (("fused",), RoleModel, ["fused"]),
    "B0 (fused, GRPO on all tokens)": (("fused",), RoleModel, ["fused"]),
}


def claim_logits(m, V, role, seqs, keys):
    ids, attn = pad_batch([s.ids for s in seqs], V.pad)
    lg = m.logits(role, ids, attn)
    return torch.stack([torch.stack([lg[b, s.q[k], V.digits] for k in keys]) for b, s in enumerate(seqs)])


def run_one(adapters, cls, group, seed=0):
    torch.manual_seed(seed)
    m = cls(ModelConfig(name="tiny", arm_adapters=adapters, seed=seed)).eval()
    V = Vocab(m.tok)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in m.peft.named_parameters():
            if "lora_B" in n:
                p.copy_(0.05 * torch.randn(p.shape, generator=g))
    cfg = PRESETS["syco16"]
    envs = [ShipHunt(cfg, 5 + i) for i in range(3)]
    fseqs = [forecast_seq(V, e, E.canon_sit(e), [0, 1, 2]) for e in envs]
    bseqs = [belief_seq(V, e) for e in envs]
    fkeys = ["success", "steps", "hit0", "hit1", "hit2"]
    bkeys = [f"cell{c}" for c in range(cfg.n_cells)]
    res = {}
    for role, seqs, keys in (("forecast", fseqs, fkeys), ("belief", bseqs, bkeys)):
        with torch.no_grad():
            base = claim_logits(m, V, role, seqs, keys)
        m.set_trainable([group])
        for p in m.parameters():
            p.grad = None
        lg = claim_logits(m, V, role, seqs, keys)
        if lg.requires_grad:      # else: the group is not in this pass's graph at all (structural zero)
            torch.log_softmax(lg, -1)[..., 3].sum().neg().backward()
        gn = float(sum(float(p.grad.pow(2).sum()) for p in m.group_params(group) if p.grad is not None) ** 0.5)
        m.set_trainable([])
        saved = [p.detach().clone() for p in m.group_params(group)]
        with torch.no_grad():
            for p in m.group_params(group):
                p.add_(0.5 * torch.randn(p.shape, generator=g))
            pert = claim_logits(m, V, role, seqs, keys)
            for p, s in zip(m.group_params(group), saved):
                p.copy_(s)
        res[role] = {"logit_maxabs_change": float((pert - base).abs().max()), "loss_grad_norm": gn}
    return res


def main(out=None):
    rows = []
    for name, (ad, cls, groups) in CONFIGS.items():
        for grp in groups:
            r = run_one(ad, cls, grp)
            for role, v in r.items():
                rows.append(dict(config=name, group=grp, pass_=role, **v))
        rows.append(dict(config=name, group="router", pass_="forecast", logit_maxabs_change=None, loss_grad_norm=None,
                         note="no relevance router in the current harness (README Not built yet, item 3)"))
    for r in rows:
        print(json.dumps(r))
    if out:
        json.dump(rows, open(out, "w"), indent=1)
    return rows


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
