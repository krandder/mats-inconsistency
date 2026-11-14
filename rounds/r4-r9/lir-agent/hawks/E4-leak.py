"""Hawk E4 leak check: does any reward-trained parameter group reach forecast-token logits?
Runs the harness RoleModel ("tiny" random Qwen3, CPU, seconds). For each arm config, perturb one
parameter group (action LoRA, router) and measure the max abs change of logits at FC positions,
plus the grad norm of an FC-position loss w.r.t. that group. 0.0 exactly = structurally no leak."""
import sys, json
sys.path = [p for p in sys.path if 'dist-packages' not in p]
sys.path.insert(0, '/tmp/lir-agent/pylib')
sys.path.insert(0, '/home/agent/workspaces/eurisko-study/lir-agent/harness')
import torch
from model.roles import RoleModel, ModelConfig, T, goal_reaches

torch.manual_seed(0)
# sequence layout: GOAL x4 | EVID item0 x3 | EVID item1 x3 | SIT x3 | PLAN x3 | FCIN x3 | FC x4 | CTRL x2
seg = [(T.GOAL, 4, -1), (T.EVID, 3, 0), (T.EVID, 3, 1), (T.SIT, 3, -1), (T.PLAN, 3, -1),
       (T.FCIN, 3, -1), (T.FC, 4, -1), (T.CTRL, 2, -1)]
types = torch.tensor([[t for t, n, _ in seg for _ in range(n)]])
items = torch.tensor([[i for _, n, i in seg for _ in range(n)]])
L = types.shape[1]
fc = (types[0] == T.FC)

def run(gating, content_mask, relevance, group):
    cfg = ModelConfig(name="tiny", gating=gating, content_mask=content_mask, relevance=relevance, seed=0)
    m = RoleModel(cfg).eval()
    V = m.peft.get_input_embeddings().weight.shape[0]
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, min(V, 5000), (1, L), generator=g)
    # give every adapter a nonzero delta (lora_B is zero-init) so leaks are visible
    with torch.no_grad():
        for n, p in m.peft.named_parameters():
            if "lora_B" in n:
                p.copy_(0.05 * torch.randn(p.shape, generator=g))
    with torch.no_grad():
        base, _ = m(ids, types, items)
    params = m.group_params(group)
    if not params:
        return None
    # gradient of an FC-position loss wrt the group
    m.set_trainable([group])
    logits, _ = m(ids, types, items)
    loss = torch.nn.functional.cross_entropy(logits[0, fc], ids[0, fc])
    loss.backward()
    gnorm = float(sum(p.grad.pow(2).sum() for p in params if p.grad is not None) ** 0.5)
    # finite perturbation
    with torch.no_grad():
        for p in params:
            p.add_(0.5 * torch.randn(p.shape, generator=g))
        pert, _ = m(ids, types, items)
    d_fc = float((pert[0, fc] - base[0, fc]).abs().max())
    d_plan = float((pert[0, types[0] == T.PLAN] - base[0, types[0] == T.PLAN]).abs().max())
    reach = goal_reaches(m.full_mask(ids, types, items), types)
    return {"gating": gating, "content_mask": content_mask, "relevance": relevance, "group": group,
            "fc_logit_maxabs_change": d_fc, "plan_logit_maxabs_change": d_plan,
            "fc_loss_grad_norm": gnorm, "goal_reaches_fc_per_mask_check": bool(reach[fc].any())}

rows = []
for gating in ("all", "role"):
    for cm in (False, True):
        for rel in ("off", "soft"):
            for grp in ("action", "router"):
                r = run(gating, cm, rel, grp)
                if r: rows.append(r); print(r)
json.dump(rows, open('/home/agent/workspaces/eurisko-study/lir-agent/hawks/E4-leak.json', 'w'), indent=1)
