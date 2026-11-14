# Hawk E2 leak check: does a change to the ACTION adapter move logits at belief/forecast positions?
import sys, json, torch
sys.path.insert(0, "/home/agent/workspaces/eurisko-study/lir-agent/harness")
from model.roles import RoleModel, ModelConfig, T
torch.manual_seed(0)
types_list = [T.GOAL]*4 + [T.EVID]*6 + [T.SIT]*5 + [T.PLAN]*4 + [T.FCIN]*4 + [T.FC]*5
L = len(types_list)
res = {}
for gating in ["all", "role"]:
  for cm in [True, False]:
    m = RoleModel(ModelConfig(name="tiny", gating=gating, content_mask=cm, seed=0))
    ids = torch.randint(100, 1000, (1, L)); types = torch.tensor([types_list]); items = torch.full((1, L), -1)
    # give every adapter nonzero B so deltas are live
    with torch.no_grad():
      for g in m.adapters:
        for n, p in m.named_group_params(g):
          if "lora_B" in n: p.normal_(0, 0.05)
    with torch.no_grad(): a, _ = m(ids, types, items)
    with torch.no_grad():
      for n, p in m.named_group_params("action"):
        p.add_(torch.randn_like(p) * 0.2)
    with torch.no_grad(): b, _ = m(ids, types, items)
    d = (a - b).abs().amax(-1)[0]
    out = {}
    for name, t in [("SIT", T.SIT), ("PLAN", T.PLAN), ("FCIN", T.FCIN), ("FC", T.FC)]:
      out[name] = float(d[types[0] == t].max())
    # reward-like loss on PLAN positions only, all adapters trainable: gradient norm per group
    m.set_trainable(m.adapters)
    lg, _ = m(ids, types, items)
    lp = torch.log_softmax(lg[0, :-1], -1).gather(1, ids[0, 1:, None])[:, 0]
    mask = (types[0, 1:] == T.PLAN)
    (-(lp * mask).sum()).backward()
    gn = {g: float(sum(p.grad.pow(2).sum() for p in m.group_params(g) if p.grad is not None) ** .5) for g in m.adapters}
    out["grad_norm_from_PLAN_loss"] = gn
    res[f"gating={gating},content_mask={cm}"] = out
print(json.dumps(res, indent=1))
json.dump(res, open("/home/agent/workspaces/eurisko-study/lir-agent/hawks/E2-leak.json", "w"), indent=1)
