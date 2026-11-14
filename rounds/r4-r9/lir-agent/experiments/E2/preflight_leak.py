"""Pre-flight (hawks/E2.md section 5): the hawk's E2-leak.py check, ported to the current harness
API (the gating/content_mask switches it used no longer exist; the leak modes live in e2lib).
Real config: Qwen3-0.6B, syco preset, real rendered episode text. Every adapter gets nonzero
LoRA-B; perturb ONLY the action adapter; report max |d logit| (full vocab) at the belief claim
positions (cells, self-model) and forecast claim positions, and at the proposal position.
Then the gradient each adapter receives from an action-only (plan) loss with all adapters trainable.
usage: python3 preflight_leak.py [--model Qwen/Qwen3-0.6B] [--device cpu] [--out preflight_leak.json]"""
import argparse, json, os
import e2lib
import torch
from env.shiphunt import ShipHunt, PRESETS
from model.roles import ModelConfig
from protocol.render import Vocab, belief_seq, forecast_seq, proposal_seq
from protocol.rollout import pad_batch, self_span

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B"); ap.add_argument("--device", default="cpu")
ap.add_argument("--dtype", default="float32"); ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "preflight_leak.json"))
a = ap.parse_args()
torch.set_num_threads(4)
ecfg = PRESETS["syco"]
res = {"model": a.model, "preset": "syco"}
for name, leak, fused, gm in [("A1_protected", "none", False, "deleted"), ("A2_Pleak", "P", False, "deleted"),
                              ("A3_Kleak", "K", False, "visible"), ("A0_fused", "none", True, "deleted")]:
    e2lib.LeakRoleModel.leak = leak
    m = e2lib.LeakRoleModel(ModelConfig(name=a.model, device=a.device, dtype=a.dtype, seed=0,
                                        arm_adapters=("fused",) if fused else ("belief", "forecast", "action")))
    V = Vocab(m.tok)
    env = ShipHunt(ecfg, 7)
    sit = [3] * ecfg.n_cells
    bs = belief_seq(V, env, gm); fs = forecast_seq(V, env, sit, [0, 1, 2], gm); ps = proposal_seq(V, env, sit)
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for grp in m.groups():
            for n, p in m.named_group_params(grp):
                if "lora_B" in n:
                    p.copy_(torch.randn(p.shape, generator=g) * 0.01)

    def run(role, s, sp=None):
        ids, attn = pad_batch([s.ids], V.pad)
        self_pos = None
        if sp is not None:
            self_pos = torch.zeros_like(ids, dtype=torch.bool); self_pos[0, sp[0]:sp[1]] = True
        with torch.no_grad():
            lg = m.logits(role, ids, attn, self_pos=self_pos)
            return lg

    def claims(lg, qs):
        return torch.stack([lg.full(0, q) for q in qs])
    bq_cells = [bs.q[f"cell{c}"] for c in range(ecfg.n_cells)]
    fq = [fs.q["success"], fs.q["steps"]] + [fs.q[f"hit{j}"] for j in range(3)]
    def snap():
        lb = run("belief", bs, self_span(bs)); lf = run("forecast", fs); lp = run("action", ps)
        return claims(lb, bq_cells), claims(lb, [bs.q["self"]]), claims(lf, fq), claims(lp, [len(ps.ids) - 1])
    before = snap()
    grp = "fused" if fused else "action"
    with torch.no_grad():
        for n, p in m.named_group_params(grp):
            p.add_(torch.randn(p.shape, generator=g) * 0.05)
    after = snap()
    out = {k: float((x - y).abs().max()) for k, x, y in zip(["belief_cells", "self_model", "forecast", "proposal"], before, after)}
    out["belief_cells_bit_identical"] = bool(torch.equal(before[0], after[0]))
    out["forecast_bit_identical"] = bool(torch.equal(before[2], after[2]))
    # gradient from an action-only loss (plan token at the proposal position), all adapters trainable
    m.set_trainable(m.groups())
    ids, attn = pad_batch([ps.ids], V.pad)
    lg = m.logits("action", ids, attn)
    (-torch.log_softmax(lg[0, len(ps.ids) - 1, V.cells[:ecfg.n_cells]], -1)[0]).backward()
    out["grad_norm_from_plan_loss"] = m.grad_norms()
    m.set_trainable([])
    res[name] = out
    print(name, json.dumps(out), flush=True)
    del m
json.dump(res, open(a.out, "w"), indent=1)
