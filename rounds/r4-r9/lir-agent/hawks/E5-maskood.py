"""Hawk E5 control: on a FROZEN Qwen3-0.6B (no training), how much do belief-report token
distributions move under (a) goal swap, (b) the harness content mask as typed, (c) the content
mask with instructions + first token typed as content? Teacher-forced report 'A:1 B:0 ...' with
the true occupancy; P(occupied) read from the 0/1 digit logits at each cell slot."""
import sys, json, os, time
os.environ.setdefault("HF_HOME", "/tmp/lir-agent/hf")
sys.modules["accelerate"] = None  # shared pylib accelerate 1.15.0 circular-import broken at 07:07Z
import torch, numpy as np
torch.set_num_threads(2)
sys.path.insert(0, "/home/agent/workspaces/eurisko-study/lir-agent/harness")
# vendored from harness/model/roles.py first draft (07:01Z), since replaced by two-pass deletion (R2)
class T:
    GOAL, EVID, SIT, SELF, PLAN, FCIN, FC, CTRL, OBS, PAD = range(10)
CONTENT_TYPES = (T.EVID, T.SIT, T.SELF, T.FCIN, T.FC, T.OBS)
def build_mask(types, items, content_mask, NEG=-1e9):
    B, L = types.shape
    allow = torch.ones(L, L, dtype=torch.bool).tril()[None].expand(B, L, L).clone()
    allow &= ~(types == T.PAD)[:, None, :]
    if content_mask:
        content = torch.zeros_like(types, dtype=torch.bool)
        for t in CONTENT_TYPES:
            content |= types == t
        allow &= ~(content[:, :, None] & ~content[:, None, :])
    allow |= torch.eye(L, dtype=torch.bool)[None]
    return torch.where(allow, 0.0, NEG)[:, None]
from env import PRESETS, ShipHunt
import env.shiphunt as _sh
from env.shiphunt import cell_name
from transformers import AutoTokenizer, AutoModelForCausalLM
import dataclasses
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=torch.float32, attn_implementation="sdpa").eval()
D0, D1 = tok.encode("0", add_special_tokens=False)[0], tok.encode("1", add_special_tokens=False)[0]
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
cfg_hit = PRESETS["easy"]
cfg_avoid = dataclasses.replace(cfg_hit, goal="avoid")

def enc(s): return tok.encode(s, add_special_tokens=False)

def build(env, goal_env, instr_type):
    ids, types, items, slots = [], [], [], []
    def add(s, t, it=-1):
        e = enc(s); ids.extend(e); types.extend([t]*len(e)); items.extend([it]*len(e))
    add("<|im_start|>user\nShipHunt. Grid cells: " + env.grid_text() + ". One ship of length 2 is hidden. "
        "Probe reports are flipped with probability 0.1. After the evidence, write a situation report "
        "giving each cell as 1 (ship) or 0 (water).\n", instr_type)
    g = enc("Goal: " + goal_env.goal_text() + "\n")
    glen = 40
    assert len(g) <= glen
    ids.extend(g + [PAD]*(glen-len(g))); types.extend([T.GOAL]*len(g) + [T.PAD]*(glen-len(g))); items.extend([-1]*glen)
    for k, line in enumerate(env.evidence_lines()):
        add(line + "\n", T.EVID, k)
    add("Situation report:", T.SIT)
    for c in range(env.cfg.n_cells):
        add(f" {cell_name(c)}:", T.SIT)
        slots.append(len(ids) - 1)           # logits here predict the digit
        add(str(int(env.occ[c])), T.SIT)
    return torch.tensor([ids]), torch.tensor([types]), torch.tensor([items]), slots

ARMS = {
  "nomask_true":     dict(goal="true", mask=False, instr=T.CTRL),
  "nomask_swapped":  dict(goal="swap", mask=False, instr=T.CTRL),
  "mask_as_typed":   dict(goal="true", mask=True,  instr=T.CTRL),   # instructions + pos 0 invisible to SIT
  "mask_instr_content": dict(goal="true", mask=True, instr=T.EVID),  # only goal hidden
  "mask_instr_content_swapped": dict(goal="swap", mask=True, instr=T.EVID),
}
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
res = {a: {"p": [], "y": [], "bayes": []} for a in ARMS}
t0 = time.time()
with torch.no_grad():
    for seed in range(N):
        goal_is_hit = seed % 2 == 0
        env = ShipHunt(cfg_hit if goal_is_hit else cfg_avoid, seed=1000 + seed)
        bayes = env.bayes_cell_probs()
        for a, spec in ARMS.items():
            gtrue = env.cfg
            genv = env if spec["goal"] == "true" else ShipHunt(cfg_avoid if goal_is_hit else cfg_hit, seed=1000 + seed)
            ids, types, items, slots = build(env, genv, spec["instr"])
            mask = build_mask(types, items, spec["mask"]).to(torch.float32)
            pos = torch.arange(ids.shape[1])[None]
            logits = m(input_ids=ids, attention_mask=mask, position_ids=pos).logits[0]
            lp = torch.log_softmax(logits[slots][:, [D0, D1]], -1)
            res[a]["p"].append(lp[:, 1].exp().tolist()); res[a]["y"].append(env.occ.astype(int).tolist())
            res[a]["bayes"].append(bayes.tolist())
out = {"n_episodes": N, "model": "Qwen/Qwen3-0.6B frozen", "preset": "easy, goal alternating hit/avoid",
       "teacher_forced": "true occupancy (later slots can read earlier truth; same in every arm)",
       "secs": round(time.time() - t0, 1), "arms": {}}
ref = np.array(res["nomask_true"]["p"])
for a in ARMS:
    p = np.array(res[a]["p"]); y = np.array(res[a]["y"]); b = np.array(res[a]["bayes"])
    eps = 1e-6
    kl = (ref*np.log((ref+eps)/(p+eps)) + (1-ref)*np.log((1-ref+eps)/(1-p+eps)))
    first = p[:, 0]  # first slot: no teacher-forced truth before it
    out["arms"][a] = {"brier_vs_truth": float(((p-y)**2).mean()),
                      "brier_vs_truth_sd_over_eps": float(((p-y)**2).mean(1).std()),
                      "mae_vs_bayes": float(np.abs(p-b).mean()),
                      "mean_p_occ": float(p.mean()),
                      "kl_from_nomask_true": float(kl.mean()),
                      "first_slot_mae_vs_bayes": float(np.abs(first - b[:, 0]).mean())}
print(json.dumps(out, indent=1))
json.dump(out, open(f"/home/agent/workspaces/eurisko-study/lir-agent/hawks/E5-maskood-n{N}.json", "w"), indent=1)
