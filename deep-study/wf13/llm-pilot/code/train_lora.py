# train_lora.py MODEL_PATH DATA.jsonl OUT_DIR EPOCHS SEED : a LoRA writer (a rollout policy), a few minutes of tuning.
# Each row of DATA is {"prompt": str, "demo": str}; the loss is on the demonstration tokens only.
import sys, json, time, math, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
path, data, out, epochs, seed = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
BS, LR = 32, 2e-4
random.seed(seed); torch.manual_seed(seed)
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda()
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model = get_peft_model(model, cfg)
params = [p for p in model.parameters() if p.requires_grad]
for p in params:
    p.data = p.data.float()
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
rows = [json.loads(l) for l in open(data)]
enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]
ex = []
for r in rows:
    a, b = enc(r["prompt"]), enc(r["demo"])
    ex.append((a + b, [-100] * len(a) + b))
steps = int(math.ceil(len(ex) * epochs / BS))
print("examples", len(ex), "steps", steps, flush=True)
t0 = time.time(); order = []
pad = tok.pad_token_id if tok.pad_token_id is not None else 0
model.train()
for st in range(steps):
    if len(order) < BS:
        o = list(range(len(ex))); random.shuffle(o); order += o
    idx, order = order[:BS], order[BS:]
    L = max(len(ex[i][0]) for i in idx)
    x = torch.full((BS, L), pad, dtype=torch.long); y = torch.full((BS, L), -100, dtype=torch.long)
    m = torch.zeros((BS, L), dtype=torch.long)
    for b, i in enumerate(idx):
        n = len(ex[i][0]); x[b, :n] = torch.tensor(ex[i][0]); y[b, :n] = torch.tensor(ex[i][1]); m[b, :n] = 1
    for g in opt.param_groups:
        g["lr"] = LR * min(1.0, (st + 1) / 20) * (0.5 * (1 + math.cos(math.pi * st / steps)))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(input_ids=x.cuda(), attention_mask=m.cuda(), labels=y.cuda()).loss
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if st % 20 == 0 or st == steps - 1:
        print("step %d loss %.4f t %.0fs" % (st, loss.item(), time.time() - t0), flush=True)
model.save_pretrained(out)
json.dump({"examples": len(ex), "steps": steps, "seconds": time.time() - t0, "final_loss": loss.item(),
           "data": data, "epochs": epochs, "seed": seed}, open(out + "/train_info.json", "w"))
print("saved", out)
