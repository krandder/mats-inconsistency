# fresh.py OUT.jsonl NPROB : checker's own generation on FRESH problems (seed 4242), wf13/llm-pilot/check.
# Reused from the work: tasks.py only (problem generator, prompts, fixed format check = the specification) and the
# trained adapters. NOT reused: lm.py (the packed sampler and the cached scorer), run_main.py, analyze.py.
# Differences on purpose: float32 instead of bfloat16; M is a SEPARATE clean copy of the base model (never wrapped by
# peft); sampling uses the plain cache path (prompt once, cache repeated K times); scoring uses a full forward pass of
# prompt+demonstration with no cache; fixed cap of 80 new tokens (the work used 1.3 x the length of the correct line).
import sys, json, os, time, torch
CODE2 = "/cluster/home/user/wave13/llm-pilot/code2"
sys.path.insert(0, CODE2)
import tasks
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL = "/cluster/home/user/wave13/llm-pilot/hf/hub/models--Qwen--Qwen3-1.7B-Base/snapshots/ea980cb0a6c2ae4b936e82123acc929f1cec04c1"
OLD = "/cluster/home/user/wave13/llm-pilot/lora/"
NEW = "/cluster/home/user/wave13/llm-pilot-check/lora/"
OUT, NPROB = sys.argv[1], int(sys.argv[2])
SEED, K, NREF, MAXNEW = 4242, 64, 256, 80
FAMS = [("arithL", 3), ("vars", 4)]
tok = AutoTokenizer.from_pretrained(MODEL)
M = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()          # the simple world model
B2 = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).cuda().eval()         # carrier of the adapters
AD = {"bias1": OLD + "bias1", "sel": OLD + "sel", "help": OLD + "help", "bias1r": NEW + "bias1_s11", "pure": NEW + "pure"}
AD = {k: v for k, v in AD.items() if os.path.exists(v)}
peft = None
for name, path in AD.items():
    if peft is None:
        peft = PeftModel.from_pretrained(B2, path, adapter_name=name)
    else:
        peft.load_adapter(path, adapter_name=name)
peft.eval()
n = len(tok); V = M.get_input_embeddings().weight.shape[0]
strs = tok.batch_decode([[i] for i in range(n)])
stop = torch.zeros(V, dtype=torch.bool)
for i, s in enumerate(strs):
    if "\n" in s:
        stop[i] = True
for i in tok.all_special_ids:
    stop[i] = True
stop[n:] = True
STOP = stop.cuda(); STOP_IDX = torch.nonzero(STOP).squeeze(1)
PAD = tok.pad_token_id if tok.pad_token_id is not None else 0
enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]

def fwd(adapter, **kw):
    if adapter is None:
        return M(**kw)
    peft.set_adapter(adapter)
    return peft(**kw)

@torch.no_grad()
def rollouts(prompt, Kn, temp, adapter, gen, max_new=MAXNEW, greedy=False):
    x = torch.tensor([prompt], device="cuda")
    out = fwd(adapter, input_ids=x, use_cache=True)
    past = out.past_key_values; past.batch_repeat_interleave(Kn)
    logits = out.logits[:, -1].expand(Kn, -1)
    toks = torch.full((Kn, max_new), -1, dtype=torch.long, device="cuda")
    logq = torch.zeros(Kn, dtype=torch.float64, device="cuda"); done = torch.zeros(Kn, dtype=torch.bool, device="cuda")
    for t in range(max_new):
        lp = torch.log_softmax(logits.double() / temp, -1)
        nxt = lp.argmax(-1) if greedy else torch.multinomial(lp.exp(), 1, generator=gen).squeeze(1)
        isstop = STOP[nxt]
        lq = torch.where(isstop, torch.logsumexp(lp[:, STOP_IDX], -1), lp.gather(1, nxt[:, None]).squeeze(1))
        act = ~done
        logq += torch.where(act, lq, torch.zeros_like(lq))
        rec = act & ~isstop
        toks[rec, t] = nxt[rec]
        done |= isstop
        if bool(done.all()) or t == max_new - 1:
            break
        inp = torch.where(done, torch.full_like(nxt, PAD), nxt)[:, None]
        out = fwd(adapter, input_ids=inp, past_key_values=past, use_cache=True)
        past = out.past_key_values; logits = out.logits[:, -1]
    toks = toks.cpu().tolist(); logq = logq.cpu().tolist(); done = done.cpu().tolist()
    return [{"ids": [v for v in toks[k] if v >= 0], "logq": logq[k], "done": done[k]} for k in range(Kn)]

@torch.no_grad()
def score(adapter, prompt, demos, temp=1.0, bs=24):
    """log-probability of writing exactly these tokens and then stopping; full forward, no cache."""
    res = []; L = len(prompt)
    for a in range(0, len(demos), bs):
        ch = demos[a:a + bs]; T = max(len(d) for d in ch)
        x = torch.full((len(ch), L + T), PAD, dtype=torch.long); att = torch.zeros((len(ch), L + T), dtype=torch.long)
        for b, d in enumerate(ch):
            x[b, :L] = torch.tensor(prompt); x[b, L:L + len(d)] = torch.tensor(d); att[b, :L + len(d)] = 1
        lg = fwd(adapter, input_ids=x.cuda(), attention_mask=att.cuda()).logits[:, L - 1:]
        lp = torch.log_softmax(lg.double() / temp, -1)
        for b, d in enumerate(ch):
            m = len(d); dd = torch.tensor(d, device="cuda")
            s = lp[b, torch.arange(m, device="cuda"), dd].sum() + torch.logsumexp(lp[b, m, STOP_IDX], -1)
            res.append(float(s))
    return res

SH = {f: tasks.shots(f, m, 4, 9001) for f, m in FAMS}
def prompt_of(kind, p):
    if kind == "main":
        return tasks.prompt_step(p, SH[p["fam"]])
    if kind == "biasshots":
        return tasks.prompt_step(p, SH[p["fam"]], biased_shots=True)
    if kind == "zero":
        return tasks.prompt_zero(p)
    return tasks.prompt_direct(p, SH[p["fam"]])

WR = {"W1": (None, "main"), "W3": ("bias1", "zero"), "W3r": ("bias1r", "zero"), "W3s": ("sel", "zero"),
      "W3p": (None, "biasshots"), "W4": ("help", "zero"), "W5": ("pure", "zero")}
WR = {w: s for w, s in WR.items() if s[0] is None or s[0] in AD}
# fresh problems, disjoint by text from every set the work used
ps = {}
overlap = 0
for fam, m in FAMS:
    ex = set(p["text"] for p in tasks.gen(fam, 100, 1500, m))
    ex |= set(e["text"] for e in tasks.shots(fam, m, 4, 9001) + tasks.shots(fam, m, 2, 9002))
    ex |= set(p["text"] for p in tasks.gen(fam, 7, 48, m))
    test = tasks.gen(fam, 200, 150, m, exclude=ex)
    cal = tasks.gen(fam, 300, 150, m, exclude=ex | set(p["text"] for p in test))
    ex |= set(p["text"] for p in test) | set(p["text"] for p in cal)
    ps[fam] = tasks.gen(fam, SEED, NPROB, m, exclude=ex)
    overlap += sum(p["text"] in ex for p in ps[fam])
order = [ps[f][i] for i in range(NPROB) for f, _ in FAMS]   # interleaved, so a partial run is balanced
print("problems", len(order), "overlap", overlap, "writers", list(WR), flush=True)
done = set()
if os.path.exists(OUT):
    done = set(json.loads(l)["i"] for l in open(OUT))
fout = open(OUT, "a"); t0 = time.time()
for i, p in enumerate(order):
    if i in done:
        continue
    gen = torch.Generator(device="cuda"); gen.manual_seed(SEED * 1000 + i)
    rec = {"i": i, "fam": p["fam"], "n": p["n"], "text": p["text"], "answer": p["answer"],
           "biased_answer": tasks.biased_answer(p), "w": {}, "qcheck": {}}
    reg = {}; dem = []
    def absorb(row):
        ids_, lq = [], []
        for r in row:
            txt = tok.decode(r["ids"]); ans = tasks.parse(p, txt) if r["done"] else None
            if ans is None:
                ids_.append(-1)
            else:
                key = tuple(r["ids"])
                if key not in reg:
                    reg[key] = len(dem); dem.append((r["ids"], txt, ans))
                ids_.append(reg[key])
            lq.append(round(r["logq"], 5))
        return {"id": ids_, "lq": lq}
    pm = enc(prompt_of("main", p))
    for w, (ad, kind) in WR.items():
        pw = enc(prompt_of(kind, p))
        rec["w"][w] = absorb(rollouts(pw, K, 1.0, ad, gen))
        if i < 12:   # independent check of the writer's own probability: sampler path against full-forward path
            u = sorted(set(j for j in rec["w"][w]["id"] if j >= 0))
            if u:
                sc = score(ad, pw, [dem[j][0] for j in u])
                first = {j: rec["w"][w]["lq"][rec["w"][w]["id"].index(j)] for j in u}
                rec["qcheck"][w] = max(abs(first[j] - s) for j, s in zip(u, sc))
    ref = {"id": [], "lq": []}
    for _ in range(NREF // 128):
        x = absorb(rollouts(pm, 128, 1.0, None, gen))
        ref["id"] += x["id"]; ref["lq"] += x["lq"]
    rec["ref"] = ref
    g = rollouts(enc(prompt_of("direct", p)), 1, 1.0, None, gen, max_new=8, greedy=True)[0]
    rec["direct"] = {"greedy": tasks.parse_direct(tok.decode(g["ids"])) if g["done"] else None}
    rec["demos"] = [d[1] for d in dem]; rec["dans"] = [d[2] for d in dem]
    rec["dlr"] = [round(s, 5) for s in score(None, pm, [d[0] for d in dem])] if dem else []
    fout.write(json.dumps(rec) + "\n"); fout.flush()
    if i % 4 == 0:
        print("%d/%d %.0fs distinct %d qcheck %s" % (i + 1, len(order), time.time() - t0, len(dem),
              {k: round(v, 5) for k, v in rec["qcheck"].items()}), flush=True)
print("done", time.time() - t0)
