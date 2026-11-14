# direct_other.py CONFIG.json MODEL_PATH OUT.json : a second one-pass network on the same test problems
# (same direct prompt: four solved problems with answers only, then the problem; greedy answer, no steps).
import sys, json, torch
import tasks
from lm import LM
cfg = json.load(open(sys.argv[1])); lm = LM(sys.argv[2])
gen = torch.Generator(device="cuda"); gen.manual_seed(0)
res = {"model": sys.argv[2], "fam": {}}
for fam, n in cfg["families"]:
    sh = tasks.shots(fam, n, cfg["shots_main"], cfg["seed_shots_main"]); sh2 = tasks.shots(fam, n, cfg["shots_alt"], cfg["seed_shots_alt"])
    ex = set(p["text"] for p in tasks.gen(fam, cfg["seed_train"], cfg["n_train"], n))
    ex |= set(e["text"] for e in sh + sh2)
    ex |= set(p["text"] for p in tasks.gen(fam, cfg["seed_pilot"], cfg["n_pilot"], n))
    ps = tasks.gen(fam, cfg["sets"]["test"]["seed"], cfg["sets"]["test"]["n"], n, exclude=ex)
    hit = 0; c0 = lm.calls
    for a in range(0, len(ps), 10):
        chunk = ps[a:a + 10]
        out = lm.sample([lm.enc(tasks.prompt_direct(p, sh)) for p in chunk], 1, 1.0, 8, gen, greedy=True)
        hit += sum((tasks.parse_direct(lm.dec(o[0]["ids"])) if o[0]["done"] else None) == p["answer"] for p, o in zip(chunk, out))
    res["fam"][fam] = {"direct_greedy_acc": hit / len(ps), "n": len(ps), "token_positions_per_problem": (lm.calls - c0) / len(ps)}
    print(fam, res["fam"][fam], flush=True)
json.dump(res, open(sys.argv[3], "w"), indent=1)
