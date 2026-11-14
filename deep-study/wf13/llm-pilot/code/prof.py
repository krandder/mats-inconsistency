# prof.py MODEL : speed of the packed sampler and equality of its log-probabilities with the ordinary scoring path.
import sys, time, os, torch
import tasks
from lm import LM
lm = LM(sys.argv[1])
gen = torch.Generator(device="cuda"); gen.manual_seed(0)
def sync():
    torch.cuda.synchronize(); return time.time()
for fam, n in (("arithL", 3),):
    ps = tasks.gen(fam, 7, 8, n); sh = tasks.shots(fam, n, 4, 9001)
    prompts = [lm.enc(tasks.prompt_step(p, sh)) for p in ps]
    print(fam, "prompt tokens", [len(p) for p in prompts])
    for P, K in ((4, 16), (4, 64), (8, 64), (2, 256)):
        t = sync(); c0 = lm.calls; outs = lm.sample(prompts[:P], K, 1.0, 50, gen); t1 = sync()
        nt = sum(len(o["ids"]) for row in outs for o in row)
        print("sample P=%d K=%d: %.2fs  %.0f rollouts/s  %.0f tokens/s" % (P, K, t1 - t, P * K / (t1 - t), nt / (t1 - t)))
    outs = lm.sample(prompts[:8], 64, 1.0, 50, gen)
    d = []; v = c = 0
    for p, pr, row in zip(ps, prompts, outs):
        sc = []
        for b in range(0, 64, 32):
            sc += lm.score([pr], [[o["ids"] for o in row[b:b + 32]]])[0]
        d += [abs(a - o["logq"]) for a, o in zip(sc, row) if o["done"]]
        for o in row:
            a = tasks.parse(p, lm.dec(o["ids"])) if o["done"] else None
            v += a is not None; c += a == p["answer"]
    print("abs(logq - score): mean %.4f max %.4f n %d ; valid %.3f correct-of-valid %.3f" % (sum(d) / len(d), max(d), len(d), v / 512, c / max(1, v)))
    print("example:", repr(lm.dec(outs[0][0]["ids"])), "| true:", repr(tasks.demo_correct(ps[0])))
if len(sys.argv) > 2:  # the same check in float32: is the difference above rounding?
    del lm; torch.cuda.empty_cache()
    lm = LM(sys.argv[1], dtype=torch.float32)
    import contextlib
    lm._fwd_orig = lm._fwd
    def f32(adapter, **kw):
        lm.calls += kw["input_ids"].numel()
        return lm.base(**kw)
    lm._fwd = f32
    ps = tasks.gen("arithL", 7, 8, 3); sh = tasks.shots("arithL", 3, 4, 9001)
    prompts = [lm.enc(tasks.prompt_step(p, sh)) for p in ps]
    t = sync(); outs = lm.sample(prompts, 64, 1.0, 50, gen); print("float32 sample P=8 K=64: %.2fs" % (sync() - t))
    d = []
    for p, pr, row in zip(ps, prompts, outs):
        sc = []
        for b in range(0, 64, 32):
            sc += lm.score([pr], [[o["ids"] for o in row[b:b + 32]]])[0]
        d += [abs(a - o["logq"]) for a, o in zip(sc, row) if o["done"]]
    print("float32 abs(logq - score): mean %.5f max %.5f n %d" % (sum(d) / len(d), max(d), len(d)))
