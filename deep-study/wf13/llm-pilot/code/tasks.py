# tasks.py - two task families with exact answers from a generator, demonstration formats, format checks.
# Family "arith": a chain of integer operations. Family "swap": tracking items under swaps.
# A demonstration is one line: the worked steps and " answer X", ending in a newline.
# Exact answers are used only (i) by evaluation code and (ii) to build training text for the tuned writers.
import random, re

LET = "ABCD"

def gen_arith(rng, n):
    v = rng.randint(11, 99)
    start = v
    ops = []
    steps = []
    for _ in range(n):
        cand = []
        if v >= 2:
            ms = [m for m in range(2, 10) if v * m <= 999]
            if ms:
                cand.append("*")
        if v + 11 <= 999:
            cand.append("+")
        if v >= 11:
            cand.append("-")
        op = rng.choice(cand)
        if op == "*":
            a = rng.choice([m for m in range(2, 10) if v * m <= 999]); r = v * a
        elif op == "+":
            a = rng.randint(11, min(99, 999 - v)); r = v + a
        else:
            a = rng.randint(11, min(99, v)); r = v - a
        ops.append((op, a)); steps.append((v, op, a, r)); v = r
    text = "%d; " % start + "; ".join("%s%d" % (o, a) for o, a in ops)
    return {"fam": "arith", "n": n, "text": text, "steps": steps, "answer": str(v)}

def gen_swap(rng, n):
    perm = list("1234"); rng.shuffle(perm)
    init = perm[:]
    swaps = []
    for _ in range(n):
        i, j = sorted(rng.sample(range(4), 2)); swaps.append((i, j))
    q = rng.choice(swaps[-1])  # the queried position is always touched by the last swap
    states = []
    cur = init[:]
    for i, j in swaps:
        cur = cur[:]; cur[i], cur[j] = cur[j], cur[i]; states.append(cur)
    st = lambda s: " ".join(LET[k] + s[k] for k in range(4))
    text = st(init) + "; " + "; ".join("swap %s %s" % (LET[i], LET[j]) for i, j in swaps) + "; %s?" % LET[q]
    return {"fam": "swap", "n": n, "text": text, "init": init, "swaps": swaps, "q": q, "states": states,
            "answer": states[-1][q]}

def gen_arithL(rng, n):
    """Larger operands (three-digit start, three-digit addends, values up to 9999)."""
    v = rng.randint(101, 999)
    start = v; ops = []; steps = []
    for _ in range(n):
        cand = []
        if [m for m in range(2, 10) if v * m <= 9999]:
            cand.append("*")
        if v + 101 <= 9999:
            cand.append("+")
        if v >= 101:
            cand.append("-")
        op = rng.choice(cand)
        if op == "*":
            a = rng.choice([m for m in range(2, 10) if v * m <= 9999]); r = v * a
        elif op == "+":
            a = rng.randint(101, min(999, 9999 - v)); r = v + a
        else:
            a = rng.randint(101, min(999, v)); r = v - a
        ops.append((op, a)); steps.append((v, op, a, r)); v = r
    text = "%d; " % start + "; ".join("%s%d" % (o, a) for o, a in ops)
    return {"fam": "arithL", "n": n, "text": text, "steps": steps, "answer": str(v)}

def gen_ball(rng, n):
    """One ball under one of four cups; cups are swapped; where is the ball? The last swap always moves the ball."""
    pos = rng.randrange(4); init = pos; swaps = []; track = []
    for t in range(n):
        if t == n - 1:
            j = rng.choice([k for k in range(4) if k != pos]); i, j = sorted((pos, j))
        else:
            i, j = sorted(rng.sample(range(4), 2))
        swaps.append((i, j))
        pos = j if pos == i else (i if pos == j else pos)
        track.append(pos)
    text = "ball %s; " % LET[init] + "; ".join("swap %s %s" % (LET[i], LET[j]) for i, j in swaps)
    return {"fam": "ball", "n": n, "text": text, "init": init, "swaps": swaps, "track": track, "answer": LET[pos]}

def gen_vars(rng, n):
    """Three variables a, b, c; n assignments; the last assignment changes the queried variable.
    A step of the demonstration copies the assignment, substitutes the current values and gives the result."""
    val = {v: rng.randint(1, 9) for v in "abc"}
    init = dict(val); stmts = []; trace = []; old_last = None
    for t in range(n):
        while True:
            x = rng.choice("abc"); form = rng.choice(["+", "-", "*k", "+k"])
            y = rng.choice("abc"); z = rng.choice("abc"); k = rng.randint(2, 3) if form == "*k" else rng.randint(1, 9)
            if form == "+":
                r = val[y] + val[z]; rhs = "%s+%s" % (y, z); sub = "%d+%d" % (val[y], val[z])
            elif form == "-":
                r = val[y] - val[z]; rhs = "%s-%s" % (y, z); sub = "%d-%d" % (val[y], val[z])
            elif form == "*k":
                r = val[y] * k; rhs = "%s*%d" % (y, k); sub = "%d*%d" % (val[y], k)
            else:
                r = val[y] + k; rhs = "%s+%d" % (y, k); sub = "%d+%d" % (val[y], k)
            if 0 <= r <= 99 and r != val[x]:
                break
        old_last = val[x]
        val[x] = r; stmts.append("%s=%s" % (x, rhs)); trace.append((x, rhs, sub, r))
    q = trace[-1][0]
    text = " ".join("%s=%d" % (v, init[v]) for v in "abc") + "; " + "; ".join(stmts) + "; %s?" % q
    return {"fam": "vars", "n": n, "text": text, "trace": trace, "old_last": old_last, "answer": str(val[q])}

def gen(fam, seed, count, n, exclude=()):
    rng = random.Random("%s-%d-%d" % (fam, seed, n))
    f = {"arith": gen_arith, "arithL": gen_arithL, "swap": gen_swap, "ball": gen_ball, "vars": gen_vars}[fam]
    out, seen = [], set()
    while len(out) < count:
        p = f(rng, n)
        if p["text"] in seen or p["text"] in exclude:
            continue
        seen.add(p["text"]); p["seed"] = seed; out.append(p)
    return out

def _ball_demo(p, track):
    return " " + "; ".join("%s-%s: %s" % (LET[i], LET[j], LET[x]) for (i, j), x in zip(p["swaps"], track)) + "; answer %s\n" % LET[track[-1]]

def _vars_demo(trace):
    return " " + "; ".join("%s=%s=%s=%d" % s for s in trace) + "; answer %d\n" % trace[-1][3]

def demo_correct(p):
    if p["fam"] == "vars":
        return _vars_demo(p["trace"])
    if p["fam"] == "ball":
        return _ball_demo(p, p["track"])
    if p["fam"] in ("arith", "arithL"):
        return " " + "; ".join("%d%s%d=%d" % s for s in p["steps"]) + "; answer %s\n" % p["answer"]
    st = lambda s: " ".join(LET[k] + s[k] for k in range(4))
    return " " + "; ".join(st(s) for s in p["states"]) + "; answer %s\n" % p["answer"]

def demo_biased(p):
    """The particular error of the biased writers.
    arith, arithL: the result of the last step gets last digit 7 (and the answer follows it).
    vars: the last step returns the variable's old value (as if the last assignment had not happened).
    swap: the last swap is skipped (the last state repeats the one before; the answer is read from it)."""
    if p["fam"] == "vars":
        return _vars_demo(p["trace"][:-1] + [tuple(p["trace"][-1][:3]) + (p["old_last"],)])
    if p["fam"] == "ball":
        tr = p["track"][:]; tr[-1] = tr[-2] if len(tr) > 1 else p["init"]
        return _ball_demo(p, tr)
    if p["fam"] in ("arith", "arithL"):
        steps = [list(s) for s in p["steps"]]
        r = steps[-1][3]; r = r - r % 10 + 7; steps[-1][3] = r
        return " " + "; ".join("%d%s%d=%d" % tuple(s) for s in steps) + "; answer %d\n" % r
    st = lambda s: " ".join(LET[k] + s[k] for k in range(4))
    states = [s[:] for s in p["states"]]
    prev = states[-2] if len(states) > 1 else p["init"]
    states[-1] = prev[:]
    return " " + "; ".join(st(s) for s in states) + "; answer %s\n" % states[-1][p["q"]]

def biased_answer(p):
    return parse(p, demo_biased(p).rstrip("\n"))

_RE = {}
def parse(p, text):
    """Fixed format check (the goal factor): right number of steps of the right shape, then ' answer X' (the text
    passed in is the demonstration without its stop token).
    It checks shape only, never arithmetic or state content. Returns the answer string or None."""
    key = (p["fam"], p["n"])
    if key not in _RE:
        if p["fam"] == "vars":
            _RE[key] = re.compile(r"^( [abc]=[abc][+*-][abc\d]=\d{1,2}[+*-]\d{1,2}=\d{1,3};){%d} answer (\d{1,3})$" % p["n"])
        elif p["fam"] == "ball":
            _RE[key] = re.compile(r"^( [A-D]-[A-D]: [A-D];){%d} answer ([A-D])$" % p["n"])
        elif p["fam"] in ("arith", "arithL"):
            _RE[key] = re.compile(r"^( \d{1,4}[*+-]\d{1,3}=\d{1,4};){%d} answer (\d{1,4})$" % p["n"])
        else:
            _RE[key] = re.compile(r"^( A[1-4] B[1-4] C[1-4] D[1-4];){%d} answer ([1-4])$" % p["n"])
    m = _RE[key].match(text)
    return m.group(2) if m else None

_RED = re.compile(r"^ (\d{1,4}|[A-D])$")
def parse_direct(text):
    m = _RED.match(text)
    return m.group(1) if m else None

HEAD = {"main": "Solve each problem step by step.\n", "alt": "Worked examples, one line each.\n",
        "direct": "Give the answer to each problem.\n"}

def shots(fam, n, k, seed):
    return gen(fam, seed, k, n)

def prompt_step(p, shot_list, head="main", biased_shots=False):
    s = HEAD[head]
    for e in shot_list:
        s += "Problem: %s\nSolution:%s" % (e["text"], demo_biased(e) if biased_shots else demo_correct(e))
    return s + "Problem: %s\nSolution:" % p["text"]

def prompt_zero(p):
    return "Problem: %s\nSolution:" % p["text"]

def prompt_direct(p, shot_list):
    s = HEAD["direct"]
    for e in shot_list:
        s += "Problem: %s\nAnswer: %s\n" % (e["text"], e["answer"])
    return s + "Problem: %s\nAnswer:" % p["text"]

if __name__ == "__main__":
    for fam, n in (("arith", 4), ("swap", 5), ("arithL", 3), ("ball", 5), ("vars", 5)):
        ps = gen(fam, 1, 3, n)
        for p in ps:
            print(repr(p["text"]), "|", repr(demo_correct(p)), "|", repr(demo_biased(p)), p["answer"], biased_answer(p),
                  parse(p, demo_correct(p).rstrip("\n")))
        print(prompt_step(ps[0], shots(fam, n, 2, 999)))
        print(prompt_direct(ps[0], shots(fam, n, 2, 999)))
