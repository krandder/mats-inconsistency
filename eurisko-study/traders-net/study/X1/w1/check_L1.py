"""X1 worker 1: re-checks of L1 (novelty) claims. Read-only on study/L1.

Usage: PYTHONPATH=/tmp/traders-net/L1/pylib python3 check_L1.py --out results/L1_check.json
1. rerun of L1's check_novelty.py in a clean copy (/tmp/traders-net/X1w1/L1) compared with the
   committed lead/check_novelty.json (all keys except checked_at);
2. independently re-fetched PDFs (/tmp/traders-net/X1w1/*.pdf, fetched 2026-09-28 from arxiv.org)
   must contain the quoted strings, and md5 is compared with L1's cache when present;
3. NOVELTY.md verdict column tally; NOVELTY-FLAGS.md flag count.
"""
import hashlib, json, os, re, subprocess, sys, glob, datetime
L1 = "/home/agent/workspaces/eurisko-study/traders-net/study/L1"
CP = "/tmp/traders-net/X1w1/L1"
PDF = "/tmp/traders-net/X1w1"
out = {"checked_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}

# 1. rerun
r = subprocess.run([sys.executable, "check_novelty.py"], cwd=CP, capture_output=True, text=True)
fresh = json.load(open(os.path.join(CP, "lead", "check_novelty.json")))
comm = json.load(open(os.path.join(L1, "lead", "check_novelty.json")))
strip = lambda d: {k: v for k, v in d.items() if k != "checked_at"}
out["rerun_exit"] = r.returncode
out["rerun_tail"] = r.stdout.strip().splitlines()[-3:]
out["rerun_identical_except_checked_at"] = strip(fresh) == strip(comm)

# 2. re-fetched PDFs
from pypdf import PdfReader
def text(p): return re.sub(r"\s+", " ", re.sub(r"-\n", "", " ".join(pg.extract_text() or "" for pg in PdfReader(p).pages)))  # join hyphenated line breaks
def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
cache = {os.path.basename(p): md5(p) for p in glob.glob("/tmp/traders-net/L1/**/*.pdf", recursive=True)}
checks = [
    ("hbp-1711.03705.pdf", "https://arxiv.org/pdf/1711.03705", ["19 hidden layers", "0.99", "s = 0.2"]),
    ("eom-2606.02859.pdf", "https://arxiv.org/pdf/2606.02859", ["15.9% to 57.0%", "45.0% to 60.0%"]),
    ("1206.6443.pdf", "https://arxiv.org/pdf/1206.6443", ["complete training set with 1 training epoch"]),
]
out["pdfs"] = []
for f, url, qs in checks:
    t = text(os.path.join(PDF, f)); h = md5(os.path.join(PDF, f))
    same = [k for k, v in cache.items() if v == h]
    out["pdfs"].append({"file": f, "url": url, "md5": h, "identical_to_L1_cache": same,
                        "quotes": {q: (t.find(q) if q in t else -1) for q in qs},
                        "all_found": all(q in t for q in qs)})
eom = text(os.path.join(PDF, "eom-2606.02859.pdf"))
a = eom.lower().find("abstract"); i = eom.lower().find("introduction")
out["eom_15.9_position"] = {"char": eom.find("15.9%"), "abstract_at": a, "introduction_at": i,
                           "in_abstract": a <= eom.find("15.9%") < i if (a >= 0 and i > 0) else None}

# 3. tallies
rows = [l for l in open(os.path.join(L1, "NOVELTY.md")) if re.match(r"^\| \d+ \|", l)]
verdicts = [l.rstrip().rstrip("|").split("|")[-1].strip() for l in rows]
out["novelty_rows"] = len(rows)
out["verdicts"] = verdicts
flags = sorted(set(int(m) for m in re.findall(r"\bF(\d+)\b", open(os.path.join(L1, "NOVELTY-FLAGS.md")).read())))
out["flags"] = {"min": flags[0], "max": flags[-1], "n_distinct": len(flags)}
out["no_flag_drops_an_arm"] = [n + 1 for n, l in enumerate(open(os.path.join(L1, "NOVELTY-FLAGS.md"))) if "drop an arm" in l.lower()]
p = sys.argv[sys.argv.index("--out") + 1]
json.dump(out, open(p, "w"), indent=1)
print(json.dumps(out, indent=1)[:4000])
