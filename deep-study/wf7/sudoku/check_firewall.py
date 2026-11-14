"""check_firewall.py -- the arms must not import the oracle. Fails (exit 1) if core.py, data.py, train_m.py,
e_distill.py or e_trustB.py mention it, or if e_search6.py imports it outside the 'multi' scoring branch."""
import re, sys
bad = []
for f in ("core.py", "data.py", "train_m.py", "e_distill.py", "e_trustB.py"):
    src = open(f).read()
    if re.search(r"^\s*(import|from)\s+(oracle|eval_lib)", src, re.M):
        bad.append(f)
src = open("e_search6.py").read()
imports = [m.start() for m in re.finditer(r"^\s*import oracle", src, re.M)]
if any(pos < src.index('elif mode == "multi"') for pos in imports) or re.search(r"^import .*oracle", src, re.M):
    bad.append("e_search6.py")
print("firewall:", "FAIL " + str(bad) if bad else "ok (arms do not import oracle.py or eval_lib.py)")
sys.exit(1 if bad else 0)
