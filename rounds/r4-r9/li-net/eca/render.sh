#!/bin/bash
# Render RESULTS.md to memo-serve/lir-agent/eca.html with the eurisko renderer, then fix the <title> and
# turn paragraphs of the form "IMG: results/NAME.png" into <img> tags (PNGs copied to eca-files/).
set -eu
S=/home/agent/workspaces/eurisko-study/li-net/eca
OUTD=/home/agent/memo-serve/lir-agent
mkdir -p $OUTD/eca-files
cp $S/results/*.png $OUTD/eca-files/ 2>/dev/null || true
python3 /home/agent/workspaces/eurisko-study/tools/render.py $S/RESULTS.md $OUTD/eca.html.tmp
python3 - "$OUTD/eca.html.tmp" <<'PY'
import re, sys
p = sys.argv[1]
h = open(p).read()
h = h.replace("<title>EURISKO Study Report</title>", "<title>E-CA: stiffness and search on a permuted cellular-automaton grid</title>")
h = re.sub(r"<p>IMG: results/([\w.\-]+\.png)</p>",
           lambda m: f'<p><img src="eca-files/{m.group(1)}" alt="{m.group(1)}" style="max-width:100%;height:auto"></p>', h)
open(p, "w").write(h)
PY
mv $OUTD/eca.html.tmp $OUTD/eca.html
grep -o "<title>[^<]*</title>" $OUTD/eca.html
