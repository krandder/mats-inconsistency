#!/bin/bash
# Render RESULTS.md to memo-serve/lir-agent/smallnet.html with the eurisko renderer, then fix the <title> and
# turn paragraphs of the form "IMG: results/NAME.png" into <img> tags (PNGs copied to smallnet-files/).
set -eu
S=/home/agent/workspaces/eurisko-study/li-net/smallnet
OUTD=/home/agent/memo-serve/lir-agent
mkdir -p $OUTD/smallnet-files
cp $S/results/*.png $OUTD/smallnet-files/ 2>/dev/null || true
python3 /home/agent/workspaces/eurisko-study/tools/render.py $S/RESULTS.md $OUTD/smallnet.html.tmp
python3 - "$OUTD/smallnet.html.tmp" <<'PY'
import re, sys
p = sys.argv[1]
h = open(p).read()
h = h.replace("<title>EURISKO Study Report</title>", "<title>ShipHunt small-net data-efficiency loop: results</title>")
h = re.sub(r"<p>IMG: results/([\w.\-]+\.png)</p>",
           lambda m: f'<p><img src="smallnet-files/{m.group(1)}" alt="{m.group(1)}" style="max-width:100%;height:auto"></p>', h)
open(p, "w").write(h)
PY
mv $OUTD/smallnet.html.tmp $OUTD/smallnet.html
grep -o "<title>[^<]*</title>" $OUTD/smallnet.html
