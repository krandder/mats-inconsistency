#!/bin/bash
# Render RESULTS.md to memo-serve/lir-agent/wm2.html with the eurisko renderer, then fix the <title> and turn
# paragraphs of the form "IMG: results/NAME.png" into <img> tags (PNGs copied to wm2-files/).
set -eu
W=/home/agent/workspaces/eurisko-study/li-net/wm2
OUTD=/home/agent/memo-serve/lir-agent
mkdir -p $OUTD/wm2-files
cp $W/results/*.png $OUTD/wm2-files/ 2>/dev/null || true
python3 /home/agent/workspaces/eurisko-study/tools/render.py $W/RESULTS.md $OUTD/wm2.html.tmp
python3 - "$OUTD/wm2.html.tmp" <<'PY'
import re, sys
p = sys.argv[1]
h = open(p).read()
h = re.sub(r"<title>[^<]*</title>", "<title>ShipHunt world-model loop: results</title>", h, count=1)
h = re.sub(r"<p>IMG: results/([\w.\-]+\.png)</p>",
           lambda m: f'<p><img src="wm2-files/{m.group(1)}" alt="{m.group(1)}" style="max-width:100%;height:auto"></p>', h)
open(p, "w").write(h)
PY
mv $OUTD/wm2.html.tmp $OUTD/wm2.html
grep -o "<title>[^<]*</title>" $OUTD/wm2.html
