#!/bin/bash
# Re-render REPORT.md -> report.html (with figures) and publish to memo-serve/lir-agent/index.html.
# Usage: bash lir-agent/report-files/render.sh   (plain tools/render.py drops the figures and the page title)
set -euo pipefail
L=/home/agent/workspaces/eurisko-study/lir-agent
W=/home/agent/memo-serve/lir-agent
python3 "$L/../tools/render.py" "$L/REPORT.md" "$L/report.html"
python3 - "$L/report.html" <<'PY'
import re, sys, datetime
p = sys.argv[1]; s = open(p).read()
s = s.replace('<title>EURISKO Study Report</title>', '<title>LIR Masks Morning Report</title>')
now = datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%MZ')
s = s.replace('Rendered from REPORT.md (2026-09-27).',
    f'Rendered {now} from lir-agent/REPORT.md by lir-agent/report-files/render.sh; numbers come from CLAIMS.md unless marked (unaudited).')
def fig(m):
    cap = (m.group(1) + m.group(3)).strip()
    alt = re.sub(r'<[^>]+>', '', cap)[:200].replace('"', '&quot;')
    return f'<figure><img src="{m.group(2)}" alt="{alt}" loading="lazy"><figcaption>{cap}</figcaption></figure>'
s, n = re.subn(r'<p>(Figure \d+\..*?)<a href="(report-files/[^"]+\.png)">[^<]*</a>(.*?)</p>', fig, s, flags=re.S)
s = s.replace('.meta{', 'figure{margin:18px 0}\nfigure img{display:block;max-width:100%;height:auto;background:#fff;'
    'border:1px solid var(--line);border-radius:6px;padding:4px}\nfigcaption{color:var(--muted);font-size:13.5px;margin-top:6px}\n.meta{', 1)
open(p, 'w').write(s); print('figures', n)
PY
mkdir -p "$W/report-files"
cp "$L"/report-files/*.png "$W/report-files/"
cp "$L/report.html" "$W/index.html"
cmp "$L/report.html" "$W/index.html" && echo "published $W/index.html"
