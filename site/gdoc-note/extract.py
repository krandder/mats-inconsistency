"""Stage 1: pull math out of the markdown (and the PDG figure captions) into placeholders,
convert the rest with markdown-it (CommonMark + tables)."""
import json, os, re, sys
from markdown_it import MarkdownIt

SRC = os.environ.get('RDN_SRC', '/home/agent/workspaces/mats/outbox/research-direction-note-v2-2026-10-16.md')
md_text = open(SRC, encoding='utf-8').read()

formulas = []          # {id, tex, display, scale}; scale = font size relative to 11pt body text
index = {}
HEADING_SCALE = {1: 24 / 11, 2: 18 / 11, 3: 14 / 11, 4: 12 / 11}   # heading sizes Google Docs gives imported <h1>-<h4>
CAPTION_SCALE = 10 / 11


def fid(tex, display, scale=1.0, bold=False):
    key = (tex, display, round(scale, 3), bold)
    if key not in index:
        index[key] = len(formulas)
        formulas.append({'id': len(formulas), 'tex': tex, 'display': display, 'scale': round(scale, 3), 'bold': bold})
    return index[key]


def boldify(h):
    """math inside <strong>, <th> or a heading gets a bold render, like the text around it"""
    def bold_id(k):
        f = formulas[int(k.group(1))]
        return f'IMATHPH{fid(f["tex"], False, f["scale"], True):04d}Z'
    return re.sub(r'<strong>.*?</strong>|<th[ >].*?</th>|<h\d>.*?</h\d>', lambda m: re.sub(r'IMATHPH(\d{4})Z', bold_id, m.group(0)), h, flags=re.S)


INLINE = re.compile(r'(?<![\\$])\$(?!\s)([^$\n]+?)(?<![\s\\])\$(?!\$)')


def inline_sub(text, scale):
    return INLINE.sub(lambda m: f'IMATHPH{fid(m.group(1).strip(), False, scale):04d}Z', text)


out_lines = []
in_fence = False
for line in md_text.split('\n'):
    if line.lstrip().startswith('```'):
        in_fence = not in_fence
        out_lines.append(line)
        continue
    if in_fence:
        out_lines.append(line)
        continue
    m = re.match(r'^(\s*)\$\$(.+?)\$\$\s*$', line)
    if m:                                   # display math -> its own paragraph, same indentation
        indent, tex = m.group(1), m.group(2).strip()
        out_lines += ['', f'{indent}DMATHPH{fid(tex, True):04d}Z', '']
        continue
    if '$$' in line:
        sys.exit(f'unhandled display math mid-line: {line[:80]}')
    h = re.match(r'^(#{1,6}) ', line)
    if line.startswith('|'):   # table cell: let "a, \, b" wrap at its comma (two images instead of one wide one)
        def wrap_at_comma(m):
            tex, parts, depth, last = m.group(1), [], 0, 0
            for i, ch in enumerate(tex):
                depth += {'{': 1, '}': -1}.get(ch, 0)
                if depth == 0 and tex.startswith(', \\, ', i):
                    parts.append(tex[last:i + 1])
                    last = i + len(', \\, ')
            parts.append(tex[last:])
            return ' '.join(f'${p.strip()}$' for p in parts)
        line = INLINE.sub(wrap_at_comma, line)
    out_lines.append(inline_sub(line, HEADING_SCALE.get(len(h.group(1)), 1.0) if h else 1.0))
text = '\n'.join(out_lines)

left = [l for l in text.split('\n') if '$' in l]
if left:
    sys.exit('leftover $ in: ' + left[0][:100])

md = MarkdownIt('commonmark', {'html': True}).enable('table')
html_body = boldify(md.render(text))

captions = {}
if os.path.exists('/tmp/rdn/figs.json'):
    for f in json.load(open('/tmp/rdn/figs.json')):
        captions[f['letter']] = md.renderInline(inline_sub(f['caption'], CAPTION_SCALE))

open('/tmp/rdn/body_placeholders.html', 'w', encoding='utf-8').write(html_body)
json.dump(captions, open('/tmp/rdn/captions.json', 'w'), indent=1, ensure_ascii=False)
json.dump(formulas, open('/tmp/rdn/formulas.json', 'w'), indent=1, ensure_ascii=False)
print(len(formulas), 'distinct formula renders;', sum(f['display'] for f in formulas), 'display;',
      sum(f['scale'] != 1.0 for f in formulas), 'scaled (headings/captions)')
print('body inline occurrences:', len(re.findall(r'IMATHPH\d{4}Z', html_body)),
      '| display:', len(re.findall(r'DMATHPH\d{4}Z', html_body)),
      '| caption inline:', sum(len(re.findall(r'IMATHPH\d{4}Z', c)) for c in captions.values()))
