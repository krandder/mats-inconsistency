"""Stage 4: assemble the Google-Docs-ready HTML: math and figures as base64 PNG data URIs, real tables."""
import base64, html, json, os, re, sys

IMG = '/tmp/rdn/img/'
body = open('/tmp/rdn/body_placeholders.html', encoding='utf-8').read()
formulas = {f['id']: f for f in json.load(open('/tmp/rdn/formulas.json'))}
meta = {m['id']: m for m in json.load(open('/tmp/rdn/img_meta.json'))}
captions = json.load(open('/tmp/rdn/captions.json'))
figs = {f['letter']: f for f in json.load(open('/tmp/rdn/figs.json'))} if os.path.exists('/tmp/rdn/figs.json') else {}
MODE = sys.argv[1] if len(sys.argv) > 1 else ('pdg' if figs else 'mermaid')
MAXW = 600


def img(file, w, h, alt, style=''):
    if w > MAXW:
        w, h = MAXW, round(h * MAXW / w)
    b64 = base64.b64encode(open(IMG + file, 'rb').read()).decode()
    return (f'<img src="data:image/png;base64,{b64}" width="{w}" height="{h}" '
            f'style="width:{w}px;height:{h}px;{style}" alt="{html.escape(alt, quote=True)}">')


def math_img(n):
    m, f = meta[n], formulas[n]
    return img(m['file'], m['css_w'], m['css_h'], f['tex'], '' if f['display'] else 'vertical-align:middle')


counts = {'inline': 0, 'display': 0, 'figure': 0}


def sub_inline(h):
    def r(mo):
        counts['inline'] += 1
        return math_img(int(mo.group(1)))
    return re.sub(r'IMATHPH(\d{4})Z', r, h)


def figure(letter):
    f = figs[letter]
    counts['figure'] += 1
    return (f'<p style="text-align:center;margin:10pt 0 2pt 0">{img(f["file"], int(f["css_w"]), int(f["css_h"]), f["caption"])}</p>\n'
            f'<p style="text-align:center;font-size:10pt;color:#444444;margin:0 0 14pt 0">{sub_inline(captions[letter])}</p>\n')


# display equations: centred paragraphs
def sub_display(mo):
    counts['display'] += 1
    return f'<p style="text-align:center">{math_img(int(mo.group(1)))}</p>'


body = re.sub(r'<p>DMATHPH(\d{4})Z</p>', sub_display, body)
assert 'DMATHPH' not in body

# the mermaid flowchart -> figure (c) from the web page builder, or a mermaid-rendered PNG
mer = re.compile(r'<pre><code class="language-mermaid">.*?</code></pre>\n?', re.S)
assert len(mer.findall(body)) == 1
if MODE == 'pdg':
    body = mer.sub(lambda _: figure('c'), body)
    k = body.index('<p>For any fixed world model')
    body = body[:k] + figure('a') + body[k:]
    mo = re.search(r'<p>In practice, we cannot evaluate.*?</p>\n?', body, re.S)
    body = body[:mo.end()] + figure('b') + body[mo.end():]
else:
    from PIL import Image
    counts['figure'] += 1
    mw, mh = Image.open(IMG + 'mermaid.png').size
    body = mer.sub(lambda _: '<p style="text-align:center">' + img('mermaid.png', mw // 2, mh // 2, 'Parameter blocks flowchart') + '</p>\n', body)

# tables: column widths that fit the widest formula in each column (Docs clips an image wider than its
# cell), the rest of the 468pt text width shared by text length; bold header text; visible borders
PAGE_PT, PAD_PT = 468, 5
from PIL import ImageFont
FONTS = {False: ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf', 110),
         True: ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf', 110)}  # Arial metrics


def units(inner, bold):
    """widths (pt, 11pt text) of the unbreakable pieces of a cell: words, with formula images glued to them"""
    out = []
    for word in re.sub(r'<br\s*/?>', ' ', inner).split():
        b = bold or '<strong>' in word or '<b>' in word
        w = sum(meta[int(n)]['css_w'] * 0.75 for n in re.findall(r'IMATHPH(\d{4})Z', word))
        out.append(w + FONTS[b].getlength(re.sub(r'<[^>]+>|IMATHPH\d{4}Z', '', html.unescape(word))) / 10)
    return out


def style_table(mo):
    t = mo.group(0)
    rows = re.findall(r'<tr>(.*?)</tr>', t, re.S)
    cells = [re.findall(r'<(th|td)(?: style="text-align:(\w+)")?>(.*?)</\1>', r, re.S) for r in rows]
    ncol = len(cells[0])
    minw, textw = [50.0] * ncol, [1.0] * ncol
    for row in cells:
        for c, (tag, _, inner) in enumerate(row):
            # bold spans: mark every word inside <strong>..</strong> as bold
            inner_b = re.sub(r'<strong>(.*?)</strong>', lambda m: ' '.join('<b>' + w for w in m.group(1).split()), inner, flags=re.S)
            u = units(inner_b, tag == 'th')
            minw[c] = max(minw[c], max(u, default=0) + 2 * PAD_PT + 3)
            textw[c] += sum(u)
    spare = max(0.0, PAGE_PT - sum(minw))
    widths = [m + spare * tw / sum(textw) for m, tw in zip(minw, textw)]
    widths = [w * PAGE_PT / sum(widths) for w in widths]

    def cell(m, c):
        tag, align, inner = m.group(1), m.group(2) or 'left', m.group(3)
        if tag == 'th':
            inner = f'<b>{inner}</b>'
        return (f'<{tag} style="width:{widths[c]:.0f}pt;text-align:{align};vertical-align:top;'
                f'border:1px solid #999999;padding:{PAD_PT}pt"><p style="text-align:{align};margin:0">{inner}</p></{tag}>')

    def row(m):
        k = iter(range(ncol))
        return re.sub(r'<(th|td)(?: style="text-align:(\w+)")?>(.*?)</\1>', lambda c: cell(c, next(k)), m.group(0), flags=re.S)

    t = re.sub(r'<tr>.*?</tr>', row, t, flags=re.S)
    print('table column widths (pt):', [round(w) for w in widths])
    return t.replace('<table>', '<table border="1" cellspacing="0" style="border-collapse:collapse;border:1px solid #999999">')


body = re.sub(r'<table>.*?</table>', style_table, body, flags=re.S)

# paragraph spacing (Docs imports bare <p> with zero space between paragraphs)
body = body.replace('<p>', '<p style="margin:0 0 8pt 0">')
body = body.replace('<li>', '<li style="margin:0 0 3pt 0">')
body = body.replace('<p style="text-align:center"><img', '<p style="text-align:center;margin:4pt 0 10pt 0"><img')

body = sub_inline(body)
assert 'IMATHPH' not in body and '$' not in re.sub(r'<img[^>]*>', '', body)

doc = ('<!DOCTYPE html>\n<html><head><meta charset="utf-8">'
       '<title>Inconsistency Is All You Need - research direction note</title></head>\n'
       '<body style="font-family:Arial,sans-serif;font-size:11pt">\n' + body + '</body></html>\n')
open('/tmp/rdn/gdoc.html', 'w', encoding='utf-8').write(doc)
n_img = doc.count('<img ')
print(MODE, counts, '| <img> total', n_img, '| html bytes', len(doc.encode()))
