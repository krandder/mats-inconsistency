"""Stage 2: render every formula with KaTeX (CDN) in headless Chromium, screenshot at 2x, crop to the ink.
Inline images keep no margin below the ink (Google Docs sits inline images on the text baseline)."""
import json, pathlib, re
from PIL import Image
import numpy as np
from playwright.sync_api import sync_playwright

CHROME = '/home/agent/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome'
OUT = pathlib.Path('/tmp/rdn/img')
OUT.mkdir(exist_ok=True)
formulas = json.load(open('/tmp/rdn/formulas.json'))
SCALE = 2


# Wide display equations get line breaks (same math, two lines) so they fit a 600px Doc column unshrunk.
def split(tex, env, cuts):
    out = tex
    for old, new in cuts:
        assert out.count(old) == 1, (old, out.count(old))
        out = out.replace(old, new)
    out = f'\\begin{{{env}}}{out}\\end{{{env}}}'
    norm = lambda s: re.sub(r'\\begin\{\w+\}|\\end\{\w+\}|\\\\|&|\\q?quad|\s', '', s)
    assert norm(out) == norm(tex), 'layout changed the math'
    return out


LAYOUT = {  # keyed by the start of the display equation's TeX
    r'\mathcal{E}(\mu; \theta, U) \;=\;': [(r'U) \;=\; \underbrace', r'U) \;=\;& \underbrace'),
                                           (r'Incompatibility}} \;+\; ', r'Incompatibility}} \\ &\;+\; ')],
    r'\mu^*_k \;=\; \arg\min_{\mu} \sum': [(r'\mu^*_k \;=\; \arg\min', r'\mu^*_k \;&=\; \arg\min'),
                                           (r'\big) \quad \stackrel', r'\big) \\ \stackrel'),
                                           (r'\quad \mu^*_k \;=\; \big(K', r'\quad \mu^*_k \;&=\; \big(K')],
    r'\text{residual}^{(l)}_r \;=\;': [(r'\text{residual}^{(l)}_r \;=\; v_r', r'\text{residual}^{(l)}_r \;&=\; v_r'),
                                       (r'\big), \qquad \ell^{(l)} \;=\; ', r'\big), \\ \ell^{(l)} \;&=\; ')],
    r'\mathrm{Score}_{\text{pol}}': [(r'\text{Consensus}) \;=\; \underbrace', r'\text{Consensus}) \;=\;& \underbrace'),
                                     (r'in } q_\theta}} \;\times\; ', r'in } q_\theta}} \\ &\;\times\; ')],
}
LAYOUT = list(LAYOUT.items()) + [   # v2 note
    (r'\mathrm{Score}_{\text{pol}}', [(r'\mathrm{Score}_{\text{pol}}(v', r'&\mathrm{Score}_{\text{pol}}(v'),
                                      (r'\text{goal variables } X_G}} \;\times\; ', r'\text{goal variables } X_G}} \\ &\qquad\;\times\; ')]),
    (r'\mathbb{E}_{\text{fused}}', [(r'\nabla U \rangle\big] \;=\; \frac', r'\nabla U \rangle\big] \;&=\; \frac'),
                                    (r'\;\ge\; 0, \qquad \mathbb{E}_{\text{masked}}', r'\;\ge\; 0, \\ \mathbb{E}_{\text{masked}}'),
                                    (r'\nabla U \rangle\big] \;=\; 0.', r'\nabla U \rangle\big] \;&=\; 0.')]),
]
for f in formulas:
    f['render_tex'] = f['tex'] if not f.get('bold') else '\\boldsymbol{' + f['tex'] + '}'
    for start, cuts in LAYOUT:
        if f['display'] and f['tex'].startswith(start) and all(f['tex'].count(o) == 1 for o, _ in cuts):
            f['render_tex'] = split(f['tex'], 'aligned', cuts)
            f['split'] = True
            break

page_html = """<!doctype html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<style>
 html,body{margin:0;background:#fff;color:#000}
 body{font-size:11pt;padding:20px}
 .row{margin:0 0 24px 0;white-space:nowrap}
 .wrap{display:inline-block;padding:8px 10px;background:#fff}
 .wrap .katex-display{margin:0}
</style></head><body><div id="root"></div></body></html>"""

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=CHROME, headless=True, args=['--no-sandbox'])
    ctx = browser.new_context(device_scale_factor=SCALE, viewport={'width': 2600, 'height': 1200})
    page = ctx.new_page()
    page.set_content(page_html, wait_until='networkidle')
    errors = page.evaluate("""(fs) => {
        const root = document.getElementById('root'); const errs = [];
        for (const f of fs) {
            const row = document.createElement('div'); row.className = 'row';
            const w = document.createElement('span'); w.className = 'wrap'; w.id = 'f' + f.id;
            w.style.fontSize = (11 * f.scale) + 'pt';
            row.appendChild(w); root.appendChild(row);
            try { katex.render(f.render_tex, w, {displayMode: f.display, throwOnError: true, strict: 'ignore'}); }
            catch (e) { errs.push([f.id, String(e)]); }
        }
        return errs;
    }""", formulas)
    page.evaluate("document.fonts.ready.then(() => true)")
    page.wait_for_timeout(500)
    fonts_ok = page.evaluate("document.fonts.check('12px KaTeX_Main') && document.fonts.check('italic 12px KaTeX_Math')")
    print('katex errors:', errors, '| fonts loaded:', fonts_ok)
    assert not errors and fonts_ok
    meta = []
    for f in formulas:
        raw = OUT / f'raw{f["id"]:04d}.png'
        page.locator(f'#f{f["id"]}').screenshot(path=str(raw))
        im = Image.open(raw).convert('RGB')
        raw.unlink()
        a = np.asarray(im).astype(int)
        ys, xs = np.where(a.min(axis=2) < 245)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        m = 2 * SCALE if not f['display'] else 4 * SCALE      # margin in device px
        mb = 0 if not f['display'] else 4 * SCALE              # inline: ink bottom = image bottom = baseline
        crop = im.crop((max(0, x0 - m), max(0, y0 - m), min(im.width, x1 + m), min(im.height, y1 + mb)))
        # pad to even device-px sizes so the CSS size (px / 2) is a whole number
        w2, h2 = crop.width + crop.width % 2, crop.height + crop.height % 2
        if (w2, h2) != crop.size:
            canvas = Image.new('RGB', (w2, h2), 'white')
            canvas.paste(crop, (0, h2 - crop.height if not f['display'] else 0))   # inline: pad on top
            crop = canvas
        name = f'{"d" if f["display"] else "i"}{f["id"]:04d}.png'
        crop.save(OUT / name, optimize=True)
        meta.append({'id': f['id'], 'display': f['display'], 'file': name,
                     'css_w': crop.width // SCALE, 'css_h': crop.height // SCALE})
    browser.close()

json.dump(meta, open('/tmp/rdn/img_meta.json', 'w'), indent=1)
for x in meta:
    if x['display']:
        f = formulas[x['id']]
        print('display', x['id'], 'css', x['css_w'], 'x', x['css_h'], 'split' if f.get('split') else '',
              'TOO WIDE' if x['css_w'] > 600 else '', f['tex'][:50])
inl = [x for x in meta if not x['display']]
print('inline css heights min/median/max:', min(x['css_h'] for x in inl),
      sorted(x['css_h'] for x in inl)[len(inl) // 2], max(x['css_h'] for x in inl))
