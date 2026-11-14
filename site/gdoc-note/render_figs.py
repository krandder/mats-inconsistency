"""Stage 3b: snapshot the web page builder's Richardson-style PDG figures (inline SVG in direction/index.html)
as 2x PNGs, and pull each caption back to text with $tex$ math (from KaTeX's annotation elements)."""
import html, json, pathlib, re, shutil
from PIL import Image
import numpy as np
from playwright.sync_api import sync_playwright

CHROME = '/home/agent/.cache/ms-playwright/chromium-1234/chrome-linux64/chrome'
PAGE = pathlib.Path('/home/agent/workspaces/mats/work/direction-page/direction/index.html')
SNAP = pathlib.Path('/tmp/rdn/builder-index-snapshot.html')
shutil.copy2(PAGE, SNAP)
src = SNAP.read_text(encoding='utf-8')

figs = []
for k, m in enumerate(re.finditer(r'<figure>(.*?)</figure>', src, flags=re.S)):
    cap_html = re.search(r'<figcaption>(.*?)</figcaption>', m.group(1), flags=re.S).group(1)
    # each KaTeX span -> $tex$ (from its annotation), then drop the remaining tags
    cap = re.sub(r'<span class="katex">.*?<annotation encoding="application/x-tex">(.*?)</annotation>.*?</span></span></span>',
                 lambda a: '$' + html.unescape(a.group(1)) + '$', cap_html, flags=re.S)
    if '<span class="katex' in cap:
        raise SystemExit(f'figure {k}: caption math not fully recovered')
    cap = html.unescape(re.sub(r'<[^>]+>', '', cap)).strip()
    letter = re.match(r'\((\w)\)', cap).group(1)
    figs.append({'letter': letter, 'caption': cap})

with sync_playwright() as p:
    browser = p.chromium.launch(executable_path=CHROME, headless=True, args=['--no-sandbox'])
    page = browser.new_context(device_scale_factor=2, viewport={'width': 1400, 'height': 1000},
                               color_scheme='light').new_page()
    page.goto(SNAP.as_uri(), wait_until='networkidle')
    page.evaluate("document.fonts.ready.then(() => true)")
    page.wait_for_timeout(500)
    print('fonts', page.evaluate("document.fonts.check('italic 12px KaTeX_Math')"))
    svgs = page.locator('figure svg.pdg')
    assert svgs.count() == len(figs), (svgs.count(), len(figs))
    caps = page.evaluate("""() => [...document.querySelectorAll('figure figcaption')].map(fc => {
        const c = fc.cloneNode(true);
        c.querySelectorAll('.katex').forEach(k => {
            const a = k.querySelector('annotation[encoding="application/x-tex"]');
            k.replaceWith(document.createTextNode('$' + a.textContent + '$'));
        });
        return c.textContent.replace(/\\s+/g, ' ').trim();
    })""")
    for f, c in zip(figs, caps):
        f['caption'] = c
    for i, f in enumerate(figs):
        el = svgs.nth(i)
        el.evaluate("e => e.style.background = '#fff'")
        css_w = el.evaluate("e => e.getBoundingClientRect().width")
        raw = f'/tmp/rdn/img/fig_{f["letter"]}_raw.png'
        el.screenshot(path=raw)
        im = Image.open(raw).convert('RGB')
        a = np.asarray(im).astype(int)
        ys, xs = np.where(a.min(axis=2) < 245)
        mg = 16
        crop = im.crop((max(0, xs.min() - mg), max(0, ys.min() - mg), min(im.width, xs.max() + 1 + mg), min(im.height, ys.max() + 1 + mg)))
        f['file'] = f'fig_{f["letter"]}.png'
        crop.save('/tmp/rdn/img/' + f['file'], optimize=True)
        f['css_w'], f['css_h'] = round(crop.width / 2, 1), round(crop.height / 2, 1)
        print(f['letter'], 'svg css width', round(css_w), '-> png css', f['css_w'], 'x', f['css_h'])
    browser.close()
json.dump(figs, open('/tmp/rdn/figs.json', 'w'), indent=1, ensure_ascii=False)
for f in figs:
    print(f['letter'], f['caption'])
