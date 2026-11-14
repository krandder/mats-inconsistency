"""Summarise the binding arms (PREREG-FA.md Amendment A3): one row per arm from diag.json and meta.json.
usage: python summarize_bind.py RUNS_DIR [--md OUT.md]"""
import glob, json, os, sys


def row(d):
    m = json.load(open(os.path.join(d, 'meta.json')))
    g = json.load(open(os.path.join(d, 'diag.json')))
    c = g['cells']
    f = lambda T: c[str(T)]['frac_history_recovered']
    att = c['32'].get('attention', {})
    br = c['32'].get('binding_response_shuffle', {})
    cur = m['curve']
    first = next((r['step'] for r in cur if r['fracT32'] is not None and r['fracT32'] >= 0.5), None)
    return dict(arm=os.path.basename(d), steps=m['steps_done'], stopped=m['stopped_on_time'], best_step=m['best_step'],
                gapT0=c['0']['gap_model'], frac8=f(8), frac32=f(32), frac64=f(64), frac128=f(128),
                hv32=c['32']['history_value'], attn=att.get('ratio_by_layer'), rms=br.get('rms_ratio'),
                skl=br.get('skl_ratio'), relabel=c['32']['kl_relabel'], first_frac50_step=first,
                hours=m['seconds'] / 3600)


def main():
    root = sys.argv[1]
    rows = [row(os.path.dirname(p)) for p in sorted(glob.glob(os.path.join(root, '*', 'diag.json')))]
    hdr = ('| arm | steps (stopped) | T0 gap | frac T8 / T32 / T64 / T128 | attn ratio L1..L4 (T32) | shuffle RMS / SKL ratio '
           '| relabel KL | first step frac32 >= .5 |\n|---|---|---|---|---|---|---|---|')
    lines = [hdr]
    for r in rows:
        lines.append(f"| {r['arm']} | {r['steps']}{' (time)' if r['stopped'] else ''} | {r['gapT0']:.3f} | "
                     f"{r['frac8']:.2f} / {r['frac32']:.2f} / {r['frac64']:.2f} / {r['frac128']:.2f} | "
                     f"{' / '.join(f'{x:.2f}' for x in r['attn'])} | {r['rms']:.3f} / {r['skl']:.3f} | {r['relabel']:.1e} | "
                     f"{r['first_frac50_step']} |")
    txt = '\n'.join(lines)
    print(txt)
    if '--md' in sys.argv:
        open(sys.argv[sys.argv.index('--md') + 1], 'w').write(txt + '\n')
    json.dump(rows, open(os.path.join(root, 'binding_summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
