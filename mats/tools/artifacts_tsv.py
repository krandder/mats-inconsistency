#!/usr/bin/env python3
"""Update mats/ARTIFACTS.tsv from a stage.py manifest, then rebuild the table with artifacts_summary.py.
usage: artifacts_tsv.py REPO_MATS_DIR MANIFEST.json [MANIFEST2.json ...]
Rows under every folder a sync re-read (and the single files of 'files' jobs) are replaced by the manifest's entries;
transcript rows are only replaced file by file. Kinds: checkpoint, array, raw-output (artifact_globs and files over the
1 MB rule), transcript (reviewer CLI logs). Each entry carries repo_path and lane from stage.py."""
import json, os, sys

M = sys.argv[1]
W = '/home/agent/workspaces/mats/work/'
KIND = {'checkpoint': 'checkpoint', 'array': 'array', 'raw-output': 'raw-output', 'large-other': 'raw-output',
        'large-log': 'raw-output', 'transcript': 'transcript'}
COLS = ('path', 'size', 'sha256', 'lane', 'general_path', 'kind')
tsv = os.path.join(M, 'ARTIFACTS.tsv')
lines = open(tsv).read().splitlines()
assert tuple(lines[0].split('\t')) == COLS, lines[0]
rows = [dict(zip(COLS, l.split('\t'))) for l in lines[1:]]
entries = [e for m in sys.argv[2:] for e in json.load(open(m))['entries']]
folder = lambda p: p[len(W):].split('/')[0] if p.startswith(W) else None
jobs = [j for m in sys.argv[2:] for j in json.load(open(m)).get('jobs', [])]
if jobs:   # replace exactly what the sync re-read: whole folders for folder jobs, single files for 'files' jobs
    prefixes = tuple(j['src'] + '/' for j in jobs if not j.get('files'))
    exact = {os.path.join(j['src'], f) for j in jobs if j.get('files') for f in j['files']}
    gone = lambda r: r['general_path'].startswith(prefixes) or r['general_path'] in exact
else:      # older manifests: whole top-level work folders
    covered = {folder(e['path']) for e in entries} - {None}
    gone = lambda r: folder(r['general_path']) in covered
# entries for files already committed (for example a large result file kept in the repo by an earlier sync) are not listed
import subprocess
repo = os.path.dirname(os.path.abspath(M))
tracked = set(subprocess.run(['git', '-C', repo, 'ls-files', '-z', 'mats'], capture_output=True, text=True).stdout.split('\0'))
entries = [e for e in entries if e['repo_path'] not in tracked]
newpaths = {e['path'] for e in entries}
keep = [r for r in rows if r['general_path'] not in newpaths and not (r['kind'] != 'transcript' and gone(r))]
for e in entries:
    keep.append(dict(path=e['repo_path'], size=str(e['size']), sha256=e['sha256'], lane=e['lane'],
                     general_path=e['path'], kind=KIND[e['kind']]))
keep.sort(key=lambda r: (r['lane'], r['path']))
assert len({r['general_path'] for r in keep}) == len(keep), 'duplicate general_path'
with open(tsv, 'w') as f:
    f.write('\t'.join(COLS) + '\n')
    for r in keep:
        f.write('\t'.join(r[k] for k in COLS) + '\n')
print(f'ARTIFACTS.tsv: {len(rows)} -> {len(keep)} rows; now run artifacts_summary.py {M}')
