#!/usr/bin/env python3
"""Stage MATS work into a repo worktree (mats/tools/stage.py). Copies text/code/small results; excludes checkpoints, arrays,
caches, backups and logs > 1 MB; records every excluded checkpoint/array/large file with size + sha256.
Usage: stage.py PLAN.json OUT_MANIFEST.json [--dry]"""
import fnmatch, hashlib, json, os, shutil, sys, time

CKPT = {'.pt', '.pth', '.ckpt', '.safetensors', '.bin'}
ARR = {'.npz', '.npy', '.pkl', '.pickle', '.h5', '.parquet'}
LOGS = {'.log', '.out', '.err'}
SKIP_DIRS = {'__pycache__', '.pytest_cache', '.venv', 'venv', '.git', 'node_modules', '.ipynb_checkpoints', '.mypy_cache'}
MB = 1 << 20

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

MARKERS = (b'OpenAI Codex v', b'hook: SessionStart', b'kimi version', b'session id: ')

def classify(rel, full, size, extra_skip_ext):
    name = os.path.basename(rel); ext = os.path.splitext(name)[1].lower()
    if ext in LOGS and size <= MB:
        head = open(full, 'rb').read(4096)
        if any(m in head for m in MARKERS): return 'artifact-transcript'
    if ext in ('.pyc', '.pyo'): return 'skip-cache'
    if '.bak' in name or name.endswith('~') or ext == '.orig' or ext == '.swp': return 'skip-backup'
    if ext in extra_skip_ext: return 'skip-build'
    if ext in CKPT: return 'artifact-checkpoint'
    if ext in ARR: return 'artifact-array'
    if size > MB: return 'artifact-raw-output'   # bulky raw run output (log, JSON, ...) under the size rule
    return 'include'

def main():
    plan = json.load(open(sys.argv[1])); out_manifest = sys.argv[2]; dry = '--dry' in sys.argv
    missing = [j['src'] for j in plan if not os.path.isdir(j['src'])] + \
        [os.path.join(j['src'], f) for j in plan if j.get('files') for f in j['files'] if not os.path.isfile(os.path.join(j['src'], f))]
    if missing:
        sys.exit('stage.py: missing sources (nothing staged): ' + ', '.join(missing))
    t0 = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    man = dict(staged_at=t0, entries=[], symlinks=[], dropped_symlinks=[], skipped_counts={}, included=[],
               jobs=[dict(src=os.path.abspath(j['src'].rstrip('/')), files=j.get('files')) for j in plan])
    for job in plan:
        src = job['src'].rstrip('/'); dst = job['dst'].rstrip('/'); files = job.get('files')
        extra = set(job.get('skip_ext', [])); skip_names = set(job.get('skip_names', [])); globs = job.get('skip_globs', []); aglobs = job.get('artifact_globs', []); tglobs = job.get('transcript_globs', []); kglobs = job.get('keep_globs', [])
        src_real = os.path.realpath(src)
        walk = []
        if files:
            walk = [(src, [], files)]
        else:
            for root, dirs, fs in os.walk(src, followlinks=False):
                keep = []
                for d in sorted(dirs):
                    fp = os.path.join(root, d)
                    if os.path.islink(fp): fs.append(d); continue
                    if d in SKIP_DIRS:
                        man['skipped_counts'][d + '/'] = man['skipped_counts'].get(d + '/', 0) + 1; continue
                    keep.append(d)
                dirs[:] = keep
                walk.append((root, None, sorted(fs)))
        for root, _, fs in walk:
            for f in fs:
                full = os.path.join(root, f); rel = os.path.relpath(full, src)
                target = os.path.join(dst, rel)
                if f in skip_names or any(fnmatch.fnmatch(rel, g) for g in globs):
                    man['skipped_counts']['named:' + rel] = 1; continue
                if os.path.islink(full):
                    lt = os.readlink(full)
                    absT = lt if os.path.isabs(lt) else os.path.normpath(os.path.join(os.path.dirname(full), lt))
                    realT = os.path.realpath(absT)
                    if os.path.exists(realT) and (realT == src_real or realT.startswith(src_real + os.sep)):
                        new = os.path.relpath(os.path.join(dst, os.path.relpath(realT, src_real)), os.path.dirname(target))
                        man['symlinks'].append(dict(path=target, src=full, was=lt, now=new))
                        if not dry:
                            os.makedirs(os.path.dirname(target), exist_ok=True)
                            if os.path.lexists(target): os.remove(target)
                            os.symlink(new, target)
                    else:
                        man['dropped_symlinks'].append(dict(src=full, target=lt, exists=os.path.exists(realT)))
                    continue
                size = os.path.getsize(full)
                c = ('artifact-transcript' if any(fnmatch.fnmatch(rel, g) for g in tglobs) else
                     'artifact-raw-output' if any(fnmatch.fnmatch(rel, g) for g in aglobs) else
                     'include' if any(fnmatch.fnmatch(rel, g) for g in kglobs) else classify(rel, full, size, extra))
                if c == 'include':
                    man['included'].append(dict(path=target, size=size))
                    if not dry:
                        os.makedirs(os.path.dirname(target), exist_ok=True)
                        shutil.copy2(full, target)
                elif c.startswith('artifact'):
                    rp = target[target.index('/mats/') + 1:] if '/mats/' in target else target
                    lane = job.get('lane') or job['dst'][job['dst'].index('/mats/') + 1:]
                    man['entries'].append(dict(kind=c.split('-', 1)[1], path=os.path.abspath(full), size=size,
                                               sha256=None if dry else sha256(full), group=job['dst'], repo_path=rp, lane=lane))
                else:
                    man['skipped_counts'][c] = man['skipped_counts'].get(c, 0) + 1
    json.dump(man, open(out_manifest, 'w'), indent=1)
    inc = sum(e['size'] for e in man['included'])
    print(f"staged_at={t0} included={len(man['included'])} files {inc/MB:.1f} MB; artifacts={len(man['entries'])} "
          f"{sum(e['size'] for e in man['entries'])/MB:.1f} MB; symlinks kept={len(man['symlinks'])} dropped={len(man['dropped_symlinks'])}; "
          f"skipped={man['skipped_counts']}")

if __name__ == '__main__':
    main()
