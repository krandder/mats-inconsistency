"""keep_valid.py SRC DST: place every fully loadable .npz from SRC into DST (skip partial files)."""
import glob, os, shutil, sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
ok = skipped = 0
for f in sorted(glob.glob(os.path.join(src, '*.npz'))):
    try:
        d = np.load(f)
        for k in d.files:
            _ = d[k].shape
    except Exception:
        skipped += 1
        continue
    t = os.path.join(dst, os.path.basename(f))
    if not os.path.exists(t) or os.path.getsize(t) != os.path.getsize(f):
        shutil.copyfile(f, t)
    ok += 1
print(f"{src}: {ok} valid files placed in {dst}, {skipped} skipped")
