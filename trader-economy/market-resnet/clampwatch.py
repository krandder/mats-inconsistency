"""One-off helper on a phase-3 VM (2026-09-29 20:55 UTC): the L3200 re-evaluations are slower than planned
(about 40 minutes of shared GPU per n = 200 block), so the clamp-20 block is kept only for the arms that
answer the runaway question (reeval.CLAMP_ARMS). reeval.py processes started before that change still run
it for every arm; this script stops such a process (by its exact PID) right after its clamp-0 L3200 block
is written, so the runner's retry pass reruns it with the new default (clamp-0 rows kept, deep evals only).
Runs for at most 3 hours; one line per stop in p3/clampwatch.log.
21:20 UTC: only processes started before the VM's current reeval.py was written are stopped (the
retry pass reruns a job whose JSON already holds its clamp-0 L3200 block; it must not be stopped).
21:45 UTC: the extreme-depth evals are kept only for reeval.DEEP_ARMS; an older process of another arm is also
stopped once its L3200 blocks are written (it would run the deep evals next)."""
import json, os, re, signal, time
from reeval import CLAMP_ARMS, DEEP_ARMS

BTIME = next(int(l.split()[1]) for l in open("/proc/stat") if l.startswith("btime"))


def started(pid):
    return BTIME + int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[19]) / os.sysconf("SC_CLK_TCK")

end = time.time() + 3 * 3600
while time.time() < end:
    for pid in [p for p in os.listdir("/proc") if p.isdigit()]:
        try:
            argv = [z.decode() for z in open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0") if z]
            cwd = os.readlink(f"/proc/{pid}/cwd")
        except OSError:
            continue
        if len(argv) < 2 or not argv[0].endswith("python3") or argv[1] != "reeval.py":
            continue
        if "--clamp" in argv or "--ckpt" not in argv or "--out" not in argv:
            continue
        ck, out = argv[argv.index("--ckpt") + 1], argv[argv.index("--out") + 1]
        arm = re.sub(r"_s\d+\.pt$", "", os.path.basename(ck))
        try:
            if started(pid) > os.stat(os.path.join(cwd, "reeval.py")).st_mtime:   # runs the current code
                continue
            rows = json.load(open(os.path.join(cwd, out)))["rows"]
        except (OSError, ValueError, KeyError):
            continue
        if not any(r["L"] == 3200 and r["clamp"] == 0.0 for r in rows):
            continue
        clamped = any(r["clamp"] > 0 for r in rows)
        # next block of the old process: an unneeded clamp-20 block (DT-recall has none), or unneeded deep evals
        extra_clamp = arm not in CLAMP_ARMS and not arm.startswith("dtr") and not clamped
        extra_deep = arm not in DEEP_ARMS and (arm not in CLAMP_ARMS or clamped)
        if extra_clamp or extra_deep:
            os.kill(int(pid), signal.SIGTERM)
            with open("p3/clampwatch.log", "a") as f:
                f.write(f"{time.strftime('%FT%TZ', time.gmtime())} stopped pid {pid} {arm} after its L3200 block"
                        f" ({'clamp block' if extra_clamp else 'deep evals'} not needed)\n")
    time.sleep(10)
