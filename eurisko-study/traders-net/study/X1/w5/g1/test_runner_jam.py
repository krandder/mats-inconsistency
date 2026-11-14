"""X1-w5 G-f: does queue-runner.sh dispatch a 2nd job while a 1st job's job-run.sh is running?
Hermetic (fake gcloud, stub job-run.sh that sleeps). Copies G1/infra (read-only source) to a temp dir.
Run: python3 test_runner_jam.py [G1_DIR]   Exit 0 always; prints JSON verdict."""
import json, os, pathlib, shutil, subprocess, sys, tempfile, time

G1 = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                  "/home/agent/workspaces/eurisko-study/traders-net/study/G1")
FAKE = r'''
gcloud() { if [[ "$*" == *"instances list"* ]]; then return 0; fi; if [[ "$*" == *"disks list"* ]]; then return 0; fi; return 0; }; export -f gcloud
'''

def trial(patch_fd):
    t = tempfile.TemporaryDirectory(dir="/tmp")
    g1 = pathlib.Path(t.name) / "study" / "G1"
    shutil.copytree(G1 / "infra", g1 / "infra", ignore=shutil.ignore_patterns("state", "tests", "__pycache__"))
    (g1 / "infra/state").mkdir()
    with (g1 / "infra/lib.sh").open("a") as f:
        f.write("\npost_status() { :; }\n")
    (g1 / "infra/job-run.sh").write_text("#!/bin/bash\necho $$ > \"$(dirname $0)/../queue/running/$1/runner.pid\"\nsleep 25\n")
    os.chmod(g1 / "infra/job-run.sh", 0o755)
    if patch_fd:  # the fix proposed in runner.log 06:54Z
        p = g1 / "infra/queue-runner.sh"
        p.write_text(p.read_text().replace('job-run.sh" "$j" </dev/null', 'job-run.sh" "$j" 7>&- </dev/null'))
    (g1.parent / "X1").mkdir()
    src = pathlib.Path(t.name) / "src"; src.mkdir(); (src / "run.sh").write_text("true\n")
    env = dict(os.environ)
    sh = lambda s: subprocess.run(["bash", "-c", FAKE + s], cwd=g1, text=True, capture_output=True, env=env)
    sh("bash infra/submit.sh --ws X1 --name a --src %s --hours 0.1" % src)
    r1 = sh("bash infra/queue-runner.sh --once")
    time.sleep(1)
    sh("bash infra/submit.sh --ws X1 --name b --src %s --hours 0.1" % src)
    r2 = sh("bash infra/queue-runner.sh --once")
    running = sorted(os.listdir(g1 / "queue/running")); pending = sorted(os.listdir(g1 / "queue/pending"))
    out = dict(patched=patch_fd, pass1=r1.stdout.strip(), pass2=r2.stdout.strip(), running=running, pending=pending)
    subprocess.run(["pkill", "-f", str(g1 / "infra/job-run.sh")])
    t.cleanup()
    return out

res = dict(checked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), source=str(G1 / "infra/queue-runner.sh"),
           as_shipped=trial(False), with_7_closed=trial(True))
res["jam_reproduced"] = res["as_shipped"]["pending"] == ["x1-b"] and res["as_shipped"]["pass2"] == ""
res["fix_works"] = "x1-b" in res["with_7_closed"]["running"]
print(json.dumps(res, indent=1))
