import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

config_path = Path(sys.argv[1]).resolve()
config_bytes = config_path.read_bytes()
config = json.loads(config_bytes)
output = Path(config["supervisor_directory"])
output.mkdir()
env = os.environ.copy()
env.update(config["environment"])
command = ["/usr/bin/time", "-v", "-o", str(output / "time.txt"),
           "timeout", "--kill-after=10s", "1800s", *config["candidate_command"]]
start = time.monotonic()
record = {"started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
          "command": command, "cwd": config["cwd"],
          "environment_overrides": config["environment"],
          "collection_plan_sha256": config["collection_plan_sha256"],
          "single_invocation": True, "automatic_retry": False}
with (output / "start.json").open("x") as stream:
    json.dump(record, stream, indent=2)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
with (output / "stdout.log").open("xb") as stdout, (output / "stderr.log").open("xb") as stderr:
    result = subprocess.run(command, cwd=config["cwd"], env=env, stdout=stdout, stderr=stderr)
record = {"finished_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
          "exit_code": result.returncode, "wall_seconds": time.monotonic() - start,
          "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
          "files": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                    for name in ("start.json", "stdout.log", "stderr.log", "time.txt")}}
with (output / "result.json").open("x") as stream:
    json.dump(record, stream, indent=2)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
print(json.dumps(record), flush=True)
sys.exit(result.returncode)
