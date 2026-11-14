"""Results JSON: build, validate, write. Schema: result_schema.json (aggbench.result/v1)."""
import datetime
import hashlib
import json
import os
import subprocess
from pathlib import Path

from . import __version__

HERE = Path(__file__).resolve().parent
SCHEMA = json.loads((HERE / "result_schema.json").read_text())


def harness_info():
    """Version, git commit, whether aggbench/ differs from that commit, sha256 of the harness sources."""
    def git(*args):
        return subprocess.run(["git", "-C", str(HERE), *args], capture_output=True, text=True).stdout.strip()
    files = sorted(p for p in HERE.rglob("*") if p.suffix in (".py", ".json") and "__pycache__" not in p.parts
                   and not p.name.startswith("baselines_"))  # generated tables, not harness code
    h = hashlib.sha256()
    for p in files:
        h.update(str(p.relative_to(HERE)).encode() + b"\0" + p.read_bytes() + b"\0")
    commit = git("rev-parse", "HEAD")
    info = {"version": __version__, "git_commit": commit or "unknown",
            "git_dirty": bool(git("status", "--porcelain", "--", str(HERE))), "sha256": h.hexdigest()}
    if not commit and os.environ.get("AGGBENCH_GIT_COMMIT"):
        # a copy without git metadata (e.g. a synced checkout on a remote GPU VM): the commit comes from the
        # environment and cleanliness is proven by the sources hashing to that commit's sha256
        expect = os.environ.get("AGGBENCH_EXPECT_SHA256", "")
        info.update(git_commit=os.environ["AGGBENCH_GIT_COMMIT"], git_dirty=info["sha256"] != expect,
                    git_source="env, clean iff sha256 == AGGBENCH_EXPECT_SHA256")
    return info


def split_record(batch, split):
    return {"split": split, "first_seed": int(batch.seeds[0]), "n": int(len(batch))}


def make_result(run_id, track, env, visibility, splits, oracle_check, methods,
                learned_arms=(), controls=None, smoke=False, extra=None, allow_dirty=False):
    """Assemble and validate a result dict.

    env: the environment object; splits: {role: split_record(...)}; methods: {name: metrics.summarize(...)};
    controls: {arm: {"random_network": method, "untrained_init": method}} for every learned arm.
    allow_dirty: accept a harness with uncommitted changes (git_dirty); recorded as harness.dirty_allowed.
    Run experiments from your own git worktree so the harness stays clean (README).
    """
    harness = harness_info()
    if allow_dirty and harness["git_dirty"]:
        harness["dirty_allowed"] = True
    result = {
        "schema": "aggbench.result/v1", "run_id": run_id, "track": track, "smoke": bool(smoke),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "harness": harness,
        "env": {"name": env.name, "config": _jsonable(env.config.__dict__)},
        "visibility": visibility, "splits": splits, "oracle_check": oracle_check,
        "methods": methods, "learned_arms": list(learned_arms), "controls": controls or {},
        "extra": extra or {},
    }
    validate(result)
    return result


def validate(result, allow_dirty=False):
    """Raise ValueError if the result breaks the schema or the experiment rules. A result stamped
    harness.git_dirty=true is rejected unless allow_dirty=True or it was built with make_result(allow_dirty=True)
    (which records harness.dirty_allowed=true)."""
    try:
        import jsonschema
        jsonschema.validate(result, SCHEMA)
    except ImportError:
        missing = [k for k in SCHEMA["required"] if k not in result]
        if missing:
            raise ValueError(f"result missing keys {missing}")
    except jsonschema.ValidationError as err:
        raise ValueError(f"result breaks schema: {err.message}") from None
    h = result["harness"]
    if h.get("git_dirty") and not (allow_dirty or h.get("dirty_allowed")):
        raise ValueError("harness.git_dirty is true: the aggbench sources differ from the recorded commit. Run from "
                         "your own git worktree at a clean commit, or pass allow_dirty=True explicitly")
    if not result["oracle_check"].get("verified"):
        raise ValueError("oracle_check not verified")
    if result["smoke"] != (result["track"] == "SMOKE"):
        raise ValueError("smoke=true exactly when track is SMOKE")
    for arm in result["learned_arms"]:
        ctl = result["controls"].get(arm)
        if not ctl:
            raise ValueError(f"learned arm {arm!r} has no controls entry")
        for name in (arm, ctl["random_network"], ctl["untrained_init"]):
            if name not in result["methods"]:
                raise ValueError(f"method {name!r} listed for arm {arm!r} has no metrics")
    if "bayes" not in result["methods"]:
        raise ValueError("methods must include 'bayes' (the ceiling)")


def write_result(directory, result, name="result.json", allow_dirty=False):
    validate(result, allow_dirty=allow_dirty)
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text(json.dumps(result, indent=2) + "\n")
    return path / name


def _jsonable(obj):
    return json.loads(json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
