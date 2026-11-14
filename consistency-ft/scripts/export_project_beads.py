#!/usr/bin/env python3
"""Export only consistency-ft's canonical Beads epic for repository collaborators."""
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
EPIC = "hub-t8k9c"
FIELDS = ("id", "title", "description", "status", "priority", "issue_type",
          "labels", "created_at", "updated_at", "closed_at", "close_reason")


def fetch(command):
    result = subprocess.run(
        ["ssh", "deploy-host", f"bd -C /home/kelvin/hub --readonly {command} {EPIC} --json"],
        check=True, text=True, capture_output=True, timeout=240,
    )
    return json.loads(result.stdout)


def main():
    raw = fetch("show") + fetch("children")
    records = {}
    for row in raw:
        identifier = row["id"]
        if identifier != EPIC and not identifier.startswith(EPIC + "."):
            raise ValueError(f"Refusing unrelated issue {identifier}")
        if "consistency-ft" not in row.get("labels", []):
            raise ValueError(f"Missing project label: {identifier}")
        record = {key: row[key] for key in FIELDS if key in row}
        record["dependencies"] = [
            {"depends_on_id": dep["depends_on_id"], "type": dep["type"]}
            for dep in row.get("dependencies", [])
        ]
        records[identifier] = record
    if EPIC not in records or len(records) < 2:
        raise ValueError("Incomplete project export")
    issues = sorted(records.values(), key=lambda row: int(row["id"].split(".")[-1]) if row["id"] != EPIC else 0)
    for issue in issues:
        for dependency in issue["dependencies"]:
            if dependency["depends_on_id"] not in records:
                raise ValueError("External dependency requires explicit review before sharing")
    snapshot = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "epic": EPIC,
        "source": "Canonical Hub Beads, read-only export",
        "live_plan": "https://consistency.futarchy.ai/plan.html",
        "issues": issues,
    }
    destination = ROOT / "project/beads.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
    lines = ["# Consistency research tasks", "",
        "The project is tracked under Beads epic **`hub-t8k9c`**. "
        "[Open the live task graph](https://consistency.futarchy.ai/plan.html) "
        "for statuses refreshed every five minutes.", "",
        "This repository includes a [project-only task snapshot](../project/beads.json), "
        "so every repository collaborator can read the task descriptions and dependencies. "
        "No access to the private Hub or fleet dashboard is needed.", "",
        "To propose work or a status change, [open a repository issue](https://github.com/futarchy-fi/consistency-ft/issues/new) "
        "and put the Beads ID in its title, or reference that ID in a pull request. "
        "A project maintainer applies accepted changes to the canonical tracker; "
        "GitHub issues do not sync automatically.", "",
        f"Snapshot exported: {snapshot['exported_at']}. The statuses below are recorded task states, "
        "not a fresh verification of whether an old operational blocker still exists.", "",
        "| Beads ID | Task | Status | Prerequisites |", "|---|---|---|---|",
    ]
    for row in issues:
        title = row["title"].replace("|", "\\|")
        dependencies = ", ".join("`" + d["depends_on_id"] + "`" for d in row["dependencies"] if d["type"] == "blocks") or "—"
        lines.append(f"| `{row['id']}` | {title} | {row['status']} | {dependencies} |")
    lines += ["", "Maintainers refresh this export with `python3 scripts/export_project_beads.py` "
              "from an authorized fleet environment, review the diff, and commit both generated files.", ""]
    (ROOT / "docs/BEADS.md").write_text("\n".join(lines))
    print(f"Exported {len(issues)} project issues to {destination}")


if __name__ == "__main__":
    main()
