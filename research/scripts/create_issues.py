#!/usr/bin/env python
"""Create GitHub labels, milestones and issues from research/issues/*.md (front matter: key, title, labels,
milestone). Cross-references written as #KEY are rewritten to real issue numbers after creation.

  python scripts/create_issues.py --repo prajak002/Falcon --dry-run
  python scripts/create_issues.py --repo prajak002/Falcon
Issues must be enabled on the repo (gh repo edit <repo> --enable-issues). Re-running skips existing titles.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

ISSUES = Path(__file__).resolve().parents[1] / "issues"
MILESTONES = {"M1: infrastructure (Nov 2)": "2026-11-02T23:59:59Z",
              "M2: main results (Nov 9)": "2026-11-09T23:59:59Z",
              "M3: paper (Nov 16)": "2026-11-16T23:59:59Z"}
LABELS = {"experiment": "1d76db", "spotlight": "fbca04", "method": "5319e7", "infrastructure": "c5def5",
          "gpu-host": "0e8a16", "critical-path": "b60205", "security": "d93f0b", "writing": "bfdadc",
          "documentation": "0075ca"}


def gh(*args, check=True):
    r = subprocess.run(["gh", *args], capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f"gh {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def parse(path: Path) -> dict:
    _, front, body = path.read_text().split("---", 2)
    meta = {k.strip(): v.strip().strip('"') for k, v in (l.split(":", 1) for l in front.strip().splitlines())}
    meta["labels"] = [x.strip() for x in meta["labels"].split(",")]
    meta["body"] = body.strip()
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    issues = [parse(p) for p in sorted(ISSUES.glob("*.md"))]
    if a.dry_run:
        for i in issues:
            print(f"[{i['milestone']}] {i['title']}  labels={i['labels']}")
        return
    for name, color in LABELS.items():
        gh("label", "create", name, "--color", color, "--force", "-R", a.repo)
    have = {m["title"] for m in json.loads(gh("api", f"repos/{a.repo}/milestones?state=all"))}
    for title, due in MILESTONES.items():
        if title not in have:
            gh("api", f"repos/{a.repo}/milestones", "-f", f"title={title}", "-f", f"due_on={due}")
    existing = {i["title"]: i["number"] for i in json.loads(gh("issue", "list", "-R", a.repo, "--state", "all",
                                                               "--limit", "500", "--json", "title,number"))}
    numbers = {}
    for i in issues:
        if i["title"] in existing:
            numbers[i["key"]] = existing[i["title"]]
            continue
        url = gh("issue", "create", "-R", a.repo, "--title", i["title"], "--body", i["body"],
                 "--milestone", i["milestone"], *sum((["--label", l] for l in i["labels"]), [])).strip()
        numbers[i["key"]] = int(url.rsplit("/", 1)[1])
        print(f"#{numbers[i['key']]:<4} {i['title']}")
    for i in issues:
        body = re.sub(r"#([A-Z]+)\b", lambda m: f"#{numbers[m[1]]}" if m[1] in numbers else m[0], i["body"])
        if body != i["body"]:
            gh("issue", "edit", str(numbers[i["key"]]), "-R", a.repo, "--body", body)


if __name__ == "__main__":
    main()
