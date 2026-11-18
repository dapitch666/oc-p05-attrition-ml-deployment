#!/usr/bin/env python3
"""Prefix the commit message with the type taken from the branch name.

On branch ``feat/predict-endpoint``, ``git commit -m "add endpoint"`` becomes
``feat: add endpoint``. Nothing changes on ``main``, on branches without a known
type, for merges, squashes and amends, or when the message already has a prefix.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

TYPES = ("feat", "fix", "test", "docs", "ci", "chore", "refactor")
BRANCH_RE = re.compile(rf"^({'|'.join(TYPES)})/")
PREFIXED_RE = re.compile(rf"^({'|'.join(TYPES)})(\([^)]+\))?!?: ")
SKIPPED_SOURCES = {"merge", "squash", "commit"}  # "commit" covers --amend, -c, -C


def current_branch() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip()


def main() -> int:
    message_file = Path(sys.argv[1])
    # pre-commit passes the message file as an argument and the source in an env var.
    source = os.environ.get("PRE_COMMIT_COMMIT_MSG_SOURCE", "")
    if source in SKIPPED_SOURCES:
        return 0

    match = BRANCH_RE.match(current_branch())
    if match is None:
        return 0
    prefix = f"{match.group(1)}: "

    lines = message_file.read_text(encoding="utf-8").split("\n")
    first = lines[0]
    if first.startswith("#") or PREFIXED_RE.match(first):
        return 0
    lines[0] = prefix + first  # empty first line (editor): the user types after the prefix
    message_file.write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
