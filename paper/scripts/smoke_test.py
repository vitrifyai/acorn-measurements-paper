#!/usr/bin/env python3
"""Fast, dependency-free integrity check for the public paper package."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REQUIRED = (
    "CITATION.cff",
    "DATA_MANIFEST.csv",
    "LICENSE",
    "README.md",
    "REPRODUCING_THE_PAPER.md",
    "SHA256SUMS",
    "paper/scripts/release_paths.py",
)
FORBIDDEN_PATH_PARTS = (
    "/home/",
    "/nas-",
    "/raid/",
    "/kriosdata/",
    "/emmadrive/",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    errors: list[str] = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            errors.append(f"missing required file: {relative}")

    checksum_file = ROOT / "SHA256SUMS"
    if checksum_file.is_file():
        for line_number, line in enumerate(checksum_file.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                expected, relative = line.split(maxsplit=1)
            except ValueError:
                errors.append(f"invalid checksum line {line_number}")
                continue
            path = ROOT / relative.strip().lstrip("*")
            if not path.is_file():
                errors.append(f"checksum target missing: {relative}")
            elif sha256(path) != expected:
                errors.append(f"checksum mismatch: {relative}")

    json_count = 0
    for path in ROOT.rglob("*.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
            json_count += 1
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append(f"invalid JSON: {path.relative_to(ROOT)} ({exc})")

    scanned = 0
    excluded = {"SHA256SUMS", "DATA_MANIFEST.csv"}
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or ".git" in path.parts
            or path.name in excluded
            or path == Path(__file__).resolve()
        ):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        scanned += 1
        if any(part in text for part in FORBIDDEN_PATH_PARTS):
            errors.append(f"workstation path found: {path.relative_to(ROOT)}")

    if errors:
        print("ACORN release smoke test: FAIL", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(
        "ACORN release smoke test: PASS "
        f"({json_count} JSON files; {scanned} text files; checksums verified)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
