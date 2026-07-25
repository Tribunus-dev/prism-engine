#!/usr/bin/env python3
"""
Authoritative production-scope unwrap/expect baseline for Prism.

Counts `.unwrap()` and `.expect()` calls per file, split by scope:
  - **production**: lines outside any `#[cfg(test)] mod tests { ... }` block.
  - **test**:       lines inside any `#[cfg(test)] mod tests { ... }` block.

The rust-quality rule (see references/rust-quality.md) permits unwraps in
test scope and requires typed errors in production scope. The production
count is the migration backlog; the test count is excluded.

Usage:
  python3 scripts/unwrap_baseline.py              # scan canonical paths (default)
  python3 scripts/unwrap_baseline.py --top 25     # show only top 25 files
  python3 scripts/unwrap_baseline.py --json       # machine-readable output
  python3 scripts/unwrap_baseline.py path/        # custom root

Excludes:
  - compute-core.legacy/  (archaeology, not production fix target)
  - target/               (cargo build artifacts)
  - /tests/ directories   (integration tests, file-level scope is test)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

UNWRAP_RE = re.compile(r"\.unwrap\(\)|\.expect\(")
ATTR_LINE = re.compile(r"^#\[cfg\(test\)\]\s*$")
MOD_TESTS_OPEN = re.compile(r"^\s*mod\s+tests\s*\{")

EXCLUDED_PATH_PARTS = ("compute-core.legacy", "target/", "/.git/")


def find_test_module_ranges(lines: list[str]) -> list[tuple[int, int]]:
    """Find (start, end) line indices for each `#[cfg(test)] mod tests { ... }` block.

    Handles the canonical `#[cfg(test)]\\nmod tests { ... }` shape where the
    `#[cfg(test)]` attribute and the `mod tests {` declaration are on
    consecutive lines. The match is by `mod tests` exactly (other test
    modules named differently are not detected; the audit captures the
    standard convention).
    """
    ranges: list[tuple[int, int]] = []
    n = len(lines)
    i = 0
    while i < n:
        if ATTR_LINE.match(lines[i].strip()):
            # Look ahead for the next non-blank, non-comment, non-attribute line.
            j = i + 1
            while j < n:
                s = lines[j].strip()
                if not s or s.startswith("//") or s.startswith("#["):
                    j += 1
                    continue
                break
            if j < n and MOD_TESTS_OPEN.match(lines[j]):
                depth = 0
                start = j
                for k in range(j, n):
                    depth += lines[k].count("{") - lines[k].count("}")
                    if depth == 0:
                        ranges.append((start, k))
                        i = k + 1
                        break
                else:
                    i = n
                continue
        i += 1
    return ranges


def classify_lines(lines: list[str]) -> tuple[set[int], set[int]]:
    """Return (production_line_set, test_line_set) for the given file lines.

    A line is `test` if it falls inside any `#[cfg(test)] mod tests { ... }` block.
    All other lines are `production`.
    """
    test_set: set[int] = set()
    for start, end in find_test_module_ranges(lines):
        for ln in range(start, end + 1):
            test_set.add(ln)
    return test_set - {ln for ln, _ in [(ln, None) for ln in test_set]}, test_set


def is_excluded(path: Path) -> bool:
    """A file is excluded if it is in a vendored, build, or test-only path."""
    p = str(path)
    for part in EXCLUDED_PATH_PARTS:
        if part in p:
            return True
    # Treat standalone /tests/*.rs files as test scope (the file-level scope
    # is test). Integration tests live in <crate>/tests/, not <crate>/src/.
    if "/tests/" in p and p.endswith(".rs"):
        return True
    return False


def count_file(path: Path) -> tuple[int, int]:
    """Return (production_count, test_count) for the file."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0, 0
    lines = text.split("\n")
    test_set = classify_lines(lines)[1]
    production = test = 0
    for i, line in enumerate(lines):
        if UNWRAP_RE.search(line):
            if i in test_set:
                test += 1
            else:
                production += 1
    return production, test


def collect_files(root: Path) -> list[Path]:
    """Find all .rs files under root, excluding vendored/build/test paths."""
    out: list[Path] = []
    for path in root.rglob("*.rs"):
        if is_excluded(path):
            continue
        out.append(path)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "root",
        nargs="?",
        default="crates",
        help="root directory to scan (default: crates/)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="show only top N files by production count (default: 30)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON instead of a table",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"error: {root} does not exist", file=sys.stderr)
        return 2

    files = collect_files(root)
    results: list[tuple[int, int, Path]] = []
    total_p = total_t = 0
    for f in files:
        p, t = count_file(f)
        if p + t > 0:
            results.append((p, t, f))
            total_p += p
            total_t += t

    results.sort(key=lambda r: (-r[0], -r[1]))

    if args.json:
        payload = {
            "root": str(root),
            "total_production": total_p,
            "total_test": total_t,
            "files": [
                {"production": p, "test": t, "path": str(f)}
                for p, t, f in results
            ],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(f'{"PROD":>5}  {"TEST":>5}  FILE')
        for p, t, f in results[: args.top]:
            print(f"{p:5d}  {t:5d}  {f}")
        print("---")
        print(f"Total production unwraps: {total_p}")
        print(f"Total test-scope unwraps:  {total_t}")
        print(f"Files with production unwraps: {sum(1 for p, _, _ in results if p > 0)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
