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
# Match any `mod <name> {` that follows `#[cfg(test)]` — the standard convention
# is `mod tests {`, but in practice modules are often named after the test group
# (e.g. `mod recovery_tests {`, `mod property_tests {`). The `#[cfg(test)]` gate
# is the authoritative marker, not the module name.
MOD_AFTER_TEST = re.compile(r"^\s*mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*\{")

EXCLUDED_PATH_PARTS = ("compute-core.legacy", "target/", "/.git/")


def find_test_module_ranges(lines: list[str]) -> list[tuple[int, int]]:
    """Find (start, end) line indices for each `#[cfg(test)] mod tests { ... }` block.

    Handles the canonical `#[cfg(test)]\\nmod tests { ... }` shape where the
    `#[cfg(test)]` attribute and the `mod tests {` declaration are on
    consecutive lines. The match is by `mod tests` exactly (other test
    modules named differently are not detected; the audit captures the
    standard convention).

    Brace counting is string-literal-aware: a `{` or `}` inside a string
    literal, char literal, line comment, or block comment does not count
    toward depth. This matters for format strings like
    `format!("got: {err}")` which contain `{` / `}` characters that would
    otherwise throw off naive counting.
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
            if j < n and MOD_AFTER_TEST.match(lines[j]):
                start = j
                end = _find_matching_brace(lines, j)
                if end is not None:
                    ranges.append((start, end))
                    i = end + 1
                    continue
                else:
                    i = n  # unbalanced — give up
        i += 1
    return ranges


def _find_matching_brace(lines: list[str], start: int) -> int | None:
    """Find the line index of the `}` that closes the `{` on `lines[start]`.

    Uses a string-literal-aware character parser. The opening brace is the
    `{` at the end of `mod tests {` on `lines[start]`. Returns the line
    containing the matching `}` (which may be the same line as a closing
    brace that shares a line with the body), or None if unbalanced.

    Handles regular strings, char literals, raw strings (r"..." and
    r#"..."#), byte strings (b"..." and br"..."), and CString/CLike
    comments. Raw string hashes (#, ##, etc.) are counted so that
    `r#"}"#` correctly closes the raw string.

    The raw-string and block-comment states are tracked ACROSS lines so
    that multi-line raw strings and block comments are correctly skipped.
    """
    depth = 0
    n = len(lines)
    in_raw_string = 0  # 0 = not in raw string, >0 = number of #s
    in_block_comment = False
    for k in range(start, n):
        line = lines[k]
        pos = 0
        # On the start line, skip the opening `mod tests {` by fast-forwarding
        # to the first `{` if present, then counting from depth=1.
        if k == start:
            # Find the `{` of `mod tests {`
            brace_pos = line.find("{")
            if brace_pos < 0:
                return None
            depth = 1
            pos = brace_pos + 1
        while pos < len(line):
            # If we're inside a multi-line raw string, look for the closing
            # pattern `"` + N hashes. The raw string started on an earlier
            # line and continues until we find the close.
            if in_raw_string > 0:
                if line[pos] == '"':
                    end_hashes = 0
                    q = pos + 1
                    while q < len(line) and line[q] == "#" and end_hashes < in_raw_string:
                        end_hashes += 1
                        q += 1
                    if end_hashes == in_raw_string:
                        in_raw_string = 0
                        pos = q
                        continue
                pos += 1
                continue
            # If we're inside a multi-line block comment, look for `*/`.
            if in_block_comment:
                if line[pos] == "*" and pos + 1 < len(line) and line[pos + 1] == "/":
                    in_block_comment = False
                    pos += 2
                    continue
                pos += 1
                continue
            ch = line[pos]
            # Fast-forward the common case: whitespace, identifier chars, digits,
            # operators, and any character that cannot start a string, comment,
            # or brace. This is the critical fix: without this, any character
            # that does not match the special cases below would loop forever.
            if ch not in 'rRbB"\'/{}':
                pos += 1
                continue
            # Raw string: r"..." or r#"..."# (any number of #s).
            if ch == "r" and pos + 1 < len(line) and line[pos + 1] in ('"', '#'):
                pos += 1  # past 'r'
                # Count leading #s
                hashes = 0
                while pos < len(line) and line[pos] == "#":
                    hashes += 1
                    pos += 1
                if pos < len(line) and line[pos] == '"':
                    pos += 1  # past opening '"'
                    in_raw_string = hashes  # may close on same line
                else:
                    continue  # not actually a raw string
                # Try to close on the same line
                while pos < len(line):
                    if line[pos] == '"':
                        end_hashes = 0
                        q = pos + 1
                        while q < len(line) and line[q] == "#" and end_hashes < hashes:
                            end_hashes += 1
                            q += 1
                        if end_hashes == hashes:
                            in_raw_string = 0
                            pos = q
                            break
                    pos += 1
                continue
            # Byte string: b"..." or br"..." or br#"..."#.
            if ch == "b" and pos + 1 < len(line) and line[pos + 1] in ('"', 'r'):
                if line[pos + 1] == "r":
                    # Could be br"..." or br#"..."# (or just an identifier like
                    # "broadcast" — only treat as byte string if pos+2 is " or #).
                    if pos + 2 >= len(line) or line[pos + 2] not in ('"', '#'):
                        pos += 1  # not a byte string, just `br` in an identifier
                        continue
                    pos += 2
                    hashes = 0
                    while pos < len(line) and line[pos] == "#":
                        hashes += 1
                        pos += 1
                    if pos < len(line) and line[pos] == '"':
                        pos += 1
                        in_raw_string = hashes
                    else:
                        continue
                    while pos < len(line):
                        if line[pos] == '"':
                            end_hashes = 0
                            q = pos + 1
                            while q < len(line) and line[q] == "#" and end_hashes < hashes:
                                end_hashes += 1
                                q += 1
                            if end_hashes == hashes:
                                in_raw_string = 0
                                pos = q
                                break
                        pos += 1
                    continue
                # line[pos + 1] == '"' here.
                pos += 2
                while pos < len(line):
                    if line[pos] == "\\":
                        pos += 2
                        continue
                    if line[pos] == '"':
                        pos += 1
                        break
                    pos += 1
                continue
            # Regular string literal
            if ch == '"':
                pos += 1
                while pos < len(line):
                    if line[pos] == "\\":
                        pos += 2
                        continue
                    if line[pos] == '"':
                        pos += 1
                        break
                    pos += 1
                continue
            # Char literal
            if ch == "'":
                pos += 1
                while pos < len(line):
                    if line[pos] == "\\":
                        pos += 2
                        continue
                    if line[pos] == "'":
                        pos += 1
                        break
                    pos += 1
                continue
            # Line comment
            if ch == "/" and pos + 1 < len(line) and line[pos + 1] == "/":
                break  # rest of line is comment
            # Block comment (multi-line)
            if ch == "/" and pos + 1 < len(line) and line[pos + 1] == "*":
                pos += 2
                in_block_comment = True
                while pos + 1 < len(line):
                    if line[pos] == "*" and line[pos + 1] == "/":
                        in_block_comment = False
                        pos += 2
                        break
                    pos += 1
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return k
            pos += 1
    return None


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
    # Sibling-file form: src/.../<module>/tests.rs. The whole file is
    # `#[cfg(test)] mod tests { ... }`, so all unwraps are test-scope.
    if path.name == "tests.rs" and "/src/" in p:
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
