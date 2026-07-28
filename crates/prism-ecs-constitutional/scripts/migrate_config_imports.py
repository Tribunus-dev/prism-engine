#!/usr/bin/env python3
"""Migrate `crate::ecs::config::` imports to `prism_ecs_constitutional::config::`.

Run as: python3 migrate_config_imports.py <files>...
"""
import re
import sys

# Patterns to handle:
# 1. use crate::ecs::config::Type;
# 2. use crate::ecs::config::{Type1, Type2};
# 3. use crate::ecs::config::module::Type;
# 4. pub use crate::ecs::config::Type as Alias;
# 5. use crate::ecs::config::parser::{Type1, Type2};
# 6. crate::ecs::config::function_call(...)
# 7. use crate::ecs::config::Type::{Variant1, Variant2};
# 8. crate::ecs::config::resolve_namespace(...)

# Regex to match `crate::ecs::config::` (with optional preceding `use ` or `pub use `)
# We need to replace just the path portion, preserving the surrounding code.

# The pattern is: (use\s+|pub\s+use\s+)?crate::ecs::config::

# We'll do a simple text replacement for now, since the pattern is well-defined.
PATTERN = re.compile(r"(\b(?:pub\s+)?use\s+)crate::ecs::config::")
PATH_PATTERN = re.compile(r"\bcrate::ecs::config::")


def migrate_file(path: str) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    new_content = PATTERN.sub(r"\1prism_ecs_constitutional::config::", content)
    # Replace remaining `crate::ecs::config::` references (in function calls, type paths, etc.)
    new_content = PATH_PATTERN.sub("prism_ecs_constitutional::config::", new_content)

    if new_content != content:
        with open(path, "w", encoding="utf-8") as f:
            f.write(new_content)
        return True
    return False


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <files>...", file=sys.stderr)
        sys.exit(1)
    changed = 0
    for path in sys.argv[1:]:
        if migrate_file(path):
            changed += 1
            print(f"updated: {path}")
    print(f"Total files updated: {changed}")


if __name__ == "__main__":
    main()
