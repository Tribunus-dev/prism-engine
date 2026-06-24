//! Hermeticity guard — no test may load configuration from a root-relative
//! `config.json` or `model.safetensors` path.  All test fixtures must live
//! under `tests/fixtures/` and be referenced via `CARGO_MANIFEST_DIR` or
//! `include_str!`.
//!
//! This guard scans tracked Rust source files (`src/**/*.rs`, `tests/**/*.rs`)
//! for forbidden patterns.  It allows explicit fixture paths under
//! `tests/fixtures/`.

// The guard must also pass when `generation-image` is disabled.
#![cfg(feature = "generation-image")]

use std::path::Path;

/// Patterns that indicate a test loads configuration relative to the crate root.
const FORBIDDEN_PATTERNS: &[&str] = &[
    r#"read_to_string("config.json")"#,
    r#"read_to_string("model.safetensors")"#,
    r#"File::open("config.json")"#,
    r#"File::open("model.safetensors")"#,
    r#"Path::new("config.json")"#,
    r#"Path::new("model.safetensors")"#,
];

#[test]
fn no_root_relative_config_reads() {
    // Walk src/ and tests/ for Rust source files
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut violations: Vec<String> = Vec::new();

    for root in &["src", "tests"] {
        let dir = manifest_dir.join(root);
        if !dir.is_dir() {
            continue;
        }
        visit_dir(&dir, &dir, &mut violations);
    }

    if violations.is_empty() {
        return;
    }

    eprintln!("Hermeticity violations found (root-relative config reads):");
    for v in &violations {
        eprintln!("  {v}");
    }
    panic!(
        "{} hermeticity violation(s) — use CARGO_MANIFEST_DIR + tests/fixtures/ paths instead",
        violations.len()
    );
}

fn visit_dir(base: &Path, dir: &Path, violations: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            visit_dir(base, path.as_path(), violations);
        } else if path.extension().is_some_and(|e| e == "rs") {
            check_file(base, &path, violations);
        }
    }
}

fn check_file(base: &Path, path: &Path, violations: &mut Vec<String>) {
    let Ok(content) = std::fs::read_to_string(path) else {
        return;
    };
    // Relative path for display
    let rel = path
        .strip_prefix(base)
        .unwrap_or(path)
        .display()
        .to_string();

    for (line_no, line) in content.lines().enumerate() {
        let trimmed = line.trim();
        // Skip comments, fixture paths, and raw-string pattern definitions
        if trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with("*")
            || trimmed.starts_with("r#") || trimmed.starts_with("]")
        {
            continue;
        }
        // Allow tests/fixtures/ paths
        if trimmed.contains("tests/fixtures/") {
            continue;
        }
        for pattern in FORBIDDEN_PATTERNS {
            if trimmed.contains(pattern) {
                violations.push(format!("{rel}:{ln}: {line}", ln = line_no + 1, line = trimmed));
            }
        }
    }
}
