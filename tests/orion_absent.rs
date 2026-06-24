//! Orion identifier absence guard.
//!
//! Scans tracked source files under the prism-engine workspace tree for
//! Orion identifiers that have been removed.  If any reappear this test
//! fails with the exact file, line, and matching content.
//!
//! The workspace root is defined as 3 directory levels up from this test
//! file (tests/ → package root → workspace root).  From there we walk
//! `prism-engine/` recursively, which also covers the `compute-core/`
//! member crate.

#![doc(hidden)]

use std::fs;
use std::path::{Path, PathBuf};

/// Case-insensitive patterns that must not appear in source files.
const FORBIDDEN: &[&str] = &[
    "orion_release_program",
    "orion_compile_mil",
    "orion_eval",
    "orion_tensor_from_external",
    "orion_tensor_from_arena",
    "orion_tensor_release",
    "orion_ane_init",
    "register_orion_rules",
    "OrionImportLedger",
    "OrionImportEntry",
    "orion_runtime",
    "orion_benchmark",
    "orion_root",
    "orion_bridge.rs",
    "ane_program_cache.rs",
    "ane_weight_dict.rs",
];

/// Tracked file extensions (lowercase).
const TRACKED_EXTENSIONS: &[&str] = &[".rs", ".mm", ".h", ".cpp", ".md", ".toml"];

/// Directory names to skip entirely.
const SKIP_DIRS: &[&str] = &["target", ".git", "node_modules"];

// ── Helpers ─────────────────────────────────────────────────────────────

/// Resolve the workspace root: 3 levels up from this test file.
///
/// At compile time `file!()` evaluates to `tests/orion_absent.rs`
/// (relative to the package root).  We join it with `CARGO_MANIFEST_DIR`
/// to get the absolute path, then walk up three times:
///
///   tests/ → package root (prism-engine/) → workspace root
fn workspace_root() -> PathBuf {
    let abs = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(Path::new(file!()));
    abs.parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .expect("workspace root is 3 levels above the test file")
        .to_path_buf()
}

/// Resolve this test file's own absolute path (to exclude from scanning).
fn self_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(Path::new(file!()))
        .canonicalize()
        .expect("test file should exist")
}

/// Return `true` when `entry` should be skipped (unsupported extension,
/// excluded directory, hidden directory).
fn should_skip(entry: &fs::DirEntry) -> bool {
    let ft = entry.file_type().ok();
    let name = entry.file_name();
    let name_str = name.to_string_lossy();

    // Skip excluded directories
    if ft.map_or(true, |t| t.is_dir() || t.is_symlink()) {
        // Skip hidden directories (like .git)
        if name_str.starts_with('.') {
            return true;
        }
        return SKIP_DIRS.iter().any(|d| name_str == *d);
    }

    // Accept known extensions
    let path = entry.path();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        let dot_ext = format!(".{}", ext.to_lowercase());
        if TRACKED_EXTENSIONS.contains(&dot_ext.as_str()) {
            return false;
        }
    }

    // Skip everything else
    true
}

/// Scan a single file for forbidden patterns, collecting matches.
fn check_file(path: &Path, matches: &mut Vec<String>) {
    let Ok(content) = fs::read_to_string(path) else {
        return;
    };

    let content_lower = content.to_lowercase();

    // Quick bail-out: no forbidden string present at all.
    if !FORBIDDEN
        .iter()
        .any(|p| content_lower.contains(&p.to_lowercase()))
    {
        return;
    }

    // Line-by-line exact match for reporting.
    for (lineno, line) in content.lines().enumerate() {
        let line_lower = line.to_lowercase();
        for pat in FORBIDDEN {
            if line_lower.contains(&pat.to_lowercase()) {
                matches.push(format!(
                    "  {}:{} -- {}",
                    path.display(),
                    lineno + 1,
                    line.trim(),
                ));
            }
        }
    }
}

/// Recursively walk a directory tree collecting forbidden matches.
fn walk_dir(dir: &Path, matches: &mut Vec<String>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };

    let own_path = self_path();

    for entry in entries {
        let Ok(entry) = entry else {
            continue;
        };

        if should_skip(&entry) {
            continue;
        }

        let path = entry.path();

        // Skip the test file itself — it necessarily contains the forbidden
        // patterns in its FORBIDDEN array and in error messages.
        if !entry.file_type().ok().map_or(false, |t| t.is_dir())
            && path.canonicalize().ok().as_deref() == Some(&own_path)
        {
            continue;
        }

        if entry.file_type().ok().map_or(false, |t| t.is_dir()) {
            walk_dir(&path, matches);
        } else {
            check_file(&path, matches);
        }
    }
}

// ── Test ────────────────────────────────────────────────────────────────

#[test]
fn no_orion_identifiers() {
    let root = workspace_root().join("prism-engine");
    let mut matches = Vec::new();
    walk_dir(&root, &mut matches);

    assert!(
        matches.is_empty(),
        "Found {} forbidden Orion identifier(s):\n{}",
        matches.len(),
        matches.join("\n"),
    );
}
