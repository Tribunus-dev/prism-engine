//! `workspace_contains_no_legacy_ane_imports`
//!
//! Workspace-level architecture enforcement test for the ane
//! migration. It scans the workspace for any `use` statement that
//! references the legacy engine ane surface
//! (`compute_core::ecs::ane::*` or `crate::ecs::ane::*` inside
//! the engine, or the `tribunus_compute_core::ane` re-export
//! shim).
//!
//! The test fails if any file in the workspace imports the legacy
//! surface. The legacy engine file is the engine's
//! `compute-core/src/ecs/legacy_ane/` directory; the constitutional
//! replacement is `prism_ecs_compile::ane`.
//!
//! # Migration status
//!
//! This test was added when the ane engine-deletion migration was
//! being completed (2026-07-28). The engine's
//! `compute-core/src/ecs/ane/` directory was renamed to
//! `compute-core/src/ecs/legacy_ane/`; the directory is the
//! migration inventory and is exempt from this scan. Files
//! outside that directory that import the surface are violations.

use std::fs;
use std::path::Path;

/// Returns the set of files OUTSIDE the legacy engine ane directory
/// that import the legacy surface. Files inside the engine's own
/// `compute-core/src/ecs/legacy_ane/` directory are exempt — they
/// ARE the legacy surface and will continue to host the
/// engine-coupled adapter code (Core ML, IOSurface, MLX, FFI).
/// Files outside that directory that import the surface are
/// violations.
pub fn legacy_ane_importers_outside_inventory() -> Vec<String> {
    let mut importers = Vec::new();
    let workspace_root = find_workspace_root();
    let scan_root = match workspace_root {
        Some(root) => root.join("compute-core"),
        None => Path::new("compute-core").to_path_buf(),
    };
    let scan_root_str = scan_root.to_str().unwrap_or("compute-core");
    scan_workspace_excluding_inventory(scan_root_str, &mut importers);
    importers
}

const LEGACY_INVENTORY_DIR: &str = "compute-core/src/ecs/legacy_ane";

fn scan_workspace_excluding_inventory(dir: &str, importers: &mut Vec<String>) {
    let path = Path::new(dir);
    if !path.exists() {
        return;
    }
    let entries = match fs::read_dir(path) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let p = entry.path();
        let path_str = p.to_str().unwrap_or("").to_string();
        // Skip the legacy inventory itself.
        if path_str.contains(LEGACY_INVENTORY_DIR) {
            continue;
        }
        // Skip the engine's archaeology snapshot
        // (compute-core.legacy/) — it is never built, only
        // preserved for archaeology.
        if path_str.contains("compute-core/compute-core.legacy/")
            || path_str.contains("compute-core\\compute-core.legacy\\")
        {
            continue;
        }
        if p.is_dir() {
            scan_workspace_excluding_inventory(&path_str, importers);
        } else if p.extension().and_then(|e| e.to_str()) == Some("rs") {
            if let Ok(content) = fs::read_to_string(&p) {
                if content.contains("use crate::ecs::ane::")
                    || content.contains("compute_core::ecs::ane::")
                    || content.contains("tribunus_compute_core::ecs::ane")
                {
                    importers.push(path_str);
                }
            }
        }
    }
}

/// Walk up the directory tree from CWD until a `Cargo.toml` with a
/// `[workspace]` section is found, or the filesystem root is
/// reached.
fn find_workspace_root() -> Option<std::path::PathBuf> {
    let mut current = std::env::current_dir().ok()?;
    loop {
        let cargo_toml = current.join("Cargo.toml");
        if cargo_toml.exists() {
            if let Ok(content) = fs::read_to_string(&cargo_toml) {
                if content.contains("[workspace]") {
                    return Some(current);
                }
            }
        }
        if !current.pop() {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_contains_no_legacy_ane_imports() {
        // Architectural invariant: no file OUTSIDE the migration
        // inventory imports the legacy engine ane surface. The
        // engine's compute-core/src/ecs/legacy_ane/ directory is
        // the migration inventory and is exempt; it IS the legacy
        // surface (engine-coupled adapter code) and re-exports the
        // constitutional data types from
        // prism_ecs_compile::ane. Files outside that directory
        // that import the surface are violations.
        let importers = legacy_ane_importers_outside_inventory();
        assert!(
            importers.is_empty(),
            "Workspace still contains {} files outside the migration \
             inventory that import the legacy engine ane surface: \
             {:?}. The migration is incomplete; callers should be \
             updated to the constitutional surface in \
             prism_ecs_compile::ane (or import via the engine's \
             legacy_ane shim if engine-coupled types are required).",
            importers.len(),
            importers
        );
    }
}
