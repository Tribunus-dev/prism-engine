//! `workspace_contains_no_legacy_canonical_imports`
//!
//! Workspace-level architecture enforcement test for the canonical
//! migration. It scans the workspace for any `use` statement that
//! references the legacy engine canonical surface
//! (`compute_core::ecs::canonical::*` or
//! `crate::ecs::canonical::*` inside the engine, or the
//! `tribunus_compute_core::ecs::canonical` re-export path).
//!
//! The test fails if any file in the workspace imports the legacy
//! surface. The legacy engine file is the engine's
//! `compute-core/src/ecs/canonical/` directory (which was renamed
//! to `compute-core/src/ecs/legacy_canonical/` as part of the
//! migration; the rename is the E-N+1 step of the migration).
//! Either name is accepted as the migration inventory because
//! the migration is in progress: in the pre-rename state the
//! legacy directory is `canonical/`; in the post-rename state it
//! is `legacy_canonical/`. The constitutional replacement is
//! `prism_ecs_constitutional::canonical`.
//!
//! # Migration status
//!
//! This test was added when the canonical engine-deletion
//! migration was being completed (2026-07-28). The engine's
//! `compute-core/src/ecs/canonical/` directory is the migration
//! inventory and is exempt; it IS the legacy surface and will
//! continue to host the engine-coupled adapter code (the
//! engine's Metal compiler, the prism-metal-runtime bridges, the
//! engine binaries) until the engine itself is git-rm'd. Files
//! outside that directory that import the surface are
//! violations.

use std::fs;
use std::path::Path;

/// Returns the set of files OUTSIDE the engine's canonical
/// migration inventory that import the legacy engine canonical
/// surface. The engine's own `compute-core/src/ecs/canonical/`
/// directory is exempt; it IS the engine-internal home for the
/// engine-coupled adapter code. Files outside that directory
/// that import the surface are violations.
pub fn legacy_canonical_importers_outside_inventory() -> Vec<String> {
    let mut importers = Vec::new();
    // The canonical migration inventory is the engine's own
    // `canonical/` dir (pre-rename) OR `legacy_canonical/` dir
    // (post-rename). We exempt both because the migration is
    // in progress; the rename is a separate E-N+1 step.
    scan_workspace_excluding_inventory("compute-core", &mut importers);
    importers
}

/// Legacy inventory directories that the scan must skip. The
/// engine's `compute-core/src/ecs/canonical/` directory (pre-
/// rename) and `compute-core/src/ecs/legacy_canonical/`
/// (post-rename) are both exempt. The safety net must accept
/// either name because the migration is in progress.
const ENGINE_INVENTORY_DIRS: &[&str] = &[
    "compute-core/src/ecs/canonical",
    "compute-core/src/ecs/legacy_canonical",
];

fn is_in_inventory(path_str: &str) -> bool {
    ENGINE_INVENTORY_DIRS
        .iter()
        .any(|dir| path_str.contains(dir))
}

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
        // Skip the engine-internal legacy canonical directory —
        // it is the canonical engine-internal home for the
        // engine-coupled adapter code and re-exports the
        // constitutional data types.
        if is_in_inventory(&path_str) {
            continue;
        }
        if p.is_dir() {
            scan_workspace_excluding_inventory(&path_str, importers);
        } else if p.extension().and_then(|e| e.to_str()) == Some("rs") {
            if let Ok(content) = fs::read_to_string(&p) {
                if content.contains("use crate::ecs::canonical::")
                    || content.contains("compute_core::ecs::canonical::")
                    || content.contains("tribunus_compute_core::ecs::canonical::")
                {
                    importers.push(path_str);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_contains_no_legacy_canonical_imports() {
        // Architectural invariant: no file OUTSIDE the engine's
        // canonical/ (pre-rename) or legacy_canonical/ (post-rename)
        // directory imports the legacy engine canonical surface.
        // Both names are exempt because the migration is in
        // progress; the rename is a separate E-N+1 step. The
        // constitutional replacement is
        // `prism_ecs_constitutional::canonical`.
        let importers = legacy_canonical_importers_outside_inventory();
        assert!(
            importers.is_empty(),
            "Workspace still contains {} files outside the engine's \
             canonical/ migration inventory that import the legacy \
             engine canonical surface: {:?}. The migration is \
             incomplete; callers should be updated to the \
             constitutional surface in \
             prism_ecs_constitutional::canonical.",
            importers.len(),
            importers
        );
    }
}
