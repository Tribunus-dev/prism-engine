//! `workspace_contains_no_legacy_kv_arena_imports`
//!
//! Workspace-level architecture enforcement test for the kv_arena
//! migration. It scans the workspace for any `use` statement that
//! references the legacy engine kv_arena surface
//! (`compute_core::ecs::kv_arena::*` or `crate::ecs::kv_arena::*`
//! inside the engine).
//!
//! The test fails if any file in the workspace imports the legacy
//! surface. The legacy engine file is the engine's
//! `compute-core/src/ecs/kv_arena/` directory; the constitutional
//! replacement is `prism_kv_cache::arena`.
//!
//! # Migration status
//!
//! This test was added when the kv_arena engine-deletion migration
//! was being completed (2026-07-27). The engine's
//! `compute-core/src/ecs/kv_arena/` directory will be deleted after
//! this test is green; until then the directory is the migration
//! inventory and is exempt from this scan.

use std::fs;
use std::path::Path;

/// Returns the set of files OUTSIDE the legacy engine kv_arena
/// directory that import the legacy surface. Files inside the
/// engine's own `compute-core/src/ecs/kv_arena/` directory are
/// exempt — they ARE the legacy surface and will be deleted after
/// the migration. Files outside that directory that import the
/// surface are violations.
pub fn legacy_kv_arena_importers_outside_inventory() -> Vec<String> {
    let mut importers = Vec::new();
    scan_workspace_excluding_inventory("compute-core", &mut importers);
    importers
}

const LEGACY_INVENTORY_DIR: &str = "compute-core/src/ecs/kv_arena";

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
        if p.is_dir() {
            scan_workspace_excluding_inventory(&path_str, importers);
        } else if p.extension().and_then(|e| e.to_str()) == Some("rs") {
            if let Ok(content) = fs::read_to_string(&p) {
                if content.contains("use crate::ecs::kv_arena::")
                    || content.contains("compute_core::ecs::kv_arena::")
                    || content.contains("tribunus_compute_core::kv_arena")
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
    fn workspace_contains_no_legacy_kv_arena_imports() {
        // Architectural invariant: no file OUTSIDE the migration
        // inventory imports the legacy engine kv_arena surface.
        // The engine's compute-core/src/ecs/kv_arena/ directory is
        // the migration inventory and is exempt; it IS the legacy
        // surface. Files outside that directory that import the
        // surface are violations.
        let importers = legacy_kv_arena_importers_outside_inventory();
        assert!(
            importers.is_empty(),
            "Workspace still contains {} files outside the migration \
             inventory that import the legacy engine kv_arena surface: \
             {:?}. The migration is incomplete; callers should be \
             updated to the constitutional surface in \
             prism_kv_cache::arena.",
            importers.len(),
            importers
        );
    }
}
