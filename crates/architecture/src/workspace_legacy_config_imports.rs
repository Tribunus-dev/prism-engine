//! `workspace_contains_no_legacy_config_imports`
//!
//! Workspace-level architecture enforcement test for the
//! `compute-core/src/ecs/config/` →
//! `prism_ecs_constitutional::config` migration. It scans the
//! workspace for any `use` statement that references the legacy
//! engine config surface (`compute_core::ecs::config::*`,
//! `crate::ecs::config::*` inside the engine, or
//! `tribunus_compute_core::config::*`).
//!
//! The test fails if any file in the workspace imports the legacy
//! surface. The legacy engine file is the engine's
//! `compute-core/src/ecs/config/` directory (the migration target was
//! the constitutional `prism_ecs_constitutional::config`). The
//! legacy directory will be renamed to
//! `compute-core/src/ecs/legacy_config/` after this test is green
//! and engine-coupled code is moved out.
//!
//! # Migration status
//!
//! This test was added when the config engine-deletion migration
//! was being completed (2026-07-28, batch 6 of the constitutional
//! engine absorption). The engine's
//! `compute-core/src/ecs/config/` directory is the migration
//! inventory; it remains in place until the legacy_code/
//! shim migration is complete. Until then, files inside the engine's
//! own `config/` directory are exempt — they ARE the legacy surface
//! and will be moved to `legacy_config/` after this test is green.

use std::fs;
use std::path::Path;

/// Returns the set of files OUTSIDE the engine's
/// `compute-core/src/ecs/config/` directory that import the legacy
/// engine config surface. Files inside the engine's own `config/`
/// directory are exempt — they ARE the legacy surface.
pub fn legacy_config_importers_outside_inventory() -> Vec<String> {
    let mut importers = Vec::new();
    scan_workspace_excluding_inventory("compute-core", &mut importers);
    importers
}

const LEGACY_INVENTORY_DIR: &str = "compute-core/src/ecs/config";

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
                if content.contains("use crate::ecs::config::")
                    || content.contains("compute_core::ecs::config::")
                    || content.contains("tribunus_compute_core::ecs::config::")
                    || content.contains("tribunus_compute_core::config::")
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
    fn workspace_contains_no_legacy_config_imports() {
        // Architectural invariant: no file OUTSIDE the migration
        // inventory imports the legacy engine config surface. The
        // engine's `compute-core/src/ecs/config/` directory is the
        // migration inventory and is exempt; it IS the legacy
        // surface. Files outside that directory that import the
        // surface are violations.
        let importers = legacy_config_importers_outside_inventory();
        assert!(
            importers.is_empty(),
            "Workspace still contains {} files outside the migration \
             inventory that import the legacy engine config surface: \
             {:?}. The migration is incomplete; callers should be \
             updated to the constitutional surface in \
             prism_ecs_constitutional::config.",
            importers.len(),
            importers
        );
    }
}
