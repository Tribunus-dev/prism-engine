//! `workspace_contains_no_legacy_config_imports`
//!
//! Workspace-level architecture enforcement test for the
//! `compute-core/src/ecs/legacy_config/` →
//! `prism_ecs_constitutional::config` migration. It scans the
//! workspace for any `use` statement that references the legacy
//! engine config surface (`compute_core::ecs::config::*`,
//! `crate::ecs::config::*` inside the engine, or
//! `tribunus_compute_core::config::*`).
//!
//! The test fails if any file in the workspace imports the legacy
//! surface. The legacy engine file is the engine's
//! `compute-core/src/ecs/legacy_config/` directory (the migration
//! target was the constitutional `prism_ecs_constitutional::config`).
//! Files inside the engine's own `legacy_config/` directory are
//! exempt — they ARE the legacy surface, re-exporting the
//! constitutional types.
//!
//! # Migration status
//!
//! This test was added when the config engine-deletion migration
//! was being completed (2026-07-28, batch 6 of the constitutional
//! engine absorption). The engine's
//! `compute-core/src/ecs/legacy_config/` directory is the engine-
//! internal re-export shim; engine binaries and tests that prefer
//! the legacy path continue to resolve.

use std::fs;
use std::path::Path;

/// Returns the set of files OUTSIDE the engine's
/// `compute-core/src/ecs/legacy_config/` directory that import the
/// legacy engine config surface. Files inside the engine's own
/// `legacy_config/` directory are exempt — they ARE the engine-
/// internal re-export shim.
pub fn legacy_config_importers_outside_inventory() -> Vec<String> {
    let mut importers = Vec::new();
    scan_workspace_excluding_inventory("compute-core", &mut importers);
    importers
}

const ENGINE_INVENTORY_DIR: &str = "compute-core/src/ecs/legacy_config";

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
        // Skip the engine-internal legacy_config directory — it
        // re-exports the constitutional data types and is the
        // canonical engine-internal home for the higher-level config
        // adapter code.
        if path_str.contains(ENGINE_INVENTORY_DIR) {
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
        // Architectural invariant: no file OUTSIDE the engine's
        // `legacy_config/` shim imports the legacy engine config
        // surface. The shim re-exports the constitutional data
        // types and is the canonical engine-internal home for the
        // higher-level config adapter code. Files outside the shim
        // that import the legacy surface are violations.
        let importers = legacy_config_importers_outside_inventory();
        assert!(
            importers.is_empty(),
            "Workspace still contains {} files outside the engine's \
             `legacy_config/` shim that import the legacy engine \
             config surface: {:?}. The migration is incomplete; \
             callers should be updated to the constitutional surface \
             in prism_ecs_constitutional::config.",
            importers.len(),
            importers
        );
    }
}
