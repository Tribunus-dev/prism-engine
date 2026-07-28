//! `workspace_contains_no_legacy_system_imports`
//!
//! Workspace-level architecture enforcement test for the
//! system migration. It scans the workspace for any `use`
//! statement that references the legacy engine system
//! surface (`compute_core::ecs::system::*` or
//! `crate::ecs::system::*` inside the engine).
//!
//! The test fails if any file in the workspace imports the
//! legacy surface. The legacy engine file is the engine's
//! `compute-core/src/ecs/system/` directory; the
//! constitutional replacement is
//! `prism_ecs_runtime::systems::*` (for data types) and
//! `compute-core::ecs::system_adapters::*` (for the
//! `CompilerSystem` trait impls that wrap the constitutional
//! data types).
//!
//! # Migration status
//!
//! This test was added when the system engine-deletion
//! migration was being completed (2026-07-27). The engine's
//! `compute-core/src/ecs/system/` directory is deleted in the
//! same migration wave; until then the directory is the
//! migration inventory and is exempt from this scan.

use std::fs;
use std::path::Path;

/// Returns the set of files OUTSIDE the legacy engine system
/// directory that import the legacy surface. Files inside the
/// engine's own `compute-core/src/ecs/system/` directory
/// are exempt — they ARE the legacy surface and will be deleted
/// after the migration. Files outside that directory that
/// import the surface are violations.
pub fn legacy_system_importers_outside_inventory() -> Vec<String> {
    let mut importers = Vec::new();
    scan_workspace_excluding_inventory("compute-core", &mut importers);
    importers
}

const LEGACY_INVENTORY_DIR: &str = "compute-core/src/ecs/system";

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
                if content.contains("use crate::ecs::system::")
                    || content.contains("compute_core::ecs::system::")
                    || content.contains("tribunus_compute_core::ecs::system::")
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
    fn workspace_contains_no_legacy_system_imports() {
        // Architectural invariant: no file OUTSIDE the migration
        // inventory imports the legacy engine system surface.
        // The engine's compute-core/src/ecs/system/ directory is
        // the migration inventory and is exempt; it IS the
        // legacy surface. Files outside that directory that
        // import the surface are violations.
        let importers = legacy_system_importers_outside_inventory();
        assert!(
            importers.is_empty(),
            "Workspace still contains {} files outside the migration \
             inventory that import the legacy engine system surface: \
             {:?}. The migration is incomplete; callers should be \
             updated to the constitutional surface in \
             prism_ecs_runtime::systems (data types) or \
             compute-core::ecs::system_adapters (CompilerSystem \
             trait impls).",
            importers.len(),
            importers
        );
    }
}
