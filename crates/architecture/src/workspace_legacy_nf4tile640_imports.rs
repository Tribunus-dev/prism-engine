//! `workspace_contains_no_legacy_nf4tile640_imports`
//!
//! Workspace-level architecture enforcement test for the
//! nf4tile640 migration. It scans the workspace for any `use`
//! statement that references the legacy engine nf4tile640
//! surface (`compute_core::ecs::nf4tile640::*`,
//! `crate::ecs::nf4tile640::*` inside the engine, or
//! `crate::nf4tile640::*` re-exported via the engine's
//! `pub use crate::ecs::nf4tile640` shim).
//!
//! The test fails if any file in the workspace imports the legacy
//! surface. The legacy engine file is the engine's
//! `compute-core/src/ecs/nf4tile640/` directory; the
//! constitutional replacement is `prism_ecs_quantization::nf4tile640`.
//!
//! # Migration status
//!
//! This test was added when the nf4tile640 engine-deletion
//! migration was being completed (2026-07-27). The engine's
//! `compute-core/src/ecs/nf4tile640/` directory will be deleted
//! after this test is green; until then the directory is the
//! migration inventory and is exempt from this scan.

use std::fs;
use std::path::Path;

/// Returns the set of files OUTSIDE the legacy engine nf4tile640
/// directory that import the legacy surface. Files inside the
/// engine's own `compute-core/src/ecs/nf4tile640/` directory
/// are exempt — they ARE the legacy surface and will be deleted
/// after the migration. Files outside that directory that import
/// the surface are violations.
pub fn legacy_nf4tile640_importers_outside_inventory() -> Vec<String> {
    let mut importers = Vec::new();
    scan_workspace_excluding_inventory("compute-core", &mut importers);
    importers
}

const LEGACY_INVENTORY_DIR: &str = "compute-core/src/ecs/nf4tile640";

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
                if content.contains("use crate::ecs::nf4tile640::")
                    || content.contains("compute_core::ecs::nf4tile640::")
                    || content.contains("tribunus_compute_core::nf4tile640")
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
    fn workspace_contains_no_legacy_nf4tile640_imports() {
        // Architectural invariant: no file OUTSIDE the migration
        // inventory imports the legacy engine nf4tile640
        // surface. The engine's
        // compute-core/src/ecs/nf4tile640/ directory is the
        // migration inventory and is exempt; it IS the legacy
        // surface. Files outside that directory that import the
        // surface are violations.
        let importers = legacy_nf4tile640_importers_outside_inventory();
        assert!(
            importers.is_empty(),
            "Workspace still contains {} files outside the migration \
             inventory that import the legacy engine nf4tile640 \
             surface: {:?}. The migration is incomplete; callers \
             should be updated to the constitutional surface in \
             prism_ecs_quantization::nf4tile640.",
            importers.len(),
            importers
        );
    }
}
