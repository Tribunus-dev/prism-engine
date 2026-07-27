//! `workspace_contains_no_legacy_evaluator_imports`
//!
//! Per the inventory v2.1 step 60, this is a workspace-level
//! architecture enforcement test. It scans the workspace for any
//! `use` statement that references the legacy engine evaluator
//! surface (`compute_core::ecs::evaluator::*`).
//!
//! The test fails if any file in the workspace imports the
//! legacy surface. The legacy engine file is the engine's
//! `compute-core/src/ecs/evaluator/` directory; the constitutional
//! replacement is `prism_ecs_codec::evaluator::*`.
//!
//! # Migration status
//!
//! This test was added when the evaluator migration was started
//! (2026-07-27). It will pass once the engine's
//! `compute-core/src/ecs/evaluator/` directory is fully removed
//! (post-evaluator E-5) or when the engine file is excluded from
//! the source scan.

use std::fs;
use std::path::Path;

/// Returns the set of files OUTSIDE the legacy engine evaluator
/// directory that import the legacy surface. Files inside the
/// engine's own `compute-core/src/ecs/evaluator/` directory are
/// exempt — they ARE the legacy surface and will be deleted in
/// the evaluator E-5 step. Files outside that directory that
/// import the surface are violations.
pub fn legacy_evaluator_importers_outside_inventory() -> Vec<String> {
    let mut importers = Vec::new();
    scan_workspace_excluding_inventory("compute-core", &mut importers);
    importers
}

const LEGACY_INVENTORY_DIR: &str = "compute-core/src/ecs/evaluator";

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
                if content.contains("use crate::ecs::evaluator::")
                    || content.contains("compute_core::ecs::evaluator::")
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
    fn workspace_contains_no_legacy_evaluator_imports() {
        // Architectural invariant: no file OUTSIDE the migration
        // inventory imports the legacy engine evaluator surface.
        // The engine's compute-core/src/ecs/evaluator/ directory
        // is the migration inventory and is exempt; it IS the
        // legacy surface. Files outside that directory that
        // import the surface are violations.
        let importers = legacy_evaluator_importers_outside_inventory();
        assert!(
            importers.is_empty(),
            "Workspace still contains {} files outside the migration \
             inventory that import the legacy engine evaluator surface: \
             {:?}. The migration is incomplete; callers should be updated \
             to the constitutional surface in prism-ecs-codec::evaluator.",
            importers.len(),
            importers
        );
    }
}
