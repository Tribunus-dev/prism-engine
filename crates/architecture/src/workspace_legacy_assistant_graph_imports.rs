//! `workspace_contains_no_legacy_assistant_graph_imports`
//!
//! Workspace-level architecture enforcement test for the
//! assistant_graph migration. It scans the workspace for any `use`
//! statement that references the legacy engine assistant_graph
//! surface (`compute_core::ecs::assistant_graph::*` or
//! `crate::ecs::assistant_graph::*` inside the engine).
//!
//! The test fails if any file in the workspace imports the legacy
//! surface. The legacy engine file is the engine's
//! `compute-core/src/ecs/assistant_graph/` directory; the
//! constitutional replacement is `prism_ecs_agent::assistant_graph`.
//!
//! # Migration status
//!
//! This test was added when the assistant_graph engine-deletion
//! migration was being completed (2026-07-27). The engine's
//! `compute-core/src/ecs/assistant_graph/` directory will be deleted
//! after this test is green; until then the directory is the
//! migration inventory and is exempt from this scan.

use std::fs;
use std::path::Path;

/// Returns the set of files OUTSIDE the legacy engine assistant_graph
/// directory that import the legacy surface. Files inside the
/// engine's own `compute-core/src/ecs/assistant_graph/` directory
/// are exempt — they ARE the legacy surface and will be deleted
/// after the migration. Files outside that directory that import
/// the surface are violations.
pub fn legacy_assistant_graph_importers_outside_inventory() -> Vec<String> {
    let mut importers = Vec::new();
    scan_workspace_excluding_inventory("compute-core", &mut importers);
    importers
}

const LEGACY_INVENTORY_DIR: &str = "compute-core/src/ecs/assistant_graph";

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
                if content.contains("use crate::ecs::assistant_graph::")
                    || content.contains("compute_core::ecs::assistant_graph::")
                    || content.contains("tribunus_compute_core::assistant_graph")
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
    fn workspace_contains_no_legacy_assistant_graph_imports() {
        // Architectural invariant: no file OUTSIDE the migration
        // inventory imports the legacy engine assistant_graph
        // surface. The engine's
        // compute-core/src/ecs/assistant_graph/ directory is the
        // migration inventory and is exempt; it IS the legacy
        // surface. Files outside that directory that import the
        // surface are violations.
        let importers = legacy_assistant_graph_importers_outside_inventory();
        assert!(
            importers.is_empty(),
            "Workspace still contains {} files outside the migration \
             inventory that import the legacy engine assistant_graph \
             surface: {:?}. The migration is incomplete; callers \
             should be updated to the constitutional surface in \
             prism_ecs_agent::assistant_graph.",
            importers.len(),
            importers
        );
    }
}
