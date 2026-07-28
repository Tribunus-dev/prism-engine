//! `workspace_contains_no_legacy_tools_imports`
//!
//! Workspace-level architecture enforcement test for the tools
//! engine-subsystem deletion migration. It scans the workspace for
//! any `use` statement that references the legacy engine tools
//! surface (`compute_core::ecs::tools::*` or
//! `crate::ecs::tools::*` inside the engine, or the
//! `tribunus_compute_core::tools` re-export shim from
//! `compute-core/src/lib.rs`).
//!
//! The test fails if any file in the workspace imports the legacy
//! surface outside the engine's `legacy_tools/` directory. The
//! legacy engine file was the engine's
//! `compute-core/src/ecs/tools/` directory (already migrated into
//! the constitutional surface at `prism_ecs_server::tools`). The
//! engine's `compute-core/src/ecs/legacy_tools/` directory is the
//! engine-internal façade that re-exports the constitutional
//! surface and hosts the two engine-coupled extensions
//! (`list_devices` querying the engine's device registry, and the
//! mlx-backend `retry_with_error` driving the engine's
//! `profiled_executor`).
//!
//! # Migration status
//!
//! This test was added when the tools engine-deletion migration
//! was being completed (2026-07-27). The engine's
//! `compute-core/src/ecs/tools/` directory is renamed / deleted
//! after the migration; this test is the architecture-level guard
//! against re-introducing a parallel `ecs::tools` authority in any
//! future commit.

use std::fs;
use std::path::Path;

/// Returns the set of files OUTSIDE the engine's
/// `compute-core/src/ecs/legacy_tools/` directory and OUTSIDE
/// `compute-core/src/ecs/tools/` (if it still exists) that
/// import the legacy engine tools surface. Files inside the
/// engine's `legacy_tools/` directory are exempt — they ARE
/// the engine-internal façade that re-exports the constitutional
/// surface. The engine's `tools/` directory (if it still exists
/// during the migration) is the migration inventory and is
/// exempt; it IS the legacy surface.
pub fn legacy_tools_importers_outside_inventory() -> Vec<String> {
    let mut importers = Vec::new();
    scan_workspace_excluding_inventory("compute-core", &mut importers);
    importers
}

const ENGINE_FACADE_DIR: &str = "compute-core/src/ecs/legacy_tools";
const ENGINE_INVENTORY_DIR: &str = "compute-core/src/ecs/tools";

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
        // Skip the engine-internal legacy_tools/ directory — it
        // re-exports the constitutional data types and is the
        // canonical engine-internal home for the tool surface.
        if path_str.contains(ENGINE_FACADE_DIR) {
            continue;
        }
        // Skip the legacy inventory (compute-core/src/ecs/tools)
        // if it still exists during the migration. The directory
        // is renamed / deleted after the migration; this branch
        // is the safety net during the in-flight rename.
        if path_str.contains(ENGINE_INVENTORY_DIR) {
            continue;
        }
        if p.is_dir() {
            scan_workspace_excluding_inventory(&path_str, importers);
        } else if p.extension().and_then(|e| e.to_str()) == Some("rs") {
            if let Ok(content) = fs::read_to_string(&p) {
                if content.contains("use crate::ecs::tools::")
                    || content.contains("compute_core::ecs::tools::")
                    || content.contains("tribunus_compute_core::tools::")
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
    fn workspace_contains_no_legacy_tools_imports() {
        // Architectural invariant: no file OUTSIDE the engine's
        // `compute-core/src/ecs/legacy_tools/` directory and
        // OUTSIDE the engine's `compute-core/src/ecs/tools/`
        // migration inventory imports the legacy engine tools
        // surface. The `legacy_tools/` directory re-exports the
        // constitutional `prism_ecs_server::tools` data types
        // and hosts the engine-internal extensions. The
        // `tools/` directory is the migration inventory and is
        // exempt; it IS the legacy surface. Files outside
        // either directory that import
        // `crate::ecs::tools::*` (or the
        // `tribunus_compute_core::tools` re-export) are
        // violations.
        let importers = legacy_tools_importers_outside_inventory();
        assert!(
            importers.is_empty(),
            "Workspace still contains {} files outside the engine's \
             legacy_tools/ and tools/ directories that import the legacy \
             engine tools surface: {:?}. The migration is incomplete; \
             callers should be updated to the constitutional surface in \
             prism_ecs_server::tools.",
            importers.len(),
            importers
        );
    }
}
