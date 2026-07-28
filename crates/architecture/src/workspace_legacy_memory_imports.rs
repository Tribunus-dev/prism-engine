//! `workspace_contains_no_legacy_memory_imports`
//!
//! Workspace-level architecture enforcement test for the memory
//! migration. It scans the workspace for any `use` statement that
//! references the legacy engine memory surface
//! (`compute_core::ecs::memory::*` or `crate::ecs::memory::*`
//! inside the engine, or the `tribunus_compute_core::memory`
//! re-export shim).
//!
//! The test fails if any file in the workspace imports the legacy
//! surface. The legacy engine file was the engine's
//! `compute-core/src/ecs/memory/` directory (already deleted in
//! E-2 of the migration). The constitutional replacement is
//! `prism_ecs_data::memory` for the data types and
//! `compute-core/src/ecs/memory_impl/` for the execution-plane
//! code.
//!
//! # Migration status
//!
//! This test was added when the memory engine-deletion migration
//! was being completed (2026-07-27). The engine's
//! `compute-core/src/ecs/memory/` directory is already deleted;
//! the test is the architecture-level guard against re-introducing
//! a parallel `ecs::memory` authority in any future commit.

use std::fs;
use std::path::Path;

/// Returns the set of files OUTSIDE the engine's
/// `compute-core/src/ecs/memory_impl/` directory that import the
/// legacy engine memory surface. Files inside the engine's
/// `memory_impl/` directory are exempt — they ARE the engine-
/// internal execution-plane home for the memory surface and
/// re-export the constitutional data types from
/// `prism_ecs_data::memory`.
pub fn legacy_memory_importers_outside_inventory() -> Vec<String> {
    let mut importers = Vec::new();
    scan_workspace_excluding_inventory("compute-core", &mut importers);
    importers
}

const ENGINE_INVENTORY_DIR: &str = "compute-core/src/ecs/memory_impl";

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
        // Skip the engine-internal memory_impl directory — it
        // re-exports the constitutional data types and is the
        // canonical engine-internal home for the execution-plane
        // code.
        if path_str.contains(ENGINE_INVENTORY_DIR) {
            continue;
        }
        if p.is_dir() {
            scan_workspace_excluding_inventory(&path_str, importers);
        } else if p.extension().and_then(|e| e.to_str()) == Some("rs") {
            if let Ok(content) = fs::read_to_string(&p) {
                if content.contains("use crate::ecs::memory::")
                    || content.contains("compute_core::ecs::memory::")
                    || content.contains("tribunus_compute_core::memory::")
                    || content.contains("tribunus_compute_core::memory;")
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
    fn workspace_contains_no_legacy_memory_imports() {
        // Architectural invariant: no file OUTSIDE the engine's
        // `compute-core/src/ecs/memory_impl/` directory imports the
        // legacy engine memory surface. The `memory_impl/`
        // directory re-exports the constitutional
        // `prism_ecs_data::memory` data types and hosts the
        // engine-internal execution-plane code. Files outside
        // `memory_impl/` that import `crate::ecs::memory::*` (or
        // the `tribunus_compute_core::memory` re-export) are
        // violations.
        let importers = legacy_memory_importers_outside_inventory();
        assert!(
            importers.is_empty(),
            "Workspace still contains {} files outside the engine's \
             memory_impl/ directory that import the legacy engine \
             memory surface: {:?}. The migration is incomplete; \
             callers should be updated to the constitutional surface \
             in prism_ecs_data::memory (data types) or the engine's \
             compute-core/src/ecs/memory_impl/ (execution-plane \
             code).",
            importers.len(),
            importers
        );
    }
}
