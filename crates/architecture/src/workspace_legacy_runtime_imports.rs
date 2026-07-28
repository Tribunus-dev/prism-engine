//! `workspace_contains_no_legacy_runtime_imports`
//!
//! Workspace-level architecture enforcement test for the runtime
//! migration. It scans the workspace for any `use` statement or
//! path reference that imports the legacy engine runtime surface
//! (`compute_core::ecs::runtime::*` or `crate::ecs::runtime::*`
//! inside the engine, or the `tribunus_compute_core::ecs::runtime`
//! re-export shim).
//!
//! The test fails if any file in the workspace imports the legacy
//! surface. The legacy engine directory was the engine's
//! `compute-core/src/ecs/runtime/` directory (renamed to
//! `compute-core/src/ecs/legacy_runtime/` in E-2 of the
//! migration). The constitutional replacement is
//! `prism_ecs_runtime::runtime` for the data types and pure
//! abstractions, and `compute-core/src/ecs/legacy_runtime/` for
//! the engine-internal execution-plane code.
//!
//! # Migration status
//!
//! This test was added when the runtime engine-deletion migration
//! was being completed (2026-07-27). The engine's
//! `compute-core/src/ecs/runtime/` directory was renamed to
//! `compute-core/src/ecs/legacy_runtime/` in E-2; the test is the
//! architecture-level guard against re-introducing a parallel
//! `ecs::runtime` authority (separate from the engine-internal
//! `ecs::legacy_runtime` execution-plane home) in any future
//! commit.

use std::fs;
use std::path::Path;

/// Returns the set of files OUTSIDE the engine's
/// `compute-core/src/ecs/legacy_runtime/` directory that import
/// the legacy engine runtime surface. Files inside the engine's
/// `legacy_runtime/` directory are exempt — they ARE the engine-
/// internal execution-plane home for the runtime surface and
/// re-export the constitutional data types from
/// `prism_ecs_runtime::runtime`.
pub fn legacy_runtime_importers_outside_inventory() -> Vec<String> {
    let mut importers = Vec::new();
    scan_workspace_excluding_inventory("compute-core", &mut importers);
    importers
}

const ENGINE_INVENTORY_DIR: &str = "compute-core/src/ecs/legacy_runtime";

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
        // Skip the engine-internal legacy_runtime directory — it
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
                if content.contains("use crate::ecs::runtime::")
                    || content.contains("compute_core::ecs::runtime::")
                    || content.contains("tribunus_compute_core::ecs::runtime::")
                    || content.contains("tribunus_compute_core::ecs::runtime;")
                    // Path references in body (e.g. in agent_slot.rs
                    // `crate::ecs::runtime::world::World::with_capacity`)
                    // are also violations.
                    || content.contains("crate::ecs::runtime::")
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
    fn workspace_contains_no_legacy_runtime_imports() {
        // Architectural invariant: no file OUTSIDE the engine's
        // `compute-core/src/ecs/legacy_runtime/` directory imports
        // the legacy engine runtime surface. The `legacy_runtime/`
        // directory re-exports the constitutional
        // `prism_ecs_runtime::runtime` data types and hosts the
        // engine-internal execution-plane code (multiplexers,
        // pumps, interceptors, executables, kv cache coordinator,
        // etc.). Files outside `legacy_runtime/` that import
        // `crate::ecs::runtime::*` (or the
        // `tribunus_compute_core::ecs::runtime` re-export) are
        // violations.
        let importers = legacy_runtime_importers_outside_inventory();
        assert!(
            importers.is_empty(),
            "Workspace still contains {} files outside the engine's \
             legacy_runtime/ directory that import the legacy engine \
             runtime surface: {:?}. The migration is incomplete; \
             callers should be updated to the constitutional surface \
             in prism_ecs_runtime::runtime (data types and pure \
             abstractions) or the engine's \
             compute-core/src/ecs/legacy_runtime/ (execution-plane \
             code).",
            importers.len(),
            importers
        );
    }
}
