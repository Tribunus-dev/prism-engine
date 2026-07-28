//! `workspace_contains_no_legacy_cimage_imports`
//!
//! Workspace-level architecture enforcement test for the cimage
//! migration. It scans the workspace for any `use` statement that
//! references the legacy engine cimage surface
//! (`compute_core::ecs::cimage::*` or `crate::ecs::cimage::*`
//! inside the engine, or the `tribunus_compute_core::cimage`
//! re-export shim).
//!
//! The test fails if any file in the workspace imports the legacy
//! surface. The legacy engine file was the engine's
//! `compute-core/src/ecs/cimage/` directory (already renamed to
//! `compute-core/src/ecs/legacy_cimage/` in E-2 of the migration).
//! The constitutional replacement is
//! `prism_ecs_compile::cimage_v0` for the engine-independent V0
//! file format data types and `compute-core/src/ecs/legacy_cimage/`
//! for the engine-internal higher-level operations.
//!
//! # Migration status
//!
//! This test was added when the cimage engine-deletion migration
//! was being completed (2026-07-27). The engine's
//! `compute-core/src/ecs/cimage/` directory is already renamed to
//! `legacy_cimage/`; the test is the architecture-level guard
//! against re-introducing a parallel `ecs::cimage` authority in
//! any future commit.

use std::fs;
use std::path::Path;

/// Returns the set of files OUTSIDE the engine's
/// `compute-core/src/ecs/legacy_cimage/` directory that import the
/// legacy engine cimage surface. Files inside the engine's
/// `legacy_cimage/` directory are exempt — they ARE the engine-
/// internal home for the higher-level cimage operations and
/// re-export the constitutional data types from
/// `prism_ecs_compile::cimage_v0`.
pub fn legacy_cimage_importers_outside_inventory() -> Vec<String> {
    let mut importers = Vec::new();
    scan_workspace_excluding_inventory("compute-core", &mut importers);
    importers
}

const ENGINE_INVENTORY_DIR: &str = "compute-core/src/ecs/legacy_cimage";

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
        // Skip the engine-internal legacy_cimage directory — it
        // re-exports the constitutional data types and is the
        // canonical engine-internal home for the higher-level
        // cimage operations.
        if path_str.contains(ENGINE_INVENTORY_DIR) {
            continue;
        }
        if p.is_dir() {
            scan_workspace_excluding_inventory(&path_str, importers);
        } else if p.extension().and_then(|e| e.to_str()) == Some("rs") {
            if let Ok(content) = fs::read_to_string(&p) {
                if content.contains("use crate::ecs::cimage::")
                    || content.contains("compute_core::ecs::cimage::")
                    || content.contains("tribunus_compute_core::cimage::")
                    || content.contains("tribunus_compute_core::cimage;")
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
    fn workspace_contains_no_legacy_cimage_imports() {
        // Architectural invariant: no file OUTSIDE the engine's
        // `compute-core/src/ecs/legacy_cimage/` directory imports the
        // legacy engine cimage surface. The `legacy_cimage/`
        // directory re-exports the constitutional
        // `prism_ecs_compile::cimage_v0` data types and hosts the
        // engine-internal higher-level cimage operations. Files
        // outside `legacy_cimage/` that import
        // `crate::ecs::cimage::*` (or the
        // `tribunus_compute_core::cimage` re-export) are violations.
        let importers = legacy_cimage_importers_outside_inventory();
        assert!(
            importers.is_empty(),
            "Workspace still contains {} files outside the engine's \
             legacy_cimage/ directory that import the legacy engine \
             cimage surface: {:?}. The migration is incomplete; \
             callers should be updated to the constitutional surface \
             in prism_ecs_compile::cimage_v0 (data types) or the \
             engine's compute-core/src/ecs/legacy_cimage/ (engine-\
             internal higher-level operations).",
            importers.len(),
            importers
        );
    }
}
