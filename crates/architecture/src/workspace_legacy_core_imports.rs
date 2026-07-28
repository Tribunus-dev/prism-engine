//! `workspace_contains_no_legacy_core_imports`
//!
//! Workspace-level architecture enforcement test for the
//! core migration. It scans the workspace for any `use`
//! statement that references the legacy engine core
//! surface (`compute_core::ecs::core::*` or
//! `crate::ecs::core::*` inside the engine).
//!
//! The test fails if any file in the workspace imports the
//! legacy surface. The legacy engine file was the engine's
//! `compute-core/src/ecs/core/` directory (121 files, 53,532
//! LOC); it was deleted in the core engine-subsystem deletion
//! migration (2026-07-27). The 121 files now live under
//! `compute-core/src/ecs/legacy_core/` as engine-internal
//! types awaiting their constitutional re-homing.
//!
//! The constitutional replacement is the per-file authority
//! table recorded in
//! `crates/prism-ecs-core/src/core/mod.rs`. The placeholder
//! `prism_ecs_core::core` module is doc-only; the per-file
//! homes are `prism_ecs_runtime`, `prism_ecs_compile`,
//! `prism_ecs_quantization`, `prism_gguf`, `prism_ane`,
//! `prism_kv_cache`, `prism_ecs_agent`, `prism_ecs_server`,
//! `prism-audio`, and `prism_ecs_codec`.
//!
//! # Migration status
//!
//! This test was added when the core engine-deletion
//! migration was being completed (2026-07-27). The engine's
//! `compute-core/src/ecs/core/` directory is already deleted;
//! the migration inventory is
//! `compute-core/src/ecs/legacy_core/` (the new path the
//! 121 files were moved to).

use std::fs;
use std::path::Path;

/// Returns the set of files OUTSIDE the legacy engine core
/// directory that import the legacy surface. Files inside
/// the engine's own `compute-core/src/ecs/legacy_core/`
/// directory are exempt — they ARE the legacy surface and
/// were moved from `compute-core/src/ecs/core/` in the
/// engine-subsystem deletion migration. Files outside that
/// directory that import the legacy `core::` path are
/// violations.
pub fn legacy_core_importers_outside_inventory() -> Vec<String> {
    let mut importers = Vec::new();
    scan_workspace_excluding_inventory("compute-core", &mut importers);
    scan_workspace_excluding_inventory("crates", &mut importers);
    scan_workspace_excluding_inventory("tests", &mut importers);
    scan_workspace_excluding_inventory("prism-bridge", &mut importers);
    importers
}

const LEGACY_INVENTORY_DIR: &str = "compute-core/src/ecs/legacy_core";

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
                if content.contains("use crate::ecs::core::")
                    || content.contains("compute_core::ecs::core::")
                    || content.contains("tribunus_compute_core::core::")
                    || content.contains("tribunus_compute_core::core;")
                    || content.contains("tribunus_compute_core::core ")
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
    fn workspace_contains_no_legacy_core_imports() {
        // Architectural invariant: no file OUTSIDE the migration
        // inventory imports the legacy engine core surface. The
        // engine's compute-core/src/ecs/legacy_core/ directory is
        // the migration inventory and is exempt; it IS the
        // legacy surface (renamed from compute-core/src/ecs/core/
        // in the engine-subsystem deletion migration). Files
        // outside that directory that import the surface are
        // violations.
        let importers = legacy_core_importers_outside_inventory();
        assert!(
            importers.is_empty(),
            "Workspace still contains {} files outside the migration \
             inventory that import the legacy engine core surface: \
             {:?}. The migration is incomplete; callers should be \
             updated to the constitutional surface (per the per-file \
             authority table in crates/prism-ecs-core/src/core/mod.rs) \
             or to the engine-internal legacy_core path.",
            importers.len(),
            importers
        );
    }
}
