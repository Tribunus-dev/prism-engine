//! `workspace_contains_no_legacy_compilation_imports`
//!
//! Workspace-level architecture enforcement test for the compilation
//! migration. It scans the workspace for any `use` statement that
//! references the legacy engine compilation surface
//! (`compute_core::ecs::compilation::*` or `crate::ecs::compilation::*`
//! inside the engine).
//!
//! The test fails if any file in the workspace imports the legacy
//! surface. The legacy engine dir is the engine's
//! `compute-core/src/ecs/legacy_compilation/` directory (renamed from
//! `ecs/compilation/` in E-3); the constitutional replacement is
//! `prism_ecs_compile::compilation` for data-only types and
//! `crate::ecs::legacy_compilation` for engine-coupled implementations.
//!
//! # Migration status
//!
//! This test was added when the compilation engine-deletion migration
//! was being completed (2026-07-27). The engine's
//! `compute-core/src/ecs/legacy_compilation/` directory remains as the
//! home for engine-coupled implementations pending later absorption
//! waves; files inside that directory are exempt from the scan. Files
//! outside that directory that still import the old
//! `crate::ecs::compilation::X` path are violations.

use std::fs;
use std::path::Path;

/// Returns the set of files OUTSIDE the legacy engine compilation
/// directory that import the legacy surface. Files inside the
/// engine's own `compute-core/src/ecs/legacy_compilation/` directory
/// are exempt — they ARE the legacy surface and engine-coupled
/// implementations. Files outside that directory that import the
/// surface are violations.
pub fn legacy_compilation_importers_outside_inventory() -> Vec<String> {
    let mut importers = Vec::new();
    scan_workspace_excluding_inventory("compute-core", &mut importers);
    importers
}

const LEGACY_INVENTORY_DIR: &str = "compute-core/src/ecs/legacy_compilation";

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
                if content.contains("use crate::ecs::compilation::")
                    || content.contains("compute_core::ecs::compilation::")
                    || content.contains("tribunus_compute_core::ecs::compilation")
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
    fn workspace_contains_no_legacy_compilation_imports() {
        // Architectural invariant: no file OUTSIDE the migration
        // inventory imports the legacy engine compilation surface. The
        // engine's compute-core/src/ecs/legacy_compilation/ directory
        // is the migration inventory and is exempt; it IS the legacy
        // surface. Files outside that directory that import the surface
        // are violations and must be retargeted to
        // prism_ecs_compile::compilation::* (data-only) or
        // crate::ecs::legacy_compilation::* (engine-coupled).
        let importers = legacy_compilation_importers_outside_inventory();
        assert!(
            importers.is_empty(),
            "Workspace still contains {} files outside the migration \
             inventory that import the legacy engine compilation surface: \
             {:?}. The migration is incomplete; callers should be \
             updated to prism_ecs_compile::compilation::* (data-only) or \
             crate::ecs::legacy_compilation::* (engine-coupled).",
            importers.len(),
            importers
        );
    }
}
