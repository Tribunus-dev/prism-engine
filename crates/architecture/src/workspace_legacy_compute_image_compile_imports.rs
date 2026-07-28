//! `workspace_contains_no_legacy_compute_image_compile_imports`
//!
//! Workspace-level architecture enforcement test for the
//! `compute_image/{compile,orchestrator}/` migration. It scans the
//! workspace for any `use` statement that references the legacy engine
//! `compile/` or `orchestrator/` path
//! (`compute-core/src/ecs/compute_image::compile::*` or
//! `compute-core/src/ecs::compute_image::orchestrator::*` inside the
//! engine).
//!
//! The test fails if any file in the workspace imports the legacy
//! surface. The engine's `compute-core/src/ecs/compute_image/`
//! directory now houses `legacy_compute_image_compile/` and
//! `legacy_compute_image_compile_orchestrator/` (renamed from
//! `compile/` and `orchestrator/` in E-3); the constitutional
//! replacement is `prism_ecs_compile::compute_image_compile` for
//! data-only types and `crate::ecs::compute_image::legacy_compute_image_compile`
//! for engine-coupled implementations.
//!
//! # Migration status
//!
//! This test was added when the `ci-compile` engine-deletion migration
//! was being completed (2026-07-27). The engine's
//! `compute-core/src/ecs/compute_image/legacy_compute_image_compile/`
//! and `…/legacy_compute_image_compile_orchestrator/` directories
//! remain as the home for engine-coupled implementations pending
//! later absorption waves; files inside those directories are exempt
//! from the scan. Files outside those directories that still import
//! the old `compute_image::compile::X` or
//! `compute_image::orchestrator::X` path are violations.

use std::fs;
use std::path::Path;

/// Returns the set of files OUTSIDE the legacy engine compute_image
/// directories that import the legacy surface. Files inside the
/// engine's own `compute-core/src/ecs/compute_image/legacy_compute_image_compile/`
/// or `…/legacy_compute_image_compile_orchestrator/` directories are
/// exempt — they ARE the legacy surface and engine-coupled
/// implementations. Files outside those directories that import the
/// surface are violations.
pub fn legacy_compute_image_compile_importers_outside_inventory() -> Vec<String> {
    let mut importers = Vec::new();
    scan_workspace_excluding_inventory("compute-core", &mut importers);
    importers
}

const LEGACY_INVENTORY_DIRS: &[&str] = &[
    "compute-core/src/ecs/compute_image/legacy_compute_image_compile",
    "compute-core/src/ecs/compute_image/legacy_compute_image_compile_orchestrator",
];

fn is_under_inventory(path_str: &str) -> bool {
    LEGACY_INVENTORY_DIRS
        .iter()
        .any(|inv| path_str.contains(inv))
}

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
        if is_under_inventory(&path_str) {
            continue;
        }
        if p.is_dir() {
            scan_workspace_excluding_inventory(&path_str, importers);
        } else if p.extension().and_then(|e| e.to_str()) == Some("rs") {
            if let Ok(content) = fs::read_to_string(&p) {
                if content.contains("compute_image::compile::")
                    || content.contains("compute_image::orchestrator::")
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
    fn workspace_contains_no_legacy_compute_image_compile_imports() {
        // Architectural invariant: no file OUTSIDE the migration
        // inventory imports the legacy engine compute_image
        // `compile/` or `orchestrator/` surface. The engine's
        // `compute-core/src/ecs/compute_image/legacy_compute_image_compile/`
        // and `…/legacy_compute_image_compile_orchestrator/`
        // directories are the migration inventory and are exempt; they
        // ARE the legacy surface. Files outside those directories that
        // import the surface are violations and must be retargeted to
        // `prism_ecs_compile::compute_image_compile::*` (data-only) or
        // `crate::ecs::compute_image::legacy_compute_image_compile::*`
        // (engine-coupled).
        let importers = legacy_compute_image_compile_importers_outside_inventory();
        assert!(
            importers.is_empty(),
            "Workspace still contains {} files outside the migration \
             inventory that import the legacy engine compute_image \
             compile/orchestrator surface: {:?}. The migration is \
             incomplete; callers should be updated to \
             prism_ecs_compile::compute_image_compile::* (data-only) or \
             crate::ecs::compute_image::legacy_compute_image_compile::* \
             (engine-coupled).",
            importers.len(),
            importers
        );
    }
}
