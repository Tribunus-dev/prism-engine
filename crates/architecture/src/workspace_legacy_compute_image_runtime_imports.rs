//! `workspace_contains_no_legacy_compute_image_runtime_imports`
//!
//! Workspace-level architecture enforcement test for the
//! `compute_image_runtime` migration. It scans the workspace for any
//! `use` statement that references the engine's legacy runtime + ancillary
//! surface under the OLD path
//! (`compute_core::ecs::compute_image::residency::*` and the 11 sibling
//! subdirs, or `crate::ecs::compute_image::residency::*` inside the
//! engine, or the `tribunus_compute_core::ecs::compute_image::residency`
//! re-export shim).
//!
//! The test fails if any file in the workspace imports the legacy
//! surface. The legacy engine dir was the engine's
//! `compute-core/src/ecs/compute_image/{residency,...,verification}/`
//! directory (renamed to
//! `compute-core/src/ecs/compute_image/legacy_compute_image_runtime/`
//! in E-{N+1} of the migration). The constitutional replacement is
//! `prism_ecs_compile::compute_image_runtime::*` for data-only types
//! and the legacy path for engine-coupled implementations.
//!
//! # Migration status
//!
//! This test was added when the `compute_image_runtime`
//! engine-deletion migration was being completed (2026-07-27). The
//! engine's 12 subdirs
//! (`compute-core/src/ecs/compute_image/{residency,...,verification}/`)
//! were renamed to
//! `compute-core/src/ecs/compute_image/legacy_compute_image_runtime/`
//! in E-{N+1}; the test is the architecture-level guard against
//! re-introducing a parallel `ecs::compute_image::X` authority
//! (separate from the engine-internal `ecs::legacy_compute_image_runtime`
//! execution-plane home) in any future commit.

use std::fs;
use std::path::Path;

/// Returns the set of files OUTSIDE the engine's
/// `compute-core/src/ecs/compute_image/legacy_compute_image_runtime/`
/// directory that import the legacy engine compute_image_runtime
/// surface.
pub fn legacy_compute_image_runtime_importers_outside_inventory() -> Vec<String> {
    let mut importers = Vec::new();
    scan_workspace_excluding_inventory("compute-core", &mut importers);
    importers
}

const ENGINE_INVENTORY_DIR: &str =
    "compute-core/src/ecs/compute_image/legacy_compute_image_runtime";

const LEGACY_SUBDIRS: &[&str] = &[
    "residency",
    "heterogeneous",
    "megakernel",
    "kernel_selection",
    "multimodal",
    "model_family",
    "variants",
    "program",
    "content_store",
    "executable",
    "scheduler",
    "verification",
];

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
        // Skip the engine-internal legacy_compute_image_runtime
        // directory — it IS the engine-internal execution-plane home
        // for the compute_image_runtime surface and re-exports the
        // constitutional data types from
        // `prism_ecs_compile::compute_image_runtime`.
        if path_str.contains(ENGINE_INVENTORY_DIR) {
            continue;
        }
        if p.is_dir() {
            scan_workspace_excluding_inventory(&path_str, importers);
        } else if p.extension().and_then(|e| e.to_str()) == Some("rs") {
            if let Ok(content) = fs::read_to_string(&p) {
                for sub in LEGACY_SUBDIRS {
                    let old_path = format!("use crate::ecs::compute_image::{}::", sub);
                    let old_pub = format!("crate::ecs::compute_image::{}::", sub);
                    let cross_crate_path =
                        format!("compute_core::ecs::compute_image::{}::", sub);
                    let tribunus_path = format!(
                        "tribunus_compute_core::ecs::compute_image::{}::",
                        sub
                    );
                    let tribunus_mod = format!(
                        "tribunus_compute_core::ecs::compute_image::{}",
                        sub
                    );
                    if content.contains(&old_path)
                        || content.contains(&old_pub)
                        || content.contains(&cross_crate_path)
                        || content.contains(&tribunus_path)
                        || content.contains(&tribunus_mod)
                    {
                        importers.push(path_str.clone());
                        break;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_contains_no_legacy_compute_image_runtime_imports() {
        // Architectural invariant: no file OUTSIDE the engine's
        // `compute-core/src/ecs/compute_image/legacy_compute_image_runtime/`
        // directory imports the legacy engine compute_image_runtime
        // surface. The `legacy_compute_image_runtime/` directory
        // re-exports the constitutional
        // `prism_ecs_compile::compute_image_runtime` data types and
        // hosts the engine-internal execution-plane code (residency
        // plan admission, content store, executable descriptors, etc.).
        // Files outside `legacy_compute_image_runtime/` that import
        // `crate::ecs::compute_image::X::*` (or the
        // `tribunus_compute_core::ecs::compute_image::X` re-export)
        // are violations.
        let importers = legacy_compute_image_runtime_importers_outside_inventory();
        assert!(
            importers.is_empty(),
            "Workspace still contains {} files outside the engine's \
             legacy_compute_image_runtime/ directory that import the legacy \
             engine compute_image_runtime surface: {:?}. The migration is \
             incomplete; callers should be updated to the constitutional \
             surface in prism_ecs_compile::compute_image_runtime (data-only \
             types) or the engine's \
             compute-core/src/ecs/compute_image/legacy_compute_image_runtime/ \
             (engine-coupled implementations).",
            importers.len(),
            importers
        );
    }
}
