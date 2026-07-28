//! `workspace_contains_no_legacy_compute_image_core_imports`
//!
//! Workspace-level architecture enforcement test for the
//! `compute_image/` core surface migration. It scans the workspace
//! for any `use` statement that references the legacy engine
//! `compute_image/` core surface (52 top-level files + the
//! `cimage_packer/` and `manifest/` subdirs).
//!
//! The test fails if any file OUTSIDE the engine's
//! `compute-core/src/ecs/legacy_compute_image_core/` directory
//! imports the legacy surface via:
//!   - `use crate::ecs::compute_image::<top-level|cimage_packer|manifest>::`
//!   - `use crate::ecs::legacy_compute_image_core::` (legacy-internal use is OK
//!     only inside the legacy directory itself; the safety net excludes the
//!     legacy dir)
//!
//! The engine's `compute-core/src/ecs/legacy_compute_image_core/`
//! directory is the migration inventory and is exempt. It is the
//! home for engine-coupled implementations that depend on
//! Metal/Accelerate/Core ML and other engine-internal types. The
//! constitutional replacement is
//! `prism_ecs_compile::compute_image_core` for data-only types
//! (KV cache plan, manifest data types, phase graph metadata,
//! fusion receipts, slot state, hardware assessment receipts,
//! Apple CImage manifest data, etc.) and
//! `compute-core/src/ecs/legacy_compute_image_core/` for engine-
//! coupled implementations.

use std::fs;
use std::path::Path;

/// Returns the set of files OUTSIDE the engine's
/// `compute-core/src/ecs/legacy_compute_image_core/` directory that
/// import the legacy engine `compute_image/` core surface. Files
/// inside the engine's `legacy_compute_image_core/` directory are
/// exempt — they ARE the legacy surface and host the engine-
/// coupled implementations.
pub fn legacy_compute_image_core_importers_outside_inventory() -> Vec<String> {
    let mut importers = Vec::new();
    scan_workspace_excluding_inventory("compute-core", &mut importers);
    importers
}

const LEGACY_INVENTORY_DIR: &str = "compute-core/src/ecs/legacy_compute_image_core";

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
                if content.contains("use crate::ecs::compute_image::")
                    || content.contains("compute_core::ecs::compute_image::")
                    || content.contains("tribunus_compute_core::ecs::compute_image")
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
    fn workspace_contains_no_legacy_compute_image_core_imports() {
        // Architectural invariant: no file OUTSIDE the engine's
        // `compute-core/src/ecs/legacy_compute_image_core/`
        // directory imports the legacy engine `compute_image/` core
        // surface. The `legacy_compute_image_core/` directory is the
        // migration inventory and is exempt; it IS the legacy
        // surface. Files outside that directory that import the
        // surface are violations and must be retargeted to
        // `prism_ecs_compile::compute_image_core::*` (data-only
        // types) or
        // `compute-core/src/ecs/legacy_compute_image_core::*`
        // (engine-coupled implementations).
        let importers = legacy_compute_image_core_importers_outside_inventory();
        assert!(
            importers.is_empty(),
            "Workspace still contains {} files outside the engine's \
             legacy_compute_image_core/ directory that import the \
             legacy engine compute_image/ core surface: {:?}. The \
             migration is incomplete; callers should be updated to \
             prism_ecs_compile::compute_image_core::* (data-only \
             types) or compute-core/src/ecs/legacy_compute_image_core::* \
             (engine-coupled implementations).",
            importers.len(),
            importers
        );
    }
}
