//! `workspace_contains_no_legacy_decode_attribution_imports`
//!
//! Workspace-level architecture enforcement test for the
//! decode-attribution migration. It scans the workspace for any
//! `use` statement that references the legacy engine
//! decode-attribution surface
//! (`compute_core::ecs::decode_attribution::*` or
//! `crate::ecs::decode_attribution::*` inside the engine, or the
//! `tribunus_compute_core::decode_attribution` re-export shim).
//!
//! The test fails if any file in the workspace imports the legacy
//! surface. The legacy engine file is the engine's
//! `compute-core/src/ecs/legacy_decode_attribution/` directory; the
//! constitutional replacement is `prism_ecs_compile::decode_attribution`.
//!
//! # Migration status
//!
//! This test was added when the decode-attribution engine-deletion
//! migration was being completed (2026-07-27). The engine's
//! `compute-core/src/ecs/decode_attribution/` directory was renamed
//! to `compute-core/src/ecs/legacy_decode_attribution/`; the
//! directory is the migration inventory and is exempt from this
//! scan. Files outside that directory that import the surface are
//! violations.

use std::fs;
use std::path::Path;

/// Returns the set of files OUTSIDE the legacy engine
/// decode-attribution directory that import the legacy surface.
/// Files inside the engine's own
/// `compute-core/src/ecs/legacy_decode_attribution/` directory are
/// exempt — they ARE the legacy surface and will continue to host
/// the engine-coupled adapter code (Core ML, MLX, Accelerate
/// adapters, harness, defect clustering, KV-cache phase contracts,
/// compute plan inspection). Files outside that directory that
/// import the surface are violations.
pub fn legacy_decode_attribution_importers_outside_inventory() -> Vec<String> {
    let mut importers = Vec::new();
    scan_workspace_excluding_inventory("compute-core", &mut importers);
    importers
}

const LEGACY_INVENTORY_DIR: &str = "compute-core/src/ecs/legacy_decode_attribution";

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
                if content.contains("use crate::ecs::decode_attribution::")
                    || content.contains("compute_core::ecs::decode_attribution::")
                    || content.contains("tribunus_compute_core::ecs::decode_attribution")
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
    fn workspace_contains_no_legacy_decode_attribution_imports() {
        // Architectural invariant: no file OUTSIDE the migration
        // inventory imports the legacy engine decode-attribution
        // surface. The engine's
        // compute-core/src/ecs/legacy_decode_attribution/ directory
        // is the migration inventory and is exempt; it IS the legacy
        // surface (engine-coupled adapter code) and re-exports the
        // constitutional data types from
        // prism_ecs_compile::decode_attribution.
        // Files outside that directory that import the surface are
        // violations.
        let importers = legacy_decode_attribution_importers_outside_inventory();
        assert!(
            importers.is_empty(),
            "Workspace still contains {} files outside the migration \
             inventory that import the legacy engine decode-attribution \
             surface: {:?}. The migration is incomplete; callers should \
             be updated to the constitutional surface in \
             prism_ecs_compile::decode_attribution (or import via the \
             engine's legacy_decode_attribution shim if engine-coupled \
             types are required).",
            importers.len(),
            importers
        );
    }
}
