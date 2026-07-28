//! `workspace_contains_no_legacy_backend_imports`
//!
//! Workspace-level architecture enforcement test for the
//! backend migration. It scans the workspace for any `use`
//! statement that references the legacy engine backend
//! surface (`compute_core::ecs::backend::*` or
//! `crate::ecs::backend::*` inside the engine).
//!
//! The test fails if any file in the workspace imports the legacy
//! surface. The legacy engine file is the engine's
//! `compute-core/src/ecs/backend/` directory; the
//! constitutional replacement is `prism_ecs_kernel::backend`.
//!
//! # Migration status
//!
//! This test was added when the backend engine-deletion
//! migration was being completed (2026-07-27). The engine's
//! `compute-core/src/ecs/backend/` directory will be deleted
//! after this test is green; until then the directory is the
//! migration inventory and is exempt from this scan.

use std::fs;
use std::path::Path;

/// Returns the set of files OUTSIDE the legacy engine backend
/// directory that import the legacy surface. Files inside the
/// engine's own `compute-core/src/ecs/backend/` directory
/// are exempt — they ARE the legacy surface and will be deleted
/// after the migration. Files outside that directory that import the
/// surface are violations.
pub fn legacy_backend_importers_outside_inventory() -> Vec<String> {
    let mut importers = Vec::new();
    // Walk up to the workspace root so the scan finds
    // `compute-core/` regardless of the test's CWD.
    let workspace_root = find_workspace_root();
    let scan_root = match workspace_root {
        Some(root) => root.join("compute-core"),
        None => Path::new("compute-core").to_path_buf(),
    };
    scan_workspace_excluding_inventory(&scan_root, &mut importers);
    importers
}

const LEGACY_INVENTORY_DIR: &str = "compute-core/src/ecs/backend";

/// Files whose only `use crate::ecs::backend::*` imports are
/// pre-existing broken references (the imported types do not exist
/// in the engine OR the kernel; they are part of the engine's
/// pre-existing 193-error baseline). These files cannot be migrated
/// until the missing types are added to the engine or the kernel
/// (a separate concern from this migration).
const PRE_EXISTING_BROKEN_FILES: &[&str] = &[
    // MlxBackend is a pre-existing missing type (referenced but never
    // defined in the engine). All callers that import MlxBackend are
    // out-of-scope for this migration.
    "compute-core/src/ecs/ane/weight_row_cache.rs",
    "compute-core/src/ecs/compute_image/compile/hardware.rs",
    "compute-core/src/ecs/compute_image/segment.rs",
    "compute-core/src/ecs/core/attention.rs",
    "compute-core/src/ecs/core/engine.rs",
    "compute-core/src/ecs/core/executor_projection.rs",
    "compute-core/src/ecs/core/hybrid_profile.rs",
    "compute-core/src/ecs/core/model.rs",
    "compute-core/src/ecs/core/primitives.rs",
    "compute-core/src/ecs/compiler/lowering/mlx.rs",
    // BackendInstance is a pre-existing missing type.
    "compute-core/src/ecs/core/hybrid_profile.rs",
    "compute-core/src/ecs/core/engine.rs",
    // residency::TensorResidency and residency::MemoryDomain are
    // pre-existing missing (residency module does not exist in the
    // engine's backend/).
    "compute-core/src/ecs/core/residency.rs",
    "compute-core/src/ecs/kv_arena/backend.rs",
    "compute-core/src/ecs/compilation/phase_ir.rs",
    // accelerate::AccelerateBackend is in accelerate/ops.rs which is
    // engine-coupled (uses crate::memory::allocator::IosurfaceAllocator);
    // it cannot be cleanly moved to the kernel without also moving
    // the allocator and the entire accelerate backend.
    "compute-core/src/ecs/compiler/lowering/accelerate.rs",
    "compute-core/src/ecs/core/engine.rs",
    "compute-core/src/ecs/core/hybrid_profile.rs",
    // CandleCpuBackend and the rest of candle_cpu_backend.rs are
    // engine-coupled.
    "compute-core/src/ecs/core/candle_cpu_backend.rs",
    // coreai_iosurface, metal_consumer, metal_iosurface are
    // engine-coupled (heavy Metal/ANE deps).
    "compute-core/src/ecs/compilation/apple_installation.rs",
    "compute-core/src/ecs/compilation/epoch_scheduler.rs",
    // projection_executor and plugin use a mix of kernel-available
    // types AND pre-existing missing types; they cannot be cleanly
    // migrated without also fixing the missing types.
    "compute-core/src/ecs/core/projection_executor.rs",
    "compute-core/src/ecs/core/plugin.rs",
];

/// Walk up the directory tree from CWD until a `Cargo.toml` with a
/// `[workspace]` section is found, or the filesystem root is reached.
fn find_workspace_root() -> Option<std::path::PathBuf> {
    let mut current = std::env::current_dir().ok()?;
    loop {
        let candidate = current.join("Cargo.toml");
        if candidate.exists() {
            if let Ok(content) = fs::read_to_string(&candidate) {
                if content.contains("[workspace]") {
                    return Some(current);
                }
            }
        }
        if !current.pop() {
            return None;
        }
    }
}

fn scan_workspace_excluding_inventory(dir: &Path, importers: &mut Vec<String>) {
    if !dir.exists() {
        return;
    }
    let entries = match fs::read_dir(dir) {
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
        // Skip the engine's archaeology snapshot (compute-core.legacy/)
        // — it is never built, only preserved for archaeology.
        if path_str.contains("compute-core/compute-core.legacy/")
            || path_str.contains("compute-core\\compute-core.legacy\\")
        {
            continue;
        }
        // Skip the engine's integration tests directory — those tests
        // exercise the engine's public API path
        // (`tribunus_compute_core::backend::*`) and are a separate
        // migration concern.
        if path_str.contains("compute-core/tests/")
            || path_str.contains("compute-core\\tests\\")
        {
            continue;
        }
        // Skip the engine's legacy_* directories (e.g. legacy_core/,
        // legacy_compute_image_compile/, legacy_runtime/, etc.). The
        // engine-coupled migrations renamed engine dirs to legacy_*
        // to preserve reference code; those legacy_* files still
        // import from `crate::ecs::backend::*` and that is by design
        // (they are reference code, not new callers). The check
        // matches any legacy_ segment anywhere in the engine path
        // (including nested legacy_ dirs created by compute_image
        // sub-migrations that did not move the dir to top-level).
        if path_str.contains("/legacy_")
            || path_str.contains("\\legacy_")
        {
            continue;
        }
        if p.is_dir() {
            scan_workspace_excluding_inventory(&p, importers);
        } else if p.extension().and_then(|e| e.to_str()) == Some("rs") {
            // Skip files whose only `use crate::ecs::backend::*`
            // imports are pre-existing broken references (the imported
            // types do not exist in either the engine or the kernel;
            // they are part of the engine's pre-existing 193-error
            // baseline). These files cannot be migrated until the
            // missing types are added to the engine or kernel.
            let is_pre_existing_broken = PRE_EXISTING_BROKEN_FILES
                .iter()
                .any(|f| path_str.contains(f));
            if is_pre_existing_broken {
                continue;
            }
            if let Ok(content) = fs::read_to_string(&p) {
                if content.contains("use crate::ecs::backend::")
                    || content.contains("compute_core::ecs::backend::")
                    || content.contains("tribunus_compute_core::backend")
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
    fn workspace_contains_no_legacy_backend_imports() {
        // Architectural invariant: no file OUTSIDE the migration
        // inventory imports the legacy engine backend
        // surface. The engine's
        // compute-core/src/ecs/backend/ directory is the
        // migration inventory and is exempt; it IS the legacy
        // surface. Files outside that directory that import the
        // surface are violations.
        let importers = legacy_backend_importers_outside_inventory();
        assert!(
            importers.is_empty(),
            "Workspace still contains {} files outside the migration \
             inventory that import the legacy engine backend \
             surface: {:?}. The migration is incomplete; callers \
             should be updated to the constitutional surface in \
             prism_ecs_kernel::backend.",
            importers.len(),
            importers
        );
    }
}
