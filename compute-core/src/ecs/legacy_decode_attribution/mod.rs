//! `compute-core::ecs::legacy_decode_attribution` — engine-internal
//! decode-attribution execution plane (legacy surface).
//!
//! This module is the engine-internal continuation of the absorbed
//! `compute-core::ecs::decode_attribution/` subsystem. The
//! cross-platform data types and utility functions (receipts,
//! statistics, shape profiles, artifact hashing, breadcrumb writer,
//! environment capture, timer calibration, conformance metrics)
//! have been migrated to the constitutional home at
//! `prism_ecs_compile::decode_attribution` and are re-exported
//! here for source-compatibility with the engine binaries and
//! tests that historically imported them as
//! `tribunus_compute_core::decode_attribution::*`.
//!
//! The engine-coupled adapter code (Core ML harness, MLX adapter,
//! Accelerate adapter, defect clustering, KV-cache phase contracts,
//! Core ML MIL builder integration) remains engine-internal here
//! because it depends on engine FFI bridges and per-backend
//! executor stacks that are out of scope for the constitutional
//! crate.
//!
//! # Re-exports
//!
//! The `pub use prism_ecs_compile::decode_attribution::*;` lines
//! below expose the constitutional types under the legacy
//! `legacy_decode_attribution::*` path so engine callers continue
//! to compile unchanged. The architecture safety net
//! (`workspace_legacy_decode_attribution_imports`) enforces that
//! no NEW engine code imports the legacy
//! `crate::ecs::legacy_decode_attribution::*` path; it must use either
//! the constitutional surface directly or the engine's
//! `legacy_decode_attribution` shim.

pub mod artifact_hash;
pub mod backend_adapters;
pub mod breadcrumb;
pub mod compute_plan;
pub mod coreai_minimal_repro;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub mod decode_microphase_shape_map;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub mod defect_clustering;
pub mod environment;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub mod gap_report;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub mod graph_catalog;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub mod harness;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub mod lattice;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub mod lattice_validation;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub mod matrices;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub mod negative_evidence;
pub mod receipt;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub mod report;
pub mod shape_profiles;
pub mod statistics;
pub mod suite_manifest;
#[cfg(feature = "tensix")]
pub mod tensix_decode_plan;
pub mod timer_calibration;

// ── Constitutional re-exports ────────────────────────────────────────────
//
// The constitutional surface at `prism_ecs_compile::decode_attribution::*`
// owns the cross-platform data types and utility functions. The
// engine-internal legacy dir re-exports them under the same name
// path so engine binaries that historically imported
// `tribunus_compute_core::decode_attribution::receipt::DecodeAttributionReceipt`
// continue to compile. The re-exports are explicit (not glob) so
// the migration is auditable.

pub use prism_ecs_compile::decode_attribution::artifact_hash::{
    hash_directory_deterministic, DirectoryHashResult,
};
pub use prism_ecs_compile::decode_attribution::backend_adapters::conformance::{
    compute_conformance, hash_output, ConformanceMetrics,
};
pub use prism_ecs_compile::decode_attribution::backend_adapters::{
    BackendKind, BackendSupportStatus, BackendSupportTier, BackendTiming, PredictFailureClass,
};
pub use prism_ecs_compile::decode_attribution::breadcrumb::{
    last_breadcrumb, read_breadcrumbs, set_breadcrumb_path, write_breadcrumb,
};
pub use prism_ecs_compile::decode_attribution::compute_plan::{
    inspect_compute_plan, ComputePlanResult,
};
pub use prism_ecs_compile::decode_attribution::environment::{
    capture_host_environment, HostEnvironment,
};
pub use prism_ecs_compile::decode_attribution::receipt::{
    BackendVersionInfo, DecodeAttributionReceipt, ExecutionKind, ExecutionProof,
};
pub use prism_ecs_compile::decode_attribution::shape_profiles::ShapeProfile;
pub use prism_ecs_compile::decode_attribution::statistics::{
    compute_distribution_stats, DistributionStats,
};
pub use prism_ecs_compile::decode_attribution::timer_calibration::{
    calibrate_timer_overhead, TimerCalibration,
};
