//! Decode Attribution Data Collection Gate.
//!
//! This module implements a measurement harness for Core ML decode
//! attribution: structured JSONL receipts capturing materialization,
//! compilation, load, warmup, and prediction timing across matrices
//! (compute-unit × graph family, shape × graph family), with reference
//! numerical conformance against the pure-Rust evaluator.

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
