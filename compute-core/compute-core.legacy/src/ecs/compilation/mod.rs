//! Compilation pipeline — phase IR, admission gate, ANE calibration lane,
//! ring-buffered staging, GPU k-means infrastructure, distill-compiler
//! foundation (phase types, activation arena, receipt manifest, memory
//! budget, calibration frontier, bridge provider trait, Level 1–3).
//!
//! Types delegate to the runtime residency contract in
//! `crate::ecs::backend::residency` for cross-backend transfer decisions.

#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub mod activation_abi;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub mod ane_eligibility;
pub mod ane_lane;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub mod apple_installation;
pub mod arena;
pub mod bench_metrics;
#[cfg(feature = "prism-backend")]
pub mod boundary_sensitivity;
pub mod bridge_provider;
pub mod cancel;
pub mod distill_core;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub mod epoch_scheduler;
pub mod failure_injector;
#[cfg(feature = "prism-backend")]
pub mod matrix_distill;
pub mod phase_ir;
pub mod phase_types;
pub mod receipt;
pub mod region_catalogue;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub mod region_planner;
#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub mod tri_lane;

// Not gated here: `level1/mod.rs` gates each Metal/CoreML-dependent submodule
// on `prism-backend` individually, so the std-only pieces (`kd_gate`) compile
// and unit-test on every host, Linux CI included.
pub mod level1;

#[cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]
pub use crate::ecs::system::gates::{LaneAdmissionGate, RiskPolicy};

#[cfg(all(target_os = "macos", feature = "prism-backend"))]
pub mod level2;
#[cfg(all(target_os = "macos", feature = "prism-backend"))]
pub mod level3;
