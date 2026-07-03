//! Compilation pipeline — phase IR, admission gate, ANE calibration lane,
//! ring-buffered staging, GPU k-means infrastructure, distill-compiler
//! foundation (phase types, activation arena, receipt manifest, memory
//! budget, calibration frontier, bridge provider trait, Level 1–3).
//!
//! Types delegate to the runtime residency contract in
//! `crate::backend::residency` for cross-backend transfer decisions.

#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub mod activation_abi;
pub mod admission;
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub mod ane_admission_gate;
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub mod ane_eligibility;
pub mod ane_lane;
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub mod apple_installation;
pub mod arena;
pub mod bridge_provider;
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub mod epoch_scheduler;
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub mod evidence_probe;
pub mod failure_injector;
pub mod frontier;
pub mod memory_budget;
pub mod phase_ir;
pub mod phase_types;
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub mod profitability;
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub mod qualification_gate;
pub mod receipt;
pub mod region_catalogue;
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub mod region_planner;
pub mod staging;
#[cfg(any(feature = "mlx-backend", feature = "prism-backend"))]
pub mod tri_lane;

#[cfg(feature = "prism-backend")]
pub mod level1;
#[cfg(all(target_os = "macos", feature = "prism-backend"))]
pub mod level2;
#[cfg(all(target_os = "macos", feature = "prism-backend"))]
pub mod level3;
