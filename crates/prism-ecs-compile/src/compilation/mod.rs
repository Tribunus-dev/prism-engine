//! Constitutional compilation surface — phase IR, activation ABI, ANE
//! calibration, distill-compiler Level 1–3, bench metrics, and receipts.
//!
//! ## Sub-modules (single authority per file)
//!
//! | Sub-module | Authority |
//! |---|---|
//! | [`phase_types`] | Phase type taxonomy (PhaseType, ElementType, TensorDescriptor). |
//! | [`phase_ir`] | Phase IR identities (CompilationId, PhaseId, RegionId, …). |
//! | [`receipt`] | Compilation receipts (BlockReceipt, EngineExecutionLog, …). |
//! | [`bench_metrics`] | Std-only perplexity, throughput, spec-decode projection. |
//! | [`cancel`] | Cooperative cancellation (CancelToken, AbortToken). |
//! | [`failure_injector`] | FailureInjector trait + Noop / EpochFailureInjector. |
//! | [`bridge_provider`] | Level 3 bridge provider trait + capability / plan types. |
//! | [`distill_core`] | KD divergence, top-1 agreement (knowledge distillation). |
//! | [`arena`] | Ring-buffered activation arena for distill passes. |
//! | [`ane_lane`] | ANE-specific lane adapters. |
//! | [`activation_abi`] | Activation ABI (ActivationAbi, SlotLeaseId, PhysicalLayout). |
//! | [`ane_eligibility`] | ANE region eligibility check. |
//! | [`apple_installation`] | Apple-specific installation surface. |
//! | [`region_catalogue`] | Region admission catalogue. |
//! | [`region_planner`] | Region planning (CoreAiIsland, ScheduledOp). |
//! | [`tri_lane`] | Apple three-lane execution (ANE / Metal / Accelerate). |
//! | [`epoch_scheduler`] | Epoch-level scheduling. |
//! | [`boundary_sensitivity`] | Boundary sensitivity analysis. |
//! | [`matrix_distill`] | Matrix distillation. |
//! | [`level1`] / [`level2`] / [`level3`] | Distill-compiler Level 1 (Metal+Accelerate) / Level 2 (Core ML) / Level 3 (routing). |
//!
//! ## Migration status
//!
//! This surface absorbed the engine's `compute-core/src/ecs/compilation/`
//! directory (37 files, 17,165 LOC) into the constitutional compile
//! crate on 2026-07-27. The engine no longer owns these types; the
//! constitutional replacement is `prism_ecs_compile::compilation::*`.
//!
//! ## Cross-crate authority
//!
//! The lane admission gate and risk policy live in
//! `prism_ecs_constitutional::admission_gates`; we re-export them
//! here for engine callers.

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
pub use crate::admission_gate_re_exports::{LaneAdmissionGate, RiskPolicy};

#[cfg(all(target_os = "macos", feature = "prism-backend"))]
pub mod level2;
#[cfg(all(target_os = "macos", feature = "prism-backend"))]
pub mod level3;
