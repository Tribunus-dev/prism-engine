//! Constitutional compilation surface — phase IR, bench metrics, KD gate,
//! receipts, and pure-data descriptors.
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
//! | [`ane_eligibility`] | ANE region eligibility check (data types only). |
//! | [`region_catalogue`] | Region admission catalogue. |
//! | [`level1::kd_gate`] | KD gate (knowledge distillation scoring) — data + pure math. |
//! | [`level3`] | Level 3 routing (gates, providers, routing). |
//!
//! ## Migration status
//!
//! This surface absorbed the engine's `compute-core/src/ecs/compilation/`
//! directory (37 files, 17,165 LOC) into the constitutional compile
//! crate on 2026-07-27. Data-only files were absorbed directly;
//! engine-coupled implementations remain at
//! `compute-core/src/ecs/legacy_compilation/` and engine callers
//! use `crate::ecs::legacy_compilation::X` for those.
//!
//! ## Cross-crate authority
//!
//! The lane admission gate and risk policy live in
//! `prism_ecs_constitutional::admission_gates`; we re-export them
//! here for engine callers via [`admission_gate_re_exports`].

pub mod admission_gate_re_exports;
pub mod ane_eligibility;
pub mod arena;
pub mod bench_metrics;
pub mod bridge_provider;
pub mod cancel;
pub mod distill_core;
pub mod failure_injector;
pub mod level1;
pub mod level3;
pub mod phase_ir;
pub mod phase_types;
pub mod receipt;
pub mod region_catalogue;

pub use admission_gate_re_exports::{LaneAdmissionGate, RiskPolicy};
