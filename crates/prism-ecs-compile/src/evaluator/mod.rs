//! Evaluator integration for evolutionary search — decomposed by
//! single authority.
//!
//! This module owns five tightly-scoped sub-modules, each with a
//! single canonical authority. The decomposition follows the
//! `changelogs/2026-07-27-godfile-engine-mapping.md` §5 axes:
//!
//! - [`canary_window`] — the bounded active-layer working set
//!   (`CanaryWindow`).
//! - [`kv_evaluator`] — KV-cache candidate evaluation
//!   (`Mi300xKvEvaluator`, `evaluate_kv_reference_cache`).
//! - [`strategy`] — search-system evaluation strategies and the
//!   canonical tree-spec speculation shapes absorbed from the
//!   engine (`MeasuredEvaluatorAdapter`,
//!   `MappedTensorEvaluationStrategy`, `DraftModelConfig`,
//!   `SpeculativeBranch`, `TreeSpecDecoder`).
//! - [`objective`] — bounded reference probe and
//!   `TernaryObjectiveEvidence` composition
//!   (`MappedTensorBehavioralProbe`, `MappedTensorProbeContext`,
//!   `BehavioralProbe` impl, `SpecHubVerification`).
//! - [`fail_closed`] — production-mode fail-closed semantics
//!   (`extract_measurements`, `evaluate_ternary_evidence`,
//!   `create_measured_evaluator_from_daemon`).
//!
//! The engine's `compute-core/src/ecs/core/speculative.rs` is
//! thinned of the canonical data types (`DraftModelConfig`,
//! `SpeculativeBranch`, `TreeSpecDecoder`, `SpecHubVerification`)
//! which now live in [`strategy`] and [`objective`]. The ANE-coupled
//! `MultiSpecDraftModel` and the MLX-coupled `spechub_verify` family
//! remain engine-side per AGENTS.md criteria 1 and 4
//! (hardware / FFI surface).

#![forbid(unsafe_code)]

pub mod canary_window;
pub mod fail_closed;
pub mod kv_evaluator;
pub mod objective;
pub mod strategy;

pub use canary_window::{CanaryWindow, CanaryWindowError};
pub use kv_evaluator::{evaluate_kv_reference_cache, Mi300xKvEvaluator};
pub use objective::{
    genome_for_format, GenericNameAdapter, MappedTensorBehavioralProbe, MappedTensorProbeContext,
    SpecHubVerification, vector_rmse,
};
pub use strategy::{
    BehavioralProbe, DraftModelConfig, MeasuredEvaluatorAdapter, MappedTensorEvaluationStrategy,
    SpeculativeBranch, TreeSpecDecoder,
};
