//! Unified runtime — the orchestrator entity that owns execution mode,
//! workload profile selection, replay admission, and the public run_* /
//! replay_aot_* surface.
//!
//! This module owns the canonical authority for taking a loaded
//! [`super::super::model::RuntimeModel`] and turning it into a stateful
//! orchestrator that callers drive through [`UnifiedRuntime::run_batch`],
//! [`UnifiedRuntime::run_prefill`], [`UnifiedRuntime::run_decode`], and the
//! [`UnifiedRuntime::replay_aot`] family. The runtime composes the
//! smaller entities — [`super::binding::CImageBindingResolver`],
//! [`super::ane_backend::EmbeddedAneRouteBackend`],
//! [`super::kernel_dispatch::KernelRouteDispatcher`],
//! [`super::xdna_dispatch::CImageXdnaRouteDispatcher`], and the
//! [`super::certification::cpu_reference_inference`] fallback — but does
//! not duplicate their authority.
//!
//! # Module layout
//!
//! The orchestrator is decomposed by responsibility:
//! - [`UnifiedRuntime`] struct + state-management methods ([`new`],
//!   [`with_backend`], [`last_workload_selection`], [`reset_kv_cache`])
//!   live in this module.
//! - [`workload`] owns workload profile selection and measured-strategy
//!   installation ([`install_measured_strategy`],
//!   [`preferred_mixed_precision_profile`], etc.).
//! - [`replay`] owns AOT replay admission and the [`replay_aot*`] family.
//! - [`ane`] (cfg-gated on Apple Silicon) owns the ANE dispatch methods
//!   ([`dispatch_ane_int8*`]).
//! - [`run`] owns the public `run_batch` / `run_prefill` / `run_decode`
//!   surface.
//! - [`dispatch`] owns the internal token-to-logits orchestration
//!   ([`dispatch_tokens`], UOp program selection, heterogeneous-plan
//!   replay, helpers).
//!
//! Everything in this module is about scheduling, validation, and
//! dispatch glue; no tensor arithmetic, no kernel contract, no ANE
//! device calls live here.

use std::collections::HashMap;

use prism_ecs_kernel::KernelBackend;
use prism_spatial_ir::WorkloadScenario;

use super::model::RuntimeModel;
use super::ExecutionMode;

pub mod ane;
pub mod dispatch;
pub mod replay;
pub mod run;
pub mod workload;

// Re-export the dispatch helpers so the test module in `super` (which uses
// `use super::*;`) can call them as if they were methods. These are
// intentionally `pub` — the dispatch layer is the orchestrator's
// authoritative token-to-logits path, and tests exercise it directly.
pub use dispatch::selected_uop_program;

/// The unified runtime is the orchestrator entity around a loaded model.
///
/// Construction lifecycle:
/// ```text
/// RuntimeModel::load(path) ──► UnifiedRuntime::new
///                                  │
///                                  ├─► with_backend (optional)
///                                  │
///                                  ├─► run_batch / run_prefill / run_decode
///                                  │
///                                  ├─► replay_aot / replay_aot_apple /
///                                  │     replay_aot_with_xdna
///                                  │
///                                  └─► reset_kv_cache
/// ```
pub struct UnifiedRuntime {
    /// Loaded model data.
    pub(super) model: RuntimeModel,
    /// Optional hardware backend for accelerated dispatch. When `None`,
    /// the runtime falls back to the CPU reference path (Phase 9+).
    pub(super) backend: Option<Box<dyn KernelBackend>>,
    /// KV cache slots for autoregressive decode (one per layer).
    pub(super) kv_cache: Option<Vec<Vec<u8>>>,
    /// Current execution mode.
    pub(super) mode: ExecutionMode,
    /// Runtime measurements can override the static plan for a concrete
    /// workload without mutating the sealed CImage artifact.
    pub(super) measured_strategy_overrides: HashMap<WorkloadScenario, String>,
    pub(super) requested_batch_size: Option<u32>,
    /// Last selected heterogeneous workload profile applied at dispatch time,
    /// retained for correlated runtime observability and debugging.
    pub(super) last_workload_selection: Option<crate::workload_search::WorkloadThroughputEvidence>,
}

impl UnifiedRuntime {
    /// Create a new unified runtime from a loaded model.
    ///
    /// Defaults to [`ExecutionMode::Batch`] with no backend and no KV cache.
    /// Call [`with_backend`](Self::with_backend) to attach a hardware
    /// accelerator, and [`run_prefill`](Self::run_prefill) to switch to
    /// autoregressive mode.
    pub fn new(model: RuntimeModel) -> Self {
        let measured_strategy_overrides = model
            .uop_workload_evidence
            .iter()
            .map(|evidence| (evidence.scenario, evidence.selected_strategy.clone()))
            .collect();
        Self {
            model,
            backend: None,
            kv_cache: None,
            mode: ExecutionMode::Batch,
            measured_strategy_overrides,
            requested_batch_size: None,
            last_workload_selection: None,
        }
    }

    /// Return the latest dispatch workload decision, when available.
    pub fn last_workload_selection(
        &self,
    ) -> Option<&crate::workload_search::WorkloadThroughputEvidence> {
        self.last_workload_selection.as_ref()
    }

    /// Attach a hardware backend.
    ///
    /// When a backend is present, all dispatch calls route through
    /// [`KernelBackend::dispatch`]. Without one, the runtime uses the CPU
    /// reference path (where available).
    pub fn with_backend(mut self, backend: Box<dyn KernelBackend>) -> Self {
        self.backend = Some(backend);
        self
    }
}
