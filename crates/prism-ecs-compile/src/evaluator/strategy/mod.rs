//! Search-facing evaluation strategy family, decomposed by single
//! authority into three sub-modules.
//!
//! The 1,000-LOC `strategy.rs` godfile (introduced during the
//! `evaluator.rs` godfile decomposition in commit `d0453c4f`)
//! is split into three single-authority sub-modules. The split
//! follows the per-authority contract that AGENTS.md and the
//! `prism-constitutional-rust-ecs` skill require of every new
//! `.rs` file:
//!
//! - [`behavioral`] — the abstract behavioral probe trait
//!   consumed by the strategy surface, and the canonical tree-spec
//!   speculation shapes ([`DraftModelConfig`], [`SpeculativeBranch`],
//!   [`TreeSpecDecoder`]) absorbed from the engine's draft/target
//!   orchestrator in commit `d0453c4f`.
//! - [`progressive`] — the bounded representation-reconstruction
//!   helpers ([`reconstruct_representation`], [`quantize_uniform`],
//!   [`quantize_ternary`]) and the [`parse_genome_from_string`]
//!   adapter that progressive stage evaluation uses to map a
//!   reference tensor through a candidate [`CandidateGenome`]
//!   representation and back to a comparable form.
//! - [`mapped`] — the constitutional [`MeasuredEvaluatorAdapter`]
//!   and [`MappedTensorEvaluationStrategy`] adapters, plus the
//!   workload and backend evaluation plumbing that drives the
//!   bounded reference probe through mixed-precision graph
//!   candidates, SpatialIR lowering, and ANE/Metal/Accelerate
//!   backend dispatch.
//!
//! The module-level authority of the original `strategy.rs` was the
//! "search-system evaluation strategy surface and the canonical
//! speculation shapes used by the engine's draft/target
//! orchestrator". After the decomposition, that authority is split
//! along the per-concern axis: the behavioral surface, the
//! progressive reconstruction helpers, and the mapped-tensor
//! strategy family. The `super::fail_closed` and
//! `super::objective` modules keep their existing authorities.
//!
//! ## Engine absorption (preserved)
//!
//! The engine's `compute-core/src/ecs/core/speculative.rs` keeps
//! the MLX-coupled helpers (criterion 4: FFI surface) and the
//! ANE-coupled `MultiSpecDraftModel` (criterion 1: hardware
//! dispatch path). The canonical data types were absorbed in
//! commit `d0453c4f` and re-export here.

#![forbid(unsafe_code)]

pub mod behavioral;
pub mod mapped;
pub mod progressive;

// Re-exports for `super` and for sibling modules
// (`super::objective`, `super::fail_closed`).
pub use behavioral::{BehavioralProbe, DraftModelConfig, SpeculativeBranch, TreeSpecDecoder};
pub use mapped::{MeasuredEvaluatorAdapter, MappedTensorEvaluationStrategy};

// Re-export the pub(crate) representation helpers at the module
// boundary so `super::objective` can keep importing
// `super::strategy::{parse_genome_from_string, quantize_ternary,
// reconstruct_representation}` without churn. The functions remain
// crate-private; only this module can re-export them.
pub(crate) use progressive::{
    parse_genome_from_string, quantize_ternary, reconstruct_representation,
};
