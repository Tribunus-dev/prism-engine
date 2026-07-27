//! Top-level heterogeneous cimage section — the sealed root container.
//!
//! This module owns the **canonical top-level descriptor** of a
//! heterogeneous execution image: the versioned
//! [`HeterogeneousExecutionImage`] struct and its [`ModelIdentity`]
//! provenance record.
//!
//! Every cimage intended for Prism Engine serving must contain one
//! [`HeterogeneousExecutionImage`]. A backend-only (Metal-only) image
//! is represented as a degenerate one-lane heterogeneous graph.
//!
//! ── Design invariants ────────────────────────────────────────────────────
//!
//! * All types are `Serialize + Deserialize` via serde for embedding
//!   in the cimage as a dedicated JSON section.
//! * Types reference [`super::shared`] vocabulary (`ExecutionLane`,
//!   `ActivationAbi`, `ContentHash`) where appropriate.
//! * The image is immutable after sealing — no mutable runtime state
//!   here.
//! * The graph is guaranteed acyclic at emission time.

use serde::{Deserialize, Serialize};

use super::admission::CompiledAdmissionPlan;
use super::concurrency::CompiledConcurrencyPlan;
use super::contract::HeterogeneousExecutionContract;
use super::evidence::CompiledEvidenceContract;
use super::fallback::CompiledFallbackPlan;
use super::lane_programs::CompiledLanePrograms;
use super::policies::CompiledExecutionPolicies;
use super::variants::CompiledPhaseGraph;
use super::resource_plan::CompiledResourcePlan;
use super::shared::ContentHash;

/// Primary top-level cimage section for heterogeneous execution.
///
/// Bundles the compiler-emitted phase graph, resource plan, lane
/// programs, concurrency plan, admission rules, fallback topology,
/// execution policies, and evidence contract into one sealed
/// artifact.
///
/// The runtime consumes this image directly via the heterogeneous
/// runtime — it does not reconstruct backend placement, resource
/// ownership, or concurrency semantics from disconnected manifests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeterogeneousExecutionImage {
    pub image_version: u32,
    pub model_identity: ModelIdentity,
    pub graph_digest: ContentHash,
    pub phase_graph: CompiledPhaseGraph,
    pub resources: CompiledResourcePlan,
    pub lane_programs: CompiledLanePrograms,
    pub concurrency: CompiledConcurrencyPlan,
    pub admission: CompiledAdmissionPlan,
    pub fallback: CompiledFallbackPlan,
    pub execution_policy: CompiledExecutionPolicies,
    pub evidence_contract: CompiledEvidenceContract,
}

/// Identity and provenance of the imported model.
///
/// Recorded at compile time and frozen into the image. The runtime
/// uses it to attribute receipts and to log migration provenance;
/// the executor does not branch on it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelIdentity {
    pub model_name: String,
    pub model_family: String,
    pub model_variant: String,
    pub canonical_graph_hash: ContentHash,
    pub compile_timestamp: String,
    pub compiler_version: String,
}
