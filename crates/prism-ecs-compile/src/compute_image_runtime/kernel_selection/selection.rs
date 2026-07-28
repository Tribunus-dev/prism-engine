//! Kernel variant selection — compile-time artifact selection and
//! selection receipts.
//!
//! At compile time the compiler benchmarks candidate kernel
//! implementations against the target profile's hardware contract. The
//! best-performing candidate is selected per operation/shape-class
//! pair, and a [`KernelSelectionReceipt`] records the selection policy
//! version, candidate artifacts, resource-fit and numerical
//! qualification outcomes, and the chosen winner.

use serde::{Deserialize, Serialize};

use crate::compute_image_runtime::ContentHash;

/// Opaque identifier for a kernel variant.
pub type KernelVariantId = String;

/// A pre-selected kernel variant for one operation/shape-class pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreselectedKernelVariant {
    /// Operation this variant implements.
    pub operation: String,
    /// Execution shape class this variant was selected for.
    pub shape_class: String,
    /// Identifier of the selected artifact.
    pub selected_artifact: String,
    /// Tiling and pipeline parameters for the selected variant.
    pub selected_configuration: KernelConfiguration,
    /// Per-candidate benchmark evidence that fed the selection.
    pub candidate_evidence: Vec<KernelCandidateEvidence>,
    /// Selection receipt recording the decision and its qualifications.
    pub selection_receipt: KernelSelectionReceipt,
}

/// Tiling and pipeline parameters for a candidate kernel variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelConfiguration {
    /// Threadgroup size (number of threads per group).
    pub threadgroup_size: u32,
    /// Tile size in the M dimension.
    pub tile_m: u32,
    /// Tile size in the N dimension.
    pub tile_n: u32,
    /// Tile size in the K dimension.
    pub tile_k: u32,
    /// Pipeline identifier (e.g., for fused pipelines).
    pub pipeline_id: String,
}

/// Per-candidate benchmark evidence for kernel selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelCandidateEvidence {
    /// Candidate identifier.
    pub candidate_id: String,
    /// Operation this candidate implements.
    pub operation: String,
    /// Tiling and pipeline configuration.
    pub configuration: KernelConfiguration,
    /// Median latency in nanoseconds.
    pub median_latency_ns: u64,
    /// Minimum observed latency in nanoseconds.
    pub min_latency_ns: u64,
    /// Whether the candidate fits the target's resources.
    pub resource_fit: bool,
    /// Whether the candidate passed numerical verification.
    pub numerical_pass: bool,
}

/// Receipt for a kernel selection decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelSelectionReceipt {
    /// Content hash of the target profile this selection was made for.
    pub target_profile_hash: ContentHash,
    /// Identifiers of all candidate artifacts considered.
    pub candidate_artifacts: Vec<String>,
    /// Number of candidates considered.
    pub candidate_count: u32,
    /// Per-candidate resource-fit outcomes.
    pub resource_fit_outcomes: Vec<String>,
    /// Per-candidate numerical qualification results.
    pub numerical_qualification_results: Vec<String>,
    /// Identifier of the candidate that was selected as the winner.
    pub selected_winner: String,
    /// Selection policy version.
    pub selection_policy_version: String,
    /// Benchmark timestamp (RFC3339 string).
    pub benchmark_timestamp: String,
}
