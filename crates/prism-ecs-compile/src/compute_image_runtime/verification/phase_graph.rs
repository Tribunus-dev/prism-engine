//! Phase-graph verification receipt — the canonical attestation that a
//! compiled executable's phase graph is structurally well-formed.

use serde::{Deserialize, Serialize};

/// Phase graph verification receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseGraphVerificationReceipt {
    /// Identity of the artifact verified (digest string).
    pub artifact_identity: String,
    /// Number of phases in the graph.
    pub phase_count: u32,
    /// Number of edges in the graph.
    pub edge_count: u32,
    /// Whether the graph is structurally valid (no cycles, all
    /// dependencies satisfied).
    pub graph_valid: bool,
    /// 64-bit content hash of the phase graph topology.
    pub graph_hash: u64,
}
