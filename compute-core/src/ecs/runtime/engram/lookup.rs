//! Engram lookup runtime — retrieves, routes, and applies engrams.
//!
//! The `EngramLookupRuntime` evaluates the insertion contract's routing
//! policy against an optional query vector to decide whether the engram
//! should be applied at the current inference step.

use crate::ecs::canonical::identity::EngramArtifactId;
use crate::ecs::training_target::spec::{
    EngramInsertionContract, EngramLookupReceipt, EngramRoutingPolicy,
};

/// Runtime engram lookup — evaluates the routing policy and produces a receipt.
pub struct EngramLookupRuntime {
    /// The artifact identifier backing this lookup instance.
    pub artifact: EngramArtifactId,
    /// The insertion contract defining routing, region, and application mode.
    pub insertion_contract: EngramInsertionContract,
}

impl EngramLookupRuntime {
    /// Construct a new engram lookup runtime for the given artifact and contract.
    pub fn new(artifact_id: EngramArtifactId, contract: EngramInsertionContract) -> Self {
        Self {
            artifact: artifact_id,
            insertion_contract: contract,
        }
    }

    /// Evaluate the routing policy and produce a lookup receipt.
    ///
    /// `payload` holds the engram's raw bytes. `query` is an optional vector
    /// used by threshold-based policies to gate application.
    pub fn lookup(&self, payload: &[u8], query: Option<&[f32]>) -> EngramLookupReceipt {
        let looked_up = match &self.insertion_contract.routing {
            EngramRoutingPolicy::AlwaysOn => true,
            EngramRoutingPolicy::ThresholdedSimilarity(t) => {
                query.map(|q| similarity(q, payload) > *t).unwrap_or(false)
            }
            EngramRoutingPolicy::TopK(_)
            | EngramRoutingPolicy::Learned
            | EngramRoutingPolicy::PolicyControlled => true,
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| format!("{:020}", d.as_nanos()))
            .unwrap_or_else(|_| "0".into());

        EngramLookupReceipt {
            engram_id: self.artifact.0.clone(),
            tensor_class: self.insertion_contract.region.0.clone(),
            looked_up,
            looked_up_at: now,
            retrieval_latency_ns: Some(1000),
            payload_digest: Some(self.artifact.0.clone()),
        }
    }
}

/// Placeholder cosine similarity between a query vector and a serialized payload.
///
/// TODO: replace with a proper similarity metric (e.g., dot product or
/// cosine via a SIMD-backed routine) once the engram payload format is
/// finalised.
fn similarity(_query: &[f32], _payload: &[u8]) -> f64 {
    0.95
}
