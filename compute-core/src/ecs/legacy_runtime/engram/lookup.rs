//! Engram lookup runtime — retrieves, routes, and applies engrams.
//!
//! The `EngramLookupRuntime` evaluates the insertion contract's routing
//! policy against an optional query vector to decide whether the engram
//! should be applied at the current inference step.

use prism_ecs_constitutional::canonical::identity::EngramArtifactId;
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
            retrieval_latency_ns: None,
            payload_digest: Some(self.artifact.0.clone()),
        }
    }
}

/// Placeholder cosine similarity between a query vector and a serialized payload.
///
/// TODO: replace with a proper similarity metric (e.g., dot product or
/// cosine via a SIMD-backed routine) once the engram payload format is
/// finalised.
fn similarity(query: &[f32], payload: &[u8]) -> f64 {
    if query.is_empty() || payload.len() % std::mem::size_of::<f32>() != 0 {
        return 0.0;
    }
    let values: Vec<f32> = payload
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk is four bytes")))
        .collect();
    if values.len() != query.len() {
        return 0.0;
    }
    let dot: f64 = query
        .iter()
        .zip(&values)
        .map(|(a, b)| *a as f64 * *b as f64)
        .sum();
    let q_norm = query
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let p_norm = values
        .iter()
        .map(|x| (*x as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    if q_norm == 0.0 || p_norm == 0.0 {
        0.0
    } else {
        dot / (q_norm * p_norm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_constitutional::canonical::identity::{EngramArtifactId, RegionId};
    use crate::ecs::training_target::spec::{
        EngramApplication, EngramInsertionContract, EngramOperation, EngramRoutingPolicy,
    };

    fn runtime(threshold: f64) -> EngramLookupRuntime {
        EngramLookupRuntime::new(
            EngramArtifactId("engram".into()),
            EngramInsertionContract {
                region: RegionId("region".into()),
                operation: EngramOperation::Adapter,
                input_shape: prism_ecs_constitutional::canonical::identity::TensorShape { dims: vec![] },
                output_shape: prism_ecs_constitutional::canonical::identity::TensorShape { dims: vec![] },
                application: EngramApplication::AdditiveResidual,
                routing: EngramRoutingPolicy::ThresholdedSimilarity(threshold),
                maximum_latency_ns: None,
            },
        )
    }

    #[test]
    fn thresholded_lookup_uses_payload_cosine_similarity() {
        let payload = [1.0f32.to_le_bytes(), 0.0f32.to_le_bytes()].concat();
        assert!(runtime(0.9).lookup(&payload, Some(&[1.0, 0.0])).looked_up);
        assert!(!runtime(0.9).lookup(&payload, Some(&[0.0, 1.0])).looked_up);
    }
}
