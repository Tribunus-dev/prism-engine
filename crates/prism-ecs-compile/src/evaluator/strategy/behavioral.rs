//! Behavioral probe trait and the canonical tree-spec speculation
//! shapes absorbed from the engine's draft/target orchestrator.
//!
//! **Single authority:** The abstract behavioral probe trait consumed
//! by the constitutional strategy surface, and the canonical
//! pure-data tree-spec speculation shapes
//! ([`DraftModelConfig`], [`SpeculativeBranch`], [`TreeSpecDecoder`])
//! absorbed from `compute-core/src/ecs/core/speculative.rs` in commit
//! `d0453c4f`. The MLX-coupled `SpecHub` verification functions
//! remain engine-side (criterion 4: FFI surface); the ANE-coupled
//! `MultiSpecDraftModel` also stays engine-side (criterion 1:
//! hardware dispatch path).

#![forbid(unsafe_code)]

use prism_ecs_ir::evolution::foundation::CandidateGenome;
use prism_ecs_ir::evolution::progressive::TernaryObjectiveEvidence;

use crate::search::SearchError;

// ---------------------------------------------------------------------------
// Tree-spec speculation shapes (absorbed from
// `compute-core/src/ecs/core/speculative.rs`)
// ---------------------------------------------------------------------------

/// Description of a draft model's architecture.
///
/// Weights are stored as group-quantized so the draft model can be
/// loaded into any backend that supports group-wise quantisation
/// (MLX, Accelerate, ANE). The struct is the canonical authority for
/// the shape of a draft model; the ANE-specific loading paths live in
/// the engine.
#[derive(Debug, Clone)]
pub struct DraftModelConfig {
    pub n_heads: u32,
    pub head_dim: u32,
    pub n_layers: u32,
}

/// One speculative branch in a tree-structured speculation.
///
/// Each branch is a sequence of draft tokens along a single path
/// through the speculation tree, together with metadata about its
/// acceptance probability and the KV-cache generation that produced
/// it.
#[derive(Debug, Clone)]
pub struct SpeculativeBranch {
    /// Draft token IDs along this branch.
    pub tokens: Vec<u32>,
    /// Estimated probability that the entire branch will be accepted by
    /// the target model.
    pub acceptance_prob: f32,
    /// Indices of the draft-model layers that generated this branch.
    pub draft_layer_indices: Vec<u32>,
    /// Provisional page IDs that the memory planner reserved for this
    /// branch's KV-cache entries.
    pub provisional_pages: Vec<u32>,
    /// Total KV-cache generation cost (bytes) for this branch.
    pub kv_generation: u64,
}

/// Tree-structured speculative decoder.
///
/// Manages a draft model and generates multiple candidate branches
/// forming a speculation tree. The target model verifies all branches
/// in a single batched forward pass; the first token (by tree order)
/// that passes the acceptance criterion is committed.
///
/// This is a pure-data canonical type — the actual proposal and
/// verification algorithms are still stubs in the engine; once a
/// concrete engine-side implementation lands, this type is the
/// authority for the shape of the result.
#[derive(Debug, Clone)]
pub struct TreeSpecDecoder {
    pub draft: DraftModelConfig,
    pub max_branches: u32,
    pub max_depth: u32,
    pub acceptance_threshold: f32,
}

impl TreeSpecDecoder {
    /// Propose a set of speculative branches from the current context.
    /// Stub — returns an empty branch list until the engine-side
    /// proposal algorithm is implemented.
    pub fn propose(&self, _context: &[u32]) -> Vec<SpeculativeBranch> {
        Vec::new()
    }

    /// Verify speculative branches against the target model's logits.
    /// Stub — returns an empty token sequence until the engine-side
    /// verification algorithm is implemented.
    pub fn verify(&mut self, _branches: &[SpeculativeBranch], _target_logits: &[f32]) -> Vec<u32> {
        Vec::new()
    }
}

// ---------------------------------------------------------------------------
// BehavioralProbe trait — abstract probe surface consumed by
// MeasuredEvaluatorAdapter and implemented by the objective layer
// ---------------------------------------------------------------------------

/// A reference-aware behavioral probe that maps a candidate genome
/// to a [`TernaryObjectiveEvidence`]. The constitutional evaluator
/// composes a [`MeasuredEvaluatorAdapter`] with a `BehavioralProbe` to
/// produce activation, logit, and router signals that a synthetic
/// evaluator cannot supply.
pub trait BehavioralProbe: Send + Sync {
    fn evaluate(
        &self,
        genome: &CandidateGenome,
        context: &[u8],
    ) -> Result<TernaryObjectiveEvidence, SearchError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_spec_decoder_stub_returns_empty() {
        let decoder = TreeSpecDecoder {
            draft: DraftModelConfig {
                n_heads: 4,
                head_dim: 32,
                n_layers: 2,
            },
            max_branches: 4,
            max_depth: 4,
            acceptance_threshold: 0.5,
        };
        assert!(decoder.propose(&[]).is_empty());
        let mut decoder = decoder;
        let branches = vec![SpeculativeBranch {
            tokens: vec![1, 2, 3],
            acceptance_prob: 0.9,
            draft_layer_indices: vec![0],
            provisional_pages: vec![],
            kv_generation: 0,
        }];
        assert!(decoder.verify(&branches, &[0.0]).is_empty());
    }
}
