//! Canonical execution graph for Gemma 4 + MTP (Multi-Token Prediction).
//!
//! Defines the phases: prefill, target decode, MTP draft, verification,
//! acceptance, and sampling. The deployment compiler resolves each phase
//! to concrete Metal kernels through the catalogue.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Phase types in the Gemma 4 + MTP execution pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MTPPhase {
    /// Embedding lookup + RoPE precomputation.
    InputEmbedding,
    /// Target model prefill over prompt tokens.
    TargetPrefill,
    /// MTP drafter predicts K future tokens in parallel.
    MTPDraft,
    /// Target model verifies draft tokens.
    TargetVerify,
    /// Accept matching prefix of draft tokens.
    Acceptance,
    /// KV commit for accepted tokens.
    KVCommit,
    /// Sample fallback token at first rejection.
    SampleFallback,
    /// Single-token decode for verified position.
    TargetDecode,
    /// Project to vocabulary and sample.
    OutputProjection,
}

/// Edge type between execution graph nodes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MTPEdge {
    pub from: MTPPhase,
    pub to: MTPPhase,
    /// How many tokens flow along this edge.
    pub token_count: usize,
    /// Whether this edge is taken conditionally (e.g. after rejection).
    pub conditional: bool,
}

/// Canonical execution graph for Gemma 4 with optional MTP speculative decoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MTPExecutionGraph {
    /// All nodes in the graph.
    pub phases: Vec<MTPPhase>,
    /// Directed edges between phases.
    pub edges: Vec<MTPEdge>,
    /// MTP depth (K future tokens), 0 = no speculation.
    pub mtp_depth: usize,
    /// Whether the drafter shares embeddings with the target.
    pub shared_embeddings: bool,
    /// Whether the drafter shares norm parameters with the target.
    pub shared_norm: bool,
    /// Maximum accepted prefix length before forced fallback.
    pub max_accepted_prefix: usize,
    /// Target-only fallback path (edges when MTP is disabled).
    pub fallback_edges: Vec<MTPEdge>,
}

impl MTPExecutionGraph {
    /// Build the default target-only graph (no speculation).
    pub fn target_only() -> Self {
        Self {
            phases: vec![
                MTPPhase::InputEmbedding,
                MTPPhase::TargetPrefill,
                MTPPhase::TargetDecode,
                MTPPhase::OutputProjection,
            ],
            edges: vec![
                MTPEdge {
                    from: MTPPhase::InputEmbedding,
                    to: MTPPhase::TargetPrefill,
                    token_count: 0,
                    conditional: false,
                },
                MTPEdge {
                    from: MTPPhase::TargetPrefill,
                    to: MTPPhase::TargetDecode,
                    token_count: 1,
                    conditional: false,
                },
                MTPEdge {
                    from: MTPPhase::TargetDecode,
                    to: MTPPhase::OutputProjection,
                    token_count: 1,
                    conditional: false,
                },
            ],
            mtp_depth: 0,
            shared_embeddings: false,
            shared_norm: false,
            max_accepted_prefix: 0,
            fallback_edges: vec![],
        }
    }

    /// Build a graph with MTP speculative decoding at the given depth.
    pub fn with_mtp(depth: usize, shared_embeddings: bool, shared_norm: bool) -> Self {
        let mut phases = vec![MTPPhase::InputEmbedding, MTPPhase::TargetPrefill];

        // After prefill, the MTP loop runs
        for _ in 0..depth {
            phases.push(MTPPhase::MTPDraft);
        }
        phases.push(MTPPhase::TargetVerify);
        phases.push(MTPPhase::Acceptance);
        phases.push(MTPPhase::KVCommit);
        phases.push(MTPPhase::SampleFallback);
        phases.push(MTPPhase::TargetDecode);
        phases.push(MTPPhase::OutputProjection);

        Self {
            phases,
            edges: vec![
                // Prefill → first draft
                MTPEdge {
                    from: MTPPhase::InputEmbedding,
                    to: MTPPhase::TargetPrefill,
                    token_count: 0,
                    conditional: false,
                },
                // Prefill → MTP draft K tokens
                MTPEdge {
                    from: MTPPhase::TargetPrefill,
                    to: MTPPhase::MTPDraft,
                    token_count: 1,
                    conditional: false,
                },
                // Draft → draft (cascade: draft K predicts draft K-1, ...)
                MTPEdge {
                    from: MTPPhase::MTPDraft,
                    to: MTPPhase::MTPDraft,
                    token_count: depth,
                    conditional: false,
                },
                // Last draft → target verification
                MTPEdge {
                    from: MTPPhase::MTPDraft,
                    to: MTPPhase::TargetVerify,
                    token_count: depth,
                    conditional: false,
                },
                // Verify → accept prefix
                MTPEdge {
                    from: MTPPhase::TargetVerify,
                    to: MTPPhase::Acceptance,
                    token_count: 0,
                    conditional: false,
                },
                // Accept → KV commit
                MTPEdge {
                    from: MTPPhase::Acceptance,
                    to: MTPPhase::KVCommit,
                    token_count: 0,
                    conditional: false,
                },
                // Accept → (if all accepted) draft next K
                MTPEdge {
                    from: MTPPhase::Acceptance,
                    to: MTPPhase::MTPDraft,
                    token_count: depth,
                    conditional: true,
                },
                // Fallback: sample at rejection → decode
                MTPEdge {
                    from: MTPPhase::SampleFallback,
                    to: MTPPhase::TargetDecode,
                    token_count: 1,
                    conditional: true,
                },
                // Decode → project
                MTPEdge {
                    from: MTPPhase::TargetDecode,
                    to: MTPPhase::OutputProjection,
                    token_count: 1,
                    conditional: false,
                },
                // Project → loop back to draft or decode
                MTPEdge {
                    from: MTPPhase::OutputProjection,
                    to: MTPPhase::MTPDraft,
                    token_count: depth,
                    conditional: true,
                },
                MTPEdge {
                    from: MTPPhase::OutputProjection,
                    to: MTPPhase::TargetDecode,
                    token_count: 1,
                    conditional: true,
                },
            ],
            mtp_depth: depth,
            shared_embeddings,
            shared_norm,
            max_accepted_prefix: depth.saturating_sub(1),
            fallback_edges: vec![MTPEdge {
                from: MTPPhase::OutputProjection,
                to: MTPPhase::TargetDecode,
                token_count: 1,
                conditional: false,
            }],
        }
    }

    /// Return the kernel semantic IDs needed for each phase on the target model.
    pub fn target_kernels(&self) -> Vec<&'static str> {
        vec![
            "gemma4.rms_norm",
            "gemma4.rope",
            "gemma4.proj_qkv",
            "gemma4.attention",
            "gemma4.proj_o",
            "gemma4.gate_up_proj",
            "gemma4.silu",
            "gemma4.down_proj",
            "gemma4.residual_add",
            "gemma4.output_proj",
        ]
    }

    /// Return the kernel semantic IDs needed for MTP drafter (subset of target).
    pub fn mtp_kernels(&self) -> Vec<&'static str> {
        if self.mtp_depth == 0 {
            return vec![];
        }
        vec![
            "gemma4.mtp_proj",
            "gemma4.mtp_attention",
            "gemma4.mtp_output",
        ]
    }

    /// Map an MTP phase to its canonical kernel semantic ID for the Metal catalogue.
    pub fn semantic_id_for_phase(phase: &MTPPhase) -> &'static str {
        match phase {
            MTPPhase::InputEmbedding => "prism.gemma4.embedding",
            MTPPhase::TargetPrefill => "prism.gemma4.prefill",
            MTPPhase::MTPDraft => "prism.gemma4.mtp_draft",
            MTPPhase::TargetVerify => "prism.gemma4.verify",
            MTPPhase::Acceptance => "prism.gemma4.acceptance",
            MTPPhase::KVCommit => "prism.gemma4.kv_commit",
            MTPPhase::SampleFallback => "prism.gemma4.sample_fallback",
            MTPPhase::TargetDecode => "prism.gemma4.decode",
            MTPPhase::OutputProjection => "prism.gemma4.output_proj",
        }
    }
}

/// Weight sharing contract between target model and MTP drafter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MTPShareContract {
    /// Whether embeddings are shared (same PhysicalSegmentId).
    pub shared_embeddings: bool,
    /// Whether normalization parameters are shared.
    pub shared_norm: bool,
    /// Whether the output head is shared.
    pub shared_output_head: bool,
    /// List of logical tensor names shared between target and drafter.
    pub shared_tensors: Vec<String>,
}

impl MTPShareContract {
    /// Build the share contract from MTP depth and checkpoint configuration.
    pub fn from_inspection(_mtp_depth: usize, config: &serde_json::Value) -> Self {
        let shared_embeddings = config
            .get("share_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);
        let shared_output_head = config
            .get("tie_word_embeddings")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        Self {
            shared_embeddings,
            shared_norm: true,
            shared_output_head,
            shared_tensors: Vec::new(),
        }
    }

    /// Resolve the weight sharing map: target tensor name -> drafter tensor name.
    ///
    /// Each boolean flag in the contract maps a conventional tensor name pair;
    /// the `shared_tensors` list adds explicit 1:1 name matches.
    pub fn resolve_share_map(&self) -> HashMap<String, String> {
        let mut map = HashMap::new();

        if self.shared_embeddings {
            map.insert(
                "embedding.weight".to_string(),
                "embedding.weight".to_string(),
            );
        }
        if self.shared_norm {
            map.insert(
                "model.norm.weight".to_string(),
                "mtp.norm.weight".to_string(),
            );
        }
        if self.shared_output_head {
            map.insert(
                "lm_head.weight".to_string(),
                "mtp.output.weight".to_string(),
            );
        }
        for tensor in &self.shared_tensors {
            map.insert(tensor.clone(), tensor.clone());
        }

        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_only_graph() {
        let g = MTPExecutionGraph::target_only();
        assert_eq!(g.phases.len(), 4);
        assert_eq!(g.edges.len(), 3);
        assert_eq!(g.mtp_depth, 0);
        assert_eq!(g.target_kernels().len(), 10);
    }

    #[test]
    fn test_mtp_graph_generates_phases() {
        let g = MTPExecutionGraph::with_mtp(3, true, true);
        assert_eq!(g.mtp_depth, 3);
        assert!(g.phases.contains(&MTPPhase::MTPDraft));
        assert!(g.phases.contains(&MTPPhase::TargetVerify));
        assert!(g.phases.contains(&MTPPhase::Acceptance));
        assert_eq!(g.mtp_kernels().len(), 3);
    }

    #[test]
    fn test_mtp_zero_depth_returns_empty_kernels() {
        let g = MTPExecutionGraph::target_only();
        assert!(g.mtp_kernels().is_empty());
    }

    #[test]
    fn test_share_contract_defaults() {
        let config = serde_json::json!({});
        let contract = MTPShareContract::from_inspection(3, &config);
        assert!(contract.shared_embeddings);
        assert!(!contract.shared_output_head);
    }
}
