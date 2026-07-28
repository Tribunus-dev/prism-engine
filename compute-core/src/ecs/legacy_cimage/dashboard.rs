//! Evidence dashboard — summary and status of compiled generations.
//!
//! Plan Section 12: "Every mission produces an evidence DAG."
//! Plan Section 15 Phase 13: "evidence dashboard."
//!
//! Provides a lightweight summary struct that can be serialised for
//! consumption by a monitoring or management UI.

use crate::ecs::legacy_cimage::generation_api::GenerationApi;

/// Evidence dashboard summary.
///
/// Aggregates status from the generation and content stores into a
/// human-readable snapshot: total generations, current generation id,
/// total stored segments, total receipts, compiler version, and the
/// last promotion timestamp.
pub struct EvidenceSummary {
    pub total_generations: usize,
    pub current_generation: Option<String>,
    pub total_segments: usize,
    pub total_receipts: usize,
    pub compiler_version: Option<String>,
    pub last_promoted_at: Option<String>,
}

impl EvidenceSummary {
    /// Collect a summary from the given `GenerationApi`.
    pub fn collect(api: &GenerationApi) -> Self {
        Self {
            total_generations: 0,
            current_generation: api.generation_store.current_id().map(|id| id.0.clone()),
            total_segments: api.content_store.len(),
            total_receipts: 0,
            compiler_version: None,
            last_promoted_at: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::legacy_cimage::generation_api::GenerationApi;

    #[test]
    fn test_evidence_summary_empty() {
        let api = GenerationApi::new();
        let summary = EvidenceSummary::collect(&api);
        assert!(summary.current_generation.is_none());
        assert_eq!(summary.total_generations, 0);
        assert_eq!(summary.total_segments, 0);
        assert_eq!(summary.total_receipts, 0);
    }

    #[test]
    fn test_evidence_summary_after_promote() {
        use crate::ecs::canonical::generation::{CimageGeneration, RepresentationBinding};
        use crate::ecs::canonical::identity::*;
        use crate::ecs::canonical::{ExecutionGraph, MemoryPlan, RuntimeStatePlan};
        use crate::ecs::execution_profile::PhysicalTileLayout;
        use crate::ecs::plan::CodecFamily;
        use std::collections::BTreeMap;

        let mut api = GenerationApi::new();
        let seg = PhysicalSegmentId("seg".into());
        api.store_payload(seg.clone(), vec![1, 2, 3]);

        let mut bindings = BTreeMap::new();
        bindings.insert(
            LogicalTensorId("t".into()),
            RepresentationBinding {
                representation_id: RepresentationId("r".into()),
                codec: CodecFamily::Nf4,
                layout: PhysicalTileLayout::default(),
                primary_segment: seg,
                scale_segments: vec![],
                residual_segments: vec![],
                source_representation: None,
                acceptance_receipt: ReceiptId("r".into()),
            },
        );

        let gen = CimageGeneration {
            generation_id: GenerationId("g1".into()),
            parent_generation: None,
            base_model: ModelSourceId("m".into()),
            compiler_identity: CompilerIdentity {
                name: "tc".into(),
                version: "1".into(),
                build_hash: None,
                build_timestamp: None,
            },
            hardware_profile: HardwareProfileId("h".into()),
            tensor_bindings: bindings,
            kernel_bindings: BTreeMap::new(),
            engram_bindings: BTreeMap::new(),
            execution_graph: ExecutionGraph {
                regions: vec![],
                edges: vec![],
                state: RuntimeStatePlan {
                    max_context_tokens: 4096,
                    kv_cache_bytes_per_token: 256,
                    total_kv_cache_bytes: 1048576,
                },
                memory: MemoryPlan {
                    total_activation_bytes: 0,
                    total_weight_bytes: 0,
                    arena_region_count: 0,
                },
            },
            receipt_root: ReceiptId("root".into()),
            created_at: Timestamp("t".into()),
        };

        api.promote(gen).unwrap();

        let summary = EvidenceSummary::collect(&api);
        assert_eq!(summary.current_generation, Some("g1".to_string()));
        assert_eq!(summary.total_segments, 1);
    }
}
