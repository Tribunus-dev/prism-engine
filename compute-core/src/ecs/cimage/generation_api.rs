//! Generation management API — inspect, promote, rollback, list.
//!
//! Plan Section 15 Phase 13: "Server APIs, progress, generation management,
//! evidence dashboard."
//!
//! Provides an HTTP-accessible API surface for managing compiled generations,
//! including listing generations, promoting new generations atomically,
//! rolling back to a parent generation, and storing content-addressed payloads.

use crate::ecs::canonical::generation::CimageGeneration;
use crate::ecs::canonical::generation::EngramBinding;
use crate::ecs::canonical::identity::*;
use crate::ecs::cimage::generation_store::{ContentStore, GenerationStore, PromotionTransaction};
use crate::ecs::training_target::spec::EngramArtifact;
use crate::ecs::evolution::evaluator::{NumericalReceipt, PerformanceReceipt};
use sha2::{Digest, Sha256};

/// Generation management API — inspect, promote, rollback, list.
///
/// Wraps the `ContentStore` and `GenerationStore` with an ergonomic,
/// HTTP-accessible interface for lifecycle operations.
pub struct GenerationApi {
    pub content_store: ContentStore,
    pub generation_store: GenerationStore,
}

/// Evidence required before a trained payload can become executable state.
#[derive(Debug, Clone)]
pub struct PromotionEvidence {
    pub numerical: NumericalReceipt,
    pub performance: PerformanceReceipt,
}

impl GenerationApi {
    /// Create a new empty API instance.
    pub fn new() -> Self {
        Self {
            content_store: ContentStore::new(),
            generation_store: GenerationStore::new(),
        }
    }

    /// List all generation IDs currently known to the store.
    ///
    pub fn list_generations(&self) -> Vec<GenerationId> {
        self.generation_store.ids()
    }

    /// Get the current (promoted) generation, if any.
    pub fn current_generation(&self) -> Option<&CimageGeneration> {
        self.generation_store.current()
    }

    /// Atomically promote a new generation.
    ///
    /// Validates that all referenced payloads exist and the parent
    /// generation is known, then commits the generation as current.
    pub fn promote(&mut self, generation: CimageGeneration) -> Result<GenerationId, String> {
        let mut tx = PromotionTransaction::new(
            &mut self.content_store,
            &mut self.generation_store,
            generation,
        );
        tx.commit()
    }

    /// Rollback to the parent generation.
    ///
    /// Fails if there is no current generation or no parent.
    pub fn rollback(&mut self) -> Result<GenerationId, String> {
        let current = self
            .generation_store
            .current()
            .ok_or_else(|| "no current generation".to_string())?;
        let parent = current
            .parent_generation
            .clone()
            .ok_or_else(|| "no parent generation".to_string())?;
        self.generation_store.set_current(parent.clone())?;
        Ok(parent)
    }

    /// Store a content-addressed payload by its canonical digest.
    pub fn store_payload(&mut self, id: PhysicalSegmentId, data: Vec<u8>) {
        self.content_store.store(id, data);
    }

    /// Store a trained engram payload and atomically promote the generation
    /// that references it. The artifact digest must be the SHA-256 digest of
    /// the exact payload bytes.
    pub fn promote_with_engram(
        &mut self,
        mut generation: CimageGeneration,
        artifact: &EngramArtifact,
        payload: Vec<u8>,
        evidence: &PromotionEvidence,
    ) -> Result<GenerationId, String> {
        if !evidence.numerical.passed {
            return Err("engram promotion rejected by numerical gate".into());
        }
        if evidence.performance.repetitions == 0
            || evidence.performance.latency_p50_ns == 0
            || evidence.performance.latency_p95_ns < evidence.performance.latency_p50_ns
        {
            return Err("engram promotion rejected by performance gate".into());
        }
        if let Some(limit) = artifact.insertion_contract.maximum_latency_ns {
            if evidence.performance.latency_p95_ns > limit {
                return Err("engram promotion exceeds insertion latency budget".into());
            }
        }
        let digest = Sha256::digest(&payload);
        let digest = digest
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        if digest != artifact.artifact_id.0 || digest != artifact.payload_segment.0 {
            return Err("engram payload digest does not match artifact identity".into());
        }
        self.store_payload(artifact.payload_segment.clone(), payload);
        generation.engram_bindings.insert(
            artifact.logical_id.clone(),
            EngramBinding {
                engram_id: artifact.logical_id.clone(),
                artifact_id: artifact.artifact_id.clone(),
                enabled: true,
                insertion_region: artifact.insertion_contract.region.clone(),
            },
        );
        self.promote(generation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::canonical::generation::RepresentationBinding;
    use crate::ecs::canonical::{ExecutionGraph, MemoryPlan, RuntimeStatePlan};
    use crate::ecs::execution_profile::PhysicalTileLayout;
    use crate::ecs::plan::CodecFamily;
    use std::collections::BTreeMap;

    #[test]
    fn test_generation_api_promote() {
        let mut api = GenerationApi::new();

        // Store a payload
        let seg_id = PhysicalSegmentId("seg1".into());
        api.store_payload(seg_id.clone(), vec![1, 2, 3, 4]);

        // Create a minimal generation
        let mut bindings = BTreeMap::new();
        bindings.insert(
            LogicalTensorId("t1".into()),
            RepresentationBinding {
                representation_id: RepresentationId("r1".into()),
                codec: CodecFamily::Nf4,
                layout: PhysicalTileLayout::default(),
                primary_segment: seg_id,
                scale_segments: vec![],
                residual_segments: vec![],
                source_representation: None,
                acceptance_receipt: ReceiptId("r".into()),
            },
        );

        let gen = CimageGeneration {
            generation_id: GenerationId("gen1".into()),
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

        use crate::ecs::canonical::identity::{EngramArtifactId, EngramId, RegionId, TensorShape};
        use crate::ecs::training_target::spec::{
            EngramApplication, EngramArtifact, EngramCodec, EngramInsertionContract,
            EngramMemoryKind, EngramOperation, EngramParameterSchema, EngramRoutingPolicy,
            PrivacyContract,
        };
        let payload = vec![1, 2, 3, 4];
        let digest = Sha256::digest(&payload)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        let artifact = EngramArtifact {
            artifact_id: EngramArtifactId(digest.clone()),
            logical_id: EngramId("e1".into()),
            format_version: 1,
            memory_kind: EngramMemoryKind::Semantic,
            codec: EngramCodec::F32,
            insertion_contract: EngramInsertionContract {
                region: RegionId("region".into()),
                operation: EngramOperation::Adapter,
                input_shape: TensorShape { dims: vec![1] },
                output_shape: TensorShape { dims: vec![1] },
                application: EngramApplication::AdditiveResidual,
                routing: EngramRoutingPolicy::AlwaysOn,
                maximum_latency_ns: None,
            },
            index_segment: None,
            payload_segment: PhysicalSegmentId(digest.clone()),
            routing_segment: None,
            parameter_schema: EngramParameterSchema {
                parameter_count: 1,
                bytes_per_parameter: 4,
                layout: "dense".into(),
            },
            training_corpus: CorpusId("corpus".into()),
            training_receipt: ReceiptId("receipt".into()),
            privacy_contract: PrivacyContract {
                purpose: "test".into(),
                retention: "test".into(),
                disclosure_class: "internal".into(),
                assimilation_permitted: false,
            },
        };
        let evidence = PromotionEvidence {
            numerical: NumericalReceipt {
                candidate_id: CandidateId("candidate".into()),
                passed: true,
                max_absolute_error: 0.0,
                max_relative_error: 0.0,
                threshold: 0.05,
            },
            performance: PerformanceReceipt {
                candidate_id: CandidateId("candidate".into()),
                latency_p50_ns: 10,
                latency_p95_ns: 12,
                encode_time_ns: 0,
                sync_time_ns: 12,
                memory_traffic_bytes: 4,
                energy_uj: None,
                repetitions: 3,
            },
        };
        let result = api.promote_with_engram(gen, &artifact, payload, &evidence);
        assert!(result.is_ok());
        assert_eq!(
            api.generation_store
                .current()
                .unwrap()
                .engram_bindings
                .len(),
            1
        );
        let stored = api
            .content_store
            .get(&artifact.payload_segment)
            .expect("promoted payload should be retrievable");
        let mut activation = vec![0.5f32];
        crate::ecs::runtime::engram::application::apply_cpu(
            &artifact.insertion_contract.application,
            &mut activation,
            stored,
        )
        .expect("promoted payload should apply");
        assert!((activation[0] - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_generation_api_rollback() {
        let mut api = GenerationApi::new();

        // Store payloads and create two generations
        let seg1 = PhysicalSegmentId("seg1".into());
        let seg2 = PhysicalSegmentId("seg2".into());
        api.store_payload(seg1.clone(), vec![1, 2, 3, 4]);
        api.store_payload(seg2.clone(), vec![5, 6, 7, 8]);

        fn make_binding(seg: PhysicalSegmentId) -> RepresentationBinding {
            RepresentationBinding {
                representation_id: RepresentationId("r".into()),
                codec: CodecFamily::Nf4,
                layout: PhysicalTileLayout::default(),
                primary_segment: seg,
                scale_segments: vec![],
                residual_segments: vec![],
                source_representation: None,
                acceptance_receipt: ReceiptId("r".into()),
            }
        }

        fn make_generation(
            id: &str,
            parent: Option<GenerationId>,
            bindings: BTreeMap<LogicalTensorId, RepresentationBinding>,
        ) -> CimageGeneration {
            CimageGeneration {
                generation_id: GenerationId(id.into()),
                parent_generation: parent,
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
            }
        }

        let mut b1 = BTreeMap::new();
        b1.insert(LogicalTensorId("t1".into()), make_binding(seg1));
        let gen1 = make_generation("gen1", None, b1);

        let mut b2 = BTreeMap::new();
        b2.insert(LogicalTensorId("t1".into()), make_binding(seg2));
        let gen2 = make_generation("gen2", Some(GenerationId("gen1".into())), b2);

        // Promote gen1, then gen2
        api.promote(gen1).unwrap();
        api.promote(gen2).unwrap();
        assert_eq!(
            api.generation_store.current_id(),
            Some(&GenerationId("gen2".into()))
        );

        // Rollback to gen1
        let rolled = api.rollback().unwrap();
        assert_eq!(rolled, GenerationId("gen1".into()));
        assert_eq!(
            api.generation_store.current_id(),
            Some(&GenerationId("gen1".into()))
        );
    }

    #[test]
    fn test_generation_api_rollback_fails_no_parent() {
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
            generation_id: GenerationId("g".into()),
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
        // Only one generation with no parent — should fail
        assert!(api.rollback().is_err());
    }

    #[test]
    fn test_generation_api_list_empty() {
        let api = GenerationApi::new();
        let list = api.list_generations();
        assert!(list.is_empty());
    }

    #[test]
    fn test_generation_api_current_none() {
        let api = GenerationApi::new();
        assert!(api.current_generation().is_none());
    }
}
