//! Generation management API — inspect, promote, rollback, list.
//!
//! Plan Section 15 Phase 13: "Server APIs, progress, generation management,
//! evidence dashboard."
//!
//! Provides an HTTP-accessible API surface for managing compiled generations,
//! including listing generations, promoting new generations atomically,
//! rolling back to a parent generation, and storing content-addressed payloads.

use prism_ecs_constitutional::canonical::generation::CimageGeneration;
use prism_ecs_constitutional::canonical::generation::EngramBinding;
use prism_ecs_constitutional::canonical::identity::*;
use prism_ecs_constitutional::canonical::provenance::{LifecycleReceiptBundle, PromotionRequest};
use crate::ecs::legacy_cimage::generation_store::{ContentStore, GenerationStore, PromotionTransaction};
use prism_ecs_ir::evolution::receipts::{NumericalReceipt, PerformanceReceipt};
use crate::ecs::training_target::engram::trainer::TrainedEngram;
use crate::ecs::training_target::spec::EngramArtifact;
use prism_ecs_ir::cimage_types::{
    EngramArtifactId as IrEngramArtifactId, EngramId as IrEngramId, RegionId as IrRegionId,
};
use sha2::{Digest, Sha256};

/// Generation management API — inspect, promote, rollback, list.
///
/// Wraps the `ContentStore` and `GenerationStore` with an ergonomic,
/// HTTP-accessible interface for lifecycle operations.
pub struct GenerationApi {
    pub content_store: ContentStore,
    pub generation_store: GenerationStore,
    /// Payload ids stored but whose generation hasn't been committed yet.
    /// Rolled back if promotion fails.
    pending_payloads: Vec<PhysicalSegmentId>,
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
            pending_payloads: Vec::new(),
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
        let result = tx.commit();
        if result.is_err() {
            // Roll back — the transaction abort removes stored payloads
            // from the content store and restores the parent generation
            // as current so it remains executable.
            tx.abort();
        }
        // Clear pending tracking regardless; on success the payloads are
        // part of a visible generation, on failure abort already removed them.
        self.pending_payloads.clear();
        result
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

    /// Rollback after promotion — requires that promotion has already
    /// happened (the current generation has a valid parent). Restores the
    /// parent as current without recompilation.
    ///
    /// Unlike `rollback()` which also returns Err on missing parent,
    /// this explicitly documents the promotion-first requirement and
    /// provides a distinct API contract for production lifecycle flows.
    pub fn rollback_after_promotion(&mut self) -> Result<GenerationId, String> {
        let current = self
            .generation_store
            .current()
            .ok_or_else(|| "no current generation — nothing to rollback from".to_string())?;
        let parent = current
            .parent_generation
            .clone()
            .ok_or_else(|| "generation has no parent — promotion did not occur".to_string())?;
        if !self.generation_store.contains(&parent) {
            return Err(format!(
                "parent generation {:?} not found — promotion chain is broken",
                parent
            ));
        }
        self.generation_store.set_current(parent.clone())?;
        Ok(parent)
    }

    /// Store a content-addressed payload by its canonical digest.
    pub fn store_payload(&mut self, id: PhysicalSegmentId, data: Vec<u8>) {
        self.content_store.store(id.clone(), data);
        self.pending_payloads.push(id);
    }

    /// Promote a generation from a `PromotionRequest` with a complete
    /// `LifecycleReceiptBundle`.
    ///
    /// 1. Verifies the receipt bundle is complete (no empty fields).
    /// 2. Verifies every payload digest in the request matches the actual
    ///    SHA-256 hash of the stored payload bytes.
    /// 3. Verifies every artifact digest is non-empty.
    /// 4. Verifies identity closure: all referenced segments and artifacts
    ///    resolve through the content store.
    /// 5. Atomically promotes the generation via `PromotionTransaction`,
    ///    preserving the parent on failure.
    pub fn promote_with_request(
        &mut self,
        request: PromotionRequest,
        receipt_bundle: LifecycleReceiptBundle,
    ) -> Result<GenerationId, String> {
        // 1. Verify receipt bundle is complete
        receipt_bundle.verify_complete()?;

        // 2. Verify every payload digest matches the stored bytes
        for (segment_id, expected_digest) in &request.payload_digests {
            let actual = self
                .content_store
                .compute_digest(segment_id)
                .ok_or_else(|| {
                    format!(
                        "payload {:?} not in content store for digest verification",
                        segment_id
                    )
                })?;
            if &actual != expected_digest {
                return Err(format!(
                    "payload digest mismatch for {:?}: expected {}, got {}",
                    segment_id, expected_digest, actual
                ));
            }
        }

        // 3. Verify artifact digests are non-empty
        for (semantic_id, expected_digest) in &request.artifact_digests {
            if expected_digest.is_empty() {
                return Err(format!("artifact digest for {:?} is empty", semantic_id));
            }
        }

        // 4. Identity closure — every referenced segment and artifact
        //    must resolve through the content store. The transaction's
        //    validate() covers tensor bindings and engram bindings.
        //    We also verify that every payload_digest segment already
        //    exists (already checked in step 2 via compute_digest).

        // 5. Atomically promote via transaction with parent preservation
        self.promote(request.generation)
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
            IrEngramId(artifact.logical_id.0.clone()),
            EngramBinding {
                engram_id: IrEngramId(artifact.logical_id.0.clone()),
                artifact_id: IrEngramArtifactId(artifact.artifact_id.0.clone()),
                enabled: true,
                insertion_region: IrRegionId(artifact.insertion_contract.region.0.clone()),
            },
        );
        self.promote(generation)
    }

    /// Promote the exact output of [`EngramTrainer::train_dataset`] after the
    /// evaluator has supplied correctness and performance evidence.
    pub fn promote_trained_engram(
        &mut self,
        generation: CimageGeneration,
        trained: &TrainedEngram,
        evidence: &PromotionEvidence,
    ) -> Result<GenerationId, String> {
        self.promote_with_engram(
            generation,
            &trained.artifact,
            trained.payload.clone(),
            evidence,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_constitutional::canonical::generation::RepresentationBinding;
    use prism_ecs_constitutional::canonical::{ExecutionGraph, MemoryPlan, RuntimeStatePlan};
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

        use prism_ecs_constitutional::canonical::identity::{EngramArtifactId, EngramId, RegionId, TensorShape};
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
        let mut base = gen.clone();
        base.generation_id = GenerationId("gen0".into());
        api.promote(base).expect("base generation should promote");
        let mut gen = gen;
        gen.parent_generation = Some(GenerationId("gen0".into()));
        let evidence = PromotionEvidence {
            numerical: NumericalReceipt {
                candidate_id: CandidateId("candidate".into()),
                passed: true,
                max_absolute_error: 0.0,
                max_relative_error: 0.0,
                threshold: 0.05,
                provenance: Vec::new(),
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
                provenance: Vec::new(),
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
        assert_eq!(
            api.rollback().expect("child generation should roll back"),
            GenerationId("gen0".into())
        );
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
    fn test_trained_engram_lifecycle() {
        use crate::ecs::training_target::engram::config::EngramTrainConfig;
        use crate::ecs::training_target::engram::dataset::EngramTrainingDataset;
        use crate::ecs::training_target::engram::trainer::EngramTrainer;
        use crate::ecs::training_target::spec::{
            EngramApplication, EngramTrainingTarget, TrainingTargetPriority,
        };

        let target = EngramTrainingTarget {
            target_id: "lifecycle.engram".into(),
            memory_kind: "semantic".into(),
            value_codec: CodecFamily::RawF32,
            lookup_policy: "always_apply".into(),
            residency: "cpu_resident".into(),
            priority: TrainingTargetPriority::Recommended,
        };
        let trainer = EngramTrainer::new(EngramTrainConfig {
            target: target.clone(),
            learning_rate: 0.5,
            max_iterations: 100,
            convergence_threshold: 1e-6,
            ..EngramTrainConfig::from_target(&target)
        });
        let dataset = EngramTrainingDataset {
            corpus_id: CorpusId("lifecycle-corpus".into()),
            train_examples: vec![vec![1.0], vec![2.0]],
            train_targets: vec![vec![1.25], vec![2.25]],
            validation_examples: vec![vec![3.0]],
            validation_targets: vec![vec![3.25]],
            holdout_examples: vec![vec![4.0]],
            holdout_targets: vec![vec![4.25]],
            interference_examples: vec![],
            activation_capture: None,
        };
        let trained = trainer.train_dataset(&dataset).expect("dataset training");
        let mut api = GenerationApi::new();
        let base = CimageGeneration {
            generation_id: GenerationId("base".into()),
            parent_generation: None,
            base_model: ModelSourceId("model".into()),
            compiler_identity: CompilerIdentity {
                name: "test".into(),
                version: "1".into(),
                build_hash: None,
                build_timestamp: None,
            },
            hardware_profile: HardwareProfileId("apple".into()),
            tensor_bindings: BTreeMap::new(),
            kernel_bindings: BTreeMap::new(),
            engram_bindings: BTreeMap::new(),
            execution_graph: ExecutionGraph {
                regions: vec![],
                edges: vec![],
                state: RuntimeStatePlan {
                    max_context_tokens: 1,
                    kv_cache_bytes_per_token: 1,
                    total_kv_cache_bytes: 1,
                },
                memory: MemoryPlan {
                    total_activation_bytes: 0,
                    total_weight_bytes: 0,
                    arena_region_count: 0,
                },
            },
            receipt_root: ReceiptId("base-receipt".into()),
            created_at: Timestamp("now".into()),
        };
        api.promote(base.clone()).expect("base promotion");
        let mut child = base;
        child.generation_id = GenerationId("trained".into());
        child.parent_generation = Some(GenerationId("base".into()));
        let evidence = PromotionEvidence {
            numerical: NumericalReceipt {
                candidate_id: CandidateId("nf4".into()),
                passed: true,
                max_absolute_error: 0.0,
                max_relative_error: 0.0,
                threshold: 0.05,
                provenance: Vec::new(),
            },
            performance: PerformanceReceipt {
                candidate_id: CandidateId("nf4".into()),
                latency_p50_ns: 10,
                latency_p95_ns: 12,
                encode_time_ns: 0,
                sync_time_ns: 12,
                memory_traffic_bytes: trained.payload.len() as u64,
                energy_uj: None,
                repetitions: 3,
                provenance: Vec::new(),
            },
        };
        api.promote_trained_engram(child, &trained, &evidence)
            .expect("trained promotion");
        let binding = api
            .current_generation()
            .unwrap()
            .engram_bindings
            .get(&trained.artifact.logical_id)
            .unwrap();
        assert_eq!(binding.artifact_id, trained.artifact.artifact_id);
        let payload = api
            .content_store
            .get(&trained.artifact.payload_segment)
            .unwrap();
        let mut activation = vec![1.0];
        crate::ecs::runtime::engram::application::apply_cpu(
            &EngramApplication::AdditiveResidual,
            &mut activation,
            payload,
        )
        .unwrap();
        assert!((activation[0] - 1.25).abs() < 1e-3);
        assert_eq!(api.rollback().unwrap(), GenerationId("base".into()));
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

    #[test]
    fn test_promote_with_request_receipt_bundle_validation() {
        let mut api = GenerationApi::new();

        // Set up a parent generation
        let seg = PhysicalSegmentId("seg".into());
        api.store_payload(seg.clone(), vec![1, 2, 3, 4]);
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
        let parent_gen = CimageGeneration {
            generation_id: GenerationId("parent".into()),
            parent_generation: None,
            base_model: ModelSourceId("m".into()),
            compiler_identity: CompilerIdentity {
                name: "tc".into(),
                version: "1".into(),
                build_hash: None,
                build_timestamp: None,
            },
            hardware_profile: HardwareProfileId("h".into()),
            tensor_bindings: bindings.clone(),
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
            receipt_root: ReceiptId("r".into()),
            created_at: Timestamp("t".into()),
        };
        api.promote(parent_gen).expect("parent should promote");

        // An empty receipt bundle should fail verify_complete
        let empty_bundle = LifecycleReceiptBundle {
            compiler_receipt: ReceiptId(String::new()),
            numerical_receipt: ReceiptId(String::new()),
            quality_receipt: ReceiptId(String::new()),
            performance_receipt: ReceiptId(String::new()),
            policy_receipt: ReceiptId(String::new()),
            promotion_receipt: ReceiptId(String::new()),
            generation_id: GenerationId(String::new()),
            sealed_at: String::new(),
        };
        assert!(
            empty_bundle.verify_complete().is_err(),
            "empty receipt bundle must fail verify_complete"
        );
        let child_gen = CimageGeneration {
            generation_id: GenerationId("child".into()),
            parent_generation: Some(GenerationId("parent".into())),
            base_model: ModelSourceId("m".into()),
            compiler_identity: CompilerIdentity {
                name: "tc".into(),
                version: "2".into(),
                build_hash: None,
                build_timestamp: None,
            },
            hardware_profile: HardwareProfileId("h".into()),
            tensor_bindings: bindings.clone(),
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
            receipt_root: ReceiptId("r2".into()),
            created_at: Timestamp("t2".into()),
        };
        let request = PromotionRequest {
            parent_generation: GenerationId("parent".into()),
            payload_digests: vec![],
            artifact_digests: vec![],
            policy_id: "policy".into(),
            receipt_bundle_id: ReceiptId("rb".into()),
            generation: child_gen,
        };
        let result = api.promote_with_request(request, empty_bundle);
        assert!(
            result.is_err(),
            "promote_with_request with empty receipt bundle should fail"
        );
        let err_msg = result.unwrap_err().to_lowercase();
        assert!(
            err_msg.contains("receipt") && err_msg.contains("empty"),
            "error should mention empty receipt, got: {err_msg}"
        );
    }

    #[test]
    fn test_promote_with_request_payload_digest_validation() {
        let mut api = GenerationApi::new();

        // Store a payload
        let payload = vec![1, 2, 3, 4];
        let seg_id = PhysicalSegmentId("seg1".into());
        api.store_payload(seg_id.clone(), payload);

        // Store the parent generation
        let mut bindings = BTreeMap::new();
        bindings.insert(
            LogicalTensorId("t".into()),
            RepresentationBinding {
                representation_id: RepresentationId("r".into()),
                codec: CodecFamily::Nf4,
                layout: PhysicalTileLayout::default(),
                primary_segment: seg_id.clone(),
                scale_segments: vec![],
                residual_segments: vec![],
                source_representation: None,
                acceptance_receipt: ReceiptId("r".into()),
            },
        );
        let parent = CimageGeneration {
            generation_id: GenerationId("parent".into()),
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
            receipt_root: ReceiptId("r".into()),
            created_at: Timestamp("t".into()),
        };
        api.promote(parent).expect("parent should promote");

        // Correct digest
        let correct_digest = Sha256::digest(&[1, 2, 3, 4])
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();
        // Wrong digest
        let wrong_digest = Sha256::digest(&[9, 9, 9])
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>();

        // Create a valid receipt bundle
        let bundle = LifecycleReceiptBundle {
            compiler_receipt: ReceiptId("c".into()),
            numerical_receipt: ReceiptId("n".into()),
            quality_receipt: ReceiptId("q".into()),
            performance_receipt: ReceiptId("p".into()),
            policy_receipt: ReceiptId("pol".into()),
            promotion_receipt: ReceiptId("pro".into()),
            generation_id: GenerationId("child".into()),
            sealed_at: "now".into(),
        };

        let mut child_bindings = BTreeMap::new();
        child_bindings.insert(
            LogicalTensorId("t".into()),
            RepresentationBinding {
                representation_id: RepresentationId("r2".into()),
                codec: CodecFamily::Int8,
                layout: PhysicalTileLayout::default(),
                primary_segment: seg_id.clone(),
                scale_segments: vec![],
                residual_segments: vec![],
                source_representation: Some(RepresentationId("r".into())),
                acceptance_receipt: ReceiptId("r2".into()),
            },
        );
        let child = CimageGeneration {
            generation_id: GenerationId("child".into()),
            parent_generation: Some(GenerationId("parent".into())),
            base_model: ModelSourceId("m".into()),
            compiler_identity: CompilerIdentity {
                name: "tc".into(),
                version: "2".into(),
                build_hash: None,
                build_timestamp: None,
            },
            hardware_profile: HardwareProfileId("h".into()),
            tensor_bindings: child_bindings,
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
            receipt_root: ReceiptId("r2".into()),
            created_at: Timestamp("t2".into()),
        };

        // Request with wrong digest should fail
        let bad_request = PromotionRequest {
            parent_generation: GenerationId("parent".into()),
            payload_digests: vec![(seg_id.clone(), wrong_digest)],
            artifact_digests: vec![],
            policy_id: "policy".into(),
            receipt_bundle_id: ReceiptId("rb".into()),
            generation: child.clone(),
        };
        let result = api.promote_with_request(bad_request, bundle.clone());
        assert!(result.is_err(), "wrong payload digest should be rejected");
        assert!(
            result.unwrap_err().contains("digest mismatch"),
            "error should mention digest mismatch"
        );

        // Request with correct digest should succeed
        let good_request = PromotionRequest {
            parent_generation: GenerationId("parent".into()),
            payload_digests: vec![(seg_id.clone(), correct_digest)],
            artifact_digests: vec![],
            policy_id: "policy".into(),
            receipt_bundle_id: ReceiptId("rb".into()),
            generation: child,
        };
        let result = api.promote_with_request(good_request, bundle);
        assert!(result.is_ok(), "correct payload digest should be accepted");
    }

    #[test]
    fn test_promote_with_request_artifact_digest_validation() {
        let mut api = GenerationApi::new();

        // Set up minimal state with a parent
        let seg = PhysicalSegmentId("seg".into());
        api.store_payload(seg.clone(), vec![1, 2, 3]);
        let mut bindings = BTreeMap::new();
        bindings.insert(
            LogicalTensorId("t".into()),
            RepresentationBinding {
                representation_id: RepresentationId("r".into()),
                codec: CodecFamily::Nf4,
                layout: PhysicalTileLayout::default(),
                primary_segment: seg.clone(),
                scale_segments: vec![],
                residual_segments: vec![],
                source_representation: None,
                acceptance_receipt: ReceiptId("r".into()),
            },
        );
        let parent = CimageGeneration {
            generation_id: GenerationId("parent".into()),
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
            receipt_root: ReceiptId("r".into()),
            created_at: Timestamp("t".into()),
        };
        api.promote(parent).expect("parent should promote");

        let bundle = LifecycleReceiptBundle {
            compiler_receipt: ReceiptId("c".into()),
            numerical_receipt: ReceiptId("n".into()),
            quality_receipt: ReceiptId("q".into()),
            performance_receipt: ReceiptId("p".into()),
            policy_receipt: ReceiptId("pol".into()),
            promotion_receipt: ReceiptId("pro".into()),
            generation_id: GenerationId("child".into()),
            sealed_at: "now".into(),
        };

        // Empty artifact digest should be rejected
        let mut child_bindings = BTreeMap::new();
        child_bindings.insert(
            LogicalTensorId("t".into()),
            RepresentationBinding {
                representation_id: RepresentationId("r2".into()),
                codec: CodecFamily::Int8,
                layout: PhysicalTileLayout::default(),
                primary_segment: seg,
                scale_segments: vec![],
                residual_segments: vec![],
                source_representation: Some(RepresentationId("r".into())),
                acceptance_receipt: ReceiptId("r2".into()),
            },
        );
        let child = CimageGeneration {
            generation_id: GenerationId("child".into()),
            parent_generation: Some(GenerationId("parent".into())),
            base_model: ModelSourceId("m".into()),
            compiler_identity: CompilerIdentity {
                name: "tc".into(),
                version: "2".into(),
                build_hash: None,
                build_timestamp: None,
            },
            hardware_profile: HardwareProfileId("h".into()),
            tensor_bindings: child_bindings,
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
            receipt_root: ReceiptId("r2".into()),
            created_at: Timestamp("t2".into()),
        };

        let bad_artifact_request = PromotionRequest {
            parent_generation: GenerationId("parent".into()),
            payload_digests: vec![],
            artifact_digests: vec![(
                crate::ecs::canonical::kernel_abi::KernelSemanticId("k1".into()),
                String::new(),
            )],
            policy_id: "policy".into(),
            receipt_bundle_id: ReceiptId("rb".into()),
            generation: child,
        };
        let result = api.promote_with_request(bad_artifact_request, bundle);
        assert!(result.is_err(), "empty artifact digest should be rejected");
    }

    #[test]
    fn test_failed_promotion_preserves_parent() {
        let mut api = GenerationApi::new();

        // Store payload and promote a parent generation
        let seg = PhysicalSegmentId("seg1".into());
        api.store_payload(seg.clone(), vec![1, 2, 3, 4]);

        let mut bindings = BTreeMap::new();
        bindings.insert(
            LogicalTensorId("t".into()),
            RepresentationBinding {
                representation_id: RepresentationId("r1".into()),
                codec: CodecFamily::Nf4,
                layout: PhysicalTileLayout::default(),
                primary_segment: seg.clone(),
                scale_segments: vec![],
                residual_segments: vec![],
                source_representation: None,
                acceptance_receipt: ReceiptId("r1".into()),
            },
        );

        let parent_gen = CimageGeneration {
            generation_id: GenerationId("parent".into()),
            parent_generation: None,
            base_model: ModelSourceId("m".into()),
            compiler_identity: CompilerIdentity {
                name: "tc".into(),
                version: "1".into(),
                build_hash: None,
                build_timestamp: None,
            },
            hardware_profile: HardwareProfileId("h".into()),
            tensor_bindings: bindings.clone(),
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
            receipt_root: ReceiptId("root1".into()),
            created_at: Timestamp("t1".into()),
        };
        api.promote(parent_gen).expect("parent should promote");

        // Store payload for a child that will fail — use a missing segment
        let child_payload = vec![5, 6, 7, 8];
        let child_seg = PhysicalSegmentId("seg_child".into());
        api.store_payload(child_seg.clone(), child_payload);

        let mut child_bindings = BTreeMap::new();
        // Reference a SEGMENT THAT DOES NOT EXIST to trigger transaction failure
        child_bindings.insert(
            LogicalTensorId("t2".into()),
            RepresentationBinding {
                representation_id: RepresentationId("r2".into()),
                codec: CodecFamily::Int8,
                layout: PhysicalTileLayout::default(),
                primary_segment: PhysicalSegmentId("MISSING_SEGMENT".into()),
                scale_segments: vec![],
                residual_segments: vec![],
                source_representation: Some(RepresentationId("r1".into())),
                acceptance_receipt: ReceiptId("r2".into()),
            },
        );

        let child_gen = CimageGeneration {
            generation_id: GenerationId("child".into()),
            parent_generation: Some(GenerationId("parent".into())),
            base_model: ModelSourceId("m".into()),
            compiler_identity: CompilerIdentity {
                name: "tc".into(),
                version: "2".into(),
                build_hash: None,
                build_timestamp: None,
            },
            hardware_profile: HardwareProfileId("h".into()),
            tensor_bindings: child_bindings,
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
            receipt_root: ReceiptId("root2".into()),
            created_at: Timestamp("t2".into()),
        };

        // Attempt promotion — should fail because MISSING_SEGMENT doesn't exist
        let result = api.promote(child_gen);
        assert!(
            result.is_err(),
            "child promotion with missing segment should fail"
        );

        // Verify parent is still current and executable
        assert_eq!(
            api.generation_store.current_id(),
            Some(&GenerationId("parent".into())),
            "parent should remain current after failed child promotion"
        );
        let current = api.current_generation().expect("parent should be current");
        assert_eq!(
            current.generation_id,
            GenerationId("parent".into()),
            "current generation should be parent"
        );
    }

    #[test]
    fn test_identity_closure_rejects_incomplete() {
        let mut api = GenerationApi::new();

        // Store payload and promote parent
        let seg = PhysicalSegmentId("seg".into());
        api.store_payload(seg.clone(), vec![1, 2, 3]);

        let mut parent_bindings = BTreeMap::new();
        parent_bindings.insert(
            LogicalTensorId("t".into()),
            RepresentationBinding {
                representation_id: RepresentationId("r".into()),
                codec: CodecFamily::Nf4,
                layout: PhysicalTileLayout::default(),
                primary_segment: seg.clone(),
                scale_segments: vec![],
                residual_segments: vec![],
                source_representation: None,
                acceptance_receipt: ReceiptId("r".into()),
            },
        );

        let parent_gen = CimageGeneration {
            generation_id: GenerationId("parent".into()),
            parent_generation: None,
            base_model: ModelSourceId("m".into()),
            compiler_identity: CompilerIdentity {
                name: "tc".into(),
                version: "1".into(),
                build_hash: None,
                build_timestamp: None,
            },
            hardware_profile: HardwareProfileId("h".into()),
            tensor_bindings: parent_bindings,
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
            receipt_root: ReceiptId("r".into()),
            created_at: Timestamp("t".into()),
        };
        api.promote(parent_gen).expect("parent should promote");

        // Build a child generation that references a payload segment
        // that does NOT exist in the content store. The validate()
        // step in the transaction should catch this.
        let mut child_bindings = BTreeMap::new();
        child_bindings.insert(
            LogicalTensorId("t".into()),
            RepresentationBinding {
                representation_id: RepresentationId("r2".into()),
                codec: CodecFamily::Int8,
                layout: PhysicalTileLayout::default(),
                primary_segment: PhysicalSegmentId("nonexistent_segment".into()),
                scale_segments: vec![],
                residual_segments: vec![],
                source_representation: Some(RepresentationId("r".into())),
                acceptance_receipt: ReceiptId("r2".into()),
            },
        );

        let child_gen = CimageGeneration {
            generation_id: GenerationId("child".into()),
            parent_generation: Some(GenerationId("parent".into())),
            base_model: ModelSourceId("m".into()),
            compiler_identity: CompilerIdentity {
                name: "tc".into(),
                version: "2".into(),
                build_hash: None,
                build_timestamp: None,
            },
            hardware_profile: HardwareProfileId("h".into()),
            tensor_bindings: child_bindings,
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
            receipt_root: ReceiptId("r2".into()),
            created_at: Timestamp("t2".into()),
        };

        // Promotion must fail because the primary segment is missing
        // from the content store (fails identity closure)
        let result = api.promote(child_gen);
        assert!(
            result.is_err(),
            "promotion should fail for generation with unresolved segment"
        );

        // Parent must still be current
        assert_eq!(
            api.generation_store.current_id(),
            Some(&GenerationId("parent".into())),
            "parent should remain current after rejection"
        );
    }
}
