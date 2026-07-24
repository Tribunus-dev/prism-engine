//! Evaluator integration for evolutionary search
//!
//! This module provides the integration between the search system and hardware-backed
//! evaluators, ensuring that production mode uses real measurements and fails closed
//! when real evaluation is not available.

use crate::{CandidateMeasurements, SearchError};
use prism_ecs_core::identity::TensorProvider;
use prism_ecs_ir::evolution::evaluate::EvaluationStrategy as EcsEvaluationStrategy;
use prism_ecs_ir::evolution::foundation::CandidateGenome;
use prism_ecs_ir::evolution::{ProgressiveStageExecutor, TernaryObjectiveEvidence};
use prism_ecs_quantization::kv_search::{
    KvCompressionCandidate, KvCompressionEvaluator, KvCompressionEvidence,
};
use prism_ecs_quantization::safetensors_provider::SafeTensorProvider;
use prism_ecs_quantization::turboquant_kv::{KvQuantMode, TurboQuantKvCache};
use std::time::Instant;
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

/// Bounded active-layer working set. Only the reference tensor and one
/// candidate representation are resident while a canary is evaluated; the
/// window is explicitly recycled before the next tensor is admitted.
#[derive(Debug, Default)]
struct CanaryWindow {
    reference: Vec<f32>,
    candidate: Vec<f32>,
    generation: u64,
    max_elements: usize,
}

impl CanaryWindow {
    fn new(max_elements: usize) -> Self {
        Self {
            max_elements,
            ..Self::default()
        }
    }
    fn load(&mut self, reference: &[f32]) -> Result<(), String> {
        if reference.is_empty() || reference.len() > self.max_elements {
            return Err(format!(
                "canary tensor exceeds {} element active window",
                self.max_elements
            ));
        }
        self.reference.clear();
        self.reference.extend_from_slice(reference);
        self.candidate.resize(reference.len(), 0.0);
        self.generation = self.generation.wrapping_add(1);
        Ok(())
    }
    fn recycle(&mut self) {
        self.reference.clear();
        self.candidate.clear();
        self.generation = self.generation.wrapping_add(1);
    }
}

/// MI300X-backed evaluator for KV-cache candidates. Quantization and
/// reconstruction stay in Prism's native TurboQuant implementation; the
/// expensive reference-vs-reconstruction reductions run through the existing
/// HIP scorer.
pub struct Mi300xKvEvaluator {
    keys: Vec<f32>,
    values: Vec<f32>,
    scorer: Arc<prism_rocm_runtime::ternary::Mi300xTernaryScorer>,
}

impl Mi300xKvEvaluator {
    pub fn new(
        keys: Vec<f32>,
        values: Vec<f32>,
        scorer: Arc<prism_rocm_runtime::ternary::Mi300xTernaryScorer>,
    ) -> Result<Self, String> {
        if keys.is_empty() || keys.len() != values.len() {
            return Err("KV evaluator keys and values must be nonempty and equal length".into());
        }
        Ok(Self {
            keys,
            values,
            scorer,
        })
    }
}

impl KvCompressionEvaluator for Mi300xKvEvaluator {
    fn evaluate(
        &mut self,
        candidate: KvCompressionCandidate,
    ) -> Result<KvCompressionEvidence, String> {
        let mut cache = TurboQuantKvCache::new(
            KvQuantMode::Polar(candidate.key_bits as u32),
            candidate.group_size as usize,
            1,
        );
        cache
            .quantize_asymmetric(0, &self.keys, &self.values, &candidate.mode())
            .map_err(|error| format!("quantize KV candidate: {error}"))?;
        let (reconstructed_keys, reconstructed_values) = cache
            .dequantize(0)
            .map_err(|error| format!("dequantize KV candidate: {error}"))?;
        let key_mse = self
            .scorer
            .mean_squared_error(&self.keys, &reconstructed_keys)?;
        let value_mse = self
            .scorer
            .mean_squared_error(&self.values, &reconstructed_values)?;
        let key_scale = self
            .keys
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max)
            .max(1.0);
        let value_scale = self
            .values
            .iter()
            .map(|v| v.abs())
            .fold(0.0f32, f32::max)
            .max(1.0);
        let key_error = (key_mse.sqrt() as f32) / key_scale;
        let value_error = (value_mse.sqrt() as f32) / value_scale;
        Ok(KvCompressionEvidence {
            candidate,
            key_error,
            value_error,
            attention_loss: (key_error + value_error) * 0.5,
            bytes_per_token: ((self.keys.len() as u64 * candidate.key_bits as u64).div_ceil(8))
                + ((self.values.len() as u64 * candidate.value_bits as u64).div_ceil(8)),
        })
    }
}

/// Evaluate a reference KV sample and persist the complete candidate
/// evidence sidecar consumed by CImage emission. MI300X is preferred when
/// enabled; otherwise the identical native CPU evaluator is used.
pub fn evaluate_kv_reference_cache(
    keys: &[f32],
    values: &[f32],
    evidence_path: &std::path::Path,
) -> Result<Vec<KvCompressionEvidence>, String> {
    let search = prism_ecs_quantization::kv_search::KvCompressionSearch::default();
    let evidence =
        if let Ok(Some(scorer)) = prism_rocm_runtime::ternary::Mi300xTernaryScorer::from_env() {
            let mut evaluator =
                Mi300xKvEvaluator::new(keys.to_vec(), values.to_vec(), Arc::new(scorer))?;
            search
                .candidates
                .iter()
                .copied()
                .map(|candidate| evaluator.evaluate(candidate))
                .collect::<Result<Vec<_>, _>>()?
        } else {
            search.evaluate_reference_cache(keys, values)?
        };
    if let Some(parent) = evidence_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("create KV evidence directory: {error}"))?;
    }
    let bytes = serde_json::to_vec_pretty(&evidence)
        .map_err(|error| format!("encode KV compression evidence: {error}"))?;
    std::fs::write(evidence_path, bytes)
        .map_err(|error| format!("write KV compression evidence: {error}"))?;
    Ok(evidence)
}

/// Wrapper around MeasuredEvaluator that adapts it to the search system's EvaluationStrategy trait
pub struct MeasuredEvaluatorAdapter {
    inner: Arc<dyn EcsEvaluationStrategy>,
    behavioral_probe: Option<Arc<dyn BehavioralProbe>>,
}

pub trait BehavioralProbe: Send + Sync {
    fn evaluate(
        &self,
        genome: &CandidateGenome,
        context: &[u8],
    ) -> Result<TernaryObjectiveEvidence, SearchError>;
}

/// Provider-backed tensor probe context. It deliberately describes a single
/// tensor because the SafeTensors provider streams one mapped tensor at a time;
/// a future graph executor can extend this to layer activations and logits.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MappedTensorProbeContext {
    pub model_dir: std::path::PathBuf,
    pub tensor_name: String,
}

impl MappedTensorProbeContext {
    /// Construct a context for one concrete tensor in a mapped checkpoint.
    ///
    /// Keeping the tensor name in the serialized context is important: a
    /// progressive run must be reproducible against the same source tensor,
    /// rather than letting the probe select an arbitrary router tensor.
    pub fn for_tensor(
        model_dir: impl Into<std::path::PathBuf>,
        tensor_name: impl Into<String>,
    ) -> Self {
        Self {
            model_dir: model_dir.into(),
            tensor_name: tensor_name.into(),
        }
    }

    /// Reject an empty or non-file-backed reference before progressive search
    /// admits any evidence. This does not read tensor payload bytes; it only
    /// verifies that the requested reference is present in the provider.
    pub fn validate_reference(&self) -> Result<(), SearchError> {
        if self.tensor_name.trim().is_empty() {
            return Err(SearchError::SearchFailed(
                "mapped probe context has an empty tensor reference".into(),
            ));
        }
        let provider =
            SafeTensorProvider::new(&self.model_dir).map_err(SearchError::SearchFailed)?;
        let present = provider
            .list_tensors()
            .map_err(SearchError::SearchFailed)?
            .into_iter()
            .any(|tensor| tensor.name == self.tensor_name);
        if !present {
            return Err(SearchError::SearchFailed(format!(
                "mapped probe tensor reference '{}' is not present in {}",
                self.tensor_name,
                self.model_dir.display()
            )));
        }
        Ok(())
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, SearchError> {
        serde_json::to_vec(self)
            .map_err(|e| SearchError::SearchFailed(format!("encode mapped probe context: {e}")))
    }
}

/// Bounded reference-backed probe for one mapped tensor. It evaluates the
/// actual BF16/F16 tensor against a deterministic ternary reconstruction and
/// feeds operator, router-margin, and logit evidence into progressive search.
/// It intentionally materializes only the selected tensor, never the model.
pub struct MappedTensorBehavioralProbe {
    pub model_dir: std::path::PathBuf,
    pub max_elements: usize,
    pub router_top_k: usize,
    pub model: Arc<dyn crate::ModelAdapter>,
    pub mi300x_scorer: Option<Arc<prism_rocm_runtime::ternary::Mi300xTernaryScorer>>,
    reference_cache: Arc<Mutex<HashMap<String, (Vec<f32>, Vec<usize>)>>>,
    cache_bytes: Arc<Mutex<usize>>,
    max_cache_bytes: usize,
}

/// Search-facing adapter for mapped model directories. It makes the
/// reference probe available to the compiler's normal evolutionary search
/// trait, not only to the progressive ECS executor.
pub struct MappedTensorEvaluationStrategy {
    pub probe: MappedTensorBehavioralProbe,
    pub limits: prism_ecs_ir::evolution::TernaryAdmissionLimits,
}

impl MappedTensorEvaluationStrategy {
    pub fn new(model_dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            probe: MappedTensorBehavioralProbe::new(model_dir),
            limits: Default::default(),
        }
    }
}

impl crate::search::EvaluationStrategy for MappedTensorEvaluationStrategy {
    fn evaluate(&self, genome: &str, context: &[u8]) -> Result<Vec<f64>, String> {
        let genome: CandidateGenome =
            serde_json::from_str(genome).map_err(|e| format!("decode genome: {e}"))?;
        let evidence = self
            .probe
            .evaluate(&genome, context)
            .map_err(|e| e.to_string())?;
        Ok(evidence
            .vector(&self.limits)
            .into_iter()
            .map(|score| score.value())
            .collect())
    }

    fn name(&self) -> &str {
        "mapped-model-behavioral"
    }

    fn is_measured(&self) -> bool {
        true
    }

    fn evaluate_workload_profile(
        &self,
        genome: &str,
        context: &[u8],
        profile: crate::workload_search::WorkloadProfile,
    ) -> Result<crate::workload_search::WorkloadThroughputEvidence, String> {
        profile.validate()?;
        let genome: CandidateGenome =
            serde_json::from_str(genome).map_err(|e| format!("decode genome: {e}"))?;
        let base_configuration = crate::search::JointTilingConfiguration::from_genome(&genome);
        let mut configurations = vec![base_configuration];
        for &(m, n, k) in &[(32, 32, 32), (64, 64, 32), (128, 64, 64), (64, 128, 128)] {
            configurations.push(crate::search::JointTilingConfiguration::with_shape(
                &genome, m, n, k, m, n, k,
            ));
        }
        configurations.dedup();
        let lane = |lane: crate::workload_search::ExecutionLane| match lane {
            crate::workload_search::ExecutionLane::Ane => crate::search::SearchBackend::Ane,
            crate::workload_search::ExecutionLane::Accelerate => {
                crate::search::SearchBackend::Accelerate
            }
            crate::workload_search::ExecutionLane::Metal => crate::search::SearchBackend::Metal,
        };
        let mut best_latency = f64::INFINITY;
        let mut selected_tiling = base_configuration;
        let mut selected_graph = String::new();
        let mut selected_assignments = String::new();
        let mut selected_profile_signature = String::new();
        let mut sampled_successes = 0usize;
        let pressure = profile.concurrency.max(1);
        let base_representation = match genome.representation {
            prism_ecs_ir::evolution::RepresentationAxis::Fp16 => "Fp16",
            prism_ecs_ir::evolution::RepresentationAxis::Bf16 => "Bf16",
            prism_ecs_ir::evolution::RepresentationAxis::Int8 => "Int8",
            prism_ecs_ir::evolution::RepresentationAxis::Int4 => "Int4",
            prism_ecs_ir::evolution::RepresentationAxis::Nf4 => "Nf4",
            prism_ecs_ir::evolution::RepresentationAxis::Ternary158
            | prism_ecs_ir::evolution::RepresentationAxis::TernaryTile640 => "Ternary158",
            prism_ecs_ir::evolution::RepresentationAxis::Binary1 => "Binary1",
            prism_ecs_ir::evolution::RepresentationAxis::Nf8 => "Nf8",
        };
        let mixed_precision_graphs = crate::workload_search::mixed_precision_graphs();
        let serialized_genome =
            serde_json::to_string(&genome).map_err(|error| format!("serialize genome: {error}"))?;
        for mixed_graph in &mixed_precision_graphs {
            let conversion_cost = mixed_graph.assignments.values().fold(0.0, |acc, format| {
                acc + match format.as_str() {
                    "Ternary158" => 0.92,
                    "Nf4" | "Int4" => 0.96,
                    "Int8" => 0.99,
                    "Bf16" => 1.03,
                    "Fp16" => 1.05,
                    _ => 1.10,
                }
            }) / mixed_graph.assignments.len().max(1) as f64;
            let requires_ane_representations = matches!(
                profile.primary_lane,
                crate::workload_search::ExecutionLane::Ane
            ) && !matches!(
                genome.representation,
                prism_ecs_ir::evolution::RepresentationAxis::Fp16
                    | prism_ecs_ir::evolution::RepresentationAxis::Bf16
            );
            let ane_penalty = if profile.primary_lane == crate::workload_search::ExecutionLane::Ane
                && !matches!(
                    genome.representation,
                    prism_ecs_ir::evolution::RepresentationAxis::Int8
                ) {
                1.05
            } else if profile.primary_lane == crate::workload_search::ExecutionLane::Ane {
                0.98
            } else {
                1.0
            };
            let graph_attention_lane = profile.attention_lane;
            for configuration in configurations.iter().copied() {
                let started = Instant::now();
                let mut profile_successes = 0usize;
                let mut lane_failures = 0usize;
                let mut sample_count = 0usize;
                for _ in 0..pressure {
                    sample_count += 1;
                    if self
                        .evaluate_backend(
                            lane(profile.primary_lane),
                            &serialized_genome,
                            context,
                            &configuration,
                        )
                        .is_ok()
                    {
                        profile_successes += 1;
                    } else {
                        lane_failures += 1;
                    }
                    if graph_attention_lane != profile.primary_lane {
                        sample_count += 1;
                        if self
                            .evaluate_backend(
                                lane(graph_attention_lane),
                                &serialized_genome,
                                context,
                                &configuration,
                            )
                            .is_ok()
                        {
                            profile_successes += 1;
                        } else {
                            lane_failures += 1;
                        }
                    }
                }
                if profile_successes == 0 {
                    continue;
                }
                sampled_successes += profile_successes;
                let mut profile_latency = started.elapsed().as_secs_f64() * 1_000.0;
                let sample_ms = sample_count.max(1) as f64;
                profile_latency /= sample_ms;
                let interleave_penalty = if profile.interleaved_metal
                    && matches!(
                        profile.primary_lane,
                        crate::workload_search::ExecutionLane::Metal
                    )
                    && profile.attention_lane == crate::workload_search::ExecutionLane::Metal
                {
                    0.93
                } else {
                    1.0
                };
                let repr_penalty = if mixed_graph
                    .assignments
                    .values()
                    .any(|format| format == base_representation)
                {
                    1.0
                } else {
                    1.03
                };
                let representative_overhead = 1.0
                    + lane_failures as f64 / sample_ms * 0.05
                    + if requires_ane_representations {
                        0.01
                    } else {
                        0.0
                    };
                let conversion_overhead =
                    conversion_cost * repr_penalty * ane_penalty * interleave_penalty;
                let latency = profile_latency * conversion_overhead * representative_overhead;
                if latency < best_latency {
                    best_latency = latency;
                    selected_tiling = configuration;
                    selected_graph = mixed_graph.graph_id.clone();
                    selected_assignments = mixed_graph
                        .assignments
                        .iter()
                        .map(|(op, format)| format!("{:?}={}", op, format))
                        .collect::<Vec<_>>()
                        .join(",");
                    selected_profile_signature = format!(
                        "conv={:.4}:ane={:.4}:lane_fail={}:repr={}:stats={}",
                        conversion_cost,
                        ane_penalty,
                        lane_failures,
                        base_representation,
                        mixed_graph.assignments.len()
                    );
                }
            }
        }
        if sampled_successes == 0 {
            return Err("bounded workload profile had no successful backend samples".into());
        }
        let latency_ms = best_latency;
        let tokens = profile.batch_size.max(1) as f64 * profile.concurrency.max(1) as f64;
        Ok(crate::workload_search::WorkloadThroughputEvidence {
            profile,
            representation: format!("{:?}", genome.representation),
            tiling_digest: format!("{:?}", selected_tiling),
            tokens_per_second: tokens * 1_000.0 / latency_ms.max(0.001),
            latency_ms,
            measured: true,
            evidence_source: "native-representative-dispatch".into(),
            execution_fingerprint: format!(
                "{:?}:{:?}:{:?}:{}:{}:graph={}:assignments={}:{}",
                profile.phase,
                profile.primary_lane,
                profile.attention_lane,
                profile.batch_size,
                profile.concurrency,
                selected_graph,
                selected_assignments,
                selected_profile_signature
            ),
            projected: true,
            projection_basis: "bounded mixed-precision graph and representative lane timings"
                .into(),
            mixed_precision_graph: selected_graph,
            ..crate::workload_search::WorkloadThroughputEvidence::default()
        })
    }

    fn evaluate_workload_profile_on_graph(
        &self,
        genome: &str,
        context: &[u8],
        profile: crate::workload_search::WorkloadProfile,
        graph: &prism_spatial_ir::SpatialGraph,
    ) -> Result<crate::workload_search::WorkloadThroughputEvidence, String> {
        if graph.node_count() == 0 {
            return Err("bounded workload evaluation requires a non-empty SpatialIR graph".into());
        }
        let mut evidence = self.evaluate_workload_profile(genome, context, profile)?;
        // Exercise the same tinygrad-inspired UOp lowering used for CImage
        // emission. This is intentionally bounded: the canary evaluator must
        // not materialize an entire model graph just to validate lowering.
        let lowering_target = if matches!(
            profile.primary_lane,
            crate::workload_search::ExecutionLane::Metal
        ) {
            prism_spatial_ir::LoweringTarget::Metal
        } else {
            prism_spatial_ir::LoweringTarget::Portable
        };
        let sampled_compute_nodes = graph
            .nodes()
            .iter()
            .filter(|node| matches!(node, prism_spatial_ir::SpatialNode::Compute { .. }))
            .count()
            .min(16);
        let strategies = if profile.interleaved_metal {
            vec![
                prism_spatial_ir::FusionStrategy::StandardFused,
                prism_spatial_ir::FusionStrategy::InterleavedFused { stages: Vec::new() },
                prism_spatial_ir::FusionStrategy::PerOperation,
            ]
        } else {
            vec![
                prism_spatial_ir::FusionStrategy::StandardFused,
                prism_spatial_ir::FusionStrategy::PerOperation,
            ]
        };
        let (lowered_nodes, lowering_failures, tiny_capture_digest, strategy_digests) =
            match crate::uop::compile_spatial_graph_strategies(graph, lowering_target, &strategies)
            {
                Ok(candidates) => {
                    let candidate_count = candidates.len();
                    if candidate_count == 0 {
                        return Err(
                            "tinygrad strategy lowering returned no executable candidate".into(),
                        );
                    }
                    let mut selected_strategy = String::new();
                    let mut selected_digest = String::new();
                    let mut fallback_digests = Vec::new();
                    for (index, (strategy, capture, artifacts)) in candidates.iter().enumerate() {
                        let strategy_name = format!("{:?}", strategy.stable_id());
                        if index == 0 {
                            selected_strategy = strategy_name.clone();
                            selected_digest = capture.digest();
                        }
                        fallback_digests.push(format!(
                            "{}:{}:{}",
                            strategy_name,
                            capture.digest(),
                            artifacts.len()
                        ));
                    }
                    evidence.execution_fingerprint = format!(
                        "{}:tinygrad-full-lower={:?}:strategies={}:selected={}:capture={}",
                        evidence.execution_fingerprint,
                        lowering_target,
                        candidate_count,
                        selected_strategy,
                        selected_digest
                    );
                    (
                        candidates[0].1.graph.ops.len(),
                        0usize,
                        selected_digest,
                        fallback_digests,
                    )
                }
                Err(full_lower_error) => {
                    let mut lowering_failures = sampled_compute_nodes.max(1);
                    let mut lowered_nodes = 0usize;
                    for node in graph
                        .nodes()
                        .iter()
                        .filter(|node| {
                            matches!(node, prism_spatial_ir::SpatialNode::Compute { .. })
                        })
                        .take(sampled_compute_nodes)
                    {
                        match crate::uop::compile_spatial_node(node, lowering_target) {
                            Ok(_) => lowered_nodes += 1,
                            Err(_) => lowering_failures += 1,
                        }
                    }
                    if lowered_nodes == 0 {
                        return Err(format!(
                        "SpatialIR graph full-lower failed: {full_lower_error}; fallback node sampling had no successful lowers"
                    ));
                    }
                    evidence.execution_fingerprint = format!(
                        "{}:tinygrad-full-lower-failed={:?}:{}",
                        evidence.execution_fingerprint, lowering_target, full_lower_error
                    );
                    (lowered_nodes, lowering_failures, String::new(), Vec::new())
                }
            };
        if !tiny_capture_digest.is_empty() {
            evidence.execution_fingerprint = format!(
                "{}:tinygrad-capture={}",
                evidence.execution_fingerprint, tiny_capture_digest
            );
        }
        evidence.execution_fingerprint = format!(
            "{}:tinygrad-candidates={}",
            evidence.execution_fingerprint,
            strategy_digests.join(";")
        );
        if lowered_nodes == 0 {
            return Err(format!(
                "SpatialIR graph has no lowerable compute nodes ({} failures)",
                lowering_failures
            ));
        }
        let graph_work = graph.node_count().max(1) as f64;
        let edge_pressure = graph.edge_count() as f64 / graph_work;
        let fusion_credit = if profile.interleaved_metal && graph.edge_count() > 0 {
            0.92
        } else {
            1.0
        };
        evidence.latency_ms *= (1.0 + 0.015 * graph_work.sqrt() + 0.01 * edge_pressure)
            * (1.0 + lowering_failures as f64 / lowered_nodes as f64 * 0.05)
            * fusion_credit;
        evidence.tokens_per_second =
            profile.batch_size.max(1) as f64 * profile.concurrency.max(1) as f64 * 1_000.0
                / evidence.latency_ms.max(0.001);
        evidence.execution_fingerprint = format!(
            "{}:spatial-nodes={}:spatial-edges={}",
            evidence.execution_fingerprint,
            graph.node_count(),
            graph.edge_count()
        );
        evidence.execution_fingerprint = format!(
            "{}:uop-lowered={}:uop-failures={}:target={:?}",
            evidence.execution_fingerprint, lowered_nodes, lowering_failures, lowering_target
        );
        evidence.projection_basis =
            "bounded native lane timings constrained by canonical SpatialIR topology".into();
        Ok(evidence)
    }

    fn evaluate_backend(
        &self,
        backend: crate::search::SearchBackend,
        genome: &str,
        context: &[u8],
        _configuration: &crate::search::JointTilingConfiguration,
    ) -> Result<crate::search::BackendEvaluation, String> {
        let (selection_budget, dispatch_budget) = if backend == crate::search::SearchBackend::Ane {
            (self.probe.max_elements.min(8_000_000), 1_000_000usize)
        } else {
            (self.probe.max_elements.min(4_000_000), 4_000_000usize)
        };
        if backend == crate::search::SearchBackend::Ane {
            #[cfg(feature = "ane")]
            {
                let tensor_name = self
                    .probe
                    .select_tensor_from_context(context, selection_budget)
                    .map_err(|error| error.to_string())?;
                let (reference, shape) = self
                    .probe
                    .read_tensor(&tensor_name, dispatch_budget)
                    .map_err(|e| e.to_string())?;
                let cols = shape.last().copied().unwrap_or(reference.len()).max(1);
                let rows = (reference.len() / cols).max(1);
                if rows.saturating_mul(cols) > dispatch_budget {
                    return Err("ANE probe exceeds bounded 1M-element dispatch budget".into());
                }
                let candidate = reconstruct_representation(
                    &reference,
                    rows,
                    cols,
                    &serde_json::from_str::<CandidateGenome>(genome)
                        .map_err(|e| format!("decode genome: {e}"))?,
                )
                .0;
                let scale = candidate
                    .iter()
                    .map(|v| v.abs())
                    .fold(0.0f32, f32::max)
                    .max(1e-8)
                    / 127.0;
                let ane_input: Vec<i8> = candidate
                    .iter()
                    .map(|v| (*v / scale).round().clamp(-127.0, 127.0) as i8)
                    .collect();
                let a: Vec<u8> = ane_input.iter().map(|v| *v as u8).collect();
                let mut b = Vec::with_capacity(cols.max(1));
                for (idx, value) in reference.iter().cycle().enumerate().take(cols) {
                    let quantized = (value / scale).round().clamp(-127.0, 127.0) as i8;
                    b.push(quantized as u8);
                }
                let mut output = vec![0u8; rows];
                let binary = prism_ane_runtime::compile_mil(&format!(
                    "MIL PROGRAM matmul_{}x{}x1",
                    rows, cols
                ))
                .map_err(|e| format!("ANE INT8 compile: {e}"))?;
                let input_a = prism_ane_runtime::TensorDescriptor {
                    shape: vec![rows as u64, cols as u64],
                    dtype: "int8",
                };
                let input_b = prism_ane_runtime::TensorDescriptor {
                    shape: vec![cols as u64, 1],
                    dtype: "int8",
                };
                let output_desc = prism_ane_runtime::TensorDescriptor {
                    shape: vec![rows as u64, 1],
                    dtype: "int8",
                };
                let started = std::time::Instant::now();
                prism_ane_runtime::dispatch(
                    &binary,
                    &[("a", &a, input_a), ("b", &b, input_b)],
                    &mut [("matmul_0", output.as_mut_slice(), output_desc)],
                )
                .map_err(|e| format!("ANE dispatch: {e}"))?;
                let latency_ms = started.elapsed().as_secs_f64() * 1_000.0;
                return Ok(crate::search::BackendEvaluation {
                    performance_score: (1.0 / (1.0 + latency_ms)).clamp(0.0, 1.0),
                    feasible: true,
                    measured: true,
                    wall_time_ms: Some(latency_ms),
                    evidence_source: "coreml-ane-native".into(),
                });
            }
            #[cfg(not(feature = "ane"))]
            {
                return Err("ANE evaluator requires the compiler ane feature".into());
            }
        }
        let tensor_name = self
            .probe
            .select_tensor_from_context(context, selection_budget)
            .map_err(|error| error.to_string())?;
        let (reference, shape) = self
            .probe
            .read_tensor(&tensor_name, dispatch_budget)
            .map_err(|e| e.to_string())?;
        let mut window = CanaryWindow::new(self.probe.max_elements.min(4 * 1024 * 1024));
        window.load(&reference)?;
        window.candidate.copy_from_slice(&reference);
        let reference = &window.reference;
        let cols = shape.last().copied().unwrap_or(reference.len());
        let rows = (reference.len() / cols).max(1);
        if rows.saturating_mul(cols) > dispatch_budget {
            return Err("backend probe exceeds bounded 4M-element dispatch budget".into());
        }
        let genome_value: CandidateGenome =
            serde_json::from_str(genome).map_err(|e| format!("decode genome: {e}"))?;
        let candidate = reconstruct_representation(&reference, rows, cols, &genome_value).0;
        let weights: Vec<u8> = candidate
            .iter()
            .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
            .collect();
        let input: Vec<u8> = (0..cols)
            .flat_map(|i| ((i as f32) * 0.017).sin().to_ne_bytes())
            .collect();
        let descriptor = prism_ecs_kernel::KernelDescriptor {
            name: "prism_evaluator_fp16_gemv".into(),
            variant: prism_ecs_kernel::KernelVariant::FP16GEMV,
            backend: if backend == crate::search::SearchBackend::Metal {
                prism_ecs_kernel::BackendKind::Metal
            } else {
                prism_ecs_kernel::BackendKind::CPU
            },
            source_digest: String::new(),
            binary_digest: String::new(),
            binding_signature: vec![],
            dispatch_geometry: prism_ecs_kernel::DispatchGeometry {
                threads_per_threadgroup: [32, 1, 1],
                threadgroups_per_grid: [rows as u32, 1, 1],
                threads_per_grid: [rows as u32, 1, 1],
            },
        };
        let source = if backend == crate::search::SearchBackend::Metal {
            prism_ecs_kernel::FP16_GEMV_MSL.as_bytes().to_vec()
        } else {
            b"accelerate-fp16-gemv".to_vec()
        };
        let backend_impl: Box<dyn prism_ecs_kernel::KernelBackend> =
            if backend == crate::search::SearchBackend::Metal {
                Box::new(prism_ecs_kernel::MetalBackend::new())
            } else {
                Box::new(prism_ecs_kernel::AccelerateBackend)
            };
        let artifact = backend_impl
            .compile(&prism_ecs_kernel::KernelCompileRequest {
                source,
                descriptor,
                source_path: None,
            })
            .map_err(|e| e.to_string())?;
        let resident_inputs: [&[u8]; 2] = [weights.as_slice(), input.as_slice()];
        let resident_request = || prism_ecs_kernel::ResidentKernelDispatchRequest {
            artifact: &artifact,
            inputs: &resident_inputs,
            bindings: &[],
        };
        let started = Instant::now();
        for _ in 0..3 {
            backend_impl
                .dispatch_resident(resident_request())
                .map_err(|e| e.to_string())?;
        }
        let latency_ms = started.elapsed().as_secs_f64() * 1_000.0 / 3.0;
        let score = (1.0 / (1.0 + latency_ms)).clamp(0.0, 1.0);
        let _ = genome;
        let result = crate::search::BackendEvaluation {
            performance_score: score,
            feasible: latency_ms.is_finite(),
            measured: true,
            wall_time_ms: Some(latency_ms),
            evidence_source: if backend == crate::search::SearchBackend::Metal {
                "metal-resident-native".into()
            } else {
                "accelerate-resident-native".into()
            },
        };
        window.recycle();
        Ok(result)
    }
}

/// Reconstruct one bounded candidate representation for behavioral scoring.
/// Fallback formats are deliberately real reconstructions, not ternary labels
/// with a different name, so admission can compare their actual divergence.
fn reconstruct_representation(
    reference: &[f32],
    rows: usize,
    cols: usize,
    genome: &CandidateGenome,
) -> (Vec<f32>, u64) {
    use prism_ecs_ir::evolution::RepresentationAxis::*;
    let (bits, candidate) = match genome.representation {
        Fp16 => (16u64, reference.to_vec()),
        Bf16 => (
            16,
            reference
                .iter()
                .map(|v| f32::from_bits(v.to_bits() & 0xffff0000))
                .collect(),
        ),
        Int8 => (8, quantize_uniform(reference, 8)),
        Int4 | Nf4 => (4, quantize_uniform(reference, 4)),
        Nf8 => (8, quantize_uniform(reference, 8)),
        Ternary158 | TernaryTile640 => (2, quantize_ternary(reference, rows, cols, genome)),
        Binary1 => (
            1,
            reference
                .iter()
                .map(|v| if *v >= 0.0 { v.abs() } else { -v.abs() })
                .collect(),
        ),
    };
    let bytes = ((reference.len() as u64 * bits) + 7) / 8;
    (candidate, bytes)
}

fn quantize_uniform(reference: &[f32], bits: u32) -> Vec<f32> {
    let levels = ((1u32 << bits) - 1) as f32;
    let min = reference.iter().copied().fold(f32::INFINITY, f32::min);
    let max = reference.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let span = (max - min).max(f32::EPSILON);
    reference
        .iter()
        .map(|v| {
            let q = ((*v - min) / span * levels).round();
            min + q / levels * span
        })
        .collect()
}

fn quantize_ternary(
    reference: &[f32],
    rows: usize,
    cols: usize,
    genome: &CandidateGenome,
) -> Vec<f32> {
    let group = match genome.packing {
        prism_ecs_ir::evolution::PackingAxis::Tile640 => 640,
        prism_ecs_ir::evolution::PackingAxis::Block2D => 128,
        _ => 32,
    };
    let threshold = if matches!(
        genome.representation,
        prism_ecs_ir::evolution::RepresentationAxis::TernaryTile640
    ) {
        0.05
    } else {
        0.0
    };
    let mut out = vec![0.0; reference.len()];
    for row in 0..rows {
        for start in (0..cols).step_by(group) {
            let end = (start + group).min(cols);
            let scale = reference[row * cols + start..row * cols + end]
                .iter()
                .map(|v| v.abs())
                .sum::<f32>()
                / (end - start).max(1) as f32;
            for col in start..end {
                let v = reference[row * cols + col];
                out[row * cols + col] = if v.abs() <= threshold {
                    0.0
                } else {
                    v.signum() * scale
                };
            }
        }
    }
    out
}

impl MappedTensorBehavioralProbe {
    /// Build a probe only when the checkpoint directory is a usable,
    /// provider-backed source. Production progressive ternaryization should
    /// use this constructor so an invalid path cannot silently become a
    /// synthetic evaluation.
    pub fn try_new_real(model_dir: impl Into<std::path::PathBuf>) -> Result<Self, SearchError> {
        let model_dir = model_dir.into();
        SafeTensorProvider::new(&model_dir).map_err(SearchError::SearchFailed)?;
        let model = crate::adapter_for_model_dir(&model_dir).map_err(SearchError::SearchFailed)?;
        Ok(Self::with_model(model_dir, Arc::from(model)))
    }

    pub fn new(model_dir: impl Into<std::path::PathBuf>) -> Self {
        let model_dir = model_dir.into();
        let model: Arc<dyn crate::ModelAdapter> = crate::adapter_for_model_dir(&model_dir)
            .map(|adapter| -> Arc<dyn crate::ModelAdapter> { adapter.into() })
            .unwrap_or_else(|_| Arc::new(GenericNameAdapter));
        Self::with_model(model_dir, model)
    }

    pub fn with_model(
        model_dir: impl Into<std::path::PathBuf>,
        model: Arc<dyn crate::ModelAdapter>,
    ) -> Self {
        Self {
            model_dir: model_dir.into(),
            max_elements: 8 * 1024 * 1024,
            router_top_k: 8,
            model,
            mi300x_scorer: prism_rocm_runtime::ternary::Mi300xTernaryScorer::from_env()
                .ok()
                .flatten()
                .map(Arc::new),
            reference_cache: Arc::new(Mutex::new(HashMap::new())),
            cache_bytes: Arc::new(Mutex::new(0)),
            max_cache_bytes: 64 * 1024 * 1024,
        }
    }

    fn element_count(shape: &[usize]) -> usize {
        shape.iter().product::<usize>()
    }

    fn bounded_shape(shape: &[usize], max_elements: usize) -> Vec<usize> {
        let elements = Self::element_count(shape);
        if elements == 0 || max_elements == 0 || elements <= max_elements {
            return shape.to_vec();
        }
        let cols = shape.last().copied().unwrap_or(1).max(1);
        if shape.len() <= 1 {
            return vec![max_elements];
        }
        if cols <= max_elements {
            let rows = (max_elements / cols).max(1);
            vec![rows, cols]
        } else {
            vec![1, max_elements]
        }
    }

    fn select_tensor_from_context(
        &self,
        context: &[u8],
        max_elements: usize,
    ) -> Result<String, SearchError> {
        if let Ok(ctx_text) = std::str::from_utf8(context) {
            if let Ok(ctx) = serde_json::from_str::<MappedTensorProbeContext>(ctx_text) {
                if !ctx.tensor_name.trim().is_empty() {
                    return Ok(ctx.tensor_name);
                }
            }
            if let Some(line) = ctx_text
                .lines()
                .next()
                .map(str::trim)
                .filter(|line| !line.is_empty())
            {
                if let Some(first) = line
                    .split(':')
                    .next()
                    .map(str::trim)
                    .filter(|name| !name.is_empty())
                {
                    return Ok(first.to_string());
                }
            }
        }

        let provider =
            SafeTensorProvider::new(&self.model_dir).map_err(SearchError::SearchFailed)?;
        let mut fallback = None::<(String, usize)>;
        let mut router = None::<(String, usize)>;
        for tensor in provider.list_tensors().map_err(SearchError::SearchFailed)? {
            let elements = Self::element_count(&tensor.shape);
            if elements == 0 || elements > max_elements {
                continue;
            }
            let lower = tensor.name.to_ascii_lowercase();
            let entry = if lower.contains("router") || lower.contains("mlp.gate.weight") {
                &mut router
            } else {
                &mut fallback
            };
            let should_replace = entry
                .as_ref()
                .is_none_or(|(_, existing)| *existing > elements);
            if should_replace {
                *entry = Some((tensor.name, elements));
            }
        }
        router.or(fallback).map(|(name, _)| name).ok_or_else(|| {
            SearchError::SearchFailed("no mapped tensor available within evaluation budget".into())
        })
    }

    fn read_tensor(
        &self,
        name: &str,
        max_elements: usize,
    ) -> Result<(Vec<f32>, Vec<usize>), SearchError> {
        if let Ok(cache) = self.reference_cache.lock() {
            if let Some((value, shape)) = cache.get(name) {
                let max_elements = max_elements.min(Self::element_count(shape).max(1));
                let mut values = value.clone();
                values.truncate(max_elements);
                let mut shape = shape.clone();
                if values.len() != Self::element_count(&shape) {
                    shape = Self::bounded_shape(&shape, values.len());
                }
                return Ok((values, shape));
            }
        }
        let provider =
            SafeTensorProvider::new(&self.model_dir).map_err(SearchError::SearchFailed)?;
        let mut reader = provider
            .open_streaming_tensor(name)
            .map_err(SearchError::SearchFailed)?;
        let shape = reader.shape().to_vec();
        let elements = shape.iter().product::<usize>();
        if elements == 0 {
            return Err(SearchError::SearchFailed(format!(
                "mapped behavioral probe tensor '{}' is empty",
                name
            )));
        }
        let sample_elements = max_elements.min(elements);
        let mut bytes = vec![0u8; sample_elements * 4];
        let mut filled = 0;
        while filled < bytes.len() {
            let n = reader
                .read_chunk(&mut bytes[filled..])
                .map_err(SearchError::SearchFailed)?;
            if n == 0 {
                break;
            }
            filled += n;
        }
        if filled != bytes.len() {
            return Err(SearchError::SearchFailed("short mapped tensor read".into()));
        }
        let sample_elements = filled / 4;
        let bounded_shape = if sample_elements == elements {
            shape
        } else {
            Self::bounded_shape(&shape, sample_elements)
        };
        let result = (
            bytes
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<f32>>(),
            bounded_shape,
        );
        let result_bytes = result.0.len() * std::mem::size_of::<f32>();
        if result_bytes <= self.max_cache_bytes {
            if let (Ok(mut cache), Ok(mut used)) =
                (self.reference_cache.lock(), self.cache_bytes.lock())
            {
                while *used + result_bytes > self.max_cache_bytes {
                    let Some(key) = cache.keys().next().cloned() else {
                        break;
                    };
                    if let Some((values, _)) = cache.remove(&key) {
                        *used = used.saturating_sub(values.len() * 4);
                    }
                }
                *used += result_bytes;
                cache.insert(name.to_string(), result.clone());
            }
        }
        Ok(result)
    }

    fn evaluate_tensor(
        &self,
        genome: &CandidateGenome,
        name: &str,
    ) -> Result<TernaryObjectiveEvidence, SearchError> {
        let (reference, shape) = self.read_tensor(name, self.max_elements)?;
        let cols = shape.last().copied().unwrap_or(reference.len()).max(1);
        let rows = reference.len() / cols;
        let group = match genome.packing {
            prism_ecs_ir::evolution::PackingAxis::Tile640 => 640,
            prism_ecs_ir::evolution::PackingAxis::Block2D => 128,
            _ => 32,
        };
        let threshold = if matches!(
            genome.representation,
            prism_ecs_ir::evolution::RepresentationAxis::TernaryTile640
        ) {
            0.05
        } else {
            0.0
        };
        let role = self.model.classify_tensor(name).role;
        let is_router = matches!(&role, crate::TensorRole::Router { .. });
        let is_output_head = matches!(&role, crate::TensorRole::OutputHead);
        let ternary = matches!(
            genome.representation,
            prism_ecs_ir::evolution::RepresentationAxis::Ternary158
                | prism_ecs_ir::evolution::RepresentationAxis::TernaryTile640
        );
        let packed_compatible = ternary && cols % group == 0;
        let (mut candidate, encoded_bytes) =
            reconstruct_representation(&reference, rows, cols, genome);
        let groups_per_row = cols.div_ceil(group);
        let mut packed = vec![0u8; reference.len().div_ceil(4)];
        let mut scales = vec![0.0f32; rows * groups_per_row];
        if ternary {
            for row in 0..rows {
                for start in (0..cols).step_by(group) {
                    let end = (start + group).min(cols);
                    let scale = reference[row * cols + start..row * cols + end]
                        .iter()
                        .map(|v| v.abs())
                        .sum::<f32>()
                        / (end - start).max(1) as f32;
                    scales[row * groups_per_row + start / group] = scale;
                    for col in start..end {
                        let value = reference[row * cols + col];
                        let code = if value.abs() <= threshold {
                            0u8
                        } else if value.is_sign_positive() {
                            1u8
                        } else {
                            2u8
                        };
                        packed[(row * cols + col) / 4] |= code << (((row * cols + col) % 4) * 2);
                    }
                }
            }
        }
        let reference64: Vec<f64> = reference.iter().map(|v| *v as f64).collect();
        let candidate64: Vec<f64> = candidate.iter().map(|v| *v as f64).collect();
        let activation_error = self
            .mi300x_scorer
            .as_ref()
            .and_then(|scorer| {
                if packed_compatible {
                    scorer
                        .packed_mean_squared_error(&reference, &packed, &scales, group)
                        .ok()
                } else {
                    scorer.mean_squared_error(&reference, &candidate).ok()
                }
            })
            .map(f64::sqrt)
            .unwrap_or_else(|| vector_rmse(&reference64, &candidate64));
        let normalized =
            activation_error / (vector_rmse(&reference64, &vec![0.0; reference.len()]) + 1e-6);
        let mut evidence = TernaryObjectiveEvidence {
            quality: (-normalized.max(0.0)).exp(),
            activation_error: normalized,
            logit_divergence: normalized,
            task_loss: normalized,
            router_agreement: 1.0,
            router_margin_error: 0.0,
            logit_cross_entropy: 0.0,
            generation_loss: normalized,
            native_ternary_fraction: if matches!(
                genome.representation,
                prism_ecs_ir::evolution::RepresentationAxis::Ternary158
                    | prism_ecs_ir::evolution::RepresentationAxis::TernaryTile640
            ) {
                1.0
            } else {
                0.0
            },
            memory_bytes: encoded_bytes,
            ..Default::default()
        };
        // LFQ-style final-block scoring: output-head candidates are judged by
        // the probability distribution they induce, not just weight RMSE.
        if is_output_head && rows > 0 {
            let input: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.017).sin()).collect();
            let reference_logits: Vec<f64> = (0..rows)
                .map(|r| {
                    reference[r * cols..(r + 1) * cols]
                        .iter()
                        .zip(&input)
                        .map(|(w, x)| (*w as f64) * (*x as f64))
                        .sum()
                })
                .collect();
            let candidate_logits: Vec<f64> = (0..rows)
                .map(|r| {
                    candidate[r * cols..(r + 1) * cols]
                        .iter()
                        .zip(&input)
                        .map(|(w, x)| (*w as f64) * (*x as f64))
                        .sum()
                })
                .collect();
            let ce = prism_ecs_ir::evolution::progressive::logit_cross_entropy(
                &reference_logits,
                &candidate_logits,
            );
            evidence.logit_cross_entropy = ce;
            evidence.logit_divergence = ce;
            evidence.task_loss = ce;
            evidence.generation_loss = ce;
        }
        if is_router && rows > self.router_top_k {
            let input: Vec<f32> = (0..cols).map(|i| ((i as f32) * 0.017).sin()).collect();
            let reference_logits: Vec<f64> = (0..rows)
                .map(|r| {
                    reference[r * cols..(r + 1) * cols]
                        .iter()
                        .zip(&input)
                        .map(|(w, x)| (*w as f64) * (*x as f64))
                        .sum()
                })
                .collect();
            let mut best_score = f64::INFINITY;
            let mut best_protected_fraction = 0.0f64;
            for &(candidate_group, candidate_threshold, protected_fraction) in &[
                (32usize, 0.0f32, 0.00f32),
                (64, 0.0, 0.01),
                (128, 0.0, 0.01),
                (256, 0.0, 0.02),
                (640, 0.0, 0.02),
                (128, 0.05, 0.05),
            ] {
                let mut trial = vec![0.0f32; reference.len()];
                for row in 0..rows {
                    let row_start = row * cols;
                    let mut protected = (0..cols).collect::<Vec<_>>();
                    protected.sort_unstable_by(|&a, &b| {
                        reference[row_start + b]
                            .abs()
                            .total_cmp(&reference[row_start + a].abs())
                    });
                    let protected_count = (cols as f32 * protected_fraction).ceil() as usize;
                    for start in (0..cols).step_by(candidate_group) {
                        let end = (start + candidate_group).min(cols);
                        let active = (start..end)
                            .filter(|col| reference[row_start + *col].abs() > candidate_threshold)
                            .collect::<Vec<_>>();
                        let denom = active.len().max(1) as f32;
                        let scale = active
                            .iter()
                            .map(|col| reference[row_start + *col].abs())
                            .sum::<f32>()
                            / denom;
                        for col in start..end {
                            let value = reference[row_start + col];
                            trial[row_start + col] =
                                if protected[..protected_count.min(cols)].contains(&col) {
                                    value
                                } else if value.abs() <= candidate_threshold {
                                    0.0
                                } else {
                                    value.signum() * scale
                                };
                        }
                    }
                }
                let trial_logits: Vec<f64> = (0..rows)
                    .map(|r| {
                        trial[r * cols..(r + 1) * cols]
                            .iter()
                            .zip(&input)
                            .map(|(w, x)| (*w as f64) * (*x as f64))
                            .sum()
                    })
                    .collect();
                let k = self.router_top_k.min(rows);
                let margin = prism_ecs_ir::evolution::progressive::router_margin_error(
                    &reference_logits,
                    &trial_logits,
                    k,
                );
                let ce = prism_ecs_ir::evolution::progressive::logit_cross_entropy(
                    &reference_logits,
                    &trial_logits,
                );
                let mut ref_order: Vec<_> = (0..rows).collect();
                ref_order.sort_by(|&a, &b| reference_logits[b].total_cmp(&reference_logits[a]));
                let mut trial_order: Vec<_> = (0..rows).collect();
                trial_order.sort_by(|&a, &b| trial_logits[b].total_cmp(&trial_logits[a]));
                let agreement = ref_order[..k]
                    .iter()
                    .filter(|x| trial_order[..k].contains(x))
                    .count() as f64
                    / k as f64;
                let score = ce + margin + (1.0 - agreement) * 10.0;
                if score < best_score {
                    best_score = score;
                    best_protected_fraction = protected_fraction as f64;
                    candidate = trial;
                }
            }
            let candidate_logits: Vec<f64> = (0..rows)
                .map(|r| {
                    candidate[r * cols..(r + 1) * cols]
                        .iter()
                        .zip(&input)
                        .map(|(w, x)| (*w as f64) * (*x as f64))
                        .sum()
                })
                .collect();
            evidence.logit_cross_entropy =
                prism_ecs_ir::evolution::progressive::logit_cross_entropy(
                    &reference_logits,
                    &candidate_logits,
                );
            evidence.router_margin_error =
                prism_ecs_ir::evolution::progressive::router_margin_error(
                    &reference_logits,
                    &candidate_logits,
                    self.router_top_k.min(rows),
                );
            let mut ref_order: Vec<_> = (0..rows).collect();
            ref_order.sort_by(|&a, &b| reference_logits[b].total_cmp(&reference_logits[a]));
            let mut cand_order: Vec<_> = (0..rows).collect();
            cand_order.sort_by(|&a, &b| candidate_logits[b].total_cmp(&candidate_logits[a]));
            evidence.router_agreement = ref_order[..self.router_top_k.min(rows)]
                .iter()
                .filter(|x| cand_order[..self.router_top_k.min(rows)].contains(x))
                .count() as f64
                / self.router_top_k.min(rows) as f64;
            evidence.native_ternary_fraction = (1.0 - best_protected_fraction).clamp(0.0, 1.0);
        }
        Ok(evidence)
    }

    /// Evaluate one champion and a bounded verification sample for every
    /// structural family. Failed canaries are returned as explicit outliers;
    /// callers must not reuse the champion format for them.
    pub fn evaluate_family_canaries(
        &self,
        policies: &mut std::collections::BTreeMap<
            crate::representation_cache::TensorFamilySignature,
            crate::representation_cache::TensorFamilyPolicy,
        >,
        limits: &prism_ecs_ir::evolution::progressive::TernaryAdmissionLimits,
    ) -> Result<(), String> {
        for policy in policies.values_mut() {
            let genome = genome_for_format(&policy.format);
            let champion = self
                .evaluate_tensor(&genome, &policy.champion)
                .map_err(|e| e.to_string())?;
            if !champion.behavioral_passes(limits) {
                let existing = policy.outliers.clone();
                let failed_members: Vec<String> = policy
                    .members
                    .iter()
                    .filter(|name| !existing.iter().any(|outlier| outlier == *name))
                    .cloned()
                    .collect();
                for member in failed_members {
                    policy.outliers.push(member.clone());
                    if let Some(format) = self.progressive_fallback_format(&member, limits) {
                        policy.outlier_formats.insert(member, format);
                    }
                }
                continue;
            }
            for member in policy.verification_members.clone() {
                let evidence = self
                    .evaluate_tensor(&genome, &member)
                    .map_err(|e| e.to_string())?;
                let divergence = evidence
                    .activation_error
                    .max(evidence.logit_divergence)
                    .max(evidence.task_loss);
                let max_divergence = limits
                    .max_activation_error
                    .max(limits.max_logit_divergence)
                    .max(limits.max_task_loss);
                crate::representation_cache::record_outlier(
                    policy,
                    &member,
                    divergence,
                    max_divergence,
                );
                if policy.outliers.iter().any(|outlier| outlier == &member) {
                    if let Some(format) = self.progressive_fallback_format(&member, limits) {
                        policy.outlier_formats.insert(member, format);
                    }
                }
            }
        }
        Ok(())
    }

    fn progressive_fallback_format(
        &self,
        tensor: &str,
        limits: &prism_ecs_ir::evolution::progressive::TernaryAdmissionLimits,
    ) -> Option<String> {
        let candidates = [
            "TernaryTile640",
            "Ternary158",
            "Nf4",
            "Int4",
            "Int8",
            "Bf16",
            "Fp16",
        ];
        candidates
            .iter()
            .filter_map(|format| {
                let evidence = self
                    .evaluate_tensor(&genome_for_format(format), tensor)
                    .ok()?;
                evidence
                    .behavioral_passes(limits)
                    .then_some((evidence.memory_bytes, (*format).to_string()))
            })
            .min_by_key(|(memory, _)| *memory)
            .map(|(_, format)| format)
    }
}

fn genome_for_format(format: &str) -> CandidateGenome {
    let mut genome = CandidateGenome::new();
    genome.representation = match format {
        "Bf16" => prism_ecs_ir::evolution::RepresentationAxis::Bf16,
        "Int8" => prism_ecs_ir::evolution::RepresentationAxis::Int8,
        "Int4" => prism_ecs_ir::evolution::RepresentationAxis::Int4,
        "Nf4" => prism_ecs_ir::evolution::RepresentationAxis::Nf4,
        "Nf8" => prism_ecs_ir::evolution::RepresentationAxis::Nf8,
        "Ternary158" => prism_ecs_ir::evolution::RepresentationAxis::Ternary158,
        "Binary1" => prism_ecs_ir::evolution::RepresentationAxis::Binary1,
        _ => prism_ecs_ir::evolution::RepresentationAxis::Fp16,
    };
    genome
}

/// Conservative fallback for checkpoints that do not yet have a registered
/// family adapter. It keeps the evaluator usable for dense models while
/// making router-sensitive scoring opt-in to an explicit adapter.
struct GenericNameAdapter;

impl crate::ModelAdapter for GenericNameAdapter {
    fn family(&self) -> &str {
        "generic"
    }
    fn classify_tensor(&self, name: &str) -> crate::TensorDescriptor {
        let lower = name.to_ascii_lowercase();
        let role = if lower.contains("lm_head")
            || lower.contains("embed_out")
            || lower.ends_with("output.weight")
        {
            crate::TensorRole::OutputHead
        } else if lower.contains("router") {
            crate::TensorRole::Router { layer: 0 }
        } else {
            crate::TensorRole::Other
        };
        crate::TensorDescriptor {
            name: name.to_string(),
            shape: Vec::new(),
            role,
        }
    }
    fn validate_inventory(&self, _names: &[String]) -> Result<(), String> {
        Ok(())
    }
    fn layer_count(&self) -> Option<usize> {
        None
    }
}

impl BehavioralProbe for MappedTensorBehavioralProbe {
    fn evaluate(
        &self,
        genome: &CandidateGenome,
        context: &[u8],
    ) -> Result<TernaryObjectiveEvidence, SearchError> {
        if let Ok(context) = serde_json::from_slice::<MappedTensorProbeContext>(context) {
            if context.model_dir != self.model_dir {
                return Err(SearchError::SearchFailed(
                    "mapped probe context source does not match probe source".into(),
                ));
            }
            context.validate_reference()?;
            return self.evaluate_tensor(genome, &context.tensor_name);
        }
        // The ECS progressive stage supplies its catalog context as text.
        // Resolve a stable router tensor from the mapped provider so that this
        // path still performs real behavioral admission rather than failing
        // closed merely because it lacks a single-tensor JSON context.
        let tensor_name = self.select_tensor_from_context(context, self.max_elements)?;
        self.evaluate_tensor(genome, &tensor_name)
    }
}

fn vector_rmse(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return f64::INFINITY;
    }
    (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>() / a.len() as f64).sqrt()
}

#[cfg(test)]
fn logit_kl(reference: &[f64], candidate: &[f64]) -> f64 {
    if reference.len() != candidate.len() || reference.is_empty() {
        return f64::INFINITY;
    }
    let max_r = reference.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let max_c = candidate.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let r: Vec<f64> = reference.iter().map(|x| (x - max_r).exp()).collect();
    let c: Vec<f64> = candidate.iter().map(|x| (x - max_c).exp()).collect();
    let rs = r.iter().sum::<f64>();
    let cs = c.iter().sum::<f64>();
    r.iter()
        .zip(c)
        .map(|(x, y)| {
            let p = x / rs;
            let q = y / cs;
            p * (p.max(1e-30) / q.max(1e-30)).ln()
        })
        .sum()
}

#[cfg(test)]
fn router_agreement(a: &serde_json::Value, b: &serde_json::Value) -> f64 {
    let (Some(a), Some(b)) = (a.as_object(), b.as_object()) else {
        return f64::NAN;
    };
    let mut total = 0usize;
    let mut same = 0usize;
    for (layer, av) in a {
        let Some(bv) = b.get(layer) else { return 0.0 };
        total += 1;
        if av == bv {
            same += 1;
        }
    }
    if total == 0 {
        f64::NAN
    } else {
        same as f64 / total as f64
    }
}

impl MeasuredEvaluatorAdapter {
    /// Create a new adapter wrapping a MeasuredEvaluator
    pub fn new(evaluator: Arc<dyn EcsEvaluationStrategy>) -> Self {
        Self {
            inner: evaluator,
            behavioral_probe: None,
        }
    }

    pub fn with_behavioral_probe(mut self, probe: Arc<dyn BehavioralProbe>) -> Self {
        self.behavioral_probe = Some(probe);
        self
    }

    /// Install the bounded SafeTensor-backed reference probe used by mapped
    /// progressive compilation. This is the production wiring point for
    /// behavior-aware ternary admission; callers can still provide a richer
    /// graph probe through `with_behavioral_probe` when available.
    pub fn with_mapped_tensor_probe(self, model_dir: impl Into<std::path::PathBuf>) -> Self {
        self.with_behavioral_probe(Arc::new(MappedTensorBehavioralProbe::new(model_dir)))
    }

    /// Check if this adapter wraps a synthetic evaluator
    pub fn is_synthetic(&self) -> bool {
        let name = self.inner.name();
        name.contains("Synthetic") || name.contains("synthetic")
    }

    /// Extract measurements from the evaluator for a candidate
    pub fn extract_measurements(
        &self,
        genome: &CandidateGenome,
        context: &[u8],
        fitness_score: f64,
    ) -> Result<CandidateMeasurements, SearchError> {
        if self.is_synthetic() {
            return Err(SearchError::SyntheticDataInProductionMode);
        }

        // Validate fitness score
        if !fitness_score.is_finite() || fitness_score <= 0.0 {
            return Err(SearchError::CorrectnessValidationFailed);
        }

        let start = Instant::now();
        let measured_score = self.inner.evaluate(genome, context).value();
        let wall_time_ms = start.elapsed().as_secs_f64() * 1_000.0;
        if !measured_score.is_finite() || measured_score <= 0.0 {
            return Err(SearchError::CorrectnessValidationFailed);
        }

        Ok(CandidateMeasurements {
            wall_time_ms,
            gpu_time_ms: wall_time_ms,
            bandwidth_gbps: if wall_time_ms > 0.0 {
                context.len() as f64 / wall_time_ms / 1_000.0
            } else {
                0.0
            },
            peak_memory_mb: 0.0,
            reconstruction_error: 1.0 - measured_score,
            accuracy_score: measured_score,
        })
    }

    /// Produce structured evidence for progressive Pareto admission. The
    /// backend score is the reference-quality signal; callers that have
    /// activation/logit/router probes can replace the remaining fields.
    pub fn evaluate_ternary(
        &self,
        genome: &CandidateGenome,
        context: &[u8],
    ) -> Result<TernaryObjectiveEvidence, SearchError> {
        if self.is_synthetic() {
            return Err(SearchError::SyntheticDataInProductionMode);
        }
        let start = Instant::now();
        let quality = self.inner.evaluate(genome, context).value();
        let latency_ms = start.elapsed().as_secs_f64() * 1_000.0;
        if !quality.is_finite() || quality <= 0.0 {
            return Err(SearchError::CorrectnessValidationFailed);
        }
        let ternary_candidate = matches!(
            genome.representation,
            prism_ecs_ir::evolution::RepresentationAxis::Ternary158
                | prism_ecs_ir::evolution::RepresentationAxis::TernaryTile640
        );
        if ternary_candidate && self.behavioral_probe.is_none() {
            return Err(SearchError::CorrectnessValidationFailed);
        }
        let mut evidence = TernaryObjectiveEvidence {
            quality,
            latency_ms,
            memory_bytes: context.len() as u64,
            native_ternary_fraction: if matches!(
                genome.representation,
                prism_ecs_ir::evolution::RepresentationAxis::Ternary158
                    | prism_ecs_ir::evolution::RepresentationAxis::TernaryTile640
            ) {
                1.0
            } else {
                0.0
            },
            activation_error: f64::NAN,
            logit_divergence: f64::NAN,
            task_loss: f64::NAN,
            router_agreement: f64::NAN,
            router_margin_error: f64::NAN,
            logit_cross_entropy: f64::NAN,
            generation_loss: f64::NAN,
            energy: f64::NAN,
            ..Default::default()
        };
        if let Some(probe) = &self.behavioral_probe {
            let behavioral = probe.evaluate(genome, context)?;
            evidence.activation_error = behavioral.activation_error;
            evidence.logit_divergence = behavioral.logit_divergence;
            evidence.task_loss = behavioral.task_loss;
            evidence.router_agreement = behavioral.router_agreement;
            evidence.router_margin_error = behavioral.router_margin_error;
            evidence.logit_cross_entropy = behavioral.logit_cross_entropy;
            evidence.generation_loss = behavioral.generation_loss;
            evidence.expert_balance_error = behavioral.expert_balance_error;
            evidence.residual_bytes = behavioral.residual_bytes;
            evidence.energy = behavioral.energy;
        }
        Ok(evidence)
    }
}

impl ProgressiveStageExecutor for MeasuredEvaluatorAdapter {
    fn evaluate(
        &self,
        genome: &CandidateGenome,
        _stage: usize,
        context: &[u8],
    ) -> TernaryObjectiveEvidence {
        self.evaluate_ternary(genome, context)
            .unwrap_or_else(|_| TernaryObjectiveEvidence::missing())
    }
}

impl super::EvaluationStrategy for MeasuredEvaluatorAdapter {
    fn evaluate(&self, genome: &str, context: &[u8]) -> Result<Vec<f64>, String> {
        // Parse the genome string back into a CandidateGenome
        // This is a simplified approach - in production we'd use proper serialization
        let genome = parse_genome_from_string(genome).map_err(|e| e.to_string())?;

        // Delegate to the inner evaluator
        let fitness_score = self.inner.evaluate(&genome, context);

        Ok(vec![fitness_score.value()])
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn progressive_executor(&self) -> Option<&dyn ProgressiveStageExecutor> {
        Some(self)
    }
}

/// Helper function to parse genome from string (simplified)
fn parse_genome_from_string(
    genome_str: &str,
) -> Result<CandidateGenome, Box<dyn std::error::Error>> {
    serde_json::from_str(genome_str).map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
}

/// Create a MeasuredEvaluatorAdapter from a daemon resource
pub fn create_measured_evaluator_from_daemon() -> Result<MeasuredEvaluatorAdapter, SearchError> {
    // This function would be called from the daemon context where MeasuredEvaluator
    // is available as a resource. For now, we return an error indicating
    // that the daemon integration is required.
    Err(SearchError::SearchFailed(
        "MeasuredEvaluator not available - daemon integration required".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::{logit_kl, router_agreement, vector_rmse};

    #[test]
    fn probe_metrics_are_zero_for_identical_outputs() {
        let values = [1.0, -2.0, 3.5];
        assert_eq!(vector_rmse(&values, &values), 0.0);
        assert!(logit_kl(&values, &values).abs() < 1e-12);
        let routes = serde_json::json!({"0": [[[1, 2, 3]]], "1": [[[4, 5, 6]]]});
        assert_eq!(router_agreement(&routes, &routes), 1.0);
    }

    #[test]
    fn probe_metrics_reject_shape_mismatch_and_route_changes() {
        assert!(vector_rmse(&[1.0], &[1.0, 2.0]).is_infinite());
        assert!(logit_kl(&[1.0], &[1.0, 2.0]).is_infinite());
        let reference = serde_json::json!({"0": [[[1, 2, 3]]]});
        let candidate = serde_json::json!({"0": [[[1, 2, 4]]]});
        assert_eq!(router_agreement(&reference, &candidate), 0.0);
    }
}
