//! Mapped-tensor search-system evaluation strategy family.
//!
//! **Single authority:** The constitutional adapters
//! ([`MeasuredEvaluatorAdapter`], [`MappedTensorEvaluationStrategy`])
//! that translate between the internal evolutionary-evidence API
//! and the search-system
//! [`crate::search::EvaluationStrategy`] trait, plus the workload and
//! backend evaluation plumbing that drives the bounded reference
//! probe. The wrapper is the only path by which an internal
//! evaluator (synthetic, measured, or behavioral) is exposed to the
//! search layer, and the plumbing is the only path by which the
//! mapped-tensor strategy actually drives the bounded reference
//! probe through mixed-precision graph candidates, SpatialIR
//! lowering, ANE/Metal/Accelerate backend dispatch, and a
//! `CanaryWindow` evidence window. Fail-closed semantics for the
//! `ProgressiveStageExecutor` impl live in
//! [`super::super::fail_closed`] so the fail-closed authority is
//! isolated to one place.

#![forbid(unsafe_code)]

use std::sync::Arc;
use std::time::Instant;

use prism_ecs_ir::evolution::evaluate::EvaluationStrategy as EcsEvaluationStrategy;
use prism_ecs_ir::evolution::foundation::CandidateGenome;
use prism_ecs_ir::evolution::progressive::TernaryAdmissionLimits;
use prism_ecs_ir::evolution::{ProgressiveStageExecutor, TernaryObjectiveEvidence};
use prism_spatial_ir::{FusionStrategy, LoweringTarget, SpatialGraph, SpatialNode};

use crate::search::{EvaluationStrategy as SearchEvaluationStrategy, SearchError};
use crate::workload_search::WorkloadProfile;

use super::behavioral::BehavioralProbe;
use super::progressive::parse_genome_from_string;

use super::super::canary_window::CanaryWindow;
use super::super::fail_closed;

// ---------------------------------------------------------------------------
// MeasuredEvaluatorAdapter — the constitutional adapter that turns an
// internal EvaluationStrategy into the search-system EvaluationStrategy
// and ProgressiveStageExecutor surface
// ---------------------------------------------------------------------------

/// Wrapper around an internal [`EcsEvaluationStrategy`] that adapts it
/// to the search system's [`SearchEvaluationStrategy`] trait.
///
/// The adapter is the only path by which an internal evaluator
/// (synthetic, measured, or behavioral) is exposed to the search
/// layer. It carries an optional [`BehavioralProbe`] for activation,
/// logit, and router scoring; fail-closed semantics are enforced in
/// the `extract_measurements` and `evaluate_ternary` methods, which
/// live in [`super::fail_closed`] so the fail-closed authority is
/// isolated to one place.
pub struct MeasuredEvaluatorAdapter {
    pub(crate) inner: Arc<dyn EcsEvaluationStrategy>,
    pub(crate) behavioral_probe: Option<Arc<dyn BehavioralProbe>>,
}

impl MeasuredEvaluatorAdapter {
    /// Create a new adapter wrapping an internal evaluator.
    pub fn new(evaluator: Arc<dyn EcsEvaluationStrategy>) -> Self {
        Self {
            inner: evaluator,
            behavioral_probe: None,
        }
    }

    /// Install a behavioral probe used for activation, logit, and
    /// router scoring.
    pub fn with_behavioral_probe(mut self, probe: Arc<dyn BehavioralProbe>) -> Self {
        self.behavioral_probe = Some(probe);
        self
    }

    /// Install the bounded SafeTensor-backed reference probe used by
    /// mapped progressive compilation. This is the production wiring
    /// point for behavior-aware ternary admission; callers can still
    /// provide a richer graph probe through `with_behavioral_probe`
    /// when available.
    pub fn with_mapped_tensor_probe(self, model_dir: impl Into<std::path::PathBuf>) -> Self {
        self.with_behavioral_probe(Arc::new(
            super::super::objective::MappedTensorBehavioralProbe::new(model_dir),
        ))
    }

    /// Check if this adapter wraps a synthetic evaluator. The check
    /// is by name convention — the search-system surface does not
    /// expose a capability flag, so this is the best signal available.
    pub fn is_synthetic(&self) -> bool {
        let name = self.inner.name();
        name.contains("Synthetic") || name.contains("synthetic")
    }
}

impl ProgressiveStageExecutor for MeasuredEvaluatorAdapter {
    fn evaluate(
        &self,
        genome: &CandidateGenome,
        _stage: usize,
        context: &[u8],
    ) -> TernaryObjectiveEvidence {
        // Fail-closed evidence composition lives in `super::fail_closed`;
        // any error there collapses to a missing-evidence marker so the
        // executor's contract (always return a value) is preserved.
        fail_closed::evaluate_ternary_evidence(self, genome, context)
            .unwrap_or_else(|_| TernaryObjectiveEvidence::missing())
    }
}

impl SearchEvaluationStrategy for MeasuredEvaluatorAdapter {
    fn evaluate(&self, genome: &str, context: &[u8]) -> Result<Vec<f64>, String> {
        let genome = parse_genome_from_string(genome).map_err(|e| e.to_string())?;
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

// ---------------------------------------------------------------------------
// MappedTensorEvaluationStrategy — search-system wrapper for the
// bounded reference probe
// ---------------------------------------------------------------------------

/// Search-facing adapter for mapped model directories. It makes the
/// reference probe available to the compiler's normal evolutionary
/// search trait, not only to the progressive ECS executor.
pub struct MappedTensorEvaluationStrategy {
    pub probe: super::super::objective::MappedTensorBehavioralProbe,
    pub limits: TernaryAdmissionLimits,
}

impl MappedTensorEvaluationStrategy {
    pub fn new(model_dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            probe: super::super::objective::MappedTensorBehavioralProbe::new(model_dir),
            limits: TernaryAdmissionLimits::default(),
        }
    }
}

impl SearchEvaluationStrategy for MappedTensorEvaluationStrategy {
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
        profile: WorkloadProfile,
    ) -> Result<crate::workload_search::WorkloadThroughputEvidence, String> {
        evaluate_workload_profile_impl(self, genome, context, profile)
    }

    fn evaluate_workload_profile_on_graph(
        &self,
        genome: &str,
        context: &[u8],
        profile: WorkloadProfile,
        graph: &SpatialGraph,
    ) -> Result<crate::workload_search::WorkloadThroughputEvidence, String> {
        evaluate_workload_profile_on_graph_impl(self, genome, context, profile, graph)
    }

    fn evaluate_backend(
        &self,
        backend: crate::search::SearchBackend,
        genome: &str,
        context: &[u8],
        _configuration: &crate::search::JointTilingConfiguration,
    ) -> Result<crate::search::BackendEvaluation, String> {
        evaluate_backend_impl(self, backend, genome, context)
    }
}

// ---------------------------------------------------------------------------
// Workload / backend evaluation plumbing — kept here because the
// wrapping strategy owns the dispatch shape; the underlying scoring
// primitive lives in `super::objective`.
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_lines)]
fn evaluate_workload_profile_impl(
    self_adapter: &MappedTensorEvaluationStrategy,
    genome: &str,
    context: &[u8],
    profile: WorkloadProfile,
) -> Result<crate::workload_search::WorkloadThroughputEvidence, String> {
    use prism_ecs_ir::evolution::RepresentationAxis;
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
        RepresentationAxis::Fp16 => "Fp16",
        RepresentationAxis::Bf16 => "Bf16",
        RepresentationAxis::Int8 => "Int8",
        RepresentationAxis::Int4 => "Int4",
        RepresentationAxis::Nf4 => "Nf4",
        RepresentationAxis::Ternary158 | RepresentationAxis::TernaryTile640 => "Ternary158",
        RepresentationAxis::Binary1 => "Binary1",
        RepresentationAxis::Nf8 => "Nf8",
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
            RepresentationAxis::Fp16 | RepresentationAxis::Bf16
        );
        let ane_penalty = if profile.primary_lane == crate::workload_search::ExecutionLane::Ane
            && !matches!(genome.representation, RepresentationAxis::Int8)
        {
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
                if self_adapter
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
                    if self_adapter
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
        projection_basis: "bounded mixed-precision graph and representative lane timings".into(),
        mixed_precision_graph: selected_graph,
        ..crate::workload_search::WorkloadThroughputEvidence::default()
    })
}

#[allow(clippy::too_many_lines)]
fn evaluate_workload_profile_on_graph_impl(
    self_adapter: &MappedTensorEvaluationStrategy,
    genome: &str,
    context: &[u8],
    profile: WorkloadProfile,
    graph: &SpatialGraph,
) -> Result<crate::workload_search::WorkloadThroughputEvidence, String> {
    if graph.node_count() == 0 {
        return Err("bounded workload evaluation requires a non-empty SpatialIR graph".into());
    }
    let mut evidence =
        evaluate_workload_profile_impl(self_adapter, genome, context, profile)?;
    // Exercise the same tinygrad-inspired UOp lowering used for CImage
    // emission. This is intentionally bounded: the canary evaluator
    // must not materialize an entire model graph just to validate
    // lowering.
    let lowering_target = if matches!(
        profile.primary_lane,
        crate::workload_search::ExecutionLane::Metal
    ) {
        LoweringTarget::Metal
    } else {
        LoweringTarget::Portable
    };
    let sampled_compute_nodes = graph
        .nodes()
        .iter()
        .filter(|node| matches!(node, SpatialNode::Compute { .. }))
        .count()
        .min(16);
    let strategies = if profile.interleaved_metal {
        vec![
            FusionStrategy::StandardFused,
            FusionStrategy::InterleavedFused { stages: Vec::new() },
            FusionStrategy::PerOperation,
        ]
    } else {
        vec![FusionStrategy::StandardFused, FusionStrategy::PerOperation]
    };
    let (lowered_nodes, lowering_failures, tiny_capture_digest, strategy_digests) =
        match crate::uop::compile_spatial_graph_strategies(graph, lowering_target, &strategies) {
            Ok(candidates) => {
                let candidate_count = candidates.len();
                if candidate_count == 0 {
                    return Err("tinygrad strategy lowering returned no executable candidate".into());
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
                    .filter(|node| matches!(node, SpatialNode::Compute { .. }))
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

#[allow(unused_variables)]
fn evaluate_backend_impl(
    self_adapter: &MappedTensorEvaluationStrategy,
    backend: crate::search::SearchBackend,
    genome: &str,
    context: &[u8],
) -> Result<crate::search::BackendEvaluation, String> {
    use super::progressive::reconstruct_representation;

    let (selection_budget, dispatch_budget) = if backend == crate::search::SearchBackend::Ane {
        (self_adapter.probe.max_elements.min(8_000_000), 1_000_000usize)
    } else {
        (self_adapter.probe.max_elements.min(4_000_000), 4_000_000usize)
    };
    if backend == crate::search::SearchBackend::Ane {
        #[cfg(feature = "ane")]
        {
            let tensor_name = self_adapter
                .probe
                .select_tensor_from_context(context, selection_budget)
                .map_err(|error| error.to_string())?;
            let (reference, shape) = self_adapter
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
    let tensor_name = self_adapter
        .probe
        .select_tensor_from_context(context, selection_budget)
        .map_err(|error| error.to_string())?;
    let (reference, shape) = self_adapter
        .probe
        .read_tensor(&tensor_name, dispatch_budget)
        .map_err(|e| e.to_string())?;
    let mut window = CanaryWindow::new(self_adapter.probe.max_elements.min(4 * 1024 * 1024));
    window
        .load(&reference)
        .map_err(|e| format!("canary window: {e}"))?;
    {
        let candidate_slot = window.candidate_mut();
        candidate_slot.copy_from_slice(&reference);
    }
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

// Re-exported for `super::fail_closed` use.
pub(crate) use super::super::objective::mapped_probe_evaluate_for_fail_closed;
