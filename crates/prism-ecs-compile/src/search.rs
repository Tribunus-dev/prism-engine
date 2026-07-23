pub use crate::SearchConfig;
use crate::{CandidateRecord, CandidateStatus, GenerationRecord, SearchTrace};
use prism_ecs_ir::evolution::{
    compile_plan::{FormatPlan, JointTilingPlan},
    emitters::EmitterKind,
    evaluate::EvaluationStrategy as EcsEvaluationStrategy,
    foundation::{CandidateGenome, FitnessScore, RepresentationAxis},
    frontier::ParetoFrontier,
    hierarchical::{FrozenHierarchy, HierarchicalStagePlan},
    joint::{JointEvolutionSystem, ScoredGenome},
    memory::{EvolutionContextKey, EvolutionReceipt, EvolutionaryMemory},
    objectives::{
        ArchiveEntry, BehaviorDescriptor, ObjectiveValue, ObjectiveVector, QualityDiversityArchive,
    },
    pareto::{DeploymentCandidate, DeploymentEvidence, DeploymentGatePolicy, DeploymentIdentity,
        DeploymentMeasurements, GateStatus, HardGate, ParetoArchive},
    progressive::ProgressiveStageExecutor,
    variation::VariationOperator,
};
use prism_ecs_source::CanonicalSource;
use prism_spatial_ir::graph::SpatialGraph;
use serde_json;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Instant;

pub struct SearchCoordinator {
    config: SearchConfig,
    trace: SearchTrace,
    memory: EvolutionaryMemory,
    surrogate: Option<Arc<dyn prism_ecs_ir::evolution::SurrogateModel>>,
    mutation_proposer: Option<Arc<dyn prism_ecs_ir::evolution::MutationProposer>>,
    runtime: prism_ecs_ir::evolution::EvolutionRuntime,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HeterogeneousScheduleEvidence {
    pub steps: usize,
    pub route_sequence: Vec<String>,
    pub zero_copy_steps: usize,
    pub estimated_latency_ns: u64,
    pub residency_windows: usize,
    pub supports_realtime_text: bool,
    pub supports_batched_text: bool,
    pub supports_batched_audio: bool,
}

/// Hardware backend evaluated by the evolutionary search.
///
/// This is deliberately local to the compiler search contract. It lets a
/// backend-aware evaluator distinguish ANE and Metal without coupling the
/// search record format to a particular kernel crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SearchBackend {
    Ane,
    Metal,
}

impl SearchBackend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ane => "ane",
            Self::Metal => "metal",
        }
    }
}

/// A joint ANE/Metal tile configuration considered for one candidate.
///
/// ANE and Metal intentionally have independent tile dimensions: the best
/// Core ML static shape is not necessarily the best Metal threadgroup/grid
/// shape. The candidate's Metal geometry supplies the threadgroup shape when
/// a profile is materialized.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct JointTilingConfiguration {
    pub ane_unit: prism_ecs_ir::evolution::foundation::AneUnitAxis,
    pub ane_tile_m: u32,
    pub ane_tile_n: u32,
    pub ane_tile_k: u32,
    pub metal_tile_m: u32,
    pub metal_tile_n: u32,
    pub metal_tile_k: u32,
    pub metal_threadgroup_width: u32,
    pub metal_threadgroup_height: u32,
}

impl JointTilingConfiguration {
    fn from_genome(genome: &CandidateGenome) -> Self {
        let geometry = &genome.metal_geometry;
        Self {
            ane_unit: genome.ane_unit.clone(),
            ane_tile_m: nearest_ane_tile(geometry.grid_tile_m),
            ane_tile_n: nearest_ane_tile(geometry.grid_tile_n),
            ane_tile_k: nearest_ane_tile(geometry.grid_tile_k),
            metal_tile_m: geometry.grid_tile_m,
            metal_tile_n: geometry.grid_tile_n,
            metal_tile_k: geometry.grid_tile_k,
            metal_threadgroup_width: geometry.threadgroup_width,
            metal_threadgroup_height: geometry.threadgroup_height,
        }
    }

    fn with_shape(
        genome: &CandidateGenome,
        ane_tile_m: u32,
        ane_tile_n: u32,
        ane_tile_k: u32,
        metal_tile_m: u32,
        metal_tile_n: u32,
        metal_tile_k: u32,
    ) -> Self {
        let geometry = &genome.metal_geometry;
        Self {
            ane_unit: genome.ane_unit.clone(),
            ane_tile_m,
            ane_tile_n,
            ane_tile_k,
            metal_tile_m,
            metal_tile_n,
            metal_tile_k,
            metal_threadgroup_width: geometry.threadgroup_width,
            metal_threadgroup_height: geometry.threadgroup_height,
        }
    }

    fn is_valid(self) -> bool {
        let metal_threadgroup_valid = prism_spatial_ir::validate_tiling_geometry(
            prism_spatial_ir::TileGeometry {
                width: self.metal_threadgroup_width as usize,
                height: self.metal_threadgroup_height as usize,
            },
            prism_spatial_ir::TilingBackend::Metal,
        )
        .is_ok();
        self.ane_tile_m > 0
            && self.ane_tile_n > 0
            && self.ane_tile_k > 0
            && self.metal_tile_m > 0
            && self.metal_tile_n > 0
            && self.metal_tile_k > 0
            && self.metal_tile_m <= 256
            && self.metal_tile_n <= 256
            && self.metal_tile_k <= 128
            && metal_threadgroup_valid
    }
}

/// Result returned by a backend-aware evaluator.
///
/// `performance_score` is the evaluator's normalized score, while
/// `wall_time_ms` is optional because an evaluator may report a device timer
/// instead of host elapsed time. No bandwidth or memory values are inferred
/// here.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct BackendEvaluation {
    pub performance_score: f64,
    pub feasible: bool,
    pub measured: bool,
    pub wall_time_ms: Option<f64>,
}

/// Per-backend evidence retained for one tiling profile.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BackendMeasurement {
    pub backend: SearchBackend,
    pub tiling: JointTilingConfiguration,
    pub feasible: bool,
    pub measured: bool,
    pub performance_score: Option<f64>,
    pub wall_time_ms: Option<f64>,
    pub evaluator: String,
    pub error: Option<String>,
}

/// Evidence for every joint tile profile tried for a candidate.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct JointTilingMeasurement {
    pub configuration: JointTilingConfiguration,
    pub ane: BackendMeasurement,
    pub metal: BackendMeasurement,
    pub both_feasible: bool,
    pub joint_score: Option<f64>,
}

/// Search-local evidence used to select a profile that works on both ANE and
/// Metal. It is serialized into each `CandidateRecord` measurement and is
/// also exposed directly on `SearchResult` for compiler callers.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct JointTilingEvidence {
    pub selected_configuration: Option<JointTilingConfiguration>,
    pub selected_score: Option<f64>,
    pub both_backends_feasible: bool,
    pub both_backends_measured: bool,
    pub profiles_evaluated: Vec<JointTilingMeasurement>,
}

impl JointTilingEvidence {
    /// True only when the selected ANE/Metal profile was actually measured on
    /// both lanes. Synthetic feasibility is intentionally insufficient for
    /// native ternary promotion.
    pub fn native_lane_ready(&self) -> bool {
        self.both_backends_feasible && self.both_backends_measured
    }

    /// Convert measured joint-lane evidence into the promotion contract.
    /// Search owns only ANE/Metal measurements; the remaining canary and
    /// replay results must be supplied by their respective validators and are
    /// never inferred from a performance score.
    pub fn native_promotion_evidence(
        &self,
        cpu_canary_passed: bool,
        accelerate_passed: bool,
        behavioral_reference_passed: bool,
        cimage_replay_passed: bool,
        ane_selected: bool,
        packed_abi_digest: impl Into<String>,
        reference_digest: impl Into<String>,
    ) -> prism_ecs_quantization::ternarization::promotion::NativeTernaryPromotionEvidence {
        use prism_ecs_quantization::ternarization::promotion::{
            BackendPass, NativeTernaryPromotionEvidence,
        };
        let lane_pass = BackendPass {
            attempted: self.native_lane_ready(),
            passed: self.native_lane_ready(),
        };
        NativeTernaryPromotionEvidence {
            cpu_canary: BackendPass {
                attempted: true,
                passed: cpu_canary_passed,
            },
            accelerate_reconstruction: BackendPass {
                attempted: true,
                passed: accelerate_passed,
            },
            metal_packed: lane_pass,
            ane_static: if ane_selected {
                lane_pass
            } else {
                BackendPass::unavailable()
            },
            cimage_replay: BackendPass {
                attempted: true,
                passed: cimage_replay_passed,
            },
            behavioral_reference: BackendPass {
                attempted: true,
                passed: behavioral_reference_passed,
            },
            activation_error: 0.0,
            router_agreement: 0.0,
            router_margin_error: 0.0,
            logit_cross_entropy: 0.0,
            generation_loss: 0.0,
            expert_balance_error: 0.0,
            ane_selected,
            packed_abi_digest: packed_abi_digest.into(),
            reference_digest: reference_digest.into(),
        }
    }
}

const JOINT_TILE_SHAPES: &[(u32, u32, u32, u32, u32, u32)] = &[
    (32, 32, 32, 32, 32, 32),
    (64, 64, 32, 64, 64, 32),
    (128, 64, 32, 128, 64, 32),
    (128, 128, 64, 256, 128, 64),
    (64, 64, 32, 128, 64, 32),
    (128, 64, 32, 64, 64, 32),
];

fn nearest_ane_tile(value: u32) -> u32 {
    [16, 32, 64, 128, 256]
        .into_iter()
        .min_by_key(|candidate| (*candidate as i64 - value as i64).unsigned_abs())
        .unwrap_or(32)
}

fn joint_tiling_configurations(genome: &CandidateGenome) -> Vec<JointTilingConfiguration> {
    let mut configurations = Vec::with_capacity(JOINT_TILE_SHAPES.len() + 1);
    configurations.push(JointTilingConfiguration::from_genome(genome));
    for ane_unit in [
        prism_ecs_ir::evolution::foundation::AneUnitAxis::Auto,
        prism_ecs_ir::evolution::foundation::AneUnitAxis::Planar,
        prism_ecs_ir::evolution::foundation::AneUnitAxis::Matrix,
    ] {
        let mut candidate = genome.clone();
        candidate.ane_unit = ane_unit;
        configurations.push(JointTilingConfiguration::from_genome(&candidate));
    }
    for &(ane_m, ane_n, ane_k, metal_m, metal_n, metal_k) in JOINT_TILE_SHAPES {
        configurations.push(JointTilingConfiguration::with_shape(
            genome, ane_m, ane_n, ane_k, metal_m, metal_n, metal_k,
        ));
    }
    configurations.sort_by_key(|configuration| {
        (
            configuration.ane_tile_m,
            configuration.ane_tile_n,
            configuration.ane_tile_k,
            format!("{:?}", configuration.ane_unit),
            configuration.metal_tile_m,
            configuration.metal_tile_n,
            configuration.metal_tile_k,
            configuration.metal_threadgroup_width,
            configuration.metal_threadgroup_height,
        )
    });
    configurations.dedup();
    configurations
        .into_iter()
        .filter(|configuration| configuration.is_valid())
        .collect()
}

fn seed_genome(index: usize) -> CandidateGenome {
    let mut genome = JointEvolutionSystem::codon_to_genome(index as u64);
    let profile = JOINT_TILE_SHAPES[index % JOINT_TILE_SHAPES.len()];
    genome.metal_geometry.grid_tile_m = profile.3;
    genome.metal_geometry.grid_tile_n = profile.4;
    genome.metal_geometry.grid_tile_k = profile.5;
    genome.metal_geometry.threadgroup_width =
        genome.metal_geometry.threadgroup_width.min(32).max(1);
    genome.metal_geometry.threadgroup_height =
        genome.metal_geometry.threadgroup_height.min(8).max(1);
    genome
}

fn harmonic_score(ane: f64, metal: f64) -> Option<f64> {
    if !ane.is_finite() || !metal.is_finite() || ane <= 0.0 || metal <= 0.0 {
        return None;
    }
    Some((2.0 * ane * metal / (ane + metal)).clamp(0.0, 1.0))
}

fn backend_measurement(
    evaluator: &dyn EvaluationStrategy,
    backend: SearchBackend,
    genome: &str,
    context: &[u8],
    configuration: JointTilingConfiguration,
) -> BackendMeasurement {
    let started = Instant::now();
    match evaluator.evaluate_backend(backend, genome, context, &configuration) {
        Ok(result) => {
            let wall_time_ms = result
                .wall_time_ms
                .filter(|value| value.is_finite() && *value >= 0.0)
                .or_else(|| Some(started.elapsed().as_secs_f64() * 1_000.0));
            let performance_score = result
                .performance_score
                .is_finite()
                .then_some(result.performance_score.clamp(0.0, 1.0));
            let feasible = result.feasible
                && performance_score.is_some_and(|score| score > 0.0)
                && configuration.is_valid();
            BackendMeasurement {
                backend,
                tiling: configuration,
                feasible,
                measured: result.measured,
                performance_score,
                wall_time_ms,
                evaluator: evaluator.name().to_string(),
                error: None,
            }
        }
        Err(error) => BackendMeasurement {
            backend,
            tiling: configuration,
            feasible: false,
            measured: false,
            performance_score: None,
            wall_time_ms: Some(started.elapsed().as_secs_f64() * 1_000.0),
            evaluator: evaluator.name().to_string(),
            error: Some(error),
        },
    }
}

fn evaluate_joint_tiling(
    evaluator: &dyn EvaluationStrategy,
    genome: &str,
    context: &[u8],
) -> JointTilingEvidence {
    let parsed_genome = serde_json::from_str::<CandidateGenome>(genome).unwrap_or_default();
    let profiles_evaluated = joint_tiling_configurations(&parsed_genome)
        .into_iter()
        .map(|configuration| {
            let ane = backend_measurement(
                evaluator,
                SearchBackend::Ane,
                genome,
                context,
                configuration,
            );
            let metal = backend_measurement(
                evaluator,
                SearchBackend::Metal,
                genome,
                context,
                configuration,
            );
            let both_feasible = ane.feasible && metal.feasible;
            let joint_score = both_feasible
                .then(|| harmonic_score(ane.performance_score?, metal.performance_score?))
                .flatten();
            JointTilingMeasurement {
                configuration,
                ane,
                metal,
                both_feasible,
                joint_score,
            }
        })
        .collect::<Vec<_>>();

    let selected = profiles_evaluated
        .iter()
        .filter(|measurement| measurement.both_feasible)
        .max_by(|left, right| {
            left.joint_score
                .partial_cmp(&right.joint_score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    let left_latency = left.ane.wall_time_ms.unwrap_or(f64::INFINITY)
                        + left.metal.wall_time_ms.unwrap_or(f64::INFINITY);
                    let right_latency = right.ane.wall_time_ms.unwrap_or(f64::INFINITY)
                        + right.metal.wall_time_ms.unwrap_or(f64::INFINITY);
                    right_latency
                        .partial_cmp(&left_latency)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

    JointTilingEvidence {
        selected_configuration: selected.map(|measurement| measurement.configuration),
        selected_score: selected.and_then(|measurement| measurement.joint_score),
        both_backends_feasible: selected.is_some(),
        both_backends_measured: selected
            .is_some_and(|measurement| measurement.ane.measured && measurement.metal.measured),
        profiles_evaluated,
    }
}

fn candidate_measurements(evidence: &JointTilingEvidence) -> serde_json::Value {
    let wall_time_ms: f64 = evidence
        .profiles_evaluated
        .iter()
        .filter(|measurement| {
            measurement.configuration
                == evidence
                    .selected_configuration
                    .unwrap_or(measurement.configuration)
        })
        .flat_map(|measurement| [measurement.ane.wall_time_ms, measurement.metal.wall_time_ms])
        .flatten()
        .sum();
    serde_json::json!({
        "wall_time_ms": wall_time_ms,
        "gpu_time_ms": 0.0,
        "bandwidth_gbps": 0.0,
        "peak_memory_mb": 0.0,
        "reconstruction_error": evidence.selected_score.map(|score| 1.0 - score),
        "accuracy_score": evidence.selected_score,
        "measurement_kind": "joint_ane_metal_tiling",
        "joint_tiling": evidence,
    })
}

/// Promote the search trace's measured records into the durable deployment
/// archive. This is deliberately done after the existing joint evaluator has
/// produced its evidence, so the deployment archive cannot accidentally admit
/// a surrogate-only score as hardware measurement.
fn deployment_archive_from_records(
    source: &CanonicalSource,
    evaluator: &dyn EvaluationStrategy,
    workload_digest: &str,
    records: &[CandidateRecord],
    gate_policy: &DeploymentGatePolicy,
) -> ParetoArchive {
    let mut archive = ParetoArchive::default();
    for record in records {
        let Ok(genome) = serde_json::from_str::<CandidateGenome>(&record.genome) else { continue };
        let Some(measurements) = record.measurements.as_ref() else { continue };
        let engram_digest = sha256_digest(
            &serde_json::to_string(&genome.engram).unwrap_or_default(),
        );
        let number = |name: &str| measurements.get(name).and_then(serde_json::Value::as_f64);
        let deployment_measurements = DeploymentMeasurements {
            quality: number("accuracy_score"),
            p50_latency_ms: number("wall_time_ms"),
            p99_latency_ms: number("wall_time_ms"),
            throughput_tokens_per_second: None,
            peak_memory_bytes: number("peak_memory_mb").map(|v| (v * 1024.0 * 1024.0) as u64),
            kv_memory_bytes: None,
            power_watts: None,
            transfer_bytes: None,
            engram_residency_bytes: None,
            engram_lookup_latency_ms: None,
            engram_hit_rate: None,
        };
        let status = match record.status {
            CandidateStatus::Evaluated => GateStatus::Passed,
            CandidateStatus::Rejected | CandidateStatus::Failed => GateStatus::Failed,
        };
        let mut candidate = DeploymentCandidate::new(
            DeploymentIdentity {
                model_digest: source.identity.source_digest.clone(),
                tokenizer_digest: "unresolved".into(),
                engram_artifact: Some(engram_digest),
                target: evaluator.name().into(),
                workload_digest: workload_digest.into(),
            },
            genome,
            0,
        );
        candidate.candidate_digest = record.candidate_digest.clone();
        let mut gates = gate_policy.evaluate(&deployment_measurements);
        gates.push(HardGate {
            name: "joint_backend_execution".into(),
            status,
            observed: record.score_vector.first().copied(),
            limit: None,
            detail: record.rejection_reason.clone().unwrap_or_else(|| "candidate evaluated".into()),
        });
        candidate.evidence = DeploymentEvidence {
            candidate_digest: record.candidate_digest.clone(),
            cimage_digest: None,
            compiler_version: env!("CARGO_PKG_VERSION").into(),
            backend_version: evaluator.name().into(),
            measurements: deployment_measurements,
            gates,
            receipt_ids: Vec::new(),
        };
        archive.insert(candidate);
    }
    archive
}

struct DefaultSearchEvaluator;

struct SearchTilingCostModel;

impl prism_spatial_ir::cost::CostModel for SearchTilingCostModel {
    fn estimate(&self, graph: &SpatialGraph) -> prism_spatial_ir::cost::CostEstimate {
        prism_spatial_ir::cost::CostEstimate {
            latency: std::time::Duration::from_nanos((graph.node_count().max(1) * 1_000) as u64),
            peak_memory: graph.node_count().max(1) as u64,
            materialized_bytes: 0,
            sync_count: graph.edge_count() as u32,
            energy: 0.0,
            error: 0.0,
        }
    }

    fn name(&self) -> &str {
        "search-joint-tiling-baseline"
    }
}

fn joint_tiling_fitness(graph: &SpatialGraph, genome: &CandidateGenome) -> f64 {
    let g = &genome.metal_geometry;
    let candidates = [
        prism_spatial_ir::graph::TileGeometry {
            width: g.threadgroup_width as usize,
            height: g.threadgroup_height as usize,
        },
        prism_spatial_ir::graph::TileGeometry {
            width: 32,
            height: 8,
        },
        prism_spatial_ir::graph::TileGeometry {
            width: 16,
            height: 16,
        },
        prism_spatial_ir::graph::TileGeometry {
            width: 64,
            height: 4,
        },
        prism_spatial_ir::graph::TileGeometry {
            width: 8,
            height: 8,
        },
    ];
    let model = SearchTilingCostModel;
    let Ok(best) = prism_spatial_ir::cost::select_best_joint_tiling(&model, graph, &candidates)
    else {
        return 0.0;
    };
    (1.0 / (1.0 + best.fitness_score())).clamp(0.0, 1.0)
}

fn schedule_fitness(graph: &SpatialGraph, tensor_keys: &[String], genome: &CandidateGenome) -> f64 {
    let format_plan = FormatPlan::from_best_genome(genome, tensor_keys);
    let Some(manifest) = prism_spatial_ir::execution_plan::lower_to_manifest(
        graph,
        prism_spatial_ir::cost::CostEstimate::zero(),
        Some(&format_plan),
    ) else {
        return 0.0;
    };
    let Some(plan) = manifest.batch_plan else {
        return 0.0;
    };
    if plan.fused_steps.is_empty() || !plan.supports_all_streamed_workloads() {
        return 0.0;
    }

    let transitions = plan
        .fused_steps
        .windows(2)
        .filter(|steps| steps[0].backend != steps[1].backend)
        .count() as f64;
    let zero_copy = plan
        .fused_steps
        .iter()
        .filter(|step| step.zero_copy)
        .count() as f64;
    let latency = plan
        .fused_steps
        .iter()
        .map(|step| step.estimated_latency_ns)
        .sum::<u64>() as f64;
    let latency_score = 1.0 / (1.0 + latency / 1_000_000.0);
    let interleave_score = (transitions / plan.fused_steps.len() as f64).min(1.0);
    let zero_copy_score = (zero_copy / plan.fused_steps.len() as f64).min(1.0);
    let tiling_score = joint_tiling_fitness(graph, genome);
    (0.35 * latency_score + 0.25 * interleave_score + 0.20 * zero_copy_score + 0.20 * tiling_score)
        .clamp(0.0, 1.0)
}

impl EvaluationStrategy for DefaultSearchEvaluator {
    fn evaluate(&self, _genome: &str, context: &[u8]) -> Result<Vec<f64>, String> {
        Ok(vec![1.0 / (1.0 + context.len() as f64)])
    }

    fn name(&self) -> &str {
        "synthetic-fallback"
    }
}

impl SearchCoordinator {
    pub fn new(config: SearchConfig) -> Self {
        Self {
            trace: SearchTrace {
                search_id: uuid::Uuid::new_v4().to_string(),
                config: config.clone(),
                generations: Vec::new(),
                pareto_frontier: Vec::new(),
                quality_diversity_archive: Vec::new(),
                best_genome: None,
                trace_digest: String::new(),
            },
            config,
            memory: EvolutionaryMemory::default(),
            surrogate: None,
            mutation_proposer: None,
            runtime: prism_ecs_ir::evolution::EvolutionRuntime::global(),
        }
    }

    pub fn with_runtime(mut self, runtime: prism_ecs_ir::evolution::EvolutionRuntime) -> Self {
        self.runtime = runtime;
        self
    }

    pub fn with_surrogate(
        mut self,
        surrogate: Arc<dyn prism_ecs_ir::evolution::SurrogateModel>,
    ) -> Self {
        self.surrogate = Some(surrogate);
        self
    }

    pub fn with_mutation_proposer(
        mut self,
        proposer: Arc<dyn prism_ecs_ir::evolution::MutationProposer>,
    ) -> Self {
        self.mutation_proposer = Some(proposer);
        self
    }

    pub fn run_search(
        &mut self,
        source: &CanonicalSource,
        graph: &SpatialGraph,
        evaluator: Option<&dyn EvaluationStrategy>,
        production_mode: bool,
    ) -> Result<SearchResult, SearchError> {
        if production_mode && evaluator.is_none() {
            return Err(SearchError::ProductionModeRequiresEvaluator);
        }

        if self.config.max_generations == 0 {
            return Ok(SearchResult {
                format_plan: None,
                trace: self.trace.clone(),
                candidates_evaluated: 0,
                generations_completed: 0,
                evaluation_route: "none".into(),
                heterogeneous_schedule: None,
                best_joint_tiling: None,
                evolution_memory: EvolutionaryMemory::default(),
                selection_receipt: SearchSelectionReceipt::build(
                    self.trace.search_id.clone(),
                    "none",
                    false,
                    &[],
                    None,
                ),
                deployment_archive: ParetoArchive::default(),
            });
        }

        let tensor_keys: Vec<String> = source
            .catalog
            .tensors
            .iter()
            .map(|t| t.name.clone())
            .collect();
        if tensor_keys.is_empty() {
            return Err(SearchError::NoTensors);
        }

        let context_str = build_search_context(source, graph);
        let context_bytes = context_str.as_bytes();
        let model_family = if source.identity.model_family.is_empty() {
            source.identity.architecture.clone()
        } else {
            source.identity.model_family.clone()
        };

        let default_evaluator;
        let evaluator = match evaluator {
            Some(evaluator) => evaluator,
            None if !production_mode => {
                default_evaluator = DefaultSearchEvaluator;
                &default_evaluator as &dyn EvaluationStrategy
            }
            None => return Err(SearchError::ProductionModeRequiresEvaluator),
        };
        if production_mode && !evaluator.is_measured() {
            return Err(SearchError::ProductionModeRequiresEvaluator);
        }

        let runtime_session = self
            .runtime
            .begin_session(model_family.clone(), evaluator.name());

        struct EvaluatorAdapter<'a> {
            inner: &'a dyn EvaluationStrategy,
            graph: &'a SpatialGraph,
            tensor_keys: &'a [String],
            joint_evidence: Mutex<HashMap<String, JointTilingEvidence>>,
        }

        impl<'a> EvaluatorAdapter<'a> {
            fn evidence_for(&self, genome: &str, context: &[u8]) -> JointTilingEvidence {
                let cache_key = format!(
                    "{}:{}",
                    sha256_digest(genome),
                    sha256_digest(&String::from_utf8_lossy(context))
                );
                if let Ok(cache) = self.joint_evidence.lock() {
                    if let Some(evidence) = cache.get(&cache_key) {
                        return evidence.clone();
                    }
                }

                let evidence = evaluate_joint_tiling(self.inner, genome, context);
                if let Ok(mut cache) = self.joint_evidence.lock() {
                    cache.insert(cache_key, evidence.clone());
                }
                evidence
            }
        }

        impl<'a> EcsEvaluationStrategy for EvaluatorAdapter<'a> {
            fn evaluate(&self, genome: &CandidateGenome, context: &[u8]) -> FitnessScore {
                let genome_str = serde_json::to_string(genome).unwrap_or_default();
                let evidence = self.evidence_for(&genome_str, context);
                let joint_score = evidence.selected_score.unwrap_or(0.0);
                let schedule_score = schedule_fitness(self.graph, self.tensor_keys, genome);
                // The harmonic backend score is the primary objective: an
                // excellent Metal result cannot hide an infeasible ANE
                // profile (or vice versa). The schedule term is a bounded
                // tie-breaker for the fused route.
                FitnessScore::new(joint_score + 0.1 * schedule_score)
            }

            fn name(&self) -> &str {
                self.inner.name()
            }
        }

        let adapter = EvaluatorAdapter {
            inner: evaluator,
            graph,
            tensor_keys: &tensor_keys,
            joint_evidence: Mutex::new(HashMap::new()),
        };

        let joint_config = prism_ecs_ir::evolution::joint::JointSearchConfig {
            population_size: self.config.population_size as usize,
            max_generations: self.config.max_generations as usize,
            stagnation_limit: self.config.early_stop_generations as usize,
            crossover_rate: 0.7,
            mutation_rate: 0.1,
            seed: None,
        };

        let joint_system = JointEvolutionSystem::new(joint_config);
        for operator in [
            VariationOperator::Representation,
            VariationOperator::Packing,
            VariationOperator::Geometry,
            VariationOperator::Decomposition,
            VariationOperator::Memory,
            VariationOperator::Fusion,
            VariationOperator::Runtime,
            VariationOperator::AneUnit,
        ] {
            if let Some(controller) = self.runtime.variation_controller(operator) {
                joint_system.merge_variation_controller(&controller);
            }
        }

        let mut population: Vec<ScoredGenome> = Vec::new();

        // Create a map to track parent relationships for lineage tracking
        let mut parent_map: HashMap<String, Vec<String>> = HashMap::new();
        let mut operator_map: HashMap<String, VariationOperator> = HashMap::new();
        let mut emitted_kinds: HashMap<String, EmitterKind> = HashMap::new();

        // Initialize population with systematic seeds plus optional semantic
        // proposals. Proposals and surrogate predictions are advisory; every
        // candidate still passes through the ordinary hardware evaluator.
        let archived_elites = self
            .runtime
            .execution_elites(self.config.population_size as usize);
        let mut initial_genomes = archived_elites;
        if let Some(first_tensor) = tensor_keys.first() {
            let tensor_family =
                prism_ecs_ir::evolution::compile_plan::classify_tensor(first_tensor);
            initial_genomes.extend(self.runtime.tensor_elites(&tensor_family, 4));
        }
        for backend in [SearchBackend::Ane, SearchBackend::Metal] {
            initial_genomes.extend(self.runtime.hardware_elites(
                &prism_ecs_ir::evolution::HardwareProfileKey {
                    backend: backend.as_str().into(),
                    device: evaluator.name().into(),
                    driver: "evaluator-reported".into(),
                },
                2,
            ));
        }
        for index in 0..self.config.population_size as usize {
            initial_genomes.push(seed_genome(index));
        }
        if let Some(proposer) = &self.mutation_proposer {
            let seed_snapshot = initial_genomes.clone();
            for genome in seed_snapshot.iter().take(4) {
                let proposal_context = EvolutionContextKey {
                    hardware: evaluator.name().to_string(),
                    model_family: model_family.clone(),
                    tensor_family: tensor_keys
                        .first()
                        .map(|key| {
                            prism_ecs_ir::evolution::compile_plan::classify_tensor(key).to_string()
                        })
                        .unwrap_or_else(|| "unknown".into()),
                };
                let mut receipts = self
                    .memory
                    .successful_mutations(&proposal_context, 0.0)
                    .cloned()
                    .collect::<Vec<_>>();
                receipts.extend(self.runtime.successful_receipts(&proposal_context, 0.0));
                receipts.sort_by(|left, right| right.improvement.total_cmp(&left.improvement));
                receipts.dedup_by(|left, right| {
                    left.parent_digest == right.parent_digest
                        && left.child_digest == right.child_digest
                        && left.measurement_receipt_digest == right.measurement_receipt_digest
                });
                for proposal in proposer.propose(genome, context_bytes, &receipts) {
                    let digest =
                        sha256_digest(&serde_json::to_string(&proposal.genome).unwrap_or_default());
                    emitted_kinds.insert(digest, EmitterKind::Semantic);
                    initial_genomes.push(proposal.genome);
                }
            }
        }
        let seed_snapshot = initial_genomes.clone();
        let emitter_context = EvolutionContextKey {
            hardware: evaluator.name().to_string(),
            model_family: model_family.clone(),
            tensor_family: tensor_keys
                .first()
                .map(|key| prism_ecs_ir::evolution::compile_plan::classify_tensor(key).to_string())
                .unwrap_or_else(|| "unknown".into()),
        };
        if let Some(controller) = self
            .runtime
            .contextual_variation_controller(&emitter_context)
        {
            joint_system.merge_variation_controller(&controller);
        }
        if self.surrogate.is_none() {
            let surrogate = self
                .runtime
                .receipt_surrogate(&emitter_context, 0.0, 10_000);
            if !surrogate.observations.is_empty() {
                self.surrogate = Some(Arc::new(Mutex::new(surrogate)));
            }
        }
        for genome in seed_snapshot.iter().take(4) {
            let (emitter, emitted) =
                self.runtime
                    .emit_candidates_for_context(genome, context_bytes, &emitter_context);
            for candidate in &emitted {
                emitted_kinds.insert(
                    sha256_digest(&serde_json::to_string(candidate).unwrap_or_default()),
                    emitter,
                );
            }
            initial_genomes.extend(emitted);
        }
        let mut seen_initial = std::collections::HashSet::new();
        initial_genomes.retain(|genome| {
            seen_initial.insert(sha256_digest(
                &serde_json::to_string(genome).unwrap_or_default(),
            ))
        });
        if let Some(surrogate) = &self.surrogate {
            initial_genomes.sort_by(|left, right| {
                let left_score = surrogate
                    .predict_with_uncertainty(left, context_bytes)
                    .map(|prediction| {
                        prediction.objectives.scalar_compatibility_score()
                            + 0.05 * prediction.uncertainty
                    })
                    .unwrap_or(0.0);
                let right_score = surrogate
                    .predict_with_uncertainty(right, context_bytes)
                    .map(|prediction| {
                        prediction.objectives.scalar_compatibility_score()
                            + 0.05 * prediction.uncertainty
                    })
                    .unwrap_or(0.0);
                right_score.total_cmp(&left_score)
            });
        }
        initial_genomes.truncate(self.config.population_size as usize);
        let mut emitted_scores: Vec<(EmitterKind, f64)> = Vec::new();
        for genome in initial_genomes {
            let scored = joint_system.estimate_and_score_genome(&genome, &adapter, context_bytes);
            if let Some(emitter) = emitted_kinds.get(&sha256_digest(
                &serde_json::to_string(&scored.genome).unwrap_or_default(),
            )) {
                emitted_scores.push((*emitter, scored.fitness.first().copied().unwrap_or(0.0)));
            }

            // Create a digest for this candidate
            let genome_str = serde_json::to_string(&scored.genome).unwrap_or_default();
            let candidate_digest = format!("gen0-{}", sha256_digest(&genome_str));

            // Store initial population with no parents
            parent_map.insert(candidate_digest.clone(), Vec::new());
            operator_map.insert(candidate_digest, VariationOperator::Unknown);

            population.push(scored);
        }
        let initial_baseline = population
            .iter()
            .filter_map(|candidate| candidate.fitness.first().copied())
            .sum::<f64>()
            / population.len().max(1) as f64;
        for (emitter, score) in emitted_scores {
            let reward = score - initial_baseline;
            self.runtime.record_emitter_reward(emitter, reward);
            self.runtime
                .record_contextual_emitter_reward(emitter_context.clone(), emitter, reward);
        }

        // Preserve the measured hardware dimensions in the live frontier;
        // scalar compatibility remains available on ScoredGenome, but is not
        // allowed to collapse the joint search to one objective.
        let mut frontier = ParetoFrontier::new(4);
        let mut quality_archive = QualityDiversityArchive::default();
        // Hydrate the live archive with persistent elites before measuring the
        // new generation. This preserves prior niches, novelty pressure, and
        // archive-first final selection across independent compile sessions.
        for entry in self
            .runtime
            .execution_archive_entries(self.config.population_size as usize)
        {
            quality_archive.insert(entry);
        }
        if let Some(first_tensor) = tensor_keys.first() {
            let tensor_family =
                prism_ecs_ir::evolution::compile_plan::classify_tensor(first_tensor);
            for entry in self
                .runtime
                .tensor_archive_entries(&tensor_family, self.config.population_size as usize)
            {
                quality_archive.insert(entry);
            }
        }
        for backend in [SearchBackend::Ane, SearchBackend::Metal] {
            let profile = prism_ecs_ir::evolution::HardwareProfileKey {
                backend: backend.as_str().into(),
                device: evaluator.name().into(),
                driver: "evaluator-reported".into(),
            };
            for entry in self.runtime.hardware_archive_entries(&profile) {
                quality_archive.insert(entry);
            }
        }
        let mut all_candidates: Vec<CandidateRecord> = Vec::new();
        let mut gen_records: Vec<GenerationRecord> = Vec::new();
        let mut digest_to_genome: HashMap<String, CandidateGenome> = HashMap::new();
        let mut previous_archive_cells = 0usize;

        for gen in 0..self.config.max_generations {
            let mut generation_candidates: Vec<CandidateRecord> = Vec::new();

            for (idx, scored_genome) in population.iter().enumerate() {
                let genome_str = serde_json::to_string(&scored_genome.genome).unwrap_or_default();
                let candidate_digest = format!("gen{}-{}", gen, sha256_digest(&genome_str));

                // Get parent digests from parent map
                let parent_digests = parent_map
                    .get(&candidate_digest)
                    .cloned()
                    .unwrap_or_default();
                let candidate_operator = operator_map
                    .get(&candidate_digest)
                    .copied()
                    .unwrap_or(VariationOperator::Unknown);

                // Store genome for lineage tracking
                digest_to_genome.insert(candidate_digest.clone(), scored_genome.genome.clone());

                // The evaluator adapter measured every backend profile during
                // scoring. Reuse that cached result so the durable trace
                // describes exactly what drove selection.
                let joint_evidence = adapter.evidence_for(&genome_str, context_bytes);
                let measurements = Some(candidate_measurements(&joint_evidence));
                let candidate_is_feasible = joint_evidence.both_backends_feasible;
                let candidate_is_measured = joint_evidence.both_backends_measured;
                let schedule_score = schedule_fitness(graph, &tensor_keys, &scored_genome.genome);
                let selected_latency =
                    joint_evidence
                        .selected_configuration
                        .and_then(|configuration| {
                            joint_evidence
                                .profiles_evaluated
                                .iter()
                                .find(|profile| profile.configuration == configuration)
                                .and_then(|profile| {
                                    Some(
                                        profile.ane.wall_time_ms.unwrap_or(0.0)
                                            + profile.metal.wall_time_ms.unwrap_or(0.0),
                                    )
                                })
                        });
                let selected_profile =
                    joint_evidence
                        .selected_configuration
                        .and_then(|configuration| {
                            joint_evidence
                                .profiles_evaluated
                                .iter()
                                .find(|profile| profile.configuration == configuration)
                        });
                let selected_ane_score = selected_profile
                    .and_then(|profile| profile.ane.performance_score)
                    .unwrap_or(0.0);
                let selected_metal_score = selected_profile
                    .and_then(|profile| profile.metal.performance_score)
                    .unwrap_or(0.0);
                let selected_ane_latency = selected_profile
                    .and_then(|profile| profile.ane.wall_time_ms)
                    .unwrap_or(f64::MAX);
                let selected_metal_latency = selected_profile
                    .and_then(|profile| profile.metal.wall_time_ms)
                    .unwrap_or(f64::MAX);
                let behavior_descriptor = BehaviorDescriptor::from_execution(
                    &scored_genome.genome,
                    selected_ane_score,
                    selected_metal_score,
                    candidate_is_measured,
                    selected_latency,
                );

                frontier.insert(
                    prism_ecs_core::Entity::new(idx as u64, 0),
                    vec![
                        FitnessScore::new(joint_evidence.selected_score.unwrap_or(0.0)),
                        FitnessScore::new(schedule_score),
                        FitnessScore::new(
                            selected_latency
                                .map(|latency| 1.0 / (1.0 + latency))
                                .unwrap_or(0.0),
                        ),
                        FitnessScore::new(if candidate_is_feasible { 1.0 } else { 0.0 }),
                    ],
                    gen as u64,
                    &Default::default(),
                );

                // Preserve independent evidence dimensions in the live
                // quality-diversity archive. The scalar frontier remains for
                // compatibility and final plan selection.
                let objectives = ObjectiveVector::new(vec![
                    ObjectiveValue::maximize(
                        "joint_backend_fidelity",
                        joint_evidence.selected_score.unwrap_or(0.0),
                    ),
                    ObjectiveValue::maximize("schedule_quality", schedule_score),
                    ObjectiveValue::minimize(
                        "measured_latency_ms",
                        selected_latency.unwrap_or(f64::MAX),
                    ),
                    ObjectiveValue::maximize(
                        "joint_feasibility",
                        if candidate_is_feasible { 1.0 } else { 0.0 },
                    ),
                    ObjectiveValue::maximize("ane_fidelity", selected_ane_score),
                    ObjectiveValue::maximize("metal_fidelity", selected_metal_score),
                    ObjectiveValue::minimize("ane_latency_ms", selected_ane_latency),
                    ObjectiveValue::minimize("metal_latency_ms", selected_metal_latency),
                ]);
                if candidate_is_measured && candidate_is_feasible {
                    if let Some(surrogate) = &self.surrogate {
                        surrogate.observe(&scored_genome.genome, objectives.clone());
                    }
                }
                quality_archive.insert(ArchiveEntry {
                    genome: scored_genome.genome.clone(),
                    objectives,
                    descriptor: behavior_descriptor,
                    generation: gen as u64,
                    novelty: 0.0,
                });

                // Validate that measurements are present and valid for production mode
                if production_mode {
                    // Ensure measurements are present
                    if measurements.is_none() {
                        return Err(SearchError::MissingMeasurements);
                    }
                    if !candidate_is_feasible {
                        return Err(SearchError::CorrectnessValidationFailed);
                    }
                    if !candidate_is_measured {
                        return Err(SearchError::SyntheticDataInProductionMode);
                    }
                }

                let candidate_record = CandidateRecord {
                    candidate_digest: candidate_digest.clone(),
                    parent_digests: parent_digests.clone(),
                    genome: genome_str.clone(),
                    tensor_scope: tensor_keys.clone(),
                    score_vector: scored_genome.fitness.clone(),
                    measurements: measurements.clone(),
                    status: if candidate_is_feasible {
                        CandidateStatus::Evaluated
                    } else {
                        CandidateStatus::Rejected
                    },
                    rejection_reason: (!candidate_is_feasible)
                        .then(|| "no jointly feasible ANE/Metal tiling profile".to_string()),
                };

                for parent_digest in &parent_digests {
                    self.runtime
                        .record_lineage(prism_ecs_ir::evolution::LineageRecord {
                            session_id: self.trace.search_id.clone(),
                            parent_digest: Some(parent_digest.clone()),
                            child_digest: candidate_digest.clone(),
                            operator: candidate_operator,
                            generation: gen as u64,
                        });
                }

                // Persist only candidates with known lineage. Seed candidates
                // are intentionally excluded because there is no mutation to
                // attribute. Hardware evidence remains the authority for the
                // recorded improvement.
                if let Some(parent_digest) = parent_digests.first() {
                    let parent_score = all_candidates
                        .iter()
                        .find(|candidate| &candidate.candidate_digest == parent_digest)
                        .and_then(|candidate| candidate.score_vector.first().copied())
                        .unwrap_or(0.0);
                    let child_score = scored_genome.fitness.first().copied().unwrap_or(0.0);
                    let receipt = EvolutionReceipt {
                        parent_digest: parent_digest.clone(),
                        child_digest: genome_str.clone(),
                        operator: candidate_operator,
                        context: EvolutionContextKey {
                            hardware: evaluator.name().to_string(),
                            model_family: model_family.clone(),
                            tensor_family: tensor_keys
                                .first()
                                .map(|key| {
                                    prism_ecs_ir::evolution::compile_plan::classify_tensor(key)
                                        .to_string()
                                })
                                .unwrap_or_else(|| "unknown".into()),
                        },
                        descriptor: BehaviorDescriptor::from_genome(&scored_genome.genome),
                        objectives: ObjectiveVector::new(vec![ObjectiveValue::maximize(
                            "candidate_fitness",
                            child_score,
                        )]),
                        improvement: child_score - parent_score,
                        measurement_receipt_digest: sha256_digest(
                            &serde_json::to_string(&measurements).unwrap_or_default(),
                        ),
                    };
                    self.memory.record(receipt.clone());
                    self.runtime.record_receipt(receipt);
                }

                generation_candidates.push(candidate_record.clone());
                all_candidates.push(candidate_record);
            }

            let best_score = frontier
                .best_by_dimension(0)
                .map(|entry| {
                    entry
                        .fitness
                        .first()
                        .map(|score| score.value())
                        .unwrap_or(0.0)
                })
                .unwrap_or(0.0);

            let diversity = if generation_candidates.is_empty() {
                0.0
            } else {
                quality_archive.cells.len() as f64 / generation_candidates.len() as f64
            }
            .clamp(0.0, 1.0);
            gen_records.push(GenerationRecord {
                generation: gen,
                candidates: all_candidates.clone(),
                best_score,
                diversity,
                timestamp: chrono::Utc::now(),
            });

            let archive_grew = quality_archive.cells.len() > previous_archive_cells;
            if gen > 0 && gen > self.config.early_stop_generations {
                let prev_best = gen_records[(gen - 1) as usize].best_score;
                if best_score <= prev_best && !archive_grew {
                    break;
                }
            }
            previous_archive_cells = quality_archive.cells.len();

            // Run next generation with proper parent tracking
            let (mut next_population_genomes, stop_reason, mut operators, mut parent_genomes) =
                joint_system.run_generation_with_feedback(
                    &population,
                    &frontier,
                    Some(&quality_archive),
                );

            if stop_reason.is_some() {
                break;
            }

            // A surrogate is advisory during cold start, but once it has
            // enough measured neighbors it becomes a hardware-evaluation
            // gate. Keep the most promising candidates and preserve an
            // uncertainty bonus so the queue does not collapse onto one
            // already-known region of the search space.
            if let Some(surrogate) = &self.surrogate {
                if surrogate.observations() >= 8 && next_population_genomes.len() > 1 {
                    let mut ranked_indices: Vec<usize> =
                        (0..next_population_genomes.len()).collect();
                    ranked_indices.sort_by(|&left, &right| {
                        let score = |candidate: &CandidateGenome| {
                            surrogate
                                .predict_with_uncertainty(candidate, context_bytes)
                                .map(|prediction| {
                                    prediction.objectives.scalar_compatibility_score()
                                        + 0.05 * prediction.uncertainty
                                })
                                .unwrap_or(f64::NEG_INFINITY)
                        };
                        score(&next_population_genomes[right].genome)
                            .total_cmp(&score(&next_population_genomes[left].genome))
                    });
                    let fraction = self.config.effective_surrogate_measurement_fraction();
                    let measurement_budget =
                        ((next_population_genomes.len() as f64 * fraction).ceil() as usize).max(1);
                    ranked_indices.truncate(measurement_budget);
                    let mut selected_genomes = Vec::with_capacity(ranked_indices.len());
                    let mut selected_operators = Vec::with_capacity(ranked_indices.len());
                    let mut selected_parents = Vec::with_capacity(ranked_indices.len());
                    for index in ranked_indices {
                        selected_genomes.push(next_population_genomes[index].clone());
                        selected_operators.push(operators[index]);
                        selected_parents.push(parent_genomes[index].clone());
                    }
                    next_population_genomes = selected_genomes;
                    operators = selected_operators;
                    parent_genomes = selected_parents;
                }
            }

            // Re-score the next generation population
            let mut next_population: Vec<ScoredGenome> = Vec::new();
            let parent_baseline = population
                .iter()
                .filter_map(|candidate| candidate.fitness.first().copied())
                .sum::<f64>()
                / population.len().max(1) as f64;
            let mut next_parent_map: HashMap<String, Vec<String>> = HashMap::new();
            let mut next_operator_map: HashMap<String, VariationOperator> = HashMap::new();
            for ((genome, operator), parents) in next_population_genomes
                .into_iter()
                .zip(operators.into_iter())
                .zip(parent_genomes.into_iter())
            {
                let scored =
                    joint_system.estimate_and_score_genome(&genome.genome, &adapter, context_bytes);
                let child_score = scored.fitness.first().copied().unwrap_or(0.0);
                joint_system.record_operator_feedback(operator, child_score - parent_baseline);
                self.runtime
                    .record_operator_reward(operator, child_score - parent_baseline);
                self.runtime.record_contextual_operator_reward(
                    emitter_context.clone(),
                    operator,
                    child_score - parent_baseline,
                );
                if operator == VariationOperator::Geometry {
                    joint_system.record_geometry_feedback(
                        &scored.genome.metal_geometry,
                        scored.genome.memory.shared_memory_bytes,
                        child_score,
                    );
                    self.runtime.record_geometry_observation(
                        &scored.genome.metal_geometry,
                        scored.genome.memory.shared_memory_bytes,
                        child_score,
                    );
                }
                let child_digest = format!(
                    "gen{}-{}",
                    gen + 1,
                    sha256_digest(&serde_json::to_string(&scored.genome).unwrap_or_default())
                );
                let parent_digests = parents
                    .iter()
                    .map(|parent| {
                        format!(
                            "gen{}-{}",
                            gen,
                            sha256_digest(&serde_json::to_string(parent).unwrap_or_default())
                        )
                    })
                    .collect();
                next_parent_map.insert(child_digest.clone(), parent_digests);
                next_operator_map.insert(child_digest, operator);
                next_population.push(scored);
            }

            population = next_population;
            parent_map = next_parent_map;
            operator_map = next_operator_map;
            self.runtime
                .persist_if_configured()
                .map_err(SearchError::SearchFailed)?;
        }
        let mut frontier_candidates: Vec<CandidateRecord> = all_candidates.clone();
        frontier_candidates.sort_by(|a, b| {
            b.score_vector
                .first()
                .partial_cmp(&a.score_vector.first())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        frontier_candidates.truncate(10);

        // Run semantic tensor-family stages after broad exploration. Each
        // stage reuses the hardware-aware adapter and freezes stabilized
        // execution axes before the next family is explored.
        let hierarchy_plan = HierarchicalStagePlan::from_tensor_keys(&tensor_keys);
        if let Some(seed) = quality_archive
            .ranked_elites()
            .first()
            .map(|entry| entry.genome.clone())
        {
            let (refined, _frozen): (CandidateGenome, FrozenHierarchy) = hierarchy_plan.run(
                seed,
                |candidate, _stage, frozen| {
                    let mut proposals = Vec::with_capacity(5);
                    for representation in [
                        RepresentationAxis::Fp16,
                        RepresentationAxis::Int8,
                        RepresentationAxis::Ternary158,
                        RepresentationAxis::TernaryTile640,
                    ] {
                        let mut proposal = candidate.clone();
                        frozen.apply(&mut proposal);
                        proposal.representation = representation;
                        proposals.push(proposal);
                    }
                    proposals.push(candidate.clone());
                    proposals
                },
                |candidate, stage, _frozen| {
                    let mut stage_context = stage.tensor_scope.join("\n").into_bytes();
                    stage_context.push(b'\n');
                    stage_context.extend_from_slice(context_bytes);
                    adapter.evaluate(candidate, &stage_context).value()
                },
            );
            let refined_score = adapter.evaluate(&refined, context_bytes).value();
            let refined_schedule = schedule_fitness(graph, &tensor_keys, &refined);
            let refined_json = serde_json::to_string(&refined).unwrap_or_default();
            let refined_evidence = adapter.evidence_for(&refined_json, context_bytes);
            let refined_profile =
                refined_evidence
                    .selected_configuration
                    .and_then(|configuration| {
                        refined_evidence
                            .profiles_evaluated
                            .iter()
                            .find(|profile| profile.configuration == configuration)
                    });
            let refined_ane = refined_profile
                .and_then(|profile| profile.ane.performance_score)
                .unwrap_or(0.0);
            let refined_metal = refined_profile
                .and_then(|profile| profile.metal.performance_score)
                .unwrap_or(0.0);
            let refined_latency = refined_profile.and_then(|profile| {
                match (profile.ane.wall_time_ms, profile.metal.wall_time_ms) {
                    (Some(ane), Some(metal)) => Some(ane + metal),
                    _ => None,
                }
            });
            let refined_feasible = refined_evidence.both_backends_feasible;
            quality_archive.insert(ArchiveEntry {
                descriptor: BehaviorDescriptor::from_execution(
                    &refined,
                    refined_ane,
                    refined_metal,
                    refined_evidence.both_backends_measured,
                    refined_latency,
                ),
                genome: refined.clone(),
                objectives: ObjectiveVector::new(vec![
                    ObjectiveValue::maximize("joint_backend_fidelity", refined_score),
                    ObjectiveValue::maximize("schedule_quality", refined_schedule),
                    ObjectiveValue::minimize(
                        "measured_latency_ms",
                        refined_latency.unwrap_or(f64::MAX),
                    ),
                    ObjectiveValue::maximize(
                        "joint_feasibility",
                        if refined_feasible { 1.0 } else { 0.0 },
                    ),
                    ObjectiveValue::maximize("ane_fidelity", refined_ane),
                    ObjectiveValue::maximize("metal_fidelity", refined_metal),
                    ObjectiveValue::minimize(
                        "ane_latency_ms",
                        refined_profile
                            .and_then(|profile| profile.ane.wall_time_ms)
                            .unwrap_or(f64::MAX),
                    ),
                    ObjectiveValue::minimize(
                        "metal_latency_ms",
                        refined_profile
                            .and_then(|profile| profile.metal.wall_time_ms)
                            .unwrap_or(f64::MAX),
                    ),
                ]),
                generation: gen_records.len() as u64,
                novelty: 1.0,
            });
        }

        // Final selection follows the same archive-first policy as parent
        // generation. The scalar frontier is retained only as a fallback for
        // legacy runs that produced no quality-diversity cell.
        let best_genome = quality_archive
            .ranked_elites()
            .first()
            .map(|entry| entry.genome.clone())
            ;

        if production_mode
            && !all_candidates
                .iter()
                .any(|candidate| matches!(candidate.status, CandidateStatus::Evaluated))
        {
            return Err(SearchError::NoJointBackendFeasibleCandidate);
        }

        let best_joint_tiling = best_genome.as_ref().map(|genome| {
            let genome_str = serde_json::to_string(genome).unwrap_or_default();
            adapter.evidence_for(&genome_str, context_bytes)
        });

        if let Some(evidence) = &best_joint_tiling {
            for backend in [SearchBackend::Ane, SearchBackend::Metal] {
                let measured = evidence
                    .profiles_evaluated
                    .iter()
                    .filter(|profile| match backend {
                        SearchBackend::Ane => profile.ane.measured,
                        SearchBackend::Metal => profile.metal.measured,
                    })
                    .count() as u64;
                self.runtime.record_hardware_profile(prism_ecs_ir::evolution::HardwareProfile {
                    key: prism_ecs_ir::evolution::HardwareProfileKey {
                        backend: backend.as_str().into(),
                        device: evaluator.name().into(),
                        driver: "evaluator-reported".into(),
                    },
                    measurements: measured,
                    last_seen_unix_ms: chrono::Utc::now().timestamp_millis().max(0) as u64,
                    metadata: serde_json::json!({ "profiles_evaluated": evidence.profiles_evaluated.len() }),
                });
            }
        }

        for receipt in &self.memory.receipts {
            self.runtime.record_receipt(receipt.clone());
        }
        if let Some(surrogate) = &self.surrogate {
            self.runtime
                .record_surrogate(prism_ecs_ir::evolution::SurrogateRecord {
                    name: surrogate.name().to_string(),
                    version: "online".into(),
                    observations: surrogate.observations(),
                    metadata: serde_json::json!({
                        "context_bytes": context_bytes.len(),
                        "candidates_evaluated": all_candidates.len(),
                    }),
                });
        }
        for entry in quality_archive.ranked_elites().into_iter().take(64) {
            self.runtime.insert_execution_elite(entry.clone());
            self.runtime.insert_tensor_elite(
                tensor_keys
                    .first()
                    .map(|key| {
                        prism_ecs_ir::evolution::compile_plan::classify_tensor(key).to_string()
                    })
                    .unwrap_or_else(|| "unknown".into()),
                entry.clone(),
            );
            self.runtime
                .record_descriptor(prism_ecs_ir::evolution::DescriptorRecord {
                    candidate_digest: sha256_digest(
                        &serde_json::to_string(&entry.genome).unwrap_or_default(),
                    ),
                    descriptor: entry.descriptor,
                    session_id: runtime_session.session_id.clone(),
                });
            if evaluator.is_measured() {
                for backend in [SearchBackend::Ane, SearchBackend::Metal] {
                    self.runtime.insert_hardware_elite(
                        prism_ecs_ir::evolution::HardwareProfileKey {
                            backend: backend.as_str().into(),
                            device: evaluator.name().into(),
                            driver: "evaluator-reported".into(),
                        },
                        entry.clone(),
                    );
                }
            }
        }
        self.runtime
            .persist_if_configured()
            .map_err(SearchError::SearchFailed)?;

        let per_tensor_overrides = best_genome.as_ref().map(|genome| {
            self.search_tensor_overrides(genome, &tensor_keys, evaluator, context_bytes)
        });
        let format_plan = best_genome.as_ref().map(|genome| {
            let mut plan = FormatPlan::from_best_genome(genome, &tensor_keys);
            // The global winner is only the seed for the tensor-wise search.
            // Promote each measured tensor override into the authoritative
            // plan consumed by lowering and CImage emission.
            if let Some(overrides) = per_tensor_overrides.as_ref() {
                for tensors in overrides.values() {
                    for (tensor, assignment) in tensors {
                        plan.per_tensor.insert(tensor.clone(), assignment.format);
                    }
                }
            }
            let plan = best_joint_tiling
                .as_ref()
                .and_then(|evidence| evidence.selected_configuration)
                .map(|configuration| {
                    plan.clone().with_joint_tiling(JointTilingPlan {
                        ane_unit: configuration.ane_unit,
                        ane_tile_m: configuration.ane_tile_m,
                        ane_tile_n: configuration.ane_tile_n,
                        ane_tile_k: configuration.ane_tile_k,
                        metal_tile_m: configuration.metal_tile_m,
                        metal_tile_n: configuration.metal_tile_n,
                        metal_tile_k: configuration.metal_tile_k,
                        metal_threadgroup_width: configuration.metal_threadgroup_width,
                        metal_threadgroup_height: configuration.metal_threadgroup_height,
                    })
                })
                .unwrap_or(plan);
            serde_json::to_string(&plan).unwrap_or_default()
        });

        let heterogeneous_schedule =
            serde_json::from_str::<FormatPlan>(format_plan.as_deref().unwrap_or("null"))
                .ok()
                .and_then(|plan| {
                    prism_spatial_ir::execution_plan::lower_to_manifest(
                        graph,
                        prism_spatial_ir::cost::CostEstimate::zero(),
                        Some(&plan),
                    )
                })
                .and_then(|manifest| manifest.batch_plan)
                .map(|plan| HeterogeneousScheduleEvidence {
                    steps: plan.fused_steps.len(),
                    route_sequence: plan
                        .route_names()
                        .iter()
                        .map(|r| (*r).to_string())
                        .collect(),
                    zero_copy_steps: plan
                        .fused_steps
                        .iter()
                        .filter(|step| step.zero_copy)
                        .count(),
                    estimated_latency_ns: plan
                        .fused_steps
                        .iter()
                        .map(|step| step.estimated_latency_ns)
                        .sum(),
                    residency_windows: plan.residency_windows.len(),
                    supports_realtime_text: plan.residency_windows.iter().all(|w| {
                        w.required_workloads.contains(
                            &prism_spatial_ir::execution_plan::ResidencyWorkload::RealtimeText,
                        )
                    }),
                    supports_batched_text: plan.residency_windows.iter().all(|w| {
                        w.required_workloads.contains(
                            &prism_spatial_ir::execution_plan::ResidencyWorkload::BatchedText,
                        )
                    }),
                    supports_batched_audio: plan.residency_windows.iter().all(|w| {
                        w.required_workloads.contains(
                            &prism_spatial_ir::execution_plan::ResidencyWorkload::BatchedAudio,
                        )
                    }),
                });

        let gen_count = gen_records.len() as u64;
        self.trace.generations = gen_records;
        self.trace.pareto_frontier = frontier_candidates;
        self.trace.quality_diversity_archive = quality_archive
            .ranked_elites()
            .into_iter()
            .cloned()
            .collect();
        self.trace.best_genome = best_genome
            .as_ref()
            .map(|g| serde_json::to_string(g).unwrap_or_default());
        self.trace.trace_digest =
            sha256_digest(&serde_json::to_string(&self.trace).unwrap_or_default());

        let selected_candidate_digest = best_genome.as_ref().and_then(|genome| {
            let serialized = serde_json::to_string(genome).ok()?;
            all_candidates
                .iter()
                .find(|candidate| candidate.genome == serialized)
                .map(|candidate| candidate.candidate_digest.clone())
        });
        let selection_receipt = SearchSelectionReceipt::build(
            self.trace.search_id.clone(),
            evaluator.name(),
            evaluator.is_measured(),
            &all_candidates,
            selected_candidate_digest,
        );
        let deployment_archive = deployment_archive_from_records(
            source,
            evaluator,
            &sha256_digest(&String::from_utf8_lossy(context_bytes)),
            &all_candidates,
            &DeploymentGatePolicy {
                min_quality: self.config.min_quality,
                max_p99_latency_ms: self.config.max_p99_latency_ms,
                max_peak_memory_bytes: self.config.max_peak_memory_bytes,
                require_measurements: true,
            },
        );
        Ok(SearchResult {
            format_plan,
            trace: self.trace.clone(),
            candidates_evaluated: all_candidates.len() as u64,
            generations_completed: gen_count,
            evaluation_route: evaluator.name().to_string(),
            heterogeneous_schedule,
            best_joint_tiling,
            evolution_memory: self.memory.clone(),
            selection_receipt,
            deployment_archive,
        })
    }

    fn search_tensor_overrides(
        &self,
        seed: &CandidateGenome,
        tensor_keys: &[String],
        evaluator: &dyn EvaluationStrategy,
        context: &[u8],
    ) -> HashMap<String, HashMap<String, prism_ecs_ir::evolution::compile_plan::PerTensorFormat>>
    {
        let mut overrides = HashMap::new();
        for tensor in tensor_keys {
            let class = prism_ecs_ir::evolution::compile_plan::classify_tensor(tensor);
            if class == "norm" || class == "embed" {
                continue;
            }
            let representations = [
                RepresentationAxis::Fp16,
                RepresentationAxis::Int8,
                RepresentationAxis::Ternary158,
                RepresentationAxis::TernaryTile640,
            ];
            let mut best: Option<(CandidateGenome, f64)> = None;
            for representation in representations {
                let mut candidate = seed.clone();
                candidate.representation = representation;
                let Ok(genome_json) = serde_json::to_string(&candidate) else {
                    continue;
                };
                let mut tensor_context = tensor.as_bytes().to_vec();
                tensor_context.extend_from_slice(b"\n");
                tensor_context.extend_from_slice(context);
                let Some(score) = evaluator
                    .evaluate(&genome_json, &tensor_context)
                    .ok()
                    .and_then(|scores| scores.into_iter().next())
                    .filter(|value| value.is_finite())
                else {
                    continue;
                };
                if best.as_ref().is_none_or(|(_, current)| score > *current) {
                    best = Some((candidate, score));
                }
            }
            if let Some((candidate, _)) = best {
                let plan = FormatPlan::from_best_genome(&candidate, &[tensor.clone()]);
                if let Some(assignment) = plan.per_tensor.get(tensor) {
                    overrides
                        .entry(class.to_string())
                        .or_insert_with(HashMap::new)
                        .insert(tensor.clone(), assignment.clone());
                }
            }
        }
        overrides.into_iter().map(|(class, tensors)| (class, tensors.into_iter().map(|(name, format)| (name, prism_ecs_ir::evolution::compile_plan::PerTensorFormat { format })).collect())).collect()
    }

    pub fn trace(&self) -> &SearchTrace {
        &self.trace
    }
}

#[derive(Debug)]
pub struct SearchResult {
    pub format_plan: Option<String>,
    pub trace: SearchTrace,
    pub candidates_evaluated: u64,
    pub generations_completed: u64,
    /// The concrete evaluator route used for candidate scoring, allowing
    /// callers to distinguish ANE-backed search from synthetic/CPU scoring.
    pub evaluation_route: String,
    pub heterogeneous_schedule: Option<HeterogeneousScheduleEvidence>,
    pub best_joint_tiling: Option<JointTilingEvidence>,
    pub evolution_memory: EvolutionaryMemory,
    /// Structured selection provenance. A fallback receipt is still useful
    /// for diagnostics, but it is explicitly ineligible as production
    /// evidence when the evaluator was not real.
    pub selection_receipt: SearchSelectionReceipt,
    /// Deployment-level archive containing only candidates with explicit
    /// admission evidence. This is the handoff to CImage selection and
    /// runtime policy, distinct from the in-generation genome frontier.
    pub deployment_archive: ParetoArchive,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SearchSelectionReceipt {
    pub schema_version: String,
    pub search_id: String,
    pub evaluator: String,
    pub evidence_source: String,
    pub production_evidence: bool,
    pub candidates_evaluated: u64,
    pub measured_candidates: u64,
    pub selected_candidate_digest: Option<String>,
    pub fallback_reason: Option<String>,
    pub receipt_digest: String,
}

impl SearchSelectionReceipt {
    fn build(
        search_id: impl Into<String>,
        evaluator: impl Into<String>,
        production_evidence: bool,
        candidates: &[CandidateRecord],
        selected_candidate_digest: Option<String>,
    ) -> Self {
        let evaluator = evaluator.into();
        let measured_candidates = candidates
            .iter()
            .filter(|candidate| {
                candidate
                    .measurements
                    .as_ref()
                    .and_then(|value| value.get("both_backends_measured"))
                    .and_then(serde_json::Value::as_bool)
                    .unwrap_or(false)
            })
            .count() as u64;
        let mut receipt = Self {
            schema_version: "prism-search-selection/1".into(),
            search_id: search_id.into(),
            evaluator,
            evidence_source: if production_evidence {
                "measured".into()
            } else {
                "synthetic-fallback".into()
            },
            production_evidence,
            candidates_evaluated: candidates.len() as u64,
            measured_candidates,
            selected_candidate_digest,
            fallback_reason: (!production_evidence).then(|| {
                "search used a non-measured evaluator; scores are diagnostic only".into()
            }),
            receipt_digest: String::new(),
        };
        let bytes = serde_json::to_vec(&receipt).unwrap_or_default();
        receipt.receipt_digest = sha256_digest(&String::from_utf8_lossy(&bytes));
        receipt
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SearchError {
    #[error("Production mode requires a real evaluator")]
    ProductionModeRequiresEvaluator,
    #[error("No tensors to search over")]
    NoTensors,
    #[error("Search failed: {0}")]
    SearchFailed(String),
    #[error("Evaluator error: {0}")]
    EvaluatorError(String),
    #[error("Synthetic data is not allowed in production mode")]
    SyntheticDataInProductionMode,
    #[error("Missing measurements for candidate")]
    MissingMeasurements,
    #[error("Correctness validation failed")]
    CorrectnessValidationFailed,
    #[error("No tiling profile was feasible on both ANE and Metal")]
    NoJointBackendFeasibleCandidate,
}

pub trait EvaluationStrategy: Send + Sync {
    fn evaluate(&self, genome: &str, context: &[u8]) -> Result<Vec<f64>, String>;

    /// Whether this evaluator supplies hardware-backed measurements.
    /// Existing evaluators remain source-compatible; the default recognizes
    /// the repository's synthetic naming convention and can be overridden by
    /// an evaluator with a stronger provenance signal.
    fn is_measured(&self) -> bool {
        false
    }

    /// Evaluate one backend for one joint tiling profile. Hardware-aware
    /// evaluators can override this to dispatch the requested backend and
    /// return device timing/feasibility directly.
    fn evaluate_backend(
        &self,
        _backend: SearchBackend,
        genome: &str,
        context: &[u8],
        _configuration: &JointTilingConfiguration,
    ) -> Result<BackendEvaluation, String> {
        let started = Instant::now();
        let score = self
            .evaluate(genome, context)?
            .first()
            .copied()
            .ok_or_else(|| "evaluator returned no score".to_string())?;
        Ok(BackendEvaluation {
            performance_score: score,
            feasible: score.is_finite() && score > 0.0,
            measured: self.is_measured(),
            wall_time_ms: Some(started.elapsed().as_secs_f64() * 1_000.0),
        })
    }

    fn name(&self) -> &str;

    /// Return the reference-aware progressive executor when this evaluator
    /// can measure activation, logits, and router agreement.  Search callers
    /// use this as an explicit capability check; the default keeps legacy and
    /// synthetic evaluators out of progressive admission.
    fn progressive_executor(&self) -> Option<&dyn ProgressiveStageExecutor> {
        None
    }
}

fn build_search_context(source: &CanonicalSource, _graph: &SpatialGraph) -> String {
    let mut ctx = String::new();
    for tensor in &source.catalog.tensors {
        ctx.push_str(&format!("{}:{:?}\n", tensor.name, tensor.shape));
    }
    ctx
}

fn sha256_digest(input: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_ir::evolution::foundation::CandidateGenome;

    #[test]
    fn compiler_profile_admission_uses_shared_metal_validation() {
        let genome = CandidateGenome::new();
        let valid = JointTilingConfiguration::from_genome(&genome);
        assert!(valid.is_valid());

        let invalid = JointTilingConfiguration {
            metal_threadgroup_width: 256,
            metal_threadgroup_height: 5,
            ..valid
        };
        assert!(!invalid.is_valid());
    }

    #[test]
    fn native_lane_promotion_requires_measured_ane_and_metal() {
        let mut evidence = JointTilingEvidence {
            selected_configuration: None,
            selected_score: None,
            both_backends_feasible: true,
            both_backends_measured: false,
            profiles_evaluated: Vec::new(),
        };
        assert!(!evidence.native_lane_ready());
        evidence.both_backends_measured = true;
        assert!(evidence.native_lane_ready());
    }

    #[test]
    fn promotion_builder_does_not_infer_unmeasured_lanes() {
        let evidence = JointTilingEvidence {
            selected_configuration: None,
            selected_score: None,
            both_backends_feasible: true,
            both_backends_measured: false,
            profiles_evaluated: Vec::new(),
        };
        let receipt =
            evidence.native_promotion_evidence(true, true, true, true, true, "abi", "reference");
        assert!(!receipt.metal_packed.passed);
        assert!(!receipt.ane_static.passed);
        assert!(!receipt.eligible());
    }

    #[test]
    fn non_measured_selection_receipt_is_explicitly_diagnostic() {
        let evaluator = DefaultSearchEvaluator;
        assert!(!evaluator.is_measured());
        let receipt = SearchSelectionReceipt::build(
            "search",
            evaluator.name(),
            evaluator.is_measured(),
            &[],
            None,
        );
        assert!(!receipt.production_evidence);
        assert_eq!(receipt.evidence_source, "synthetic-fallback");
        assert!(receipt.fallback_reason.is_some());
    }
}
