//! Foundation types for evolutionary program search.
//!
//! Stage 0 of the evolutionary cimage plan. These types define the search
//! space, program representation, cost model, and MIL fragment format.
//!
//! No ECS systems here — those come in Stage 1+.

use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::ecs::canonical::identity::CandidateId;
use crate::ecs::canonical::kernel_abi::{ArtifactProvenance, KernelImplementationId};
use crate::ecs::canonical::provenance::MeasuredCandidateRecord;
use crate::ecs::cimage::PhysicalTileLayout;
use crate::ecs::component::backend::BackendTarget;
use crate::ecs::plan::CodecFamily;
use crate::ecs::quantization::contract::TernaryCandidateRecipe;
use crate::ecs::CompEntity;

use crate::ecs::Entity;

// ── Search infrastructure ───────────────────────────────────────────────

/// Describes one search over a tensor's execution program for a target backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolveCandidate {
    pub tensor_id: String,
    pub target_backend: BackendTarget,
    pub format: CodecFamily,
    pub program: EvolveProgram,
    pub measured_cost: Option<CostMetrics>,
    pub generation: u64,
    pub parents: Vec<String>,
}

/// Cost measurement for one candidate on one backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostMetrics {
    pub wall_ns: u64,
    pub energy_uj: Option<u64>,
    pub alu_cycles: Option<u64>,
    pub bandwidth_bytes: u64,
}

/// The search state machine for one tensor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionState {
    pub tensor_id: String,
    pub target_backend: BackendTarget,
    pub seed_program: EvolveProgram,
    pub population: Vec<Entity>,
    pub generation: u64,
    pub best_cost: Option<CostMetrics>,
    pub best_candidate: Option<Entity>,
    pub converged: bool,
    pub search_config: SearchConfig,
    pub receipt_store: Vec<ReceiptMetadata>,
    /// Persisted candidate records for every evaluated candidate.
    pub records: Vec<MeasuredCandidateRecord>,
}

/// Configuration for one search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    pub population_size: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub max_generations: usize,
    pub convergence_threshold: f64,
    pub cost_function: CostFunction,
}

/// How to compute candidate fitness.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CostFunction {
    WallTime,
    Energy,
    Bandwidth,
    Weighted {
        wall: f64,
        energy: f64,
        bandwidth: f64,
    },
}

/// The program space a search explores.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EvolveProgram {
    MetalShader(String),
    MilProgram(MilProgramFragment),
    CustomPack {
        tile_m: usize,
        tile_n: usize,
        tile_k: usize,
        instructions: Vec<CustomInstruction>,
    },
    FusedGroupRef(CompEntity),
}

/// A custom instruction for a discovered decomposition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CustomInstruction {
    LoadWeight {
        offset: u64,
        format: CodecFamily,
    },
    Dequantize {
        src: usize,
        dst: usize,
        codebook: EvolveCodebookRef,
    },
    Accumulate {
        src: usize,
        dst: usize,
    },
    ReduceAdd {
        srcs: Vec<usize>,
        dst: usize,
    },
    Fma {
        a: usize,
        b: usize,
        c: usize,
    },
    StoreOutput {
        src: usize,
        offset: u64,
    },
}

/// A reference to a codebook (e.g., NF4 lookup table, ternary sign map).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EvolveCodebookRef {
    pub name: String,
    pub offset: u64,
    pub length: u64,
}

// ── MIL (ANE dataflow) types ───────────────────────────────────────────

/// A compiled MIL program fragment for one tensor on the ANE.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MilProgramFragment {
    pub ops: Vec<MilOp>,
    pub schedule: MilSchedule,
    pub sram_budget: u64,
}

/// ANE dataflow operations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum MilOp {
    MatMul {
        lhs: usize,
        rhs: usize,
        output: usize,
    },
    Conv1x1 {
        input: usize,
        weight: usize,
        output: usize,
    },
    Add {
        lhs: usize,
        rhs: usize,
        output: usize,
    },
    Mul {
        lhs: usize,
        rhs: usize,
        output: usize,
    },
    Load {
        buffer: usize,
        offset: u64,
        size: u64,
    },
    Store {
        buffer: usize,
        offset: u64,
    },
    Activation {
        kind: String,
        input: usize,
        output: usize,
    },
    Norm {
        kind: String,
        input: usize,
        weight: usize,
        output: usize,
    },
}

/// Schedule for one MIL program on the ANE.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MilSchedule {
    pub units: Vec<MilUnit>,
    pub sync_points: Vec<usize>,
}

/// One neuron/unit assignment in the MIL schedule.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MilUnit {
    pub op_range: Range<usize>,
    pub assigned_neuron: usize,
    pub sram_usage: u64,
}

/// Provenance record for an evolved program embedded in ExecutionView.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionProvenance {
    pub tensor_id: String,
    pub generation: u64,
    pub parent_candidates: Vec<String>,
    pub best_cost: CostMetrics,
    pub generation_count: u64,
}

// ── Typed EvolutionCandidate (Sections 3.5, 10) ─────────────────────────

/// A search candidate with a typed genome and evaluation receipts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionCandidate {
    pub candidate_id: CandidateId,
    pub parent_ids: Vec<CandidateId>,
    pub generation: u64,
    pub genome: CandidateGenome,
    pub compiled_artifacts: Vec<KernelImplementationId>,
    pub correctness_receipt: Option<StaticValidationReceipt>,
    pub quality_receipt: Option<NumericalReceipt>,
    pub performance_receipt: Option<PerformanceReceipt>,
    pub ternary_recipe: Option<TernaryCandidateRecipe>,
    pub fitness: Option<FitnessVector>,
    pub status: CandidateStatus,
}

/// Static validation receipt — validates ABI, device limits, constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaticValidationReceipt {
    pub candidate_id: CandidateId,
    pub passed: bool,
    pub violations: Vec<String>,
    pub validated_at: String,
    /// Provenance chain for compiled artifacts linked to this receipt.
    pub provenance: Vec<ArtifactProvenance>,
}

/// Numerical validation receipt — compares candidate output to reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericalReceipt {
    pub candidate_id: CandidateId,
    pub passed: bool,
    pub max_absolute_error: f64,
    pub max_relative_error: f64,
    pub threshold: f64,
    /// Provenance chain for compiled artifacts linked to this receipt.
    pub provenance: Vec<ArtifactProvenance>,
}

/// Performance measurement receipt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReceipt {
    pub candidate_id: CandidateId,
    pub latency_p50_ns: u64,
    pub latency_p95_ns: u64,
    pub encode_time_ns: u64,
    pub sync_time_ns: u64,
    pub memory_traffic_bytes: u64,
    pub energy_uj: Option<u64>,
    pub repetitions: usize,
    /// Provenance chain for compiled artifacts linked to this receipt.
    pub provenance: Vec<ArtifactProvenance>,
}

/// Metadata linking evaluation receipts to their candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReceiptMetadata {
    pub candidate_id: CandidateId,
    pub static_receipt: Option<StaticValidationReceipt>,
    pub numerical_receipt: Option<NumericalReceipt>,
    pub performance_receipt: Option<PerformanceReceipt>,
}

/// The typed genome for a search candidate.
/// Representation, packing, Metal geometry, decomposition, memory, fusion, engram, and runtime genes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateGenome {
    pub representation: CodecFamily,
    pub packing: PhysicalTileLayout,
    pub metal_geometry: MetalGeometry,
    pub decomposition: DecompositionStrategy,
    pub memory_config: MemoryConfig,
    pub fusion_strategy: Option<FusionStrategy>,
    pub engram_config: Option<EngramGene>,
    pub kernel_variant: String,
}

/// Metal dispatch geometry for a kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalGeometry {
    pub grid_width: u32,
    pub grid_height: u32,
    pub simd_width: u32,
    pub threadgroup_width: u32,
    pub threadgroup_height: u32,
    pub threadgroup_depth: u32,
}

/// How a kernel is decomposed across the GPU.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecompositionStrategy {
    /// Split K dimension into the given number of tiles.
    SplitK(u32),
    /// Reduction tree with given branching factor.
    ReductionTree(u32),
    /// Single sequential pass.
    Sequential,
    /// Warp-level reduction — uses warp shuffle instructions for fast intra-warp reduction.
    WarpReduction,
    /// Partial dot product — decomposes matmul into partial dot products accumulated across threadgroups.
    PartialDotProduct,
    /// Fused gate-up — combines element-wise gating with the reduction in a single kernel.
    FusedGateUp,
}

/// Memory configuration for a kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub vector_width: u32,
    pub cache_policy: String,
    pub threadgroup_staging: u64,
}

/// How operations are fused together.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Group a set of named operations together.
    OperationGrouping(Vec<String>),
    /// A named fused region.
    FusedRegion(String),
    /// No fusion.
    None,
}

/// Engram configuration gene — codec, capacity, insertion point, routing threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngramGene {
    pub codec: String,
    pub capacity: usize,
    pub insertion_point: String,
    pub routing_threshold: f64,
}

/// Lifecycle status of an evolution candidate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum CandidateStatus {
    /// Candidate created but not yet validated.
    Created,
    /// Genome validated, artifacts not yet compiled.
    Validated,
    /// Artifacts compiled successfully.
    Compiled,
    /// Correctness validated against reference.
    Correct,
    /// Performance measured on target hardware.
    Measured,
    /// Validation failed at some stage.
    Failed,
    /// Promoted to the next generation.
    Promoted,
}

/// Multi-dimensional fitness for an evolution candidate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitnessVector {
    pub task_quality: f64,
    pub interference: f64,
    pub operator_error: f64,
    pub memory_bytes: u64,
    pub latency_p50_ns: u64,
    pub latency_p95_ns: u64,
    pub energy_uj: Option<u64>,
    pub compile_cost_ms: u64,
}

// ── RepeatabilityConfig ─────────────────────────────────────────────────

/// Configuration for repeatability checks during performance measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepeatabilityConfig {
    /// Number of warm-up repetitions before measured runs begin.
    pub warm_up_repetitions: usize,
    /// Minimum repetitions required for a valid measurement.
    pub min_repetitions: usize,
    /// Maximum acceptable coefficient of variation (std/mean) across measured runs.
    pub max_variance: f64,
}

impl Default for RepeatabilityConfig {
    fn default() -> Self {
        Self {
            warm_up_repetitions: 3,
            min_repetitions: 3,
            max_variance: 0.15,
        }
    }
}

impl EvolutionState {
    /// Add a measured candidate record and its linked receipts to the evolution state.
    ///
    /// Persists the candidate evidence so the search, frontier, replay, and promotion
    /// paths all reference the same data.
    pub fn add_candidate_record(
        &mut self,
        record: MeasuredCandidateRecord,
        static_receipt: Option<StaticValidationReceipt>,
        numerical_receipt: Option<NumericalReceipt>,
        performance_receipt: Option<PerformanceReceipt>,
    ) {
        self.records.push(record);
        self.receipt_store.push(ReceiptMetadata {
            candidate_id: static_receipt
                .as_ref()
                .map(|r| &r.candidate_id)
                .or_else(|| numerical_receipt.as_ref().map(|r| &r.candidate_id))
                .or_else(|| performance_receipt.as_ref().map(|r| &r.candidate_id))
                .cloned()
                .unwrap_or_else(|| CandidateId("unknown".into())),
            static_receipt,
            numerical_receipt,
            performance_receipt,
        });
    }
}
