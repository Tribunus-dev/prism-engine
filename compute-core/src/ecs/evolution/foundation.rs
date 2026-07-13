//! Foundation types for evolutionary program search.
//!
//! Stage 0 of the evolutionary cimage plan. These types define the search
//! space, program representation, cost model, and MIL fragment format.
//!
//! No ECS systems here — those come in Stage 1+.

use std::ops::Range;

use serde::{Deserialize, Serialize};

use crate::ecs::canonical::identity::{CandidateId, ReceiptId};
use crate::ecs::canonical::kernel_abi::KernelImplementationId;
use crate::ecs::cimage::PhysicalTileLayout;
use crate::ecs::component::backend::BackendTarget;
use crate::ecs::plan::CodecFamily;
use crate::ecs::CompEntity;

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
    pub population: Vec<CompEntity>,
    pub generation: u64,
    pub best_cost: Option<CostMetrics>,
    pub best_candidate: Option<CompEntity>,
    pub converged: bool,
    pub search_config: SearchConfig,
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
    pub correctness_receipt: Option<ReceiptId>,
    pub quality_receipt: Option<ReceiptId>,
    pub performance_receipt: Option<ReceiptId>,
    pub fitness: Option<FitnessVector>,
    pub status: CandidateStatus,
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
