//! Foundation identity and genome types for evolutionary search.
//!
//! Provides the low-level identity wrappers (`CandidateId`, `LogicalTensorId`),
//! scoring types (`FitnessScore`), and the 8-dimensional genome
//! (`CandidateGenome`) that the evolution pipeline searches over.

use serde::{Deserialize, Serialize};

// ── Identity Types ──────────────────────────────────────────────────────────

/// Identifier for an evolution candidate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct CandidateId(pub String);

impl From<&str> for CandidateId {
    fn from(s: &str) -> Self {
        CandidateId(s.to_string())
    }
}

impl From<String> for CandidateId {
    fn from(s: String) -> Self {
        CandidateId(s)
    }
}

/// Identifier for a logical tensor within a model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct LogicalTensorId(pub String);

impl From<&str> for LogicalTensorId {
    fn from(s: &str) -> Self {
        LogicalTensorId(s.to_string())
    }
}

impl From<String> for LogicalTensorId {
    fn from(s: String) -> Self {
        LogicalTensorId(s)
    }
}

// ── Scoring ─────────────────────────────────────────────────────────────────

/// A scalar fitness score (higher is better).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FitnessScore(pub f64);

impl FitnessScore {
    /// Perfect fitness.
    pub const MAX: FitnessScore = FitnessScore(1.0);
    /// Worst possible fitness.
    pub const MIN: FitnessScore = FitnessScore(0.0);

    /// Create a new fitness score clamped to [0, 1].
    pub fn new(value: f64) -> Self {
        FitnessScore(value.clamp(0.0, 1.0))
    }

    /// Return the raw value.
    pub fn value(&self) -> f64 {
        self.0
    }
}

impl From<f64> for FitnessScore {
    fn from(value: f64) -> Self {
        FitnessScore::new(value)
    }
}

// ── CandidateGenome ─────────────────────────────────────────────────────────

/// The 8-dimensional typed genome explored by the evolution pipeline.
///
/// Each dimension controls one axis of the compilation search space:
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateGenome {
    /// Representation strategy (e.g. fp16, int8, nf4, ternary, binary).
    pub representation: RepresentationAxis,
    /// Packing strategy (e.g. tile640, block2d, planar).
    pub packing: PackingAxis,
    /// Metal geometry tile sizes (threadgroup + grid dimensions).
    pub metal_geometry: MetalGeometryAxis,
    /// Decomposition strategy (tiling, partitioning, fusion boundaries).
    pub decomposition: DecompositionAxis,
    /// Memory configuration (shared memory, caching, allocation).
    pub memory: MemoryAxis,
    /// Fusion strategy (op fusion, kernel fusion, pipeline fusion).
    pub fusion: FusionAxis,
    /// Engram gene: how this candidate's execution plan is encoded.
    pub engram: EngramAxis,
    /// Runtime execution parameters (dispatch width, sync depth).
    pub runtime: RuntimeAxis,
}

impl CandidateGenome {
    /// Create a new genome with all axes set to their defaults.
    pub fn new() -> Self {
        Self::default()
    }
}

impl Default for CandidateGenome {
    fn default() -> Self {
        Self {
            representation: RepresentationAxis::default(),
            packing: PackingAxis::default(),
            metal_geometry: MetalGeometryAxis::default(),
            decomposition: DecompositionAxis::default(),
            memory: MemoryAxis::default(),
            fusion: FusionAxis::default(),
            engram: EngramAxis::default(),
            runtime: RuntimeAxis::default(),
        }
    }
}

// ── Genome Axis Types ───────────────────────────────────────────────────────

/// Representation strategy axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RepresentationAxis {
    Fp16,
    Bf16,
    Int8,
    Int4,
    Nf4,
    Nf8,
    Ternary158,
    Binary1,
}

impl Default for RepresentationAxis {
    fn default() -> Self {
        RepresentationAxis::Fp16
    }
}

/// Packing strategy axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PackingAxis {
    Tile640,
    Block2D,
    Planar,
    Interleaved,
}

impl Default for PackingAxis {
    fn default() -> Self {
        PackingAxis::Tile640
    }
}

/// Metal geometry axis — threadgroup and grid dimensions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalGeometryAxis {
    pub threadgroup_width: u32,
    pub threadgroup_height: u32,
    pub grid_tile_m: u32,
    pub grid_tile_n: u32,
    pub grid_tile_k: u32,
}

impl Default for MetalGeometryAxis {
    fn default() -> Self {
        Self {
            threadgroup_width: 32,
            threadgroup_height: 8,
            grid_tile_m: 64,
            grid_tile_n: 64,
            grid_tile_k: 32,
        }
    }
}

/// Decomposition strategy axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecompositionAxis {
    /// Split along M dimension only.
    SplitM,
    /// Split along M and N dimensions.
    SplitMN,
    /// Split along M, N, and K dimensions.
    SplitMNK,
    /// No decomposition — flat.
    Flat,
}

impl Default for DecompositionAxis {
    fn default() -> Self {
        DecompositionAxis::SplitMN
    }
}

/// Memory configuration axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAxis {
    /// Shared memory per threadgroup in bytes.
    pub shared_memory_bytes: u32,
    /// Whether to prefetch operands into shared memory.
    pub prefetch: bool,
    /// Whether to double-buffer shared memory loads.
    pub double_buffer: bool,
}

impl Default for MemoryAxis {
    fn default() -> Self {
        Self {
            shared_memory_bytes: 32768,
            prefetch: true,
            double_buffer: false,
        }
    }
}

/// Fusion strategy axis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionAxis {
    /// No fusion — each op compiled separately.
    None,
    /// Element-wise ops fused with matmul.
    ElementWise,
    /// Full kernel fusion at pipeline boundaries.
    KernelFusion,
}

impl Default for FusionAxis {
    fn default() -> Self {
        FusionAxis::ElementWise
    }
}

/// Engram gene — encoding strategy for the execution plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EngramAxis {
    /// Direct dispatch (single kernel, no indirection).
    Direct,
    /// Indirect dispatch table.
    Indirect,
    /// JIT-compiled execution plan.
    Jit,
}

impl Default for EngramAxis {
    fn default() -> Self {
        EngramAxis::Direct
    }
}

/// Runtime execution parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeAxis {
    /// Dispatch width — number of concurrent dispatches.
    pub dispatch_width: u32,
    /// Synchronization depth — number of sync barriers between dispatches.
    pub sync_depth: u32,
    /// Whether to use async command encoding.
    pub async_encoding: bool,
}

impl Default for RuntimeAxis {
    fn default() -> Self {
        Self {
            dispatch_width: 1,
            sync_depth: 1,
            async_encoding: false,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fitness_score_clamping() {
        let f = FitnessScore::new(1.5);
        assert_eq!(f.value(), 1.0);
        let f = FitnessScore::new(-0.5);
        assert_eq!(f.value(), 0.0);
        let f = FitnessScore::new(0.75);
        assert_eq!(f.value(), 0.75);
    }

    #[test]
    fn candidate_id_from_str() {
        let id: CandidateId = "test-candidate-42".into();
        assert_eq!(id.0, "test-candidate-42");
    }

    #[test]
    fn logical_tensor_id_roundtrip() {
        let id = LogicalTensorId("layer0.attention.q_proj".to_string());
        assert_eq!(id.0, "layer0.attention.q_proj");
    }

    #[test]
    fn default_genome() {
        let g = CandidateGenome::new();
        assert_eq!(g.representation as i32, RepresentationAxis::Fp16 as i32);
        assert_eq!(g.decomposition as i32, DecompositionAxis::SplitMN as i32);
    }
}
