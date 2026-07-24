//! Foundation identity and genome types for evolutionary search.
//!
//! Provides the low-level identity wrappers (`CandidateId`, `LogicalTensorId`),
//! scoring types (`FitnessScore`), and the 9-dimensional genome
//! (`CandidateGenome`) that the evolution pipeline searches over.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct CandidateId(pub String);

impl From<&str> for CandidateId { fn from(s: &str) -> Self { CandidateId(s.to_string()) } }
impl From<String> for CandidateId { fn from(s: String) -> Self { CandidateId(s) } }

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LogicalTensorId(pub String);

impl From<&str> for LogicalTensorId { fn from(s: &str) -> Self { LogicalTensorId(s.to_string()) } }
impl From<String> for LogicalTensorId { fn from(s: String) -> Self { LogicalTensorId(s) } }

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FitnessScore(pub f64);

impl FitnessScore {
    pub const MAX: FitnessScore = FitnessScore(1.0);
    pub const MIN: FitnessScore = FitnessScore(0.0);
    pub fn new(value: f64) -> Self { FitnessScore(value.clamp(0.0, 1.0)) }
    pub fn value(&self) -> f64 { self.0 }
}
impl From<f64> for FitnessScore { fn from(value: f64) -> Self { FitnessScore::new(value) } }

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GenomeAxis {
    Representation,
    Packing,
    MetalGeometry,
    Decomposition,
    Memory,
    Fusion,
    Engram,
    Runtime,
    AneUnit,
}

pub const GENOME_AXIS_COUNT: usize = 9;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct GenomeAxisSet(pub u16);

impl GenomeAxisSet {
    pub const fn empty() -> Self { Self(0) }
    pub const fn all() -> Self { Self((1u16 << GENOME_AXIS_COUNT) - 1) }
    pub const fn from_axis(axis: GenomeAxis) -> Self { Self(1u16 << axis as u8) }
    pub const fn union(self, other: Self) -> Self { Self(self.0 | other.0) }
    pub const fn contains(self, axis: GenomeAxis) -> bool { (self.0 & (1u16 << axis as u8)) != 0 }
    pub const fn count(self) -> u32 { self.0.count_ones() }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateGenome {
    pub representation: RepresentationAxis,
    pub packing: PackingAxis,
    pub metal_geometry: MetalGeometryAxis,
    pub decomposition: DecompositionAxis,
    pub memory: MemoryAxis,
    pub fusion: FusionAxis,
    pub engram: EngramAxis,
    pub runtime: RuntimeAxis,
    pub ane_unit: AneUnitAxis,
}

impl CandidateGenome { pub fn new() -> Self { Self::default() } }
impl Default for CandidateGenome {
    fn default() -> Self {
        Self {
            representation: RepresentationAxis::default(), packing: PackingAxis::default(),
            metal_geometry: MetalGeometryAxis::default(), decomposition: DecompositionAxis::default(),
            memory: MemoryAxis::default(), fusion: FusionAxis::default(), engram: EngramAxis::default(),
            runtime: RuntimeAxis::default(), ane_unit: AneUnitAxis::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RepresentationAxis { Fp16, Bf16, Int8, Int4, Nf4, Nf8, Ternary158, TernaryTile640, Binary1 }
impl Default for RepresentationAxis { fn default() -> Self { Self::Fp16 } }

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AneUnitAxis { Auto, Planar, Matrix }
impl Default for AneUnitAxis { fn default() -> Self { Self::Auto } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PackingAxis { Tile640, Block2D, Planar, Interleaved }
impl Default for PackingAxis { fn default() -> Self { Self::Tile640 } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalGeometryAxis { pub threadgroup_width: u32, pub threadgroup_height: u32, pub grid_tile_m: u32, pub grid_tile_n: u32, pub grid_tile_k: u32 }
impl Default for MetalGeometryAxis { fn default() -> Self { Self { threadgroup_width: 32, threadgroup_height: 8, grid_tile_m: 64, grid_tile_n: 64, grid_tile_k: 32 } } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecompositionAxis { SplitM, SplitMN, SplitMNK, Flat }
impl Default for DecompositionAxis { fn default() -> Self { Self::SplitMN } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAxis { pub shared_memory_bytes: u32, pub prefetch: bool, pub double_buffer: bool }
impl Default for MemoryAxis { fn default() -> Self { Self { shared_memory_bytes: 32768, prefetch: true, double_buffer: false } } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionAxis { None, ElementWise, KernelFusion }
impl Default for FusionAxis { fn default() -> Self { Self::ElementWise } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EngramAxis { Direct, Indirect, Jit }
impl Default for EngramAxis { fn default() -> Self { Self::Direct } }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeAxis { pub dispatch_width: u32, pub sync_depth: u32, pub async_encoding: bool }
impl Default for RuntimeAxis { fn default() -> Self { Self { dispatch_width: 1, sync_depth: 1, async_encoding: false } } }

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn fitness_score_clamping() { assert_eq!(FitnessScore::new(1.5).value(), 1.0); assert_eq!(FitnessScore::new(-0.5).value(), 0.0); }
    #[test] fn candidate_id_from_str() { let id: CandidateId = "test-candidate-42".into(); assert_eq!(id.0, "test-candidate-42"); }
    #[test] fn logical_tensor_id_roundtrip() { let id = LogicalTensorId("layer0.attention.q_proj".to_string()); assert_eq!(id.0, "layer0.attention.q_proj"); }
    #[test] fn default_genome() { let g = CandidateGenome::new(); assert_eq!(g.representation as i32, RepresentationAxis::Fp16 as i32); assert_eq!(g.decomposition as i32, DecompositionAxis::SplitMN as i32); }
    #[test] fn all_axes_are_representable() { let all = GenomeAxisSet::all(); assert_eq!(all.count() as usize, GENOME_AXIS_COUNT); assert!(all.contains(GenomeAxis::AneUnit)); }
}
