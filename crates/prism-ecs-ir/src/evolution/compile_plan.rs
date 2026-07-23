//! Compilation plan types for evolution-driven format and tile assignment.
//!
//! Each tensor in a compilation plan is assigned a quantization format and
//! tile geometry by the evolutionary search. These types provide the ECS
//! components and query functions that backends use to read assignments.

use crate::evolution::foundation::CandidateGenome;
use crate::evolution::mutation_table::TensorFormat;
use prism_ecs_core::{Component, Entity, World};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct JointTilingPlan { pub ane_unit: crate::evolution::foundation::AneUnitAxis, pub ane_tile_m:u32,pub ane_tile_n:u32,pub ane_tile_k:u32,pub metal_tile_m:u32,pub metal_tile_n:u32,pub metal_tile_k:u32,pub metal_threadgroup_width:u32,pub metal_threadgroup_height:u32 }
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct PerTensorFormat { pub format: TensorFormat }
pub fn classify_tensor(name:&str)->String { let n=name.to_ascii_lowercase(); if n.contains("norm"){"norm".into()} else if n.contains("embed"){"embed".into()} else if n.contains("attn"){"attention".into()} else if n.contains("mlp")||n.contains("ffn"){"mlp".into()} else {"other".into()} }

// ── Components ──────────────────────────────────────────────────────────────

/// Marker component for an entity that is a compilation plan.
#[derive(Debug, Clone, Copy)]
pub struct CompilePlan;

impl Component for CompilePlan {}

/// Marker component for an entity that references a compilation plan.
#[derive(Debug, Clone, Copy)]
pub struct CompilePlanMarker;

impl Component for CompilePlanMarker {}

/// Reference from a tensor entity to its parent compilation plan.
#[derive(Debug, Clone, Copy)]
pub struct CompilePlanRef(pub Entity);

impl Component for CompilePlanRef {}

/// Tile sizes for a matmul operation in a compilation plan.
#[derive(Debug, Clone, Copy)]
pub struct TileSizes {
    pub tile_m: u32,
    pub tile_n: u32,
    pub tile_k: u32,
}

impl Component for TileSizes {}

impl TileSizes {
    pub fn new(tile_m: u32, tile_n: u32, tile_k: u32) -> Self {
        Self {
            tile_m,
            tile_n,
            tile_k,
        }
    }
}

/// Format assignment for a tensor within a compilation plan.
#[derive(Debug, Clone, Copy)]
pub struct FormatAssignment(pub TensorFormat);

impl Component for FormatAssignment {}

// ── Query Functions ─────────────────────────────────────────────────────────

/// Look up the format and operation assigned to a tensor entity.
///
/// Returns `Some((format, operation))` if a `FormatAssignment` component
/// exists, or `None` if the tensor has no assigned format (defaults apply).
///
/// The operation is determined from the format when no explicit
/// `TensorOperation` component is present.
pub fn get_assigned_format(
    world: &World,
    tensor: Entity,
) -> Option<(
    TensorFormat,
    crate::evolution::mutation_table::TensorOperation,
)> {
    let fmt = world.get_component::<FormatAssignment>(tensor)?.0;
    // Derive the default operation from the format
    let op = match fmt {
        TensorFormat::Ternary158 => crate::evolution::mutation_table::TensorOperation::TernaryGemm,
        TensorFormat::Binary1 => {
            crate::evolution::mutation_table::TensorOperation::BinaryPopcountGemm
        }
        TensorFormat::Int4 => crate::evolution::mutation_table::TensorOperation::Int4DequantMatmul,
        _ => crate::evolution::mutation_table::TensorOperation::Matmul,
    };
    Some((fmt, op))
}

/// Resolve the tile sizes for a matmul operation from its compilation plan.
///
/// Follows `CompilePlanRef` from the matmul entity to the plan entity, then
/// reads the `TileSizes` component. Falls back to default tile sizes when no
/// plan or tile sizes exist.
pub fn resolve_matmul_tile(
    world: &World,
    matmul_op: Entity,
    m: u32,
    n: u32,
    k: u32,
) -> (u32, u32, u32) {
    if let Some(plan_ref) = world.get_component::<CompilePlanRef>(matmul_op) {
        if let Some(tiles) = world.get_component::<TileSizes>(plan_ref.0) {
            return (tiles.tile_m, tiles.tile_n, tiles.tile_k);
        }
    }
    let _ = (m, n, k);
    (64, 64, 32)
}

// ── FormatPlan ───────────────────────────────────────────────────────────────

/// Thread-safe format assignment extracted from evolution search.
///
/// Maps tensor keys to their assigned quantization formats. This is the
/// non-ECS representation passed across thread boundaries to the compiler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatPlan {
    /// Per-tensor format assignment, keyed by tensor key
    /// (e.g. "model.layers.0.self_attn.q_proj.weight").
    pub per_tensor: HashMap<String, TensorFormat>,
    #[serde(default)] pub joint_tiling: Option<JointTilingPlan>,
}

impl FormatPlan {
    /// Create an empty format plan.
    pub fn new() -> Self {
        Self {
            per_tensor: HashMap::new(), joint_tiling: None,
        }
    }

    /// Build a format plan from the best genome in an evolution search.
    ///
    /// Maps the genome's representation axis to a per-tensor format
    /// assignment for all tensors in `tensor_keys`.
    pub fn from_best_genome(genome: &CandidateGenome, tensor_keys: &[String]) -> Self {
        let format = Self::representation_to_format(&genome.representation);
        let mut per_tensor = HashMap::new();
        for key in tensor_keys {
            per_tensor.insert(key.clone(), format);
        }
        Self { per_tensor, joint_tiling: None }
    }

    /// Convert a `RepresentationAxis` to the corresponding `TensorFormat`.
    fn representation_to_format(
        repr: &crate::evolution::foundation::RepresentationAxis,
    ) -> TensorFormat {
        use crate::evolution::foundation::RepresentationAxis;
        match repr {
            RepresentationAxis::Fp16 => TensorFormat::Fp16,
            RepresentationAxis::Bf16 => TensorFormat::Bf16,
            RepresentationAxis::Int8 => TensorFormat::Int8,
            RepresentationAxis::Int4 => TensorFormat::Int4,
            RepresentationAxis::Nf4 => TensorFormat::Nf4,
            RepresentationAxis::Nf8 => TensorFormat::Nf8,
            RepresentationAxis::Ternary158 => TensorFormat::Ternary158,
            RepresentationAxis::TernaryTile640 => TensorFormat::Ternary158,
            RepresentationAxis::Binary1 => TensorFormat::Binary1,
        }
    }

    /// Get the assigned format for a tensor key, if present.
    pub fn get(&self, key: &str) -> Option<TensorFormat> {
        self.per_tensor.get(key).copied()
    }
}

impl FormatPlan { pub fn with_joint_tiling(mut self, joint: JointTilingPlan)->Self { self.joint_tiling=Some(joint); self } }

impl Default for FormatPlan {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use prism_ecs_core::{EntityKind, World};

    use super::*;

    #[test]
    fn compile_plan_components() {
        let mut world = World::new();
        let plan = world
            .spawn(EntityKind::Pipeline, Some("test-plan".into()))
            .unwrap();
        let plan_e: Entity = plan.into();
        world.add_component(plan_e, CompilePlan).unwrap();
        world
            .add_component(plan_e, TileSizes::new(128, 128, 64))
            .unwrap();

        let tensor = world
            .spawn(EntityKind::Tensor, Some("test-tensor".into()))
            .unwrap();
        let tensor_e: Entity = tensor.into();
        world
            .add_component(tensor_e, CompilePlanRef(plan_e))
            .unwrap();
        world
            .add_component(tensor_e, FormatAssignment(TensorFormat::Ternary158))
            .unwrap();

        let (fmt, _op) = get_assigned_format(&world, tensor_e).unwrap();
        assert_eq!(fmt, TensorFormat::Ternary158);

        let (tm, tn, tk) = resolve_matmul_tile(&world, tensor_e, 1024, 1024, 1024);
        assert_eq!(tm, 128);
        assert_eq!(tn, 128);
        assert_eq!(tk, 64);
    }

    #[test]
    fn default_tiles_when_no_plan() {
        let world = World::new();
        let fake = Entity::new(999, 0);
        let (tm, tn, tk) = resolve_matmul_tile(&world, fake, 1024, 1024, 1024);
        assert_eq!(tm, 64);
        assert_eq!(tn, 64);
        assert_eq!(tk, 32);
    }

    #[test]
    fn format_plan_from_genome() {
        let genome = CandidateGenome::new();
        let keys = vec!["w1".to_string(), "w2".to_string()];
        let plan = FormatPlan::from_best_genome(&genome, &keys);
        assert_eq!(plan.per_tensor.len(), 2);
        assert_eq!(plan.get("w1"), Some(TensorFormat::Fp16));
    }

    #[test]
    fn format_plan_empty() {
        let plan = FormatPlan::new();
        assert!(plan.per_tensor.is_empty());
        assert_eq!(plan.get("anything"), None);
    }
}
