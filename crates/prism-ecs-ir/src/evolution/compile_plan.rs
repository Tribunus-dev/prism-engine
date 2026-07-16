//! Compilation plan types for evolution-driven format and tile assignment.
//!
//! Each tensor in a compilation plan is assigned a quantization format and
//! tile geometry by the evolutionary search. These types provide the ECS
//! components and query functions that backends use to read assignments.

use crate::evolution::mutation_table::TensorFormat;
use prism_ecs_core::{Component, Entity, World};

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
) -> Option<(TensorFormat, crate::evolution::mutation_table::TensorOperation)> {
    let fmt = world.get_component::<FormatAssignment>(tensor)?.0;
    // Derive the default operation from the format
    let op = match fmt {
        TensorFormat::Ternary158 => crate::evolution::mutation_table::TensorOperation::TernaryGemm,
        TensorFormat::Binary1 => crate::evolution::mutation_table::TensorOperation::BinaryPopcountGemm,
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
    _m: u32,
    _n: u32,
    _k: u32,
) -> (u32, u32, u32) {
    if let Some(plan_ref) = world.get_component::<CompilePlanRef>(matmul_op) {
        if let Some(tiles) = world.get_component::<TileSizes>(plan_ref.0) {
            return (tiles.tile_m, tiles.tile_n, tiles.tile_k);
        }
    }
    // Default tile sizes: 64×64×32
    (64, 64, 32)
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
        let fake = Entity(999, 0);
        let (tm, tn, tk) = resolve_matmul_tile(&world, fake, 1024, 1024, 1024);
        assert_eq!(tm, 64);
        assert_eq!(tn, 64);
        assert_eq!(tk, 32);
    }
}
