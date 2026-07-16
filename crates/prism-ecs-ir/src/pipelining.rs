//! Software pipelining system — overlap shared memory loads with compute.
//!
//! Analyzes block-level operations to identify shared memory loads and stores
//! (MemRef with memory_space == 3, the GPU shared/tile memory convention) and
//! groups them into pipeline stages. Each stage represents shared memory
//! operations that can be issued while compute is still in-flight.
//!
//! After pipelining, each operation in a pipeline stage carries a
//! [`PipelineLoadMarker`] or [`PipelineStoreMarker`] component with its stage
//! index. The program op itself receives a [`PipelineConfig`] component with
//! the total depth.

use prism_ecs_core::{Component, Entity, World};

use crate::block::{block_ops, BlockOps};
use crate::ir_attrs::Attribute;
use crate::ir_types::Type;
use crate::op::{op_name, operands, RegionRef};

// ── Pipeline components ──────────────────────────────────────────────────────

/// Pipeline configuration — attached to the program op after pipelining.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PipelineConfig {
    /// Pipeline depth (0 = no pipelining possible).
    pub depth: u32,
    /// Current pipeline stage being scheduled (priming / steady / drain).
    pub stage: u32,
}
impl Component for PipelineConfig {}

/// Parameter component that overrides the pipeline depth for a program.
///
/// When attached to a program entity, [`pipeline_program`] uses this depth
/// directly instead of computing it from shared memory patterns.  A depth of
/// 0 means no pipelining.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PipelineDepth(pub u32);
impl Component for PipelineDepth {}
impl From<u32> for PipelineDepth {
    fn from(d: u32) -> Self {
        Self(d)
    }
}

/// Marks a shared memory load with its pipeline stage index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PipelineLoadMarker(pub u32);

impl Component for PipelineLoadMarker {}
impl From<u32> for PipelineLoadMarker {
    fn from(stage: u32) -> Self {
        PipelineLoadMarker(stage)
    }
}

/// Marks a shared memory store with its pipeline stage index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PipelineStoreMarker(pub u32);

impl Component for PipelineStoreMarker {}
impl From<u32> for PipelineStoreMarker {
    fn from(stage: u32) -> Self {
        PipelineStoreMarker(stage)
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Check whether a value entity has a MemRef type pointing to shared memory
/// (memory_space == 3, the Metal/GPU tile-memory convention).
fn is_shared_memref(world: &World, val: Entity) -> bool {
    match crate::value::value_type(world, val) {
        Some(Type::MemRef(mr)) => match mr.memory_space.as_ref() {
            Attribute::Integer(val, _) => *val == 3,
            _ => false,
        },
        _ => false,
    }
}

/// True when `op` is a `memref.load` whose source memref lives in shared memory.
fn is_shared_load(world: &World, op: Entity) -> bool {
    if op_name(world, op).as_deref() != Some("memref.load") {
        return false;
    }
    let args = operands(world, op);
    args.first().map_or(false, |&v| is_shared_memref(world, v))
}

/// True when `op` is a `memref.store` whose destination lives in shared memory.
fn is_shared_store(world: &World, op: Entity) -> bool {
    if op_name(world, op).as_deref() != Some("memref.store") {
        return false;
    }
    let args = operands(world, op);
    // Last operand is the destination memref.
    args.last().map_or(false, |&v| is_shared_memref(world, v))
}

/// Extract the ordered list of ops from a program's first block.
fn program_block_ops(world: &World, program: Entity) -> Result<Vec<Entity>, String> {
    // Programs may either carry BlockOps directly (single-block program) or
    // own regions with blocks.
    if let Some(block) = world.get_component::<BlockOps>(program) {
        return Ok(block.0.clone());
    }

    // Try the region path: RegionRef → region → first block → ops.
    let regions = world
        .get_component::<RegionRef>(program)
        .map(|r| r.0.clone())
        .ok_or_else(|| "program has neither BlockOps nor RegionRef".to_string())?;

    let first_region = *regions
        .first()
        .ok_or_else(|| "program has zero regions".to_string())?;

    let blocks = crate::region::region_blocks(world, first_region);

    let first_block = blocks
        .first()
        .copied()
        .ok_or_else(|| "region has zero blocks".to_string())?;

    Ok(block_ops(world, first_block))
}

/// Group consecutive indices (contiguous runs) into stages, marking each op
/// with the given marker component.
fn mark_pipeline_stages<T: Component + From<u32>>(
    world: &mut World,
    ops: &[Entity],
    indices: &[usize],
) -> Result<u32, String> {
    if indices.is_empty() {
        return Ok(0);
    }

    let mut stage: u32 = 0;
    let mut prev: Option<usize> = None;

    for &idx in indices {
        // A new stage starts when there is a gap of more than one op between
        // consecutive loads/stores.
        if prev.map_or(true, |p| idx > p + 1) {
            if prev.is_some() {
                stage += 1;
            }
        }
        world
            .add_component(ops[idx], T::from(stage))
            .map_err(|e| format!("failed to add pipeline marker: {:?}", e))?;
        prev = Some(idx);
    }

    Ok(stage + 1) // number of stages = last stage index + 1
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Analyze a program and partition its shared memory loads/stores into
/// pipeline stages.
///
/// Returns the pipeline depth (number of stages), or an error if the program
/// structure cannot be traversed. A depth of 0 means no pipelining is possible
/// (no shared memory ops found).
///
/// After a successful call the program entity carries a [`PipelineConfig`]
/// component and each pipelined op carries a [`PipelineLoadMarker`] or
/// [`PipelineStoreMarker`].
pub fn pipeline_program(world: &mut World, program: Entity) -> Result<u32, String> {
    let ops = program_block_ops(world, program)?;

    // Check for an explicit PipelineDepth override on the program entity.
    if let Some(pd) = world.get_component::<PipelineDepth>(program) {
        let depth = pd.0;
        if depth == 0 {
            // Depth 0 means no pipelining — attach a zero-depth config.
            world
                .add_component(program, PipelineConfig { depth: 0, stage: 0 })
                .map_err(|e| format!("failed to add PipelineConfig: {:?}", e))?;
            return Ok(0);
        }
        // Explicit depth > 0: still analyze shared memory ops for stage
        // markers, but use the explicit depth in PipelineConfig.
        let load_indices: Vec<usize> = ops
            .iter()
            .enumerate()
            .filter(|(_, &op)| is_shared_load(world, op))
            .map(|(i, _)| i)
            .collect();

        let store_indices: Vec<usize> = ops
            .iter()
            .enumerate()
            .filter(|(_, &op)| is_shared_store(world, op))
            .map(|(i, _)| i)
            .collect();

        mark_pipeline_stages::<PipelineLoadMarker>(world, &ops, &load_indices)?;
        mark_pipeline_stages::<PipelineStoreMarker>(world, &ops, &store_indices)?;

        world
            .add_component(program, PipelineConfig { depth, stage: 0 })
            .map_err(|e| format!("failed to add PipelineConfig: {:?}", e))?;
        return Ok(depth);
    }

    // No explicit override — compute depth from shared memory ops.
    // Identify shared memory loads and stores by position.
    let load_indices: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|(_, &op)| is_shared_load(world, op))
        .map(|(i, _)| i)
        .collect();

    let store_indices: Vec<usize> = ops
        .iter()
        .enumerate()
        .filter(|(_, &op)| is_shared_store(world, op))
        .map(|(i, _)| i)
        .collect();

    // Group into stages (contiguous runs of loads / stores).
    let load_depth = mark_pipeline_stages::<PipelineLoadMarker>(world, &ops, &load_indices)?;
    let store_depth = mark_pipeline_stages::<PipelineStoreMarker>(world, &ops, &store_indices)?;
    let depth = load_depth.max(store_depth);

    world
        .add_component(program, PipelineConfig { depth, stage: 0 })
        .map_err(|e| format!("failed to add PipelineConfig: {:?}", e))?;

    Ok(depth)
}

/// Estimate the ideal pipeline depth based on shared memory latency vs. compute
/// latency for a given program.
///
/// Shared memory loads on Apple Silicon have roughly ~50 cycle latency. Compute
/// ops such as `linalg.dot` or `arith.mulf` take ~4-16 cycles.  This heuristic
/// uses `max(1, shared_mem_latency / avg_compute_latency)` with sensible
/// defaults when no concrete ops are found.
///
/// Unlike [`pipeline_program`], this function is read-only and does not mutate
/// the world.
pub fn estimate_pipeline_depth(world: &World, program: Entity) -> u32 {
    const SHARED_MEM_LATENCY: f64 = 50.0; // cycles
    const DEFAULT_COMPUTE_LATENCY: f64 = 10.0; // cycles

    let ops = match program_block_ops(world, program) {
        Ok(o) => o,
        Err(_) => return 0,
    };

    if ops.is_empty() {
        return 0;
    }

    // Compute average latency of known compute ops. Non-compute ops (loads,
    // stores, control flow) are skipped — they do not overlap with loads.
    let mut total_latency = 0.0f64;
    let mut count = 0usize;

    for &op in &ops {
        let lat = compute_op_latency(world, op);
        if lat > 0.0 {
            total_latency += lat;
            count += 1;
        }
    }

    let avg = if count > 0 {
        total_latency / count as f64
    } else {
        DEFAULT_COMPUTE_LATENCY
    };

    let depth = (SHARED_MEM_LATENCY / avg).ceil() as u32;
    depth.max(1)
}

/// Return the estimated latency (in cycles) of a compute operation, or 0.0 for
/// non-compute ops (loads, stores, control flow).
fn compute_op_latency(world: &World, op: Entity) -> f64 {
    match op_name(world, op).as_deref() {
        // Compute ops
        Some("linalg.dot") => 8.0,
        Some("linalg.matmul") => 16.0,
        Some("linalg.conv") => 12.0,
        Some("arith.addf") | Some("arith.addi") => 4.0,
        Some("arith.mulf") | Some("arith.muli") => 6.0,
        Some("arith.divf") | Some("arith.divi") => 12.0,
        Some("arith.subf") | Some("arith.subi") => 4.0,
        // Non-compute (loads, stores, control flow) → 0.0
        _ => 0.0,
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_attrs::Attribute;
    use crate::ir_types::{MemRefType, Type};
    use crate::op::{OpMarker, OpName, Operands, Results};
    use crate::value::{Uses, ValueDef, ValueType};
    use prism_ecs_core::{EntityKind, World};

    /// Helper: create a value entity with the given producer op and index.
    fn make_placeholder(world: &mut World) -> Entity {
        // Spawn an entity used as a dummy defining entity for external buffer
        // references (e.g., shared memory allocations not produced by any op).
        world.spawn(EntityKind::Node, None).unwrap().into()
    }

    /// Helper: create a value entity with the given producer op and index.
    fn make_value(world: &mut World, producer: Entity, idx: u32, ty: Type) -> Entity {
        let val: Entity = world.spawn(EntityKind::Node, None).unwrap().into();
        world
            .add_component(val, ValueDef::op_result(producer, idx))
            .unwrap();
        world.add_component(val, ValueType(ty)).unwrap();
        world.add_component(val, Uses(vec![])).unwrap();
        val
    }

    fn shared_memref(shape: Vec<u64>) -> Type {
        Type::MemRef(MemRefType::new(
            shape,
            Type::f32(),
            Attribute::integer(3, Type::i32()),
            Attribute::UnitAttr,
        ))
    }

    /// Create a global-memory MemRef type (memory_space == 0).
    fn global_memref(shape: Vec<u64>) -> Type {
        Type::MemRef(MemRefType::new(
            shape,
            Type::f32(),
            Attribute::integer(0, Type::i32()),
            Attribute::UnitAttr,
        ))
    }

    /// Create an op entity with a single result value.
    fn make_op(
        world: &mut World,
        name: &str,
        op_operands: &[Entity],
        result_type: Type,
    ) -> (Entity, Entity) {
        let op: Entity = world.spawn(EntityKind::Node, None).unwrap().into();
        world.add_component(op, OpMarker).unwrap();
        world.add_component(op, OpName(name.to_string())).unwrap();
        world
            .add_component(op, Operands(op_operands.to_vec()))
            .unwrap();

        let val = make_value(world, op, 0, result_type);
        world.add_component(op, Results(vec![val])).unwrap();

        for &operand in op_operands {
            if let Some(u) = world.get_component_mut::<Uses>(operand) {
                u.0.push(op);
            }
        }

        (op, val)
    }

    /// Create an op with no results (e.g., memref.store).
    fn make_op_no_result(world: &mut World, name: &str, op_operands: &[Entity]) -> Entity {
        let op: Entity = world.spawn(EntityKind::Node, None).unwrap().into();
        world.add_component(op, OpMarker).unwrap();
        world.add_component(op, OpName(name.to_string())).unwrap();
        world
            .add_component(op, Operands(op_operands.to_vec()))
            .unwrap();
        world.add_component(op, Results(vec![])).unwrap();
        op
    }

    /// Create a program entity with a single block.
    fn make_program(world: &mut World, ops: &[Entity]) -> Entity {
        let prog: Entity = world
            .spawn(EntityKind::Node, Some("program".into()))
            .unwrap()
            .into();
        world.add_component(prog, OpMarker).unwrap();
        world
            .add_component(prog, OpName("func.func".to_string()))
            .unwrap();
        world.add_component(prog, BlockOps(ops.to_vec())).unwrap();
        world.add_component(prog, RegionRef(vec![])).unwrap();
        world.add_component(prog, Results(vec![])).unwrap();
        world.add_component(prog, Operands(vec![])).unwrap();
        prog
    }

    // ── Tests ─────────────────────────────────────────────────────────────

    #[test]
    fn pipeline_load_and_dot_returns_depth_gt_zero() {
        let mut world = World::new();
        let placeholder = make_placeholder(&mut world);

        // Program: [shared_load(A)] → [shared_load(B)] → [linalg.dot(A_loaded, B_loaded)]
        // Create a shared-memory memref value.
        let shared_src_a = make_value(&mut world, placeholder, 0, shared_memref(vec![16, 16]));
        let shared_src_b = make_value(&mut world, placeholder, 0, shared_memref(vec![16, 16]));

        // Load from shared memory — produces a scalar f32.
        let (load_a, loaded_a) = make_op(&mut world, "memref.load", &[shared_src_a], Type::f32());
        let (load_b, loaded_b) = make_op(&mut world, "memref.load", &[shared_src_b], Type::f32());

        // Dot product consuming the loaded values.
        let (_dot_op, _dot_val) =
            make_op(&mut world, "linalg.dot", &[loaded_a, loaded_b], Type::f32());

        let prog = make_program(&mut world, &[load_a, load_b, _dot_op]);

        let depth = pipeline_program(&mut world, prog).unwrap();
        assert!(
            depth > 0,
            "expected non-zero pipeline depth for shared load + dot"
        );

        // Both loads should be marked as stage 0 (consecutive).
        let la_marker = world
            .get_component::<PipelineLoadMarker>(load_a)
            .expect("load_a should have PipelineLoadMarker");
        assert_eq!(la_marker.0, 0, "load_a should be stage 0");

        let lb_marker = world
            .get_component::<PipelineLoadMarker>(load_b)
            .expect("load_b should have PipelineLoadMarker");
        assert_eq!(lb_marker.0, 0, "load_b should be stage 0");

        // The dot op should NOT have a load marker.
        assert!(
            world.get_component::<PipelineLoadMarker>(_dot_op).is_none(),
            "dot op should not be marked as a shared load"
        );

        // The program should have PipelineConfig with matching depth.
        let cfg = world
            .get_component::<PipelineConfig>(prog)
            .expect("program should have PipelineConfig");
        assert_eq!(
            cfg.depth, depth,
            "PipelineConfig depth should match returned depth"
        );
        assert_eq!(cfg.stage, 0, "PipelineConfig stage should be 0 initially");
    }

    #[test]
    fn pipeline_no_memory_ops_returns_zero() {
        let mut world = World::new();

        // Program with only compute ops (no shared memory loads/stores).
        let (addf_op, addf_val) = make_op(&mut world, "arith.addf", &[], Type::f32());
        let (_mulf_op, _) = make_op(&mut world, "arith.mulf", &[addf_val], Type::f32());

        let prog = make_program(&mut world, &[addf_op, _mulf_op]);

        let depth = pipeline_program(&mut world, prog).unwrap();
        assert_eq!(
            depth, 0,
            "expected depth 0 for program with no shared memory ops"
        );

        // PipelineConfig should still be present with depth 0.
        let cfg = world
            .get_component::<PipelineConfig>(prog)
            .expect("program should have PipelineConfig");
        assert_eq!(cfg.depth, 0);
    }

    #[test]
    fn pipeline_shared_loads_spanning_stages() {
        let mut world = World::new();
        let placeholder = make_placeholder(&mut world);

        // Program: [load_A] → [load_B] → [dot(load_A, load_B)] → [load_C] → [load_D] → [dot(load_C, load_D)]
        // Loads A and B are consecutive (stage 0), loads C and D are consecutive (stage 1).
        let a_val = make_value(&mut world, placeholder, 0, shared_memref(vec![16]));
        let b_val = make_value(&mut world, placeholder, 0, shared_memref(vec![16]));
        let c_val = make_value(&mut world, placeholder, 0, shared_memref(vec![16]));
        let d_val = make_value(&mut world, placeholder, 0, shared_memref(vec![16]));

        let (load_a, l_a) = make_op(&mut world, "memref.load", &[a_val], Type::f32());
        let (load_b, l_b) = make_op(&mut world, "memref.load", &[b_val], Type::f32());
        let (dot_1, _) = make_op(&mut world, "linalg.dot", &[l_a, l_b], Type::f32());
        let (load_c, l_c) = make_op(&mut world, "memref.load", &[c_val], Type::f32());
        let (load_d, l_d) = make_op(&mut world, "memref.load", &[d_val], Type::f32());
        let (_dot_2, _) = make_op(&mut world, "linalg.dot", &[l_c, l_d], Type::f32());

        let prog = make_program(&mut world, &[load_a, load_b, dot_1, load_c, load_d, _dot_2]);

        let depth = pipeline_program(&mut world, prog).unwrap();
        assert_eq!(
            depth, 2,
            "expected depth 2 for two interleaved load clusters"
        );

        // Verify stage assignments.
        assert_eq!(
            world.get_component::<PipelineLoadMarker>(load_a).unwrap().0,
            0
        );
        assert_eq!(
            world.get_component::<PipelineLoadMarker>(load_b).unwrap().0,
            0
        );
        assert_eq!(
            world.get_component::<PipelineLoadMarker>(load_c).unwrap().0,
            1
        );
        assert_eq!(
            world.get_component::<PipelineLoadMarker>(load_d).unwrap().0,
            1
        );
    }

    #[test]
    fn pipeline_global_memref_not_pipelined() {
        let mut world = World::new();
        let placeholder = make_placeholder(&mut world);

        // Load from global memory (memory_space = 0) should not be pipelined.
        let global_src = make_value(&mut world, placeholder, 0, global_memref(vec![16]));
        let (load_g, loaded_g) = make_op(&mut world, "memref.load", &[global_src], Type::f32());
        let (_dot, _) = make_op(&mut world, "linalg.dot", &[loaded_g], Type::f32());

        let prog = make_program(&mut world, &[load_g, _dot]);

        let depth = pipeline_program(&mut world, prog).unwrap();
        assert_eq!(
            depth, 0,
            "global memory loads should not create pipeline stages"
        );
        assert!(
            world.get_component::<PipelineLoadMarker>(load_g).is_none(),
            "global memory load should not be marked"
        );
    }

    #[test]
    fn pipeline_shared_store() {
        let mut world = World::new();
        let placeholder = make_placeholder(&mut world);

        // Shared memory store: memref.store(value, shared_memref)
        let loaded_val = make_value(&mut world, placeholder, 0, Type::f32());
        let shared_dst = make_value(&mut world, placeholder, 0, shared_memref(vec![16]));

        let store_op = make_op_no_result(&mut world, "memref.store", &[loaded_val, shared_dst]);
        let prog = make_program(&mut world, &[store_op]);

        let depth = pipeline_program(&mut world, prog).unwrap();
        assert!(depth > 0, "shared store should create pipeline stages");

        let sm = world
            .get_component::<PipelineStoreMarker>(store_op)
            .expect("store should have PipelineStoreMarker");
        assert_eq!(sm.0, 0, "store should be stage 0");
    }

    #[test]
    fn estimate_depth_with_compute_ops() {
        let mut world = World::new();
        let placeholder = make_placeholder(&mut world);

        let (addf_op, addf_val) = make_op(&mut world, "arith.addf", &[], Type::f32());
        let shared_src = make_value(&mut world, placeholder, 0, shared_memref(vec![16]));
        let (load_op, loaded) = make_op(&mut world, "memref.load", &[shared_src], Type::f32());
        let (_dot, _) = make_op(&mut world, "linalg.dot", &[addf_val, loaded], Type::f32());

        let prog = make_program(&mut world, &[load_op, addf_op, _dot]);

        let depth = estimate_pipeline_depth(&world, prog);
        // With addf(4) + dot(8) = avg 6 → 50/6 ≈ 8.3 → ceil → 9
        assert!(depth > 0, "expected non-zero estimated depth");
        assert_eq!(depth, 9, "50 / avg(4, 8) = 9");
    }

    #[test]
    fn estimate_depth_empty_program() {
        let mut world = World::new();
        let prog = make_program(&mut world, &[]);
        assert_eq!(
            estimate_pipeline_depth(&world, prog),
            0,
            "empty program -> 0"
        );
    }

    #[test]
    fn pipeline_depth_override_uses_explicit_depth() {
        let mut world = World::new();
        let placeholder = make_placeholder(&mut world);

        // Shared loads that would normally produce depth 2 (two load clusters
        // separated by a dot op).
        let a_val = make_value(&mut world, placeholder, 0, shared_memref(vec![16]));
        let b_val = make_value(&mut world, placeholder, 0, shared_memref(vec![16]));
        let c_val = make_value(&mut world, placeholder, 0, shared_memref(vec![16]));
        let d_val = make_value(&mut world, placeholder, 0, shared_memref(vec![16]));

        let (load_a, l_a) = make_op(&mut world, "memref.load", &[a_val], Type::f32());
        let (load_b, l_b) = make_op(&mut world, "memref.load", &[b_val], Type::f32());
        let (dot_1, _) = make_op(&mut world, "linalg.dot", &[l_a, l_b], Type::f32());
        let (load_c, l_c) = make_op(&mut world, "memref.load", &[c_val], Type::f32());
        let (load_d, l_d) = make_op(&mut world, "memref.load", &[d_val], Type::f32());
        let (_dot_2, _) = make_op(&mut world, "linalg.dot", &[l_c, l_d], Type::f32());

        let prog = make_program(&mut world, &[load_a, load_b, dot_1, load_c, load_d, _dot_2]);

        // Attach explicit depth 3 (overrides the natural depth of 2).
        world.add_component(prog, PipelineDepth(3)).unwrap();

        let depth = pipeline_program(&mut world, prog).unwrap();
        assert_eq!(
            depth, 3,
            "expected depth 3 from explicit PipelineDepth override"
        );

        // PipelineConfig should carry the overridden depth.
        let cfg = world
            .get_component::<PipelineConfig>(prog)
            .expect("program should have PipelineConfig");
        assert_eq!(
            cfg.depth, 3,
            "PipelineConfig depth should be the overridden value"
        );

        // Stage markers should still be assigned from the actual shared memory
        // pattern (two clusters, stages 0 and 1).
        assert_eq!(
            world.get_component::<PipelineLoadMarker>(load_a).unwrap().0,
            0
        );
        assert_eq!(
            world.get_component::<PipelineLoadMarker>(load_b).unwrap().0,
            0
        );
        assert_eq!(
            world.get_component::<PipelineLoadMarker>(load_c).unwrap().0,
            1
        );
        assert_eq!(
            world.get_component::<PipelineLoadMarker>(load_d).unwrap().0,
            1
        );
    }

    #[test]
    fn pipeline_depth_override_zero_skips_pipelining() {
        let mut world = World::new();
        let placeholder = make_placeholder(&mut world);

        let shared_src = make_value(&mut world, placeholder, 0, shared_memref(vec![16]));
        let (load_op, _loaded) = make_op(&mut world, "memref.load", &[shared_src], Type::f32());

        let prog = make_program(&mut world, &[load_op]);

        // Attach explicit depth 0 — skip pipelining.
        world.add_component(prog, PipelineDepth(0)).unwrap();

        let depth = pipeline_program(&mut world, prog).unwrap();
        assert_eq!(depth, 0, "expected depth 0 when PipelineDepth(0) is set");

        // No marker should be attached.
        assert!(
            world.get_component::<PipelineLoadMarker>(load_op).is_none(),
            "no load marker should be attached when depth is 0"
        );

        let cfg = world
            .get_component::<PipelineConfig>(prog)
            .expect("program should have PipelineConfig");
        assert_eq!(cfg.depth, 0, "PipelineConfig depth should be 0");
    }

    #[test]
    fn pipeline_no_depth_override_uses_standard_analysis() {
        let mut world = World::new();

        let (addf_op, addf_val) = make_op(&mut world, "arith.addf", &[], Type::f32());
        let placeholder = make_placeholder(&mut world);
        let shared_src = make_value(&mut world, placeholder, 0, shared_memref(vec![16]));
        let (load_op, loaded) = make_op(&mut world, "memref.load", &[shared_src], Type::f32());
        let (_dot, _) = make_op(&mut world, "linalg.dot", &[addf_val, loaded], Type::f32());

        let prog = make_program(&mut world, &[load_op, addf_op, _dot]);

        // No PipelineDepth attached — standard contiguity analysis.
        let depth = pipeline_program(&mut world, prog).unwrap();
        assert_eq!(depth, 1, "one shared load cluster = depth 1");

        let cfg = world
            .get_component::<PipelineConfig>(prog)
            .expect("program should have PipelineConfig");
        assert_eq!(cfg.depth, 1, "fallback depth from shared memory analysis");
    }

    #[test]
    fn pipeline_estimate_fallback_not_affected_by_pipeline_depth() {
        // Verify estimate_pipeline_depth is unaffected — it stays read-only and
        // never consults PipelineDepth.
        let mut world = World::new();
        let placeholder = make_placeholder(&mut world);
        let shared_src = make_value(&mut world, placeholder, 0, shared_memref(vec![16]));
        let (load_op, loaded) = make_op(&mut world, "memref.load", &[shared_src], Type::f32());
        let (_dot, _) = make_op(&mut world, "linalg.dot", &[loaded, loaded], Type::f32());

        let prog = make_program(&mut world, &[load_op, _dot]);

        let estimated = estimate_pipeline_depth(&world, prog);
        // dot=8 cycles, load skipped → 50/8 = 6.25 → ceil → 7
        assert_eq!(estimated, 7, "50 / dot(8) = 7");
    }
}
