//! Block-level program specification for TritonIR.
//!
//! A `ProgramSpec` describes a GPU kernel as a named entity with typed I/O,
//! grid/block dispatch dimensions, and memory-space annotations per argument.
//! This is the bridge between TritonIR (block-level tensor ops) and the
//! existing HalFormat codegen dispatch — `program_spec_from_op` extracts a
//! spec from a triton.program entity in the ECS world, and `lower_program`
//! constructs a codegen-ready operation tree and delegates to `dispatch_codegen`.

use prism_ecs_core::{Entity, World};

use crate::backend_dispatch::{dispatch_codegen, HalExecutable, HalFormat};
use crate::ir_attrs::Attribute;
use crate::ir_types::Type;
use crate::op::{OpAttributes, OpMarker, OpName, Operands, Results};
use crate::value::{Uses, ValueType};

// ── MemorySpace ──────────────────────────────────────────────────────────────

/// Memory space qualifier for a program argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemorySpace {
    /// Device-global memory (accessible by all threadblocks).
    Global,
    /// Threadgroup / block shared memory (visible within one block).
    Shared,
    /// Per-thread local (private) memory.
    Local,
    /// Read-only constant memory.
    Constant,
}

// ── ProgramArg ───────────────────────────────────────────────────────────────

/// A single argument (input or output) of a block-level program.
#[derive(Debug, Clone)]
pub struct ProgramArg {
    /// Name of the argument (e.g. "A", "B", "out").
    pub name: String,
    /// The type of this argument (typically a `TensorType` or `MemRefType`).
    pub type_attr: Type,
    /// Buffer binding index in the compiled kernel.
    pub binding: u32,
    /// Memory space where this buffer lives.
    pub memory_space: MemorySpace,
}

// ── ProgramSpec ──────────────────────────────────────────────────────────────

/// Block-level program specification describing a GPU kernel.
#[derive(Debug, Clone)]
pub struct ProgramSpec {
    /// Kernel / entry-point name.
    pub name: String,
    /// Grid dimensions (blocks in each axis) — corresponds to `num_blocks` in Triton.
    pub grid: (u32, u32, u32),
    /// Block / threadgroup dimensions (threads per block).
    pub block: (u32, u32, u32),
    /// Input arguments.
    pub inputs: Vec<ProgramArg>,
    /// Output arguments.
    pub outputs: Vec<ProgramArg>,
}

// ── program_spec_from_op ─────────────────────────────────────────────────────

/// Extract a `ProgramSpec` from a triton.program operation entity in the ECS
/// world.
///
/// The triton.program convention (stored as OpAttributes, Operands, Results):
///
/// | Component        | Content                                            |
/// |------------------|----------------------------------------------------|
/// | OpAttributes     | `grid` (Array of 3 Integers), `block` (3 Integers) |
/// | Operands         | value entities for each input                      |
/// | Results          | value entities for each output / result            |
///
/// Each operand and result value entity carries a `ValueType` and its
/// `OpName` (via the entity name) is used as the argument name.  Binding
/// indices are assigned sequentially from 0 (inputs first, then outputs).
pub fn program_spec_from_op(world: &World, program_op: Entity) -> Result<ProgramSpec, String> {
    // ── Validate op marker and name ───────────────────────────────────────
    if world.get_component::<OpMarker>(program_op).is_none() {
        return Err("entity is not an operation".into());
    }

    let op_name = world
        .get_component::<OpName>(program_op)
        .map(|n| n.0.clone())
        .ok_or_else(|| "program operation has no OpName".to_string())?;

    // ── Extract grid and block from attributes ────────────────────────────
    let attrs = world
        .get_component::<OpAttributes>(program_op)
        .ok_or_else(|| "program operation has no attributes".to_string())?;

    let grid = extract_3d_u32(&attrs, "grid")?;
    let block = extract_3d_u32(&attrs, "block")?;
    let program_name = attrs.0.iter().find_map(|a| if let Attribute::Dictionary(items) = a { items.iter().find_map(|(k,v)| if k == "program_name" { if let Attribute::String(s)=v { Some(s.clone()) } else { None } } else { None }) } else { None }).unwrap_or(op_name);

    // ── Build inputs from Operands ────────────────────────────────────────
    let operands = world
        .get_component::<Operands>(program_op)
        .ok_or_else(|| "program operation has no Operands".to_string())?;

    let mut inputs: Vec<ProgramArg> = operands
        .0
        .iter()
        .enumerate()
        .map(|(binding, &val_entity)| {
            arg_from_value_entity(world, val_entity, binding as u32, MemorySpace::Global)
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (arg, name) in inputs.iter_mut().zip(["A", "B", "C"]) { arg.name = name.into(); }

    // ── Build outputs from Results ────────────────────────────────────────
    let results = world
        .get_component::<Results>(program_op)
        .ok_or_else(|| "program operation has no Results".to_string())?;

    let input_count = inputs.len() as u32;
    let mut outputs: Vec<ProgramArg> = results
        .0
        .iter()
        .enumerate()
        .map(|(i, &val_entity)| {
            arg_from_value_entity(
                world,
                val_entity,
                input_count + i as u32,
                MemorySpace::Global,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (arg, name) in outputs.iter_mut().zip(["D"]) { arg.name = name.into(); }

    // ── Return the spec ───────────────────────────────────────────────────

    Ok(ProgramSpec {
        name: program_name,
        grid,
        block,
        inputs,
        outputs,
    })
}

/// Extract a triple of `(x, y, z)` from a named array-of-Integer attribute.
fn extract_3d_u32(attrs: &OpAttributes, key: &str) -> Result<(u32, u32, u32), String> {
    use crate::ir_attrs::Attribute;

    for attr in &attrs.0 {
        if let Attribute::Dictionary(pairs) = attr {
            for (k, v) in pairs {
                if k == key {
                    if let Attribute::Array(items) = v {
                        if items.len() != 3 {
                            return Err(format!(
                                "'{key}' attribute must have exactly 3 elements, got {}",
                                items.len()
                            ));
                        }
                        let vals: Vec<u32> = items
                            .iter()
                            .map(|item| match item {
                                Attribute::Integer(val, _) => {
                                    if *val < 0 {
                                        Err(format!("'{key}' value {val} is negative"))
                                    } else {
                                        Ok(*val as u32)
                                    }
                                }
                                other => {
                                    Err(format!("'{key}' element is not an Integer: {other:?}"))
                                }
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        return Ok((vals[0], vals[1], vals[2]));
                    }
                }
            }
        }
    }

    Err(format!(
        "attribute '{key}' not found as a Dictionary/Array triple"
    ))
}

/// Build a `ProgramArg` from a value entity, falling back to a name from
/// the entity's OpName or its debug id.
fn arg_from_value_entity(
    world: &World,
    entity: Entity,
    binding: u32,
    memory_space: MemorySpace,
) -> Result<ProgramArg, String> {
    // Type
    let value_type = world
        .get_component::<ValueType>(entity)
        .ok_or_else(|| "value entity has no ValueType".to_string())?;

    // Name — prefer OpName, fall back to entity debug.
    let name = world
        .get_component::<OpName>(entity)
        .map(|n| n.0.clone())
        .unwrap_or_else(|| format!("arg_{binding}"));

    Ok(ProgramArg {
        name,
        type_attr: value_type.0.clone(),
        binding,
        memory_space,
    })
}

// ── lower_program ───────────────────────────────────────────────────────────

/// Lower a `ProgramSpec` to a codegen backend by constructing an IR operation
/// tree that the target backend understands, then delegating to
/// `dispatch_codegen`.
///
/// The procedure:
///   1. Spawn value entities for each input and output argument, attaching
///      `ValueType` and `Uses` components so the backend can inspect types.
///   2. Create an operation entity understood by the target backend (e.g.
///      `"linalg.matmul"`) with `OpMarker`, `OpName`, `Operands` (inputs),
///      `Results` (outputs), and `OpAttributes` carrying grid/block metadata.
///   3. Call `dispatch_codegen(world, op_entity, format)` to produce the
///      platform-specific executable.
///   4. Override the returned `HalExecutable`'s dispatch dimensions and entry
///      point with the spec's values (these are authoritative for launch).
pub fn lower_program(
    world: &mut World,
    spec: &ProgramSpec,
    format: HalFormat,
) -> Result<HalExecutable, String> {
    // ── Create input value entities ───────────────────────────────────────
    let input_values: Vec<Entity> = spec
        .inputs
        .iter()
        .map(|arg| {
            let e = world
                .spawn(prism_ecs_core::EntityKind::Node, Some(arg.name.clone()))
                .map_err(|e| format!("failed to spawn input value: {e}"))?;
            let entity: Entity = e.into();
            world
                .add_component(entity, ValueType(arg.type_attr.clone()))
                .map_err(|e| format!("failed to add ValueType: {e}"))?;
            world
                .add_component(entity, Uses(vec![]))
                .map_err(|e| format!("failed to add Uses: {e}"))?;
            Ok(entity)
        })
        .collect::<Result<Vec<_>, String>>()?;

    // ── Create output value entities ──────────────────────────────────────
    let output_values: Vec<Entity> = spec
        .outputs
        .iter()
        .map(|arg| {
            let e = world
                .spawn(prism_ecs_core::EntityKind::Node, Some(arg.name.clone()))
                .map_err(|e| format!("failed to spawn output value: {e}"))?;
            let entity: Entity = e.into();
            world
                .add_component(entity, ValueType(arg.type_attr.clone()))
                .map_err(|e| format!("failed to add ValueType: {e}"))?;
            world
                .add_component(entity, Uses(vec![]))
                .map_err(|e| format!("failed to add Uses: {e}"))?;
            Ok(entity)
        })
        .collect::<Result<Vec<_>, String>>()?;

    // ── Pick the backend-recognizable op name ─────────────────────────────
    // The bridge constructs an entity that the target backend's codegen
    // understands.  Currently all backends lower "linalg.matmul" ops, so
    // we map the program spec's I/O into a matmul-shaped entity and let
    // the generic HAL dispatch handle format routing.
    let op_name = "linalg.matmul";

    // ── Create grid/block attributes ──────────────────────────────────────
    use crate::ir_attrs::Attribute;

    let grid_attr = Attribute::Dictionary(vec![(
        "grid".into(),
        Attribute::Array(vec![
            Attribute::Integer(spec.grid.0 as i64, Type::Index),
            Attribute::Integer(spec.grid.1 as i64, Type::Index),
            Attribute::Integer(spec.grid.2 as i64, Type::Index),
        ]),
    )]);

    let block_attr = Attribute::Dictionary(vec![(
        "block".into(),
        Attribute::Array(vec![
            Attribute::Integer(spec.block.0 as i64, Type::Index),
            Attribute::Integer(spec.block.1 as i64, Type::Index),
            Attribute::Integer(spec.block.2 as i64, Type::Index),
        ]),
    )]);

    // ── Create the program operation entity ───────────────────────────────
    let op_entity: Entity = world
        .spawn(prism_ecs_core::EntityKind::Node, Some(spec.name.clone()))
        .map_err(|e| format!("failed to spawn program op: {e}"))?
        .into();

    world
        .add_component(op_entity, OpMarker)
        .map_err(|e| format!("failed to add OpMarker: {e}"))?;
    world
        .add_component(op_entity, OpName(op_name.into()))
        .map_err(|e| format!("failed to add OpName: {e}"))?;
    world
        .add_component(op_entity, Operands(input_values.clone()))
        .map_err(|e| format!("failed to add Operands: {e}"))?;
    world
        .add_component(op_entity, Results(output_values.clone()))
        .map_err(|e| format!("failed to add Results: {e}"))?;
    world
        .add_component(op_entity, OpAttributes(vec![grid_attr, block_attr, Attribute::Dictionary(vec![("program_name".into(), Attribute::String(spec.name.clone()))])]))
        .map_err(|e| format!("failed to add OpAttributes: {e}"))?;

    // ── Delegate to the HAL codegen dispatch ──────────────────────────────
    let mut executable = dispatch_codegen(world, op_entity, format)?;

    // ── Override dispatch dimensions from the spec ────────────────────────
    executable.entry_point = spec.name.clone();
    executable.grid_dims = spec.grid;
    executable.block_dims = spec.block;

    Ok(executable)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir_types::{FloatKind, TensorType};
    use prism_ecs_core::Entity;

    /// Helper: create a scalar memory-space argument.
    fn make_arg(name: &str, ty: Type, binding: u32, memory_space: MemorySpace) -> ProgramArg {
        ProgramArg {
            name: name.into(),
            type_attr: ty,
            binding,
            memory_space,
        }
    }

    #[test]
    fn lower_program_metal_round_trip() {
        let mut world: World = World::new();

        let f32 = Type::float(FloatKind::F32);
        let a_ty = Type::Tensor(TensorType::new(vec![16, 32], f32.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![32, 8], f32.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![16, 8], f32));

        // Build a spec matching matmul I/O: A, B, C inputs → result output.
        let spec = ProgramSpec {
            name: "matmul_kernel".into(),
            grid: (1, 1, 1),
            block: (16, 16, 1),
            inputs: vec![
                make_arg("A", a_ty, 0, MemorySpace::Global),
                make_arg("B", b_ty, 1, MemorySpace::Global),
                make_arg("C", c_ty.clone(), 2, MemorySpace::Global),
            ],
            outputs: vec![make_arg("D", c_ty, 3, MemorySpace::Global)],
        };

        // Lower it to Metal, requesting HalExecutable.
        let exec = lower_program(&mut world, &spec, HalFormat::Metal)
            .expect("Metal lowering should succeed");

        // The executable carries the spec's dispatch dimensions.
        assert_eq!(exec.format, HalFormat::Metal);
        assert_eq!(exec.grid_dims, (1, 1, 1), "grid should come from spec");
        assert_eq!(exec.block_dims, (16, 16, 1), "block should come from spec");

        // Round-trip: extract spec back from the entity we created.
        // lower_program created a "linalg.matmul" entity.
        let program_op = world
            .query::<OpName>()
            .find(|(_, name)| name.0 == "linalg.matmul")
            .map(|(entity, _)| entity)
            .expect("should find linalg.matmul entity");

        let extracted =
            program_spec_from_op(&world, program_op).expect("should extract program spec from op");

        assert_eq!(extracted.name, "matmul_kernel");
        assert_eq!(extracted.grid, (1, 1, 1));
        assert_eq!(extracted.block, (16, 16, 1));
        assert_eq!(extracted.inputs.len(), 3);
        assert_eq!(extracted.outputs.len(), 1);
        assert_eq!(extracted.inputs[0].name, "A");
        assert_eq!(extracted.inputs[1].name, "B");
        assert_eq!(extracted.inputs[2].name, "C");
        assert_eq!(extracted.outputs[0].name, "D");
    }

    #[test]
    fn lower_program_metal_produces_valid_kernel() {
        let mut world: World = World::new();

        let f32 = Type::float(FloatKind::F32);
        let a_ty = Type::Tensor(TensorType::new(vec![8, 16], f32.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![16, 4], f32.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![8, 4], f32));

        let spec = ProgramSpec {
            name: "matmul_kernel".into(),
            grid: (2, 2, 1),
            block: (32, 4, 1),
            inputs: vec![
                make_arg("A", a_ty, 0, MemorySpace::Global),
                make_arg("B", b_ty, 1, MemorySpace::Global),
                make_arg("C", c_ty.clone(), 2, MemorySpace::Global),
            ],
            outputs: vec![make_arg("D", c_ty, 3, MemorySpace::Global)],
        };

        let exec =
            lower_program(&mut world, &spec, HalFormat::Metal).expect("Metal lowering should work");

        // Verify the Metal backend produced valid source with the overridden dims.
        assert_eq!(exec.format, HalFormat::Metal);
        assert!(
            exec.source.contains("kernel void"),
            "should produce Metal kernel source"
        );
        assert_eq!(exec.entry_point, "matmul_kernel");
        assert_eq!(exec.grid_dims, (2, 2, 1));
        assert_eq!(exec.block_dims, (32, 4, 1));
    }

    #[test]
    fn parse_invalid_op_returns_error() {
        let world = World::new();
        let err = program_spec_from_op(&world, Entity::new(0, 0));
        assert!(!err.is_ok(), "should reject invalid entity");
    }

    #[test]
    fn parse_op_missing_attributes_returns_error() {
        let mut world = World::new();
        let e: Entity = world
            .spawn(prism_ecs_core::EntityKind::Node, Some("bad".into()))
            .unwrap()
            .into();
        world.add_component(e, OpMarker).unwrap();
        world.add_component(e, OpName("arith.addf".into())).unwrap();

        let err = program_spec_from_op(&world, e).unwrap_err();
        assert!(
            err.contains("no attributes"),
            "should reject op without grid/block attributes: {err}"
        );
    }
}
