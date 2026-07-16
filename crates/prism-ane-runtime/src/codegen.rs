//! ANE code generation — lowers ECS-native IR operations to Apple MIL programs.
//!
//! Originally migrated from `prism-ecs-ir::backend_ane` into its own runtime
//! crate. Follows the prism-metal-runtime pattern: compile source → binary →
//! dispatch → evidence.
//!
//! The ANE uses a graph-of-layers programming model rather than GPU-style
//! grid dispatch.  Each layer is a named operation with typed inputs and
//! outputs.  The generated output is a textual MIL (Apple Neural Engine's
//! intermediate language) program description.

use prism_ecs_core::{Entity, World};

use prism_ecs_ir::ir_types::{FloatKind, FloatType, TensorType, Type};
use prism_ecs_ir::op::{op_name, operands};
use prism_ecs_ir::value::ValueType;

/// Error type for ANE lowering failures.
#[derive(Debug)]
pub enum AneLowerError {
    /// The operation is not one that can be lowered to ANE MIL.
    UnsupportedOp(String),
    /// A required operand or result is missing.
    MissingOperand(String),
    /// An operand or result is missing a type annotation.
    MissingType(String),
}

// ── Element type mapping ─────────────────────────────────────────────────────

/// Map an IR element type to its MIL scalar type name.
fn element_type_to_mil(ty: &Type) -> &'static str {
    match ty {
        Type::Float(FloatType { kind }) => match kind {
            FloatKind::F16 => "float16",
            FloatKind::BF16 => "bfloat16",
            FloatKind::F32 => "float32",
            FloatKind::F64 => "float64",
            FloatKind::F8E4M3 | FloatKind::F8E5M2 => "float8",
        },
        Type::Integer(_) => "int32",
        Type::Index
        | Type::NoneType
        | Type::Function(_)
        | Type::Tensor(_)
        | Type::Vector(_)
        | Type::Complex(_) => "float16",
    }
}

/// Extract a `TensorType` from a value entity's `ValueType` component.
fn require_tensor(world: &World, entity: Entity, label: &str) -> Result<TensorType, AneLowerError> {
    let value_ty = world
        .get_component::<ValueType>(entity)
        .ok_or_else(|| AneLowerError::MissingType(format!("{label} is missing ValueType")))?;

    match &value_ty.0 {
        Type::Tensor(t) => Ok(t.clone()),
        other => Err(AneLowerError::MissingType(format!(
            "{label} has non-tensor type {other:?}"
        ))),
    }
}

// ── Emitters ─────────────────────────────────────────────────────────────────

/// Emit a MIL program for a matmul operation.
#[rustfmt::skip]
fn emit_matmul_mil(m: u64, n: u64, k: u64, mil_type: &str) -> String {
    format!(
        r#"// ANE Program: matmul_{m}x{k}x{n}
// Input: A[{m},{k}], B[{k},{n}]
// Output: C[{m},{n}]
MIL PROGRAM matmul_{m}x{k}x{n} {{
  layer @0 = matmul(inputs: [A, B], output: C,
    M: {m}, K: {k}, N: {n},
    type: {mil_type})
}}
"#,
        m = m, n = n, k = k, mil_type = mil_type,
    )
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Lower a `linalg.matmul` to an ANE MIL program.
///
/// Given a `linalg.matmul` op consuming operands `A`, `B`, `C` where the
/// semantics are `C += A @ B`, this function emits a textual MIL pseudo-code
/// program describing the neural network layer.
///
/// Each operand **must** carry a `tensor<...>` type with a 2-D shape so that
/// the dimensions `M`, `K`, `N` can be extracted.
pub fn lower_matmul_to_ane(world: &World, matmul_op: Entity) -> Result<String, AneLowerError> {
    // 1. Verify the op is a matmul
    let name = op_name(world, matmul_op).unwrap_or_default();
    if name != "linalg.matmul" {
        return Err(AneLowerError::UnsupportedOp(format!(
            "expected 'linalg.matmul', got '{name}'"
        )));
    }

    // 2. Read operands
    let op_operands = operands(world, matmul_op);
    if op_operands.len() < 3 {
        return Err(AneLowerError::MissingOperand(format!(
            "matmul requires 3 operands (A, B, C), got {}",
            op_operands.len()
        )));
    }

    let a = op_operands[0];
    let b = op_operands[1];
    let c = op_operands[2];

    // 3. Extract tensor shapes
    let a_tensor = require_tensor(world, a, "operand A")?;
    let b_tensor = require_tensor(world, b, "operand B")?;
    let c_tensor = require_tensor(world, c, "operand C")?;

    // 4. Validate shapes: A[M,K] x B[K,N] = C[M,N]
    let shape_ok =
        a_tensor.shape.len() == 2 && b_tensor.shape.len() == 2 && c_tensor.shape.len() == 2;

    if !shape_ok {
        return Err(AneLowerError::MissingType(
            "all matmul operands must be 2-D tensors".into(),
        ));
    }

    let m = a_tensor.shape[0];
    let k_a = a_tensor.shape[1];
    let k_b = b_tensor.shape[0];
    let n = b_tensor.shape[1];
    let c_m = c_tensor.shape[0];
    let c_n = c_tensor.shape[1];

    if k_a != k_b {
        return Err(AneLowerError::MissingType(format!(
            "matmul dimension mismatch: A[1] = {k_a} ≠ B[0] = {k_b}"
        )));
    }
    if m != c_m || n != c_n {
        return Err(AneLowerError::MissingType(format!(
            "matmul result shape mismatch: expected [{m}, {n}], got [{c_m}, {c_n}]"
        )));
    }

    // 5. Emit MIL program
    let mil_type = element_type_to_mil(&a_tensor.element_type);
    Ok(emit_matmul_mil(m, n, k_a, mil_type))
}

/// Lower any supported root IR operation to ANE MIL.
///
/// Dispatches to the appropriate lowering function based on the operation name.
pub fn lower_to_ane(world: &World, root_op: Entity) -> Result<String, AneLowerError> {
    match op_name(world, root_op).as_deref() {
        Some("linalg.matmul") => lower_matmul_to_ane(world, root_op),
        Some(name) => Err(AneLowerError::UnsupportedOp(format!(
            "no ANE lowering available for '{name}'"
        ))),
        None => Err(AneLowerError::UnsupportedOp("operation has no name".into())),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use prism_ecs_core::{EntityKind, World};
    use prism_ecs_ir::ir_types::{FloatKind, TensorType, Type};
    use prism_ecs_ir::op::{OpMarker, OpName, Operands, Results};
    use prism_ecs_ir::value::{Uses, ValueType};

    // ── helpers ───────────────────────────────────────────────────────────

    fn create_matmul_op(world: &mut World, a_ty: Type, b_ty: Type, c_ty: Type) -> Entity {
        let a: Entity = world
            .spawn(EntityKind::Node, Some("A".into()))
            .unwrap()
            .into();
        world.add_component(a, ValueType(a_ty)).unwrap();
        world.add_component(a, Uses(vec![])).unwrap();

        let b: Entity = world
            .spawn(EntityKind::Node, Some("B".into()))
            .unwrap()
            .into();
        world.add_component(b, ValueType(b_ty)).unwrap();
        world.add_component(b, Uses(vec![])).unwrap();

        let c: Entity = world
            .spawn(EntityKind::Node, Some("C".into()))
            .unwrap()
            .into();
        world.add_component(c, ValueType(c_ty.clone())).unwrap();
        world.add_component(c, Uses(vec![])).unwrap();

        let result: Entity = world
            .spawn(EntityKind::Node, Some("result".into()))
            .unwrap()
            .into();
        world.add_component(result, ValueType(c_ty)).unwrap();
        world.add_component(result, Uses(vec![])).unwrap();

        let op: Entity = world
            .spawn(EntityKind::Node, Some("matmul".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("linalg.matmul".into()))
            .unwrap();
        world.add_component(op, Operands(vec![a, b, c])).unwrap();
        world.add_component(op, Results(vec![result])).unwrap();
        op
    }

    // ── lower_matmul_to_ane ───────────────────────────────────────────────

    #[test]
    fn lower_matmul_to_ane_produces_mil() {
        let mut world = World::new();

        let f16 = Type::float(FloatKind::F16);
        let a_ty = Type::Tensor(TensorType::new(vec![4, 8], f16.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![8, 16], f16.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![4, 16], f16));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let mil = lower_matmul_to_ane(&world, op).expect("ANE lowering failed");

        assert!(
            mil.contains("MIL PROGRAM"),
            "expected 'MIL PROGRAM' in output, got:\n{mil}"
        );
        assert!(
            mil.contains("matmul"),
            "expected 'matmul' in output, got:\n{mil}"
        );
        assert!(
            mil.contains("M: 4"),
            "expected 'M: 4' in output, got:\n{mil}"
        );
        assert!(
            mil.contains("K: 8"),
            "expected 'K: 8' in output, got:\n{mil}"
        );
        assert!(
            mil.contains("N: 16"),
            "expected 'N: 16' in output, got:\n{mil}"
        );
        assert!(
            mil.contains("float16"),
            "expected 'float16' type in output, got:\n{mil}"
        );
    }

    #[test]
    fn lower_matmul_to_ane_rejects_non_matmul() {
        let mut world = World::new();
        let op: Entity = world
            .spawn(EntityKind::Node, Some("not_matmul".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("linalg.fill".into()))
            .unwrap();

        let err = lower_matmul_to_ane(&world, op).expect_err("should have failed");
        match err {
            AneLowerError::UnsupportedOp(msg) => {
                assert!(msg.contains("linalg.fill"));
            }
            other => panic!("expected UnsupportedOp, got {other:?}"),
        }
    }

    #[test]
    fn lower_matmul_to_ane_rejects_missing_operands() {
        let mut world = World::new();
        let op: Entity = world
            .spawn(EntityKind::Node, Some("matmul".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("linalg.matmul".into()))
            .unwrap();
        world.add_component(op, Operands(vec![])).unwrap();

        let err = lower_matmul_to_ane(&world, op).expect_err("should have failed");
        match err {
            AneLowerError::MissingOperand(_) => {}
            other => panic!("expected MissingOperand, got {other:?}"),
        }
    }

    #[test]
    fn lower_matmul_to_ane_rejects_dimension_mismatch() {
        let mut world = World::new();

        let f16 = Type::float(FloatKind::F16);
        // A[4,8] x B[16,16] — K doesn't match
        let a_ty = Type::Tensor(TensorType::new(vec![4, 8], f16.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![16, 16], f16.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![4, 16], f16));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let err = lower_matmul_to_ane(&world, op).expect_err("should fail");
        match err {
            AneLowerError::MissingType(msg) => {
                assert!(msg.contains("dimension mismatch"));
            }
            other => panic!("expected MissingType, got {other:?}"),
        }
    }

    // ── lower_to_ane ──────────────────────────────────────────────────────

    #[test]
    fn lower_to_ane_dispatches_matmul() {
        let mut world = World::new();

        let f16 = Type::float(FloatKind::F16);
        let a_ty = Type::Tensor(TensorType::new(vec![2, 3], f16.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![3, 4], f16.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![2, 4], f16));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let mil = lower_to_ane(&world, op).expect("ANE lowering failed");

        assert!(
            mil.contains("MIL PROGRAM"),
            "expected 'MIL PROGRAM' in output, got:\n{mil}"
        );
        assert!(
            mil.contains("matmul_2x3x4"),
            "expected matmul_2x3x4, got:\n{mil}"
        );
    }

    #[test]
    fn lower_to_ane_rejects_unknown_op() {
        let mut world = World::new();
        let op: Entity = world
            .spawn(EntityKind::Node, Some("bogus".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("unknown.dance".into()))
            .unwrap();

        let err = lower_to_ane(&world, op).expect_err("should fail");
        match err {
            AneLowerError::UnsupportedOp(msg) => {
                assert!(msg.contains("unknown.dance"));
            }
            other => panic!("expected UnsupportedOp, got {other:?}"),
        }
    }
}
