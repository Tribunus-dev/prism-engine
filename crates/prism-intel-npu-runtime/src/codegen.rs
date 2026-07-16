//! Intel NPU (Meteor Lake+) code generation for ECS-native IR operations.
//!
//! Lowers high-level IR operations (e.g., `linalg.matmul`) into Intel NPU
//! graph description pseudo-IR, matching Intel's NPU compiler interface /
//! Level Zero driver model.
//!
//! The Intel NPU uses a graph-of-tasks programming model rather than a
//! GPU-style grid dispatch. Each task is a named operation with typed
//! inputs and outputs running on a dedicated NPU execution unit.
//! The generated output is a textual graph description for the Intel NPU.

use prism_ecs_core::{Entity, World};

use prism_ecs_ir::ir_types::{FloatKind, FloatType, Signedness, TensorType, Type};
use prism_ecs_ir::op::{op_name, operands};
use prism_ecs_ir::value::ValueType;

/// Error type for Intel NPU lowering failures.
#[derive(Debug)]
pub enum IntelNpuLowerError {
    /// The operation is not one that can be lowered to Intel NPU.
    UnsupportedOp(String),
    /// A required operand or result is missing.
    MissingOperand(String),
    /// An operand or result is missing a type annotation.
    MissingType(String),
}

// ── Element type mapping ─────────────────────────────────────────────────────

/// Map an IR element type to its Intel NPU precision name.
fn element_type_to_intel_precision(ty: &Type) -> &'static str {
    match ty {
        Type::Float(FloatType { kind }) => match kind {
            FloatKind::F16 => "FP16",
            FloatKind::BF16 => "BF16",
            FloatKind::F32 => "FP32",
            FloatKind::F64 => "FP64",
            FloatKind::F8E4M3 | FloatKind::F8E5M2 => "FP8",
        },
        Type::Integer(int_ty) => match int_ty.signedness {
            Signedness::Signed => "INT",
            Signedness::Unsigned | Signedness::Signless => "UINT",
        },
        Type::Index
        | Type::NoneType
        | Type::Function(_)
        | Type::Tensor(_)
        | Type::Vector(_)
        | Type::Complex(_) => "FP16",
    }
}

/// Extract a `TensorType` from a value entity's `ValueType` component.
fn require_tensor(
    world: &World,
    entity: Entity,
    label: &str,
) -> Result<TensorType, IntelNpuLowerError> {
    let value_ty = world
        .get_component::<ValueType>(entity)
        .ok_or_else(|| IntelNpuLowerError::MissingType(format!("{label} is missing ValueType")))?;

    match &value_ty.0 {
        Type::Tensor(t) => Ok(t.clone()),
        other => Err(IntelNpuLowerError::MissingType(format!(
            "{label} has non-tensor type {other:?}"
        ))),
    }
}

// ── Emitters ─────────────────────────────────────────────────────────────────

/// Emit an Intel NPU graph description for a matmul operation.
#[rustfmt::skip]
fn emit_intel_npu_graph(m: u64, n: u64, k: u64, precision: &str) -> String {
    format!(
        r#"// Intel NPU Graph
// matmul {m}x{k}x{n} ({precision}) on NPU 3720
TASK @0: matmul
  INPUTS: A[{m},{k}], B[{k},{n}]
  OUTPUTS: C[{m},{n}]
  PRECISION: {precision}
  EXECUTION_UNIT: NPU
"#,
        m = m, n = n, k = k, precision = precision,
    )
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Lower a `linalg.matmul` to an Intel NPU graph description.
///
/// Given a `linalg.matmul` op consuming operands `A`, `B`, `C` where the
/// semantics are `C += A @ B`, this function emits a textual Intel NPU
/// graph pseudo-IR program.
///
/// Each operand **must** carry a `tensor<...>` type with a 2-D shape so that
/// the dimensions `M`, `K`, `N` can be extracted.
pub fn lower_matmul_to_intel_npu(
    world: &World,
    matmul_op: Entity,
) -> Result<String, IntelNpuLowerError> {
    // 1. Verify the op is a matmul
    let name = op_name(world, matmul_op).unwrap_or_default();
    if name != "linalg.matmul" {
        return Err(IntelNpuLowerError::UnsupportedOp(format!(
            "expected 'linalg.matmul', got '{name}'"
        )));
    }

    // 2. Read operands
    let op_operands = operands(world, matmul_op);
    if op_operands.len() < 3 {
        return Err(IntelNpuLowerError::MissingOperand(format!(
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
        return Err(IntelNpuLowerError::MissingType(
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
        return Err(IntelNpuLowerError::MissingType(format!(
            "matmul dimension mismatch: A[1] = {k_a} != B[0] = {k_b}"
        )));
    }
    if m != c_m || n != c_n {
        return Err(IntelNpuLowerError::MissingType(format!(
            "matmul result shape mismatch: expected [{m}, {n}], got [{c_m}, {c_n}]"
        )));
    }

    // 5. Emit Intel NPU graph
    let precision = element_type_to_intel_precision(&a_tensor.element_type);
    Ok(emit_intel_npu_graph(m, n, k_a, precision))
}

/// Lower any supported root IR operation to Intel NPU graph.
///
/// Dispatches to the appropriate lowering function based on the operation name.
pub fn lower_to_intel_npu(world: &World, root_op: Entity) -> Result<String, IntelNpuLowerError> {
    match op_name(world, root_op).as_deref() {
        Some("linalg.matmul") => lower_matmul_to_intel_npu(world, root_op),
        Some(name) => Err(IntelNpuLowerError::UnsupportedOp(format!(
            "no Intel NPU lowering available for '{name}'"
        ))),
        None => Err(IntelNpuLowerError::UnsupportedOp(
            "operation has no name".into(),
        )),
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
        world
            .add_component(result, ValueType(c_ty.clone()))
            .unwrap();
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

    // ── lower_matmul_to_intel_npu ─────────────────────────────────────────

    #[test]
    fn lower_matmul_to_intel_npu_produces_graph() {
        let mut world = World::new();

        let f16 = Type::float(FloatKind::F16);
        let a_ty = Type::Tensor(TensorType::new(vec![4, 8], f16.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![8, 16], f16.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![4, 16], f16));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let graph = lower_matmul_to_intel_npu(&world, op).expect("Intel NPU lowering failed");

        assert!(
            graph.contains("Intel NPU"),
            "expected 'Intel NPU' in output, got:\n{graph}"
        );
        assert!(
            graph.contains("matmul"),
            "expected 'matmul' in output, got:\n{graph}"
        );
        assert!(
            graph.contains("PRECISION"),
            "expected 'PRECISION' in output, got:\n{graph}"
        );
        assert!(
            graph.contains("FP16"),
            "expected 'FP16' in output, got:\n{graph}"
        );
        assert!(
            graph.contains("TASK @0: matmul"),
            "expected 'TASK @0: matmul' in output, got:\n{graph}"
        );
        assert!(
            graph.contains("A[4,8]"),
            "expected 'A[4,8]' in output, got:\n{graph}"
        );
        assert!(
            graph.contains("B[8,16]"),
            "expected 'B[8,16]' in output, got:\n{graph}"
        );
        assert!(
            graph.contains("C[4,16]"),
            "expected 'C[4,16]' in output, got:\n{graph}"
        );
        assert!(
            graph.contains("EXECUTION_UNIT: NPU"),
            "expected 'EXECUTION_UNIT: NPU' in output, got:\n{graph}"
        );
    }

    #[test]
    fn lower_matmul_to_intel_npu_rejects_non_matmul() {
        let mut world = World::new();
        let op: Entity = world
            .spawn(EntityKind::Node, Some("not_matmul".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("linalg.fill".into()))
            .unwrap();

        let err = lower_matmul_to_intel_npu(&world, op).expect_err("should have failed");
        match err {
            IntelNpuLowerError::UnsupportedOp(msg) => {
                assert!(msg.contains("linalg.fill"));
            }
            other => panic!("expected UnsupportedOp, got {other:?}"),
        }
    }

    #[test]
    fn lower_matmul_to_intel_npu_rejects_missing_operands() {
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

        let err = lower_matmul_to_intel_npu(&world, op).expect_err("should have failed");
        match err {
            IntelNpuLowerError::MissingOperand(_) => {}
            other => panic!("expected MissingOperand, got {other:?}"),
        }
    }

    #[test]
    fn lower_matmul_to_intel_npu_rejects_dimension_mismatch() {
        let mut world = World::new();

        let f16 = Type::float(FloatKind::F16);
        // A[4,8] x B[16,16] -- K doesn't match
        let a_ty = Type::Tensor(TensorType::new(vec![4, 8], f16.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![16, 16], f16.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![4, 16], f16));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let err = lower_matmul_to_intel_npu(&world, op).expect_err("should fail");
        match err {
            IntelNpuLowerError::MissingType(msg) => {
                assert!(msg.contains("dimension mismatch"));
            }
            other => panic!("expected MissingType, got {other:?}"),
        }
    }

    // ── lower_to_intel_npu ────────────────────────────────────────────────

    #[test]
    fn lower_to_intel_npu_dispatches_matmul() {
        let mut world = World::new();

        let f16 = Type::float(FloatKind::F16);
        let a_ty = Type::Tensor(TensorType::new(vec![2, 3], f16.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![3, 4], f16.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![2, 4], f16));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let graph = lower_to_intel_npu(&world, op).expect("Intel NPU lowering failed");

        assert!(
            graph.contains("Intel NPU"),
            "expected 'Intel NPU' in output, got:\n{graph}"
        );
        assert!(
            graph.contains("matmul"),
            "expected 'matmul' in output, got:\n{graph}"
        );
        assert!(
            graph.contains("PRECISION"),
            "expected 'PRECISION' in output, got:\n{graph}"
        );
    }

    #[test]
    fn lower_to_intel_npu_rejects_unknown_op() {
        let mut world = World::new();
        let op: Entity = world
            .spawn(EntityKind::Node, Some("bogus".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("unknown.dance".into()))
            .unwrap();

        let err = lower_to_intel_npu(&world, op).expect_err("should fail");
        match err {
            IntelNpuLowerError::UnsupportedOp(msg) => {
                assert!(msg.contains("unknown.dance"));
            }
            other => panic!("expected UnsupportedOp, got {other:?}"),
        }
    }
}
