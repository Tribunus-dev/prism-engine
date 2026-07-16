//! AMD XDNA NPU code generation for ECS-native IR operations.
//!
//! Lowers high-level IR operations (e.g., `linalg.matmul`) into XDNA graph
//! pseudo-code describing a graph-level program suitable for the AMD Ryzen AI
//! NPU (XDNA architecture with AIE2/AIE2P engines).
//!
//! The XDNA NPU uses a graph-of-nodes programming model where each node maps
//! to a hardware-scheduled AI Engine (AIE) tile operation. Unlike GPU-style
//! grid dispatch, the NPU compiler accepts a graph-level description and
//! handles tiling, routing, and resource allocation internally.

use prism_ecs_core::{Entity, World};

use prism_ecs_ir::ir_types::FloatType;
use prism_ecs_ir::ir_types::{FloatKind, TensorType, Type};
use prism_ecs_ir::op::{op_name, operands};
use prism_ecs_ir::value::ValueType;

/// Error type for AMD XDNA NPU lowering failures.
#[derive(Debug)]
pub enum AmdNpuLowerError {
    /// The operation is not one that can be lowered to XDNA NPU IR.
    UnsupportedOp(String),
    /// A required operand or result is missing.
    MissingOperand(String),
    /// An operand or result is missing a type annotation.
    MissingType(String),
}

// ── Element type mapping ─────────────────────────────────────────────────────

/// Map an IR element type to its XDNA data type name.
fn element_type_to_xdna(ty: &Type) -> &'static str {
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
fn require_tensor(
    world: &World,
    entity: Entity,
    label: &str,
) -> Result<TensorType, AmdNpuLowerError> {
    let value_ty = world
        .get_component::<ValueType>(entity)
        .ok_or_else(|| AmdNpuLowerError::MissingType(format!("{label} is missing ValueType")))?;

    match &value_ty.0 {
        Type::Tensor(t) => Ok(t.clone()),
        other => Err(AmdNpuLowerError::MissingType(format!(
            "{label} has non-tensor type {other:?}"
        ))),
    }
}

// ── Emitters ─────────────────────────────────────────────────────────────────

/// Emit an XDNA NPU graph description for a matmul operation.
#[rustfmt::skip]
fn emit_matmul_xdna(m: u64, n: u64, k: u64, xdna_type: &str) -> String {
    format!(
        r#"// AMD XDNA NPU Graph
// matmul {m}x{k}x{n} ({xdna_type})
NODE @0: matmul(A[M,K], B[K,N]) -> C[M,N]
  ENGINE: AIE2
  DATA_TYPE: {xdna_type}
  TILING: {{M: 64, K: 64, N: 64}}
"#,
        m = m, n = n, k = k, xdna_type = xdna_type,
    )
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Lower a `linalg.matmul` to an AMD XDNA NPU graph description.
///
/// Given a `linalg.matmul` op consuming operands `A`, `B`, `C` where the
/// semantics are `C += A @ B`, this function emits a textual graph-level
/// pseudo-code description targeting the AMD XDNA NPU architecture.
///
/// Each operand **must** carry a `tensor<...>` type with a 2-D shape so that
/// the dimensions `M`, `K`, `N` can be extracted.
pub fn lower_matmul_to_amd_npu(
    world: &World,
    matmul_op: Entity,
) -> Result<String, AmdNpuLowerError> {
    // 1. Verify the op is a matmul
    let name = op_name(world, matmul_op).unwrap_or_default();
    if name != "linalg.matmul" {
        return Err(AmdNpuLowerError::UnsupportedOp(format!(
            "expected 'linalg.matmul', got '{name}'"
        )));
    }

    // 2. Read operands
    let op_operands = operands(world, matmul_op);
    if op_operands.len() < 3 {
        return Err(AmdNpuLowerError::MissingOperand(format!(
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
        return Err(AmdNpuLowerError::MissingType(
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
        return Err(AmdNpuLowerError::MissingType(format!(
            "matmul dimension mismatch: A[1] = {k_a} ≠ B[0] = {k_b}"
        )));
    }
    if m != c_m || n != c_n {
        return Err(AmdNpuLowerError::MissingType(format!(
            "matmul result shape mismatch: expected [{m}, {n}], got [{c_m}, {c_n}]"
        )));
    }

    // 5. Emit XDNA graph description
    let xdna_type = element_type_to_xdna(&a_tensor.element_type);
    Ok(emit_matmul_xdna(m, n, k_a, xdna_type))
}

/// Lower any supported root IR operation to AMD XDNA NPU graph IR.
///
/// Dispatches to the appropriate lowering function based on the operation name.
pub fn lower_to_amd_npu(world: &World, root_op: Entity) -> Result<String, AmdNpuLowerError> {
    match op_name(world, root_op).as_deref() {
        Some("linalg.matmul") => lower_matmul_to_amd_npu(world, root_op),
        Some(name) => Err(AmdNpuLowerError::UnsupportedOp(format!(
            "no AMD XDNA NPU lowering available for '{name}'"
        ))),
        None => Err(AmdNpuLowerError::UnsupportedOp(
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

    // ── lower_matmul_to_amd_npu ───────────────────────────────────────────

    #[test]
    fn lower_matmul_produces_xdna_graph() {
        let mut world = World::new();

        let f16 = Type::float(FloatKind::F16);
        let a_ty = Type::Tensor(TensorType::new(vec![4, 8], f16.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![8, 16], f16.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![4, 16], f16));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let xdna = lower_matmul_to_amd_npu(&world, op).expect("AMD XDNA lowering failed");

        assert!(xdna.contains("XDNA"), "expected 'XDNA', got:\n{xdna}");
        assert!(xdna.contains("AIE2"), "expected 'AIE2', got:\n{xdna}");
        assert!(xdna.contains("matmul"), "expected 'matmul', got:\n{xdna}");
        assert!(
            xdna.contains("matmul 4x8x16"),
            "expected '4x8x16', got:\n{xdna}"
        );
        assert!(xdna.contains("float16"), "expected 'float16', got:\n{xdna}");
        assert!(xdna.contains("TILING"), "expected 'TILING', got:\n{xdna}");
        assert!(
            xdna.contains("NODE @0: matmul"),
            "expected 'NODE @0: matmul', got:\n{xdna}"
        );
    }

    #[test]
    fn lower_matmul_rejects_non_matmul() {
        let mut world = World::new();
        let op: Entity = world
            .spawn(EntityKind::Node, Some("not_matmul".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("linalg.fill".into()))
            .unwrap();

        let err = lower_matmul_to_amd_npu(&world, op).expect_err("should have failed");
        match err {
            AmdNpuLowerError::UnsupportedOp(msg) => {
                assert!(msg.contains("linalg.fill"));
            }
            other => panic!("expected UnsupportedOp, got {other:?}"),
        }
    }

    #[test]
    fn lower_matmul_rejects_missing_operands() {
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

        let err = lower_matmul_to_amd_npu(&world, op).expect_err("should have failed");
        match err {
            AmdNpuLowerError::MissingOperand(_) => {}
            other => panic!("expected MissingOperand, got {other:?}"),
        }
    }

    #[test]
    fn lower_matmul_rejects_dimension_mismatch() {
        let mut world = World::new();

        let f16 = Type::float(FloatKind::F16);
        // A[4,8] x B[16,16] — K doesn't match
        let a_ty = Type::Tensor(TensorType::new(vec![4, 8], f16.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![16, 16], f16.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![4, 16], f16));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let err = lower_matmul_to_amd_npu(&world, op).expect_err("should fail");
        match err {
            AmdNpuLowerError::MissingType(msg) => {
                assert!(msg.contains("dimension mismatch"));
            }
            other => panic!("expected MissingType, got {other:?}"),
        }
    }

    // ── lower_to_amd_npu ──────────────────────────────────────────────────

    #[test]
    fn lower_to_amd_npu_dispatches_matmul() {
        let mut world = World::new();

        let f16 = Type::float(FloatKind::F16);
        let a_ty = Type::Tensor(TensorType::new(vec![2, 3], f16.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![3, 4], f16.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![2, 4], f16));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let xdna = lower_to_amd_npu(&world, op).expect("AMD XDNA lowering failed");

        assert!(
            xdna.contains("XDNA"),
            "expected 'XDNA' in output, got:\n{xdna}"
        );
        assert!(
            xdna.contains("AIE2"),
            "expected 'AIE2' in output, got:\n{xdna}"
        );
    }

    #[test]
    fn lower_to_amd_npu_rejects_unknown_op() {
        let mut world = World::new();
        let op: Entity = world
            .spawn(EntityKind::Node, Some("bogus".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("unknown.dance".into()))
            .unwrap();

        let err = lower_to_amd_npu(&world, op).expect_err("should fail");
        match err {
            AmdNpuLowerError::UnsupportedOp(msg) => {
                assert!(msg.contains("unknown.dance"));
            }
            other => panic!("expected UnsupportedOp, got {other:?}"),
        }
    }
}
