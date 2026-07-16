//! AMDGCN assembly generation for ECS-native IR operations.
//!
//! Lowers high-level IR operations (e.g., `linalg.matmul`) into AMDGCN
//! assembly source suitable for compilation by the ROCm toolchain targeting
//! RDNA 3 (gfx1100).
//!
//! The generated kernels use a flat 1-D thread dispatch with innermost
//! K-dimension loop and device-global pointers for the matmul operands.

use std::fmt;

use prism_ecs_core::{Entity, World};

use crate::evolution::{resolve_matmul_tile, CompilePlanMarker, CompilePlanRef, TileSizes};
use crate::ir_types::{FloatKind, TensorType, Type};
use crate::op::{op_name, operands};
use crate::value::ValueType;

/// Error type for AMDGCN lowering failures.
#[derive(Debug)]
pub enum AmdgpuLowerError {
    /// The operation is not one that can be lowered to AMDGCN.
    UnsupportedOp(String),
    /// A required operand or result is missing.
    MissingOperand(String),
    /// An operand or result is missing a type annotation.
    MissingType(String),
}

impl fmt::Display for AmdgpuLowerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AmdgpuLowerError::UnsupportedOp(msg) => write!(f, "unsupported op: {}", msg),
            AmdgpuLowerError::MissingOperand(msg) => write!(f, "missing operand: {}", msg),
            AmdgpuLowerError::MissingType(msg) => write!(f, "missing type: {}", msg),
        }
    }
}

impl std::error::Error for AmdgpuLowerError {}

// ── Element type mapping ─────────────────────────────────────────────────────

/// Map an IR element type to its AMDGCN scalar type abbreviation.
fn element_type_to_amdgpu(ty: &Type) -> &'static str {
    match ty {
        Type::Float(ft) => match ft.kind {
            FloatKind::F16 => "f16",
            FloatKind::F32 => "f32",
            FloatKind::F64 => "f64",
            FloatKind::BF16 => "bf16",
            FloatKind::F8E4M3 | FloatKind::F8E5M2 => "f32", // promote to f32
        },
        Type::Integer(int_ty) if int_ty.width <= 8 => "i8",
        Type::Integer(int_ty) if int_ty.width <= 16 => "i16",
        Type::Integer(int_ty) if int_ty.width <= 32 => "i32",
        Type::Integer(_) => "i64",
        _ => "f32",
    }
}

/// Extract a `TensorType` from a value entity's `ValueType` component.
fn require_tensor(
    world: &World,
    entity: Entity,
    label: &str,
) -> Result<TensorType, AmdgpuLowerError> {
    let vt = world
        .get_component::<ValueType>(entity)
        .map(|v| v.0.clone())
        .ok_or_else(|| AmdgpuLowerError::MissingType(format!("{} has no ValueType", label)))?;

    match vt {
        Type::Tensor(t) => Ok(t),
        _ => Err(AmdgpuLowerError::MissingType(format!(
            "{} is not a tensor type",
            label
        ))),
    }
}

// ── Kernel emission ──────────────────────────────────────────────────────────

/// Emit an AMDGCN assembly kernel for a 2-D matrix multiply `C += A @ B`.
///
/// Each thread computes one output element of C through a 1-D dispatch
/// that linearly walks the K dimension. The kernel accepts three device-
/// global pointers plus dimension constants passed in scalar registers.
///
/// The kernel uses the following register convention (caller-specified
/// via s-bank):
///   - s[0:1] — A matrix base pointer
///   - s[2:3] — B matrix base pointer
///   - s[4:5] — C matrix base pointer
///
/// Dimensions are baked into the assembly as immediate literals.
#[rustfmt::skip]
fn emit_matmul_kernel(
    m: u64, n: u64, k: u64,
    tile_m: u64,
    tile_n: u64,
    tile_k: u64,
    amdgpu_type: &str,
) -> String {
    let has_tiles = tile_m != m || tile_n != n || tile_k != k;
    let label = if has_tiles {
        format!("matmul_{}x{}x{}_tile_{}x{}x{}", m, k, n, tile_m, tile_k, tile_n)
    } else {
        format!("matmul_{}x{}x{}", m, k, n)
    };
    // Pick the FMA mnemonic based on the element type.
    let fma_mnemonic = match amdgpu_type {
        "f16"  => "v_fma_f16",
        "f64"  => "v_fma_f64",
        "bf16" => "v_fma_bf16",
        _      => "v_fma_f32",        // f32, i8, i16, i32 → f32
    };

    format!(
        "\
.amdgpu_target gfx1100

.rodata

.text

.globl {label}
.align 256
.type {label},@function
{label}:
    ; Tile sizes: {tile_m}x{tile_k}x{tile_n} (MxKxN threadblock)
    ; Matrix multiply:  C[M,N] += A[M,K] @ B[K,N]
    ; Arguments:        s[0:1] = A_ptr, s[2:3] = B_ptr, s[4:5] = C_ptr
    ; Dimensions:       M={m}, K={k}, N={n}

    ; Load kernel arguments — device-global buffer addresses.
    s_load_dwordx4 s[0:3], s[0:1], 0x0
    s_load_dwordx2 s[4:5], s[0:1], 0x10
    s_waitcnt lgkmcnt(0)

    ; Allocate VGPRs.
    ;   v0  = row index (i)
    ;   v1  = column index (j)
    ;   v2  = accumulator
    ;   v3  = loop index (k)
    ;   v4  = A[i,k] load temporary
    ;   v5  = B[k,j] load temporary
    ;   v6  = address temporary

    v_mov_b32 v0, 0x0                 ; row i = 0 (single work-item per element)
    v_mov_b32 v1, 0x0                 ; col j = 0
    v_mov_b32 v2, 0x0                 ; accumulator = 0

    v_mov_b32 v3, 0x0                 ; k = 0

.Lloop_{label}:
    ; Address: A[i,k]  →  i * K + k
    v_mad_u32_u24 v4, v0, {k}, v3
    global_load_dword v4, v4, s[0:1]  ; v4 = A[i*K + k]
    s_waitcnt vmcnt(0)

    ; Address: B[k,j]  →  k * N + j
    v_mad_u32_u24 v5, v3, {n}, v1
    global_load_dword v5, v5, s[2:3]  ; v5 = B[k*N + j]
    s_waitcnt vmcnt(0)

    ; Accumulate: C[i,j] += A[i,k] * B[k,j]
    {fma} v2, v4, v5, v2

    ; k++
    v_add_u32 v3, v3, 1
    v_cmp_lt_u32 vcc, v3, {k}
    s_cbranch_vccnz .Lloop_{label}

    ; Store result C[i,j]
    ; Address:  i * N + j
    v_mad_u32_u24 v6, v0, {n}, v1
    global_store_dword v[6:7], v2, s[4:5]

    s_endpgm
",
        label = label,
        m = m,
        n = n,
        k = k,
        fma = fma_mnemonic,
    )
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Lower a `linalg.matmul` to AMDGCN assembly source.
///
/// Given a `linalg.matmul` op consuming operands `A`, `B`, `C` where the
/// semantics are `C += A @ B`, this function emits AMDGCN assembly source
/// implementing the matrix multiplication as a flat 1-D kernel — one
/// work-item per output element `C[i][j]`.
///
/// Each operand **must** carry a `tensor<...>` type with a 2-D shape so
/// that the dimensions `M`, `K`, `N` can be extracted.
pub fn lower_matmul_to_amdgpu(
    world: &World,
    matmul_op: Entity,
) -> Result<String, AmdgpuLowerError> {
    // 1. Verify the op is a matmul
    let name = op_name(world, matmul_op).unwrap_or_default();
    if name != "linalg.matmul" {
        return Err(AmdgpuLowerError::UnsupportedOp(format!(
            "expected 'linalg.matmul', got '{}'",
            name
        )));
    }

    // 2. Read operands
    let op_operands = operands(world, matmul_op);
    if op_operands.len() < 3 {
        return Err(AmdgpuLowerError::MissingOperand(format!(
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
        return Err(AmdgpuLowerError::MissingType(
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
        return Err(AmdgpuLowerError::MissingType(format!(
            "matmul dimension mismatch: A[1] = {k_a} ≠ B[0] = {k_b}"
        )));
    }
    if m != c_m || n != c_n {
        return Err(AmdgpuLowerError::MissingType(format!(
            "matmul result shape mismatch: expected [{m}, {n}], got [{c_m}, {c_n}]"
        )));
    }

    // 5. Emit kernel source
    let (tile_m, tile_n, tile_k) = resolve_matmul_tile(world, matmul_op, m, n, k_a);
    let amdgpu_type = element_type_to_amdgpu(&a_tensor.element_type);
    Ok(emit_matmul_kernel(
        m,
        n,
        k_a,
        tile_m,
        tile_n,
        tile_k,
        amdgpu_type,
    ))
}

/// Lower any supported root IR operation to AMDGCN source.
///
/// Dispatches to the appropriate lowering function based on the operation name.
pub fn lower_to_amdgpu(world: &World, root_op: Entity) -> Result<String, AmdgpuLowerError> {
    match op_name(world, root_op).as_deref() {
        Some("linalg.matmul") => lower_matmul_to_amdgpu(world, root_op),
        Some(name) => Err(AmdgpuLowerError::UnsupportedOp(format!(
            "no AMDGCN lowering available for '{name}'"
        ))),
        None => Err(AmdgpuLowerError::UnsupportedOp(
            "operation has no name".into(),
        )),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use prism_ecs_core::{EntityKind, World};

    use super::*;
    use crate::ir_types::{FloatKind, TensorType, Type};
    use crate::op::{OpAttributes, OpMarker, OpName, Operands, Results};
    use crate::value::{Uses, ValueDef, ValueType};

    // ── Helpers ──────────────────────────────────────────────────────────

    fn create_value(world: &mut World, label: &str, ty: Type) -> Entity {
        let entity: Entity = world
            .spawn(EntityKind::Node, Some(label.into()))
            .unwrap()
            .into();
        world
            .add_component(entity, ValueDef::op_result(Entity::new(0, 1), 0))
            .unwrap();
        world.add_component(entity, ValueType(ty)).unwrap();
        world.add_component(entity, Uses(vec![])).unwrap();
        entity
    }

    fn create_matmul_op(world: &mut World, a_ty: Type, b_ty: Type, c_ty: Type) -> Entity {
        let a = create_value(world, "A", a_ty.clone());
        let b = create_value(world, "B", b_ty.clone());
        let c = create_value(world, "C", c_ty.clone());
        let _result = create_value(world, "result", c_ty);

        let op: Entity = world
            .spawn(EntityKind::Node, Some("matmul".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("linalg.matmul".into()))
            .unwrap();
        world.add_component(op, Operands(vec![a, b, c])).unwrap();
        world.add_component(op, Results(vec![])).unwrap();
        world.add_component(op, OpAttributes(vec![])).unwrap();
        op
    }

    // ── Tests ────────────────────────────────────────────────────────────

    #[test]
    fn lower_matmul_f32_2x3x4() {
        let mut world = World::new();

        let f32 = Type::float(FloatKind::F32);
        let a_ty = Type::Tensor(TensorType::new(vec![2, 3], f32.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![3, 4], f32.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![2, 4], f32));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let source = lower_matmul_to_amdgpu(&world, op).expect("lowering failed");

        // Verify key AMDGCN structural elements
        assert!(
            source.contains(".amdgpu_target"),
            "missing '.amdgpu_target' in:\n{}",
            source
        );
        assert!(
            source.contains("v_fma_f32"),
            "missing 'v_fma_f32' in:\n{}",
            source
        );
        assert!(
            source.contains("global_store_dword"),
            "missing 'global_store_dword' in:\n{}",
            source
        );
        assert!(
            source.contains("matmul_2x3x4"),
            "missing 'matmul_2x3x4' in:\n{}",
            source
        );
        assert!(
            source.contains("s_endpgm"),
            "missing 's_endpgm' in:\n{}",
            source
        );

        // Verify dimensions appear in the source
        assert!(source.contains("M=2"), "missing 'M=2' in:\n{}", source);
        assert!(source.contains("K=3"), "missing 'K=3' in:\n{}", source);
        assert!(source.contains("N=4"), "missing 'N=4' in:\n{}", source);
    }

    #[test]
    fn lower_matmul_f16_1x2x2() {
        let mut world = World::new();

        let f16 = Type::float(FloatKind::F16);
        let a_ty = Type::Tensor(TensorType::new(vec![1, 2], f16.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![2, 2], f16.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![1, 2], f16));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let source = lower_matmul_to_amdgpu(&world, op).expect("lowering failed");

        // f16 should emit v_fma_f16
        assert!(
            source.contains("v_fma_f16"),
            "missing 'v_fma_f16' in:\n{}",
            source
        );
        assert!(
            source.contains(".amdgpu_target"),
            "missing '.amdgpu_target' in:\n{}",
            source
        );
    }

    #[test]
    fn lower_matmul_f64_2x2x2() {
        let mut world = World::new();

        let f64 = Type::float(FloatKind::F64);
        let a_ty = Type::Tensor(TensorType::new(vec![2, 2], f64.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![2, 2], f64.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![2, 2], f64));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let source = lower_matmul_to_amdgpu(&world, op).expect("lowering failed");

        assert!(source.contains("v_fma_f64"), "missing 'v_fma_f64' in:\n",);
        assert!(
            source.contains(".amdgpu_target"),
            "missing '.amdgpu_target'"
        );
    }

    #[test]
    fn lower_to_amdgpu_dispatches_matmul() {
        let mut world = World::new();

        let f32 = Type::float(FloatKind::F32);
        let a_ty = Type::Tensor(TensorType::new(vec![4, 8], f32.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![8, 16], f32.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![4, 16], f32));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let source = lower_to_amdgpu(&world, op).expect("dispatching failed");

        assert!(source.contains(".amdgpu_target"));
        assert!(source.contains("matmul_4x8x16"));
    }

    #[test]
    fn unsupported_op_returns_error() {
        let mut world = World::new();
        let op: Entity = world
            .spawn(EntityKind::Node, Some("unknown".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("arith.addf".into()))
            .unwrap();
        world.add_component(op, Operands(vec![])).unwrap();
        world.add_component(op, Results(vec![])).unwrap();
        world.add_component(op, OpAttributes(vec![])).unwrap();

        let err = lower_to_amdgpu(&world, op).unwrap_err();
        match err {
            AmdgpuLowerError::UnsupportedOp(_) => {} // expected
            _ => panic!("expected UnsupportedOp, got {:?}", err),
        }
    }

    #[test]
    fn missing_operand_returns_error() {
        let mut world = World::new();
        let op: Entity = world
            .spawn(EntityKind::Node, Some("matmul".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("linalg.matmul".into()))
            .unwrap();
        world.add_component(op, Operands(vec![])).unwrap(); // no operands
        world.add_component(op, Results(vec![])).unwrap();
        world.add_component(op, OpAttributes(vec![])).unwrap();

        let err = lower_matmul_to_amdgpu(&world, op).unwrap_err();
        match err {
            AmdgpuLowerError::MissingOperand(_) => {} // expected
            _ => panic!("expected MissingOperand, got {:?}", err),
        }
    }

    #[test]
    fn missing_type_returns_error() {
        let mut world = World::new();

        // Create a value with no ValueType component
        let a_val: Entity = world
            .spawn(EntityKind::Node, Some("A".into()))
            .unwrap()
            .into();

        let b_val: Entity = world
            .spawn(EntityKind::Node, Some("B".into()))
            .unwrap()
            .into();

        let c_val: Entity = world
            .spawn(EntityKind::Node, Some("C".into()))
            .unwrap()
            .into();

        let op: Entity = world
            .spawn(EntityKind::Node, Some("matmul".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("linalg.matmul".into()))
            .unwrap();
        world
            .add_component(op, Operands(vec![a_val, b_val, c_val]))
            .unwrap();
        world.add_component(op, Results(vec![])).unwrap();
        world.add_component(op, OpAttributes(vec![])).unwrap();

        let err = lower_matmul_to_amdgpu(&world, op).unwrap_err();
        match err {
            AmdgpuLowerError::MissingType(_) => {} // expected
            _ => panic!("expected MissingType, got {:?}", err),
        }
    }

    #[test]
    fn dimension_mismatch_returns_error() {
        let mut world = World::new();

        let f32 = Type::float(FloatKind::F32);
        // A[2,3] but B[4,5] — mismatch on K
        let a_ty = Type::Tensor(TensorType::new(vec![2, 3], f32.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![4, 5], f32.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![2, 5], f32));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let err = lower_matmul_to_amdgpu(&world, op).unwrap_err();
        match err {
            AmdgpuLowerError::MissingType(_) => {} // expected
            _ => panic!("expected MissingType for dim mismatch, got {:?}", err),
        }
    }

    #[test]
    fn not_a_matmul_returns_error() {
        let mut world = World::new();
        let op: Entity = world
            .spawn(EntityKind::Node, Some("fill".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("linalg.fill".into()))
            .unwrap();
        world.add_component(op, Operands(vec![])).unwrap();
        world.add_component(op, Results(vec![])).unwrap();
        world.add_component(op, OpAttributes(vec![])).unwrap();

        let err = lower_matmul_to_amdgpu(&world, op).unwrap_err();
        match err {
            AmdgpuLowerError::UnsupportedOp(_) => {} // expected
            _ => panic!("expected UnsupportedOp, got {:?}", err),
        }
    }

    #[test]
    fn error_display_and_error_trait() {
        let err = AmdgpuLowerError::UnsupportedOp("test op".into());
        let msg = format!("{}", err);
        assert!(msg.contains("unsupported op"));
        assert!(msg.contains("test op"));

        // Verify it implements std::error::Error
        let err_ref: &dyn std::error::Error = &err;
        assert!(err_ref.to_string().contains("unsupported op"));
    }

    #[test]
    fn error_display_missing_operand() {
        let err = AmdgpuLowerError::MissingOperand("missing A".into());
        assert_eq!(format!("{}", err), "missing operand: missing A");
    }

    #[test]
    fn error_display_missing_type() {
        let err = AmdgpuLowerError::MissingType("no type".into());
        assert_eq!(format!("{}", err), "missing type: no type");
    }

    #[test]
    fn bf16_emits_v_fma_bf16() {
        let mut world = World::new();

        let bf16 = Type::float(FloatKind::BF16);
        let a_ty = Type::Tensor(TensorType::new(vec![1, 1], bf16.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![1, 1], bf16.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![1, 1], bf16));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let source = lower_matmul_to_amdgpu(&world, op).expect("lowering failed");

        assert!(
            source.contains("v_fma_bf16"),
            "missing 'v_fma_bf16' in:\n{}",
            source
        );
        assert!(source.contains(".amdgpu_target"));
    }
}
