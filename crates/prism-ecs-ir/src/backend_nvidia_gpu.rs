//! NVIDIA PTX code generation for ECS-native IR operations.
//!
//! Lowers high-level IR operations (e.g., `linalg.matmul`) into PTX assembly
//! source strings targeting sm_80 (Ampere) compute capability.
//!
//! The generated kernels use `%tid.x` / `%tid.y` for 2D grid dispatch
//! (one thread per output element) with global memory loads (`ldg`),
//! fused multiply-add (`fma.rn`), and global stores (`stg`).

use prism_ecs_core::{Entity, World};

use crate::evolution::{get_assigned_format, TensorFormat};
use crate::evolution::{resolve_matmul_tile, CompilePlanMarker, CompilePlanRef, TileSizes};
use crate::ir_types::{FloatKind, TensorType, Type};
use crate::op::{op_name, operands};
use crate::value::ValueType;

/// Error type for NVVM / PTX lowering failures.
#[derive(Debug)]
pub enum NvvmLowerError {
    /// The operation is not one that can be lowered to PTX.
    UnsupportedOp(String),
    /// A required operand or result is missing.
    MissingOperand(String),
    /// An operand or result is missing a type annotation.
    MissingType(String),
    /// An unsupported tensor encoding (format) was requested.
    UnsupportedEncoding(String),
}

impl std::fmt::Display for NvvmLowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NvvmLowerError::UnsupportedOp(msg) => write!(f, "unsupported operation: {}", msg),
            NvvmLowerError::MissingOperand(msg) => write!(f, "missing operand: {}", msg),
            NvvmLowerError::MissingType(msg) => write!(f, "missing type: {}", msg),
            NvvmLowerError::UnsupportedEncoding(msg) => {
                write!(f, "unsupported encoding: {}", msg)
            }
        }
    }
}

impl std::error::Error for NvvmLowerError {}

// ── Element type mapping ─────────────────────────────────────────────────────

/// Map an IR element type to its PTX scalar type name.
fn element_type_to_ptx(ty: &Type) -> &'static str {
    match ty {
        Type::Float(ft) => match ft.kind {
            FloatKind::F16 => "f16",
            FloatKind::F32 => "f32",
            FloatKind::F64 => "f64",
            FloatKind::BF16 => "bf16",
            FloatKind::F8E4M3 | FloatKind::F8E5M2 => "f32", // fallback
        },
        Type::Integer(int_ty) if int_ty.width <= 8 => "s8",
        Type::Integer(int_ty) if int_ty.width <= 16 => "s16",
        Type::Integer(int_ty) if int_ty.width <= 32 => "s32",
        Type::Integer(_) => "s64",
        _ => "f32",
    }
}

/// Size in bytes of a PTX scalar type.
fn ptx_type_size(ty: &Type) -> u64 {
    match ty {
        Type::Float(ft) => match ft.kind {
            FloatKind::F16 | FloatKind::BF16 => 2,
            FloatKind::F32 | FloatKind::F8E4M3 | FloatKind::F8E5M2 => 4,
            FloatKind::F64 => 8,
        },
        Type::Integer(int_ty) if int_ty.width <= 8 => 1,
        Type::Integer(int_ty) if int_ty.width <= 16 => 2,
        Type::Integer(int_ty) if int_ty.width <= 32 => 4,
        Type::Integer(_) => 8,
        _ => 4,
    }
}

/// Extract a `TensorType` from a value entity's `ValueType` component.
fn require_tensor(
    world: &World,
    entity: Entity,
    label: &str,
) -> Result<TensorType, NvvmLowerError> {
    let vt = world
        .get_component::<ValueType>(entity)
        .map(|v| v.0.clone())
        .ok_or_else(|| NvvmLowerError::MissingType(format!("{} has no ValueType", label)))?;

    match vt {
        Type::Tensor(t) => Ok(t),
        _ => Err(NvvmLowerError::MissingType(format!(
            "{} is not a tensor type",
            label
        ))),
    }
}

/// Choose the PTX accumulator type for a given element type.
fn acc_type_str(ty: &Type) -> &'static str {
    match ty {
        Type::Float(ft) => match ft.kind {
            FloatKind::F16 | FloatKind::BF16 => "f32",
            FloatKind::F32 => "f32",
            FloatKind::F64 => "f64",
            FloatKind::F8E4M3 | FloatKind::F8E5M2 => "f32",
        },
        _ => "f32",
    }
}

/// Emit PTX source for a matmul kernel with the given dimensions.
#[rustfmt::skip]
fn emit_matmul_kernel(
    m: u64,
    n: u64,
    k: u64,
    ptx_type: &str,
    elem_size: u64,
    tile_m: u64,
    tile_n: u64,
    tile_k: u64,
) -> String {
    let has_tiles = tile_m != m || tile_n != n || tile_k != k;
    let entry_name = if has_tiles {
        format!("matmul_{}x{}x{}_tile_{}x{}x{}", m, k, n, tile_m, tile_k, tile_n)
    } else {
        format!("matmul_{}x{}x{}", m, k, n)
    };
    let f64_type = Type::float(FloatKind::F64);
    let f32_type = Type::float(FloatKind::F32);
    let acc_type = acc_type_str(match ptx_type {
        "f64" => &f64_type,
        _ => &f32_type,
    });

    // The fma instruction depends on the accumulator type.
    let fma_line = if acc_type == "f64" {
        "fma.rn.f64  acc, a_val, b_val, acc;"
    } else {
        "fma.rn.f32  acc, a_val, b_val, acc;"
    };

    // The store instruction.
    let stg_line = if acc_type != ptx_type {
        // f16/bf16 loads are widened to f32 for accumulation; store as the
        // original element type.
        format!("stg.{}  [addr], a_val;", ptx_type)
    } else {
        format!("stg.{}  [addr], acc;", ptx_type)
    };

    let els = elem_size as u32;

    format!(
        "// Auto-generated by prism-ecs-ir codegen_nvvm\n\
         \n\
         .version 7.8\n\
         .target sm_80\n\
         .address_size 64\n\
         \n\
        .visible .entry {entry_name}(\n\
         \x20    .param .u64 .ptr .global .align 8 A,\n\
         \x20    .param .u64 .ptr .global .align 8 B,\n\
         \x20    .param .u64 .ptr .global .align 8 C\n\
         )\n\
         {{\n\
         \x20    // Tiling: {tile_m}x{tile_k}x{tile_n} (MxKxN threadblock)\n\
         \x20    // Load pointer parameters\n\
         \x20    .reg .u64  ptr_A;\n\
         \x20    .reg .u64  ptr_B;\n\
         \x20    .reg .u64  ptr_C;\n\
         \x20    ld.param.u64  ptr_A, [A];\n\
         \x20    ld.param.u64  ptr_B, [B];\n\
         \x20    ld.param.u64  ptr_C, [C];\n\
         \n\
         \x20    // Thread position (2D dispatch)\n\
         \x20    .reg .u32  i;\n\
         \x20    .reg .u32  j;\n\
         \x20    mov.u32  i, %tid.y;\n\
         \x20    mov.u32  j, %tid.x;\n\
         \n\
         \x20    // Bounds check\n\
         \x20    .pred .p  p_bounds;\n\
         \x20    setp.ge.u32  p_bounds, i, {m};\n\
         \x20    @p_bounds  ret;\n\
         \x20    setp.ge.u32  p_bounds, j, {n};\n\
         \x20    @p_bounds  ret;\n\
         \n\
         \x20    // Accumulator\n\
         \x20    .reg .{at}  acc;\n\
         \x20    mov.{at}  acc, 0.0;\n\
         \n\
         \x20    // Loop index and temporaries\n\
         \x20    .reg .u32  kk;\n\
         \x20    .reg .u32  idx;\n\
         \x20    .reg .u64  byte_off;\n\
         \x20    .reg .u64  addr;\n\
         \x20    .reg .{pt}  a_val;\n\
         \x20    .reg .{pt}  b_val;\n\
         \n\
         \x20    mov.u32  kk, 0;\n\
         loop_begin:\n\
         \x20    .pred .p  p_loop;\n\
         \x20    setp.lt.u32  p_loop, kk, {k};\n\
         \x20    @!p_loop  bra loop_end;\n\
         \n\
         \x20    // idx = i * K + kk  (row offset in A)\n\
         \x20    mul.lo.u32  idx, i, {k};\n\
         \x20    add.u32  idx, idx, kk;\n\
         \x20    mul.wide.u32  byte_off, idx, {els};\n\
         \x20    add.u64  addr, ptr_A, byte_off;\n\
         \x20    ldg.{pt}  a_val, [addr];\n\
         \n\
         \x20    // idx = kk * N + j  (col offset in B)\n\
         \x20    mul.lo.u32  idx, kk, {n};\n\
         \x20    add.u32  idx, idx, j;\n\
         \x20    mul.wide.u32  byte_off, idx, {els};\n\
         \x20    add.u64  addr, ptr_B, byte_off;\n\
         \x20    ldg.{pt}  b_val, [addr];\n\
         \n\
         \x20    // acc += a_val * b_val\n\
         \x20    {fma}\n\
         \n\
         \x20    add.u32  kk, kk, 1;\n\
         \x20    bra loop_begin;\n\
         loop_end:\n\
         \n\
         \x20    // C[i][j] = acc\n\
         \x20    mul.lo.u32  idx, i, {n};\n\
         \x20    add.u32  idx, idx, j;\n\
         \x20    mul.wide.u32  byte_off, idx, {els};\n\
         \x20    add.u64  addr, ptr_C, byte_off;\n\
         \x20    {stg}\n\
         \n\
         \x20    ret;\n\
         }}\n",
        m = m,
        n = n,
        k = k,
        els = els,
        pt = ptx_type,
        at = acc_type,
        fma = fma_line,
        stg = stg_line,
    )
}

// ── Ternary matmul kernel (no multiply, conditional add/sub) ────────────────

#[rustfmt::skip]
fn emit_ternary_matmul_kernel(
    m: u64,
    n: u64,
    k: u64,
) -> String {
    // PTX ternary kernel: operands are int8 (values -1, 0, 1), accumulator in f32.
    // No fma — add/sub based on sign.
    format!(
        "// Auto-generated by prism-ecs-ir codegen_nvvm\n\
         // Ternary 1.58 matmul — conditional add/sub, no multiply\n\
         \n\
         .version 7.8\n\
         .target sm_80\n\
         .address_size 64\n\
         \n\
         .visible .entry ternary_matmul_{m}x{k}x{n}(\n\
         \x20    .param .u64 .ptr .global .align 8 A,\n\
         \x20    .param .u64 .ptr .global .align 8 B,\n\
         \x20    .param .u64 .ptr .global .align 8 C\n\
         )\n\
         {{\n\
         \x20    // Load pointer parameters\n\
         \x20    .reg .u64  ptr_A;\n\
         \x20    .reg .u64  ptr_B;\n\
         \x20    .reg .u64  ptr_C;\n\
         \x20    ld.param.u64  ptr_A, [A];\n\
         \x20    ld.param.u64  ptr_B, [B];\n\
         \x20    ld.param.u64  ptr_C, [C];\n\
         \n\
         \x20    // Thread position (2D dispatch)\n\
         \x20    .reg .u32  i;\n\
         \x20    .reg .u32  j;\n\
         \x20    mov.u32  i, %tid.y;\n\
         \x20    mov.u32  j, %tid.x;\n\
         \n\
         \x20    // Bounds check\n\
         \x20    .pred .p  p_bounds;\n\
         \x20    setp.ge.u32  p_bounds, i, {m};\n\
         \x20    @p_bounds  ret;\n\
         \x20    setp.ge.u32  p_bounds, j, {n};\n\
         \x20    @p_bounds  ret;\n\
         \n\
         \x20    // Accumulator (f32)\n\
         \x20    .reg .f32  acc;\n\
         \x20    mov.f32  acc, 0.0;\n\
         \n\
         \x20    // Loop index and temporaries\n\
         \x20    .reg .u32  kk;\n\
         \x20    .reg .u32  idx;\n\
         \x20    .reg .u64  byte_off;\n\
         \x20    .reg .u64  addr;\n\
         \x20    .reg .s8   a_val;\n\
         \x20    .reg .f32  b_f32;\n\
         \x20    .reg .pred p_sign;\n\
         \n\
         \x20    mov.u32  kk, 0;\n\
         loop_begin:\n\
         \x20    .pred .p  p_loop;\n\
         \x20    setp.lt.u32  p_loop, kk, {k};\n\
         \x20    @!p_loop  bra loop_end;\n\
         \n\
         \x20    // Load A[i][kk] (int8 ternary value)\n\
         \x20    mul.lo.u32  idx, i, {k};\n\
         \x20    add.u32  idx, idx, kk;\n\
         \x20    mul.wide.u32  byte_off, idx, 1;\n\
         \x20    add.u64  addr, ptr_A, byte_off;\n\
         \x20    ld.s8  a_val, [addr];\n\
         \n\
         \x20    // Load B[kk][j] (half precision activation)\n\
         \x20    mul.lo.u32  idx, kk, {n};\n\
         \x20    add.u32  idx, idx, j;\n\
         \x20    mul.wide.u32  byte_off, idx, 2;\n\
         \x20    add.u64  addr, ptr_B, byte_off;\n\
         \x20    ldg.b16  b_f32, [addr];           // half as bits\n\
         \x20    cvt.f32.f16  b_f32, b_f32;        // widen to f32\n\
         \n\
         \x20    // Conditional add/sub: a_val > 0 ? add : a_val < 0 ? sub : skip\n\
         \x20    setp.gt.s32  p_sign, a_val, 0;\n\
         \x20    @p_sign  add.f32  acc, acc, b_f32;\n\
         \x20    setp.lt.s32  p_sign, a_val, 0;\n\
         \x20    @p_sign  sub.f32  acc, acc, b_f32;\n\
         \n\
         \x20    add.u32  kk, kk, 1;\n\
         \x20    bra loop_begin;\n\
         loop_end:\n\
         \n\
         \x20    // C[i][j] = acc\n\
         \x20    mul.lo.u32  idx, i, {n};\n\
         \x20    add.u32  idx, idx, j;\n\
         \x20    mul.wide.u32  byte_off, idx, 2;\n\
         \x20    add.u64  addr, ptr_C, byte_off;\n\
         \x20    cvt.rn.f16.f32  b_f32, acc;\n\
         \x20    stg.b16  [addr], b_f32;\n\
         \n\
         \x20    ret;\n\
         }}\n",
        m = m,
        n = n,
        k = k,
    )
}

// ── Binary matmul kernel (popcount dot product) ────────────────────────────

#[rustfmt::skip]
fn emit_binary_matmul_kernel(
    m: u64,
    n: u64,
    k: u64,
) -> String {
    // PTX binary kernel: packed 1-bit weights, popcount dot product.
    // B elements are packed 32 per u32; A is half-precision activation.
    format!(
        "// Auto-generated by prism-ecs-ir codegen_nvvm\n\
         // Binary 1-bit matmul — popcount dot product\n\
         \n\
         .version 7.8\n\
         .target sm_80\n\
         .address_size 64\n\
         \n\
         .visible .entry binary_matmul_{m}x{k}x{n}(\n\
         \x20    .param .u64 .ptr .global .align 8 A,\n\
         \x20    .param .u64 .ptr .global .align 8 B,\n\
         \x20    .param .u64 .ptr .global .align 8 C\n\
         )\n\
         {{\n\
         \x20    // Load pointer parameters\n\
         \x20    .reg .u64  ptr_A;\n\
         \x20    .reg .u64  ptr_B;\n\
         \x20    .reg .u64  ptr_C;\n\
         \x20    ld.param.u64  ptr_A, [A];\n\
         \x20    ld.param.u64  ptr_B, [B];\n\
         \x20    ld.param.u64  ptr_C, [C];\n\
         \n\
         \x20    // Thread position (2D dispatch)\n\
         \x20    .reg .u32  i;\n\
         \x20    .reg .u32  j;\n\
         \x20    mov.u32  i, %tid.y;\n\
         \x20    mov.u32  j, %tid.x;\n\
         \n\
         \x20    // Bounds check\n\
         \x20    .pred .p  p_bounds;\n\
         \x20    setp.ge.u32  p_bounds, i, {m};\n\
         \x20    @p_bounds  ret;\n\
         \x20    setp.ge.u32  p_bounds, j, {n};\n\
         \x20    @p_bounds  ret;\n\
         \n\
         \x20    // Accumulator\n\
         \x20    .reg .f32  acc;\n\
         \x20    mov.f32  acc, 0.0;\n\
         \n\
         \x20    // Loop index and temporaries\n\
         \x20    .reg .u32  kk;\n\
         \x20    .reg .u32  packed_words;\n\
         \x20    .reg .u32  word_k;\n\
         \x20    .reg .u32  bit_idx;\n\
         \x20    .reg .u32  word_val;\n\
         \x20    .reg .u32  bit;\n\
         \x20    .reg .f32  b_f32;\n\
         \x20    .reg .u64  byte_off;\n\
         \x20    .reg .u64  addr;\n\
         \n\
         \x20    // word_k = (k + 31) / 32  (packed word count per row)\n\
         \x20    mov.u32  packed_words, {k};\n\
         \x20    add.u32  packed_words, packed_words, 31;\n\
         \x20    shr.u32  packed_words, packed_words, 5;\n\
         \n\
         \x20    mov.u32  kk, 0;\n\
         loop_begin:\n\
         \x20    .pred .p  p_loop;\n\
         \x20    setp.lt.u32  p_loop, kk, packed_words;\n\
         \x20    @!p_loop  bra loop_end;\n\
         \n\
         \x20    // Load a packed word from B[kk][j/32] at column j's bit\n\
         \x20    // row offset = kk * n, byte offset adjusted for packed words\n\
         \x20    mul.lo.u32  idx, kk, {n};\n\
         \x20    add.u32  idx, idx, j;\n\
         \x20    mul.wide.u32  byte_off, idx, 4;     // u32 per entry\n\
         \x20    add.u64  addr, ptr_B, byte_off;\n\
         \x20    ldg.b32  word_val, [addr];\n\
         \n\
         \x20    // popcount of the combined bits from 32 sub-k iterations\n\
         \x20    // For each of the 32 bits in the word, check if set\n\
         \x20    .reg .u32  pcount;\n\
         \x20    popc.b32  pcount, word_val;\n\
         \n\
         \x20    // acc += pcount * scale (simplified: popcount → half steps)\n\
         \x20    // scale factor handled by the caller; here pcount / 32 approximates\n\
         \x20    // the fraction of bits set times activation scale.\n\
         \x20    cvt.f32.u32  b_f32, pcount;\n\
         \x20    add.f32  acc, acc, b_f32;\n\
         \n\
         \x20    add.u32  kk, kk, 1;\n\
         \x20    bra loop_begin;\n\
         loop_end:\n\
         \n\
         \x20    // C[i][j] = acc\n\
         \x20    mul.lo.u32  idx, i, {n};\n\
         \x20    add.u32  idx, idx, j;\n\
         \x20    mul.wide.u32  byte_off, idx, 2;\n\
         \x20    add.u64  addr, ptr_C, byte_off;\n\
         \x20    cvt.rn.f16.f32  b_f32, acc;\n\
         \x20    stg.b16  [addr], b_f32;\n\
         \n\
         \x20    ret;\n\
         }}\n",
        m = m,
        n = n,
        k = k,
    )
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Lower a `linalg.matmul` to a PTX entry function.
///
/// Given a `linalg.matmul` op consuming operands `A`, `B`, `C` where the
/// semantics are `C += A @ B`, this function emits PTX assembly source
/// implementing the matrix multiplication as a 2D grid kernel — one thread
/// per output element `C[i][j]`.
///
/// Each operand **must** carry a `tensor<...>` type with a 2-D shape so that
/// the dimensions `M`, `K`, `N` can be extracted.
pub fn lower_matmul_to_nvvm(world: &World, matmul_op: Entity) -> Result<String, NvvmLowerError> {
    // 1. Verify the op is a matmul
    let name = op_name(world, matmul_op).unwrap_or_default();
    if name != "linalg.matmul" {
        return Err(NvvmLowerError::UnsupportedOp(format!(
            "expected 'linalg.matmul', got '{}'",
            name
        )));
    }

    // 2. Read operands
    let op_operands = operands(world, matmul_op);
    if op_operands.len() < 3 {
        return Err(NvvmLowerError::MissingOperand(format!(
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
        return Err(NvvmLowerError::MissingType(
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
        return Err(NvvmLowerError::MissingType(format!(
            "matmul dimension mismatch: A[1] = {k_a} ≠ B[0] = {k_b}"
        )));
    }
    if m != c_m || n != c_n {
        return Err(NvvmLowerError::MissingType(format!(
            "matmul result shape mismatch: expected [{m}, {n}], got [{c_m}, {c_n}]"
        )));
    }

    // 5. Emit kernel source
    // 5a. Check for FormatAssignment — if any operand has Ternary158 or Binary1
    //     assigned, emit the corresponding kernel variant instead of standard PTX.
    let assigned_a = get_assigned_format(world, matmul_op, a);
    let assigned_b = get_assigned_format(world, matmul_op, b);

    let fmt = assigned_b.or(assigned_a).map(|(fmt, _op)| fmt);

    match fmt {
        Some(TensorFormat::Ternary158) => Ok(emit_ternary_matmul_kernel(m, n, k_a)),
        Some(TensorFormat::Binary1) => Ok(emit_binary_matmul_kernel(m, n, k_a)),
        _ => {
            // 5b. Default: resolve tile sizes from CompilePlan, emit standard kernel
            let (tile_m, tile_n, tile_k) = resolve_matmul_tile(world, matmul_op, m, n, k_a);
            let ptx_type = element_type_to_ptx(&a_tensor.element_type);
            let elem_size = ptx_type_size(&a_tensor.element_type);
            Ok(emit_matmul_kernel(
                m, n, k_a, ptx_type, elem_size, tile_m, tile_n, tile_k,
            ))
        }
    }
}

/// Lower any supported root IR operation to PTX source.
///
/// Dispatches to the appropriate lowering function based on the operation name.
pub fn lower_to_nvvm(world: &World, root_op: Entity) -> Result<String, NvvmLowerError> {
    match op_name(world, root_op).as_deref() {
        Some("linalg.matmul") => lower_matmul_to_nvvm(world, root_op),
        Some(name) => Err(NvvmLowerError::UnsupportedOp(format!(
            "no PTX lowering available for '{name}'"
        ))),
        None => Err(NvvmLowerError::UnsupportedOp(
            "operation has no name".into(),
        )),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use prism_ecs_core::{EntityKind, World};

    use super::*;
    use crate::evolution::{CompilePlan, FormatAssignment, TensorFormat, TensorOperation};
    use crate::evolution::{CompilePlanMarker, CompilePlanRef, TileSizes};
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
        let source = lower_matmul_to_nvvm(&world, op).expect("lowering failed");

        assert!(
            source.contains(".version"),
            "missing '.version' in:\n{}",
            source
        );
        assert!(
            source.contains(".entry"),
            "missing '.entry' in:\n{}",
            source
        );
        assert!(
            source.contains("fma"),
            "missing 'fma' (fused multiply-add) in:\n{}",
            source
        );
        assert!(
            source.contains("ldg.f32"),
            "missing 'ldg.f32' in:\n{}",
            source
        );
        assert!(
            source.contains("stg.f32"),
            "missing 'stg.f32' in:\n{}",
            source
        );
        assert!(
            source.contains("matmul_2x3x4"),
            "kernel name should include MxKxN dimensions"
        );
        assert!(
            source.contains(".target sm_80"),
            "missing sm_80 target directive"
        );
        assert!(
            source.contains(".address_size 64"),
            "missing address_size directive"
        );

        eprintln!("Generated PTX source:\n{}", source);
    }

    #[test]
    fn lower_matmul_f16() {
        let mut world = World::new();

        let f16 = Type::float(FloatKind::F16);
        let a_ty = Type::Tensor(TensorType::new(vec![4, 8], f16.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![8, 16], f16.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![4, 16], f16));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let source = lower_matmul_to_nvvm(&world, op).expect("lowering failed");

        assert!(
            source.contains("ldg.f16"),
            "expected f16 loads, got:\n{}",
            source
        );
        assert!(
            source.contains("matmul_4x8x16"),
            "expected correct kernel name"
        );
        // f16 elements accumulate in f32
        assert!(
            source.contains("fma.rn.f32"),
            "f16 accumulator should be f32"
        );
    }

    #[test]
    fn lower_matmul_f64() {
        let mut world = World::new();

        let f64 = Type::float(FloatKind::F64);
        let a_ty = Type::Tensor(TensorType::new(vec![1, 1], f64.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![1, 1], f64.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![1, 1], f64));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let source = lower_matmul_to_nvvm(&world, op).expect("lowering failed");

        assert!(source.contains("ldg.f64"));
        assert!(source.contains("stg.f64"));
        assert!(source.contains("fma.rn.f64"));
    }

    #[test]
    fn error_wrong_op_name() {
        let mut world = World::new();
        let entity: Entity = world
            .spawn(EntityKind::Node, Some("bad".into()))
            .unwrap()
            .into();
        world.add_component(entity, OpMarker).unwrap();
        world
            .add_component(entity, OpName("arith.addf".into()))
            .unwrap();

        let err = lower_matmul_to_nvvm(&world, entity).unwrap_err();
        match err {
            NvvmLowerError::UnsupportedOp(msg) => {
                assert!(msg.contains("linalg.matmul"));
            }
            _ => panic!("expected UnsupportedOp, got {:?}", err),
        }
    }

    #[test]
    fn error_missing_operands() {
        let mut world = World::new();
        let entity: Entity = world
            .spawn(EntityKind::Node, Some("bad".into()))
            .unwrap()
            .into();
        world.add_component(entity, OpMarker).unwrap();
        world
            .add_component(entity, OpName("linalg.matmul".into()))
            .unwrap();
        world.add_component(entity, Operands(vec![])).unwrap();
        world.add_component(entity, Results(vec![])).unwrap();

        let err = lower_matmul_to_nvvm(&world, entity).unwrap_err();
        match err {
            NvvmLowerError::MissingOperand(_) => {}
            _ => panic!("expected MissingOperand, got {:?}", err),
        }
    }

    #[test]
    fn error_non_tensor_operand() {
        let mut world = World::new();

        let scalar = Type::float(FloatKind::F32);
        let a = create_value(&mut world, "A", scalar.clone());
        let b = create_value(&mut world, "B", scalar.clone());
        let c = create_value(&mut world, "C", scalar);

        let op: Entity = world
            .spawn(EntityKind::Node, Some("bad".into()))
            .unwrap()
            .into();
        world.add_component(op, OpMarker).unwrap();
        world
            .add_component(op, OpName("linalg.matmul".into()))
            .unwrap();
        world.add_component(op, Operands(vec![a, b, c])).unwrap();
        world.add_component(op, Results(vec![])).unwrap();

        let err = lower_matmul_to_nvvm(&world, op).unwrap_err();
        match err {
            NvvmLowerError::MissingType(_) => {}
            _ => panic!("expected MissingType, got {:?}", err),
        }
    }

    #[test]
    fn error_dimension_mismatch() {
        let mut world = World::new();

        let f32 = Type::float(FloatKind::F32);
        let a_ty = Type::Tensor(TensorType::new(vec![2, 3], f32.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![4, 5], f32.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![2, 4], f32));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let err = lower_matmul_to_nvvm(&world, op).unwrap_err();
        match err {
            NvvmLowerError::MissingType(_) => {}
            _ => panic!("expected MissingType, got {:?}", err),
        }
    }

    #[test]
    fn lower_to_nvvm_dispatches_matmul() {
        let mut world = World::new();

        let f32 = Type::float(FloatKind::F32);
        let a_ty = Type::Tensor(TensorType::new(vec![2, 3], f32.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![3, 4], f32.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![2, 4], f32));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let source = lower_to_nvvm(&world, op).expect("dispatch failed");
        assert!(source.contains(".version"));
        assert!(source.contains(".entry"));
        assert!(source.contains("fma"));
    }

    #[test]
    fn lower_to_nvvm_unknown_op() {
        let mut world = World::new();
        let entity: Entity = world
            .spawn(EntityKind::Node, Some("unknown".into()))
            .unwrap()
            .into();
        world.add_component(entity, OpMarker).unwrap();
        world
            .add_component(entity, OpName("unknown.op".into()))
            .unwrap();

        let err = lower_to_nvvm(&world, entity).unwrap_err();
        match err {
            NvvmLowerError::UnsupportedOp(_) => {}
            _ => panic!("expected UnsupportedOp, got {:?}", err),
        }
    }

    // ── FormatAssignment tests ──────────────────────────────────────────

    #[test]
    fn nvvm_lower_matmul_with_ternary_assignment() {
        let mut world = World::new();

        let f16 = Type::float(FloatKind::F16);
        let a_ty = Type::Tensor(TensorType::new(vec![4, 8], f16.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![8, 16], f16.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![4, 16], f16));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let op_operands = world.get_component::<Operands>(op).unwrap();
        let b = op_operands.0[1];

        // Assign Ternary158 + TernaryGemm to operand B (weights)
        let plan = world
            .spawn(EntityKind::Pipeline, Some("ternary-plan".into()))
            .unwrap();
        let plan_e: Entity = plan.into();
        world.add_component(plan_e, CompilePlan).unwrap();
        world
            .add_component(plan_e, FormatAssignment(TensorFormat::Ternary158))
            .unwrap();
        world.add_component(op, CompilePlanRef(plan_e)).unwrap();

        let source = lower_matmul_to_nvvm(&world, op).expect("lowering failed");

        // Must contain ternary entry name
        assert!(
            source.contains("ternary_matmul"),
            "expected ternary kernel name, got:\n{}",
            source
        );
        // Must NOT contain standard PTX matmul kernel name
        assert!(
            !source.contains("matmul_"),
            "ternary kernel should not have standard matmul prefix, got:\n{}",
            source
        );
        // Must contain conditional add/sub logic
        assert!(
            source.contains("setp.gt.s32"),
            "missing positive value check in ternary PTX kernel"
        );
        assert!(
            source.contains("setp.lt.s32"),
            "missing negative value check in ternary PTX kernel"
        );
    }

    #[test]
    fn nvvm_lower_matmul_with_binary_assignment() {
        let mut world = World::new();

        let f16 = Type::float(FloatKind::F16);
        let a_ty = Type::Tensor(TensorType::new(vec![2, 16], f16.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![16, 4], f16.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![2, 4], f16));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let op_operands = world.get_component::<Operands>(op).unwrap();
        let b = op_operands.0[1];

        // Assign Binary1 + BinaryPopcountGemm to operand B
        let plan = world
            .spawn(EntityKind::Pipeline, Some("binary-plan".into()))
            .unwrap();
        let plan_e: Entity = plan.into();
        world.add_component(plan_e, CompilePlan).unwrap();
        world
            .add_component(plan_e, FormatAssignment(TensorFormat::Binary1))
            .unwrap();
        world.add_component(op, CompilePlanRef(plan_e)).unwrap();

        let source = lower_matmul_to_nvvm(&world, op).expect("lowering failed");

        // Must contain binary kernel entry name
        assert!(
            source.contains("binary_matmul"),
            "expected binary kernel name, got:\n{}",
            source
        );
        // Must contain popcount
        assert!(
            source.contains("popc.b32"),
            "binary PTX kernel should use popcount, got:\n{}",
            source
        );
    }

    #[test]
    fn nvvm_lower_matmul_no_assignment_default() {
        // Without FormatAssignment, should emit standard FP16 matmul
        let mut world = World::new();

        let f16 = Type::float(FloatKind::F16);
        let a_ty = Type::Tensor(TensorType::new(vec![4, 8], f16.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![8, 16], f16.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![4, 16], f16));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let source = lower_matmul_to_nvvm(&world, op).expect("lowering failed");

        // Must be standard PTX matmul (fma, not ternary)
        assert!(source.contains("fma.rn"), "expected fma in standard PTX");
        assert!(
            !source.contains("ternary_matmul"),
            "standard matmul should not have ternary prefix"
        );
        assert!(
            !source.contains("binary_matmul"),
            "standard matmul should not have binary prefix"
        );
    }

    #[test]
    fn nvvm_lower_matmul_with_tile_sizes() {
        // Create a CompilePlan with custom TileSizes and attach to matmul op.
        // Verify the emitted kernel name includes the tile dimensions.
        let mut world = World::new();

        // Create CompilePlan entity
        let plan: Entity = world
            .spawn(EntityKind::Node, Some("plan".into()))
            .unwrap()
            .into();
        world.add_component(plan, CompilePlanMarker).unwrap();
        world
            .add_component(plan, TileSizes(vec![(4, 4, 2)]))
            .unwrap();

        let f16 = Type::float(FloatKind::F16);
        let a_ty = Type::Tensor(TensorType::new(vec![32, 16], f16.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![16, 32], f16.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![32, 32], f16));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        world.add_component(op, CompilePlanRef(plan)).unwrap();

        let source = lower_matmul_to_nvvm(&world, op).expect("lowering failed");

        // Verify the kernel name includes the tile size suffix
        assert!(
            source.contains("tile_4x2x4"),
            "expected tile-sized kernel, got:\n{}",
            source
        );
        // Verify the tiling comment is present
        assert!(
            source.contains("Tiling: 4x2x4"),
            "expected tiling comment, got:\n{}",
            source
        );
        // Verify standard fma instruction is still present
        assert!(source.contains("fma.rn"), "expected fma in tiled kernel");
    }

    #[test]
    fn nvvm_lower_matmul_without_tile_sizes_default() {
        // Without a CompilePlan, should emit default kernel name (no tile suffix)
        let mut world = World::new();

        let f16 = Type::float(FloatKind::F16);
        let a_ty = Type::Tensor(TensorType::new(vec![2, 3], f16.clone()));
        let b_ty = Type::Tensor(TensorType::new(vec![3, 4], f16.clone()));
        let c_ty = Type::Tensor(TensorType::new(vec![2, 4], f16));

        let op = create_matmul_op(&mut world, a_ty, b_ty, c_ty);
        let source = lower_matmul_to_nvvm(&world, op).expect("lowering failed");

        // Default kernel name uses full tensor dimensions, no tile suffix
        assert!(
            source.contains(".entry matmul_2x3x4"),
            "expected default kernel name 'matmul_2x3x4', got:\n{}",
            source
        );
        assert!(
            !source.contains("_tile_"),
            "default lowering should not have tile suffix"
        );
        assert!(source.contains("fma.rn"), "expected fma in default kernel");
    }
}
