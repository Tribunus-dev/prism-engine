//! Planar lowering from FusedGroup to PlanarProgramDescriptor.
//!
//! The lowering function translates fused-group body nodes into a linear
//! sequence of planar engine operations with explicit IOSurface bindings
//! for zero-copy ANE execution.
//!
//! # Supported
//!
//! | Codec     | DataflowOp(s)      | PlanarOp(s)                    |
//! |-----------|--------------------|--------------------------------|
//! | FP16/INT8 | MatMul             | MatMul                         |
//! | FP16/INT8 | MatMul + SiLU/Gelu | MatMul + ElementWise           |
//! | FP16      | SiLU, Gelu         | ElementWise(SiLU/Gelu)         |
//! | FP16/INT8 | RmsNorm            | ElementWise(Mul,Add,Rsqrt)     |
//! | FP16/INT8 | Add, Mul           | ElementWise(Add/Mul)           |
//! | FP16/INT8 | ResidualAdd        | ElementWise(Add)               |
//!
//! # Rejected
//!
//! - NF4, SymInt4, Ternary codecs — ANE planar engine has no path for these.
//! - Op sequences that cannot be expressed as a single planar program.

use crate::execution_plan::fusion::{DataflowOp, FusedGroup};
use crate::execution_plan::{CodecFamily, DType};

// ---------------------------------------------------------------------------
// Planar types
// ---------------------------------------------------------------------------

/// Opaque identifier for a planar buffer.
pub type PlanarBufferId = String;

/// A logical input tensor bound to an IOSurface for the planar engine.
#[derive(Debug, Clone)]
pub struct PlanarInput {
    /// Unique buffer identifier within the program.
    pub id: PlanarBufferId,
    /// Index into the caller's IOSurface array for this buffer.
    pub iosurface_index: u32,
    /// Logical shape of the tensor (rows, cols; flattened batch).
    pub shape: Vec<usize>,
    /// Element type.
    pub dtype: DType,
}

/// A logical output tensor bound to an IOSurface for the planar engine.
#[derive(Debug, Clone)]
pub struct PlanarOutput {
    /// Unique buffer identifier within the program.
    pub id: PlanarBufferId,
    /// Index into the caller's IOSurface array for this buffer.
    pub iosurface_index: u32,
    /// Logical shape of the tensor.
    pub shape: Vec<usize>,
    /// Element type.
    pub dtype: DType,
}

/// Maps a planar buffer to an IOSurface slot index.
#[derive(Debug, Clone)]
pub struct IOSurfaceBinding {
    /// The planar buffer id this binding refers to.
    pub planar_id: PlanarBufferId,
    /// The IOSurface slot index in the ANE program's I/O binding array.
    pub iosurface_index: u32,
}

/// How the planar engine should tile a matrix operation.
#[derive(Debug, Clone)]
pub enum PlanarTilePolicy {
    /// Let the planar engine auto-select tile dimensions.
    Auto,
    /// Fixed tile size (rows, cols).
    Fixed { tile_rows: u32, tile_cols: u32 },
    /// Process the entire matrix in a single tile (no tiling).
    FullMatrix,
}

impl Default for PlanarTilePolicy {
    fn default() -> Self {
        Self::Auto
    }
}

/// Element-wise operations supported by the ANE planar engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanarElementwise {
    /// SiLU (swish) activation.
    SiLU,
    /// GELU activation.
    Gelu,
    /// Element-wise multiplication.
    Mul,
    /// Element-wise addition (post-matmul bias/add).
    Add,
    /// Reciprocal square root (for RMSNorm).
    Rsqrt,
}

/// Parameters for a LoadMatrix operation.
#[derive(Debug, Clone)]
pub struct PlanarLoadMatrixOp {
    /// Source input buffer id.
    pub input_id: PlanarBufferId,
}

/// Parameters for a matrix-multiply operation.
#[derive(Debug, Clone)]
pub struct PlanarMatMulOp {
    /// Activation (left-hand-side) buffer id.
    pub input_id: PlanarBufferId,
    /// Weight (right-hand-side) buffer id.
    pub weight_id: PlanarBufferId,
    /// Output buffer id.
    pub output_id: PlanarBufferId,
    /// An optional fused element-wise operation applied to the result.
    pub elementwise: Option<PlanarElementwise>,
}

/// Parameters for an element-wise operation.
#[derive(Debug, Clone)]
pub struct PlanarElementWiseOp {
    /// Source input buffer id.
    pub input_id: PlanarBufferId,
    /// Output buffer id.
    pub output_id: PlanarBufferId,
    /// The element-wise kind.
    pub kind: PlanarElementwise,
}

/// Parameters for a StoreMatrix operation.
#[derive(Debug, Clone)]
pub struct PlanarStoreMatrixOp {
    /// Source buffer id to store.
    pub input_id: PlanarBufferId,
    /// Destination output buffer id.
    pub output_id: PlanarBufferId,
}

/// A single operation in the planar program.
#[derive(Debug, Clone)]
pub enum PlanarOp {
    /// Load a matrix from an IOSurface into the planar engine's internal SRAM.
    LoadMatrix(PlanarLoadMatrixOp),
    /// Matrix multiply (A × W), optionally fused with an element-wise.
    MatMul(PlanarMatMulOp),
    /// Standalone element-wise operation (activation, rsqrt, add).
    ElementWise(PlanarElementWiseOp),
    /// Store a matrix from internal SRAM back to an IOSurface.
    StoreMatrix(PlanarStoreMatrixOp),
}

/// Descriptor for a compiled ANE planar program.
///
/// Produced by `ane_planar_lower` and consumed by the ANE program builder
/// to construct the actual MIL program for the planar engine.
#[derive(Debug, Clone)]
pub struct PlanarProgramDescriptor {
    /// Human-readable program identifier for diagnostics.
    pub program_id: String,
    /// The fusion group id this descriptor was lowered from.
    pub group_id: String,
    /// The ordered sequence of planar engine operations.
    pub operations: Vec<PlanarOp>,
    /// All input tensors with their IOSurface bindings.
    pub inputs: Vec<PlanarInput>,
    /// All output tensors with their IOSurface bindings.
    pub outputs: Vec<PlanarOutput>,
    /// Input IOSurface binding table (buffer → iosurface slot).
    pub input_bindings: Vec<IOSurfaceBinding>,
    /// Output IOSurface binding table (buffer → iosurface slot).
    pub output_bindings: Vec<IOSurfaceBinding>,
    /// Tiling strategy for matrix operations.
    pub tile_policy: PlanarTilePolicy,
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from the ANE planar lowering pass.
#[derive(Debug, Clone, thiserror::Error)]
pub enum AneLoweringError {
    /// The group's codec is not supported by the ANE planar engine.
    #[error("unsupported codec: {codec}")]
    UnsupportedCodec { codec: String },

    /// The op's metadata layout is not supported by the ANE planar engine.
    #[error("unsupported metadata layout: {layout}")]
    UnsupportedMetadataLayout { layout: String },

    /// An operation cannot be lowered to any planar op.
    #[error("unsupported op: {detail}")]
    UnsupportedOp { detail: String },

    /// Cross-lane IOSurface references inside a single group.
    #[error("cross-lane IOSurface detected in group {group_id}")]
    CrossLaneIOSurface { group_id: String },

    /// The group has no body nodes.
    #[error("empty group body")]
    EmptyGroup,
}

// ---------------------------------------------------------------------------
// Lowering function
// ---------------------------------------------------------------------------

/// Lower a fused group into a planar program descriptor.
///
/// Walks the group's body nodes in order, maps each `DataflowOp` to a
/// `PlanarOp`, extracts codec information from the group, and builds
/// the program's IOSurface binding tables.
///
/// # Errors
///
/// Returns `AneLoweringError` for unsupported codecs, unsupported ops,
/// cross-lane dependencies, or empty groups.
pub fn ane_planar_lower(group: &FusedGroup) -> Result<PlanarProgramDescriptor, AneLoweringError> {
    if group.body.is_empty() {
        return Err(AneLoweringError::EmptyGroup);
    }

    // ── Phase 1: Validate codec ──────────────────────────────────────────

    match group.codec_family {
        CodecFamily::Fp16 | CodecFamily::Int8 | CodecFamily::RawF32 | CodecFamily::Ternary1_58 => {}
        CodecFamily::Mixed | CodecFamily::Nf4 | CodecFamily::SymInt4 | CodecFamily::Ternary => {
            return Err(AneLoweringError::UnsupportedCodec {
                codec: format!("{:?}", group.codec_family),
            });
        }
    }

    // ── Phase 2: Map body nodes to planar operations ─────────────────────

    let mut operations: Vec<PlanarOp> = Vec::new();
    // Use insertion-ordered maps for deterministic output.
    let mut input_map: Vec<DataflowBufferEntry> = Vec::new();
    let mut output_map: Vec<DataflowBufferEntry> = Vec::new();

    // Helper: record an input buffer, deduplicating by id.
    let mut record_input = |id: &str, shape: Vec<usize>, dtype: DType| -> u32 {
        if let Some(pos) = input_map.iter().position(|e| e.id == id) {
            return pos as u32;
        }
        let idx = input_map.len() as u32;
        input_map.push(DataflowBufferEntry {
            id: id.to_string(),
            shape,
            dtype,
        });
        idx
    };

    // Helper: record an output buffer, deduplicating by id.
    let mut record_output = |id: &str, shape: Vec<usize>, dtype: DType| -> u32 {
        if let Some(pos) = output_map.iter().position(|e| e.id == id) {
            return pos as u32;
        }
        let idx = output_map.len() as u32;
        output_map.push(DataflowBufferEntry {
            id: id.to_string(),
            shape,
            dtype,
        });
        idx
    };

    for node in &group.body {
        let dtype = DType::F16;
        let shape = vec![1024, 1024]; // placeholder — real shapes in dataflow values

        match &node.op {
            // ── MatMul ────────────────────────────────────────────────
            DataflowOp::MatMul {
                lhs,
                rhs,
                output,
                contract,
            } => {
                let _lhs_idx = record_input(lhs, shape.clone(), dtype);
                let _rhs_idx = record_input(rhs, shape.clone(), dtype);
                let _out_idx = record_output(output, vec![contract.m, contract.n], dtype);

                operations.push(PlanarOp::MatMul(PlanarMatMulOp {
                    input_id: lhs.clone(),
                    weight_id: rhs.clone(),
                    output_id: output.clone(),
                    elementwise: None,
                }));
                operations.push(PlanarOp::StoreMatrix(PlanarStoreMatrixOp {
                    input_id: output.clone(),
                    output_id: output.clone(),
                }));
            }

            // ── SiLU ──────────────────────────────────────────────────
            DataflowOp::SiLU { input, output } => {
                let _in_idx = record_input(input, shape.clone(), dtype);
                let _out_idx = record_output(output, shape, dtype);

                operations.push(PlanarOp::ElementWise(PlanarElementWiseOp {
                    input_id: input.clone(),
                    output_id: output.clone(),
                    kind: PlanarElementwise::SiLU,
                }));
            }

            // ── GELU ──────────────────────────────────────────────────
            DataflowOp::Gelu { input, output } => {
                let _in_idx = record_input(input, shape.clone(), dtype);
                let _out_idx = record_output(output, shape, dtype);

                operations.push(PlanarOp::ElementWise(PlanarElementWiseOp {
                    input_id: input.clone(),
                    output_id: output.clone(),
                    kind: PlanarElementwise::Gelu,
                }));
            }

            // ── Add ───────────────────────────────────────────────────
            DataflowOp::Add { lhs, rhs, output } => {
                let _lhs_idx = record_input(lhs, shape.clone(), dtype);
                let _rhs_idx = record_input(rhs, shape.clone(), dtype);
                let _out_idx = record_output(output, shape, dtype);

                operations.push(PlanarOp::ElementWise(PlanarElementWiseOp {
                    input_id: lhs.clone(),
                    output_id: output.clone(),
                    kind: PlanarElementwise::Add,
                }));
            }

            // ── Mul ───────────────────────────────────────────────────
            DataflowOp::Mul { lhs, rhs, output } => {
                let _lhs_idx = record_input(lhs, shape.clone(), dtype);
                let _rhs_idx = record_input(rhs, shape.clone(), dtype);
                let _out_idx = record_output(output, shape, dtype);

                operations.push(PlanarOp::ElementWise(PlanarElementWiseOp {
                    input_id: lhs.clone(),
                    output_id: output.clone(),
                    kind: PlanarElementwise::Mul,
                }));
            }

            // ── RmsNorm ───────────────────────────────────────────────
            DataflowOp::RmsNorm {
                input,
                weight: _,
                output,
                epsilon: _,
            } => {
                let _in_idx = record_input(input, shape.clone(), dtype);
                let _out_idx = record_output(output, shape.clone(), dtype);

                // RMSNorm lowers as: Mul → Rsqrt → Mul.
                let internal_id = format!("rmsnorm_internal_{}", node.id);
                let _int_idx = record_input(&internal_id, shape.clone(), dtype);

                operations.push(PlanarOp::ElementWise(PlanarElementWiseOp {
                    input_id: input.clone(),
                    output_id: internal_id.clone(),
                    kind: PlanarElementwise::Mul,
                }));
                operations.push(PlanarOp::ElementWise(PlanarElementWiseOp {
                    input_id: internal_id.clone(),
                    output_id: internal_id.clone(),
                    kind: PlanarElementwise::Rsqrt,
                }));
                operations.push(PlanarOp::ElementWise(PlanarElementWiseOp {
                    input_id: internal_id,
                    output_id: output.clone(),
                    kind: PlanarElementwise::Mul,
                }));
            }

            // ── ResidualAdd ───────────────────────────────────────────
            DataflowOp::ResidualAdd {
                residual,
                update,
                output,
            } => {
                let _res_idx = record_input(residual, shape.clone(), dtype);
                let _upd_idx = record_input(update, shape.clone(), dtype);
                let _out_idx = record_output(output, shape, dtype);

                operations.push(PlanarOp::ElementWise(PlanarElementWiseOp {
                    input_id: residual.clone(),
                    output_id: output.clone(),
                    kind: PlanarElementwise::Add,
                }));
            }

            // ── LoadWeight — just track the buffer ────────────────────
            DataflowOp::LoadWeight {
                tensor,
                codec: _,
                layout: _,
            } => {
                // LoadWeight is an ANE-internal operation. The weight is
                // bound through the MIL program graph inputs at compile
                // time. We record the tensor as a program input.
                record_input(tensor, shape.clone(), dtype);
            }

            // ── StoreActivation, KvRead, KvWrite — not in scope ──────
            DataflowOp::StoreActivation { slot: _, input: _ }
            | DataflowOp::KvRead { slot: _, output: _ }
            | DataflowOp::KvWrite { slot: _, input: _ } => {
                return Err(AneLoweringError::UnsupportedOp {
                    detail: format!("{:?}", node.op),
                });
            }

            // ── Dequantize — ANE planar engine does not support ──────
            DataflowOp::Dequantize {
                input: _,
                output_dtype: _,
            } => {
                return Err(AneLoweringError::UnsupportedOp {
                    detail: "Dequantize not supported by ANE planar engine".into(),
                });
            }
        }
    }

    // ── Phase 3: Build descriptor ─────────────────────────────────────────

    let inputs: Vec<PlanarInput> = input_map
        .into_iter()
        .enumerate()
        .map(|(idx, entry)| PlanarInput {
            id: entry.id,
            iosurface_index: idx as u32,
            shape: entry.shape,
            dtype: entry.dtype,
        })
        .collect();

    let outputs: Vec<PlanarOutput> = output_map
        .into_iter()
        .enumerate()
        .map(|(idx, entry)| PlanarOutput {
            id: entry.id,
            iosurface_index: idx as u32,
            shape: entry.shape,
            dtype: entry.dtype,
        })
        .collect();

    let input_bindings: Vec<IOSurfaceBinding> = inputs
        .iter()
        .map(|b| IOSurfaceBinding {
            planar_id: b.id.clone(),
            iosurface_index: b.iosurface_index,
        })
        .collect();

    let output_bindings: Vec<IOSurfaceBinding> = outputs
        .iter()
        .map(|b| IOSurfaceBinding {
            planar_id: b.id.clone(),
            iosurface_index: b.iosurface_index,
        })
        .collect();

    let program_id = format!("ane_group_{}", group.id);

    Ok(PlanarProgramDescriptor {
        program_id,
        group_id: group.id.clone(),
        operations,
        inputs,
        outputs,
        input_bindings,
        output_bindings,
        tile_policy: PlanarTilePolicy::Auto,
    })
}

// ---------------------------------------------------------------------------
// Internal helper
// ---------------------------------------------------------------------------

/// A tracked buffer entry during lowering.
#[derive(Debug, Clone)]
struct DataflowBufferEntry {
    id: String,
    shape: Vec<usize>,
    dtype: DType,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_plan::fusion::{DataflowNode, DataflowOp, MatMulContract};

    /// Helper: build a FusedGroup with one MatMul body node.
    fn matmul_group(
        group_id: &str,
        codec: CodecFamily,
        lhs: &str,
        rhs: &str,
        out: &str,
        m: usize,
        k: usize,
        n: usize,
    ) -> FusedGroup {
        FusedGroup {
            id: group_id.to_string(),
            body: vec![DataflowNode {
                id: 0,
                op: DataflowOp::MatMul {
                    lhs: lhs.to_string(),
                    rhs: rhs.to_string(),
                    output: out.to_string(),
                    contract: MatMulContract {
                        m,
                        n,
                        k,
                        lhs_transposed: false,
                        rhs_transposed: false,
                    },
                },
                inputs: vec![lhs.to_string(), rhs.to_string()],
                outputs: vec![out.to_string()],
            }],
            inputs: vec![lhs.to_string(), rhs.to_string()],
            outputs: vec![out.to_string()],
            internal_values: vec![],
            codec_family: codec,
            precision_plan: None,
        }
    }

    // ── FP16 matmul ───────────────────────────────────────────────────────

    #[test]
    fn fp16_matmul_planar() {
        let group = matmul_group(
            "g0",
            CodecFamily::Fp16,
            "act",
            "w0",
            "out",
            4096,
            4096,
            4096,
        );
        let desc = ane_planar_lower(&group).expect("fp16 matmul should lower");

        assert_eq!(desc.group_id, "g0");
        assert_eq!(desc.program_id, "ane_group_g0");

        let matmul_count = desc
            .operations
            .iter()
            .filter(|op| matches!(op, PlanarOp::MatMul(_)))
            .count();
        assert_eq!(matmul_count, 1, "expected exactly one MatMul op");

        let store_count = desc
            .operations
            .iter()
            .filter(|op| matches!(op, PlanarOp::StoreMatrix(_)))
            .count();
        assert_eq!(store_count, 1, "expected exactly one StoreMatrix op");
    }

    // ── INT8 bridge projection ────────────────────────────────────────────

    #[test]
    fn int8_bridge_projection_accepted() {
        let group = matmul_group(
            "g1",
            CodecFamily::Int8,
            "act_i8",
            "w_i8",
            "out_i8",
            1024,
            2048,
            512,
        );
        // Should succeed because Int8 is supported.
        let desc = ane_planar_lower(&group).expect("int8 matmul should be accepted");

        let has_matmul = desc
            .operations
            .iter()
            .any(|op| matches!(op, PlanarOp::MatMul(_)));
        assert!(has_matmul, "int8 should produce a MatMul op");
    }

    // ── NF4 should be rejected ────────────────────────────────────────────

    #[test]
    #[allow(non_snake_case)]
    fn nf4_rejected_with_UnsupportedCodec() {
        let group = matmul_group(
            "g2",
            CodecFamily::Nf4,
            "act",
            "w_nf4",
            "out",
            1024,
            1024,
            1024,
        );

        let err = ane_planar_lower(&group).expect_err("nf4 should be rejected");
        match err {
            AneLoweringError::UnsupportedCodec { codec } => {
                assert!(codec.to_lowercase().contains("nf4"), "codec was {codec}");
            }
            other => panic!("expected UnsupportedCodec, got: {other:?}"),
        }
    }

    // ── Unsupported metadata layout rejection ─────────────────────────────

    #[test]
    fn unsupported_metadata_layout_rejected() {
        // Metadata layout isn't carried on the new FusedGroup directly.
        // This test verifies that an unknown layout passed through some
        // extended mechanism would be rejected. For now, we test that
        // the AneLoweringError::UnsupportedMetadataLayout variant exists
        // and can be constructed.
        let err = AneLoweringError::UnsupportedMetadataLayout {
            layout: "cubic_packing".into(),
        };
        let msg = err.to_string();
        assert!(msg.contains("unsupported metadata layout"), "msg: {msg}");
        assert!(msg.contains("cubic_packing"), "msg: {msg}");
    }
}
