//! This module owns the canonical authority for the `UOp` data type — the
//! `UOpId` identifier, the `UOpKind` op-kind enum, and the `UOp` struct that
//! the phase graph uses to represent every executable node.
//! It does not own graph mutation, validation, or lowering.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd, Serialize, Deserialize)]
pub struct UOpId(pub u32);

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UOpKind {
    Input {
        name: String,
    },
    Const {
        value: f32,
    },
    Add,
    Mul,
    Sub,
    Div,
    Maximum,
    Minimum,
    /// Elementwise select: nonzero condition selects the true value.
    Where,
    Relu,
    Neg,
    Exp,
    Sqrt,
    Abs,
    Log,
    Tanh,
    Sin,
    Cos,
    Gelu,
    /// Elementwise power with a compile-time scalar exponent.
    Pow {
        exponent: f32,
    },
    /// Elementwise dtype conversion. Runtime buffers use f32 transport, so
    /// non-f32 targets are represented by their deterministic quantized f32
    /// value until a typed buffer ABI is selected.
    Cast {
        from: String,
        to: String,
    },
    /// Permute row-major tensor axes; permutation maps output axes to input axes.
    Transpose {
        permutation: Vec<usize>,
    },
    /// Metadata-only view change. Reshape preserves row-major storage and
    /// therefore does not require a kernel or a new allocation.
    Reshape,
    RmsNorm {
        rows: usize,
        features: usize,
        epsilon: f32,
    },
    LayerNorm {
        rows: usize,
        features: usize,
        epsilon: f32,
    },
    /// Rotary position embedding over the final dimension. The inputs are
    /// `x[rows, features]`, `cos[rows, features / 2]`, and
    /// `sin[rows, features / 2]`.
    Rope {
        rows: usize,
        features: usize,
    },
    ReduceSum,
    ReduceMax,
    ReduceMin,
    ReduceSumAxis {
        axis: usize,
    },
    ReduceMaxAxis {
        axis: usize,
    },
    ReduceMinAxis {
        axis: usize,
    },
    SoftmaxAxis {
        axis: usize,
    },
    Attention {
        seq: usize,
        head: usize,
        scale: f32,
    },
    AttentionBatched {
        batch: usize,
        seq: usize,
        head: usize,
        scale: f32,
    },
    MatMul {
        m: usize,
        k: usize,
        n: usize,
    },
    Conv2d {
        batch: usize,
        in_channels: usize,
        height: usize,
        width: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: usize,
        padding: usize,
    },
    /// Embedding lookup: `weight[vocab, features]` indexed by
    /// `indices[rows]`, producing `output[rows, features]`.
    Gather {
        rows: usize,
        vocab: usize,
        features: usize,
    },
    /// Indexed row updates: `base[rows, features]`, `indices[updates]`, and
    /// `updates[updates, features]` produce a shape-preserving tensor.
    Scatter {
        rows: usize,
        updates: usize,
        features: usize,
    },
    /// Diagonal state-space scan. For each row, `state = decay * state +
    /// input_gain * x` and `output = output_gain * state`.
    Ssm {
        rows: usize,
        features: usize,
    },
    Output {
        name: String,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UOp {
    pub id: UOpId,
    pub kind: UOpKind,
    pub src: Vec<UOpId>,
    pub shape: Vec<u64>,
}
