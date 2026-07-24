//! A deliberately small UOp-style compiler core for SpatialIR.
//!
//! This is not a second semantic IR. It is the compact executable-kernel IR
//! below SpatialIR: graph rewrites and scheduling happen here, while target
//! lowering and artifact assembly remain explicit Prism responsibilities.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

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

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TinyGraph {
    pub ops: Vec<UOp>,
}

#[derive(Debug, Error, PartialEq)]
pub enum GraphError {
    #[error("UOp {0:?} is missing")]
    Missing(UOpId),
    #[error("graph contains duplicate UOp id {0:?}")]
    DuplicateId(UOpId),
    #[error("graph contains duplicate output name '{0}'")]
    DuplicateOutput(String),
    #[error("graph contains duplicate input name '{0}'")]
    DuplicateInput(String),
    #[error("graph contains an empty input name")]
    EmptyInputName,
    #[error("graph contains an empty output name")]
    EmptyOutputName,
    #[error("UOp {0:?} has an invalid source")]
    InvalidSource(UOpId),
    #[error("graph contains a cycle")]
    Cycle,
    #[error("UOp {0:?} has an invalid arity")]
    InvalidArity(UOpId),
    #[error("target cannot render operation {0:?}")]
    UnsupportedOperation(UOpKind),
    #[error("graph serialization failed: {0}")]
    Serialization(String),
    #[error("missing runtime input '{0}'")]
    MissingInput(String),
    #[error("runtime input '{0}' has the wrong length")]
    InputLength(String),
    #[error("runtime operation {0:?} has incompatible input lengths")]
    ShapeMismatch(UOpId),
}

impl TinyGraph {
    pub fn add(&mut self, kind: UOpKind, src: Vec<UOpId>, shape: Vec<u64>) -> UOpId {
        let id = UOpId(self.ops.len() as u32);
        self.ops.push(UOp {
            id,
            kind,
            src,
            shape,
        });
        id
    }

    /// Add a mean reduction using the compact UOp vocabulary. Mean is
    /// intentionally represented as `ReduceSumAxis` followed by scalar
    /// division, keeping the semantic IR small while allowing the optimizer
    /// and backend lowering to reuse the existing reduction and elementwise
    /// implementations.
    pub fn add_mean_axis(&mut self, input: UOpId, axis: usize) -> Result<UOpId, GraphError> {
        let source = self
            .ops
            .iter()
            .find(|op| op.id == input)
            .ok_or(GraphError::Missing(input))?;
        let reduce = *source
            .shape
            .get(axis)
            .ok_or(GraphError::ShapeMismatch(input))?;
        if reduce == 0 {
            return Err(GraphError::ShapeMismatch(input));
        }
        let mut output_shape = source.shape.clone();
        output_shape.remove(axis);
        let sum = self.add(
            UOpKind::ReduceSumAxis { axis },
            vec![input],
            output_shape.clone(),
        );
        let divisor = self.add(
            UOpKind::Const {
                value: reduce as f32,
            },
            vec![],
            vec![1],
        );
        Ok(self.add(UOpKind::Div, vec![sum, divisor], output_shape))
    }

    /// Add a whole-tensor mean as `ReduceSum` followed by scalar division.
    pub fn add_mean(&mut self, input: UOpId) -> Result<UOpId, GraphError> {
        let source = self
            .ops
            .iter()
            .find(|op| op.id == input)
            .ok_or(GraphError::Missing(input))?;
        let count = element_count(&source.shape);
        if count == 0 {
            return Err(GraphError::ShapeMismatch(input));
        }
        let sum = self.add(UOpKind::ReduceSum, vec![input], vec![1]);
        let divisor = self.add(
            UOpKind::Const {
                value: count as f32,
            },
            vec![],
            vec![1],
        );
        Ok(self.add(UOpKind::Div, vec![sum, divisor], vec![1]))
    }

    pub fn validate(&self) -> Result<(), GraphError> {
        let ids: BTreeSet<_> = self.ops.iter().map(|op| op.id).collect();
        if ids.len() != self.ops.len() {
            let mut seen = BTreeSet::new();
            if let Some(duplicate) = self.ops.iter().map(|op| op.id).find(|id| !seen.insert(*id)) {
                return Err(GraphError::DuplicateId(duplicate));
            }
        }
        let mut output_names = BTreeSet::new();
        let mut input_names = BTreeSet::new();
        for op in &self.ops {
            for source in &op.src {
                if !ids.contains(source) {
                    return Err(GraphError::InvalidSource(op.id));
                }
            }
            if op.shape.is_empty() || op.shape.contains(&0) {
                return Err(GraphError::ShapeMismatch(op.id));
            }
            if op
                .shape
                .iter()
                .try_fold(1usize, |count, dimension| {
                    count.checked_mul(*dimension as usize)
                })
                .is_none()
            {
                return Err(GraphError::ShapeMismatch(op.id));
            }
            let expected = match op.kind {
                UOpKind::Add
                | UOpKind::Mul
                | UOpKind::Sub
                | UOpKind::Div
                | UOpKind::Maximum
                | UOpKind::Minimum
                | UOpKind::MatMul { .. } => 2,
                UOpKind::Where => 3,
                UOpKind::Conv2d { .. } => 3,
                UOpKind::Gather { .. } => 2,
                UOpKind::Scatter { .. } => 3,
                UOpKind::Ssm { .. } => 4,
                UOpKind::RmsNorm { .. } => 2,
                UOpKind::LayerNorm { .. } => 3,
                UOpKind::Rope { .. } => 3,
                UOpKind::Attention { .. } => 3,
                UOpKind::AttentionBatched { .. } => 3,
                UOpKind::Transpose { .. } => 1,
                UOpKind::Reshape => 1,
                UOpKind::Relu
                | UOpKind::Neg
                | UOpKind::Exp
                | UOpKind::Sqrt
                | UOpKind::Abs
                | UOpKind::Log
                | UOpKind::Tanh
                | UOpKind::Sin
                | UOpKind::Cos
                | UOpKind::Gelu
                | UOpKind::Pow { .. }
                | UOpKind::Cast { .. }
                | UOpKind::ReduceSum
                | UOpKind::ReduceMax
                | UOpKind::ReduceMin
                | UOpKind::ReduceSumAxis { .. }
                | UOpKind::ReduceMaxAxis { .. }
                | UOpKind::ReduceMinAxis { .. }
                | UOpKind::SoftmaxAxis { .. }
                | UOpKind::Output { .. } => 1,
                _ => 0,
            };
            if op.src.len() != expected {
                return Err(GraphError::InvalidArity(op.id));
            }
            if let UOpKind::Transpose { permutation } = &op.kind {
                let source = self
                    .ops
                    .iter()
                    .find(|candidate| candidate.id == op.src[0])
                    .unwrap();
                if permutation.len() != source.shape.len()
                    || permutation.iter().copied().collect::<BTreeSet<_>>().len()
                        != permutation.len()
                    || permutation.iter().any(|axis| *axis >= source.shape.len())
                    || op.shape.len() != permutation.len()
                    || permutation
                        .iter()
                        .enumerate()
                        .any(|(out_axis, source_axis)| {
                            op.shape[out_axis] != source.shape[*source_axis]
                        })
                {
                    return Err(GraphError::ShapeMismatch(op.id));
                }
            }
            match op.kind {
                UOpKind::Cast { ref from, ref to } => {
                    if !is_supported_dtype(from) || !is_supported_dtype(to) {
                        return Err(GraphError::ShapeMismatch(op.id));
                    }
                }
                UOpKind::Pow { exponent } if !exponent.is_finite() => {
                    return Err(GraphError::ShapeMismatch(op.id));
                }
                UOpKind::ReduceSum | UOpKind::ReduceMax | UOpKind::ReduceMin => {
                    if op.shape != vec![1] {
                        return Err(GraphError::ShapeMismatch(op.id));
                    }
                }
                UOpKind::Reshape => {
                    let source = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    if element_count(&source.shape) != element_count(&op.shape) {
                        return Err(GraphError::ShapeMismatch(op.id));
                    }
                }
                UOpKind::Output { .. } => {
                    let source = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    if element_count(&source.shape) != element_count(&op.shape) {
                        return Err(GraphError::ShapeMismatch(op.id));
                    }
                }
                UOpKind::Where => {
                    let condition = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let when_true = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[1])
                        .unwrap();
                    let when_false = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[2])
                        .unwrap();
                    if broadcast_shape(&condition.shape, &when_true.shape)
                        .and_then(|shape| broadcast_shape(&shape, &when_false.shape))
                        .as_deref()
                        != Some(op.shape.as_slice())
                    {
                        return Err(GraphError::ShapeMismatch(op.id));
                    }
                }
                UOpKind::Add
                | UOpKind::Mul
                | UOpKind::Sub
                | UOpKind::Div
                | UOpKind::Maximum
                | UOpKind::Minimum => {
                    let left = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let right = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[1])
                        .unwrap();
                    if broadcast_shape(&left.shape, &right.shape).as_deref()
                        != Some(op.shape.as_slice())
                    {
                        return Err(GraphError::ShapeMismatch(op.id));
                    }
                }
                UOpKind::MatMul { m, k, n } => {
                    let left = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let right = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[1])
                        .unwrap();
                    if left.shape != vec![m as u64, k as u64]
                        || right.shape != vec![k as u64, n as u64]
                        || op.shape != vec![m as u64, n as u64]
                    {
                        return Err(GraphError::ShapeMismatch(op.id));
                    }
                }
                UOpKind::Conv2d {
                    batch,
                    in_channels,
                    height,
                    width,
                    out_channels,
                    kernel_h,
                    kernel_w,
                    stride,
                    padding,
                } => {
                    let input = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let weight = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[1])
                        .unwrap();
                    let bias = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[2])
                        .unwrap();
                    if stride == 0
                        || height + 2 * padding < kernel_h
                        || width + 2 * padding < kernel_w
                    {
                        return Err(GraphError::ShapeMismatch(op.id));
                    }
                    let out_h = (height + 2 * padding - kernel_h) / stride + 1;
                    let out_w = (width + 2 * padding - kernel_w) / stride + 1;
                    if input.shape
                        != vec![
                            batch as u64,
                            in_channels as u64,
                            height as u64,
                            width as u64,
                        ]
                        || weight.shape
                            != vec![
                                out_channels as u64,
                                in_channels as u64,
                                kernel_h as u64,
                                kernel_w as u64,
                            ]
                        || bias.shape != vec![out_channels as u64]
                        || op.shape
                            != vec![
                                batch as u64,
                                out_channels as u64,
                                out_h as u64,
                                out_w as u64,
                            ]
                    {
                        return Err(GraphError::ShapeMismatch(op.id));
                    }
                }
                UOpKind::AttentionBatched {
                    batch,
                    seq,
                    head,
                    scale,
                } => {
                    let expected = vec![batch as u64, seq as u64, head as u64];
                    for source in &op.src {
                        if self
                            .ops
                            .iter()
                            .find(|candidate| candidate.id == *source)
                            .unwrap()
                            .shape
                            != expected
                        {
                            return Err(GraphError::ShapeMismatch(op.id));
                        }
                    }
                    if op.shape != expected || !scale.is_finite() {
                        return Err(GraphError::ShapeMismatch(op.id));
                    }
                }
                UOpKind::RmsNorm {
                    rows,
                    features,
                    epsilon,
                } => {
                    let input = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let weight = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[1])
                        .unwrap();
                    let expected = vec![rows as u64, features as u64];
                    if !epsilon.is_finite()
                        || epsilon < 0.0
                        || input.shape != expected
                        || weight.shape != vec![features as u64]
                        || op.shape != expected
                    {
                        return Err(GraphError::ShapeMismatch(op.id));
                    }
                }
                UOpKind::Attention { seq, head, scale } => {
                    let q = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let k = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[1])
                        .unwrap();
                    let v = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[2])
                        .unwrap();
                    let expected = vec![seq as u64, head as u64];
                    if !scale.is_finite()
                        || q.shape != expected
                        || k.shape != expected
                        || v.shape != expected
                        || op.shape != expected
                    {
                        return Err(GraphError::ShapeMismatch(op.id));
                    }
                }
                UOpKind::Rope { rows, features } => {
                    let x = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let cos = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[1])
                        .unwrap();
                    let sin = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[2])
                        .unwrap();
                    if features == 0
                        || features % 2 != 0
                        || x.shape != vec![rows as u64, features as u64]
                        || cos.shape != vec![rows as u64, (features / 2) as u64]
                        || sin.shape != cos.shape
                        || op.shape != x.shape
                    {
                        return Err(GraphError::ShapeMismatch(op.id));
                    }
                }
                UOpKind::Gather {
                    rows,
                    vocab,
                    features,
                } => {
                    let weight = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let indices = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[1])
                        .unwrap();
                    if vocab == 0
                        || features == 0
                        || weight.shape != vec![vocab as u64, features as u64]
                        || indices.shape != vec![rows as u64]
                        || op.shape != vec![rows as u64, features as u64]
                    {
                        return Err(GraphError::ShapeMismatch(op.id));
                    }
                }
                UOpKind::Scatter {
                    rows,
                    updates,
                    features,
                } => {
                    let base = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let indices = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[1])
                        .unwrap();
                    let values = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[2])
                        .unwrap();
                    if rows == 0
                        || updates == 0
                        || features == 0
                        || base.shape != vec![rows as u64, features as u64]
                        || indices.shape != vec![updates as u64]
                        || values.shape != vec![updates as u64, features as u64]
                        || op.shape != base.shape
                    {
                        return Err(GraphError::ShapeMismatch(op.id));
                    }
                }
                UOpKind::Ssm { rows, features } => {
                    let input = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let decay = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[1])
                        .unwrap();
                    let input_gain = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[2])
                        .unwrap();
                    let output_gain = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[3])
                        .unwrap();
                    let expected = vec![rows as u64, features as u64];
                    let vector = vec![features as u64];
                    if rows == 0
                        || features == 0
                        || input.shape != expected
                        || op.shape != expected
                        || decay.shape != vector
                        || input_gain.shape != vector
                        || output_gain.shape != vector
                    {
                        return Err(GraphError::ShapeMismatch(op.id));
                    }
                }
                UOpKind::LayerNorm {
                    rows,
                    features,
                    epsilon,
                } => {
                    let input = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let weight = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[1])
                        .unwrap();
                    let bias = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[2])
                        .unwrap();
                    let expected = vec![rows as u64, features as u64];
                    let parameter = vec![features as u64];
                    if !epsilon.is_finite()
                        || epsilon < 0.0
                        || rows == 0
                        || features == 0
                        || input.shape != expected
                        || op.shape != expected
                        || weight.shape != parameter
                        || bias.shape != parameter
                    {
                        return Err(GraphError::ShapeMismatch(op.id));
                    }
                }
                UOpKind::Relu
                | UOpKind::Neg
                | UOpKind::Exp
                | UOpKind::Sqrt
                | UOpKind::Abs
                | UOpKind::Log
                | UOpKind::Tanh
                | UOpKind::Sin
                | UOpKind::Cos
                | UOpKind::Gelu
                | UOpKind::ReduceSumAxis { .. }
                | UOpKind::ReduceMaxAxis { .. }
                | UOpKind::ReduceMinAxis { .. }
                | UOpKind::SoftmaxAxis { .. } => {
                    let source = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    if let UOpKind::ReduceSumAxis { axis }
                    | UOpKind::ReduceMaxAxis { axis }
                    | UOpKind::ReduceMinAxis { axis } = op.kind
                    {
                        let expected = source
                            .shape
                            .iter()
                            .enumerate()
                            .filter_map(|(index, dim)| (index != axis).then_some(*dim))
                            .collect::<Vec<_>>();
                        if axis >= source.shape.len()
                            || op.shape.len() + 1 != source.shape.len()
                            || op.shape != expected
                        {
                            return Err(GraphError::ShapeMismatch(op.id));
                        }
                    }
                    if let UOpKind::SoftmaxAxis { axis } = op.kind {
                        if axis >= source.shape.len() || op.shape != source.shape {
                            return Err(GraphError::ShapeMismatch(op.id));
                        }
                    }
                    if element_count(&source.shape) != element_count(&op.shape) {
                        if !matches!(
                            op.kind,
                            UOpKind::ReduceSum | UOpKind::ReduceMax | UOpKind::ReduceMin
                        ) && !matches!(
                            op.kind,
                            UOpKind::ReduceSumAxis { .. }
                                | UOpKind::ReduceMaxAxis { .. }
                                | UOpKind::ReduceMinAxis { .. }
                        ) {
                            return Err(GraphError::ShapeMismatch(op.id));
                        }
                        if let UOpKind::ReduceSumAxis { axis }
                        | UOpKind::ReduceMaxAxis { axis }
                        | UOpKind::ReduceMinAxis { axis } = op.kind
                        {
                            if axis >= source.shape.len()
                                || op.shape.len() + 1 != source.shape.len()
                                || op.shape
                                    != source
                                        .shape
                                        .iter()
                                        .enumerate()
                                        .filter_map(|(index, dim)| (index != axis).then_some(*dim))
                                        .collect::<Vec<_>>()
                            {
                                return Err(GraphError::ShapeMismatch(op.id));
                            }
                        }
                    }
                }
                _ => {}
            }
            if let UOpKind::Output { name } = &op.kind {
                if name.is_empty() {
                    return Err(GraphError::EmptyOutputName);
                }
                if !output_names.insert(name.clone()) {
                    return Err(GraphError::DuplicateOutput(name.clone()));
                }
            }
            if let UOpKind::Input { name } = &op.kind {
                if name.is_empty() {
                    return Err(GraphError::EmptyInputName);
                }
                if !input_names.insert(name.clone()) {
                    return Err(GraphError::DuplicateInput(name.clone()));
                }
            }
        }
        self.schedule().map(|_| ())
    }

    /// Apply semantics-preserving local rewrites while keeping stable UOp
    /// identities. Keeping this pass small is intentional: later passes can
    /// add target-independent rules without introducing backend concepts.
    pub fn optimize(&self) -> Result<Self, GraphError> {
        self.validate()?;
        let mut optimized = self.clone();
        for index in 0..optimized.ops.len() {
            let op = &optimized.ops[index];
            if !matches!(
                op.kind,
                UOpKind::Add
                    | UOpKind::Mul
                    | UOpKind::Sub
                    | UOpKind::Div
                    | UOpKind::Maximum
                    | UOpKind::Minimum
            ) || op.src.len() != 2
            {
                continue;
            }
            let left = optimized
                .ops
                .iter()
                .find(|candidate| candidate.id == op.src[0]);
            let right = optimized
                .ops
                .iter()
                .find(|candidate| candidate.id == op.src[1]);
            if let (
                Some(UOp {
                    kind: UOpKind::Const { value: left },
                    ..
                }),
                Some(UOp {
                    kind: UOpKind::Const { value: right },
                    ..
                }),
            ) = (left, right)
            {
                let value = match op.kind {
                    UOpKind::Add => left + right,
                    UOpKind::Mul => left * right,
                    UOpKind::Sub => left - right,
                    UOpKind::Div => left / right,
                    UOpKind::Maximum => left.max(*right),
                    UOpKind::Minimum => left.min(*right),
                    _ => unreachable!(),
                };
                optimized.ops[index].kind = UOpKind::Const { value };
                optimized.ops[index].src.clear();
            }
        }
        // Collapse repeated pure elementwise expressions. UOp IDs remain
        // stable for capture/provenance consumers; the duplicate node is
        // turned into a shaped constant after all users are redirected.
        let mut canonical = BTreeMap::<(String, Vec<UOpId>, Vec<u64>), UOpId>::new();
        for index in 0..optimized.ops.len() {
            let op = &optimized.ops[index];
            let pure = matches!(
                op.kind,
                UOpKind::Add
                    | UOpKind::Mul
                    | UOpKind::Sub
                    | UOpKind::Div
                    | UOpKind::Maximum
                    | UOpKind::Minimum
                    | UOpKind::Relu
                    | UOpKind::Neg
                    | UOpKind::Exp
                    | UOpKind::Sqrt
                    | UOpKind::Abs
                    | UOpKind::Log
                    | UOpKind::Tanh
                    | UOpKind::Sin
                    | UOpKind::Cos
                    | UOpKind::Gelu
                    | UOpKind::Pow { .. }
                    | UOpKind::Where
                    | UOpKind::Cast { .. }
                    | UOpKind::MatMul { .. }
                    | UOpKind::Attention { .. }
                    | UOpKind::AttentionBatched { .. }
                    | UOpKind::RmsNorm { .. }
                    | UOpKind::LayerNorm { .. }
                    | UOpKind::Rope { .. }
                    | UOpKind::Gather { .. }
                    | UOpKind::Scatter { .. }
                    | UOpKind::Ssm { .. }
                    | UOpKind::Conv2d { .. }
                    | UOpKind::ReduceSum
                    | UOpKind::ReduceMax
                    | UOpKind::ReduceMin
                    | UOpKind::ReduceSumAxis { .. }
                    | UOpKind::ReduceMaxAxis { .. }
                    | UOpKind::ReduceMinAxis { .. }
                    | UOpKind::SoftmaxAxis { .. }
            );
            if !pure {
                continue;
            }
            let mut canonical_sources = op.src.clone();
            if matches!(
                op.kind,
                UOpKind::Add | UOpKind::Mul | UOpKind::Maximum | UOpKind::Minimum
            ) {
                canonical_sources.sort_unstable();
            }
            let key = (
                format!("{:?}", op.kind),
                canonical_sources,
                op.shape.clone(),
            );
            if let Some(canonical_id) = canonical.get(&key).copied() {
                let duplicate_id = op.id;
                for consumer in &mut optimized.ops {
                    for source in &mut consumer.src {
                        if *source == duplicate_id {
                            *source = canonical_id;
                        }
                    }
                }
                optimized.ops[index].kind = UOpKind::Const { value: 0.0 };
                optimized.ops[index].src.clear();
            } else {
                canonical.insert(key, op.id);
            }
        }

        // Reach a local fixed point after rewrites.  A single forward pass is
        // insufficient when folding exposes another fold (for example
        // `(2 + 3) * 4`) or when a duplicated subexpression becomes a
        // constant.  Keep this target-independent and deliberately scalar;
        // the renderer remains responsible for tensor code generation.
        loop {
            let mut changed = false;
            for index in 0..optimized.ops.len() {
                let op = optimized.ops[index].clone();
                let source = |id: UOpId| optimized.ops.iter().find(|candidate| candidate.id == id);
                let unary_value = op.src.first().and_then(|id| {
                    source(*id).and_then(|candidate| match candidate.kind {
                        UOpKind::Const { value } => Some(value),
                        _ => None,
                    })
                });
                let folded = match (op.kind, unary_value) {
                    (UOpKind::Relu, Some(value)) => Some(value.max(0.0)),
                    (UOpKind::Neg, Some(value)) => Some(-value),
                    (UOpKind::Exp, Some(value)) => Some(value.exp()),
                    (UOpKind::Sqrt, Some(value)) => Some(value.sqrt()),
                    (UOpKind::Abs, Some(value)) => Some(value.abs()),
                    (UOpKind::Log, Some(value)) => Some(value.ln()),
                    (UOpKind::Tanh, Some(value)) => Some(value.tanh()),
                    (UOpKind::Sin, Some(value)) => Some(value.sin()),
                    (UOpKind::Cos, Some(value)) => Some(value.cos()),
                    (UOpKind::Gelu, Some(value)) => Some(
                        0.5 * value
                            * (1.0
                                + (std::f32::consts::FRAC_2_SQRT_PI
                                    * (value + 0.044715 * value.powi(3)))
                                .tanh()),
                    ),
                    (UOpKind::Pow { exponent }, Some(value)) => Some(value.powf(exponent)),
                    (UOpKind::Cast { from, to }, Some(value)) => Some(cast_f32(value, &from, &to)),
                    _ => None,
                };
                if let Some(value) = folded {
                    optimized.ops[index].kind = UOpKind::Const { value };
                    optimized.ops[index].src.clear();
                    changed = true;
                }
            }
            for index in 0..optimized.ops.len() {
                let op = optimized.ops[index].clone();
                if !matches!(op.kind, UOpKind::Where) || op.src.len() != 3 {
                    continue;
                }
                let values = op
                    .src
                    .iter()
                    .map(|id| {
                        optimized
                            .ops
                            .iter()
                            .find(|candidate| candidate.id == *id)
                            .and_then(|candidate| match candidate.kind {
                                UOpKind::Const { value } => Some(value),
                                _ => None,
                            })
                    })
                    .collect::<Option<Vec<_>>>();
                if let Some(values) = values {
                    optimized.ops[index].kind = UOpKind::Const {
                        value: if values[0] != 0.0 {
                            values[1]
                        } else {
                            values[2]
                        },
                    };
                    optimized.ops[index].src.clear();
                    changed = true;
                }
            }
            // Fold reductions of a shaped constant.  A Const UOp represents
            // one repeated value, so every output lane has the same result;
            // retaining a shaped Const keeps downstream shape validation
            // intact without materializing a reduction kernel.
            for index in 0..optimized.ops.len() {
                let op = optimized.ops[index].clone();
                let Some(source) = op
                    .src
                    .first()
                    .and_then(|id| optimized.ops.iter().find(|candidate| candidate.id == *id))
                else {
                    continue;
                };
                let UOpKind::Const { value } = source.kind else {
                    continue;
                };
                let folded = match op.kind {
                    UOpKind::ReduceSum => Some(value * element_count(&source.shape) as f32),
                    UOpKind::ReduceMax | UOpKind::ReduceMin => Some(value),
                    UOpKind::ReduceSumAxis { axis } => source
                        .shape
                        .get(axis)
                        .map(|dimension| value * *dimension as f32),
                    UOpKind::ReduceMaxAxis { .. } | UOpKind::ReduceMinAxis { .. } => Some(value),
                    UOpKind::SoftmaxAxis { axis } => source
                        .shape
                        .get(axis)
                        .filter(|dimension| **dimension > 0)
                        .map(|dimension| 1.0 / *dimension as f32),
                    _ => None,
                };
                if let Some(value) = folded {
                    optimized.ops[index].kind = UOpKind::Const { value };
                    optimized.ops[index].src.clear();
                    changed = true;
                }
            }
            // Fold a matmul when both operands are repeated constants.  Every
            // output element is the same dot product, so the compact Const
            // representation remains shape-correct.
            for index in 0..optimized.ops.len() {
                let op = optimized.ops[index].clone();
                let is_attention = matches!(
                    op.kind,
                    UOpKind::Attention { .. } | UOpKind::AttentionBatched { .. }
                );
                let is_gather = matches!(op.kind, UOpKind::Gather { .. });
                let norm_kind = match op.kind {
                    UOpKind::RmsNorm { epsilon, .. } => Some((false, epsilon)),
                    UOpKind::LayerNorm { .. } => Some((true, 0.0)),
                    _ => None,
                };
                let conv_kernel = match op.kind {
                    UOpKind::Conv2d {
                        in_channels,
                        kernel_h,
                        kernel_w,
                        ..
                    } => Some(
                        in_channels
                            .saturating_mul(kernel_h)
                            .saturating_mul(kernel_w),
                    ),
                    _ => None,
                };
                let matmul_k = match op.kind {
                    UOpKind::MatMul { k, .. } => Some(k),
                    _ => None,
                };
                if !is_attention
                    && !is_gather
                    && norm_kind.is_none()
                    && matmul_k.is_none()
                    && conv_kernel.is_none()
                {
                    continue;
                }
                let expected_arity = if is_attention || conv_kernel.is_some() {
                    3
                } else if norm_kind.is_some() {
                    if norm_kind.unwrap().0 {
                        3
                    } else {
                        2
                    }
                } else {
                    2
                };
                if op.src.len() != expected_arity {
                    continue;
                }
                let constants = op
                    .src
                    .iter()
                    .filter_map(|id| {
                        optimized
                            .ops
                            .iter()
                            .find(|candidate| candidate.id == *id)
                            .and_then(|candidate| match candidate.kind {
                                UOpKind::Const { value } => Some(value),
                                _ => None,
                            })
                    })
                    .collect::<Vec<_>>();
                if constants.len() == expected_arity {
                    let value = if is_attention {
                        constants[2]
                    } else if is_gather {
                        constants[0]
                    } else if let Some((layer_norm, epsilon)) = norm_kind {
                        if layer_norm {
                            constants[2]
                        } else {
                            constants[0]
                                * (constants[0] * constants[0] + epsilon).sqrt().recip()
                                * constants[1]
                        }
                    } else if let Some(kernel_elements) = conv_kernel {
                        constants[2] + constants[0] * constants[1] * kernel_elements as f32
                    } else {
                        constants[0] * constants[1] * matmul_k.unwrap() as f32
                    };
                    optimized.ops[index].kind = UOpKind::Const { value };
                    optimized.ops[index].src.clear();
                    changed = true;
                }
            }
            // Eliminate neutral elementwise operands.  Keep the original UOp
            // id as a tombstone and redirect consumers so serialized graph
            // identities remain stable while the lowered graph contains no
            // redundant kernel work.
            for index in 0..optimized.ops.len() {
                let op = optimized.ops[index].clone();
                let replacement = match op.kind {
                    UOpKind::Add | UOpKind::Sub | UOpKind::Mul | UOpKind::Div
                        if op.src.len() == 2 =>
                    {
                        let left = optimized
                            .ops
                            .iter()
                            .find(|candidate| candidate.id == op.src[0]);
                        let right = optimized
                            .ops
                            .iter()
                            .find(|candidate| candidate.id == op.src[1]);
                        let left_scalar = left.and_then(|candidate| match candidate.kind {
                            UOpKind::Const { value } => Some(value),
                            _ => None,
                        });
                        let right_scalar = right.and_then(|candidate| match candidate.kind {
                            UOpKind::Const { value } => Some(value),
                            _ => None,
                        });
                        match (op.kind, left_scalar, right_scalar) {
                            (UOpKind::Add, _, Some(0.0)) => Some(op.src[0]),
                            (UOpKind::Add, Some(0.0), _) => Some(op.src[1]),
                            (UOpKind::Sub, _, Some(0.0)) => Some(op.src[0]),
                            (UOpKind::Mul, _, Some(1.0)) => Some(op.src[0]),
                            (UOpKind::Mul, Some(1.0), _) => Some(op.src[1]),
                            (UOpKind::Div, _, Some(1.0)) => Some(op.src[0]),
                            _ => None,
                        }
                    }
                    _ => None,
                };
                if let Some(replacement) = replacement {
                    let rewritten = op.id;
                    for consumer in &mut optimized.ops {
                        for source in &mut consumer.src {
                            if *source == rewritten {
                                *source = replacement;
                            }
                        }
                    }
                    optimized.ops[index].kind = UOpKind::Const { value: 0.0 };
                    optimized.ops[index].src.clear();
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        Ok(optimized)
    }

    fn prune_unreachable(&self) -> Result<Self, GraphError> {
        self.validate()?;
        let outputs: Vec<UOpId> = self
            .ops
            .iter()
            .filter_map(|op| matches!(op.kind, UOpKind::Output { .. }).then_some(op.id))
            .collect();
        if outputs.is_empty() {
            return Ok(self.clone());
        }
        let mut reachable = BTreeSet::new();
        let mut pending = outputs;
        while let Some(id) = pending.pop() {
            if !reachable.insert(id) {
                continue;
            }
            let op = self
                .ops
                .iter()
                .find(|candidate| candidate.id == id)
                .unwrap();
            pending.extend(op.src.iter().copied());
        }
        let mut pruned = self.clone();
        pruned.ops.retain(|op| reachable.contains(&op.id));
        Ok(pruned)
    }

    /// Stable Kahn schedule. Stable ordering makes captures reproducible.
    pub fn schedule(&self) -> Result<Vec<UOpId>, GraphError> {
        let mut indegree = BTreeMap::new();
        let mut users: BTreeMap<UOpId, Vec<UOpId>> = BTreeMap::new();
        for op in &self.ops {
            indegree.insert(op.id, op.src.len());
            for src in &op.src {
                users.entry(*src).or_default().push(op.id);
            }
        }
        let mut ready: BTreeSet<_> = indegree
            .iter()
            .filter_map(|(id, n)| (*n == 0).then_some(*id))
            .collect();
        let mut result = Vec::with_capacity(self.ops.len());
        while let Some(id) = ready.pop_first() {
            result.push(id);
            for user in users.get(&id).into_iter().flatten() {
                let n = indegree.get_mut(user).unwrap();
                *n -= 1;
                if *n == 0 {
                    ready.insert(*user);
                }
            }
        }
        (result.len() == self.ops.len())
            .then_some(result)
            .ok_or(GraphError::Cycle)
    }

    pub fn lower(&self, target: LoweringTarget) -> Result<CapturePlan, GraphError> {
        let graph = self.optimize()?.prune_unreachable()?;
        let mut groups = Vec::new();
        let mut current = Vec::new();
        let mut use_counts: BTreeMap<UOpId, usize> = BTreeMap::new();
        for op in &graph.ops {
            for source in &op.src {
                *use_counts.entry(*source).or_default() += 1;
            }
        }
        for id in graph.schedule()? {
            let op = graph.ops.iter().find(|op| op.id == id).unwrap();
            let is_broadcast_binary = matches!(
                op.kind,
                UOpKind::Add
                    | UOpKind::Mul
                    | UOpKind::Sub
                    | UOpKind::Div
                    | UOpKind::Maximum
                    | UOpKind::Minimum
            ) && op.src.iter().any(|source| {
                let source = graph
                    .ops
                    .iter()
                    .find(|candidate| candidate.id == *source)
                    .unwrap();
                source.shape != op.shape
                    && !matches!(source.kind, UOpKind::Const { .. } if source.shape == vec![1])
            });
            if is_broadcast_binary {
                if !current.is_empty() {
                    groups.push(KernelGroup {
                        ops: std::mem::take(&mut current),
                    });
                }
                current.push(KernelOp::from_broadcast_op(op, &graph));
                continue;
            }
            match op.kind {
                UOpKind::Add
                | UOpKind::Mul
                | UOpKind::Sub
                | UOpKind::Div
                | UOpKind::Maximum
                | UOpKind::Minimum
                | UOpKind::Relu
                | UOpKind::Neg
                | UOpKind::Exp
                | UOpKind::Sqrt
                | UOpKind::Abs
                | UOpKind::Log
                | UOpKind::Tanh
                | UOpKind::Sin
                | UOpKind::Cos
                | UOpKind::Gelu
                | UOpKind::Pow { .. }
                | UOpKind::Where
                | UOpKind::Cast { .. } => {
                    let broadcast_group =
                        matches!(current.first(), Some(KernelOp::BroadcastBinary { .. }));
                    let broadcast_postlude_supported = matches!(
                        op.kind,
                        UOpKind::Relu
                            | UOpKind::Neg
                            | UOpKind::Exp
                            | UOpKind::Sqrt
                            | UOpKind::Abs
                            | UOpKind::Log
                            | UOpKind::Tanh
                            | UOpKind::Sin
                            | UOpKind::Cos
                            | UOpKind::Gelu
                    );
                    if broadcast_group && !broadcast_postlude_supported {
                        groups.push(KernelGroup {
                            ops: std::mem::take(&mut current),
                        });
                    }
                    if matches!(op.kind, UOpKind::Where) && !current.is_empty() {
                        groups.push(KernelGroup {
                            ops: std::mem::take(&mut current),
                        });
                    }
                    // A group is a straight-line kernel. Do not fuse two
                    // independent branches merely because they are adjacent
                    // in the topological order.
                    let depends_on_group = op.src.iter().any(|src| {
                        current
                            .iter()
                            .any(|candidate: &KernelOp| candidate.id() == *src)
                    });
                    let group_has_fork = current
                        .iter()
                        .any(|candidate| use_counts.get(&candidate.id()).copied().unwrap_or(0) > 1);
                    if !current.is_empty() && (!depends_on_group || group_has_fork) {
                        groups.push(KernelGroup {
                            ops: std::mem::take(&mut current),
                        });
                    }
                    current.push(KernelOp::from_graph_op(op, &graph));
                }
                UOpKind::ReduceSum | UOpKind::ReduceMax | UOpKind::ReduceMin => {
                    if !current.is_empty() {
                        groups.push(KernelGroup {
                            ops: std::mem::take(&mut current),
                        });
                    }
                    groups.push(KernelGroup {
                        ops: vec![KernelOp::from_graph_op(op, &graph)],
                    });
                }
                UOpKind::ReduceSumAxis { .. } => {
                    if !current.is_empty() {
                        groups.push(KernelGroup {
                            ops: std::mem::take(&mut current),
                        });
                    }
                    groups.push(KernelGroup {
                        ops: vec![KernelOp::from_graph_op(op, &graph)],
                    });
                }
                UOpKind::ReduceMaxAxis { .. } => {
                    if !current.is_empty() {
                        groups.push(KernelGroup {
                            ops: std::mem::take(&mut current),
                        });
                    }
                    groups.push(KernelGroup {
                        ops: vec![KernelOp::from_graph_op(op, &graph)],
                    });
                }
                UOpKind::ReduceMinAxis { .. } => {
                    if !current.is_empty() {
                        groups.push(KernelGroup {
                            ops: std::mem::take(&mut current),
                        });
                    }
                    groups.push(KernelGroup {
                        ops: vec![KernelOp::from_graph_op(op, &graph)],
                    });
                }
                UOpKind::SoftmaxAxis { .. } => {
                    if !current.is_empty() {
                        groups.push(KernelGroup {
                            ops: std::mem::take(&mut current),
                        });
                    }
                    groups.push(KernelGroup {
                        ops: vec![KernelOp::from_graph_op(op, &graph)],
                    });
                }
                UOpKind::Attention { .. } => {
                    if !current.is_empty() {
                        groups.push(KernelGroup {
                            ops: std::mem::take(&mut current),
                        });
                    }
                    groups.push(KernelGroup {
                        ops: vec![KernelOp::from_graph_op(op, &graph)],
                    });
                }
                UOpKind::AttentionBatched { .. } => {
                    if !current.is_empty() {
                        groups.push(KernelGroup {
                            ops: std::mem::take(&mut current),
                        });
                    }
                    groups.push(KernelGroup {
                        ops: vec![KernelOp::from_graph_op(op, &graph)],
                    });
                }
                UOpKind::RmsNorm { .. } => {
                    if !current.is_empty() {
                        groups.push(KernelGroup {
                            ops: std::mem::take(&mut current),
                        });
                    }
                    groups.push(KernelGroup {
                        ops: vec![KernelOp::from_graph_op(op, &graph)],
                    });
                }
                UOpKind::LayerNorm { .. } => {
                    if !current.is_empty() {
                        groups.push(KernelGroup {
                            ops: std::mem::take(&mut current),
                        });
                    }
                    groups.push(KernelGroup {
                        ops: vec![KernelOp::from_graph_op(op, &graph)],
                    });
                }
                UOpKind::Rope { .. } => {
                    if !current.is_empty() {
                        groups.push(KernelGroup {
                            ops: std::mem::take(&mut current),
                        });
                    }
                    groups.push(KernelGroup {
                        ops: vec![KernelOp::from_graph_op(op, &graph)],
                    });
                }
                UOpKind::Gather { .. } => {
                    if !current.is_empty() {
                        groups.push(KernelGroup {
                            ops: std::mem::take(&mut current),
                        });
                    }
                    groups.push(KernelGroup {
                        ops: vec![KernelOp::from_graph_op(op, &graph)],
                    });
                }
                UOpKind::Scatter { .. } => {
                    if !current.is_empty() {
                        groups.push(KernelGroup {
                            ops: std::mem::take(&mut current),
                        });
                    }
                    groups.push(KernelGroup {
                        ops: vec![KernelOp::from_graph_op(op, &graph)],
                    });
                }
                UOpKind::Ssm { .. } => {
                    if !current.is_empty() {
                        groups.push(KernelGroup {
                            ops: std::mem::take(&mut current),
                        });
                    }
                    groups.push(KernelGroup {
                        ops: vec![KernelOp::from_graph_op(op, &graph)],
                    });
                }
                UOpKind::Transpose { .. } => {
                    if !current.is_empty() {
                        groups.push(KernelGroup {
                            ops: std::mem::take(&mut current),
                        });
                    }
                    groups.push(KernelGroup {
                        ops: vec![KernelOp::from_graph_op(op, &graph)],
                    });
                }
                UOpKind::MatMul { .. } => {
                    if !current.is_empty() {
                        groups.push(KernelGroup {
                            ops: std::mem::take(&mut current),
                        });
                    }
                    groups.push(KernelGroup {
                        ops: vec![KernelOp::from_graph_op(op, &graph)],
                    });
                }
                UOpKind::Conv2d { .. } => {
                    if !current.is_empty() {
                        groups.push(KernelGroup {
                            ops: std::mem::take(&mut current),
                        });
                    }
                    groups.push(KernelGroup {
                        ops: vec![KernelOp::from_graph_op(op, &graph)],
                    });
                }
                _ if !current.is_empty() => {
                    groups.push(KernelGroup {
                        ops: std::mem::take(&mut current),
                    });
                }
                _ => {}
            }
        }
        if !current.is_empty() {
            groups.push(KernelGroup { ops: current });
        }
        let kernels: Vec<LoweredKernel> = groups
            .into_iter()
            .map(|group| {
                let (source, source_digest) = render_kernel(&group, target);
                let output_elements = group.ops.last().and_then(|op| {
                    graph
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.id())
                        .and_then(|candidate| {
                            candidate.shape.iter().try_fold(1usize, |count, dimension| {
                                count.checked_mul(*dimension as usize)
                            })
                        })
                });
                LoweredKernel {
                    group,
                    source,
                    source_digest,
                    output_elements,
                }
            })
            .collect();
        let kernel_count = kernels.len();
        let memory_plan = graph.memory_plan()?;
        Ok(CapturePlan {
            target,
            kernels,
            graph_op_count: graph.ops.len(),
            graph,
            memory_plan,
            replay: ReplayPlan {
                command_ids: (0..kernel_count).map(|id| id as u32).collect(),
                synchronization_points: (0..kernel_count).map(|id| id as u32).collect(),
                persistent: false,
            },
        })
    }

    /// Lower the graph using an explicitly selected fusion layout.
    ///
    /// The ordinary [`Self::lower`] path emits maximal safe straight-line
    /// groups. Strategy search can request a different executable capture:
    /// per-operation materializes every boundary, while interleaved fusion
    /// partitions each safe group into the requested number of stages.
    /// Persistent megakernels retain the standard group layout but receive a
    /// stable replay plan suitable for repeated submission.
    pub fn lower_with_fusion_strategy(
        &self,
        target: LoweringTarget,
        strategy: &crate::fused_ops::FusionStrategy,
    ) -> Result<CapturePlan, GraphError> {
        let mut capture = self.lower(target)?;
        let stage_count = match strategy {
            crate::fused_ops::FusionStrategy::PerOperation => None,
            crate::fused_ops::FusionStrategy::InterleavedFused { stages } => {
                // The graph adapter may request an interleaved candidate
                // before it has a concrete evolutionary stage payload. An
                // empty request still denotes an interleaved alternative,
                // so infer two stages for multi-op kernels instead of
                // silently collapsing it into StandardFused.
                Some(if stages.is_empty() { 2 } else { stages.len() })
            }
            crate::fused_ops::FusionStrategy::StandardFused
            | crate::fused_ops::FusionStrategy::PersistentMegakernel { .. } => Some(1),
        };
        let mut groups = Vec::new();
        for kernel in capture.kernels {
            match stage_count {
                None => groups.extend(
                    kernel
                        .group
                        .ops
                        .into_iter()
                        .map(|op| KernelGroup { ops: vec![op] }),
                ),
                Some(stages) if stages > 1 && kernel.group.ops.len() > 1 => {
                    let chunk = kernel.group.ops.len().div_ceil(stages);
                    groups.extend(
                        kernel
                            .group
                            .ops
                            .chunks(chunk)
                            .map(|ops| KernelGroup { ops: ops.to_vec() }),
                    );
                }
                _ => groups.push(kernel.group),
            }
        }
        if matches!(
            strategy,
            crate::fused_ops::FusionStrategy::PersistentMegakernel { .. }
        ) {
            groups = Self::merge_persistent_elementwise_groups(groups, &capture.graph);
        }
        capture.kernels = groups
            .into_iter()
            .map(|group| {
                let (source, source_digest) = render_kernel(&group, target);
                let output_elements = group.ops.last().and_then(|op| {
                    capture
                        .graph
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.id())
                        .and_then(|candidate| {
                            candidate.shape.iter().try_fold(1usize, |count, dimension| {
                                count.checked_mul(*dimension as usize)
                            })
                        })
                });
                LoweredKernel {
                    group,
                    source,
                    source_digest,
                    output_elements,
                }
            })
            .collect();
        capture.replay = ReplayPlan {
            command_ids: (0..capture.kernels.len()).map(|id| id as u32).collect(),
            synchronization_points: (0..capture.kernels.len()).map(|id| id as u32).collect(),
            persistent: matches!(
                strategy,
                crate::fused_ops::FusionStrategy::PersistentMegakernel { .. }
            ),
        };
        capture.validate().map_err(|error| {
            GraphError::Serialization(format!(
                "fusion strategy produced an invalid capture: {error}"
            ))
        })?;
        Ok(capture)
    }

    fn merge_persistent_elementwise_groups(
        groups: Vec<KernelGroup>,
        graph: &TinyGraph,
    ) -> Vec<KernelGroup> {
        if groups.len() < 2 {
            return groups;
        }
        let eligible = |group: &KernelGroup| {
            if group.ops.is_empty() {
                return false;
            }
            let shape = group.ops.first().and_then(|kernel_op| {
                graph
                    .ops
                    .iter()
                    .find(|op| op.id == kernel_op.id())
                    .map(|op| op.shape.clone())
            });
            shape.is_some_and(|shape| {
                group.ops.iter().all(|kernel_op| {
                    let Some(op) = graph.ops.iter().find(|op| op.id == kernel_op.id()) else {
                        return false;
                    };
                    op.shape == shape
                        && matches!(
                            kernel_op,
                            KernelOp::Add { .. }
                                | KernelOp::Mul { .. }
                                | KernelOp::Sub { .. }
                                | KernelOp::Div { .. }
                                | KernelOp::Maximum { .. }
                                | KernelOp::Minimum { .. }
                                | KernelOp::Relu { .. }
                                | KernelOp::Neg { .. }
                                | KernelOp::Exp { .. }
                                | KernelOp::Sqrt { .. }
                                | KernelOp::Abs { .. }
                                | KernelOp::Log { .. }
                                | KernelOp::Tanh { .. }
                                | KernelOp::Sin { .. }
                                | KernelOp::Cos { .. }
                                | KernelOp::Gelu { .. }
                                | KernelOp::Pow { .. }
                                | KernelOp::Cast { .. }
                        )
                })
            })
        };
        if !groups.iter().all(eligible) {
            return groups;
        }
        for pair in groups.windows(2) {
            let previous = pair[0].ops.last().map(KernelOp::id);
            let first = pair[1].ops.first().map(KernelOp::id);
            let chained = previous.zip(first).is_some_and(|(previous, first)| {
                graph
                    .ops
                    .iter()
                    .find(|op| op.id == first)
                    .is_some_and(|op| op.src.contains(&previous))
            });
            if !chained {
                return groups;
            }
        }
        vec![KernelGroup {
            ops: groups.into_iter().flat_map(|group| group.ops).collect(),
        }]
    }

    /// Deterministic scalar reference execution for capture validation.
    /// Backends can use this as a behavioral oracle before publishing an
    /// executable artifact.
    pub fn execute_f32(
        &self,
        inputs: &BTreeMap<String, Vec<f32>>,
    ) -> Result<BTreeMap<String, Vec<f32>>, GraphError> {
        self.validate()?;
        let mut values: BTreeMap<UOpId, Vec<f32>> = BTreeMap::new();
        let mut outputs = BTreeMap::new();
        for id in self.schedule()? {
            let op = self.ops.iter().find(|op| op.id == id).unwrap();
            let value = match &op.kind {
                UOpKind::Input { name } => {
                    let value = inputs
                        .get(name)
                        .ok_or_else(|| GraphError::MissingInput(name.clone()))?;
                    let expected = element_count(&op.shape);
                    if value.len() != expected {
                        return Err(GraphError::InputLength(name.clone()));
                    }
                    value.clone()
                }
                UOpKind::Const { value } => vec![*value; element_count(&op.shape)],
                UOpKind::Reshape => values.get(&op.src[0]).unwrap().clone(),
                UOpKind::Add
                | UOpKind::Mul
                | UOpKind::Sub
                | UOpKind::Div
                | UOpKind::Maximum
                | UOpKind::Minimum => {
                    let left = values.get(&op.src[0]).unwrap();
                    let right = values.get(&op.src[1]).unwrap();
                    let left_shape = &self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap()
                        .shape;
                    let right_shape = &self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[1])
                        .unwrap()
                        .shape;
                    if broadcast_shape(left_shape, right_shape).as_deref()
                        != Some(op.shape.as_slice())
                    {
                        return Err(GraphError::ShapeMismatch(op.id));
                    }
                    (0..element_count(&op.shape))
                        .map(|index| {
                            let lhs = left[broadcast_index(index, &op.shape, left_shape)];
                            let rhs = right[broadcast_index(index, &op.shape, right_shape)];
                            match op.kind {
                                UOpKind::Add => lhs + rhs,
                                UOpKind::Mul => lhs * rhs,
                                UOpKind::Sub => lhs - rhs,
                                UOpKind::Div => lhs / rhs,
                                UOpKind::Maximum => lhs.max(rhs),
                                UOpKind::Minimum => lhs.min(rhs),
                                _ => unreachable!(),
                            }
                        })
                        .collect()
                }
                UOpKind::Where => {
                    let condition = values.get(&op.src[0]).unwrap();
                    let when_true = values.get(&op.src[1]).unwrap();
                    let when_false = values.get(&op.src[2]).unwrap();
                    let condition_shape = &self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap()
                        .shape;
                    let true_shape = &self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[1])
                        .unwrap()
                        .shape;
                    let false_shape = &self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[2])
                        .unwrap()
                        .shape;
                    (0..element_count(&op.shape))
                        .map(|index| {
                            if condition[broadcast_index(index, &op.shape, condition_shape)] != 0.0
                            {
                                when_true[broadcast_index(index, &op.shape, true_shape)]
                            } else {
                                when_false[broadcast_index(index, &op.shape, false_shape)]
                            }
                        })
                        .collect()
                }
                UOpKind::Relu => values
                    .get(&op.src[0])
                    .unwrap()
                    .iter()
                    .map(|value| value.max(0.0))
                    .collect(),
                UOpKind::Neg => values
                    .get(&op.src[0])
                    .unwrap()
                    .iter()
                    .map(|value| -*value)
                    .collect(),
                UOpKind::Exp => values
                    .get(&op.src[0])
                    .unwrap()
                    .iter()
                    .map(|value| value.exp())
                    .collect(),
                UOpKind::Sqrt => values
                    .get(&op.src[0])
                    .unwrap()
                    .iter()
                    .map(|value| value.sqrt())
                    .collect(),
                UOpKind::Abs => values
                    .get(&op.src[0])
                    .unwrap()
                    .iter()
                    .map(|value| value.abs())
                    .collect(),
                UOpKind::Log => values
                    .get(&op.src[0])
                    .unwrap()
                    .iter()
                    .map(|value| value.ln())
                    .collect(),
                UOpKind::Tanh => values
                    .get(&op.src[0])
                    .unwrap()
                    .iter()
                    .map(|value| value.tanh())
                    .collect(),
                UOpKind::Sin => values
                    .get(&op.src[0])
                    .unwrap()
                    .iter()
                    .map(|value| value.sin())
                    .collect(),
                UOpKind::Cos => values
                    .get(&op.src[0])
                    .unwrap()
                    .iter()
                    .map(|value| value.cos())
                    .collect(),
                UOpKind::Gelu => values
                    .get(&op.src[0])
                    .unwrap()
                    .iter()
                    .map(|value| {
                        0.5 * *value
                            * (1.0
                                + (std::f32::consts::FRAC_2_SQRT_PI
                                    * (*value + 0.044715 * value.powi(3)))
                                .tanh())
                    })
                    .collect(),
                UOpKind::Pow { exponent } => values
                    .get(&op.src[0])
                    .unwrap()
                    .iter()
                    .map(|value| value.powf(*exponent))
                    .collect(),
                UOpKind::Cast { from, to } => values
                    .get(&op.src[0])
                    .unwrap()
                    .iter()
                    .map(|value| cast_f32(*value, from, to))
                    .collect(),
                UOpKind::RmsNorm {
                    rows,
                    features,
                    epsilon,
                } => {
                    let input = values.get(&op.src[0]).unwrap();
                    let weight = values.get(&op.src[1]).unwrap();
                    let mut output = vec![0.0; rows * features];
                    for row in 0..*rows {
                        let base = row * *features;
                        let mean_sq = input[base..base + features]
                            .iter()
                            .map(|v| v * v)
                            .sum::<f32>()
                            / *features as f32;
                        let inv = 1.0 / (mean_sq + epsilon).sqrt();
                        for col in 0..*features {
                            output[base + col] = input[base + col] * inv * weight[col];
                        }
                    }
                    output
                }
                UOpKind::ReduceMinAxis { axis } => {
                    let source = values.get(&op.src[0]).unwrap();
                    let source_op = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let outer: usize = source_op.shape[..*axis]
                        .iter()
                        .map(|dim| *dim as usize)
                        .product();
                    let reduce = source_op.shape[*axis] as usize;
                    let inner: usize = source_op.shape[*axis + 1..]
                        .iter()
                        .map(|dim| *dim as usize)
                        .product();
                    let mut output = Vec::with_capacity(outer * inner);
                    for outer_index in 0..outer {
                        for inner_index in 0..inner {
                            let mut minimum = f32::INFINITY;
                            for step in 0..reduce {
                                minimum = minimum.min(
                                    source[(outer_index * reduce + step) * inner + inner_index],
                                );
                            }
                            output.push(minimum);
                        }
                    }
                    output
                }
                UOpKind::Conv2d {
                    batch,
                    in_channels,
                    height,
                    width,
                    out_channels,
                    kernel_h,
                    kernel_w,
                    stride,
                    padding,
                } => {
                    let input = values.get(&op.src[0]).unwrap();
                    let weight = values.get(&op.src[1]).unwrap();
                    let bias = values.get(&op.src[2]).unwrap();
                    let out_h = (height + 2 * padding - kernel_h) / stride + 1;
                    let out_w = (width + 2 * padding - kernel_w) / stride + 1;
                    let mut output = vec![0.0; batch * out_channels * out_h * out_w];
                    for b in 0..*batch {
                        for oc in 0..*out_channels {
                            for oh in 0..out_h {
                                for ow in 0..out_w {
                                    let mut sum = bias[oc];
                                    for ic in 0..*in_channels {
                                        for kh in 0..*kernel_h {
                                            for kw in 0..*kernel_w {
                                                let ih = oh * *stride + kh;
                                                let iw = ow * *stride + kw;
                                                if ih + *padding >= *padding
                                                    && iw + *padding >= *padding
                                                    && ih + *padding < *height + *padding
                                                    && iw + *padding < *width + *padding
                                                {
                                                    let src_h = ih as isize - *padding as isize;
                                                    let src_w = iw as isize - *padding as isize;
                                                    if src_h >= 0
                                                        && src_w >= 0
                                                        && (src_h as usize) < *height
                                                        && (src_w as usize) < *width
                                                    {
                                                        sum += input[((b * *in_channels + ic)
                                                            * *height
                                                            + src_h as usize)
                                                            * *width
                                                            + src_w as usize]
                                                            * weight[(((oc * *in_channels + ic)
                                                                * *kernel_h
                                                                + kh)
                                                                * *kernel_w)
                                                                + kw];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    output[((b * *out_channels + oc) * out_h + oh) * out_w + ow] =
                                        sum;
                                }
                            }
                        }
                    }
                    output
                }
                UOpKind::LayerNorm {
                    rows,
                    features,
                    epsilon,
                } => {
                    let input = values.get(&op.src[0]).unwrap();
                    let weight = values.get(&op.src[1]).unwrap();
                    let bias = values.get(&op.src[2]).unwrap();
                    let mut output = vec![0.0; rows * features];
                    for row in 0..*rows {
                        let base = row * *features;
                        let mean =
                            input[base..base + features].iter().sum::<f32>() / *features as f32;
                        let variance = input[base..base + features]
                            .iter()
                            .map(|v| {
                                let centered = *v - mean;
                                centered * centered
                            })
                            .sum::<f32>()
                            / *features as f32;
                        let inv = (variance + epsilon).sqrt().recip();
                        for col in 0..*features {
                            output[base + col] =
                                (input[base + col] - mean) * inv * weight[col] + bias[col];
                        }
                    }
                    output
                }
                UOpKind::Rope { rows, features } => {
                    let input = values.get(&op.src[0]).unwrap();
                    let cos = values.get(&op.src[1]).unwrap();
                    let sin = values.get(&op.src[2]).unwrap();
                    let half = *features / 2;
                    let mut output = vec![0.0; rows * features];
                    for row in 0..*rows {
                        let x_base = row * *features;
                        let angle_base = row * half;
                        for pair in 0..half {
                            let x0 = input[x_base + 2 * pair];
                            let x1 = input[x_base + 2 * pair + 1];
                            let c = cos[angle_base + pair];
                            let s = sin[angle_base + pair];
                            output[x_base + 2 * pair] = x0 * c - x1 * s;
                            output[x_base + 2 * pair + 1] = x0 * s + x1 * c;
                        }
                    }
                    output
                }
                UOpKind::Gather {
                    rows,
                    vocab,
                    features,
                } => {
                    let weight = values.get(&op.src[0]).unwrap();
                    let indices = values.get(&op.src[1]).unwrap();
                    let mut output = vec![0.0; rows * features];
                    for row in 0..*rows {
                        let raw = indices[row];
                        if !raw.is_finite()
                            || raw < 0.0
                            || raw.fract() != 0.0
                            || raw >= *vocab as f32
                        {
                            return Err(GraphError::ShapeMismatch(op.id));
                        }
                        let source = raw as usize * *features;
                        output[row * *features..(row + 1) * *features]
                            .copy_from_slice(&weight[source..source + *features]);
                    }
                    output
                }
                UOpKind::Scatter {
                    rows,
                    updates,
                    features,
                } => {
                    let base = values.get(&op.src[0]).unwrap();
                    let indices = values.get(&op.src[1]).unwrap();
                    let update_values = values.get(&op.src[2]).unwrap();
                    let mut output = base.clone();
                    for (update, raw) in indices.iter().take(*updates).enumerate() {
                        if !raw.is_finite()
                            || *raw < 0.0
                            || raw.fract() != 0.0
                            || *raw >= *rows as f32
                        {
                            return Err(GraphError::ShapeMismatch(op.id));
                        }
                        let raw = *raw;
                        let destination = raw as usize * *features;
                        let source = update * *features;
                        output[destination..destination + *features]
                            .copy_from_slice(&update_values[source..source + *features]);
                    }
                    output
                }
                UOpKind::Ssm { rows, features } => {
                    let input = values.get(&op.src[0]).unwrap();
                    let decay = values.get(&op.src[1]).unwrap();
                    let input_gain = values.get(&op.src[2]).unwrap();
                    let output_gain = values.get(&op.src[3]).unwrap();
                    let mut state = vec![0.0; *features];
                    let mut output = vec![0.0; rows * features];
                    for row in 0..*rows {
                        for feature in 0..*features {
                            let index = row * *features + feature;
                            state[feature] = decay[feature] * state[feature]
                                + input_gain[feature] * input[index];
                            output[index] = output_gain[feature] * state[feature];
                        }
                    }
                    output
                }
                UOpKind::Transpose { permutation } => {
                    let source_op = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let input = values.get(&op.src[0]).unwrap();
                    let mut source_strides = vec![1usize; source_op.shape.len()];
                    for axis in (0..source_op.shape.len().saturating_sub(1)).rev() {
                        source_strides[axis] =
                            source_strides[axis + 1] * source_op.shape[axis + 1] as usize;
                    }
                    let mut output = vec![0.0; element_count(&op.shape)];
                    for (out_linear, destination) in output.iter_mut().enumerate() {
                        let mut remainder = out_linear;
                        let mut source_linear = 0usize;
                        for out_axis in (0..op.shape.len()).rev() {
                            let coordinate = remainder % op.shape[out_axis] as usize;
                            remainder /= op.shape[out_axis] as usize;
                            source_linear += coordinate * source_strides[permutation[out_axis]];
                        }
                        *destination = input[source_linear];
                    }
                    output
                }
                UOpKind::ReduceSum => vec![values.get(&op.src[0]).unwrap().iter().sum()],
                UOpKind::ReduceMax => vec![values
                    .get(&op.src[0])
                    .unwrap()
                    .iter()
                    .copied()
                    .fold(f32::NEG_INFINITY, f32::max)],
                UOpKind::ReduceMin => vec![values
                    .get(&op.src[0])
                    .unwrap()
                    .iter()
                    .copied()
                    .fold(f32::INFINITY, f32::min)],
                UOpKind::ReduceSumAxis { axis } => {
                    let source = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let input = values.get(&op.src[0]).unwrap();
                    let outer: usize = source.shape[..*axis]
                        .iter()
                        .map(|dim| *dim as usize)
                        .product();
                    let reduce = source.shape[*axis] as usize;
                    let inner: usize = source.shape[*axis + 1..]
                        .iter()
                        .map(|dim| *dim as usize)
                        .product();
                    let mut output = vec![0.0; outer * inner];
                    for out in 0..outer {
                        for col in 0..inner {
                            output[out * inner + col] = (0..reduce)
                                .map(|step| input[(out * reduce + step) * inner + col])
                                .sum();
                        }
                    }
                    output
                }
                UOpKind::ReduceMaxAxis { axis } => {
                    let source = values.get(&op.src[0]).unwrap();
                    let source_op = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let outer: usize = source_op.shape[..*axis]
                        .iter()
                        .map(|dim| *dim as usize)
                        .product();
                    let reduce = source_op.shape[*axis] as usize;
                    let inner: usize = source_op.shape[*axis + 1..]
                        .iter()
                        .map(|dim| *dim as usize)
                        .product();
                    let mut output = Vec::with_capacity(outer * inner);
                    for outer_index in 0..outer {
                        for inner_index in 0..inner {
                            let mut maximum = f32::NEG_INFINITY;
                            for step in 0..reduce {
                                maximum = maximum.max(
                                    source[(outer_index * reduce + step) * inner + inner_index],
                                );
                            }
                            output.push(maximum);
                        }
                    }
                    output
                }
                UOpKind::SoftmaxAxis { axis } => {
                    let source = self
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let input = values.get(&op.src[0]).unwrap();
                    let outer: usize = source.shape[..*axis]
                        .iter()
                        .map(|dim| *dim as usize)
                        .product();
                    let reduce = source.shape[*axis] as usize;
                    let inner: usize = source.shape[*axis + 1..]
                        .iter()
                        .map(|dim| *dim as usize)
                        .product();
                    let mut output = vec![0.0; outer * reduce * inner];
                    for out in 0..outer {
                        for col in 0..inner {
                            let max = (0..reduce)
                                .map(|step| input[(out * reduce + step) * inner + col])
                                .fold(f32::NEG_INFINITY, f32::max);
                            let denominator: f32 = (0..reduce)
                                .map(|step| {
                                    (input[(out * reduce + step) * inner + col] - max).exp()
                                })
                                .sum();
                            for step in 0..reduce {
                                output[(out * reduce + step) * inner + col] =
                                    (input[(out * reduce + step) * inner + col] - max).exp()
                                        / denominator;
                            }
                        }
                    }
                    output
                }
                UOpKind::Attention { seq, head, scale } => {
                    let q = values.get(&op.src[0]).unwrap();
                    let k = values.get(&op.src[1]).unwrap();
                    let v = values.get(&op.src[2]).unwrap();
                    let mut output = vec![0.0; *seq * *head];
                    for query in 0..*seq {
                        let mut scores = vec![0.0; *seq];
                        for key in 0..*seq {
                            scores[key] = (0..*head)
                                .map(|dim| q[query * *head + dim] * k[key * *head + dim])
                                .sum::<f32>()
                                * *scale;
                        }
                        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let denominator: f32 =
                            scores.iter().map(|score| (*score - max).exp()).sum();
                        for dim in 0..*head {
                            output[query * *head + dim] = (0..*seq)
                                .map(|key| {
                                    ((*scores.get(key).unwrap() - max).exp() / denominator)
                                        * v[key * *head + dim]
                                })
                                .sum();
                        }
                    }
                    output
                }
                UOpKind::AttentionBatched {
                    batch,
                    seq,
                    head,
                    scale,
                } => {
                    let q = values.get(&op.src[0]).unwrap();
                    let k = values.get(&op.src[1]).unwrap();
                    let v = values.get(&op.src[2]).unwrap();
                    let mut output = vec![0.0; *batch * *seq * *head];
                    for batch_index in 0..*batch {
                        let base = batch_index * *seq * *head;
                        for query in 0..*seq {
                            let mut scores = vec![0.0; *seq];
                            for key in 0..*seq {
                                scores[key] = (0..*head)
                                    .map(|dim| {
                                        q[base + query * *head + dim] * k[base + key * *head + dim]
                                    })
                                    .sum::<f32>()
                                    * *scale;
                            }
                            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            let denominator: f32 =
                                scores.iter().map(|score| (*score - max).exp()).sum();
                            for dim in 0..*head {
                                output[base + query * *head + dim] = (0..*seq)
                                    .map(|key| {
                                        ((*scores.get(key).unwrap() - max).exp() / denominator)
                                            * v[base + key * *head + dim]
                                    })
                                    .sum();
                            }
                        }
                    }
                    output
                }
                UOpKind::MatMul { m, k, n } => {
                    let (m, k, n) = (*m, *k, *n);
                    let left = values.get(&op.src[0]).unwrap();
                    let right = values.get(&op.src[1]).unwrap();
                    let mut output = vec![0.0; m * n];
                    for row in 0..m {
                        for col in 0..n {
                            output[row * n + col] = (0..k)
                                .map(|inner| left[row * k + inner] * right[inner * n + col])
                                .sum();
                        }
                    }
                    output
                }
                UOpKind::Output { name } => {
                    let value = values.get(&op.src[0]).unwrap().clone();
                    outputs.insert(name.clone(), value.clone());
                    value
                }
            };
            values.insert(op.id, value);
        }
        Ok(outputs)
    }

    pub fn memory_plan(&self) -> Result<MemoryPlan, GraphError> {
        self.validate()?;
        let schedule = self.schedule()?;
        let position: BTreeMap<UOpId, usize> = schedule
            .iter()
            .enumerate()
            .map(|(index, id)| (*id, index))
            .collect();
        let mut last_use = BTreeMap::new();
        for op in &self.ops {
            last_use.insert(op.id, position[&op.id]);
        }
        for op in &self.ops {
            for source in &op.src {
                last_use.insert(*source, last_use[source].max(position[&op.id]));
            }
        }
        let mut active: Vec<(usize, usize)> = Vec::new();
        let mut allocations = Vec::new();
        let mut slot_count = 0;
        let mut slot_capacity: Vec<usize> = Vec::new();
        for (command, id) in schedule.iter().enumerate() {
            active.retain(|(end, _)| *end >= command);
            let op = self.ops.iter().find(|op| op.id == *id).unwrap();
            if matches!(
                op.kind,
                UOpKind::Input { .. }
                    | UOpKind::Const { .. }
                    | UOpKind::Reshape
                    | UOpKind::Output { .. }
            ) {
                continue;
            }
            let elements = op
                .shape
                .iter()
                .try_fold(1usize, |count, dimension| {
                    count.checked_mul(*dimension as usize)
                })
                .ok_or(GraphError::ShapeMismatch(*id))?;
            let slot = (0..slot_count)
                .find(|candidate| {
                    slot_capacity[*candidate] >= elements
                        && active.iter().all(|(_, assigned)| assigned != candidate)
                })
                .unwrap_or_else(|| {
                    let slot = slot_count;
                    slot_count += 1;
                    slot_capacity.push(0);
                    slot
                });
            slot_capacity[slot] = slot_capacity[slot].max(elements);
            active.push((last_use[id], slot));
            allocations.push(BufferAllocation {
                value: *id,
                slot,
                elements,
                first_command: command,
                last_command: last_use[id],
            });
        }
        Ok(MemoryPlan {
            allocations,
            slot_count,
        })
    }
}

fn is_supported_dtype(dtype: &str) -> bool {
    matches!(dtype, "f32" | "f16" | "bf16" | "i8" | "u8" | "i32" | "u32")
}

fn cast_f32(value: f32, from: &str, to: &str) -> f32 {
    debug_assert!(matches!(
        from,
        "f32" | "f16" | "bf16" | "i8" | "u8" | "i32" | "u32"
    ));
    match to {
        "f32" => value,
        "f16" => half::f16::from_f32(value).to_f32(),
        "bf16" => half::bf16::from_f32(value).to_f32(),
        "i8" => value.clamp(i8::MIN as f32, i8::MAX as f32).trunc(),
        "u8" => value.clamp(0.0, u8::MAX as f32).trunc(),
        "i32" => value.clamp(i32::MIN as f32, i32::MAX as f32).trunc(),
        "u32" => value.clamp(0.0, u32::MAX as f32).trunc(),
        _ => unreachable!("validated cast target"),
    }
}

fn element_count(shape: &[u64]) -> usize {
    shape
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim as usize))
        .unwrap_or(0)
}

/// NumPy/tinygrad-style trailing-dimension broadcasting. A dimension is
/// compatible when it is equal or one; missing leading dimensions behave as
/// one. Keeping this helper in the compact IR makes shape semantics shared by
/// validation, reference execution, and future backend index lowering.
fn broadcast_shape(left: &[u64], right: &[u64]) -> Option<Vec<u64>> {
    let rank = left.len().max(right.len());
    let mut shape = vec![1; rank];
    for (axis, output_dim) in shape.iter_mut().take(rank).enumerate() {
        let left_dim = left
            .get(left.len().wrapping_sub(rank - axis))
            .copied()
            .unwrap_or(1);
        let right_dim = right
            .get(right.len().wrapping_sub(rank - axis))
            .copied()
            .unwrap_or(1);
        if left_dim != right_dim && left_dim != 1 && right_dim != 1 {
            return None;
        }
        *output_dim = left_dim.max(right_dim);
    }
    Some(shape)
}

fn broadcast_index(index: usize, output_shape: &[u64], input_shape: &[u64]) -> usize {
    if input_shape == output_shape {
        return index;
    }
    let rank_delta = output_shape.len() - input_shape.len();
    let mut input_index = 0usize;
    let mut stride = 1usize;
    for input_axis in (0..input_shape.len()).rev() {
        let output_axis = input_axis + rank_delta;
        let output_stride = output_shape[output_axis + 1..]
            .iter()
            .map(|dim| *dim as usize)
            .product::<usize>();
        let coordinate = (index / output_stride.max(1)) % output_shape[output_axis] as usize;
        let input_coordinate = if input_shape[input_axis] == 1 {
            0
        } else {
            coordinate
        };
        input_index += input_coordinate * stride;
        stride *= input_shape[input_axis] as usize;
    }
    input_index
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoweringTarget {
    Cpu,
    Metal,
    Portable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum KernelOp {
    BroadcastBinary {
        id: UOpId,
        operation: BroadcastBinaryOperation,
        lhs_shape: Vec<u64>,
        rhs_shape: Vec<u64>,
        output_shape: Vec<u64>,
    },
    Add {
        id: UOpId,
        scalar: Option<f32>,
        scalar_left: bool,
    },
    Mul {
        id: UOpId,
        scalar: Option<f32>,
        scalar_left: bool,
    },
    Sub {
        id: UOpId,
        scalar: Option<f32>,
        scalar_left: bool,
    },
    Div {
        id: UOpId,
        scalar: Option<f32>,
        scalar_left: bool,
    },
    Maximum {
        id: UOpId,
        scalar: Option<f32>,
        scalar_left: bool,
    },
    Minimum {
        id: UOpId,
        scalar: Option<f32>,
        scalar_left: bool,
    },
    Where {
        id: UOpId,
        condition_shape: Vec<u64>,
        true_shape: Vec<u64>,
        false_shape: Vec<u64>,
        output_shape: Vec<u64>,
    },
    Relu {
        id: UOpId,
    },
    Neg {
        id: UOpId,
    },
    Exp {
        id: UOpId,
    },
    Sqrt {
        id: UOpId,
    },
    Abs {
        id: UOpId,
    },
    Log {
        id: UOpId,
    },
    Tanh {
        id: UOpId,
    },
    Sin {
        id: UOpId,
    },
    Cos {
        id: UOpId,
    },
    Gelu {
        id: UOpId,
    },
    Pow {
        id: UOpId,
        exponent: f32,
    },
    Cast {
        id: UOpId,
        from: String,
        to: String,
    },
    Transpose {
        id: UOpId,
        permutation: Vec<usize>,
        input_shape: Vec<u64>,
        output_shape: Vec<u64>,
    },
    RmsNorm {
        id: UOpId,
        rows: usize,
        features: usize,
        epsilon: f32,
    },
    LayerNorm {
        id: UOpId,
        rows: usize,
        features: usize,
        epsilon: f32,
    },
    Rope {
        id: UOpId,
        rows: usize,
        features: usize,
    },
    Gather {
        id: UOpId,
        rows: usize,
        vocab: usize,
        features: usize,
    },
    Scatter {
        id: UOpId,
        rows: usize,
        updates: usize,
        features: usize,
    },
    Ssm {
        id: UOpId,
        rows: usize,
        features: usize,
    },
    ReduceSum {
        id: UOpId,
        elements: usize,
    },
    ReduceMax {
        id: UOpId,
        elements: usize,
    },
    ReduceMin {
        id: UOpId,
        elements: usize,
    },
    ReduceSumAxis {
        id: UOpId,
        outer: usize,
        reduce: usize,
        inner: usize,
    },
    ReduceMaxAxis {
        id: UOpId,
        outer: usize,
        reduce: usize,
        inner: usize,
    },
    ReduceMinAxis {
        id: UOpId,
        outer: usize,
        reduce: usize,
        inner: usize,
    },
    SoftmaxAxis {
        id: UOpId,
        outer: usize,
        reduce: usize,
        inner: usize,
    },
    Attention {
        id: UOpId,
        seq: usize,
        head: usize,
        scale: f32,
    },
    AttentionBatched {
        id: UOpId,
        batch: usize,
        seq: usize,
        head: usize,
        scale: f32,
    },
    MatMul {
        id: UOpId,
        m: usize,
        k: usize,
        n: usize,
    },
    Conv2d {
        id: UOpId,
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BroadcastBinaryOperation {
    Add,
    Mul,
    Sub,
    Div,
    Maximum,
    Minimum,
}
impl KernelOp {
    fn id(&self) -> UOpId {
        match self {
            Self::BroadcastBinary { id, .. }
            | Self::Add { id, .. }
            | Self::Mul { id, .. }
            | Self::Sub { id, .. }
            | Self::Div { id, .. }
            | Self::Maximum { id, .. }
            | Self::Minimum { id, .. }
            | Self::Where { id, .. }
            | Self::Relu { id }
            | Self::Neg { id }
            | Self::Exp { id }
            | Self::Sqrt { id }
            | Self::Abs { id }
            | Self::Log { id }
            | Self::Tanh { id }
            | Self::Sin { id }
            | Self::Cos { id }
            | Self::Gelu { id }
            | Self::Pow { id, .. }
            | Self::Cast { id, .. }
            | Self::Transpose { id, .. }
            | Self::RmsNorm { id, .. }
            | Self::LayerNorm { id, .. } => *id,
            Self::Rope { id, .. } => *id,
            Self::Gather { id, .. } => *id,
            Self::Scatter { id, .. } => *id,
            Self::Ssm { id, .. } => *id,
            Self::Conv2d { id, .. } => *id,
            Self::ReduceSum { id, .. } => *id,
            Self::ReduceMax { id, .. } => *id,
            Self::ReduceMin { id, .. } => *id,
            Self::ReduceSumAxis { id, .. } => *id,
            Self::ReduceMaxAxis { id, .. } => *id,
            Self::ReduceMinAxis { id, .. } => *id,
            Self::SoftmaxAxis { id, .. } => *id,
            Self::Attention { id, .. } => *id,
            Self::AttentionBatched { id, .. } => *id,
            Self::MatMul { id, .. } => *id,
        }
    }
}
impl KernelOp {
    fn from_broadcast_op(op: &UOp, graph: &TinyGraph) -> Self {
        let source_shape = |index: usize| {
            graph
                .ops
                .iter()
                .find(|candidate| candidate.id == op.src[index])
                .unwrap()
                .shape
                .clone()
        };
        let operation = match op.kind {
            UOpKind::Add => BroadcastBinaryOperation::Add,
            UOpKind::Mul => BroadcastBinaryOperation::Mul,
            UOpKind::Sub => BroadcastBinaryOperation::Sub,
            UOpKind::Div => BroadcastBinaryOperation::Div,
            UOpKind::Maximum => BroadcastBinaryOperation::Maximum,
            UOpKind::Minimum => BroadcastBinaryOperation::Minimum,
            _ => unreachable!("broadcast lowering only accepts binary elementwise ops"),
        };
        Self::BroadcastBinary {
            id: op.id,
            operation,
            lhs_shape: source_shape(0),
            rhs_shape: source_shape(1),
            output_shape: op.shape.clone(),
        }
    }

    fn from_graph_op(op: &UOp, graph: &TinyGraph) -> Self {
        match op.kind {
            UOpKind::Add => Self::Add {
                id: op.id,
                scalar: scalar_operand(op, graph),
                scalar_left: scalar_is_left(op, graph),
            },
            UOpKind::Mul => Self::Mul {
                id: op.id,
                scalar: scalar_operand(op, graph),
                scalar_left: scalar_is_left(op, graph),
            },
            UOpKind::Sub => Self::Sub {
                id: op.id,
                scalar: scalar_operand(op, graph),
                scalar_left: scalar_is_left(op, graph),
            },
            UOpKind::Div => Self::Div {
                id: op.id,
                scalar: scalar_operand(op, graph),
                scalar_left: scalar_is_left(op, graph),
            },
            UOpKind::Maximum => Self::Maximum {
                id: op.id,
                scalar: scalar_operand(op, graph),
                scalar_left: scalar_is_left(op, graph),
            },
            UOpKind::Minimum => Self::Minimum {
                id: op.id,
                scalar: scalar_operand(op, graph),
                scalar_left: scalar_is_left(op, graph),
            },
            UOpKind::Where => {
                let shape = |index: usize| {
                    graph
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[index])
                        .unwrap()
                        .shape
                        .clone()
                };
                Self::Where {
                    id: op.id,
                    condition_shape: shape(0),
                    true_shape: shape(1),
                    false_shape: shape(2),
                    output_shape: op.shape.clone(),
                }
            }
            UOpKind::Relu => Self::Relu { id: op.id },
            UOpKind::Neg => Self::Neg { id: op.id },
            UOpKind::Exp => Self::Exp { id: op.id },
            UOpKind::Sqrt => Self::Sqrt { id: op.id },
            UOpKind::Abs => Self::Abs { id: op.id },
            UOpKind::Log => Self::Log { id: op.id },
            UOpKind::Tanh => Self::Tanh { id: op.id },
            UOpKind::Sin => Self::Sin { id: op.id },
            UOpKind::Cos => Self::Cos { id: op.id },
            UOpKind::Gelu => Self::Gelu { id: op.id },
            UOpKind::Pow { exponent } => Self::Pow {
                id: op.id,
                exponent,
            },
            UOpKind::Cast { ref from, ref to } => Self::Cast {
                id: op.id,
                from: from.clone(),
                to: to.clone(),
            },
            UOpKind::Transpose { ref permutation } => {
                let input = graph
                    .ops
                    .iter()
                    .find(|candidate| candidate.id == op.src[0])
                    .unwrap();
                Self::Transpose {
                    id: op.id,
                    permutation: permutation.clone(),
                    input_shape: input.shape.clone(),
                    output_shape: op.shape.clone(),
                }
            }
            UOpKind::RmsNorm {
                rows,
                features,
                epsilon,
            } => Self::RmsNorm {
                id: op.id,
                rows,
                features,
                epsilon,
            },
            UOpKind::LayerNorm {
                rows,
                features,
                epsilon,
            } => Self::LayerNorm {
                id: op.id,
                rows,
                features,
                epsilon,
            },
            UOpKind::Rope { rows, features } => Self::Rope {
                id: op.id,
                rows,
                features,
            },
            UOpKind::Gather {
                rows,
                vocab,
                features,
            } => Self::Gather {
                id: op.id,
                rows,
                vocab,
                features,
            },
            UOpKind::Scatter {
                rows,
                updates,
                features,
            } => Self::Scatter {
                id: op.id,
                rows,
                updates,
                features,
            },
            UOpKind::Ssm { rows, features } => Self::Ssm {
                id: op.id,
                rows,
                features,
            },
            UOpKind::ReduceSum => Self::ReduceSum {
                id: op.id,
                elements: element_count(
                    &graph
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[0])
                        .expect("validated reduction source")
                        .shape,
                ),
            },
            UOpKind::ReduceMax => Self::ReduceMax {
                id: op.id,
                elements: element_count(
                    &graph
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[0])
                        .expect("validated reduction source")
                        .shape,
                ),
            },
            UOpKind::ReduceMin => Self::ReduceMin {
                id: op.id,
                elements: element_count(
                    &graph
                        .ops
                        .iter()
                        .find(|candidate| candidate.id == op.src[0])
                        .expect("validated reduction source")
                        .shape,
                ),
            },
            UOpKind::ReduceSumAxis { axis } => {
                let source = graph
                    .ops
                    .iter()
                    .find(|candidate| candidate.id == op.src[0])
                    .expect("validated reduction source");
                Self::ReduceSumAxis {
                    id: op.id,
                    outer: source.shape[..axis]
                        .iter()
                        .map(|dim| *dim as usize)
                        .product(),
                    reduce: source.shape[axis] as usize,
                    inner: source.shape[axis + 1..]
                        .iter()
                        .map(|dim| *dim as usize)
                        .product(),
                }
            }
            UOpKind::ReduceMaxAxis { axis } => {
                let source = graph
                    .ops
                    .iter()
                    .find(|candidate| candidate.id == op.src[0])
                    .expect("validated reduction source");
                Self::ReduceMaxAxis {
                    id: op.id,
                    outer: source.shape[..axis]
                        .iter()
                        .map(|dim| *dim as usize)
                        .product(),
                    reduce: source.shape[axis] as usize,
                    inner: source.shape[axis + 1..]
                        .iter()
                        .map(|dim| *dim as usize)
                        .product(),
                }
            }
            UOpKind::ReduceMinAxis { axis } => {
                let source = graph
                    .ops
                    .iter()
                    .find(|candidate| candidate.id == op.src[0])
                    .expect("validated reduction source");
                Self::ReduceMinAxis {
                    id: op.id,
                    outer: source.shape[..axis]
                        .iter()
                        .map(|dim| *dim as usize)
                        .product(),
                    reduce: source.shape[axis] as usize,
                    inner: source.shape[axis + 1..]
                        .iter()
                        .map(|dim| *dim as usize)
                        .product(),
                }
            }
            UOpKind::SoftmaxAxis { axis } => {
                let source = graph
                    .ops
                    .iter()
                    .find(|candidate| candidate.id == op.src[0])
                    .expect("validated softmax source");
                Self::SoftmaxAxis {
                    id: op.id,
                    outer: source.shape[..axis]
                        .iter()
                        .map(|dim| *dim as usize)
                        .product(),
                    reduce: source.shape[axis] as usize,
                    inner: source.shape[axis + 1..]
                        .iter()
                        .map(|dim| *dim as usize)
                        .product(),
                }
            }
            UOpKind::Attention { seq, head, scale } => Self::Attention {
                id: op.id,
                seq,
                head,
                scale,
            },
            UOpKind::AttentionBatched {
                batch,
                seq,
                head,
                scale,
            } => Self::AttentionBatched {
                id: op.id,
                batch,
                seq,
                head,
                scale,
            },
            UOpKind::MatMul { m, k, n } => Self::MatMul { id: op.id, m, k, n },
            UOpKind::Conv2d {
                batch,
                in_channels,
                height,
                width,
                out_channels,
                kernel_h,
                kernel_w,
                stride,
                padding,
            } => Self::Conv2d {
                id: op.id,
                batch,
                in_channels,
                height,
                width,
                out_channels,
                kernel_h,
                kernel_w,
                stride,
                padding,
            },
            _ => unreachable!(),
        }
    }
}

fn scalar_operand(op: &UOp, graph: &TinyGraph) -> Option<f32> {
    op.src.iter().find_map(|source| {
        graph
            .ops
            .iter()
            .find(|candidate| candidate.id == *source)
            .and_then(|candidate| match candidate.kind {
                UOpKind::Const { value } => Some(value),
                _ => None,
            })
    })
}

fn scalar_is_left(op: &UOp, graph: &TinyGraph) -> bool {
    op.src
        .first()
        .and_then(|source| graph.ops.iter().find(|candidate| candidate.id == *source))
        .is_some_and(|candidate| matches!(candidate.kind, UOpKind::Const { .. }))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelGroup {
    pub ops: Vec<KernelOp>,
}

type BroadcastBinaryShape = (BroadcastBinaryOperation, Vec<u64>, Vec<u64>, Vec<u64>);

type Convolution2dShape = (
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
);

impl KernelGroup {
    fn ops_after_broadcast(&self) -> &[KernelOp] {
        self.ops.get(1..).unwrap_or(&[])
    }

    pub fn broadcast_binary_shape(&self) -> Option<BroadcastBinaryShape> {
        match self.ops.first() {
            Some(KernelOp::BroadcastBinary {
                operation,
                lhs_shape,
                rhs_shape,
                output_shape,
                ..
            }) => Some((
                *operation,
                lhs_shape.clone(),
                rhs_shape.clone(),
                output_shape.clone(),
            )),
            _ => None,
        }
    }

    pub fn broadcast_program(&self) -> Option<String> {
        self.broadcast_binary_shape()?;
        let mut program = Vec::new();
        for op in self.ops_after_broadcast() {
            program.push(match op {
                KernelOp::Relu { .. } => "relu",
                KernelOp::Neg { .. } => "neg",
                KernelOp::Exp { .. } => "exp",
                KernelOp::Sqrt { .. } => "sqrt",
                KernelOp::Abs { .. } => "abs",
                KernelOp::Log { .. } => "log",
                KernelOp::Tanh { .. } => "tanh",
                KernelOp::Sin { .. } => "sin",
                KernelOp::Cos { .. } => "cos",
                KernelOp::Gelu { .. } => "gelu",
                _ => return None,
            });
        }
        Some(program.join(","))
    }

    pub fn transpose_shape(&self) -> Option<(Vec<usize>, Vec<u64>, Vec<u64>)> {
        match self.ops.as_slice() {
            [KernelOp::Transpose {
                permutation,
                input_shape,
                output_shape,
                ..
            }] => Some((
                permutation.clone(),
                input_shape.clone(),
                output_shape.clone(),
            )),
            _ => None,
        }
    }

    pub fn elementwise_program(&self) -> Option<String> {
        let mut program = Vec::with_capacity(self.ops.len());
        for op in &self.ops {
            let name = match op {
                KernelOp::Add { scalar: None, .. } => "add",
                KernelOp::Mul { scalar: None, .. } => "mul",
                KernelOp::Sub { scalar: None, .. } => "sub",
                KernelOp::Div { scalar: None, .. } => "div",
                KernelOp::Maximum { scalar: None, .. } => "maximum",
                KernelOp::Minimum { scalar: None, .. } => "minimum",
                KernelOp::Relu { .. } => "relu",
                KernelOp::Neg { .. } => "neg",
                KernelOp::Exp { .. } => "exp",
                KernelOp::Sqrt { .. } => "sqrt",
                KernelOp::Abs { .. } => "abs",
                KernelOp::Log { .. } => "log",
                KernelOp::Tanh { .. } => "tanh",
                KernelOp::Sin { .. } => "sin",
                KernelOp::Cos { .. } => "cos",
                KernelOp::Gelu { .. } => "gelu",
                _ => return None,
            };
            program.push(name);
        }
        (!program.is_empty()).then(|| program.join(","))
    }

    pub fn op_ids(&self) -> Vec<UOpId> {
        self.ops.iter().map(KernelOp::id).collect()
    }

    pub fn unary_elementwise_program(&self) -> Option<String> {
        let mut program = Vec::with_capacity(self.ops.len());
        for op in &self.ops {
            let name = match op {
                KernelOp::Relu { .. } => "relu",
                KernelOp::Neg { .. } => "neg",
                KernelOp::Exp { .. } => "exp",
                KernelOp::Sqrt { .. } => "sqrt",
                KernelOp::Abs { .. } => "abs",
                KernelOp::Log { .. } => "log",
                KernelOp::Tanh { .. } => "tanh",
                KernelOp::Sin { .. } => "sin",
                KernelOp::Cos { .. } => "cos",
                KernelOp::Gelu { .. } => "gelu",
                _ => return None,
            };
            program.push(name);
        }
        (!program.is_empty()).then(|| program.join(","))
    }

    pub fn elementwise_variant(&self) -> Option<&'static str> {
        match self.ops.as_slice() {
            [KernelOp::Relu { .. }] => Some("relu"),
            [KernelOp::Neg { .. }] => Some("neg"),
            [KernelOp::Exp { .. }] => Some("exp"),
            [KernelOp::Sqrt { .. }] => Some("sqrt"),
            [KernelOp::Abs { .. }] => Some("abs"),
            [KernelOp::Log { .. }] => Some("log"),
            [KernelOp::Tanh { .. }] => Some("tanh"),
            [KernelOp::Sin { .. }] => Some("sin"),
            [KernelOp::Cos { .. }] => Some("cos"),
            [KernelOp::Gelu { .. }] => Some("gelu"),
            _ => None,
        }
    }

    pub fn binary_elementwise_variant(&self) -> Option<&'static str> {
        match self.ops.as_slice() {
            [KernelOp::Add { scalar: None, .. }] => Some("add"),
            [KernelOp::Mul { scalar: None, .. }] => Some("mul"),
            [KernelOp::Sub { scalar: None, .. }] => Some("sub"),
            [KernelOp::Div { scalar: None, .. }] => Some("div"),
            [KernelOp::Maximum { scalar: None, .. }] => Some("maximum"),
            [KernelOp::Minimum { scalar: None, .. }] => Some("minimum"),
            _ => None,
        }
    }

    pub fn scalar_elementwise_variant(&self) -> Option<String> {
        let (operation, scalar, scalar_left) = match self.ops.as_slice() {
            [KernelOp::Add {
                scalar: Some(value),
                scalar_left,
                ..
            }] => ("add", *value, *scalar_left),
            [KernelOp::Mul {
                scalar: Some(value),
                scalar_left,
                ..
            }] => ("mul", *value, *scalar_left),
            [KernelOp::Sub {
                scalar: Some(value),
                scalar_left,
                ..
            }] => ("sub", *value, *scalar_left),
            [KernelOp::Div {
                scalar: Some(value),
                scalar_left,
                ..
            }] => ("div", *value, *scalar_left),
            [KernelOp::Maximum {
                scalar: Some(value),
                scalar_left,
                ..
            }] => ("maximum", *value, *scalar_left),
            [KernelOp::Minimum {
                scalar: Some(value),
                scalar_left,
                ..
            }] => ("minimum", *value, *scalar_left),
            _ => return None,
        };
        Some(format!(
            "{operation}|{scalar}|{}",
            if scalar_left { 1 } else { 0 }
        ))
    }

    fn reduction(&self) -> Option<usize> {
        match self.ops.as_slice() {
            [KernelOp::ReduceSum { elements, .. }] => Some(*elements),
            _ => None,
        }
    }

    pub fn max_reduction(&self) -> Option<usize> {
        match self.ops.as_slice() {
            [KernelOp::ReduceMax { elements, .. }] => Some(*elements),
            _ => None,
        }
    }

    pub fn min_reduction(&self) -> Option<usize> {
        match self.ops.as_slice() {
            [KernelOp::ReduceMin { elements, .. }] => Some(*elements),
            _ => None,
        }
    }

    pub fn axis_reduction(&self) -> Option<(usize, usize, usize)> {
        match self.ops.as_slice() {
            [KernelOp::ReduceSumAxis {
                outer,
                reduce,
                inner,
                ..
            }] => Some((*outer, *reduce, *inner)),
            _ => None,
        }
    }

    pub fn max_axis_reduction(&self) -> Option<(usize, usize, usize)> {
        match self.ops.as_slice() {
            [KernelOp::ReduceMaxAxis {
                outer,
                reduce,
                inner,
                ..
            }] => Some((*outer, *reduce, *inner)),
            _ => None,
        }
    }

    pub fn min_axis_reduction(&self) -> Option<(usize, usize, usize)> {
        match self.ops.as_slice() {
            [KernelOp::ReduceMinAxis {
                outer,
                reduce,
                inner,
                ..
            }] => Some((*outer, *reduce, *inner)),
            _ => None,
        }
    }

    pub fn softmax_shape(&self) -> Option<(usize, usize, usize)> {
        match self.ops.as_slice() {
            [KernelOp::SoftmaxAxis {
                outer,
                reduce,
                inner,
                ..
            }] => Some((*outer, *reduce, *inner)),
            _ => None,
        }
    }

    pub fn attention_shape(&self) -> Option<(usize, usize, f32)> {
        match self.ops.as_slice() {
            [KernelOp::Attention {
                seq, head, scale, ..
            }] => Some((*seq, *head, *scale)),
            _ => None,
        }
    }

    pub fn batched_attention_shape(&self) -> Option<(usize, usize, usize, f32)> {
        match self.ops.as_slice() {
            [KernelOp::AttentionBatched {
                batch,
                seq,
                head,
                scale,
                ..
            }] => Some((*batch, *seq, *head, *scale)),
            _ => None,
        }
    }

    pub fn rms_norm_shape(&self) -> Option<(usize, usize, f32)> {
        match self.ops.as_slice() {
            [KernelOp::RmsNorm {
                rows,
                features,
                epsilon,
                ..
            }] => Some((*rows, *features, *epsilon)),
            _ => None,
        }
    }

    pub fn layer_norm_shape(&self) -> Option<(usize, usize, f32)> {
        match self.ops.as_slice() {
            [KernelOp::LayerNorm {
                rows,
                features,
                epsilon,
                ..
            }] => Some((*rows, *features, *epsilon)),
            _ => None,
        }
    }

    pub fn rope_shape(&self) -> Option<(usize, usize)> {
        match self.ops.as_slice() {
            [KernelOp::Rope { rows, features, .. }] => Some((*rows, *features)),
            _ => None,
        }
    }

    pub fn gather_shape(&self) -> Option<(usize, usize, usize)> {
        match self.ops.as_slice() {
            [KernelOp::Gather {
                rows,
                vocab,
                features,
                ..
            }] => Some((*rows, *vocab, *features)),
            _ => None,
        }
    }

    pub fn scatter_shape(&self) -> Option<(usize, usize, usize)> {
        match self.ops.as_slice() {
            [KernelOp::Scatter {
                rows,
                updates,
                features,
                ..
            }] => Some((*rows, *updates, *features)),
            _ => None,
        }
    }

    pub fn ssm_shape(&self) -> Option<(usize, usize)> {
        match self.ops.as_slice() {
            [KernelOp::Ssm { rows, features, .. }] => Some((*rows, *features)),
            _ => None,
        }
    }

    pub fn is_reduction(&self) -> bool {
        self.reduction().is_some()
            || self.max_reduction().is_some()
            || self.min_reduction().is_some()
            || self.axis_reduction().is_some()
            || self.max_axis_reduction().is_some()
            || self.min_axis_reduction().is_some()
    }

    pub fn input_elements(&self) -> Option<usize> {
        self.reduction()
            .or_else(|| self.max_reduction())
            .or_else(|| self.min_reduction())
            .or_else(|| {
                self.axis_reduction()
                    .or_else(|| self.max_axis_reduction())
                    .or_else(|| self.min_axis_reduction())
                    .map(|(outer, reduce, inner)| outer * reduce * inner)
            })
    }

    pub fn matmul_shape(&self) -> Option<(usize, usize, usize)> {
        match self.ops.as_slice() {
            [KernelOp::MatMul { m, k, n, .. }] => Some((*m, *k, *n)),
            _ => None,
        }
    }

    pub fn conv2d_shape(&self) -> Option<Convolution2dShape> {
        match self.ops.as_slice() {
            [KernelOp::Conv2d {
                batch,
                in_channels,
                height,
                width,
                out_channels,
                kernel_h,
                kernel_w,
                stride,
                padding,
                ..
            }] => Some((
                *batch,
                *in_channels,
                *height,
                *width,
                *out_channels,
                *kernel_h,
                *kernel_w,
                *stride,
                *padding,
            )),
            _ => None,
        }
    }

    pub fn requires_rhs(&self) -> bool {
        self.ops.iter().any(|op| {
            matches!(
                op,
                KernelOp::BroadcastBinary { .. }
                    | KernelOp::Add { scalar: None, .. }
                    | KernelOp::Mul { scalar: None, .. }
                    | KernelOp::Sub { scalar: None, .. }
                    | KernelOp::Div { scalar: None, .. }
                    | KernelOp::Maximum { scalar: None, .. }
                    | KernelOp::Minimum { scalar: None, .. }
            )
        })
    }

    pub fn requires_tertiary_input(&self) -> bool {
        self.ops
            .iter()
            .any(|op| matches!(op, KernelOp::Where { .. }))
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoweredKernel {
    pub group: KernelGroup,
    pub source: String,
    /// Exact output element count retained for generic elementwise ABI
    /// construction. Older serialized captures may omit it.
    #[serde(default)]
    pub output_elements: Option<usize>,
    /// Digest of deterministic target source, suitable for compiler provenance.
    pub source_digest: String,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CapturePlan {
    pub target: LoweringTarget,
    pub kernels: Vec<LoweredKernel>,
    pub graph_op_count: usize,
    pub replay: ReplayPlan,
    pub graph: TinyGraph,
    pub memory_plan: MemoryPlan,
}

/// TinyJIT-style capture cache. The first invocation lowers and validates a
/// graph; subsequent invocations with the same graph digest and target reuse
/// the immutable command sequence and kernel payloads.
#[derive(Debug, Default, Clone, PartialEq, Serialize, Deserialize)]
pub struct TinyJitCache {
    captures: BTreeMap<String, CapturePlan>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TinyJitArchive {
    version: u32,
    captures: BTreeMap<String, TinyJitArchiveEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TinyJitArchiveEntry {
    capture: CapturePlan,
    identity_digest: String,
}

impl TinyJitCache {
    fn key(
        graph: &TinyGraph,
        target: LoweringTarget,
        strategy: Option<&crate::fused_ops::FusionStrategy>,
    ) -> Result<String, GraphError> {
        let optimized = graph.optimize()?;
        let bytes = serde_json::to_vec(&(target, strategy, optimized))
            .map_err(|error| GraphError::Serialization(error.to_string()))?;
        let mut digest = Sha256::new();
        digest.update(bytes);
        Ok(hex_digest(digest.finalize()))
    }

    pub fn capture(
        &mut self,
        graph: &TinyGraph,
        target: LoweringTarget,
    ) -> Result<(String, bool), GraphError> {
        self.capture_with_strategy(
            graph,
            target,
            &crate::fused_ops::FusionStrategy::StandardFused,
        )
    }

    /// Capture and cache a strategy-specific executable layout. Strategy is
    /// part of the identity so a per-operation or interleaved capture can
    /// never reuse the standard fused command sequence.
    pub fn capture_with_strategy(
        &mut self,
        graph: &TinyGraph,
        target: LoweringTarget,
        strategy: &crate::fused_ops::FusionStrategy,
    ) -> Result<(String, bool), GraphError> {
        let key = Self::key(graph, target, Some(strategy))?;
        if !self.captures.contains_key(&key) {
            self.captures.insert(
                key.clone(),
                graph.lower_with_fusion_strategy(target, strategy)?,
            );
            Ok((key, true))
        } else {
            Ok((key, false))
        }
    }

    /// Materialize a set of strategy alternatives in the same cache. The
    /// returned entries preserve caller order and include the cache-hit bit,
    /// allowing a workload calibrator to compile each executable alternative
    /// once and then benchmark or replay it without rebuilding the graph.
    pub fn capture_strategies(
        &mut self,
        graph: &TinyGraph,
        target: LoweringTarget,
        strategies: &[crate::fused_ops::FusionStrategy],
    ) -> Result<Vec<(crate::fused_ops::FusionStrategy, String, bool)>, GraphError> {
        for (index, strategy) in strategies.iter().enumerate() {
            if strategies[..index].contains(strategy) {
                return Err(GraphError::Serialization(format!(
                    "duplicate fusion strategy at index {index}: {strategy:?}"
                )));
            }
        }
        let mut captures = Vec::with_capacity(strategies.len());
        for strategy in strategies {
            let (key, inserted) = self.capture_with_strategy(graph, target, strategy)?;
            captures.push((strategy.clone(), key, inserted));
        }
        Ok(captures)
    }

    /// Serialize the immutable TinyJIT command cache for durable artifact
    /// storage. Captures are validated before encoding so malformed state
    /// cannot be published as a reusable executable cache.
    pub fn export_bytes(&self) -> Result<Vec<u8>, GraphError> {
        let mut captures = BTreeMap::new();
        for (key, capture) in &self.captures {
            capture
                .validate()
                .map_err(|error| GraphError::Serialization(format!("capture '{key}': {error}")))?;
            let bytes = serde_json::to_vec(&(key, capture))
                .map_err(|error| GraphError::Serialization(error.to_string()))?;
            let mut digest = Sha256::new();
            digest.update(bytes);
            captures.insert(
                key.clone(),
                TinyJitArchiveEntry {
                    capture: capture.clone(),
                    identity_digest: hex_digest(digest.finalize()),
                },
            );
        }
        serde_json::to_vec(&TinyJitArchive {
            version: 1,
            captures,
        })
        .map_err(|error| GraphError::Serialization(error.to_string()))
    }

    /// Import a persisted TinyJIT cache and re-run capture validation before
    /// making any entry available for replay.
    pub fn import_bytes(bytes: &[u8]) -> Result<Self, GraphError> {
        let archive: TinyJitArchive = serde_json::from_slice(bytes)
            .map_err(|error| GraphError::Serialization(error.to_string()))?;
        if archive.version != 1 {
            return Err(GraphError::Serialization(format!(
                "unsupported TinyJIT archive version {}",
                archive.version
            )));
        }
        let mut captures = BTreeMap::new();
        for (key, entry) in archive.captures {
            let bytes = serde_json::to_vec(&(key.clone(), &entry.capture))
                .map_err(|error| GraphError::Serialization(error.to_string()))?;
            let mut digest = Sha256::new();
            digest.update(bytes);
            if hex_digest(digest.finalize()) != entry.identity_digest {
                return Err(GraphError::Serialization(format!(
                    "TinyJIT capture '{key}' identity digest mismatch"
                )));
            }
            let capture = entry.capture;
            capture
                .validate()
                .map_err(|error| GraphError::Serialization(format!("capture '{key}': {error}")))?;
            captures.insert(key, capture);
        }
        Ok(Self { captures })
    }

    pub fn get(&self, key: &str) -> Option<&CapturePlan> {
        self.captures.get(key)
    }

    /// Evict one captured command sequence so a changed lowering policy or
    /// target capability cannot reuse stale TinyJIT state.
    pub fn invalidate(&mut self, key: &str) -> bool {
        self.captures.remove(key).is_some()
    }

    /// Evict every captured command sequence while preserving the cache
    /// object for reuse by a long-lived compiler session.
    pub fn clear(&mut self) {
        self.captures.clear();
    }

    pub fn replay<E: CaptureExecutor>(
        &self,
        key: &str,
        executor: &mut E,
    ) -> Result<ExecutionReceipt, String> {
        self.captures
            .get(key)
            .ok_or_else(|| format!("TinyJIT capture '{key}' is not cached"))?
            .replay(executor)
    }

    pub fn len(&self) -> usize {
        self.captures.len()
    }

    pub fn is_empty(&self) -> bool {
        self.captures.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BufferAllocation {
    pub value: UOpId,
    pub slot: usize,
    /// Minimum number of f32 elements required by this value.
    #[serde(default)]
    pub elements: usize,
    pub first_command: usize,
    pub last_command: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryPlan {
    pub allocations: Vec<BufferAllocation>,
    pub slot_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayPlan {
    pub command_ids: Vec<u32>,
    pub synchronization_points: Vec<u32>,
    /// Whether the command sequence is intended for persistent replay.
    /// Persistent executors can submit the complete sequence in one call;
    /// the default hook below preserves correctness for simpler executors.
    #[serde(default)]
    pub persistent: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionReceipt {
    pub target: LoweringTarget,
    /// Digest of the complete validated capture, including graph, memory,
    /// replay, and kernel metadata.
    #[serde(default)]
    pub capture_digest: String,
    pub command_ids: Vec<u32>,
    pub kernel_digests: Vec<String>,
    #[serde(default)]
    pub persistent: bool,
    pub replayed: bool,
}

pub trait CaptureExecutor {
    fn dispatch(&mut self, command_id: u32, kernel: &LoweredKernel) -> Result<(), String>;
    fn synchronize(&mut self, command_id: u32) -> Result<(), String>;

    fn dispatch_persistent(
        &mut self,
        command_ids: &[u32],
        kernels: &[LoweredKernel],
    ) -> Result<(), String> {
        for (command_id, kernel) in command_ids.iter().zip(kernels) {
            self.dispatch(*command_id, kernel)?;
        }
        Ok(())
    }

    /// Submit a persistent command sequence together with its barrier points.
    /// Device executors that can encode barriers inside a command buffer may
    /// override this hook; the compatibility implementation only supports a
    /// final barrier and rejects interior points before submission.
    fn dispatch_persistent_with_sync_points(
        &mut self,
        command_ids: &[u32],
        kernels: &[LoweredKernel],
        synchronization_points: &[u32],
    ) -> Result<(), String> {
        if let Some(last_command) = command_ids.last() {
            if synchronization_points
                .iter()
                .any(|point| point != last_command)
            {
                return Err(
                    "persistent executor does not support interior synchronization points".into(),
                );
            }
        }
        self.dispatch_persistent(command_ids, kernels)?;
        if let Some(last_command) = command_ids.last() {
            if synchronization_points.contains(last_command) {
                self.synchronize(*last_command)?;
            }
        }
        Ok(())
    }
}

impl CapturePlan {
    pub fn validate(&self) -> Result<(), String> {
        self.graph
            .validate()
            .map_err(|error| format!("capture graph is invalid: {error}"))?;
        if self.graph_op_count != self.graph.ops.len() {
            return Err(format!(
                "capture graph operation count mismatch: recorded {}, embedded {}",
                self.graph_op_count,
                self.graph.ops.len()
            ));
        }
        if self.replay.command_ids.len() != self.kernels.len() {
            return Err("capture command count does not match kernel count".into());
        }
        let expected_command_ids: Vec<u32> = (0..self.kernels.len() as u32).collect();
        if self.replay.command_ids != expected_command_ids {
            return Err("capture command IDs are not canonical".into());
        }
        let graph_ids: BTreeSet<UOpId> = self.graph.ops.iter().map(|op| op.id).collect();
        for kernel in &self.kernels {
            if kernel.group.ops.is_empty() {
                return Err("capture contains an empty kernel group".into());
            }
            if kernel.source.is_empty() {
                return Err(format!(
                    "capture kernel {:?} has empty rendered source",
                    kernel.group.op_ids()
                ));
            }
            if kernel
                .group
                .op_ids()
                .iter()
                .any(|id| !graph_ids.contains(id))
            {
                return Err("capture kernel references a missing graph UOp".into());
            }
            if let Some(recorded_elements) = kernel.output_elements {
                let output_id = kernel
                    .group
                    .ops
                    .last()
                    .map(KernelOp::id)
                    .ok_or_else(|| "capture kernel has no terminal UOp".to_string())?;
                let expected_elements = self
                    .graph
                    .ops
                    .iter()
                    .find(|op| op.id == output_id)
                    .and_then(|op| {
                        op.shape.iter().try_fold(1usize, |count, dimension| {
                            count.checked_mul(*dimension as usize)
                        })
                    })
                    .ok_or_else(|| "capture terminal UOp has invalid shape".to_string())?;
                if recorded_elements != expected_elements {
                    return Err(format!(
                        "capture kernel output geometry mismatch: recorded {recorded_elements}, expected {expected_elements}"
                    ));
                }
            }
            let mut digest = Sha256::new();
            digest.update(kernel.source.as_bytes());
            if kernel.source_digest != hex_digest(digest.finalize()) {
                return Err(format!(
                    "kernel {:?} source digest does not match source",
                    kernel.group.op_ids()
                ));
            }
        }
        if self
            .replay
            .command_ids
            .windows(2)
            .any(|ids| ids[0] >= ids[1])
        {
            return Err("capture command ids must be strictly increasing".into());
        }
        if self.memory_plan.slot_count
            < self
                .memory_plan
                .allocations
                .iter()
                .map(|allocation| allocation.slot)
                .max()
                .map_or(0, |slot| slot + 1)
        {
            return Err("capture memory plan references an unavailable slot".into());
        }
        if self
            .memory_plan
            .allocations
            .iter()
            .any(|allocation| allocation.first_command > allocation.last_command)
        {
            return Err("capture memory allocation has an invalid lifetime".into());
        }
        let mut allocation_ids = BTreeSet::new();
        for allocation in &self.memory_plan.allocations {
            if !allocation_ids.insert(allocation.value) {
                return Err(format!(
                    "capture memory plan allocates UOp {:?} more than once",
                    allocation.value
                ));
            }
            let value = self
                .graph
                .ops
                .iter()
                .find(|op| op.id == allocation.value)
                .ok_or_else(|| {
                    format!(
                        "capture memory plan references missing UOp {:?}",
                        allocation.value
                    )
                })?;
            let expected_elements = value
                .shape
                .iter()
                .try_fold(1usize, |count, dimension| {
                    count.checked_mul(*dimension as usize)
                })
                .ok_or_else(|| "capture memory value shape overflows element count".to_string())?;
            if allocation.elements != expected_elements {
                return Err(format!(
                    "capture memory allocation for {:?} records {} elements; expected {}",
                    allocation.value, allocation.elements, expected_elements
                ));
            }
        }
        if self
            .replay
            .synchronization_points
            .iter()
            .any(|point| !self.replay.command_ids.contains(point))
        {
            return Err("capture synchronization point references an unknown command".into());
        }
        if self
            .replay
            .synchronization_points
            .windows(2)
            .any(|points| points[0] >= points[1])
        {
            return Err("capture synchronization points are not canonical".into());
        }
        Ok(())
    }

    pub fn digest(&self) -> String {
        let bytes = serde_json::to_vec(self).expect("CapturePlan must be serializable");
        let mut digest = Sha256::new();
        digest.update(bytes);
        hex_digest(digest.finalize())
    }

    /// Verify that an execution receipt still describes this exact capture.
    /// Receipts are transportable evidence, so callers must validate them
    /// before using them for promotion, calibration, or artifact provenance.
    pub fn validate_receipt(&self, receipt: &ExecutionReceipt) -> Result<(), String> {
        self.validate()
            .map_err(|error| format!("capture: {error}"))?;
        if !receipt.replayed {
            return Err("execution receipt is not marked as replayed".into());
        }
        if receipt.target != self.target {
            return Err("execution receipt target does not match capture".into());
        }
        if receipt.capture_digest != self.digest() {
            return Err("execution receipt capture digest does not match capture".into());
        }
        if receipt.command_ids != self.replay.command_ids {
            return Err("execution receipt command sequence does not match capture".into());
        }
        if receipt.persistent != self.replay.persistent {
            return Err("execution receipt persistence mode does not match capture".into());
        }
        let expected_kernel_digests = self
            .kernels
            .iter()
            .map(|kernel| kernel.source_digest.clone())
            .collect::<Vec<_>>();
        if receipt.kernel_digests != expected_kernel_digests {
            return Err("execution receipt kernel digests do not match capture".into());
        }
        Ok(())
    }

    pub fn replay<E: CaptureExecutor>(&self, executor: &mut E) -> Result<ExecutionReceipt, String> {
        self.validate()?;
        if self.replay.persistent {
            executor.dispatch_persistent_with_sync_points(
                &self.replay.command_ids,
                &self.kernels,
                &self.replay.synchronization_points,
            )?;
        } else {
            for (command_id, kernel) in self.replay.command_ids.iter().zip(&self.kernels) {
                executor.dispatch(*command_id, kernel)?;
                if self.replay.synchronization_points.contains(command_id) {
                    executor.synchronize(*command_id)?;
                }
            }
        }
        Ok(ExecutionReceipt {
            target: self.target,
            capture_digest: self.digest(),
            command_ids: self.replay.command_ids.clone(),
            kernel_digests: self
                .kernels
                .iter()
                .map(|kernel| kernel.source_digest.clone())
                .collect(),
            persistent: self.replay.persistent,
            replayed: true,
        })
    }
}

fn render_broadcast_index(
    input_shape: &[u64],
    output_shape: &[u64],
    prefix: &str,
) -> (String, String) {
    let rank_delta = output_shape.len() - input_shape.len();
    let mut declarations = String::new();
    for axis in 0..output_shape.len() {
        declarations.push_str(&format!(
            "unsigned {prefix}c{axis} = (id / {}u) % {}u; ",
            output_shape[axis + 1..].iter().product::<u64>().max(1),
            output_shape[axis]
        ));
    }
    let mut expression = String::from("0u");
    for axis in 0..input_shape.len() {
        let output_axis = axis + rank_delta;
        if input_shape[axis] != 1 {
            let stride = input_shape[axis + 1..].iter().product::<u64>().max(1);
            expression.push_str(&format!(" + {prefix}c{output_axis} * {stride}u"));
        }
    }
    (declarations, expression)
}

fn render_kernel(group: &KernelGroup, target: LoweringTarget) -> (String, String) {
    if let Some((operation, lhs_shape, rhs_shape, output_shape)) = group.broadcast_binary_shape() {
        let (lhs_declarations, lhs_index) = render_broadcast_index(&lhs_shape, &output_shape, "l");
        let (rhs_declarations, rhs_index) = render_broadcast_index(&rhs_shape, &output_shape, "r");
        let expression = match operation {
            BroadcastBinaryOperation::Add => "lhs_value + rhs_value",
            BroadcastBinaryOperation::Mul => "lhs_value * rhs_value",
            BroadcastBinaryOperation::Sub => "lhs_value - rhs_value",
            BroadcastBinaryOperation::Div => "lhs_value / rhs_value",
            BroadcastBinaryOperation::Maximum => "max(lhs_value, rhs_value)",
            BroadcastBinaryOperation::Minimum => "min(lhs_value, rhs_value)",
        };
        let mut postlude = String::new();
        for op in group.ops_after_broadcast() {
            postlude.push_str(match op {
                KernelOp::Relu { .. } => " value = max(value, 0.0f);",
                KernelOp::Neg { .. } => " value = -value;",
                KernelOp::Exp { .. } => " value = expf(value);",
                KernelOp::Sqrt { .. } => " value = sqrtf(value);",
                KernelOp::Abs { .. } => " value = fabsf(value);",
                KernelOp::Log { .. } => " value = logf(value);",
                KernelOp::Tanh { .. } => " value = tanhf(value);",
                KernelOp::Sin { .. } => " value = sinf(value);",
                KernelOp::Cos { .. } => " value = cosf(value);",
                KernelOp::Gelu { .. } => " value = 0.5f * value * (1.0f + tanhf(0.79788456f * (value + 0.044715f * value * value * value)));",
                KernelOp::Pow { exponent, .. } => return {
                    let _ = exponent;
                    (String::new(), String::new())
                },
                _ => return (String::new(), String::new()),
            });
        }
        let elements = output_shape.iter().product::<u64>();
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_broadcast_binary(device const float* x [[buffer(0)]], device const float* rhs [[buffer(1)]], device float* output [[buffer(2)]], uint id [[thread_position_in_grid]]) {{ if (id < {elements}u) {{ {lhs_declarations}{rhs_declarations} float lhs_value = x[{lhs_index}]; float rhs_value = rhs[{rhs_index}]; float value = {expression}; {postlude} output[id] = value; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_broadcast_binary(const float* x, const float* rhs, float* output, unsigned id) {{ if (id < {elements}u) {{ {lhs_declarations}{rhs_declarations} float lhs_value = x[{lhs_index}]; float rhs_value = rhs[{rhs_index}]; float value = {expression}; {postlude} output[id] = value; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let [KernelOp::Where {
        condition_shape,
        true_shape,
        false_shape,
        output_shape,
        ..
    }] = group.ops.as_slice()
    {
        let (condition_declarations, condition_index) =
            render_broadcast_index(condition_shape, output_shape, "c");
        let (true_declarations, true_index) = render_broadcast_index(true_shape, output_shape, "t");
        let (false_declarations, false_index) =
            render_broadcast_index(false_shape, output_shape, "f");
        let elements = output_shape.iter().product::<u64>();
        let source: String = match target {
            LoweringTarget::Metal => format!("#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_where(device const float* condition [[buffer(0)]], device const float* when_true [[buffer(1)]], device const float* when_false [[buffer(2)]], device float* output [[buffer(3)]], uint id [[thread_position_in_grid]]) {{ if (id < {elements}u) {{ {condition_declarations}{true_declarations}{false_declarations} output[id] = condition[{condition_index}] != 0.0f ? when_true[{true_index}] : when_false[{false_index}]; }} }}"),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!("void prism_where(const float* condition, const float* when_true, const float* when_false, float* output, unsigned id) {{ if (id < {elements}u) {{ {condition_declarations}{true_declarations}{false_declarations} output[id] = condition[{condition_index}] != 0.0f ? when_true[{true_index}] : when_false[{false_index}]; }} }}"),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let [KernelOp::Cast { from: _, to, .. }] = group.ops.as_slice() {
        let conversion = match to.as_str() {
            "f32" | "f16" | "bf16" => "v",
            "i8" => "(float)clamp((int)v, -128, 127)",
            "u8" => "(float)clamp((int)v, 0, 255)",
            "i32" => "(float)(int)clamp(v, -2147483648.0f, 2147483647.0f)",
            "u32" => "(float)(uint)clamp(v, 0.0f, 4294967295.0f)",
            _ => unreachable!("validated cast target"),
        };
        let source = match target {
            LoweringTarget::Metal => format!("#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_cast(device const float* x [[buffer(0)]], device float* output [[buffer(1)]], uint id [[thread_position_in_grid]]) {{ float v = x[id]; output[id] = {conversion}; }}"),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!("void prism_cast(const float* x, float* output, unsigned id) {{ float v = x[id]; output[id] = {conversion}; }}"),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((permutation, input_shape, output_shape)) = group.transpose_shape() {
        let input_strides: Vec<usize> = (0..input_shape.len())
            .map(|axis| {
                input_shape[axis + 1..]
                    .iter()
                    .map(|d| *d as usize)
                    .product()
            })
            .collect();
        let output_dims: Vec<usize> = output_shape.iter().map(|d| *d as usize).collect();
        let output_elements: usize = output_dims.iter().product();
        let mut coord = String::new();
        let mut source_index = String::from("0");
        for (axis, source_axis) in permutation.iter().enumerate() {
            coord.push_str(&format!(
                "unsigned c{axis} = (id / {}u) % {}u; ",
                output_dims[axis + 1..].iter().product::<usize>().max(1),
                output_dims[axis]
            ));
            source_index.push_str(&format!(" + c{axis} * {}u", input_strides[*source_axis]));
        }
        let source = match target {
            LoweringTarget::Metal => format!("#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_transpose(device const float* x [[buffer(0)]], device float* output [[buffer(1)]], uint id [[thread_position_in_grid]]) {{ if (id < {output_elements}u) {{ {coord} output[id] = x[{source_index}]; }} }}"),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!("void prism_transpose(const float* x, float* output, unsigned id) {{ if (id < {output_elements}u) {{ {coord} output[id] = x[{source_index}]; }} }}"),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((rows, features)) = group.ssm_shape() {
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_ssm(device const float* input [[buffer(0)]], device const float* decay [[buffer(1)]], device const float* input_gain [[buffer(2)]], device const float* output_gain [[buffer(3)]], device float* output [[buffer(4)]], uint id [[thread_position_in_grid]]) {{ if (id < {rows}u * {features}u) {{ uint row = id / {features}u; uint feature = id % {features}u; float state = 0.0; for (uint step = 0; step <= row; ++step) state = decay[feature] * state + input_gain[feature] * input[step * {features}u + feature]; output[id] = output_gain[feature] * state; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_ssm(const float* input, const float* decay, const float* input_gain, const float* output_gain, float* output, unsigned id) {{ if (id < {rows}u * {features}u) {{ unsigned row = id / {features}u; unsigned feature = id % {features}u; float state = 0.0f; for (unsigned step = 0; step <= row; ++step) state = decay[feature] * state + input_gain[feature] * input[step * {features}u + feature]; output[id] = output_gain[feature] * state; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((rows, vocab, features)) = group.gather_shape() {
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_gather(device const float* weight [[buffer(0)]], device const float* indices [[buffer(1)]], device float* output [[buffer(2)]], uint id [[thread_position_in_grid]]) {{ if (id < {rows}u * {features}u) {{ uint row = id / {features}u; uint col = id % {features}u; uint index = uint(indices[row]); if (index < {vocab}u) output[id] = weight[index * {features}u + col]; else output[id] = 0.0; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_gather(const float* weight, const float* indices, float* output, unsigned id) {{ if (id < {rows}u * {features}u) {{ unsigned row = id / {features}u; unsigned col = id % {features}u; unsigned index = (unsigned)indices[row]; output[id] = index < {vocab}u ? weight[index * {features}u + col] : 0.0f; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((rows, updates, features)) = group.scatter_shape() {
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_scatter(device const float* base [[buffer(0)]], device const float* indices [[buffer(1)]], device const float* updates [[buffer(2)]], device float* output [[buffer(3)]], uint id [[thread_position_in_grid]]) {{ if (id < {rows}u * {features}u) {{ output[id] = base[id]; for (uint update = 0; update < {updates}u; ++update) {{ uint index = uint(indices[update]); if (index == id / {features}u) output[id] = updates[update * {features}u + id % {features}u]; }} }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_scatter(const float* base, const float* indices, const float* updates, float* output, unsigned id) {{ if (id < {rows}u * {features}u) {{ output[id] = base[id]; for (unsigned update = 0; update < {updates}u; ++update) {{ unsigned index = (unsigned)indices[update]; if (index == id / {features}u) output[id] = updates[update * {features}u + id % {features}u]; }} }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((rows, features)) = group.rope_shape() {
        let elements = rows * (features / 2);
        let half = features / 2;
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_rope(device const float* x [[buffer(0)]], device const float* cosv [[buffer(1)]], device const float* sinv [[buffer(2)]], device float* output [[buffer(3)]], uint id [[thread_position_in_grid]]) {{ if (id < {elements}u) {{ uint row = id / {half}u; uint pair = id % {half}u; float c = cosv[id]; float s = sinv[id]; uint base = row * {features}u + pair * 2u; float a = x[base]; float b = x[base + 1u]; output[base] = a * c - b * s; output[base + 1u] = a * s + b * c; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_rope(const float* x, const float* cosv, const float* sinv, float* output, unsigned id) {{ if (id < {elements}u) {{ unsigned row = id / {half}u; unsigned pair = id % {half}u; float c = cosv[id]; float s = sinv[id]; unsigned base = row * {features}u + pair * 2u; float a = x[base]; float b = x[base + 1u]; output[base] = a * c - b * s; output[base + 1u] = a * s + b * c; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((batch, seq, head, scale)) = group.batched_attention_shape() {
        let elements = batch * seq * head;
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_attention_batched(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const float* v [[buffer(2)]], device float* output [[buffer(3)]], uint id [[thread_position_in_grid]]) {{ if (id < {elements}u) {{ uint batch = id / ({seq}u * {head}u); uint rem = id % ({seq}u * {head}u); uint query = rem / {head}u; uint dim = rem % {head}u; uint base = batch * {seq}u * {head}u; float scores[{seq}]; float max_v = -INFINITY; for (uint key = 0; key < {seq}u; ++key) {{ float score = 0.0; for (uint d = 0; d < {head}u; ++d) score += q[base + query * {head}u + d] * k[base + key * {head}u + d]; scores[key] = score * {scale}; max_v = max(max_v, scores[key]); }} float denom = 0.0; for (uint key = 0; key < {seq}u; ++key) denom += exp(scores[key] - max_v); float result = 0.0; for (uint key = 0; key < {seq}u; ++key) result += exp(scores[key] - max_v) / denom * v[base + key * {head}u + dim]; output[id] = result; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_attention_batched(const float* q, const float* k, const float* v, float* output, unsigned id) {{ if (id < {elements}u) {{ unsigned rem = id % ({seq}u * {head}u); unsigned query = rem / {head}u; unsigned dim = rem % {head}u; unsigned base = (id / ({seq}u * {head}u)) * {seq}u * {head}u; float scores[{seq}]; float max_v = -INFINITY; for (unsigned key = 0; key < {seq}u; ++key) {{ float score = 0.0f; for (unsigned d = 0; d < {head}u; ++d) score += q[base + query * {head}u + d] * k[base + key * {head}u + d]; scores[key] = score * {scale}f; if (scores[key] > max_v) max_v = scores[key]; }} float denom = 0.0f; for (unsigned key = 0; key < {seq}u; ++key) denom += expf(scores[key] - max_v); float result = 0.0f; for (unsigned key = 0; key < {seq}u; ++key) result += expf(scores[key] - max_v) / denom * v[base + key * {head}u + dim]; output[id] = result; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((seq, head, scale)) = group.attention_shape() {
        let elements = seq * head;
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_attention(device const float* q [[buffer(0)]], device const float* k [[buffer(1)]], device const float* v [[buffer(2)]], device float* output [[buffer(3)]], uint id [[thread_position_in_grid]]) {{ if (id < {elements}u) {{ uint query = id / {head}u; uint dim = id % {head}u; float scores[{seq}]; float max_v = -INFINITY; for (uint key = 0; key < {seq}u; ++key) {{ float score = 0.0; for (uint d = 0; d < {head}u; ++d) score += q[query * {head}u + d] * k[key * {head}u + d]; scores[key] = score * {scale}; max_v = max(max_v, scores[key]); }} float denom = 0.0; for (uint key = 0; key < {seq}u; ++key) denom += exp(scores[key] - max_v); float result = 0.0; for (uint key = 0; key < {seq}u; ++key) result += exp(scores[key] - max_v) / denom * v[key * {head}u + dim]; output[id] = result; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_attention(const float* q, const float* k, const float* v, float* output, unsigned id) {{ if (id < {elements}u) {{ unsigned query = id / {head}u; unsigned dim = id % {head}u; float scores[{seq}]; float max_v = -INFINITY; for (unsigned key = 0; key < {seq}u; ++key) {{ float score = 0.0f; for (unsigned d = 0; d < {head}u; ++d) score += q[query * {head}u + d] * k[key * {head}u + d]; scores[key] = score * {scale}f; if (scores[key] > max_v) max_v = scores[key]; }} float denom = 0.0f; for (unsigned key = 0; key < {seq}u; ++key) denom += expf(scores[key] - max_v); float result = 0.0f; for (unsigned key = 0; key < {seq}u; ++key) result += expf(scores[key] - max_v) / denom * v[key * {head}u + dim]; output[id] = result; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((rows, features, epsilon)) = group.rms_norm_shape() {
        let elements = rows * features;
        let source = match target {
            LoweringTarget::Metal => format!("#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_rms_norm(device const float* x [[buffer(0)]], device const float* weight [[buffer(1)]], device float* output [[buffer(2)]], uint id [[thread_position_in_grid]]) {{ if (id < {elements}u) {{ uint row = id / {features}u; float mean_sq = 0.0; for (uint i = 0; i < {features}u; ++i) {{ float v = x[row * {features}u + i]; mean_sq += v * v; }} mean_sq /= {features}u; output[id] = x[id] * rsqrt(mean_sq + {epsilon}); output[id] *= weight[id % {features}u]; }} }}"),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!("void prism_rms_norm(const float* x, const float* weight, float* output, unsigned id) {{ if (id < {elements}u) {{ unsigned row = id / {features}u; float mean_sq = 0.0f; for (unsigned i = 0; i < {features}u; ++i) {{ float v = x[row * {features}u + i]; mean_sq += v * v; }} mean_sq /= {features}u; output[id] = x[id] / sqrtf(mean_sq + {epsilon}f) * weight[id % {features}u]; }} }}"),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((rows, features, epsilon)) = group.layer_norm_shape() {
        let elements = rows * features;
        let source = match target {
            LoweringTarget::Metal => format!("#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_layer_norm(device const float* x [[buffer(0)]], device const float* weight [[buffer(1)]], device const float* bias [[buffer(2)]], device float* output [[buffer(3)]], uint id [[thread_position_in_grid]]) {{ if (id < {elements}u) {{ uint row = id / {features}u; float mean = 0.0; for (uint i = 0; i < {features}u; ++i) mean += x[row * {features}u + i]; mean /= {features}u; float variance = 0.0; for (uint i = 0; i < {features}u; ++i) {{ float centered = x[row * {features}u + i] - mean; variance += centered * centered; }} variance /= {features}u; output[id] = (x[id] - mean) * rsqrt(variance + {epsilon}) * weight[id % {features}u] + bias[id % {features}u]; }} }}"),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!("void prism_layer_norm(const float* x, const float* weight, const float* bias, float* output, unsigned id) {{ if (id < {elements}u) {{ unsigned row = id / {features}u; float mean = 0.0f; for (unsigned i = 0; i < {features}u; ++i) mean += x[row * {features}u + i]; mean /= {features}u; float variance = 0.0f; for (unsigned i = 0; i < {features}u; ++i) {{ float centered = x[row * {features}u + i] - mean; variance += centered * centered; }} variance /= {features}u; output[id] = (x[id] - mean) / sqrtf(variance + {epsilon}f) * weight[id % {features}u] + bias[id % {features}u]; }} }}"),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((
        batch,
        in_channels,
        height,
        width,
        out_channels,
        kernel_h,
        kernel_w,
        stride,
        padding,
    )) = group.conv2d_shape()
    {
        let out_h = (height + 2 * padding - kernel_h) / stride + 1;
        let out_w = (width + 2 * padding - kernel_w) / stride + 1;
        let elements = batch * out_channels * out_h * out_w;
        let source = match target {
            LoweringTarget::Metal => format!("#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_conv2d(device const float* x [[buffer(0)]], device const float* weight [[buffer(1)]], device const float* bias [[buffer(2)]], device float* output [[buffer(3)]], uint id [[thread_position_in_grid]]) {{ if (id < {elements}u) {{ uint ow = id % {out_w}u; uint oh = (id / {out_w}u) % {out_h}u; uint oc = (id / ({out_h}u * {out_w}u)) % {out_channels}u; uint b = id / ({out_channels}u * {out_h}u * {out_w}u); float sum = bias[oc]; for (uint ic = 0; ic < {in_channels}u; ++ic) for (uint kh = 0; kh < {kernel_h}u; ++kh) for (uint kw = 0; kw < {kernel_w}u; ++kw) {{ int ih = int(oh * {stride}u + kh) - int({padding}u); int iw = int(ow * {stride}u + kw) - int({padding}u); if (ih >= 0 && iw >= 0 && ih < {height} && iw < {width}) sum += x[((b * {in_channels}u + ic) * {height}u + uint(ih)) * {width}u + uint(iw)] * weight[(((oc * {in_channels}u + ic) * {kernel_h}u + kh) * {kernel_w}u) + kw]; }} output[id] = sum; }} }}"),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!("void prism_conv2d(const float* x, const float* weight, const float* bias, float* output, unsigned id) {{ if (id < {elements}u) {{ unsigned ow = id % {out_w}u; unsigned oh = (id / {out_w}u) % {out_h}u; unsigned oc = (id / ({out_h}u * {out_w}u)) % {out_channels}u; unsigned b = id / ({out_channels}u * {out_h}u * {out_w}u); float sum = bias[oc]; for (unsigned ic = 0; ic < {in_channels}u; ++ic) for (unsigned kh = 0; kh < {kernel_h}u; ++kh) for (unsigned kw = 0; kw < {kernel_w}u; ++kw) {{ int ih = (int)(oh * {stride}u + kh) - (int){padding}; int iw = (int)(ow * {stride}u + kw) - (int){padding}; if (ih >= 0 && iw >= 0 && ih < {height} && iw < {width}) sum += x[((b * {in_channels}u + ic) * {height}u + (unsigned)ih) * {width}u + (unsigned)iw] * weight[(((oc * {in_channels}u + ic) * {kernel_h}u + kh) * {kernel_w}u) + kw]; }} output[id] = sum; }} }}"),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((m, k, n)) = group.matmul_shape() {
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_matmul(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]], device float* output [[buffer(2)]], uint id [[thread_position_in_grid]]) {{ if (id < {m}u * {n}u) {{ uint row = id / {n}u; uint col = id % {n}u; float v = 0.0; for (uint inner = 0; inner < {k}u; ++inner) v += a[row * {k}u + inner] * b[inner * {n}u + col]; output[id] = v; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_matmul(const float* a, const float* b, float* output, unsigned id) {{ if (id < {m}u * {n}u) {{ unsigned row = id / {n}u; unsigned col = id % {n}u; float v = 0.0f; for (unsigned inner = 0; inner < {k}u; ++inner) v += a[row * {k}u + inner] * b[inner * {n}u + col]; output[id] = v; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some(elements) = group.reduction() {
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_reduce_sum(device const float* x [[buffer(0)]], device float* output [[buffer(1)]], uint id [[thread_position_in_grid]]) {{ if (id == 0) {{ float v = 0.0; for (uint i = 0; i < {elements}; ++i) v += x[i]; output[0] = v; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_reduce_sum(const float* x, float* output, unsigned id) {{ if (id == 0) {{ float v = 0.0f; for (unsigned i = 0; i < {elements}; ++i) v += x[i]; output[0] = v; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some(elements) = group.max_reduction() {
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_reduce_max(device const float* x [[buffer(0)]], device float* output [[buffer(1)]], uint id [[thread_position_in_grid]]) {{ if (id == 0) {{ float v = -INFINITY; for (uint i = 0; i < {elements}; ++i) v = max(v, x[i]); output[0] = v; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_reduce_max(const float* x, float* output, unsigned id) {{ if (id == 0) {{ float v = -INFINITY; for (unsigned i = 0; i < {elements}; ++i) v = fmaxf(v, x[i]); output[0] = v; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some(elements) = group.min_reduction() {
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_reduce_min(device const float* x [[buffer(0)]], device float* output [[buffer(1)]], uint id [[thread_position_in_grid]]) {{ if (id == 0) {{ float v = INFINITY; for (uint i = 0; i < {elements}; ++i) v = min(v, x[i]); output[0] = v; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_reduce_min(const float* x, float* output, unsigned id) {{ if (id == 0) {{ float v = INFINITY; for (unsigned i = 0; i < {elements}; ++i) v = fminf(v, x[i]); output[0] = v; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((outer, reduce, inner)) = group.softmax_shape() {
        let elements = outer * reduce * inner;
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_softmax_axis(device const float* x [[buffer(0)]], device float* output [[buffer(1)]], uint id [[thread_position_in_grid]]) {{ if (id < {elements}u) {{ uint outer = id / ({reduce}u * {inner}u); uint rem = id % ({reduce}u * {inner}u); uint step = rem / {inner}u; uint inner = rem % {inner}u; float max_v = -INFINITY; for (uint i = 0; i < {reduce}u; ++i) max_v = max(max_v, x[(outer * {reduce}u + i) * {inner}u + inner]); float denom = 0.0; for (uint i = 0; i < {reduce}u; ++i) denom += exp(x[(outer * {reduce}u + i) * {inner}u + inner] - max_v); output[id] = exp(x[id] - max_v) / denom; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_softmax_axis(const float* x, float* output, unsigned id) {{ if (id < {elements}u) {{ unsigned outer = id / ({reduce}u * {inner}u); unsigned rem = id % ({reduce}u * {inner}u); unsigned inner = rem % {inner}u; float max_v = -INFINITY; for (unsigned i = 0; i < {reduce}u; ++i) {{ float v = x[(outer * {reduce}u + i) * {inner}u + inner]; if (v > max_v) max_v = v; }} float denom = 0.0f; for (unsigned i = 0; i < {reduce}u; ++i) denom += expf(x[(outer * {reduce}u + i) * {inner}u + inner] - max_v); output[id] = expf(x[id] - max_v) / denom; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((outer, reduce, inner)) = group.axis_reduction() {
        let output_elements = outer * inner;
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_reduce_sum_axis(device const float* x [[buffer(0)]], device float* output [[buffer(1)]], uint id [[thread_position_in_grid]]) {{ if (id < {output_elements}u) {{ uint outer = id / {inner}u; uint inner = id % {inner}u; float v = 0.0; for (uint step = 0; step < {reduce}u; ++step) v += x[(outer * {reduce}u + step) * {inner}u + inner]; output[id] = v; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_reduce_sum_axis(const float* x, float* output, unsigned id) {{ if (id < {output_elements}u) {{ unsigned outer = id / {inner}u; unsigned inner = id % {inner}u; float v = 0.0f; for (unsigned step = 0; step < {reduce}u; ++step) v += x[(outer * {reduce}u + step) * {inner}u + inner]; output[id] = v; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((outer, reduce, inner)) = group.max_axis_reduction() {
        let output_elements = outer * inner;
        let source = match target {
            LoweringTarget::Metal => format!(
                "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_reduce_max_axis(device const float* x [[buffer(0)]], device float* output [[buffer(1)]], uint id [[thread_position_in_grid]]) {{ if (id < {output_elements}u) {{ uint outer = id / {inner}u; uint inner = id % {inner}u; float v = -INFINITY; for (uint step = 0; step < {reduce}u; ++step) v = max(v, x[(outer * {reduce}u + step) * {inner}u + inner]); output[id] = v; }} }}"
            ),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!(
                "void prism_reduce_max_axis(const float* x, float* output, unsigned id) {{ if (id < {output_elements}u) {{ unsigned outer = id / {inner}u; unsigned inner = id % {inner}u; float v = -INFINITY; for (unsigned step = 0; step < {reduce}u; ++step) v = fmaxf(v, x[(outer * {reduce}u + step) * {inner}u + inner]); output[id] = v; }} }}"
            ),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    if let Some((outer, reduce, inner)) = group.min_axis_reduction() {
        let output_elements = outer * inner;
        let source = match target {
            LoweringTarget::Metal => format!("#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_reduce_min_axis(device const float* x [[buffer(0)]], device float* output [[buffer(1)]], uint id [[thread_position_in_grid]]) {{ if (id < {output_elements}u) {{ uint outer = id / {inner}u; uint inner = id % {inner}u; float v = INFINITY; for (uint step = 0; step < {reduce}u; ++step) v = min(v, x[(outer * {reduce}u + step) * {inner}u + inner]); output[id] = v; }} }}"),
            LoweringTarget::Cpu | LoweringTarget::Portable => format!("void prism_reduce_min_axis(const float* x, float* output, unsigned id) {{ if (id < {output_elements}u) {{ unsigned outer = id / {inner}u; unsigned inner = id % {inner}u; float v = INFINITY; for (unsigned step = 0; step < {reduce}u; ++step) v = fminf(v, x[(outer * {reduce}u + step) * {inner}u + inner]); output[id] = v; }} }}"),
        };
        let mut digest = Sha256::new();
        digest.update(source.as_bytes());
        return (source, hex_digest(digest.finalize()));
    }
    let mut source = match target {
        LoweringTarget::Metal => {
            if group.requires_rhs() {
                String::from("#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_kernel(device const float* x [[buffer(0)]], device const float* rhs [[buffer(1)]], device float* output [[buffer(2)]], uint id [[thread_position_in_grid]]) { float v = x[id];")
            } else {
                String::from(
                    "#include <metal_stdlib>\nusing namespace metal;\nkernel void prism_kernel(device const float* x [[buffer(0)]], device float* output [[buffer(1)]], uint id [[thread_position_in_grid]]) { float v = x[id];",
                )
            }
        }
        LoweringTarget::Cpu | LoweringTarget::Portable => {
            if group.requires_rhs() {
                String::from(
                    "void prism_kernel(const float* x, const float* rhs, float* output, unsigned id) { float v = x[id];",
                )
            } else {
                String::from("void prism_kernel(const float* x, float* output, unsigned id) { float v = x[id];")
            }
        }
    };
    for op in &group.ops {
        match op {
            KernelOp::BroadcastBinary { .. } => {
                unreachable!("shape-aware broadcast requires a dedicated kernel ABI")
            }
            KernelOp::ReduceSum { .. } => unreachable!("reductions use the dedicated renderer"),
            KernelOp::ReduceMax { .. } => unreachable!("reductions use the dedicated renderer"),
            KernelOp::ReduceMin { .. } => unreachable!("reductions use the dedicated renderer"),
            KernelOp::ReduceSumAxis { .. } => unreachable!("reductions use the dedicated renderer"),
            KernelOp::ReduceMaxAxis { .. } => unreachable!("reductions use the dedicated renderer"),
            KernelOp::ReduceMinAxis { .. } => unreachable!("reductions use the dedicated renderer"),
            KernelOp::SoftmaxAxis { .. } => unreachable!("softmax uses the dedicated renderer"),
            KernelOp::Attention { .. } => unreachable!("attention uses the dedicated renderer"),
            KernelOp::AttentionBatched { .. } => {
                unreachable!("attention uses the dedicated renderer")
            }
            KernelOp::RmsNorm { .. } => unreachable!("rms norm uses the dedicated renderer"),
            KernelOp::LayerNorm { .. } => unreachable!("layer norm uses the dedicated renderer"),
            KernelOp::Rope { .. } => unreachable!("rope uses the dedicated renderer"),
            KernelOp::Gather { .. } => unreachable!("gather uses the dedicated renderer"),
            KernelOp::Scatter { .. } => unreachable!("scatter uses the dedicated renderer"),
            KernelOp::Ssm { .. } => unreachable!("ssm uses the dedicated renderer"),
            KernelOp::MatMul { .. } => unreachable!("matmul uses the dedicated renderer"),
            KernelOp::Conv2d { .. } => unreachable!("conv2d uses the dedicated renderer"),
            KernelOp::Where { .. } => unreachable!("where uses the dedicated renderer"),
            KernelOp::Transpose { .. } => unreachable!("transpose uses the dedicated renderer"),
            KernelOp::Add {
                scalar,
                scalar_left,
                ..
            } => source.push_str(&render_binary(*scalar, *scalar_left, "+")),
            KernelOp::Mul {
                scalar,
                scalar_left,
                ..
            } => source.push_str(&render_binary(*scalar, *scalar_left, "*")),
            KernelOp::Sub {
                scalar,
                scalar_left,
                ..
            } => source.push_str(&render_binary(*scalar, *scalar_left, "-")),
            KernelOp::Div {
                scalar,
                scalar_left,
                ..
            } => source.push_str(&render_binary(*scalar, *scalar_left, "/")),
            KernelOp::Maximum {
                scalar,
                scalar_left,
                ..
            } => source.push_str(&render_extremum(*scalar, *scalar_left, "max")),
            KernelOp::Minimum {
                scalar,
                scalar_left,
                ..
            } => source.push_str(&render_extremum(*scalar, *scalar_left, "min")),
            KernelOp::Relu { .. } => source.push_str(" v = v > 0.0 ? v : 0.0;"),
            KernelOp::Neg { .. } => source.push_str(" v = -v;"),
            KernelOp::Exp { .. } => source.push_str(if matches!(target, LoweringTarget::Metal) {
                " v = exp(v);"
            } else {
                " v = expf(v);"
            }),
            KernelOp::Sqrt { .. } => source.push_str(if matches!(target, LoweringTarget::Metal) {
                " v = sqrt(v);"
            } else {
                " v = sqrtf(v);"
            }),
            KernelOp::Abs { .. } => source.push_str(if matches!(target, LoweringTarget::Metal) {
                " v = abs(v);"
            } else {
                " v = fabsf(v);"
            }),
            KernelOp::Log { .. } => source.push_str(if matches!(target, LoweringTarget::Metal) {
                " v = log(v);"
            } else {
                " v = logf(v);"
            }),
            KernelOp::Tanh { .. } => source.push_str(if matches!(target, LoweringTarget::Metal) {
                " v = tanh(v);"
            } else {
                " v = tanhf(v);"
            }),
            KernelOp::Sin { .. } => source.push_str(if matches!(target, LoweringTarget::Metal) {
                " v = sin(v);"
            } else {
                " v = sinf(v);"
            }),
            KernelOp::Cos { .. } => source.push_str(if matches!(target, LoweringTarget::Metal) {
                " v = cos(v);"
            } else {
                " v = cosf(v);"
            }),
            KernelOp::Gelu { .. } => source.push_str(if matches!(target, LoweringTarget::Metal) {
                " v = 0.5 * v * (1.0 + tanh(0.7978845608 * (v + 0.044715 * v * v * v)));"
            } else {
                " v = 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));"
            }),
            KernelOp::Pow { exponent, .. } => {
                source.push_str(&if matches!(target, LoweringTarget::Metal) {
                    format!(" v = pow(v, {exponent});")
                } else {
                    format!(" v = powf(v, {exponent}f);")
                })
            }
            KernelOp::Cast { to, .. } => source.push_str(match to.as_str() {
                "f32" | "f16" | "bf16" => "",
                "i8" => " v = fminf(fmaxf(truncf(v), -128.0f), 127.0f);",
                "u8" => " v = fminf(fmaxf(truncf(v), 0.0f), 255.0f);",
                "i32" => " v = fminf(fmaxf(truncf(v), -2147483648.0f), 2147483647.0f);",
                "u32" => " v = fminf(fmaxf(truncf(v), 0.0f), 4294967295.0f);",
                _ => unreachable!("validated cast target"),
            }),
        }
    }
    source.push_str(" output[id] = v; }");
    let mut digest = Sha256::new();
    digest.update(source.as_bytes());
    (source, hex_digest(digest.finalize()))
}

fn render_binary(scalar: Option<f32>, scalar_left: bool, operator: &str) -> String {
    let scalar = scalar.map(|value| value.to_string());
    let left = if scalar.is_some() && scalar_left {
        scalar.as_deref().unwrap()
    } else {
        "v"
    };
    let right = if scalar.is_some() && scalar_left {
        "v"
    } else {
        scalar.as_deref().unwrap_or("rhs[id]")
    };
    format!(" v = {left} {operator} {right};")
}

fn render_extremum(scalar: Option<f32>, scalar_left: bool, function: &str) -> String {
    let scalar = scalar.map(|value| value.to_string());
    let left = if scalar.is_some() && scalar_left {
        scalar.as_deref().unwrap()
    } else {
        "v"
    };
    let right = if scalar.is_some() && scalar_left {
        "v"
    } else {
        scalar.as_deref().unwrap_or("rhs[id]")
    };
    format!(" v = {function}({left}, {right});")
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    bytes
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn schedules_and_fuses_elementwise_chain() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
        let one = graph.add(UOpKind::Const { value: 1.0 }, vec![], vec![1]);
        let add = graph.add(UOpKind::Add, vec![x, one], vec![4]);
        let relu = graph.add(UOpKind::Relu, vec![add], vec![4]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![4]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert_eq!(capture.kernels.len(), 1);
        assert_eq!(capture.kernels[0].group.ops.len(), 2);
        assert!(!capture.kernels[0].source_digest.is_empty());
        assert!(capture.kernels[0].source.contains("prism_kernel"));
    }

    #[test]
    fn optimizer_eliminates_neutral_elementwise_uops_before_lowering() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
        let zero = graph.add(UOpKind::Const { value: 0.0 }, vec![], vec![1]);
        let one = graph.add(UOpKind::Const { value: 1.0 }, vec![], vec![1]);
        let add = graph.add(UOpKind::Add, vec![x, zero], vec![4]);
        let mul = graph.add(UOpKind::Mul, vec![add, one], vec![4]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![mul], vec![4]);

        let optimized = graph.optimize().unwrap();
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        let output = graph
            .execute_f32(&BTreeMap::from([(
                String::from("x"),
                vec![1.0, -2.0, 3.0, 4.0],
            )]))
            .unwrap();

        assert!(optimized
            .ops
            .iter()
            .any(|op| { matches!(op.kind, UOpKind::Output { .. }) && op.src == vec![x] }));
        assert!(capture.kernels.is_empty());
        assert_eq!(output["y"], vec![1.0, -2.0, 3.0, 4.0]);
    }

    #[test]
    fn mean_axis_reuses_minimal_sum_and_scalar_division_uops() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
        let mean = graph.add_mean_axis(input, 1).unwrap();
        graph.add(
            UOpKind::Output {
                name: "mean".into(),
            },
            vec![mean],
            vec![2],
        );
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert_eq!(capture.kernels.len(), 2);
        let outputs = graph
            .execute_f32(&BTreeMap::from([(
                "x".into(),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            )]))
            .unwrap();
        assert_eq!(outputs["mean"], vec![2.0, 5.0]);
    }

    #[test]
    fn whole_tensor_mean_reuses_reduce_sum_and_scalar_division() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 2]);
        let mean = graph.add_mean(input).unwrap();
        graph.add(
            UOpKind::Output {
                name: "mean".into(),
            },
            vec![mean],
            vec![1],
        );
        let output = graph
            .execute_f32(&BTreeMap::from([("x".into(), vec![1.0, 2.0, 3.0, 6.0])]))
            .unwrap();
        assert_eq!(output["mean"], vec![3.0]);
    }

    #[test]
    fn validation_rejects_non_scalar_whole_tensor_reduction_output() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let sum = graph.add(UOpKind::ReduceSum, vec![input], vec![2]);
        graph.add(UOpKind::Output { name: "sum".into() }, vec![sum], vec![2]);
        assert!(matches!(
            graph.validate(),
            Err(GraphError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn validation_rejects_invalid_specialized_parameters() {
        let mut cast = TinyGraph::default();
        let input = cast.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        cast.add(
            UOpKind::Cast {
                from: "fp8".into(),
                to: "f32".into(),
            },
            vec![input],
            vec![2],
        );
        assert!(matches!(cast.validate(), Err(GraphError::ShapeMismatch(_))));

        let mut attention = TinyGraph::default();
        let q = attention.add(UOpKind::Input { name: "q".into() }, vec![], vec![2, 2]);
        let k = attention.add(UOpKind::Input { name: "k".into() }, vec![], vec![2, 2]);
        let v = attention.add(UOpKind::Input { name: "v".into() }, vec![], vec![2, 2]);
        attention.add(
            UOpKind::Attention {
                seq: 2,
                head: 2,
                scale: f32::NAN,
            },
            vec![q, k, v],
            vec![2, 2],
        );
        assert!(matches!(
            attention.validate(),
            Err(GraphError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn cast_reference_and_rendered_integer_bounds_agree() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
        let cast = graph.add(
            UOpKind::Cast {
                from: "f32".into(),
                to: "i32".into(),
            },
            vec![input],
            vec![4],
        );
        graph.add(UOpKind::Output { name: "y".into() }, vec![cast], vec![4]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("2147483647.0f"));
        let output = graph
            .execute_f32(&BTreeMap::from([(
                "x".into(),
                vec![f32::MAX, f32::MIN, 2.9, -2.9],
            )]))
            .unwrap();
        assert_eq!(output["y"], vec![2147483647.0, -2147483648.0, 2.0, -2.0]);
    }

    #[test]
    fn strategy_lowering_emits_distinct_executable_capture_layouts() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
        let one = graph.add(UOpKind::Const { value: 1.0 }, vec![], vec![1]);
        let add = graph.add(UOpKind::Add, vec![x, one], vec![4]);
        let relu = graph.add(UOpKind::Relu, vec![add], vec![4]);
        let exp = graph.add(UOpKind::Exp, vec![relu], vec![4]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![exp], vec![4]);

        let standard = graph
            .lower_with_fusion_strategy(
                LoweringTarget::Portable,
                &crate::fused_ops::FusionStrategy::StandardFused,
            )
            .unwrap();
        let per_operation = graph
            .lower_with_fusion_strategy(
                LoweringTarget::Portable,
                &crate::fused_ops::FusionStrategy::PerOperation,
            )
            .unwrap();
        let interleaved = graph
            .lower_with_fusion_strategy(
                LoweringTarget::Portable,
                &crate::fused_ops::FusionStrategy::InterleavedFused {
                    stages: vec![vec![crate::fused_ops::FusableOp::FpGemv]; 2],
                },
            )
            .unwrap();

        assert_eq!(standard.kernels.len(), 1);
        assert_eq!(per_operation.kernels.len(), 3);
        assert_eq!(interleaved.kernels.len(), 2);
        assert_eq!(per_operation.replay.command_ids, vec![0, 1, 2]);
        assert!(per_operation.validate().is_ok());
        assert!(interleaved.validate().is_ok());
    }

    #[test]
    fn empty_interleaved_stage_request_still_splits_multi_op_kernel() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![4]);
        let exp = graph.add(UOpKind::Exp, vec![relu], vec![4]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![exp], vec![4]);
        let capture = graph
            .lower_with_fusion_strategy(
                LoweringTarget::Portable,
                &crate::fused_ops::FusionStrategy::InterleavedFused { stages: vec![] },
            )
            .unwrap();
        assert_eq!(capture.kernels.len(), 2);
        assert!(capture.validate().is_ok());
    }

    #[test]
    fn fusion_does_not_consume_forked_intermediate_values() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let one = graph.add(UOpKind::Const { value: 1.0 }, vec![], vec![1]);
        let base = graph.add(UOpKind::Add, vec![x, one], vec![2]);
        let relu = graph.add(UOpKind::Relu, vec![base], vec![2]);
        let scaled = graph.add(UOpKind::Mul, vec![base, one], vec![2]);
        graph.add(UOpKind::Output { name: "a".into() }, vec![relu], vec![2]);
        graph.add(UOpKind::Output { name: "b".into() }, vec![scaled], vec![2]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        // `base * 1` is rewritten to `base`; the two consumers still remain
        // separate because the shared intermediate is forked.
        assert_eq!(capture.kernels.len(), 2);
        assert_eq!(capture.kernels[0].group.ops.len(), 1);
    }

    struct RecordingExecutor(Vec<String>);
    impl CaptureExecutor for RecordingExecutor {
        fn dispatch(&mut self, command_id: u32, _kernel: &LoweredKernel) -> Result<(), String> {
            self.0.push(format!("dispatch:{command_id}"));
            Ok(())
        }
        fn synchronize(&mut self, command_id: u32) -> Result<(), String> {
            self.0.push(format!("sync:{command_id}"));
            Ok(())
        }
    }

    #[test]
    fn capture_replays_commands_and_returns_receipt() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
        let one = graph.add(UOpKind::Const { value: 1.0 }, vec![], vec![1]);
        let add = graph.add(UOpKind::Add, vec![x, one], vec![4]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![add], vec![4]);
        let capture = graph.lower(LoweringTarget::Cpu).unwrap();
        let mut executor = RecordingExecutor(Vec::new());
        let receipt = capture.replay(&mut executor).unwrap();
        assert!(receipt.replayed);
        assert_eq!(receipt.capture_digest, capture.digest());
        capture.validate_receipt(&receipt).unwrap();
        let mut tampered = receipt.clone();
        tampered.command_ids.push(99);
        assert!(capture.validate_receipt(&tampered).is_err());
        assert_eq!(executor.0, vec!["dispatch:0", "sync:0"]);
    }

    #[test]
    fn tiny_jit_cache_captures_once_and_reuses_by_graph_and_target() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![2]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![2]);
        let mut cache = TinyJitCache::default();
        let (first_key, first_capture) = cache.capture(&graph, LoweringTarget::Portable).unwrap();
        let (second_key, second_capture) = cache.capture(&graph, LoweringTarget::Portable).unwrap();
        assert!(first_capture);
        assert!(!second_capture);
        assert_eq!(first_key, second_key);
        assert_eq!(cache.len(), 1);
        assert!(cache.get(&first_key).unwrap().replay.command_ids == vec![0]);
        let mut executor = RecordingExecutor(Vec::new());
        let receipt = cache.replay(&first_key, &mut executor).unwrap();
        assert!(receipt.replayed);
        assert_eq!(executor.0, vec!["dispatch:0", "sync:0"]);
    }

    #[test]
    fn tiny_jit_cache_supports_capture_invalidation() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![2]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![2]);
        let mut cache = TinyJitCache::default();
        let (key, _) = cache.capture(&graph, LoweringTarget::Portable).unwrap();
        assert!(cache.invalidate(&key));
        assert!(!cache.invalidate(&key));
        assert!(cache.is_empty());
        cache.capture(&graph, LoweringTarget::Portable).unwrap();
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn tiny_jit_cache_keys_strategy_specific_capture_layouts() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![4]);
        let exp = graph.add(UOpKind::Exp, vec![relu], vec![4]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![exp], vec![4]);
        let mut cache = TinyJitCache::default();
        let (standard_key, _) = cache.capture(&graph, LoweringTarget::Portable).unwrap();
        let (per_op_key, first) = cache
            .capture_with_strategy(
                &graph,
                LoweringTarget::Portable,
                &crate::fused_ops::FusionStrategy::PerOperation,
            )
            .unwrap();
        let (_, second) = cache
            .capture_with_strategy(
                &graph,
                LoweringTarget::Portable,
                &crate::fused_ops::FusionStrategy::PerOperation,
            )
            .unwrap();
        assert_ne!(standard_key, per_op_key);
        assert!(first);
        assert!(!second);
        assert_eq!(cache.get(&standard_key).unwrap().kernels.len(), 1);
        assert_eq!(cache.get(&per_op_key).unwrap().kernels.len(), 2);
    }

    #[test]
    fn tiny_jit_cache_materializes_requested_strategy_set_once() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![4]);
        let exp = graph.add(UOpKind::Exp, vec![relu], vec![4]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![exp], vec![4]);
        let strategies = vec![
            crate::fused_ops::FusionStrategy::StandardFused,
            crate::fused_ops::FusionStrategy::PerOperation,
            crate::fused_ops::FusionStrategy::PersistentMegakernel {
                search_generation: 7,
            },
        ];
        let mut cache = TinyJitCache::default();
        let first = cache
            .capture_strategies(&graph, LoweringTarget::Portable, &strategies)
            .unwrap();
        assert!(first.iter().all(|(_, _, inserted)| *inserted));
        assert_eq!(cache.len(), strategies.len());
        let second = cache
            .capture_strategies(&graph, LoweringTarget::Portable, &strategies)
            .unwrap();
        assert!(second.iter().all(|(_, _, inserted)| !*inserted));
        assert_eq!(
            first.iter().map(|(_, key, _)| key).collect::<Vec<_>>(),
            second.iter().map(|(_, key, _)| key).collect::<Vec<_>>()
        );
        let duplicate = cache.capture_strategies(
            &graph,
            LoweringTarget::Portable,
            &[
                crate::fused_ops::FusionStrategy::StandardFused,
                crate::fused_ops::FusionStrategy::StandardFused,
            ],
        );
        assert!(duplicate.is_err());
    }

    #[test]
    fn tiny_jit_cache_round_trips_through_validated_bytes() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![2]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![2]);
        let mut cache = TinyJitCache::default();
        let (key, _) = cache.capture(&graph, LoweringTarget::Portable).unwrap();
        let bytes = cache.export_bytes().unwrap();
        let restored = TinyJitCache::import_bytes(&bytes).unwrap();
        assert_eq!(restored.len(), 1);
        assert_eq!(restored.get(&key), cache.get(&key));
    }

    #[test]
    fn tiny_jit_cache_import_rejects_malformed_bytes() {
        let error = TinyJitCache::import_bytes(b"not a cache").unwrap_err();
        assert!(matches!(error, GraphError::Serialization(_)));
    }

    #[test]
    fn tiny_jit_cache_import_rejects_identity_tampering() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
        let output = graph.add(UOpKind::Output { name: "y".into() }, vec![input], vec![1]);
        assert_eq!(output, UOpId(1));
        let mut cache = TinyJitCache::default();
        cache.capture(&graph, LoweringTarget::Portable).unwrap();
        let bytes = cache.export_bytes().unwrap();
        let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let digest = value["captures"]
            .as_object_mut()
            .and_then(|entries| entries.values_mut().next())
            .and_then(|entry| entry.get_mut("identity_digest"))
            .and_then(|value| value.as_str())
            .unwrap()
            .to_owned();
        value["captures"]
            .as_object_mut()
            .unwrap()
            .values_mut()
            .next()
            .unwrap()["identity_digest"] = serde_json::Value::String(format!("{digest}tampered"));
        let tampered = serde_json::to_vec(&value).unwrap();
        assert!(TinyJitCache::import_bytes(&tampered).is_err());
    }

    #[test]
    fn persistent_capture_uses_single_submission_and_final_sync() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![4]);
        let exp = graph.add(UOpKind::Exp, vec![relu], vec![4]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![exp], vec![4]);
        let capture = graph
            .lower_with_fusion_strategy(
                LoweringTarget::Portable,
                &crate::fused_ops::FusionStrategy::PersistentMegakernel {
                    search_generation: 9,
                },
            )
            .unwrap();
        assert!(capture.replay.persistent);
        assert_eq!(capture.kernels.len(), 1);
        assert_eq!(capture.kernels[0].group.ops.len(), 2);
        let mut executor = RecordingExecutor(Vec::new());
        let receipt = capture.replay(&mut executor).unwrap();
        assert!(receipt.persistent);
        assert_eq!(executor.0, vec!["dispatch:0", "sync:0"]);
    }

    #[test]
    fn persistent_metadata_defaults_for_legacy_serialized_records() {
        let replay: ReplayPlan = serde_json::from_value(serde_json::json!({
            "command_ids": [0],
            "synchronization_points": [0]
        }))
        .unwrap();
        assert!(!replay.persistent);
        let receipt: ExecutionReceipt = serde_json::from_value(serde_json::json!({
            "target": "Portable",
            "command_ids": [0],
            "kernel_digests": ["digest"],
            "replayed": true
        }))
        .unwrap();
        assert!(!receipt.persistent);
    }

    #[test]
    fn reference_executor_produces_behavioral_output() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![3]);
        let two = graph.add(UOpKind::Const { value: 2.0 }, vec![], vec![1]);
        let scaled = graph.add(UOpKind::Mul, vec![x, two], vec![3]);
        let positive = graph.add(UOpKind::Relu, vec![scaled], vec![3]);
        graph.add(
            UOpKind::Output { name: "y".into() },
            vec![positive],
            vec![3],
        );
        let mut inputs = BTreeMap::new();
        inputs.insert("x".into(), vec![-1.0, 2.0, 3.0]);
        let outputs = graph.execute_f32(&inputs).unwrap();
        assert_eq!(outputs["y"], vec![0.0, 4.0, 6.0]);
    }

    #[test]
    fn optimizer_folds_constant_binary_ops() {
        let mut graph = TinyGraph::default();
        let left = graph.add(UOpKind::Const { value: 2.0 }, vec![], vec![1]);
        let right = graph.add(UOpKind::Const { value: 3.0 }, vec![], vec![1]);
        let sum = graph.add(UOpKind::Add, vec![left, right], vec![1]);
        let output = graph.add(UOpKind::Output { name: "y".into() }, vec![sum], vec![1]);
        let optimized = graph.optimize().unwrap();
        assert!(
            matches!(optimized.ops[sum.0 as usize].kind, UOpKind::Const { value } if value == 5.0)
        );
        assert_eq!(optimized.ops[output.0 as usize].src, vec![sum]);
    }

    #[test]
    fn optimizer_reaches_fixed_point_for_nested_unary_and_binary_folds() {
        let mut graph = TinyGraph::default();
        let left = graph.add(UOpKind::Const { value: 2.0 }, vec![], vec![1]);
        let right = graph.add(UOpKind::Const { value: 3.0 }, vec![], vec![1]);
        let sum = graph.add(UOpKind::Add, vec![left, right], vec![1]);
        let relu = graph.add(UOpKind::Relu, vec![sum], vec![1]);
        let output = graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![1]);
        let optimized = graph.optimize().unwrap();
        assert!(
            matches!(optimized.ops[sum.0 as usize].kind, UOpKind::Const { value } if value == 5.0)
        );
        assert!(
            matches!(optimized.ops[relu.0 as usize].kind, UOpKind::Const { value } if value == 5.0)
        );
        assert_eq!(optimized.ops[output.0 as usize].src, vec![relu]);
    }

    #[test]
    fn optimizer_folds_constant_where() {
        let mut graph = TinyGraph::default();
        let condition = graph.add(UOpKind::Const { value: 1.0 }, vec![], vec![2]);
        let when_true = graph.add(UOpKind::Const { value: 7.0 }, vec![], vec![2]);
        let when_false = graph.add(UOpKind::Const { value: -3.0 }, vec![], vec![2]);
        let selected = graph.add(
            UOpKind::Where,
            vec![condition, when_true, when_false],
            vec![2],
        );
        let optimized = graph.optimize().unwrap();
        assert!(
            matches!(optimized.ops[selected.0 as usize].kind, UOpKind::Const { value } if value == 7.0)
        );
    }

    #[test]
    fn optimizer_folds_constant_cast_with_runtime_cast_semantics() {
        let mut graph = TinyGraph::default();
        let value = graph.add(UOpKind::Const { value: 300.75 }, vec![], vec![1]);
        let cast = graph.add(
            UOpKind::Cast {
                from: "f32".into(),
                to: "u8".into(),
            },
            vec![value],
            vec![1],
        );
        graph.add(UOpKind::Output { name: "out".into() }, vec![cast], vec![1]);
        let optimized = graph.optimize().unwrap();
        assert!(matches!(
            optimized.ops[cast.0 as usize].kind,
            UOpKind::Const { value } if value == 255.0
        ));
        assert!(optimized.ops[cast.0 as usize].src.is_empty());
    }

    #[test]
    fn optimizer_folds_constant_reductions() {
        let mut graph = TinyGraph::default();
        let value = graph.add(UOpKind::Const { value: 2.0 }, vec![], vec![2, 3]);
        let sum = graph.add(UOpKind::ReduceSum, vec![value], vec![1]);
        let max = graph.add(UOpKind::ReduceMax, vec![value], vec![1]);
        let axis = graph.add(UOpKind::ReduceSumAxis { axis: 1 }, vec![value], vec![2]);
        graph.add(UOpKind::Output { name: "sum".into() }, vec![sum], vec![1]);
        graph.add(UOpKind::Output { name: "max".into() }, vec![max], vec![1]);
        graph.add(
            UOpKind::Output {
                name: "axis".into(),
            },
            vec![axis],
            vec![2],
        );
        let optimized = graph.optimize().unwrap();
        assert!(
            matches!(optimized.ops[sum.0 as usize].kind, UOpKind::Const { value } if value == 12.0)
        );
        assert!(
            matches!(optimized.ops[max.0 as usize].kind, UOpKind::Const { value } if value == 2.0)
        );
        assert!(
            matches!(optimized.ops[axis.0 as usize].kind, UOpKind::Const { value } if value == 6.0)
        );
    }

    #[test]
    fn optimizer_folds_constant_matmul() {
        let mut graph = TinyGraph::default();
        let lhs = graph.add(UOpKind::Const { value: 2.0 }, vec![], vec![2, 3]);
        let rhs = graph.add(UOpKind::Const { value: 4.0 }, vec![], vec![3, 5]);
        let product = graph.add(
            UOpKind::MatMul { m: 2, k: 3, n: 5 },
            vec![lhs, rhs],
            vec![2, 5],
        );
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![product],
            vec![2, 5],
        );
        let optimized = graph.optimize().unwrap();
        assert!(matches!(
            optimized.ops[product.0 as usize].kind,
            UOpKind::Const { value } if value == 24.0
        ));
    }

    #[test]
    fn optimizer_folds_constant_softmax_axis() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Const { value: 7.0 }, vec![], vec![2, 4]);
        let softmax = graph.add(UOpKind::SoftmaxAxis { axis: 1 }, vec![input], vec![2, 4]);
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![softmax],
            vec![2, 4],
        );
        let optimized = graph.optimize().unwrap();
        assert!(matches!(
            optimized.ops[softmax.0 as usize].kind,
            UOpKind::Const { value } if (value - 0.25).abs() < f32::EPSILON
        ));
    }

    #[test]
    fn optimizer_folds_constant_attention_to_value_stream() {
        let mut graph = TinyGraph::default();
        let q = graph.add(UOpKind::Const { value: 1.0 }, vec![], vec![2, 3]);
        let k = graph.add(UOpKind::Const { value: 2.0 }, vec![], vec![2, 3]);
        let v = graph.add(UOpKind::Const { value: 9.0 }, vec![], vec![2, 3]);
        let attention = graph.add(
            UOpKind::Attention {
                seq: 2,
                head: 3,
                scale: 1.0,
            },
            vec![q, k, v],
            vec![2, 3],
        );
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![attention],
            vec![2, 3],
        );
        let optimized = graph.optimize().unwrap();
        assert!(matches!(
            optimized.ops[attention.0 as usize].kind,
            UOpKind::Const { value } if value == 9.0
        ));
    }

    #[test]
    fn optimizer_cse_reuses_duplicate_matmul() {
        let mut graph = TinyGraph::default();
        let lhs = graph.add(UOpKind::Input { name: "lhs".into() }, vec![], vec![2, 3]);
        let rhs = graph.add(UOpKind::Input { name: "rhs".into() }, vec![], vec![3, 4]);
        let first = graph.add(
            UOpKind::MatMul { m: 2, k: 3, n: 4 },
            vec![lhs, rhs],
            vec![2, 4],
        );
        let second = graph.add(
            UOpKind::MatMul { m: 2, k: 3, n: 4 },
            vec![lhs, rhs],
            vec![2, 4],
        );
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![second],
            vec![2, 4],
        );
        let optimized = graph.optimize().unwrap();
        assert_eq!(optimized.ops[second.0 as usize].src.len(), 0);
        assert_eq!(optimized.ops.last().unwrap().src, vec![first]);
    }

    #[test]
    fn optimizer_cse_reuses_duplicate_axis_reduction() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 4]);
        let first = graph.add(UOpKind::ReduceSumAxis { axis: 1 }, vec![input], vec![2]);
        let second = graph.add(UOpKind::ReduceSumAxis { axis: 1 }, vec![input], vec![2]);
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![second],
            vec![2],
        );
        let optimized = graph.optimize().unwrap();
        assert!(
            matches!(optimized.ops[second.0 as usize].kind, UOpKind::Const { value } if value == 0.0)
        );
        assert_eq!(optimized.ops.last().unwrap().src, vec![first]);
    }

    #[test]
    fn optimizer_folds_constant_conv2d() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Const { value: 2.0 }, vec![], vec![1, 2, 3, 3]);
        let weight = graph.add(UOpKind::Const { value: 3.0 }, vec![], vec![4, 2, 1, 1]);
        let bias = graph.add(UOpKind::Const { value: 5.0 }, vec![], vec![4]);
        let conv = graph.add(
            UOpKind::Conv2d {
                batch: 1,
                in_channels: 2,
                height: 3,
                width: 3,
                out_channels: 4,
                kernel_h: 1,
                kernel_w: 1,
                stride: 1,
                padding: 0,
            },
            vec![input, weight, bias],
            vec![1, 4, 3, 3],
        );
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![conv],
            vec![1, 4, 3, 3],
        );
        let optimized = graph.optimize().unwrap();
        assert!(matches!(
            optimized.ops[conv.0 as usize].kind,
            UOpKind::Const { value } if value == 17.0
        ));
    }

    #[test]
    fn optimizer_cse_reuses_duplicate_conv2d() {
        let mut graph = TinyGraph::default();
        let input = graph.add(
            UOpKind::Input { name: "x".into() },
            vec![],
            vec![1, 2, 3, 3],
        );
        let weight = graph.add(
            UOpKind::Input { name: "w".into() },
            vec![],
            vec![4, 2, 1, 1],
        );
        let bias = graph.add(UOpKind::Input { name: "b".into() }, vec![], vec![4]);
        let kind = UOpKind::Conv2d {
            batch: 1,
            in_channels: 2,
            height: 3,
            width: 3,
            out_channels: 4,
            kernel_h: 1,
            kernel_w: 1,
            stride: 1,
            padding: 0,
        };
        let first = graph.add(kind.clone(), vec![input, weight, bias], vec![1, 4, 3, 3]);
        let second = graph.add(kind, vec![input, weight, bias], vec![1, 4, 3, 3]);
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![second],
            vec![1, 4, 3, 3],
        );
        let optimized = graph.optimize().unwrap();
        assert!(
            matches!(optimized.ops[second.0 as usize].kind, UOpKind::Const { value } if value == 0.0)
        );
        assert_eq!(optimized.ops.last().unwrap().src, vec![first]);
    }

    #[test]
    fn optimizer_folds_constant_gather() {
        let mut graph = TinyGraph::default();
        let weight = graph.add(UOpKind::Const { value: 8.0 }, vec![], vec![16, 4]);
        let indices = graph.add(UOpKind::Const { value: 3.0 }, vec![], vec![2]);
        let gather = graph.add(
            UOpKind::Gather {
                rows: 2,
                vocab: 16,
                features: 4,
            },
            vec![weight, indices],
            vec![2, 4],
        );
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![gather],
            vec![2, 4],
        );
        let optimized = graph.optimize().unwrap();
        assert!(matches!(
            optimized.ops[gather.0 as usize].kind,
            UOpKind::Const { value } if value == 8.0
        ));
    }

    #[test]
    fn optimizer_folds_constant_normalization() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Const { value: 2.0 }, vec![], vec![1, 4]);
        let weight = graph.add(UOpKind::Const { value: 3.0 }, vec![], vec![4]);
        let rms = graph.add(
            UOpKind::RmsNorm {
                rows: 1,
                features: 4,
                epsilon: 0.0,
            },
            vec![input, weight],
            vec![1, 4],
        );
        let bias = graph.add(UOpKind::Const { value: 5.0 }, vec![], vec![4]);
        let layer = graph.add(
            UOpKind::LayerNorm {
                rows: 1,
                features: 4,
                epsilon: 1e-5,
            },
            vec![input, weight, bias],
            vec![1, 4],
        );
        graph.add(
            UOpKind::Output { name: "rms".into() },
            vec![rms],
            vec![1, 4],
        );
        graph.add(
            UOpKind::Output {
                name: "layer".into(),
            },
            vec![layer],
            vec![1, 4],
        );
        let optimized = graph.optimize().unwrap();
        assert!(
            matches!(optimized.ops[rms.0 as usize].kind, UOpKind::Const { value } if value == 3.0)
        );
        assert!(
            matches!(optimized.ops[layer.0 as usize].kind, UOpKind::Const { value } if value == 5.0)
        );
    }

    #[test]
    fn reference_executor_supports_trailing_dimension_broadcast() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
        let bias = graph.add(
            UOpKind::Input {
                name: "bias".into(),
            },
            vec![],
            vec![3],
        );
        let y = graph.add(UOpKind::Add, vec![x, bias], vec![2, 3]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![y], vec![2, 3]);
        let inputs = BTreeMap::from([
            ("x".into(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ("bias".into(), vec![10.0, 20.0, 30.0]),
        ]);
        let outputs = graph.execute_f32(&inputs).unwrap();
        assert_eq!(outputs["y"], vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn validation_accepts_broadcast_where_and_rejects_incompatible_shapes() {
        let mut graph = TinyGraph::default();
        let condition = graph.add(
            UOpKind::Input {
                name: "condition".into(),
            },
            vec![],
            vec![2, 1],
        );
        let when_true = graph.add(
            UOpKind::Input {
                name: "true".into(),
            },
            vec![],
            vec![2, 3],
        );
        let when_false = graph.add(
            UOpKind::Input {
                name: "false".into(),
            },
            vec![],
            vec![3],
        );
        let selected = graph.add(
            UOpKind::Where,
            vec![condition, when_true, when_false],
            vec![2, 3],
        );
        graph.add(
            UOpKind::Output { name: "y".into() },
            vec![selected],
            vec![2, 3],
        );
        assert!(graph.validate().is_ok());

        let mut invalid = graph.clone();
        invalid.ops[2].shape = vec![4];
        assert!(matches!(
            invalid.validate(),
            Err(GraphError::ShapeMismatch(_))
        ));
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert_eq!(capture.kernels.len(), 1);
        assert!(capture.kernels[0].source.contains("prism_where"));
        assert!(capture.kernels[0].source.contains("cc0"));
    }

    #[test]
    fn lowering_supports_shape_aware_broadcast_kernel_abi() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
        let bias = graph.add(
            UOpKind::Input {
                name: "bias".into(),
            },
            vec![],
            vec![3],
        );
        let y = graph.add(UOpKind::Add, vec![x, bias], vec![2, 3]);
        let y = graph.add(UOpKind::Relu, vec![y], vec![2, 3]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![y], vec![2, 3]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert_eq!(capture.kernels.len(), 1);
        assert!(capture.kernels[0].source.contains("prism_broadcast_binary"));
        assert!(capture.kernels[0].source.contains("max(value, 0.0f)"));
    }

    #[test]
    fn lowering_separates_where_after_broadcast_binary() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
        let bias = graph.add(
            UOpKind::Input {
                name: "bias".into(),
            },
            vec![],
            vec![3],
        );
        let condition = graph.add(
            UOpKind::Input {
                name: "condition".into(),
            },
            vec![],
            vec![2, 1],
        );
        let added = graph.add(UOpKind::Add, vec![x, bias], vec![2, 3]);
        let selected = graph.add(UOpKind::Where, vec![condition, added, x], vec![2, 3]);
        graph.add(
            UOpKind::Output { name: "y".into() },
            vec![selected],
            vec![2, 3],
        );
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert_eq!(capture.kernels.len(), 2);
        assert!(capture.kernels[0].source.contains("prism_broadcast_binary"));
        assert!(capture.kernels[1].source.contains("prism_where"));
        assert!(capture
            .kernels
            .iter()
            .all(|kernel| !kernel.source.is_empty()));
    }

    #[test]
    fn lowering_separates_cast_after_broadcast_binary() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
        let bias = graph.add(
            UOpKind::Input {
                name: "bias".into(),
            },
            vec![],
            vec![3],
        );
        let added = graph.add(UOpKind::Add, vec![x, bias], vec![2, 3]);
        let cast = graph.add(
            UOpKind::Cast {
                from: "f32".into(),
                to: "i8".into(),
            },
            vec![added],
            vec![2, 3],
        );
        graph.add(UOpKind::Output { name: "y".into() }, vec![cast], vec![2, 3]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert_eq!(capture.kernels.len(), 2);
        assert!(capture
            .kernels
            .iter()
            .all(|kernel| !kernel.source.is_empty()));
        assert!(capture.kernels[1].source.contains("prism_cast"));
    }

    #[test]
    fn reshape_is_a_kernel_free_row_major_alias() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
        let reshaped = graph.add(UOpKind::Reshape, vec![input], vec![3, 2]);
        graph.add(
            UOpKind::Output { name: "y".into() },
            vec![reshaped],
            vec![3, 2],
        );
        let mut inputs = BTreeMap::new();
        inputs.insert("x".into(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            graph.execute_f32(&inputs).unwrap()["y"],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        );
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert!(capture.kernels.is_empty());
        assert_eq!(capture.memory_plan.slot_count, 0);
    }

    #[test]
    fn reshape_rejects_element_count_changes() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
        graph.add(UOpKind::Reshape, vec![input], vec![4, 2]);
        assert!(matches!(
            graph.validate(),
            Err(GraphError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn optimizer_folds_constant_extrema() {
        let mut graph = TinyGraph::default();
        let left = graph.add(UOpKind::Const { value: -2.0 }, vec![], vec![1]);
        let right = graph.add(UOpKind::Const { value: 3.0 }, vec![], vec![1]);
        let maximum = graph.add(UOpKind::Maximum, vec![left, right], vec![1]);
        let minimum = graph.add(UOpKind::Minimum, vec![left, right], vec![1]);
        graph.add(
            UOpKind::Output {
                name: "maximum".into(),
            },
            vec![maximum],
            vec![1],
        );
        graph.add(
            UOpKind::Output {
                name: "minimum".into(),
            },
            vec![minimum],
            vec![1],
        );
        let optimized = graph.optimize().unwrap();
        assert!(matches!(
            optimized.ops[maximum.0 as usize].kind,
            UOpKind::Const { value } if value == 3.0
        ));
        assert!(matches!(
            optimized.ops[minimum.0 as usize].kind,
            UOpKind::Const { value } if value == -2.0
        ));
    }

    #[test]
    fn optimizer_eliminates_duplicate_elementwise_subexpressions() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let first = graph.add(UOpKind::Relu, vec![x], vec![2]);
        let second = graph.add(UOpKind::Relu, vec![x], vec![2]);
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![second],
            vec![2],
        );
        let optimized = graph.optimize().unwrap();
        assert!(matches!(
            optimized.ops[second.0 as usize].kind,
            UOpKind::Const { value } if value == 0.0
        ));
        assert_eq!(optimized.ops[graph.ops.len() - 1].src, vec![first]);
        let mut inputs = BTreeMap::new();
        inputs.insert("x".into(), vec![-1.0, 2.0]);
        assert_eq!(
            optimized.execute_f32(&inputs).unwrap()["out"],
            vec![0.0, 2.0]
        );
    }

    #[test]
    fn lowering_prunes_dead_rewritten_uops_from_capture_graph() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let first = graph.add(UOpKind::Relu, vec![x], vec![2]);
        let duplicate = graph.add(UOpKind::Relu, vec![x], vec![2]);
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![duplicate],
            vec![2],
        );
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert!(capture.graph.ops.iter().any(|op| op.id == first));
        assert!(!capture.graph.ops.iter().any(|op| op.id == duplicate));
        assert!(capture.graph_op_count < graph.ops.len());
    }

    #[test]
    fn capture_digest_is_deterministic() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![1]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![1]);
        let left = graph.lower(LoweringTarget::Portable).unwrap();
        let right = graph.lower(LoweringTarget::Portable).unwrap();
        assert_eq!(left.digest(), right.digest());
        assert_eq!(left.digest().len(), 64);
    }

    #[test]
    fn memory_plan_assigns_reusable_slots() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
        let first = graph.add(UOpKind::Relu, vec![input], vec![1]);
        let second = graph.add(UOpKind::Relu, vec![first], vec![1]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![second], vec![1]);
        let plan = graph.memory_plan().unwrap();
        assert_eq!(plan.allocations.len(), 2);
        assert_eq!(plan.slot_count, 2);
        assert!(plan.allocations[0].last_command <= plan.allocations[1].first_command);
    }

    #[test]
    fn memory_plan_does_not_reuse_a_slot_for_a_larger_value() {
        let mut graph = TinyGraph::default();
        let narrow = graph.add(
            UOpKind::Input {
                name: "narrow".into(),
            },
            vec![],
            vec![2],
        );
        let reduce = graph.add(UOpKind::ReduceSum, vec![narrow], vec![1]);
        let wide = graph.add(
            UOpKind::Input {
                name: "wide".into(),
            },
            vec![],
            vec![4],
        );
        let expanded = graph.add(UOpKind::Add, vec![reduce, wide], vec![4]);
        let output = graph.add(UOpKind::Relu, vec![expanded], vec![4]);
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![output],
            vec![4],
        );

        let plan = graph.memory_plan().unwrap();
        assert_eq!(plan.allocations[0].elements, 1);
        assert_eq!(plan.allocations[1].elements, 4);
        assert_eq!(plan.allocations[2].elements, 4);
        assert_ne!(plan.allocations[0].slot, plan.allocations[2].slot);
    }

    #[test]
    fn validation_rejects_incompatible_elementwise_shapes() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let y = graph.add(UOpKind::Input { name: "y".into() }, vec![], vec![3]);
        graph.add(UOpKind::Add, vec![x, y], vec![2]);
        assert!(matches!(
            graph.validate(),
            Err(GraphError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn validation_rejects_empty_and_zero_sized_uop_shapes() {
        let mut empty = TinyGraph::default();
        empty.add(
            UOpKind::Input {
                name: "empty".into(),
            },
            vec![],
            vec![],
        );
        assert!(matches!(
            empty.validate(),
            Err(GraphError::ShapeMismatch(_))
        ));

        let mut zero = TinyGraph::default();
        zero.add(
            UOpKind::Input {
                name: "zero".into(),
            },
            vec![],
            vec![2, 0],
        );
        assert!(matches!(zero.validate(), Err(GraphError::ShapeMismatch(_))));

        let mut overflowing = TinyGraph::default();
        overflowing.add(
            UOpKind::Input {
                name: "overflowing".into(),
            },
            vec![],
            vec![usize::MAX as u64, 2],
        );
        assert!(matches!(
            overflowing.validate(),
            Err(GraphError::ShapeMismatch(_))
        ));

        let mut mismatched_output = TinyGraph::default();
        let input = mismatched_output.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        mismatched_output.add(UOpKind::Output { name: "out".into() }, vec![input], vec![1]);
        assert!(matches!(
            mismatched_output.validate(),
            Err(GraphError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn validation_rejects_equal_size_but_non_broadcastable_shapes() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 2]);
        let y = graph.add(UOpKind::Input { name: "y".into() }, vec![], vec![4]);
        graph.add(UOpKind::Add, vec![x, y], vec![2, 2]);
        assert!(matches!(
            graph.validate(),
            Err(GraphError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn validation_rejects_output_shape_mismatch() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
        graph.add(UOpKind::Output { name: "out".into() }, vec![input], vec![5]);
        assert!(matches!(
            graph.validate(),
            Err(GraphError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn validation_rejects_malformed_layer_norm_parameters() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
        let weight = graph.add(UOpKind::Input { name: "w".into() }, vec![], vec![2]);
        let bias = graph.add(UOpKind::Input { name: "b".into() }, vec![], vec![2]);
        let norm = graph.add(
            UOpKind::LayerNorm {
                rows: 2,
                features: 3,
                epsilon: 1e-5,
            },
            vec![input, weight, bias],
            vec![2, 3],
        );
        graph.add(UOpKind::Output { name: "y".into() }, vec![norm], vec![2, 3]);
        assert!(matches!(
            graph.validate(),
            Err(GraphError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn validation_rejects_duplicate_ids_and_output_names() {
        let duplicate = TinyGraph {
            ops: vec![
                UOp {
                    id: UOpId(0),
                    kind: UOpKind::Input { name: "x".into() },
                    src: vec![],
                    shape: vec![1],
                },
                UOp {
                    id: UOpId(0),
                    kind: UOpKind::Input { name: "y".into() },
                    src: vec![],
                    shape: vec![1],
                },
            ],
        };
        assert!(matches!(
            duplicate.validate(),
            Err(GraphError::DuplicateId(UOpId(0)))
        ));

        let mut outputs = TinyGraph::default();
        let input = outputs.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
        outputs.add(UOpKind::Output { name: "y".into() }, vec![input], vec![1]);
        outputs.add(UOpKind::Output { name: "y".into() }, vec![input], vec![1]);
        assert!(matches!(
            outputs.validate(),
            Err(GraphError::DuplicateOutput(_))
        ));

        let mut inputs = TinyGraph::default();
        inputs.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
        inputs.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
        assert!(matches!(
            inputs.validate(),
            Err(GraphError::DuplicateInput(_))
        ));

        let mut empty_input = TinyGraph::default();
        empty_input.add(
            UOpKind::Input {
                name: String::new(),
            },
            vec![],
            vec![1],
        );
        assert!(matches!(
            empty_input.validate(),
            Err(GraphError::EmptyInputName)
        ));

        let mut empty_output = TinyGraph::default();
        let input = empty_output.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
        empty_output.add(
            UOpKind::Output {
                name: String::new(),
            },
            vec![input],
            vec![1],
        );
        assert!(matches!(
            empty_output.validate(),
            Err(GraphError::EmptyOutputName)
        ));
    }

    #[test]
    fn capture_validation_checks_replay_and_memory_invariants() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![1]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![1]);
        let mut capture = graph.lower(LoweringTarget::Portable).unwrap();
        capture.replay.synchronization_points.push(99);
        assert!(capture.validate().is_err());
    }

    #[test]
    fn capture_validation_rejects_tampered_graph_operation_count() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![1]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![1]);
        let mut capture = graph.lower(LoweringTarget::Portable).unwrap();
        capture.graph_op_count += 1;
        let error = capture.validate().unwrap_err();
        assert!(error.contains("operation count mismatch"));
    }

    #[test]
    fn capture_validation_rejects_noncanonical_command_ids() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![1]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![1]);
        let mut capture = graph.lower(LoweringTarget::Portable).unwrap();
        capture.replay.command_ids[0] = 7;
        let error = capture.validate().unwrap_err();
        assert!(error.contains("canonical"));
    }

    #[test]
    fn capture_validation_rejects_noncanonical_synchronization_points() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![1]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![1]);
        let mut capture = graph.lower(LoweringTarget::Portable).unwrap();
        capture.replay.synchronization_points = vec![0, 0];
        let error = capture.validate().unwrap_err();
        assert!(error.contains("synchronization points are not canonical"));
    }

    #[test]
    fn capture_validation_rejects_tampered_kernel_source() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![1]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![1]);
        let mut capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert!(capture.validate().is_ok());
        capture.kernels[0].source.push_str(" tampered");
        assert!(capture.validate().is_err());
    }

    #[test]
    fn capture_validation_rejects_empty_kernel_source() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![1]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![1]);
        let mut capture = graph.lower(LoweringTarget::Portable).unwrap();
        capture.kernels[0].source.clear();
        let mut digest = Sha256::new();
        digest.update(capture.kernels[0].source.as_bytes());
        capture.kernels[0].source_digest = hex_digest(digest.finalize());
        let error = capture.validate().unwrap_err();
        assert!(error.contains("empty rendered source"));
    }

    #[test]
    fn capture_validation_rejects_tampered_output_geometry() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let relu = graph.add(UOpKind::Relu, vec![input], vec![2]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![2]);
        let mut capture = graph.lower(LoweringTarget::Portable).unwrap();
        capture.kernels[0].output_elements = Some(3);
        assert!(capture.validate().is_err());
    }

    #[test]
    fn renderer_preserves_scalar_operand_order() {
        let mut graph = TinyGraph::default();
        let two = graph.add(UOpKind::Const { value: 2.0 }, vec![], vec![1]);
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
        let sub = graph.add(UOpKind::Sub, vec![two, x], vec![1]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![sub], vec![1]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("v = 2 - v"));
    }

    #[test]
    fn negation_fuses_and_executes() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let neg = graph.add(UOpKind::Neg, vec![input], vec![2]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![neg], vec![2]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert_eq!(capture.kernels[0].output_elements, Some(2));
        assert!(capture.kernels[0].source.contains("v = -v"));
        let mut inputs = BTreeMap::new();
        inputs.insert("x".into(), vec![-2.0, 3.0]);
        assert_eq!(graph.execute_f32(&inputs).unwrap()["y"], vec![2.0, -3.0]);
    }

    #[test]
    fn exp_and_sqrt_render_and_execute() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
        let exp = graph.add(UOpKind::Exp, vec![input], vec![1]);
        let root = graph.add(UOpKind::Sqrt, vec![exp], vec![1]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![root], vec![1]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("expf(v)"));
        assert!(capture.kernels[0].source.contains("sqrtf(v)"));
        let mut inputs = BTreeMap::new();
        inputs.insert("x".into(), vec![0.0]);
        let value = graph.execute_f32(&inputs).unwrap()["y"][0];
        assert!((value - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pow_renders_and_executes_scalar_exponent() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![3]);
        let pow = graph.add(UOpKind::Pow { exponent: 2.0 }, vec![input], vec![3]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![pow], vec![3]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("powf(v, 2f)"));
        let output = graph
            .execute_f32(&BTreeMap::from([("x".into(), vec![-2.0, 3.0, 0.5])]))
            .unwrap();
        assert_eq!(output["y"], vec![4.0, 9.0, 0.25]);
    }

    #[test]
    fn sin_and_cos_render_and_execute() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let sin = graph.add(UOpKind::Sin, vec![input], vec![2]);
        let cos = graph.add(UOpKind::Cos, vec![sin], vec![2]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![cos], vec![2]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("sinf(v)"));
        assert!(capture.kernels[0].source.contains("cosf(v)"));
        let mut inputs = BTreeMap::new();
        inputs.insert("x".into(), vec![0.0, std::f32::consts::FRAC_PI_2]);
        let output = graph.execute_f32(&inputs).unwrap();
        assert!((output["y"][0] - 1.0).abs() < 1e-6);
        assert!((output["y"][1] - 1.0f32.cos()).abs() < 1e-6);
    }

    #[test]
    fn renderer_declares_rhs_for_two_input_elementwise_ops() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let y = graph.add(UOpKind::Input { name: "y".into() }, vec![], vec![2]);
        let sum = graph.add(UOpKind::Add, vec![x, y], vec![2]);
        graph.add(UOpKind::Output { name: "z".into() }, vec![sum], vec![2]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("float* rhs"));
        assert!(capture.kernels[0].source.contains("rhs[id]"));
    }

    #[test]
    fn maximum_and_minimum_render_execute_and_preserve_rhs_abi() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![3]);
        let y = graph.add(UOpKind::Input { name: "y".into() }, vec![], vec![3]);
        let upper = graph.add(UOpKind::Maximum, vec![x, y], vec![3]);
        let lower = graph.add(UOpKind::Minimum, vec![upper, y], vec![3]);
        graph.add(UOpKind::Output { name: "out".into() }, vec![lower], vec![3]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert!(capture
            .kernels
            .iter()
            .all(|kernel| kernel.group.requires_rhs()));
        assert!(
            capture
                .kernels
                .iter()
                .any(|kernel| kernel.source.contains("max(v, rhs[id])")),
            "{:#?}",
            capture.kernels
        );
        assert!(
            capture
                .kernels
                .iter()
                .any(|kernel| kernel.source.contains("min(v, rhs[id])")),
            "{:#?}",
            capture.kernels
        );
        let outputs = graph
            .execute_f32(&BTreeMap::from([
                ("x".into(), vec![-2.0, 4.0, 1.0]),
                ("y".into(), vec![1.0, 3.0, 2.0]),
            ]))
            .unwrap();
        assert_eq!(outputs["out"], vec![1.0, 3.0, 2.0]);
    }

    #[test]
    fn unary_math_family_renders_and_executes() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
        let abs = graph.add(UOpKind::Abs, vec![x], vec![2]);
        let log = graph.add(UOpKind::Log, vec![abs], vec![2]);
        let tanh = graph.add(UOpKind::Tanh, vec![log], vec![2]);
        graph.add(UOpKind::Output { name: "out".into() }, vec![tanh], vec![2]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        let source = &capture.kernels[0].source;
        assert!(source.contains("fabsf(v)"));
        assert!(source.contains("logf(v)"));
        assert!(source.contains("tanhf(v)"));
        let outputs = graph
            .execute_f32(&BTreeMap::from([("x".into(), vec![-1.0, 2.0])]))
            .unwrap();
        assert!((outputs["out"][0] - 0.0).abs() < 1e-6);
        assert!((outputs["out"][1] - 2.0f32.ln().tanh()).abs() < 1e-6);
    }

    #[test]
    fn reduce_sum_has_shape_changing_kernel_and_reference_behavior() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
        let sum = graph.add(UOpKind::ReduceSum, vec![x], vec![1]);
        graph.add(UOpKind::Output { name: "sum".into() }, vec![sum], vec![1]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert_eq!(capture.kernels.len(), 1);
        assert!(capture.kernels[0]
            .source
            .contains("for (unsigned i = 0; i < 4; ++i)"));
        assert!(capture.kernels[0].source.contains("output[0] = v"));
        let outputs = graph
            .execute_f32(&BTreeMap::from([("x".into(), vec![1.0, -2.0, 3.0, 4.0])]))
            .unwrap();
        assert_eq!(outputs["sum"], vec![6.0]);
    }

    #[test]
    fn reduce_max_has_dedicated_kernel_and_reference_behavior() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
        let maximum = graph.add(UOpKind::ReduceMax, vec![input], vec![1]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![maximum], vec![1]);
        let mut inputs = BTreeMap::new();
        inputs.insert("x".into(), vec![-2.0, 7.0, 3.0, 1.0]);
        assert_eq!(graph.execute_f32(&inputs).unwrap()["y"], vec![7.0]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("prism_reduce_max"));
    }

    #[test]
    fn reduce_min_has_dedicated_kernel_and_reference_behavior() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
        let minimum = graph.add(UOpKind::ReduceMin, vec![input], vec![1]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![minimum], vec![1]);
        let mut inputs = BTreeMap::new();
        inputs.insert("x".into(), vec![-2.0, 7.0, 3.0, 1.0]);
        assert_eq!(graph.execute_f32(&inputs).unwrap()["y"], vec![-2.0]);
        assert!(graph.lower(LoweringTarget::Portable).unwrap().kernels[0]
            .source
            .contains("prism_reduce_min"));
    }

    #[test]
    fn reduce_max_axis_has_dedicated_kernel_and_reference_behavior() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
        let maximum = graph.add(UOpKind::ReduceMaxAxis { axis: 1 }, vec![input], vec![2]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![maximum], vec![2]);
        let mut inputs = BTreeMap::new();
        inputs.insert("x".into(), vec![1.0, 7.0, 3.0, 9.0, 2.0, 4.0]);
        assert_eq!(graph.execute_f32(&inputs).unwrap()["y"], vec![7.0, 9.0]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("prism_reduce_max_axis"));
    }

    #[test]
    fn reduce_min_axis_has_dedicated_kernel_and_reference_behavior() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
        let minimum = graph.add(UOpKind::ReduceMinAxis { axis: 1 }, vec![input], vec![2]);
        graph.add(UOpKind::Output { name: "y".into() }, vec![minimum], vec![2]);
        let mut inputs = BTreeMap::new();
        inputs.insert("x".into(), vec![1.0, 7.0, 3.0, 9.0, 2.0, 4.0]);
        assert_eq!(graph.execute_f32(&inputs).unwrap()["y"], vec![1.0, 2.0]);
        assert!(graph.lower(LoweringTarget::Portable).unwrap().kernels[0]
            .source
            .contains("prism_reduce_min_axis"));
    }

    #[test]
    fn axis_reduction_rejects_same_element_count_wrong_shape() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 1]);
        let maximum = graph.add(UOpKind::ReduceMaxAxis { axis: 1 }, vec![input], vec![2, 1]);
        graph.add(
            UOpKind::Output { name: "y".into() },
            vec![maximum],
            vec![2, 1],
        );
        assert_eq!(graph.validate(), Err(GraphError::ShapeMismatch(maximum)));
    }

    #[test]
    fn matmul_validates_renders_and_executes() {
        let mut graph = TinyGraph::default();
        let a = graph.add(UOpKind::Input { name: "a".into() }, vec![], vec![2, 3]);
        let b = graph.add(UOpKind::Input { name: "b".into() }, vec![], vec![3, 2]);
        let product = graph.add(UOpKind::MatMul { m: 2, k: 3, n: 2 }, vec![a, b], vec![2, 2]);
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![product],
            vec![2, 2],
        );
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert_eq!(capture.kernels.len(), 1);
        assert!(capture.kernels[0].source.contains("prism_matmul"));
        assert!(capture.kernels[0].source.contains("inner < 3u"));
        let outputs = graph
            .execute_f32(&BTreeMap::from([
                ("a".into(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                ("b".into(), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
            ]))
            .unwrap();
        assert_eq!(outputs["out"], vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn axis_reduction_validates_renders_and_executes() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
        let sum = graph.add(UOpKind::ReduceSumAxis { axis: 1 }, vec![x], vec![2]);
        graph.add(UOpKind::Output { name: "sum".into() }, vec![sum], vec![2]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("prism_reduce_sum_axis"));
        assert!(capture.kernels[0].source.contains("step < 3u"));
        let outputs = graph
            .execute_f32(&BTreeMap::from([(
                "x".into(),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            )]))
            .unwrap();
        assert_eq!(outputs["sum"], vec![6.0, 15.0]);
    }

    #[test]
    fn softmax_axis_is_stable_and_normalizes_each_row() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
        let softmax = graph.add(UOpKind::SoftmaxAxis { axis: 1 }, vec![x], vec![2, 3]);
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![softmax],
            vec![2, 3],
        );
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("prism_softmax_axis"));
        let output = graph
            .execute_f32(&BTreeMap::from([(
                "x".into(),
                vec![1000.0, 1001.0, 1002.0, 0.0, 0.0, 0.0],
            )]))
            .unwrap();
        for row in output["out"].chunks_exact(3) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
        assert!(output["out"].iter().all(|value| value.is_finite()));
    }

    #[test]
    fn fused_attention_matches_scaled_dot_product_reference() {
        let mut graph = TinyGraph::default();
        let q = graph.add(UOpKind::Input { name: "q".into() }, vec![], vec![2, 2]);
        let k = graph.add(UOpKind::Input { name: "k".into() }, vec![], vec![2, 2]);
        let v = graph.add(UOpKind::Input { name: "v".into() }, vec![], vec![2, 2]);
        let attention = graph.add(
            UOpKind::Attention {
                seq: 2,
                head: 2,
                scale: 0.5,
            },
            vec![q, k, v],
            vec![2, 2],
        );
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![attention],
            vec![2, 2],
        );
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("prism_attention"));
        let output = graph
            .execute_f32(&BTreeMap::from([
                ("q".into(), vec![1.0, 0.0, 0.0, 1.0]),
                ("k".into(), vec![1.0, 0.0, 0.0, 1.0]),
                ("v".into(), vec![2.0, 4.0, 8.0, 16.0]),
            ]))
            .unwrap();
        assert!(output["out"].iter().all(|value| value.is_finite()));
        assert!(output["out"][0] > 2.0 && output["out"][0] < 8.0);
        assert!(output["out"][1] > 4.0 && output["out"][1] < 16.0);
    }

    #[test]
    fn batched_attention_preserves_batch_independence() {
        let mut graph = TinyGraph::default();
        let q = graph.add(UOpKind::Input { name: "q".into() }, vec![], vec![2, 2, 1]);
        let k = graph.add(UOpKind::Input { name: "k".into() }, vec![], vec![2, 2, 1]);
        let v = graph.add(UOpKind::Input { name: "v".into() }, vec![], vec![2, 2, 1]);
        let attention = graph.add(
            UOpKind::AttentionBatched {
                batch: 2,
                seq: 2,
                head: 1,
                scale: 1.0,
            },
            vec![q, k, v],
            vec![2, 2, 1],
        );
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![attention],
            vec![2, 2, 1],
        );
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0]
            .source
            .contains("prism_attention_batched"));
        let output = graph
            .execute_f32(&BTreeMap::from([
                ("q".into(), vec![1.0, 0.0, 0.0, 1.0]),
                ("k".into(), vec![1.0, 0.0, 0.0, 1.0]),
                ("v".into(), vec![2.0, 4.0, 8.0, 16.0]),
            ]))
            .unwrap();
        assert!(output["out"].iter().all(|value| value.is_finite()));
        assert!(output["out"][0] < output["out"][2]);
    }

    #[test]
    fn gelu_renders_and_matches_reference_approximation() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![3]);
        let gelu = graph.add(UOpKind::Gelu, vec![x], vec![3]);
        graph.add(UOpKind::Output { name: "out".into() }, vec![gelu], vec![3]);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("tanhf"));
        let output = graph
            .execute_f32(&BTreeMap::from([("x".into(), vec![-1.0, 0.0, 1.0])]))
            .unwrap();
        assert!(output["out"][1].abs() < 1e-6);
        assert!(output["out"][0] < 0.0 && output["out"][2] > 0.0);
    }

    #[test]
    fn rms_norm_renders_and_matches_reference() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 2]);
        let weight = graph.add(
            UOpKind::Input {
                name: "weight".into(),
            },
            vec![],
            vec![2],
        );
        let norm = graph.add(
            UOpKind::RmsNorm {
                rows: 2,
                features: 2,
                epsilon: 1e-5,
            },
            vec![x, weight],
            vec![2, 2],
        );
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![norm],
            vec![2, 2],
        );
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("prism_rms_norm"));
        let output = graph
            .execute_f32(&BTreeMap::from([
                ("x".into(), vec![3.0, 4.0, 0.0, 2.0]),
                ("weight".into(), vec![2.0, 0.5]),
            ]))
            .unwrap();
        assert!((output["out"][0] - 1.697056).abs() < 1e-3);
        assert!((output["out"][1] - 0.565685).abs() < 1e-3);
    }

    #[test]
    fn layer_norm_renders_and_matches_reference() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1, 2]);
        let weight = graph.add(
            UOpKind::Input {
                name: "weight".into(),
            },
            vec![],
            vec![2],
        );
        let bias = graph.add(
            UOpKind::Input {
                name: "bias".into(),
            },
            vec![],
            vec![2],
        );
        let norm = graph.add(
            UOpKind::LayerNorm {
                rows: 1,
                features: 2,
                epsilon: 1e-5,
            },
            vec![x, weight, bias],
            vec![1, 2],
        );
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![norm],
            vec![1, 2],
        );
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("prism_layer_norm"));
        let output = graph
            .execute_f32(&BTreeMap::from([
                ("x".into(), vec![1.0, 3.0]),
                ("weight".into(), vec![2.0, 0.5]),
                ("bias".into(), vec![1.0, -1.0]),
            ]))
            .unwrap();
        assert!((output["out"][0] + 1.0).abs() < 1e-3);
        assert!((output["out"][1] + 0.5).abs() < 1e-3);
    }

    #[test]
    fn rope_validates_and_matches_reference_rotation() {
        let mut graph = TinyGraph::default();
        let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1, 4]);
        let cos = graph.add(UOpKind::Input { name: "cos".into() }, vec![], vec![1, 2]);
        let sin = graph.add(UOpKind::Input { name: "sin".into() }, vec![], vec![1, 2]);
        let rope = graph.add(
            UOpKind::Rope {
                rows: 1,
                features: 4,
            },
            vec![x, cos, sin],
            vec![1, 4],
        );
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![rope],
            vec![1, 4],
        );
        let mut inputs = BTreeMap::new();
        inputs.insert("x".into(), vec![1.0, 2.0, 3.0, 4.0]);
        inputs.insert("cos".into(), vec![0.0, 1.0]);
        inputs.insert("sin".into(), vec![1.0, 0.0]);
        let outputs = graph.execute_f32(&inputs).unwrap();
        assert_eq!(outputs["out"], vec![-2.0, 1.0, 3.0, 4.0]);
    }

    #[test]
    fn transpose_executes_nontrivial_rank_three_permutation() {
        let mut graph = TinyGraph::default();
        let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3, 4]);
        let transpose = graph.add(
            UOpKind::Transpose {
                permutation: vec![2, 0, 1],
            },
            vec![input],
            vec![4, 2, 3],
        );
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![transpose],
            vec![4, 2, 3],
        );
        let input_values = (0..24).map(|value| value as f32).collect::<Vec<_>>();
        let output = graph
            .execute_f32(&BTreeMap::from([(String::from("x"), input_values)]))
            .unwrap();
        let expected = (0..4)
            .flat_map(|c| {
                (0..2).flat_map(move |a| (0..3).map(move |b| (a * 12 + b * 4 + c) as f32))
            })
            .collect::<Vec<_>>();
        assert_eq!(output["out"], expected);
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("prism_transpose"));
    }

    #[test]
    fn gather_validates_and_executes_embedding_lookup() {
        let mut graph = TinyGraph::default();
        let weight = graph.add(
            UOpKind::Input {
                name: "weight".into(),
            },
            vec![],
            vec![3, 2],
        );
        let indices = graph.add(
            UOpKind::Input {
                name: "indices".into(),
            },
            vec![],
            vec![2],
        );
        let gather = graph.add(
            UOpKind::Gather {
                rows: 2,
                vocab: 3,
                features: 2,
            },
            vec![weight, indices],
            vec![2, 2],
        );
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![gather],
            vec![2, 2],
        );
        let mut inputs = BTreeMap::new();
        inputs.insert("weight".into(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        inputs.insert("indices".into(), vec![2.0, 0.0]);
        assert_eq!(
            graph.execute_f32(&inputs).unwrap()["out"],
            vec![5.0, 6.0, 1.0, 2.0]
        );
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("prism_gather"));
        assert!(capture.validate().is_ok());
    }

    #[test]
    fn scatter_validates_executes_and_uses_last_update() {
        let mut graph = TinyGraph::default();
        let base = graph.add(
            UOpKind::Input {
                name: "base".into(),
            },
            vec![],
            vec![3, 2],
        );
        let indices = graph.add(
            UOpKind::Input {
                name: "indices".into(),
            },
            vec![],
            vec![2],
        );
        let updates = graph.add(
            UOpKind::Input {
                name: "updates".into(),
            },
            vec![],
            vec![2, 2],
        );
        let scatter = graph.add(
            UOpKind::Scatter {
                rows: 3,
                updates: 2,
                features: 2,
            },
            vec![base, indices, updates],
            vec![3, 2],
        );
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![scatter],
            vec![3, 2],
        );
        let inputs = BTreeMap::from([
            ("base".into(), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]),
            ("indices".into(), vec![1.0, 1.0]),
            ("updates".into(), vec![7.0, 8.0, 9.0, 10.0]),
        ]);
        assert_eq!(
            graph.execute_f32(&inputs).unwrap()["out"],
            vec![0.0, 0.0, 9.0, 10.0, 2.0, 2.0]
        );
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("prism_scatter"));
        assert!(capture.validate().is_ok());
    }

    #[test]
    fn ssm_validates_and_executes_diagonal_scan() {
        let mut graph = TinyGraph::default();
        let input = graph.add(
            UOpKind::Input {
                name: "input".into(),
            },
            vec![],
            vec![2, 2],
        );
        let decay = graph.add(
            UOpKind::Input {
                name: "decay".into(),
            },
            vec![],
            vec![2],
        );
        let input_gain = graph.add(
            UOpKind::Input {
                name: "input_gain".into(),
            },
            vec![],
            vec![2],
        );
        let output_gain = graph.add(
            UOpKind::Input {
                name: "output_gain".into(),
            },
            vec![],
            vec![2],
        );
        let scan = graph.add(
            UOpKind::Ssm {
                rows: 2,
                features: 2,
            },
            vec![input, decay, input_gain, output_gain],
            vec![2, 2],
        );
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![scan],
            vec![2, 2],
        );
        let mut inputs = BTreeMap::new();
        inputs.insert("input".into(), vec![1.0, 2.0, 3.0, 4.0]);
        inputs.insert("decay".into(), vec![0.5, 0.25]);
        inputs.insert("input_gain".into(), vec![1.0, 1.0]);
        inputs.insert("output_gain".into(), vec![2.0, 4.0]);
        assert_eq!(
            graph.execute_f32(&inputs).unwrap()["out"],
            vec![2.0, 8.0, 7.0, 18.0]
        );
    }

    #[test]
    fn conv2d_renders_and_executes_nchw() {
        let mut graph = TinyGraph::default();
        let x = graph.add(
            UOpKind::Input { name: "x".into() },
            vec![],
            vec![1, 1, 3, 3],
        );
        let weight = graph.add(
            UOpKind::Input {
                name: "weight".into(),
            },
            vec![],
            vec![1, 1, 2, 2],
        );
        let bias = graph.add(
            UOpKind::Input {
                name: "bias".into(),
            },
            vec![],
            vec![1],
        );
        let conv = graph.add(
            UOpKind::Conv2d {
                batch: 1,
                in_channels: 1,
                height: 3,
                width: 3,
                out_channels: 1,
                kernel_h: 2,
                kernel_w: 2,
                stride: 1,
                padding: 0,
            },
            vec![x, weight, bias],
            vec![1, 1, 2, 2],
        );
        graph.add(
            UOpKind::Output { name: "out".into() },
            vec![conv],
            vec![1, 1, 2, 2],
        );
        let capture = graph.lower(LoweringTarget::Portable).unwrap();
        assert!(capture.kernels[0].source.contains("prism_conv2d"));
        let output = graph
            .execute_f32(&BTreeMap::from([
                (
                    "x".into(),
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                ),
                ("weight".into(), vec![1.0, 0.0, 0.0, 1.0]),
                ("bias".into(), vec![0.0]),
            ]))
            .unwrap();
        assert_eq!(output["out"], vec![6.0, 8.0, 12.0, 14.0]);
    }

    #[test]
    fn conv2d_rejects_invalid_geometry_without_underflow() {
        let mut graph = TinyGraph::default();
        let x = graph.add(
            UOpKind::Input { name: "x".into() },
            vec![],
            vec![1, 1, 2, 2],
        );
        let weight = graph.add(
            UOpKind::Input {
                name: "weight".into(),
            },
            vec![],
            vec![1, 1, 5, 5],
        );
        let bias = graph.add(
            UOpKind::Input {
                name: "bias".into(),
            },
            vec![],
            vec![1],
        );
        graph.add(
            UOpKind::Conv2d {
                batch: 1,
                in_channels: 1,
                height: 2,
                width: 2,
                out_channels: 1,
                kernel_h: 5,
                kernel_w: 5,
                stride: 1,
                padding: 0,
            },
            vec![x, weight, bias],
            vec![1, 1, 0, 0],
        );
        assert!(matches!(
            graph.validate(),
            Err(GraphError::ShapeMismatch(_))
        ));
    }
}
