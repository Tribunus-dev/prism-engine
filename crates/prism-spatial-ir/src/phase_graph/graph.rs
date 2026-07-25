//! This module owns the canonical authority for the `TinyGraph` mutable
//! graph structure, the `GraphError` typed error, and every graph mutation,
//! validation, scheduling, and reference-execution method.
//! It does not own the UOp data type (see `uop.rs`), the kernel layer (see
//! `kernel_op.rs` and `kernel_group.rs`), or replay submission (see
//! `capture.rs`).

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

use crate::phase_graph::capture::{CapturePlan, LoweredKernel};
use crate::phase_graph::kernel_op::{KernelOp, LoweringTarget};
use crate::phase_graph::kernel_group::KernelGroup;
use crate::phase_graph::plan::{BufferAllocation, MemoryPlan, ReplayPlan};
use crate::phase_graph::render::render_kernel;
use crate::phase_graph::shape::{broadcast_index, broadcast_shape, cast_f32, element_count, is_supported_dtype};
use crate::phase_graph::uop::{UOp, UOpId, UOpKind};

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
                    // WAIVER: `op.src[0]` is the validated source for this
                    // `Transpose` op (the `ids.contains(source)` check above
                    // passes only when the source exists in the graph).
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
                        // WAIVER: same validated-source guard as above.
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
                        // WAIVER: same validated-source guard as above.
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
                        // WAIVER: same validated-source guard as above.
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let when_true = self
                        .ops
                        .iter()
                        // WAIVER: same validated-source guard as above.
                        .find(|candidate| candidate.id == op.src[1])
                        .unwrap();
                    let when_false = self
                        .ops
                        .iter()
                        // WAIVER: same validated-source guard as above.
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
                        // WAIVER: same validated-source guard as above.
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let right = self
                        .ops
                        .iter()
                        // WAIVER: same validated-source guard as above.
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
                        // WAIVER: same validated-source guard as above.
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let right = self
                        .ops
                        .iter()
                        // WAIVER: same validated-source guard as above.
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
                        // WAIVER: same validated-source guard as above.
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let weight = self
                        .ops
                        .iter()
                        // WAIVER: same validated-source guard as above.
                        .find(|candidate| candidate.id == op.src[1])
                        .unwrap();
                    let bias = self
                        .ops
                        .iter()
                        // WAIVER: same validated-source guard as above.
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
                            // WAIVER: same validated-source guard as above.
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
                        // WAIVER: same validated-source guard as above.
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let weight = self
                        .ops
                        .iter()
                        // WAIVER: same validated-source guard as above.
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
                        // WAIVER: same validated-source guard as above.
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let k = self
                        .ops
                        .iter()
                        // WAIVER: same validated-source guard as above.
                        .find(|candidate| candidate.id == op.src[1])
                        .unwrap();
                    let v = self
                        .ops
                        .iter()
                        // WAIVER: same validated-source guard as above.
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
                        // WAIVER: same validated-source guard as above.
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let cos = self
                        .ops
                        .iter()
                        // WAIVER: same validated-source guard as above.
                        .find(|candidate| candidate.id == op.src[1])
                        .unwrap();
                    let sin = self
                        .ops
                        .iter()
                        // WAIVER: same validated-source guard as above.
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
                        // WAIVER: same validated-source guard as above.
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let indices = self
                        .ops
                        .iter()
                        // WAIVER: same validated-source guard as above.
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
                        // WAIVER: same validated-source guard as above.
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let indices = self
                        .ops
                        .iter()
                        // WAIVER: same validated-source guard as above.
                        .find(|candidate| candidate.id == op.src[1])
                        .unwrap();
                    let values = self
                        .ops
                        .iter()
                        // WAIVER: same validated-source guard as above.
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
                        // WAIVER: same validated-source guard as above.
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let decay = self
                        .ops
                        .iter()
                        // WAIVER: same validated-source guard as above.
                        .find(|candidate| candidate.id == op.src[1])
                        .unwrap();
                    let input_gain = self
                        .ops
                        .iter()
                        // WAIVER: same validated-source guard as above.
                        .find(|candidate| candidate.id == op.src[2])
                        .unwrap();
                    let output_gain = self
                        .ops
                        .iter()
                        // WAIVER: same validated-source guard as above.
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
                        // WAIVER: same validated-source guard as above.
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap();
                    let weight = self
                        .ops
                        .iter()
                        // WAIVER: same validated-source guard as above.
                        .find(|candidate| candidate.id == op.src[1])
                        .unwrap();
                    let bias = self
                        .ops
                        .iter()
                        // WAIVER: same validated-source guard as above.
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
                        // WAIVER: same validated-source guard as above.
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
                    // WAIVER: the outer `matches!` guard above restricts this
                    // match to the six binary arithmetic variants, so all
                    // other `UOpKind` arms are unreachable.
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
            //
            // WAIVER (function-scope, covers the `norm_kind.unwrap().0` and
            // `matmul_k.unwrap()` calls below): the outer `if norm_kind.is_some()
            // / if matmul_k.is_some()` guards establish the `Option` shape, and
            // the subsequent `Some(...)` / `_ => None` arms of the enclosing
            // `match` block make the `unwrap` infallible. This is the canonical
            // "Option-then-match-then-unwrap" pattern.
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
                // WAIVER: `id` is a UOp id collected from the graph's own
                // `ops` vector, so the linear search is infallible.
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
        // WAIVER (loop-scope, covers the `indegree.get_mut(user).unwrap()`
            // call below): the `users` map records every consumer of every
            // op, and `indegree` was populated for every op.id, so any
            // user surfaced by the Kahn schedule must already have a slot
            // in both maps. The unwrap is infallible by construction.
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
        // WAIVER (loop-scope, covers the `graph.ops.iter().find(|op| op.id == id).unwrap()`
            // and `graph.ops.iter().find(|candidate| candidate.id == *source).unwrap()`
            // calls below): `id` is produced by `graph.schedule()?` which
            // enumerates `graph.ops`, and `source` is a UOp id from a
            // validated `op.src` (the same validation that produced
            // `graph`), so the linear lookups are infallible.
        for id in graph.schedule()? {
            let op = graph
                .ops
                .iter()
                // WAIVER: `id` came from `graph.schedule()` which iterates
                // the graph's own `ops` vector, so the lookup is infallible.
                .find(|op| op.id == id)
                .unwrap();
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
                    // WAIVER: `source` is a UOp id from a validated
                    // `op.src`, present in the same graph.
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
                        // WAIVER: `op.id()` is the validated UOp id
                        // recorded when the group was constructed from the
                        // scheduled graph, so the lookup is infallible.
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
                        // WAIVER: same validated-source guard as in
                        // `lower`; `op.id()` was set from the original
                        // graph when the group was built.
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
        // WAIVER (function-scope, covers the `values.get(&op.src[N]).unwrap()`
        // and `scores.get(key).unwrap()` calls in the body below): every UOp
        // referenced by `op.src[N]` is a validated source (the outer
        // `self.validate()` call rejected any unknown source), and the loop
        // follows the topological schedule from `self.schedule()?` so each
        // source's value has been inserted into `values` before its consumer
        // reads it. `scores` is a `vec![0.0; *seq]` allocated on the same
        // line, so any `scores.get(key)` lookup is infallible.
        let mut values: BTreeMap<UOpId, Vec<f32>> = BTreeMap::new();
        let mut outputs = BTreeMap::new();
        for id in self.schedule()? {
            let op = self
                .ops
                .iter()
                // WAIVER: `id` is a UOp id produced by
                // `self.schedule()`, which enumerates the graph's own
                // `ops` vector.
                .find(|op| op.id == id)
                .unwrap();
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
                UOpKind::Reshape => values
                    .get(&op.src[0])
                    // WAIVER: the arity and source-presence checks in
                    // `validate` ensure `op.src[0]` is a valid UOp and a
                    // `Reshape` op has exactly one source. The execution
                    // order is a topological schedule, so the source value
                    // is present in `values` by the time we read it.
                    .unwrap()
                    .clone(),
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
                        // WAIVER: same validated-source guard as above.
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap()
                        .shape;
                    let right_shape = &self
                        .ops
                        .iter()
                        // WAIVER: same validated-source guard as above.
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
                                // WAIVER: the outer arm restricts this match
                                // to the six binary arithmetic variants.
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
                        // WAIVER: same validated-source guard as above.
                        .find(|candidate| candidate.id == op.src[0])
                        .unwrap()
                        .shape;
                    let true_shape = &self
                        .ops
                        .iter()
                        // WAIVER: same validated-source guard as above.
                        .find(|candidate| candidate.id == op.src[1])
                        .unwrap()
                        .shape;
                    let false_shape = &self
                        .ops
                        .iter()
                        // WAIVER: same validated-source guard as above.
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
                        // WAIVER: same validated-source guard as above.
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
                        // WAIVER: same validated-source guard as above.
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
                        // WAIVER: same validated-source guard as above.
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
                        // WAIVER: same validated-source guard as above.
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
                        // WAIVER: same validated-source guard as above.
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
        // WAIVER (function-scope, covers the `position[&op.id]` /
            // `last_use[source]` / `self.ops.iter().find(|op| op.id == *id).unwrap()`
            // calls below): the schedule was produced by
            // `self.schedule()?` which enumerates the graph's own `ops`
            // vector, and `last_use` is populated for every op.id before
            // any consumer reads it, so the `unwrap`/`[]` index ops are
            // infallible by construction.
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
            let op = self
                .ops
                .iter()
                // WAIVER: `id` is a UOp id produced by
                // `self.schedule()`, which enumerates the graph's own
                // `ops` vector.
                .find(|op| op.id == *id)
                .unwrap();
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
