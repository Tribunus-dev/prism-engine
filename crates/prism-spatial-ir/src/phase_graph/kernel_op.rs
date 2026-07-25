//! This module owns the canonical authority for the `LoweringTarget` enum,
//! the `KernelOp` executable-kernel variant, the `BroadcastBinaryOperation`
//! subset, and their `id` / `from_broadcast_op` / `from_graph_op` impls.
//! It does not own graph mutation, kernel rendering, or replay submission.

use serde::{Deserialize, Serialize};

use crate::phase_graph::graph::TinyGraph;
use crate::phase_graph::scalar::{scalar_is_left, scalar_operand};
use crate::phase_graph::shape::element_count;
use crate::phase_graph::uop::{UOp, UOpId, UOpKind};

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
    pub(crate) fn id(&self) -> UOpId {
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
    pub(crate) fn from_broadcast_op(op: &UOp, graph: &TinyGraph) -> Self {
        let source_shape = |index: usize| {
            graph
                .ops
                .iter()
                // WAIVER: `op.src[index]` is the validated UOp source for the
                // binary op being lowered; `validate` rejects unknown sources
                // before this lowering pass runs, so the lookup is infallible.
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
            // WAIVER: the caller (`TinyGraph::lower`) only routes binary
            // elementwise `UOpKind` variants into this function, so every
            // other variant is unreachable.
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

    pub(crate) fn from_graph_op(op: &UOp, graph: &TinyGraph) -> Self {
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
                        // WAIVER: same validated-source guard as above.
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
                    // WAIVER: validated UOp source — see from_broadcast_op.
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
                        // WAIVER: reduction source is validated by
                        // `TinyGraph::validate` (reduction arity + shape)
                        // before this lowering pass runs.
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
            // WAIVER: the caller's match arm in `TinyGraph::lower` only feeds
            // the variant set above into this function, so any other
            // `UOpKind` (Input/Const/Reshape/Output) is unreachable here.
            _ => unreachable!(),
        }
    }
}
