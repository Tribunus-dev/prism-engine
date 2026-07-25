//! This module owns the canonical authority for the `KernelGroup` struct and
//! the helper methods that expose a group's executable shape, kernel-ABI
//! variant, and buffer requirements to downstream consumers.
//! It does not own graph mutation, kernel rendering, or replay submission.

use serde::{Deserialize, Serialize};

use crate::phase_graph::kernel_op::{BroadcastBinaryOperation, KernelOp};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KernelGroup {
    pub ops: Vec<KernelOp>,
}

pub(crate) type BroadcastBinaryShape = (BroadcastBinaryOperation, Vec<u64>, Vec<u64>, Vec<u64>);

pub(crate) type Convolution2dShape = (
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
    pub(crate) fn ops_after_broadcast(&self) -> &[KernelOp] {
        self.ops.get(1..)
            // WAIVER: `ops_after_broadcast` is only called from `render_kernel`
            // after the caller has pattern-matched a `BroadcastBinary` head,
            // which guarantees `self.ops` is non-empty.
            .unwrap_or(&[])
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

    pub fn op_ids(&self) -> Vec<crate::phase_graph::uop::UOpId> {
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

    pub(crate) fn reduction(&self) -> Option<usize> {
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
