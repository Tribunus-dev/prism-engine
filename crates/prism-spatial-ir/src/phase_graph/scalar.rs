//! This module owns the canonical authority for the `scalar_operand` and
//! `scalar_is_left` helpers used to detect constant-fused binary operands
//! during kernel lowering.
//! It does not own UOp identity, graph mutation, or kernel rendering.

use crate::phase_graph::graph::TinyGraph;
use crate::phase_graph::uop::{UOp, UOpKind};

pub(crate) fn scalar_operand(op: &UOp, graph: &TinyGraph) -> Option<f32> {
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

pub(crate) fn scalar_is_left(op: &UOp, graph: &TinyGraph) -> bool {
    op.src
        .first()
        .and_then(|source| graph.ops.iter().find(|candidate| candidate.id == *source))
        .is_some_and(|candidate| matches!(candidate.kind, UOpKind::Const { .. }))
}
