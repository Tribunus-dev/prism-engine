//! Schedule-level types for the fusion compiler pipeline.
//!
//! These types represent the operation-granularity IR that `FusionScheduler`,
//! `planner`, and the test harness consume — distinct from the node-based
//! graph IR in `fusion.rs`.
//!
//! Each type is independently serializable and carries no references into
//! the dataflow graph.

use serde::{Deserialize, Serialize};

use crate::ecs::execution_profile::ExecutionView;

/// Discriminant for dataflow operations at the schedule level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataflowOpKind {
    RmsNorm,
    QkvProjection,
    AttentionScore,
    AttentionApply,
    OProjectionResidual,
    MlpGateUp,
    MlpActivation,
    MlpDownResidual,
    BridgeProjection,
    VisionPatchProjection,
    TtsProjection,
    TokenEmbedding,
    LmHead,
}

/// A concrete schedule-level operation instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataflowOp {
    pub op_index: usize,
    pub step_name: String,
    pub op_kind: DataflowOpKind,
    pub execution_view: ExecutionView,
    pub input_tensors: Vec<usize>,
    pub output_tensors: Vec<usize>,
    pub arithmetic_intensity: Option<f64>,
}

/// Schedule-level dataflow graph: ops + tensor shapes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataflowGraph {
    pub ops: Vec<DataflowOp>,
    pub tensor_shapes: Vec<TensorDescriptor>,
}

/// Logical tensor descriptor for buffer planning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDescriptor {
    pub shape: Vec<usize>,
    pub dtype: String,
    pub byte_size: usize,
}

/// Combined dispatch information for a fused group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DispatchInfo {
    pub threadgroups: [u32; 3],
    pub threads_per_group: [u32; 3],
}

/// A group of dataflow ops fused into a single scheduling unit.
///
/// When fusion is disabled or no pattern matches, each group contains
/// exactly one op.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedGroup {
    pub group_id: usize,
    pub ops: Vec<DataflowOp>,
    pub combined_dispatch_shape: Option<DispatchInfo>,
    pub has_fused_kernel: bool,
    pub fusion_pattern: Option<String>,
}

/// The set of fusion patterns a backend supports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionCapabilities {
    pub supported_patterns: Vec<FusionPattern>,
    pub max_fused_ops: usize,
}

/// An identified fusion pattern — a sequence of op kinds that can be fused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FusionPattern {
    QkvFused,
    MlpGateActivation,
    AttentionFused,
    OProjectionResidual,
    Custom([DataflowOpKind; 3]),
}

// ── Convenience function ─────────────────────────────────────────────────

/// Convenience function: fuse a resolved dataflow graph into groups.
///
/// Delegates to a simple greedy scheduler.  When no capabilities are
/// provided, fusion is effectively disabled and every op becomes a
/// singleton group.
pub fn fuse_and_schedule(
    graph: &DataflowGraph,
    capabilities: &[FusionCapabilities],
) -> Vec<FusedGroup> {
    if capabilities.is_empty() || capabilities.iter().all(|c| c.supported_patterns.is_empty()) {
        return graph
            .ops
            .iter()
            .enumerate()
            .map(|(i, op)| FusedGroup {
                group_id: i,
                ops: vec![op.clone()],
                combined_dispatch_shape: None,
                has_fused_kernel: false,
                fusion_pattern: None,
            })
            .collect();
    }

    let known_patterns: std::collections::HashSet<FusionPattern> = capabilities
        .iter()
        .flat_map(|c| c.supported_patterns.iter().copied())
        .collect();

    let max_ops = capabilities
        .iter()
        .map(|c| c.max_fused_ops)
        .max()
        .unwrap_or(1);

    let mut groups: Vec<FusedGroup> = Vec::new();
    let mut gid: usize = 0;
    let mut i: usize = 0;

    while i < graph.ops.len() {
        let remaining = graph.ops.len().saturating_sub(i);
        let mut matched = false;
        let mut consumed = 0;

        if remaining >= 2 {
            let window = remaining.min(max_ops);
            for len in (2..=window).rev() {
                let candidate = &graph.ops[i..i + len];
                let pattern = match len {
                    2 => {
                        let a = candidate[0].op_kind;
                        let b = candidate[1].op_kind;
                        if a == DataflowOpKind::QkvProjection && b == DataflowOpKind::RmsNorm {
                            FusionPattern::QkvFused
                        } else if a == DataflowOpKind::AttentionScore
                            && b == DataflowOpKind::AttentionApply
                        {
                            FusionPattern::AttentionFused
                        } else if a == DataflowOpKind::MlpGateUp
                            && b == DataflowOpKind::MlpActivation
                        {
                            FusionPattern::MlpGateActivation
                        } else if a == DataflowOpKind::OProjectionResidual
                            && b == DataflowOpKind::MlpGateUp
                        {
                            FusionPattern::OProjectionResidual
                        } else {
                            FusionPattern::Custom([a, b, a])
                        }
                    }
                    _ => {
                        FusionPattern::Custom([
                            candidate[0].op_kind,
                            candidate[1].op_kind,
                            candidate[2].op_kind,
                        ])
                    }
                };
                if known_patterns.contains(&pattern) {
                    matched = true;
                    consumed = len;
                    break;
                }
            }
        }

        if matched {
            let ops: Vec<DataflowOp> = graph.ops[i..i + consumed].to_vec();
            let label = Some(
                ops.iter()
                    .map(|o| o.step_name.as_str())
                    .collect::<Vec<&str>>()
                    .join("+"),
            );
            groups.push(FusedGroup {
                group_id: gid,
                ops,
                combined_dispatch_shape: None,
                has_fused_kernel: true,
                fusion_pattern: label,
            });
            gid += 1;
            i += consumed;
        } else {
            groups.push(FusedGroup {
                group_id: gid,
                ops: vec![graph.ops[i].clone()],
                combined_dispatch_shape: None,
                has_fused_kernel: false,
                fusion_pattern: None,
            });
            gid += 1;
            i += 1;
        }
    }

    groups
}
