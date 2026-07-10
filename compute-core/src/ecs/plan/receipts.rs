//! Fusion-specific evidence receipts for planning and runtime.
//!
//! These receipts capture dimensional and diagnostic data about the fusion
//! pipeline stages: graph structure, scheduling decisions, lowering choices,
//! and region-level grouping. Each receipt is serializable and designed for
//! observability, audit trails, and performance regression detection.

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use super::fusion::DataflowGraph;
use super::fusion_scheduler::{
    BackendTarget, FusionEvaluation, FusionSchedule, FusionSupportLevel,
};
use super::KernelSpecializationKey;

// ── UnsupportedFusionReason ───────────────────────────────────────────

/// Why a fusion candidate was rejected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnsupportedFusionReason {
    /// No matching pattern was found in backend capabilities.
    NoMatchingPattern,
    /// The fused group exceeds the backend's max op count.
    OpCountExceedsCapabilities,
    /// The execution view is incompatible with the backend.
    IncompatibleExecutionView,
    /// The backend does not support this operation.
    BackendUnsupported,
    /// Memory pressure exceeded the available budget.
    MemoryPressureExceeded,
    /// Other rejection reason (human-readable).
    Other(String),
}

// ── DataflowGraphReceipt ──────────────────────────────────────────────

/// Receipt capturing the dimensionality of a resolved dataflow graph.
///
/// Records structural counts so planners and profilers can correlate
/// graph complexity with downstream scheduling and lowering decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataflowGraphReceipt {
    /// Number of dataflow operation nodes in the graph.
    pub node_count: usize,
    /// Number of directed edges (data dependencies between nodes).
    pub edge_count: usize,
    /// Number of logical buffer values flowing through the graph.
    pub value_count: usize,
    /// Model-layer identifier (e.g. `"gemma4_layer_12"`, `"vision_encoder"`).
    pub layer_id: String,
}

// ── FusionScheduleReceipt ─────────────────────────────────────────────

/// Receipt for a complete fusion scheduling pass.
///
/// Summarises how many fused groups were produced and what evaluation
/// results — both accepted and rejected — the scheduler emitted.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionScheduleReceipt {
    /// Number of fused groups in the schedule (includes singletons).
    pub group_count: usize,
    /// Per-evaluation receipts for each fusion opportunity examined.
    pub evaluations: Vec<FusionEvaluationReceipt>,
}

// ── FusionEvaluationReceipt ───────────────────────────────────────────

/// Receipt for a single fusion evaluation.
///
/// Records which source nodes were considered, whether a fusion was
/// accepted (and to which backend), and any rejections that occurred.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionEvaluationReceipt {
    /// Indices of the dataflow ops considered for fusion.
    pub source_nodes: Vec<usize>,
    /// The backend target selected for fusion, if any.
    pub selected_target: Option<BackendTarget>,
    /// Rejection reasons for fusion attempts that were declined.
    pub rejected: Vec<UnsupportedFusionReason>,
    /// Estimated bytes saved by materializing this fused group
    /// (e.g. avoided intermediate buffer allocation).
    pub materialization_saved_bytes: u64,
}

// ── BackendLoweringReceipt ────────────────────────────────────────────

/// How ready a lowered kernel is for execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LoweringReadiness {
    /// Only a descriptor / template — not yet compiled into an executable form.
    DescriptorOnly,
    /// Compilation validated — ready for execution (PSO cached, lowered).
    Executable,
}

impl Default for LoweringReadiness {
    fn default() -> Self {
        Self::DescriptorOnly
    }
}

/// Receipt for a backend lowering decision.
///
/// Records which backend was chosen, the specialization key digest,
/// and the readiness level for execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendLoweringReceipt {
    /// The backend target the fused group was lowered to.
    pub target: BackendTarget,
    /// Hex digest of the `KernelSpecializationKey` used for this lowering.
    pub specialization_key_digest: String,
    /// Human-readable identifier for the fusion pattern, derived from
    /// the kernel template (e.g. `"FusedGateUpActivation"`).
    pub fusion_pattern_id: String,
    /// Readiness level — whether the lowered kernel is compiled/executable.
    #[serde(default)]
    pub readiness: LoweringReadiness,
}

// ── RegionFusionReceipt ───────────────────────────────────────────────

/// Receipt for a region-level fused group.
///
/// Captures how a fused group was placed into an execution region for
/// command-buffer planning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionFusionReceipt {
    /// Identifier of the execution region (e.g. `"region_decode_0"`).
    pub region_id: String,
    /// The fused group id assigned during scheduling.
    pub fused_group_id: String,
    /// Total number of dataflow ops in this fused group.
    pub total_ops: usize,
    /// Estimated scratch buffer bytes for this region's fused group.
    pub estimated_scratch_bytes: usize,
}

// ── Collection functions ──────────────────────────────────────────────

/// Collect a receipt from a resolved `DataflowGraph`.
pub fn collect_graph_receipt(graph: &DataflowGraph) -> DataflowGraphReceipt {
    DataflowGraphReceipt {
        node_count: graph.nodes.len(),
        edge_count: graph.edges.len(),
        value_count: graph.values.len(),
        layer_id: graph.layer_id.clone(),
    }
}

/// Collect a receipt from a `FusionSchedule`.
pub fn collect_schedule_receipt(schedule: &FusionSchedule) -> FusionScheduleReceipt {
    let group_count = schedule.groups.len();
    let evaluations: Vec<FusionEvaluationReceipt> = schedule
        .receipts
        .iter()
        .map(collect_evaluation_receipt)
        .collect();
    FusionScheduleReceipt {
        group_count,
        evaluations,
    }
}

/// Internal helper — convert one `FusionEvaluation` into its receipt form.
fn collect_evaluation_receipt(eval: &FusionEvaluation) -> FusionEvaluationReceipt {
    let selected_target = eval.selected.as_ref().map(|c| c.target);
    let rejected: Vec<UnsupportedFusionReason> = eval
        .rejected
        .iter()
        .map(|r| UnsupportedFusionReason::Other(r.reason.clone()))
        .collect();
    let materialization_saved_bytes = eval
        .selected
        .as_ref()
        .map(|c| match c.support {
            FusionSupportLevel::Full => c.lowering_cost.scratch_bytes,
            _ => 0,
        })
        .unwrap_or(0);
    FusionEvaluationReceipt {
        source_nodes: eval.source_nodes.clone(),
        selected_target,
        rejected,
        materialization_saved_bytes,
    }
}

/// Collect a receipt from a backend lowering decision.
pub fn collect_lowering_receipt(
    target: BackendTarget,
    key: &KernelSpecializationKey,
) -> BackendLoweringReceipt {
    let mut hasher = DefaultHasher::new();
    key.hash(&mut hasher);
    let digest = format!("{:016x}", hasher.finish());
    let fusion_pattern_id = format!("{:?}", key.template_id);
    BackendLoweringReceipt {
        target,
        specialization_key_digest: digest,
        fusion_pattern_id,
        readiness: LoweringReadiness::default(),
    }
}

// ── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ecs::plan::fusion::{
        DataflowGraph, DataflowNode, DataflowOp, FusedGroup,
    };
    use crate::ecs::plan::fusion_scheduler::{
        BackendTarget, FusionCandidate, FusionEvaluation, FusionRejection, FusionSchedule,
        FusionSupportLevel, LoweringCost,
    };
    use crate::ecs::plan::{
        AffineMode, Axis, CodecFamily, DType, ExecutionPhase, HardwareProfileId,
        KernelSpecializationKey, KernelTemplateId, MetadataLayout, TileShape,
    };
    use std::collections::HashMap;

    fn sample_graph() -> DataflowGraph {
        DataflowGraph {
            nodes: vec![
                DataflowNode {
                    id: 0,
                    op: DataflowOp::RmsNorm {
                        input: "a".into(),
                        weight: "rms_w".into(),
                        output: "b".into(),
                        epsilon: 1e-6,
                    },
                    inputs: vec!["a".into()],
                    outputs: vec!["b".into()],
                },
                DataflowNode {
                    id: 1,
                    op: DataflowOp::MatMul {
                        lhs: "b".into(),
                        rhs: "gate_proj.weight".into(),
                        output: "gate_out".into(),
                        contract: crate::ecs::plan::fusion::MatMulContract {
                            m: 1,
                            n: 8192,
                            k: 2048,
                            lhs_transposed: false,
                            rhs_transposed: true,
                        },
                    },
                    inputs: vec!["b".into(), "gate_proj.weight".into()],
                    outputs: vec!["gate_out".into()],
                },
                DataflowNode {
                    id: 2,
                    op: DataflowOp::SiLU {
                        input: "gate_out".into(),
                        output: "gated".into(),
                    },
                    inputs: vec!["gate_out".into()],
                    outputs: vec!["gated".into()],
                },
            ],
            edges: vec![],
            values: HashMap::new(),
            layer_id: "test_layer_0".into(),
        }
    }

    #[test]
    fn graph_receipt_counts_match() {
        let graph = sample_graph();
        let receipt = collect_graph_receipt(&graph);

        assert_eq!(receipt.node_count, 3);
        assert_eq!(receipt.edge_count, 0);
        assert_eq!(receipt.value_count, 0);
        assert_eq!(receipt.layer_id, "test_layer_0");
    }

    #[test]
    fn schedule_receipt_records_rejections() {
        let schedule = FusionSchedule {
            groups: vec![],
            receipts: vec![
                FusionEvaluation {
                    source_nodes: vec![0, 1],
                    candidates: vec![],
                    selected: None,
                    rejected: vec![
                        FusionRejection {
                            group_id: "0".into(),
                            target: BackendTarget::MetalFusedGpu,
                            reason: "NoMatchingPattern".into(),
                        },
                        FusionRejection {
                            group_id: "0".into(),
                            target: BackendTarget::AccelerateRayonCpu,
                            reason: "OpCountExceedsCapabilities".into(),
                        },
                    ],
                },
                FusionEvaluation {
                    source_nodes: vec![2],
                    candidates: vec![],
                    selected: Some(FusionCandidate {
                        group: FusedGroup {
                            id: "g0".into(),
                            body: vec![],
                            inputs: vec![],
                            outputs: vec![],
                            internal_values: vec![],
                            codec_family: crate::ecs::plan::CodecFamily::RawF32,
                            precision_plan: None,
                        },
                        target: BackendTarget::MetalFusedGpu,
                        support: FusionSupportLevel::Full,
                        lowering_cost: LoweringCost {
                            estimated_us: 0.0,
                            bytes_read: 0,
                            bytes_written: 0,
                            scratch_bytes: 4096,
                            thread_count: 128,
                            materialization_cost: 0.0,
                        },
                    }),
                    rejected: vec![],
                },
            ],
        };

        let receipt = collect_schedule_receipt(&schedule);

        assert_eq!(receipt.evaluations.len(), 2);

        let eval0 = &receipt.evaluations[0];
        assert_eq!(eval0.source_nodes, vec![0, 1]);
        assert!(eval0.selected_target.is_none());
        assert_eq!(eval0.rejected.len(), 2);
        assert_eq!(
            eval0.rejected[0],
            UnsupportedFusionReason::Other("NoMatchingPattern".to_string())
        );
        assert_eq!(
            eval0.rejected[1],
            UnsupportedFusionReason::Other("OpCountExceedsCapabilities".to_string())
        );
        assert_eq!(eval0.materialization_saved_bytes, 0);

        let eval1 = &receipt.evaluations[1];
        assert_eq!(eval1.source_nodes, vec![2]);
        assert_eq!(eval1.selected_target, Some(BackendTarget::MetalFusedGpu));
        assert!(eval1.rejected.is_empty());
        assert_eq!(eval1.materialization_saved_bytes, 4096);
    }

    #[test]
    fn lowering_receipt_contains_specialization_digest() {
        let key = KernelSpecializationKey {
            template_id: KernelTemplateId::FusedGateUpActivation,
            execution_phase: ExecutionPhase::Prefill,
            codec: CodecFamily::Int8,
            tile_shape: TileShape::tile640_decode(),
            group_size: 32,
            group_axis: Axis::PackedContiguous,
            affine_mode: AffineMode::ScaleOnly,
            metadata_layout: MetadataLayout::AdjacentTile,
            input_dtype: DType::F32,
            output_dtype: DType::F16,
            hardware_profile: HardwareProfileId::AppleMBaseMemoryBound,
            mode_flags: 0,
        };

        let receipt = collect_lowering_receipt(BackendTarget::MetalFusedGpu, &key);

        assert_eq!(receipt.target, BackendTarget::MetalFusedGpu);
        assert_eq!(receipt.specialization_key_digest.len(), 16);
        assert!(receipt
            .specialization_key_digest
            .chars()
            .all(|c| c.is_ascii_hexdigit()));
        assert_eq!(receipt.fusion_pattern_id, "FusedGateUpActivation");

        let key2 = KernelSpecializationKey {
            group_size: 128,
            ..key.clone()
        };
        let receipt2 = collect_lowering_receipt(BackendTarget::AccelerateRayonCpu, &key2);
        assert_ne!(
            receipt.specialization_key_digest,
            receipt2.specialization_key_digest
        );
    }
}
