//! Backend-neutral dataflow graph IR for fusion scheduling.
//!
//! Phase 1 of the Fusion Compiler IR pipeline:
//!   PolicyResolver -> LayoutResolver -> DataflowGraphBuilder -> FusionScheduler
//!   -> BackendLowering -> ExecutionPlanner -> ExecutionRegion -> RegionEncoder
//!
//! This module defines the intermediate representation (IR) types for
//! representing model layer compute as a dataflow graph prior to backend
//! lowering and fusion scheduling.

use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::ecs::plan::{CodecFamily, DType, precision_plan::PrecisionPlan};
use crate::execution_profile::{GroupAxis, MetadataLayout, PhysicalTileLayout};

// ---------------------------------------------------------------------------
// Core type aliases
// ---------------------------------------------------------------------------

/// Opaque identifier for a buffer in the dataflow graph.
pub type DataflowBufferId = String;

/// Reference to a named tensor in the model's weight or KV-cache store.
pub type DataflowTensorRef = String;

// ---------------------------------------------------------------------------
// Value and residency
// ---------------------------------------------------------------------------

/// A value (buffer) flowing through the dataflow graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataflowValue {
    pub id: DataflowBufferId,
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub current_residency: ValueResidency,
}

/// Where a value is currently resident in the memory hierarchy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueResidency {
    CpuResident,
    GpuResident,
    AneResident,
    SharedUnified,
    Unknown,
}

// ---------------------------------------------------------------------------
// Edge
// ---------------------------------------------------------------------------

/// A directed edge in the dataflow graph from a producer node to a consumer
/// node, carrying a specific buffer value.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataflowEdge {
    /// Index into `DataflowGraph.nodes` for the producing node.
    pub producer: usize,
    /// Index into `DataflowGraph.nodes` for the consuming node.
    pub consumer: usize,
    /// The buffer value transmitted along this edge.
    pub value: DataflowBufferId,
}

// ---------------------------------------------------------------------------
// Node
// ---------------------------------------------------------------------------

/// A single operation node in the dataflow graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataflowNode {
    pub id: usize,
    pub op: DataflowOp,
    pub inputs: Vec<DataflowBufferId>,
    pub outputs: Vec<DataflowBufferId>,
}

// ---------------------------------------------------------------------------
// Graph
// ---------------------------------------------------------------------------

/// A backend-neutral dataflow graph for one layer's compute.
///
/// Nodes represent operations, edges represent data dependencies, and values
/// describe the buffers flowing between nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataflowGraph {
    pub nodes: Vec<DataflowNode>,
    pub edges: Vec<DataflowEdge>,
    pub values: HashMap<DataflowBufferId, DataflowValue>,
    pub layer_id: String,
}

// ---------------------------------------------------------------------------
// MatMul contract (fresh)
// ---------------------------------------------------------------------------

/// Describes the mathematical shape of a matrix multiplication.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct MatMulContract {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub lhs_transposed: bool,
    pub rhs_transposed: bool,
}

// ---------------------------------------------------------------------------
// DataflowOp
// ---------------------------------------------------------------------------

/// Discriminant for DataflowOp variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataflowOpKind {
    LoadWeight,
    LoadActivation,
    Dequantize,
    MatMul,
    RmsNorm,
    SiLU,
    Gelu,
    Mul,
    Add,
    ResidualAdd,
    StoreActivation,
    KvRead,
    KvWrite,
    EngramLookup,
}

/// The operation kind carried by a dataflow node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataflowOp {
    /// Load a weight tensor from the model store into a graph buffer.
    LoadWeight {
        tensor: DataflowTensorRef,
        codec: CodecFamily,
        layout: PhysicalTileLayout,
    },
    /// Dequantize a quantized buffer to a wider dtype.
    Dequantize {
        input: DataflowBufferId,
        output_dtype: DType,
    },
    /// Matrix multiply: output = lhs @ rhs (with optional transposes in contract).
    MatMul {
        lhs: DataflowBufferId,
        rhs: DataflowBufferId,
        output: DataflowBufferId,
        contract: MatMulContract,
    },
    /// RMS normalization: output = rms_norm(input, weight).
    RmsNorm {
        input: DataflowBufferId,
        weight: DataflowTensorRef,
        output: DataflowBufferId,
        epsilon: f32,
    },
    /// SiLU activation: output = silu(input).
    SiLU {
        input: DataflowBufferId,
        output: DataflowBufferId,
    },
    /// GELU activation: output = gelu(input).
    Gelu {
        input: DataflowBufferId,
        output: DataflowBufferId,
    },
    /// Element-wise multiply: output = lhs * rhs.
    Mul {
        lhs: DataflowBufferId,
        rhs: DataflowBufferId,
        output: DataflowBufferId,
    },
    /// Element-wise add: output = lhs + rhs.
    Add {
        lhs: DataflowBufferId,
        rhs: DataflowBufferId,
        output: DataflowBufferId,
    },
    /// Fused residual add: output = residual + update.
    ResidualAdd {
        residual: DataflowBufferId,
        update: DataflowBufferId,
        output: DataflowBufferId,
    },
    /// Persist an activation to a named slot (used for cross-layer sharing).
    StoreActivation {
        slot: String,
        input: DataflowBufferId,
    },
    /// Read a previously stored KV entry from a named slot.
    KvRead {
        slot: String,
        output: DataflowBufferId,
    },
    /// Write a new KV entry to a named slot.
    KvWrite {
        slot: String,
        input: DataflowBufferId,
    },
}

impl DataflowOp {
    /// Return the `DataflowOpKind` discriminant for this operation.
    pub fn kind(&self) -> DataflowOpKind {
        match self {
            DataflowOp::LoadWeight { .. } => DataflowOpKind::LoadWeight,
            DataflowOp::Dequantize { .. } => DataflowOpKind::Dequantize,
            DataflowOp::MatMul { .. } => DataflowOpKind::MatMul,
            DataflowOp::RmsNorm { .. } => DataflowOpKind::RmsNorm,
            DataflowOp::SiLU { .. } => DataflowOpKind::SiLU,
            DataflowOp::Gelu { .. } => DataflowOpKind::Gelu,
            DataflowOp::Mul { .. } => DataflowOpKind::Mul,
            DataflowOp::Add { .. } => DataflowOpKind::Add,
            DataflowOp::ResidualAdd { .. } => DataflowOpKind::ResidualAdd,
            DataflowOp::StoreActivation { .. } => DataflowOpKind::StoreActivation,
            DataflowOp::KvRead { .. } => DataflowOpKind::KvRead,
            DataflowOp::KvWrite { .. } => DataflowOpKind::KvWrite,
            // LoadActivation and EngramLookup are not yet defined in DataflowOp.
            // When added, update the match here.
        }
    }
}

// ---------------------------------------------------------------------------
// FusedGroup
// ---------------------------------------------------------------------------

/// Derived semantics for a fused group, describing its codec, data types,
/// physical layout, and mixed-precision plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedGroupSemantics {
    /// Codec family from the first LoadWeight node.
    pub codec_family: Option<CodecFamily>,
    /// Input dtype for the group.
    pub input_dtype: Option<DType>,
    /// Output dtype for the group.
    pub output_dtype: Option<DType>,
    /// Physical tile layout from the first LoadWeight node.
    pub physical_layout: Option<PhysicalTileLayout>,
    /// Group size from the first LoadWeight tile layout.
    pub group_size: Option<u32>,
    /// Group axis from the first LoadWeight tile layout.
    pub group_axis: Option<GroupAxis>,
    /// Metadata layout from the first LoadWeight tile layout.
    pub metadata_layout: Option<MetadataLayout>,
    /// Whether the group contains mixed codec families.
    pub mixed_codec: bool,
    /// Whether mixed precision is active.
    pub mixed_precision: bool,
    /// Whether the group contains a LoadWeight node.
    pub has_weight_load: bool,
    /// Whether the group has a materialization boundary node.
    pub has_materialization_boundary: bool,
    /// Optional PrecisionPlan for mixed-precision groups.
    pub precision_plan: Option<PrecisionPlan>,
}

/// Errors that can arise when deriving semantics for a fused group.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FusionSemanticError {
    /// Group body is empty.
    EmptyGroup,
    /// No LoadWeight node found in group.
    NoLoadWeightNode,
    /// Multiple conflicting codec families found.
    ConflictingCodecFamilies,
    /// Mixed codec group is missing a PrecisionPlan.
    MissingPrecisionPlan,
}

impl FusedGroup {
    /// Derive semantics by scanning the group body.
    ///
    /// Collects codec family, layout parameters, and mixed-codec detection
    /// from LoadWeight nodes. Returns errors when semantics cannot be derived
    /// (empty group, no LoadWeight, conflicting codecs).
    pub fn derive_semantics(&self) -> Result<FusedGroupSemantics, FusionSemanticError> {
        if self.body.is_empty() {
            return Err(FusionSemanticError::EmptyGroup);
        }

        let mut codec_family: Option<CodecFamily> = None;
        let mut physical_layout: Option<PhysicalTileLayout> = None;
        let mut has_weight_load = false;
        let mut has_materialization_boundary = false;
        let mut mixed_codec = false;

        for node in &self.body {
            match &node.op {
                DataflowOp::LoadWeight { codec, layout, .. } => {
                    if !has_weight_load {
                        // First LoadWeight: capture codec and layout.
                        codec_family = Some(*codec);
                        physical_layout = Some(layout.clone());
                    } else {
                        // Subsequent LoadWeight: check for codec conflict.
                        if let Some(prev_codec) = codec_family {
                            if *codec != prev_codec {
                                mixed_codec = true;
                            }
                        }
                    }
                    has_weight_load = true;
                }
                DataflowOp::StoreActivation { .. }
                | DataflowOp::KvRead { .. }
                | DataflowOp::KvWrite { .. } => {
                    has_materialization_boundary = true;
                }
                _ => {}
            }
        }

        // Extract group_size, group_axis, metadata_layout from physical layout.
        let (group_size, group_axis, metadata_layout) = match &physical_layout {
            Some(layout) => (
                Some(layout.group_size),
                Some(layout.group_axis),
                Some(layout.metadata_layout),
            ),
            None => (None, None, None),
        };

        // A group is mixed-precision only if it has a mixed codec AND a PrecisionPlan.
        let mixed_precision = mixed_codec && self.precision_plan.is_some();

        Ok(FusedGroupSemantics {
            codec_family,
            input_dtype: None,
            output_dtype: None,
            physical_layout,
            group_size,
            group_axis,
            metadata_layout,
            mixed_codec,
            mixed_precision,
            has_weight_load,
            has_materialization_boundary,
            precision_plan: self.precision_plan.clone(),
        })
    }
}

/// A fused subgraph that can be lowered to a single backend kernel call.
///
/// `body` contains the fused `DataflowNode` instances. `inputs` and `outputs`
/// are the boundary buffer IDs that cross the group boundary. `internal_values`
/// are buffers produced and consumed entirely within the group and need not be
/// materialized outside it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusedGroup {
    pub id: String,
    pub body: Vec<DataflowNode>,
    pub inputs: Vec<DataflowBufferId>,
    pub outputs: Vec<DataflowBufferId>,
    pub internal_values: Vec<DataflowBufferId>,
    pub codec_family: CodecFamily,
    pub precision_plan: Option<PrecisionPlan>,
}

// ---------------------------------------------------------------------------
// FusedGroupSemantics
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Legacy schedule-level types — used by fusion_scheduler.rs and planar_lowering.rs
// ---------------------------------------------------------------------------

/// Discriminant for schedule-level operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ScheduledOpKind {
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

/// A concrete schedule-level operation instance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledOp {
    pub op_index: usize,
    pub step_name: String,
    pub op_kind: ScheduledOpKind,
    pub execution_view: crate::execution_profile::ExecutionView,
    pub input_tensors: Vec<usize>,
    pub output_tensors: Vec<usize>,
    pub arithmetic_intensity: Option<f64>,
}

/// Schedule-level dataflow graph: ops + tensor shapes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleGraph {
    pub ops: Vec<ScheduledOp>,
    pub tensor_shapes: Vec<TensorDescriptor>,
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
    Custom([ScheduledOpKind; 3]),
}

/// A fused group at the schedule level.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduleFusedGroup {
    pub group_id: usize,
    pub ops: Vec<ScheduledOp>,
    pub combined_dispatch_shape: Option<DispatchInfo>,
    pub has_fused_kernel: bool,
    pub fusion_pattern: Option<String>,
}

/// Convenience function: fuse a resolved dataflow graph into groups.
pub fn fuse_and_schedule(
    graph: &ScheduleGraph,
    capabilities: &[FusionCapabilities],
) -> Vec<ScheduleFusedGroup> {
    if capabilities.is_empty() || capabilities.iter().all(|c| c.supported_patterns.is_empty()) {
        return graph
            .ops
            .iter()
            .enumerate()
            .map(|(i, op)| ScheduleFusedGroup {
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

    let mut groups: Vec<ScheduleFusedGroup> = Vec::new();
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
                        if a == ScheduledOpKind::QkvProjection && b == ScheduledOpKind::RmsNorm {
                            FusionPattern::QkvFused
                        } else if a == ScheduledOpKind::AttentionScore && b == ScheduledOpKind::AttentionApply {
                            FusionPattern::AttentionFused
                        } else if a == ScheduledOpKind::MlpGateUp && b == ScheduledOpKind::MlpActivation {
                            FusionPattern::MlpGateActivation
                        } else if a == ScheduledOpKind::OProjectionResidual && b == ScheduledOpKind::MlpGateUp {
                            FusionPattern::OProjectionResidual
                        } else {
                            FusionPattern::Custom([a, b, a])
                        }
                    }
                    _ if len >= 3 => {
                        FusionPattern::Custom([
                            candidate[0].op_kind,
                            candidate[1].op_kind,
                            candidate[2].op_kind,
                        ])
                    }
                    _ => FusionPattern::Custom([ScheduledOpKind::RmsNorm; 3]),
                };
                if known_patterns.contains(&pattern) {
                    matched = true;
                    consumed = len;
                    break;
                }
            }
        }

        if matched {
            let ops: Vec<ScheduledOp> = graph.ops[i..i + consumed].to_vec();
            let label = Some(
                ops.iter()
                    .map(|o| o.step_name.as_str())
                    .collect::<Vec<&str>>()
                    .join("+"),
            );
            groups.push(ScheduleFusedGroup {
                group_id: gid,
                ops,
                combined_dispatch_shape: None,
                has_fused_kernel: true,
                fusion_pattern: label,
            });
            gid += 1;
            i += consumed;
        } else {
            groups.push(ScheduleFusedGroup {
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

// ---------------------------------------------------------------------------
// DataflowGraph methods
// ---------------------------------------------------------------------------

impl DataflowGraph {
    /// Returns node indices in valid topological (FIFO) order.
    ///
    /// Every edge's producer appears before its consumer in the result.
    /// Nodes with no dependencies within the graph (e.g. external inputs)
    /// are emitted first. Uses Kahn's algorithm with a FIFO queue for
    /// deterministic, intuitive ordering.
    pub fn topological_sort(&self) -> Vec<usize> {
        let n = self.nodes.len();
        if n == 0 {
            return Vec::new();
        }

        // Build in-degree counts: how many graph-internal edges point into each node.
        let mut in_degree = vec![0usize; n];
        for edge in &self.edges {
            in_degree[edge.consumer] += 1;
        }

        let mut queue: VecDeque<usize> = VecDeque::new();
        for (i, deg) in in_degree.iter().enumerate() {
            if *deg == 0 {
                queue.push_back(i);
            }
        }

        let mut result = Vec::with_capacity(n);
        while let Some(node_idx) = queue.pop_front() {
            result.push(node_idx);
            for edge in &self.edges {
                if edge.producer == node_idx {
                    let consumer = edge.consumer;
                    in_degree[consumer] = in_degree[consumer].saturating_sub(1);
                    if in_degree[consumer] == 0 {
                        queue.push_back(consumer);
                    }
                }
            }
        }

        result
    }

    /// Returns the node index that produces a given buffer value, if any.
    pub fn producer_of(&self, value_id: &str) -> Option<usize> {
        self.edges
            .iter()
            .find(|e| e.value == value_id)
            .map(|e| e.producer)
            .or_else(|| {
                self.nodes
                    .iter()
                    .find(|n| n.outputs.contains(&value_id.to_string()))
                    .map(|n| n.id)
            })
    }

    /// Returns the node indices that consume a given buffer value.
    pub fn consumers_of(&self, value_id: &str) -> Vec<usize> {
        self.edges
            .iter()
            .filter(|e| e.value == value_id)
            .map(|e| e.consumer)
            .collect()
    }

    /// Returns node indices where values must be materialized at runtime
    /// boundaries: cross-layer aliasing, KV cache reads/writes, and
    /// StoreActivation nodes.
    pub fn materialization_boundaries(&self) -> Vec<usize> {
        self.nodes
            .iter()
            .filter(|n| {
                matches!(
                    n.op,
                    DataflowOp::StoreActivation { .. }
                        | DataflowOp::KvRead { .. }
                        | DataflowOp::KvWrite { .. }
                        | DataflowOp::LoadWeight { .. }
                )
            })
            .map(|n| n.id)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// DataflowGraphBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing `DataflowGraph` instances, including canonical
/// reference graphs like the Gemma decoder MLP.
pub struct DataflowGraphBuilder;

impl DataflowGraphBuilder {
    /// Builds a canonical Gemma decoder MLP dataflow graph.
    ///
    /// The graph has 7 nodes:
    ///   1. RMSNorm(activation) -> normalized
    ///   2. Gate MatMul(normalized, gate_proj.weight) -> gate_out
    ///   3. Up MatMul(normalized, up_proj.weight) -> up_out
    ///   4. SiLU(gate_out) -> gated
    ///   5. Mul(gated, up_out) -> gated_up
    ///   6. Down MatMul(gated_up, down_proj.weight) -> down_out
    ///   7. ResidualAdd(layer_input, down_out) -> layer_output
    pub fn build_mlp() -> DataflowGraph {
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        let mut values = HashMap::new();

        let activation = "activation".to_string();
        let normalized = "normalized".to_string();
        let gate_out = "gate_out".to_string();
        let up_out = "up_out".to_string();
        let gated = "gated".to_string();
        let gated_up = "gated_up".to_string();
        let down_out = "down_out".to_string();
        let layer_output = "layer_output".to_string();

        let val = |id: &str, dtype: DType, shape: Vec<usize>| DataflowValue {
            id: id.to_string(),
            dtype,
            shape,
            current_residency: ValueResidency::Unknown,
        };
        values.insert(activation.clone(), val("activation", DType::F32, vec![1, 2048]));
        values.insert(normalized.clone(), val("normalized", DType::F32, vec![1, 2048]));
        values.insert(gate_out.clone(), val("gate_out", DType::F32, vec![1, 8192]));
        values.insert(up_out.clone(), val("up_out", DType::F32, vec![1, 8192]));
        values.insert(gated.clone(), val("gated", DType::F32, vec![1, 8192]));
        values.insert(gated_up.clone(), val("gated_up", DType::F32, vec![1, 8192]));
        values.insert(down_out.clone(), val("down_out", DType::F32, vec![1, 2048]));
        values.insert(layer_output.clone(), val("layer_output", DType::F32, vec![1, 2048]));

        // Node 0: RMSNorm
        nodes.push(DataflowNode {
            id: 0,
            op: DataflowOp::RmsNorm {
                input: activation.clone(),
                weight: "input_layernorm.weight".to_string(),
                output: normalized.clone(),
                epsilon: 1e-6,
            },
            inputs: vec![activation.clone()],
            outputs: vec![normalized.clone()],
        });

        // Node 1: Gate MatMul
        nodes.push(DataflowNode {
            id: 1,
            op: DataflowOp::MatMul {
                lhs: normalized.clone(),
                rhs: "gate_proj.weight".to_string(),
                output: gate_out.clone(),
                contract: MatMulContract {
                    m: 1,
                    n: 8192,
                    k: 2048,
                    lhs_transposed: false,
                    rhs_transposed: true,
                },
            },
            inputs: vec![normalized.clone()],
            outputs: vec![gate_out.clone()],
        });

        // Node 2: Up MatMul
        nodes.push(DataflowNode {
            id: 2,
            op: DataflowOp::MatMul {
                lhs: normalized.clone(),
                rhs: "up_proj.weight".to_string(),
                output: up_out.clone(),
                contract: MatMulContract {
                    m: 1,
                    n: 8192,
                    k: 2048,
                    lhs_transposed: false,
                    rhs_transposed: true,
                },
            },
            inputs: vec![normalized.clone()],
            outputs: vec![up_out.clone()],
        });

        // Node 3: SiLU
        nodes.push(DataflowNode {
            id: 3,
            op: DataflowOp::SiLU {
                input: gate_out.clone(),
                output: gated.clone(),
            },
            inputs: vec![gate_out.clone()],
            outputs: vec![gated.clone()],
        });

        // Node 4: Mul
        nodes.push(DataflowNode {
            id: 4,
            op: DataflowOp::Mul {
                lhs: gated.clone(),
                rhs: up_out.clone(),
                output: gated_up.clone(),
            },
            inputs: vec![gated.clone(), up_out.clone()],
            outputs: vec![gated_up.clone()],
        });

        // Node 5: Down MatMul
        nodes.push(DataflowNode {
            id: 5,
            op: DataflowOp::MatMul {
                lhs: gated_up.clone(),
                rhs: "down_proj.weight".to_string(),
                output: down_out.clone(),
                contract: MatMulContract {
                    m: 1,
                    n: 2048,
                    k: 8192,
                    lhs_transposed: false,
                    rhs_transposed: true,
                },
            },
            inputs: vec![gated_up.clone()],
            outputs: vec![down_out.clone()],
        });

        // Node 6: ResidualAdd
        nodes.push(DataflowNode {
            id: 6,
            op: DataflowOp::ResidualAdd {
                residual: activation.clone(),
                update: down_out.clone(),
                output: layer_output.clone(),
            },
            inputs: vec![activation.clone(), down_out.clone()],
            outputs: vec![layer_output.clone()],
        });

        // Edges
        edges.push(DataflowEdge { producer: 0, consumer: 1, value: normalized.clone() });
        edges.push(DataflowEdge { producer: 0, consumer: 2, value: normalized.clone() });
        edges.push(DataflowEdge { producer: 1, consumer: 3, value: gate_out.clone() });
        edges.push(DataflowEdge { producer: 3, consumer: 4, value: gated.clone() });
        edges.push(DataflowEdge { producer: 2, consumer: 4, value: up_out.clone() });
        edges.push(DataflowEdge { producer: 4, consumer: 5, value: gated_up.clone() });
        edges.push(DataflowEdge { producer: 5, consumer: 6, value: down_out.clone() });

        DataflowGraph {
            nodes,
            edges,
            values,
            layer_id: "gemma_decoder_mlp".to_string(),
        }
    }
}

// Backward-compatibility aliases for Phase 5-7 consumers.
// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// An empty graph must produce an empty topological sort.
    #[test]
    fn empty_graph() {
        let graph = DataflowGraph {
            nodes: vec![],
            edges: vec![],
            values: HashMap::new(),
            layer_id: "empty".to_string(),
        };
        let order = graph.topological_sort();
        assert!(order.is_empty());
    }

    /// The canonical Gemma MLP graph must have 7 nodes, valid topological
    /// order, and correct producer/consumer queries.
    #[test]
    fn gemma_mlp_graph_structure() {
        let graph = DataflowGraphBuilder::build_mlp();

        assert_eq!(graph.nodes.len(), 7, "MLP must have 7 nodes");

        assert!(matches!(graph.nodes[0].op, DataflowOp::RmsNorm { .. }));
        assert!(matches!(graph.nodes[1].op, DataflowOp::MatMul { .. }));
        assert!(matches!(graph.nodes[2].op, DataflowOp::MatMul { .. }));
        assert!(matches!(graph.nodes[3].op, DataflowOp::SiLU { .. }));
        assert!(matches!(graph.nodes[4].op, DataflowOp::Mul { .. }));
        assert!(matches!(graph.nodes[5].op, DataflowOp::MatMul { .. }));
        assert!(matches!(graph.nodes[6].op, DataflowOp::ResidualAdd { .. }));

        // Topological sort - must be valid
        let order = graph.topological_sort();
        assert_eq!(order.len(), 7, "topological sort must include all 7 nodes");

        // Verify topological validity: every edge's producer appears before consumer
        let pos: std::collections::HashMap<usize, usize> = order
            .iter()
            .enumerate()
            .map(|(i, &n)| (n, i))
            .collect();
        for edge in &graph.edges {
            assert!(
                pos[&edge.producer] < pos[&edge.consumer],
                "edge {} -> {} violates topological order",
                edge.producer,
                edge.consumer
            );
        }

        // With FIFO order, we expect intuitive [0, 1, 2, 3, 4, 5, 6]
        assert_eq!(order, vec![0, 1, 2, 3, 4, 5, 6]);

        // Producer/consumer queries
        assert_eq!(graph.producer_of("normalized"), Some(0));
        assert_eq!(graph.producer_of("gate_out"), Some(1));
        assert_eq!(graph.producer_of("gated"), Some(3));
        assert_eq!(graph.producer_of("nonexistent"), None);

        let consumers_normalized = graph.consumers_of("normalized");
        assert_eq!(consumers_normalized.len(), 2);
        assert!(consumers_normalized.contains(&1));
        assert!(consumers_normalized.contains(&2));

        let consumers_gated_up = graph.consumers_of("gated_up");
        assert_eq!(consumers_gated_up, vec![5]);

        let boundaries = graph.materialization_boundaries();
        assert!(boundaries.is_empty(), "bare MLP has no boundaries");
    }

    /// FusedGroup must serialize and deserialize correctly.
    #[test]
    fn fused_group_roundtrip() {
        let node = DataflowNode {
            id: 1,
            op: DataflowOp::MatMul {
                lhs: "normalized".into(),
                rhs: "gate_proj.weight".into(),
                output: "gate_out".into(),
                contract: MatMulContract {
                    m: 1, n: 8192, k: 2048,
                    lhs_transposed: false,
                    rhs_transposed: true,
                },
            },
            inputs: vec!["normalized".to_string()],
            outputs: vec!["gate_out".to_string()],
        };
        let group = FusedGroup {
            id: "42".to_string(),
            body: vec![node],
            inputs: vec!["normalized".to_string()],
            outputs: vec!["gated_up".to_string()],
            internal_values: vec!["gate_out".to_string(), "up_out".to_string()],
            codec_family: CodecFamily::Nf4,
            precision_plan: None,
        };

        let json = serde_json::to_string(&group).expect("serialize FusedGroup");
        let restored: FusedGroup = serde_json::from_str(&json).expect("deserialize FusedGroup");

        assert_eq!(restored.id, "42");
        assert_eq!(restored.body.len(), 1);
        assert_eq!(restored.body[0].id, 1);
        assert_eq!(restored.inputs, vec!["normalized"]);
        assert_eq!(restored.outputs, vec!["gated_up"]);
        assert_eq!(restored.internal_values, vec!["gate_out", "up_out"]);
    }

    /// All DataflowOp variants must serialize and deserialize.
    #[test]
    fn dataflow_op_roundtrip() {
        use crate::execution_profile::{
            GroupAxis, MetadataLayout, StorageOrder, TileFamily, TileShape as ProfileTileShape,
        };

        let ops: Vec<DataflowOp> = vec![
            DataflowOp::LoadWeight {
                tensor: "w1".into(),
                codec: CodecFamily::Nf4,
                layout: PhysicalTileLayout {
                    format: "nf4_tile640".into(),
                    tile_family: TileFamily::tile640(),
                    logical_shape: [2048, 8192],
                    storage_order: StorageOrder::RowMajor,
                    tile_shape: ProfileTileShape { rows: 640, cols: 640 },
                    group_size: 32,
                    group_axis: GroupAxis::PackedContiguous,
                    metadata_layout: MetadataLayout::AdjacentTile,
                    padding_policy: "zero".into(),
                    alignment_bytes: 256,
                    interleave: "none".into(),
                },
            },
            DataflowOp::Dequantize { input: "quantized".into(), output_dtype: DType::F32 },
            DataflowOp::MatMul {
                lhs: "a".into(), rhs: "b".into(), output: "c".into(),
                contract: MatMulContract { m: 1, n: 8192, k: 2048, lhs_transposed: false, rhs_transposed: true },
            },
            DataflowOp::RmsNorm { input: "x".into(), weight: "ln.weight".into(), output: "y".into(), epsilon: 1e-6 },
            DataflowOp::SiLU { input: "x".into(), output: "y".into() },
            DataflowOp::Gelu { input: "x".into(), output: "y".into() },
            DataflowOp::Mul { lhs: "a".into(), rhs: "b".into(), output: "c".into() },
            DataflowOp::Add { lhs: "a".into(), rhs: "b".into(), output: "c".into() },
            DataflowOp::ResidualAdd { residual: "res".into(), update: "upd".into(), output: "out".into() },
            DataflowOp::StoreActivation { slot: "act_0".into(), input: "x".into() },
            DataflowOp::KvRead { slot: "k_0".into(), output: "k".into() },
            DataflowOp::KvWrite { slot: "v_0".into(), input: "v".into() },
        ];

        for op in &ops {
            let json = serde_json::to_string(op).expect("serialize DataflowOp");
            let _restored: DataflowOp = serde_json::from_str(&json).expect("deserialize DataflowOp");
        }
    }
}
