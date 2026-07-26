//! Fusion analysis — dataflow graph construction and fusion-group
//! identification from layer role assignments.
//!
//! This module owns the canonical authority for the fusion analysis
//! step that runs between model loading and kernel selection:
//!
//! 1. **Graph construction** — for each `Layer`, collect its tensors
//!    by `CanonicalRole`, and build a `DataflowGraph` containing the
//!    canonical MLP triplet (Gate-SiLU-Mul-Down) plus any standalone
//!    projection / norm roles.
//! 2. **Fusion-group identification** — find the `MatMul` roots in
//!    the graph and record which elementwise ops (SiLU, Gelu, Mul,
//!    Add) are inlined into each root's kernel.
//! 3. **Dispatch entity emission** — the caller (the schedule) stages
//!    one `Dispatch` entity per fusion group through a `WorldTxn`,
//!    with a `FusionGroup` component recording the root op kind and
//!    fused op kinds.
//!
//! ## Authority boundary
//!
//! This module does **not** own:
//! - The dataflow graph IR itself (owned by the spatial IR / fusion
//!   scheduler).
//! - The kernel lowerer (owned by `prism-ecs-kernel`).
//! - The dispatch entity lifecycle (owned by fusion scheduling).
//!
//! All exposed types are pure value types. The module never mutates
//! the world directly.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Role / graph node / edge
// ---------------------------------------------------------------------------

/// Canonical tensor role — vendor-neutral, model-family independent.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CanonicalRole {
    Embedding,
    AttnNorm(u32),
    Q(u32),
    K(u32),
    V(u32),
    O(u32),
    QNorm(u32),
    KNorm(u32),
    MlpNorm(u32),
    Gate(u32),
    Up(u32),
    Down(u32),
    GateEx(u32, u32),
    UpEx(u32, u32),
    DownEx(u32, u32),
    RouterWeight(u32),
    SharedGate,
    SharedUp,
    SharedDown,
    SharedGateL(u32),
    SharedUpL(u32),
    SharedDownL(u32),
    CompressWeight(u32),
    IndexerWeight(u32),
    WindowK(u32),
    WindowV(u32),
    HCWeight(u32),
    FinalNorm,
    LmHead,
}

/// Op kind in the dataflow graph.
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
    AneMatMul,
    AneConv1x1,
    AneLoadWeight,
    AneStoreOutput,
}

/// A node in the dataflow graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataflowNode {
    pub id: usize,
    pub op: DataflowOp,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
}

/// Op shape — the typed payload of a `DataflowNode`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DataflowOp {
    LoadWeight {
        tensor: String,
        codec: String,
        layout: String,
    },
    Dequantize {
        input: String,
        output_dtype: String,
    },
    MatMul {
        lhs: String,
        rhs: String,
        output: String,
        contract: MatMulContract,
    },
    RmsNorm {
        input: String,
        weight: String,
        output: String,
        epsilon: f64,
    },
    SiLU {
        input: String,
        output: String,
    },
    Gelu {
        input: String,
        output: String,
    },
    Mul {
        lhs: String,
        rhs: String,
        output: String,
    },
    Add {
        lhs: String,
        rhs: String,
        output: String,
    },
    ResidualAdd {
        residual: String,
        update: String,
        output: String,
    },
    StoreActivation {
        slot: String,
        input: String,
    },
    KvRead {
        slot: String,
        output: String,
    },
    KvWrite {
        slot: String,
        input: String,
    },
    EngramLookup {
        engram_id: String,
        weights: String,
        output: String,
    },
    AneMatMul {
        lhs: String,
        rhs: String,
        output: String,
        contract: MatMulContract,
        sram_budget: u32,
    },
    AneConv1x1 {
        input: String,
        weight: String,
        output: String,
        sram_budget: u32,
    },
    AneLoadWeight {
        tensor: String,
        codec: String,
        layout: String,
        target_sram_region: u32,
    },
    AneStoreOutput {
        input: String,
        offset: u32,
    },
}

/// MatMul shape contract — describes the GEMM dimensions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatMulContract {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub lhs_transposed: bool,
    pub rhs_transposed: bool,
}

/// A value carried by the dataflow graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataflowValue {
    pub id: String,
    pub dtype: String,
    pub shape: Vec<u32>,
    pub current_residency: ValueResidency,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueResidency {
    Unknown,
    Cpu,
    Gpu,
    Ane,
}

/// A directed edge in the dataflow graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataflowEdge {
    pub producer: usize,
    pub consumer: usize,
    pub value: String,
}

/// The full dataflow graph for a layer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DataflowGraph {
    pub nodes: Vec<DataflowNode>,
    pub edges: Vec<DataflowEdge>,
    pub values: BTreeMap<String, DataflowValue>,
    pub layer_id: String,
}

/// A fusion group — one root op plus zero or more fused elementwise
/// ops that can be inlined into the root kernel.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FusionGroupSpec {
    pub root_op_kind: String,
    pub fused_op_kinds: Vec<String>,
}

impl prism_ecs_core::Component for FusionGroupSpec {}

/// Stable handle to a layer's dataflow graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DataflowGraphHandle(pub String);

impl prism_ecs_core::Component for DataflowGraphHandle {}

impl std::fmt::Display for CanonicalRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CanonicalRole::Embedding => write!(f, "Embedding"),
            CanonicalRole::AttnNorm(l) => write!(f, "AttnNorm({l})"),
            CanonicalRole::Q(l) => write!(f, "Q({l})"),
            CanonicalRole::K(l) => write!(f, "K({l})"),
            CanonicalRole::V(l) => write!(f, "V({l})"),
            CanonicalRole::O(l) => write!(f, "O({l})"),
            CanonicalRole::QNorm(l) => write!(f, "QNorm({l})"),
            CanonicalRole::KNorm(l) => write!(f, "KNorm({l})"),
            CanonicalRole::MlpNorm(l) => write!(f, "MlpNorm({l})"),
            CanonicalRole::Gate(l) => write!(f, "Gate({l})"),
            CanonicalRole::Up(l) => write!(f, "Up({l})"),
            CanonicalRole::Down(l) => write!(f, "Down({l})"),
            CanonicalRole::GateEx(l, e) => write!(f, "GateEx({l},{e})"),
            CanonicalRole::UpEx(l, e) => write!(f, "UpEx({l},{e})"),
            CanonicalRole::DownEx(l, e) => write!(f, "DownEx({l},{e})"),
            CanonicalRole::RouterWeight(l) => write!(f, "RouterWeight({l})"),
            CanonicalRole::SharedGate => write!(f, "SharedGate"),
            CanonicalRole::SharedUp => write!(f, "SharedUp"),
            CanonicalRole::SharedDown => write!(f, "SharedDown"),
            CanonicalRole::SharedGateL(l) => write!(f, "SharedGateL({l})"),
            CanonicalRole::SharedUpL(l) => write!(f, "SharedUpL({l})"),
            CanonicalRole::SharedDownL(l) => write!(f, "SharedDownL({l})"),
            CanonicalRole::CompressWeight(l) => write!(f, "CompressWeight({l})"),
            CanonicalRole::IndexerWeight(l) => write!(f, "IndexerWeight({l})"),
            CanonicalRole::WindowK(l) => write!(f, "WindowK({l})"),
            CanonicalRole::WindowV(l) => write!(f, "WindowV({l})"),
            CanonicalRole::HCWeight(l) => write!(f, "HCWeight({l})"),
            CanonicalRole::FinalNorm => write!(f, "FinalNorm"),
            CanonicalRole::LmHead => write!(f, "LmHead"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum FusionAnalysisError {
    #[error("graph for layer `{0}` is empty")]
    EmptyGraph(String),
}

// ---------------------------------------------------------------------------
// Graph construction
// ---------------------------------------------------------------------------

/// Build a `DataflowGraph` for a single layer, given the layer's roles.
///
/// For MLP layers (Gate + Up + Down roles present) this synthesises
/// the canonical MLP subgraph (Gate MatMul → SiLU; Up MatMul; SiLU ⊕
/// Up → Mul; Mul → Down MatMul). For attention layers (Q, K, V, O)
/// each projection becomes an independent MatMul node. For standalone
/// MatMul roles, a single MatMul node is emitted.
pub fn build_graph_for_layer(
    layer_id: u32,
    roles: &[CanonicalRole],
) -> DataflowGraph {
    let mut nodes: Vec<DataflowNode> = Vec::new();
    let mut edges: Vec<DataflowEdge> = Vec::new();
    let mut values: BTreeMap<String, DataflowValue> = BTreeMap::new();
    let mut node_id: usize = 0;

    let emit_matmul = |nodes: &mut Vec<DataflowNode>,
                       values: &mut BTreeMap<String, DataflowValue>,
                       node_id: &mut usize,
                       lhs_buf: &str,
                       rhs_tensor: &str,
                       output_buf: &str|
     -> usize {
        let id = *node_id;
        values.entry(lhs_buf.to_string()).or_insert_with(|| DataflowValue {
            id: lhs_buf.to_string(),
            dtype: "F32".into(),
            shape: vec![1, 2048],
            current_residency: ValueResidency::Unknown,
        });
        values.entry(output_buf.to_string()).or_insert_with(|| DataflowValue {
            id: output_buf.to_string(),
            dtype: "F32".into(),
            shape: vec![1, 2048],
            current_residency: ValueResidency::Unknown,
        });
        nodes.push(DataflowNode {
            id,
            op: DataflowOp::MatMul {
                lhs: lhs_buf.to_string(),
                rhs: rhs_tensor.to_string(),
                output: output_buf.to_string(),
                contract: MatMulContract {
                    m: 1,
                    n: 2048,
                    k: 2048,
                    lhs_transposed: false,
                    rhs_transposed: true,
                },
            },
            inputs: vec![lhs_buf.to_string()],
            outputs: vec![output_buf.to_string()],
        });
        *node_id += 1;
        id
    };

    if has_mlp_triplet(roles) {
        let activation = "mlp_activation";
        let gate_out = "gate_out";
        let up_out = "up_out";
        let gated = "gated";
        let gated_up = "gated_up";
        let down_out = "down_out";

        let _gate_id = emit_matmul(
            &mut nodes,
            &mut values,
            &mut node_id,
            activation,
            "Gate(0)",
            gate_out,
        );
        let _up_id = emit_matmul(
            &mut nodes,
            &mut values,
            &mut node_id,
            activation,
            "Up(0)",
            up_out,
        );

        let silu_id = nodes.len();
        values.entry(gate_out.to_string()).or_insert_with(|| DataflowValue {
            id: gate_out.to_string(),
            dtype: "F32".into(),
            shape: vec![1, 2048],
            current_residency: ValueResidency::Unknown,
        });
        values.entry(gated.to_string()).or_insert_with(|| DataflowValue {
            id: gated.to_string(),
            dtype: "F32".into(),
            shape: vec![1, 2048],
            current_residency: ValueResidency::Unknown,
        });
        nodes.push(DataflowNode {
            id: silu_id,
            op: DataflowOp::SiLU {
                input: gate_out.to_string(),
                output: gated.to_string(),
            },
            inputs: vec![gate_out.to_string()],
            outputs: vec![gated.to_string()],
        });
        node_id += 1;

        let mul_id = nodes.len();
        nodes.push(DataflowNode {
            id: mul_id,
            op: DataflowOp::Mul {
                lhs: gated.to_string(),
                rhs: up_out.to_string(),
                output: gated_up.to_string(),
            },
            inputs: vec![gated.to_string(), up_out.to_string()],
            outputs: vec![gated_up.to_string()],
        });
        node_id += 1;

        let _down_id = emit_matmul(
            &mut nodes,
            &mut values,
            &mut node_id,
            gated_up,
            "Down(0)",
            down_out,
        );

        edges.push(DataflowEdge {
            producer: 0,
            consumer: 2,
            value: gate_out.to_string(),
        });
        edges.push(DataflowEdge {
            producer: 2,
            consumer: 3,
            value: gated.to_string(),
        });
        edges.push(DataflowEdge {
            producer: 1,
            consumer: 3,
            value: up_out.to_string(),
        });
        edges.push(DataflowEdge {
            producer: 3,
            consumer: 4,
            value: gated_up.to_string(),
        });
    }

    for role in roles {
        if matches!(role, CanonicalRole::Gate(_) | CanonicalRole::Up(_) | CanonicalRole::Down(_))
            && has_mlp_triplet(roles)
        {
            continue;
        }
        match role_to_op_kind(role) {
            Some(DataflowOpKind::MatMul) => {
                let tensor_name = role.to_string();
                let output_buf = format!("{}_output", tensor_name);
                emit_matmul(
                    &mut nodes,
                    &mut values,
                    &mut node_id,
                    &format!("{}_input", tensor_name),
                    &tensor_name,
                    &output_buf,
                );
            }
            Some(DataflowOpKind::RmsNorm) => {
                let tensor_name = role.to_string();
                let input_buf = format!("{}_input", tensor_name);
                let output_buf = format!("{}_output", tensor_name);
                values.entry(input_buf.clone()).or_insert_with(|| DataflowValue {
                    id: input_buf.clone(),
                    dtype: "F32".into(),
                    shape: vec![1, 2048],
                    current_residency: ValueResidency::Unknown,
                });
                values.entry(output_buf.clone()).or_insert_with(|| DataflowValue {
                    id: output_buf.clone(),
                    dtype: "F32".into(),
                    shape: vec![1, 2048],
                    current_residency: ValueResidency::Unknown,
                });
                let id = node_id;
                nodes.push(DataflowNode {
                    id,
                    op: DataflowOp::RmsNorm {
                        input: input_buf.clone(),
                        weight: tensor_name,
                        output: output_buf.clone(),
                        epsilon: 1e-6,
                    },
                    inputs: vec![input_buf],
                    outputs: vec![output_buf],
                });
                node_id += 1;
            }
            _ => {}
        }
    }

    DataflowGraph {
        nodes,
        edges,
        values,
        layer_id: format!("layer_{}", layer_id),
    }
}

/// Check whether a role set contains the canonical MLP triplet.
pub fn has_mlp_triplet(roles: &[CanonicalRole]) -> bool {
    let has_gate = roles.iter().any(|r| matches!(r, CanonicalRole::Gate(_)));
    let has_up = roles.iter().any(|r| matches!(r, CanonicalRole::Up(_)));
    let has_down = roles.iter().any(|r| matches!(r, CanonicalRole::Down(_)));
    has_gate && has_up && has_down
}

/// Map a `CanonicalRole` to the `DataflowOpKind` it contributes.
pub fn role_to_op_kind(role: &CanonicalRole) -> Option<DataflowOpKind> {
    match role {
        CanonicalRole::Gate(_)
        | CanonicalRole::Up(_)
        | CanonicalRole::Down(_)
        | CanonicalRole::Q(_)
        | CanonicalRole::K(_)
        | CanonicalRole::V(_)
        | CanonicalRole::O(_)
        | CanonicalRole::GateEx(_, _)
        | CanonicalRole::UpEx(_, _)
        | CanonicalRole::DownEx(_, _)
        | CanonicalRole::SharedGate
        | CanonicalRole::SharedUp
        | CanonicalRole::SharedDown
        | CanonicalRole::SharedGateL(_)
        | CanonicalRole::SharedUpL(_)
        | CanonicalRole::SharedDownL(_)
        | CanonicalRole::CompressWeight(_)
        | CanonicalRole::IndexerWeight(_)
        | CanonicalRole::WindowK(_)
        | CanonicalRole::WindowV(_)
        | CanonicalRole::HCWeight(_)
        | CanonicalRole::LmHead => Some(DataflowOpKind::MatMul),
        CanonicalRole::AttnNorm(_) | CanonicalRole::MlpNorm(_) | CanonicalRole::FinalNorm => {
            Some(DataflowOpKind::RmsNorm)
        }
        CanonicalRole::Embedding
        | CanonicalRole::QNorm(_)
        | CanonicalRole::KNorm(_)
        | CanonicalRole::RouterWeight(_) => None,
    }
}

/// Produce a concise label for an op, used as `root_op_kind` / fused kind.
pub fn op_kind_label(op: &DataflowOp) -> String {
    match op {
        DataflowOp::LoadWeight { .. } => "LoadWeight",
        DataflowOp::Dequantize { .. } => "Dequantize",
        DataflowOp::MatMul { .. } => "MatMul",
        DataflowOp::RmsNorm { .. } => "RmsNorm",
        DataflowOp::SiLU { .. } => "SiLU",
        DataflowOp::Gelu { .. } => "Gelu",
        DataflowOp::Mul { .. } => "Mul",
        DataflowOp::Add { .. } => "Add",
        DataflowOp::ResidualAdd { .. } => "ResidualAdd",
        DataflowOp::StoreActivation { .. } => "StoreActivation",
        DataflowOp::KvRead { .. } => "KvRead",
        DataflowOp::KvWrite { .. } => "KvWrite",
        DataflowOp::EngramLookup { .. } => "EngramLookup",
        DataflowOp::AneMatMul { .. } => "AneMatMul",
        DataflowOp::AneConv1x1 { .. } => "AneConv1x1",
        DataflowOp::AneLoadWeight { .. } => "AneLoadWeight",
        DataflowOp::AneStoreOutput { .. } => "AneStoreOutput",
    }
    .to_string()
}

/// Analyse the dataflow graph to identify fusion opportunities.
pub fn analyse_fusion_groups(graph: &DataflowGraph) -> Vec<FusionGroupSpec> {
    // Build a consumer map: which buffer names are consumed by which nodes.
    let mut buffer_consumers: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for edge in &graph.edges {
        buffer_consumers
            .entry(edge.value.as_str())
            .or_default()
            .push(edge.consumer);
    }

    let mut groups = Vec::new();

    for node in &graph.nodes {
        let root_kind = op_kind_label(&node.op);
        if !matches!(node.op, DataflowOp::MatMul { .. } | DataflowOp::AneMatMul { .. }) {
            continue;
        }
        let mut fused = Vec::new();
        for output in &node.outputs {
            if let Some(consumers) = buffer_consumers.get(output.as_str()) {
                for &consumer_idx in consumers {
                    if let Some(consumer) = graph.nodes.get(consumer_idx) {
                        if matches!(
                            consumer.op,
                            DataflowOp::SiLU { .. }
                                | DataflowOp::Gelu { .. }
                                | DataflowOp::Mul { .. }
                                | DataflowOp::Add { .. }
                        ) {
                            fused.push(op_kind_label(&consumer.op));
                        }
                    }
                }
            }
        }
        groups.push(FusionGroupSpec {
            root_op_kind: root_kind,
            fused_op_kinds: fused,
        });
    }

    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_mlp_triplet_detects_all_three_roles() {
        let roles = vec![
            CanonicalRole::Gate(0),
            CanonicalRole::Up(0),
            CanonicalRole::Down(0),
        ];
        assert!(has_mlp_triplet(&roles));
    }

    #[test]
    fn has_mlp_triplet_rejects_missing_role() {
        let roles = vec![CanonicalRole::Gate(0), CanonicalRole::Up(0)];
        assert!(!has_mlp_triplet(&roles));
    }

    #[test]
    fn role_to_op_kind_maps_projections_to_matmul() {
        assert_eq!(role_to_op_kind(&CanonicalRole::Q(1)), Some(DataflowOpKind::MatMul));
        assert_eq!(role_to_op_kind(&CanonicalRole::LmHead), Some(DataflowOpKind::MatMul));
    }

    #[test]
    fn role_to_op_kind_maps_norms_to_rms_norm() {
        assert_eq!(
            role_to_op_kind(&CanonicalRole::AttnNorm(1)),
            Some(DataflowOpKind::RmsNorm)
        );
        assert_eq!(
            role_to_op_kind(&CanonicalRole::FinalNorm),
            Some(DataflowOpKind::RmsNorm)
        );
    }

    #[test]
    fn role_to_op_kind_returns_none_for_embeddings() {
        assert_eq!(role_to_op_kind(&CanonicalRole::Embedding), None);
        assert_eq!(role_to_op_kind(&CanonicalRole::RouterWeight(0)), None);
    }

    #[test]
    fn build_mlp_graph_has_5_nodes() {
        let roles = vec![
            CanonicalRole::Gate(0),
            CanonicalRole::Up(0),
            CanonicalRole::Down(0),
        ];
        let g = build_graph_for_layer(0, &roles);
        assert_eq!(g.nodes.len(), 5, "MLP graph has Gate, Up, SiLU, Mul, Down");
    }

    #[test]
    fn build_mlp_graph_has_4_edges() {
        let roles = vec![
            CanonicalRole::Gate(0),
            CanonicalRole::Up(0),
            CanonicalRole::Down(0),
        ];
        let g = build_graph_for_layer(0, &roles);
        assert_eq!(g.edges.len(), 4);
    }

    #[test]
    fn build_attention_layer_emits_4_matmuls() {
        let roles = vec![
            CanonicalRole::Q(0),
            CanonicalRole::K(0),
            CanonicalRole::V(0),
            CanonicalRole::O(0),
        ];
        let g = build_graph_for_layer(0, &roles);
        let matmul_count = g
            .nodes
            .iter()
            .filter(|n| matches!(n.op, DataflowOp::MatMul { .. }))
            .count();
        assert_eq!(matmul_count, 4);
    }

    #[test]
    fn build_layer_with_norm_emits_rms_norm() {
        let roles = vec![CanonicalRole::AttnNorm(0)];
        let g = build_graph_for_layer(0, &roles);
        let rms_count = g
            .nodes
            .iter()
            .filter(|n| matches!(n.op, DataflowOp::RmsNorm { .. }))
            .count();
        assert_eq!(rms_count, 1);
    }

    #[test]
    fn analyse_mlp_finds_one_root_with_fused_ops() {
        let roles = vec![
            CanonicalRole::Gate(0),
            CanonicalRole::Up(0),
            CanonicalRole::Down(0),
        ];
        let g = build_graph_for_layer(0, &roles);
        let groups = analyse_fusion_groups(&g);
        let matmul_roots: Vec<&FusionGroupSpec> = groups
            .iter()
            .filter(|g| g.root_op_kind == "MatMul")
            .collect();
        assert!(!matmul_roots.is_empty());
        // At least one root should have at least one fused op.
        let has_fused = matmul_roots.iter().any(|g| !g.fused_op_kinds.is_empty());
        assert!(has_fused);
    }

    #[test]
    fn op_kind_label_round_trip() {
        let op = DataflowOp::MatMul {
            lhs: "a".into(),
            rhs: "b".into(),
            output: "c".into(),
            contract: MatMulContract {
                m: 1,
                n: 2048,
                k: 2048,
                lhs_transposed: false,
                rhs_transposed: false,
            },
        };
        assert_eq!(op_kind_label(&op), "MatMul");
        let op = DataflowOp::SiLU {
            input: "x".into(),
            output: "y".into(),
        };
        assert_eq!(op_kind_label(&op), "SiLU");
    }
}
