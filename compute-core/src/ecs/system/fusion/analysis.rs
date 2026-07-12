use crate::ecs::adapter::CanonicalRole;
use crate::ecs::component::fusion::{DataflowGraphHandle, FusionGroup};
use crate::ecs::component::tensor::{CanonicalRoleComp, LayerIndex};
use crate::ecs::plan::fusion::{
    DataflowEdge, DataflowGraph, DataflowNode, DataflowOp, DataflowOpKind, DataflowValue,
    MatMulContract, ValueResidency,
};
use crate::ecs::plan::DType;
use crate::ecs::{CompEntity, CompWorld, CompilerSystem, EntityKind, SchedulePhase};

use std::collections::HashMap;

pub struct FusionAnalysisSystem;
impl CompilerSystem for FusionAnalysisSystem {
    fn name(&self) -> &str {
        "FusionAnalysisSystem"
    }
    fn phase(&self) -> SchedulePhase {
        SchedulePhase::FusionDispatch
    }
    fn run(&self, world: &mut CompWorld) -> anyhow::Result<()> {
        let layers: Vec<CompEntity> = world.entities_of_kind(EntityKind::Layer);
        let all_tensors: Vec<CompEntity> = world.entities_of_kind(EntityKind::Tensor);

        for layer in &layers {
            let layer_idx = match world.get_component::<LayerIndex>(*layer) {
                Some(idx) => idx.0,
                None => continue,
            };

            // Collect tensors whose CanonicalRole carries this layer index.
            let layer_tensors: Vec<CompEntity> = all_tensors
                .iter()
                .filter(|t| {
                    world
                        .get_component::<CanonicalRoleComp>(**t)
                        .map(|c| role_layer(&c.0) == Some(layer_idx))
                        .unwrap_or(false)
                })
                .copied()
                .collect();

            if layer_tensors.is_empty() {
                continue;
            }

            // Detect which roles are present for this layer.
            let roles: Vec<CanonicalRole> = layer_tensors
                .iter()
                .filter_map(|t| world.get_component::<CanonicalRoleComp>(*t).map(|c| c.0))
                .collect();

            // Build a DataflowGraph with both MatMul roots and synthetic
            // elementwise ops for any MLP pattern present.
            let graph = build_graph_for_layer(world, &layer_tensors, &roles, layer_idx);

            // Analyse the graph: find MatMul roots and fuse elementwise ops.
            let fusion_groups = analyse_fusion_groups(&graph);

            // Create one Dispatch entity per fusion group.
            for group in &fusion_groups {
                let dispatch = world.spawn(EntityKind::Dispatch, None);
                world.add_component(
                    dispatch,
                    FusionGroup {
                        root_op_kind: group.root_op_kind.clone(),
                        fused_op_kinds: group.fused_op_kinds.clone(),
                        binding_slots: (1 + group.fused_op_kinds.len()) as u32,
                        accepted: true,
                        reject_reason: None,
                    },
                );
            }

            // Attach a handle so downstream systems can reference the graph.
            world.add_component(
                *layer,
                DataflowGraphHandle(format!("fusion_graph_layer_{}", layer_idx)),
            );
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers: role → layer index
// ---------------------------------------------------------------------------

/// Extract the layer index from a `CanonicalRole`, if one is embedded.
fn role_layer(role: &CanonicalRole) -> Option<u32> {
    match role {
        CanonicalRole::AttnNorm(l)
        | CanonicalRole::Q(l)
        | CanonicalRole::K(l)
        | CanonicalRole::V(l)
        | CanonicalRole::O(l)
        | CanonicalRole::MlpNorm(l)
        | CanonicalRole::Gate(l)
        | CanonicalRole::Up(l)
        | CanonicalRole::Down(l)
        | CanonicalRole::QNorm(l)
        | CanonicalRole::KNorm(l)
        | CanonicalRole::CompressWeight(l)
        | CanonicalRole::IndexerWeight(l)
        | CanonicalRole::WindowK(l)
        | CanonicalRole::WindowV(l)
        | CanonicalRole::HCWeight(l) => Some(*l),
        CanonicalRole::GateEx(l, _) | CanonicalRole::UpEx(l, _) | CanonicalRole::DownEx(l, _) => {
            Some(*l)
        }
        CanonicalRole::RouterWeight(l)
        | CanonicalRole::SharedGateL(l)
        | CanonicalRole::SharedUpL(l)
        | CanonicalRole::SharedDownL(l) => Some(*l),
        CanonicalRole::Embedding
        | CanonicalRole::FinalNorm
        | CanonicalRole::LmHead
        | CanonicalRole::SharedGate
        | CanonicalRole::SharedUp
        | CanonicalRole::SharedDown => None,
    }
}

/// Map a `CanonicalRole` to the `DataflowOpKind` it contributes.
fn role_to_op_kind(role: &CanonicalRole) -> Option<DataflowOpKind> {
    match role {
        // All weight projections become MatMul.
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
        // Normalisation roles become RmsNorm.
        CanonicalRole::AttnNorm(_) | CanonicalRole::MlpNorm(_) | CanonicalRole::FinalNorm => {
            Some(DataflowOpKind::RmsNorm)
        }
        // Non-compute roles (embedding, router, Q/K-norm) don't produce graph
        // nodes at the fusion-analysis level.
        CanonicalRole::Embedding
        | CanonicalRole::QNorm(_)
        | CanonicalRole::KNorm(_)
        | CanonicalRole::RouterWeight(_) => None,
    }
}

/// Quick check: does the role set contain the canonical MLP triplet?
fn has_mlp_triplet(roles: &[CanonicalRole]) -> bool {
    let has_gate = roles.iter().any(|r| matches!(r, CanonicalRole::Gate(_)));
    let has_up = roles.iter().any(|r| matches!(r, CanonicalRole::Up(_)));
    let has_down = roles.iter().any(|r| matches!(r, CanonicalRole::Down(_)));
    has_gate && has_up && has_down
}

// ---------------------------------------------------------------------------
// Graph construction
// ---------------------------------------------------------------------------

/// Build a `DataflowGraph` from a layer's tensor entities.
///
/// For MLP layers (Gate + Up + Down roles present) this synthesises the
/// canonical MLP subgraph:
///
///   Gate MatMul ──→ SiLU ──┐
///                           ├──→ Mul ──→ Down MatMul
///   Up  MatMul  ───────────┘
///
/// For attention layers (Q, K, V, O), each projection becomes an independent
/// MatMul node (the attention fusion is handled downstream).
/// For standalone MatMul roles, a single MatMul node is emitted.
fn build_graph_for_layer(
    _world: &CompWorld,
    _tensors: &[CompEntity],
    roles: &[CanonicalRole],
    layer_idx: u32,
) -> DataflowGraph {
    let mut nodes: Vec<DataflowNode> = Vec::new();
    let mut edges: Vec<DataflowEdge> = Vec::new();
    let mut values: HashMap<String, DataflowValue> = HashMap::new();
    let mut node_id: usize = 0;

    // Shared value factory.
    let val = |id: &str, dtype: DType| -> DataflowValue {
        DataflowValue {
            id: id.to_string(),
            dtype,
            shape: vec![1, 2048],
            current_residency: ValueResidency::Unknown,
        }
    };

    // Emit a MatMul node against a specific weight-tensor role.
    let emit_matmul = |nodes: &mut Vec<DataflowNode>,
                       values: &mut HashMap<String, DataflowValue>,
                       node_id: &mut usize,
                       lhs_buf: &str,
                       rhs_tensor: &str,
                       output_buf: &str|
     -> usize {
        let id = *node_id;
        values
            .entry(lhs_buf.to_string())
            .or_insert_with(|| val(lhs_buf, DType::F32));
        values
            .entry(output_buf.to_string())
            .or_insert_with(|| val(output_buf, DType::F32));
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

    let emit_silu = |nodes: &mut Vec<DataflowNode>,
                     values: &mut HashMap<String, DataflowValue>,
                     node_id: &mut usize,
                     input_buf: &str,
                     output_buf: &str|
     -> usize {
        let id = *node_id;
        values
            .entry(input_buf.to_string())
            .or_insert_with(|| val(input_buf, DType::F32));
        values
            .entry(output_buf.to_string())
            .or_insert_with(|| val(output_buf, DType::F32));
        nodes.push(DataflowNode {
            id,
            op: DataflowOp::SiLU {
                input: input_buf.to_string(),
                output: output_buf.to_string(),
            },
            inputs: vec![input_buf.to_string()],
            outputs: vec![output_buf.to_string()],
        });
        *node_id += 1;
        id
    };

    let emit_mul = |nodes: &mut Vec<DataflowNode>,
                    values: &mut HashMap<String, DataflowValue>,
                    node_id: &mut usize,
                    lhs: &str,
                    rhs: &str,
                    output_buf: &str|
     -> usize {
        let id = *node_id;
        values
            .entry(lhs.to_string())
            .or_insert_with(|| val(lhs, DType::F32));
        values
            .entry(rhs.to_string())
            .or_insert_with(|| val(rhs, DType::F32));
        values
            .entry(output_buf.to_string())
            .or_insert_with(|| val(output_buf, DType::F32));
        nodes.push(DataflowNode {
            id,
            op: DataflowOp::Mul {
                lhs: lhs.to_string(),
                rhs: rhs.to_string(),
                output: output_buf.to_string(),
            },
            inputs: vec![lhs.to_string(), rhs.to_string()],
            outputs: vec![output_buf.to_string()],
        });
        *node_id += 1;
        id
    };

    // ------------------------------------------------------------------
    // MLP pattern with canonical Gate-SiLU-Mul-Down fusion topology
    // ------------------------------------------------------------------
    if has_mlp_triplet(roles) {
        // Node 0: Gate MatMul (normalized → gate_out)
        let activation = "mlp_activation";
        let gate_out = "gate_out";
        let up_out = "up_out";
        let gated = "gated";
        let gated_up = "gated_up";
        let down_out = "down_out";

        let gate_id = emit_matmul(
            &mut nodes,
            &mut values,
            &mut node_id,
            activation,
            &CanonicalRole::Gate(0).to_string(),
            gate_out,
        );

        // Node 1: Up MatMul (normalized → up_out)
        let up_id = emit_matmul(
            &mut nodes,
            &mut values,
            &mut node_id,
            activation,
            &CanonicalRole::Up(0).to_string(),
            up_out,
        );

        // Edge: normalized value flows to both Gate and Up.
        // Only edge from node 0 → node 1 since both read the same activation.
        // (node IDs are 0 and 1; the activation buffer ties them.)

        // Node 2: SiLU(gate_out) → gated
        let silu_id = emit_silu(&mut nodes, &mut values, &mut node_id, gate_out, gated);

        // Edge: gate_out from Gate → SiLU
        edges.push(DataflowEdge {
            producer: gate_id,
            consumer: silu_id,
            value: gate_out.to_string(),
        });

        // Node 3: Mul(gated, up_out) → gated_up
        let mul_id = emit_mul(
            &mut nodes,
            &mut values,
            &mut node_id,
            gated,
            up_out,
            gated_up,
        );

        // Edges: gated from SiLU → Mul; up_out from Up → Mul
        edges.push(DataflowEdge {
            producer: silu_id,
            consumer: mul_id,
            value: gated.to_string(),
        });
        edges.push(DataflowEdge {
            producer: up_id,
            consumer: mul_id,
            value: up_out.to_string(),
        });

        // Node 4: Down MatMul(gated_up → down_out)
        let down_id = emit_matmul(
            &mut nodes,
            &mut values,
            &mut node_id,
            gated_up,
            &CanonicalRole::Down(0).to_string(),
            down_out,
        );

        // Edge: gated_up from Mul → Down
        edges.push(DataflowEdge {
            producer: mul_id,
            consumer: down_id,
            value: gated_up.to_string(),
        });
    }

    // ------------------------------------------------------------------
    // Independent projection & norm roles (MLP roles already handled above)
    // ------------------------------------------------------------------
    for role in roles {
        // Skip roles already handled in the MLP block.
        if matches!(
            role,
            CanonicalRole::Gate(_) | CanonicalRole::Up(_) | CanonicalRole::Down(_)
        ) && has_mlp_triplet(roles)
        {
            continue;
        }

        if role_to_op_kind(role) == Some(DataflowOpKind::MatMul) {
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
        } else if role_to_op_kind(role) == Some(DataflowOpKind::RmsNorm) {
            let tensor_name = role.to_string();
            let input_buf = format!("{}_input", tensor_name);
            let output_buf = format!("{}_output", tensor_name);
            let id = node_id;
            values
                .entry(input_buf.clone())
                .or_insert_with(|| val(&input_buf, DType::F32));
            values
                .entry(output_buf.clone())
                .or_insert_with(|| val(&output_buf, DType::F32));
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
    }

    DataflowGraph {
        nodes,
        edges,
        values,
        layer_id: format!("layer_{}", layer_idx),
    }
}

// ---------------------------------------------------------------------------
// Fusion analysis
// ---------------------------------------------------------------------------

/// A single fusion group derived from analysis.
struct FusionGroupSpec {
    root_op_kind: String,
    fused_op_kinds: Vec<String>,
}

/// Analyse the dataflow graph to identify fusion opportunities.
///
/// **Root ops** are `MatMul` nodes (weight projections with reduction).
/// **Fused ops** are elementwise operations (SiLU, Gelu, Mul, Add) that
/// directly consume a MatMul output — they are inlined into the root kernel.
fn analyse_fusion_groups(graph: &DataflowGraph) -> Vec<FusionGroupSpec> {
    let topo = graph.topological_sort();

    // Identify root MatMul nodes.
    let mut root_indices: Vec<usize> = Vec::new();
    for &idx in &topo {
        if idx < graph.nodes.len() && matches!(graph.nodes[idx].op, DataflowOp::MatMul { .. }) {
            root_indices.push(idx);
        }
    }

    // Build a consumer map: which buffer names are consumed by which nodes.
    let mut buffer_consumers: HashMap<&str, Vec<usize>> = HashMap::new();
    for edge in &graph.edges {
        buffer_consumers
            .entry(edge.value.as_str())
            .or_default()
            .push(edge.consumer);
    }

    let mut groups = Vec::with_capacity(root_indices.len());

    for &root_idx in &root_indices {
        if root_idx >= graph.nodes.len() {
            continue;
        }
        let root_node = &graph.nodes[root_idx];
        let root_kind = op_kind_label(&root_node.op);
        let mut fused = Vec::new();

        for output in &root_node.outputs {
            if let Some(consumers) = buffer_consumers.get(output.as_str()) {
                for &consumer_idx in consumers {
                    if consumer_idx >= graph.nodes.len() {
                        continue;
                    }
                    let consumer = &graph.nodes[consumer_idx];
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

        groups.push(FusionGroupSpec {
            root_op_kind: root_kind,
            fused_op_kinds: fused,
        });
    }

    groups
}

/// Produce a concise label for an op, used as `root_op_kind` / fused kind.
fn op_kind_label(op: &DataflowOp) -> String {
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
        DataflowOp::AneMatMul { .. } => "AneMatMul",
        DataflowOp::AneConv1x1 { .. } => "AneConv1x1",
        DataflowOp::AneLoadWeight { .. } => "AneLoadWeight",
        DataflowOp::AneStoreOutput { .. } => "AneStoreOutput",
    }
    .to_string()
}
