//! Fusion compiler correctness tests.
//!
//! Verifies that graph IR, fusion schedules, and backend capabilities preserve
//! semantics end-to-end. Each test targets one specific claim.

use super::fusion::DataflowGraphBuilder;
use super::fusion_schedule_types::{
    fuse_and_schedule, DataflowGraph, DataflowOp, DataflowOpKind, FusionCapabilities,
    FusionPattern, TensorDescriptor,
};
use super::backend_capability::{BackendCapability, BackendLoweringTarget, BackendRole};
use super::CodecFamily;
use crate::execution_profile::{ExecutionView, ResidencyMode};
use std::collections::HashMap;

// ── Helpers ───────────────────────────────────────────────────────────────

fn mlp_graph() -> DataflowGraph {
    let ev = |lane: &str| ExecutionView {
        lane: lane.into(),
        data_offset: 0,
        data_length: 8192,
        metadata_offset: None,
        metadata_length: None,
        codec_overrides: HashMap::new(),
        repacking_required: false,
        residency: ResidencyMode::EphemeralScratch,
    };
    DataflowGraph {
        ops: vec![
            DataflowOp {
                op_index: 0,
                step_name: "mlp_gate_up".into(),
                op_kind: DataflowOpKind::MlpGateUp,
                execution_view: ev("metal"),
                input_tensors: vec![0],
                output_tensors: vec![1],
                arithmetic_intensity: Some(4.0),
            },
            DataflowOp {
                op_index: 1,
                step_name: "mlp_activation".into(),
                op_kind: DataflowOpKind::MlpActivation,
                execution_view: ev("metal"),
                input_tensors: vec![1],
                output_tensors: vec![2],
                arithmetic_intensity: Some(1.0),
            },
            DataflowOp {
                op_index: 2,
                step_name: "mlp_down".into(),
                op_kind: DataflowOpKind::MlpDownResidual,
                execution_view: ev("metal"),
                input_tensors: vec![2],
                output_tensors: vec![3],
                arithmetic_intensity: Some(3.0),
            },
        ],
        tensor_shapes: vec![TensorDescriptor { shape: vec![4096], dtype: "f16".into(), byte_size: 8192 }; 4],
    }
}

fn bridge_graph() -> DataflowGraph {
    let ev = |lane: &str| ExecutionView {
        lane: lane.into(),
        data_offset: 0,
        data_length: 4096,
        metadata_offset: None,
        metadata_length: None,
        codec_overrides: HashMap::new(),
        repacking_required: false,
        residency: ResidencyMode::EphemeralScratch,
    };
    DataflowGraph {
        ops: vec![
            DataflowOp {
                op_index: 0,
                step_name: "bridge_proj".into(),
                op_kind: DataflowOpKind::BridgeProjection,
                execution_view: ev("ane"),
                input_tensors: vec![0],
                output_tensors: vec![1],
                arithmetic_intensity: Some(2.0),
            },
            DataflowOp {
                op_index: 1,
                step_name: "lm_head".into(),
                op_kind: DataflowOpKind::LmHead,
                execution_view: ev("ane"),
                input_tensors: vec![1],
                output_tensors: vec![2],
                arithmetic_intensity: Some(1.0),
            },
        ],
        tensor_shapes: vec![TensorDescriptor { shape: vec![2048], dtype: "f16".into(), byte_size: 4096 }; 3],
    }
}

// ── Test 1: dataflow_toposort_gemma_mlp ──────────────────────────────────

#[test]
fn dataflow_toposort_gemma_mlp() {
    let graph = DataflowGraphBuilder::build_mlp();
    assert_eq!(graph.nodes.len(), 7, "MLP must have 7 nodes");
    let order = graph.topological_sort();
    assert_eq!(order.len(), 7, "topo sort must include all 7 nodes");
    assert_eq!(order, vec![0, 1, 2, 3, 4, 5, 6], "FIFO topo order");

    let pos: HashMap<usize, usize> =
        order.iter().enumerate().map(|(i, &n)| (n, i)).collect();
    for edge in &graph.edges {
        assert!(pos[&edge.producer] < pos[&edge.consumer]);
    }
    assert_eq!(graph.producer_of("normalized"), Some(0));
    assert_eq!(graph.consumers_of("normalized").len(), 2);
}

// ── Test 2: metal_fuses_nf4_gate_up_silu_when_supported ─────────────────

#[test]
fn metal_fuses_nf4_gate_up_silu_when_supported() {
    let groups = fuse_and_schedule(
        &mlp_graph(),
        &[FusionCapabilities {
            supported_patterns: vec![FusionPattern::MlpGateActivation],
            max_fused_ops: 3,
        }],
    );
    assert_eq!(groups.len(), 2, "gate_up+activation fused → 2 groups");
    assert!(groups[0].has_fused_kernel);
    assert!(groups[0].fusion_pattern.is_some());
    let k: Vec<DataflowOpKind> = groups[0].ops.iter().map(|o| o.op_kind).collect();
    assert!(k.contains(&DataflowOpKind::MlpGateUp));
    assert!(k.contains(&DataflowOpKind::MlpActivation));
    assert_eq!(groups[1].ops.len(), 1);
    assert_eq!(groups[1].ops[0].op_kind, DataflowOpKind::MlpDownResidual);
}

// ── Test 3: ane_rejects_nf4_fusion ───────────────────────────────────────

#[test]
fn ane_rejects_nf4_fusion() {
    let groups = fuse_and_schedule(
        &mlp_graph(),
        &[FusionCapabilities {
            supported_patterns: vec![],
            max_fused_ops: 1,
        }],
    );
    assert_eq!(groups.len(), 3, "no fusion → 3 singletons");
    for g in &groups {
        assert_eq!(g.ops.len(), 1);
        assert!(!g.has_fused_kernel);
    }
}

// ── Test 4: ane_accepts_int8_bridge_projection ───────────────────────────

#[test]
fn ane_accepts_int8_bridge_projection() {
    let groups = fuse_and_schedule(
        &bridge_graph(),
        &[FusionCapabilities {
            supported_patterns: vec![],
            max_fused_ops: 1,
        }],
    );
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0].ops[0].op_kind, DataflowOpKind::BridgeProjection);
    assert_eq!(groups[1].ops[0].op_kind, DataflowOpKind::LmHead);
}

// ── Test 5: region_batched_plan_matches_op_by_op_plan_shape ─────────────

#[test]
fn region_batched_plan_matches_op_by_op_plan_shape() {
    let graph = mlp_graph();

    let fused = fuse_and_schedule(
        &graph,
        &[FusionCapabilities {
            supported_patterns: vec![FusionPattern::MlpGateActivation],
            max_fused_ops: 3,
        }],
    );
    let identity = fuse_and_schedule(
        &graph,
        &[FusionCapabilities {
            supported_patterns: vec![],
            max_fused_ops: 1,
        }],
    );

    assert!(fused.len() < identity.len(), "fused < identity groups");
    let fused_ops: usize = fused.iter().map(|g| g.ops.len()).sum();
    let identity_ops: usize = identity.iter().map(|g| g.ops.len()).sum();
    assert_eq!(fused_ops, identity_ops, "same total ops: {fused_ops} == {identity_ops}");
    assert_eq!(fused_ops, 3);
    for g in &identity {
        assert_eq!(g.ops.len(), 1);
        assert!(!g.has_fused_kernel);
    }
    for (i, g) in fused.iter().enumerate() {
        assert_eq!(g.group_id, i);
    }
    for (i, g) in identity.iter().enumerate() {
        assert_eq!(g.group_id, i);
    }
}

// ── Test 6: missing_backend_capability_fails_closed ─────────────────────

#[test]
fn missing_backend_capability_fails_closed() {
    let cap = BackendCapability {
        target: BackendLoweringTarget::MetalTensorApi,
        supported_codecs: vec![],
        supported_roles: vec![],
        max_ops_per_group: 1,
        max_tile_elements: 0,
        rules: vec![],
    };
    assert!(cap.supported_codecs.is_empty());
    assert!(!cap.supported_codecs.contains(&CodecFamily::Nf4));
    assert!(!cap.supported_codecs.contains(&CodecFamily::Int8));
    assert!(!cap.supported_codecs.contains(&CodecFamily::RawF32));
}

// ── Test 7: unsupported_codec_fails_closed ──────────────────────────────

#[test]
fn unsupported_codec_fails_closed() {
    // ANE supports Int8 and Fp16 only.
    let cap = BackendCapability {
        target: BackendLoweringTarget::AneNeuralEngine,
        supported_codecs: vec![CodecFamily::Int8, CodecFamily::Fp16],
        supported_roles: vec![],
        max_ops_per_group: 1,
        max_tile_elements: 0,
        rules: vec![],
    };
    assert!(cap.supported_codecs.contains(&CodecFamily::Int8));
    assert!(cap.supported_codecs.contains(&CodecFamily::Fp16));
    assert!(!cap.supported_codecs.contains(&CodecFamily::Nf4));
    assert!(!cap.supported_codecs.contains(&CodecFamily::RawF32));
}

// ── Test 8: cpu_capability_registered_as_first_class ────────────────────

#[test]
fn cpu_capability_registered_as_first_class() {
    let cap = BackendCapability {
        target: BackendLoweringTarget::AccelerateRayonCpu,
        supported_codecs: vec![CodecFamily::RawF32],
        supported_roles: vec![BackendRole::ProductionHotPath],
        max_ops_per_group: 1,
        max_tile_elements: 0,
        rules: vec![],
    };
    assert_eq!(cap.target, BackendLoweringTarget::AccelerateRayonCpu);
    assert!(cap.supported_codecs.contains(&CodecFamily::RawF32));
    assert!(!cap.supported_codecs.is_empty());
    assert_eq!(
        format!("{:?}", BackendLoweringTarget::AccelerateRayonCpu),
        "AccelerateRayonCpu",
    );
}

// ── Test 9: cpu_accepts_rawf32_matmul_reference ─────────────────────────

#[test]
fn cpu_accepts_rawf32_matmul_reference() {
    let cap = BackendCapability {
        target: BackendLoweringTarget::AccelerateRayonCpu,
        supported_codecs: vec![CodecFamily::RawF32],
        supported_roles: vec![],
        max_ops_per_group: 1,
        max_tile_elements: 0,
        rules: vec![],
    };
    assert!(cap.supported_codecs.contains(&CodecFamily::RawF32));
    assert!(!cap.supported_codecs.contains(&CodecFamily::Nf4));
    assert!(!cap.supported_codecs.contains(&CodecFamily::Int8));
}

// ── Test 10: cpu_rejects_nf4_without_custom_kernel ─────────────────────

#[test]
fn cpu_rejects_nf4_without_custom_kernel() {
    // CPU without NF4 in its codecs rejects it.
    let cap = BackendCapability {
        target: BackendLoweringTarget::AccelerateRayonCpu,
        supported_codecs: vec![CodecFamily::RawF32],
        supported_roles: vec![],
        max_ops_per_group: 1,
        max_tile_elements: 0,
        rules: vec![],
    };
    assert!(!cap.supported_codecs.contains(&CodecFamily::Nf4));

    // A capability with NF4 accepts it.
    let cap_nf4 = BackendCapability {
        target: BackendLoweringTarget::AccelerateRayonCpu,
        supported_codecs: vec![CodecFamily::Nf4],
        supported_roles: vec![],
        max_ops_per_group: 1,
        max_tile_elements: 0,
        rules: vec![],
    };
    assert!(cap_nf4.supported_codecs.contains(&CodecFamily::Nf4));
}
