//! Fusion hardening pass — adversarial and edge-case test suite.
//!
//! Each test targets one specific claim about the fusion scheduling pipeline:
//! codec selection, mixed-precision planning, backend rejection, policy
//! promotion, training target resolution, and kernel validation.
//!
//! The suite is designed to be deterministic and backend-independent.

use super::backend_capability::{
    BackendCapabilityRegistry, BackendLoweringTarget, BackendRole, default_registry,
};
use super::fusion::{
    DataflowGraph, DataflowNode, DataflowOp, FusedGroup, FusionSemanticError, MatMulContract,
};
use super::fusion_scheduler::{
    FusionScheduler, FusionPolicy, FusionSelectionPolicy, FusionSchedule, FusionError,
};
use super::{CodecFamily, ExecutionMode, ScheduledKernelOp, KernelOpKind};
use crate::execution_plan::precision_plan::{
    PrecisionOverride, PrecisionOverrideReason, PrecisionPlan, PrecisionScope,
    PrecisionSelectionBasis, PrecisionSelector,
};
use crate::execution_profile::PhysicalTileLayout;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a fused group with one LoadWeight op for the given codec.
fn single_codec_group(codec: CodecFamily) -> FusedGroup {
    FusedGroup {
        id: "test".into(),
        body: vec![DataflowNode {
            id: 0,
            op: DataflowOp::LoadWeight {
                tensor: "w".into(),
                codec,
                layout: PhysicalTileLayout::default(),
            },
            inputs: vec![],
            outputs: vec!["loaded".into()],
        }],
        inputs: vec!["loaded".into()],
        outputs: vec!["loaded".into()],
        internal_values: vec![],
        codec_family: codec,
        precision_plan: None,
    }
}

/// Build a fused group with a LoadWeight->MatMul->SiLU chain using one codec.
#[allow(dead_code)]
fn load_matmul_silu_group(codec: CodecFamily) -> FusedGroup {
    let nodes = vec![
        DataflowNode {
            id: 0,
            op: DataflowOp::LoadWeight {
                tensor: "w".into(),
                codec,
                layout: PhysicalTileLayout::default(),
            },
            inputs: vec![],
            outputs: vec!["w_loaded".into()],
        },
        DataflowNode {
            id: 1,
            op: DataflowOp::MatMul {
                lhs: "input".into(),
                rhs: "w_loaded".into(),
                output: "mm_out".into(),
                contract: MatMulContract {
                    m: 1,
                    n: 8192,
                    k: 2048,
                    lhs_transposed: false,
                    rhs_transposed: true,
                },
            },
            inputs: vec!["input".into(), "w_loaded".into()],
            outputs: vec!["mm_out".into()],
        },
        DataflowNode {
            id: 2,
            op: DataflowOp::SiLU {
                input: "mm_out".into(),
                output: "silu_out".into(),
            },
            inputs: vec!["mm_out".into()],
            outputs: vec!["silu_out".into()],
        },
    ];
    FusedGroup {
        id: "lms".into(),
        body: nodes,
        inputs: vec!["input".into()],
        outputs: vec!["silu_out".into()],
        internal_values: vec!["w_loaded".into(), "mm_out".into()],
        codec_family: codec,
        precision_plan: None,
    }
}

/// Build a mixed-codec group with two LoadWeight ops (Nf4 + Int8).
fn mixed_codec_group() -> FusedGroup {
    let nodes = vec![
        DataflowNode {
            id: 0,
            op: DataflowOp::LoadWeight {
                tensor: "w_nf4".into(),
                codec: CodecFamily::Nf4,
                layout: PhysicalTileLayout::default(),
            },
            inputs: vec![],
            outputs: vec!["w_nf4".into()],
        },
        DataflowNode {
            id: 1,
            op: DataflowOp::LoadWeight {
                tensor: "w_int8".into(),
                codec: CodecFamily::Int8,
                layout: PhysicalTileLayout::default(),
            },
            inputs: vec![],
            outputs: vec!["w_int8".into()],
        },
    ];
    FusedGroup {
        id: "mixed".into(),
        body: nodes,
        inputs: vec![],
        outputs: vec!["w_nf4".into(), "w_int8".into()],
        internal_values: vec![],
        codec_family: CodecFamily::Mixed,
        precision_plan: None,
    }
}

/// Schedule helper using the FusionScheduler with given mode.
fn schedule_with(
    reg: BackendCapabilityRegistry,
    graph: &DataflowGraph,
    mode: ExecutionMode,
) -> Result<FusionSchedule, FusionError> {
    let scheduler = FusionScheduler::new(reg);
    let policy = FusionPolicy {
        max_group_size: 8,
        allow_materialization: true,
        allow_research_fusions: false,
        execution_mode: mode,
    };
    let sel_policy = FusionSelectionPolicy::default();
    scheduler.schedule(graph, &policy, &sel_policy, BackendRole::ProductionHotPath)
}

/// Build an empty DataflowGraph for testing schedule error paths.
#[allow(dead_code)]
fn empty_graph() -> DataflowGraph {
    DataflowGraph {
        nodes: vec![],
        edges: vec![],
        values: HashMap::new(),
        layer_id: "hardening_test".into(),
    }
}

/// Build a minimal single-node DataflowGraph for testing.
fn single_node_graph(codec: CodecFamily) -> DataflowGraph {
    let node = DataflowNode {
        id: 0,
        op: DataflowOp::LoadWeight {
            tensor: "w".into(),
            codec,
            layout: PhysicalTileLayout::default(),
        },
        inputs: vec![],
        outputs: vec!["w".into()],
    };
    DataflowGraph {
        nodes: vec![node],
        edges: vec![],
        values: HashMap::new(),
        layer_id: "hardening_test".into(),
    }
}

// ── Test 1: derive_semantics_from_loadweight ──────────────────────────

#[test]
fn derive_semantics_from_loadweight() {
    let group = load_matmul_silu_group(CodecFamily::Nf4);
    let semantics = group.derive_semantics().expect("should derive semantics");
    assert_eq!(semantics.codec_family, Some(CodecFamily::Nf4));
    assert!(semantics.has_weight_load);
    assert!(!semantics.mixed_codec);
}

// ── Test 2: mixed_codec_without_precision_plan_rejected ─────────────────

#[test]
fn mixed_codec_without_precision_plan_rejected() {
    let group = mixed_codec_group();
    let semantics = group.derive_semantics().expect("should derive semantics");
    assert!(semantics.mixed_codec);
    assert!(semantics.precision_plan.is_none());

    let reg = default_registry();
    for target in &[
        BackendLoweringTarget::MetalFusedGpu,
        BackendLoweringTarget::AnePlanarEngine,
        BackendLoweringTarget::CoreMlHighLevel,
        BackendLoweringTarget::AccelerateRayonCpu,
    ] {
        let result = reg.evaluate(*target, &group, BackendRole::ProductionHotPath);
        assert!(
            !result.supported,
            "{:?} should reject mixed group without PrecisionPlan",
            target
        );
    }
}

// ── Test 3: nf4_group_selects_metal_only ─────────────────────────────

#[test]
fn nf4_group_selects_metal_only() {
    let group = load_matmul_silu_group(CodecFamily::Nf4);
    let reg = default_registry();

    let ane =
        reg.evaluate(BackendLoweringTarget::AnePlanarEngine, &group, BackendRole::ProductionHotPath);
    assert!(!ane.supported, "ANE must reject NF4");

    let cpu =
        reg.evaluate(BackendLoweringTarget::AccelerateRayonCpu, &group, BackendRole::ProductionHotPath);
    assert!(!cpu.supported, "CPU must reject NF4");

    let metal =
        reg.evaluate(BackendLoweringTarget::MetalFusedGpu, &group, BackendRole::ProductionHotPath);
    assert!(metal.supported, "Metal must accept NF4 group");
}

// ── Test 4: compile_mode_fails_without_backend ─────────────────────────

#[test]
fn compile_mode_fails_without_backend() {
    let reg = BackendCapabilityRegistry::new();
    let graph = single_node_graph(CodecFamily::Nf4);
    let result = schedule_with(reg, &graph, ExecutionMode::Compile);
    assert!(result.is_err(), "Compile mode with empty registry should fail");
    match result {
        Err(FusionError::NoViableBackend { .. }) => {} // expected
        other => panic!("Expected NoViableBackend error, got: {:?}", other),
    }
}

// ── Test 5: explore_mode_records_without_backend ──────────────────────

#[test]
fn explore_mode_records_without_backend() {
    let reg = BackendCapabilityRegistry::new();
    let graph = single_node_graph(CodecFamily::Nf4);
    let result = schedule_with(reg, &graph, ExecutionMode::Explore);
    assert!(
        result.is_ok(),
        "Explore mode with empty registry should succeed (records only, no error)"
    );
}
