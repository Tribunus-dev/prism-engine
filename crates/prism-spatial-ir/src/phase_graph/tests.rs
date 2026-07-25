//! Test suite for the `phase_graph` subsystem. Extracted out of
//! `phase_graph/mod.rs` to keep the directory index under the 200-LOC
//! `mod.rs` rule; the tests are unchanged from the original
//! `tinygrad_core.rs` test module.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use crate::phase_graph::capture::{CaptureExecutor, LoweredKernel, TinyJitCache};
use crate::phase_graph::graph::{GraphError, TinyGraph};
use crate::phase_graph::kernel_op::LoweringTarget;
use crate::phase_graph::plan::{ExecutionReceipt, ReplayPlan};
use crate::phase_graph::render::hex_digest;
use crate::phase_graph::uop::{UOp, UOpId, UOpKind};

#[test]
fn schedules_and_fuses_elementwise_chain() {
    let mut graph = TinyGraph::default();
    let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
    let one = graph.add(UOpKind::Const { value: 1.0 }, vec![], vec![1]);
    let add = graph.add(UOpKind::Add, vec![x, one], vec![4]);
    let relu = graph.add(UOpKind::Relu, vec![add], vec![4]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![4]);
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert_eq!(capture.kernels.len(), 1);
    assert_eq!(capture.kernels[0].group.ops.len(), 2);
    assert!(!capture.kernels[0].source_digest.is_empty());
    assert!(capture.kernels[0].source.contains("prism_kernel"));
}

#[test]
fn optimizer_eliminates_neutral_elementwise_uops_before_lowering() {
    let mut graph = TinyGraph::default();
    let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
    let zero = graph.add(UOpKind::Const { value: 0.0 }, vec![], vec![1]);
    let one = graph.add(UOpKind::Const { value: 1.0 }, vec![], vec![1]);
    let add = graph.add(UOpKind::Add, vec![x, zero], vec![4]);
    let mul = graph.add(UOpKind::Mul, vec![add, one], vec![4]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![mul], vec![4]);

    let optimized = graph.optimize().unwrap();
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    let output = graph
        .execute_f32(&BTreeMap::from([(
            String::from("x"),
            vec![1.0, -2.0, 3.0, 4.0],
        )]))
        .unwrap();

    assert!(optimized
        .ops
        .iter()
        .any(|op| { matches!(op.kind, UOpKind::Output { .. }) && op.src == vec![x] }));
    assert!(capture.kernels.is_empty());
    assert_eq!(output["y"], vec![1.0, -2.0, 3.0, 4.0]);
}

#[test]
fn mean_axis_reuses_minimal_sum_and_scalar_division_uops() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
    let mean = graph.add_mean_axis(input, 1).unwrap();
    graph.add(
        UOpKind::Output {
            name: "mean".into(),
        },
        vec![mean],
        vec![2],
    );
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert_eq!(capture.kernels.len(), 2);
    let outputs = graph
        .execute_f32(&BTreeMap::from([(
            "x".into(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )]))
        .unwrap();
    assert_eq!(outputs["mean"], vec![2.0, 5.0]);
}

#[test]
fn whole_tensor_mean_reuses_reduce_sum_and_scalar_division() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 2]);
    let mean = graph.add_mean(input).unwrap();
    graph.add(
        UOpKind::Output {
            name: "mean".into(),
        },
        vec![mean],
        vec![1],
    );
    let output = graph
        .execute_f32(&BTreeMap::from([("x".into(), vec![1.0, 2.0, 3.0, 6.0])]))
        .unwrap();
    assert_eq!(output["mean"], vec![3.0]);
}

#[test]
fn validation_rejects_non_scalar_whole_tensor_reduction_output() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
    let sum = graph.add(UOpKind::ReduceSum, vec![input], vec![2]);
    graph.add(UOpKind::Output { name: "sum".into() }, vec![sum], vec![2]);
    assert!(matches!(
        graph.validate(),
        Err(GraphError::ShapeMismatch(_))
    ));
}

#[test]
fn validation_rejects_invalid_specialized_parameters() {
    let mut cast = TinyGraph::default();
    let input = cast.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
    cast.add(
        UOpKind::Cast {
            from: "fp8".into(),
            to: "f32".into(),
        },
        vec![input],
        vec![2],
    );
    assert!(matches!(cast.validate(), Err(GraphError::ShapeMismatch(_))));

    let mut attention = TinyGraph::default();
    let q = attention.add(UOpKind::Input { name: "q".into() }, vec![], vec![2, 2]);
    let k = attention.add(UOpKind::Input { name: "k".into() }, vec![], vec![2, 2]);
    let v = attention.add(UOpKind::Input { name: "v".into() }, vec![], vec![2, 2]);
    attention.add(
        UOpKind::Attention {
            seq: 2,
            head: 2,
            scale: f32::NAN,
        },
        vec![q, k, v],
        vec![2, 2],
    );
    assert!(matches!(
        attention.validate(),
        Err(GraphError::ShapeMismatch(_))
    ));
}

#[test]
fn cast_reference_and_rendered_integer_bounds_agree() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
    let cast = graph.add(
        UOpKind::Cast {
            from: "f32".into(),
            to: "i32".into(),
        },
        vec![input],
        vec![4],
    );
    graph.add(UOpKind::Output { name: "y".into() }, vec![cast], vec![4]);
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert!(capture.kernels[0].source.contains("2147483647.0f"));
    let output = graph
        .execute_f32(&BTreeMap::from([(
            "x".into(),
            vec![f32::MAX, f32::MIN, 2.9, -2.9],
        )]))
        .unwrap();
    assert_eq!(output["y"], vec![2147483647.0, -2147483648.0, 2.0, -2.0]);
}

#[test]
fn strategy_lowering_emits_distinct_executable_capture_layouts() {
    let mut graph = TinyGraph::default();
    let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
    let one = graph.add(UOpKind::Const { value: 1.0 }, vec![], vec![1]);
    let add = graph.add(UOpKind::Add, vec![x, one], vec![4]);
    let relu = graph.add(UOpKind::Relu, vec![add], vec![4]);
    let exp = graph.add(UOpKind::Exp, vec![relu], vec![4]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![exp], vec![4]);

    let standard = graph
        .lower_with_fusion_strategy(
            LoweringTarget::Portable,
            &crate::fused_ops::FusionStrategy::StandardFused,
        )
        .unwrap();
    let per_operation = graph
        .lower_with_fusion_strategy(
            LoweringTarget::Portable,
            &crate::fused_ops::FusionStrategy::PerOperation,
        )
        .unwrap();
    let interleaved = graph
        .lower_with_fusion_strategy(
            LoweringTarget::Portable,
            &crate::fused_ops::FusionStrategy::InterleavedFused {
                stages: vec![vec![crate::fused_ops::FusableOp::FpGemv]; 2],
            },
        )
        .unwrap();

    assert_eq!(standard.kernels.len(), 1);
    assert_eq!(per_operation.kernels.len(), 3);
    assert_eq!(interleaved.kernels.len(), 2);
    assert_eq!(per_operation.replay.command_ids, vec![0, 1, 2]);
    assert!(per_operation.validate().is_ok());
    assert!(interleaved.validate().is_ok());
}

#[test]
fn empty_interleaved_stage_request_still_splits_multi_op_kernel() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
    let relu = graph.add(UOpKind::Relu, vec![input], vec![4]);
    let exp = graph.add(UOpKind::Exp, vec![relu], vec![4]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![exp], vec![4]);
    let capture = graph
        .lower_with_fusion_strategy(
            LoweringTarget::Portable,
            &crate::fused_ops::FusionStrategy::InterleavedFused { stages: vec![] },
        )
        .unwrap();
    assert_eq!(capture.kernels.len(), 2);
    assert!(capture.validate().is_ok());
}

#[test]
fn fusion_does_not_consume_forked_intermediate_values() {
    let mut graph = TinyGraph::default();
    let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
    let one = graph.add(UOpKind::Const { value: 1.0 }, vec![], vec![1]);
    let base = graph.add(UOpKind::Add, vec![x, one], vec![2]);
    let relu = graph.add(UOpKind::Relu, vec![base], vec![2]);
    let scaled = graph.add(UOpKind::Mul, vec![base, one], vec![2]);
    graph.add(UOpKind::Output { name: "a".into() }, vec![relu], vec![2]);
    graph.add(UOpKind::Output { name: "b".into() }, vec![scaled], vec![2]);
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    // `base * 1` is rewritten to `base`; the two consumers still remain
    // separate because the shared intermediate is forked.
    assert_eq!(capture.kernels.len(), 2);
    assert_eq!(capture.kernels[0].group.ops.len(), 1);
}

struct RecordingExecutor(Vec<String>);
impl CaptureExecutor for RecordingExecutor {
    fn dispatch(&mut self, command_id: u32, _kernel: &LoweredKernel) -> Result<(), String> {
        self.0.push(format!("dispatch:{command_id}"));
        Ok(())
    }
    fn synchronize(&mut self, command_id: u32) -> Result<(), String> {
        self.0.push(format!("sync:{command_id}"));
        Ok(())
    }
}

#[test]
fn capture_replays_commands_and_returns_receipt() {
    let mut graph = TinyGraph::default();
    let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
    let one = graph.add(UOpKind::Const { value: 1.0 }, vec![], vec![1]);
    let add = graph.add(UOpKind::Add, vec![x, one], vec![4]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![add], vec![4]);
    let capture = graph.lower(LoweringTarget::Cpu).unwrap();
    let mut executor = RecordingExecutor(Vec::new());
    let receipt = capture.replay(&mut executor).unwrap();
    assert!(receipt.replayed);
    assert_eq!(receipt.capture_digest, capture.digest());
    capture.validate_receipt(&receipt).unwrap();
    let mut tampered = receipt.clone();
    tampered.command_ids.push(99);
    assert!(capture.validate_receipt(&tampered).is_err());
    assert_eq!(executor.0, vec!["dispatch:0", "sync:0"]);
}

#[test]
fn tiny_jit_cache_captures_once_and_reuses_by_graph_and_target() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
    let relu = graph.add(UOpKind::Relu, vec![input], vec![2]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![2]);
    let mut cache = TinyJitCache::default();
    let (first_key, first_capture) = cache.capture(&graph, LoweringTarget::Portable).unwrap();
    let (second_key, second_capture) = cache.capture(&graph, LoweringTarget::Portable).unwrap();
    assert!(first_capture);
    assert!(!second_capture);
    assert_eq!(first_key, second_key);
    assert_eq!(cache.len(), 1);
    assert!(cache.get(&first_key).unwrap().replay.command_ids == vec![0]);
    let mut executor = RecordingExecutor(Vec::new());
    let receipt = cache.replay(&first_key, &mut executor).unwrap();
    assert!(receipt.replayed);
    assert_eq!(executor.0, vec!["dispatch:0", "sync:0"]);
}

#[test]
fn tiny_jit_cache_supports_capture_invalidation() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
    let relu = graph.add(UOpKind::Relu, vec![input], vec![2]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![2]);
    let mut cache = TinyJitCache::default();
    let (key, _) = cache.capture(&graph, LoweringTarget::Portable).unwrap();
    assert!(cache.invalidate(&key));
    assert!(!cache.invalidate(&key));
    assert!(cache.is_empty());
    cache.capture(&graph, LoweringTarget::Portable).unwrap();
    cache.clear();
    assert!(cache.is_empty());
}

#[test]
fn tiny_jit_cache_keys_strategy_specific_capture_layouts() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
    let relu = graph.add(UOpKind::Relu, vec![input], vec![4]);
    let exp = graph.add(UOpKind::Exp, vec![relu], vec![4]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![exp], vec![4]);
    let mut cache = TinyJitCache::default();
    let (standard_key, _) = cache.capture(&graph, LoweringTarget::Portable).unwrap();
    let (per_op_key, first) = cache
        .capture_with_strategy(
            &graph,
            LoweringTarget::Portable,
            &crate::fused_ops::FusionStrategy::PerOperation,
        )
        .unwrap();
    let (_, second) = cache
        .capture_with_strategy(
            &graph,
            LoweringTarget::Portable,
            &crate::fused_ops::FusionStrategy::PerOperation,
        )
        .unwrap();
    assert_ne!(standard_key, per_op_key);
    assert!(first);
    assert!(!second);
    assert_eq!(cache.get(&standard_key).unwrap().kernels.len(), 1);
    assert_eq!(cache.get(&per_op_key).unwrap().kernels.len(), 2);
}

#[test]
fn tiny_jit_cache_materializes_requested_strategy_set_once() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
    let relu = graph.add(UOpKind::Relu, vec![input], vec![4]);
    let exp = graph.add(UOpKind::Exp, vec![relu], vec![4]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![exp], vec![4]);
    let strategies = vec![
        crate::fused_ops::FusionStrategy::StandardFused,
        crate::fused_ops::FusionStrategy::PerOperation,
        crate::fused_ops::FusionStrategy::PersistentMegakernel {
            search_generation: 7,
        },
    ];
    let mut cache = TinyJitCache::default();
    let first = cache
        .capture_strategies(&graph, LoweringTarget::Portable, &strategies)
        .unwrap();
    assert!(first.iter().all(|(_, _, inserted)| *inserted));
    assert_eq!(cache.len(), strategies.len());
    let second = cache
        .capture_strategies(&graph, LoweringTarget::Portable, &strategies)
        .unwrap();
    assert!(second.iter().all(|(_, _, inserted)| !*inserted));
    assert_eq!(
        first.iter().map(|(_, key, _)| key).collect::<Vec<_>>(),
        second.iter().map(|(_, key, _)| key).collect::<Vec<_>>()
    );
    let duplicate = cache.capture_strategies(
        &graph,
        LoweringTarget::Portable,
        &[
            crate::fused_ops::FusionStrategy::StandardFused,
            crate::fused_ops::FusionStrategy::StandardFused,
        ],
    );
    assert!(duplicate.is_err());
}

#[test]
fn tiny_jit_cache_round_trips_through_validated_bytes() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
    let relu = graph.add(UOpKind::Relu, vec![input], vec![2]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![2]);
    let mut cache = TinyJitCache::default();
    let (key, _) = cache.capture(&graph, LoweringTarget::Portable).unwrap();
    let bytes = cache.export_bytes().unwrap();
    let restored = TinyJitCache::import_bytes(&bytes).unwrap();
    assert_eq!(restored.len(), 1);
    assert_eq!(restored.get(&key), cache.get(&key));
}

#[test]
fn tiny_jit_cache_import_rejects_malformed_bytes() {
    let error = TinyJitCache::import_bytes(b"not a cache").unwrap_err();
    assert!(matches!(error, GraphError::Serialization(_)));
}

#[test]
fn tiny_jit_cache_import_rejects_identity_tampering() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
    let output = graph.add(UOpKind::Output { name: "y".into() }, vec![input], vec![1]);
    assert_eq!(output, UOpId(1));
    let mut cache = TinyJitCache::default();
    cache.capture(&graph, LoweringTarget::Portable).unwrap();
    let bytes = cache.export_bytes().unwrap();
    let mut value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let digest = value["captures"]
        .as_object_mut()
        .and_then(|entries| entries.values_mut().next())
        .and_then(|entry| entry.get_mut("identity_digest"))
        .and_then(|value| value.as_str())
        .unwrap()
        .to_owned();
    value["captures"]
        .as_object_mut()
        .unwrap()
        .values_mut()
        .next()
        .unwrap()["identity_digest"] = serde_json::Value::String(format!("{digest}tampered"));
    let tampered = serde_json::to_vec(&value).unwrap();
    assert!(TinyJitCache::import_bytes(&tampered).is_err());
}

#[test]
fn persistent_capture_uses_single_submission_and_final_sync() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
    let relu = graph.add(UOpKind::Relu, vec![input], vec![4]);
    let exp = graph.add(UOpKind::Exp, vec![relu], vec![4]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![exp], vec![4]);
    let capture = graph
        .lower_with_fusion_strategy(
            LoweringTarget::Portable,
            &crate::fused_ops::FusionStrategy::PersistentMegakernel {
                search_generation: 9,
            },
        )
        .unwrap();
    assert!(capture.replay.persistent);
    assert_eq!(capture.kernels.len(), 1);
    assert_eq!(capture.kernels[0].group.ops.len(), 2);
    let mut executor = RecordingExecutor(Vec::new());
    let receipt = capture.replay(&mut executor).unwrap();
    assert!(receipt.persistent);
    assert_eq!(executor.0, vec!["dispatch:0", "sync:0"]);
}

#[test]
fn persistent_metadata_defaults_for_legacy_serialized_records() {
    let replay: ReplayPlan = serde_json::from_value(serde_json::json!({
        "command_ids": [0],
        "synchronization_points": [0]
    }))
    .unwrap();
    assert!(!replay.persistent);
    let receipt: ExecutionReceipt = serde_json::from_value(serde_json::json!({
        "target": "Portable",
        "command_ids": [0],
        "kernel_digests": ["digest"],
        "replayed": true
    }))
    .unwrap();
    assert!(!receipt.persistent);
}

#[test]
fn reference_executor_produces_behavioral_output() {
    let mut graph = TinyGraph::default();
    let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![3]);
    let two = graph.add(UOpKind::Const { value: 2.0 }, vec![], vec![1]);
    let scaled = graph.add(UOpKind::Mul, vec![x, two], vec![3]);
    let positive = graph.add(UOpKind::Relu, vec![scaled], vec![3]);
    graph.add(
        UOpKind::Output { name: "y".into() },
        vec![positive],
        vec![3],
    );
    let mut inputs = BTreeMap::new();
    inputs.insert("x".into(), vec![-1.0, 2.0, 3.0]);
    let outputs = graph.execute_f32(&inputs).unwrap();
    assert_eq!(outputs["y"], vec![0.0, 4.0, 6.0]);
}

#[test]
fn optimizer_folds_constant_binary_ops() {
    let mut graph = TinyGraph::default();
    let left = graph.add(UOpKind::Const { value: 2.0 }, vec![], vec![1]);
    let right = graph.add(UOpKind::Const { value: 3.0 }, vec![], vec![1]);
    let sum = graph.add(UOpKind::Add, vec![left, right], vec![1]);
    let output = graph.add(UOpKind::Output { name: "y".into() }, vec![sum], vec![1]);
    let optimized = graph.optimize().unwrap();
    assert!(
        matches!(optimized.ops[sum.0 as usize].kind, UOpKind::Const { value } if value == 5.0)
    );
    assert_eq!(optimized.ops[output.0 as usize].src, vec![sum]);
}

#[test]
fn optimizer_reaches_fixed_point_for_nested_unary_and_binary_folds() {
    let mut graph = TinyGraph::default();
    let left = graph.add(UOpKind::Const { value: 2.0 }, vec![], vec![1]);
    let right = graph.add(UOpKind::Const { value: 3.0 }, vec![], vec![1]);
    let sum = graph.add(UOpKind::Add, vec![left, right], vec![1]);
    let relu = graph.add(UOpKind::Relu, vec![sum], vec![1]);
    let output = graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![1]);
    let optimized = graph.optimize().unwrap();
    assert!(
        matches!(optimized.ops[sum.0 as usize].kind, UOpKind::Const { value } if value == 5.0)
    );
    assert!(
        matches!(optimized.ops[relu.0 as usize].kind, UOpKind::Const { value } if value == 5.0)
    );
    assert_eq!(optimized.ops[output.0 as usize].src, vec![relu]);
}

#[test]
fn optimizer_folds_constant_where() {
    let mut graph = TinyGraph::default();
    let condition = graph.add(UOpKind::Const { value: 1.0 }, vec![], vec![2]);
    let when_true = graph.add(UOpKind::Const { value: 7.0 }, vec![], vec![2]);
    let when_false = graph.add(UOpKind::Const { value: -3.0 }, vec![], vec![2]);
    let selected = graph.add(
        UOpKind::Where,
        vec![condition, when_true, when_false],
        vec![2],
    );
    let optimized = graph.optimize().unwrap();
    assert!(
        matches!(optimized.ops[selected.0 as usize].kind, UOpKind::Const { value } if value == 7.0)
    );
}

#[test]
fn optimizer_folds_constant_cast_with_runtime_cast_semantics() {
    let mut graph = TinyGraph::default();
    let value = graph.add(UOpKind::Const { value: 300.75 }, vec![], vec![1]);
    let cast = graph.add(
        UOpKind::Cast {
            from: "f32".into(),
            to: "u8".into(),
        },
        vec![value],
        vec![1],
    );
    graph.add(UOpKind::Output { name: "out".into() }, vec![cast], vec![1]);
    let optimized = graph.optimize().unwrap();
    assert!(matches!(
        optimized.ops[cast.0 as usize].kind,
        UOpKind::Const { value } if value == 255.0
    ));
    assert!(optimized.ops[cast.0 as usize].src.is_empty());
}

#[test]
fn optimizer_folds_constant_reductions() {
    let mut graph = TinyGraph::default();
    let value = graph.add(UOpKind::Const { value: 2.0 }, vec![], vec![2, 3]);
    let sum = graph.add(UOpKind::ReduceSum, vec![value], vec![1]);
    let max = graph.add(UOpKind::ReduceMax, vec![value], vec![1]);
    let axis = graph.add(UOpKind::ReduceSumAxis { axis: 1 }, vec![value], vec![2]);
    graph.add(UOpKind::Output { name: "sum".into() }, vec![sum], vec![1]);
    graph.add(UOpKind::Output { name: "max".into() }, vec![max], vec![1]);
    graph.add(
        UOpKind::Output {
            name: "axis".into(),
        },
        vec![axis],
        vec![2],
    );
    let optimized = graph.optimize().unwrap();
    assert!(
        matches!(optimized.ops[sum.0 as usize].kind, UOpKind::Const { value } if value == 12.0)
    );
    assert!(
        matches!(optimized.ops[max.0 as usize].kind, UOpKind::Const { value } if value == 2.0)
    );
    assert!(
        matches!(optimized.ops[axis.0 as usize].kind, UOpKind::Const { value } if value == 6.0)
    );
}

#[test]
fn optimizer_folds_constant_matmul() {
    let mut graph = TinyGraph::default();
    let lhs = graph.add(UOpKind::Const { value: 2.0 }, vec![], vec![2, 3]);
    let rhs = graph.add(UOpKind::Const { value: 4.0 }, vec![], vec![3, 5]);
    let product = graph.add(
        UOpKind::MatMul { m: 2, k: 3, n: 5 },
        vec![lhs, rhs],
        vec![2, 5],
    );
    graph.add(
        UOpKind::Output { name: "out".into() },
        vec![product],
        vec![2, 5],
    );
    let optimized = graph.optimize().unwrap();
    assert!(matches!(
        optimized.ops[product.0 as usize].kind,
        UOpKind::Const { value } if value == 24.0
    ));
}

#[test]
fn optimizer_folds_constant_softmax_axis() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Const { value: 7.0 }, vec![], vec![2, 4]);
    let softmax = graph.add(UOpKind::SoftmaxAxis { axis: 1 }, vec![input], vec![2, 4]);
    graph.add(
        UOpKind::Output { name: "out".into() },
        vec![softmax],
        vec![2, 4],
    );
    let optimized = graph.optimize().unwrap();
    assert!(matches!(
        optimized.ops[softmax.0 as usize].kind,
        UOpKind::Const { value } if (value - 0.25).abs() < f32::EPSILON
    ));
}

#[test]
fn optimizer_folds_constant_attention_to_value_stream() {
    let mut graph = TinyGraph::default();
    let q = graph.add(UOpKind::Const { value: 1.0 }, vec![], vec![2, 3]);
    let k = graph.add(UOpKind::Const { value: 2.0 }, vec![], vec![2, 3]);
    let v = graph.add(UOpKind::Const { value: 9.0 }, vec![], vec![2, 3]);
    let attention = graph.add(
        UOpKind::Attention {
            seq: 2,
            head: 3,
            scale: 1.0,
        },
        vec![q, k, v],
        vec![2, 3],
    );
    graph.add(
        UOpKind::Output { name: "out".into() },
        vec![attention],
        vec![2, 3],
    );
    let optimized = graph.optimize().unwrap();
    assert!(matches!(
        optimized.ops[attention.0 as usize].kind,
        UOpKind::Const { value } if value == 9.0
    ));
}

#[test]
fn optimizer_cse_reuses_duplicate_matmul() {
    let mut graph = TinyGraph::default();
    let lhs = graph.add(UOpKind::Input { name: "lhs".into() }, vec![], vec![2, 3]);
    let rhs = graph.add(UOpKind::Input { name: "rhs".into() }, vec![], vec![3, 4]);
    let first = graph.add(
        UOpKind::MatMul { m: 2, k: 3, n: 4 },
        vec![lhs, rhs],
        vec![2, 4],
    );
    let second = graph.add(
        UOpKind::MatMul { m: 2, k: 3, n: 4 },
        vec![lhs, rhs],
        vec![2, 4],
    );
    graph.add(
        UOpKind::Output { name: "out".into() },
        vec![second],
        vec![2, 4],
    );
    let optimized = graph.optimize().unwrap();
    assert_eq!(optimized.ops[second.0 as usize].src.len(), 0);
    assert_eq!(optimized.ops.last().unwrap().src, vec![first]);
}

#[test]
fn optimizer_cse_reuses_duplicate_axis_reduction() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 4]);
    let first = graph.add(UOpKind::ReduceSumAxis { axis: 1 }, vec![input], vec![2]);
    let second = graph.add(UOpKind::ReduceSumAxis { axis: 1 }, vec![input], vec![2]);
    graph.add(
        UOpKind::Output { name: "out".into() },
        vec![second],
        vec![2],
    );
    let optimized = graph.optimize().unwrap();
    assert!(
        matches!(optimized.ops[second.0 as usize].kind, UOpKind::Const { value } if value == 0.0)
    );
    assert_eq!(optimized.ops.last().unwrap().src, vec![first]);
}

#[test]
fn optimizer_folds_constant_conv2d() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Const { value: 2.0 }, vec![], vec![1, 2, 3, 3]);
    let weight = graph.add(UOpKind::Const { value: 3.0 }, vec![], vec![4, 2, 1, 1]);
    let bias = graph.add(UOpKind::Const { value: 5.0 }, vec![], vec![4]);
    let conv = graph.add(
        UOpKind::Conv2d {
            batch: 1,
            in_channels: 2,
            height: 3,
            width: 3,
            out_channels: 4,
            kernel_h: 1,
            kernel_w: 1,
            stride: 1,
            padding: 0,
        },
        vec![input, weight, bias],
        vec![1, 4, 3, 3],
    );
    graph.add(
        UOpKind::Output { name: "out".into() },
        vec![conv],
        vec![1, 4, 3, 3],
    );
    let optimized = graph.optimize().unwrap();
    assert!(matches!(
        optimized.ops[conv.0 as usize].kind,
        UOpKind::Const { value } if value == 17.0
    ));
}

#[test]
fn optimizer_cse_reuses_duplicate_conv2d() {
    let mut graph = TinyGraph::default();
    let input = graph.add(
        UOpKind::Input { name: "x".into() },
        vec![],
        vec![1, 2, 3, 3],
    );
    let weight = graph.add(
        UOpKind::Input { name: "w".into() },
        vec![],
        vec![4, 2, 1, 1],
    );
    let bias = graph.add(UOpKind::Input { name: "b".into() }, vec![], vec![4]);
    let kind = UOpKind::Conv2d {
        batch: 1,
        in_channels: 2,
        height: 3,
        width: 3,
        out_channels: 4,
        kernel_h: 1,
        kernel_w: 1,
        stride: 1,
        padding: 0,
    };
    let first = graph.add(kind.clone(), vec![input, weight, bias], vec![1, 4, 3, 3]);
    let second = graph.add(kind, vec![input, weight, bias], vec![1, 4, 3, 3]);
    graph.add(
        UOpKind::Output { name: "out".into() },
        vec![second],
        vec![1, 4, 3, 3],
    );
    let optimized = graph.optimize().unwrap();
    assert!(
        matches!(optimized.ops[second.0 as usize].kind, UOpKind::Const { value } if value == 0.0)
    );
    assert_eq!(optimized.ops.last().unwrap().src, vec![first]);
}

#[test]
fn optimizer_folds_constant_gather() {
    let mut graph = TinyGraph::default();
    let weight = graph.add(UOpKind::Const { value: 8.0 }, vec![], vec![16, 4]);
    let indices = graph.add(UOpKind::Const { value: 3.0 }, vec![], vec![2]);
    let gather = graph.add(
        UOpKind::Gather {
            rows: 2,
            vocab: 16,
            features: 4,
        },
        vec![weight, indices],
        vec![2, 4],
    );
    graph.add(
        UOpKind::Output { name: "out".into() },
        vec![gather],
        vec![2, 4],
    );
    let optimized = graph.optimize().unwrap();
    assert!(matches!(
        optimized.ops[gather.0 as usize].kind,
        UOpKind::Const { value } if value == 8.0
    ));
}

#[test]
fn optimizer_folds_constant_normalization() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Const { value: 2.0 }, vec![], vec![1, 4]);
    let weight = graph.add(UOpKind::Const { value: 3.0 }, vec![], vec![4]);
    let rms = graph.add(
        UOpKind::RmsNorm {
            rows: 1,
            features: 4,
            epsilon: 0.0,
        },
        vec![input, weight],
        vec![1, 4],
    );
    let bias = graph.add(UOpKind::Const { value: 5.0 }, vec![], vec![4]);
    let layer = graph.add(
        UOpKind::LayerNorm {
            rows: 1,
            features: 4,
            epsilon: 1e-5,
        },
        vec![input, weight, bias],
        vec![1, 4],
    );
    graph.add(
        UOpKind::Output { name: "rms".into() },
        vec![rms],
        vec![1, 4],
    );
    graph.add(
        UOpKind::Output {
            name: "layer".into(),
        },
        vec![layer],
        vec![1, 4],
    );
    let optimized = graph.optimize().unwrap();
    assert!(
        matches!(optimized.ops[rms.0 as usize].kind, UOpKind::Const { value } if value == 3.0)
    );
    assert!(
        matches!(optimized.ops[layer.0 as usize].kind, UOpKind::Const { value } if value == 5.0)
    );
}

#[test]
fn reference_executor_supports_trailing_dimension_broadcast() {
    let mut graph = TinyGraph::default();
    let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
    let bias = graph.add(
        UOpKind::Input {
            name: "bias".into(),
        },
        vec![],
        vec![3],
    );
    let y = graph.add(UOpKind::Add, vec![x, bias], vec![2, 3]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![y], vec![2, 3]);
    let inputs = BTreeMap::from([
        ("x".into(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ("bias".into(), vec![10.0, 20.0, 30.0]),
    ]);
    let outputs = graph.execute_f32(&inputs).unwrap();
    assert_eq!(outputs["y"], vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
fn validation_accepts_broadcast_where_and_rejects_incompatible_shapes() {
    let mut graph = TinyGraph::default();
    let condition = graph.add(
        UOpKind::Input {
            name: "condition".into(),
        },
        vec![],
        vec![2, 1],
    );
    let when_true = graph.add(
        UOpKind::Input {
            name: "true".into(),
        },
        vec![],
        vec![2, 3],
    );
    let when_false = graph.add(
        UOpKind::Input {
            name: "false".into(),
        },
        vec![],
        vec![3],
    );
    let selected = graph.add(
        UOpKind::Where,
        vec![condition, when_true, when_false],
        vec![2, 3],
    );
    graph.add(
        UOpKind::Output { name: "y".into() },
        vec![selected],
        vec![2, 3],
    );
    assert!(graph.validate().is_ok());

    let mut invalid = graph.clone();
    invalid.ops[2].shape = vec![4];
    assert!(matches!(
        invalid.validate(),
        Err(GraphError::ShapeMismatch(_))
    ));
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert_eq!(capture.kernels.len(), 1);
    assert!(capture.kernels[0].source.contains("prism_where"));
    assert!(capture.kernels[0].source.contains("cc0"));
}

#[test]
fn lowering_supports_shape_aware_broadcast_kernel_abi() {
    let mut graph = TinyGraph::default();
    let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
    let bias = graph.add(
        UOpKind::Input {
            name: "bias".into(),
        },
        vec![],
        vec![3],
    );
    let y = graph.add(UOpKind::Add, vec![x, bias], vec![2, 3]);
    let y = graph.add(UOpKind::Relu, vec![y], vec![2, 3]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![y], vec![2, 3]);
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert_eq!(capture.kernels.len(), 1);
    assert!(capture.kernels[0].source.contains("prism_broadcast_binary"));
    assert!(capture.kernels[0].source.contains("max(value, 0.0f)"));
}

#[test]
fn lowering_separates_where_after_broadcast_binary() {
    let mut graph = TinyGraph::default();
    let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
    let bias = graph.add(
        UOpKind::Input {
            name: "bias".into(),
        },
        vec![],
        vec![3],
    );
    let condition = graph.add(
        UOpKind::Input {
            name: "condition".into(),
        },
        vec![],
        vec![2, 1],
    );
    let added = graph.add(UOpKind::Add, vec![x, bias], vec![2, 3]);
    let selected = graph.add(UOpKind::Where, vec![condition, added, x], vec![2, 3]);
    graph.add(
        UOpKind::Output { name: "y".into() },
        vec![selected],
        vec![2, 3],
    );
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert_eq!(capture.kernels.len(), 2);
    assert!(capture.kernels[0].source.contains("prism_broadcast_binary"));
    assert!(capture.kernels[1].source.contains("prism_where"));
    assert!(capture
        .kernels
        .iter()
        .all(|kernel| !kernel.source.is_empty()));
}

#[test]
fn lowering_separates_cast_after_broadcast_binary() {
    let mut graph = TinyGraph::default();
    let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
    let bias = graph.add(
        UOpKind::Input {
            name: "bias".into(),
        },
        vec![],
        vec![3],
    );
    let added = graph.add(UOpKind::Add, vec![x, bias], vec![2, 3]);
    let cast = graph.add(
        UOpKind::Cast {
            from: "f32".into(),
            to: "i8".into(),
        },
        vec![added],
        vec![2, 3],
    );
    graph.add(UOpKind::Output { name: "y".into() }, vec![cast], vec![2, 3]);
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert_eq!(capture.kernels.len(), 2);
    assert!(capture
        .kernels
        .iter()
        .all(|kernel| !kernel.source.is_empty()));
    assert!(capture.kernels[1].source.contains("prism_cast"));
}

#[test]
fn reshape_is_a_kernel_free_row_major_alias() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
    let reshaped = graph.add(UOpKind::Reshape, vec![input], vec![3, 2]);
    graph.add(
        UOpKind::Output { name: "y".into() },
        vec![reshaped],
        vec![3, 2],
    );
    let mut inputs = BTreeMap::new();
    inputs.insert("x".into(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(
        graph.execute_f32(&inputs).unwrap()["y"],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert!(capture.kernels.is_empty());
    assert_eq!(capture.memory_plan.slot_count, 0);
}

#[test]
fn reshape_rejects_element_count_changes() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
    graph.add(UOpKind::Reshape, vec![input], vec![4, 2]);
    assert!(matches!(
        graph.validate(),
        Err(GraphError::ShapeMismatch(_))
    ));
}

#[test]
fn optimizer_folds_constant_extrema() {
    let mut graph = TinyGraph::default();
    let left = graph.add(UOpKind::Const { value: -2.0 }, vec![], vec![1]);
    let right = graph.add(UOpKind::Const { value: 3.0 }, vec![], vec![1]);
    let maximum = graph.add(UOpKind::Maximum, vec![left, right], vec![1]);
    let minimum = graph.add(UOpKind::Minimum, vec![left, right], vec![1]);
    graph.add(
        UOpKind::Output {
            name: "maximum".into(),
        },
        vec![maximum],
        vec![1],
    );
    graph.add(
        UOpKind::Output {
            name: "minimum".into(),
        },
        vec![minimum],
        vec![1],
    );
    let optimized = graph.optimize().unwrap();
    assert!(matches!(
        optimized.ops[maximum.0 as usize].kind,
        UOpKind::Const { value } if value == 3.0
    ));
    assert!(matches!(
        optimized.ops[minimum.0 as usize].kind,
        UOpKind::Const { value } if value == -2.0
    ));
}

#[test]
fn optimizer_eliminates_duplicate_elementwise_subexpressions() {
    let mut graph = TinyGraph::default();
    let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
    let first = graph.add(UOpKind::Relu, vec![x], vec![2]);
    let second = graph.add(UOpKind::Relu, vec![x], vec![2]);
    graph.add(
        UOpKind::Output { name: "out".into() },
        vec![second],
        vec![2],
    );
    let optimized = graph.optimize().unwrap();
    assert!(matches!(
        optimized.ops[second.0 as usize].kind,
        UOpKind::Const { value } if value == 0.0
    ));
    assert_eq!(optimized.ops[graph.ops.len() - 1].src, vec![first]);
    let mut inputs = BTreeMap::new();
    inputs.insert("x".into(), vec![-1.0, 2.0]);
    assert_eq!(
        optimized.execute_f32(&inputs).unwrap()["out"],
        vec![0.0, 2.0]
    );
}

#[test]
fn lowering_prunes_dead_rewritten_uops_from_capture_graph() {
    let mut graph = TinyGraph::default();
    let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
    let first = graph.add(UOpKind::Relu, vec![x], vec![2]);
    let duplicate = graph.add(UOpKind::Relu, vec![x], vec![2]);
    graph.add(
        UOpKind::Output { name: "out".into() },
        vec![duplicate],
        vec![2],
    );
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert!(capture.graph.ops.iter().any(|op| op.id == first));
    assert!(!capture.graph.ops.iter().any(|op| op.id == duplicate));
    assert!(capture.graph_op_count < graph.ops.len());
}

#[test]
fn capture_digest_is_deterministic() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
    let relu = graph.add(UOpKind::Relu, vec![input], vec![1]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![1]);
    let left = graph.lower(LoweringTarget::Portable).unwrap();
    let right = graph.lower(LoweringTarget::Portable).unwrap();
    assert_eq!(left.digest(), right.digest());
    assert_eq!(left.digest().len(), 64);
}

#[test]
fn memory_plan_assigns_reusable_slots() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
    let first = graph.add(UOpKind::Relu, vec![input], vec![1]);
    let second = graph.add(UOpKind::Relu, vec![first], vec![1]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![second], vec![1]);
    let plan = graph.memory_plan().unwrap();
    assert_eq!(plan.allocations.len(), 2);
    assert_eq!(plan.slot_count, 2);
    assert!(plan.allocations[0].last_command <= plan.allocations[1].first_command);
}

#[test]
fn memory_plan_does_not_reuse_a_slot_for_a_larger_value() {
    let mut graph = TinyGraph::default();
    let narrow = graph.add(
        UOpKind::Input {
            name: "narrow".into(),
        },
        vec![],
        vec![2],
    );
    let reduce = graph.add(UOpKind::ReduceSum, vec![narrow], vec![1]);
    let wide = graph.add(
        UOpKind::Input {
            name: "wide".into(),
        },
        vec![],
        vec![4],
    );
    let expanded = graph.add(UOpKind::Add, vec![reduce, wide], vec![4]);
    let output = graph.add(UOpKind::Relu, vec![expanded], vec![4]);
    graph.add(
        UOpKind::Output { name: "out".into() },
        vec![output],
        vec![4],
    );

    let plan = graph.memory_plan().unwrap();
    assert_eq!(plan.allocations[0].elements, 1);
    assert_eq!(plan.allocations[1].elements, 4);
    assert_eq!(plan.allocations[2].elements, 4);
    assert_ne!(plan.allocations[0].slot, plan.allocations[2].slot);
}

#[test]
fn validation_rejects_incompatible_elementwise_shapes() {
    let mut graph = TinyGraph::default();
    let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
    let y = graph.add(UOpKind::Input { name: "y".into() }, vec![], vec![3]);
    graph.add(UOpKind::Add, vec![x, y], vec![2]);
    assert!(matches!(
        graph.validate(),
        Err(GraphError::ShapeMismatch(_))
    ));
}

#[test]
fn validation_rejects_empty_and_zero_sized_uop_shapes() {
    let mut empty = TinyGraph::default();
    empty.add(
        UOpKind::Input {
            name: "empty".into(),
        },
        vec![],
        vec![],
    );
    assert!(matches!(
        empty.validate(),
        Err(GraphError::ShapeMismatch(_))
    ));

    let mut zero = TinyGraph::default();
    zero.add(
        UOpKind::Input {
            name: "zero".into(),
        },
        vec![],
        vec![2, 0],
    );
    assert!(matches!(zero.validate(), Err(GraphError::ShapeMismatch(_))));

    let mut overflowing = TinyGraph::default();
    overflowing.add(
        UOpKind::Input {
            name: "overflowing".into(),
        },
        vec![],
        vec![usize::MAX as u64, 2],
    );
    assert!(matches!(
        overflowing.validate(),
        Err(GraphError::ShapeMismatch(_))
    ));

    let mut mismatched_output = TinyGraph::default();
    let input = mismatched_output.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
    mismatched_output.add(UOpKind::Output { name: "out".into() }, vec![input], vec![1]);
    assert!(matches!(
        mismatched_output.validate(),
        Err(GraphError::ShapeMismatch(_))
    ));
}

#[test]
fn validation_rejects_equal_size_but_non_broadcastable_shapes() {
    let mut graph = TinyGraph::default();
    let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 2]);
    let y = graph.add(UOpKind::Input { name: "y".into() }, vec![], vec![4]);
    graph.add(UOpKind::Add, vec![x, y], vec![2, 2]);
    assert!(matches!(
        graph.validate(),
        Err(GraphError::ShapeMismatch(_))
    ));
}

#[test]
fn validation_rejects_output_shape_mismatch() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
    graph.add(UOpKind::Output { name: "out".into() }, vec![input], vec![5]);
    assert!(matches!(
        graph.validate(),
        Err(GraphError::ShapeMismatch(_))
    ));
}

#[test]
fn validation_rejects_malformed_layer_norm_parameters() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
    let weight = graph.add(UOpKind::Input { name: "w".into() }, vec![], vec![2]);
    let bias = graph.add(UOpKind::Input { name: "b".into() }, vec![], vec![2]);
    let norm = graph.add(
        UOpKind::LayerNorm {
            rows: 2,
            features: 3,
            epsilon: 1e-5,
        },
        vec![input, weight, bias],
        vec![2, 3],
    );
    graph.add(UOpKind::Output { name: "y".into() }, vec![norm], vec![2, 3]);
    assert!(matches!(
        graph.validate(),
        Err(GraphError::ShapeMismatch(_))
    ));
}

#[test]
fn validation_rejects_duplicate_ids_and_output_names() {
    let duplicate = TinyGraph {
        ops: vec![
            UOp {
                id: UOpId(0),
                kind: UOpKind::Input { name: "x".into() },
                src: vec![],
                shape: vec![1],
            },
            UOp {
                id: UOpId(0),
                kind: UOpKind::Input { name: "y".into() },
                src: vec![],
                shape: vec![1],
            },
        ],
    };
    assert!(matches!(
        duplicate.validate(),
        Err(GraphError::DuplicateId(UOpId(0)))
    ));

    let mut outputs = TinyGraph::default();
    let input = outputs.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
    outputs.add(UOpKind::Output { name: "y".into() }, vec![input], vec![1]);
    outputs.add(UOpKind::Output { name: "y".into() }, vec![input], vec![1]);
    assert!(matches!(
        outputs.validate(),
        Err(GraphError::DuplicateOutput(_))
    ));

    let mut inputs = TinyGraph::default();
    inputs.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
    inputs.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
    assert!(matches!(
        inputs.validate(),
        Err(GraphError::DuplicateInput(_))
    ));

    let mut empty_input = TinyGraph::default();
    empty_input.add(
        UOpKind::Input {
            name: String::new(),
        },
        vec![],
        vec![1],
    );
    assert!(matches!(
        empty_input.validate(),
        Err(GraphError::EmptyInputName)
    ));

    let mut empty_output = TinyGraph::default();
    let input = empty_output.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
    empty_output.add(
        UOpKind::Output {
            name: String::new(),
        },
        vec![input],
        vec![1],
    );
    assert!(matches!(
        empty_output.validate(),
        Err(GraphError::EmptyOutputName)
    ));
}

#[test]
fn capture_validation_checks_replay_and_memory_invariants() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
    let relu = graph.add(UOpKind::Relu, vec![input], vec![1]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![1]);
    let mut capture = graph.lower(LoweringTarget::Portable).unwrap();
    capture.replay.synchronization_points.push(99);
    assert!(capture.validate().is_err());
}

#[test]
fn capture_validation_rejects_tampered_graph_operation_count() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
    let relu = graph.add(UOpKind::Relu, vec![input], vec![1]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![1]);
    let mut capture = graph.lower(LoweringTarget::Portable).unwrap();
    capture.graph_op_count += 1;
    let error = capture.validate().unwrap_err();
    assert!(error.contains("operation count mismatch"));
}

#[test]
fn capture_validation_rejects_noncanonical_command_ids() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
    let relu = graph.add(UOpKind::Relu, vec![input], vec![1]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![1]);
    let mut capture = graph.lower(LoweringTarget::Portable).unwrap();
    capture.replay.command_ids[0] = 7;
    let error = capture.validate().unwrap_err();
    assert!(error.contains("canonical"));
}

#[test]
fn capture_validation_rejects_noncanonical_synchronization_points() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
    let relu = graph.add(UOpKind::Relu, vec![input], vec![1]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![1]);
    let mut capture = graph.lower(LoweringTarget::Portable).unwrap();
    capture.replay.synchronization_points = vec![0, 0];
    let error = capture.validate().unwrap_err();
    assert!(error.contains("synchronization points are not canonical"));
}

#[test]
fn capture_validation_rejects_tampered_kernel_source() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
    let relu = graph.add(UOpKind::Relu, vec![input], vec![1]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![1]);
    let mut capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert!(capture.validate().is_ok());
    capture.kernels[0].source.push_str(" tampered");
    assert!(capture.validate().is_err());
}

#[test]
fn capture_validation_rejects_empty_kernel_source() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
    let relu = graph.add(UOpKind::Relu, vec![input], vec![1]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![1]);
    let mut capture = graph.lower(LoweringTarget::Portable).unwrap();
    capture.kernels[0].source.clear();
    let mut digest = Sha256::new();
    digest.update(capture.kernels[0].source.as_bytes());
    capture.kernels[0].source_digest = hex_digest(digest.finalize());
    let error = capture.validate().unwrap_err();
    assert!(error.contains("empty rendered source"));
}

#[test]
fn capture_validation_rejects_tampered_output_geometry() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
    let relu = graph.add(UOpKind::Relu, vec![input], vec![2]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![relu], vec![2]);
    let mut capture = graph.lower(LoweringTarget::Portable).unwrap();
    capture.kernels[0].output_elements = Some(3);
    assert!(capture.validate().is_err());
}

#[test]
fn renderer_preserves_scalar_operand_order() {
    let mut graph = TinyGraph::default();
    let two = graph.add(UOpKind::Const { value: 2.0 }, vec![], vec![1]);
    let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
    let sub = graph.add(UOpKind::Sub, vec![two, x], vec![1]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![sub], vec![1]);
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert!(capture.kernels[0].source.contains("v = 2 - v"));
}

#[test]
fn negation_fuses_and_executes() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
    let neg = graph.add(UOpKind::Neg, vec![input], vec![2]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![neg], vec![2]);
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert_eq!(capture.kernels[0].output_elements, Some(2));
    assert!(capture.kernels[0].source.contains("v = -v"));
    let mut inputs = BTreeMap::new();
    inputs.insert("x".into(), vec![-2.0, 3.0]);
    assert_eq!(graph.execute_f32(&inputs).unwrap()["y"], vec![2.0, -3.0]);
}

#[test]
fn exp_and_sqrt_render_and_execute() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1]);
    let exp = graph.add(UOpKind::Exp, vec![input], vec![1]);
    let root = graph.add(UOpKind::Sqrt, vec![exp], vec![1]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![root], vec![1]);
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert!(capture.kernels[0].source.contains("expf(v)"));
    assert!(capture.kernels[0].source.contains("sqrtf(v)"));
    let mut inputs = BTreeMap::new();
    inputs.insert("x".into(), vec![0.0]);
    let value = graph.execute_f32(&inputs).unwrap()["y"][0];
    assert!((value - 1.0).abs() < 1e-6);
}

#[test]
fn pow_renders_and_executes_scalar_exponent() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![3]);
    let pow = graph.add(UOpKind::Pow { exponent: 2.0 }, vec![input], vec![3]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![pow], vec![3]);
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert!(capture.kernels[0].source.contains("powf(v, 2f)"));
    let output = graph
        .execute_f32(&BTreeMap::from([("x".into(), vec![-2.0, 3.0, 0.5])]))
        .unwrap();
    assert_eq!(output["y"], vec![4.0, 9.0, 0.25]);
}

#[test]
fn sin_and_cos_render_and_execute() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
    let sin = graph.add(UOpKind::Sin, vec![input], vec![2]);
    let cos = graph.add(UOpKind::Cos, vec![sin], vec![2]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![cos], vec![2]);
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert!(capture.kernels[0].source.contains("sinf(v)"));
    assert!(capture.kernels[0].source.contains("cosf(v)"));
    let mut inputs = BTreeMap::new();
    inputs.insert("x".into(), vec![0.0, std::f32::consts::FRAC_PI_2]);
    let output = graph.execute_f32(&inputs).unwrap();
    assert!((output["y"][0] - 1.0).abs() < 1e-6);
    assert!((output["y"][1] - 1.0f32.cos()).abs() < 1e-6);
}

#[test]
fn renderer_declares_rhs_for_two_input_elementwise_ops() {
    let mut graph = TinyGraph::default();
    let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
    let y = graph.add(UOpKind::Input { name: "y".into() }, vec![], vec![2]);
    let sum = graph.add(UOpKind::Add, vec![x, y], vec![2]);
    graph.add(UOpKind::Output { name: "z".into() }, vec![sum], vec![2]);
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert!(capture.kernels[0].source.contains("float* rhs"));
    assert!(capture.kernels[0].source.contains("rhs[id]"));
}

#[test]
fn maximum_and_minimum_render_execute_and_preserve_rhs_abi() {
    let mut graph = TinyGraph::default();
    let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![3]);
    let y = graph.add(UOpKind::Input { name: "y".into() }, vec![], vec![3]);
    let upper = graph.add(UOpKind::Maximum, vec![x, y], vec![3]);
    let lower = graph.add(UOpKind::Minimum, vec![upper, y], vec![3]);
    graph.add(UOpKind::Output { name: "out".into() }, vec![lower], vec![3]);
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert!(capture
        .kernels
        .iter()
        .all(|kernel| kernel.group.requires_rhs()));
    assert!(
        capture
            .kernels
            .iter()
            .any(|kernel| kernel.source.contains("max(v, rhs[id])")),
        "{:#?}",
        capture.kernels
    );
    assert!(
        capture
            .kernels
            .iter()
            .any(|kernel| kernel.source.contains("min(v, rhs[id])")),
        "{:#?}",
        capture.kernels
    );
    let outputs = graph
        .execute_f32(&BTreeMap::from([
            ("x".into(), vec![-2.0, 4.0, 1.0]),
            ("y".into(), vec![1.0, 3.0, 2.0]),
        ]))
        .unwrap();
    assert_eq!(outputs["out"], vec![1.0, 3.0, 2.0]);
}

#[test]
fn unary_math_family_renders_and_executes() {
    let mut graph = TinyGraph::default();
    let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2]);
    let abs = graph.add(UOpKind::Abs, vec![x], vec![2]);
    let log = graph.add(UOpKind::Log, vec![abs], vec![2]);
    let tanh = graph.add(UOpKind::Tanh, vec![log], vec![2]);
    graph.add(UOpKind::Output { name: "out".into() }, vec![tanh], vec![2]);
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    let source = &capture.kernels[0].source;
    assert!(source.contains("fabsf(v)"));
    assert!(source.contains("logf(v)"));
    assert!(source.contains("tanhf(v)"));
    let outputs = graph
        .execute_f32(&BTreeMap::from([("x".into(), vec![-1.0, 2.0])]))
        .unwrap();
    assert!((outputs["out"][0] - 0.0).abs() < 1e-6);
    assert!((outputs["out"][1] - 2.0f32.ln().tanh()).abs() < 1e-6);
}

#[test]
fn reduce_sum_has_shape_changing_kernel_and_reference_behavior() {
    let mut graph = TinyGraph::default();
    let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
    let sum = graph.add(UOpKind::ReduceSum, vec![x], vec![1]);
    graph.add(UOpKind::Output { name: "sum".into() }, vec![sum], vec![1]);
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert_eq!(capture.kernels.len(), 1);
    assert!(capture.kernels[0]
        .source
        .contains("for (unsigned i = 0; i < 4; ++i)"));
    assert!(capture.kernels[0].source.contains("output[0] = v"));
    let outputs = graph
        .execute_f32(&BTreeMap::from([("x".into(), vec![1.0, -2.0, 3.0, 4.0])]))
        .unwrap();
    assert_eq!(outputs["sum"], vec![6.0]);
}

#[test]
fn reduce_max_has_dedicated_kernel_and_reference_behavior() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
    let maximum = graph.add(UOpKind::ReduceMax, vec![input], vec![1]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![maximum], vec![1]);
    let mut inputs = BTreeMap::new();
    inputs.insert("x".into(), vec![-2.0, 7.0, 3.0, 1.0]);
    assert_eq!(graph.execute_f32(&inputs).unwrap()["y"], vec![7.0]);
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert!(capture.kernels[0].source.contains("prism_reduce_max"));
}

#[test]
fn reduce_min_has_dedicated_kernel_and_reference_behavior() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![4]);
    let minimum = graph.add(UOpKind::ReduceMin, vec![input], vec![1]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![minimum], vec![1]);
    let mut inputs = BTreeMap::new();
    inputs.insert("x".into(), vec![-2.0, 7.0, 3.0, 1.0]);
    assert_eq!(graph.execute_f32(&inputs).unwrap()["y"], vec![-2.0]);
    assert!(graph.lower(LoweringTarget::Portable).unwrap().kernels[0]
        .source
        .contains("prism_reduce_min"));
}

#[test]
fn reduce_max_axis_has_dedicated_kernel_and_reference_behavior() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
    let maximum = graph.add(UOpKind::ReduceMaxAxis { axis: 1 }, vec![input], vec![2]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![maximum], vec![2]);
    let mut inputs = BTreeMap::new();
    inputs.insert("x".into(), vec![1.0, 7.0, 3.0, 9.0, 2.0, 4.0]);
    assert_eq!(graph.execute_f32(&inputs).unwrap()["y"], vec![7.0, 9.0]);
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert!(capture.kernels[0].source.contains("prism_reduce_max_axis"));
}

#[test]
fn reduce_min_axis_has_dedicated_kernel_and_reference_behavior() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
    let minimum = graph.add(UOpKind::ReduceMinAxis { axis: 1 }, vec![input], vec![2]);
    graph.add(UOpKind::Output { name: "y".into() }, vec![minimum], vec![2]);
    let mut inputs = BTreeMap::new();
    inputs.insert("x".into(), vec![1.0, 7.0, 3.0, 9.0, 2.0, 4.0]);
    assert_eq!(graph.execute_f32(&inputs).unwrap()["y"], vec![1.0, 2.0]);
    assert!(graph.lower(LoweringTarget::Portable).unwrap().kernels[0]
        .source
        .contains("prism_reduce_min_axis"));
}

#[test]
fn axis_reduction_rejects_same_element_count_wrong_shape() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 1]);
    let maximum = graph.add(UOpKind::ReduceMaxAxis { axis: 1 }, vec![input], vec![2, 1]);
    graph.add(
        UOpKind::Output { name: "y".into() },
        vec![maximum],
        vec![2, 1],
    );
    assert_eq!(graph.validate(), Err(GraphError::ShapeMismatch(maximum)));
}

#[test]
fn matmul_validates_renders_and_executes() {
    let mut graph = TinyGraph::default();
    let a = graph.add(UOpKind::Input { name: "a".into() }, vec![], vec![2, 3]);
    let b = graph.add(UOpKind::Input { name: "b".into() }, vec![], vec![3, 2]);
    let product = graph.add(UOpKind::MatMul { m: 2, k: 3, n: 2 }, vec![a, b], vec![2, 2]);
    graph.add(
        UOpKind::Output { name: "out".into() },
        vec![product],
        vec![2, 2],
    );
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert_eq!(capture.kernels.len(), 1);
    assert!(capture.kernels[0].source.contains("prism_matmul"));
    assert!(capture.kernels[0].source.contains("inner < 3u"));
    let outputs = graph
        .execute_f32(&BTreeMap::from([
            ("a".into(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ("b".into(), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]),
        ]))
        .unwrap();
    assert_eq!(outputs["out"], vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn axis_reduction_validates_renders_and_executes() {
    let mut graph = TinyGraph::default();
    let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
    let sum = graph.add(UOpKind::ReduceSumAxis { axis: 1 }, vec![x], vec![2]);
    graph.add(UOpKind::Output { name: "sum".into() }, vec![sum], vec![2]);
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert!(capture.kernels[0].source.contains("prism_reduce_sum_axis"));
    assert!(capture.kernels[0].source.contains("step < 3u"));
    let outputs = graph
        .execute_f32(&BTreeMap::from([(
            "x".into(),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        )]))
        .unwrap();
    assert_eq!(outputs["sum"], vec![6.0, 15.0]);
}

#[test]
fn softmax_axis_is_stable_and_normalizes_each_row() {
    let mut graph = TinyGraph::default();
    let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3]);
    let softmax = graph.add(UOpKind::SoftmaxAxis { axis: 1 }, vec![x], vec![2, 3]);
    graph.add(
        UOpKind::Output { name: "out".into() },
        vec![softmax],
        vec![2, 3],
    );
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert!(capture.kernels[0].source.contains("prism_softmax_axis"));
    let output = graph
        .execute_f32(&BTreeMap::from([(
            "x".into(),
            vec![1000.0, 1001.0, 1002.0, 0.0, 0.0, 0.0],
        )]))
        .unwrap();
    for row in output["out"].chunks_exact(3) {
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
    assert!(output["out"].iter().all(|value| value.is_finite()));
}

#[test]
fn fused_attention_matches_scaled_dot_product_reference() {
    let mut graph = TinyGraph::default();
    let q = graph.add(UOpKind::Input { name: "q".into() }, vec![], vec![2, 2]);
    let k = graph.add(UOpKind::Input { name: "k".into() }, vec![], vec![2, 2]);
    let v = graph.add(UOpKind::Input { name: "v".into() }, vec![], vec![2, 2]);
    let attention = graph.add(
        UOpKind::Attention {
            seq: 2,
            head: 2,
            scale: 0.5,
        },
        vec![q, k, v],
        vec![2, 2],
    );
    graph.add(
        UOpKind::Output { name: "out".into() },
        vec![attention],
        vec![2, 2],
    );
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert!(capture.kernels[0].source.contains("prism_attention"));
    let output = graph
        .execute_f32(&BTreeMap::from([
            ("q".into(), vec![1.0, 0.0, 0.0, 1.0]),
            ("k".into(), vec![1.0, 0.0, 0.0, 1.0]),
            ("v".into(), vec![2.0, 4.0, 8.0, 16.0]),
        ]))
        .unwrap();
    assert!(output["out"].iter().all(|value| value.is_finite()));
    assert!(output["out"][0] > 2.0 && output["out"][0] < 8.0);
    assert!(output["out"][1] > 4.0 && output["out"][1] < 16.0);
}

#[test]
fn batched_attention_preserves_batch_independence() {
    let mut graph = TinyGraph::default();
    let q = graph.add(UOpKind::Input { name: "q".into() }, vec![], vec![2, 2, 1]);
    let k = graph.add(UOpKind::Input { name: "k".into() }, vec![], vec![2, 2, 1]);
    let v = graph.add(UOpKind::Input { name: "v".into() }, vec![], vec![2, 2, 1]);
    let attention = graph.add(
        UOpKind::AttentionBatched {
            batch: 2,
            seq: 2,
            head: 1,
            scale: 1.0,
        },
        vec![q, k, v],
        vec![2, 2, 1],
    );
    graph.add(
        UOpKind::Output { name: "out".into() },
        vec![attention],
        vec![2, 2, 1],
    );
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert!(capture.kernels[0]
        .source
        .contains("prism_attention_batched"));
    let output = graph
        .execute_f32(&BTreeMap::from([
            ("q".into(), vec![1.0, 0.0, 0.0, 1.0]),
            ("k".into(), vec![1.0, 0.0, 0.0, 1.0]),
            ("v".into(), vec![2.0, 4.0, 8.0, 16.0]),
        ]))
        .unwrap();
    assert!(output["out"].iter().all(|value| value.is_finite()));
    assert!(output["out"][0] < output["out"][2]);
}

#[test]
fn gelu_renders_and_matches_reference_approximation() {
    let mut graph = TinyGraph::default();
    let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![3]);
    let gelu = graph.add(UOpKind::Gelu, vec![x], vec![3]);
    graph.add(UOpKind::Output { name: "out".into() }, vec![gelu], vec![3]);
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert!(capture.kernels[0].source.contains("tanhf"));
    let output = graph
        .execute_f32(&BTreeMap::from([("x".into(), vec![-1.0, 0.0, 1.0])]))
        .unwrap();
    assert!(output["out"][1].abs() < 1e-6);
    assert!(output["out"][0] < 0.0 && output["out"][2] > 0.0);
}

#[test]
fn rms_norm_renders_and_matches_reference() {
    let mut graph = TinyGraph::default();
    let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 2]);
    let weight = graph.add(
        UOpKind::Input {
            name: "weight".into(),
        },
        vec![],
        vec![2],
    );
    let norm = graph.add(
        UOpKind::RmsNorm {
            rows: 2,
            features: 2,
            epsilon: 1e-5,
        },
        vec![x, weight],
        vec![2, 2],
    );
    graph.add(
        UOpKind::Output { name: "out".into() },
        vec![norm],
        vec![2, 2],
    );
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert!(capture.kernels[0].source.contains("prism_rms_norm"));
    let output = graph
        .execute_f32(&BTreeMap::from([
            ("x".into(), vec![3.0, 4.0, 0.0, 2.0]),
            ("weight".into(), vec![2.0, 0.5]),
        ]))
        .unwrap();
    assert!((output["out"][0] - 1.697056).abs() < 1e-3);
    assert!((output["out"][1] - 0.565685).abs() < 1e-3);
}

#[test]
fn layer_norm_renders_and_matches_reference() {
    let mut graph = TinyGraph::default();
    let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1, 2]);
    let weight = graph.add(
        UOpKind::Input {
            name: "weight".into(),
        },
        vec![],
        vec![2],
    );
    let bias = graph.add(
        UOpKind::Input {
            name: "bias".into(),
        },
        vec![],
        vec![2],
    );
    let norm = graph.add(
        UOpKind::LayerNorm {
            rows: 1,
            features: 2,
            epsilon: 1e-5,
        },
        vec![x, weight, bias],
        vec![1, 2],
    );
    graph.add(
        UOpKind::Output { name: "out".into() },
        vec![norm],
        vec![1, 2],
    );
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert!(capture.kernels[0].source.contains("prism_layer_norm"));
    let output = graph
        .execute_f32(&BTreeMap::from([
            ("x".into(), vec![1.0, 3.0]),
            ("weight".into(), vec![2.0, 0.5]),
            ("bias".into(), vec![1.0, -1.0]),
        ]))
        .unwrap();
    assert!((output["out"][0] + 1.0).abs() < 1e-3);
    assert!((output["out"][1] + 0.5).abs() < 1e-3);
}

#[test]
fn rope_validates_and_matches_reference_rotation() {
    let mut graph = TinyGraph::default();
    let x = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![1, 4]);
    let cos = graph.add(UOpKind::Input { name: "cos".into() }, vec![], vec![1, 2]);
    let sin = graph.add(UOpKind::Input { name: "sin".into() }, vec![], vec![1, 2]);
    let rope = graph.add(
        UOpKind::Rope {
            rows: 1,
            features: 4,
        },
        vec![x, cos, sin],
        vec![1, 4],
    );
    graph.add(
        UOpKind::Output { name: "out".into() },
        vec![rope],
        vec![1, 4],
    );
    let mut inputs = BTreeMap::new();
    inputs.insert("x".into(), vec![1.0, 2.0, 3.0, 4.0]);
    inputs.insert("cos".into(), vec![0.0, 1.0]);
    inputs.insert("sin".into(), vec![1.0, 0.0]);
    let outputs = graph.execute_f32(&inputs).unwrap();
    assert_eq!(outputs["out"], vec![-2.0, 1.0, 3.0, 4.0]);
}

#[test]
fn transpose_executes_nontrivial_rank_three_permutation() {
    let mut graph = TinyGraph::default();
    let input = graph.add(UOpKind::Input { name: "x".into() }, vec![], vec![2, 3, 4]);
    let transpose = graph.add(
        UOpKind::Transpose {
            permutation: vec![2, 0, 1],
        },
        vec![input],
        vec![4, 2, 3],
    );
    graph.add(
        UOpKind::Output { name: "out".into() },
        vec![transpose],
        vec![4, 2, 3],
    );
    let input_values = (0..24).map(|value| value as f32).collect::<Vec<_>>();
    let output = graph
        .execute_f32(&BTreeMap::from([(String::from("x"), input_values)]))
        .unwrap();
    let expected = (0..4)
        .flat_map(|c| {
            (0..2).flat_map(move |a| (0..3).map(move |b| (a * 12 + b * 4 + c) as f32))
        })
        .collect::<Vec<_>>();
    assert_eq!(output["out"], expected);
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert!(capture.kernels[0].source.contains("prism_transpose"));
}

#[test]
fn gather_validates_and_executes_embedding_lookup() {
    let mut graph = TinyGraph::default();
    let weight = graph.add(
        UOpKind::Input {
            name: "weight".into(),
        },
        vec![],
        vec![3, 2],
    );
    let indices = graph.add(
        UOpKind::Input {
            name: "indices".into(),
        },
        vec![],
        vec![2],
    );
    let gather = graph.add(
        UOpKind::Gather {
            rows: 2,
            vocab: 3,
            features: 2,
        },
        vec![weight, indices],
        vec![2, 2],
    );
    graph.add(
        UOpKind::Output { name: "out".into() },
        vec![gather],
        vec![2, 2],
    );
    let mut inputs = BTreeMap::new();
    inputs.insert("weight".into(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    inputs.insert("indices".into(), vec![2.0, 0.0]);
    assert_eq!(
        graph.execute_f32(&inputs).unwrap()["out"],
        vec![5.0, 6.0, 1.0, 2.0]
    );
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert!(capture.kernels[0].source.contains("prism_gather"));
    assert!(capture.validate().is_ok());
}

#[test]
fn scatter_validates_executes_and_uses_last_update() {
    let mut graph = TinyGraph::default();
    let base = graph.add(
        UOpKind::Input {
            name: "base".into(),
        },
        vec![],
        vec![3, 2],
    );
    let indices = graph.add(
        UOpKind::Input {
            name: "indices".into(),
        },
        vec![],
        vec![2],
    );
    let updates = graph.add(
        UOpKind::Input {
            name: "updates".into(),
        },
        vec![],
        vec![2, 2],
    );
    let scatter = graph.add(
        UOpKind::Scatter {
            rows: 3,
            updates: 2,
            features: 2,
        },
        vec![base, indices, updates],
        vec![3, 2],
    );
    graph.add(
        UOpKind::Output { name: "out".into() },
        vec![scatter],
        vec![3, 2],
    );
    let inputs = BTreeMap::from([
        ("base".into(), vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]),
        ("indices".into(), vec![1.0, 1.0]),
        ("updates".into(), vec![7.0, 8.0, 9.0, 10.0]),
    ]);
    assert_eq!(
        graph.execute_f32(&inputs).unwrap()["out"],
        vec![0.0, 0.0, 9.0, 10.0, 2.0, 2.0]
    );
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert!(capture.kernels[0].source.contains("prism_scatter"));
    assert!(capture.validate().is_ok());
}

#[test]
fn ssm_validates_and_executes_diagonal_scan() {
    let mut graph = TinyGraph::default();
    let input = graph.add(
        UOpKind::Input {
            name: "input".into(),
        },
        vec![],
        vec![2, 2],
    );
    let decay = graph.add(
        UOpKind::Input {
            name: "decay".into(),
        },
        vec![],
        vec![2],
    );
    let input_gain = graph.add(
        UOpKind::Input {
            name: "input_gain".into(),
        },
        vec![],
        vec![2],
    );
    let output_gain = graph.add(
        UOpKind::Input {
            name: "output_gain".into(),
        },
        vec![],
        vec![2],
    );
    let scan = graph.add(
        UOpKind::Ssm {
            rows: 2,
            features: 2,
        },
        vec![input, decay, input_gain, output_gain],
        vec![2, 2],
    );
    graph.add(
        UOpKind::Output { name: "out".into() },
        vec![scan],
        vec![2, 2],
    );
    let mut inputs = BTreeMap::new();
    inputs.insert("input".into(), vec![1.0, 2.0, 3.0, 4.0]);
    inputs.insert("decay".into(), vec![0.5, 0.25]);
    inputs.insert("input_gain".into(), vec![1.0, 1.0]);
    inputs.insert("output_gain".into(), vec![2.0, 4.0]);
    assert_eq!(
        graph.execute_f32(&inputs).unwrap()["out"],
        vec![2.0, 8.0, 7.0, 18.0]
    );
}

#[test]
fn conv2d_renders_and_executes_nchw() {
    let mut graph = TinyGraph::default();
    let x = graph.add(
        UOpKind::Input { name: "x".into() },
        vec![],
        vec![1, 1, 3, 3],
    );
    let weight = graph.add(
        UOpKind::Input {
            name: "weight".into(),
        },
        vec![],
        vec![1, 1, 2, 2],
    );
    let bias = graph.add(
        UOpKind::Input {
            name: "bias".into(),
        },
        vec![],
        vec![1],
    );
    let conv = graph.add(
        UOpKind::Conv2d {
            batch: 1,
            in_channels: 1,
            height: 3,
            width: 3,
            out_channels: 1,
            kernel_h: 2,
            kernel_w: 2,
            stride: 1,
            padding: 0,
        },
        vec![x, weight, bias],
        vec![1, 1, 2, 2],
    );
    graph.add(
        UOpKind::Output { name: "out".into() },
        vec![conv],
        vec![1, 1, 2, 2],
    );
    let capture = graph.lower(LoweringTarget::Portable).unwrap();
    assert!(capture.kernels[0].source.contains("prism_conv2d"));
    let output = graph
        .execute_f32(&BTreeMap::from([
            (
                "x".into(),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            ),
            ("weight".into(), vec![1.0, 0.0, 0.0, 1.0]),
            ("bias".into(), vec![0.0]),
        ]))
        .unwrap();
    assert_eq!(output["out"], vec![6.0, 8.0, 12.0, 14.0]);
}

#[test]
fn conv2d_rejects_invalid_geometry_without_underflow() {
    let mut graph = TinyGraph::default();
    let x = graph.add(
        UOpKind::Input { name: "x".into() },
        vec![],
        vec![1, 1, 2, 2],
    );
    let weight = graph.add(
        UOpKind::Input {
            name: "weight".into(),
        },
        vec![],
        vec![1, 1, 5, 5],
    );
    let bias = graph.add(
        UOpKind::Input {
            name: "bias".into(),
        },
        vec![],
        vec![1],
    );
    graph.add(
        UOpKind::Conv2d {
            batch: 1,
            in_channels: 1,
            height: 2,
            width: 2,
            out_channels: 1,
            kernel_h: 5,
            kernel_w: 5,
            stride: 1,
            padding: 0,
        },
        vec![x, weight, bias],
        vec![1, 1, 0, 0],
    );
    assert!(matches!(
        graph.validate(),
        Err(GraphError::ShapeMismatch(_))
    ));
}
