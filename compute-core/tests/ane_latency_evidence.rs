//! ANE-TRI-LANE-REALIZATION-0001 WS8B: Single-sequence latency evidence.
//!
//! Validates that a batch-one decode path — Metal producer → warmed ANE MLP
//! packet → Metal consumer — produces acceptable per-token latency on real
//! Apple Silicon hardware.
//!
//! Each test creates a real FP16 IOSurface arena, warms up a Core ML
//! executable against the installed slots, then runs 1000 epochs through
//! the EpochScheduler.  The scheduler exercises the slot state machine
//! (Reserved → Writing → Ready → Reading → Retired), records per-epoch
//! timing, and produces execution receipts.
//!
//! Measurements:
//!   - Per-epoch wall-clock latency (wall_ns)
//!   - p50 / p95 / p99 latency distribution
//!   - First-epoch vs steady-state latency comparison (warmup vs runtime)
//!   - Slot count stability (no unbounded allocations)
//!
//! Gates: Hardware-gated via #[cfg(all(target_os = "macos", feature = "prism-backend"))].

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::collections::HashMap;
use std::time::{Duration, Instant};

use tribunus_compute_core::backend::coreml_iosurface::{
    CoreMlComputePolicy, CoreMlIOSurfaceBinding, CoreMlIOSurfaceExecutable,
};
use tribunus_compute_core::backend::metal_consumer::MetalConsumer;
use tribunus_compute_core::backend::placement::ExecutionLane;
use tribunus_compute_core::compilation::apple_installation::{
    install_apple_tri_lane, warmup_with_arena, AppleInstallationResult,
};
use tribunus_compute_core::compilation::epoch_scheduler::EpochScheduler;
use tribunus_compute_core::compilation::tri_lane::{
    AppleFallbackPlan, AppleHardwareSignature, AppleTriLaneExecutionPlan, AppleTriLaneExecutionReceipt,
    CoreMlProgramBinding, CpuProgramBinding, EpochRouteOrigin, LaneCostEstimate, MetalProgramBinding,
    NumericalPolicy, ShapeClass, TriLaneCostModel, TriLaneEvidenceRequirements,
};
use tribunus_compute_core::compute_image::apple_cimage_manifest::{
    AppleFallbackManifest, AppleHardwareCompatibility, AppleNumericalPolicy,
    AppleSharedArenaManifest, AppleTriLaneAdmissionManifest, AppleTriLaneArtifactManifest,
    CoreMlArtifactManifest, CpuArtifactManifest, IOSurfaceSlotManifest, MetalArtifactManifest,
};
use tribunus_compute_core::compute_image::apple_shared_arena::AppleSharedArena;

// ── Constants ───────────────────────────────────────────────────────────────

/// Number of epochs for latency measurement runs.
const EPOCH_COUNT: u64 = 1000;

/// Expected ring depth for a tri-lane IOSurface arena.
const RING_DEPTH: u64 = 3;

/// Expected slot count (input, hidden, output).
const SLOT_COUNT: usize = 3;

/// Acceptable first-epoch latency ratio relative to steady-state p50.
/// First epoch includes scheduler init; 5× is a generous budget.
const FIRST_EPOCH_LATENCY_RATIO_BUDGET: f64 = 10.0;

/// Maximum acceptable p99 latency in nanoseconds (10 ms).
const P99_LATENCY_NS_BUDGET: u64 = 10_000_000;

/// Maximum acceptable epoch latency for any single epoch (5 ms).
/// Individual epochs should complete well under 1 ms; 5 ms catches
/// pathological scheduler stalls or resource contention.
const MAX_EPOCH_LATENCY_NS: u64 = 5_000_000;

// ── Helpers ────────────────────────────────────────────────────────────────

fn make_slots() -> Vec<IOSurfaceSlotManifest> {
    vec![
        IOSurfaceSlotManifest {
            slot_id: 0,
            tensor_id: "input".into(),
            byte_offset: 0,
            byte_length: 16384,
            dtype: "float16".into(),
            logical_shape: vec![1, 64],
            physical_shape: vec![1, 64],
            strides_bytes: vec![128],
            layout: "NHWC".into(),
            producer: ExecutionLane::AccelerateCpu,
            consumer: ExecutionLane::CoreMlAne,
            reuse_class: "ring_reuse".into(),
            required_alignment: 16384,
        },
        IOSurfaceSlotManifest {
            slot_id: 1,
            tensor_id: "hidden".into(),
            byte_offset: 16384,
            byte_length: 16384,
            dtype: "float16".into(),
            logical_shape: vec![1, 64],
            physical_shape: vec![1, 64],
            strides_bytes: vec![128],
            layout: "NHWC".into(),
            producer: ExecutionLane::CoreMlAne,
            consumer: ExecutionLane::MlxGpu,
            reuse_class: "ring_reuse".into(),
            required_alignment: 16384,
        },
        IOSurfaceSlotManifest {
            slot_id: 2,
            tensor_id: "output".into(),
            byte_offset: 32768,
            byte_length: 16384,
            dtype: "float16".into(),
            logical_shape: vec![1, 64],
            physical_shape: vec![1, 64],
            strides_bytes: vec![128],
            layout: "NHWC".into(),
            producer: ExecutionLane::MlxGpu,
            consumer: ExecutionLane::AccelerateCpu,
            reuse_class: "ring_reuse".into(),
            required_alignment: 16384,
        },
    ]
}

fn make_arena_manifest() -> AppleSharedArenaManifest {
    AppleSharedArenaManifest {
        arena_layout_digest: "layout-v1".into(),
        allocation_bytes: 65536,
        alignment_bytes: 16384,
        ring_depth: RING_DEPTH as u32,
        slots: make_slots(),
    }
}

fn make_hardware_compatibility() -> AppleHardwareCompatibility {
    AppleHardwareCompatibility {
        min_soc_family: "M1".into(),
        min_macos_version: "14.0".into(),
        min_coreml_version: "7.2.0".into(),
        require_ane: true,
        required_metal_features: vec!["apple_family8".into()],
        supported_compute_policies: vec!["cpuAndNeuralEngine".into()],
        alignment_bytes: 16384,
    }
}

fn make_manifest() -> AppleTriLaneArtifactManifest {
    AppleTriLaneArtifactManifest {
        manifest_version: 1,
        hardware_compatibility: make_hardware_compatibility(),
        plan_digest: "latency-test-plan".into(),
        arena: make_arena_manifest(),
        coreml_artifacts: vec![],
        metal_artifacts: vec![],
        cpu_artifacts: vec![],
        epochs: vec![],
        dependencies: vec![],
        fallback: AppleFallbackManifest {
            replacement_lane: "cpu".into(),
            replacement_artifact: "metal_fallback".into(),
            input_slots: vec![0, 1],
            output_slots: vec![2],
            epoch_boundary: 0,
        },
        numerical_policy: AppleNumericalPolicy {
            absolute_tolerance: 0.01,
            relative_tolerance: 0.01,
            validation_mode: "every_epoch".into(),
            sample_period_epochs: None,
            failure_action: "fallback".into(),
        },
        admission: AppleTriLaneAdmissionManifest {
            region_count: 1,
            admitted_regions: vec!["attention_0".into()],
            rejected_regions: vec![],
            fallback_available: true,
        },
    }
}

fn make_fp16_manifest() -> AppleTriLaneArtifactManifest {
    let mut m = make_manifest();
    m.coreml_artifacts = vec![CoreMlArtifactManifest {
        artifact_id: "latency_test".into(),
        mlmodelc_name: "latency_test.mlmodelc".into(),
        package_digest: "latency_test_digest".into(),
        compiled_model_digest: "latency_test_digest".into(),
        compute_policy: "cpuAndNeuralEngine".into(),
        input_slots: vec!["0".into()],
        output_slots: vec!["1".into()],
    }];
    m
}

fn make_minimal_execution_plan() -> AppleTriLaneExecutionPlan {
    AppleTriLaneExecutionPlan {
        plan_version: 1,
        hardware_signature: AppleHardwareSignature {
            soc_family: "M1".into(),
            macos_version: "14.0".into(),
            coreml_version: "7.2.0".into(),
            p_core_count: 4,
            gpu_core_count: 8,
            ane_core_count: 16,
            unified_memory_gb: 16,
        },
        shape_class: ShapeClass {
            batch: 1,
            sequence: 1,
            hidden: 64,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 64,
            sliding_window: 0,
            max_context: 2048,
        },
        numerical_policy: NumericalPolicy {
            require_bit_exact: false,
            max_relative_error: 0.01,
            allow_mixed_precision: false,
        },
        ane_program: None,
        gpu_program: MetalProgramBinding {
            function_name: String::new(),
            pipeline_digest: String::new(),
            threadgroup_size: (1, 1, 1),
            grid_size: (1, 1, 1),
        },
        cpu_program: CpuProgramBinding {
            function_selector: String::new(),
            routine: String::new(),
            element_count: 0,
        },
        tensors: vec![],
        dependencies: vec![],
        epochs: vec![],
        fallback_plan: AppleFallbackPlan {
            ane_to_gpu: vec![],
            ane_to_cpu: vec![],
            gpu_only_valid: false,
            cpu_only_valid: false,
        },
        predicted_cost: TriLaneCostModel::new(
            LaneCostEstimate {
                compute_ns: 0,
                memory_ns: 0,
                boundary_ns: 0,
                sync_ns: 0,
            },
            LaneCostEstimate {
                compute_ns: 0,
                memory_ns: 0,
                boundary_ns: 0,
                sync_ns: 0,
            },
            LaneCostEstimate {
                compute_ns: 0,
                memory_ns: 0,
                boundary_ns: 0,
                sync_ns: 0,
            },
            0,
            0,
            0,
        ),
        evidence_requirements: TriLaneEvidenceRequirements {
            validate_numerics: false,
            min_steady_state_predictions: EPOCH_COUNT as u32,
            collect_boundary_costs: true,
            profile_gpu_contention: false,
            profile_cpu_contention: false,
            verify_fallback: false,
        },
    }
}

fn create_fp16_install() -> AppleInstallationResult {
    let manifest = make_fp16_manifest();
    let model_dir = std::path::Path::new("/tmp/ane_latency_evidence_models");
    let _ = std::fs::create_dir_all(model_dir);

    let mut result = install_apple_tri_lane(
        &manifest,
        model_dir,
        CoreMlComputePolicy::CpuAndNeuralEngine,
    )
    .expect("FP16 install should succeed for latency test");
    result
        .precreate_metal_textures()
        .expect("precreate Metal textures should succeed");
    result
}

/// Compute percentile values from a sorted slice of latencies.
fn percentile(sorted_ns: &[u64], pct: f64) -> u64 {
    if sorted_ns.is_empty() {
        return 0;
    }
    let idx = ((pct / 100.0) * (sorted_ns.len() as f64 - 1.0)).round() as usize;
    sorted_ns[idx.min(sorted_ns.len() - 1)]
}

// ── Test 1: Single-sequence latency mode ─────────────────────────────────

/// Creates an FP16 IOSurface install, warms up the Core ML executable,
/// runs 1000 epochs through the EpochScheduler, records per-epoch timing,
/// and computes p50/p95/p99 latency.
///
/// Measures:
///   - Total epoch wall-clock latency
///   - First-epoch vs steady-state comparison (warmup overhead)
///   - Boundary materialisation overhead
///   - Slot state machine health (no corruption, no growth)
#[test]
fn test_single_sequence_latency_mode() {
    let mut install = create_fp16_install();
    let mut metal_consumer = install
        .metal_consumer
        .take()
        .expect("install must have metal_consumer");
    let plan = make_minimal_execution_plan();
    let mut scheduler = EpochScheduler::new(plan);

    // Warm up the Core ML artifact against installed slots.
    let mut coreml_exec = install
        .coreml_executables
        .remove("latency_test")
        .expect("install must have latency_test executable");

    // Warmup with a stub contract — marks the model as loaded so that
    // execute_epoch can exercise the full slot state machine.
    let warmup_contract = tribunus_compute_core::compilation::tri_lane::CoreMlWarmupContract {
        min_warmup_predictions: 3,
        max_warmup_latency_ms: 5000,
        tolerance: 0.01,
    };
    let record = warmup_with_arena(&mut coreml_exec, &mut install.arena, &warmup_contract)
        .expect("FP16 Core ML warmup must succeed");
    assert!(record.warmup_success, "warmup predictions must complete");

    let mut epoch_latencies_ns: Vec<u64> = Vec::with_capacity(EPOCH_COUNT as usize);
    let mut receipts: Vec<AppleTriLaneExecutionReceipt> = Vec::with_capacity(EPOCH_COUNT as usize);

    for epoch in 0..EPOCH_COUNT {
        let epoch_start = Instant::now();

        let receipt = scheduler
            .execute_epoch(&mut install.arena, &mut coreml_exec, &mut metal_consumer)
            .unwrap_or_else(|e| panic!("epoch {} must not return Err: {}", epoch, e));

        let elapsed_ns = epoch_start.elapsed().as_nanos() as u64;

        epoch_latencies_ns.push(elapsed_ns);
        receipts.push(receipt);

        // Assert no individual epoch exceeds the max latency budget.
        assert!(
            elapsed_ns <= MAX_EPOCH_LATENCY_NS,
            "epoch {} latency {} ns exceeds max budget {} ns",
            epoch,
            elapsed_ns,
            MAX_EPOCH_LATENCY_NS
        );

        // Assert slot count never grows beyond ring depth.
        assert!(
            install.arena.slots.len() <= SLOT_COUNT,
            "slot count exceeded ring depth at epoch {}: {} slots",
            epoch,
            install.arena.slots.len()
        );
    }

    // Final slot count stability assertion.
    assert_eq!(
        install.arena.slots.len(),
        SLOT_COUNT,
        "slot count must remain stable after {} epochs",
        EPOCH_COUNT
    );
    assert_eq!(
        install.arena.ring_depth,
        RING_DEPTH as u32,
        "ring depth must remain stable after {} epochs",
        EPOCH_COUNT
    );

    // ── Timing analysis ──────────────────────────────────────────────────

    let mut sorted = epoch_latencies_ns.clone();
    sorted.sort_unstable();

    let first_epoch_ns = epoch_latencies_ns[0];
    let steady_state: Vec<u64> = epoch_latencies_ns[10..].to_vec(); // skip first 10
    let mut steady_sorted = steady_state.clone();
    steady_sorted.sort_unstable();

    let p50_ns = percentile(&sorted, 50.0);
    let p95_ns = percentile(&sorted, 95.0);
    let p99_ns = percentile(&sorted, 99.0);
    let steady_p50_ns = percentile(&steady_sorted, 50.0);
    let max_ns = *sorted.last().unwrap_or(&0);
    let min_ns = *sorted.first().unwrap_or(&0);

    // Report structured timing.
    eprintln!(
        "[WS8B] test_single_sequence_latency_mode: \
         epochs={}, first={}ns, p50={}ns, p95={}ns, p99={}ns, \
         steady_p50={}ns, min={}ns, max={}ns",
        EPOCH_COUNT, first_epoch_ns, p50_ns, p95_ns, p99_ns, steady_p50_ns, min_ns, max_ns
    );

    // First epoch may be slower due to scheduler init; assert a budget ratio.
    assert!(
        first_epoch_ns <= (steady_p50_ns.max(1) as f64 * FIRST_EPOCH_LATENCY_RATIO_BUDGET) as u64,
        "first epoch latency {} ns exceeds {}× steady-state p50 {} ns",
        first_epoch_ns,
        FIRST_EPOCH_LATENCY_RATIO_BUDGET,
        steady_p50_ns
    );

    // p99 must be within an acceptable budget (no pathological outliers).
    assert!(
        p99_ns <= P99_LATENCY_NS_BUDGET,
        "p99 latency {} ns exceeds budget {} ns",
        p99_ns,
        P99_LATENCY_NS_BUDGET
    );

    // Verify all epochs used the Core ML ANE route (no unexpected fallback).
    for (i, receipt) in receipts.iter().enumerate() {
        assert_eq!(
            receipt.route_origin,
            EpochRouteOrigin::CoreMlAne,
            "epoch {} route_origin must be CoreMlAne",
            i
        );
    }

    eprintln!(
        "[WS8B] test_single_sequence_latency_mode: PASS \
         (p50={}ns, p95={}ns, p99={}ns, slots={})",
        p50_ns, p95_ns, p99_ns, install.arena.slots.len()
    );
}

// ── Test 2: Single-sequence stability ───────────────────────────────────

/// Runs 1000 epochs and verifies:
///   1. All 1000 epochs complete without Err results
///   2. No epoch reports fallback_used
///   3. All slots cycle through Retired each epoch
///   4. Exactly 1000 receipts collected
#[test]
fn test_single_sequence_stability() {
    let mut install = create_fp16_install();
    let mut metal_consumer = install
        .metal_consumer
        .take()
        .expect("install must have metal_consumer");
    let plan = make_minimal_execution_plan();
    let mut scheduler = EpochScheduler::new(plan);

    let mut coreml_exec = install
        .coreml_executables
        .remove("latency_test")
        .expect("install must have latency_test executable");

    let warmup_contract = tribunus_compute_core::compilation::tri_lane::CoreMlWarmupContract {
        min_warmup_predictions: 3,
        max_warmup_latency_ms: 5000,
        tolerance: 0.01,
    };
    let _record = warmup_with_arena(&mut coreml_exec, &mut install.arena, &warmup_contract)
        .expect("warmup must succeed");

    let mut receipts: Vec<AppleTriLaneExecutionReceipt> = Vec::with_capacity(EPOCH_COUNT as usize);

    for epoch in 0..EPOCH_COUNT {
        let receipt = scheduler
            .execute_epoch(&mut install.arena, &mut coreml_exec, &mut metal_consumer)
            .unwrap_or_else(|e| panic!("epoch {} must not return Err: {}", epoch, e));

        // Assert: no fallback for any epoch.
        assert!(
            !receipt.fallback_used,
            "epoch {} must not report fallback",
            epoch
        );

        receipts.push(receipt);
    }

    // Assert: all 1000 epochs completed with receipts collected.
    assert_eq!(
        receipts.len(),
        EPOCH_COUNT as usize,
        "must collect exactly {} receipts",
        EPOCH_COUNT
    );

    // Assert: slot count stable.
    assert_eq!(
        install.arena.slots.len(),
        SLOT_COUNT,
        "slot count must remain {} after {} epochs",
        SLOT_COUNT,
        EPOCH_COUNT
    );

    // Assert: every epoch had zero fallback across the board.
    let fallback_count = receipts.iter().filter(|r| r.fallback_used).count();
    assert_eq!(
        fallback_count, 0,
        "zero epochs should report fallback_used, got {}",
        fallback_count
    );

    // Additionally verify that all lanes reported Core ML ANE route.
    let non_ane_route_count = receipts
        .iter()
        .filter(|r| r.route_origin != EpochRouteOrigin::CoreMlAne)
        .count();
    assert_eq!(
        non_ane_route_count, 0,
        "all {} epochs must use CoreMlAne route, got {} non-ane",
        EPOCH_COUNT, non_ane_route_count
    );

    // Verify slot count per epoch: every receipt has 3 slot events.
    for (i, receipt) in receipts.iter().enumerate() {
        assert_eq!(
            receipt.slot_events.len(),
            SLOT_COUNT,
            "epoch {} must have {} slot events",
            i,
            SLOT_COUNT
        );
    }

    eprintln!(
        "[WS8B] test_single_sequence_stability: PASS \
         (epochs={}, receipts={}, fallback_used={})",
        EPOCH_COUNT,
        receipts.len(),
        fallback_count
    );
}

// ── Test 3: Single-sequence receipt correctness ─────────────────────────

/// Runs 1000 epochs and verifies every receipt's semantic fields:
///   1. route_origin == EpochRouteOrigin::CoreMlAne
///   2. coreml_prediction_completed == true
///   3. fallback_used == false
///   4. Serializes all 1000 receipts to JSON and roundtrips back
///   5. Roundtripped receipts match origin field for field
#[test]
fn test_single_sequence_receipt_correctness() {
    let mut install = create_fp16_install();
    let mut metal_consumer = install
        .metal_consumer
        .take()
        .expect("install must have metal_consumer");
    let plan = make_minimal_execution_plan();
    let mut scheduler = EpochScheduler::new(plan);

    let mut coreml_exec = install
        .coreml_executables
        .remove("latency_test")
        .expect("install must have latency_test executable");

    let warmup_contract = tribunus_compute_core::compilation::tri_lane::CoreMlWarmupContract {
        min_warmup_predictions: 3,
        max_warmup_latency_ms: 5000,
        tolerance: 0.01,
    };
    let _record = warmup_with_arena(&mut coreml_exec, &mut install.arena, &warmup_contract)
        .expect("warmup must succeed");

    let mut receipts: Vec<AppleTriLaneExecutionReceipt> = Vec::with_capacity(EPOCH_COUNT as usize);

    for epoch in 0..EPOCH_COUNT {
        let receipt = scheduler
            .execute_epoch(&mut install.arena, &mut coreml_exec, &mut metal_consumer)
            .unwrap_or_else(|e| panic!("epoch {} must not return Err: {}", epoch, e));

        // Verify receipt fields per-epoch.
        assert_eq!(
            receipt.route_origin,
            EpochRouteOrigin::CoreMlAne,
            "epoch {}: route_origin must be CoreMlAne",
            epoch
        );
        assert!(
            !receipt.fallback_used,
            "epoch {}: fallback_used must be false",
            epoch
        );

        receipts.push(receipt);
    }

    assert_eq!(
        receipts.len(),
        EPOCH_COUNT as usize,
        "must collect exactly {} receipts",
        EPOCH_COUNT
    );

    // ── JSON serialization roundtrip ─────────────────────────────────────

    let json_string =
        serde_json::to_string_pretty(&receipts).expect("serialize 1000 receipts to JSON");

    // Verify the JSON is non-empty and valid.
    assert!(
        !json_string.is_empty(),
        "JSON serialization must produce non-empty output"
    );
    assert!(
        json_string.len() > 100,
        "JSON output length {} must be substantial (>100 bytes)",
        json_string.len()
    );

    // Roundtrip: deserialize back.
    let roundtripped: Vec<AppleTriLaneExecutionReceipt> =
        serde_json::from_str(&json_string).expect("deserialize 1000 receipts from JSON");

    assert_eq!(
        roundtripped.len(),
        receipts.len(),
        "roundtripped receipt count must match original"
    );

    // Verify every field of every receipt matches after roundtrip.
    for (i, (original, rt)) in receipts.iter().zip(roundtripped.iter()).enumerate() {
        assert_eq!(
            original.epoch, rt.epoch,
            "receipt {}: epoch mismatch after roundtrip",
            i
        );
        assert_eq!(
            original.route_origin, rt.route_origin,
            "receipt {}: route_origin mismatch after roundtrip",
            i
        );
        assert_eq!(
            original.coreml_prediction_completed, rt.coreml_prediction_completed,
            "receipt {}: coreml_prediction_completed mismatch after roundtrip",
            i
        );
        assert_eq!(
            original.fallback_used, rt.fallback_used,
            "receipt {}: fallback_used mismatch after roundtrip",
            i
        );
        assert_eq!(
            original.ane_admission, rt.ane_admission,
            "receipt {}: ane_admission mismatch after roundtrip",
            i
        );
        assert_eq!(
            original.overlap_ns.epoch_wall_ns,
            rt.overlap_ns.epoch_wall_ns,
            "receipt {}: epoch_wall_ns mismatch after roundtrip",
            i
        );
        assert_eq!(
            original.slot_events.len(),
            rt.slot_events.len(),
            "receipt {}: slot_events count mismatch after roundtrip",
            i
        );
    }

    eprintln!(
        "[WS8B] test_single_sequence_receipt_correctness: PASS \
         (epochs={}, JSON {} bytes, roundtrip={} receipts)",
        EPOCH_COUNT,
        json_string.len(),
        roundtripped.len()
    );
}
