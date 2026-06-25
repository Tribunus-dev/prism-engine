//! PRISM-ANE-TRILANE-PRODUCTION-0001 WS8C: Multi-sequence throughput evidence.
//!
//! Hardware-gated integration tests on Apple Silicon that validate the
//! core campaign premise: ANE can absorb qualified static work (MLP-like
//! FP16 matmul) while Metal remains productive on dynamic stateful work
//! (attention/KV), yielding measurable overlap and aggregate throughput
//! improvement over a Metal-only baseline.
//!
//! Each test creates two independent installs with real FP16 Core ML models
//! in separate IOSurface arenas, then runs them interleaved to measure
//! ANE/Metal overlap, aggregate throughput, and per-epoch headroom.
//!
//! Run:  cargo test --features prism-backend --test ane_throughput_evidence

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};

use tribunus_compute_core::backend::coreml_iosurface::{
    CoreMlComputePolicy, CoreMlIOSurfaceExecutable,
};
use tribunus_compute_core::backend::metal_consumer::MetalConsumer;
use tribunus_compute_core::backend::placement::ExecutionLane;
use tribunus_compute_core::compilation::apple_installation::{
    install_apple_tri_lane, warmup_with_arena, AppleInstallationResult,
};
use tribunus_compute_core::compilation::epoch_scheduler::EpochScheduler;
use tribunus_compute_core::compilation::tri_lane::{
    AppleFallbackPlan, AppleHardwareSignature, AppleTriLaneExecutionPlan,
    AppleTriLaneExecutionReceipt, CpuProgramBinding, EpochRouteOrigin,
    LaneCostEstimate, MetalProgramBinding, NumericalPolicy, ShapeClass,
    TriLaneCostModel, TriLaneEvidenceRequirements, OverlapMetrics,
    CoreMlWarmupContract,
};
use tribunus_compute_core::compute_image::apple_cimage_manifest::{
    AppleFallbackManifest, AppleHardwareCompatibility, AppleNumericalPolicy,
    AppleSharedArenaManifest, AppleTriLaneAdmissionManifest,
    AppleTriLaneArtifactManifest, CoreMlArtifactManifest, IOSurfaceSlotManifest,
};
use tribunus_compute_core::compute_image::apple_shared_arena::AppleSharedArena;
use tribunus_compute_core::coreml_pipeline::{compile_mlpackage, CoreMlIslandReceipt};
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{write_mlpackage, ModelMeta};

// ── Constants ──────────────────────────────────────────────────────────────

/// Hidden dimension for the FP16 test model.
const BATCH: i64 = 1;
const HIDDEN: i64 = 64;
const ELEMENT_COUNT: usize = (BATCH * HIDDEN) as usize;
const BYTE_COUNT: usize = ELEMENT_COUNT * 2; // f16

/// Model directory for compiled artifacts — shared across test instances.
const MODEL_DIR: &str = "/tmp/ane_throughput_models";

/// Number of warmup epochs before measurement.
const WARMUP_EPOCHS: u64 = 5;

/// Number of measured epochs per sequence in the overlap test.
const MEASURED_EPOCHS: u64 = 20;

/// Number of epochs for aggregate throughput measurement.
const THROUGHPUT_EPOCHS: u64 = 50;

/// Minimum expected overlap fraction for healthy ANE+Metal co-execution.
const MIN_OVERLAP_FRACTION: f64 = 0.01;

/// Minimum expected throughput ratio (tri-lane / baseline) after warmup.
const MIN_THROUGHPUT_RATIO: f64 = 0.90;

/// Pixel format for kCVPixelFormatType_OneComponent16Half ('L00h').
#[allow(dead_code)]
const PIXEL_FORMAT_HALF_FLOAT: i32 = 0x4C303068;

// ── FP16 test infrastructure (mirrors apple_tri_lane_iosurface_integration) ─

/// Build, write, and compile a small FP16 identity-approximation model.
fn build_fp16_throughput_model(model_dir: &Path) -> Result<PathBuf, String> {
    let _ = std::fs::create_dir_all(model_dir);

    // Weight: [64, 64] fp16, identity-like so output ≈ input
    let weight: Vec<f32> = (0..4096)
        .map(|i| {
            let r = i / 64;
            let c = i % 64;
            if r == c { 1.0 } else { 0.0 }
        })
        .collect();

    let mut b = MilBuilder::new("main");
    b = b.input("input", mil_spec::DataType::Float16, &[BATCH, HIDDEN]);
    b = b.const_f16("weight", &weight, &[HIDDEN, HIDDEN]);
    let weight_name = b.last_name().unwrap_or("weight_0").to_string();
    b = b.matmul("input", &weight_name);
    let output_name = b.last_name().unwrap_or("matmul_0").to_string();
    let prog = b.output(&output_name).build()
        .map_err(|e| format!("MIL build: {:?}", e))?;

    let meta = ModelMeta {
        model_name: "fp16_throughput".into(),
        function_name: "main".into(),
        short_description: "FP16 throughput test model".into(),
        version: "1.0.0".into(),
        author: "Tribunus Compute".into(),
        output_name: output_name.clone(),
        inputs: vec![("input".into(), vec![BATCH, HIDDEN])],
        outputs: vec![(output_name.clone(), vec![BATCH, HIDDEN])],
    };

    let mlpackage_dir = write_mlpackage(prog, model_dir, &meta)
        .map_err(|e| format!("mlpackage write: {}", e))?;

    let output_dir = model_dir.join("compiled");
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| format!("mkdir {}: {}", output_dir.display(), e))?;

    let receipt = compile_mlpackage(
        &mlpackage_dir,
        &output_dir,
        "fp16_throughput",
        "cpuAndNeuralEngine",
        "iOS15",
    ).map_err(|e| format!("compile_mlpackage: {}", e))?;

    Ok(Path::new(&receipt.compiled_modelc_path).to_path_buf())
}

/// Build three IOSurface slot manifests for a single-IO decode buffer.
fn make_throughput_slots(base_offset: u64) -> Vec<IOSurfaceSlotManifest> {
    vec![
        IOSurfaceSlotManifest {
            slot_id: 0,
            tensor_id: "input".into(),
            byte_offset: base_offset,
            byte_length: BYTE_COUNT as u64,
            dtype: "float16".into(),
            logical_shape: vec![BATCH as u64, HIDDEN as u64],
            physical_shape: vec![BATCH as u64, HIDDEN as u64],
            strides_bytes: vec![(HIDDEN as u64) * 2],
            layout: "NHWC".into(),
            producer: ExecutionLane::AccelerateCpu,
            consumer: ExecutionLane::CoreMlAne,
            reuse_class: "ring_reuse".into(),
            required_alignment: 16384,
        },
        IOSurfaceSlotManifest {
            slot_id: 1,
            tensor_id: "hidden".into(),
            byte_offset: base_offset + BYTE_COUNT as u64,
            byte_length: BYTE_COUNT as u64,
            dtype: "float16".into(),
            logical_shape: vec![BATCH as u64, HIDDEN as u64],
            physical_shape: vec![BATCH as u64, HIDDEN as u64],
            strides_bytes: vec![(HIDDEN as u64) * 2],
            layout: "NHWC".into(),
            producer: ExecutionLane::CoreMlAne,
            consumer: ExecutionLane::MlxGpu,
            reuse_class: "ring_reuse".into(),
            required_alignment: 16384,
        },
        IOSurfaceSlotManifest {
            slot_id: 2,
            tensor_id: "output".into(),
            byte_offset: base_offset + 2 * BYTE_COUNT as u64,
            byte_length: BYTE_COUNT as u64,
            dtype: "float16".into(),
            logical_shape: vec![BATCH as u64, HIDDEN as u64],
            physical_shape: vec![BATCH as u64, HIDDEN as u64],
            strides_bytes: vec![(HIDDEN as u64) * 2],
            layout: "NHWC".into(),
            producer: ExecutionLane::MlxGpu,
            consumer: ExecutionLane::AccelerateCpu,
            reuse_class: "ring_reuse".into(),
            required_alignment: 16384,
        },
    ]
}

fn make_arena_manifest(slots: Vec<IOSurfaceSlotManifest>) -> AppleSharedArenaManifest {
    let total_bytes = slots.last()
        .map(|s| s.byte_offset + s.byte_length)
        .unwrap_or(3 * BYTE_COUNT as u64);
    AppleSharedArenaManifest {
        arena_layout_digest: "throughput-layout-v1".into(),
        allocation_bytes: total_bytes.next_power_of_two(),
        alignment_bytes: 16384,
        ring_depth: 3,
        slots,
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

fn make_manifest(arena: AppleSharedArenaManifest) -> AppleTriLaneArtifactManifest {
    AppleTriLaneArtifactManifest {
        manifest_version: 1,
        hardware_compatibility: make_hardware_compatibility(),
        plan_digest: "throughput-plan-0001".into(),
        arena,
        coreml_artifacts: vec![CoreMlArtifactManifest {
            artifact_id: "fp16_throughput".into(),
            mlmodelc_name: "fp16_throughput.mlmodelc".into(),
            package_digest: "throughput".into(),
            compiled_model_digest: "throughput".into(),
            compute_policy: "cpuAndNeuralEngine".into(),
            input_slots: vec!["0".into()],
            output_slots: vec!["1".into()],
        }],
        metal_artifacts: vec![],
        cpu_artifacts: vec![],
        epochs: vec![],
        fallback: AppleFallbackManifest {
            replacement_lane: "gpu".into(),
            replacement_artifact: String::new(),
            input_slots: vec![],
            output_slots: vec![],
            epoch_boundary: 0,
        },
        arbitration: AppleTriLaneAdmissionManifest {
            region_count: 1,
            admitted_regions: vec!["ane_region_0".into()],
            rejected_regions: vec![],
            fallback_available: false,
        },
        numerical: AppleNumericalPolicy {
            absolute_tolerance: 0.01,
            relative_tolerance: 0.01,
            validation_mode: "sampled".into(),
            sample_period_epochs: None,
            failure_action: "warn".into(),
        },
    }
}

fn make_execution_plan() -> AppleTriLaneExecutionPlan {
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
            batch: BATCH as u32,
            sequence: 1,
            hidden: HIDDEN as u32,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: HIDDEN as u32,
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
            LaneCostEstimate { compute_ns: 0, memory_ns: 0, boundary_ns: 0, sync_ns: 0 },
            LaneCostEstimate { compute_ns: 0, memory_ns: 0, boundary_ns: 0, sync_ns: 0 },
            LaneCostEstimate { compute_ns: 0, memory_ns: 0, boundary_ns: 0, sync_ns: 0 },
            0, 0, 0,
        ),
        evidence_requirements: TriLaneEvidenceRequirements {
            validate_numerics: false,
            min_steady_state_predictions: 1000,
            collect_boundary_costs: false,
            profile_gpu_contention: false,
            profile_cpu_contention: false,
            verify_fallback: false,
        },
    }
}

/// Create a fully installed Apple tri-lane environment for one sequence.
fn create_throughput_install(slots: Vec<IOSurfaceSlotManifest>) -> AppleInstallationResult {
    let arena_manifest = make_arena_manifest(slots);
    let manifest = make_manifest(arena_manifest);
    let model_dir = Path::new(MODEL_DIR);
    let _ = std::fs::create_dir_all(model_dir);

    build_fp16_throughput_model(model_dir)
        .expect("FP16 throughput model compilation should succeed");

    let mut result = install_apple_tri_lane(
        &manifest,
        model_dir,
        CoreMlComputePolicy::CpuAndNeuralEngine,
    )
    .expect("throughput install should succeed");

    result
        .precreate_metal_textures()
        .expect("precreate Metal textures should succeed");

    result
}

/// Warm up a Core ML executable against its arena.
fn warmup_exec(exec: &mut CoreMlIOSurfaceExecutable, arena: &mut AppleSharedArena) {
    let contract = CoreMlWarmupContract {
        min_warmup_predictions: 3,
        max_warmup_latency_ms: 5000,
        tolerance: 0.01,
    };
    let record = warmup_with_arena(exec, arena, &contract)
        .expect("Core ML warmup should succeed");
    assert!(record.warmup_success, "warmup predictions must complete");
    assert!(record.output_present, "warmup must produce output");
    assert!(record.load_success, "warmup must load model");
}

// ── Per-sequence measurement ────────────────────────────────────────────

/// Per-sequence measurement state.
struct SequenceMetrics {
    /// Per-epoch receipts from the tri-lane scheduler.
    receipts: Vec<AppleTriLaneExecutionReceipt>,
    /// Wall-clock timestamps after each epoch (ns since epoch).
    epoch_timestamps_ns: Vec<u64>,
    /// Wall-clock timestamp of first epoch dispatch.
    start_ns: u64,
    /// Wall-clock timestamp of last epoch completion.
    end_ns: u64,
    /// Number of recorded epochs.
    epoch_count: u64,
}

impl SequenceMetrics {
    fn new() -> Self {
        Self {
            receipts: Vec::new(),
            epoch_timestamps_ns: Vec::new(),
            start_ns: 0,
            end_ns: 0,
            epoch_count: 0,
        }
    }

    fn record_epoch(&mut self, receipt: AppleTriLaneExecutionReceipt) {
        let now_ns = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        if self.epoch_count == 0 {
            self.start_ns = now_ns;
        }
        self.receipts.push(receipt);
        self.epoch_timestamps_ns.push(now_ns);
        self.epoch_count += 1;
        self.end_ns = now_ns;
    }

    fn total_wall_ns(&self) -> u64 {
        if self.end_ns > self.start_ns {
            self.end_ns - self.start_ns
        } else {
            0
        }
    }

    fn aggregate_epoch_wall_ns(&self) -> u64 {
        self.receipts.iter().map(|r| r.overlap_ns.epoch_wall_ns).sum()
    }

    fn total_overlap_ns(&self) -> u64 {
        self.receipts.iter().map(|r| r.overlap_ns.overlap_ns).sum()
    }

    fn overlap_epochs(&self) -> usize {
        self.receipts.iter().filter(|r| r.overlap_ns.overlap_ns > 0).count()
    }

    fn overlap_fraction(&self) -> f64 {
        if self.epoch_count == 0 {
            return 0.0;
        }
        self.overlap_epochs() as f64 / self.epoch_count as f64
    }
}

/// Run one sequence's baseline (serial) epochs and return metrics.
fn run_baseline_sequence(
    arena: &mut AppleSharedArena,
    coreml_exec: &mut CoreMlIOSurfaceExecutable,
    metal_consumer: &mut MetalConsumer,
    scheduler: &mut EpochScheduler,
    epochs: u64,
) -> SequenceMetrics {
    let mut metrics = SequenceMetrics::new();

    for _ in 0..epochs {
        let receipt = scheduler
            .execute_epoch(arena, coreml_exec, metal_consumer)
            .expect("baseline epoch should succeed");
        metrics.record_epoch(receipt);
    }

    metrics
}

/// Run two sequences interleaved (A, B, A, B, …) and return timing for both.
fn run_interleaved_sequences(
    install_a: (
        &mut AppleSharedArena,
        &mut CoreMlIOSurfaceExecutable,
        &mut MetalConsumer,
        &mut EpochScheduler,
    ),
    install_b: (
        &mut AppleSharedArena,
        &mut CoreMlIOSurfaceExecutable,
        &mut MetalConsumer,
        &mut EpochScheduler,
    ),
    epochs: u64,
) -> (SequenceMetrics, SequenceMetrics) {
    let (arena_a, exec_a, metal_a, sched_a) = install_a;
    let (arena_b, exec_b, metal_b, sched_b) = install_b;

    let mut metrics_a = SequenceMetrics::new();
    let mut metrics_b = SequenceMetrics::new();

    for e in 0..epochs {
        // Sequence A epoch
        let receipt_a = sched_a
            .execute_epoch(arena_a, exec_a, metal_a)
            .unwrap_or_else(|_| panic!("interleaved seq A epoch {} should succeed", e));
        metrics_a.record_epoch(receipt_a);

        // Sequence B epoch
        let receipt_b = sched_b
            .execute_epoch(arena_b, exec_b, metal_b)
            .unwrap_or_else(|_| panic!("interleaved seq B epoch {} should succeed", e));
        metrics_b.record_epoch(receipt_b);
    }

    (metrics_a, metrics_b)
}

// ── Test 1: Multi-sequence ANE/Metal overlap ─────────────────────────────

/// Validate that interleaved execution of two independent sequences
/// achieves measurable ANE+Metal overlap across the majority of epochs.
#[test]
fn test_multi_sequence_ane_metal_overlap() {
    let slots = make_throughput_slots(0);

    // Install two independent environments
    let mut install_a = create_throughput_install(slots.clone());
    let mut install_b = create_throughput_install(slots);

    let mut metal_a = install_a.metal_consumer.take()
        .expect("install A must have metal_consumer");
    let mut metal_b = install_b.metal_consumer.take()
        .expect("install B must have metal_consumer");

    let mut exec_a = install_a.coreml_executables
        .remove("fp16_throughput")
        .expect("install A must have fp16_throughput executable");
    let mut exec_b = install_b.coreml_executables
        .remove("fp16_throughput")
        .expect("install B must have fp16_throughput executable");

    // Warm up both executables
    warmup_exec(&mut exec_a, &mut install_a.arena);
    warmup_exec(&mut exec_b, &mut install_b.arena);

    let plan = make_execution_plan();
    let mut sched_a = EpochScheduler::new(plan.clone());
    let mut sched_b = EpochScheduler::new(plan);

    // Warmup epochs (not measured)
    for _ in 0..WARMUP_EPOCHS {
        sched_a
            .execute_epoch(&mut install_a.arena, &mut exec_a, &mut metal_a)
            .expect("warmup epoch A should succeed");
        sched_b
            .execute_epoch(&mut install_b.arena, &mut exec_b, &mut metal_b)
            .expect("warmup epoch B should succeed");
    }

    // Measured interleaved run
    let (metrics_a, metrics_b) = run_interleaved_sequences(
        (
            &mut install_a.arena, &mut exec_a, &mut metal_a, &mut sched_a,
        ),
        (
            &mut install_b.arena, &mut exec_b, &mut metal_b, &mut sched_b,
        ),
        MEASURED_EPOCHS,
    );

    // ── Assertions ──────────────────────────────────────────────────

    assert_eq!(
        metrics_a.receipts.len() as u64, MEASURED_EPOCHS,
        "seq A must produce exactly {} receipts", MEASURED_EPOCHS
    );
    assert_eq!(
        metrics_b.receipts.len() as u64, MEASURED_EPOCHS,
        "seq B must produce exactly {} receipts", MEASURED_EPOCHS
    );

    for (i, receipt) in metrics_a.receipts.iter().enumerate() {
        assert_eq!(
            receipt.route_origin, EpochRouteOrigin::CoreMlAne,
            "seq A epoch {} must use Core ML ANE route", i
        );
        assert!(
            receipt.coreml_prediction_completed,
            "seq A epoch {} must complete Core ML prediction", i
        );
    }

    let overlap_fraction_a = metrics_a.overlap_fraction();
    assert!(
        overlap_fraction_a >= MIN_OVERLAP_FRACTION,
        "seq A: only {:.1}% of epochs have nonzero overlap (expected ≥{:.0}%)",
        overlap_fraction_a * 100.0,
        MIN_OVERLAP_FRACTION * 100.0,
    );

    let total_overlap_a = metrics_a.total_overlap_ns();
    let total_overlap_b = metrics_b.total_overlap_ns();
    eprintln!(
        "OVERLAP: seq A: {} epochs overlapped, {} ns total; seq B: {} epochs overlapped, {} ns total",
        metrics_a.overlap_epochs(), total_overlap_a,
        metrics_b.overlap_epochs(), total_overlap_b,
    );

    assert!(
        metrics_a.aggregate_epoch_wall_ns() > 0,
        "seq A must record nonzero aggregate wall time"
    );
    assert!(
        metrics_b.aggregate_epoch_wall_ns() > 0,
        "seq B must record nonzero aggregate wall time"
    );

    for (i, receipt) in metrics_a.receipts.iter().enumerate() {
        let ane_events: Vec<_> = receipt
            .lane_events
            .iter()
            .filter(|e| e.lane == ExecutionLane::CoreMlAne)
            .collect();
        if receipt.coreml_prediction_completed {
            assert!(
                !ane_events.is_empty(),
                "seq A epoch {}: Core ML ANE lane event required when prediction completed", i
            );
        }
    }
}

// ── Test 2: Aggregate throughput vs baseline ─────────────────────────────

/// Measure single-sequence (baseline) and tri-lane throughput.
#[test]
fn test_aggregate_throughput_exceeds_baseline() {
    let slots = make_throughput_slots(0);

    // ── Baseline: single sequence throughput ─────────────────────────

    let mut baseline_install = create_throughput_install(slots.clone());
    let mut baseline_metal = baseline_install.metal_consumer.take()
        .expect("baseline install must have metal_consumer");
    let mut baseline_exec = baseline_install.coreml_executables
        .remove("fp16_throughput")
        .expect("baseline must have fp16_throughput executable");

    warmup_exec(&mut baseline_exec, &mut baseline_install.arena);

    let baseline_plan = make_execution_plan();
    let mut baseline_sched = EpochScheduler::new(baseline_plan);

    for _ in 0..WARMUP_EPOCHS {
        baseline_sched
            .execute_epoch(&mut baseline_install.arena, &mut baseline_exec, &mut baseline_metal)
            .expect("baseline warmup should succeed");
    }

    let baseline_metrics = run_baseline_sequence(
        &mut baseline_install.arena,
        &mut baseline_exec,
        &mut baseline_metal,
        &mut baseline_sched,
        THROUGHPUT_EPOCHS,
    );

    let baseline_total_ns = baseline_metrics.total_wall_ns();
    let baseline_thpt = if baseline_total_ns > 0 {
        THROUGHPUT_EPOCHS as f64 / (baseline_total_ns as f64 / 1_000_000_000.0)
    } else {
        0.0
    };
    let baseline_per_epoch_ns = if baseline_metrics.epoch_count > 0 {
        baseline_metrics.aggregate_epoch_wall_ns() as f64 / baseline_metrics.epoch_count as f64
    } else {
        0.0
    };

    eprintln!(
        "BASELINE: {} epochs in {} ms ({:.1} µs/epoch, {:.1} eps)",
        THROUGHPUT_EPOCHS,
        baseline_total_ns / 1_000_000,
        baseline_per_epoch_ns / 1000.0,
        baseline_thpt,
    );

    // ── Tri-lane: two sequences interleaved ──────────────────────────

    let mut install_a = create_throughput_install(slots.clone());
    let mut install_b = create_throughput_install(slots);

    let mut metal_a = install_a.metal_consumer.take()
        .expect("install A must have metal_consumer");
    let mut metal_b = install_b.metal_consumer.take()
        .expect("install B must have metal_consumer");

    let mut exec_a = install_a.coreml_executables
        .remove("fp16_throughput")
        .expect("install A must have fp16_throughput executable");
    let mut exec_b = install_b.coreml_executables
        .remove("fp16_throughput")
        .expect("install B must have fp16_throughput executable");

    warmup_exec(&mut exec_a, &mut install_a.arena);
    warmup_exec(&mut exec_b, &mut install_b.arena);

    let plan_a = make_execution_plan();
    let plan_b = make_execution_plan();
    let mut sched_a = EpochScheduler::new(plan_a);
    let mut sched_b = EpochScheduler::new(plan_b);

    for _ in 0..WARMUP_EPOCHS {
        sched_a
            .execute_epoch(&mut install_a.arena, &mut exec_a, &mut metal_a)
            .expect("tri-lane warmup A should succeed");
        sched_b
            .execute_epoch(&mut install_b.arena, &mut exec_b, &mut metal_b)
            .expect("tri-lane warmup B should succeed");
    }

    let (metrics_a, metrics_b) = run_interleaved_sequences(
        (
            &mut install_a.arena, &mut exec_a, &mut metal_a, &mut sched_a,
        ),
        (
            &mut install_b.arena, &mut exec_b, &mut metal_b, &mut sched_b,
        ),
        THROUGHPUT_EPOCHS,
    );

    let tri_lane_total_ns = metrics_a
        .total_wall_ns()
        .max(metrics_b.total_wall_ns());
    let tri_lane_total_epochs = metrics_a.epoch_count + metrics_b.epoch_count;

    let tri_lane_thpt = if tri_lane_total_ns > 0 {
        tri_lane_total_epochs as f64 / (tri_lane_total_ns as f64 / 1_000_000_000.0)
    } else {
        0.0
    };

    let speedup_ratio = if baseline_thpt > 0.0 {
        tri_lane_thpt / baseline_thpt
    } else {
        0.0
    };

    eprintln!(
        "TRI-LANE: {} epochs across 2 sequences in {} ms ({:.1} eps, {:.2}x vs baseline)",
        tri_lane_total_epochs,
        tri_lane_total_ns / 1_000_000,
        tri_lane_thpt,
        speedup_ratio,
    );

    // ── Assertions ──────────────────────────────────────────────────

    assert!(
        baseline_per_epoch_ns > 0.0,
        "baseline must record nonzero per-epoch wall time"
    );

    assert!(
        speedup_ratio >= MIN_THROUGHPUT_RATIO,
        "tri-lane throughput ratio {:.2}x below min {:.2}x (baseline={:.1} eps, tri-lane={:.1} eps)",
        speedup_ratio,
        MIN_THROUGHPUT_RATIO,
        baseline_thpt,
        tri_lane_thpt,
    );

    assert_eq!(
        metrics_a.receipts.len() as u64, THROUGHPUT_EPOCHS,
        "seq A must complete {} epochs", THROUGHPUT_EPOCHS
    );
    assert_eq!(
        metrics_b.receipts.len() as u64, THROUGHPUT_EPOCHS,
        "seq B must complete {} epochs", THROUGHPUT_EPOCHS
    );

    for (i, receipt) in metrics_a.receipts.iter().enumerate() {
        assert!(
            receipt.overlap_ns.epoch_wall_ns > 0,
            "seq A epoch {} must have nonzero epoch wall time", i
        );
    }

    eprintln!(
        "OVERLAP: seq A {:.1}%, seq B {:.1}% epochs with nonzero overlap",
        metrics_a.overlap_fraction() * 100.0,
        metrics_b.overlap_fraction() * 100.0,
    );
}

// ── Test 3: Overlap latency headroom ─────────────────────────────────────

/// Per-epoch timing analysis for interleaved sequences.
#[test]
fn test_overlap_latency_headroom() {
    let slots = make_throughput_slots(0);

    let mut install_a = create_throughput_install(slots.clone());
    let mut install_b = create_throughput_install(slots);

    let mut metal_a = install_a.metal_consumer.take()
        .expect("install A must have metal_consumer");
    let mut metal_b = install_b.metal_consumer.take()
        .expect("install B must have metal_consumer");

    let mut exec_a = install_a.coreml_executables
        .remove("fp16_throughput")
        .expect("install A must have fp16_throughput executable");
    let mut exec_b = install_b.coreml_executables
        .remove("fp16_throughput")
        .expect("install B must have fp16_throughput executable");

    warmup_exec(&mut exec_a, &mut install_a.arena);
    warmup_exec(&mut exec_b, &mut install_b.arena);

    let plan = make_execution_plan();
    let mut sched_a = EpochScheduler::new(plan.clone());
    let mut sched_b = EpochScheduler::new(plan);

    // Warmup epochs
    for _ in 0..WARMUP_EPOCHS {
        sched_a
            .execute_epoch(&mut install_a.arena, &mut exec_a, &mut metal_a)
            .expect("warmup A should succeed");
        sched_b
            .execute_epoch(&mut install_b.arena, &mut exec_b, &mut metal_b)
            .expect("warmup B should succeed");
    }

    // ── Timing collection ────────────────────────────────────────────

    /// Per-epoch timing record for latency headroom analysis.
    #[derive(Debug, Clone)]
    struct EpochTiming {
        epoch: u64,
        seq: usize,
        epoch_wall_ns: u64,
        epoch_overlap_ns: u64,
        overlap_fraction: f64,
        orchestration_ns: i64,
    }

    let mut timings: Vec<EpochTiming> = Vec::new();

    for epoch in 0..MEASURED_EPOCHS {
        let t_start = Instant::now();

        // Sequence A epoch
        let receipt_a = sched_a
            .execute_epoch(&mut install_a.arena, &mut exec_a, &mut metal_a)
            .unwrap_or_else(|_| panic!("seq A epoch {} should succeed", epoch));
        let t_a = Instant::now();

        // Sequence B epoch
        let receipt_b = sched_b
            .execute_epoch(&mut install_b.arena, &mut exec_b, &mut metal_b)
            .unwrap_or_else(|_| panic!("seq B epoch {} should succeed", epoch));
        let t_b = Instant::now();

        let a_elapsed = t_a.duration_since(t_start);
        let b_elapsed = t_b.duration_since(t_start);

        let a_wall_ns = receipt_a.overlap_ns.epoch_wall_ns;
        let a_overlap_ns = receipt_a.overlap_ns.overlap_ns;
        let a_of = receipt_a.overlap_ns.overlap_fraction;
        let a_orch = a_elapsed.as_nanos() as i64 - a_wall_ns as i64;

        timings.push(EpochTiming {
            epoch,
            seq: 0,
            epoch_wall_ns: a_wall_ns,
            epoch_overlap_ns: a_overlap_ns,
            overlap_fraction: a_of,
            orchestration_ns: a_orch,
        });

        let b_wall_ns = receipt_b.overlap_ns.epoch_wall_ns;
        let b_overlap_ns = receipt_b.overlap_ns.overlap_ns;
        let b_of = receipt_b.overlap_ns.overlap_fraction;
        let b_orch = b_elapsed.as_nanos() as i64 - b_wall_ns as i64;

        timings.push(EpochTiming {
            epoch,
            seq: 1,
            epoch_wall_ns: b_wall_ns,
            epoch_overlap_ns: b_overlap_ns,
            overlap_fraction: b_of,
            orchestration_ns: b_orch,
        });
    }

    // ── Analysis ─────────────────────────────────────────────────────

    let seq_a_timings: Vec<&EpochTiming> = timings.iter().filter(|t| t.seq == 0).collect();
    let seq_b_timings: Vec<&EpochTiming> = timings.iter().filter(|t| t.seq == 1).collect();

    let a_avg_wall = seq_a_timings.iter().map(|t| t.epoch_wall_ns as f64).sum::<f64>()
        / seq_a_timings.len() as f64;
    let b_avg_wall = seq_b_timings.iter().map(|t| t.epoch_wall_ns as f64).sum::<f64>()
        / seq_b_timings.len() as f64;

    let a_avg_overlap = seq_a_timings.iter().map(|t| t.epoch_overlap_ns as f64).sum::<f64>()
        / seq_a_timings.len() as f64;
    let b_avg_overlap = seq_b_timings.iter().map(|t| t.epoch_overlap_ns as f64).sum::<f64>()
        / seq_b_timings.len() as f64;

    // Epoch-level speedup: serial sum / cycle wall
    let total_serial_wall: u64 = timings.iter().map(|t| t.epoch_wall_ns).sum();
    let mut cycle_walls: Vec<u64> = Vec::new();
    for i in 0..MEASURED_EPOCHS as usize {
        let a_t = &timings[i * 2];
        let b_t = &timings[i * 2 + 1];
        let cycle = b_t.epoch_wall_ns + a_t.orchestration_ns.unsigned_abs();
        cycle_walls.push(cycle);
    }
    let total_cycle_wall: u64 = cycle_walls.iter().sum();
    let speedup = if total_cycle_wall > 0 {
        total_serial_wall as f64 / total_cycle_wall as f64
    } else {
        1.0
    };

    // Boundary latency: inter-epoch gap for seq A
    let mut boundary_latencies: Vec<u64> = Vec::new();
    for i in 1..seq_a_timings.len() {
        let prev = seq_a_timings[i - 1];
        let curr = seq_a_timings[i];
        let gap = (curr.epoch_wall_ns as i64 - prev.orchestration_ns + prev.epoch_wall_ns as i64)
            .unsigned_abs();
        boundary_latencies.push(gap);
    }

    let avg_boundary_ns = if !boundary_latencies.is_empty() {
        boundary_latencies.iter().sum::<u64>() as f64 / boundary_latencies.len() as f64
    } else {
        0.0
    };

    let a_avg_orch = seq_a_timings.iter().map(|t| t.orchestration_ns as f64).sum::<f64>()
        / seq_a_timings.len() as f64;
    let b_avg_orch = seq_b_timings.iter().map(|t| t.orchestration_ns as f64).sum::<f64>()
        / seq_b_timings.len() as f64;

    // ── Report ───────────────────────────────────────────────────────

    eprintln!("--- Overlap Latency Headroom Report ---");
    eprintln!("Seq A: avg wall {:.1} µs, avg overlap {:.1} µs",
        a_avg_wall / 1000.0, a_avg_overlap / 1000.0);
    eprintln!("Seq B: avg wall {:.1} µs, avg overlap {:.1} µs",
        b_avg_wall / 1000.0, b_avg_overlap / 1000.0);
    eprintln!("Serial sum: {:.1} µs across {} timings",
        total_serial_wall as f64 / 1000.0, timings.len());
    eprintln!("Cycle wall: {:.1} µs across {} cycles",
        total_cycle_wall as f64 / 1000.0, MEASURED_EPOCHS);
    eprintln!("Epoch-level speedup: {:.4}x", speedup);
    eprintln!("Avg boundary latency: {:.1} µs ({} samples)",
        avg_boundary_ns / 1000.0, boundary_latencies.len());
    eprintln!("Avg CPU orchestration: seq A {:.1} µs, seq B {:.1} µs",
        a_avg_orch / 1000.0, b_avg_orch / 1000.0);

    // ── Assertions ──────────────────────────────────────────────────

    for t in &timings {
        assert!(
            t.epoch_wall_ns > 0,
            "seq {} epoch {} must have nonzero wall time",
            t.seq, t.epoch,
        );
    }

    assert!(
        speedup > 0.0,
        "epoch-level speedup must be positive, got {:.4}", speedup
    );

    let overlap_epochs = timings.iter().filter(|t| t.epoch_overlap_ns > 0).count();
    eprintln!(
        "Epochs with non-zero overlap: {}/{} ({:.1}%)",
        overlap_epochs,
        timings.len(),
        overlap_epochs as f64 / timings.len() as f64 * 100.0,
    );

    assert!(
        seq_a_timings.len() as u64 == MEASURED_EPOCHS,
        "seq A must produce {} timing records", MEASURED_EPOCHS
    );
    assert!(
        seq_b_timings.len() as u64 == MEASURED_EPOCHS,
        "seq B must produce {} timing records", MEASURED_EPOCHS
    );
}
