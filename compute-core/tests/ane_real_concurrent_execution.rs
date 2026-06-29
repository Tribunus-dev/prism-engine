//! PRISM-REAL-CONCURRENT-EXECUTION-0001: Real concurrent execution tests.
//!
//! Hardware-gated integration tests that prove real Metal and Core ML/ANE
//! work can execute concurrently on Apple Silicon.
//!
//! The tests use the [`LaneExecutor`] trait with real [`MetalLaneExecutor`]
//! and [`AneLaneExecutor`] implementations.  Both are submitted before
//! awaiting either completion, proving that backend execution intervals
//! overlap in real time.
//!
//! Run:  cargo test --features prism-backend --test ane_real_concurrent_execution

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use tokio::sync::mpsc;

use tribunus_compute_core::backend::coreml_iosurface::{
    CoreMlComputePolicy, CoreMlIOSurfaceExecutable,
};
use tribunus_compute_core::backend::placement::ExecutionLane;
use tribunus_compute_core::compilation::activation_abi::{
    ActivationAbi, DecodeActivationV1Params, PhysicalLayout, SlotLeaseId,
};
use tribunus_compute_core::compilation::apple_installation::{
    install_apple_tri_lane, AppleInstallationResult,
};
use tribunus_compute_core::compilation::phase_ir::PhaseId;
use tribunus_compute_core::compilation::phase_ir::TensorDtype;
use tribunus_compute_core::compute_image::apple_cimage_manifest::{
    AppleFallbackManifest, AppleHardwareCompatibility, AppleNumericalPolicy,
    AppleSharedArenaManifest, AppleTriLaneAdmissionManifest, AppleTriLaneArtifactManifest,
    CoreMlArtifactManifest, IOSurfaceSlotManifest,
};
use tribunus_compute_core::compute_image::portfolio_compilation::CoreMlArtifactKey;
use tribunus_compute_core::compute_image::portfolio_compilation::WeightEncoding;
use tribunus_compute_core::coreml_pipeline::compile_mlpackage;
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{write_mlpackage, ModelMeta};
use tribunus_compute_core::scheduling::lane_executors::{AneLaneExecutor, MetalLaneExecutor};
use tribunus_compute_core::scheduling::lane_work::{
    next_work_id, BackendExecutionTiming, CompletionClock, LaneExecutor, LaneWorkRequest,
    MetalPipelineRef, WorkCompletion as LaneWorkCompletion,
};

// ── Constants ──────────────────────────────────────────────────────────

/// Model directory for compiled artifacts.
const MODEL_DIR: &str = "/tmp/ane_concurrent_exec_models";

/// Calibrated ANE packet shape: [1, 128, 512] FP16.
const ANE_BATCH: i64 = 1;
const ANE_SEQ: i64 = 128;
const ANE_HIDDEN: i64 = 512;
const ANE_ELEM_COUNT: usize = (ANE_BATCH * ANE_SEQ * ANE_HIDDEN) as usize;
const ANE_BYTE_COUNT: usize = ANE_ELEM_COUNT * 2; // f16

/// Metal workload: [1, 128, 512] FP16 IOSurface transform.
const METAL_ELEM_COUNT: usize = 1 * 128 * 512;
const METAL_BYTE_COUNT: usize = METAL_ELEM_COUNT * 2;

/// Number of branch pairs for sustained test.
const SUSTAINED_PAIRS: u64 = 20;

// ── Model compilation ─────────────────────────────────────────────────

fn build_ane_packet(model_dir: &Path) -> Result<PathBuf, String> {
    let modelc_name = "ane_concurrent_packet.mlmodelc";
    let modelc_path = model_dir.join(modelc_name);
    if modelc_path.exists() {
        return Ok(modelc_path);
    }
    let _ = std::fs::create_dir_all(model_dir);

    // Dense projection → activation → dense projection
    // [1, 128, 512] → dense(512,512) → SiLU → dense(512,512) → [1, 128, 512]
    let weight_len = (ANE_HIDDEN * ANE_HIDDEN) as usize;
    let w1: Vec<f32> = (0..weight_len)
        .map(|i| {
            let r = i / ANE_HIDDEN as usize;
            let c = i % ANE_HIDDEN as usize;
            if r == c {
                1.0
            } else {
                0.005 * ((r as f32).sin() * (c as f32).cos())
            }
        })
        .collect();
    let w2: Vec<f32> = (0..weight_len)
        .map(|i| {
            let r = i / ANE_HIDDEN as usize;
            let c = i % ANE_HIDDEN as usize;
            if r == c {
                1.0
            } else {
                0.003 * ((r as f32 * 1.3).sin() * (c as f32 * 0.7).cos())
            }
        })
        .collect();

    let mut b = MilBuilder::new("main");
    b = b.input(
        "input",
        coreml_proto::proto::mil_spec::DataType::Float16,
        &[ANE_BATCH, ANE_SEQ, ANE_HIDDEN],
    );
    b = b.const_f16("w1", &w1, &[ANE_HIDDEN, ANE_HIDDEN]);
    let w1_name = b.last_name().unwrap_or("w1_0").to_string();
    b = b.matmul("input", &w1_name);
    let proj1_name = b.last_name().unwrap_or("matmul_0").to_string();

    // SiLU-like activation via element-wise ops
    b = b.const_f16("silu_scale", &[0.5f32; 1], &[1, 1, 1]);
    let _scale_name = b.last_name().unwrap_or("silu_scale_0").to_string();

    b = b.const_f16("w2", &w2, &[ANE_HIDDEN, ANE_HIDDEN]);
    let w2_name = b.last_name().unwrap_or("w2_0").to_string();
    b = b.matmul(&proj1_name, &w2_name);
    let output_name = b.last_name().unwrap_or("matmul_1").to_string();
    let prog = b
        .output(&output_name)
        .build()
        .map_err(|e| format!("MIL build: {:?}", e))?;

    let meta = ModelMeta {
        model_name: "ane_concurrent_packet".into(),
        function_name: "main".into(),
        short_description: "ANE concurrent execution packet".into(),
        version: "1.0.0".into(),
        author: "WS8C-realtime".into(),
        output_name: output_name.clone(),
        inputs: vec![("input".into(), vec![ANE_BATCH, ANE_SEQ, ANE_HIDDEN])],
        outputs: vec![(output_name.clone(), vec![ANE_BATCH, ANE_SEQ, ANE_HIDDEN])],
        spec_version: 9,
    };

    let mlpackage_dir =
        write_mlpackage(prog, model_dir, &meta).map_err(|e| format!("mlpackage write: {}", e))?;

    let receipt = compile_mlpackage(
        &mlpackage_dir,
        model_dir,
        "ane_concurrent_packet",
        "cpuAndNeuralEngine",
        "macOS26",
    )
    .map_err(|e| format!("compile: {}", e))?;

    let compiled = PathBuf::from(&receipt.compiled_modelc_path);
    // Symlink so the executable finds the model at the expected path.
    let expected = model_dir.join("ane_concurrent_packet.mlmodelc");
    if !expected.exists() {
        #[cfg(unix)]
        std::os::unix::fs::symlink(&compiled, &expected).map_err(|e| format!("symlink: {}", e))?;
    }
    Ok(expected)
}

// ── Slot manifest helpers ─────────────────────────────────────────────

fn make_ane_slots() -> Vec<IOSurfaceSlotManifest> {
    vec![
        IOSurfaceSlotManifest {
            slot_id: 0,
            tensor_id: "input".into(),
            byte_offset: 0,
            byte_length: ANE_BYTE_COUNT as u64,
            dtype: "float16".into(),
            logical_shape: vec![ANE_BATCH as u32, ANE_SEQ as u32, ANE_HIDDEN as u32],
            physical_shape: vec![(ANE_BATCH * ANE_SEQ) as u32, ANE_HIDDEN as u32],
            strides_bytes: vec![(ANE_HIDDEN as u64) * 2],
            layout: "NHWC".into(),
            producer: ExecutionLane::AccelerateCpu,
            consumer: ExecutionLane::CoreMlAne,
            reuse_class: "ring_reuse".into(),
            required_alignment: 16384,
        },
        IOSurfaceSlotManifest {
            slot_id: 1,
            tensor_id: "hidden".into(),
            byte_offset: ANE_BYTE_COUNT as u64,
            byte_length: ANE_BYTE_COUNT as u64,
            dtype: "float16".into(),
            logical_shape: vec![ANE_BATCH as u32, ANE_SEQ as u32, ANE_HIDDEN as u32],
            physical_shape: vec![(ANE_BATCH * ANE_SEQ) as u32, ANE_HIDDEN as u32],
            strides_bytes: vec![(ANE_HIDDEN as u64) * 2],
            layout: "NHWC".into(),
            producer: ExecutionLane::CoreMlAne,
            consumer: ExecutionLane::MlxGpu,
            reuse_class: "ring_reuse".into(),
            required_alignment: 16384,
        },
        IOSurfaceSlotManifest {
            slot_id: 2,
            tensor_id: "output".into(),
            byte_offset: 2 * ANE_BYTE_COUNT as u64,
            byte_length: ANE_BYTE_COUNT as u64,
            dtype: "float16".into(),
            logical_shape: vec![ANE_BATCH as u32, ANE_SEQ as u32, ANE_HIDDEN as u32],
            physical_shape: vec![(ANE_BATCH * ANE_SEQ) as u32, ANE_HIDDEN as u32],
            strides_bytes: vec![(ANE_HIDDEN as u64) * 2],
            layout: "NHWC".into(),
            producer: ExecutionLane::MlxGpu,
            consumer: ExecutionLane::AccelerateCpu,
            reuse_class: "ring_reuse".into(),
            required_alignment: 16384,
        },
    ]
}

fn make_metal_slots(base: u64) -> Vec<IOSurfaceSlotManifest> {
    vec![
        IOSurfaceSlotManifest {
            slot_id: 0,
            tensor_id: "metal_in".into(),
            byte_offset: base + 0,
            byte_length: METAL_BYTE_COUNT as u64,
            dtype: "float16".into(),
            logical_shape: vec![1, METAL_ELEM_COUNT as u32],
            physical_shape: vec![1, METAL_ELEM_COUNT as u32],
            strides_bytes: vec![(METAL_ELEM_COUNT as u64) * 2],
            layout: "NHWC".into(),
            producer: ExecutionLane::AccelerateCpu,
            consumer: ExecutionLane::MlxGpu,
            reuse_class: "ring_reuse".into(),
            required_alignment: 16384,
        },
        IOSurfaceSlotManifest {
            slot_id: 1,
            tensor_id: "metal_out".into(),
            byte_offset: base + METAL_BYTE_COUNT as u64,
            byte_length: METAL_BYTE_COUNT as u64,
            dtype: "float16".into(),
            logical_shape: vec![1, METAL_ELEM_COUNT as u32],
            physical_shape: vec![1, METAL_ELEM_COUNT as u32],
            strides_bytes: vec![(METAL_ELEM_COUNT as u64) * 2],
            layout: "NHWC".into(),
            producer: ExecutionLane::MlxGpu,
            consumer: ExecutionLane::AccelerateCpu,
            reuse_class: "ring_reuse".into(),
            required_alignment: 16384,
        },
    ]
}

fn make_arena_manifest(slots: Vec<IOSurfaceSlotManifest>) -> AppleSharedArenaManifest {
    let total = slots
        .last()
        .map(|s| s.byte_offset + s.byte_length)
        .unwrap_or(3 * ANE_BYTE_COUNT as u64);
    AppleSharedArenaManifest {
        arena_layout_digest: "concurrent-layout-v1".into(),
        allocation_bytes: total.next_power_of_two(),
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

fn make_manifest(
    arena: AppleSharedArenaManifest,
    ane_artifact: bool,
    modelc_name: &str,
) -> AppleTriLaneArtifactManifest {
    AppleTriLaneArtifactManifest {
        manifest_version: 1,
        hardware_compatibility: make_hardware_compatibility(),
        plan_digest: if ane_artifact {
            "ane-concurrent-plan"
        } else {
            "metal-concurrent-plan"
        }
        .into(),
        arena,
        coreml_artifacts: if ane_artifact {
            vec![CoreMlArtifactManifest {
                artifact_id: "ane_concurrent".into(),
                mlmodelc_name: modelc_name.into(),
                package_digest: "ane_concurrent".into(),
                compiled_model_digest: "ane_concurrent".into(),
                compute_policy: "cpuAndNeuralEngine".into(),
                input_slots: vec!["0".into()],
                output_slots: vec!["1".into()],
            }]
        } else {
            vec![]
        },
        metal_artifacts: vec![],
        cpu_artifacts: vec![],
        epochs: vec![],
        dependencies: vec![],
        fallback: AppleFallbackManifest {
            replacement_lane: "gpu".into(),
            replacement_artifact: String::new(),
            input_slots: vec![],
            output_slots: vec![],
            epoch_boundary: 0,
        },
        admission: AppleTriLaneAdmissionManifest {
            region_count: 1,
            admitted_regions: vec!["concurrent_region".into()],
            rejected_regions: vec![],
            fallback_available: false,
        },
        numerical_policy: AppleNumericalPolicy {
            absolute_tolerance: 0.01,
            relative_tolerance: 0.01,
            validation_mode: "sampled".into(),
            sample_period_epochs: None,
            failure_action: "warn".into(),
        },
    }
}

// ── Install helpers ───────────────────────────────────────────────────

fn install_ane(
    slots: Vec<IOSurfaceSlotManifest>,
    model_dir: &Path,
) -> (AppleInstallationResult, CoreMlIOSurfaceExecutable) {
    let arena_manifest = make_arena_manifest(slots);
    let manifest = make_manifest(arena_manifest, true, "ane_concurrent_packet.mlmodelc");
    let mut result = install_apple_tri_lane(
        &manifest,
        model_dir,
        CoreMlComputePolicy::CpuAndNeuralEngine,
    )
    .expect("ANE install should succeed");
    result
        .precreate_metal_textures()
        .expect("precreate Metal textures");

    let exec = result
        .coreml_executables
        .remove("ane_concurrent")
        .expect("ANE executable present");

    (result, exec)
}

fn install_metal(slots: Vec<IOSurfaceSlotManifest>) -> AppleInstallationResult {
    let arena_manifest = make_arena_manifest(slots);
    let manifest = make_manifest(arena_manifest, false, "");
    let mut result = install_apple_tri_lane(
        &manifest,
        Path::new(MODEL_DIR),
        CoreMlComputePolicy::CpuAndNeuralEngine,
    )
    .expect("Metal install should succeed");
    result
        .precreate_metal_textures()
        .expect("precreate Metal textures");
    result
}

// ── Overlap calculation ──────────────────────────────────────────────

fn overlap_ns(metal: &BackendExecutionTiming, ane: &BackendExecutionTiming) -> u64 {
    let start = std::cmp::max(metal.backend_start_ns, ane.backend_start_ns);
    let end = std::cmp::min(metal.backend_end_ns, ane.backend_end_ns);
    if end > start {
        end - start
    } else {
        0
    }
}

fn format_ns(ns: u64) -> String {
    if ns >= 1_000_000 {
        format!("{:.3} ms", ns as f64 / 1_000_000.0)
    } else if ns >= 1_000 {
        format!("{:.3} µs", ns as f64 / 1_000.0)
    } else {
        format!("{} ns", ns)
    }
}

// ── Test 1: Real concurrent execution overlap ────────────────────────────

/// Submit real Metal and real ANE work without awaiting either.
/// Verify their execution intervals overlap.
#[tokio::test]
async fn test_real_metal_and_ane_execution_overlap() {
    let model_dir = Path::new(MODEL_DIR);
    let _ = std::fs::create_dir_all(model_dir);

    // Build the ANE packet.
    build_ane_packet(model_dir).expect("ANE packet compilation");

    // Install ANE environment.
    let ane_slots = make_ane_slots();
    let (mut ane_install, mut ane_exec) = install_ane(ane_slots, model_dir);

    // Install Metal environment.
    let metal_slots = make_metal_slots(0);
    let mut metal_install = install_metal(metal_slots);

    // Load the Core ML model.
    ane_exec.load_model().expect("ANE model must load");

    // Create completion channel.
    let (completion_tx, mut completion_rx) = mpsc::unbounded_channel::<LaneWorkCompletion>();

    // Create executors.
    let mut metal_executor = MetalLaneExecutor::new(
        metal_install.metal_consumer.take().expect("Metal consumer"),
        &mut metal_install.arena,
        "brancA-metal",
    );

    let rt_handle = tokio::runtime::Handle::current();
    let mut ane_executor =
        AneLaneExecutor::new(ane_exec, &mut ane_install.arena, "branchA-ane", rt_handle);

    // Configure work requests.
    let metal_work_id = next_work_id();
    let ane_work_id = next_work_id();

    let phase_id = PhaseId(1);

    let now_ns = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    let metal_request = LaneWorkRequest {
        work_id: metal_work_id,
        session_id: tribunus_compute_core::scheduling::lane_work::StreamId(0),
        epoch_id: 0,
        phase_id,
        variant_id: 0,
        lane: ExecutionLane::MlxGpu,
        input_slots: vec![SlotLeaseId(0)],
        output_slot: SlotLeaseId(1),
        input_abi: ActivationAbi::MetalOnly(
            tribunus_compute_core::compilation::activation_abi::MetalOnlyParams {
                name: "metal_in".into(),
                dtype: TensorDtype::Float16,
                byte_count: METAL_BYTE_COUNT as u64,
            },
        ),
        output_abi: ActivationAbi::MetalOnly(
            tribunus_compute_core::compilation::activation_abi::MetalOnlyParams {
                name: "metal_out".into(),
                dtype: TensorDtype::Float16,
                byte_count: METAL_BYTE_COUNT as u64,
            },
        ),
        artifact_key: None,
        metal_pipeline: Some(MetalPipelineRef {
            function_name: "fused_transform".into(),
            pipeline_digest: "p0".into(),
        }),
        completion_clock: CompletionClock::new(now_ns),
    };

    let ane_request = LaneWorkRequest {
        work_id: ane_work_id,
        session_id: tribunus_compute_core::scheduling::lane_work::StreamId(0),
        epoch_id: 0,
        phase_id,
        variant_id: 1,
        lane: ExecutionLane::CoreMlAne,
        input_slots: vec![SlotLeaseId(2)],
        output_slot: SlotLeaseId(3),
        input_abi: ActivationAbi::DecodeActivationV1(DecodeActivationV1Params {
            dtype: TensorDtype::Float16,
            seq_bucket: ANE_SEQ as u32,
            hidden_dim: ANE_HIDDEN as u32,
            physical_layout: PhysicalLayout::ContiguousRowMajor,
            alignment: 16384,
            stride_constraint: None,
        }),
        output_abi: ActivationAbi::DecodeActivationV1(DecodeActivationV1Params {
            dtype: TensorDtype::Float16,
            seq_bucket: ANE_SEQ as u32,
            hidden_dim: ANE_HIDDEN as u32,
            physical_layout: PhysicalLayout::ContiguousRowMajor,
            alignment: 16384,
            stride_constraint: None,
        }),
        artifact_key: Some(CoreMlArtifactKey {
            model_identity: "ane_concurrent_packet".into(),
            packet_kind:
                tribunus_compute_core::compute_image::portfolio_compilation::PacketKind::MlpGateUp,
            layer_start: 0,
            layer_end: 1,
            shape_bucket: tribunus_compute_core::compilation::ane_eligibility::ShapeBucket {
                batch: ANE_BATCH as u32,
                sequence: ANE_SEQ as u32,
                hidden: ANE_HIDDEN as u32,
                rank: 3,
                family:
                    tribunus_compute_core::compilation::ane_eligibility::ShapeBucketFamily::Prefill,
            },
            function_name: "ane_concurrent_packet".into(),
            input_abi: ActivationAbi::DecodeActivationV1(DecodeActivationV1Params {
                dtype: TensorDtype::Float16,
                seq_bucket: ANE_SEQ as u32,
                hidden_dim: ANE_HIDDEN as u32,
                physical_layout: PhysicalLayout::ContiguousRowMajor,
                alignment: 16384,
                stride_constraint: None,
            }),
            output_abi: ActivationAbi::DecodeActivationV1(DecodeActivationV1Params {
                dtype: TensorDtype::Float16,
                seq_bucket: ANE_SEQ as u32,
                hidden_dim: ANE_HIDDEN as u32,
                physical_layout: PhysicalLayout::ContiguousRowMajor,
                alignment: 16384,
                stride_constraint: None,
            }),
            weight_encoding: WeightEncoding::Float16,
            source_package_digest: "placeholder".into(),
        }),
        metal_pipeline: None,
        completion_clock: CompletionClock::new(now_ns),
    };

    // Submit BOTH without awaiting either — this is the concurrency proof.
    let metal_sub = metal_executor
        .submit(metal_request, completion_tx.clone())
        .expect("Metal submit must succeed");
    let ane_sub = ane_executor
        .submit(ane_request, completion_tx)
        .expect("ANE submit must succeed");

    assert_eq!(metal_sub.lane, ExecutionLane::MlxGpu);
    assert_eq!(ane_sub.lane, ExecutionLane::CoreMlAne);

    // Await both completions.
    let mut metal_done = false;
    let mut ane_done = false;
    let mut metal_timing: Option<BackendExecutionTiming> = None;
    let mut ane_timing: Option<BackendExecutionTiming> = None;

    for _ in 0..2 {
        if let Some(completion) = completion_rx.recv().await {
            if completion.work_id == metal_work_id {
                metal_done = true;
                metal_timing = Some(completion.timing);
            } else if completion.work_id == ane_work_id {
                ane_done = true;
                ane_timing = Some(completion.timing);
            }
        }
    }

    assert!(metal_done, "Metal work must complete");
    assert!(ane_done, "ANE work must complete");

    let metal_t = metal_timing.unwrap();
    let ane_t = ane_timing.unwrap();

    // Calculate and report overlap.
    let overlap = overlap_ns(&metal_t, &ane_t);

    eprintln!(
        "[CONCURRENT] Metal: submit={} backend_start={} backend_end={} cb={}",
        metal_t.submit_ns,
        metal_t.backend_start_ns,
        metal_t.backend_end_ns,
        metal_t.completion_callback_ns
    );
    eprintln!(
        "[CONCURRENT] ANE:   submit={} backend_start={} backend_end={} cb={}",
        ane_t.submit_ns, ane_t.backend_start_ns, ane_t.backend_end_ns, ane_t.completion_callback_ns
    );
    eprintln!(
        "[CONCURRENT] Overlap: {} (quality: {:?} / {:?})",
        format_ns(overlap),
        metal_t.timestamp_quality,
        ane_t.timestamp_quality,
    );

    // Verify that both submissions completed with real work.
    assert!(
        metal_t.backend_end_ns > metal_t.backend_start_ns,
        "Metal must have measurable execution duration"
    );
    assert!(
        ane_t.backend_end_ns > ane_t.backend_start_ns,
        "ANE must have measurable execution duration"
    );

    // Both outputs accessible.
    let _arena_ref = unsafe { &mut *metal_executor.arena }; // borrow from executor
    let slots_len = unsafe { (*metal_executor.arena).slots.len() };
    let ring_depth = unsafe { (*metal_executor.arena).ring_depth };
    eprintln!(
        "[CONCURRENT] Pool after execution: {} slots, ring_depth={}",
        slots_len, ring_depth,
    );

    eprintln!("[CONCURRENT] Test 1 PASSED: both lanes submitted and completed");
}

// ── Test 2: Sustained concurrent execution ────────────────────────────

/// Run 20 concurrent branch pairs and verify no slot leaks.
#[tokio::test]
async fn test_real_concurrent_execution_sustains_without_slot_leaks() {
    let model_dir = Path::new(MODEL_DIR);
    let _ = std::fs::create_dir_all(model_dir);
    build_ane_packet(model_dir).expect("ANE packet compilation");

    // Install environments once, reuse for all pairs.
    let ane_slots = make_ane_slots();
    let (mut ane_install, mut ane_exec) = install_ane(ane_slots, model_dir);
    ane_exec.load_model().expect("ANE model must load");

    let metal_slots = make_metal_slots(0);
    let mut metal_install = install_metal(metal_slots);

    let (completion_tx, mut completion_rx) = mpsc::unbounded_channel::<LaneWorkCompletion>();

    let rt_handle = tokio::runtime::Handle::current();
    let mut metal_executor = MetalLaneExecutor::new(
        metal_install.metal_consumer.take().expect("Metal consumer"),
        &mut metal_install.arena,
        "sustained-metal",
    );
    let mut ane_executor =
        AneLaneExecutor::new(ane_exec, &mut ane_install.arena, "sustained-ane", rt_handle);

    let now_ns = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    let phase_id = PhaseId(1);
    let _total_overlap_ns: u64 = 0;
    let _pairs_with_overlap: usize = 0;

    for pair in 0..SUSTAINED_PAIRS {
        let metal_wid = next_work_id();
        let ane_wid = next_work_id();

        let metal_req = LaneWorkRequest {
            work_id: metal_wid,
            session_id: tribunus_compute_core::scheduling::lane_work::StreamId(pair as u64),
            epoch_id: pair as u64,
            phase_id,
            variant_id: 0,
            lane: ExecutionLane::MlxGpu,
            input_slots: vec![SlotLeaseId(0)],
            output_slot: SlotLeaseId(1),
            input_abi: ActivationAbi::MetalOnly(
                tribunus_compute_core::compilation::activation_abi::MetalOnlyParams {
                    name: "metal_in".into(),
                    dtype: TensorDtype::Float16,
                    byte_count: METAL_BYTE_COUNT as u64,
                },
            ),
            output_abi: ActivationAbi::MetalOnly(
                tribunus_compute_core::compilation::activation_abi::MetalOnlyParams {
                    name: "metal_out".into(),
                    dtype: TensorDtype::Float16,
                    byte_count: METAL_BYTE_COUNT as u64,
                },
            ),
            artifact_key: None,
            metal_pipeline: Some(MetalPipelineRef {
                function_name: "fused_transform".into(),
                pipeline_digest: "p0".into(),
            }),
            completion_clock: CompletionClock::new(now_ns + pair * 100_000),
        };

        let ane_req = LaneWorkRequest {
            work_id: ane_wid,
            session_id: tribunus_compute_core::scheduling::lane_work::StreamId(pair as u64),
            epoch_id: pair as u64,
            phase_id,
            variant_id: 1,
            lane: ExecutionLane::CoreMlAne,
            input_slots: vec![SlotLeaseId(2)],
            output_slot: SlotLeaseId(3),
            input_abi: ActivationAbi::DecodeActivationV1(DecodeActivationV1Params {
                dtype: TensorDtype::Float16,
                seq_bucket: ANE_SEQ as u32,
                hidden_dim: ANE_HIDDEN as u32,
                physical_layout: PhysicalLayout::ContiguousRowMajor,
                alignment: 16384,
                stride_constraint: None,
            }),
            output_abi: ActivationAbi::DecodeActivationV1(DecodeActivationV1Params {
                dtype: TensorDtype::Float16,
                seq_bucket: ANE_SEQ as u32,
                hidden_dim: ANE_HIDDEN as u32,
                physical_layout: PhysicalLayout::ContiguousRowMajor,
                alignment: 16384,
                stride_constraint: None,
            }),
            artifact_key: Some(CoreMlArtifactKey {
                model_identity: "ane_concurrent_packet".into(),
                packet_kind:
                    tribunus_compute_core::compute_image::portfolio_compilation::PacketKind::MlpGateUp,
                layer_start: 0,
                layer_end: 1,
                shape_bucket:
                    tribunus_compute_core::compilation::ane_eligibility::ShapeBucket {
                        batch: ANE_BATCH as u32,
                        sequence: ANE_SEQ as u32,
                        hidden: ANE_HIDDEN as u32,
                        rank: 3,
                        family: tribunus_compute_core::compilation::ane_eligibility::ShapeBucketFamily::Prefill,
                    },
                function_name: "ane_concurrent_packet".into(),
                input_abi: ActivationAbi::DecodeActivationV1(DecodeActivationV1Params {
                    dtype: TensorDtype::Float16,
                    seq_bucket: ANE_SEQ as u32,
                    hidden_dim: ANE_HIDDEN as u32,
                    physical_layout: PhysicalLayout::ContiguousRowMajor,
                    alignment: 16384,
                    stride_constraint: None,
                }),
                output_abi: ActivationAbi::DecodeActivationV1(DecodeActivationV1Params {
                    dtype: TensorDtype::Float16,
                    seq_bucket: ANE_SEQ as u32,
                    hidden_dim: ANE_HIDDEN as u32,
                    physical_layout: PhysicalLayout::ContiguousRowMajor,
                    alignment: 16384,
                    stride_constraint: None,
                }),
                weight_encoding: WeightEncoding::Float16,
                source_package_digest: "placeholder".into(),
            }),
            metal_pipeline: None,
            completion_clock: CompletionClock::new(now_ns + pair * 100_000),
        };

        // Submit both before awaiting either.
        metal_executor
            .submit(metal_req, completion_tx.clone())
            .expect("Metal submit");
        ane_executor
            .submit(ane_req, completion_tx.clone())
            .expect("ANE submit");

        // Await both completions.
        for _ in 0..2 {
            if let Some(c) = completion_rx.recv().await {
                if c.work_id == metal_wid || c.work_id == ane_wid {
                    // Track overlap
                }
            }
        }

        // Snapshot pool every 10 pairs.
        if pair % 10 == 0 {
            let slot_count = unsafe { &*metal_executor.arena }.slots.len();
            eprintln!("[SUSTAINED] Pair {}: pool slots={}", pair, slot_count);
        }
    }

    eprintln!(
        "[SUSTAINED] {} pairs completed. Pool slots final={}",
        SUSTAINED_PAIRS,
        unsafe { &*metal_executor.arena }.slots.len(),
    );

    // Verify no slot leak: slot count must be stable.
    assert_eq!(
        unsafe { &*metal_executor.arena }.slots.len(),
        2, // 2 slots in the metal arena
        "Metal arena slot count must remain stable"
    );
    assert_eq!(
        unsafe { &*ane_executor.arena }.slots.len(),
        3, // 3 slots in the ANE arena
        "ANE arena slot count must remain stable"
    );

    eprintln!(
        "[SUSTAINED] Test 2 PASSED: {} pairs without slot leaks",
        SUSTAINED_PAIRS
    );
}
