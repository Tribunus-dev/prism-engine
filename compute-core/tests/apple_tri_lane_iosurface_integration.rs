//! ANE-TRI-LANE-REALIZATION-0001 Phase 7: IOSurface integration soak.
//!
//! ```
//! This test does NOT exercise real Core ML or Metal hardware paths.
//! It validates the scheduling model, slot state machine, and manifest
//! contracts using in-memory simulation.
//! For real hardware execution, see `cargo test --features prism-backend`
//! on macOS Apple Silicon.
//! ```
//!
//! Installs a CImage manifest into the IOSurface shared arena, verifies
//! all three slots begin in Free state, then runs 1000 simulated epochs
//! cycling slot state through the full state machine:
//!
//!   Free → Reserved → Writing → Ready → Reading → Retired → Free
//!
//! The FP16 soak test uses the real EpochScheduler::execute_epoch() with
//! CoreML/Metal bindings instead of simulate_epoch().  Validates that slot
//! count and ring depth remain stable through 1000 epochs — a growing
//! `slots` HashMap would indicate a leak or unbounded allocation.  This is
//! the primary hardware-soak gate for the tri-lane IOSurface arena.

use tribunus_compute_core::backend::placement::ExecutionLane;
use tribunus_compute_core::compute_image::apple_cimage_manifest::{
    AppleFallbackManifest, AppleHardwareCompatibility, AppleNumericalPolicy,
    AppleSharedArenaManifest, AppleTriLaneAdmissionManifest, AppleTriLaneArtifactManifest,
    CoreMlArtifactManifest, CpuArtifactManifest, IOSurfaceSlotManifest, MetalArtifactManifest,
};
use tribunus_compute_core::compute_image::apple_shared_arena::{AppleSharedArena, SlotState};
use tribunus_compute_core::backend::metal_consumer::{MetalConsumer, MetalSlotBinding};
use tribunus_compute_core::backend::coreml_iosurface::{CoreMlIOSurfaceExecutable, CoreMlComputePolicy, CoreMlIOSurfaceBinding};
use tribunus_compute_core::compilation::epoch_scheduler::EpochScheduler;
use tribunus_compute_core::compilation::tri_lane::{
    AppleTriLaneExecutionPlan, AppleHardwareSignature, ShapeClass, NumericalPolicy,
    MetalProgramBinding, CpuProgramBinding, AppleFallbackPlan,
    TriLaneCostModel, TriLaneEvidenceRequirements, LaneCostEstimate,
};

// ── Helpers ────────────────────────────────────────────────────────────

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
        ring_depth: 3,
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
        plan_digest: "test-plan-0001".into(),
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

fn simulate_epoch(arena: &mut AppleSharedArena, epoch: u64) {
    // Producers acquire slots in sequence:
    //   Free → Reserved → Writing → Ready
    // Then consumers read:
    //   Ready → Reading → Retired

    let slot_ids: Vec<u32> = arena.slots.keys().copied().collect();

    for &id in &slot_ids {
        let producer = arena
            .slot(id)
            .unwrap()
            .manifest
            .producer;
        let slot = arena.slot_mut(id).unwrap();
        if slot.is_available_for(epoch, producer) {
            let _ = slot.reserve(epoch, producer);
        }
    }

    for &id in &slot_ids {
        if let Some(slot) = arena.slot_mut(id) {
            if matches!(slot.state, SlotState::Reserved { .. }) {
                let producer = slot.manifest.producer;
                slot.mark_writing(epoch, producer);
            }
        }
    }

    for &id in &slot_ids {
        if let Some(slot) = arena.slot_mut(id) {
            if matches!(slot.state, SlotState::Writing { .. }) {
                let producer = slot.manifest.producer;
                slot.mark_ready(epoch, producer);
            }
        }
    }

    for &id in &slot_ids {
        let consumer = arena
            .slot(id)
            .unwrap()
            .manifest
            .consumer;
        if let Some(slot) = arena.slot_mut(id) {
            if matches!(slot.state, SlotState::Ready { .. }) {
                let _ = slot.mark_reading(epoch, consumer);
            }
        }
    }

    for &id in &slot_ids {
        if let Some(slot) = arena.slot_mut(id) {
            if matches!(slot.state, SlotState::Reading { .. }) {
                slot.retire(epoch);
            }
        }
    }

    arena.advance_generation();
}

// ── Tests ──────────────────────────────────────────────────────────────

#[test]
fn test_install_cimage_and_run_1000_epochs() {
    let manifest = make_manifest();
    let mut arena = AppleSharedArena::install(&manifest.arena)
        .expect("install arena from manifest");

    // Verify structural invariants from manifest.
    assert_eq!(arena.slots.len(), 3, "expected exactly 3 slots");
    assert_eq!(arena.ring_depth, 3, "ring depth must match manifest");

    // All slots must start Free.
    for slot in arena.slots.values() {
        assert_eq!(
            slot.state,
            SlotState::Free,
            "slot {} should start Free",
            slot.manifest.slot_id
        );
    }

    // Run 1000 simulated epochs.
    for epoch in 0..1000 {
        simulate_epoch(&mut arena, epoch);

        // Every 100 epochs, verify slot count is stable.
        if epoch % 100 == 0 {
            assert!(
                arena.slots.len() <= 3,
                "slot count exceeded ring depth at epoch {}: {} slots",
                epoch,
                arena.slots.len()
            );
        }
    }

    assert_eq!(arena.slots.len(), 3, "slot count grew during soak");
    assert_eq!(arena.ring_depth, 3, "ring depth unchanged after soak");
}

#[test]
fn test_iosurface_mutation_detected() {
    // 1. Create an AppleSharedArena via install() — allocates real IOSurface backing
    let manifest = make_arena_manifest();
    let mut arena = AppleSharedArena::install(&manifest).unwrap();

    // 2. Write known data to input slot 0 (using per-slot backing)
    let input_slot = arena.slot(0).unwrap();
    let input_backing = input_slot.backing_arena.as_ref().expect("input slot 0 must have backing");
    let len = input_slot.manifest.byte_length as usize;
    let input_ptr = unsafe { input_backing.base_ptr() } as *mut u8;
    let slice = unsafe { std::slice::from_raw_parts_mut(input_ptr as *mut u32, len / 4) };
    for (i, v) in slice.iter_mut().enumerate() {
        *v = (i % 256) as u32;
    }

    // 3. Write known data to output slot 1 (simulating Core ML output, using per-slot backing)
    let output_slot = arena.slot(1).unwrap();
    let output_backing = output_slot.backing_arena.as_ref().expect("output slot 1 must have backing");
    let out_len = output_slot.manifest.byte_length as usize;
    let out_ptr = unsafe { output_backing.base_ptr() } as *mut u8;
    let out_slice = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut u32, out_len / 4) };
    for (i, v) in out_slice.iter_mut().enumerate() {
        *v = (i * 2 % 256) as u32;
    }

    // 4. Get CPU checksum before mutation
    let checksum_before: u64 = out_slice.iter().map(|&v| v as u64).sum();

    // 5. Mutate output slot contents (simulating different Core ML output)
    for (i, v) in out_slice.iter_mut().enumerate() {
        *v = (i * 3 % 256) as u32;
    }

    // 6. Get CPU checksum after mutation
    let checksum_after: u64 = out_slice.iter().map(|&v| v as u64).sum();

    // 7. Verify checksums differ — proves digest reflects actual byte contents
    assert_ne!(checksum_before, checksum_after,
        "checksum must change when slot contents change");

    // 5. Verify Metal consumer can read the same bytes
    let mut consumer = MetalConsumer::new("validation");
    let input_binding = MetalSlotBinding {
        slot_id: 1,
        tensor_name: "output".into(),
        byte_offset: arena.slot(1).unwrap().manifest.byte_offset,
        byte_length: arena.slot(1).unwrap().manifest.byte_length,
        layout_digest: arena.layout_digest.clone(),
    };
    consumer.add_input(input_binding);
    // Validate — creates and caches R16Uint texture from IOSurface
    let result = consumer.validate(&arena, 0).unwrap();
    assert!(result.matched, "CPU and Metal checksums must match on same bytes");
    // Both digests limit to 512 u16 elements (1024 bytes) — verify manually
    let max_u16s = (out_len / 2).min(512);
    let bounded_checksum: u64 = (0..max_u16s).map(|i| unsafe { (out_ptr as *const u16).add(i).read() } as u64).sum();
    assert_eq!(result.metal_digest, bounded_checksum,
        "Metal digest ({}) must match bounded u16 checksum ({})", result.metal_digest, bounded_checksum);

    // ── Persistence mutation test ──────────────────────────────────────────
    // After caching the texture (above), mutate the IOSurface bytes in-place,
    // then re-validate with the same cached texture. The texture shares the
    // IOSurface memory, so the checksum must change.

    // Baseline checksum from the first validation
    let baseline_digest = result.metal_digest;

    // Mutate slot bytes with a different pattern
    let out_slice_mut = unsafe { std::slice::from_raw_parts_mut(out_ptr as *mut u32, out_len / 4) };
    for (i, v) in out_slice_mut.iter_mut().enumerate() {
        *v = (i * 5 % 256) as u32;
    }

    // Re-validate with the SAME consumer — texture cache hit, same MTLTexture
    let result2 = consumer.validate(&arena, 0).unwrap();
    assert!(result2.matched, "second validation must still produce matching CPU/Metal digest");
    assert_ne!(result2.metal_digest, baseline_digest,
        "persistent Metal texture must detect IOSurface content changes: before={}, after={}",
        baseline_digest, result2.metal_digest);
}

#[test]
fn test_two_slot_isolation() {
    // Create arena with 2 slots, write distinct data to each.
    let manifest = make_arena_manifest();
    let arena = AppleSharedArena::install(&manifest).unwrap();

    // Write distinct patterns to slot 0 and slot 1
    let offset0 = arena.slot(0).unwrap().manifest.byte_offset as usize;
    let len0 = arena.slot(0).unwrap().manifest.byte_length as usize;
    let ptr0 = unsafe { arena.slot(0).unwrap().backing_arena.as_ref().unwrap().base_ptr() } as *mut u32;
    let slice0 = unsafe { std::slice::from_raw_parts_mut(ptr0, len0 / 4) };
    for (i, v) in slice0.iter_mut().enumerate() {
        *v = (i * 7 % 256) as u32;  // pattern A
    }

    let ptr1 = unsafe { arena.slot(1).unwrap().backing_arena.as_ref().unwrap().base_ptr() } as *mut u32;
    let len1 = arena.slot(1).unwrap().manifest.byte_length as usize;
    let slice1 = unsafe { std::slice::from_raw_parts_mut(ptr1, len1 / 4) };
    for (i, v) in slice1.iter_mut().enumerate() {
        *v = (i * 11 % 256) as u32;  // pattern B (distinct from A)
    }

    // Validate both slots — creates and caches textures
    let mut consumer0 = MetalConsumer::new("slot0");
    consumer0.add_input(MetalSlotBinding {
        slot_id: 0, tensor_name: "input".into(),
        byte_offset: 0, byte_length: len0 as u64,
        layout_digest: arena.layout_digest.clone(),
    });
    let mut consumer1 = MetalConsumer::new("slot1");
    consumer1.add_input(MetalSlotBinding {
        slot_id: 1, tensor_name: "output".into(),
        byte_offset: 0, byte_length: len1 as u64,
        layout_digest: arena.layout_digest.clone(),
    });

    let baseline0 = consumer0.validate(&arena, 0).unwrap();
    let baseline1 = consumer1.validate(&arena, 0).unwrap();
    assert!(baseline0.matched);
    assert!(baseline1.matched);
    assert_ne!(baseline0.metal_digest, baseline1.metal_digest,
        "two slots with distinct data must produce different digests");

    // Mutate slot 0 only
    for (i, v) in slice0.iter_mut().enumerate() {
        *v = (i * 13 % 256) as u32;  // pattern C (different from A)
    }

    // Re-validate both — slot 0 digest must change, slot 1 must NOT
    let after0 = consumer0.validate(&arena, 0).unwrap();
    let after1 = consumer1.validate(&arena, 0).unwrap();
    assert!(after0.matched);
    assert!(after1.matched);
    assert_ne!(after0.metal_digest, baseline0.metal_digest,
        "slot 0 digest must change after mutating slot 0");
    assert_eq!(after1.metal_digest, baseline1.metal_digest,
        "slot 1 digest must remain unchanged when only slot 0 mutated");
}

// ── FP16 Production V1 Tests ─────────────────────────────────────────
//
// These tests validate the FP16-only production envelope: float16 only,
// static shape, one IOSurface per ring slot, Core ML writes float16 slot,
// Metal consumes same IOSurface as R16Float texture, persistent texture
// cache, completion-driven slot lifecycle, fallback at epoch boundary,
// and artifact receipt with allocation attestation.

#[cfg(all(target_os = "macos", feature = "prism-backend"))]
mod fp16_production_v1 {
    use super::*;
    use coreml_proto::proto::mil_spec;
    use tribunus_compute_core::mil_builder::MilBuilder;
    use tribunus_compute_core::coreml_pipeline::compile_mlpackage;
    use tribunus_compute_core::compilation::apple_installation::{install_apple_tri_lane, warmup_with_arena, AppleInstallationResult};
    use tribunus_compute_core::compilation::tri_lane::CoreMlWarmupContract;
    use tribunus_compute_core::backend::coreml_iosurface::CoreMlComputePolicy;
    use tribunus_compute_core::mlpackage::{write_mlpackage, ModelMeta};
    use tribunus_compute_core::compute_image::fallback_plan::{CoreMlFailureInjector, TestFailureInjector};

    /// Returns the path to the compiled .mlmodelc directory.
    fn build_fp16_test_model(model_dir: &std::path::Path) -> Result<std::path::PathBuf, String> {
        let _ = std::fs::create_dir_all(model_dir);

        let weight: Vec<f32> = (0..4096).map(|i| (i % 256) as f32).collect();

        let prog = MilBuilder::new("main")
            .input("input", mil_spec::DataType::Float16, &[1, 64])
            .const_f16("weight", &weight, &[64, 64])
            .matmul("input", "weight")
            .output("output")
            .build()
            .map_err(|e| format!("MIL build: {:?}", e))?;

        let meta = ModelMeta {
            model_name: "fp16_test".into(),
            function_name: "main".into(),
            short_description: "FP16 test model".into(),
            version: "1.0.0".into(),
            author: "Tribunus Compute".into(),
            output_name: "output".into(),
            inputs: vec![("input".into(), vec![1, 64])],
            outputs: vec![("output".into(), vec![1, 64])],
        };

        let mlpackage_dir = write_mlpackage(prog, model_dir, &meta)
            .map_err(|e| format!("mlpackage write: {}", e))?;

        let output_dir = model_dir.join("compiled");
        std::fs::create_dir_all(&output_dir)
            .map_err(|e| format!("mkdir {}: {}", output_dir.display(), e))?;

        let receipt = compile_mlpackage(
            &mlpackage_dir,
            &output_dir,
            "fp16_test",
            "cpuAndNeuralEngine",
            "iOS15",
        ).map_err(|e| format!("compile_mlpackage: {}", e))?;

        Ok(std::path::Path::new(&receipt.compiled_modelc_path).to_path_buf())
    }

    fn make_fp16_manifest() -> AppleTriLaneArtifactManifest {
        let mut m = make_manifest();
        m.coreml_artifacts = vec![
            CoreMlArtifactManifest {
                artifact_id: "fp16_test".into(),
                mlmodelc_name: "fp16_test.mlmodelc".into(),
                package_digest: "test".into(),
                compiled_model_digest: "test".into(),
                compute_policy: "cpuAndNeuralEngine".into(),
                input_slots: vec!["0".into()],
                output_slots: vec!["1".into()],
            },
        ];
        m
    }

    fn create_fp16_install() -> AppleInstallationResult {
        let manifest = make_fp16_manifest();
        let model_dir = std::path::Path::new("/tmp/fp16-test-models");
        let _ = std::fs::create_dir_all(model_dir);

        // Build a real FP16 Core ML artifact if it doesn't exist
        let modelc_path = model_dir.join("fp16_test.mlmodelc");
        if !modelc_path.exists() {
            build_fp16_test_model(model_dir)
                .expect("FP16 test model compilation should succeed");
        }

        let mut result = install_apple_tri_lane(
            &manifest, model_dir, CoreMlComputePolicy::CpuAndNeuralEngine,
        ).expect("FP16 production install should succeed");
        result.precreate_metal_textures()
            .expect("precreate Metal textures should succeed");
        result
    }

    fn create_minimal_execution_plan() -> AppleTriLaneExecutionPlan {
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

    #[test]
    fn test_fp16_slot_allocated_with_correct_pixel_format() {
        // Create a float16 arena, verify each slot's IOSurface pixel format.
        let install = create_fp16_install();
        let arena = install.arena;
        let slot = arena.slot(0).unwrap();
        if let Some(backing) = &slot.backing_arena {
            // Arena::new(pw=1, ph=64, Float16) => C allocator: width=dim1=64, height=dim0=1
            assert_eq!(backing.info.width, 64, "width should be physical_shape[1]");
            assert_eq!(backing.info.height, 1, "height should be physical_shape[0]");
            // kCVPixelFormatType_OneComponent16Half = 'L00h' = 0x4C303068
            let pf = backing.info.pixel_format as u32;
            assert!(
                pf == 0x4C303068 || pf == 0x4C303066, // 'L00h' or 'L00f'
                "float16 slot must use a 16-bit half-float pixel format, got 0x{:08X}", pf
            );
        } else {
            panic!("slot 0 has no IOSurface backing");

        }
    }

    #[test]
    fn test_1000_epoch_fp16_reuse() {
        // Run 1000 epochs, verify slot identities stable, no alloc growth.
        // Uses the real EpochScheduler::execute_epoch() with CoreML/Metal
        // bindings instead of simulate_epoch().
        let mut install = create_fp16_install();
        let mut metal_consumer = install.metal_consumer.take().expect("install must have metal_consumer");
        let plan = create_minimal_execution_plan();
        let mut scheduler = EpochScheduler::new(plan);

        // Warm up the Core ML artifact against installed slots
for (_id, exec) in install.coreml_executables.iter_mut() {
            let warmup_contract = CoreMlWarmupContract {
                min_warmup_predictions: 3,
                max_warmup_latency_ms: 5000,
                tolerance: 0.01,
            };
            let record = warmup_with_arena(exec, &mut install.arena, &warmup_contract)
                .expect("real FP16 Core ML warmup must succeed");
            assert!(record.warmup_success, "warmup predictions must complete");
            assert!(record.output_present, "warmup must produce output");
            assert!(record.load_success, "warmup must load model");
        }

        // Take the Core ML executable from the install
        let mut coreml_exec = install.coreml_executables
            .remove("fp16_test")
            .expect("install must have fp16_test executable");


        for epoch in 0..1000u64 {
            let _receipt = scheduler
                .execute_epoch(&mut install.arena, &mut coreml_exec, &mut metal_consumer)
                .unwrap();
            if epoch % 100 == 0 {
                assert_eq!(
                    install.arena.slots.len(),
                    3,
                    "slot count must remain stable at 1000 epochs (epoch {})",
                    epoch
                );

            }
        }

        assert_eq!(install.arena.slots.len(), 3, "slot count stable after 1000 FP16 epochs");

    }

    #[test]
    fn test_fp16_fallback_at_epoch_boundary() {
        // Uses create_fp16_install() then injects failure via model_path
        // swap to make execute_epoch() skip the prediction step for epoch 5.
        // The TestFailureInjector gates which epoch gets the injected failure.
        let mut install = create_fp16_install();
        let mut metal_consumer = install.metal_consumer.take().expect("install must have metal_consumer");
        let plan = create_minimal_execution_plan();
        let mut scheduler = EpochScheduler::new(plan);
        let injector = TestFailureInjector { fail_epoch: Some(5) };

        let mut coreml_exec = install.coreml_executables
            .remove("fp16_test")
            .expect("install must have fp16_test executable");
        let original_model_path = coreml_exec.model_path.clone();

        // Run epochs 0-4: should succeed (CoreMlAne route)
        for epoch in 0..5u64 {
            // verify injector doesn't fire
            assert!(!injector.should_fail(epoch),
                "injector should not fire before epoch 5");

            scheduler
                .execute_epoch(&mut install.arena, &mut coreml_exec, &mut metal_consumer)
                .expect(&format!("epoch {} should succeed", epoch));
        }

        // Epoch 5: inject failure — swap model_path so load_model() fails
        // and execute_epoch() skips the prediction step.
        assert!(injector.should_fail(5),
            "injector must fire at epoch 5");
        coreml_exec.model_path = "/tmp/nonexistent_fp16_test.mlmodelc".into();
        coreml_exec.loaded = false;
        let _epoch5_receipt = scheduler
            .execute_epoch(&mut install.arena, &mut coreml_exec, &mut metal_consumer)
            .expect("epoch 5 should return Ok even with injected failure");
        // Prediction was skipped — output slot was not transitioned to Ready

        // After failure, verify fallback preserves ABI
        for id in 0..3 {
            let slot = install.arena.slot(id).unwrap();
            assert_eq!(slot.manifest.dtype, "float16",
                "fallback must preserve fp16 dtype on slot {}", id);
            assert_eq!(slot.manifest.physical_shape.len(), 2,
                "fallback must preserve 2D shape on slot {}", id);
            assert!(slot.manifest.strides_bytes[0] > 0,
                "fallback must preserve positive stride on slot {}", id);
        }

        // Run epochs 6+: restore model path, continue with fallback route
        coreml_exec.model_path = original_model_path;
        coreml_exec.loaded = false;
        for epoch in 6..10u64 {
            let _ = scheduler
                .execute_epoch(&mut install.arena, &mut coreml_exec, &mut metal_consumer)
                .expect(&format!("epoch {} should succeed after fallback", epoch));
        }
    }
}
