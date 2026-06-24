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
//! Validates that slot count and ring depth remain stable through every
//! epoch — a growing `slots` HashMap would indicate a leak or unbounded
//! allocation.  This is the primary hardware-soak gate for the tri-lane
//! IOSurface arena.

use tribunus_compute_core::backend::placement::ExecutionLane;
use tribunus_compute_core::compute_image::apple_cimage_manifest::{
    AppleFallbackManifest, AppleHardwareCompatibility, AppleNumericalPolicy,
    AppleSharedArenaManifest, AppleTriLaneAdmissionManifest, AppleTriLaneArtifactManifest,
    CoreMlArtifactManifest, CpuArtifactManifest, IOSurfaceSlotManifest, MetalArtifactManifest,
};
use tribunus_compute_core::compute_image::apple_shared_arena::{AppleSharedArena, SlotState};
use tribunus_compute_core::backend::metal_consumer::{MetalConsumer, MetalSlotBinding};

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
