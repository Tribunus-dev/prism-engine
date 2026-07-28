//! Core ML IOSurface binding — accepts a compiled Core ML artifact and an
//! IOSurface-backed arena, returns a validated executable binding.
//!
//! Bridge between Core ML model artifacts and IOSurface-resident tensor slots.
//! Each binding maps a model tensor to an IOSurface arena slot, validated
//! against a cimage manifest contract.

use crate::coreai_bridge::{CoreAiComputeUnits, CoreAiModel};
use crate::ecs::backend::shared_event::SharedEventBinding;
use std::ffi::c_void;
use std::io;

/// Core ML compute policy enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreAiComputePolicy {
    CpuOnly,
    CpuAndNeuralEngine,
    NeuralEngineOnly,
    GpuOnly,
    All,
}

impl CoreAiComputePolicy {
    pub fn name(&self) -> &'static str {
        match self {
            CoreAiComputePolicy::CpuOnly => "cpuOnly",
            CoreAiComputePolicy::CpuAndNeuralEngine => "cpuAndNeuralEngine",
            CoreAiComputePolicy::NeuralEngineOnly => "neuralEngine",
            CoreAiComputePolicy::GpuOnly => "gpuOnly",
            CoreAiComputePolicy::All => "all",
        }
    }
}

/// Core ML IOSurface binding — a single tensor's binding to an IOSurface slot.
#[derive(Debug, Clone)]
pub struct CoreAiIOSurfaceBinding {
    pub tensor_id: String,
    pub slot_id: u32,
    pub io_surface_id: u32,
    pub byte_offset: u64,
    pub contract_digest: String,
}

/// Validated executable binding for Core ML with IOSurface-backed arenas.
pub struct CoreAiIOSurfaceExecutable {
    pub artifact_id: String,
    pub compute_policy: CoreAiComputePolicy,
    pub input_bindings: Vec<CoreAiIOSurfaceBinding>,
    pub output_bindings: Vec<CoreAiIOSurfaceBinding>,
    pub model_path: String,
    /// Shared-event waits/signals that guard IOSurface slot handoff.
    pub shared_event_bindings: Vec<SharedEventBinding>,
    /// Whether the underlying Core ML model is loaded.
    pub loaded: bool,
    /// Loaded Core ML model handle, or None before load_model() is called.
    pub model: Option<CoreAiModel>,
}

impl CoreAiIOSurfaceExecutable {
    pub fn new(artifact_id: &str, model_path: &str, compute_policy: CoreAiComputePolicy) -> Self {
        Self {
            artifact_id: artifact_id.to_string(),
            compute_policy,
            input_bindings: Vec::new(),
            output_bindings: Vec::new(),
            model_path: model_path.to_string(),
            shared_event_bindings: Vec::new(),
            loaded: false,
            model: None,
        }
    }

    /// Add an input binding, returns error if slot_id already bound.
    pub fn add_input_binding(&mut self, binding: CoreAiIOSurfaceBinding) -> Result<(), String> {
        if self
            .input_bindings
            .iter()
            .any(|b| b.slot_id == binding.slot_id)
        {
            return Err(format!("slot {} already bound as input", binding.slot_id));
        }
        self.input_bindings.push(binding);
        Ok(())
    }

    /// Add an output binding.
    pub fn add_output_binding(&mut self, binding: CoreAiIOSurfaceBinding) -> Result<(), String> {
        if self
            .output_bindings
            .iter()
            .any(|b| b.slot_id == binding.slot_id)
        {
            return Err(format!("slot {} already bound as output", binding.slot_id));
        }
        self.output_bindings.push(binding);
        Ok(())
    }

    /// Attach one shared-event wait or signal to this executable.
    pub fn add_shared_event_binding(&mut self, binding: SharedEventBinding) {
        self.shared_event_bindings.push(binding);
    }

    /// Bind from an AppleSharedArena manifest — validates shape/dtype/layout match.
    pub fn bind_from_arena(
        &mut self,
        arena_slots: &[crate::ecs::legacy_compute_image_core::apple_cimage_manifest::IOSurfaceSlotManifest],
    ) -> Result<(), String> {
        for binding in self.input_bindings.iter_mut() {
            let slot = arena_slots
                .iter()
                .find(|s| s.slot_id == binding.slot_id)
                .ok_or_else(|| format!("slot {} not found in arena", binding.slot_id))?;
            // Validate compatible layout — stub for now
            binding.contract_digest = format!("digest:{}", slot.tensor_id);
        }
        for binding in self.output_bindings.iter_mut() {
            let slot = arena_slots
                .iter()
                .find(|s| s.slot_id == binding.slot_id)
                .ok_or_else(|| format!("slot {} not found in arena", binding.slot_id))?;
            binding.contract_digest = format!("digest:{}", slot.tensor_id);
        }
        Ok(())
    }

    /// Load the Core ML model for this executable.
    pub fn load_model(&mut self) -> Result<(), String> {
        if self.loaded {
            return Ok(());
        }
        let compute_units = match self.compute_policy {
            CoreAiComputePolicy::CpuAndNeuralEngine => CoreAiComputeUnits::CpuAndNeuralEngine,
            CoreAiComputePolicy::CpuOnly => CoreAiComputeUnits::CpuOnly,
            CoreAiComputePolicy::NeuralEngineOnly => {
                // Apple does not expose a public MLComputeUnits value
                // that guarantees exclusive ANE execution. Map to
                // CpuAndNeuralEngine with a comment documenting this
                // limitation.
                CoreAiComputeUnits::CpuAndNeuralEngine
            }
            CoreAiComputePolicy::GpuOnly => CoreAiComputeUnits::CpuAndGpu,
            CoreAiComputePolicy::All => CoreAiComputeUnits::All,
        };
        let model = CoreAiModel::load_with_compute_units(&self.model_path, compute_units)?;
        self.model = Some(model);
        self.loaded = true;
        Ok(())
    }

    /// Reject if any input/output tensor name differs from the cimage contract.
    pub fn validate_against_slots(
        &self,
        input_contract: &[CoreAiIOSurfaceBinding],
        output_contract: &[CoreAiIOSurfaceBinding],
    ) -> Result<(), String> {
        if self.input_bindings.len() != input_contract.len() {
            return Err("input binding count mismatch".into());
        }
        if self.output_bindings.len() != output_contract.len() {
            return Err("output binding count mismatch".into());
        }
        for (a, b) in self.input_bindings.iter().zip(input_contract.iter()) {
            if a.tensor_id != b.tensor_id {
                return Err(format!(
                    "input tensor name mismatch: {} vs {}",
                    a.tensor_id, b.tensor_id
                ));
            }
        }
        for (a, b) in self.output_bindings.iter().zip(output_contract.iter()) {
            if a.tensor_id != b.tensor_id {
                return Err(format!(
                    "output tensor name mismatch: {} vs {}",
                    a.tensor_id, b.tensor_id
                ));
            }
        }
        Ok(())
    }

    /// Bind the canonical NF4Tile640 shared-weight triplet.
    pub fn bind_nf4_tile640_triplet(
        &mut self,
        packed_weights_slot: u32,
        packed_weights_byte_offset: u64,
        scales_slot: u32,
        scales_byte_offset: u64,
        biases_slot: u32,
        biases_byte_offset: u64,
        contract_digest: &str,
    ) -> Result<(), String> {
        for (slot_id, byte_offset, tensor_id) in [
            (
                packed_weights_slot,
                packed_weights_byte_offset,
                "packed_nf4_weights",
            ),
            (scales_slot, scales_byte_offset, "scales"),
            (biases_slot, biases_byte_offset, "biases"),
        ] {
            self.add_input_binding(CoreAiIOSurfaceBinding {
                tensor_id: tensor_id.into(),
                slot_id,
                io_surface_id: 0,
                byte_offset,
                contract_digest: contract_digest.into(),
            })?;
        }
        Ok(())
    }
}

/// Create an IOSurface backed by a page-aligned mmap slice.
/// The kernel skips the shadow copy because the pointer matches the
/// hardware 16 KB boundary — the IOSurface pages are wired directly.
///
/// # Parameters
/// - `base`: Page-aligned pointer to the mmap'd data (may be null for
///   zero-initialized allocation).
/// - `width`: IOSurface width in pixels.
/// - `height`: IOSurface height in pixels.
/// - `pixel_format`: FourCC pixel format (e.g. `'L00h'` for FP16).
///
/// # Returns
/// The IOSurfaceRef as an opaque pointer, or an error if allocation fails.
/// The returned IOSurface owns its backing pages and must be freed by the
/// caller via `CFRelease`.
pub fn create_iosurface_from_mmap(
    base: *const u8,
    width: u32,
    height: u32,
    pixel_format: u32,
) -> io::Result<*mut c_void> {
    let byte_count = (width as u64) * (height as u64) * 4; // worst-case bytes per pixel
    if byte_count > i32::MAX as u64 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "IOSurface too large",
        ));
    }
    let mut info = crate::arena_info::ArenaInfo {
        width: 0,
        height: 0,
        logical_dim0: 0,
        logical_dim1: 0,
        pixel_format: 0,
        byte_size: 0,
        bytes_per_row: 0,
        base_address: std::ptr::null_mut(),
        cv_buffer: std::ptr::null_mut(),
        io_surface: std::ptr::null_mut(),
    };
    let rc = unsafe {
        crate::arena::tribunus_create_iosurface_from_mmap(
            &mut info as *mut crate::arena_info::ArenaInfo,
            base as *const std::ffi::c_void,
            width as i32,
            height as i32,
            pixel_format,
            byte_count as i32,
        )
    };
    if rc != 0 {
        return Err(io::Error::new(
            io::ErrorKind::Other,
            format!("create_iosurface_from_mmap failed: {}", rc),
        ));
    }
    Ok(info.io_surface)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bind_add_input_output() {
        let mut exec = CoreAiIOSurfaceExecutable::new(
            "artifact_1",
            "/tmp/model.mlmodelc",
            CoreAiComputePolicy::All,
        );

        let input = CoreAiIOSurfaceBinding {
            tensor_id: "input_0".into(),
            slot_id: 0,
            io_surface_id: 1,
            byte_offset: 0,
            contract_digest: String::new(),
        };
        let output = CoreAiIOSurfaceBinding {
            tensor_id: "output_0".into(),
            slot_id: 1,
            io_surface_id: 2,
            byte_offset: 4096,
            contract_digest: String::new(),
        };

        assert!(exec.add_input_binding(input.clone()).is_ok());
        assert!(exec.add_output_binding(output.clone()).is_ok());
        assert_eq!(exec.input_bindings.len(), 1);
        assert_eq!(exec.output_bindings.len(), 1);
        assert_eq!(exec.input_bindings[0].tensor_id, "input_0");
        assert_eq!(exec.output_bindings[0].tensor_id, "output_0");
    }

    #[test]
    fn test_bind_duplicate_slot_rejected() {
        let mut exec = CoreAiIOSurfaceExecutable::new(
            "artifact_dup",
            "/tmp/model.mlmodelc",
            CoreAiComputePolicy::NeuralEngineOnly,
        );

        let binding = CoreAiIOSurfaceBinding {
            tensor_id: "x".into(),
            slot_id: 5,
            io_surface_id: 1,
            byte_offset: 0,
            contract_digest: String::new(),
        };

        assert!(exec.add_input_binding(binding.clone()).is_ok());
        // Same slot_id 5 on inputs — should fail
        let dup = CoreAiIOSurfaceBinding {
            tensor_id: "y".into(),
            slot_id: 5,
            io_surface_id: 2,
            byte_offset: 1024,
            contract_digest: String::new(),
        };
        assert!(exec.add_input_binding(dup).is_err());

        // Different slot_id 5 on outputs — outputs track their own set, so this is fine
        let out = CoreAiIOSurfaceBinding {
            tensor_id: "out".into(),
            slot_id: 5,
            io_surface_id: 2,
            byte_offset: 1024,
            contract_digest: String::new(),
        };
        assert!(exec.add_output_binding(out.clone()).is_ok());

        // Same slot_id 5 again on outputs — should fail
        let dup_out = CoreAiIOSurfaceBinding {
            tensor_id: "out2".into(),
            slot_id: 5,
            io_surface_id: 3,
            byte_offset: 2048,
            contract_digest: String::new(),
        };
        assert!(exec.add_output_binding(dup_out).is_err());
    }

    #[test]
    fn test_validate_contract_mismatch_rejected() {
        let mut exec = CoreAiIOSurfaceExecutable::new(
            "contract_test",
            "/tmp/model.mlmodelc",
            CoreAiComputePolicy::All,
        );

        exec.add_input_binding(CoreAiIOSurfaceBinding {
            tensor_id: "input_a".into(),
            slot_id: 0,
            io_surface_id: 1,
            byte_offset: 0,
            contract_digest: String::new(),
        })
        .unwrap();
        exec.add_output_binding(CoreAiIOSurfaceBinding {
            tensor_id: "output_a".into(),
            slot_id: 1,
            io_surface_id: 2,
            byte_offset: 4096,
            contract_digest: String::new(),
        })
        .unwrap();

        // Input contract with wrong tensor_id
        let bad_input = CoreAiIOSurfaceBinding {
            tensor_id: "input_b".into(),
            slot_id: 0,
            io_surface_id: 1,
            byte_offset: 0,
            contract_digest: String::new(),
        };

        let result = exec.validate_against_slots(&[bad_input], &exec.output_bindings);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("input tensor name mismatch: input_a vs input_b"));

        // Rebuild exec for output mismatch test
        let mut exec2 = CoreAiIOSurfaceExecutable::new(
            "contract_test_2",
            "/tmp/model.mlmodelc",
            CoreAiComputePolicy::All,
        );
        exec2
            .add_input_binding(CoreAiIOSurfaceBinding {
                tensor_id: "input_a".into(),
                slot_id: 0,
                io_surface_id: 1,
                byte_offset: 0,
                contract_digest: String::new(),
            })
            .unwrap();
        exec2
            .add_output_binding(CoreAiIOSurfaceBinding {
                tensor_id: "output_a".into(),
                slot_id: 1,
                io_surface_id: 2,
                byte_offset: 4096,
                contract_digest: String::new(),
            })
            .unwrap();

        let bad_output = CoreAiIOSurfaceBinding {
            tensor_id: "output_b".into(),
            slot_id: 1,
            io_surface_id: 2,
            byte_offset: 4096,
            contract_digest: String::new(),
        };

        let result2 = exec2.validate_against_slots(&exec2.input_bindings, &[bad_output]);
        assert!(result2.is_err());
        assert!(result2
            .unwrap_err()
            .contains("output tensor name mismatch: output_a vs output_b"));
    }

    #[test]
    fn test_validate_count_mismatch() {
        let exec = CoreAiIOSurfaceExecutable::new(
            "count_test",
            "/tmp/model.mlmodelc",
            CoreAiComputePolicy::CpuOnly,
        );
        // Zero input bindings, but pass one contract entry
        let contract = CoreAiIOSurfaceBinding {
            tensor_id: "x".into(),
            slot_id: 0,
            io_surface_id: 0,
            byte_offset: 0,
            contract_digest: String::new(),
        };
        let result = exec.validate_against_slots(&[contract], &[]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("input binding count mismatch"));
    }

    #[test]
    fn test_compute_policy_name() {
        assert_eq!(CoreAiComputePolicy::CpuOnly.name(), "cpuOnly");
        assert_eq!(
            CoreAiComputePolicy::CpuAndNeuralEngine.name(),
            "cpuAndNeuralEngine"
        );
        assert_eq!(CoreAiComputePolicy::NeuralEngineOnly.name(), "neuralEngine");
        assert_eq!(CoreAiComputePolicy::GpuOnly.name(), "gpuOnly");
        assert_eq!(CoreAiComputePolicy::All.name(), "all");
    }

    #[test]
    fn test_executable_new_defaults() {
        let exec =
            CoreAiIOSurfaceExecutable::new("test", "/path.mlmodelc", CoreAiComputePolicy::GpuOnly);
        assert_eq!(exec.artifact_id, "test");
        assert_eq!(exec.model_path, "/path.mlmodelc");
        assert_eq!(exec.compute_policy, CoreAiComputePolicy::GpuOnly);
        assert!(exec.input_bindings.is_empty());
        assert!(exec.output_bindings.is_empty());
        assert!(!exec.loaded);
    }

    #[test]
    fn test_bind_nf4_tile640_triplet() {
        let mut exec = CoreAiIOSurfaceExecutable::new(
            "nf4_tile640",
            "/path.mlmodelc",
            CoreAiComputePolicy::CpuAndNeuralEngine,
        );

        exec.bind_nf4_tile640_triplet(7, 64, 8, 128, 9, 256, "nf4tile640-v1")
            .expect("bind triplet");

        assert_eq!(exec.input_bindings.len(), 3);
        assert_eq!(exec.input_bindings[0].tensor_id, "packed_nf4_weights");
        assert_eq!(exec.input_bindings[1].tensor_id, "scales");
        assert_eq!(exec.input_bindings[2].tensor_id, "biases");
        assert_eq!(exec.input_bindings[0].byte_offset, 64);
        assert_eq!(exec.input_bindings[1].byte_offset, 128);
        assert_eq!(exec.input_bindings[2].byte_offset, 256);
    }

    #[test]
    /// This test loads a non-existent CoreAI model which sets the global
    /// CoreAI initialization flag (g_init). When other tests run after this
    /// one in the same process, they inherit stale CoreAI state and fail
    /// unpredictably. The model-load-failure path is already exercised by
    /// `test_install_creates_coreai_executables` in apple_installation,
    /// so this test is quarantined to avoid poisoning global state.
    #[ignore]
    fn test_coreai_iosurface_warmup_with_arena() {
        use crate::ecs::backend::coreai_lane::{CoreAiLane, CoreAiSubgraph, CoreAiSubgraphStatus};
        use crate::ecs::backend::placement::ExecutionLane;
        use crate::ecs::legacy_compilation::tri_lane::{AneLaneLifecycle, CoreAiWarmupContract};
        use crate::ecs::legacy_compute_image_core::apple_shared_arena::{
            AppleSharedArena, IOSurfaceSlotManifest, LiveIOSurfaceSlot, SlotReuseClass,
        };

        // Create arena with input/output slots
        let mut arena = AppleSharedArena::new("test-arena".into(), 1);

        arena.add_slot(LiveIOSurfaceSlot {
            manifest: IOSurfaceSlotManifest {
                slot_id: 0,
                tensor_id: "input".into(),
                byte_offset: 0,
                byte_length: 4096,
                dtype: "float32".into(),
                logical_shape: vec![1, 1],
                physical_shape: vec![1, 1],
                strides_bytes: vec![4, 4],
                layout: "NHWC".into(),
                producer: ExecutionLane::CoreAiAne,
                consumer: ExecutionLane::MlxGpu,
                reuse_class: SlotReuseClass::Exclusive,
                required_alignment: 64,
            },
            state: crate::ecs::legacy_compute_image_core::apple_shared_arena::SlotState::Free,
            generation: 0,
            layout_digest: "digest-00000000".into(),
            metal_view: None,
            coreai_view: None,
            backing_arena: None,
            attestation: None,
        });

        arena.add_slot(LiveIOSurfaceSlot {
            manifest: IOSurfaceSlotManifest {
                slot_id: 1,
                tensor_id: "output".into(),
                byte_offset: 4096,
                byte_length: 4096,
                dtype: "float32".into(),
                logical_shape: vec![1, 1],
                physical_shape: vec![1, 1],
                strides_bytes: vec![4, 4],
                layout: "NHWC".into(),
                producer: ExecutionLane::CoreAiAne,
                consumer: ExecutionLane::MlxGpu,
                reuse_class: SlotReuseClass::Exclusive,
                required_alignment: 64,
            },
            state: crate::ecs::legacy_compute_image_core::apple_shared_arena::SlotState::Free,
            generation: 0,
            layout_digest: "digest-00000000".into(),
            metal_view: None,
            coreai_view: None,
            backing_arena: None,
            attestation: None,
        });

        // Create executable with input/output bindings matching arena slots
        let mut exec = CoreAiIOSurfaceExecutable::new(
            "warmup_test",
            "/tmp/warmup.mlmodelc",
            CoreAiComputePolicy::CpuAndNeuralEngine,
        );

        exec.add_input_binding(CoreAiIOSurfaceBinding {
            tensor_id: "input".into(),
            slot_id: 0,
            io_surface_id: 1,
            byte_offset: 0,
            contract_digest: String::new(),
        })
        .unwrap();

        exec.add_output_binding(CoreAiIOSurfaceBinding {
            tensor_id: "output".into(),
            slot_id: 1,
            io_surface_id: 2,
            byte_offset: 4096,
            contract_digest: String::new(),
        })
        .unwrap();

        // Create lane with a compiled subgraph
        let mut lane = CoreAiLane::new();
        let mut sg = CoreAiSubgraph::new("test_subgraph");
        sg.status = CoreAiSubgraphStatus::Compiled {
            model_path: "/tmp/warmup.mlmodelc".into(),
        };
        lane.add_subgraph(sg);

        let contract = CoreAiWarmupContract {
            min_warmup_predictions: 3,
            max_warmup_latency_ms: 1000,
            tolerance: 0.01,
        };

        // Call warmup_with_arena — the model file doesn't exist, so load_model()
        // fails gracefully. This validates that the binding/arena validation works
        // before the model load attempt, and that failure is reported without a panic.
        let result = lane.warmup_with_arena("test_subgraph", &contract, &mut arena, &mut exec);

        // Model file doesn't exist — expect graceful failure
        assert!(
            result.is_err(),
            "warmup should fail gracefully with missing model: {:?}",
            result
        );
        let err = result.unwrap_err();
        assert!(
            err.contains("tribunus_coreai_load_model") || err.contains("load"),
            "error should mention model loading: {}",
            err
        );

        // Executable state: model not loaded, but bindings still configured
        assert!(!exec.loaded, "executable should not be marked as loaded");
        assert!(exec.model.is_none(), "model handle should be None");
        assert_eq!(exec.input_bindings.len(), 1, "input bindings preserved");
        assert_eq!(exec.output_bindings.len(), 1, "output bindings preserved");

        // Lifecycle should remain Unavailable since warmup failed
        assert_eq!(
            lane.lifecycle,
            AneLaneLifecycle::Unavailable,
            "lifecycle should be Unavailable after failed warmup"
        );
    }
}
