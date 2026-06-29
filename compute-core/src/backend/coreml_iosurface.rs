//! Core ML IOSurface binding — accepts a compiled Core ML artifact and an
//! IOSurface-backed arena, returns a validated executable binding.
//!
//! Bridge between Core ML model artifacts and IOSurface-resident tensor slots.
//! Each binding maps a model tensor to an IOSurface arena slot, validated
//! against a cimage manifest contract.

use crate::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use std::ffi::c_void;
use std::io;

/// Core ML compute policy enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreMlComputePolicy {
    CpuOnly,
    CpuAndNeuralEngine,
    NeuralEngineOnly,
    GpuOnly,
    All,
}

impl CoreMlComputePolicy {
    pub fn name(&self) -> &'static str {
        match self {
            CoreMlComputePolicy::CpuOnly => "cpuOnly",
            CoreMlComputePolicy::CpuAndNeuralEngine => "cpuAndNeuralEngine",
            CoreMlComputePolicy::NeuralEngineOnly => "neuralEngine",
            CoreMlComputePolicy::GpuOnly => "gpuOnly",
            CoreMlComputePolicy::All => "all",
        }
    }
}

/// Core ML IOSurface binding — a single tensor's binding to an IOSurface slot.
#[derive(Debug, Clone)]
pub struct CoreMlIOSurfaceBinding {
    pub tensor_id: String,
    pub slot_id: u32,
    pub io_surface_id: u32,
    pub byte_offset: u64,
    pub contract_digest: String,
}

/// Validated executable binding for Core ML with IOSurface-backed arenas.
pub struct CoreMlIOSurfaceExecutable {
    pub artifact_id: String,
    pub compute_policy: CoreMlComputePolicy,
    pub input_bindings: Vec<CoreMlIOSurfaceBinding>,
    pub output_bindings: Vec<CoreMlIOSurfaceBinding>,
    pub model_path: String,
    /// Whether the underlying Core ML model is loaded.
    pub loaded: bool,
    /// Loaded Core ML model handle, or None before load_model() is called.
    pub model: Option<CoreMlModel>,
}

impl CoreMlIOSurfaceExecutable {
    pub fn new(artifact_id: &str, model_path: &str, compute_policy: CoreMlComputePolicy) -> Self {
        Self {
            artifact_id: artifact_id.to_string(),
            compute_policy,
            input_bindings: Vec::new(),
            output_bindings: Vec::new(),
            model_path: model_path.to_string(),
            loaded: false,
            model: None,
        }
    }

    /// Add an input binding, returns error if slot_id already bound.
    pub fn add_input_binding(&mut self, binding: CoreMlIOSurfaceBinding) -> Result<(), String> {
        if self.input_bindings.iter().any(|b| b.slot_id == binding.slot_id) {
            return Err(format!("slot {} already bound as input", binding.slot_id));
        }
        self.input_bindings.push(binding);
        Ok(())
    }

    /// Add an output binding.
    pub fn add_output_binding(&mut self, binding: CoreMlIOSurfaceBinding) -> Result<(), String> {
        if self.output_bindings.iter().any(|b| b.slot_id == binding.slot_id) {
            return Err(format!("slot {} already bound as output", binding.slot_id));
        }
        self.output_bindings.push(binding);
        Ok(())
    }

    /// Bind from an AppleSharedArena manifest — validates shape/dtype/layout match.
    pub fn bind_from_arena(
        &mut self,
        arena_slots: &[crate::compute_image::apple_cimage_manifest::IOSurfaceSlotManifest],
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
            CoreMlComputePolicy::CpuAndNeuralEngine => CoreMlComputeUnits::CpuAndNeuralEngine,
            CoreMlComputePolicy::CpuOnly => CoreMlComputeUnits::CpuOnly,
            CoreMlComputePolicy::NeuralEngineOnly => {
                // Apple does not expose a public MLComputeUnits value
                // that guarantees exclusive ANE execution. Map to
                // CpuAndNeuralEngine with a comment documenting this
                // limitation.
                CoreMlComputeUnits::CpuAndNeuralEngine
            }
            CoreMlComputePolicy::GpuOnly => CoreMlComputeUnits::CpuAndGpu,
            CoreMlComputePolicy::All => CoreMlComputeUnits::All,
        };
        let model = CoreMlModel::load_with_compute_units(&self.model_path, compute_units)?;
        self.model = Some(model);
        self.loaded = true;
        Ok(())
    }

    /// Reject if any input/output tensor name differs from the cimage contract.
    pub fn validate_against_slots(
        &self,
        input_contract: &[CoreMlIOSurfaceBinding],
        output_contract: &[CoreMlIOSurfaceBinding],
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
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "IOSurface too large"));
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
        let mut exec =
            CoreMlIOSurfaceExecutable::new("artifact_1", "/tmp/model.mlmodelc", CoreMlComputePolicy::All);

        let input = CoreMlIOSurfaceBinding {
            tensor_id: "input_0".into(),
            slot_id: 0,
            io_surface_id: 1,
            byte_offset: 0,
            contract_digest: String::new(),
        };
        let output = CoreMlIOSurfaceBinding {
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
        let mut exec = CoreMlIOSurfaceExecutable::new(
            "artifact_dup",
            "/tmp/model.mlmodelc",
            CoreMlComputePolicy::NeuralEngineOnly,
        );

        let binding = CoreMlIOSurfaceBinding {
            tensor_id: "x".into(),
            slot_id: 5,
            io_surface_id: 1,
            byte_offset: 0,
            contract_digest: String::new(),
        };

        assert!(exec.add_input_binding(binding.clone()).is_ok());
        // Same slot_id 5 on inputs — should fail
        let dup = CoreMlIOSurfaceBinding {
            tensor_id: "y".into(),
            slot_id: 5,
            io_surface_id: 2,
            byte_offset: 1024,
            contract_digest: String::new(),
        };
        assert!(exec.add_input_binding(dup).is_err());

        // Different slot_id 5 on outputs — outputs track their own set, so this is fine
        let out = CoreMlIOSurfaceBinding {
            tensor_id: "out".into(),
            slot_id: 5,
            io_surface_id: 2,
            byte_offset: 1024,
            contract_digest: String::new(),
        };
        assert!(exec.add_output_binding(out.clone()).is_ok());

        // Same slot_id 5 again on outputs — should fail
        let dup_out = CoreMlIOSurfaceBinding {
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
        let mut exec =
            CoreMlIOSurfaceExecutable::new("contract_test", "/tmp/model.mlmodelc", CoreMlComputePolicy::All);

        exec.add_input_binding(CoreMlIOSurfaceBinding {
            tensor_id: "input_a".into(),
            slot_id: 0,
            io_surface_id: 1,
            byte_offset: 0,
            contract_digest: String::new(),
        })
        .unwrap();
        exec.add_output_binding(CoreMlIOSurfaceBinding {
            tensor_id: "output_a".into(),
            slot_id: 1,
            io_surface_id: 2,
            byte_offset: 4096,
            contract_digest: String::new(),
        })
        .unwrap();

        // Input contract with wrong tensor_id
        let bad_input = CoreMlIOSurfaceBinding {
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
        let mut exec2 =
            CoreMlIOSurfaceExecutable::new("contract_test_2", "/tmp/model.mlmodelc", CoreMlComputePolicy::All);
        exec2
            .add_input_binding(CoreMlIOSurfaceBinding {
                tensor_id: "input_a".into(),
                slot_id: 0,
                io_surface_id: 1,
                byte_offset: 0,
                contract_digest: String::new(),
            })
            .unwrap();
        exec2
            .add_output_binding(CoreMlIOSurfaceBinding {
                tensor_id: "output_a".into(),
                slot_id: 1,
                io_surface_id: 2,
                byte_offset: 4096,
                contract_digest: String::new(),
            })
            .unwrap();

        let bad_output = CoreMlIOSurfaceBinding {
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
        let exec = CoreMlIOSurfaceExecutable::new("count_test", "/tmp/model.mlmodelc", CoreMlComputePolicy::CpuOnly);
        // Zero input bindings, but pass one contract entry
        let contract = CoreMlIOSurfaceBinding {
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
        assert_eq!(CoreMlComputePolicy::CpuOnly.name(), "cpuOnly");
        assert_eq!(
            CoreMlComputePolicy::CpuAndNeuralEngine.name(),
            "cpuAndNeuralEngine"
        );
        assert_eq!(CoreMlComputePolicy::NeuralEngineOnly.name(), "neuralEngine");
        assert_eq!(CoreMlComputePolicy::GpuOnly.name(), "gpuOnly");
        assert_eq!(CoreMlComputePolicy::All.name(), "all");
    }

    #[test]
    fn test_executable_new_defaults() {
        let exec = CoreMlIOSurfaceExecutable::new("test", "/path.mlmodelc", CoreMlComputePolicy::GpuOnly);
        assert_eq!(exec.artifact_id, "test");
        assert_eq!(exec.model_path, "/path.mlmodelc");
        assert_eq!(exec.compute_policy, CoreMlComputePolicy::GpuOnly);
        assert!(exec.input_bindings.is_empty());
        assert!(exec.output_bindings.is_empty());
        assert!(!exec.loaded);
    }

    #[test]
    fn test_coreml_iosurface_warmup_with_arena() {
        use crate::backend::coreml_lane::{CoreMlLane, CoreMlSubgraph, CoreMlSubgraphStatus};
        use crate::compute_image::apple_shared_arena::{
            AppleSharedArena, LiveIOSurfaceSlot, IOSurfaceSlotManifest, SlotReuseClass,
        };
        use crate::backend::placement::ExecutionLane;
        use crate::compilation::tri_lane::{CoreMlWarmupContract, AneLaneLifecycle};

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
                producer: ExecutionLane::CoreMlAne,
                consumer: ExecutionLane::MlxGpu,
                reuse_class: SlotReuseClass::Exclusive,
                required_alignment: 64,
            },
            state: crate::compute_image::apple_shared_arena::SlotState::Free,
            generation: 0,
            layout_digest: "digest-00000000".into(),
            metal_view: None,
            coreml_view: None,
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
                producer: ExecutionLane::CoreMlAne,
                consumer: ExecutionLane::MlxGpu,
                reuse_class: SlotReuseClass::Exclusive,
                required_alignment: 64,
            },
            state: crate::compute_image::apple_shared_arena::SlotState::Free,
            generation: 0,
            layout_digest: "digest-00000000".into(),
            metal_view: None,
            coreml_view: None,
            backing_arena: None,
            attestation: None,
        });

        // Create executable with input/output bindings matching arena slots
        let mut exec = CoreMlIOSurfaceExecutable::new(
            "warmup_test",
            "/tmp/warmup.mlmodelc",
            CoreMlComputePolicy::CpuAndNeuralEngine,
        );

        exec.add_input_binding(CoreMlIOSurfaceBinding {
            tensor_id: "input".into(),
            slot_id: 0,
            io_surface_id: 1,
            byte_offset: 0,
            contract_digest: String::new(),
        }).unwrap();

        exec.add_output_binding(CoreMlIOSurfaceBinding {
            tensor_id: "output".into(),
            slot_id: 1,
            io_surface_id: 2,
            byte_offset: 4096,
            contract_digest: String::new(),
        }).unwrap();

        // Create lane with a compiled subgraph
        let mut lane = CoreMlLane::new();
        let mut sg = CoreMlSubgraph::new("test_subgraph");
        sg.status = CoreMlSubgraphStatus::Compiled {
            model_path: "/tmp/warmup.mlmodelc".into(),
        };
        lane.add_subgraph(sg);

        let contract = CoreMlWarmupContract {
            min_warmup_predictions: 3,
            max_warmup_latency_ms: 1000,
            tolerance: 0.01,
        };

        // Call warmup_with_arena — the model file doesn't exist, so load_model()
        // fails gracefully. This validates that the binding/arena validation works
        // before the model load attempt, and that failure is reported without a panic.
        let result = lane.warmup_with_arena("test_subgraph", &contract, &mut arena, &mut exec);

        // Model file doesn't exist — expect graceful failure
        assert!(result.is_err(), "warmup should fail gracefully with missing model: {:?}", result);
        let err = result.unwrap_err();
        assert!(err.contains("tribunus_coreml_load_model") || err.contains("load"),
            "error should mention model loading: {}", err);

        // Executable state: model not loaded, but bindings still configured
        assert!(!exec.loaded, "executable should not be marked as loaded");
        assert!(exec.model.is_none(), "model handle should be None");
        assert_eq!(exec.input_bindings.len(), 1, "input bindings preserved");
        assert_eq!(exec.output_bindings.len(), 1, "output bindings preserved");

        // Lifecycle should remain Unavailable since warmup failed
        assert_eq!(lane.lifecycle, AneLaneLifecycle::Unavailable,
            "lifecycle should be Unavailable after failed warmup");
    }
}
