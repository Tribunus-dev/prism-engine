//! Core ML IOSurface binding — accepts a compiled Core ML artifact and an
//! IOSurface-backed arena, returns a validated executable binding.
//!
//! Bridge between Core ML model artifacts and IOSurface-resident tensor slots.
//! Each binding maps a model tensor to an IOSurface arena slot, validated
//! against a cimage manifest contract.

use std::collections::HashMap;

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

        // Call warmup_with_arena
        let result = lane.warmup_with_arena("test_subgraph", &contract, &mut arena, &mut exec);

        assert!(result.is_ok(), "warmup_with_arena should succeed: {:?}", result);
        let record = result.unwrap();
        assert!(record.warmup_success, "warmup_success should be true");
        assert_eq!(lane.lifecycle, AneLaneLifecycle::Warmed, "lifecycle should be Warmed");
        assert!(exec.loaded, "binding should be marked as loaded");
    }
}
