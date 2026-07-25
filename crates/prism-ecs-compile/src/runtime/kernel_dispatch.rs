//! Kernel route dispatcher — composition of CPU / Accelerate / Metal / ANE /
//! XDNA backends behind the [`RouteDispatch`] trait.
//!
//! This module owns the canonical authority for fanning an AOT fused-step
//! schedule out to the right concrete backend at replay time. The
//! [`KernelRouteDispatcher`] is the production implementation of
//! [`RouteDispatch`] that [`super::UnifiedRuntime::replay_aot_routed`]
//! drives; it holds borrowed handles to a [`super::model::RuntimeModel`]
//! and the various backend implementations.
//!
//! The dispatcher does not own the dispatch helpers themselves — those live
//! on [`super::unified::UnifiedRuntime`] for ANE and on the AMD NPU runtime
//! for XDNA. The dispatcher's job is to translate the trait's per-route
//! methods into the right backend call (and the right kernel name in the
//! loaded descriptor set).

use std::collections::HashMap;

use prism_ecs_kernel::{BackendKind, KernelBackend, KernelDispatchRequest};
use prism_spatial_ir::RouteDispatch;
use prism_spatial_ir::ResolvedBuffer;
use prism_spatial_ir::execution_plan::FusedScheduleStep;

use super::ane_backend::AneRouteBackend;
use super::model::RuntimeModel;

/// XDNA-specific route contract used by [`super::unified::UnifiedRuntime`]
/// when replaying on an AMD NPU device. The dispatcher treats this as
/// optional — when no XDNA backend is configured, XDNA steps fail closed.
pub trait XdnaRouteBackend {
    fn dispatch_xdna(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String>;
}

/// Runtime composition of the concrete CPU/Accelerate/Metal kernel backends
/// and the ANE IOSurface backend. This is the production implementation of
/// [`RouteDispatch`] used by [`super::UnifiedRuntime::replay_aot_routed`].
pub struct KernelRouteDispatcher<'a> {
    pub model: &'a RuntimeModel,
    pub ane: &'a mut dyn AneRouteBackend,
    pub accelerate: &'a dyn KernelBackend,
    pub metal: &'a dyn KernelBackend,
    pub cpu: &'a dyn KernelBackend,
    pub xdna: Option<&'a mut dyn XdnaRouteBackend>,
}

impl KernelRouteDispatcher<'_> {
    fn dispatch_kernel(
        &self,
        backend: &dyn KernelBackend,
        expected_backend: BackendKind,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        let names = kernel_names_for_backend(&self.model.kernel_descriptors, expected_backend);
        let name = names
            .get(step.step_id)
            .ok_or_else(|| format!("no compiled kernel for AOT step {}", step.step_id))?;
        let artifact = self
            .model
            .kernel_artifact(name)
            .map_err(|error| error.to_string())?;
        let input_payloads = inputs
            .iter()
            .map(|buffer| {
                buffer
                    .payload
                    .clone()
                    .or_else(|| self.model.tensors.get(&buffer.name).cloned())
                    .ok_or_else(|| format!("no payload for runtime binding '{}'", buffer.name))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let bindings = artifact
            .manifest
            .kernels
            .first()
            .map(|descriptor| descriptor.binding_signature.clone())
            .unwrap_or_default();
        let result = backend
            .dispatch(&KernelDispatchRequest {
                artifact,
                inputs: input_payloads,
                bindings,
            })
            .map_err(|error| error.to_string())?;
        if result.outputs.len() != outputs.len() {
            return Err(format!(
                "backend returned {} outputs for {} scheduled bindings",
                result.outputs.len(),
                outputs.len()
            ));
        }
        for (binding, payload) in outputs.iter_mut().zip(result.outputs.iter()) {
            if binding.byte_length != 0 && binding.byte_length != payload.len() {
                return Err(format!(
                    "backend output for '{}' has {} bytes; binding requires {}",
                    binding.name,
                    payload.len(),
                    binding.byte_length
                ));
            }
            binding.byte_length = payload.len();
            binding.payload = Some(payload.clone());
        }
        Ok(())
    }
}

/// Return the descriptors of all compiled kernels for one backend, sorted
/// by kernel name. This is the canonical lookup the kernel route dispatcher
/// uses to pick a kernel for an AOT step; tests rely on the same ordering.
pub fn kernel_names_for_backend(
    descriptors: &HashMap<String, prism_ecs_kernel::KernelDescriptor>,
    backend: BackendKind,
) -> Vec<&String> {
    let mut names: Vec<&String> = descriptors
        .iter()
        .filter_map(|(name, descriptor)| (descriptor.backend == backend).then_some(name))
        .collect();
    names.sort();
    names
}

impl RouteDispatch for KernelRouteDispatcher<'_> {
    fn ensure_residency(&mut self, window_id: usize) -> Result<(), String> {
        let plan = self
            .model
            .execution_plan
            .as_ref()
            .ok_or_else(|| "cannot ensure residency without an execution plan".to_string())?;
        let window = plan.residency_windows.get(window_id).ok_or_else(|| {
            format!("residency window {window_id} is not present in the AOT plan")
        })?;
        let cimage_bytes = self
            .model
            .mapped_cimage
            .as_ref()
            .map(|mapped| mapped.len() as u64)
            .or_else(|| {
                std::fs::metadata(&self.model.cimage_path)
                    .ok()
                    .map(|m| m.len())
            })
            .ok_or_else(|| "cannot determine CImage size for residency validation".to_string())?;
        if window.model_bytes == 0 || window.model_bytes > cimage_bytes {
            return Err(format!(
                "residency window {window_id} requires {} bytes, CImage has {}",
                window.model_bytes, cimage_bytes
            ));
        }
        Ok(())
    }

    fn dispatch_ane_planar(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        self.ane.dispatch_planar(step, inputs, outputs)
    }

    fn dispatch_ane_matrix(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        self.ane.dispatch_matrix(step, inputs, outputs)
    }

    fn dispatch_accelerate(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        _outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        self.dispatch_kernel(self.accelerate, BackendKind::CPU, step, inputs, _outputs)
    }

    fn dispatch_metal(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        _outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        self.dispatch_kernel(self.metal, BackendKind::Metal, step, inputs, _outputs)
    }

    fn dispatch_cpu(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        _outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        self.dispatch_kernel(self.cpu, BackendKind::CPU, step, inputs, _outputs)
    }

    fn dispatch_xdna(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        self.xdna
            .as_deref_mut()
            .ok_or_else(|| "XDNA route is not configured in KernelRouteDispatcher".to_string())?
            .dispatch_xdna(step, inputs, outputs)
    }

    fn synchronize(&mut self, _step: &FusedScheduleStep) -> Result<(), String> {
        Ok(())
    }
}
