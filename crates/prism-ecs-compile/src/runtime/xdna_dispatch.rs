//! XDNA route dispatcher — native AMD NPU dispatch for loaded CImage XDNA
//! artifacts.
//!
//! This module owns the canonical authority for dispatching an AOT fused
//! step onto an AMD XDNA device through the `prism_amd_npu_runtime`
//! contract. It implements both the generic [`RouteDispatch`] trait (so
//! the kernel route dispatcher can compose it with ANE / CPU / Metal
//! backends) and the [`super::kernel_dispatch::XdnaRouteBackend`] contract
//! (so the same code path is used whether the route is invoked through
//! `replay_aot_routed` or the AMD-NPU-specific entry point).
//!
//! Like the kernel route dispatcher, this type is a borrowed view over a
//! [`super::model::RuntimeModel`] — it does not own the model data and
//! never mutates it.

use std::collections::HashMap;

use prism_amd_npu_runtime::{XdnaArtifact, XdnaExecutionPhase, XdnaRuntime};
use prism_spatial_ir::BufferStorage;
use prism_spatial_ir::RouteDispatch;
use prism_spatial_ir::ResolvedBuffer;
use prism_spatial_ir::execution_plan::FusedScheduleStep;

use super::kernel_dispatch::XdnaRouteBackend;
use super::model::RuntimeModel;

/// Native XDNA route for a loaded CImage. This keeps XDNA execution inside
/// the same dependency-aware scheduler as CPU, GPU, and ANE work while
/// leaving device submission to the Prism-owned `XdnaDevice` contract.
pub struct CImageXdnaRouteDispatcher<'a, D> {
    pub model: &'a RuntimeModel,
    pub runtime: XdnaRuntime,
    pub device: D,
    /// Phase used when the scheduler dispatches this XDNA island. Callers
    /// switch this before replaying a prefill or decode schedule.
    pub phase: XdnaExecutionPhase,
}

impl<'a, D> CImageXdnaRouteDispatcher<'a, D> {
    pub fn new(model: &'a RuntimeModel, device: D) -> Result<Self, String> {
        if model.xdna_artifacts.is_empty() {
            return Err("CImage has no native XDNA artifacts".into());
        }
        Ok(Self {
            model,
            runtime: XdnaRuntime::new(),
            device,
            phase: XdnaExecutionPhase::Decode,
        })
    }

    pub fn set_phase(&mut self, phase: XdnaExecutionPhase) {
        self.phase = phase;
    }

    fn unsupported(route: &str) -> Result<(), String> {
        Err(format!("XDNA route cannot dispatch {route} work"))
    }

    pub(super) fn payloads_for_inputs(&self, inputs: &[ResolvedBuffer]) -> HashMap<String, Vec<u8>> {
        inputs
            .iter()
            .filter_map(|input| {
                input
                    .payload
                    .clone()
                    .or_else(|| self.model.tensors.get(&input.name).cloned())
                    .map(|payload| (input.name.clone(), payload))
            })
            .collect()
    }
}

impl<D: prism_amd_npu_runtime::XdnaCommandSubmitter> RouteDispatch
    for CImageXdnaRouteDispatcher<'_, D>
{
    fn ensure_residency(&mut self, _window_id: usize) -> Result<(), String> {
        Ok(())
    }

    fn dispatch_ane_planar(
        &mut self,
        _: &FusedScheduleStep,
        _: &[ResolvedBuffer],
        _: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        Self::unsupported("ANE planar")
    }

    fn dispatch_ane_matrix(
        &mut self,
        _: &FusedScheduleStep,
        _: &[ResolvedBuffer],
        _: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        Self::unsupported("ANE matrix")
    }

    fn dispatch_accelerate(
        &mut self,
        _: &FusedScheduleStep,
        _: &[ResolvedBuffer],
        _: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        Self::unsupported("Accelerate")
    }

    fn dispatch_metal(
        &mut self,
        _: &FusedScheduleStep,
        _: &[ResolvedBuffer],
        _: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        Self::unsupported("Metal")
    }

    fn dispatch_cpu(
        &mut self,
        _: &FusedScheduleStep,
        _: &[ResolvedBuffer],
        _: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        Self::unsupported("CPU")
    }

    fn dispatch_xdna(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        let artifact_name = step.model_id.as_deref().unwrap_or("main");
        let artifact: &XdnaArtifact = self
            .model
            .xdna_artifact(artifact_name)
            .or_else(|| self.model.xdna_artifacts.get("main"))
            .ok_or_else(|| format!("no XDNA artifact for route {artifact_name}"))?;
        let mut payloads = self.payloads_for_inputs(inputs);
        // The native matmul lowering uses stable operand buffers A/B/C while
        // scheduler bindings use source-model tensor names. Bind by operand
        // position as the canonical fallback, then retain named bindings for
        // artifacts that expose model-native buffer identifiers.
        for (buffer_id, input) in ["A", "B", "C"].into_iter().zip(inputs.iter()) {
            if let Some(payload) = input.payload.as_ref() {
                payloads
                    .entry(buffer_id.into())
                    .or_insert_with(|| payload.clone());
            }
        }
        let command = artifact
            .command_buffer()
            .map_err(|error| format!("build XDNA command buffer: {error}"))?;
        self.runtime.submit_bound_artifact_phase_with_payloads(
            artifact,
            &command,
            self.phase,
            &payloads,
            &mut self.device,
        )?;
        if let Some(payload) =
            self.runtime
                .download_buffer(&artifact.program, "C", &mut self.device)?
        {
            if let Some(output) = outputs.first_mut() {
                output.byte_length = payload.len();
                output.payload = Some(payload);
                output.storage = BufferStorage::RuntimeOwned;
                output.zero_copy = false;
                output.file_offset = None;
            }
        }
        Ok(())
    }

    fn synchronize(&mut self, _: &FusedScheduleStep) -> Result<(), String> {
        Ok(())
    }
}

impl<D: prism_amd_npu_runtime::XdnaCommandSubmitter> XdnaRouteBackend
    for CImageXdnaRouteDispatcher<'_, D>
{
    fn dispatch_xdna(
        &mut self,
        step: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        RouteDispatch::dispatch_xdna(self, step, inputs, outputs)
    }
}
