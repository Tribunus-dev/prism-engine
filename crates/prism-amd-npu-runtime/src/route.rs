//! XDNA implementation of Prism's heterogeneous route contract.

use crate::{XdnaArtifact, XdnaDevice, XdnaExecutionPhase, XdnaRuntime};
use prism_spatial_ir::execution_plan::FusedScheduleStep;
use prism_spatial_ir::scheduler::{BufferStorage, ResolvedBuffer, RouteDispatch};
use std::collections::HashMap;

pub struct XdnaRouteExecutor<D> {
    pub runtime: XdnaRuntime,
    pub device: D,
    pub artifact: XdnaArtifact,
    pub phase: XdnaExecutionPhase,
    active_window: Option<usize>,
}

impl<D> XdnaRouteExecutor<D> {
    pub fn new(device: D, artifact: XdnaArtifact) -> Result<Self, String> {
        artifact.validate()?;
        Ok(Self {
            runtime: XdnaRuntime::new(),
            device,
            artifact,
            phase: XdnaExecutionPhase::Decode,
            active_window: None,
        })
    }

    pub fn set_phase(&mut self, phase: XdnaExecutionPhase) {
        self.phase = phase;
    }
}

impl<D: XdnaDevice> RouteDispatch for XdnaRouteExecutor<D> {
    fn ensure_residency(&mut self, window_id: usize) -> Result<(), String> {
        if self.active_window != Some(window_id) {
            self.active_window = Some(window_id);
        }
        Ok(())
    }

    fn dispatch_ane_planar(
        &mut self,
        _: &FusedScheduleStep,
        _: &[ResolvedBuffer],
        _: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        Err("XDNA executor cannot dispatch ANE planar work".into())
    }
    fn dispatch_ane_matrix(
        &mut self,
        _: &FusedScheduleStep,
        _: &[ResolvedBuffer],
        _: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        Err("XDNA executor cannot dispatch ANE matrix work".into())
    }
    fn dispatch_accelerate(
        &mut self,
        _: &FusedScheduleStep,
        _: &[ResolvedBuffer],
        _: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        Err("XDNA executor cannot dispatch Accelerate work".into())
    }
    fn dispatch_metal(
        &mut self,
        _: &FusedScheduleStep,
        _: &[ResolvedBuffer],
        _: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        Err("XDNA executor cannot dispatch Metal work".into())
    }
    fn dispatch_cpu(
        &mut self,
        _: &FusedScheduleStep,
        _: &[ResolvedBuffer],
        _: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        Err("XDNA executor cannot dispatch CPU work".into())
    }

    fn dispatch_xdna(
        &mut self,
        _: &FusedScheduleStep,
        inputs: &[ResolvedBuffer],
        outputs: &mut [ResolvedBuffer],
    ) -> Result<(), String> {
        let mut payloads = inputs
            .iter()
            .filter_map(|input| {
                input
                    .payload
                    .as_ref()
                    .map(|payload| (input.name.clone(), payload.clone()))
            })
            .collect::<HashMap<_, _>>();
        for (buffer_id, input) in ["A", "B", "C"].into_iter().zip(inputs.iter()) {
            if let Some(payload) = input.payload.as_ref() {
                payloads
                    .entry(buffer_id.into())
                    .or_insert_with(|| payload.clone());
            }
        }
        self.runtime.submit_phase_with_payloads(
            &self.artifact,
            self.phase,
            &payloads,
            &mut self.device,
        )?;
        if let Some(payload) =
            self.runtime
                .download_buffer(&self.artifact.program, "C", &mut self.device)?
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
