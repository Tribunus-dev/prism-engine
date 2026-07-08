//! Metal region encoder — implements [`RegionEncoder`] for Metal command buffers.
//!
//! Walks an [`ExecutionRegion`]'s ops, looks up PSOs from the cache, sets up
//! compute encoders, binds buffers, and dispatches threadgroups.

use std::collections::HashMap;
use metal::Device;

use crate::execution_plan::ExecutionRegion;
use crate::execution_plan::pso_cache::PsoCache;
use crate::execution_plan::region_encoder::{
    RegionEncoder, RegionExecutionHandle, RegionEncoderError,
};

/// Metal compute pipeline state, alias for the metal-rs type.
pub type MetalPipelineState = metal::ComputePipelineState;

/// Metal region encoder — owns the device, command queue, and buffer registry.
pub struct MetalRegionEncoder {
    device: Device,
    command_queue: metal::CommandQueue,
    buffers: HashMap<String, metal::Buffer>,
}

impl MetalRegionEncoder {
    /// Create a new metal region encoder from the given device.
    pub fn new(device: &Device) -> Self {
        let queue = device.new_command_queue();
        Self {
            device: device.clone(),
            command_queue: queue,
            buffers: HashMap::new(),
        }
    }

    /// Access the device (for test helpers and composite use).
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Register a buffer by name so the encoder can bind it during dispatch.
    pub fn register_buffer(&mut self, name: &str, buffer: metal::Buffer) {
        self.buffers.insert(name.to_string(), buffer);
    }

    /// Remove a registered buffer.
    pub fn unregister_buffer(&mut self, name: &str) -> Option<metal::Buffer> {
        self.buffers.remove(name)
    }
}

impl RegionEncoder for MetalRegionEncoder {
    type PipelineState = MetalPipelineState;

    #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
    fn new(device: &Device) -> Self {
        MetalRegionEncoder::new(device)
    }

    fn encode_region(
        &mut self,
        region: &ExecutionRegion,
        pso_cache: &mut dyn PsoCache<PipelineState = Self::PipelineState>,
    ) -> Result<RegionExecutionHandle, RegionEncoderError> {
        let cb = self.command_queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        let mut ops_encoded = 0u32;

        for op in &region.ops {
            // Look up or compile the pipeline state for this kernel specialisation.
            let pso = pso_cache
                .get_or_create(&op.specialization, &Default::default())
                .map_err(|e| format!("PSO error for op {}: {:?}", op.op_id, e))?;
            encoder.set_compute_pipeline_state(&pso);

            // Bind each buffer to its slot from the encoder's internal registry.
            for binding in &op.bindings {
                if let Some(buf) = self.buffers.get(&binding.buffer_id) {
                    encoder.set_buffer(binding.slot as u64, Some(buf), binding.offset);
                }
            }

            // Dispatch the threadgroups.
            let tg = metal::MTLSize::new(
                op.dispatch_shape.threadgroup_m as u64,
                op.dispatch_shape.threadgroup_n as u64,
                op.dispatch_shape.threadgroup_p as u64,
            );
            let grid = metal::MTLSize::new(
                op.dispatch_shape.grid_x as u64,
                op.dispatch_shape.grid_y as u64,
                op.dispatch_shape.grid_z as u64,
            );
            encoder.dispatch_thread_groups(grid, tg);
            ops_encoded += 1;
        }

        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();

        Ok(RegionExecutionHandle {
            region_id: region.region_id.clone(),
            command_buffer_index: 0,
            compute_encoder_count: 1,
            ops_encoded,
            barriers_inserted: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that region encoder construction succeeds on Metal hardware.
    /// Uses cfg(target_os = "macos") — the module gate already provides this,
    /// but the per-item guard makes the intention explicit.
    #[test]
    #[cfg_attr(not(target_os = "macos"), ignore = "Metal is macOS-only")]
    fn create_encoder() {
        let device = Device::system_default()
            .expect("Metal device should be available");
        let _encoder = MetalRegionEncoder::new(&device);
    }
}
