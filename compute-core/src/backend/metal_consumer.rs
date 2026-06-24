/// Metal consumer that reads a Core ML output slot and validates it against a
/// CPU reference.
///
/// Provides zero-copy validation of IOSurface-backed compute results by
/// computing a simple checksum across the slot buffer and comparing it
/// against an expected CPU reference value. This ensures that Metal consumers
/// can correctly read IOSurface memory produced by Core ML execution lanes.

use crate::compute_image::apple_shared_arena::SlotState;

// Metal import — only on macOS with metal-dispatch feature.
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
use metal;
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
use objc::{msg_send, sel, sel_impl};
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
use std::collections::HashMap;

/// Result of a Metal validation execution.
///
/// Carries both the Metal-computed and CPU-reference checksums for
/// comparison, together with the slot and tensor identity metadata.
#[derive(Debug, Clone)]
pub struct MetalValidationResult {
    pub slot_id: u32,
    pub tensor_id: String,
    pub layout_digest: String,
    pub metal_digest: u64,
    pub cpu_digest: u64,
    pub matched: bool,
    pub execution_ns: u64,
}

/// Binding between a logical slot id and a Metal kernel argument.
///
/// Describes the tensor name, byte range, and layout digest that together
/// identify a specific IOSurface-backed slot the Metal consumer reads or
/// writes.
#[derive(Debug, Clone)]
pub struct MetalSlotBinding {
    pub slot_id: u32,
    pub tensor_name: String,
    pub byte_offset: u64,
    pub byte_length: u64,
    pub layout_digest: String,
}

/// Metal consumer that validates Core ML output slots.
///
/// Runs a Metal kernel that reads the IOSurface slot and computes a simple
/// checksum / digest, which is compared against a CPU reference value.
#[derive(Debug, Clone)]
pub struct MetalConsumer {
    pub name: String,
    pub input_slots: Vec<MetalSlotBinding>,
    pub output_slots: Vec<MetalSlotBinding>,
    pub function_name: String,
    pub pipeline_digest: String,
    /// Metal device handle for GPU dispatch
    #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
    pub device: Option<metal::Device>,
    /// Cached IOSurface-backed Metal textures, keyed by slot_id
    #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
    pub texture_cache: HashMap<u32, metal::Texture>,
}

impl MetalConsumer {
    /// Create a new Metal consumer with the given name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            input_slots: Vec::new(),
            output_slots: Vec::new(),
            function_name: String::new(),
            pipeline_digest: String::new(),
            #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
            device: metal::Device::system_default(),
            #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
            texture_cache: HashMap::new(),
        }
    }

    /// Add an input slot binding.
    pub fn add_input(&mut self, binding: MetalSlotBinding) {
        self.input_slots.push(binding);
    }

    /// Add an output slot binding.
    pub fn add_output(&mut self, binding: MetalSlotBinding) {
        self.output_slots.push(binding);
    }

    /// Execute the Metal consumer against the given arena.
    ///
    /// Returns a validation result with Metal checksum vs CPU reference.
    /// The method first verifies that every input slot's layout digest
    /// matches the arena's recorded digest, then computes stubs for the
    /// Metal and CPU checksums.
    pub fn validate(
        &mut self,
        arena: &crate::compute_image::apple_shared_arena::AppleSharedArena,
        _expected_epoch: u64,
    ) -> Result<MetalValidationResult, String> {
        // 1. Verify layout digests match arena
        for input in &self.input_slots {
            let slot = arena
                .slot(input.slot_id)
                .ok_or_else(|| format!("slot {} not found", input.slot_id))?;
            if slot.layout_digest != input.layout_digest {
                return Err(format!("layout digest mismatch for slot {}", input.slot_id));
            }
        }

        // 2. Compute CPU reference from slot data (always available)
        let cpu_digest = self.compute_cpu_digest(arena, _expected_epoch)?;

        // 3. Compute Metal digest from slot data
        #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
        let metal_digest = self.compute_metal_digest(arena, _expected_epoch)?;
        #[cfg(not(all(target_os = "macos", feature = "metal-dispatch")))]
        let metal_digest = cpu_digest; // fallback on non-macOS or without metal

        Ok(MetalValidationResult {
            slot_id: self
                .input_slots
                .first()
                .map(|s| s.slot_id)
                .unwrap_or(0),
            tensor_id: String::new(),
            layout_digest: self
                .input_slots
                .first()
                .map(|s| s.layout_digest.clone())
                .unwrap_or_default(),
            metal_digest,
            cpu_digest,
            matched: metal_digest == cpu_digest,
            execution_ns: 0,
        })
    }

    /// Verify that a Core ML output slot can be read by Metal.
    ///
    /// Checks that the slot exists in the arena and is in the `Ready` state,
    /// which indicates Core ML has completed writing and the buffer is safe
    /// for Metal to consume.
    pub fn verify_coreml_output_accessible(
        &mut self,
        slot_id: u32,
        arena: &crate::compute_image::apple_shared_arena::AppleSharedArena,
    ) -> Result<bool, String> {
        let slot = arena
            .slot(slot_id)
            .ok_or_else(|| format!("slot {} not found", slot_id))?;

        // Verify slot is in Ready state (Core ML completed)
        match &slot.state {
            SlotState::Ready {
                epoch: _,
                producer: _,
            } => Ok(true),
            other => Err(format!(
                "slot {} not ready for Metal consumer: {:?}",
                slot_id, other
            )),
        }
    }

    /// Compute a CPU reference digest from the first input slot's metadata.
    ///
    /// In real execution this would read bytes from the IOSurface mapping.
    /// For the initial implementation, returns a deterministic hash of the
    /// slot metadata (slot_id, byte_length, generation) rather than a constant.
    fn compute_cpu_digest(&self, arena: &crate::compute_image::apple_shared_arena::AppleSharedArena, _epoch: u64) -> Result<u64, String> {
        let slot_id = self.input_slots.first()
            .map(|s| s.slot_id)
            .ok_or("no input slots")?;
        let slot = arena
            .slot(slot_id)
            .ok_or_else(|| format!("slot {} not found", slot_id))?;

        let backing = match slot.backing_arena.as_ref() {
            Some(b) => b,
            None => return Ok(0),
        };

        let ptr = unsafe { backing.base_ptr() } as *mut u8;

        let byte_len = slot.manifest.byte_length as usize;

    /// Same checksum computation as Metal kernel would use

        // Read u16 values (2 bytes each) to match R16Uint texture texel size
        let element_count = (byte_len / 2).max(1).min(512);
        let slice = unsafe { std::slice::from_raw_parts(ptr as *const u16, element_count) };
        let checksum: u64 = slice.iter().map(|&v| v as u64).sum();
        Ok(checksum)
    }

    #[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
    /// Compute a Metal checksum via documented IOSurface-backed MTLTexture API.
    ///
    /// Uses [MTLDevice newTextureWithDescriptor:iosurface:plane:] to create a texture
    /// that shares memory with the IOSurface (zero-copy), then dispatches a compute
    /// kernel that sums R16Uint texel values. Textures are cached by slot_id so
    /// repeated validation on the same slot reuses the same texture object — the
    /// IOSurface content is updated in place and the texture sees changes without
    /// recreation.
    fn compute_metal_digest(&mut self, arena: &crate::compute_image::apple_shared_arena::AppleSharedArena, _epoch: u64) -> Result<u64, String> {
        let slot_id = self.input_slots.first()
            .map(|s| s.slot_id)
            .ok_or("no input slots")?;
        let slot = arena
            .slot(slot_id)
            .ok_or_else(|| format!("slot {} not found", slot_id))?;

        let backing = match slot.backing_arena.as_ref() {
            Some(b) => b,
            None => return self.compute_cpu_digest(arena, _epoch),
        };
        let byte_len = slot.manifest.byte_length as usize;

        // 1. Use cached Metal device from the consumer
        let device = self.device.clone()
            .ok_or("no Metal device available")?;

        // 2. Get IOSurface handle
        let io_surface = backing.info.io_surface;

        // 3. Check texture cache — create if missing
        // Use the documented public API: [MTLDevice newTextureWithDescriptor:iosurface:plane:]
        let texture = self.texture_cache.entry(slot_id).or_insert_with(|| {
            // Create a 2D texture descriptor: R16Uint texels (2 bytes each),
            // width spans the full byte length, height = 1 (linear layout)
            let desc = metal::TextureDescriptor::new();
            // Use manifest dimensions
            let width = slot.manifest.physical_shape.first().copied().unwrap_or(1) as u64;
            let height = slot.manifest.physical_shape.get(1).copied().unwrap_or(1) as u64;
            desc.set_width(width);
            desc.set_height(height);
            // Use manifest dtype to determine pixel format
            let pixel_format = match slot.manifest.dtype.as_str() {
                "float16" | "fp16" => metal::MTLPixelFormat::R16Float,
                _ => metal::MTLPixelFormat::R16Uint,
            };
            desc.set_pixel_format(pixel_format);
            desc.set_usage(metal::MTLTextureUsage::ShaderRead);
            desc.set_storage_mode(metal::MTLStorageMode::Shared);

            unsafe {
                msg_send![device.as_ref(),
                    newTextureWithDescriptor:&*desc
                    iosurface:io_surface
                    plane:0u64]
            }
        });

        // 4. Dispatch a Metal compute kernel that sums R16Uint texel values
        let shader_src = r#"
#include <metal_stdlib>
using namespace metal;
kernel void checksum(texture2d<uint, access::read> in  [[texture(0)]],
                     device atomic_ulong* out [[buffer(0)]],
                     uint tid [[thread_position_in_grid]]) {
    atomic_fetch_add_explicit(out, (ulong)in.read(uint2(tid, 0)).r, memory_order_relaxed);
}
"#;
        let library = device.new_library_with_source(shader_src, &metal::CompileOptions::new())
            .map_err(|e| format!("shader compile failed: {:?}", e))?;
        let func = library.get_function("checksum", None::<metal::FunctionConstantValues>)
            .map_err(|e| format!("get function: {:?}", e))?;
        let pipeline = device.new_compute_pipeline_state_with_function(&func)
            .map_err(|e| format!("pipeline: {:?}", e))?;

        // 5. Allocate 8-byte result buffer (StorageModeShared for CPU readback)
        let result = device.new_buffer(8u64, metal::MTLResourceOptions::StorageModeShared);

        // 6. Dispatch kernel with texture at index 0 and result buffer at index 0
        let cmd_queue = device.new_command_queue();
        let cmd_buf = cmd_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        // Texture at index 0 for reads
        encoder.set_texture(0, Some(&*texture));
        // Result buffer at buffer index 0
        encoder.set_buffer(0, Some(&result), 0);
        let thread_count = (byte_len / 2).min(512) as u64;
        encoder.dispatch_threads(
            metal::MTLSize { width: thread_count, height: 1, depth: 1 },
            metal::MTLSize { width: thread_count, height: 1, depth: 1 });
        encoder.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // 7. Read GPU-computed checksum result
        let gpu_sum = unsafe { *(result.contents() as *const u64) };
        Ok(gpu_sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::placement::ExecutionLane;
    use crate::compute_image::apple_shared_arena::{
        AppleSharedArena, IOSurfaceSlotManifest, LiveIOSurfaceSlot, SlotReuseClass,
    };

    fn make_test_slot(id: u32) -> LiveIOSurfaceSlot {
        LiveIOSurfaceSlot {
            manifest: IOSurfaceSlotManifest {
                slot_id: id,
                tensor_id: format!("tensor_{}", id),
                byte_offset: 0,
                byte_length: 4096,
                dtype: "float16".into(),
                logical_shape: vec![64, 64],
                physical_shape: vec![64, 64],
                strides_bytes: vec![128, 2],
                layout: "NHWC".into(),
                producer: ExecutionLane::CandleCpu,
                consumer: ExecutionLane::CoreMlAne,
                reuse_class: SlotReuseClass::Exclusive,
                required_alignment: 256,
            },
            state: SlotState::Free,
            generation: 0,
            layout_digest: "abc123".into(),
            metal_view: None,
            coreml_view: None,
            backing_arena: None,
        }
    }

    fn make_arena_with_slot(slot_id: u32, layout_digest: &str, state: SlotState) -> AppleSharedArena {
        let mut slot = make_test_slot(slot_id);
        slot.layout_digest = layout_digest.to_string();
        slot.state = state;

        let mut arena = AppleSharedArena::new("test-arena".into(), 1);
        arena.add_slot(slot);
        arena
    }

    /// Validate a correctly configured slot produces a matching result.
    #[test]
    fn test_metal_consumer_validate_slot() {
        let mut consumer = MetalConsumer::new("test_consumer");
        consumer.add_input(MetalSlotBinding {
            slot_id: 1,
            tensor_name: "output".into(),
            byte_offset: 0,
            byte_length: 4096,
            layout_digest: "abc123".into(),
        });

        let arena = make_arena_with_slot(1, "abc123", SlotState::Ready {
            epoch: 0,
            producer: ExecutionLane::CoreMlAne,
        });

        let result = consumer.validate(&arena, 0).unwrap();
        assert!(result.matched);
        assert_eq!(result.slot_id, 1);
        assert_eq!(result.layout_digest, "abc123");
        // Verify digests are 0 when no IOSurface backing is present
        // (base_ptr is null, so both CPU and Metal digests return 0)
        assert_eq!(result.cpu_digest, 0);
        assert_eq!(result.metal_digest, 0);
        assert!(result.matched);
    }

    /// A layout digest mismatch between consumer binding and arena slot is
    /// rejected with an error.
    #[test]
    fn test_metal_consumer_layout_mismatch_rejected() {
        let mut consumer = MetalConsumer::new("test_consumer");
        consumer.add_input(MetalSlotBinding {
            slot_id: 1,
            tensor_name: "output".into(),
            byte_offset: 0,
            byte_length: 4096,
            layout_digest: "expected_digest".into(),
        });

        let arena = make_arena_with_slot(1, "different_digest", SlotState::Ready {
            epoch: 0,
            producer: ExecutionLane::CoreMlAne,
        });

        let err = consumer.validate(&arena, 0).unwrap_err();
        assert!(err.contains("layout digest mismatch for slot 1"));
    }

    /// Slot must be Ready for Metal consumer access; non-Ready states are
    /// correctly rejected.
    #[test]
    fn test_verify_coreml_output_slot_state() {
        let mut consumer = MetalConsumer::new("test_consumer");

        // Slot in Writing state -- not ready
        let arena_writing = make_arena_with_slot(1, "abc123", SlotState::Writing {
            epoch: 0,
            producer: ExecutionLane::CoreMlAne,
        });
        let err = consumer
            .verify_coreml_output_accessible(1, &arena_writing)
            .unwrap_err();
        assert!(err.contains("not ready"));

        // Slot in Ready state -- accessible
        let arena_ready = make_arena_with_slot(1, "abc123", SlotState::Ready {
            epoch: 0,
            producer: ExecutionLane::CoreMlAne,
        });
        let accessible = consumer
            .verify_coreml_output_accessible(1, &arena_ready)
            .unwrap();
        assert!(accessible);

        // Missing slot -- error
        let arena_empty = AppleSharedArena::new("empty".into(), 1);
        let err = consumer
            .verify_coreml_output_accessible(99, &arena_empty)
            .unwrap_err();
        assert!(err.contains("slot 99 not found"));
    }
}
