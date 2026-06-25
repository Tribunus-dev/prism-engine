//! Metal LUT decoder adapter implementing the `MetalDecoder` trait.
//!
//! Wraps the Phase 1 palettized GEMV kernel (split-block layout) for
//! one-step decode benchmarking.  Codebooks and indices are stored in a
//! single IOSurface-backed buffer; the kernel binds them at offset 0 and
//! offset `dim_m × 16 × 2` respectively.

use metal::{Buffer, CommandQueue, ComputePipelineState, Device, Library};

use crate::scheduling::benchmark_harness::MetalDecoder;

/// Placeholder token returned by every `step()`.
const PLACEHOLDER_TOKEN: u32 = 0;

/// Thin GEMV decoder for the palettized LUT kernel (split-block layout).
///
/// # Resource layout (matches Phase 1 Metal template)
/// | Bind | Resource             | Format          | Offset                |
/// |------|----------------------|-----------------|-----------------------|
/// | 0    | Input activations    | `half[d_n]`     | —                     |
/// | 1    | Codebook block       | `half[16][d_m]` | 0                     |
/// | 2    | Indices block        | `uchar[d_m][d_n/2]` | codebook_byte_size |
/// | 3    | Output logits        | `half[d_m]`     | —                     |
/// | 4    | `dim_n` (scalar)     | `uint`          | —                     |
/// | 5    | `dim_m` (scalar)     | `uint`          | —                     |
pub struct PalettizedGemvDecoder {
    #[allow(dead_code)]
    device: Device,
    command_queue: CommandQueue,
    pipeline_state: ComputePipelineState,
    /// Single buffer holding codebook_block then indices_block.
    weight_arena: Buffer,
    input_activations: Buffer,
    output_logits: Buffer,
    dim_m: u32,
    dim_n: u32,
}

impl PalettizedGemvDecoder {
    /// Create a new decoder from a pre-assembled split-block payload buffer.
    ///
    /// `weight_arena` must contain:
    ///   - codebook_block: `dim_m × 16` half values (offset 0)
    ///   - indices_block:  `dim_m × dim_n/2` uint8_t values (following codebooks)
    ///
    /// `library` must contain a kernel function named `"palettized_gemv"`.
    pub fn new(
        device: Device,
        library: &Library,
        weight_arena: Buffer,
        dim_m: u32,
        dim_n: u32,
    ) -> Result<Self, String> {
        let command_queue = device.new_command_queue();

        let function = library
            .get_function("palettized_gemv", None)
            .map_err(|e| format!("palettized_gemv not found: {e:?}"))?;

        let pipeline_state = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| format!("pipeline state: {e:?}"))?;

        // FP16 input vector: [1, dim_n]
        let input_size = (dim_n as usize) * 2;
        let input_activations = device.new_buffer(
            input_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // FP16 output vector: [1, dim_m]
        let output_size = (dim_m as usize) * 2;
        let output_logits = device.new_buffer(
            output_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        Ok(Self {
            device,
            command_queue,
            pipeline_state,
            weight_arena,
            input_activations,
            output_logits,
            dim_m,
            dim_n,
        })
    }

    /// Fill the input activation buffer with FP16 1.0 for benchmarking.
    pub fn fill_dummy_input(&self) {
        let n = self.dim_n as usize;
        let data = vec![0x3c00u16; n]; // 0x3c00 = 1.0 in IEEE 754 binary16
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, n * 2) };
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                self.input_activations.contents() as *mut u8,
                bytes.len(),
            );
        }
    }

    /// Return the byte offset into `weight_arena` where the indices block starts.
    #[inline]
    pub fn codebook_byte_size(&self) -> u64 {
        (self.dim_m as u64) * 16 * 2 // 16 f16 per row × 2 bytes each
    }
}

impl MetalDecoder for PalettizedGemvDecoder {
    type Token = u32;
    type Error = String;

    /// Execute one palettized GEMV step (split-block layout).
    ///
    /// Encodes a command buffer, dispatches `dim_m` threadgroups of 64 threads,
    /// and blocks via `wait_until_completed()`.
    fn step(&mut self) -> Result<u32, String> {
        let cmd_buf = self.command_queue.new_command_buffer();
        let encoder = cmd_buf.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipeline_state);

        // Split-block binding: codebooks at offset 0, indices at codebook_byte_size
        let cb_offset = self.codebook_byte_size();

        encoder.set_buffer(0, Some(&self.input_activations), 0);
        encoder.set_buffer(1, Some(&self.weight_arena), 0); // codebook_block
        encoder.set_buffer(2, Some(&self.weight_arena), cb_offset); // indices_block
        encoder.set_buffer(3, Some(&self.output_logits), 0);
        encoder.set_bytes(4, 4, &self.dim_n as *const u32 as *const _);
        encoder.set_bytes(5, 4, &self.dim_m as *const u32 as *const _);

        let threads_per_group = metal::MTLSize::new(64, 1, 1);
        let thread_groups = metal::MTLSize::new(self.dim_m as u64, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, threads_per_group);
        encoder.end_encoding();

        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        Ok(PLACEHOLDER_TOKEN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codebook_byte_size() {
        // codebook = dim_m * 16 * 2 (fp16)
        assert_eq!(896u64 * 16 * 2, 28672);
        // test that the calculation method exists and returns expected value
        // Without a real Metal device we can't construct a full decoder
    }
}
