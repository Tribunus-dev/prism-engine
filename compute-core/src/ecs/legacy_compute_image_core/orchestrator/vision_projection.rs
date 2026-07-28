#![cfg(any(
    feature = "mlx-backend",
    feature = "prism-backend",
    feature = "prism-backend-ios"
))]
//! Metal-accelerated vision embedder projection for Gemma 4 Unified.
//!
//! Implements the three-stage vision pipeline:
//!   1. `vision_patch_embed` — patch dense projection + norm + positions
//!   2. `vision_pool_soft_tokens` — average-pool to soft tokens
//!   3. `vision_final_projection` — project to decoder embedding space
//!
//! This is Phase A (multimodal preparation + projection), separate from
//! the decoder megakernel (Phase B).

use crate::ecs::canonical::kernel_abi::KernelSemanticId;
use crate::ecs::compute_image::legacy_compute_image_runtime::megakernel::kernels::HIDDEN_DIM;
use crate::ecs::metal_backend::catalogue_source_for;
use metal::*;

/// Vision embedder projection pipeline.
pub struct VisionProjectionPipeline {
    patch_embed_pso: ComputePipelineState,
    pool_pso: ComputePipelineState,
    final_proj_pso: ComputePipelineState,
}

/// Input to the vision projection pipeline.
pub struct VisionProjectionInput {
    /// Raw patch pixel features (patch_ln1 already applied by CPU preprocessor).
    /// Shape: [num_patches × 6912] as FP16.
    pub patch_pixels: Buffer,
    /// Number of patches.
    pub num_patches: u32,
    /// Patch grid width.
    pub patches_w: u32,
    /// Patch grid height.
    pub patches_h: u32,
    /// Soft-token pooling kernel size (default: 2).
    pub soft_token_kernel: u32,
}

/// Weights for the vision embedder.
pub struct VisionEmbedderWeights {
    /// Patch dense projection weights [3840 × packed_6912] — ternary-packed uint32.
    pub patch_dense_weights: Buffer,
    /// Patch dense projection scales [3840 × groups] — FP16.
    pub patch_dense_scales: Buffer,
    /// Patch layer norm 2 weight [3840] — FP16.
    pub patch_ln2_weight: Buffer,
    /// Patch layer norm 2 bias [3840] — FP16.
    pub patch_ln2_bias: Buffer,
    /// Learned 2D position embeddings [max_positions × 2 × 3840] — FP16.
    pub pos_embedding: Buffer,
    /// Max position grid dimension.
    pub max_positions: u32,
    /// Position norm weight [3840] — FP16.
    pub pos_norm_weight: Buffer,
    /// Position norm bias [3840] — FP16.
    pub pos_norm_bias: Buffer,
    /// Final embedding projection weights [3840 × packed_3840] — ternary-packed uint32.
    pub embed_proj_weights: Buffer,
    /// Final embedding projection scales [3840 × groups] — FP16.
    pub embed_proj_scales: Buffer,
}

impl VisionProjectionPipeline {
    /// Compile the vision projection Metal shaders.
    pub fn new(device: &Device) -> Result<Self, String> {
        let shader_src =
            catalogue_source_for(&KernelSemanticId("prism.vision.projection.v1".into()))
                .ok_or_else(|| "no source for vision_projection".to_string())?;
        let library = device
            .new_library_with_source(&shader_src, &CompileOptions::new())
            .map_err(|e| format!("vision projection shader compile: {}", e))?;

        let embed_fn = library
            .get_function("vision_patch_embed", None)
            .map_err(|e| format!("vision_patch_embed not found: {e}"))?;
        let patch_embed_pso = device
            .new_compute_pipeline_state_with_function(&embed_fn)
            .map_err(|e| format!("vision_patch_embed PSO: {e}"))?;

        let pool_fn = library
            .get_function("vision_pool_soft_tokens", None)
            .map_err(|e| format!("vision_pool_soft_tokens not found: {e}"))?;
        let pool_pso = device
            .new_compute_pipeline_state_with_function(&pool_fn)
            .map_err(|e| format!("vision_pool_soft_tokens PSO: {e}"))?;

        let proj_fn = library
            .get_function("vision_final_projection", None)
            .map_err(|e| format!("vision_final_projection not found: {e}"))?;
        let final_proj_pso = device
            .new_compute_pipeline_state_with_function(&proj_fn)
            .map_err(|e| format!("vision_final_projection PSO: {e}"))?;

        Ok(Self {
            patch_embed_pso,
            pool_pso,
            final_proj_pso,
        })
    }

    /// Run the full vision projection pipeline.
    /// Returns a Metal buffer containing decoder-width embeddings
    /// shaped [num_soft_tokens × 3840] as FP16.
    pub fn project(
        &self,
        device: &Device,
        command_queue: &CommandQueue,
        input: &VisionProjectionInput,
        weights: &VisionEmbedderWeights,
    ) -> Result<(Buffer, u32), String> {
        let num_soft_tokens =
            input.num_patches / (input.soft_token_kernel * input.soft_token_kernel);

        // Allocate intermediate buffers
        let after_positions = device.new_buffer(
            (input.num_patches as u64) * HIDDEN_DIM as u64 * 2, // FP16 = 2 bytes
            MTLResourceOptions::StorageModeShared,
        );
        let soft_tokens_buf = device.new_buffer(
            (num_soft_tokens as u64) * HIDDEN_DIM as u64 * 2,
            MTLResourceOptions::StorageModeShared,
        );
        let output_buf = device.new_buffer(
            (num_soft_tokens as u64) * HIDDEN_DIM as u64 * 2,
            MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = command_queue.new_command_buffer();

        // ── Stage 1: Patch embed ──
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.patch_embed_pso);
            encoder.set_buffer(0, Some(&input.patch_pixels), 0);
            encoder.set_buffer(1, Some(&weights.patch_dense_weights), 0);
            encoder.set_buffer(2, Some(&weights.patch_dense_scales), 0);
            encoder.set_buffer(3, Some(&weights.patch_ln2_weight), 0);
            encoder.set_buffer(4, Some(&weights.patch_ln2_bias), 0);
            encoder.set_buffer(5, Some(&weights.pos_embedding), 0);
            encoder.set_buffer(6, Some(&weights.pos_norm_weight), 0);
            encoder.set_buffer(7, Some(&weights.pos_norm_bias), 0);
            // buffers 8/9 reserved for embed_proj_weights/scales (unused in this pass)
            encoder.set_buffer(10, Some(&after_positions), 0);

            // Constants via setBytes
            let constants: [u32; 5] = [
                input.num_patches,
                input.patches_w,
                input.patches_h,
                input.soft_token_kernel,
                weights.max_positions,
            ];
            encoder.set_bytes(
                11,
                std::mem::size_of_val(&constants) as u64,
                &constants as *const u32 as *const std::ffi::c_void,
            );

            let threads_per_group = MTLSize::new(256, 1, 1);
            let groups = MTLSize::new(input.num_patches as u64, 1, 1);
            encoder.dispatch_thread_groups(groups, threads_per_group);
            encoder.end_encoding();
        }

        // ── Stage 2: Soft-token pooling ──
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.pool_pso);
            encoder.set_buffer(0, Some(&after_positions), 0);
            encoder.set_buffer(1, Some(&soft_tokens_buf), 0);

            let constants: [u32; 4] = [
                input.num_patches,
                input.patches_w,
                input.patches_h,
                input.soft_token_kernel,
            ];
            encoder.set_bytes(
                2,
                std::mem::size_of_val(&constants) as u64,
                &constants as *const u32 as *const std::ffi::c_void,
            );

            let threads_per_group = MTLSize::new(256, 1, 1);
            let groups = MTLSize::new(num_soft_tokens as u64, 1, 1);
            encoder.dispatch_thread_groups(groups, threads_per_group);
            encoder.end_encoding();
        }

        // ── Stage 3: Final projection ──
        {
            let encoder = command_buffer.new_compute_command_encoder();
            encoder.set_compute_pipeline_state(&self.final_proj_pso);
            encoder.set_buffer(0, Some(&soft_tokens_buf), 0);
            encoder.set_buffer(1, Some(&weights.embed_proj_weights), 0);
            encoder.set_buffer(2, Some(&weights.embed_proj_scales), 0);
            encoder.set_buffer(3, Some(&output_buf), 0);

            encoder.set_bytes(
                4,
                std::mem::size_of::<u32>() as u64,
                &num_soft_tokens as *const u32 as *const std::ffi::c_void,
            );

            let threads_per_group = MTLSize::new(256, 1, 1);
            let groups = MTLSize::new(num_soft_tokens as u64, 1, 1);
            encoder.dispatch_thread_groups(groups, threads_per_group);
            encoder.end_encoding();
        }

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok((output_buf, num_soft_tokens))
    }
}
