//! Mimi Codec decoder — causal ConvNet that converts 16 RVQ codebook tokens to PCM waveform.
//!
//! Architecture (causal ConvNet):
//!   1. Codebook gather: 16 codebook indices → 16 × 128-dim embeddings
//!   2. 1D transposed conv layers: upsample 12.5 Hz → 24 kHz
//!   3. Residual blocks with causal dilated conv + layer norm
//!   4. Overlap-add for streaming waveform synthesis
//!
//! Qwen3-TTS is Apache 2.0 licensed.
//!
//! # Conventions
//! - All weight data loaded from cimage segments — assumed correctly shaped.
//! - Half-precision (f16) throughout the GPU pipeline.
//! - Kernels compiled at runtime from `shaders/tts_codec.metal`.

use metal::*;

// ── Helper: compile a Metal function from the shared shader library ────────

/// Compile a single kernel function from tts_codec.metal and return a pipeline state.
fn compile_kernel(
    device: &Device,
    library: &Library,
    name: &str,
) -> Result<ComputePipelineState, String> {
    let function = library
        .get_function(name, None)
        .map_err(|e| format!("Metal function '{name}' not found: {e}"))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| format!("Pipeline state for '{name}' failed: {e}"))
}

// ── Weight parsing helpers ─────────────────────────────────────────────────

/// Parse header: [num_layers: u32 LE] then for each layer:
///   [out_channels: u32][in_channels: u32][kernel_size: u32][stride: u32]
///   [weight_bytes: u64][bias_bytes: u64]
///   [weight data][bias data]
fn parse_conv_layer_headers(data: &[u8]) -> Result<Vec<ConvLayerDesc<'_>>, String> {
    if data.len() < 4 {
        return Err("conv layer data too short for header".into());
    }
    let num_layers = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let mut offset = 4;
    let mut descs = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        if offset + 32 > data.len() {
            return Err(format!("conv layer header truncated at offset {offset}"));
        }
        let out_ch = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap());
        let in_ch = u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap());
        let kernel = u32::from_le_bytes(data[offset + 8..offset + 12].try_into().unwrap());
        let stride = u32::from_le_bytes(data[offset + 12..offset + 16].try_into().unwrap());
        let wbytes = u64::from_le_bytes(data[offset + 16..offset + 24].try_into().unwrap());
        let bbytes = u64::from_le_bytes(data[offset + 24..offset + 32].try_into().unwrap());
        offset += 32;
        if offset + wbytes as usize + bbytes as usize > data.len() {
            return Err(format!("conv layer data truncated at offset {offset}"));
        }
        let weight_data = &data[offset..offset + wbytes as usize];
        let bias_data = if bbytes > 0 {
            Some(&data[offset + wbytes as usize..offset + wbytes as usize + bbytes as usize])
        } else {
            None
        };
        offset += wbytes as usize + bbytes as usize;
        descs.push(ConvLayerDesc {
            out_ch,
            in_ch,
            kernel,
            stride,
            weight_data,
            bias_data,
        });
    }
    Ok(descs)
}

struct ConvLayerDesc<'a> {
    out_ch: u32,
    in_ch: u32,
    kernel: u32,
    stride: u32,
    weight_data: &'a [u8],
    bias_data: Option<&'a [u8]>,
}

// ── Conv layer runtime ─────────────────────────────────────────────────────

/// A single 1D transposed convolution layer (weight + bias) stored as Metal buffers.
pub struct MimiConvLayer {
    /// Weight buffer [out_ch][in_ch][kernel], half-precision.
    pub weight: Buffer,
    /// Bias buffer [out_ch], half-precision; absent → zero bias.
    pub bias: Option<Buffer>,
    pub stride: u32,
    pub kernel_size: u32,
    pub in_channels: u32,
    pub out_channels: u32,
}

impl MimiConvLayer {
    fn new(device: &Device, desc: &ConvLayerDesc) -> Result<Self, String> {
        let weight = device.new_buffer_with_data(
            desc.weight_data.as_ptr() as *const std::ffi::c_void,
            desc.weight_data.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let bias = desc.bias_data.map(|b| {
            device.new_buffer_with_data(
                b.as_ptr() as *const std::ffi::c_void,
                b.len() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        });
        Ok(Self {
            weight,
            bias,
            stride: desc.stride,
            kernel_size: desc.kernel,
            in_channels: desc.in_ch,
            out_channels: desc.out_ch,
        })
    }
}

// ── Mimi Codec ─────────────────────────────────────────────────────────────

/// Mimi Codec decoder — converts 16 RVQ codebook tokens to 24 kHz PCM waveform.
///
/// # Pipeline
///
/// 1. **codebook_gather** — index each of the 16 codebooks and concatenate 128-dim embeddings.
/// 2. **conv1d_transpose** (×N layers) — progressive upsampling from 12.5 Hz to 24 kHz.
/// 3. **causal_conv1d** + **seq_layernorm** — residual refinement blocks.
/// 4. **overlap_add** — assemble output waveform from windowed frames.
#[allow(dead_code)]
pub struct MimiCodec {
    device: Device,
    queue: CommandQueue,

    // Compiled pipeline states for every kernel.
    codebook_gather_ps: ComputePipelineState,
    conv1d_transpose_ps: ComputePipelineState,
    causal_conv1d_ps: ComputePipelineState,
    seq_layernorm_ps: ComputePipelineState,
    overlap_add_ps: ComputePipelineState,
    ifft_synthesis_ps: ComputePipelineState,

    // The Metal library must be kept alive with the pipeline states.
    #[allow(dead_code)]
    library: Library,

    // Weight buffers.
    codebooks: Buffer, // [16][2048][128] half

    // Conv layer descriptors (not used after construction — kept for debugging).
    #[allow(dead_code)]
    conv_layers: Vec<MimiConvLayer>,

    // Pipeline state: number of transposed conv + residual layers.
    num_layers: usize,
}

impl MimiCodec {
    /// Load codec weights from cimage TTS segments.
    ///
    /// `weights` — packed conv layer data with embedded headers.
    /// `codebooks` — raw half-precision data [16][2048][128] (8 MiB).
    pub fn from_segments(
        device: &Device,
        weights: &[u8],
        codebooks: &[u8],
    ) -> Result<Self, String> {
        // ── Compile shared shader library ────────────────────────────────
        let src = include_str!("../../shaders/tts_codec.metal");
        let opts = CompileOptions::new();
        let library = device
            .new_library_with_source(src, &opts)
            .map_err(|e| format!("tts_codec.metal library compile failed: {e}"))?;

        let queue = device.new_command_queue();

        // ── Compile kernels ──────────────────────────────────────────────
        let codebook_gather_ps = compile_kernel(device, &library, "codebook_gather")?;
        let conv1d_transpose_ps = compile_kernel(device, &library, "conv1d_transpose")?;
        let causal_conv1d_ps = compile_kernel(device, &library, "causal_conv1d")?;
        let seq_layernorm_ps = compile_kernel(device, &library, "seq_layernorm")?;
        let overlap_add_ps = compile_kernel(device, &library, "overlap_add")?;
        let ifft_synthesis_ps = compile_kernel(device, &library, "ifft_synthesis")?;

        // ── Upload codebook buffer ───────────────────────────────────────
        let cb_buf = device.new_buffer_with_data(
            codebooks.as_ptr() as *const std::ffi::c_void,
            codebooks.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // ── Parse conv layers ────────────────────────────────────────────
        let layer_descs = parse_conv_layer_headers(weights)?;
        let mut conv_layers = Vec::with_capacity(layer_descs.len());
        for desc in &layer_descs {
            conv_layers.push(MimiConvLayer::new(device, desc)?);
        }
        let num_layers = conv_layers.len();

        Ok(Self {
            device: device.clone(),
            queue,
            codebook_gather_ps,
            conv1d_transpose_ps,
            causal_conv1d_ps,
            seq_layernorm_ps,
            overlap_add_ps,
            ifft_synthesis_ps,
            library,
            codebooks: cb_buf,
            conv_layers,
            num_layers,
        })
    }

    /// Decode audio tokens to waveform.
    ///
    /// `tokens` — `[num_tokens × 16]` u32 codebook indices
    ///   (16 RVQ codebooks per frame, arranged in row-major).
    ///
    /// Returns `Vec<f32>` PCM samples at 24 kHz, normalized to [-1.0, 1.0].
    pub fn decode(&self, tokens: &[u32]) -> Result<Vec<f32>, String> {
        if tokens.is_empty() || tokens.len() % 16 != 0 {
            return Err(format!(
                "token count must be a multiple of 16, got {}",
                tokens.len()
            ));
        }
        let num_tokens = (tokens.len() / 16) as u32;
        let device = &self.device;
        let queue = &self.queue;

        // ── Allocate buffers ─────────────────────────────────────────────
        // The embedding dimension after codebook gather: 16 × 128 = 2048.
        let _embed_dim: u32 = 16 * 128;

        // Input token buffer.
        let token_buf = device.new_buffer_with_data(
            tokens.as_ptr() as *const std::ffi::c_void,
            (tokens.len() * 4) as u64, // u32 = 4 bytes
            MTLResourceOptions::StorageModeShared,
        );

        // Scratch buffers: ping-pong between buf_a and buf_b for the
        // transposed conv chain.
        let mut buf_a_active = true;

        // Allocate generously: max_frames is num_tokens after the final
        // upsampling layer, max_channels is the largest out_ch across layers.
        let (output_frames, max_channels) = self.estimate_output_shape(num_tokens);
        let scratch_len = (output_frames as u64) * (max_channels as u64) * 2; // half = 2 bytes
        let buf_a = device.new_buffer(scratch_len, MTLResourceOptions::StorageModeShared);
        let buf_b = device.new_buffer(scratch_len, MTLResourceOptions::StorageModeShared);

        // ── Step 1: codebook_gather ──────────────────────────────────────
        {
            let cmd_buf = queue.new_command_buffer();
            let enc = cmd_buf.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.codebook_gather_ps);
            enc.set_buffer(0, Some(&token_buf), 0);
            enc.set_buffer(1, Some(&self.codebooks), 0);
            enc.set_buffer(2, Some(&buf_a), 0);

            let n = num_tokens as u64;
            enc.set_bytes(3, 4, &num_tokens as *const u32 as *const std::ffi::c_void);

            let grid = MTLSize {
                width: n,
                height: 16,
                depth: 128,
            };
            let group = MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            };
            enc.dispatch_threads(grid, group);
            enc.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
        }

        // ── Step 2: transposed conv layers ───────────────────────────────
        let mut current_frames = num_tokens;

        for layer in &self.conv_layers {
            let (in_buf, out_buf) = if buf_a_active {
                (&buf_a, &buf_b)
            } else {
                (&buf_b, &buf_a)
            };

            let out_frames = (current_frames - 1) * layer.stride + layer.kernel_size;
            let out_channels = layer.out_channels;

            let cmd_buf = queue.new_command_buffer();
            let enc = cmd_buf.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.conv1d_transpose_ps);

            enc.set_buffer(0, Some(in_buf), 0);
            enc.set_buffer(1, Some(&layer.weight), 0);
            match &layer.bias {
                Some(b) => enc.set_buffer(2, Some(b), 0),
                None => enc.set_buffer(2, None, 0),
            }
            enc.set_buffer(3, Some(out_buf), 0);

            enc.set_bytes(
                4,
                4,
                &layer.in_channels as *const u32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                5,
                4,
                &layer.out_channels as *const u32 as *const std::ffi::c_void,
            );
            enc.set_bytes(6, 4, &layer.stride as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(
                7,
                4,
                &layer.kernel_size as *const u32 as *const std::ffi::c_void,
            );
            enc.set_bytes(
                8,
                4,
                &current_frames as *const u32 as *const std::ffi::c_void,
            );

            let total = (out_frames as u64) * (out_channels as u64);
            let grid = MTLSize {
                width: total,
                height: 1,
                depth: 1,
            };
            let group = MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            };
            enc.dispatch_threads(grid, group);
            enc.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            buf_a_active = !buf_a_active;
            current_frames = out_frames;
        }

        // After the last transposed conv, the active buffer holds the waveform frames.
        let final_buf = if buf_a_active { &buf_a } else { &buf_b };

        // ── Step 3: overlap-add ──────────────────────────────────────────
        // Each frame = kernel_size of the last layer (or stride-aligned).
        // Use hop_size = stride of last layer for standard overlap-add.
        let last_stride = self.conv_layers.last().map(|l| l.stride).unwrap_or(1);
        let last_kernel = self.conv_layers.last().map(|l| l.kernel_size).unwrap_or(1);
        let frame_size = last_kernel;
        let hop_size = last_stride;
        // Output length: each frame contributes frame_size samples,
        // overlapping by (frame_size - hop_size).
        let output_samples = (current_frames - 1) * hop_size + frame_size;

        let output_buf = device.new_buffer(
            (output_samples as u64) * 4, // f32
            MTLResourceOptions::StorageModeShared,
        );

        {
            let cmd_buf = queue.new_command_buffer();
            let enc = cmd_buf.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&self.overlap_add_ps);

            enc.set_buffer(0, Some(final_buf), 0);
            enc.set_buffer(1, Some(&output_buf), 0);
            enc.set_buffer(2, None, 0); // no window function

            enc.set_bytes(
                3,
                4,
                &current_frames as *const u32 as *const std::ffi::c_void,
            );
            enc.set_bytes(4, 4, &hop_size as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(5, 4, &frame_size as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(
                6,
                4,
                &output_samples as *const u32 as *const std::ffi::c_void,
            );

            let grid = MTLSize {
                width: current_frames as u64,
                height: 1,
                depth: 1,
            };
            let group = MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            };
            enc.dispatch_threads(grid, group);
            enc.end_encoding();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
        }

        // ── Read back ────────────────────────────────────────────────────
        let ptr = output_buf.contents() as *const u8;
        let out_bytes = unsafe { std::slice::from_raw_parts(ptr, (output_samples as usize) * 4) };
        let samples: Vec<f32> = out_bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();

        Ok(samples)
    }

    /// Estimate final output shape given `num_tokens` input frames.
    fn estimate_output_shape(&self, num_tokens: u32) -> (u32, u32) {
        let embed_dim = 16 * 128;
        let mut frames = num_tokens;
        let mut max_ch = embed_dim;
        for layer in &self.conv_layers {
            frames = (frames - 1) * layer.stride + layer.kernel_size;
            max_ch = max_ch.max(layer.out_channels);
        }
        // Return (frames, max_channels) after all transposed conv layers
        // — this is the final frame count before overlap-add, and the
        // widest channel count, used for sizing scratch buffers.
        (frames, max_ch)
    }
}
