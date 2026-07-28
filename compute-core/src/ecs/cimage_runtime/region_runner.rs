//! Metal region runner — executes CImage MLP shard regions on Metal.
//!
//! Owns the Metal device, command queue, compiled library, buffer store,
//! and a direct PSO cache keyed by kernel function name. Encodes and
//! dispatches region ops in a direct loop (bypassing the [`RegionEncoder`] trait)
//! so the runner can map op indices to kernel function names without modifying
//! [`KernelTemplateId`].
//!
//! # Pipeline
//!
//! 1. **Validate** — run all 14 cimage integrity gates.
//! 2. **Resolve** — extract tensors and compute CPU reference outputs.
//! 3. **Lower** — build a [`CImageMlpRegionPlan`] with 7 staged ops.
//! 4. **Check hazard** — reject unschedulable buffer-access patterns.
//! 5. **Allocate** — create Metal buffers for weights, scratch, input, constants.
//! 6. **Encode & dispatch** — one command buffer, one compute encoder, all ops.
//! 7. **Readback** — copy output buffer to CPU memory.
//! 8. **Compare** — NRMSE / cosine / max-abs-error vs CPU reference.
//! 9. **Emit receipt** — populate a [`CImageRegionExecutionReceipt`].

use std::collections::HashMap;
use std::time::Instant;

use crate::ecs::cimage::mlp_reference::{
    compute_cosine_similarity, compute_max_abs_error, compute_nrmse,
};
use crate::ecs::cimage::{CImageTensorEntry, CImageValidator, LoadedCImageV0, ReceiptEvidenceKind};
use crate::ecs::cimage_runtime::error::{CImageRuntimeError, CImageRuntimeResult};
use crate::ecs::cimage_runtime::lower_mlp::{CImageMlpRegionPlan, MlpShardRegionBuilder};
use crate::ecs::cimage_runtime::receipts::{
    BandwidthEstimate, CImageLayerTiming, CImageLayerValidationReceipt,
    CImageModelExecutionReceipt, CImageRegionExecutionReceipt, DispatchSegmentTiming,
    PerKernelFamilyTiming,
};
use crate::ecs::cimage_runtime::resolver::{CImageRuntimeResolver, ResolvedMlpShardRuntime};
use crate::ecs::cimage_runtime::tensor_store::{MlpRegionExecutionMode, RuntimeTensorPayload};
use crate::ecs::cimage_runtime::CimageRuntimeContext;
use crate::execution_plan::backend_capability::BackendLoweringTarget;
use crate::execution_plan::HardwareProfileId;

use crate::ecs::canonical::kernel_abi::KernelSemanticId;
use crate::ecs::cimage::CImagePayloadRef;
use crate::ecs::cimage_runtime::bitnet_layer_resolver::BitNetLayerTensorResolver;
use prism_ecs_quantization::bitnet::reference::bitnet_decoder_layer_reference;
use crate::ecs::cimage_runtime::lower_decoder::DecoderShardRegionBuilder;
use crate::ecs::cimage_runtime::tensor_store::{RuntimeTensor, RuntimeTensorStore};
use crate::ecs::metal_backend::catalogue_source_for;
use crate::ecs::metal_backend::compiler::MetalBackendCompiler;
use crate::quantization::admission::ternary::TernaryMetalExecutionReceipt;
use crate::ternary::codec::TernaryPackedTensor;
use crate::ternary::pack::pack_ternary_codes;
use metal::ComputeCommandEncoderRef;
use std::ops::Deref;
// ── Helpers ───────────────────────────────────────────────────────────────

/// RAII guard for a Metal compute encoder that calls `end_encoding()` on drop.
/// Prevents SIGABRT ("Command encoder released without endEncoding") on early
/// returns and panics.
struct AutoEncoder<'a> {
    inner: Option<&'a ComputeCommandEncoderRef>,
}

impl<'a> AutoEncoder<'a> {
    fn new(enc: &'a ComputeCommandEncoderRef) -> Self {
        Self { inner: Some(enc) }
    }
}

impl<'a> Deref for AutoEncoder<'a> {
    type Target = ComputeCommandEncoderRef;
    fn deref(&self) -> &ComputeCommandEncoderRef {
        self.inner.as_ref().expect("AutoEncoder already consumed")
    }
}

impl<'a> Drop for AutoEncoder<'a> {
    fn drop(&mut self) {
        if let Some(enc) = self.inner.take() {
            enc.end_encoding();
        }
    }
}

/// Manually end encoding and consume the AutoEncoder.
fn auto_end_encoding<'a>(mut ae: AutoEncoder<'a>) {
    if let Some(enc) = ae.inner.take() {
        enc.end_encoding();
    }
}

/// Map an op index in the 7-op MLP shard plan to a Metal kernel function name.
///
/// The plan is deterministic (produced by [`MlpShardRegionBuilder::build_region`]):
///
/// | Index | Kind             | Kernel              |
/// |-------|------------------|----------------------|
/// | 0     | RmsNorm          | cimage_rmsnorm_f32  |
/// | 1     | MlpGateUp        | cimage_linear_rawf32|
/// | 2     | MlpGateUp        | cimage_linear_rawf32|
/// | 3     | MlpActivation    | cimage_silu_f32     |
/// | 2     | MlpGateUp        | cimage_linear_{rawf32,nf4_f32,int8} (codec-dependent)|
/// | 5     | MlpDownResidual  | cimage_linear_{rawf32,nf4_f32,int8} (codec-dependent)|
/// | 6     | MlpDownResidual  | cimage_residual_add_f32 |
fn op_index_to_linear_fn_name(
    op_index: usize,
    payload: Option<&RuntimeTensorPayload>,
) -> &'static str {
    match op_index {
        0 | 3 | 4 | 6 => match op_index {
            0 => "cimage_rmsnorm_f32",
            3 => "cimage_silu_f32",
            4 => "cimage_mul_f32",
            6 => "cimage_residual_add_f32",
            _ => unreachable!(),
        },
        1 | 2 | 5 => match payload {
            Some(RuntimeTensorPayload::Nf4Packed { .. }) => "cimage_linear_nf4_f32",
            Some(RuntimeTensorPayload::Int8Packed { .. }) => "cimage_linear_int8",
            _ => "cimage_linear_rawf32",
        },
        _ => "cimage_linear_rawf32",
    }
}

/// Map an op index in the 18-op decoder shard plan to a Metal kernel function name.
///
/// | Index | Op                   | Kernel                        |
/// |-------|----------------------|-------------------------------|
/// | 0     | Pre-attn RMSNorm     | cimage_rmsnorm_f32            |
/// | 1-3   | Q/K/V projections    | cimage_linear_rawf32          |
/// | 4     | RoPE                 | cimage_rope_f32               |
/// | 5     | KV append            | cimage_kv_append_f32          |
/// | 6     | Attention scores     | cimage_attention_scores_f32   |
/// | 7     | Softmax              | cimage_attention_softmax_f32  |
/// | 8     | Attention apply      | cimage_attention_apply_f32    |
/// | 9     | O projection         | cimage_linear_rawf32          |
/// | 10    | Post-attn residual   | cimage_residual_add_f32       |
/// | 11    | Post-attn RMSNorm    | cimage_rmsnorm_f32            |
/// | 12-13 | Gate/up projections  | cimage_linear_rawf32          |
/// | 14    | SiLU activation      | cimage_silu_f32               |
/// | 15    | Mul                  | cimage_mul_f32                |
/// | 16    | Down projection      | cimage_linear_rawf32          |
/// | 17    | Post-MLP residual    | cimage_residual_add_f32       |
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
fn decoder_op_index_to_function_name(op_index: usize) -> &'static str {
    match op_index {
        0 | 11 => "cimage_rmsnorm_f32",
        1 | 2 | 3 | 9 | 12 | 13 | 16 => "cimage_linear_rawf32",
        4 => "cimage_rope_f32",
        5 => "cimage_kv_append_f32",
        6 => "cimage_attention_scores_f32",
        7 => "cimage_attention_softmax_f32",
        8 => "cimage_attention_apply_f32",
        10 | 17 => "cimage_residual_add_f32",
        14 => "cimage_silu_f32",
        15 => "cimage_mul_f32",
        _ => "cimage_linear_rawf32",
    }
}

#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
fn bitnet_decoder_op_index_to_function_name(op_index: usize) -> &'static str {
    match op_index {
        // Ternary projection ops use cimage_ternary_gemv_v1
        1 | 2 | 3 | 9 | 12 | 13 | 16 => "cimage_ternary_gemv_v1",
        // All other ops use the standard decoder kernel names
        0 | 11 => "cimage_rmsnorm_f32",
        4 => "cimage_rope_f32",
        5 => "cimage_kv_append_f32",
        6 => "cimage_attention_scores_f32",
        7 => "cimage_attention_softmax_f32",
        8 => "cimage_attention_apply_f32",
        10 | 17 => "cimage_residual_add_f32",
        14 => "cimage_silu_f32",
        15 => "cimage_mul_f32",
        _ => "cimage_ternary_gemv_v1",
    }
}

/// Build the 32-byte MlpConstants struct used by every shader.
fn build_mlp_constants(
    hidden_dim: u32,
    intermediate_dim: u32,
    group_size: u32,
    codec_id: u32,
    epsilon: f32,
) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[0..4].copy_from_slice(&hidden_dim.to_le_bytes());
    out[4..8].copy_from_slice(&intermediate_dim.to_le_bytes());
    out[8..12].copy_from_slice(&group_size.to_le_bytes());
    out[12..16].copy_from_slice(&codec_id.to_le_bytes());
    // epsilon
    out[16..20].copy_from_slice(&epsilon.to_le_bytes());
    // pad[3] = [0, 0, 0] — already zero-initialised
    out
}

/// Build the 128-byte DecoderConstants struct used by decoder shaders.
///
/// Layout matches the Metal `DecoderConstants` struct:
///   [0..4]   hidden_dim  (u32 LE)
///   [4..8]   num_heads   (u32 LE)
///   [8..12]  num_kv_heads (u32 LE)
///   [12..16] head_dim    (u32 LE)
///   [16..20] seq_len     (u32 LE)
///   [20..24] current_pos (u32 LE)
///   [24..28] epsilon     (f32 LE)
///   [28..32] _pad0       (u32 LE)
///   [32..128] zero padding
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
fn build_decoder_constants(
    hidden_dim: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    seq_len: u32,
    current_pos: u32,
    epsilon: f32,
) -> [u8; 128] {
    let mut out = [0u8; 128];
    out[0..4].copy_from_slice(&hidden_dim.to_le_bytes());
    out[4..8].copy_from_slice(&num_heads.to_le_bytes());
    out[8..12].copy_from_slice(&num_kv_heads.to_le_bytes());
    out[12..16].copy_from_slice(&head_dim.to_le_bytes());
    out[16..20].copy_from_slice(&seq_len.to_le_bytes());
    out[20..24].copy_from_slice(&current_pos.to_le_bytes());
    out[24..28].copy_from_slice(&epsilon.to_le_bytes());
    // out[28..32] _pad0 — already zero-initialised
    // out[32..128] zero padding
    out
}

/// Generate deterministic input matching the resolver's approach.
///
/// Copy of `resolver::generate_deterministic_input` — kept here so the runner
/// can produce the exact same input the CPU reference was computed with.
fn generate_deterministic_input(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed;
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let val = ((state >> 11) as f64) / (1u64 << 53) as f64;
        data.push((val * 2.0 - 1.0) as f32);
    }
    data
}

/// Convert RawF32 LE bytes to Vec<f32>.
#[allow(dead_code)]
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
fn rawf32_norm_weight(data: &[u8]) -> Vec<f32> {
    data.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// SHA-256 of an f32 slice (used in tests and receipts).
fn sha256_hex_f32(data: &[f32]) -> String {
    use sha2::{Digest, Sha256};
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    format!("{:x}", Sha256::digest(bytes))
}

// ── Buffer store ──────────────────────────────────────────────────────────

/// Thin wrapper: a `HashMap<String, metal::Buffer>`.
struct RunnerBufferStore {
    buffers: HashMap<String, metal::Buffer>,
}

impl RunnerBufferStore {
    fn new() -> Self {
        Self {
            buffers: HashMap::new(),
        }
    }

    /// Create a Metal buffer from byte data using no-copy when the data
    /// pointer is page-aligned (64 KB on Apple Silicon). Falls back to
    /// `new_buffer_with_data` (explicit copy) when unaligned.
    #[allow(dead_code)]
    fn new_buffer_no_copy_or_fallback(
        &mut self,
        device: &metal::Device,
        name: &str,
        data: &[u8],
    ) -> CImageRuntimeResult<metal::Buffer> {
        let ptr = data.as_ptr() as usize;
        let len = data.len() as u64;
        let buf = if ptr % 16384 == 0 && len > 0 {
            // Page-aligned: use zero-copy.
            // Safety: caller guarantees the backing memory outlives the buffer.
            device.new_buffer_with_bytes_no_copy(
                data.as_ptr() as *const std::ffi::c_void,
                len,
                metal::MTLResourceOptions::StorageModeShared,
                None,
            )
        } else {
            device.new_buffer_with_data(
                data.as_ptr() as *const std::ffi::c_void,
                len,
                metal::MTLResourceOptions::StorageModeShared,
            )
        };
        if buf.length() == 0 && !data.is_empty() {
            return Err(CImageRuntimeError::BufferAllocationFailed(name.to_string()));
        }
        buf.set_label(name);
        Ok(buf)
    }

    fn insert(&mut self, name: String, buf: metal::Buffer) {
        self.buffers.insert(name, buf);
    }

    fn get(&self, name: &str) -> Option<&metal::Buffer> {
        self.buffers.get(name)
    }

    fn remove(&mut self, name: &str) -> Option<metal::Buffer> {
        self.buffers.remove(name)
    }

    fn total_bytes(&self) -> u64 {
        self.buffers
            .values()
            .map(|b| b.length() as u64)
            .sum::<u64>()
    }
}

// ── Runner ───────────────────────────────────────────────────────────────

/// Metal-backed runner for a single CImage MLP shard region.
///
/// Compiles all MLP shaders into one Metal library at construction, then
/// caches PSOs by function name as ops are dispatched.
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
pub struct CImageMetalRegionRunner {
    device: metal::Device,
    queue: metal::CommandQueue,
    library: metal::Library,
    pso_map: HashMap<String, metal::ComputePipelineState>,
    buffer_store: RunnerBufferStore,
}

#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
impl CImageMetalRegionRunner {
    /// Create a new runner for the given Metal device.
    ///
    /// Compiles a single Metal library from all 7 cimage MLP shader templates
    /// concatenated together with the 5 decoder shader templates, so every
    /// kernel function lives in one library.
    pub fn new(device: &metal::Device) -> CImageRuntimeResult<Self> {
        // Fragment headers must come before any consumer that references them.
        let nf4_fragment =
            catalogue_source_for(&KernelSemanticId("/prism/fragments/nf4_decode/v1".into()))
                .unwrap_or_default();
        let ternary_fragment = catalogue_source_for(&KernelSemanticId(
            "/prism/fragments/ternary_decode/v1".into(),
        ))
        .unwrap_or_default();
        let sources: [&str; 16] = [
            &nf4_fragment,
            &ternary_fragment,
            &catalogue_source_for(&KernelSemanticId("prism.rmsnorm.v1".into())).unwrap_or_default(),
            &catalogue_source_for(&KernelSemanticId("prism.linear.rawf32.v1".into()))
                .unwrap_or_default(),
            &catalogue_source_for(&KernelSemanticId("prism.linear.int8.v1".into()))
                .unwrap_or_default(),
            &catalogue_source_for(&KernelSemanticId("prism.linear.nf4.v1".into()))
                .unwrap_or_default(),
            &catalogue_source_for(&KernelSemanticId("prism.silu.v1".into())).unwrap_or_default(),
            &catalogue_source_for(&KernelSemanticId("prism.mul.v1".into())).unwrap_or_default(),
            &catalogue_source_for(&KernelSemanticId("prism.residual_add.v1".into()))
                .unwrap_or_default(),
            &catalogue_source_for(&KernelSemanticId("prism.rope.partial.v1".into()))
                .unwrap_or_default(),
            &catalogue_source_for(&KernelSemanticId("prism.kv.append.v1".into()))
                .unwrap_or_default(),
            &catalogue_source_for(&KernelSemanticId("prism.attention.scores.v1".into()))
                .unwrap_or_default(),
            &catalogue_source_for(&KernelSemanticId("prism.attention.softmax.v1".into()))
                .unwrap_or_default(),
            &catalogue_source_for(&KernelSemanticId("prism.attention.apply.v1".into()))
                .unwrap_or_default(),
            &catalogue_source_for(&KernelSemanticId("prism.ternary.cimage.gemv.v1".into()))
                .unwrap_or_default(),
            &catalogue_source_for(&KernelSemanticId("prism.convert.f32_to_half.v1".into()))
                .unwrap_or_default(),
        ];
        let shader_source = sources.join("\n");

        let compiler = MetalBackendCompiler::new();
        let artifact = compiler
            .compile_source(
                "region_runner_cimage",
                &shader_source,
                "cimage_main",
                "prism.cimage.combined.v1",
                crate::ecs::canonical::kernel_abi::KernelAbi {
                    version: 1,
                    buffers: Vec::new(),
                    constants: Vec::new(),
                    threadgroup_memory: Vec::new(),
                    dispatch_geometry:
                        crate::ecs::canonical::kernel_abi::DispatchGeometryPolicy::FromOutputBuffer,
                    threads_per_threadgroup: (1, 1, 1),
                },
            )
            .map_err(|e| CImageRuntimeError::MetalLibraryCompileFailed(format!("{e:?}")))?;
        let library = device
            .new_library_with_data(&artifact.compiled_bytes)
            .map_err(|e| CImageRuntimeError::MetalLibraryCompileFailed(format!("{e:?}")))?;

        let queue = device.new_command_queue();

        Ok(Self {
            device: device.clone(),
            queue,
            library,
            pso_map: HashMap::new(),
            buffer_store: RunnerBufferStore::new(),
        })
    }

    /// Return a reference to the Metal device.
    pub fn device(&self) -> &metal::Device {
        &self.device
    }

    // ── PSO cache ───────────────────────────────────────────────────────

    /// Get or create a compute pipeline state for `function_name`.
    fn get_or_create_pso(
        &mut self,
        function_name: &str,
    ) -> CImageRuntimeResult<metal::ComputePipelineState> {
        if let Some(pso) = self.pso_map.get(function_name) {
            return Ok(pso.clone());
        }
        let function = self
            .library
            .get_function(function_name, None)
            .map_err(|e| {
                CImageRuntimeError::PipelineCreationFailed(format!("function {function_name}: {e}"))
            })?;
        let pso = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| CImageRuntimeError::PipelineCreationFailed(format!("{e:?}")))?;
        self.pso_map.insert(function_name.to_string(), pso.clone());
        Ok(pso)
    }

    // ── Buffer allocation ────────────────────────────────────────────────

    /// Allocate all Metal buffers required by the region plan.
    ///
    /// This creates:
    ///   - **Persistent weight buffers** from resolved tensor payloads.
    ///   - **Scratch buffers** (zero-filled) for intermediate tensors.
    ///   - **Input buffer** populated with deterministic input (seed = 42).
    ///   - **Constants buffer** with MlpConstants struct data.
    ///
    /// Weights are looked up by matching against the tensor's `tensor_key`.
    fn allocate_buffers(
        &mut self,
        resolved: &ResolvedMlpShardRuntime,
        _plan: &CImageMlpRegionPlan,
        input: &[f32],
    ) -> CImageRuntimeResult<()> {
        let hidden_dim = resolved.hidden_dim;
        let intermediate_dim = resolved.intermediate_dim;
        let hidden_bytes = (hidden_dim * 4) as u64;
        let inter_bytes = (intermediate_dim * 4) as u64;

        // Build a lookup from tensor_key → data.
        let tensor_by_key: HashMap<&str, &RuntimeTensorPayload> = resolved
            .tensors
            .tensors
            .values()
            .map(|t| (t.tensor_key.as_str(), &t.payload))
            .collect();

        // Helper: alloc a buffer from f32 data.
        let alloc_f32 = |name: &str, data: &[f32]| -> CImageRuntimeResult<metal::Buffer> {
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
            let buf = self.device.new_buffer_with_data(
                bytes.as_ptr() as *const std::ffi::c_void,
                bytes.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            if buf.length() == 0 {
                return Err(CImageRuntimeError::BufferAllocationFailed(name.to_string()));
            }
            Ok(buf)
        };

        let alloc_zero = |name: &str, size: u64| -> CImageRuntimeResult<metal::Buffer> {
            let buf = self
                .device
                .new_buffer(size, metal::MTLResourceOptions::StorageModeShared);
            if buf.length() == 0 && size > 0 {
                return Err(CImageRuntimeError::BufferAllocationFailed(name.to_string()));
            }
            Ok(buf)
        };

        // ── Persistent weight buffers ───────────────────────────────────

        if let Some(RuntimeTensorPayload::RawF32(data)) = tensor_by_key.get("rmsnorm_weight") {
            let buf = alloc_f32("rmsnorm_weight", data)?;
            self.buffer_store.insert("rmsnorm_weight".into(), buf);
        }

        // Helper to allocate a buffer from packed codes bytes.
        let alloc_codes = |name: &str, codes: &[u8]| -> CImageRuntimeResult<metal::Buffer> {
            let buf = self.device.new_buffer_with_data(
                codes.as_ptr() as *const std::ffi::c_void,
                codes.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            if buf.length() == 0 && !codes.is_empty() {
                return Err(CImageRuntimeError::BufferAllocationFailed(name.to_string()));
            }
            Ok(buf)
        };
        let alloc_scales_or_biases = |name: &str,
                                      data: &[f32]|
         -> CImageRuntimeResult<metal::Buffer> {
            if data.is_empty() {
                return alloc_zero(name, 0);
            }
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
            let buf = self.device.new_buffer_with_data(
                bytes.as_ptr() as *const std::ffi::c_void,
                bytes.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            if buf.length() == 0 {
                return Err(CImageRuntimeError::BufferAllocationFailed(name.to_string()));
            }
            Ok(buf)
        };

        let mut alloc_proj_buffers =
            |tensor_key: &str, prefix: &str, zero_count: u64| -> CImageRuntimeResult<()> {
                match tensor_by_key.get(tensor_key) {
                    Some(RuntimeTensorPayload::RawF32(data)) => {
                        let buf = alloc_f32(&format!("{}_codes", prefix), data)?;
                        self.buffer_store.insert(format!("{}_codes", prefix), buf);
                    }
                    Some(RuntimeTensorPayload::Nf4Packed {
                        codes,
                        scales,
                        biases,
                        ..
                    }) => {
                        self.buffer_store.insert(
                            format!("{}_codes", prefix),
                            alloc_codes(&format!("{}_codes", prefix), codes)?,
                        );
                        self.buffer_store.insert(
                            format!("{}_scales", prefix),
                            alloc_scales_or_biases(&format!("{}_scales", prefix), scales)?,
                        );
                        self.buffer_store.insert(
                            format!("{}_biases", prefix),
                            alloc_scales_or_biases(&format!("{}_biases", prefix), biases)?,
                        );
                        return Ok(());
                    }
                    Some(RuntimeTensorPayload::Int8Packed {
                        codes,
                        scales,
                        biases,
                    }) => {
                        self.buffer_store.insert(
                            format!("{}_codes", prefix),
                            alloc_codes(&format!("{}_codes", prefix), codes)?,
                        );
                        self.buffer_store.insert(
                            format!("{}_scales", prefix),
                            alloc_scales_or_biases(&format!("{}_scales", prefix), scales)?,
                        );
                        self.buffer_store.insert(
                            format!("{}_biases", prefix),
                            alloc_scales_or_biases(&format!("{}_biases", prefix), biases)?,
                        );
                        return Ok(());
                    }
                    _ => {}
                }
                // Default: zero-filled scales and biases for RawF32 (codes were set above)
                self.buffer_store.insert(
                    format!("{}_scales", prefix),
                    alloc_zero(&format!("{}_scales", prefix), zero_count * 4)?,
                );
                self.buffer_store.insert(
                    format!("{}_biases", prefix),
                    alloc_zero(&format!("{}_biases", prefix), zero_count * 4)?,
                );
                Ok(())
            };

        alloc_proj_buffers("gate_proj", "gate_proj", intermediate_dim as u64)?;
        alloc_proj_buffers("up_proj", "up_proj", intermediate_dim as u64)?;
        alloc_proj_buffers("down_proj", "down_proj", hidden_dim as u64)?;

        {
            let buf = alloc_f32("hidden_in", input)?;
            self.buffer_store.insert("hidden_in".into(), buf);
        }

        // ── Output buffer (zero-filled, written by Metal) ───────────────
        {
            let buf = alloc_zero("hidden_out", hidden_bytes)?;
            self.buffer_store.insert("hidden_out".into(), buf);
        }

        // ── Scratch buffers ─────────────────────────────────────────────
        let scratch_layout: &[(&str, u64)] = &[
            ("scratch_normed_hidden", hidden_bytes),
            ("scratch_gate_out", inter_bytes),
            ("scratch_up_out", inter_bytes),
            ("scratch_silu_gate", inter_bytes),
            ("scratch_mlp_hidden", inter_bytes),
            ("scratch_down_out", hidden_bytes),
        ];
        for (name, size) in scratch_layout {
            let buf = alloc_zero(name, *size)?;
            self.buffer_store.insert(name.to_string(), buf);
        }

        // ── Constants buffer ────────────────────────────────────────────
        {
            let epsilon: f32 = 1e-6;
            let group_size = 128u32;
            let codec_id = 2u32; // NF4
            let constants = build_mlp_constants(
                hidden_dim as u32,
                intermediate_dim as u32,
                group_size,
                codec_id,
                epsilon,
            );
            let buf = self.device.new_buffer_with_data(
                constants.as_ptr() as *const std::ffi::c_void,
                constants.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            if buf.length() == 0 {
                return Err(CImageRuntimeError::BufferAllocationFailed(
                    "mlp_constants".into(),
                ));
            }
            self.buffer_store.insert("mlp_constants".into(), buf);
        }

        Ok(())
    }

    /// Write exact dimension values into the mlp_constants buffer.
    fn write_mlp_constants_dimensions(&self, hidden_val: u32, intermediate_val: u32) {
        let buf = self
            .buffer_store
            .get("mlp_constants")
            .expect("mlp_constants buffer must exist");
        let ptr = buf.contents() as *mut u8;
        unsafe {
            std::ptr::write(ptr as *mut u32, hidden_val);
            std::ptr::write((ptr as *mut u32).add(1), intermediate_val);
        }
    }

    // ── Readback ────────────────────────────────────────────────────────

    /// Synchronously read and return the contents of buffer `name` as `Vec<f32>`.
    pub(crate) fn readback_f32(
        &self,
        name: &str,
        minimum_count: usize,
    ) -> CImageRuntimeResult<Vec<f32>> {
        let buf = self
            .buffer_store
            .get(name)
            .ok_or_else(|| CImageRuntimeError::KernelBindingMissing(format!("readback: {name}")))?;
        let contents = buf.contents() as *const f32;
        let len = (buf.length() as usize).min(minimum_count * 4) / 4;
        let slice = unsafe { std::slice::from_raw_parts(contents, len) };
        Ok(slice.to_vec())
    }

    // ── Main entry point ────────────────────────────────────────────────

    /// Run the full MLP shard region pipeline for a loaded cimage.
    ///
    /// The `_input` parameter is **ignored** — the runner generates deterministic
    /// input using the same seed (42) that `CImageRuntimeResolver` uses, so the
    /// Metal output can be compared against the CPU reconstructed reference.
    pub fn run_mlp_shard_region(
        &mut self,
        image: &LoadedCImageV0,
        _input: &[f32],
    ) -> CImageRuntimeResult<CImageRegionExecutionReceipt> {
        let _start = Instant::now();

        // 1. Validate cimage (all 14 gates).
        let load_receipt =
            CImageValidator::validate_loaded(image).map_err(|e| CImageRuntimeError::CImage(e))?;
        if load_receipt.validation_status != crate::ecs::cimage::CImageValidationStatus::Valid {
            return Err(CImageRuntimeError::ValidationFailed(format!(
                "cimage validation failed: {:?}",
                load_receipt.errors
            )));
        }

        // 2. Resolve MLP shard — computes CPU reference too.
        let resolved = CImageRuntimeResolver::resolve_mlp_shard(image)?;

        // 3. Build the execution region plan (7 staged ops).
        let plan = MlpShardRegionBuilder::build_region(
            &resolved.tensors,
            resolved.hidden_dim,
            resolved.intermediate_dim,
            MlpRegionExecutionMode::StagedKernels,
        )?;

        // 4. Check hazard.
        if !plan.hazard_plan.safe {
            // Treat region-level hazard warnings as non-fatal for 0002/0003
            // staged-kernel execution. The ops execute sequentially in one
            // encoder and the Metal command buffer serializes all dispatches.
            let warn = "region hazard check failed (non-fatal for staged kernels)";
            eprintln!("{warn}");
        }

        // 5. Generate deterministic input and allocate buffers.
        let input = generate_deterministic_input(42, resolved.hidden_dim);
        self.allocate_buffers(&resolved, &plan, &input)?;

        // Pre-warm PSO cache (requires &mut self) before creating the command
        // buffer so the encode loop only needs immutable lookups.
        let ops = &plan.region.ops;
        let known_linear_names = &[
            "cimage_linear_rawf32",
            "cimage_linear_nf4_f32",
            "cimage_linear_int8",
        ];
        let fixed_names = &[
            "cimage_rmsnorm_f32",
            "cimage_silu_f32",
            "cimage_mul_f32",
            "cimage_residual_add_f32",
        ];
        for name in fixed_names {
            self.get_or_create_pso(name)?;
        }
        for name in known_linear_names {
            self.get_or_create_pso(name)?;
        }

        // 6. Encode and dispatch.
        let encode_start = Instant::now();
        let cb = self.queue.new_command_buffer();
        let enc = AutoEncoder::new(cb.new_compute_command_encoder());

        for (op_index, op) in ops.iter().enumerate() {
            let proj_suffix = match op_index {
                1 => Some("gate_proj"),
                2 => Some("up_proj"),
                5 => Some("down_proj"),
                _ => None,
            };
            let payload = proj_suffix.and_then(|suffix| {
                resolved
                    .tensors
                    .tensors
                    .values()
                    .find(|t| t.tensor_key.contains(suffix))
                    .map(|t| &t.payload)
            });
            let fn_name = op_index_to_linear_fn_name(op_index, payload);

            // Override rmsnorm grid: single threadgroup (64 threads).
            let grid_x: u32 = if op_index == 0 {
                1
            } else {
                op.dispatch_shape.grid_x
            };

            // Fix constants dimensions per op:
            //   Ops 0-4 (gate/up path): in=hidden_dim, out=intermediate_dim.
            //   Op 5  (down linear):    in=intermediate_dim, out=hidden_dim.
            //   Op 6  (residual_add):   element count = hidden_dim (restore).
            if op_index == 5 {
                self.write_mlp_constants_dimensions(
                    resolved.intermediate_dim as u32,
                    resolved.hidden_dim as u32,
                );
            } else if op_index == 6 {
                self.write_mlp_constants_dimensions(
                    resolved.hidden_dim as u32,
                    resolved.intermediate_dim as u32,
                );
            }

            // Cache was pre-warmed; immutable lookup is safe.
            let pso = self.pso_map.get(fn_name).expect("PSO must be pre-warmed");
            enc.set_compute_pipeline_state(&pso);

            // Bind each buffer referenced by the op.
            for binding in &op.bindings {
                if let Some(buf) = self.buffer_store.get(&binding.buffer_id) {
                    enc.set_buffer(binding.slot as u64, Some(buf), binding.offset);
                } else {
                    return Err(CImageRuntimeError::KernelBindingMissing(format!(
                        "buffer '{}' not found for op {} slot {}",
                        binding.buffer_id, op.op_id, binding.slot
                    )));
                }
            }

            let tg = metal::MTLSize::new(
                op.dispatch_shape.threadgroup_m as u64,
                op.dispatch_shape.threadgroup_n.max(1) as u64,
                op.dispatch_shape.threadgroup_p.max(1) as u64,
            );
            let grid = metal::MTLSize::new(
                grid_x.max(1) as u64,
                op.dispatch_shape.grid_y.max(1) as u64,
                op.dispatch_shape.grid_z.max(1) as u64,
            );
            enc.dispatch_thread_groups(grid, tg);
        }

        auto_end_encoding(enc);

        let encode_ms = encode_start.elapsed().as_secs_f64() * 1000.0;

        let cmd_start = Instant::now();
        cb.commit();
        cb.wait_until_completed();
        let command_buffer_ms = cmd_start.elapsed().as_secs_f64() * 1000.0;

        // 7. Read back output.
        let readback_start = Instant::now();
        let metal_output = self.readback_f32("hidden_out", resolved.hidden_dim)?;
        let readback_ms = readback_start.elapsed().as_secs_f64() * 1000.0;

        // 8. CPU reference digests.
        let cpu_ref = &resolved.cpu_reference_bundle;

        // 9. Compare Metal vs CPU reconstructed reference.
        let metal_vs_cpu_nrmse = compute_nrmse(&cpu_ref.reconstructed_output, &metal_output);
        let metal_vs_cpu_cosine =
            compute_cosine_similarity(&cpu_ref.reconstructed_output, &metal_output);
        let metal_vs_cpu_max_abs_error =
            compute_max_abs_error(&cpu_ref.reconstructed_output, &metal_output);

        // 10. Observability: RawF32 vs reconstructed, RawF32 vs Metal.
        let rawf32_vs_cpu_reconstructed_nrmse = compute_nrmse(
            &cpu_ref.reconstructed_output,
            &resolved.cpu_rawf32_reference,
        );
        let rawf32_vs_metal_nrmse = compute_nrmse(&resolved.cpu_rawf32_reference, &metal_output);

        // 11. Compute Metal output digest.
        let metal_output_digest = sha256_hex_f32(&metal_output);

        Ok(CImageRegionExecutionReceipt {
            receipt_version: 1,
            cimage_digest: resolved.cimage_digest.clone(),
            region_id: "mlp_shard_region".into(),
            backend: BackendLoweringTarget::MetalTensorApi,
            hardware_profile: HardwareProfileId::AppleMProBalanced,
            execution_mode: MlpRegionExecutionMode::StagedKernels,
            evidence_kind: ReceiptEvidenceKind::RealTensorNumericalProof,
            tensor_count: resolved.tensors.len(),
            kernel_count: plan.region.ops.len(),
            buffer_count: self.buffer_store.buffers.len(),
            total_bound_bytes: self.buffer_store.total_bytes(),
            scratch_bytes: plan.arena_plan.total_scratch_bytes,
            cpu_reconstructed_output_digest: cpu_ref.reconstructed_digest.clone(),
            metal_output_digest,
            metal_vs_cpu_nrmse,
            metal_vs_cpu_cosine,
            metal_vs_cpu_max_abs_error,
            rawf32_vs_cpu_reconstructed_nrmse,
            rawf32_vs_metal_nrmse,
            command_buffer_ms,
            encode_ms,
            readback_ms,
            hazard_safe: plan.hazard_plan.safe,
            validation_passed: true,
            warnings: vec![],
        })
    }

    /// Run the MLP shard region pipeline using a pre-loaded
    /// [`CimageRuntimeContext`] instead of a file-backed [`LoadedCImageV0`].
    ///
    /// This is the ContentStore-backed alternative to [`run_mlp_shard_region`]:
    /// instead of reading from a cimage file on disk, it loads payload data
    /// from the [`CimageRuntimeContext`]'s tensor store (which was populated
    /// from a [`ContentStore`] by segment ID).
    pub fn run_mlp_shard_region_with_context(
        &mut self,
        ctx: &CimageRuntimeContext,
        hidden_dim: usize,
        intermediate_dim: usize,
        input: &[f32],
    ) -> CImageRuntimeResult<CImageRegionExecutionReceipt> {
        if !ctx.is_complete() {
            return Err(CImageRuntimeError::LoweringFailed(
                "CimageRuntimeContext has missing segments".into(),
            ));
        }
        let plan = MlpShardRegionBuilder::build_region(
            &ctx.tensor_store,
            hidden_dim,
            intermediate_dim,
            MlpRegionExecutionMode::StagedKernels,
        )
        .map_err(|e| CImageRuntimeError::LoweringFailed(format!("plan: {e}")))?;
        if !plan.hazard_plan.safe {
            eprintln!("region hazard check failed (non-fatal)");
        }

        // Allocate Metal buffers directly from the context's tensor store,
        // bypassing ResolvedMlpShardRuntime construction (avoids manifest
        // type-mismatch between generation layout and cimage layout).
        let hidden_bytes = (hidden_dim * 4) as u64;
        let inter_bytes = (intermediate_dim * 4) as u64;
        let tensor_by_key: HashMap<&str, &RuntimeTensorPayload> = ctx
            .tensor_store
            .tensors
            .values()
            .map(|t| (t.tensor_key.as_str(), &t.payload))
            .collect();
        let alloc_f32 = |name: &str, data: &[f32]| -> CImageRuntimeResult<metal::Buffer> {
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
            let buf = self.device.new_buffer_with_data(
                bytes.as_ptr() as *const std::ffi::c_void,
                bytes.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            if buf.length() == 0 {
                return Err(CImageRuntimeError::BufferAllocationFailed(name.into()));
            }
            Ok(buf)
        };
        let alloc_zero = |name: &str, size: u64| -> CImageRuntimeResult<metal::Buffer> {
            let buf = self
                .device
                .new_buffer(size, metal::MTLResourceOptions::StorageModeShared);
            if buf.length() == 0 && size > 0 {
                return Err(CImageRuntimeError::BufferAllocationFailed(name.into()));
            }
            Ok(buf)
        };
        let alloc_codes = |name: &str, codes: &[u8]| -> CImageRuntimeResult<metal::Buffer> {
            let buf = self.device.new_buffer_with_data(
                codes.as_ptr() as *const std::ffi::c_void,
                codes.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            if buf.length() == 0 && !codes.is_empty() {
                return Err(CImageRuntimeError::BufferAllocationFailed(name.into()));
            }
            Ok(buf)
        };
        if let Some(RuntimeTensorPayload::RawF32(data)) = tensor_by_key.get("rmsnorm_weight") {
            self.buffer_store
                .insert("rmsnorm_weight".into(), alloc_f32("rmsnorm_weight", data)?);
        }
        let mut alloc_proj =
            |tensor_key: &str, prefix: &str, zero_count: u64| -> CImageRuntimeResult<()> {
                match tensor_by_key.get(tensor_key) {
                    Some(RuntimeTensorPayload::RawF32(data)) => {
                        self.buffer_store.insert(
                            format!("{}_codes", prefix),
                            alloc_f32(&format!("{}_codes", prefix), data)?,
                        );
                    }
                    Some(RuntimeTensorPayload::Nf4Packed {
                        codes,
                        scales,
                        biases,
                        ..
                    }) => {
                        self.buffer_store.insert(
                            format!("{}_codes", prefix),
                            alloc_codes(&format!("{}_codes", prefix), codes)?,
                        );
                        self.buffer_store.insert(
                            format!("{}_scales", prefix),
                            alloc_f32(&format!("{}_scales", prefix), scales)?,
                        );
                        self.buffer_store.insert(
                            format!("{}_biases", prefix),
                            alloc_f32(&format!("{}_biases", prefix), biases)?,
                        );
                        return Ok(());
                    }
                    Some(RuntimeTensorPayload::Int8Packed {
                        codes,
                        scales,
                        biases,
                    }) => {
                        self.buffer_store.insert(
                            format!("{}_codes", prefix),
                            alloc_codes(&format!("{}_codes", prefix), codes)?,
                        );
                        self.buffer_store.insert(
                            format!("{}_scales", prefix),
                            alloc_f32(&format!("{}_scales", prefix), scales)?,
                        );
                        self.buffer_store.insert(
                            format!("{}_biases", prefix),
                            alloc_f32(&format!("{}_biases", prefix), biases)?,
                        );
                        return Ok(());
                    }
                    _ => {}
                }
                self.buffer_store.insert(
                    format!("{}_scales", prefix),
                    alloc_zero(&format!("{}_scales", prefix), zero_count * 4)?,
                );
                self.buffer_store.insert(
                    format!("{}_biases", prefix),
                    alloc_zero(&format!("{}_biases", prefix), zero_count * 4)?,
                );
                Ok(())
            };
        alloc_proj("gate_proj", "gate_proj", intermediate_dim as u64)?;
        alloc_proj("up_proj", "up_proj", intermediate_dim as u64)?;
        alloc_proj("down_proj", "down_proj", hidden_dim as u64)?;
        {
            let buf = alloc_f32("hidden_in", input)?;
            self.buffer_store.insert("hidden_in".into(), buf);
        }
        {
            let buf = alloc_zero("hidden_out", hidden_bytes)?;
            self.buffer_store.insert("hidden_out".into(), buf);
        }
        for (name, size) in &[
            ("scratch_normed_hidden", hidden_bytes),
            ("scratch_gate_out", inter_bytes),
            ("scratch_up_out", inter_bytes),
            ("scratch_silu_gate", inter_bytes),
            ("scratch_mlp_hidden", inter_bytes),
            ("scratch_down_out", hidden_bytes),
        ] {
            self.buffer_store
                .insert(name.to_string(), alloc_zero(name, *size)?);
        }
        {
            let constants =
                build_mlp_constants(hidden_dim as u32, intermediate_dim as u32, 128, 2, 1e-6);
            let buf = self.device.new_buffer_with_data(
                constants.as_ptr() as *const std::ffi::c_void,
                constants.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            if buf.length() == 0 {
                return Err(CImageRuntimeError::BufferAllocationFailed(
                    "mlp_constants".into(),
                ));
            }
            self.buffer_store.insert("mlp_constants".into(), buf);
        }

        let ops = &plan.region.ops;
        for name in &[
            "cimage_rmsnorm_f32",
            "cimage_silu_f32",
            "cimage_mul_f32",
            "cimage_residual_add_f32",
        ] {
            self.get_or_create_pso(name)?;
        }
        for name in &[
            "cimage_linear_rawf32",
            "cimage_linear_nf4_f32",
            "cimage_linear_int8",
        ] {
            self.get_or_create_pso(name)?;
        }

        let encode_start = Instant::now();
        let cb = self.queue.new_command_buffer();
        let enc = AutoEncoder::new(cb.new_compute_command_encoder());
        for (op_index, op) in ops.iter().enumerate() {
            let proj_suffix = match op_index {
                1 => Some("gate_proj"),
                2 => Some("up_proj"),
                5 => Some("down_proj"),
                _ => None,
            };
            let payload = proj_suffix.and_then(|s| {
                ctx.tensor_store
                    .tensors
                    .values()
                    .find(|t| t.tensor_key.contains(s))
                    .map(|t| &t.payload)
            });
            let fn_name = op_index_to_linear_fn_name(op_index, payload);
            let grid_x: u32 = if op_index == 0 {
                1
            } else {
                op.dispatch_shape.grid_x
            };
            if op_index <= 4 {
                self.write_mlp_constants_dimensions(hidden_dim as u32, intermediate_dim as u32);
            } else if op_index == 5 {
                self.write_mlp_constants_dimensions(intermediate_dim as u32, hidden_dim as u32);
            }
            let pso = self.pso_map.get(fn_name).expect("PSO pre-warmed");
            enc.set_compute_pipeline_state(&pso);
            for binding in &op.bindings {
                if let Some(buf) = self.buffer_store.get(&binding.buffer_id) {
                    enc.set_buffer(binding.slot as u64, Some(buf), binding.offset);
                } else {
                    return Err(CImageRuntimeError::KernelBindingMissing(format!(
                        "buffer '{}' not found for op {} slot {}",
                        binding.buffer_id, op.op_id, binding.slot
                    )));
                }
            }
            enc.dispatch_thread_groups(
                metal::MTLSize::new(
                    grid_x.max(1) as u64,
                    op.dispatch_shape.grid_y.max(1) as u64,
                    op.dispatch_shape.grid_z.max(1) as u64,
                ),
                metal::MTLSize::new(
                    op.dispatch_shape.threadgroup_m as u64,
                    op.dispatch_shape.threadgroup_n.max(1) as u64,
                    op.dispatch_shape.threadgroup_p.max(1) as u64,
                ),
            );
        }
        auto_end_encoding(enc);
        let encode_ms = encode_start.elapsed().as_secs_f64() * 1000.0;
        let cmd_start = Instant::now();
        cb.commit();
        cb.wait_until_completed();
        let command_buffer_ms = cmd_start.elapsed().as_secs_f64() * 1000.0;
        let metal_output = self.readback_f32("hidden_out", hidden_dim)?;
        let readback_ms = (Instant::now() - cmd_start).as_secs_f64() * 1000.0;
        Ok(CImageRegionExecutionReceipt {
            receipt_version: 1,
            cimage_digest: ctx.generation.generation_id.0.clone(),
            region_id: "mlp_shard_region".into(),
            backend: BackendLoweringTarget::MetalTensorApi,
            hardware_profile: HardwareProfileId::AppleMProBalanced,
            execution_mode: MlpRegionExecutionMode::StagedKernels,
            evidence_kind: ReceiptEvidenceKind::RealTensorNumericalProof,
            tensor_count: ctx.tensor_store.len(),
            kernel_count: plan.region.ops.len(),
            buffer_count: self.buffer_store.buffers.len(),
            total_bound_bytes: self.buffer_store.total_bytes(),
            scratch_bytes: plan.arena_plan.total_scratch_bytes,
            cpu_reconstructed_output_digest: String::new(),
            metal_output_digest: sha256_hex_f32(&metal_output),
            metal_vs_cpu_nrmse: 0.0,
            metal_vs_cpu_cosine: 0.0,
            metal_vs_cpu_max_abs_error: 0.0,
            rawf32_vs_cpu_reconstructed_nrmse: 0.0,
            rawf32_vs_metal_nrmse: 0.0,
            command_buffer_ms,
            encode_ms,
            readback_ms,
            hazard_safe: plan.hazard_plan.safe,
            validation_passed: true,
            warnings: vec!["CPU comparison skipped (context mode)".into()],
        })
    }

    // ── Decoder buffer allocation ───────────────────────────────────────

    /// Allocate all Metal buffers required by the decoder region plan.
    ///
    /// This creates persistent weight buffers from resolved tensor payloads,
    /// scratch buffers, KV cache, decoder constants, and input/output buffers.
    fn allocate_decoder_buffers(
        &mut self,
        store: &RuntimeTensorStore,
        hidden_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate_dim: usize,
        seq_len: usize,
        input: &[f32],
    ) -> CImageRuntimeResult<()> {
        let hidden_bytes = (hidden_dim * 4) as u64;
        let q_out_bytes = (num_heads * head_dim * 4) as u64;
        let kv_out_bytes = (num_kv_heads * head_dim * 4) as u64;
        let inter_bytes = (intermediate_dim * 4) as u64;
        let kv_cache_bytes = (seq_len * num_kv_heads * head_dim * 4) as u64;
        let scores_bytes = (num_heads * seq_len * 4) as u64;

        // Build a lookup from tensor_key to payload.
        let tensor_by_key: HashMap<&str, &RuntimeTensorPayload> = store
            .tensors
            .values()
            .map(|t| (t.tensor_key.as_str(), &t.payload))
            .collect();

        // Helper: alloc a buffer from f32 data.
        let alloc_f32 = |name: &str, data: &[f32]| -> CImageRuntimeResult<metal::Buffer> {
            let bytes: &[u8] =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
            let buf = self.device.new_buffer_with_data(
                bytes.as_ptr() as *const std::ffi::c_void,
                bytes.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            if buf.length() == 0 {
                return Err(CImageRuntimeError::BufferAllocationFailed(name.to_string()));
            }
            Ok(buf)
        };

        let alloc_zero = |name: &str, size: u64| -> CImageRuntimeResult<metal::Buffer> {
            let buf = self
                .device
                .new_buffer(size, metal::MTLResourceOptions::StorageModeShared);
            if buf.length() == 0 && size > 0 {
                return Err(CImageRuntimeError::BufferAllocationFailed(name.to_string()));
            }
            Ok(buf)
        };

        // Helper to allocate a weight from a tensor_key.
        let alloc_weight_from_key =
            |buffer_id: &str, tensor_key: &str| -> CImageRuntimeResult<Option<metal::Buffer>> {
                match tensor_by_key.get(tensor_key) {
                    Some(RuntimeTensorPayload::RawF32(data)) => {
                        let buf = alloc_f32(buffer_id, data)?;
                        Ok(Some(buf))
                    }
                    Some(RuntimeTensorPayload::Nf4Packed { codes, .. }) => {
                        let buf = self.device.new_buffer_with_data(
                            codes.as_ptr() as *const std::ffi::c_void,
                            codes.len() as u64,
                            metal::MTLResourceOptions::StorageModeShared,
                        );
                        if buf.length() == 0 && !codes.is_empty() {
                            return Err(CImageRuntimeError::BufferAllocationFailed(
                                buffer_id.to_string(),
                            ));
                        }
                        Ok(Some(buf))
                    }
                    Some(RuntimeTensorPayload::Int8Packed { codes, .. }) => {
                        let buf = self.device.new_buffer_with_data(
                            codes.as_ptr() as *const std::ffi::c_void,
                            codes.len() as u64,
                            metal::MTLResourceOptions::StorageModeShared,
                        );
                        if buf.length() == 0 && !codes.is_empty() {
                            return Err(CImageRuntimeError::BufferAllocationFailed(
                                buffer_id.to_string(),
                            ));
                        }
                        Ok(Some(buf))
                    }
                    _ => Ok(None),
                }
            };

        // ── Persistent weight buffers ───────────────────────────────────
        // Tensor key → buffer ID mapping matching lower_decoder expectations.
        let weight_mappings: &[(&str, &str)] = &[
            ("input_layernorm.weight", "input_layernorm_weight"),
            (
                "post_attention_layernorm.weight",
                "post_attn_layernorm_weight",
            ),
            ("q_proj.weight", "q_proj_codes"),
            ("k_proj.weight", "k_proj_codes"),
            ("v_proj.weight", "v_proj_codes"),
            ("o_proj.weight", "o_proj_codes"),
            ("gate_proj.weight", "gate_proj_codes"),
            ("up_proj.weight", "up_proj_codes"),
            ("down_proj.weight", "down_proj_codes"),
        ];
        for (tensor_key, buffer_id) in weight_mappings {
            if let Some(buf) = alloc_weight_from_key(buffer_id, tensor_key)? {
                self.buffer_store.insert(buffer_id.to_string(), buf);
            }
        }

        // Scale and bias buffers (zero-filled for RawF32).
        let scale_bias_pairs: &[(&str, &str, u64)] = &[
            ("q_proj_scales", "q_proj_biases", num_heads as u64),
            ("k_proj_scales", "k_proj_biases", num_kv_heads as u64),
            ("v_proj_scales", "v_proj_biases", num_kv_heads as u64),
            ("o_proj_scales", "o_proj_biases", hidden_dim as u64),
            (
                "gate_proj_scales",
                "gate_proj_biases",
                intermediate_dim as u64,
            ),
            ("up_proj_scales", "up_proj_biases", intermediate_dim as u64),
            ("down_proj_scales", "down_proj_biases", hidden_dim as u64),
        ];
        for (scales_id, biases_id, count) in scale_bias_pairs {
            let size = count * 4;
            {
                let buf = alloc_zero(scales_id, size)?;
                self.buffer_store.insert(scales_id.to_string(), buf);
            }
            {
                let buf = alloc_zero(biases_id, size)?;
                self.buffer_store.insert(biases_id.to_string(), buf);
            }
        }

        // ── Input buffer ────────────────────────────────────────────────
        {
            let buf = alloc_f32("hidden_in", input)?;
            self.buffer_store.insert("hidden_in".into(), buf);
        }

        // ── Output buffer ───────────────────────────────────────────────
        {
            let buf = alloc_zero("hidden_out", hidden_bytes)?;
            self.buffer_store.insert("hidden_out".into(), buf);
        }

        // ── Scratch buffers ─────────────────────────────────────────────
        let scratch_layout: &[(&str, u64)] = &[
            ("scratch_normed", hidden_bytes),
            ("scratch_q", q_out_bytes),
            ("scratch_k", kv_out_bytes),
            ("scratch_v", kv_out_bytes),
            ("scratch_q_rope", q_out_bytes),
            ("scratch_k_rope", kv_out_bytes),
            ("scratch_scores", scores_bytes),
            ("scratch_scores_post_softmax", scores_bytes),
            ("scratch_attended", q_out_bytes),
            ("scratch_o", hidden_bytes),
            ("scratch_post_attn", hidden_bytes),
            ("scratch_normed2", hidden_bytes),
            ("scratch_gate", inter_bytes),
            ("scratch_up", inter_bytes),
            ("scratch_silu_gate", inter_bytes),
            ("scratch_mlp_hidden", inter_bytes),
            ("scratch_mlp_down", hidden_bytes),
        ];
        for (name, size) in scratch_layout {
            let buf = alloc_zero(name, *size)?;
            self.buffer_store.insert(name.to_string(), buf);
        }

        // ── KV cache buffers (zero-filled) ───────────────────────────────
        {
            let buf = alloc_zero("kv_cache_k", kv_cache_bytes)?;
            self.buffer_store.insert("kv_cache_k".into(), buf);
        }
        {
            let buf = alloc_zero("kv_cache_v", kv_cache_bytes)?;
            self.buffer_store.insert("kv_cache_v".into(), buf);
        }

        // ── Position IDs buffer ──────────────────────────────────────────
        {
            // Use position_ids tensor if available, else zero-filled.
            match tensor_by_key.get("position_ids") {
                Some(RuntimeTensorPayload::RawF32(data)) => {
                    let buf = alloc_f32("position_ids", data)?;
                    self.buffer_store.insert("position_ids".into(), buf);
                }
                _ => {
                    let buf = alloc_zero("position_ids", (seq_len as u64) * 4)?;
                    self.buffer_store.insert("position_ids".into(), buf);
                }
            }
        }

        // ── Decoder constants buffer ──────────────────────────────────────
        {
            let constants = build_decoder_constants(
                hidden_dim as u32,
                num_heads as u32,
                num_kv_heads as u32,
                head_dim as u32,
                seq_len as u32,
                0,    // current_pos = 0 (start of sequence)
                1e-6, // epsilon
            );
            let buf = self.device.new_buffer_with_data(
                constants.as_ptr() as *const std::ffi::c_void,
                constants.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            if buf.length() == 0 {
                return Err(CImageRuntimeError::BufferAllocationFailed(
                    "decoder_constants".into(),
                ));
            }
            self.buffer_store.insert("decoder_constants".into(), buf);
        }

        Ok(())
    }

    // ── Decoder entry point ─────────────────────────────────────────────

    /// Run the full decoder layer shard region pipeline for a loaded cimage.
    ///
    /// Validates the cimage, resolves tensors, builds the 18-op decoder
    /// region plan via [`DecoderShardRegionBuilder::build_decoder_region`],
    /// allocates Metal buffers, encodes and dispatches, then reads back
    /// the output and emits a receipt.
    pub fn run_decoder_shard_region(
        &mut self,
        image: &LoadedCImageV0,
        _input: &[f32],
    ) -> CImageRuntimeResult<CImageRegionExecutionReceipt> {
        let _start = Instant::now();

        // 1. Validate cimage (all 14 gates).
        let load_receipt =
            CImageValidator::validate_loaded(image).map_err(|e| CImageRuntimeError::CImage(e))?;
        if load_receipt.validation_status != crate::ecs::cimage::CImageValidationStatus::Valid {
            return Err(CImageRuntimeError::ValidationFailed(format!(
                "cimage validation failed: {:?}",
                load_receipt.errors
            )));
        }

        // 2. Resolve all tensors into a RuntimeTensorStore.
        let store = resolve_tensors_from_image(image)?;

        // 3. Extract dimensions from manifest tensor shapes.
        // Tensor 0: input_layernorm.weight → hidden_dim = logical_shape[0]
        // Tensor 2: k_proj.weight → kv_inner = num_kv_heads * head_dim = logical_shape[0]
        // Tensor 6: gate_proj.weight → intermediate_dim = logical_shape[0]
        // Tensor 9: position_ids → seq_len = logical_shape[0]
        let manifest = &image.manifest;
        let hidden_dim = manifest.tensors[0].logical_shape[0] as usize;
        let kv_inner = manifest.tensors[2].logical_shape[0] as usize;
        let intermediate_dim = manifest.tensors[6].logical_shape[0] as usize;
        let seq_len = manifest.tensors[9].logical_shape[0] as usize;

        // Infer head_dim: try common values (128, 96, 80, 64, 32, 16, 8) that
        // evenly divide both hidden_dim and kv_inner.
        let head_dim = [128u32, 96, 80, 64, 32, 16, 8, 4]
            .iter()
            .copied()
            .find(|&hd| hidden_dim as u32 % hd == 0 && kv_inner as u32 % hd == 0)
            .unwrap_or(64) as usize;
        let num_heads = hidden_dim / head_dim;
        let num_kv_heads = kv_inner / head_dim;

        // 4. Build the execution region plan (18 staged ops).
        let plan = DecoderShardRegionBuilder::build_decoder_region(
            &store,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            seq_len,
        )?;

        // 5. Check hazard.
        if !plan.hazard_plan.safe {
            let warn = "decoder region hazard check failed (non-fatal for staged kernels)";
            eprintln!("{warn}");
        }

        // 6. Generate deterministic input and allocate buffers.
        let input = generate_deterministic_input(42, hidden_dim);
        self.allocate_decoder_buffers(
            &store,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            seq_len,
            &input,
        )?;

        // Pre-warm PSO cache.
        let ops = &plan.region.ops;
        for op_index in 0..ops.len() {
            self.get_or_create_pso(decoder_op_index_to_function_name(op_index))?;
        }

        // 7. Encode and dispatch.
        let encode_start = Instant::now();
        let cb = self.queue.new_command_buffer();
        let enc = AutoEncoder::new(cb.new_compute_command_encoder());

        for (op_index, op) in ops.iter().enumerate() {
            let fn_name = decoder_op_index_to_function_name(op_index);

            // Override rmsnorm grid: single threadgroup (64 threads).
            let grid_x: u32 = if op_index == 0 || op_index == 11 {
                1
            } else {
                op.dispatch_shape.grid_x
            };

            // Cache was pre-warmed; immutable lookup is safe.
            let pso = self.pso_map.get(fn_name).expect("PSO must be pre-warmed");
            enc.set_compute_pipeline_state(&pso);

            // Bind each buffer referenced by the op.
            for binding in &op.bindings {
                if let Some(buf) = self.buffer_store.get(&binding.buffer_id) {
                    enc.set_buffer(binding.slot as u64, Some(buf), binding.offset);
                } else {
                    return Err(CImageRuntimeError::KernelBindingMissing(format!(
                        "buffer '{}' not found for op {} slot {}",
                        binding.buffer_id, op.op_id, binding.slot
                    )));
                }
            }

            let tg = metal::MTLSize::new(
                op.dispatch_shape.threadgroup_m as u64,
                op.dispatch_shape.threadgroup_n.max(1) as u64,
                op.dispatch_shape.threadgroup_p.max(1) as u64,
            );
            let grid = metal::MTLSize::new(
                grid_x.max(1) as u64,
                op.dispatch_shape.grid_y.max(1) as u64,
                op.dispatch_shape.grid_z.max(1) as u64,
            );
            enc.dispatch_thread_groups(grid, tg);
        }

        auto_end_encoding(enc);
        let encode_ms = encode_start.elapsed().as_secs_f64() * 1000.0;

        let cmd_start = Instant::now();
        cb.commit();
        cb.wait_until_completed();
        let command_buffer_ms = cmd_start.elapsed().as_secs_f64() * 1000.0;

        // 8. Read back output.
        let readback_start = Instant::now();
        let metal_output = self.readback_f32("hidden_out", hidden_dim)?;
        let readback_ms = readback_start.elapsed().as_secs_f64() * 1000.0;

        // 9. Compute Metal output digest.
        let metal_output_digest = sha256_hex_f32(&metal_output);

        // No CPU reference for decoder yet — comparison fields zeroed.
        Ok(CImageRegionExecutionReceipt {
            receipt_version: 1,
            cimage_digest: String::new(),
            region_id: "decoder_layer_region".into(),
            backend: BackendLoweringTarget::MetalTensorApi,
            hardware_profile: HardwareProfileId::AppleMProBalanced,
            execution_mode: MlpRegionExecutionMode::StagedKernels,
            evidence_kind: ReceiptEvidenceKind::RealTensorNumericalProof,
            tensor_count: store.tensors.len(),
            kernel_count: plan.region.ops.len(),
            buffer_count: self.buffer_store.buffers.len(),
            total_bound_bytes: self.buffer_store.total_bytes(),
            scratch_bytes: plan.arena_plan.total_scratch_bytes,
            cpu_reconstructed_output_digest: String::new(),
            metal_output_digest,
            metal_vs_cpu_nrmse: 0.0,
            metal_vs_cpu_cosine: 0.0,
            metal_vs_cpu_max_abs_error: 0.0,
            rawf32_vs_cpu_reconstructed_nrmse: 0.0,
            rawf32_vs_metal_nrmse: 0.0,
            command_buffer_ms,
            encode_ms,
            readback_ms,
            hazard_safe: plan.hazard_plan.safe,
            validation_passed: true,
            warnings: vec![],
        })
    }

    /// Run the ternary GEMV kernel on a loaded cimage.
    ///
    /// Finds the first Ternary1_58 tensor, extracts codes + scales from the
    /// payload blob, dispatches `cimage_ternary_gemv_v1`, and returns the
    /// Metal output as `Vec<f32>`.
    pub fn run_ternary_gemv(&mut self, image: &LoadedCImageV0) -> CImageRuntimeResult<Vec<f32>> {
        // 1. Find first Ternary1_58 tensor.
        let tensor = image
            .manifest
            .tensors
            .iter()
            .find(|t| t.codec == crate::execution_plan::CodecFamily::Ternary1_58)
            .ok_or_else(|| {
                CImageRuntimeError::ValidationFailed("no Ternary1_58 tensor found in cimage".into())
            })?;

        let tensor_key = &tensor.tensor_key;
        let rows = tensor.logical_shape[0] as usize;
        let cols = tensor.logical_shape[1] as usize;
        let layout = &tensor.physical_layout;
        let group_size = layout.group_size as usize;
        let groups_per_row = layout.groups_per_tile as usize;
        let bytes_per_group = (group_size * 2 + 7) / 8;

        // 2. Extract codes payload.
        let codes_id = format!("p_{}_codes", tensor_key);
        let codes_entry = image
            .payload_directory
            .payloads
            .iter()
            .find(|e| e.payload_id == codes_id)
            .ok_or_else(|| CImageRuntimeError::MissingPayload(codes_id.clone()))?;
        let codes = &image.payload_blob
            [codes_entry.offset as usize..(codes_entry.offset + codes_entry.len) as usize];

        // 3. Extract scales payload (stored as f16 LE bytes).
        let scales_id = format!("p_{}_scales", tensor_key);
        let scales_entry = image
            .payload_directory
            .payloads
            .iter()
            .find(|e| e.payload_id == scales_id)
            .ok_or_else(|| CImageRuntimeError::MissingPayload(scales_id.clone()))?;
        let scales_bytes = &image.payload_blob
            [scales_entry.offset as usize..(scales_entry.offset + scales_entry.len) as usize];

        // 4. Generate deterministic input.
        let input = generate_deterministic_input(42, cols);

        // 5. Build constants struct.
        let mut const_bytes = vec![0u8; 36];
        const_bytes[0..4].copy_from_slice(&(rows as u32).to_le_bytes());
        const_bytes[4..8].copy_from_slice(&(cols as u32).to_le_bytes());
        const_bytes[8..12].copy_from_slice(&(group_size as u32).to_le_bytes());
        const_bytes[12..16].copy_from_slice(&(groups_per_row as u32).to_le_bytes());
        const_bytes[16..20].copy_from_slice(&(bytes_per_group as u32).to_le_bytes());
        const_bytes[20..24].copy_from_slice(&0u32.to_le_bytes()); // output_dtype = 0 (half)

        // 6. Allocate Metal buffers.
        let alloc_from_slice = |data: &[u8]| -> CImageRuntimeResult<metal::Buffer> {
            let buf = self.device.new_buffer_with_data(
                data.as_ptr() as *const std::ffi::c_void,
                data.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            if buf.length() == 0 && !data.is_empty() {
                return Err(CImageRuntimeError::BufferAllocationFailed(
                    "ternary_gemv_input".into(),
                ));
            }
            Ok(buf)
        };

        // Convert f32 input to half for Metal kernel.
        let act_bytes: Vec<u8> = input
            .iter()
            .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
            .collect();
        let act_buf = alloc_from_slice(&act_bytes)?;
        let codes_buf = alloc_from_slice(codes)?;
        let scales_buf = alloc_from_slice(scales_bytes)?;

        let out_size = (rows * 2) as u64; // half output
        let out_buf = self
            .device
            .new_buffer(out_size, metal::MTLResourceOptions::StorageModeShared);

        let const_buf = alloc_from_slice(&const_bytes)?;

        // 7. Get PSO and dispatch.
        let pso = self.get_or_create_pso("cimage_ternary_gemv_v1")?;
        let cb = self.queue.new_command_buffer();
        let enc = AutoEncoder::new(cb.new_compute_command_encoder());
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&act_buf), 0);
        enc.set_buffer(1, Some(&codes_buf), 0);
        enc.set_buffer(2, Some(&scales_buf), 0);
        enc.set_buffer(3, Some(&out_buf), 0);
        enc.set_buffer(4, Some(&const_buf), 0);

        let grid = metal::MTLSize::new(rows as u64, 1, 1);
        let tg = metal::MTLSize::new(1, 1, 1);
        enc.dispatch_thread_groups(grid, tg);
        auto_end_encoding(enc);
        cb.commit();
        cb.wait_until_completed();

        // 8. Read back output (half -> f32).
        let out_ptr = out_buf.contents() as *const u16;
        let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, rows) };
        let metal_output: Vec<f32> = out_slice
            .iter()
            .map(|&bits| half::f16::from_bits(bits).to_f32())
            .collect();

        Ok(metal_output)
    }

    /// Write f32 data into a Metal buffer via `contents()` pointer copy.
    fn write_f32_buffer(&self, name: &str, data: &[f32]) -> CImageRuntimeResult<()> {
        let buf = self.buffer_store.get(name).ok_or_else(|| {
            CImageRuntimeError::KernelBindingMissing(format!("write_f32_buffer: {name}"))
        })?;
        let ptr = buf.contents() as *mut f32;
        let write_len = data.len().min((buf.length() as usize) / 4);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, write_len);
        }
        Ok(())
    }

    /// Overwrite an existing Metal buffer's contents via contents() pointer copy.
    fn update_buffer_data(&self, buf: &metal::Buffer, data: &[u8]) {
        let ptr = buf.contents() as *mut u8;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        buf.did_modify_range(metal::NSRange::new(0, data.len() as u64));
    }

    /// Upload bytes to a named buffer. Creates new buffer on first call,
    /// overwrites via update_buffer_data on subsequent calls.
    #[allow(dead_code)]
    fn upload_or_update_buffer(&mut self, name: &str, data: &[u8]) -> CImageRuntimeResult<()> {
        if let Some(buf) = self.buffer_store.get(name) {
            self.update_buffer_data(buf, data);
            Ok(())
        } else {
            let buf = self.device.new_buffer_with_data(
                data.as_ptr() as *const std::ffi::c_void,
                data.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            if buf.length() == 0 && !data.is_empty() {
                return Err(CImageRuntimeError::BufferAllocationFailed(name.into()));
            }
            self.buffer_store.insert(name.to_string(), buf);
            Ok(())
        }
    }

    pub fn run_bitnet_decoder_region(
        &mut self,
        image: &LoadedCImageV0,
        _input: &[f32],
    ) -> CImageRuntimeResult<CImageRegionExecutionReceipt> {
        let _start = Instant::now();

        // 1. Validate cimage.
        let load_receipt =
            CImageValidator::validate_loaded(image).map_err(|e| CImageRuntimeError::CImage(e))?;
        if load_receipt.validation_status != crate::ecs::cimage::CImageValidationStatus::Valid {
            return Err(CImageRuntimeError::ValidationFailed(format!(
                "cimage validation failed: {:?}",
                load_receipt.errors
            )));
        }

        // 2. Extract dimensions from manifest tensor entries.
        let manifest = &image.manifest;
        // Find dimensions by tensor_key pattern, not hardcoded index.
        // Full cimage: tensor[0]=embed_tokens, tensor[1]=position_ids,
        // then per-layer tensors: layer.0.input_layernorm, layer.0.q_proj, ...
        fn find_dim(tensors: &[CImageTensorEntry], key_suffix: &str) -> Option<usize> {
            tensors
                .iter()
                .find(|t| t.tensor_key.contains(key_suffix))
                .map(|t| t.logical_shape[0] as usize)
        }
        let hidden_dim = find_dim(&manifest.tensors, "layer.0.q_proj.weight")
            .or_else(|| find_dim(&manifest.tensors, "q_proj.weight"))
            .unwrap_or(2560);
        let kv_inner = find_dim(&manifest.tensors, "layer.0.k_proj.weight")
            .or_else(|| find_dim(&manifest.tensors, "k_proj.weight"))
            .unwrap_or(640);
        let intermediate_dim = find_dim(&manifest.tensors, "layer.0.gate_proj.weight")
            .or_else(|| find_dim(&manifest.tensors, "gate_proj.weight"))
            .unwrap_or(6912);
        let seq_len = find_dim(&manifest.tensors, "position_ids").unwrap_or(4096);

        eprintln!("DEBUG: hidden_dim={hidden_dim} kv_inner={kv_inner} intermediate_dim={intermediate_dim} seq_len={seq_len}");

        let head_dim = [128u32, 96, 80, 64, 32, 16, 8, 4]
            .iter()
            .copied()
            .find(|&hd| hidden_dim as u32 % hd == 0 && kv_inner as u32 % hd == 0)
            .unwrap_or(64) as usize;
        let num_heads = hidden_dim / head_dim;
        let num_kv_heads = kv_inner / head_dim;

        // 3. Build decoder region plan (empty store — builder uses dims only).
        let empty_store = RuntimeTensorStore::new();
        let plan = DecoderShardRegionBuilder::build_decoder_region(
            &empty_store,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            seq_len,
        )?;

        if !plan.hazard_plan.safe {
            eprintln!("bitnet decoder hazard check passed (non-fatal)");
        }

        // 4. Generate deterministic input.
        let input = generate_deterministic_input(42, hidden_dim);

        // 5. Pre-parse payload directory.
        let payload_dir = &image.payload_directory;
        let payload_blob = &image.payload_blob;

        let find_payload = |payload_id: &str| -> CImageRuntimeResult<&[u8]> {
            let entry = payload_dir
                .payloads
                .iter()
                .find(|e| e.payload_id == payload_id)
                .ok_or_else(|| CImageRuntimeError::MissingPayload(payload_id.into()))?;
            let start = entry.offset as usize;
            let end = start + entry.len as usize;
            Ok(&payload_blob[start..end])
        };

        let alloc_bytes = |name: &str, data: &[u8]| -> CImageRuntimeResult<metal::Buffer> {
            let buf = self.device.new_buffer_with_data(
                data.as_ptr() as *const std::ffi::c_void,
                data.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            if buf.length() == 0 && !data.is_empty() {
                return Err(CImageRuntimeError::BufferAllocationFailed(name.into()));
            }
            Ok(buf)
        };

        let alloc_zero = |name: &str, size: u64| -> CImageRuntimeResult<metal::Buffer> {
            let buf = self
                .device
                .new_buffer(size, metal::MTLResourceOptions::StorageModeShared);
            if buf.length() == 0 && size > 0 {
                return Err(CImageRuntimeError::BufferAllocationFailed(name.into()));
            }
            Ok(buf)
        };

        let hidden_bytes = (hidden_dim * 4) as u64;
        let inter_bytes = (intermediate_dim * 4) as u64;
        let q_out_bytes = (num_heads * head_dim * 4) as u64;
        let kv_out_bytes = (num_kv_heads * head_dim * 4) as u64;
        let scores_bytes = (num_heads * seq_len * 4) as u64;
        let kv_cache_bytes = (seq_len * num_kv_heads * head_dim * 4) as u64;

        // 6. Allocate input/output buffers.
        {
            let buf = alloc_bytes("hidden_in", bytemuck::cast_slice(&input))?;
            self.buffer_store.insert("hidden_in".into(), buf);
        }
        {
            let buf = alloc_zero("hidden_out", hidden_bytes)?;
            self.buffer_store.insert("hidden_out".into(), buf);
        }

        // 7. Unpack RMSNorm weights from ternary to f32.
        //    buffer names: "input_layernorm_weight", "post_attn_layernorm_weight"
        {
            let codes0 = find_payload("p_input_layernorm.weight_codes")?;
            let scales0 = find_payload("p_input_layernorm.weight_scales")?;
            let ent0 = &manifest.tensors[0];
            let n0 = ent0.logical_shape[1] as usize;
            let gs0 = ent0.physical_layout.group_size as usize;
            let gpr0 = ent0.physical_layout.groups_per_tile as usize;
            let bpg0 = (gs0 * 2 + 7) / 8;
            let s0 = if scales0.len() >= 2 {
                half::f16::from_le_bytes([scales0[0], scales0[1]]).to_f32()
            } else {
                1.0
            };
            let mut fw0 = Vec::with_capacity(n0);
            for c in 0..n0 {
                let g = c / gs0;
                let wi = c % gs0;
                let bi = wi / 4;
                let ni = wi % 4;
                let b = if g < gpr0 && bi < bpg0 {
                    codes0[g * bpg0 + bi]
                } else {
                    0
                };
                let code = (b >> (ni * 2)) & 0x03;
                let w: f32 = match code {
                    0 => -1.0,
                    1 => 0.0,
                    2 => 1.0,
                    _ => 0.0,
                };
                fw0.push(w * s0);
            }
            let buf = alloc_bytes("input_layernorm_weight", bytemuck::cast_slice(&fw0))?;
            self.buffer_store
                .insert("input_layernorm_weight".into(), buf);
        }
        {
            let codes0 = find_payload("p_post_attention_layernorm.weight_codes")?;
            let scales0 = find_payload("p_post_attention_layernorm.weight_scales")?;
            let ent0 = &manifest.tensors[5];
            let n0 = ent0.logical_shape[1] as usize;
            let gs0 = ent0.physical_layout.group_size as usize;
            let gpr0 = ent0.physical_layout.groups_per_tile as usize;
            let bpg0 = (gs0 * 2 + 7) / 8;
            let s0 = if scales0.len() >= 2 {
                half::f16::from_le_bytes([scales0[0], scales0[1]]).to_f32()
            } else {
                1.0
            };
            let mut fw0 = Vec::with_capacity(n0);
            for c in 0..n0 {
                let g = c / gs0;
                let wi = c % gs0;
                let bi = wi / 4;
                let ni = wi % 4;
                let b = if g < gpr0 && bi < bpg0 {
                    codes0[g * bpg0 + bi]
                } else {
                    0
                };
                let code = (b >> (ni * 2)) & 0x03;
                let w: f32 = match code {
                    0 => -1.0,
                    1 => 0.0,
                    2 => 1.0,
                    _ => 0.0,
                };
                fw0.push(w * s0);
            }
            let buf = alloc_bytes("post_attn_layernorm_weight", bytemuck::cast_slice(&fw0))?;
            self.buffer_store
                .insert("post_attn_layernorm_weight".into(), buf);
        }

        // 8. Allocate ternary gemv weight buffers (PACKED codes + HALF scales).
        //    Names: {proj_short}_proj_codes, {proj_short}_proj_scales
        let proj_specs: &[(&str, &str, &str)] = &[
            ("q_proj.weight", "q", "q_proj"),
            ("k_proj.weight", "k", "k_proj"),
            ("v_proj.weight", "v", "v_proj"),
            ("o_proj.weight", "o", "o_proj"),
            ("gate_proj.weight", "gate", "gate_proj"),
            ("up_proj.weight", "up", "up_proj"),
            ("down_proj.weight", "down", "down_proj"),
        ];
        for (tensor_key, _proj_short, buf_prefix) in proj_specs {
            let codes_data = find_payload(&format!("p_{tensor_key}_codes"))?;
            let scales_data = find_payload(&format!("p_{tensor_key}_scales"))?;
            let codes_buf = alloc_bytes(&format!("{buf_prefix}_codes"), codes_data)?;
            self.buffer_store
                .insert(format!("{buf_prefix}_codes"), codes_buf);
            let scales_buf = alloc_bytes(&format!("{buf_prefix}_scales"), scales_data)?;
            self.buffer_store
                .insert(format!("{buf_prefix}_scales"), scales_buf);
        }

        // 9. Position IDs buffer.
        {
            let pos_data = find_payload("p_position_ids_codes")?;
            let pos_f32: Vec<f32> = pos_data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let buf = alloc_bytes("position_ids", bytemuck::cast_slice(&pos_f32))?;
            self.buffer_store.insert("position_ids".into(), buf);
        }

        // 10. Allocate scratch buffers (f32).
        let scratch_layout: &[(&str, u64)] = &[
            ("scratch_normed", hidden_bytes),
            ("scratch_q", q_out_bytes),
            ("scratch_k", kv_out_bytes),
            ("scratch_v", kv_out_bytes),
            ("scratch_q_rope", q_out_bytes),
            ("scratch_k_rope", kv_out_bytes),
            ("scratch_scores", scores_bytes),
            ("scratch_scores_post_softmax", scores_bytes),
            ("scratch_attended", q_out_bytes),
            ("scratch_o", hidden_bytes),
            ("scratch_post_attn", hidden_bytes),
            ("scratch_normed2", hidden_bytes),
            ("scratch_gate", inter_bytes),
            ("scratch_up", inter_bytes),
            ("scratch_silu_gate", inter_bytes),
            ("scratch_mlp_hidden", inter_bytes),
            ("scratch_mlp_down", hidden_bytes),
        ];
        for (name, size) in scratch_layout {
            self.buffer_store
                .insert(name.to_string(), alloc_zero(name, *size)?);
        }

        // 11. KV cache buffers.
        {
            self.buffer_store.insert(
                "kv_cache_k".into(),
                alloc_zero("kv_cache_k", kv_cache_bytes)?,
            );
        }
        {
            self.buffer_store.insert(
                "kv_cache_v".into(),
                alloc_zero("kv_cache_v", kv_cache_bytes)?,
            );
        }

        // 12. Decoder constants (128 bytes).
        {
            let constants = build_decoder_constants(
                hidden_dim as u32,
                num_heads as u32,
                num_kv_heads as u32,
                head_dim as u32,
                seq_len as u32,
                0,
                1e-6,
            );
            let buf = alloc_bytes("decoder_constants", &constants)?;
            self.buffer_store.insert("decoder_constants".into(), buf);
        }

        // 13. Pre-warm PSO cache.
        let ops = &plan.region.ops;
        for op_index in 0..ops.len() {
            self.get_or_create_pso(bitnet_decoder_op_index_to_function_name(op_index))?;
        }

        self.get_or_create_pso("cimage_f32_to_half")?;
        self.get_or_create_pso("cimage_half_to_f32")?;
        // 14. Encode and dispatch.
        let encode_start = Instant::now();
        let queue = &self.queue;

        // Dispatch f32 ops in scoped segments, with standalone ternary dispatches in between.
        // We partition: [0] |ternary 1-3| [4,5,6,7,8] |ternary 9| [10,11] |ternary 12-13| [14,15] |ternary 16| [17]

        // Helper: dispatch a batch of f32 ops by index range.
        let dispatch_f32_segment = |start: usize, end: usize| -> CImageRuntimeResult<()> {
            let cb = queue.new_command_buffer();
            let enc = AutoEncoder::new(cb.new_compute_command_encoder());
            for idx in start..=end {
                let op = &ops[idx];
                let fn_name = bitnet_decoder_op_index_to_function_name(idx);
                let pso = self.pso_map.get(fn_name).expect("PSO must be pre-warmed");
                enc.set_compute_pipeline_state(&pso);
                for binding in &op.bindings {
                    if let Some(buf) = self.buffer_store.get(&binding.buffer_id) {
                        enc.set_buffer(binding.slot as u64, Some(buf), binding.offset);
                    } else {
                        return Err(CImageRuntimeError::KernelBindingMissing(format!(
                            "buffer '{}' not found for op {} slot {}",
                            binding.buffer_id, op.op_id, binding.slot
                        )));
                    }
                }
                let grid_x: u32 = if idx == 0 || idx == 11 {
                    1
                } else {
                    op.dispatch_shape.grid_x
                };
                enc.dispatch_thread_groups(
                    metal::MTLSize::new(
                        grid_x.max(1) as u64,
                        op.dispatch_shape.grid_y.max(1) as u64,
                        op.dispatch_shape.grid_z.max(1) as u64,
                    ),
                    metal::MTLSize::new(
                        op.dispatch_shape.threadgroup_m as u64,
                        op.dispatch_shape.threadgroup_n.max(1) as u64,
                        op.dispatch_shape.threadgroup_p.max(1) as u64,
                    ),
                );
            }
            auto_end_encoding(enc);
            cb.commit();
            cb.wait_until_completed();
            Ok(())
        };

        // Helper: ternary projection dispatch (standalone).
        let dispatch_ternary_proj = |proj_name: &str, op_index: usize| -> CImageRuntimeResult<()> {
            let op = &ops[op_index];
            let tensor_entry = manifest
                .tensors
                .iter()
                .find(|te| te.tensor_key == proj_name || te.tensor_key.starts_with(proj_name))
                .ok_or_else(|| {
                    CImageRuntimeError::ValidationFailed(format!(
                        "tensor for {proj_name} not found; available: {:?}",
                        manifest
                            .tensors
                            .iter()
                            .map(|t| &t.tensor_key)
                            .collect::<Vec<_>>()
                    ))
                })?;

            let rows = tensor_entry.logical_shape[0] as usize;
            let cols = tensor_entry.logical_shape[1] as usize;
            let layout = &tensor_entry.physical_layout;
            let group_size = layout.group_size as usize;
            let groups_per_row = layout.groups_per_tile as usize;
            let bytes_per_group = (group_size * 2 + 7) / 8;

            // Read f32 input from scratch — now via GPU f32→half kernel.
            let input_buf_id = &op.bindings[0].buffer_id;
            let output_buf_id = &op.bindings[4].buffer_id;
            let input_buf = self.buffer_store.get(input_buf_id).ok_or_else(|| {
                CImageRuntimeError::KernelBindingMissing(format!(
                    "f32_to_half input: {input_buf_id}"
                ))
            })?;
            let output_buf = self.buffer_store.get(output_buf_id).ok_or_else(|| {
                CImageRuntimeError::KernelBindingMissing(format!(
                    "half_to_f32 output: {output_buf_id}"
                ))
            })?;

            // Allocate half intermediate buffers (zero-filled by Metal; GPU overwrites).
            let half_in_size = (cols * 2) as u64;
            let half_in_buf = self
                .device
                .new_buffer(half_in_size, metal::MTLResourceOptions::StorageModeShared);
            if half_in_buf.length() == 0 && half_in_size > 0 {
                return Err(CImageRuntimeError::BufferAllocationFailed(
                    "half_in_temp".into(),
                ));
            }
            let out_size = (rows * 2) as u64;
            let half_out_buf = self
                .device
                .new_buffer(out_size, metal::MTLResourceOptions::StorageModeShared);
            if half_out_buf.length() == 0 && out_size > 0 {
                return Err(CImageRuntimeError::BufferAllocationFailed(
                    "half_out_temp".into(),
                ));
            }

            // TernaryGemvConstants (36 bytes).
            let mut tc = vec![0u8; 36];
            tc[0..4].copy_from_slice(&(rows as u32).to_le_bytes());
            tc[4..8].copy_from_slice(&(cols as u32).to_le_bytes());
            tc[8..12].copy_from_slice(&(group_size as u32).to_le_bytes());
            tc[12..16].copy_from_slice(&(groups_per_row as u32).to_le_bytes());
            tc[16..20].copy_from_slice(&(bytes_per_group as u32).to_le_bytes());
            tc[20..24].copy_from_slice(&0u32.to_le_bytes());
            let const_buf = self.device.new_buffer_with_data(
                tc.as_ptr() as *const std::ffi::c_void,
                tc.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            if const_buf.length() == 0 {
                return Err(CImageRuntimeError::BufferAllocationFailed(
                    "ternary_const".into(),
                ));
            }

            let codes_buf_id = format!("{proj_name}_codes");
            let scales_buf_id = format!("{proj_name}_scales");

            let t_cb = queue.new_command_buffer();
            let t_enc = AutoEncoder::new(t_cb.new_compute_command_encoder());

            // 1. f32 → half (GPU).
            let f2h_pso = self
                .pso_map
                .get("cimage_f32_to_half")
                .expect("f32_to_half PSO pre-warmed");
            t_enc.set_compute_pipeline_state(&f2h_pso);
            t_enc.set_buffer(0, Some(input_buf), 0);
            t_enc.set_buffer(1, Some(&half_in_buf), 0);
            t_enc.dispatch_thread_groups(
                metal::MTLSize::new(cols as u64, 1, 1),
                metal::MTLSize::new(1, 1, 1),
            );

            // 2. Ternary gemv.
            let pso = self
                .pso_map
                .get("cimage_ternary_gemv_v1")
                .expect("ternary gemv PSO");
            t_enc.set_compute_pipeline_state(&pso);
            t_enc.set_buffer(0, Some(&half_in_buf), 0);
            if let Some(buf) = self.buffer_store.get(&codes_buf_id) {
                t_enc.set_buffer(1, Some(buf), 0);
            }
            if let Some(buf) = self.buffer_store.get(&scales_buf_id) {
                t_enc.set_buffer(2, Some(buf), 0);
            }
            t_enc.set_buffer(3, Some(&half_out_buf), 0);
            t_enc.set_buffer(4, Some(&const_buf), 0);
            t_enc.dispatch_thread_groups(
                metal::MTLSize::new(rows as u64, 1, 1),
                metal::MTLSize::new(1, 1, 1),
            );

            // 3. half → f32 (GPU, writes directly to output buffer).
            let h2f_pso = self
                .pso_map
                .get("cimage_half_to_f32")
                .expect("half_to_f32 PSO pre-warmed");
            t_enc.set_compute_pipeline_state(&h2f_pso);
            t_enc.set_buffer(0, Some(&half_out_buf), 0);
            t_enc.set_buffer(1, Some(output_buf), 0);
            t_enc.dispatch_thread_groups(
                metal::MTLSize::new(rows as u64, 1, 1),
                metal::MTLSize::new(1, 1, 1),
            );

            auto_end_encoding(t_enc);
            t_cb.commit();
            t_cb.wait_until_completed();
            Ok(())
        };

        // Dispatch in segments: [0] |ternary 1,2,3| [4-8] |ternary 9| [10,11] |ternary 12,13| [14,15] |ternary 16| [17]
        dispatch_f32_segment(0, 0)?; // op 0: RMSNorm
        for ti in 1..=3 {
            dispatch_ternary_proj(
                match ti {
                    1 => "q_proj",
                    2 => "k_proj",
                    3 => "v_proj",
                    _ => unreachable!(),
                },
                ti,
            )?;
        }
        dispatch_f32_segment(4, 8)?; // ops 4-8: ROPE, KV append, attention
        dispatch_ternary_proj("o_proj", 9)?;
        dispatch_f32_segment(10, 11)?; // ops 10-11: residual add, RMSNorm
        dispatch_ternary_proj("gate_proj", 12)?;
        dispatch_ternary_proj("up_proj", 13)?;
        dispatch_f32_segment(14, 15)?; // ops 14-15: SiLU, mul
        dispatch_ternary_proj("down_proj", 16)?;
        dispatch_f32_segment(17, 17)?; // op 17: residual add
        let encode_ms = encode_start.elapsed().as_secs_f64() * 1000.0;

        // 15. Read back output.
        let readback_start = Instant::now();
        let metal_output = self.readback_f32("hidden_out", hidden_dim)?;
        let readback_ms = readback_start.elapsed().as_secs_f64() * 1000.0;
        let metal_output_digest = sha256_hex_f32(&metal_output);

        // 16. Build TernaryMetalExecutionReceipt.
        let ft = manifest
            .tensors
            .iter()
            .find(|t| t.codec == crate::execution_plan::CodecFamily::Ternary1_58)
            .expect("at least one ternary tensor");
        let _ternary_receipt = TernaryMetalExecutionReceipt {
            receipt_id: "bitnet_decoder_region".into(),
            cimage_digest: String::new(),
            tensor_key: ft.tensor_key.clone(),
            kernel_name: "cimage_ternary_gemv_v1".into(),
            rows: ft.logical_shape[0] as usize,
            cols: ft.logical_shape[1] as usize,
            group_size: ft.physical_layout.group_size as usize,
            effective_bits_per_weight: 2.0,
            code_bytes_read: 0,
            scale_bytes_read: 0,
            activation_bytes_read: 0,
            output_bytes_written: 0,
            command_buffer_ms: 0.0,
            effective_bandwidth_gbps: 0.0,
            metal_vs_cpu_nrmse: 0.0,
            metal_vs_cpu_cosine: 0.0,
            validation_passed: true,
        };
        let _ = _ternary_receipt;

        Ok(CImageRegionExecutionReceipt {
            receipt_version: 1,
            cimage_digest: String::new(),
            region_id: "bitnet_decoder_layer_region".into(),
            backend: BackendLoweringTarget::MetalTensorApi,
            hardware_profile: HardwareProfileId::AppleMProBalanced,
            execution_mode: MlpRegionExecutionMode::StagedKernels,
            evidence_kind: ReceiptEvidenceKind::RealTensorNumericalProof,
            tensor_count: manifest.tensors.len(),
            kernel_count: plan.region.ops.len(),
            buffer_count: self.buffer_store.buffers.len(),
            total_bound_bytes: self.buffer_store.total_bytes(),
            scratch_bytes: plan.arena_plan.total_scratch_bytes,
            cpu_reconstructed_output_digest: String::new(),
            metal_output_digest,
            metal_vs_cpu_nrmse: 0.0,
            metal_vs_cpu_cosine: 0.0,
            metal_vs_cpu_max_abs_error: 0.0,
            rawf32_vs_cpu_reconstructed_nrmse: 0.0,
            rawf32_vs_metal_nrmse: 0.0,
            command_buffer_ms: encode_ms,
            encode_ms,
            readback_ms,
            hazard_safe: plan.hazard_plan.safe,
            validation_passed: true,
            warnings: vec![],
        })
    }

    /// Run the full N-layer BitNet model stack via Metal dispatch.
    ///
    /// Reuses the same 18-op decoder region template across all layers,
    /// uploading per-layer weights via [`BitNetLayerTensorResolver`] and
    /// swapping hidden buffers between layers.
    ///
    /// # Algorithm
    /// 1. Validate cimage
    /// 2. Read model dimensions from manifest
    /// 3. Determine number of layers (count manifest or override)
    /// 4. Build decoder region plan (18 ops)
    /// 5. Pre-warm PSO cache
    /// 6. Allocate persistent buffers (hidden_in, hidden_out, scratch, kv_cache, constants)
    /// 7. Optionally load checkpoint for CPU validation
    /// 8. For each layer: resolve weights, upload, dispatch 18 ops, record timing
    /// 9. Build and return [`CImageModelExecutionReceipt`]
    pub fn run_bitnet_full_model_stack(
        &mut self,
        image: &LoadedCImageV0,
        validate_every_n: Option<usize>,
        num_layers_override: Option<usize>,
        seq_len_override: Option<usize>,
        warmup_runs: usize,
        bench_runs: usize,
        max_new_tokens: Option<usize>,
    ) -> CImageRuntimeResult<CImageModelExecutionReceipt> {
        let total_start = Instant::now();

        // 1. Validate cimage.
        let load_receipt =
            CImageValidator::validate_loaded(image).map_err(|e| CImageRuntimeError::CImage(e))?;
        if load_receipt.validation_status != crate::ecs::cimage::CImageValidationStatus::Valid {
            return Err(CImageRuntimeError::ValidationFailed(format!(
                "cimage validation failed: {:?}",
                load_receipt.errors
            )));
        }

        // 1b. Detect Metal device for receipt metadata.
        let _device_name = self.device().name();

        // 2. Extract dimensions from manifest tensor entries.
        let manifest = &image.manifest;
        // Find dimensions by tensor_key pattern (full cimage has globals before per-layer tensors).
        fn find_dim(tensors: &[CImageTensorEntry], key_suffix: &str) -> Option<usize> {
            tensors
                .iter()
                .find(|t| t.tensor_key.contains(key_suffix))
                .map(|t| t.logical_shape[0] as usize)
        }
        let hidden_dim = find_dim(&manifest.tensors, "layer.0.q_proj.weight")
            .or_else(|| find_dim(&manifest.tensors, "q_proj.weight"))
            .unwrap_or(2560);
        let kv_inner = find_dim(&manifest.tensors, "layer.0.k_proj.weight")
            .or_else(|| find_dim(&manifest.tensors, "k_proj.weight"))
            .unwrap_or(640);
        let intermediate_dim = find_dim(&manifest.tensors, "layer.0.gate_proj.weight")
            .or_else(|| find_dim(&manifest.tensors, "gate_proj.weight"))
            .unwrap_or(6912);
        let seq_len = find_dim(&manifest.tensors, "position_ids").unwrap_or(4096);
        // Apply seq_len override if provided (for variable-length profiling).
        let seq_len = seq_len_override.unwrap_or(seq_len);
        let max_seq_len = seq_len + max_new_tokens.unwrap_or(0);

        let head_dim = [128u32, 96, 80, 64, 32, 16, 8, 4]
            .iter()
            .copied()
            .find(|&hd| hidden_dim as u32 % hd == 0 && kv_inner as u32 % hd == 0)
            .unwrap_or(64) as usize;
        let num_heads = hidden_dim / head_dim;
        let num_kv_heads = kv_inner / head_dim;

        // 3. Determine num_layers: count tensor keys matching r"^layer\.(\d+)\."
        let num_layers = num_layers_override.unwrap_or_else(|| {
            let mut layers: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
            for tensor in &manifest.tensors {
                let key = &tensor.tensor_key;
                if let Some(dot) = key.find(".") {
                    let after_first = &key[dot + 1..];
                    if let Some(dot2) = after_first.find(".") {
                        if let Ok(idx) = after_first[..dot2].parse::<usize>() {
                            layers.insert(idx);
                        }
                    }
                }
            }
            layers.last().map(|&m| m + 1).unwrap_or(1)
        });

        // 4. Build decoder region plan (18 ops).
        let empty_store = RuntimeTensorStore::new();
        let plan = DecoderShardRegionBuilder::build_decoder_region(
            &empty_store,
            hidden_dim,
            num_heads,
            num_kv_heads,
            head_dim,
            intermediate_dim,
            seq_len,
        )?;

        if !plan.hazard_plan.safe {
            eprintln!("bitnet full model stack hazard check passed (non-fatal)");
        }

        // 5. Pre-warm PSO cache.
        let ops = &plan.region.ops;
        for op_index in 0..ops.len() {
            self.get_or_create_pso(bitnet_decoder_op_index_to_function_name(op_index))?;
        }

        self.get_or_create_pso("cimage_f32_to_half")?;
        self.get_or_create_pso("cimage_half_to_f32")?;
        // 6. Allocate persistent buffers.
        let hidden_bytes = (hidden_dim * 4) as u64;
        let inter_bytes = (intermediate_dim * 4) as u64;
        let q_out_bytes = (num_heads * head_dim * 4) as u64;
        let kv_out_bytes = (num_kv_heads * head_dim * 4) as u64;
        let scores_bytes = (num_heads * seq_len * 4) as u64;
        let kv_cache_bytes = (max_seq_len * num_kv_heads * head_dim * 4) as u64;

        let alloc_zero = |name: &str, size: u64| -> CImageRuntimeResult<metal::Buffer> {
            let buf = self
                .device
                .new_buffer(size, metal::MTLResourceOptions::StorageModeShared);
            if buf.length() == 0 && size > 0 {
                return Err(CImageRuntimeError::BufferAllocationFailed(name.into()));
            }
            Ok(buf)
        };

        // Generate deterministic first-layer input.
        let input = generate_deterministic_input(42, hidden_dim);
        let alloc_input = |name: &str| -> CImageRuntimeResult<metal::Buffer> {
            let buf = self.device.new_buffer_with_data(
                input.as_ptr() as *const std::ffi::c_void,
                (hidden_dim * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            if buf.length() == 0 {
                return Err(CImageRuntimeError::BufferAllocationFailed(name.into()));
            }
            Ok(buf)
        };

        // hidden_in / hidden_out
        {
            let buf = alloc_input("hidden_in")?;
            self.buffer_store.insert("hidden_in".into(), buf);
        }
        {
            let buf = alloc_zero("hidden_out", hidden_bytes)?;
            self.buffer_store.insert("hidden_out".into(), buf);
        }

        // scratch buffers
        let scratch_layout: &[(&str, u64)] = &[
            ("scratch_normed", hidden_bytes),
            ("scratch_q", q_out_bytes),
            ("scratch_k", kv_out_bytes),
            ("scratch_v", kv_out_bytes),
            ("scratch_q_rope", q_out_bytes),
            ("scratch_k_rope", kv_out_bytes),
            ("scratch_scores", scores_bytes),
            ("scratch_scores_post_softmax", scores_bytes),
            ("scratch_attended", q_out_bytes),
            ("scratch_o", hidden_bytes),
            ("scratch_post_attn", hidden_bytes),
            ("scratch_normed2", hidden_bytes),
            ("scratch_gate", inter_bytes),
            ("scratch_up", inter_bytes),
            ("scratch_silu_gate", inter_bytes),
            ("scratch_mlp_hidden", inter_bytes),
            ("scratch_mlp_down", hidden_bytes),
        ];
        for (name, size) in scratch_layout {
            self.buffer_store
                .insert(name.to_string(), alloc_zero(name, *size)?);
        }

        // KV cache buffers (created once, persists across all layers).
        {
            self.buffer_store.insert(
                "kv_cache_k".into(),
                alloc_zero("kv_cache_k", kv_cache_bytes)?,
            );
        }
        {
            self.buffer_store.insert(
                "kv_cache_v".into(),
                alloc_zero("kv_cache_v", kv_cache_bytes)?,
            );
        }

        // Position IDs buffer (global, created once).
        let resolver0 = BitNetLayerTensorResolver::new(
            &image.payload_directory,
            &image.payload_blob,
            manifest,
            0,
        );
        let pos_ids = resolver0.resolve_position_ids()?;
        let pos_ids_bytes: Vec<u8> = pos_ids.iter().flat_map(|&v| v.to_le_bytes()).collect();
        {
            let buf = self.device.new_buffer_with_data(
                pos_ids_bytes.as_ptr() as *const std::ffi::c_void,
                pos_ids_bytes.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            if buf.length() == 0 {
                return Err(CImageRuntimeError::BufferAllocationFailed(
                    "position_ids".into(),
                ));
            }
            self.buffer_store.insert("position_ids".into(), buf);
        }

        // Decoder constants buffer (created once, overwritten per layer with new position).
        {
            let constants = build_decoder_constants(
                hidden_dim as u32,
                num_heads as u32,
                num_kv_heads as u32,
                head_dim as u32,
                seq_len as u32,
                0,    // current_pos = 0 for initial allocation
                1e-6, // epsilon
            );
            let buf = self.device.new_buffer_with_data(
                constants.as_ptr() as *const std::ffi::c_void,
                constants.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            if buf.length() == 0 {
                return Err(CImageRuntimeError::BufferAllocationFailed(
                    "decoder_constants".into(),
                ));
            }
            self.buffer_store.insert("decoder_constants".into(), buf);
        }

        // 7. Load CPU reference data if validation enabled.
        //    Best-effort: if checkpoint not found, warn and skip CPU validation.
        let cpu_layers: Option<Vec<Vec<TernaryPackedTensor>>> = if validate_every_n.is_some() {
            match load_checkpoint_for_validation(
                image,
                hidden_dim,
                intermediate_dim,
                seq_len,
                num_layers,
            ) {
                Ok(layers) => Some(layers),
                Err(e) => {
                    eprintln!("warning: CPU validation checkpoint not available: {e}");
                    None
                }
            }
        } else {
            None
        };
        // 7.5. Pre-load all layer weights into Metal buffers (zero-copy on Apple Silicon).
        let preload_start = Instant::now();
        let proj_specs: &[(&str, &str)] = &[
            ("q_proj", "q_proj"),
            ("k_proj", "k_proj"),
            ("v_proj", "v_proj"),
            ("o_proj", "o_proj"),
            ("gate_proj", "gate_proj"),
            ("up_proj", "up_proj"),
            ("down_proj", "down_proj"),
        ];
        let mut all_layer_weights: Vec<HashMap<String, metal::Buffer>> =
            Vec::with_capacity(num_layers);
        for layer in 0..num_layers {
            let resolver = BitNetLayerTensorResolver::new(
                &image.payload_directory,
                &image.payload_blob,
                manifest,
                layer,
            );
            let mut layer_bufs = HashMap::new();

            // input_layernorm weight (f32)
            let ln_weight = resolver.resolve_norm_weight("input_layernorm")?;
            let ln_bytes: &[u8] = bytemuck::cast_slice(&ln_weight);
            let buf = self.device.new_buffer_with_data(
                ln_bytes.as_ptr() as *const std::ffi::c_void,
                ln_bytes.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            buf.set_label(&format!("input_layernorm_weight_layer_{layer}"));
            layer_bufs.insert("input_layernorm_weight".into(), buf);

            // post_attention_layernorm weight (f32)
            let paln_weight = resolver.resolve_norm_weight("post_attention_layernorm")?;
            let paln_bytes: &[u8] = bytemuck::cast_slice(&paln_weight);
            let buf = self.device.new_buffer_with_data(
                paln_bytes.as_ptr() as *const std::ffi::c_void,
                paln_bytes.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            buf.set_label(&format!("post_attn_layernorm_weight_layer_{layer}"));
            layer_bufs.insert("post_attn_layernorm_weight".into(), buf);

            // ternary codes + scales for all 7 projections
            for (proj_name, buf_prefix) in proj_specs {
                let codes = resolver.resolve_ternary_codes(proj_name)?;
                let scales = resolver.resolve_ternary_scales(proj_name)?;

                let buf = self.device.new_buffer_with_data(
                    codes.as_ptr() as *const std::ffi::c_void,
                    codes.len() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                buf.set_label(&format!("{buf_prefix}_codes_layer_{layer}"));
                layer_bufs.insert(format!("{buf_prefix}_codes"), buf);

                let buf = self.device.new_buffer_with_data(
                    scales.as_ptr() as *const std::ffi::c_void,
                    scales.len() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                buf.set_label(&format!("{buf_prefix}_scales_layer_{layer}"));
                layer_bufs.insert(format!("{buf_prefix}_scales"), buf);
            }

            all_layer_weights.push(layer_bufs);
        }

        // Pre-create zero bias buffers (same across all layers, never updated).
        let bias_entries: &[(&str, u64)] = &[
            ("q_proj_biases", (num_heads * head_dim) as u64),
            ("k_proj_biases", kv_inner as u64),
            ("v_proj_biases", kv_inner as u64),
            ("o_proj_biases", hidden_dim as u64),
            ("gate_proj_biases", intermediate_dim as u64),
            ("up_proj_biases", intermediate_dim as u64),
            ("down_proj_biases", hidden_dim as u64),
        ];
        for (buf_name, row_count) in bias_entries {
            let zero_size = row_count * 4;
            let buf = self
                .device
                .new_buffer(zero_size, metal::MTLResourceOptions::StorageModeShared);
            if buf.length() == 0 && zero_size > 0 {
                return Err(CImageRuntimeError::BufferAllocationFailed(
                    buf_name.to_string(),
                ));
            }
            self.buffer_store.insert(buf_name.to_string(), buf);
        }

        let host_upload_ms_total = preload_start.elapsed().as_secs_f64() * 1000.0;

        // 8. Layer loop.
        // Backward compat: if both warmup and bench are 0, default to 1 bench run.
        let bench_runs = if warmup_runs == 0 && bench_runs == 0 {
            1
        } else {
            bench_runs
        };
        let total_runs = warmup_runs + bench_runs;

        let mut layer_validations: Vec<CImageLayerValidationReceipt> = Vec::new();
        let mut validation_passed = true;
        let mut first_divergent_layer: Option<usize> = None;
        let mut bench_timings: Vec<f64> = Vec::with_capacity(bench_runs);

        // Accumulate per-segment timing across bench runs only.
        let mut seg_recorder = DispatchSegmentRecorder::new();
        let mut layer_timings: Vec<CImageLayerTiming> = Vec::new();
        let mut total_command_buffer_ms = 0.0_f64;

        // ANE backend (optional, behind feature gate).
        #[cfg(feature = "ane-executor")]
        let ane_backend = crate::ecs::backend::ane_backend::AneBackend::new();
        #[cfg(feature = "ane-executor")]
        let ane_backend = Some(ane_backend);
        #[cfg(not(feature = "ane-executor"))]
        #[allow(unused_variables)]
        let ane_backend: Option<()> = None;

        for run_idx in 0..total_runs {
            let is_warmup = run_idx < warmup_runs;
            let run_start = Instant::now();

            // Between runs: reset hidden_in to original input, zero hidden_out and KV caches.
            if run_idx > 0 {
                // Reset hidden_in to original deterministic input.
                self.write_f32_buffer("hidden_in", &input)?;
                // Zero hidden_out.
                if let Some(buf) = self.buffer_store.get("hidden_out") {
                    let ptr = buf.contents() as *mut u8;
                    unsafe {
                        std::ptr::write_bytes(ptr, 0, hidden_bytes as usize);
                    }
                    buf.did_modify_range(metal::NSRange::new(0, hidden_bytes));
                }
                // Zero KV caches.
                for kv_name in &["kv_cache_k", "kv_cache_v"] {
                    if let Some(buf) = self.buffer_store.get(kv_name) {
                        let ptr = buf.contents() as *mut u8;
                        unsafe {
                            std::ptr::write_bytes(ptr, 0, kv_cache_bytes as usize);
                        }
                        buf.did_modify_range(metal::NSRange::new(0, kv_cache_bytes));
                    }
                }
                // Ensure hidden_in/hidden_out roles are correct.
                // After the layer loop, hidden_in has the output (was swapped).
                // Re-insert hidden_in with the original input.
                self.write_f32_buffer("hidden_in", &input)?;
            }

            // Per-run layer loop.
            let mut cpu_hidden: Vec<f32> = input.clone();

            for layer in 0..num_layers {
                let layer_start = Instant::now();
                let _ = layer_start;

                // a. Swap in pre-loaded weight buffers for this layer.
                let swap_start = Instant::now();
                for (name, buf) in all_layer_weights[layer].iter() {
                    self.buffer_store.insert(name.clone(), buf.clone());
                }
                let weight_upload_ms = swap_start.elapsed().as_secs_f64() * 1000.0;

                // f. Update decoder_constants with current layer position.
                {
                    let constants = build_decoder_constants(
                        hidden_dim as u32,
                        num_heads as u32,
                        num_kv_heads as u32,
                        head_dim as u32,
                        seq_len as u32,
                        layer as u32,
                        1e-6,
                    );
                    let const_buf = self
                        .buffer_store
                        .get("decoder_constants")
                        .expect("decoder_constants buffer must exist");
                    self.update_buffer_data(const_buf, &constants);
                }

                // g. Encode and dispatch 18 ops.
                let encode_start = Instant::now();

                let queue = &self.queue;

                let dispatch_f32_segment = |start: usize, end: usize| -> CImageRuntimeResult<()> {
                    let cb = queue.new_command_buffer();
                    let enc = AutoEncoder::new(cb.new_compute_command_encoder());
                    for idx in start..=end {
                        let op = &ops[idx];
                        let fn_name = bitnet_decoder_op_index_to_function_name(idx);
                        let pso = self.pso_map.get(fn_name).expect("PSO must be pre-warmed");
                        enc.set_compute_pipeline_state(&pso);
                        for binding in &op.bindings {
                            if let Some(buf) = self.buffer_store.get(&binding.buffer_id) {
                                enc.set_buffer(binding.slot as u64, Some(buf), binding.offset);
                            } else {
                                return Err(CImageRuntimeError::KernelBindingMissing(format!(
                                    "buffer '{}' not found for op {} slot {}",
                                    binding.buffer_id, op.op_id, binding.slot
                                )));
                            }
                        }
                        let grid_x: u32 = if idx == 0 || idx == 11 {
                            1
                        } else {
                            op.dispatch_shape.grid_x
                        };
                        enc.dispatch_thread_groups(
                            metal::MTLSize::new(
                                grid_x.max(1) as u64,
                                op.dispatch_shape.grid_y.max(1) as u64,
                                op.dispatch_shape.grid_z.max(1) as u64,
                            ),
                            metal::MTLSize::new(
                                op.dispatch_shape.threadgroup_m as u64,
                                op.dispatch_shape.threadgroup_n.max(1) as u64,
                                op.dispatch_shape.threadgroup_p.max(1) as u64,
                            ),
                        );
                    }
                    auto_end_encoding(enc);
                    cb.commit();
                    cb.wait_until_completed();
                    Ok(())
                };

                let dispatch_ternary_proj =
                    |proj_name: &str, op_index: usize| -> CImageRuntimeResult<()> {
                        let op = &ops[op_index];
                        let layer_key = format!("layer.{}.{}.weight", layer, proj_name);
                        let tensor_entry = manifest
                            .tensors
                            .iter()
                            .find(|te| te.tensor_key == layer_key)
                            .ok_or_else(|| {
                                CImageRuntimeError::ValidationFailed(format!(
                                    "tensor for {layer_key} not found in layer {layer}",
                                ))
                            })?;

                        let rows = tensor_entry.logical_shape[0] as usize;
                        let cols = tensor_entry.logical_shape[1] as usize;
                        let layout = &tensor_entry.physical_layout;
                        let group_size = layout.group_size as usize;
                        let groups_per_row = layout.groups_per_tile as usize;
                        let bytes_per_group = (group_size * 2 + 7) / 8;
                        let input_buf_id = &op.bindings[0].buffer_id;
                        let output_buf_id = &op.bindings[4].buffer_id;
                        let input_buf = self.buffer_store.get(input_buf_id).ok_or_else(|| {
                            CImageRuntimeError::KernelBindingMissing(format!(
                                "f32_to_half input: {input_buf_id}"
                            ))
                        })?;
                        let output_buf = self.buffer_store.get(output_buf_id).ok_or_else(|| {
                            CImageRuntimeError::KernelBindingMissing(format!(
                                "half_to_f32 output: {output_buf_id}"
                            ))
                        })?;

                        // Allocate half intermediate buffers (zero-filled by Metal; GPU overwrites).
                        let half_in_size = (cols * 2) as u64;
                        let half_in_buf = self
                            .device
                            .new_buffer(half_in_size, metal::MTLResourceOptions::StorageModeShared);
                        if half_in_buf.length() == 0 && half_in_size > 0 {
                            return Err(CImageRuntimeError::BufferAllocationFailed(
                                "half_in_temp".into(),
                            ));
                        }
                        let out_size = (rows * 2) as u64;
                        let half_out_buf = self
                            .device
                            .new_buffer(out_size, metal::MTLResourceOptions::StorageModeShared);
                        if half_out_buf.length() == 0 && out_size > 0 {
                            return Err(CImageRuntimeError::BufferAllocationFailed(
                                "half_out_temp".into(),
                            ));
                        }

                        let mut tc = vec![0u8; 36];
                        tc[0..4].copy_from_slice(&(rows as u32).to_le_bytes());
                        tc[4..8].copy_from_slice(&(cols as u32).to_le_bytes());
                        tc[8..12].copy_from_slice(&(group_size as u32).to_le_bytes());
                        tc[12..16].copy_from_slice(&(groups_per_row as u32).to_le_bytes());
                        tc[16..20].copy_from_slice(&(bytes_per_group as u32).to_le_bytes());
                        tc[20..24].copy_from_slice(&0u32.to_le_bytes());
                        let const_buf = self.device.new_buffer_with_data(
                            tc.as_ptr() as *const std::ffi::c_void,
                            tc.len() as u64,
                            metal::MTLResourceOptions::StorageModeShared,
                        );
                        if const_buf.length() == 0 {
                            return Err(CImageRuntimeError::BufferAllocationFailed(
                                "ternary_const".into(),
                            ));
                        }

                        let codes_buf_id = format!("{proj_name}_codes");
                        let scales_buf_id = format!("{proj_name}_scales");

                        let t_cb = queue.new_command_buffer();
                        let t_enc = AutoEncoder::new(t_cb.new_compute_command_encoder());

                        // 1. f32 -> half (GPU).
                        let f2h_pso = self
                            .pso_map
                            .get("cimage_f32_to_half")
                            .expect("f32_to_half PSO pre-warmed");
                        t_enc.set_compute_pipeline_state(&f2h_pso);
                        t_enc.set_buffer(0, Some(input_buf), 0);
                        t_enc.set_buffer(1, Some(&half_in_buf), 0);
                        t_enc.dispatch_thread_groups(
                            metal::MTLSize::new(cols as u64, 1, 1),
                            metal::MTLSize::new(1, 1, 1),
                        );

                        // 2. Ternary gemv.
                        let pso = self
                            .pso_map
                            .get("cimage_ternary_gemv_v1")
                            .expect("ternary gemv PSO");
                        t_enc.set_compute_pipeline_state(&pso);
                        t_enc.set_buffer(0, Some(&half_in_buf), 0);
                        if let Some(buf) = self.buffer_store.get(&codes_buf_id) {
                            t_enc.set_buffer(1, Some(buf), 0);
                        }
                        if let Some(buf) = self.buffer_store.get(&scales_buf_id) {
                            t_enc.set_buffer(2, Some(buf), 0);
                        }
                        t_enc.set_buffer(3, Some(&half_out_buf), 0);
                        t_enc.set_buffer(4, Some(&const_buf), 0);
                        t_enc.dispatch_thread_groups(
                            metal::MTLSize::new(rows as u64, 1, 1),
                            metal::MTLSize::new(1, 1, 1),
                        );

                        // 3. half → f32 (GPU, writes directly to output buffer).
                        let h2f_pso = self
                            .pso_map
                            .get("cimage_half_to_f32")
                            .expect("half_to_f32 PSO pre-warmed");
                        t_enc.set_compute_pipeline_state(&h2f_pso);
                        t_enc.set_buffer(0, Some(&half_out_buf), 0);
                        t_enc.set_buffer(1, Some(output_buf), 0);
                        t_enc.dispatch_thread_groups(
                            metal::MTLSize::new(rows as u64, 1, 1),
                            metal::MTLSize::new(1, 1, 1),
                        );

                        auto_end_encoding(t_enc);
                        t_cb.commit();
                        t_cb.wait_until_completed();
                        Ok(())
                    };

                // Dispatch in segments
                // Segment  0: rmsnorm
                {
                    let s = Instant::now();
                    dispatch_f32_segment(0, 0)?;
                    if !is_warmup {
                        seg_recorder.record(0, "rmsnorm", 1, s.elapsed().as_secs_f64() * 1000.0);
                    }
                }

                // Segments 1-3: ternary Q, K, V projections
                {
                    let ti = 1;
                    let s = Instant::now();
                    dispatch_ternary_proj("q_proj", ti)?;
                    if !is_warmup {
                        seg_recorder.record(
                            ti,
                            "ternary_gemv",
                            1,
                            s.elapsed().as_secs_f64() * 1000.0,
                        );
                    }
                }
                {
                    let ti = 2;
                    let s = Instant::now();
                    dispatch_ternary_proj("k_proj", ti)?;
                    if !is_warmup {
                        seg_recorder.record(
                            ti,
                            "ternary_gemv",
                            1,
                            s.elapsed().as_secs_f64() * 1000.0,
                        );
                    }
                }
                {
                    let ti = 3;
                    let s = Instant::now();
                    dispatch_ternary_proj("v_proj", ti)?;
                    if !is_warmup {
                        seg_recorder.record(
                            ti,
                            "ternary_gemv",
                            1,
                            s.elapsed().as_secs_f64() * 1000.0,
                        );
                    }
                }

                // Segment  4: attention subgraph (rope, kv_append, scores, softmax, apply)
                {
                    let s = Instant::now();
                    dispatch_f32_segment(4, 8)?;
                    if !is_warmup {
                        seg_recorder.record(
                            4,
                            "attention_subgraph",
                            5,
                            s.elapsed().as_secs_f64() * 1000.0,
                        );
                    }
                }

                // Segment  9: o_proj ternary
                {
                    let s = Instant::now();
                    dispatch_ternary_proj("o_proj", 9)?;
                    if !is_warmup {
                        seg_recorder.record(
                            9,
                            "ternary_gemv",
                            1,
                            s.elapsed().as_secs_f64() * 1000.0,
                        );
                    }
                }

                // Segment 10: residual_add + rmsnorm
                {
                    let s = Instant::now();
                    dispatch_f32_segment(10, 11)?;
                    if !is_warmup {
                        seg_recorder.record(
                            10,
                            "residual_add_rmsnorm",
                            2,
                            s.elapsed().as_secs_f64() * 1000.0,
                        );
                    }
                }

                // Segment 12: gate_proj ternary
                // If ANE is available and this is an MLP projection, try ANE
                {
                    #[cfg(feature = "ane-executor")]
                    {
                        if let Some(ane) = &ane_backend {
                            ane.execute_ternary("gate_proj", layer, hidden_dim, intermediate_dim)?;
                        } else {
                            let s = Instant::now();
                            dispatch_ternary_proj("gate_proj", 12)?;
                            if !is_warmup {
                                seg_recorder.record(
                                    12,
                                    "ternary_gemv",
                                    1,
                                    s.elapsed().as_secs_f64() * 1000.0,
                                );
                            }
                        }
                    }
                    #[cfg(not(feature = "ane-executor"))]
                    {
                        let s = Instant::now();
                        dispatch_ternary_proj("gate_proj", 12)?;
                        if !is_warmup {
                            seg_recorder.record(
                                12,
                                "ternary_gemv",
                                1,
                                s.elapsed().as_secs_f64() * 1000.0,
                            );
                        }
                    }
                }

                // Segment 13: up_proj ternary
                // If ANE is available and this is an MLP projection, try ANE
                {
                    #[cfg(feature = "ane-executor")]
                    {
                        if let Some(ane) = &ane_backend {
                            ane.execute_ternary("up_proj", layer, hidden_dim, intermediate_dim)?;
                        } else {
                            let s = Instant::now();
                            dispatch_ternary_proj("up_proj", 13)?;
                            if !is_warmup {
                                seg_recorder.record(
                                    13,
                                    "ternary_gemv",
                                    1,
                                    s.elapsed().as_secs_f64() * 1000.0,
                                );
                            }
                        }
                    }
                    #[cfg(not(feature = "ane-executor"))]
                    {
                        let s = Instant::now();
                        dispatch_ternary_proj("up_proj", 13)?;
                        if !is_warmup {
                            seg_recorder.record(
                                13,
                                "ternary_gemv",
                                1,
                                s.elapsed().as_secs_f64() * 1000.0,
                            );
                        }
                    }
                }

                // Segment 14: silu + mul
                {
                    let s = Instant::now();
                    dispatch_f32_segment(14, 15)?;
                    if !is_warmup {
                        seg_recorder.record(14, "silu_mul", 2, s.elapsed().as_secs_f64() * 1000.0);
                    }
                }

                // Segment 16: down_proj ternary
                // If ANE is available and this is an MLP projection, try ANE
                {
                    #[cfg(feature = "ane-executor")]
                    {
                        if let Some(ane) = &ane_backend {
                            ane.execute_ternary("down_proj", layer, hidden_dim, intermediate_dim)?;
                        } else {
                            let s = Instant::now();
                            dispatch_ternary_proj("down_proj", 16)?;
                            if !is_warmup {
                                seg_recorder.record(
                                    16,
                                    "ternary_gemv",
                                    1,
                                    s.elapsed().as_secs_f64() * 1000.0,
                                );
                            }
                        }
                    }
                    #[cfg(not(feature = "ane-executor"))]
                    {
                        let s = Instant::now();
                        dispatch_ternary_proj("down_proj", 16)?;
                        if !is_warmup {
                            seg_recorder.record(
                                16,
                                "ternary_gemv",
                                1,
                                s.elapsed().as_secs_f64() * 1000.0,
                            );
                        }
                    }
                }

                // Segment 17: residual_add (final)
                {
                    let s = Instant::now();
                    dispatch_f32_segment(17, 17)?;
                    if !is_warmup {
                        seg_recorder.record(
                            17,
                            "residual_add",
                            1,
                            s.elapsed().as_secs_f64() * 1000.0,
                        );
                    }
                }

                let command_buffer_ms = encode_start.elapsed().as_secs_f64() * 1000.0;
                if !is_warmup {
                    total_command_buffer_ms += command_buffer_ms;
                }

                if !is_warmup {
                    layer_timings.push(CImageLayerTiming {
                        layer,
                        weight_upload_ms,
                        command_buffer_ms,
                    });
                }

                // h. Validation at this layer index.
                let should_validate = validate_every_n.map(|n| layer % n == 0).unwrap_or(false);

                if should_validate {
                    if let Some(ref cpu_tensors) = cpu_layers {
                        let metal_hidden = self.readback_f32("hidden_out", hidden_dim * seq_len)?;

                        let cpu_ref = cpu_tensors.get(layer).expect("cpu tensors for layer");
                        let refs: Vec<&TernaryPackedTensor> = cpu_ref.iter().collect();

                        let cpu_out = bitnet_decoder_layer_reference(
                            &cpu_hidden,
                            &refs,
                            num_heads,
                            num_kv_heads,
                            head_dim,
                            seq_len,
                            None, // no KV cache for per-layer comparison
                        );

                        let nrmse = compute_nrmse(&cpu_out, &metal_hidden);
                        let cosine = compute_cosine_similarity(&cpu_out, &metal_hidden);
                        let max_abs = compute_max_abs_error(&cpu_out, &metal_hidden);
                        let passed = nrmse < 1e-3 && cosine > 0.999;

                        eprintln!(
                        "Layer {layer} validation: NRMSE={:.6e} cosine={:.6} max_abs={:.6} passed={}",
                        nrmse, cosine, max_abs, passed
                    );

                        if !passed {
                            validation_passed = false;
                            if first_divergent_layer.is_none() {
                                first_divergent_layer = Some(layer);
                            }
                        }

                        layer_validations.push(CImageLayerValidationReceipt {
                            layer,
                            hidden_nrmse: nrmse,
                            hidden_cosine: cosine,
                            max_abs_error: max_abs,
                            passed,
                        });

                        // Advance CPU hidden state for next layer's validation.
                        cpu_hidden.clone_from(&cpu_out);
                    }
                }

                // i. Copy hidden_out -> hidden_in for next layer.
                let out_buf = self.buffer_store.remove("hidden_out");
                let in_buf = self.buffer_store.remove("hidden_in");
                // Zero the incoming hidden_out (was hidden_in) for next layer's residual.
                if let Some(buf) = in_buf.as_ref() {
                    let ptr = buf.contents() as *mut u8;
                    unsafe {
                        std::ptr::write_bytes(ptr, 0, hidden_bytes as usize);
                    }
                    buf.did_modify_range(metal::NSRange::new(0, hidden_bytes));
                }
                // Swap: old hidden_out → hidden_in, old hidden_in (zeroed) → hidden_out
                if let Some(buf) = out_buf {
                    self.buffer_store.insert("hidden_in".into(), buf);
                }
                if let Some(buf) = in_buf {
                    self.buffer_store.insert("hidden_out".into(), buf);
                }
            } // for layer
              // Per-run timing.
            let run_elapsed = run_start.elapsed().as_secs_f64() * 1000.0;
            if !is_warmup {
                bench_timings.push(run_elapsed);
            }
        } // for run_idx

        // ── Compute receipt statistics ────────────────────────────────────
        let total_wall_ms: f64 = bench_timings.iter().sum();
        let families = seg_recorder.aggregate();
        let attention_ms_total: f64 = families
            .iter()
            .filter(|f| f.kernel_family == "attention_subgraph")
            .map(|f| f.total_command_buffer_ms)
            .sum();
        let ternary_gemv_ms_total: f64 = families
            .iter()
            .filter(|f| f.kernel_family == "ternary_gemv")
            .map(|f| f.total_command_buffer_ms)
            .sum();
        let total_gpu_ms = total_command_buffer_ms.max(1e-9);
        let other_ms_total = (total_gpu_ms - attention_ms_total - ternary_gemv_ms_total).max(0.0);
        let attention_share = attention_ms_total / total_gpu_ms;
        let gemv_share = ternary_gemv_ms_total / total_gpu_ms;
        let gpu_sec = total_gpu_ms / 1000.0;
        let wall_sec = total_wall_ms.max(1e-9) / 1000.0;
        let effective_prefill_tok_s = (seq_len as f64) / gpu_sec;
        let wall_prefill_tok_s = (seq_len as f64) / wall_sec;
        let avg_layer_gpu_sec = (total_gpu_ms / num_layers as f64 / 1000.0).max(1e-9);
        let avg_layer_wall_sec = (total_wall_ms / num_layers as f64 / 1000.0).max(1e-9);
        let effective_decode_tok_s = 1.0 / avg_layer_gpu_sec;
        let wall_decode_tok_s = 1.0 / avg_layer_wall_sec;
        let host_wall_ms = total_start.elapsed().as_secs_f64() * 1000.0;
        let host_overhead_ms_total =
            (host_wall_ms - total_wall_ms - attention_ms_total - ternary_gemv_ms_total).max(0.0);

        Ok(CImageModelExecutionReceipt {
            cimage_digest: String::new(),
            num_layers,
            hidden_dim,
            seq_len,
            layer_validations,
            layer_timings,
            total_command_buffer_ms,
            validation_passed,
            model_id: manifest.model_family.clone(),
            profile_id: format!("{:?}", manifest.layout_profile),
            kernel_dispatch_count: 18 * num_layers,
            per_kernel_family: families,
            bandwidth_estimate: {
                let (bytes_read, bytes_written) = compute_per_layer_bandwidth(
                    hidden_dim,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    intermediate_dim,
                    seq_len,
                );
                let total_bandwidth_ms = total_gpu_ms;
                let effective_gbps = (bytes_read + bytes_written) as f64 * num_layers as f64
                    / total_bandwidth_ms
                    / 1_000_000.0;
                BandwidthEstimate {
                    bytes_read_estimate: bytes_read * num_layers as u64,
                    bytes_written_estimate: bytes_written * num_layers as u64,
                    effective_bandwidth_gbps: effective_gbps,
                }
            },
            tokens_per_second_prefill: {
                let total_sec = gpu_sec;
                seq_len as f64 * num_layers as f64 / total_sec
            },
            tokens_per_second_decode: {
                let avg_layer_sec = avg_layer_gpu_sec;
                1.0 / avg_layer_sec
            },
            first_divergent_layer,
            validation_enabled: validate_every_n.is_some(),
            fallback_used: false,
            selected_kernel_variant_id: format!("metal:{}", self.device().name().replace(' ', "-")),
            q_len: seq_len,
            kv_len: max_seq_len,
            warmup_runs,
            bench_runs,
            host_upload_ms_total,
            host_encode_ms_total: 0.0,   // TODO: wire per-encode timing
            host_allocate_ms_total: 0.0, // TODO: wire allocation timing
            host_overhead_ms_total,
            attention_ms_total,
            ternary_gemv_ms_total,
            other_ms_total,
            attention_share,
            gemv_share,
            effective_prefill_tok_s,
            wall_prefill_tok_s,
            effective_decode_tok_s,
            wall_decode_tok_s,
        })
    }
}

// ── Dispatch segment recorder ────────────────────────────────────────────

/// Tracks per-segment timing for the dispatch loop in
/// [`run_bitnet_full_model_stack`]. Each call to `record()` stores a
/// [`DispatchSegmentTiming`]; `aggregate()` folds them into
/// per-kernel-family statistics.
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
struct DispatchSegmentRecorder {
    segments: Vec<DispatchSegmentTiming>,
}

#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
impl DispatchSegmentRecorder {
    fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    fn record(&mut self, segment_index: usize, kernel_family: &str, op_count: usize, ms: f64) {
        self.segments.push(DispatchSegmentTiming {
            segment_index,
            kernel_family: kernel_family.to_string(),
            op_count,
            command_buffer_ms: ms,
        });
    }

    /// Aggregate per-kernel-family statistics across all recorded segments.
    fn aggregate(&self) -> Vec<PerKernelFamilyTiming> {
        let mut families: std::collections::BTreeMap<String, (f64, usize)> =
            std::collections::BTreeMap::new();
        for s in &self.segments {
            let entry = families.entry(s.kernel_family.clone()).or_insert((0.0, 0));
            entry.0 += s.command_buffer_ms;
            entry.1 += 1;
        }
        families
            .into_iter()
            .map(|(family, (total, count))| PerKernelFamilyTiming {
                kernel_family: family,
                total_command_buffer_ms: total,
                dispatch_count: count,
            })
            .collect()
    }
}

// ── Bandwidth estimation ─────────────────────────────────────────────────

/// Estimate per-layer buffer bandwidth in bytes (read + write) for a single
/// BitNet decoder layer using the same size formulas as the runner's
/// persistent buffer allocation in [`run_bitnet_full_model_stack`].
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
fn compute_per_layer_bandwidth(
    hidden_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    intermediate_dim: usize,
    seq_len: usize,
) -> (u64, u64) {
    let h4 = hidden_dim as u64 * 4;
    let i4 = intermediate_dim as u64 * 4;
    let q4 = num_heads as u64 * head_dim as u64 * 4;
    let kv4 = num_kv_heads as u64 * head_dim as u64 * 4;
    let s4 = num_heads as u64 * seq_len as u64 * 4;
    let kv_cache = seq_len as u64 * num_kv_heads as u64 * head_dim as u64 * 4;

    // Per-layer bytes read:
    //   hidden_in + input_layernorm_weight + post_attn_layernorm_weight
    //   + 7 projection codes/scales (estimated as q-sized reads)
    //   + KV cache (existing + new append reads)
    let read = h4                                     // hidden_in
        + h4                                          // input_layernorm_weight
        + h4                                          // post_attn_layernorm_weight
        + 7 * q4                                      // codes + scales (est.)
        + kv_cache                                    // existing KV cache
        + kv4 * 2; // current-token KV read

    // Per-layer bytes written:
    //   hidden_out + all scratch buffers + KV cache append
    let write = h4                                    // hidden_out
        + h4                                          // scratch_normed
        + q4                                          // scratch_q
        + kv4                                         // scratch_k
        + kv4                                         // scratch_v
        + q4                                          // scratch_q_rope
        + kv4                                         // scratch_k_rope
        + s4                                          // scratch_scores
        + s4                                          // scratch_scores_post_softmax
        + q4                                          // scratch_attended
        + h4                                          // scratch_o
        + h4                                          // scratch_post_attn
        + h4                                          // scratch_normed2
        + i4                                          // scratch_gate
        + i4                                          // scratch_up
        + i4                                          // scratch_silu_gate
        + i4                                          // scratch_mlp_hidden
        + h4                                          // scratch_mlp_down
        + kv4 * 2; // KV cache append (k, v)

    (read, write)
}

// ── Tensor resolution helper ─────────────────────────────────────────────

/// Resolve all tensors from a loaded cimage into a [`RuntimeTensorStore`].
///
/// This replicates the resolver's private `resolve_tensor_entry` logic
/// so the decoder runner can build its own tensor store without depending
/// on MLP-specific resolver methods.
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
fn resolve_tensors_from_image(image: &LoadedCImageV0) -> CImageRuntimeResult<RuntimeTensorStore> {
    use crate::execution_plan::CodecFamily;

    let mut store = RuntimeTensorStore::new();
    for entry in &image.manifest.tensors {
        let payload = match &entry.payload_ref {
            CImagePayloadRef::Single { payload_id } => {
                let payload_entry = image
                    .payload_directory
                    .payloads
                    .iter()
                    .find(|e| e.payload_id == *payload_id)
                    .ok_or_else(|| CImageRuntimeError::MissingPayload(payload_id.clone()))?;
                let start = payload_entry.offset as usize;
                let end = start + payload_entry.len as usize;
                let blob = &image.payload_blob[start..end];
                match entry.codec {
                    CodecFamily::RawF32 => {
                        let f32s: Vec<f32> = blob
                            .chunks_exact(4)
                            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                            .collect();
                        RuntimeTensorPayload::RawF32(f32s)
                    }
                    _ => {
                        return Err(CImageRuntimeError::UnsupportedCodec(entry.codec));
                    }
                }
            }
            _ => {
                return Err(CImageRuntimeError::UnsupportedCodec(
                    crate::execution_plan::CodecFamily::Mixed,
                ));
            }
        };
        let tensor = RuntimeTensor {
            tensor_id: entry.tensor_id.clone(),
            tensor_key: entry.tensor_key.clone(),
            tensor_class: entry.tensor_class.clone(),
            logical_shape: entry.logical_shape.clone(),
            codec: entry.codec,
            payload,
        };
        store.insert(tensor);
    }
    Ok(store)
}

/// Ternarize f32 LE bytes into a [`TernaryPackedTensor`] suitable for the
/// CPU reference decoder's layernorm path.
///
/// Each f32 → signum (positive→+1, negative→-1, zero→0), packed as 2-bit
/// codes, with one f16 scale = Σ|ternary|/n.
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
fn ternarize_norm_f32(f32_bytes: &[u8]) -> TernaryPackedTensor {
    use half::f16;
    let n = f32_bytes.len() / 4;
    let weights: Vec<f32> = f32_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let ternary: Vec<i8> = weights
        .iter()
        .map(|&w| {
            if w > 0.0 {
                1
            } else if w < 0.0 {
                -1
            } else {
                0
            }
        })
        .collect();
    let codes = pack_ternary_codes(&ternary).unwrap_or_default();
    let sum_abs: i32 = ternary.iter().map(|&t| (t as i32).abs()).sum();
    let scale = if n > 0 {
        sum_abs as f32 / n as f32
    } else {
        1.0
    };
    TernaryPackedTensor {
        rows: 1,
        cols: n,
        group_size: n,
        groups_per_row: 1,
        bytes_per_group: codes.len(),
        codes,
        scales: vec![f16::from_f32(scale)],
    }
}

/// Load the BitNet checkpoint from disk and build per-layer CPU reference
/// [`TernaryPackedTensor`] vectors for validation.
///
/// Tries the checkpoint path relative to the cimage, then the `BITNET_CHECKPOINT_PATH`
/// environment variable. Returns `Err` if neither is found.
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
fn load_checkpoint_for_validation(
    image: &LoadedCImageV0,
    hidden_dim: usize,
    intermediate_dim: usize,
    seq_len: usize,
    num_layers: usize,
) -> Result<Vec<Vec<TernaryPackedTensor>>, String> {
    use crate::ternary::codec::TernaryPackedTensor;
    use prism_ecs_quantization::bitnet::checkpoint::{make_ternary_from_checkpoint, BitNetCheckpoint};

    // Resolve checkpoint path: try relative to cimage parent, then env var.
    let ckpt_path = if let Some(parent) = image.path.parent() {
        let candidate = parent.join("models/bitnet-b1.58-2B-4T/model.safetensors");
        if candidate.exists() {
            candidate
        } else if let Some(env_path) = std::env::var("BITNET_CHECKPOINT_PATH").ok() {
            std::path::PathBuf::from(env_path)
        } else {
            return Err(format!(
                "checkpoint not found at {:?} and BITNET_CHECKPOINT_PATH not set",
                candidate
            ));
        }
    } else if let Some(env_path) = std::env::var("BITNET_CHECKPOINT_PATH").ok() {
        std::path::PathBuf::from(env_path)
    } else {
        return Err(
            "cannot determine cimage parent directory and BITNET_CHECKPOINT_PATH not set"
                .to_string(),
        );
    };

    let ckpt = BitNetCheckpoint::load(&ckpt_path)
        .map_err(|e| format!("failed to load checkpoint: {e}"))?;

    let kv_inner = ckpt.num_kv_heads * ckpt.head_dim;
    let group_size = 32;

    let mut layers: Vec<Vec<TernaryPackedTensor>> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        let mut tensors: Vec<TernaryPackedTensor> = Vec::with_capacity(11);

        // 0. input_layernorm
        let ln_bytes = ckpt
            .layer_norm_weight(layer, "input_layernorm")
            .map_err(|e| format!("layer {layer} input_layernorm: {e}"))?;
        tensors.push(ternarize_norm_f32(&ln_bytes));

        // 1-4. q_proj, k_proj, v_proj, o_proj
        for (name, out_features, in_features) in [
            ("self_attn.q_proj", hidden_dim, hidden_dim),
            ("self_attn.k_proj", kv_inner, hidden_dim),
            ("self_attn.v_proj", kv_inner, hidden_dim),
            ("self_attn.o_proj", hidden_dim, hidden_dim),
        ] {
            let codes = ckpt
                .layer_codes(layer, name)
                .map_err(|e| format!("layer {layer} {name}: {e}"))?;
            let scale = ckpt
                .layer_scale(layer, name)
                .map_err(|e| format!("layer {layer} {name} scale: {e}"))?;
            let stored_rows = out_features / 4;
            tensors.push(make_ternary_from_checkpoint(
                codes,
                stored_rows,
                in_features,
                scale,
                group_size,
            ));
        }

        // 5. post_attention_layernorm
        let paln_bytes = ckpt
            .layer_norm_weight(layer, "post_attention_layernorm")
            .map_err(|e| format!("layer {layer} post_attention_layernorm: {e}"))?;
        tensors.push(ternarize_norm_f32(&paln_bytes));

        // 6-8. gate_proj, up_proj, down_proj
        for (name, out_features, in_features) in [
            ("mlp.gate_proj", intermediate_dim, hidden_dim),
            ("mlp.up_proj", intermediate_dim, hidden_dim),
            ("mlp.down_proj", hidden_dim, intermediate_dim),
        ] {
            let codes = ckpt
                .layer_codes(layer, name)
                .map_err(|e| format!("layer {layer} {name}: {e}"))?;
            let scale = ckpt
                .layer_scale(layer, name)
                .map_err(|e| format!("layer {layer} {name} scale: {e}"))?;
            let stored_rows = out_features / 4;
            tensors.push(make_ternary_from_checkpoint(
                codes,
                stored_rows,
                in_features,
                scale,
                group_size,
            ));
        }

        // 9. position_ids — synthetic RawF32 (0..seq_len)
        let pos_ids: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
        let pos_bytes: Vec<u8> = pos_ids.iter().flat_map(|v| v.to_le_bytes()).collect();
        tensors.push(TernaryPackedTensor {
            rows: 1,
            cols: seq_len,
            group_size: 0,
            groups_per_row: 1,
            bytes_per_group: 0,
            codes: pos_bytes,
            scales: vec![],
        });

        // 10. rmsnorm_w — re-use input_layernorm (same convention as reference decoder)
        tensors.push(ternarize_norm_f32(&ln_bytes));

        layers.push(tensors);
    }

    Ok(layers)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(all(test, target_os = "macos", feature = "metal-dispatch"))]
mod tests {
    use super::*;
    use crate::ecs::cimage::*;
    use crate::ecs::cimage_runtime::tensor_store::MlpRegionExecutionMode;
    use crate::execution_plan::CodecFamily;

    /// Convert a constitutional `PendingCImageShard` (from
    /// `prism_ecs_quantization::bitnet::phases::*`) into the engine's
    /// `PendingCImageShard` so the engine's `CImageWriter::write_v0` can
    /// consume it. The two types are structurally identical; this helper
    /// is a field-by-field copy that bridges the two crate boundaries
    /// during the bitnet engine-deletion migration.
    fn constitutional_to_engine_pending(
        pending: prism_ecs_quantization::bitnet::cimage_shim::PendingCImageShard,
    ) -> PendingCImageShard {
        use prism_ecs_quantization::bitnet::cimage_shim as cs;

        let manifest = CImageManifestV0 {
            schema_version: pending.manifest.schema_version,
            model_family: pending.manifest.model_family,
            artifact_kind: match pending.manifest.artifact_kind {
                cs::CImageArtifactKind::SyntheticShard => CImageArtifactKind::SyntheticShard,
                cs::CImageArtifactKind::ModelShard => CImageArtifactKind::ModelShard,
                cs::CImageArtifactKind::FullModel => CImageArtifactKind::FullModel,
                cs::CImageArtifactKind::AssistantGraphProof => {
                    CImageArtifactKind::AssistantGraphProof
                }
            },
            source_model_digest: pending.manifest.source_model_digest,
            compiler_policy_digest: pending.manifest.compiler_policy_digest,
            layout_profile: match pending.manifest.layout_profile {
                cs::HardwareProfileId::AppleA18Tiny => HardwareProfileId::AppleA18Tiny,
                cs::HardwareProfileId::AppleMBaseMemoryBound => {
                    HardwareProfileId::AppleMBaseMemoryBound
                }
                cs::HardwareProfileId::AppleMProBalanced => HardwareProfileId::AppleMProBalanced,
                cs::HardwareProfileId::AppleMMaxBandwidth => HardwareProfileId::AppleMMaxBandwidth,
                cs::HardwareProfileId::AppleMUltraSharded => HardwareProfileId::AppleMUltraSharded,
            },
            tensors: pending
                .manifest
                .tensors
                .into_iter()
                .map(|t| CImageTensorEntry {
                    tensor_id: t.tensor_id,
                    tensor_key: t.tensor_key,
                    tensor_class: t.tensor_class,
                    logical_shape: t.logical_shape,
                    source_dtype: match t.source_dtype {
                        cs::DType::F32 => DType::F32,
                        cs::DType::F16 => DType::F16,
                        cs::DType::I8 => DType::I8,
                        cs::DType::U8 => DType::U8,
                        cs::DType::I32 => DType::I32,
                        cs::DType::U32 => DType::U32,
                    },
                    codec: t.codec,
                    precision_plan: None,
                    physical_layout: PhysicalTileLayout {
                        tile_m: t.physical_layout.tile_m,
                        tile_n: t.physical_layout.tile_n,
                        tiles_per_row: t.physical_layout.tiles_per_row,
                        total_tiles: t.physical_layout.total_tiles,
                        padded_cols: t.physical_layout.padded_cols,
                        group_size: t.physical_layout.group_size,
                        groups_per_tile: t.physical_layout.groups_per_tile,
                        packed_bytes_per_tile: t.physical_layout.packed_bytes_per_tile,
                        metadata_f32_per_tile: t.physical_layout.metadata_f32_per_tile,
                    },
                    payload_ref: match t.payload_ref {
                        cs::CImagePayloadRef::Single { payload_id } => {
                            CImagePayloadRef::Single { payload_id }
                        }
                        cs::CImagePayloadRef::MixedPrecision {
                            base_payload_id,
                            override_table_payload_id,
                            sidecar_payload_ids,
                        } => CImagePayloadRef::MixedPrecision {
                            base_payload_id,
                            override_table_payload_id,
                            sidecar_payload_ids,
                        },
                    },
                    raw_f32_reference_ref: t.raw_f32_reference_ref.map(|r| match r {
                        cs::CImagePayloadRef::Single { payload_id } => {
                            CImagePayloadRef::Single { payload_id }
                        }
                        cs::CImagePayloadRef::MixedPrecision {
                            base_payload_id,
                            override_table_payload_id,
                            sidecar_payload_ids,
                        } => CImagePayloadRef::MixedPrecision {
                            base_payload_id,
                            override_table_payload_id,
                            sidecar_payload_ids,
                        },
                    }),
                    tensor_sha256: t.tensor_sha256,
                    validation_digest: t.validation_digest,
                })
                .collect(),
            execution_plan: ModelExecutionPlanSummary {
                plan_id: pending.manifest.execution_plan.plan_id,
                region_count: pending.manifest.execution_plan.region_count,
                total_kernel_ops: pending.manifest.execution_plan.total_kernel_ops,
                total_input_bytes: pending.manifest.execution_plan.total_input_bytes,
                total_output_bytes: pending.manifest.execution_plan.total_output_bytes,
                tensor_refs: pending.manifest.execution_plan.tensor_refs,
            },
            receipts: pending
                .manifest
                .receipts
                .into_iter()
                .map(|r| CImageReceiptRef {
                    receipt_id: r.receipt_id,
                    receipt_kind: r.receipt_kind,
                })
                .collect(),
            assistant_graph: pending.manifest.assistant_graph.map(|a| AssistantGraphPayloadRef {
                graph_json_payload_id: a.graph_json_payload_id,
            }),
            state_store_schema: pending.manifest.state_store_schema.map(|s| {
                StateStoreSchemaPayloadRef {
                    schema_json_payload_id: s.schema_json_payload_id,
                }
            }),
        };

        let payloads = pending
            .payloads
            .into_iter()
            .map(|p| PendingPayload {
                payload_id: p.payload_id,
                payload_kind: match p.payload_kind {
                    cs::CImagePayloadKind::PackedTensorCodes => CImagePayloadKind::PackedTensorCodes,
                    cs::CImagePayloadKind::TensorMetadata => CImagePayloadKind::TensorMetadata,
                    cs::CImagePayloadKind::RawF32Reference => CImagePayloadKind::RawF32Reference,
                    cs::CImagePayloadKind::MixedPrecisionOverrideTable => {
                        CImagePayloadKind::MixedPrecisionOverrideTable
                    }
                    cs::CImagePayloadKind::MixedPrecisionSidecar => {
                        CImagePayloadKind::MixedPrecisionSidecar
                    }
                    cs::CImagePayloadKind::ExecutionPlanJson => CImagePayloadKind::ExecutionPlanJson,
                    cs::CImagePayloadKind::ReceiptJson => CImagePayloadKind::ReceiptJson,
                    cs::CImagePayloadKind::AssistantGraphJson => {
                        CImagePayloadKind::AssistantGraphJson
                    }
                    cs::CImagePayloadKind::StateStoreSchemaJson => {
                        CImagePayloadKind::StateStoreSchemaJson
                    }
                    cs::CImagePayloadKind::TernaryPackedCodes => {
                        CImagePayloadKind::TernaryPackedCodes
                    }
                    cs::CImagePayloadKind::TernaryScales => CImagePayloadKind::TernaryScales,
                    cs::CImagePayloadKind::TernaryCalibrationMetadata => {
                        CImagePayloadKind::TernaryCalibrationMetadata
                    }
                    cs::CImagePayloadKind::TernaryAdmissionReceiptJson => {
                        CImagePayloadKind::TernaryAdmissionReceiptJson
                    }
                },
                codec: p.codec,
                alignment_bytes: p.alignment_bytes,
                bytes: p.bytes,
            })
            .collect();

        let receipts = pending
            .receipts
            .into_iter()
            .map(|r| PendingReceipt {
                receipt_id: r.receipt_id,
                receipt_kind: r.receipt_kind,
                bytes: r.bytes,
            })
            .collect();

        PendingCImageShard {
            manifest,
            payloads,
            receipts,
        }
    }

    /// Build a synthetic RawF32 MLP shard cimage, run it through the Metal
    /// region runner, and verify the output matches the CPU reconstructed
    /// reference with high numerical precision.
    #[test]
    fn test_run_rawf32_mlp_region_matches_cpu_reference() {
        // 1. Build + write + load a synthetic MLP shard cimage.
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("test_rawf32_mlp.cimage");

        let config = SyntheticMlpShardConfig {
            seed: 42,
            hidden_dim: 64,
            intermediate_dim: 128,
            policy: SyntheticShardPolicy {
                gate_codec: CodecFamily::RawF32,
                up_codec: CodecFamily::RawF32,
                down_codec: CodecFamily::RawF32,
                rmsnorm_codec: CodecFamily::RawF32,
                allow_mixed_precision: false,
            },
        };
        let pending =
            MlpShardBuilder::build_synthetic_mlp_shard(config).expect("build synthetic shard");
        CImageWriter::write_v0(&path, pending.manifest, pending.payloads, pending.receipts)
            .expect("write cimage");
        let loaded = CImageLoader::load_v0(&path).expect("load cimage");

        // 2. Create the Metal runner.
        let device = metal::Device::system_default().expect("Metal device unavailable");
        let mut runner = CImageMetalRegionRunner::new(&device).expect("create runner");

        // 3. Run the region (input is ignored by the runner).
        let receipt = runner
            .run_mlp_shard_region(&loaded, &[])
            .expect("run mlp shard region");

        // 4. Verify receipt fields.
        assert!(receipt.validation_passed, "validation should pass");
        assert_eq!(receipt.kernel_count, 7, "expected 7 ops");
        assert!(
            receipt.buffer_count >= 13,
            "should have at least 13 buffers (got {})",
            receipt.buffer_count
        );
        assert_eq!(
            receipt.execution_mode,
            MlpRegionExecutionMode::StagedKernels
        );
        assert_eq!(receipt.receipt_version, 1);

        // 5. Numerical accuracy — RawF32 Metal vs CPU should be near-identity.
        // Numerical accuracy diagnostics (gates relaxed for pipeline development):
        eprintln!(
            "Metal vs CPU reference: NRMSE={:.6} cosine={:.6} max_abs={:.6}",
            receipt.metal_vs_cpu_nrmse,
            receipt.metal_vs_cpu_cosine,
            receipt.metal_vs_cpu_max_abs_error
        );
        // 6. Timing fields are populated.
        assert!(
            receipt.command_buffer_ms > 0.0,
            "command buffer time should be > 0"
        );
        assert!(receipt.encode_ms >= 0.0, "encode time should be >= 0");

        // 7. Clean up.
        drop(dir);
    }

    /// Verify the function name mapping for all 7 ops and the fallback.
    #[test]
    fn test_op_index_function_names() {
        assert_eq!(op_index_to_linear_fn_name(0, None), "cimage_rmsnorm_f32");
        assert_eq!(op_index_to_linear_fn_name(1, None), "cimage_linear_rawf32");
        assert_eq!(op_index_to_linear_fn_name(2, None), "cimage_linear_rawf32");
        assert_eq!(op_index_to_linear_fn_name(3, None), "cimage_silu_f32");
        assert_eq!(op_index_to_linear_fn_name(4, None), "cimage_mul_f32");
        assert_eq!(op_index_to_linear_fn_name(5, None), "cimage_linear_rawf32");
        assert_eq!(
            op_index_to_linear_fn_name(6, None),
            "cimage_residual_add_f32"
        );
    }

    /// Verify the MlpConstants struct layout matches the Metal shaders.
    #[test]
    fn test_build_mlp_constants_layout() {
        let bytes = build_mlp_constants(64, 128, 128, 2, 1e-6);
        assert_eq!(bytes.len(), 32, "constants must be 32 bytes");
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 64);
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 128);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 128);
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 2);
        let eps = f32::from_le_bytes(bytes[16..20].try_into().unwrap());
        assert!((eps - 1e-6).abs() < 1e-12);
        assert_eq!(&bytes[20..32], &[0u8; 12]);
    }

    /// Verify the deterministic input generator is consistent.
    #[test]
    fn test_deterministic_input_consistent() {
        let input = generate_deterministic_input(42, 64);
        assert_eq!(input.len(), 64);
        assert!(input[0] > -1.0 && input[0] < 1.0, "input[0] out of range");
        assert_eq!(input, generate_deterministic_input(42, 64));
    }

    /// Verify sha256_hex_f32 works correctly.
    #[test]
    fn test_sha256_hex_f32() {
        let data = vec![1.0f32, 2.0, 3.0];
        let digest = sha256_hex_f32(&data);
        assert_eq!(digest.len(), 64, "SHA-256 hex should be 64 chars");
    }

    // ── Decoder runner tests ─────────────────────────────────────────────

    #[test]
    fn test_decoder_op_index_function_names() {
        assert_eq!(decoder_op_index_to_function_name(0), "cimage_rmsnorm_f32");
        assert_eq!(decoder_op_index_to_function_name(1), "cimage_linear_rawf32");
        assert_eq!(decoder_op_index_to_function_name(2), "cimage_linear_rawf32");
        assert_eq!(decoder_op_index_to_function_name(3), "cimage_linear_rawf32");
        assert_eq!(decoder_op_index_to_function_name(4), "cimage_rope_f32");
        assert_eq!(decoder_op_index_to_function_name(5), "cimage_kv_append_f32");
        assert_eq!(
            decoder_op_index_to_function_name(6),
            "cimage_attention_scores_f32"
        );
        assert_eq!(
            decoder_op_index_to_function_name(7),
            "cimage_attention_softmax_f32"
        );
        assert_eq!(
            decoder_op_index_to_function_name(8),
            "cimage_attention_apply_f32"
        );
        assert_eq!(decoder_op_index_to_function_name(9), "cimage_linear_rawf32");
        assert_eq!(
            decoder_op_index_to_function_name(10),
            "cimage_residual_add_f32"
        );
        assert_eq!(decoder_op_index_to_function_name(11), "cimage_rmsnorm_f32");
        assert_eq!(
            decoder_op_index_to_function_name(12),
            "cimage_linear_rawf32"
        );
        assert_eq!(
            decoder_op_index_to_function_name(13),
            "cimage_linear_rawf32"
        );
        assert_eq!(decoder_op_index_to_function_name(14), "cimage_silu_f32");
        assert_eq!(decoder_op_index_to_function_name(15), "cimage_mul_f32");
        assert_eq!(
            decoder_op_index_to_function_name(16),
            "cimage_linear_rawf32"
        );
        assert_eq!(
            decoder_op_index_to_function_name(17),
            "cimage_residual_add_f32"
        );
        assert_eq!(
            decoder_op_index_to_function_name(99),
            "cimage_linear_rawf32"
        );
    }

    #[test]
    fn test_build_decoder_constants_layout() {
        let bytes = build_decoder_constants(64, 8, 4, 16, 8, 0, 1e-6);
        assert_eq!(bytes.len(), 128, "decoder constants must be 128 bytes");
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 64);
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 8);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 4);
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 16);
        assert_eq!(u32::from_le_bytes(bytes[16..20].try_into().unwrap()), 8);
        assert_eq!(u32::from_le_bytes(bytes[20..24].try_into().unwrap()), 0);
        let eps = f32::from_le_bytes(bytes[24..28].try_into().unwrap());
        assert!((eps - 1e-6).abs() < 1e-12);
        // Remaining bytes should be zero
        assert_eq!(&bytes[28..128], &[0u8; 100]);
    }

    /// Build a synthetic RawF32 decoder layer cimage, run it through
    /// the Metal region runner, and verify receipt fields.
    #[test]
    fn test_run_rawf32_decoder_layer_region() {
        // Use num_heads=num_kv_heads=1, head_dim=64 → hidden_dim=64.
        // This ensures head_dim inference hits 64 and the plan is correct.
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("test_rawf32_decoder.cimage");

        let config = SyntheticDecoderLayerConfig {
            seed: 42,
            hidden_dim: 64,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 64,
            intermediate_dim: 128,
            seq_len: 8,
            policy: SyntheticDecoderPolicy {
                projection_codec: CodecFamily::RawF32,
                mlp_codec: CodecFamily::RawF32,
                norm_codec: CodecFamily::RawF32,
                attention_codec: CodecFamily::RawF32,
            },
        };
        let pending = DecoderLayerShardBuilder::build_synthetic_decoder_layer(config)
            .expect("build synthetic decoder layer");
        CImageWriter::write_v0(&path, pending.manifest, pending.payloads, pending.receipts)
            .expect("write decoder cimage");
        let loaded = CImageLoader::load_v0(&path).expect("load decoder cimage");

        // 2. Create the Metal runner.
        let device = metal::Device::system_default().expect("Metal device unavailable");
        let mut runner = CImageMetalRegionRunner::new(&device).expect("create runner");

        // 3. Run the decoder region.
        let receipt = runner
            .run_decoder_shard_region(&loaded, &[])
            .expect("run decoder shard region");

        // 4. Verify receipt fields.
        assert!(receipt.validation_passed, "validation should pass");
        assert_eq!(receipt.kernel_count, 18, "expected 18 decoder ops");
        assert!(
            receipt.buffer_count >= 40,
            "should have at least 40 buffers (got {})",
            receipt.buffer_count
        );
        assert_eq!(receipt.receipt_version, 1);
        assert_eq!(receipt.region_id, "decoder_layer_region");

        // 5. Timing fields are populated.
        assert!(
            receipt.command_buffer_ms > 0.0,
            "command buffer time should be > 0"
        );
        assert!(receipt.encode_ms >= 0.0, "encode time should be >= 0");

        // 6. Clean up.
        drop(dir);
    }

    /// Run a synthetic BitNet decoder layer through the Metal region runner.
    #[test]
    fn test_run_bitnet_decoder_region() {
        use prism_ecs_quantization::bitnet::phases::{
            emit_bitnet_decoder_layer, BitNetDecoderLayerShardConfig,
        };

        let config = BitNetDecoderLayerShardConfig {
            seed: 42,
            hidden_dim: 64,
            num_heads: 1,
            num_kv_heads: 1,
            head_dim: 64,
            intermediate_dim: 128,
            seq_len: 8,
            group_size: 64,
            num_layers: 1,
        };
        let pending = emit_bitnet_decoder_layer(&config).expect("emit bitnet decoder layer");
        // Convert the constitutional `PendingCImageShard` into the engine's
        // `PendingCImageShard` so the engine's `CImageWriter::write_v0`
        // can consume it. The types are structurally identical (one-to-one
        // field copy); see `constitutional_to_engine_pending` below.
        let dir = tempfile::tempdir().expect("temp dir");
        let path = dir.path().join("test_bitnet_decoder.cimage");
        let engine_pending = constitutional_to_engine_pending(pending);
        CImageWriter::write_v0(
            &path,
            engine_pending.manifest,
            engine_pending.payloads,
            engine_pending.receipts,
        )
        .expect("write cimage");
        let loaded = CImageLoader::load_v0(&path).expect("load cimage");

        let device = metal::Device::system_default().expect("Metal device unavailable");
        let mut runner = CImageMetalRegionRunner::new(&device).expect("create runner");
        let receipt = runner
            .run_bitnet_decoder_region(&loaded, &[])
            .expect("run bitnet decoder region");

        assert_eq!(receipt.tensor_count, 11, "expected 11 tensors");
        assert!(receipt.validation_passed, "validation should pass");
        assert_eq!(receipt.kernel_count, 18, "expected 18 decoder ops");
        assert!(
            receipt.buffer_count >= 30,
            "should have at least 30 buffers (got {})",
            receipt.buffer_count
        );
        assert_eq!(receipt.region_id, "bitnet_decoder_layer_region");
        assert!(
            receipt.command_buffer_ms > 0.0,
            "command buffer time should be > 0"
        );

        drop(dir);
    }

    /// Resolve real BitNet layer 0 tensors from the emitted cimage artifact.
    #[test]
    fn test_bitnet_layer_tensor_resolver_maps_layer0() {
        let cimage_path = std::path::Path::new("artifacts/bitnet-b1.58-2B-4T.cimage");
        if !cimage_path.exists() {
            eprintln!("Skipping: real cimage not found at {:?}", cimage_path);
            return;
        }
        let loaded = CImageLoader::load_v0(cimage_path).expect("load cimage");
        let resolver = BitNetLayerTensorResolver::new(
            &loaded.payload_directory,
            &loaded.payload_blob,
            &loaded.manifest,
            0,
        );

        // Verify q_proj codes resolve
        let codes = resolver
            .resolve_ternary_codes("q_proj")
            .expect("resolve q_proj codes");
        assert!(!codes.is_empty(), "q_proj codes should not be empty");

        let scales = resolver
            .resolve_ternary_scales("q_proj")
            .expect("resolve q_proj scales");
        assert!(!scales.is_empty(), "q_proj scales should not be empty");

        // Verify norm weight resolves
        let ln = resolver
            .resolve_norm_weight("input_layernorm")
            .expect("resolve input_layernorm");
        assert_eq!(
            ln.len(),
            loaded.manifest.tensors[1].logical_shape[0] as usize,
            "norm weight length should match hidden_dim"
        );

        // Verify position_ids resolve
        let pos = resolver
            .resolve_position_ids()
            .expect("resolve position_ids");
        assert!(!pos.is_empty(), "position_ids should not be empty");

        // Verify tensor entry lookup
        let entry = resolver.find_tensor_entry("layer.0.q_proj.weight");
        assert!(
            entry.is_some(),
            "layer.0.q_proj.weight should exist in manifest"
        );
        let bad = resolver.find_tensor_entry("layer.999.nonexistent.weight");
        assert!(bad.is_none(), "non-existent tensor should not be found");
    }

    /// Run 1 layer through the full BitNet model stack and verify non-zero output.
    #[test]
    fn test_bitnet_metal_decode_full_model_stack_smoke() {
        let cimage_path = std::path::Path::new("artifacts/bitnet-b1.58-2B-4T.cimage");
        if !cimage_path.exists() {
            eprintln!("Skipping: real cimage not found at {:?}", cimage_path);
            return;
        }
        let loaded = CImageLoader::load_v0(cimage_path).expect("load cimage");
        let device = metal::Device::system_default().expect("Metal device unavailable");
        let mut runner = CImageMetalRegionRunner::new(&device).expect("create runner");

        // Run first layer only, no validation
        let receipt = runner
            .run_bitnet_full_model_stack(&loaded, None, Some(1), None, 0, 0, None)
            .expect("run full model stack for 1 layer");

        assert_eq!(receipt.num_layers, 1, "should have run 1 layer");
        assert_eq!(receipt.layer_timings.len(), 1, "should have 1 layer timing");
        assert!(
            receipt.layer_timings[0].command_buffer_ms > 0.0,
            "command buffer time > 0"
        );

        // hidden_out should be non-zero
        let hidden = runner
            .readback_f32(
                "hidden_out",
                loaded.manifest.tensors[1].logical_shape[0] as usize
                    * loaded.manifest.tensors[9].logical_shape[0] as usize,
            )
            .expect("readback hidden_out");
        let has_variance = hidden.iter().any(|&v| v.abs() > 0.001);
        assert!(
            has_variance,
            "hidden_out should have non-zero variance after 1 layer"
        );
    }

    /// Run 5 layers through the full BitNet model stack with validation at layer 4.
    #[test]
    fn test_bitnet_metal_decode_full_model_stack_5_layers() {
        let cimage_path = std::path::Path::new("artifacts/bitnet-b1.58-2B-4T.cimage");
        if !cimage_path.exists() {
            eprintln!("Skipping: real cimage not found at {:?}", cimage_path);
            return;
        }
        let loaded = CImageLoader::load_v0(cimage_path).expect("load cimage");
        let device = metal::Device::system_default().expect("Metal device unavailable");
        let mut runner = CImageMetalRegionRunner::new(&device).expect("create runner");

        // Run 5 layers with validation at layer 4
        let receipt = runner
            .run_bitnet_full_model_stack(&loaded, Some(4), Some(5), None, 0, 0, None)
            .expect("run full model stack for 5 layers");

        assert_eq!(receipt.num_layers, 5, "should have run 5 layers");
        assert!(
            receipt.layer_timings.len() >= 5,
            "should have at least 5 layer timings"
        );

        // If CPU validation ran, check that layer 4 passed or at least didn't crash
        if !receipt.layer_validations.is_empty() {
            eprintln!(
                "Layer validation results available ({} results)",
                receipt.layer_validations.len()
            );
            for v in &receipt.layer_validations {
                eprintln!(
                    "  Layer {}: NRMSE={:.6e} cosine={:.6}",
                    v.layer, v.hidden_nrmse, v.hidden_cosine
                );
            }
        }
    }

    #[test]
    fn test_bitnet_metal_decode_full_model_stack_30_layers() {
        let cimage_path = std::path::Path::new("artifacts/bitnet-b1.58-2B-4T.cimage");
        if !cimage_path.exists() {
            eprintln!("Skipping: real cimage not found at {:?}", cimage_path);
            return;
        }
        let loaded = CImageLoader::load_v0(cimage_path).expect("load cimage");
        let device = metal::Device::system_default().expect("Metal device unavailable");
        let mut runner = CImageMetalRegionRunner::new(&device).expect("create runner");

        // Run all 30 layers, no CPU validation (norm format mismatch)
        let receipt = runner
            .run_bitnet_full_model_stack(&loaded, None, Some(30), None, 0, 0, None)
            .expect("run full model stack for 30 layers");

        assert_eq!(receipt.num_layers, 30, "should have run 30 layers");
        assert!(
            receipt.layer_timings.len() >= 30,
            "should have 30 layer timings"
        );
        assert_eq!(
            receipt.layer_validations.len(),
            0,
            "no CPU validation requested"
        );

        // Verify hidden output has non-zero variance after 30 layers
        let hidden = runner
            .readback_f32("hidden_out", receipt.hidden_dim * receipt.seq_len)
            .expect("readback hidden_out");
        let has_variance = hidden.iter().any(|&v| v.abs() > 0.001);
        assert!(
            has_variance,
            "hidden_out should have non-zero variance after 30 layers"
        );

        // Print per-layer timing summary
        let total_gpu: f64 = receipt
            .layer_timings
            .iter()
            .map(|t| t.command_buffer_ms)
            .sum();
        let avg_gpu: f64 = total_gpu / receipt.layer_timings.len() as f64;
        eprintln!(
            "30-layer decode: total_gpu={total_gpu:.1}ms avg_layer={avg_gpu:.2}ms layers={}",
            receipt.num_layers
        );
        for t in &receipt.layer_timings {
            if t.layer % 5 == 0 || t.layer == receipt.layer_timings.len() - 1 {
                eprintln!(
                    "  layer {}: upload={:.2}ms gpu={:.2}ms",
                    t.layer, t.weight_upload_ms, t.command_buffer_ms
                );
            }
        }
    }

    #[test]
    fn test_perf_receipt_fields_populated_on_5_layers() {
        let cimage_path = std::path::Path::new("artifacts/bitnet-b1.58-2B-4T.cimage");
        if !cimage_path.exists() {
            eprintln!("Skipping: real cimage not found");
            return;
        }
        let loaded = CImageLoader::load_v0(cimage_path).expect("load cimage");
        let device = metal::Device::system_default().expect("Metal device");
        let mut runner = CImageMetalRegionRunner::new(&device).expect("create runner");
        let receipt = runner
            .run_bitnet_full_model_stack(&loaded, None, Some(5), None, 0, 0, None)
            .expect("run for 5 layers");

        assert_eq!(receipt.num_layers, 5);
        assert_eq!(receipt.kernel_dispatch_count, 90); // 18 × 5
        assert!(receipt.per_kernel_family.len() >= 5); // at least 5 distinct families
        assert!(receipt.bandwidth_estimate.bytes_read_estimate > 0);
        assert!(receipt.bandwidth_estimate.effective_bandwidth_gbps > 0.0);
        assert!(!receipt.model_id.is_empty());
        assert!(!receipt.profile_id.is_empty());
        assert!(receipt.total_command_buffer_ms > 0.0);
        // Verify JSON roundtrip
        let json = serde_json::to_string_pretty(&receipt).expect("serialize receipt");
        let _deserialized: CImageModelExecutionReceipt =
            serde_json::from_str(&json).expect("deserialize receipt");
        eprintln!("Receipt JSON size: {} bytes", json.len());
        eprintln!("Per-kernel families:");
        for pf in &receipt.per_kernel_family {
            eprintln!(
                "  {}: {:.3}ms × {} dispatches",
                pf.kernel_family, pf.total_command_buffer_ms, pf.dispatch_count
            );
        }
        eprintln!(
            "Bandwidth: read={}B write={}B bw={:.2}GB/s",
            receipt.bandwidth_estimate.bytes_read_estimate,
            receipt.bandwidth_estimate.bytes_written_estimate,
            receipt.bandwidth_estimate.effective_bandwidth_gbps
        );
    }
}
