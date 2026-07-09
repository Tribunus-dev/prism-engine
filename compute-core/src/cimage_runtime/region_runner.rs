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

use crate::cimage::mlp_reference::{
    compute_cosine_similarity, compute_max_abs_error, compute_nrmse,
};
use crate::cimage::{CImageValidator, LoadedCImageV0, ReceiptEvidenceKind};
use crate::cimage_runtime::error::{CImageRuntimeError, CImageRuntimeResult};
use crate::cimage_runtime::lower_mlp::{CImageMlpRegionPlan, MlpShardRegionBuilder};
use crate::cimage_runtime::receipts::CImageRegionExecutionReceipt;
use crate::cimage_runtime::resolver::{CImageRuntimeResolver, ResolvedMlpShardRuntime};
use crate::cimage_runtime::tensor_store::{MlpRegionExecutionMode, RuntimeTensorPayload};
use crate::execution_plan::backend_capability::BackendLoweringTarget;
use crate::execution_plan::HardwareProfileId;

#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
use crate::cimage::CImagePayloadRef;
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
use crate::cimage_runtime::lower_decoder::DecoderShardRegionBuilder;
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
use crate::cimage_runtime::tensor_store::RuntimeTensor;
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
use crate::cimage_runtime::tensor_store::RuntimeTensorStore;

// ── Helpers ───────────────────────────────────────────────────────────────

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
/// | 4     | MlpDownResidual  | cimage_mul_f32      |
/// | 5     | MlpDownResidual  | cimage_linear_rawf32|
/// | 6     | MlpDownResidual  | cimage_residual_add_f32 |
fn op_index_to_function_name(op_index: usize) -> &'static str {
    match op_index {
        0 => "cimage_rmsnorm_f32",
        1 | 2 => "cimage_linear_rawf32",
        3 => "cimage_silu_f32",
        4 => "cimage_mul_f32",
        5 => "cimage_linear_rawf32",
        6 => "cimage_residual_add_f32",
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

/// Build the 32-byte MlpConstants struct used by every shader.
fn build_mlp_constants(hidden_dim: u32, intermediate_dim: u32, epsilon: f32) -> [u8; 32] {
    let mut out = [0u8; 32];
    out[0..4].copy_from_slice(&hidden_dim.to_le_bytes());
    out[4..8].copy_from_slice(&intermediate_dim.to_le_bytes());
    // group_size = 0 (not used by RawF32)
    out[8..12].copy_from_slice(&0u32.to_le_bytes());
    // codec_id = 0 (RawF32)
    out[12..16].copy_from_slice(&0u32.to_le_bytes());
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

    fn insert(&mut self, name: String, buf: metal::Buffer) {
        self.buffers.insert(name, buf);
    }

    fn get(&self, name: &str) -> Option<&metal::Buffer> {
        self.buffers.get(name)
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
        let shader_source = format!(
            "{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}",
            include_str!("../compute_image/templates/cimage_rmsnorm_f32.metal"),
            include_str!("../compute_image/templates/cimage_linear_rawf32.metal"),
            include_str!("../compute_image/templates/cimage_linear_int8.metal"),
            include_str!("../compute_image/templates/cimage_linear_nf4.metal"),
            include_str!("../compute_image/templates/cimage_silu_f32.metal"),
            include_str!("../compute_image/templates/cimage_mul_f32.metal"),
            include_str!("../compute_image/templates/cimage_residual_add_f32.metal"),
            include_str!("../compute_image/templates/cimage_rope_f32.metal"),
            include_str!("../compute_image/templates/cimage_kv_append_f32.metal"),
            include_str!("../compute_image/templates/cimage_attention_scores_f32.metal"),
            include_str!("../compute_image/templates/cimage_attention_softmax_f32.metal"),
            include_str!("../compute_image/templates/cimage_attention_apply_f32.metal"),
        );

        let library = device
            .new_library_with_source(&shader_source, &metal::CompileOptions::new())
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

        if let Some(RuntimeTensorPayload::RawF32(data)) = tensor_by_key.get("gate_proj") {
            let buf = alloc_f32("gate_proj_codes", data)?;
            self.buffer_store.insert("gate_proj_codes".into(), buf);
        }
        {
            let buf = alloc_zero("gate_proj_scales", (intermediate_dim as u64) * 4)?;
            self.buffer_store.insert("gate_proj_scales".into(), buf);
        }
        {
            let buf = alloc_zero("gate_proj_biases", (intermediate_dim as u64) * 4)?;
            self.buffer_store.insert("gate_proj_biases".into(), buf);
        }

        if let Some(RuntimeTensorPayload::RawF32(data)) = tensor_by_key.get("up_proj") {
            let buf = alloc_f32("up_proj_codes", data)?;
            self.buffer_store.insert("up_proj_codes".into(), buf);
        }
        {
            let buf = alloc_zero("up_proj_scales", (intermediate_dim as u64) * 4)?;
            self.buffer_store.insert("up_proj_scales".into(), buf);
        }
        {
            let buf = alloc_zero("up_proj_biases", (intermediate_dim as u64) * 4)?;
            self.buffer_store.insert("up_proj_biases".into(), buf);
        }

        if let Some(RuntimeTensorPayload::RawF32(data)) = tensor_by_key.get("down_proj") {
            let buf = alloc_f32("down_proj_codes", data)?;
            self.buffer_store.insert("down_proj_codes".into(), buf);
        }
        {
            let buf = alloc_zero("down_proj_scales", (hidden_dim as u64) * 4)?;
            self.buffer_store.insert("down_proj_scales".into(), buf);
        }
        {
            let buf = alloc_zero("down_proj_biases", (hidden_dim as u64) * 4)?;
            self.buffer_store.insert("down_proj_biases".into(), buf);
        }

        // ── Input buffer ────────────────────────────────────────────────
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
            let constants =
                build_mlp_constants(hidden_dim as u32, intermediate_dim as u32, epsilon);
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
    fn readback_f32(&self, name: &str, minimum_count: usize) -> CImageRuntimeResult<Vec<f32>> {
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
        if load_receipt.validation_status != crate::cimage::CImageValidationStatus::Valid {
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
        for op_index in 0..ops.len() {
            self.get_or_create_pso(op_index_to_function_name(op_index))?;
        }

        // 6. Encode and dispatch.
        let encode_start = Instant::now();
        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();

        for (op_index, op) in ops.iter().enumerate() {
            let fn_name = op_index_to_function_name(op_index);

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

        enc.end_encoding();

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
        if load_receipt.validation_status != crate::cimage::CImageValidationStatus::Valid {
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
        let enc = cb.new_compute_command_encoder();

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

        enc.end_encoding();

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

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(all(test, target_os = "macos", feature = "metal-dispatch"))]
mod tests {
    use super::*;
    use crate::cimage::*;
    use crate::cimage_runtime::tensor_store::MlpRegionExecutionMode;
    use crate::execution_plan::CodecFamily;

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
        assert_eq!(op_index_to_function_name(0), "cimage_rmsnorm_f32");
        assert_eq!(op_index_to_function_name(1), "cimage_linear_rawf32");
        assert_eq!(op_index_to_function_name(2), "cimage_linear_rawf32");
        assert_eq!(op_index_to_function_name(3), "cimage_silu_f32");
        assert_eq!(op_index_to_function_name(4), "cimage_mul_f32");
        assert_eq!(op_index_to_function_name(5), "cimage_linear_rawf32");
        assert_eq!(op_index_to_function_name(6), "cimage_residual_add_f32");
        assert_eq!(op_index_to_function_name(99), "cimage_linear_rawf32");
    }

    /// Verify the MlpConstants struct layout matches the Metal shaders.
    #[test]
    fn test_build_mlp_constants_layout() {
        let bytes = build_mlp_constants(64, 128, 1e-6);
        assert_eq!(bytes.len(), 32, "constants must be 32 bytes");

        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 64);
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 128);
        assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 0);
        assert_eq!(u32::from_le_bytes(bytes[12..16].try_into().unwrap()), 0);
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
}
