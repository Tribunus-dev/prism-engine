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
    /// concatenated together, so every kernel function lives in one library.
    pub fn new(device: &metal::Device) -> CImageRuntimeResult<Self> {
        let shader_source = format!(
            "{}\n{}\n{}\n{}\n{}\n{}\n{}",
            include_str!("../compute_image/templates/cimage_rmsnorm_f32.metal"),
            include_str!("../compute_image/templates/cimage_linear_rawf32.metal"),
            include_str!("../compute_image/templates/cimage_linear_int8.metal"),
            include_str!("../compute_image/templates/cimage_linear_nf4.metal"),
            include_str!("../compute_image/templates/cimage_silu_f32.metal"),
            include_str!("../compute_image/templates/cimage_mul_f32.metal"),
            include_str!("../compute_image/templates/cimage_residual_add_f32.metal"),
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
            return Err(CImageRuntimeError::HazardViolation(
                "region hazard check failed".into(),
            ));
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
        assert!(receipt.hazard_safe, "hazard check should pass");
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
        assert!(
            receipt.metal_vs_cpu_nrmse < 1e-3,
            "NRMSE too high: {:.6}",
            receipt.metal_vs_cpu_nrmse
        );
        assert!(
            receipt.metal_vs_cpu_cosine > 0.999,
            "cosine too low: {:.6}",
            receipt.metal_vs_cpu_cosine
        );
        assert!(
            receipt.metal_vs_cpu_max_abs_error < 1.0,
            "max abs error too high: {:.6}",
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
}
