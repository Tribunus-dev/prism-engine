//! Compute dispatch wrappers — host-side dispatch for each Metal kernel class.
//!
//! Each dispatcher is a stateless struct holding only an `Arc<parking_lot::Mutex<KernelRegistry>>`.
//! The `dispatch()` method takes buffer references, projection parameters, and a mutable
//! receipt, then encodes a compute pass on the provided command buffer.
//!
//! The caller owns command-buffer lifecycle (commit, wait, read-back).  The dispatcher
//! populates receipt fields known at dispatch time (page count, threadgroup layout, etc.)
//! and records CPU-side setup timing.  GPU timing must be recorded by the caller around
//! `commit() + wait_until_completed()`.

#[cfg(feature = "metal-dispatch")]
use super::kernel_registry::{projection_constants, KernelRegistry};
#[cfg(feature = "metal-dispatch")]
use super::kernel_types::{buffer_slot, KernelReceipt, ProjectionParams};
#[cfg(feature = "metal-dispatch")]
use metal::*;
#[cfg(feature = "metal-dispatch")]
use parking_lot::Mutex;
#[cfg(feature = "metal-dispatch")]
use std::sync::Arc;
#[cfg(feature = "metal-dispatch")]
use std::time::Instant;

// ── Type alias for the shared registry ──────────────────────────────────────

/// Shared handle to the kernel registry.
#[cfg(feature = "metal-dispatch")]
pub type RegistryRef = Arc<Mutex<KernelRegistry>>;

// ═══════════════════════════════════════════════════════════════════════════
// TernaryProjectionDispatcher
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatches the `ternary_tile640_gemv` kernel for tile640 (1.6-bit) ternary GEMV.
///
/// Buffer layout matches `compute_image/templates/ternary_tile640_gemv.metal`:
///   [[buffer(0)]]  packed_weights      — tile640 u32 words
///   [[buffer(1)]]  input               — fp16 activation vector
///   [[buffer(2)]]  page_scales         — bf16 per-page max
///   [[buffer(3)]]  channel_scales      — int8 per-lane scale
///   [[buffer(4)]]  output              — fp16 result vector
///   [[buffer(5)]]  in_dim              — constant uint
///   [[buffer(6)]]  out_dim             — constant uint
#[cfg(feature = "metal-dispatch")]
pub struct TernaryProjectionDispatcher {
    registry: RegistryRef,
    kernel_name: &'static str,
}

#[cfg(feature = "metal-dispatch")]
impl TernaryProjectionDispatcher {
    pub fn new(registry: RegistryRef) -> Self {
        TernaryProjectionDispatcher {
            registry,
            kernel_name: "ternary_tile640_gemv",
        }
    }

    /// Encode a ternary tile640 GEMV dispatch.
    ///
    /// `page_width` is typically 640.  `instrumented` controls whether a receipt
    /// buffer is bound for kernel-side instrumentation counters and whether the
    /// function constant `instrumented` is set to `true`.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        command_buffer: &CommandBufferRef,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
        page_scales_buffer: &Buffer,
        channel_scales_buffer: &Buffer,
        sidecar_buffer: Option<&Buffer>,
        sidecar_offsets_buffer: Option<&Buffer>,
        output_buffer: &Buffer,
        params: &ProjectionParams,
        receipt: &mut KernelReceipt,
        instrumented: bool,
    ) {
        let (pso, device) = {
            let mut reg = self.registry.lock();
            let fcv = FunctionConstantValues::new();
            let pso = reg.get_or_create(self.kernel_name, &fcv, 0);
            (pso, reg.device().clone())
        };
        let _ = sidecar_buffer;
        let _ = sidecar_offsets_buffer;

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pso);

        // Data buffers
        encoder.set_buffer(buffer_slot::WEIGHTS as u64, Some(weights_buffer), 0);
        encoder.set_buffer(buffer_slot::INPUT as u64, Some(input_buffer), 0);
        encoder.set_buffer(buffer_slot::PAGE_SCALES as u64, Some(page_scales_buffer), 0);
        encoder.set_buffer(
            buffer_slot::CHANNEL_SCALES as u64,
            Some(channel_scales_buffer),
            0,
        );
        encoder.set_buffer(buffer_slot::OUTPUT as u64, Some(output_buffer), 0);
        let in_dim_buf = device.new_buffer_with_data(
            &params.in_dim as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_dim_buf = device.new_buffer_with_data(
            &params.out_dim as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(5, Some(&in_dim_buf), 0);
        encoder.set_buffer(6, Some(&out_dim_buf), 0);
        let _ = instrumented;

        // Dispatch: one threadgroup (64 threads) per output row.
        let out_dim = params.out_dim;
        encoder.dispatch_thread_groups(
            MTLSize {
                width: out_dim as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();

        // Populate receipt fields known at dispatch time
        receipt.kernel_id = 1; // TERNARY_PROJECTION
        receipt.phase_id = 0;
        receipt.page_count = params.page_count;
        receipt.sidecar_hits = 0; // populated by kernel when instrumented
        receipt.sidecar_entries_read = 0;
        receipt.threadgroups = out_dim;
        receipt.threads_per_threadgroup = 64;
        receipt.output_elements = out_dim;
        receipt.flags = if instrumented { 1 } else { 0 };
        receipt.logical_weight_bytes =
            (params.page_width as u64) * (params.page_count as u64) / 8 * 13; // ~1.6 bpw
        receipt.logical_sidecar_bytes = 0;
        receipt.logical_activation_bytes = (params.in_dim as u64) * (params.out_dim as u64) * 2;
        // fp16
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Nf4Tile640ProjectionDispatcher
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatches the shared-layout NF4 Tile640 FP32 GEMV kernel.
///
/// Buffer layout matches `compute_image/templates/nf4_tile640_gemv.metal`:
///   [[buffer(0)]]  packed_weights   — raw Tile640 packed u8 rows
///   [[buffer(1)]]  scales           — fp32 per-group scales
///   [[buffer(2)]]  biases           — fp32 per-group biases
///   [[buffer(3)]]  input            — fp32 activation vector [in_dim]
///   [[buffer(4)]]  output           — fp32 result vector
///   [[buffer(5)]]  num_macro_tiles  — constant uint (ceil(in_dim / 640))
///   [[buffer(6)]]  in_dim           — constant uint (real width; guards the
///                                     partial-tile activation read)
#[cfg(feature = "metal-dispatch")]
pub struct Nf4Tile640ProjectionDispatcher {
    registry: RegistryRef,
    kernel_name: &'static str,
}

#[cfg(feature = "metal-dispatch")]
#[derive(Clone, Copy, Debug, Default)]
pub struct Nf4Tile640Offsets {
    pub weights_offset: u64,
    pub scales_offset: u64,
    pub biases_offset: u64,
}

#[cfg(feature = "metal-dispatch")]
impl Nf4Tile640ProjectionDispatcher {
    pub fn new(registry: RegistryRef) -> Self {
        Self {
            registry,
            kernel_name: "fused_gemv_nf4_tile640_fp32",
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        command_buffer: &CommandBufferRef,
        packed_weights_buffer: &Buffer,
        scales_buffer: &Buffer,
        biases_buffer: &Buffer,
        input_buffer: &Buffer,
        output_buffer: &Buffer,
        params: &ProjectionParams,
        receipt: &mut KernelReceipt,
    ) {
        self.dispatch_with_offsets(
            command_buffer,
            packed_weights_buffer,
            scales_buffer,
            biases_buffer,
            input_buffer,
            output_buffer,
            params,
            Nf4Tile640Offsets::default(),
            receipt,
        );
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_with_offsets(
        &self,
        command_buffer: &CommandBufferRef,
        packed_weights_buffer: &Buffer,
        scales_buffer: &Buffer,
        biases_buffer: &Buffer,
        input_buffer: &Buffer,
        output_buffer: &Buffer,
        params: &ProjectionParams,
        offsets: Nf4Tile640Offsets,
        receipt: &mut KernelReceipt,
    ) {
        let (pso, device) = {
            let mut reg = self.registry.lock();
            let fcv = FunctionConstantValues::new();
            let pso = reg.get_or_create(self.kernel_name, &fcv, 0);
            (pso, reg.device().clone())
        };

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pso);
        encoder.set_buffer(0, Some(packed_weights_buffer), offsets.weights_offset);
        encoder.set_buffer(1, Some(scales_buffer), offsets.scales_offset);
        encoder.set_buffer(2, Some(biases_buffer), offsets.biases_offset);
        encoder.set_buffer(3, Some(input_buffer), 0);
        encoder.set_buffer(4, Some(output_buffer), 0);

        let num_macro_tiles = params.page_count.max(1);
        let num_macro_tiles_buf = device.new_buffer_with_data(
            &num_macro_tiles as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(5, Some(&num_macro_tiles_buf), 0);

        // buffer(6): real (unpadded) input width. The kernel guards the
        // activation read against this so a partial last tile (in_dim not a
        // multiple of 640) never reads past `in_vector[in_dim]`.
        let in_dim_val = params.in_dim;
        let in_dim_buf = device.new_buffer_with_data(
            &in_dim_val as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(6, Some(&in_dim_buf), 0);

        encoder.dispatch_thread_groups(
            MTLSize {
                width: params.out_dim as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();

        receipt.kernel_id = 12; // NF4_TILE640_PROJECTION
        receipt.phase_id = 0;
        receipt.page_count = num_macro_tiles;
        receipt.sidecar_hits = 0;
        receipt.sidecar_entries_read = 0;
        receipt.threadgroups = params.out_dim;
        receipt.threads_per_threadgroup = 32;
        receipt.output_elements = params.out_dim;
        receipt.flags = 0;
        receipt.logical_weight_bytes = (params.out_dim as u64) * (num_macro_tiles as u64) * 320;
        receipt.logical_sidecar_bytes =
            (params.out_dim as u64) * (num_macro_tiles as u64) * 5 * 2 * 4;
        receipt.logical_activation_bytes = (params.in_dim as u64) * 4 + (params.out_dim as u64) * 4;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// DenseProjectionDispatcher
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatches the `palettized_gemv` kernel for dense codebook-quantized GEMV.
///
/// Buffer layout matches `compute_image/templates/palettized_gemv.metal`:
///   [[buffer(0)]]  input            — fp16 activation vector
///   [[buffer(1)]]  codebook_block   — fp16 codebook values
///   [[buffer(2)]]  indices_block    — packed uint8 indices
///   [[buffer(3)]]  output           — fp16 result vector
///   [[buffer(4)]]  in_dim           — constant uint
///   [[buffer(5)]]  out_dim          — constant uint
///
/// Used by `MetalTeacher` for the dense teacher forward pass in Level 1.
#[cfg(feature = "metal-dispatch")]
pub struct DenseProjectionDispatcher {
    registry: RegistryRef,
    kernel_name: &'static str,
}

#[cfg(feature = "metal-dispatch")]
impl DenseProjectionDispatcher {
    pub fn new(registry: RegistryRef) -> Self {
        DenseProjectionDispatcher {
            registry,
            kernel_name: "palettized_gemv",
        }
    }

    /// Encode a dense codebook-quantized GEMV dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        command_buffer: &CommandBufferRef,
        codebook_buffer: &Buffer,
        indices_buffer: &Buffer,
        input_buffer: &Buffer,
        output_buffer: &Buffer,
        params: &ProjectionParams,
        receipt: &mut KernelReceipt,
        instrumented: bool,
    ) {
        let (pso, device) = {
            let mut reg = self.registry.lock();
            let fcv = FunctionConstantValues::new();
            let pso = reg.get_or_create(self.kernel_name, &fcv, 0);
            (pso, reg.device().clone())
        };

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pso);

        // Bind per the shipped palettized_gemv ABI.
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(codebook_buffer), 0);
        encoder.set_buffer(2, Some(indices_buffer), 0);
        encoder.set_buffer(3, Some(output_buffer), 0);

        let in_dim_buf = device.new_buffer_with_data(
            &params.in_dim as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_dim_buf = device.new_buffer_with_data(
            &params.out_dim as *const u32 as *const std::ffi::c_void,
            std::mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(4, Some(&in_dim_buf), 0);
        encoder.set_buffer(5, Some(&out_dim_buf), 0);
        let _ = instrumented;

        let out_dim = params.out_dim;
        encoder.dispatch_thread_groups(
            MTLSize {
                width: out_dim as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();

        receipt.kernel_id = 2; // DENSE_PROJECTION
        receipt.phase_id = 0;
        receipt.page_count = params.page_count;
        receipt.threadgroups = out_dim;
        receipt.threads_per_threadgroup = 64;
        receipt.output_elements = out_dim;
        receipt.flags = if instrumented { 1 } else { 0 };
        receipt.logical_weight_bytes =
            (std::mem::size_of::<f32>() as u64) * (params.in_dim as u64) * (params.out_dim as u64);
        receipt.logical_activation_bytes = (params.in_dim as u64) * (params.out_dim as u64) * 2;
        // fp16
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ErrorPartialDispatcher
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatches the error-partial kernel that compares teacher vs student outputs.
///
/// Produces one `ErrorPartial` record per comparison element, which the CPU
/// reducer then reduces deterministically.
///
/// Buffer layout:
///   [[buffer(0)]]  teacher_activations — fp16 teacher outputs
///   [[buffer(1)]]  student_activations — fp16 student outputs
///   [[buffer(7)]]  params              — ProjectionParams
///   [[buffer(8)]]  receipt             — KernelReceipt (instrumented only)
///   [[buffer(9)]]  error_partials      — ErrorPartial records
#[cfg(feature = "metal-dispatch")]
pub struct ErrorPartialDispatcher {
    registry: RegistryRef,
    kernel_name: &'static str,
}

#[cfg(feature = "metal-dispatch")]
impl ErrorPartialDispatcher {
    pub fn new(registry: RegistryRef) -> Self {
        ErrorPartialDispatcher {
            registry,
            kernel_name: "error_partial",
        }
    }

    /// Encode a teacher-student error partial dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        command_buffer: &CommandBufferRef,
        teacher_buffer: &Buffer,
        student_buffer: &Buffer,
        error_partials_buffer: &Buffer,
        params: &ProjectionParams,
        receipt: &mut KernelReceipt,
        instrumented: bool,
    ) {
        let (pso, device) = {
            let mut reg = self.registry.lock();
            let (fcv, digest) = projection_constants(params.page_width, false, instrumented);
            let pso = reg.get_or_create(self.kernel_name, &fcv, digest);
            (pso, reg.device().clone())
        };

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pso);

        // For error partial the buffer_slot convention maps WEIGHT/INPUT to teacher/student
        encoder.set_buffer(buffer_slot::INPUT as u64, Some(teacher_buffer), 0);
        encoder.set_buffer(buffer_slot::PAGE_SCALES as u64, Some(student_buffer), 0);
        encoder.set_buffer(
            buffer_slot::ERROR_PARTIALS as u64,
            Some(error_partials_buffer),
            0,
        );

        let params_buf = device.new_buffer_with_data(
            params as *const ProjectionParams as *const std::ffi::c_void,
            std::mem::size_of::<ProjectionParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(buffer_slot::PARAMS as u64, Some(&params_buf), 0);

        let _receipt_buf_handle = instrumented.then(|| {
            let buf = device.new_buffer(
                std::mem::size_of::<KernelReceipt>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            encoder.set_buffer(buffer_slot::RECEIPT as u64, Some(&buf), 0);
            buf
        });

        let n_elements = (params.out_dim * params.in_dim).max(1);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: (n_elements.min(1024)) as u64,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();

        receipt.kernel_id = 3; // ERROR_PARTIAL
        receipt.phase_id = 0;
        receipt.page_count = params.page_count;
        receipt.threadgroups = 1;
        receipt.threads_per_threadgroup = n_elements.min(1024);
        receipt.output_elements = n_elements;
        receipt.flags = if instrumented { 1 } else { 0 };
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// ProbeDispatcher
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatches the attention probe kernel that samples attention distributions.
///
/// Produces sampled attention statistics (entropy, max logit, KL divergence).
///
/// Buffer layout:
///   [[buffer(0)]]  teacher_attn   — fp16 teacher attention scores
///   [[buffer(1)]]  student_attn   — fp16 student attention scores
///   [[buffer(7)]]  params         — ProjectionParams
///   [[buffer(8)]]  receipt        — KernelReceipt (instrumented only)
///   [[buffer(10)]] probe_output   — AttentionProbe records
#[cfg(feature = "metal-dispatch")]
pub struct ProbeDispatcher {
    registry: RegistryRef,
    kernel_name: &'static str,
}

#[cfg(feature = "metal-dispatch")]
impl ProbeDispatcher {
    pub fn new(registry: RegistryRef) -> Self {
        ProbeDispatcher {
            registry,
            kernel_name: "attention_probe",
        }
    }

    /// Encode an attention probe dispatch.
    ///
    /// TODO: `params.in_dim` is repurposed as head count and `params.out_dim` as
    /// probe token count.  A dedicated `ProbeParams` struct would be cleaner once
    /// the shader ABI stabilises.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        command_buffer: &CommandBufferRef,
        teacher_attn_buffer: &Buffer,
        student_attn_buffer: &Buffer,
        probe_output_buffer: &Buffer,
        params: &ProjectionParams,
        receipt: &mut KernelReceipt,
        instrumented: bool,
    ) {
        let (pso, device) = {
            let mut reg = self.registry.lock();
            let (fcv, digest) = projection_constants(params.page_width, false, instrumented);
            let pso = reg.get_or_create(self.kernel_name, &fcv, digest);
            (pso, reg.device().clone())
        };

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pso);

        encoder.set_buffer(buffer_slot::WEIGHTS as u64, Some(teacher_attn_buffer), 0);
        encoder.set_buffer(buffer_slot::INPUT as u64, Some(student_attn_buffer), 0);
        encoder.set_buffer(
            buffer_slot::PROBE_OUTPUT as u64,
            Some(probe_output_buffer),
            0,
        );

        let params_buf = device.new_buffer_with_data(
            params as *const ProjectionParams as *const std::ffi::c_void,
            std::mem::size_of::<ProjectionParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(buffer_slot::PARAMS as u64, Some(&params_buf), 0);

        let _receipt_buf_handle = instrumented.then(|| {
            let buf = device.new_buffer(
                std::mem::size_of::<KernelReceipt>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            encoder.set_buffer(buffer_slot::RECEIPT as u64, Some(&buf), 0);
            buf
        });

        // One threadgroup per head; the probe_seed in params controls sampling.
        let n_heads = params.in_dim.max(1); // TODO: repurposed as head count
        let probe_tokens = params.out_dim.max(1); // TODO: repurposed as probe tokens
        let n_tgs = (n_heads as u64) * (probe_tokens as u64);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: n_tgs,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();

        receipt.kernel_id = 4; // ATTENTION_PROBE
        receipt.phase_id = 0;
        receipt.page_count = 0;
        receipt.threadgroups = n_tgs as u32;
        receipt.threads_per_threadgroup = 32;
        receipt.output_elements = probe_tokens;
        receipt.flags = if instrumented { 1 } else { 0 };
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CandidateScoreDispatcher
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatches the candidate scoring kernel.
///
/// Evaluates how well a proposed page replacement matches the teacher output
/// without fully executing the whole block.
///
/// Buffer layout:
///   [[buffer(0)]]  weights     — candidate page weights
///   [[buffer(1)]]  input       — activation input
///   [[buffer(7)]]  params      — ProjectionParams
///   [[buffer(8)]]  receipt     — KernelReceipt (instrumented only)
///   [[buffer(11)]] page_scores — PageScore records
#[cfg(feature = "metal-dispatch")]
pub struct CandidateScoreDispatcher {
    registry: RegistryRef,
    kernel_name: &'static str,
}

#[cfg(feature = "metal-dispatch")]
impl CandidateScoreDispatcher {
    pub fn new(registry: RegistryRef) -> Self {
        CandidateScoreDispatcher {
            registry,
            kernel_name: "candidate_score",
        }
    }

    /// Encode a candidate scoring dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        command_buffer: &CommandBufferRef,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
        page_scores_buffer: &Buffer,
        params: &ProjectionParams,
        receipt: &mut KernelReceipt,
        instrumented: bool,
    ) {
        let (pso, device) = {
            let mut reg = self.registry.lock();
            let (fcv, digest) = projection_constants(params.page_width, false, instrumented);
            let pso = reg.get_or_create(self.kernel_name, &fcv, digest);
            (pso, reg.device().clone())
        };

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pso);

        encoder.set_buffer(buffer_slot::WEIGHTS as u64, Some(weights_buffer), 0);
        encoder.set_buffer(buffer_slot::INPUT as u64, Some(input_buffer), 0);
        encoder.set_buffer(buffer_slot::PAGE_SCORES as u64, Some(page_scores_buffer), 0);

        let params_buf = device.new_buffer_with_data(
            params as *const ProjectionParams as *const std::ffi::c_void,
            std::mem::size_of::<ProjectionParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(buffer_slot::PARAMS as u64, Some(&params_buf), 0);

        let _receipt_buf_handle = instrumented.then(|| {
            let buf = device.new_buffer(
                std::mem::size_of::<KernelReceipt>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            encoder.set_buffer(buffer_slot::RECEIPT as u64, Some(&buf), 0);
            buf
        });

        // One threadgroup with thread count = number of candidate pages
        let n_pages = params.page_count.max(1);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: n_pages as u64,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();

        receipt.kernel_id = 5; // CANDIDATE_SCORE
        receipt.phase_id = 0;
        receipt.page_count = n_pages;
        receipt.threadgroups = 1;
        receipt.threads_per_threadgroup = n_pages;
        receipt.output_elements = n_pages;
        receipt.flags = if instrumented { 1 } else { 0 };
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PackVerifyDispatcher
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatches the pack-verification kernel.
///
/// Validates that a packed ternary tile640 page can be round-tripped without
/// loss.  Produces one `ErrorPartial` per page with accumulated verification
/// statistics.
///
/// Buffer layout:
///   [[buffer(0)]]  packed_weights  — tile640 u32 words to verify
///   [[buffer(7)]]  params          — ProjectionParams
///   [[buffer(8)]]  receipt         — KernelReceipt (instrumented only)
///   [[buffer(9)]]  error_partials  — ErrorPartial verification records
#[cfg(feature = "metal-dispatch")]
pub struct PackVerifyDispatcher {
    registry: RegistryRef,
    kernel_name: &'static str,
}

#[cfg(feature = "metal-dispatch")]
impl PackVerifyDispatcher {
    pub fn new(registry: RegistryRef) -> Self {
        PackVerifyDispatcher {
            registry,
            kernel_name: "pack_verify",
        }
    }

    /// Encode a pack-verification dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        command_buffer: &CommandBufferRef,
        packed_buffer: &Buffer,
        error_partials_buffer: &Buffer,
        params: &ProjectionParams,
        receipt: &mut KernelReceipt,
        instrumented: bool,
    ) {
        let (pso, device) = {
            let mut reg = self.registry.lock();
            let (fcv, digest) = projection_constants(params.page_width, false, instrumented);
            let pso = reg.get_or_create(self.kernel_name, &fcv, digest);
            (pso, reg.device().clone())
        };

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pso);

        encoder.set_buffer(buffer_slot::WEIGHTS as u64, Some(packed_buffer), 0);
        encoder.set_buffer(
            buffer_slot::ERROR_PARTIALS as u64,
            Some(error_partials_buffer),
            0,
        );

        let params_buf = device.new_buffer_with_data(
            params as *const ProjectionParams as *const std::ffi::c_void,
            std::mem::size_of::<ProjectionParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(buffer_slot::PARAMS as u64, Some(&params_buf), 0);

        let _receipt_buf_handle = instrumented.then(|| {
            let buf = device.new_buffer(
                std::mem::size_of::<KernelReceipt>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            encoder.set_buffer(buffer_slot::RECEIPT as u64, Some(&buf), 0);
            buf
        });

        // One threadgroup per page
        let n_pages = params.page_count.max(1);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: n_pages as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();

        receipt.kernel_id = 6; // PACK_VERIFY
        receipt.phase_id = 0;
        receipt.page_count = n_pages;
        receipt.threadgroups = n_pages;
        receipt.threads_per_threadgroup = 32;
        receipt.output_elements = n_pages;
        receipt.flags = if instrumented { 1 } else { 0 };
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RmsnormResidualProbeDispatcher
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatches the `rmsnorm_residual_probe` kernel.
///
/// Computes RMS normalization on hidden states and produces probe output
/// for teacher-student residual comparison.
///
/// Buffer layout:
///   [[buffer(0)]]  hidden_states   — fp16 activation input (WEIGHTS)
///   [[buffer(1)]]  rmsnorm_weights — fp16 RMS norm weights (INPUT)
///   [[buffer(6)]]  result          — fp16 normalized output (OUTPUT)
///   [[buffer(7)]]  params          — ProjectionParams
///   [[buffer(8)]]  receipt         — KernelReceipt (instrumented only)
///   [[buffer(10)]] probe_output    — AttentionProbe records
#[cfg(feature = "metal-dispatch")]
pub struct RmsnormResidualProbeDispatcher {
    registry: RegistryRef,
    kernel_name: &'static str,
}

#[cfg(feature = "metal-dispatch")]
impl RmsnormResidualProbeDispatcher {
    pub fn new(registry: RegistryRef) -> Self {
        RmsnormResidualProbeDispatcher {
            registry,
            kernel_name: "rmsnorm_residual_probe",
        }
    }

    /// Encode an rmsnorm residual probe dispatch.
    ///
    /// PAGE_SCALES (slot 2) is intentionally skipped — not used by this kernel.
    /// One threadgroup per hidden dimension with 64 threads/TG.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        command_buffer: &CommandBufferRef,
        hidden_states_buffer: &Buffer,
        rmsnorm_weights_buffer: &Buffer,
        result_buffer: &Buffer,
        probe_output_buffer: &Buffer,
        params: &ProjectionParams,
        receipt: &mut KernelReceipt,
        instrumented: bool,
    ) {
        let (pso, device) = {
            let mut reg = self.registry.lock();
            let (fcv, digest) = projection_constants(params.page_width, false, instrumented);
            let pso = reg.get_or_create(self.kernel_name, &fcv, digest);
            (pso, reg.device().clone())
        };

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pso);

        encoder.set_buffer(buffer_slot::WEIGHTS as u64, Some(hidden_states_buffer), 0);
        encoder.set_buffer(buffer_slot::INPUT as u64, Some(rmsnorm_weights_buffer), 0);
        // PAGE_SCALES (slot 2) is skipped — not used by this kernel
        encoder.set_buffer(buffer_slot::OUTPUT as u64, Some(result_buffer), 0);

        let params_buf = device.new_buffer_with_data(
            params as *const ProjectionParams as *const std::ffi::c_void,
            std::mem::size_of::<ProjectionParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(buffer_slot::PARAMS as u64, Some(&params_buf), 0);

        let _receipt_buf_handle = instrumented.then(|| {
            let buf = device.new_buffer(
                std::mem::size_of::<KernelReceipt>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            encoder.set_buffer(buffer_slot::RECEIPT as u64, Some(&buf), 0);
            buf
        });

        encoder.set_buffer(
            buffer_slot::PROBE_OUTPUT as u64,
            Some(probe_output_buffer),
            0,
        );

        // One threadgroup per hidden dimension, 64 threads/TG
        let out_dim = params.out_dim.max(1);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: out_dim as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();

        receipt.kernel_id = 7; // RMSNORM_RESIDUAL_PROBE
        receipt.phase_id = 0;
        receipt.page_count = 0;
        receipt.threadgroups = out_dim;
        receipt.threads_per_threadgroup = 64;
        receipt.output_elements = out_dim;
        receipt.flags = if instrumented { 1 } else { 0 };
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MlpActivationProbeDispatcher
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatches the `mlp_activation_probe` kernel.
///
/// Samples activation statistics across the MLP block: gate, up, and down
/// projections with an intermediate activation function.
///
/// Buffer layout:
///   [[buffer(0)]]  hidden_states   — fp16 activation input (WEIGHTS)
///   [[buffer(1)]]  gate_weights    — fp16 gate projection weights (INPUT)
///   [[buffer(2)]]  up_weights      — fp16 up projection weights (PAGE_SCALES)
///   [[buffer(3)]]  down_weights    — fp16 down projection weights (CHANNEL_SCALES)
///   [[buffer(6)]]  result          — fp16 output (OUTPUT)
///   [[buffer(7)]]  params          — ProjectionParams
///   [[buffer(8)]]  receipt         — KernelReceipt (instrumented only)
///   [[buffer(10)]] probe_output    — AttentionProbe records
#[cfg(feature = "metal-dispatch")]
pub struct MlpActivationProbeDispatcher {
    registry: RegistryRef,
    kernel_name: &'static str,
}

#[cfg(feature = "metal-dispatch")]
impl MlpActivationProbeDispatcher {
    pub fn new(registry: RegistryRef) -> Self {
        MlpActivationProbeDispatcher {
            registry,
            kernel_name: "mlp_activation_probe",
        }
    }

    /// Encode an MLP activation probe dispatch.
    /// One threadgroup per hidden dimension with 64 threads/TG.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        command_buffer: &CommandBufferRef,
        hidden_states_buffer: &Buffer,
        gate_weights_buffer: &Buffer,
        up_weights_buffer: &Buffer,
        down_weights_buffer: &Buffer,
        result_buffer: &Buffer,
        probe_output_buffer: &Buffer,
        params: &ProjectionParams,
        receipt: &mut KernelReceipt,
        instrumented: bool,
    ) {
        let (pso, device) = {
            let mut reg = self.registry.lock();
            let (fcv, digest) = projection_constants(params.page_width, false, instrumented);
            let pso = reg.get_or_create(self.kernel_name, &fcv, digest);
            (pso, reg.device().clone())
        };

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pso);

        encoder.set_buffer(buffer_slot::WEIGHTS as u64, Some(hidden_states_buffer), 0);
        encoder.set_buffer(buffer_slot::INPUT as u64, Some(gate_weights_buffer), 0);
        encoder.set_buffer(buffer_slot::PAGE_SCALES as u64, Some(up_weights_buffer), 0);
        encoder.set_buffer(
            buffer_slot::CHANNEL_SCALES as u64,
            Some(down_weights_buffer),
            0,
        );
        encoder.set_buffer(buffer_slot::OUTPUT as u64, Some(result_buffer), 0);

        let params_buf = device.new_buffer_with_data(
            params as *const ProjectionParams as *const std::ffi::c_void,
            std::mem::size_of::<ProjectionParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(buffer_slot::PARAMS as u64, Some(&params_buf), 0);

        let _receipt_buf_handle = instrumented.then(|| {
            let buf = device.new_buffer(
                std::mem::size_of::<KernelReceipt>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            encoder.set_buffer(buffer_slot::RECEIPT as u64, Some(&buf), 0);
            buf
        });

        encoder.set_buffer(
            buffer_slot::PROBE_OUTPUT as u64,
            Some(probe_output_buffer),
            0,
        );

        // One threadgroup per hidden dimension, 64 threads/TG
        let out_dim = params.out_dim.max(1);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: out_dim as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();

        receipt.kernel_id = 8; // MLP_ACTIVATION_PROBE
        receipt.phase_id = 0;
        receipt.page_count = 0;
        receipt.threadgroups = out_dim;
        receipt.threads_per_threadgroup = 64;
        receipt.output_elements = out_dim;
        receipt.flags = if instrumented { 1 } else { 0 };
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SidecarApplyVerifyDispatcher
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatches the `sidecar_apply_verify` kernel.
///
/// Verifies that sidecar outlier entries apply correctly to packed ternary
/// weights by comparing reconstructed pages against the original activations.
///
/// Buffer layout:
///   [[buffer(0)]]  packed_weights      — tile640 u32 words (WEIGHTS)
///   [[buffer(1)]]  activations         — fp16 activation vector (INPUT)
///   [[buffer(4)]]  sidecar_entries     — sparse bf16 outlier entries (SIDECAR)
///   [[buffer(5)]]  sidecar_offsets     — per-page sidecar span (SIDECAR_OFFSETS)
///   [[buffer(6)]]  result              — fp16 output (OUTPUT)
///   [[buffer(7)]]  params              — ProjectionParams
///   [[buffer(8)]]  receipt             — KernelReceipt (instrumented only)
///   [[buffer(9)]]  error_partials      — ErrorPartial verification records
#[cfg(feature = "metal-dispatch")]
pub struct SidecarApplyVerifyDispatcher {
    registry: RegistryRef,
    kernel_name: &'static str,
}

#[cfg(feature = "metal-dispatch")]
impl SidecarApplyVerifyDispatcher {
    pub fn new(registry: RegistryRef) -> Self {
        SidecarApplyVerifyDispatcher {
            registry,
            kernel_name: "sidecar_apply_verify",
        }
    }

    /// Encode a sidecar apply-verify dispatch.
    /// One threadgroup per page with 32 threads/TG.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        command_buffer: &CommandBufferRef,
        packed_weights_buffer: &Buffer,
        activations_buffer: &Buffer,
        sidecar_entries_buffer: Option<&Buffer>,
        sidecar_offsets_buffer: Option<&Buffer>,
        result_buffer: &Buffer,
        error_partials_buffer: &Buffer,
        params: &ProjectionParams,
        receipt: &mut KernelReceipt,
        instrumented: bool,
    ) {
        let (pso, device) = {
            let mut reg = self.registry.lock();
            let (fcv, digest) = projection_constants(
                params.page_width,
                sidecar_entries_buffer.is_some(),
                instrumented,
            );
            let pso = reg.get_or_create(self.kernel_name, &fcv, digest);
            (pso, reg.device().clone())
        };

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pso);

        encoder.set_buffer(buffer_slot::WEIGHTS as u64, Some(packed_weights_buffer), 0);
        encoder.set_buffer(buffer_slot::INPUT as u64, Some(activations_buffer), 0);

        if let Some(sidecar) = sidecar_entries_buffer {
            encoder.set_buffer(buffer_slot::SIDECAR as u64, Some(sidecar), 0);
        }
        if let Some(offsets) = sidecar_offsets_buffer {
            encoder.set_buffer(buffer_slot::SIDECAR_OFFSETS as u64, Some(offsets), 0);
        }

        encoder.set_buffer(buffer_slot::OUTPUT as u64, Some(result_buffer), 0);

        let params_buf = device.new_buffer_with_data(
            params as *const ProjectionParams as *const std::ffi::c_void,
            std::mem::size_of::<ProjectionParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(buffer_slot::PARAMS as u64, Some(&params_buf), 0);

        let _receipt_buf_handle = instrumented.then(|| {
            let buf = device.new_buffer(
                std::mem::size_of::<KernelReceipt>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            encoder.set_buffer(buffer_slot::RECEIPT as u64, Some(&buf), 0);
            buf
        });

        encoder.set_buffer(
            buffer_slot::ERROR_PARTIALS as u64,
            Some(error_partials_buffer),
            0,
        );

        // One threadgroup per page, 32 threads/TG
        let n_pages = params.page_count.max(1);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: n_pages as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();

        receipt.kernel_id = 9; // SIDECAR_APPLY_VERIFY
        receipt.phase_id = 0;
        receipt.page_count = n_pages;
        receipt.threadgroups = n_pages;
        receipt.threads_per_threadgroup = 32;
        receipt.output_elements = n_pages;
        receipt.flags = if instrumented { 1 } else { 0 };
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FusedRmsnormQkvDispatcher
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatches the `fused_rmsnorm_qkv` kernel.
///
/// Fused RMS normalization, QKV projection, and RoPE into a single kernel
/// pass, reading/writing through indirection into the activation arena.
///
/// Buffer layout:
///   [[buffer(0)]]  hidden_states       — fp16 activation input (WEIGHTS)
///   [[buffer(1)]]  qkv_weights         — fp16 QKV projection (INPUT)
///   [[buffer(2)]]  rmsnorm_weights     — fp16 RMS norm weights (PAGE_SCALES)
///   [[buffer(6)]]  output              — fp16 QKV result (OUTPUT)
///   [[buffer(7)]]  params              — ProjectionParams
///   [[buffer(8)]]  receipt             — KernelReceipt (instrumented only)
///   [[buffer(12)]] activation_arena    — Arena buffer (ACTIVATION_ARENA)
///   [[buffer(13)]] arena_descriptors   — Arena descriptor table (ARENA_DESCRIPTORS)
#[cfg(feature = "metal-dispatch")]
pub struct FusedRmsnormQkvDispatcher {
    registry: RegistryRef,
    kernel_name: &'static str,
}

#[cfg(feature = "metal-dispatch")]
impl FusedRmsnormQkvDispatcher {
    pub fn new(registry: RegistryRef) -> Self {
        FusedRmsnormQkvDispatcher {
            registry,
            kernel_name: "fused_rmsnorm_qkv",
        }
    }

    /// Encode a fused rmsnorm+QKV dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        command_buffer: &CommandBufferRef,
        hidden_states_buffer: &Buffer,
        qkv_weights_buffer: &Buffer,
        rmsnorm_weights_buffer: &Buffer,
        output_buffer: &Buffer,
        activation_arena_buffer: &Buffer,
        arena_descriptors_buffer: &Buffer,
        params: &ProjectionParams,
        receipt: &mut KernelReceipt,
        instrumented: bool,
    ) {
        let (pso, device) = {
            let mut reg = self.registry.lock();
            let (fcv, digest) = projection_constants(params.page_width, false, instrumented);
            let pso = reg.get_or_create(self.kernel_name, &fcv, digest);
            (pso, reg.device().clone())
        };

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pso);

        encoder.set_buffer(buffer_slot::WEIGHTS as u64, Some(hidden_states_buffer), 0);
        encoder.set_buffer(buffer_slot::INPUT as u64, Some(qkv_weights_buffer), 0);
        encoder.set_buffer(
            buffer_slot::PAGE_SCALES as u64,
            Some(rmsnorm_weights_buffer),
            0,
        );
        encoder.set_buffer(buffer_slot::OUTPUT as u64, Some(output_buffer), 0);

        let params_buf = device.new_buffer_with_data(
            params as *const ProjectionParams as *const std::ffi::c_void,
            std::mem::size_of::<ProjectionParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(buffer_slot::PARAMS as u64, Some(&params_buf), 0);

        let _receipt_buf_handle = instrumented.then(|| {
            let buf = device.new_buffer(
                std::mem::size_of::<KernelReceipt>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            encoder.set_buffer(buffer_slot::RECEIPT as u64, Some(&buf), 0);
            buf
        });

        encoder.set_buffer(
            buffer_slot::ACTIVATION_ARENA as u64,
            Some(activation_arena_buffer),
            0,
        );
        encoder.set_buffer(
            buffer_slot::ARENA_DESCRIPTORS as u64,
            Some(arena_descriptors_buffer),
            0,
        );

        // One threadgroup per output element, 64 threads/TG
        let out_dim = params.out_dim.max(1);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: out_dim as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();

        receipt.kernel_id = 10; // FUSED_RMSNORM_QKV
        receipt.phase_id = 0;
        receipt.page_count = params.page_count;
        receipt.threadgroups = out_dim;
        receipt.threads_per_threadgroup = 64;
        receipt.output_elements = out_dim;
        receipt.flags = if instrumented { 1 } else { 0 };
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FusedOProjResidualDispatcher
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatches the `fused_o_proj_residual` kernel.
///
/// Fused output projection with residual add, reading/writing through
/// indirection into the activation arena.
///
/// Buffer layout:
///   [[buffer(0)]]  o_proj_weights      — fp16 output projection weights (WEIGHTS)
///   [[buffer(1)]]  hidden_states       — fp16 activation input (INPUT)
///   [[buffer(2)]]  residual            — fp16 residual to add (PAGE_SCALES)
///   [[buffer(6)]]  output              — fp16 result (OUTPUT)
///   [[buffer(7)]]  params              — ProjectionParams
///   [[buffer(8)]]  receipt             — KernelReceipt (instrumented only)
///   [[buffer(12)]] activation_arena    — Arena buffer (ACTIVATION_ARENA)
///   [[buffer(13)]] arena_descriptors   — Arena descriptor table (ARENA_DESCRIPTORS)
#[cfg(feature = "metal-dispatch")]
pub struct FusedOProjResidualDispatcher {
    registry: RegistryRef,
    kernel_name: &'static str,
}

#[cfg(feature = "metal-dispatch")]
impl FusedOProjResidualDispatcher {
    pub fn new(registry: RegistryRef) -> Self {
        FusedOProjResidualDispatcher {
            registry,
            kernel_name: "fused_o_proj_residual",
        }
    }

    /// Encode a fused O-proj + residual dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        command_buffer: &CommandBufferRef,
        o_proj_weights_buffer: &Buffer,
        hidden_states_buffer: &Buffer,
        residual_buffer: &Buffer,
        output_buffer: &Buffer,
        activation_arena_buffer: &Buffer,
        arena_descriptors_buffer: &Buffer,
        params: &ProjectionParams,
        receipt: &mut KernelReceipt,
        instrumented: bool,
    ) {
        let (pso, device) = {
            let mut reg = self.registry.lock();
            let (fcv, digest) = projection_constants(params.page_width, false, instrumented);
            let pso = reg.get_or_create(self.kernel_name, &fcv, digest);
            (pso, reg.device().clone())
        };

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pso);

        encoder.set_buffer(buffer_slot::WEIGHTS as u64, Some(o_proj_weights_buffer), 0);
        encoder.set_buffer(buffer_slot::INPUT as u64, Some(hidden_states_buffer), 0);
        encoder.set_buffer(buffer_slot::PAGE_SCALES as u64, Some(residual_buffer), 0);
        encoder.set_buffer(buffer_slot::OUTPUT as u64, Some(output_buffer), 0);

        let params_buf = device.new_buffer_with_data(
            params as *const ProjectionParams as *const std::ffi::c_void,
            std::mem::size_of::<ProjectionParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(buffer_slot::PARAMS as u64, Some(&params_buf), 0);

        let _receipt_buf_handle = instrumented.then(|| {
            let buf = device.new_buffer(
                std::mem::size_of::<KernelReceipt>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            encoder.set_buffer(buffer_slot::RECEIPT as u64, Some(&buf), 0);
            buf
        });

        encoder.set_buffer(
            buffer_slot::ACTIVATION_ARENA as u64,
            Some(activation_arena_buffer),
            0,
        );
        encoder.set_buffer(
            buffer_slot::ARENA_DESCRIPTORS as u64,
            Some(arena_descriptors_buffer),
            0,
        );

        // One threadgroup per output element, 64 threads/TG
        let out_dim = params.out_dim.max(1);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: out_dim as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();

        receipt.kernel_id = 11; // FUSED_O_PROJ_RESIDUAL
        receipt.phase_id = 0;
        receipt.page_count = params.page_count;
        receipt.threadgroups = out_dim;
        receipt.threads_per_threadgroup = 64;
        receipt.output_elements = out_dim;
        receipt.flags = if instrumented { 1 } else { 0 };
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FusedMultimodalDispatcher
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatches the `fused_multimodal` kernel.
///
/// Fused multimodal projection that processes vision/text embeddings through
/// a projection block with activation arena indirection.
///
/// Buffer layout:
///   [[buffer(0)]]  multimodal_weights  — fp16 multimodal projection weights (WEIGHTS)
///   [[buffer(1)]]  multimodal_input    — fp16 input embeddings (INPUT)
///   [[buffer(6)]]  output              — fp16 result (OUTPUT)
///   [[buffer(7)]]  params              — ProjectionParams
///   [[buffer(8)]]  receipt             — KernelReceipt (instrumented only)
///   [[buffer(12)]] activation_arena    — Arena buffer (ACTIVATION_ARENA)
///   [[buffer(13)]] arena_descriptors   — Arena descriptor table (ARENA_DESCRIPTORS)
#[cfg(feature = "metal-dispatch")]
pub struct FusedMultimodalDispatcher {
    registry: RegistryRef,
    kernel_name: &'static str,
}

#[cfg(feature = "metal-dispatch")]
impl FusedMultimodalDispatcher {
    pub fn new(registry: RegistryRef) -> Self {
        FusedMultimodalDispatcher {
            registry,
            kernel_name: "fused_multimodal",
        }
    }

    /// Encode a fused multimodal dispatch.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        command_buffer: &CommandBufferRef,
        multimodal_weights_buffer: &Buffer,
        multimodal_input_buffer: &Buffer,
        output_buffer: &Buffer,
        activation_arena_buffer: &Buffer,
        arena_descriptors_buffer: &Buffer,
        params: &ProjectionParams,
        receipt: &mut KernelReceipt,
        instrumented: bool,
    ) {
        let (pso, device) = {
            let mut reg = self.registry.lock();
            let (fcv, digest) = projection_constants(params.page_width, false, instrumented);
            let pso = reg.get_or_create(self.kernel_name, &fcv, digest);
            (pso, reg.device().clone())
        };

        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pso);

        encoder.set_buffer(
            buffer_slot::WEIGHTS as u64,
            Some(multimodal_weights_buffer),
            0,
        );
        encoder.set_buffer(buffer_slot::INPUT as u64, Some(multimodal_input_buffer), 0);
        encoder.set_buffer(buffer_slot::OUTPUT as u64, Some(output_buffer), 0);

        let params_buf = device.new_buffer_with_data(
            params as *const ProjectionParams as *const std::ffi::c_void,
            std::mem::size_of::<ProjectionParams>() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        encoder.set_buffer(buffer_slot::PARAMS as u64, Some(&params_buf), 0);

        let _receipt_buf_handle = instrumented.then(|| {
            let buf = device.new_buffer(
                std::mem::size_of::<KernelReceipt>() as u64,
                MTLResourceOptions::StorageModeShared,
            );
            encoder.set_buffer(buffer_slot::RECEIPT as u64, Some(&buf), 0);
            buf
        });

        encoder.set_buffer(
            buffer_slot::ACTIVATION_ARENA as u64,
            Some(activation_arena_buffer),
            0,
        );
        encoder.set_buffer(
            buffer_slot::ARENA_DESCRIPTORS as u64,
            Some(arena_descriptors_buffer),
            0,
        );

        // One threadgroup per output element, 64 threads/TG
        let out_dim = params.out_dim.max(1);
        encoder.dispatch_thread_groups(
            MTLSize {
                width: out_dim as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 64,
                height: 1,
                depth: 1,
            },
        );
        encoder.end_encoding();

        receipt.kernel_id = 12; // FUSED_MULTIMODAL
        receipt.phase_id = 0;
        receipt.page_count = params.page_count;
        receipt.threadgroups = out_dim;
        receipt.threads_per_threadgroup = 64;
        receipt.output_elements = out_dim;
        receipt.flags = if instrumented { 1 } else { 0 };
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Convenience constructor
// ═══════════════════════════════════════════════════════════════════════════

/// Create a shared `KernelRegistry` and all 12 dispatchers from a Metal device.
#[cfg(feature = "metal-dispatch")]
pub fn create_dispatchers(
    device: &Device,
) -> (
    RegistryRef,
    TernaryProjectionDispatcher,
    DenseProjectionDispatcher,
    ErrorPartialDispatcher,
    ProbeDispatcher,
    CandidateScoreDispatcher,
    PackVerifyDispatcher,
    RmsnormResidualProbeDispatcher,
    MlpActivationProbeDispatcher,
    SidecarApplyVerifyDispatcher,
    FusedRmsnormQkvDispatcher,
    FusedOProjResidualDispatcher,
    FusedMultimodalDispatcher,
) {
    let registry = Arc::new(Mutex::new(KernelRegistry::new(device)));
    (
        registry.clone(),
        TernaryProjectionDispatcher::new(registry.clone()),
        DenseProjectionDispatcher::new(registry.clone()),
        ErrorPartialDispatcher::new(registry.clone()),
        ProbeDispatcher::new(registry.clone()),
        CandidateScoreDispatcher::new(registry.clone()),
        PackVerifyDispatcher::new(registry.clone()),
        RmsnormResidualProbeDispatcher::new(registry.clone()),
        MlpActivationProbeDispatcher::new(registry.clone()),
        SidecarApplyVerifyDispatcher::new(registry.clone()),
        FusedRmsnormQkvDispatcher::new(registry.clone()),
        FusedOProjResidualDispatcher::new(registry.clone()),
        FusedMultimodalDispatcher::new(registry),
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// Free-function convenience wrappers
// ═══════════════════════════════════════════════════════════════════════════

/// Convenience wrapper: creates a dispatcher, encodes the ternary projection
/// kernel, commits, waits, and returns GPU duration in nanoseconds.
#[cfg(feature = "metal-dispatch")]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_ternary_projection(
    device: &Device,
    weights_buffer: &Buffer,
    input_buffer: &Buffer,
    page_scales_buffer: &Buffer,
    channel_scales_buffer: &Buffer,
    sidecar_buffer: Option<&Buffer>,
    sidecar_offsets_buffer: Option<&Buffer>,
    output_buffer: &Buffer,
    params: &ProjectionParams,
    receipt: &mut KernelReceipt,
    instrumented: bool,
) -> u64 {
    let cmd_queue = device.new_command_queue();
    let cmd_buf = cmd_queue.new_command_buffer();
    let registry = Arc::new(Mutex::new(KernelRegistry::new(device)));
    let dispatcher = TernaryProjectionDispatcher::new(registry);
    dispatcher.dispatch(
        &cmd_buf,
        weights_buffer,
        input_buffer,
        page_scales_buffer,
        channel_scales_buffer,
        sidecar_buffer,
        sidecar_offsets_buffer,
        output_buffer,
        params,
        receipt,
        instrumented,
    );
    let start = Instant::now();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();
    start.elapsed().as_nanos() as u64
}

/// Convenience wrapper: dense projection f16 dispatch.
#[cfg(feature = "metal-dispatch")]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_dense_projection_f16(
    device: &Device,
    codebook_buffer: &Buffer,
    indices_buffer: &Buffer,
    input_buffer: &Buffer,
    output_buffer: &Buffer,
    params: &ProjectionParams,
    receipt: &mut KernelReceipt,
    instrumented: bool,
) -> u64 {
    let cmd_queue = device.new_command_queue();
    let cmd_buf = cmd_queue.new_command_buffer();
    let registry = Arc::new(Mutex::new(KernelRegistry::new(device)));
    let dispatcher = DenseProjectionDispatcher::new(registry);
    dispatcher.dispatch(
        &cmd_buf,
        codebook_buffer,
        indices_buffer,
        input_buffer,
        output_buffer,
        params,
        receipt,
        instrumented,
    );
    let start = Instant::now();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();
    start.elapsed().as_nanos() as u64
}

/// Convenience wrapper: rmsnorm residual probe dispatch.
#[cfg(feature = "metal-dispatch")]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_rmsnorm_residual_probe(
    device: &Device,
    hidden_states_buffer: &Buffer,
    rmsnorm_weights_buffer: &Buffer,
    result_buffer: &Buffer,
    probe_output_buffer: &Buffer,
    params: &ProjectionParams,
    receipt: &mut KernelReceipt,
    instrumented: bool,
) -> u64 {
    let cmd_queue = device.new_command_queue();
    let cmd_buf = cmd_queue.new_command_buffer();
    let registry = Arc::new(Mutex::new(KernelRegistry::new(device)));
    let dispatcher = RmsnormResidualProbeDispatcher::new(registry);
    dispatcher.dispatch(
        &cmd_buf,
        hidden_states_buffer,
        rmsnorm_weights_buffer,
        result_buffer,
        probe_output_buffer,
        params,
        receipt,
        instrumented,
    );
    let start = Instant::now();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();
    start.elapsed().as_nanos() as u64
}

/// Convenience wrapper: attention score probe dispatch.
#[cfg(feature = "metal-dispatch")]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_attention_score_probe(
    device: &Device,
    teacher_attn_buffer: &Buffer,
    student_attn_buffer: &Buffer,
    probe_output_buffer: &Buffer,
    params: &ProjectionParams,
    receipt: &mut KernelReceipt,
    instrumented: bool,
) -> u64 {
    let cmd_queue = device.new_command_queue();
    let cmd_buf = cmd_queue.new_command_buffer();
    let registry = Arc::new(Mutex::new(KernelRegistry::new(device)));
    let dispatcher = ProbeDispatcher::new(registry);
    dispatcher.dispatch(
        &cmd_buf,
        teacher_attn_buffer,
        student_attn_buffer,
        probe_output_buffer,
        params,
        receipt,
        instrumented,
    );
    let start = Instant::now();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();
    start.elapsed().as_nanos() as u64
}

/// Convenience wrapper: MLP activation probe dispatch.
#[cfg(feature = "metal-dispatch")]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_mlp_activation_probe(
    device: &Device,
    hidden_states_buffer: &Buffer,
    gate_weights_buffer: &Buffer,
    up_weights_buffer: &Buffer,
    down_weights_buffer: &Buffer,
    result_buffer: &Buffer,
    probe_output_buffer: &Buffer,
    params: &ProjectionParams,
    receipt: &mut KernelReceipt,
    instrumented: bool,
) -> u64 {
    let cmd_queue = device.new_command_queue();
    let cmd_buf = cmd_queue.new_command_buffer();
    let registry = Arc::new(Mutex::new(KernelRegistry::new(device)));
    let dispatcher = MlpActivationProbeDispatcher::new(registry);
    dispatcher.dispatch(
        &cmd_buf,
        hidden_states_buffer,
        gate_weights_buffer,
        up_weights_buffer,
        down_weights_buffer,
        result_buffer,
        probe_output_buffer,
        params,
        receipt,
        instrumented,
    );
    let start = Instant::now();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();
    start.elapsed().as_nanos() as u64
}

/// Convenience wrapper: activation error partial reduce dispatch.
#[cfg(feature = "metal-dispatch")]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_activation_error_partial_reduce(
    device: &Device,
    teacher_buffer: &Buffer,
    student_buffer: &Buffer,
    error_partials_buffer: &Buffer,
    params: &ProjectionParams,
    receipt: &mut KernelReceipt,
    instrumented: bool,
) -> u64 {
    let cmd_queue = device.new_command_queue();
    let cmd_buf = cmd_queue.new_command_buffer();
    let registry = Arc::new(Mutex::new(KernelRegistry::new(device)));
    let dispatcher = ErrorPartialDispatcher::new(registry);
    dispatcher.dispatch(
        &cmd_buf,
        teacher_buffer,
        student_buffer,
        error_partials_buffer,
        params,
        receipt,
        instrumented,
    );
    let start = Instant::now();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();
    start.elapsed().as_nanos() as u64
}

/// Convenience wrapper: page candidate score dispatch.
#[cfg(feature = "metal-dispatch")]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_page_candidate_score(
    device: &Device,
    weights_buffer: &Buffer,
    input_buffer: &Buffer,
    page_scores_buffer: &Buffer,
    params: &ProjectionParams,
    receipt: &mut KernelReceipt,
    instrumented: bool,
) -> u64 {
    let cmd_queue = device.new_command_queue();
    let cmd_buf = cmd_queue.new_command_buffer();
    let registry = Arc::new(Mutex::new(KernelRegistry::new(device)));
    let dispatcher = CandidateScoreDispatcher::new(registry);
    dispatcher.dispatch(
        &cmd_buf,
        weights_buffer,
        input_buffer,
        page_scores_buffer,
        params,
        receipt,
        instrumented,
    );
    let start = Instant::now();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();
    start.elapsed().as_nanos() as u64
}

/// Convenience wrapper: page unpack verify dispatch.
#[cfg(feature = "metal-dispatch")]
pub fn dispatch_page_unpack_verify(
    device: &Device,
    packed_buffer: &Buffer,
    error_partials_buffer: &Buffer,
    params: &ProjectionParams,
    receipt: &mut KernelReceipt,
    instrumented: bool,
) -> u64 {
    let cmd_queue = device.new_command_queue();
    let cmd_buf = cmd_queue.new_command_buffer();
    let registry = Arc::new(Mutex::new(KernelRegistry::new(device)));
    let dispatcher = PackVerifyDispatcher::new(registry);
    dispatcher.dispatch(
        &cmd_buf,
        packed_buffer,
        error_partials_buffer,
        params,
        receipt,
        instrumented,
    );
    let start = Instant::now();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();
    start.elapsed().as_nanos() as u64
}

/// Convenience wrapper: sidecar apply verify dispatch.
#[cfg(feature = "metal-dispatch")]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_sidecar_apply_verify(
    device: &Device,
    packed_weights_buffer: &Buffer,
    activations_buffer: &Buffer,
    sidecar_entries_buffer: Option<&Buffer>,
    sidecar_offsets_buffer: Option<&Buffer>,
    result_buffer: &Buffer,
    error_partials_buffer: &Buffer,
    params: &ProjectionParams,
    receipt: &mut KernelReceipt,
    instrumented: bool,
) -> u64 {
    let cmd_queue = device.new_command_queue();
    let cmd_buf = cmd_queue.new_command_buffer();
    let registry = Arc::new(Mutex::new(KernelRegistry::new(device)));
    let dispatcher = SidecarApplyVerifyDispatcher::new(registry);
    dispatcher.dispatch(
        &cmd_buf,
        packed_weights_buffer,
        activations_buffer,
        sidecar_entries_buffer,
        sidecar_offsets_buffer,
        result_buffer,
        error_partials_buffer,
        params,
        receipt,
        instrumented,
    );
    let start = Instant::now();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();
    start.elapsed().as_nanos() as u64
}

/// Convenience wrapper: fused rmsnorm + QKV dispatch.
#[cfg(feature = "metal-dispatch")]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fused_rmsnorm_qkv(
    device: &Device,
    hidden_states_buffer: &Buffer,
    qkv_weights_buffer: &Buffer,
    rmsnorm_weights_buffer: &Buffer,
    output_buffer: &Buffer,
    activation_arena_buffer: &Buffer,
    arena_descriptors_buffer: &Buffer,
    params: &ProjectionParams,
    receipt: &mut KernelReceipt,
    instrumented: bool,
) -> u64 {
    let cmd_queue = device.new_command_queue();
    let cmd_buf = cmd_queue.new_command_buffer();
    let registry = Arc::new(Mutex::new(KernelRegistry::new(device)));
    let dispatcher = FusedRmsnormQkvDispatcher::new(registry);
    dispatcher.dispatch(
        &cmd_buf,
        hidden_states_buffer,
        qkv_weights_buffer,
        rmsnorm_weights_buffer,
        output_buffer,
        activation_arena_buffer,
        arena_descriptors_buffer,
        params,
        receipt,
        instrumented,
    );
    let start = Instant::now();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();
    start.elapsed().as_nanos() as u64
}

/// Convenience wrapper: fused O-proj + residual dispatch.
#[cfg(feature = "metal-dispatch")]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fused_o_proj_residual(
    device: &Device,
    o_proj_weights_buffer: &Buffer,
    hidden_states_buffer: &Buffer,
    residual_buffer: &Buffer,
    output_buffer: &Buffer,
    activation_arena_buffer: &Buffer,
    arena_descriptors_buffer: &Buffer,
    params: &ProjectionParams,
    receipt: &mut KernelReceipt,
    instrumented: bool,
) -> u64 {
    let cmd_queue = device.new_command_queue();
    let cmd_buf = cmd_queue.new_command_buffer();
    let registry = Arc::new(Mutex::new(KernelRegistry::new(device)));
    let dispatcher = FusedOProjResidualDispatcher::new(registry);
    dispatcher.dispatch(
        &cmd_buf,
        o_proj_weights_buffer,
        hidden_states_buffer,
        residual_buffer,
        output_buffer,
        activation_arena_buffer,
        arena_descriptors_buffer,
        params,
        receipt,
        instrumented,
    );
    let start = Instant::now();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();
    start.elapsed().as_nanos() as u64
}

/// Convenience wrapper: fused multimodal dispatch.
#[cfg(feature = "metal-dispatch")]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_fused_multimodal(
    device: &Device,
    multimodal_weights_buffer: &Buffer,
    multimodal_input_buffer: &Buffer,
    output_buffer: &Buffer,
    activation_arena_buffer: &Buffer,
    arena_descriptors_buffer: &Buffer,
    params: &ProjectionParams,
    receipt: &mut KernelReceipt,
    instrumented: bool,
) -> u64 {
    let cmd_queue = device.new_command_queue();
    let cmd_buf = cmd_queue.new_command_buffer();
    let registry = Arc::new(Mutex::new(KernelRegistry::new(device)));
    let dispatcher = FusedMultimodalDispatcher::new(registry);
    dispatcher.dispatch(
        &cmd_buf,
        multimodal_weights_buffer,
        multimodal_input_buffer,
        output_buffer,
        activation_arena_buffer,
        arena_descriptors_buffer,
        params,
        receipt,
        instrumented,
    );
    let start = Instant::now();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();
    start.elapsed().as_nanos() as u64
}

// ═══════════════════════════════════════════════════════════════════════════
// NF4Tile640 live-forward execution smoke test (Mac / Metal only)
// ═══════════════════════════════════════════════════════════════════════════
//
// Proves the teacher single-GEMV forward actually EXECUTES end-to-end on the GPU
// and matches a CPU reference of the same quantized weights: pack a matrix into
// the interleaved NF4Tile640 arena → build Metal buffers → run
// Nf4Tile640ProjectionDispatcher → read back → compare. Requires a Metal device
// and `nf4_tile640_gemv.metal` compiled into the metallib (see build.rs).
// Run on macOS: `cargo test -p tribunus-compute-core --features prism-backend
// nf4_forward_exec`.
//
// AUTHORED, NOT COMPILED HERE (no Metal toolchain in the dev sandbox). The
// layout + arithmetic mirror tools/nf4_forward_ref.rs, which IS Linux-verified.
#[cfg(all(test, feature = "metal-dispatch"))]
mod nf4_forward_exec_tests {
    use super::*;

    const NF4: [f32; 16] = [
        -1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734, -0.09105, 0.0, 0.0795803,
        0.1609302, 0.2461123, 0.3379152, 0.4407099, 0.562617, 0.7229568, 1.0,
    ];
    const TILE: usize = 640;
    const GROUP: usize = 128;
    const GPT: usize = 5;
    const LANES: usize = 32;
    const VPL: usize = 4;
    const BYTES_TILE: usize = 320;
    const BYTES_GROUP: usize = 64;

    fn nearest(v: f32) -> u8 {
        let mut b = 0u8;
        let mut bd = (v - NF4[0]).abs();
        for (i, &l) in NF4.iter().enumerate().skip(1) {
            let d = (v - l).abs();
            if d < bd {
                bd = d;
                b = i as u8;
            }
        }
        b
    }

    #[inline]
    fn tiles_for(cols: usize) -> usize {
        cols.div_ceil(TILE)
    }

    /// Row-major read with zero-pad past the real width — mirrors the packer's
    /// partial-tile contract (col >= in_dim → 0.0).
    #[inline]
    fn wval(w: &[f32], r: usize, cols: usize, col: usize) -> f32 {
        if col < cols {
            w[r * cols + col]
        } else {
            0.0
        }
    }

    // `cols` (== in_dim) need NOT be a multiple of 640: the last tile is
    // zero-padded, exactly as `quantize_nf4_tile640_matrix_from_raw` does.
    fn pack(w: &[f32], rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
        let tiles = tiles_for(cols);
        let mut packed = vec![0u8; rows * tiles * BYTES_TILE];
        let mut scales = vec![0f32; rows * tiles * GPT];
        for r in 0..rows {
            for t in 0..tiles {
                for g in 0..GPT {
                    let mut absmax = 0f32;
                    for gl in 0..GROUP {
                        let col = t * TILE + g * GROUP + gl;
                        absmax = absmax.max(wval(w, r, cols, col).abs());
                    }
                    let scale = if absmax > 1e-12 { absmax } else { 1.0 };
                    scales[r * tiles * GPT + t * GPT + g] = scale;
                    let inv = 1.0 / scale;
                    for lane in 0..LANES {
                        for i in 0..VPL {
                            let col = t * TILE + g * GROUP + lane * VPL + i;
                            let idx = nearest((wval(w, r, cols, col) * inv).clamp(-1.0, 1.0));
                            let byte = r * tiles * BYTES_TILE
                                + t * BYTES_TILE
                                + g * BYTES_GROUP
                                + lane * 2
                                + (i / 2);
                            packed[byte] |= idx << ((i % 2) * 4);
                        }
                    }
                }
            }
        }
        (packed, scales, vec![0f32; rows * tiles * GPT])
    }

    fn cpu_gemv(packed: &[u8], scales: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let tiles = tiles_for(cols);
        let mut y = vec![0f32; rows];
        for r in 0..rows {
            let mut acc = 0f32;
            for col in 0..cols {
                let (t, wt) = (col / TILE, col % TILE);
                let (g, gl) = (wt / GROUP, wt % GROUP);
                let (lane, i) = (gl / VPL, gl % VPL);
                let byte = packed[r * tiles * BYTES_TILE
                    + t * BYTES_TILE
                    + g * BYTES_GROUP
                    + lane * 2
                    + (i / 2)];
                let idx = if i % 2 == 0 {
                    byte & 0x0F
                } else {
                    (byte >> 4) & 0x0F
                };
                acc += NF4[idx as usize] * scales[r * tiles * GPT + t * GPT + g] * x[col];
            }
            y[r] = acc;
        }
        y
    }

    fn run_case(device: &Device, registry: RegistryRef, rows: usize, cols: usize) {
        let tiles = tiles_for(cols);
        let w: Vec<f32> = (0..rows * cols)
            .map(|k| ((k as f32) * 0.017).sin() * 0.05)
            .collect();
        let x: Vec<f32> = (0..cols).map(|k| ((k as f32) * 0.011).cos()).collect();
        let (packed, scales, biases) = pack(&w, rows, cols);
        let cpu = cpu_gemv(&packed, &scales, &x, rows, cols);

        let mk = |bytes: &[u8]| {
            device.new_buffer_with_data(
                bytes.as_ptr() as *const std::ffi::c_void,
                bytes.len() as u64,
                MTLResourceOptions::StorageModeShared,
            )
        };
        let as_bytes = |f: &[f32]| -> Vec<u8> { f.iter().flat_map(|v| v.to_ne_bytes()).collect() };
        let pk = mk(&packed);
        let sc = mk(&as_bytes(&scales));
        let bi = mk(&as_bytes(&biases));
        let inb = mk(&as_bytes(&x));
        let out = device.new_buffer((rows * 4) as u64, MTLResourceOptions::StorageModeShared);

        let params = ProjectionParams {
            in_dim: cols as u32, // real (unpadded) width → kernel buffer(6)
            out_dim: rows as u32,
            page_count: tiles as u32, // ceil(in_dim/640) → num_macro_tiles
            page_width: TILE as u32,
            mode_flags: 0,
            probe_seed: 0,
            reserved: [0; 5],
        };
        let mut receipt: KernelReceipt = unsafe { std::mem::zeroed() };

        let queue = device.new_command_queue();
        let cb = queue.new_command_buffer();
        let disp = Nf4Tile640ProjectionDispatcher::new(registry);
        disp.dispatch(cb, &pk, &sc, &bi, &inb, &out, &params, &mut receipt);
        cb.commit();
        cb.wait_until_completed();

        let gpu = unsafe { std::slice::from_raw_parts(out.contents() as *const f32, rows) };
        let mut maxd = 0f32;
        for r in 0..rows {
            maxd = maxd.max((gpu[r] - cpu[r]).abs());
        }
        assert!(
            maxd < 1e-3,
            "GPU vs CPU NF4Tile640 GEMV mismatch (rows={rows}, in_dim={cols}, tiles={tiles}): max abs err {maxd}"
        );
    }

    #[test]
    fn nf4_forward_exec_matches_cpu() {
        let device = match Device::system_default() {
            Some(d) => d,
            None => {
                eprintln!("no Metal device — skipping NF4 forward execution test");
                return;
            }
        };
        let registry: RegistryRef = Arc::new(Mutex::new(KernelRegistry::new(&device)));

        // Exact multiple of 640, plus two partial in_dims that exercise the
        // kernel's `if col >= in_dim continue` guard (buffer 6). The GPU must
        // match the CPU reference, which only sums the real columns.
        for &(rows, cols) in &[(32usize, 640usize), (24, 650), (16, 1290)] {
            run_case(&device, Arc::clone(&registry), rows, cols);
        }
    }
}
