//! Level 1 dense teacher forward executor via Accelerate vDSP.
//!
//! Loads teacher weights, runs a dense FP16 GEMV forward pass — tries a real
//! Metal kernel dispatch via `DenseProjectionDispatcher` when the compiled
//! metallib is available, falling back to CPU Accelerate vDSP (AMX coprocessor).
//! Stateless: receives a microbatch index and produces activations into an
//! internal output buffer that the scheduler and reducer access for
//! deterministic CPU-side comparisons.

use crate::calibration::accelerate::dot_product;
#[cfg(feature = "metal-dispatch")]
use crate::compute_image::compile::kernel_dispatch::{
    create_dispatchers, DenseProjectionDispatcher, RegistryRef,
};
#[cfg(feature = "metal-dispatch")]
use crate::compute_image::compile::kernel_types::{KernelReceipt, ProjectionParams};
#[cfg(feature = "metal-dispatch")]
use half;
#[cfg(feature = "metal-dispatch")]
use metal::*;

/// Default hidden dimension used for the teacher weight matrix.
const HIDDEN_DIM: usize = 3840;

/// Simulated FP16 teacher weights and output buffer.
struct WeightStore {
    /// Flat weight matrix: out_dim × in_dim, row-major.
    data: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
}

impl WeightStore {
    fn new(in_dim: usize, out_dim: usize) -> Self {
        let n = in_dim * out_dim;
        let mut data = vec![0.0f32; n];
        // Seed with a deterministic sinusoidal pattern (synthetic proxy for
        // real teacher weights loaded from a checkpoint).  The amplitude is
        // low (0.01) so the dot-product output is in a reasonable FP16 range.
        for i in 0..n {
            data[i] = ((i as f64).sin() * 0.01) as f32;
        }
        WeightStore {
            data,
            in_dim,
            out_dim,
        }
    }
}

/// The dense teacher forward executor.
///
/// Wraps the Accelerate/vDSP-based dense GEMV that runs the teacher model on
/// a microbatch of tokens.  Stateless — each call receives a microbatch index
/// and produces output activations into the internal output buffer.
/// ⚠ SYNTHETIC PROXY — NOT the real model. Weights are a deterministic
/// sinusoid (`WeightStore::new`) and the Metal path uploads a compact
/// approximation/codebook, not a dense checkpoint matrix. It exists to
/// exercise scheduling, memory ceilings, and reducer plumbing at realistic
/// SHAPES. Any accuracy/MSE/cosine number measured against it validates the
/// pipeline's mechanics, not model quality — do not interpret Level 1/2 gate
/// metrics as model fidelity while this proxy is the teacher.
/// The checkpoint-backed teacher is [`Gemma4Teacher`] (below); today it feeds
/// the model-level KD gate (`level1::kd_gate` via `distill_worker`), and the
/// per-layer gate graduation is blocked on per-layer real-teacher activations
/// (kernels/PER_OP_FORWARD_PLAN.md Stage 0/7).
pub struct MetalTeacher {
    weights: WeightStore,
    output: Vec<f32>,
    sample_input: Vec<f32>,
    /// Shared Metal pipeline registry, lazily created on first dispatch attempt.
    #[cfg(feature = "metal-dispatch")]
    dispatch_state: Option<(DenseProjectionDispatcher, RegistryRef)>,
}

impl MetalTeacher {
    /// Create a new MetalTeacher with default 3840×3840 weight matrix.
    pub fn new() -> Self {
        let in_dim = HIDDEN_DIM;
        let out_dim = HIDDEN_DIM;
        // Pre-generate the synthetic sample input (doesn't vary per microbatch
        // in this synthetic Level 1 mode; real weights would load from disk).
        let sample_input: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f64).cos() * 0.1) as f32)
            .collect();

        MetalTeacher {
            weights: WeightStore::new(in_dim, out_dim),
            output: vec![0.0f32; out_dim],
            sample_input,
            #[cfg(feature = "metal-dispatch")]
            dispatch_state: None,
        }
    }

    /// Create a MetalTeacher with a specific weight matrix shape.
    pub fn with_shape(in_dim: usize, out_dim: usize) -> Self {
        let sample_input: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f64).cos() * 0.1) as f32)
            .collect();

        MetalTeacher {
            weights: WeightStore::new(in_dim, out_dim),
            output: vec![0.0f32; out_dim],
            sample_input,
            #[cfg(feature = "metal-dispatch")]
            dispatch_state: None,
        }
    }

    /// Execute dense teacher forward via Accelerate vDSP.
    ///
    /// Tries a real Metal dispatch via `DenseProjectionDispatcher` when the
    /// compiled metallib is available.  Falls back to Accelerate vDSP (AMX/NEON)
    /// dot product when Metal compilation failed or the feature is disabled.
    pub fn forward(&mut self, _microbatch: usize, _slot_id: u64) {
        #[cfg(feature = "metal-dispatch")]
        if self.try_metal_dispatch() {
            return;
        }

        let out_dim = self.weights.out_dim;
        let in_dim = self.weights.in_dim;

        for o in 0..out_dim {
            let start = o * in_dim;
            let row = &self.weights.data[start..start + in_dim];
            // Accelerate vDSP dot product — hits AMX coprocessor on M-series.
            self.output[o] = dot_product(row, &self.sample_input);
        }
    }

    /// Access the internal output buffer for reduction.
    pub fn output(&self) -> &[f32] {
        &self.output
    }

    /// Return the hidden dimension (output width).
    pub fn hidden_dim(&self) -> usize {
        self.weights.out_dim
    }

    /// Attempt a Metal dispatch for the dense forward pass.
    ///
    /// Creates the pipeline registry on first call, then dispatches via
    /// `DenseProjectionDispatcher`.  Returns `true` on success.
    /// Returns `false` on any failure — the caller falls back to Accelerate vDSP.
    /// The existing vDSP path remains the production fallback; this method is
    /// opportunistic and silences all errors.
    #[cfg(feature = "metal-dispatch")]
    fn try_metal_dispatch(&mut self) -> bool {
        let (dispatcher, registry) = match &self.dispatch_state {
            Some((d, r)) => (d, r),
            None => {
                let device = match Device::system_default() {
                    Some(d) => d,
                    None => return false,
                };
                let (reg, _, dense, ..) = create_dispatchers(&device);
                self.dispatch_state = Some((dense, reg.clone()));
                let (d, r) = self.dispatch_state.as_ref().unwrap();
                (d, r)
            }
        };

        let device = registry.lock().device().clone();

        let in_dim = self.weights.in_dim as u32;
        let out_dim = self.weights.out_dim as u32;
        let page_width = 640u32;
        let page_count = (in_dim + page_width - 1) / page_width;

        // Convert f32 sample input to fp16 for the Metal buffer.
        let mut input_fp16: Vec<u16> = Vec::with_capacity(self.sample_input.len());
        for &v in &self.sample_input {
            input_fp16.push(half::f16::from_f32(v).to_bits());
        }

        // Build the 16-entry per-row codebook expected by palettized_gemv.
        // This is a compact approximation of the dense teacher weights, not a
        // full dense matrix upload.
        let mut codebook_fp16 = Vec::with_capacity(self.weights.out_dim * 16);
        let sample_width = self.weights.in_dim.min(16);
        for row in 0..self.weights.out_dim {
            let row_offset = row * self.weights.in_dim;
            for i in 0..16 {
                let src_col = if sample_width == 0 {
                    0
                } else {
                    i % sample_width
                };
                let v = self.weights.data[row_offset + src_col];
                codebook_fp16.push(half::f16::from_f32(v).to_bits());
            }
        }

        let params = ProjectionParams {
            in_dim,
            out_dim,
            page_count,
            page_width,
            mode_flags: 0,
            probe_seed: 0,
            reserved: [0u32; 5],
        };

        let queue = device.new_command_queue();
        let cmd_buf = queue.new_command_buffer();

        // Input buffer: fp16 activations
        let input_buf = device.new_buffer_with_data(
            input_fp16.as_ptr() as *const std::ffi::c_void,
            (input_fp16.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Output buffer: fp16 result
        let output_buf =
            device.new_buffer((out_dim as u64) * 2, MTLResourceOptions::StorageModeShared);

        let mut receipt = KernelReceipt {
            kernel_id: 0,
            phase_id: 0,
            page_count: 0,
            sidecar_hits: 0,
            sidecar_entries_read: 0,
            threadgroups: 0,
            threads_per_threadgroup: 0,
            output_elements: 0,
            flags: 0,
            logical_weight_bytes: 0,
            logical_sidecar_bytes: 0,
            logical_activation_bytes: 0,
        };

        // Build packed 4-bit indices (two per byte) matching palettized_gemv.
        let idx_count = (in_dim as usize * out_dim as usize) / 2;
        let mut indices = vec![0u8; idx_count];
        for row in 0..out_dim as usize {
            let row_base = row * (in_dim as usize / 2);
            for c in 0..(in_dim as usize / 2) {
                let idx_lo = ((c * 2) % 16) as u8;
                let idx_hi = ((c * 2 + 1) % 16) as u8;
                indices[row_base + c] = idx_lo | (idx_hi << 4);
            }
        }

        let codebook = device.new_buffer_with_data(
            codebook_fp16.as_ptr() as *const std::ffi::c_void,
            (codebook_fp16.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let indices_buf = device.new_buffer_with_data(
            indices.as_ptr() as *const std::ffi::c_void,
            indices.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        dispatcher.dispatch(
            cmd_buf,
            &codebook,
            &indices_buf,
            &input_buf,
            &output_buf,
            &params,
            &mut receipt,
            false, // instrumented — off for now
        );

        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        // Read back fp16 output.
        let ptr = output_buf.contents() as *const u16;
        let slice = unsafe { std::slice::from_raw_parts(ptr, out_dim as usize) };
        for (i, &half_bits) in slice.iter().enumerate() {
            self.output[i] = half::f16::from_bits(half_bits).to_f32();
        }

        true
    }
}

impl Default for MetalTeacher {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Real Gemma 4 teacher forward (full graph via the Orchestrator megakernel)
// ═══════════════════════════════════════════════════════════════════════════
//
// `MetalTeacher` above is the Level-1 *numerical-parity proxy* (one synthetic
// dense GEMV for reducer comparison). This is the REAL teacher: it runs the
// full Gemma 4 forward over a compiled `.cimage` and returns logits, which the
// distillation loop scores against the student (see compilation::distill_core,
// compilation::bench_metrics).
//
// ARCHITECTURE: Gemma 4's forward — embed → 48 layers of
// {RMSNorm, QKV, attention, O-proj, FFN gate/up/SiLU/down} → final norm →
// logits, with the KV cache threaded across positions — is implemented as a
// single fused **megakernel** inside [`Orchestrator`], NOT as a chain of per-op
// dispatchers. The per-op dispatchers in `kernel_dispatch` cover projections
// (incl. `Nf4Tile640ProjectionDispatcher`), fused RMSNorm+QKV and
// O-proj+residual, plus numerical *probes* — but attention/SDPA, FFN, embedding,
// and the logit projection are fused into the megakernel. Re-implementing them
// op-by-op would duplicate (and diverge from) the proven megakernel, so the
// teacher delegates to it. `decode_token_logits` (added on Orchestrator) is the
// hook that surfaces the megakernel's per-step logits.
#[cfg(feature = "prism-backend")]
pub struct Gemma4Teacher {
    orch: crate::compute_image::orchestrator::Orchestrator,
}

#[cfg(feature = "prism-backend")]
impl Gemma4Teacher {
    /// Load a teacher `.cimage` (batch 1, NF4/native — no int4 expansion).
    pub fn load(cimage: impl AsRef<std::path::Path>) -> Result<Self, String> {
        let orch = crate::compute_image::orchestrator::Orchestrator::from_cimage(
            cimage.as_ref(),
            1,
            false,
        )?;
        Ok(Self { orch })
    }

    /// Full-graph forward: prefill `prompt[..n-1]` to build the KV context, then
    /// decode the last token and return the next-position logit vector.
    ///
    /// NOTE: this advances the KV cache. Call [`load`] again (or reset) for an
    /// independent context — successive calls continue the same sequence.
    pub fn logits_after(&mut self, prompt: &[u32]) -> Result<Vec<f32>, String> {
        if prompt.is_empty() {
            return Err("Gemma4Teacher: empty prompt".into());
        }
        if prompt.len() > 1 {
            self.orch.prefill_text(&prompt[..prompt.len() - 1])?;
        }
        let (_next, logits) = self.orch.decode_token_logits(prompt[prompt.len() - 1])?;
        Ok(logits)
    }

    /// Teacher-forced pass over `tokens`: per-position next-token logits — the
    /// distillation / perplexity signal. Feeds the true tokens in order
    /// (advancing the KV cache).
    ///
    /// Returns ONE flat row-major `[positions × vocab]` buffer plus the vocab
    /// width. Rows are appended as they decode, so the resident footprint is
    /// the flat buffer plus a single transient row — never two copies of the
    /// full logits. (The earlier `Vec<Vec<f32>>` shape held the nested rows
    /// AND the flattened copy alive simultaneously during scoring — a ~2×
    /// transient peak the low-memory validation lane cannot afford; see the
    /// `kd_gate` module docs for the budget math.)
    pub fn teacher_forced_flat(&mut self, tokens: &[u32]) -> Result<(Vec<f32>, usize), String> {
        let mut flat: Vec<f32> = Vec::new();
        let mut vocab = 0usize;
        for (i, &t) in tokens.iter().enumerate() {
            let (_next, logits) = self.orch.decode_token_logits(t)?;
            if i == 0 {
                vocab = logits.len();
                if vocab == 0 {
                    return Err("teacher_forced_flat: empty logit row at position 0".into());
                }
                // One reservation for the whole pass — no growth reallocations.
                flat.reserve_exact(vocab * tokens.len());
            } else if logits.len() != vocab {
                return Err(format!(
                    "teacher_forced_flat: ragged logit row at position {i}: {} vs vocab {vocab}",
                    logits.len()
                ));
            }
            flat.extend_from_slice(&logits);
        }
        Ok((flat, vocab))
    }

    /// Borrow the underlying orchestrator (e.g. to reset slots between eval docs).
    pub fn orchestrator_mut(&mut self) -> &mut crate::compute_image::orchestrator::Orchestrator {
        &mut self.orch
    }
}
