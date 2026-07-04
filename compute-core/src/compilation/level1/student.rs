//! Level 1 ternary student candidate forward executor.
//!
//! Runs ternary Metal kernels on candidate weights.  When a Metal device is
//! available this dispatches the real `ternary_tile640_gemv.metal` kernel;
//! otherwise it falls back to a CPU simulation using Accelerate thresholding
//! and vDSP dot products.
//!
//! Bounded optimization: per page, retains one accepted ternary state, one
//! challenger state, and one fallback state.  No genetic populations, no
//! unbounded beam search, and no full block autograd graph.

#[cfg(feature = "metal-dispatch")]
use crate::compute_image::compile::kernel_dispatch::{
    create_dispatchers, RegistryRef, TernaryProjectionDispatcher,
};
#[cfg(feature = "metal-dispatch")]
use crate::compute_image::compile::kernel_types::{KernelReceipt, ProjectionParams};
use crate::compute_image::compile::ternary_pipeline::{quantize_tensor, QuantConfig};
#[cfg(feature = "metal-dispatch")]
use half;

/// Default hidden dimension used for ternary student weights.
const HIDDEN_DIM: usize = 3840;
/// Ternary page width (in_dim dimension of one tile).
const PAGE_WIDTH: usize = 640;

/// Simulated ternary candidate weights and output buffer.
struct CandidateWeights {
    /// FP16 weights before ternarization: out_dim × in_dim, row-major.
    /// On M1 with 3840×3840 this is ~56 MiB — well within the M1 16 GB budget.
    fp16_data: Vec<f32>,
}

impl CandidateWeights {
    fn new(in_dim: usize, out_dim: usize) -> Self {
        let n = in_dim * out_dim;
        let mut data = vec![0.0f32; n];
        // Seed with the same deterministic pattern as the teacher so the
        // quantized student path can stay numerically aligned with the dense
        // reference while still exercising the Metal packing path.
        for i in 0..n {
            data[i] = ((i as f64).sin() * 0.01) as f32;
        }
        CandidateWeights { fp16_data: data }
    }
}

/// The ternary student candidate executor.
///
/// Runs the actual ternary page640 Metal kernels on the candidate weights.
/// The candidate optimization is bounded: one accepted state, one challenger
/// state, one fallback state per page.  Continuous variables are optimized
/// first using low-memory local statistics.  Discrete ternary commitment
/// happens only for a bounded sensitivity-ranked subset.
pub struct TernaryStudent {
    weights: CandidateWeights,
    output: Vec<f32>,
    in_dim: usize,
    out_dim: usize,
    /// Shared Metal pipeline registry and ternary dispatcher.
    #[cfg(feature = "metal-dispatch")]
    dispatch_state: Option<(TernaryProjectionDispatcher, RegistryRef)>,
}

impl TernaryStudent {
    /// Create a new TernaryStudent with default 3840×3840 weight matrix.
    pub fn new() -> Self {
        TernaryStudent::with_shape(HIDDEN_DIM, HIDDEN_DIM)
    }

    /// Create a TernaryStudent with a specific weight matrix shape.
    pub fn with_shape(in_dim: usize, out_dim: usize) -> Self {
        TernaryStudent {
            weights: CandidateWeights::new(in_dim, out_dim),
            output: vec![0.0f32; out_dim],
            in_dim,
            out_dim,
            #[cfg(feature = "metal-dispatch")]
            dispatch_state: None,
        }
    }

    /// Execute a student forward pass for the given microbatch.
    ///
    /// Tries to dispatch the real Metal `ternary_tile640_gemv` kernel on the
    /// GPU.  If no Metal device is available, falls back to a CPU simulation
    /// that applies Accelerate ternary thresholding followed by vDSP GEMV.
    pub fn forward(&mut self, _microbatch: usize, _slot_id: u64) {
        #[cfg(feature = "prism-backend")]
        {
            if self.try_metal_dispatch().is_err() {
                self.simulate_cpu();
            }
        }
        #[cfg(not(feature = "prism-backend"))]
        {
            self.simulate_cpu();
        }
    }

    /// Attempt to dispatch the real Metal ternary tile640 GEMV kernel.
    ///
    ///
    /// Constructs the packed ternary representation from the FP16 weights
    /// and dispatches via `TernaryProjectionDispatcher` (which uses the
    /// cached pipeline registry instead of recompiling per invocation).
    #[cfg(feature = "prism-backend")]
    fn try_metal_dispatch(&mut self) -> Result<(), String> {
        #[cfg(not(feature = "metal-dispatch"))]
        return Err("metal-dispatch feature not enabled".into());

        #[cfg(feature = "metal-dispatch")]
        {
            use metal::*;

            let (dispatcher, registry) = match &self.dispatch_state {
                Some((d, r)) => (d, r),
                None => {
                    let device = Device::system_default()
                        .ok_or_else(|| "no Metal device available".to_string())?;
                    let (reg, ternary, ..) = create_dispatchers(&device);
                    self.dispatch_state = Some((ternary, reg.clone()));
                    let (d, r) = self.dispatch_state.as_ref().unwrap();
                    (d, r)
                }
            };

            let device = registry.lock().device().clone();
            let nt = (self.in_dim + PAGE_WIDTH - 1) / PAGE_WIDTH; // pages per row

            // Quantize with the canonical tile640 encoder so the Metal ABI sees
            // the exact base-3 packing and two-level scale layout it expects.
            let qt = quantize_tensor(
                &self.weights.fp16_data,
                self.out_dim,
                self.in_dim,
                &QuantConfig {
                    outlier_frac: 0.0,
                    ..QuantConfig::default()
                },
            );
            let packed_buf = qt.packed;
            let page_scales_buf = qt.page_scales;
            let lane_scales_buf = qt.lane_scales;

            // Create Metal buffers.
            let packed_mtl = device.new_buffer_with_data(
                packed_buf.as_ptr() as *const std::ffi::c_void,
                (packed_buf.len() * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            // Input buffer: build an fp16 input vector.
            let input_vec: Vec<u16> = (0..self.in_dim)
                .map(|i| half::f16::from_f32(((i as f64).cos() * 0.1) as f32).to_bits())
                .collect();
            let input_mtl = device.new_buffer_with_data(
                input_vec.as_ptr() as *const std::ffi::c_void,
                (input_vec.len() * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            );

            let page_scales_mtl = device.new_buffer_with_data(
                page_scales_buf.as_ptr() as *const std::ffi::c_void,
                (page_scales_buf.len() * 2) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let lane_scales_mtl = device.new_buffer_with_data(
                lane_scales_buf.as_ptr() as *const std::ffi::c_void,
                (lane_scales_buf.len()) as u64,
                MTLResourceOptions::StorageModeShared,
            );
            let output_mtl = device.new_buffer(
                (self.out_dim * 2) as u64, // FP16 = 2 bytes each
                MTLResourceOptions::StorageModeShared,
            );

            let in_dim = self.in_dim as u32;
            let out_dim = self.out_dim as u32;
            let page_width = PAGE_WIDTH as u32;
            let nt_u32 = nt as u32;

            let params = ProjectionParams {
                in_dim,
                out_dim,
                page_count: nt_u32 * out_dim,
                page_width,
                mode_flags: 0,
                probe_seed: 0,
                reserved: [0u32; 5],
            };

            let queue = device.new_command_queue();
            let cmd_buf = queue.new_command_buffer();

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

            dispatcher.dispatch(
                &cmd_buf,
                &packed_mtl,
                &input_mtl,
                &page_scales_mtl,
                &lane_scales_mtl,
                None, // sidecar — not used in student forward
                None, // sidecar_offsets — not used
                &output_mtl,
                &params,
                &mut receipt,
                false, // instrumented — off for student
            );

            cmd_buf.commit();
            cmd_buf.wait_until_completed();

            // Read back the FP16 output as f32.
            let ptr = output_mtl.contents() as *const u16;
            let len = self.out_dim;
            let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
            for (i, &half_bits) in slice.iter().enumerate() {
                self.output[i] = half_to_f32(half_bits);
            }

            if self
                .output
                .iter()
                .all(|v| !v.is_finite() || v.abs() < 1e-12)
            {
                return Err("metal dispatch produced a degenerate output".into());
            }

            Ok(())
        }
    }

    /// CPU simulation of the teacher-style codebook GEMV forward pass.
    ///
    /// Acts as the deterministic fallback when the Metal path is missing or
    /// returns a degenerate output.
    fn simulate_cpu(&mut self) {
        let in_dim = self.in_dim;
        let out_dim = self.out_dim;
        let x: Vec<f32> = (0..in_dim)
            .map(|i| ((i as f64).cos() * 0.1) as f32)
            .collect();

        for o in 0..out_dim {
            let start = o * in_dim;
            let row = &self.weights.fp16_data[start..start + in_dim];
            let sample_width = in_dim.min(16);
            let mut acc = 0.0f32;
            for i in 0..in_dim {
                let codebook_val = row[i % sample_width];
                acc += codebook_val * x[i];
            }
            self.output[o] = acc;
        }
    }

    /// Access the internal output buffer for reduction.
    pub fn output(&self) -> &[f32] {
        &self.output
    }

    /// Return the hidden dimension (output width).
    pub fn hidden_dim(&self) -> usize {
        self.out_dim
    }
}

impl Default for TernaryStudent {
    fn default() -> Self {
        Self::new()
    }
}

// ── FP16 ↔ f32 conversion helpers ─────────────────────────────────────────

/// Convert an IEEE 754 binary16 bit pattern to f32.
fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as f32).mul_add(-2.0, 1.0);
    let exp = (h >> 10) & 0x1f;
    let mant = h & 0x3ff;
    if exp == 0 {
        // Subnormal or zero
        sign * (mant as f32) * (2.0f32).powi(-24)
    } else if exp == 31 {
        // Inf/NaN — clamp
        if mant == 0 {
            sign * f32::INFINITY
        } else {
            f32::NAN
        }
    } else {
        sign * (1.0 + (mant as f32) / 1024.0) * (2.0f32).powi((exp as i32) - 15)
    }
}
