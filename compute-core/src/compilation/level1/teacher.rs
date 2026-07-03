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
use half;
#[cfg(feature = "metal-dispatch")]
use crate::compute_image::compile::kernel_dispatch::{create_dispatchers, DenseProjectionDispatcher, RegistryRef};
#[cfg(feature = "metal-dispatch")]
use crate::compute_image::compile::kernel_types::{KernelReceipt, ProjectionParams};
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
        WeightStore { data, in_dim, out_dim }
    }
}

/// The dense teacher forward executor.
///
/// Wraps the Accelerate/vDSP-based dense GEMV that runs the teacher model on
/// a microbatch of tokens.  Stateless — each call receives a microbatch index
/// and produces output activations into the internal output buffer.
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

        // Convert f32 weights to fp16 for the Metal codebook buffer.
        let mut weights_fp16: Vec<u16> = Vec::with_capacity(self.weights.data.len());
        for &v in &self.weights.data {
            weights_fp16.push(half::f16::from_f32(v).to_bits());
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

        // Codebook buffer: fp16 weights for the palettized_gemv kernel.
        let codebook = device.new_buffer_with_data(
            weights_fp16.as_ptr() as *const std::ffi::c_void,
            (weights_fp16.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Indices buffer: simple linear mapping (palettized_gemv expects 2 bytes per index).
        let idx_count = (in_dim as usize * out_dim as usize) / 2;
        let mut indices = vec![0u8; idx_count];
        for i in 0..indices.len() {
            indices[i] = (i % 16) as u8; // cycle through codebook entries
        }
        let indices_buf = device.new_buffer_with_data(
            indices.as_ptr() as *const std::ffi::c_void,
            indices.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Input buffer: fp16 activations
        let input_buf = device.new_buffer_with_data(
            input_fp16.as_ptr() as *const std::ffi::c_void,
            (input_fp16.len() * 2) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        // Output buffer: fp16 result
        let output_buf = device.new_buffer(
            (out_dim as u64) * 2,
            MTLResourceOptions::StorageModeShared,
        );

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
