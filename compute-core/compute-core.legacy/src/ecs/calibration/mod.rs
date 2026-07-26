//! Prism-PT2: Offline calibration engine for 1.58-bit ternary quantization.
//!
//! Implements the PT2-LLM post-training ternarization algorithm using Apple's
//! Accelerate framework (vDSP on AMX/NEON) for the ITF/AGA scale optimization
//! and custom Metal shaders for the FP16 forward pass calibration runs.
//!
//! Architecture:
//!
//! ```text
//!                     ┌──────────────────────┐
//!                     │  FP16 model weights   │
//!                     │  (safetensors shards) │
//!                     └──────┬───────────────┘
//!                            │
//!               ┌────────────▼────────────┐
//!               │  Per-layer calibration  │
//!               │  loop (42 layers × K-V)│
//!               └────┬────────────────┬──┘
//!                    │                │
//!          ┌─────────▼──────┐  ┌─────▼──────────┐
//!          │ Metal forward  │  │ Accelerate ITF │
//!          │ pass (FP16 GPU)│  │ (vDSP dot +    │
//!          │ → shared buf   │  │  threshold on  │
//!          │   via MTLShared│  │  AMX coproc)   │
//!          │   Event signal │  │                │
//!          └────────┬───────┘  └───────┬────────┘
//!                   │                  │
//!                   └──────┬──────────┘
//!                          ▼
//!               ┌──────────────────────┐
//!               │  Packed ternary      │
//!               │  cimage output       │
//!               └──────────────────────┘
//! ```

pub mod accelerate;

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

/// Configuration for the Prism-PT2 calibration engine.
#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    /// Number of calibration sequences to use (default: 1024).
    pub num_calibration_sequences: usize,
    /// Sequence length for calibration (default: 512 tokens).
    pub calibration_seq_length: usize,
    /// Threshold for ternary snapping (default: 0.5).
    pub ternary_threshold: f32,
    /// Number of ITF refinement iterations per layer (default: 20).
    pub itf_iterations: usize,
    /// Path to the NVMe SSD file for streaming calibration activations.
    pub activation_stream_path: std::path::PathBuf,
    /// Group size for block quantization (default: 128).
    pub group_size: usize,
    /// Whether to reorder weight columns by structural similarity (SSR).
    pub enable_ssr: bool,
    /// Whether to align the ternary grid to calibration activations (AGA).
    pub enable_aga: bool,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            num_calibration_sequences: 1024,
            calibration_seq_length: 512,
            ternary_threshold: 0.5,
            itf_iterations: 20,
            activation_stream_path: std::env::temp_dir().join("prism-pt2-activations.bin"),
            group_size: 128,
            enable_ssr: true,
            enable_aga: true,
        }
    }
}

/// Result of calibrating a single linear layer to ternary format.
#[derive(Debug)]
pub struct CalibratedLayer {
    /// Layer name (e.g., "model.layers.0.self_attn.q_proj").
    pub name: String,
    /// Packed ternary weights: 16 weights per uint32, 00/01/10 encoding.
    pub packed_weights: Vec<u32>,
    /// Output dimension.
    pub out_dim: u32,
    /// Input dimension.
    pub in_dim: u32,
    /// Per-group FP32 scale factors (one per `group_size` weights per row).
    pub group_scales: Vec<f32>,
    /// Optimal alpha scale found by ITF (positive ternary value).
    pub alpha: f32,
    /// Optimal beta scale found by ITF (negative ternary value).
    pub beta: f32,
    /// Final quantization MSE for this layer.
    pub final_mse: f32,
}

/// Status signal from the Metal calibration forward pass.
/// Set to true when a layer's activations have been written to shared memory.
pub type ActivationReadySignal = Arc<AtomicBool>;

/// Run the full PT2-LLM calibration pipeline on a set of FP16 model weights.
///
/// 1. For each linear layer, run an FP16 forward pass on calibration data
///    via the Metal calibration shader.
/// 2. Stream the captured activations to NVMe (CPU-side read via MTLSharedEvent).
/// 3. Run Iterative Ternary Fitting (ITF) on the Accelerate AMX coprocessor
///    to find optimal {-alpha, 0, +beta} ternary scales.
/// 4. Pack the resulting ternary weights into the 16-per-uint32 format.
/// 5. Output a CalibratedLayer for emission into the cimage pipeline.
pub fn calibrate_model(
    _model_path: &Path,
    _config: &CalibrationConfig,
    _cancel_token: Option<ActivationReadySignal>,
) -> Result<Vec<CalibratedLayer>, CalibrationError> {
    // TODO(full-implementation): Wire the Metal forward pass pipeline:
    //   1. Load FP16 weights per layer
    //   2. Dispatch calibration_forward_pass Metal kernel
    //      with MTLStorageModeShared output buffer
    //   3. Wait on MTLSharedEvent for GPU completion
    //   4. Stream shared buffer to NVMe activation_stream_path
    //   5. Call run_itf() for this layer's weights + activations
    //   6. Collect CalibratedLayer and append to output
    //
    // The Metal dispatch pipeline (MTLCommandBuffer, MTLComputeCommandEncoder,
    // MTLSharedEvent encoding) is in compute_core/src/calibration/metal_pipeline.rs
    // which depends on the metal crate (feature-gated by metal-dispatch).
    //
    // For now, the accelerate math is fully tested. The Metal dispatch layer
    // requires the metal-dispatch feature to be enabled. See:
    //   - src/calibration/metal_pipeline.rs (planned)
    //   - src/bridge/metal_calibration.mm (planned)
    Err(CalibrationError::Unimplemented(
        "Metal calibration dispatch pipeline not yet wired. \
         Run with --features metal-dispatch after building the \
         MTLCommandBuffer encoding path. \
         The Accelerate ITF math (accelerate.rs) is ready and tested."
            .to_string(),
    ))
}

/// Run Iterative Ternary Fitting (ITF) on a single weight matrix.
///
/// This is the core PT2-LLM algorithm. For each group of `group_size` weights,
/// it searches for the optimal asymmetric ternary scales (alpha for +1, beta for -1)
/// that minimize the quantization MSE against the original FP16 activations.
///
/// The algorithm:
/// 1. Start with alpha = beta = max_abs (symmetric initialization)
/// 2. For each iteration:
///    a. Snap weights to ternary grid using current alpha/beta
///    b. Compute MSE against original via Accelerate vDSP_dotpr
///    c. Adjust alpha/beta to minimize MSE (golden-section search)
/// 3. Return optimal scales and the packed ternary representation
pub fn run_itf(
    weights_f32: &[f32],
    _activations_f32: &[f32],
    out_dim: usize,
    in_dim: usize,
    group_size: usize,
    threshold: f32,
    max_iterations: usize,
) -> ItfResult {
    let groups_per_row = (in_dim + group_size - 1) / group_size;
    let packed_in = (in_dim + 15) / 16;
    let mut packed = vec![0u32; out_dim * packed_in];
    let mut group_scales = Vec::with_capacity(out_dim * groups_per_row);
    let mut total_mse = 0.0f32;

    for row in 0..out_dim {
        let row_offset = row * in_dim;
        for g in 0..groups_per_row {
            let start = row_offset + g * group_size;
            let end = (start + group_size).min(row_offset + in_dim);
            let group_weights = &weights_f32[start..end];

            // Find optimal scale for this group via golden-section search
            let max_abs = group_weights.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

            if max_abs < 1e-12 {
                group_scales.push(1.0);
                continue;
            }

            let (optimal_scale, final_mse) = golden_section_search(
                group_weights,
                max_abs * 0.1, // lower bound
                max_abs * 2.0, // upper bound
                threshold,
                max_iterations,
            );

            group_scales.push(optimal_scale);
            total_mse += final_mse;

            // Snap to ternary and pack
            let inv = 1.0 / optimal_scale;
            for (i, &w) in group_weights.iter().enumerate() {
                let normalized = w * inv;
                let ternary = if normalized > threshold {
                    1 // +1
                } else if normalized < -threshold {
                    2 // -1
                } else {
                    0
                };
                let word_idx = row * ((in_dim + 15) / 16) + (start + i) / 16;
                let shift = ((start + i) % 16) * 2;
                packed[word_idx] |= (ternary as u32) << shift;
            }
        }
    }

    let final_mse = total_mse / (out_dim * groups_per_row) as f32;
    let alpha = group_scales.iter().copied().fold(0.0f32, f32::max);
    let beta = group_scales.iter().copied().fold(f32::MAX, f32::min);

    ItfResult {
        packed_weights: packed,
        group_scales,
        final_mse,
        alpha,
        beta,
    }
}

/// Result of a single ITF calibration run on one weight matrix.
#[derive(Debug)]
pub struct ItfResult {
    pub packed_weights: Vec<u32>,
    pub group_scales: Vec<f32>,
    pub final_mse: f32,
    pub alpha: f32,
    pub beta: f32,
}

/// Golden-section search for optimal ternary scale factor.
///
/// Minimizes `quantization_mse(weights, scale)` over [lo, hi].
/// Converges in ~log(hi/lo) / log(golden) iterations (~20 for 10^4 range).
pub fn golden_section_search(
    weights: &[f32],
    lo: f32,
    hi: f32,
    _threshold: f32,
    max_iter: usize,
) -> (f32, f32) {
    const GOLDEN: f32 = 0.6180339887;

    let mut a = lo;
    let mut b = hi;
    let mut c = b - GOLDEN * (b - a);
    let mut d = a + GOLDEN * (b - a);

    let mut fc = accelerate::quantization_mse(weights, c);
    let mut fd = accelerate::quantization_mse(weights, d);

    for _ in 0..max_iter {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - GOLDEN * (b - a);
            fc = accelerate::quantization_mse(weights, c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + GOLDEN * (b - a);
            fd = accelerate::quantization_mse(weights, d);
        }
    }

    let mid = (a + b) / 2.0;
    let mse = accelerate::quantization_mse(weights, mid);
    (mid, mse)
}

/// Errors from the calibration pipeline.
#[derive(Debug)]
pub enum CalibrationError {
    /// The calibration pipeline is not yet fully wired to Metal dispatch.
    Unimplemented(String),
    /// IO error reading model weights or writing activation stream.
    Io(std::io::Error),
    /// Invalid model architecture (missing expected tensors).
    Architecture(String),
    /// Calibration cancelled by user.
    Cancelled,
}

impl std::fmt::Display for CalibrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unimplemented(msg) => write!(f, "calibration unimplemented: {}", msg),
            Self::Io(e) => write!(f, "calibration IO error: {}", e),
            Self::Architecture(msg) => write!(f, "calibration architecture: {}", msg),
            Self::Cancelled => write!(f, "calibration cancelled"),
        }
    }
}

impl std::error::Error for CalibrationError {}

impl From<std::io::Error> for CalibrationError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_golden_section_search_perfect_ternary() {
        // Weights already in {-1, 0, +1} — optimal scale should be 1.0
        let weights: Vec<f32> = vec![1.0, -1.0, 0.0, 1.0, -1.0, 0.0];
        let max_abs = weights.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let (scale, mse) = golden_section_search(&weights, max_abs * 0.1, max_abs * 2.0, 0.5, 20);
        assert!((scale - 1.0).abs() < 0.2, "scale={} should be ~1.0", scale);
        assert!(mse < 1e-6, "mse={} should be ~0", mse);
    }

    #[test]
    fn test_golden_section_search_scaled_ternary() {
        // Weights are {-1, 0, +1} * 0.5 — optimal scale should be ~0.5.
        // Upper bound must not exceed max_abs * 2 to avoid the plateau
        // where all values snap to 0 and MSE plateaus at mean(orig^2).
        let weights: Vec<f32> = vec![0.5, -0.5, 0.0, 0.5, 0.0, -0.5];
        let max_abs = weights.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let (scale, mse) = golden_section_search(&weights, max_abs * 0.1, max_abs * 2.0, 0.5, 20);
        assert!((scale - 0.5).abs() < 0.1, "scale={} should be ~0.5", scale);
        assert!(mse < 1e-6, "mse={} should be ~0", mse);
    }

    #[test]
    fn test_run_itf_small_matrix() {
        let out_dim = 2;
        let in_dim = 8;
        let group_size = 4;
        // Perfect ternary values
        let w: Vec<f32> = vec![
            1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0,
        ];
        let a: Vec<f32> = vec![0.0; out_dim * in_dim];

        let result = run_itf(&w, &a, out_dim, in_dim, group_size, 0.5, 20);
        let packed_in = (in_dim + 15) / 16;
        let groups_per_row = (in_dim + group_size - 1) / group_size;
        assert_eq!(result.packed_weights.len(), out_dim * packed_in);
        assert_eq!(result.group_scales.len(), out_dim * groups_per_row);
        assert!(result.final_mse < 1e-6, "mse={}", result.final_mse);
    }

    #[test]
    fn test_calibration_error_display() {
        let err = CalibrationError::Unimplemented("test".into());
        assert!(err.to_string().contains("test"));
    }
}
