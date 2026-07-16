//! GPTQ (Post-Training Quantization via GPTQ) codec family.
//!
//! Reference: https://arxiv.org/abs/2210.17323
//! Implements the optimal-brain-damage style layer-wise quantization that
//! compensates each quantized column's error into the remaining columns using
//! the precomputed Hessian inverse.
//!
//! Sources:
//! - llm-compressor/src/llmcompressor/modifiers/gptq/gptq_quantize.py
//! - AutoGPTQ/auto_gptq/nn_modules/qlinear/qlinear_cuda.py

use serde::{Deserialize, Serialize};

// ── GptqCodecFamily ───────────────────────────────────────────────────────

/// GPTQ codec family — parameter space for Cholesky-based quantization.
///
/// Fields correspond to the GPTQ algorithm exposed in llm-compressor's
/// `quantize_weight` and AutoGPTQ's `QuantLinear`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GptqCodecFamily {
    /// Columns per quantization group.
    pub group_size: usize,
    /// Dampening fraction on the Hessian diagonal to improve numerical
    /// stability.  llm-compressor default: 0.01.
    pub damp_percent: f32,
    /// Block size for column-wise quantization.  Process this many columns
    /// at once within the outer loop.  Default: 128.
    pub block_size: usize,
    /// Descending activation ordering — reorder columns by Hessian diagonal
    /// so the most important columns are quantized first (less error to
    /// propagate).  Corresponds to `actorder` in llm-compressor.
    pub desc_act: bool,
    /// Number of bits per element (default 4).
    pub n_bit: u32,
    /// Whether quantization is symmetric (no zero-point).
    pub sym: bool,
    /// True group-size strategy (vs channel or block strategy).
    /// Matches QuantizationStrategy::GROUP from compressed-tensors.
    pub use_group_strategy: bool,
}

impl Default for GptqCodecFamily {
    fn default() -> Self {
        Self {
            group_size: 128,
            damp_percent: 0.01,
            block_size: 128,
            desc_act: true,
            n_bit: 4,
            sym: false,
            use_group_strategy: true,
        }
    }
}

// ── GptqSaliency ──────────────────────────────────────────────────────────

/// Hessian-based per-channel importance for GPTQ.
///
/// GPTQ uses the full Hessian matrix `H = sum(sqrt(2) * x * x^T)` over
/// calibration inputs to determine the optimal quantization order and
/// error compensation.  The diagonal of H in descending order determines
/// which columns are quantized first (desc_act = True).
///
/// Reference: `accumulate_hessian` in llm-compressor accumulates
/// `H += sqrt(2) * inp @ inp.t()` for each module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GptqSaliency {
    /// Full Hessian matrix stored as a flat Vec (row-major).
    /// Shape: `[num_columns * num_columns]`.
    pub hessian: Vec<f32>,
    /// Number of columns (in_features of the weight matrix).
    pub num_columns: usize,
    /// Number of calibration samples accumulated.
    pub num_samples: usize,
    /// Permutation indices from activation ordering.
    /// None if `desc_act = False`.
    pub perm: Option<Vec<usize>>,
}

impl GptqSaliency {
    /// Create a new empty Hessian for `num_columns` features.
    pub fn empty(num_columns: usize) -> Self {
        Self {
            hessian: vec![0.0_f32; num_columns * num_columns],
            num_columns,
            num_samples: 0,
            perm: None,
        }
    }

    /// Retrieve the diagonal of the Hessian matrix.
    pub fn diagonal(&self) -> Vec<f32> {
        self.hessian
            .chunks(self.num_columns)
            .enumerate()
            .map(|(i, row)| row[i])
            .collect()
    }

    /// Column importance ranks (larger diagonal → more important).
    /// Returns indices sorted by descending diagonal value.
    pub fn importance_rank(&self) -> Vec<usize> {
        let diag = self.diagonal();
        let mut indices: Vec<usize> = (0..diag.len()).collect();
        indices.sort_by(|&a, &b| {
            diag[b]
                .partial_cmp(&diag[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices
    }
}

// ── GptqQuantResult ───────────────────────────────────────────────────────

/// Result of GPTQ quantization for a single module.
///
/// Corresponds to the return value of `quantize_weight` in llm-compressor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GptqQuantResult {
    /// Quantized weight tensor (packed).
    pub qweight: Vec<u8>,
    /// Scale per group.
    pub scales: Vec<f32>,
    /// Zero-point per group.
    pub zeros: Vec<f32>,
    /// Group index mapping column → group (for desc_act).
    pub g_idx: Option<Vec<u32>>,
    /// Total quantization loss (sum over rows of column-loss).
    pub loss: f32,
}

// ── GptqSweepGrid ─────────────────────────────────────────────────────────

/// Sweep grid for GPTQ hyper-parameter exploration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GptqSweepGrid {
    /// Group sizes to try.
    pub group_sizes: Vec<usize>,
    /// Dampening fractions to try.
    pub damp_percents: Vec<f32>,
    /// Block sizes to try.
    pub block_sizes: Vec<usize>,
    /// Whether to try desc_act.
    pub try_desc_act: bool,
}

impl Default for GptqSweepGrid {
    fn default() -> Self {
        Self {
            group_sizes: vec![32, 64, 128],
            damp_percents: vec![0.001, 0.01, 0.1],
            block_sizes: vec![64, 128],
            try_desc_act: true,
        }
    }
}
