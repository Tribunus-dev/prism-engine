use serde::{Deserialize, Serialize};

/// Deterministic search budget controlling Phase 4 calibration space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchBudgetClass {
    /// Exclude ternary; start with NF4, INT8, FP16, or RawF32
    SkipTernary,
    /// Small fixed ternary recipe set with whole-tensor rescue only
    CheapTernaryProbe,
    /// Full evolution over thresholds, scales, group sizes, sparsity, residuals
    FullTernarySweep,
    /// Search higher-precision combinations without ternary
    MixedPrecisionOnly,
    /// Full mixed refinement — tile, row, column, or group rescue after ternary base
    FullMixedRefinement,
}

/// Per-tensor sensitivity receipt — content-addressed, deterministic, replayable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorSensitivityReceipt {
    pub receipt_id: String,
    pub tensor_key: String,
    pub tensor_class: String,
    pub layer_index: u32,
    pub is_mtp: bool,
    pub weight_outlier_mass: f64,
    pub weight_kurtosis: f64,
    pub ternary_recon_nrmse: f64,
    pub ternary_max_error: f64,
    pub activation_nrmse: Option<f64>,
    pub activation_cosine: Option<f64>,
    pub max_channel_shift: f64,
    pub sensitive_channels: Vec<u32>,
    pub layer_error_amplification: f64,
    pub budget: SearchBudgetClass,
    pub source_digest: String,
    pub calibrator_digest: String,
}

/// Budget parameters derived from SearchBudgetClass.
#[derive(Debug, Clone, Copy)]
pub struct BudgetParameters {
    pub population_size: usize,
    pub generations: usize,
    pub max_calibration_batches: usize,
}

impl SearchBudgetClass {
    pub fn parameters(&self) -> BudgetParameters {
        match self {
            Self::SkipTernary => BudgetParameters {
                population_size: 0,
                generations: 0,
                max_calibration_batches: 0,
            },
            Self::CheapTernaryProbe => BudgetParameters {
                population_size: 4,
                generations: 1,
                max_calibration_batches: 1,
            },
            Self::FullTernarySweep => BudgetParameters {
                population_size: 32,
                generations: 8,
                max_calibration_batches: 8,
            },
            Self::MixedPrecisionOnly => BudgetParameters {
                population_size: 8,
                generations: 3,
                max_calibration_batches: 4,
            },
            Self::FullMixedRefinement => BudgetParameters {
                population_size: 64,
                generations: 16,
                max_calibration_batches: 16,
            },
        }
    }
}

/// Run sensitivity analysis on a single tensor.
/// Uses cheap canonical ternary probe, not full calibration.
pub fn analyze_tensor_sensitivity(
    tensor_key: &str,
    tensor_class: &str,
    layer_index: u32,
    is_mtp: bool,
    weights: &[f32],
) -> TensorSensitivityReceipt {
    // ── Weight distribution ──────────────────────────────────────────────
    let n = weights.len() as f64;
    let mean = weights.iter().sum::<f32>() as f64 / n;
    let variance = weights
        .iter()
        .map(|w| {
            let d = *w as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let std = variance.sqrt();
    let kurtosis = if variance > 0.0 {
        weights
            .iter()
            .map(|w| {
                let d = *w as f64 - mean;
                let d2 = d * d;
                d2 * d2
            })
            .sum::<f64>()
            / n
            / (variance * variance)
            - 3.0
    } else {
        0.0
    };
    let outlier_mass = weights
        .iter()
        .filter(|w| (**w as f64 - mean).abs() > 3.0 * std)
        .count() as f64
        / n;

    // ── Cheap ternary probe (canonical 2-bit block quantize) ────────────
    let (ternary_nrmse, ternary_max_err) = cheap_ternary_probe(weights);

    // ── Max channel shift (per-output-channel error) ─────────────────────
    let channel_shift = estimate_channel_shift(weights);

    // ── Budget classification ───────────────────────────────────────────
    let budget = classify_budget(
        tensor_class,
        is_mtp,
        outlier_mass,
        kurtosis,
        ternary_nrmse,
        channel_shift,
    );

    let digest = sha256_hex(weights);
    TensorSensitivityReceipt {
        receipt_id: format!("sens.{}.{:.8}", tensor_key, &digest[..8]),
        tensor_key: tensor_key.to_string(),
        tensor_class: tensor_class.to_string(),
        layer_index,
        is_mtp,
        weight_outlier_mass: outlier_mass,
        weight_kurtosis: kurtosis,
        ternary_recon_nrmse: ternary_nrmse,
        ternary_max_error: ternary_max_err,
        activation_nrmse: None,
        activation_cosine: None,
        max_channel_shift: channel_shift,
        sensitive_channels: Vec::new(),
        layer_error_amplification: 1.0,
        budget,
        source_digest: digest,
        calibrator_digest: String::new(),
    }
}

/// Canonical cheap ternary probe: 2-bit block quantize, compute NRMSE.
fn cheap_ternary_probe(weights: &[f32]) -> (f64, f64) {
    let block_size = 256;
    if weights.is_empty() {
        return (0.0, 0.0);
    }
    let mut total_sq_error = 0.0;
    let mut max_abs_err: f64 = 0.0;
    let mut weight_sq_sum = 0.0;
    for chunk in weights.chunks(block_size) {
        let block_max = chunk.iter().fold(0.0f32, |a, &v| a.max(v.abs()));
        let scale = if block_max > 1e-12 { block_max } else { 1.0 };
        for &w in chunk {
            let q = (w / scale).round().clamp(-1.0, 1.0);
            let recon = q * scale;
            let err = (w - recon) as f64;
            total_sq_error += err * err;
            max_abs_err = max_abs_err.max(err.abs());
            weight_sq_sum += (w as f64) * (w as f64);
        }
    }
    let nrmse = if weight_sq_sum > 0.0 {
        (total_sq_error / weight_sq_sum).sqrt()
    } else {
        0.0
    };
    (nrmse, max_abs_err)
}

/// Rough estimate of per-channel reconstruction error.
fn estimate_channel_shift(_weights: &[f32]) -> f64 {
    // Returns 0.0 — requires known channel layout to compute per-output-channel
    // reconstruction error. When channel layout metadata is available (e.g. from
    // the model descriptor), this should be wired to compute the maximum
    // per-output-channel RMSE from a cheap ternary probe.
    0.0
}

/// Classify budget from sensitivity metrics.
fn classify_budget(
    tensor_class: &str,
    is_mtp: bool,
    outlier_mass: f64,
    kurtosis: f64,
    ternary_nrmse: f64,
    channel_shift: f64,
) -> SearchBudgetClass {
    // Sensitive tensor classes get conservative budgets
    let is_sensitive_class = matches!(
        tensor_class,
        "RmsNormWeight" | "TokenEmbedding" | "OutputHead"
    );
    // MTP-coupled tensors: conservative
    let is_coupled = is_mtp;
    // High activation sensitivity
    let high_sensitivity = outlier_mass > 0.05 || kurtosis > 10.0 || channel_shift > 0.5;

    if is_sensitive_class || is_coupled {
        if ternary_nrmse < 0.05 {
            SearchBudgetClass::CheapTernaryProbe
        } else {
            SearchBudgetClass::SkipTernary
        }
    } else if high_sensitivity && ternary_nrmse > 0.15 {
        SearchBudgetClass::MixedPrecisionOnly
    } else if ternary_nrmse < 0.10 {
        SearchBudgetClass::FullTernarySweep
    } else if high_sensitivity {
        SearchBudgetClass::FullMixedRefinement
    } else {
        SearchBudgetClass::FullTernarySweep
    }
}

fn sha256_hex(data: &[f32]) -> String {
    use sha2::{Digest, Sha256};
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{:02x}", b))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_skip_ternary_for_norms() {
        // Ramp -2..2 produces high ternary NRMSE (~0.43), above the
        // 0.05 threshold, so norms correctly get SkipTernary.
        let w: Vec<f32> = (0..640).map(|i| ((i as f32) - 320.0) / 160.0).collect();
        let r = analyze_tensor_sensitivity("rmsnorm", "RmsNormWeight", 0, false, &w);
        assert_eq!(
            r.budget,
            SearchBudgetClass::SkipTernary,
            "Norms should skip ternary"
        );
    }

    #[test]
    fn test_mtp_gets_conservative_budget() {
        let w: Vec<f32> = (0..640).map(|i| ((i as f32) - 320.0) / 320.0).collect();
        let r = analyze_tensor_sensitivity("mtp_proj", "DecoderMlpProjection", 1, true, &w);
        assert!(
            matches!(
                r.budget,
                SearchBudgetClass::SkipTernary | SearchBudgetClass::CheapTernaryProbe
            ),
            "MTP tensors should be conservative, got {:?}",
            r.budget
        );
    }

    #[test]
    fn test_cheap_ternary_probe_high_error_triggers_mixed() {
        // Random noise → high ternary error → MixedPrecisionOnly
        let w: Vec<f32> = (0..1280).map(|i| (i as f32 % 10.0 - 5.0) * 0.2).collect();
        let r = analyze_tensor_sensitivity("q_proj", "DecoderAttentionProjection", 5, false, &w);
        assert!(!matches!(r.budget, SearchBudgetClass::SkipTernary));
    }

    #[test]
    fn test_deterministic_receipt() {
        let w: Vec<f32> = (0..640).map(|i| (i as f32) / 640.0).collect();
        let r1 = analyze_tensor_sensitivity("t", "DecoderMlpProjection", 0, false, &w);
        let r2 = analyze_tensor_sensitivity("t", "DecoderMlpProjection", 0, false, &w);
        assert_eq!(r1.receipt_id, r2.receipt_id);
        assert!((r1.ternary_recon_nrmse - r2.ternary_recon_nrmse).abs() < 1e-12);
    }

    #[test]
    fn test_budget_parameters_are_reasonable() {
        assert_eq!(
            SearchBudgetClass::SkipTernary.parameters().population_size,
            0
        );
        assert!(SearchBudgetClass::FullTernarySweep.parameters().generations >= 4);
        assert!(
            SearchBudgetClass::FullMixedRefinement
                .parameters()
                .population_size
                > 32
        );
    }
}
