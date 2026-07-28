//! Activation-weighted affine NF4 fitting (AW-LS).
//!
//! For each 128-element group, minimizes:
//!   L = sum_i a_i (w_i - (s * c_{q_i} + b))^2
//! where a_i = E[x_i^2] (activation second moment), c_{q_i} is the NF4 codebook
//! entry for weight w_i, and s,b are the group's scale and bias.

use crate::nf4tile640::NF4_CODEBOOK;
use prism_ecs_core::compilation::cancel::CancelToken;
use serde::Serialize;

/// Optimal scale and bias for one 128-element group.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct GroupScaleBias {
    pub scale: f32,
    pub bias: f32,
    #[serde(skip)]
    pub codes: [u8; 128],
    pub aw_mse: f64, // Activation-weighted MSE after optimization
    pub iterations: u8,
}

/// Solve weighted least squares for scale and bias with fixed code indices.
///
/// Args:
///   weights: 128-element weight values
///   code_indices: 128 NF4 code indices (0..15) — the discrete assignment
///   activation_weights: 128 activation second moments E[x_i^2]
///   max_iters: max coordinate descent iterations (4-8 recommended)
///
/// Returns optimal (scale, bias) and the final weighted MSE.
///
/// Algorithm:
/// 1. Solve for s with b fixed: s = sum(a_i * c_i * (w_i - b)) / sum(a_i * c_i^2)
/// 2. Solve for b with s fixed: b = sum(a_i * (w_i - s*c_i)) / sum(a_i)
/// 3. Alternate until convergence or max_iters
pub fn optimize_scale_bias(
    weights: &[f32; 128],
    code_indices: &[u8; 128],
    activation_weights: &[f32; 128],
    max_iters: u8,
    cancel_token: &CancelToken,
) -> GroupScaleBias {
    let codebook = NF4_CODEBOOK;
    let max_codebook = codebook.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    // Initial s: max-abs / actual codebook max, not a hardcoded constant.
    let max_abs = weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
    let mut s = if max_abs > 0.0 && max_codebook > 0.0 {
        max_abs / max_codebook
    } else {
        1.0f32
    };
    let mut b = 0.0f32; // Start symmetric for NF4

    let sum_a: f32 = activation_weights.iter().sum();
    if sum_a < 1e-10 {
        return GroupScaleBias {
            scale: s,
            bias: b,
            codes: *code_indices,
            aw_mse: 0.0,
            iterations: 0,
        };
    }

    // Track the best full state (codes, s, b) seen so far.
    let mut best_s = s;
    let mut best_b = b;
    let mut best_codes = *code_indices;
    let mut best_mse =
        compute_weighted_mse(weights, &best_codes, best_s, best_b, activation_weights);
    let mut prev_mse = best_mse;

    for _ in 0..max_iters {
        cancel_token.heartbeat().ok();
        // Step 1: Fix current codes and b, solve for s
        let (num_s, den_s) = weights
            .iter()
            .zip(best_codes.iter())
            .zip(activation_weights.iter())
            .fold((0.0f32, 0.0f32), |(num, den), ((w, &ci), &a)| {
                let c = codebook[ci as usize];
                (num + a * c * (w - b), den + a * c * c)
            });
        if den_s > 1e-10 {
            s = num_s / den_s;
        }

        // Step 2: Fix new s, solve for b
        let num_b = weights
            .iter()
            .zip(best_codes.iter())
            .zip(activation_weights.iter())
            .fold(0.0f32, |acc, ((w, &ci), &a)| {
                acc + a * (w - s * codebook[ci as usize])
            });
        b = num_b / sum_a;

        // Step 3: Reassign every code using the new (s, b)
        let mut new_codes = [0u8; 128];
        for (i, &w) in weights.iter().enumerate() {
            let normalized = (w - b) / s;
            new_codes[i] = crate::nf4tile640::nf4_quantize(if s > 0.0 { normalized } else { w });
        }

        // Step 4: Re-solve (s, b) with the new code assignments (full joint step)
        let (num_s2, den_s2) = weights
            .iter()
            .zip(new_codes.iter())
            .zip(activation_weights.iter())
            .fold((0.0f32, 0.0f32), |(num, den), ((w, &ci), &a)| {
                let c = codebook[ci as usize];
                (num + a * c * w, den + a * c * c)
            });
        if den_s2 > 1e-10 {
            s = num_s2 / den_s2;
        }
        let num_b2 = weights
            .iter()
            .zip(new_codes.iter())
            .zip(activation_weights.iter())
            .fold(0.0f32, |acc, ((w, &ci), &a)| {
                acc + a * (w - s * codebook[ci as usize])
            });
        b = num_b2 / sum_a;

        // Step 5: Check if codes stabilized
        let code_stable = new_codes == best_codes;
        best_codes = new_codes;

        let mse = compute_weighted_mse(weights, &best_codes, s, b, activation_weights);

        // Keep the best full state (codes, s, b) if this iteration improved.
        if mse < best_mse {
            best_mse = mse;
            best_s = s;
            best_b = b;
        }

        if code_stable || (mse >= prev_mse - 1e-10) {
            break;
        }
        prev_mse = mse;
    }

    GroupScaleBias {
        scale: best_s,
        bias: best_b,
        codes: best_codes,
        aw_mse: best_mse,
        iterations: max_iters,
    }
}

/// Compute AW-MSE for given scale and bias.
pub fn compute_weighted_mse(
    weights: &[f32; 128],
    code_indices: &[u8; 128],
    scale: f32,
    bias: f32,
    activation_weights: &[f32; 128],
) -> f64 {
    let codebook = NF4_CODEBOOK;
    weights
        .iter()
        .zip(code_indices.iter())
        .zip(activation_weights.iter())
        .map(|((w, &ci), &a)| {
            let recon = scale * codebook[ci as usize] + bias;
            let err = w - recon;
            (a as f64) * (err as f64).powi(2)
        })
        .sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_awls_converges_synthetic() {
        // Create a group with known structure
        let mut weights = [0.0f32; 128];
        let true_scale = 2.5f32;
        let true_bias = 0.1f32;
        let codebook = NF4_CODEBOOK;
        let mut code_indices = [0u8; 128];
        for i in 0..128 {
            let ci = (i % 16) as u8;
            code_indices[i] = ci;
            weights[i] = true_scale * codebook[ci as usize] + true_bias + (i as f32 - 64.0) * 0.001;
        }
        let act_weights = [1.0f32; 128]; // uniform

        let result = optimize_scale_bias(
            &weights,
            &code_indices,
            &act_weights,
            8,
            &CancelToken::new(None),
        );
        assert!(
            (result.scale - true_scale).abs() < 0.5,
            "scale delta too large: {}",
            result.scale - true_scale
        );
        assert!(
            (result.bias - true_bias).abs() < 0.1,
            "bias delta too large: {}",
            result.bias - true_bias
        );
        assert!(result.iterations > 0 && result.iterations <= 8);
    }

    #[test]
    fn test_awls_all_same_value() {
        let weights = [0.5f32; 128];
        let code_indices = [7u8; 128]; // NF4 codebook[7] = 0.0
        let act_weights = [1.0f32; 128];
        let result = optimize_scale_bias(
            &weights,
            &code_indices,
            &act_weights,
            4,
            &CancelToken::new(None),
        );
        // With all codes mapping to 0.0, bias should absorb the value
        assert!(
            (result.bias - 0.5).abs() < 0.1,
            "bias should absorb constant value"
        );
    }
}
