//! Ternary tile640 quantization pipeline (v7) — std-only and engine-decoupled.
//!
//! Layered tile640 ternary quantization with absmean scale, per-lane
//! micro-scaling, optional outlier extraction, and a two-level scale
//! encoding (bf16 page-max + int8 per-lane relative).
//!
//! Authority: tile640 ternary quantization math. Std-only — no engine,
//! no MLX, no Metal. Engine-coupled wrappers (kernel dispatch, GPU
//! packers) live in the engine's `legacy_compute_image_compile/`.

/// Number of trits packed into one u32 (base-3 encoding).
pub const LANE: usize = 20; // trits per packed u32 word
/// Number of weights per tile (= LANE × 32 lanes).
pub const PAGE: usize = 640; // 32 lanes × 20

/// Rounding strategy for the deadzone quantizer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rounding {
    /// Standard b1.58 deadzone rounding.
    DeadzoneAbsmean,
    /// Stochastic rounding (probability proportional to |w|/γ).
    Stochastic,
}

/// Quantization configuration.
#[derive(Clone, Copy, Debug)]
pub struct QuantConfig {
    /// Rounding strategy.
    pub rounding: Rounding,
    /// Deadzone as a fraction of γ; 0.5 = standard b1.58.
    pub tau: f32,
    /// Error diffusion (scoped strictly within a 20-weight lane).
    pub error_diffusion: bool,
    /// Least-squares α* on locked non-zero trits.
    pub optimize_scale: bool,
    /// Fraction of |w| per page pulled to bf16 sidecar.
    pub outlier_frac: f32,
    /// PRNG seed for stochastic rounding.
    pub seed: u64,
}

impl Default for QuantConfig {
    /// Evidence-backed defaults (see module docs / quant_lab A/B).
    fn default() -> Self {
        Self {
            rounding: Rounding::DeadzoneAbsmean,
            tau: 0.5,               // standard threshold; τ<0.5 gated for A/B
            error_diffusion: false, // did not help on synthetic; gated
            optimize_scale: true,
            outlier_frac: 0.005, // the dominant fidelity lever on real weights
            seed: 0x9E3779B9,
        }
    }
}

/// A quantized weight tensor.
#[derive(Default)]
pub struct QuantizedTensor {
    /// Output dimension (rows).
    pub out_dim: usize,
    /// Input dimension (cols).
    pub in_dim: usize,
    /// Packed ternary words: `out_dim × ceil(in_dim/640) × 32`.
    pub packed: Vec<u32>,
    /// FP16 page-max scales (one per page).
    pub page_scales: Vec<u16>,
    /// Int8 relative lane scales (one per 20-weight lane).
    pub lane_scales: Vec<u8>,
    /// Outliers: `(flat_index, bf16_bits)` pairs.
    pub outliers: Vec<(u32, u16)>,
}

/// Convert f32 to BF16 bits (round-to-nearest-even).
#[inline]
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    if !x.is_finite() {
        // preserve inf/nan sign/quiet — bf16 shares f32's exponent range.
        return (x.to_bits() >> 16) as u16;
    }
    let b = x.to_bits();
    let round = 0x7fff + ((b >> 16) & 1); // round-to-nearest-even
    ((b.wrapping_add(round)) >> 16) as u16
}

/// Convert BF16 bits to f32.
#[inline]
pub fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

/// Quantize one weight matrix (row-major `[out_dim × in_dim]`) to v7 artifacts.
pub fn quantize_tensor(
    w: &[f32],
    out_dim: usize,
    in_dim: usize,
    cfg: &QuantConfig,
) -> QuantizedTensor {
    assert_eq!(w.len(), out_dim * in_dim, "weight slice length mismatch");
    let nt = (in_dim + PAGE - 1) / PAGE; // pages per row
    let lanes_per_page = PAGE / LANE; // 32
    let mut out = QuantizedTensor {
        out_dim,
        in_dim,
        ..Default::default()
    };
    let mut rng = Lcg(cfg.seed);

    for r in 0..out_dim {
        for p in 0..nt {
            let col0 = p * PAGE;
            let cols = PAGE.min(in_dim - col0);
            // gather page weights
            let mut pw = vec![0.0f32; cols];
            for c in 0..cols {
                pw[c] = w[r * in_dim + col0 + c];
            }

            // 1. Outlier extraction (per page) — must precede scale calc.
            let mut extracted = vec![false; cols];
            if cfg.outlier_frac > 0.0 {
                let k = ((cfg.outlier_frac * cols as f32).round() as usize).min(cols);
                if k > 0 {
                    let mut idx: Vec<usize> = (0..cols).collect();
                    idx.sort_by(|&a, &b| {
                        pw[b]
                            .abs()
                            .partial_cmp(&pw[a].abs())
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    for &c in idx.iter().take(k) {
                        extracted[c] = true;
                        out.outliers
                            .push(((r * in_dim + col0 + c) as u32, f32_to_bf16_bits(pw[c])));
                    }
                }
            }

            // Per-page two-level scale: gather each lane's γ, then encode.
            let mut lane_gamma = vec![0.0f32; lanes_per_page];
            let mut lane_trits: Vec<[i8; LANE]> = vec![[0i8; LANE]; lanes_per_page];

            for l in 0..lanes_per_page {
                let lo = l * LANE;
                if lo >= cols {
                    lane_gamma[l] = 0.0;
                    continue;
                }
                let hi = (lo + LANE).min(cols);

                // 2. Per-lane absmean over non-outlier weights.
                let (mut s, mut cnt) = (0.0f32, 0usize);
                for c in lo..hi {
                    if !extracted[c] {
                        s += pw[c].abs();
                        cnt += 1;
                    }
                }
                let mut gamma = if cnt > 0 {
                    (s / cnt as f32).max(1e-12)
                } else {
                    1e-12
                };

                // 3. Rounding (+ error diffusion), scoped WITHIN this lane only.
                let mut carry = 0.0f32;
                for c in lo..hi {
                    if extracted[c] {
                        carry = 0.0;
                        continue;
                    }
                    let we = pw[c] + if cfg.error_diffusion { carry } else { 0.0 };
                    let t: i8 = match cfg.rounding {
                        Rounding::DeadzoneAbsmean => {
                            if we.abs() > cfg.tau * gamma {
                                if we > 0.0 {
                                    1
                                } else {
                                    -1
                                }
                            } else {
                                0
                            }
                        }
                        Rounding::Stochastic => {
                            let pr = (we.abs() / gamma).clamp(0.0, 1.0);
                            if rng.unit() < pr {
                                if we > 0.0 {
                                    1
                                } else {
                                    -1
                                }
                            } else {
                                0
                            }
                        }
                    };
                    lane_trits[l][c - lo] = t;
                    if cfg.error_diffusion {
                        carry = we - (t as f32) * gamma;
                    }
                }
                // carry deliberately dropped at the lane boundary.

                // 4. Least-squares optimal scale on locked non-zero trits.
                if cfg.optimize_scale {
                    let (mut num, mut den) = (0.0f32, 0.0f32);
                    for c in lo..hi {
                        let t = lane_trits[l][c - lo];
                        if t != 0 {
                            num += pw[c] * t as f32;
                            den += 1.0;
                        }
                    }
                    if den > 0.0 {
                        gamma = (num / den).abs().max(1e-12);
                    }
                }
                lane_gamma[l] = gamma;
            }

            // Two-level encode: bf16 page-max + int8 per-lane relative.
            let page_max = lane_gamma
                .iter()
                .cloned()
                .fold(0.0f32, f32::max)
                .max(1e-12);
            out.page_scales.push(f32_to_bf16_bits(page_max));
            for l in 0..lanes_per_page {
                let q = ((lane_gamma[l] / page_max) * 127.0)
                    .round()
                    .clamp(0.0, 127.0) as u8;
                out.lane_scales.push(q);
            }

            // Pack trits into tile640 base-3 words (20 trits / u32, LSB first).
            for l in 0..lanes_per_page {
                let mut word: u32 = 0;
                let mut pow: u32 = 1;
                for vi in 0..LANE {
                    let d: u32 = match lane_trits[l][vi] {
                        1 => 1,
                        -1 => 2,
                        _ => 0,
                    };
                    word = word.wrapping_add(d.wrapping_mul(pow));
                    pow = pow.wrapping_mul(3);
                }
                out.packed.push(word);
            }
        }
    }
    out
}

/// Dequantize (reference / for tests + oracle parity). Reconstructs the dense
/// ternary contribution; add outliers separately.
pub fn dequantize(qt: &QuantizedTensor) -> Vec<f32> {
    let nt = (qt.in_dim + PAGE - 1) / PAGE;
    let lanes_per_page = PAGE / LANE;
    let mut w = vec![0.0f32; qt.out_dim * qt.in_dim];
    for r in 0..qt.out_dim {
        for p in 0..nt {
            let page_idx = r * nt + p;
            let page_max = bf16_bits_to_f32(qt.page_scales[page_idx]);
            let col0 = p * PAGE;
            for l in 0..lanes_per_page {
                let scale = page_max * (qt.lane_scales[page_idx * lanes_per_page + l] as f32 / 127.0);
                let word = qt.packed[page_idx * lanes_per_page + l];
                let mut rem = word;
                for vi in 0..LANE {
                    let d = rem % 3;
                    rem /= 3;
                    let col = col0 + l * LANE + vi;
                    if col >= qt.in_dim {
                        break;
                    }
                    let tv = if d == 1 {
                        1.0
                    } else if d == 2 {
                        -1.0
                    } else {
                        0.0
                    };
                    w[r * qt.in_dim + col] = tv * scale;
                }
            }
        }
    }
    // outlier add-back (dense reconstruction)
    for &(idx, bits) in &qt.outliers {
        w[idx as usize] = bf16_bits_to_f32(bits);
    }
    w
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_roundtrip_range() {
        for &x in &[0.0f32, 1.0, -0.02, 3.5e3, -1.7e-3] {
            let y = bf16_bits_to_f32(f32_to_bf16_bits(x));
            assert!((x - y).abs() <= 0.01 * x.abs().max(1e-3), "{x} -> {y}");
        }
    }

    #[test]
    fn pack_unpack_roundtrip_and_scale() {
        let (o, i) = (4usize, 1280usize);
        let mut w = vec![0.0f32; o * i];
        let mut s = 0x1234u64;
        for v in w.iter_mut() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.1;
        }
        let qt = quantize_tensor(&w, o, i, &QuantConfig::default());
        assert_eq!(qt.packed.len(), o * (i / PAGE) * (PAGE / LANE));
        assert_eq!(qt.lane_scales.len(), qt.packed.len());
        let recon = dequantize(&qt);
        let (mut se, mut den) = (0.0f64, 0.0f64);
        for k in 0..w.len() {
            se += (w[k] - recon[k]).powi(2) as f64;
            den += (w[k] as f64).powi(2);
        }
        let rel = (se / den).sqrt();
        assert!(rel < 0.9, "rel L2 {rel} unexpectedly high");
    }

    #[test]
    fn error_diffusion_stays_within_lane() {
        let (o, i) = (1usize, 640usize);
        let mut w = vec![0.0f32; i];
        w[LANE - 1] = 1.0;
        let cfg = QuantConfig {
            error_diffusion: true,
            outlier_frac: 0.0,
            ..QuantConfig::default()
        };
        let qt = quantize_tensor(&w, o, i, &cfg);
        assert_eq!(
            qt.packed[1], 0,
            "error diffusion leaked across the lane boundary"
        );
    }
}
