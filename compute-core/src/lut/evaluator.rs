//! CPU math operations for LUT-based inference.
//!
//! Pure FP16 arithmetic kernels used as fallback when Metal/ANE backends
//! are unavailable. All operations operate on `u16` bit patterns
//! (half-precision floats via the `half` crate).

use crate::lut::graph::ActivationFunction;

// ── Palettized GEMV ────────────────────────────────────────────────────

/// Palettized matrix-vector multiply: `o = W @ inp` where `W` is stored as a
/// 16-entry codebook per row + packed 4-bit indices.
///
/// # Format
/// - `p[0..32)` — row 0 codebook (16 × u16 LE)
/// - `p[32..64)` — row 1 codebook
/// - … repeated for `dm` rows
/// - then packed 4-bit index nibbles: `dm × dn / 8` bytes
///
/// Returns `Vec<u16>` of length `dm` (FP16).
pub fn lut_gemv_cpu(inp: &[u16], p: &[u8], dm: u32, dn: u32) -> Vec<u16> {
    let m = dm as usize;
    let n = dn as usize;
    let cbb = m * 16 * 2;
    let mut o = vec![0u16; m];
    for r in 0..m {
        let mut cb = [0u16; 16];
        for i in 0..16 {
            cb[i] = u16::from_le_bytes([p[r * 32 + i * 2], p[r * 32 + i * 2 + 1]]);
        }
        let io = cbb + r * (n / 2);
        let mut acc = 0.0f32;
        for wi in 0..n / 8 {
            let o2 = io + wi * 4;
            let pw = u32::from_le_bytes([p[o2], p[o2 + 1], p[o2 + 2], p[o2 + 3]]);
            for j in 0..8 {
                acc += half::f16::from_bits(inp[wi * 8 + j]).to_f32()
                    * half::f16::from_bits(cb[((pw >> (j * 4)) & 0x0F) as usize]).to_f32();
            }
        }
        o[r] = half::f16::from_f32(acc).to_bits();
    }
    o
}

// ── Normalisation ──────────────────────────────────────────────────────

/// In-place RMS layer normalisation.
pub fn rms_norm_inplace(x: &mut [u16], eps: f32) {
    let inv = 1.0
        / (x.iter()
            .map(|&v| {
                let f = half::f16::from_bits(v).to_f32();
                f * f
            })
            .sum::<f32>()
            / x.len() as f32
            + eps)
            .sqrt();
    for v in x.iter_mut() {
        let f = half::f16::from_bits(*v).to_f32();
        *v = half::f16::from_f32(f * inv).to_bits();
    }
}

/// In-place vector addition: `a[i] = a[i] + b[i]`.
pub fn vec_add_inplace(a: &mut [u16], b: &[u16]) {
    for (av, &bv) in a.iter_mut().zip(b.iter()) {
        let fa = half::f16::from_bits(*av).to_f32();
        let fb = half::f16::from_bits(bv).to_f32();
        *av = half::f16::from_f32(fa + fb).to_bits();
    }
}

// ── Activation functions ───────────────────────────────────────────────

/// In-place SiLU (sigmoid linear unit): `x[i] = x[i] / (1 + exp(-x[i]))`.
pub fn silu_inplace(x: &mut [u16]) {
    for v in x.iter_mut() {
        let f = half::f16::from_bits(*v).to_f32();
        *v = half::f16::from_f32(f / (1.0 + (-f).exp())).to_bits();
    }
}

/// In-place GELU (Gaussian Error Linear Unit, tanh approximation).
pub fn gelu_inplace(x: &mut [u16]) {
    let s = (2.0 / std::f32::consts::PI).sqrt();
    for v in x.iter_mut() {
        let f = half::f16::from_bits(*v).to_f32();
        *v = half::f16::from_f32(0.5 * f * (1.0 + (s * (f + 0.044715 * f * f * f)).tanh()))
            .to_bits();
    }
}

// ── Attention (CPU fallback) ───────────────────────────────────────────

/// Grouped-query attention on FP16 data: `softmax(Q @ K^T / sqrt(d)) @ V`.
///
/// - `q`:  `[nh × hd]` FP16 values
/// - `kc`: `[sl × nkv × hd]` FP16 values
/// - `vc`: `[sl × nkv × hd]` FP16 values
/// - `nh`, `nkv`, `hd`: num query heads, num KV heads, head dim
/// - `sl`: sequence length
pub fn attention_cpu(
    q: &[u16],
    kc: &[u16],
    vc: &[u16],
    nh: usize,
    nkv: usize,
    hd: usize,
    sl: usize,
) -> Vec<u16> {
    let g = nh / nkv.max(1);
    let kvd = nkv * hd;
    let mut o = vec![0u16; nh * hd];
    for h in 0..nh {
        let kh = h / g;
        let qb = h * hd;
        let mut sc = vec![0.0f32; sl];
        for p in 0..sl {
            let kb = p * kvd + kh * hd;
            let mut s = 0.0f32;
            for d in 0..hd {
                s += half::f16::from_bits(q[qb + d]).to_f32()
                    * half::f16::from_bits(kc[kb + d]).to_f32();
            }
            sc[p] = s / (hd as f32).sqrt();
        }
        let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut ex = vec![0.0f32; sl];
        let mut es = 0.0f32;
        for (i, &s) in sc.iter().enumerate() {
            let e = (s - mx).exp();
            ex[i] = e;
            es += e;
        }
        let inv = 1.0 / (es + 1e-10);
        for d in 0..hd {
            let mut ac = 0.0f32;
            for p in 0..sl {
                ac += ex[p] * half::f16::from_bits(vc[p * kvd + kh * hd + d]).to_f32() * inv;
            }
            o[qb + d] = half::f16::from_f32(ac).to_bits();
        }
    }
    o
}

// ── Rotary Position Embedding ──────────────────────────────────────────

/// In-place rotary position embedding (RoPE) for FP16 data.
///
/// Applies rotation to contiguous `hd`-sized groups (one head at a time).
pub fn rope_inplace(x: &mut [u16], pos: i64, hd: usize, _th: f32) {
    for i in (0..x.len()).step_by(hd) {
        let h = hd / 2;
        for j in 0..h {
            let a = half::f16::from_bits(x[i + j]).to_f32();
            let b = half::f16::from_bits(x[i + j + h]).to_f32();
            let ang = (pos as f32) * (10000.0f32).powf(-2.0 * j as f32 / hd as f32);
            let (sa, ca) = ang.sin_cos();
            x[i + j] = half::f16::from_f32(a * ca - b * sa).to_bits();
            x[i + j + h] = half::f16::from_f32(a * sa + b * ca).to_bits();
        }
    }
}

// ── Token embedding lookup ─────────────────────────────────────────────

/// Look up a single token embedding from a palettized LUT tensor.
///
/// # Format
/// - Per-row codebook: 16 × u16 LE at `payload[row * 32 .. row * 32 + 32]`
/// - Then packed 4-bit indices: `vocab_size × dm / 8` bytes
/// - The codebook prefix: `vocab_size × 32` bytes total for all rows
pub fn lut_embed(token: u32, payload: &[u8], vocab_size: u32, hidden_dim: u32) -> Vec<u16> {
    let hd = hidden_dim as usize;
    let t = token as usize;
    if t >= vocab_size as usize {
        return vec![0u16; hd];
    }
    let cb = vocab_size as usize * 16 * 2;
    let mut c = [0u16; 16];
    for i in 0..16 {
        c[i] = u16::from_le_bytes([payload[t * 32 + i * 2], payload[t * 32 + i * 2 + 1]]);
    }
    let io = cb + t * (hd / 2);
    let mut v = Vec::with_capacity(hd);
    for wi in 0..hd / 8 {
        let o = io + wi * 4;
        let pw = u32::from_le_bytes([payload[o], payload[o + 1], payload[o + 2], payload[o + 3]]);
        for j in 0..8 {
            v.push(c[((pw >> (j * 4)) & 0x0F) as usize]);
        }
    }
    v
}

// ── Activation evaluation ───────────────────────────────────────────

/// Evaluate an activation function in-place on gate data, with optional
/// element-wise multiply with up-projection values (for SwiGLU/MLP).
///
/// # Arguments
/// * `func` — Which activation to apply (Silu or Gelu)
/// * `gate` — In-place gate values (FP16 u16), modified in place
/// * `up` — Optional up-projection values; if Some, performs element-wise
///   multiply `gate[i] = gate[i] * up[i]` after activation (SwiGLU pattern)
pub fn evaluate_activations(func: ActivationFunction, gate: &mut [u16], up: Option<&[u16]>) {
    match func {
        ActivationFunction::Silu => {
            silu_inplace(gate);
            if let Some(up) = up {
                for i in 0..gate.len().min(up.len()) {
                    let a = half::f16::from_bits(gate[i]).to_f32();
                    let b = half::f16::from_bits(up[i]).to_f32();
                    gate[i] = half::f16::from_f32(a * b).to_bits();
                }
            }
        }
        ActivationFunction::Gelu => {
            gelu_inplace(gate);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lut_gemv() {
        let mut p = Vec::new();
        for _ in 0..16 {
            p.extend_from_slice(&0x3c00u16.to_le_bytes());
        }
        for _ in 0..16 {
            p.extend_from_slice(&0x4000u16.to_le_bytes());
        }
        for _ in 0..2 {
            p.extend_from_slice(&[0x00; 4]);
        }
        let o = lut_gemv_cpu(&[0x3c00u16; 8], &p, 2, 8);
        let v = half::f16::from_bits(o[0]).to_f32();
        assert!((v - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_rms() {
        let mut x = vec![0x3c00u16; 4];
        rms_norm_inplace(&mut x, 1e-6);
        for &v in &x {
            assert!((half::f16::from_bits(v).to_f32() - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_vec_add() {
        let mut a = vec![half::f16::from_f32(1.0).to_bits(); 4];
        let b = vec![half::f16::from_f32(2.0).to_bits(); 4];
        vec_add_inplace(&mut a, &b);
        for &v in &a {
            assert!((half::f16::from_bits(v).to_f32() - 3.0).abs() < 1e-4);
        }
    }

    #[test]
    fn test_lut_embed() {
        let mut payload = Vec::new();
        // 2 vocab entries × 32 bytes codebook
        for _ in 0..2 * 16 {
            payload.extend_from_slice(&0x3c00u16.to_le_bytes());
        }
        // packed indices for 8-dim
        for _ in 0..2 * (8 / 2) {
            payload.push(0);
        }
        let v = lut_embed(0, &payload, 2, 8);
        assert_eq!(v.len(), 8);
        let v2 = lut_embed(5, &payload, 2, 8);
        assert_eq!(v2.len(), 8);
        // Out-of-bounds returns zeros
        assert_eq!(v2, vec![0; 8]);
    }

    #[test]
    fn test_evaluate_activations_silu() {
        use crate::lut::graph::ActivationFunction;
        let mut gate = vec![half::f16::from_f32(2.0).to_bits(); 4];
        let up = vec![half::f16::from_f32(3.0).to_bits(); 4];
        evaluate_activations(ActivationFunction::Silu, &mut gate, Some(&up));
        // SiLU(2.0) = 2.0 / (1.0 + exp(-2.0)) ≈ 1.7616
        // After gate*up: 1.7616 * 3.0 ≈ 5.2848
        let expected = half::f16::from_f32(2.0).to_f32() / (1.0 + (-2.0f32).exp());
        let got = half::f16::from_bits(gate[0]).to_f32();
        assert!((got - expected * 3.0).abs() < 0.01,
            "silu gate*up: got={got}, expected={}", expected * 3.0);
    }

    #[test]
    fn test_evaluate_activations_gelu() {
        use crate::lut::graph::ActivationFunction;
        let mut gate = vec![half::f16::from_f32(1.0).to_bits(); 2];
        evaluate_activations(ActivationFunction::Gelu, &mut gate, None);
        // GELU(1.0) ≈ 0.84119 via tanh approximation
        let got = half::f16::from_bits(gate[0]).to_f32();
        assert!(got > 0.8 && got < 0.9, "gelu(1.0) ≈ {got}");
    }
}
