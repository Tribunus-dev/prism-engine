//! CPU math op fallback definitions.
//!
//! These are defined here so the workspace crate compiles without the
//! `prism-backend` feature (which provides them via
//! `compute-core/src/lut/evaluator.rs`).  When `prism-backend` is enabled
//! the engine imports from compute-core instead.
//!
//! TODO: Remove this file once `prism-backend` becomes a required dependency.

// ── Palettized GEMV ────────────────────────────────────────────────────

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

pub fn vec_add_inplace(a: &mut [u16], b: &[u16]) {
    for (av, &bv) in a.iter_mut().zip(b.iter()) {
        let fa = half::f16::from_bits(*av).to_f32();
        let fb = half::f16::from_bits(bv).to_f32();
        *av = half::f16::from_f32(fa + fb).to_bits();
    }
}

// ── Activation functions ───────────────────────────────────────────────

pub fn silu_inplace(x: &mut [u16]) {
    for v in x.iter_mut() {
        let f = half::f16::from_bits(*v).to_f32();
        *v = half::f16::from_f32(f / (1.0 + (-f).exp())).to_bits();
    }
}

pub fn gelu_inplace(x: &mut [u16]) {
    let s = (2.0 / std::f32::consts::PI).sqrt();
    for v in x.iter_mut() {
        let f = half::f16::from_bits(*v).to_f32();
        *v = half::f16::from_f32(0.5 * f * (1.0 + (s * (f + 0.044715 * f * f * f)).tanh()))
            .to_bits();
    }
}

// ── Attention (CPU fallback) ───────────────────────────────────────────

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
