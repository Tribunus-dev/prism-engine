//! nf4tile640_ref.rs — CPU reference for the NF4Tile640 teacher forward, and
//! the accuracy ground-truth for the teacher-vs-student benchmark.
//!
//! Validates the exact dequant + GEMV the Metal NF4 kernel must reproduce, using
//! the real NF4 codebook and packing convention read from the compiler:
//!   * codebook: the 16 normal quantiles in compile/quantize.rs::NF4_CODEBOOK
//!   * packing:  2 NF4 indices per byte, LOW nibble = even element
//!               (coreai_pipeline.rs: codebook[byte & 0x0F] then codebook[byte>>4])
//!   * scale:    one f32 per 128-element group (absmax), bias = 0 (symmetric)
//!   * dequant:  w = NF4_CODEBOOK[idx] * scale[group]
//!
//! Build & run:  rustc -O tools/nf4tile640_ref.rs -o /tmp/nf4ref && /tmp/nf4ref

const NF4_CODEBOOK: [f32; 16] = [
    -1.0, -0.8480, -0.5698, -0.3940, -0.2419, -0.1057, 0.0, 0.1057, 0.2419, 0.3940, 0.5698, 0.8480,
    1.0, 1.2588, 1.5862, 2.0,
];
const GROUP: usize = 128;

fn nearest_nf4(normalized: f32) -> u8 {
    // nearest over the FULL codebook (correct NF4). NOTE: the CPU-fallback
    // quantize_nf4_group clamps to [-1,1] first, which makes codes 13–15
    // (values >1.0) unreachable — a quantizer quirk worth fixing separately.
    let mut best = 0u8;
    let mut bd = (normalized - NF4_CODEBOOK[0]).abs();
    for (i, &lv) in NF4_CODEBOOK.iter().enumerate().skip(1) {
        let d = (normalized - lv).abs();
        if d < bd { bd = d; best = i as u8; }
    }
    best
}

/// Quantize one row to NF4Tile640: returns (packed u8 [cols/2], scales [cols/128]).
fn quantize_row(w: &[f32]) -> (Vec<u8>, Vec<f32>) {
    let cols = w.len();
    let ngroups = cols / GROUP;
    let mut scales = vec![0f32; ngroups];
    let mut packed = vec![0u8; cols / 2];
    for g in 0..ngroups {
        let lo = g * GROUP;
        let absmax = w[lo..lo + GROUP].iter().fold(0f32, |a, &v| a.max(v.abs()));
        let scale = if absmax > 1e-12 { absmax } else { 1.0 };
        scales[g] = scale;
        for e in lo..lo + GROUP {
            let idx = nearest_nf4(w[e] / scale);
            let byte = e / 2;
            if e % 2 == 0 {
                packed[byte] |= idx; // low nibble = even element
            } else {
                packed[byte] |= idx << 4;
            }
        }
    }
    (packed, scales)
}

/// Kernel-equivalent dequant + GEMV in fp32: y[o] = Σ_e codebook[idx]*scale · x[e].
fn nf4_gemv(packed: &[u8], scales: &[f32], x: &[f32], cols: usize) -> f32 {
    let mut acc = 0f32;
    for e in 0..cols {
        let byte = packed[e / 2];
        let idx = if e % 2 == 0 { byte & 0x0F } else { (byte >> 4) & 0x0F } as usize;
        let w = NF4_CODEBOOK[idx] * scales[e / GROUP];
        acc += w * x[e];
    }
    acc
}

fn main() {
    let (rows, cols) = (64usize, 1280usize); // cols multiple of 128 and 640
    fn urand(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 40) as f32) / ((1u64 << 24) as f32) // [0,1)
    }
    fn gauss(s: &mut u64) -> f32 {
        let u1 = urand(s).max(1e-7);
        let u2 = urand(s);
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
    let mut s = 0xC0FFEEu64;
    let w: Vec<f32> = (0..rows * cols).map(|_| 0.02 * gauss(&mut s)).collect();
    let x: Vec<f32> = (0..cols).map(|_| gauss(&mut s)).collect();

    // Per-row quantize, then compare dequant+GEMV vs direct reference and
    // measure NF4 reconstruction error (the teacher's own quant fidelity).
    let (mut max_gemv_err, mut se, mut den) = (0f64, 0f64, 0f64);
    for r in 0..rows {
        let row = &w[r*cols..(r+1)*cols];
        let (packed, scales) = quantize_row(row);
        // direct reference: dequant to a dense row, dot with x
        let mut yref = 0f64;
        let mut recon = vec![0f32; cols];
        for e in 0..cols {
            let byte = packed[e/2];
            let idx = if e%2==0 { byte & 0x0F } else { (byte>>4)&0x0F } as usize;
            recon[e] = NF4_CODEBOOK[idx]*scales[e/GROUP];
            yref += (recon[e] as f64)*(x[e] as f64);
            se += ((row[e]-recon[e]) as f64).powi(2); den += (row[e] as f64).powi(2);
        }
        let yk = nf4_gemv(&packed, &scales, &x, cols) as f64;
        max_gemv_err = max_gemv_err.max((yref-yk).abs());
    }
    let rel_l2 = (se/den).sqrt();
    println!("[gemv parity] max|ref - kernel_equiv| = {:.3e}  {}", max_gemv_err, if max_gemv_err<1e-3 {"PASS"} else {"FAIL"});
    println!("[accuracy] NF4 reconstruction rel-L2 vs fp32 = {:.4}  (teacher ground-truth fidelity)", rel_l2);
    // NF4 should represent Gaussian weights far better than ternary (~0.5):
    println!("[context] ternary student rel-L2 was ~0.5 in quant_lab; NF4 teacher should be far lower.");
    assert!(max_gemv_err < 1e-3, "gemv parity failed");
    assert!(rel_l2 < 0.15, "NF4 reconstruction worse than expected");
    println!("\nNF4TILE640 REFERENCE VERIFIED");
}
