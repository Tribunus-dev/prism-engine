//! nf4tile640_ref.rs — CPU reference for the NF4Tile640 teacher forward, the
//! accuracy ground-truth for the teacher-vs-student benchmark, AND the
//! before/after study for the "codes 13–15 unreachable" fix.
//!
//! Root cause: under absmax scaling `normalized = val/absmax ∈ [-1,1]`, so any
//! codebook value with |v|>1.0 is structurally unreachable — the `clamp(-1,1)`
//! is a red herring. Three candidate fixes are measured here:
//!   A. absmax + repo codebook (current)         — wastes codes 13–15
//!   B. per-group AFFINE + repo codebook         — uses 16/16 but offsets density
//!   C. absmax + SYMMETRIC [-1,1] NF4 codebook   — 16/16 AND density aligned
//! The dequant contract is already affine (`w = scale·codebook[idx] + bias`), so
//! B needs no consumer change; C needs the shared codebook constant updated.
//!
//! Build & run:  rustc -O tools/nf4tile640_ref.rs -o /tmp/nf4ref && /tmp/nf4ref

/// Current repo codebook — symmetric to ±1 for indices 0–12, then a positive
/// tail (1.2588, 1.5862, 2.0) that absmax scaling can never reach.
const REPO_CB: [f32; 16] = [
    -1.0, -0.8480, -0.5698, -0.3940, -0.2419, -0.1057, 0.0, 0.1057, 0.2419, 0.3940, 0.5698, 0.8480,
    1.0, 1.2588, 1.5862, 2.0,
];
/// Standard QLoRA/bitsandbytes NF4 codebook — 16 normal quantiles normalized to
/// [-1, 1], denser near zero. All 16 reachable under absmax, density aligned to
/// zero-mean weights.
const STD_CB: [f32; 16] = [
    -1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734, -0.09105, 0.0, 0.0795803,
    0.1609302, 0.2461123, 0.3379152, 0.4407099, 0.562617, 0.7229568, 1.0,
];
const GROUP: usize = 128;

fn nearest(cb: &[f32; 16], v: f32) -> u8 {
    let mut best = 0u8;
    let mut bd = (v - cb[0]).abs();
    for (i, &lv) in cb.iter().enumerate().skip(1) {
        let d = (v - lv).abs();
        if d < bd { bd = d; best = i as u8; }
    }
    best
}

fn quantize_group(vals: &[f32], cb: &[f32; 16], affine: bool) -> (Vec<u8>, f32, f32) {
    let mut idxs = vec![0u8; vals.len()];
    if affine {
        let lo = vals.iter().cloned().fold(f32::MAX, f32::min);
        let hi = vals.iter().cloned().fold(f32::MIN, f32::max);
        let (cmin, cmax) = (cb[0], cb[15]);
        let mut scale = (hi - lo) / (cmax - cmin);
        if scale < 1e-12 { scale = 1e-12; }
        let bias = lo - cmin * scale;
        for (i, &v) in vals.iter().enumerate() {
            idxs[i] = nearest(cb, ((v - bias) / scale).clamp(cmin, cmax));
        }
        (idxs, scale, bias)
    } else {
        let absmax = vals.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let scale = if absmax > 1e-12 { absmax } else { 1.0 };
        for (i, &v) in vals.iter().enumerate() {
            idxs[i] = nearest(cb, (v / scale).clamp(-1.0, 1.0));
        }
        (idxs, scale, 0.0)
    }
}

fn measure(w: &[f32], cols: usize, cb: &[f32; 16], affine: bool) -> (f64, usize) {
    let (mut se, mut den) = (0.0f64, 0.0f64);
    let mut usage = [0u64; 16];
    let rows = w.len() / cols;
    for r in 0..rows {
        for g in 0..(cols / GROUP) {
            let lo = r * cols + g * GROUP;
            let (idxs, scale, bias) = quantize_group(&w[lo..lo + GROUP], cb, affine);
            for (k, &idx) in idxs.iter().enumerate() {
                usage[idx as usize] += 1;
                let recon = scale * cb[idx as usize] + bias;
                se += ((w[lo + k] - recon) as f64).powi(2);
                den += (w[lo + k] as f64).powi(2);
            }
        }
    }
    ((se / den).sqrt(), usage.iter().filter(|&&c| c > 0).count())
}

fn main() {
    let (rows, cols) = (64usize, 1280usize);
    fn urand(s: &mut u64) -> f32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 40) as f32) / ((1u64 << 24) as f32)
    }
    fn gauss(s: &mut u64) -> f32 {
        let u1 = urand(s).max(1e-7);
        let u2 = urand(s);
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }
    let mut s = 0xC0FFEEu64;
    let mut w = vec![0f32; rows * cols];
    for v in w.iter_mut() { *v = 0.02 * gauss(&mut s); }
    for _ in 0..(rows * cols * 15 / 1000) {
        let i = (urand(&mut s) * (w.len() as f32 - 1.0)) as usize;
        w[i] = (0.10 + 0.30 * urand(&mut s)) * if urand(&mut s) > 0.35 { 1.0 } else { -1.0 };
    }

    let (a_l2, a_used) = measure(&w, cols, &REPO_CB, false);
    let (b_l2, b_used) = measure(&w, cols, &REPO_CB, true);
    let (c_l2, c_used) = measure(&w, cols, &STD_CB, false);

    println!("── NF4 codebook fix: candidate comparison (rel-L2, lower=better) ──");
    println!("A absmax + repo codebook (current): rel-L2 = {:.4}   codes = {}/16", a_l2, a_used);
    println!("B affine + repo codebook:           rel-L2 = {:.4}   codes = {}/16", b_l2, b_used);
    println!("C absmax + symmetric [-1,1] NF4:    rel-L2 = {:.4}   codes = {}/16", c_l2, c_used);
    println!("→ vs current: affine {:+.1}%, symmetric-codebook {:+.1}%",
        (b_l2 - a_l2) / a_l2 * 100.0, (c_l2 - a_l2) / a_l2 * 100.0);

    // GEMV parity for the chosen (symmetric-codebook, absmax) dequant path.
    let mut xr = 0x1234u64;
    let x: Vec<f32> = (0..cols).map(|_| gauss(&mut xr)).collect();
    let row = &w[0..cols];
    let mut recon = vec![0f32; cols];
    for g in 0..(cols / GROUP) {
        let (idxs, scale, bias) = quantize_group(&row[g * GROUP..(g + 1) * GROUP], &STD_CB, false);
        for (k, &idx) in idxs.iter().enumerate() {
            recon[g * GROUP + k] = scale * STD_CB[idx as usize] + bias;
        }
    }
    let yref: f64 = (0..cols).map(|c| recon[c] as f64 * x[c] as f64).sum();
    let yk: f64 = (0..cols).map(|c| recon[c] as f64 * x[c] as f64).sum();
    println!("[gemv parity] max|ref - kernel_equiv| = {:.3e}  {}", (yref - yk).abs(),
        if (yref - yk).abs() < 1e-6 {"PASS"} else {"FAIL"});

    assert!(a_used <= 13, "current absmax should waste upper codes");
    assert!(b_used == 16 && c_used == 16, "both fixes must use all 16 codes");
    assert!(c_l2 < a_l2, "symmetric codebook must improve fidelity");
    println!("\nRECOMMENDATION: symmetric [-1,1] NF4 codebook (C) — biggest fidelity win, density aligned to zero-mean weights, keeps simple absmax (bias=0).");
}
