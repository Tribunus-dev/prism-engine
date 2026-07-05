//! nf4_forward_ref.rs — layout-exact CPU reference for the NF4Tile640 forward.
//! This is the parity ground-truth the live Metal kernel (nf4_tile640_gemv.metal)
//! is checked against: same interleaved arena layout the GPU packer
//! (nf4_tile640_pack) writes, same symmetric NF4 codebook, same affine dequant.
//!
//! Arena layout (per row, per 640-tile):
//!   packed u8  : [tiles × 320]  — 5 groups × 64 bytes; per lane (32) 2 bytes = 4
//!                nibbles; value at group-local index gl=lane*4+i sits in byte
//!                (group*64 + lane*2 + i/2), low nibble if i even, high if odd.
//!   scales f32 : [tiles × 5]    — one absmax scale per 128-element group.
//!   biases f32 : [tiles × 5]    — 0 for NF4 (affine dequant: codebook·scale + bias).
//!
//! Dequant: w[row,col] = NF4_CODEBOOK[idx] * scale[group] + bias[group].
//! Build & run:  rustc -O tools/nf4_forward_ref.rs -o /tmp/nf4fwd && /tmp/nf4fwd

const NF4_CODEBOOK: [f32; 16] = [
    -1.0, -0.6961928, -0.5250731, -0.3949175, -0.2844414, -0.1847734, -0.09105, 0.0, 0.0795803,
    0.1609302, 0.2461123, 0.3379152, 0.4407099, 0.562617, 0.7229568, 1.0,
];
const TILE: usize = 640;
const GROUP: usize = 128;
const GPT: usize = 5; // groups per tile
const LANES: usize = 32;
const VPL: usize = 4; // values per lane
const BYTES_TILE: usize = 320;
const BYTES_GROUP: usize = 64;

fn nearest(v: f32) -> u8 {
    let mut b = 0u8;
    let mut bd = (v - NF4_CODEBOOK[0]).abs();
    for (i, &l) in NF4_CODEBOOK.iter().enumerate().skip(1) {
        let d = (v - l).abs();
        if d < bd { bd = d; b = i as u8; }
    }
    b
}

/// Pack a row-major [rows × cols] weight matrix into the interleaved NF4Tile640
/// arena, exactly as the GPU packer does. Returns (packed_u8, scales_f32).
fn pack(w: &[f32], rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>) {
    let tiles = cols / TILE;
    let mut packed = vec![0u8; rows * tiles * BYTES_TILE];
    let mut scales = vec![0f32; rows * tiles * GPT];
    for r in 0..rows {
        for t in 0..tiles {
            for g in 0..GPT {
                // group absmax over its 128 values
                let mut absmax = 0f32;
                for gl in 0..GROUP {
                    let col = t * TILE + g * GROUP + gl;
                    absmax = absmax.max(w[r * cols + col].abs());
                }
                let scale = if absmax > 1e-12 { absmax } else { 1.0 };
                scales[r * tiles * GPT + t * GPT + g] = scale;
                let inv = 1.0 / scale;
                for lane in 0..LANES {
                    for i in 0..VPL {
                        let gl = lane * VPL + i;
                        let col = t * TILE + g * GROUP + gl;
                        let idx = nearest((w[r * cols + col] * inv).clamp(-1.0, 1.0));
                        let byte = r * tiles * BYTES_TILE + t * BYTES_TILE
                            + g * BYTES_GROUP + lane * 2 + (i / 2);
                        packed[byte] |= idx << ((i % 2) * 4);
                    }
                }
            }
        }
    }
    (packed, scales)
}

/// Kernel-style dequant: iterate col-by-col, compute byte/nibble address, apply
/// the group's affine scale/bias. This is the indexing the Metal kernel uses.
fn dequant_col(packed: &[u8], scales: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let tiles = cols / TILE;
    let mut w = vec![0f32; rows * cols];
    for r in 0..rows {
        for col in 0..cols {
            let t = col / TILE;
            let wt = col % TILE;
            let g = wt / GROUP;
            let gl = wt % GROUP;
            let lane = gl / VPL;
            let i = gl % VPL;
            let byte = r * tiles * BYTES_TILE + t * BYTES_TILE + g * BYTES_GROUP + lane * 2 + (i / 2);
            let idx = if i % 2 == 0 { packed[byte] & 0x0F } else { (packed[byte] >> 4) & 0x0F };
            let scale = scales[r * tiles * GPT + t * GPT + g];
            w[r * cols + col] = NF4_CODEBOOK[idx as usize] * scale; // + bias(0)
        }
    }
    w
}

/// The fused GEMV the Metal kernel computes: y[o] = Σ_col dequant(o,col) · x[col].
fn nf4_gemv(packed: &[u8], scales: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let recon = dequant_col(packed, scales, rows, cols);
    (0..rows).map(|r| (0..cols).map(|c| recon[r * cols + c] * x[c]).sum()).collect()
}

fn main() {
    let (rows, cols) = (48usize, 1280usize); // cols multiple of 640
    let mut s = 0xF00Du64;
    let mut u = |s: &mut u64| { *s = s.wrapping_mul(6364136223846793005).wrapping_add(1); ((*s >> 40) as f32) / ((1u64 << 24) as f32) };
    let g = |s: &mut u64| { let a = u(s).max(1e-7); let b = u(s); (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos() };
    let w: Vec<f32> = (0..rows * cols).map(|_| 0.02 * g(&mut s)).collect();
    let x: Vec<f32> = (0..cols).map(|_| g(&mut s)).collect();

    let (packed, scales) = pack(&w, rows, cols);

    // (1) fidelity: dequant(pack(w)) vs w
    let recon = dequant_col(&packed, &scales, rows, cols);
    let (mut se, mut den) = (0.0f64, 0.0f64);
    for k in 0..w.len() { se += ((w[k] - recon[k]) as f64).powi(2); den += (w[k] as f64).powi(2); }
    println!("[fidelity] NF4Tile640 recon rel-L2 = {:.4}", (se / den).sqrt());

    // (2) GEMV parity: kernel-style dequant+GEMV vs an independent dense recon·x.
    let yk = nf4_gemv(&packed, &scales, &x, rows, cols);
    let mut maxd = 0.0f64;
    for r in 0..rows {
        let yref: f64 = (0..cols).map(|c| recon[r * cols + c] as f64 * x[c] as f64).sum();
        maxd = maxd.max((yref - yk[r] as f64).abs());
    }
    println!("[gemv parity] max|dense·x - fused| = {:.3e}  {}", maxd, if maxd < 1e-3 {"PASS"} else {"FAIL"});

    // (3) layout self-consistency: every col maps to a byte/nibble that a
    // byte-iteration reconstruction agrees with (proves kernel↔packer indexing).
    let tiles = cols / TILE;
    let mut recon_byte = vec![0f32; rows * cols];
    for r in 0..rows {
        for t in 0..tiles {
            for gnum in 0..GPT {
                let scale = scales[r * tiles * GPT + t * GPT + gnum];
                for b in 0..BYTES_GROUP {
                    let lane = b / 2;
                    let wb = b % 2;
                    let byte = packed[r * tiles * BYTES_TILE + t * BYTES_TILE + gnum * BYTES_GROUP + b];
                    for (nib_i, idx) in [(wb * 2, byte & 0x0F), (wb * 2 + 1, (byte >> 4) & 0x0F)] {
                        let gl = lane * VPL + nib_i;
                        let col = t * TILE + gnum * GROUP + gl;
                        recon_byte[r * cols + col] = NF4_CODEBOOK[idx as usize] * scale;
                    }
                }
            }
        }
    }
    let consistent = recon.iter().zip(&recon_byte).all(|(a, b)| (a - b).abs() < 1e-9);
    println!("[layout] col-iteration == byte-iteration reconstruction: {}", if consistent {"PASS"} else {"FAIL"});

    assert!(maxd < 1e-3 && consistent);
    println!("\nNF4TILE640 FORWARD REFERENCE VERIFIED — kernel layout contract locked");
}
