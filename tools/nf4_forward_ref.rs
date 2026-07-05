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
//! tiles = ceil(in_dim / 640). When in_dim is NOT a multiple of 640 the last
//! tile is PARTIAL: columns [in_dim, tiles*640) are zero-padded by the packer
//! (value 0 → nearest NF4 code 7 = 0.0 → dequants to 0), and the kernel guards
//! its activation read with `if col >= in_dim continue`. The kernel-emulation
//! GEMV below reproduces that guard exactly so the partial path is verified here.
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

fn tiles_for(cols: usize) -> usize {
    cols.div_ceil(TILE)
}

fn nearest(v: f32) -> u8 {
    let mut b = 0u8;
    let mut bd = (v - NF4_CODEBOOK[0]).abs();
    for (i, &l) in NF4_CODEBOOK.iter().enumerate().skip(1) {
        let d = (v - l).abs();
        if d < bd { bd = d; b = i as u8; }
    }
    b
}

/// Row-major weight read with zero-pad past the real width. `col < cols` returns
/// the true weight; the padded tail of a partial last tile reads as 0.0 —
/// exactly what the GPU/CPU packers write there.
#[inline]
fn wval(w: &[f32], r: usize, cols: usize, col: usize) -> f32 {
    if col < cols { w[r * cols + col] } else { 0.0 }
}

/// Pack a row-major [rows × cols] weight matrix into the interleaved NF4Tile640
/// arena, exactly as the GPU packer does. `cols` need NOT be a multiple of 640:
/// the last tile is zero-padded. Returns (packed_u8, scales_f32).
fn pack(w: &[f32], rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>) {
    let tiles = tiles_for(cols);
    let mut packed = vec![0u8; rows * tiles * BYTES_TILE];
    let mut scales = vec![0f32; rows * tiles * GPT];
    for r in 0..rows {
        for t in 0..tiles {
            for g in 0..GPT {
                // group absmax over its 128 values (padded slots contribute 0)
                let mut absmax = 0f32;
                for gl in 0..GROUP {
                    let col = t * TILE + g * GROUP + gl;
                    absmax = absmax.max(wval(w, r, cols, col).abs());
                }
                let scale = if absmax > 1e-12 { absmax } else { 1.0 };
                scales[r * tiles * GPT + t * GPT + g] = scale;
                let inv = 1.0 / scale;
                for lane in 0..LANES {
                    for i in 0..VPL {
                        let gl = lane * VPL + i;
                        let col = t * TILE + g * GROUP + gl;
                        let idx = nearest((wval(w, r, cols, col) * inv).clamp(-1.0, 1.0));
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

/// Kernel-style dequant of the REAL columns [0, cols): iterate col-by-col,
/// compute byte/nibble address, apply the group's affine scale/bias.
fn dequant_col(packed: &[u8], scales: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let tiles = tiles_for(cols);
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

/// Dense reference GEMV over the real columns: y[o] = Σ_col recon(o,col) · x[col].
fn dense_gemv(recon: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f64> {
    (0..rows)
        .map(|r| (0..cols).map(|c| recon[r * cols + c] as f64 * x[c] as f64).sum::<f64>())
        .collect()
}

/// Kernel-emulation GEMV: mirrors the EXACT loop structure of
/// fused_gemv_nf4_tile640_fp32 — it walks the full padded grid `tiles*TILE`,
/// reads the interleaved nibble per (tile, group, lane, i), and applies the
/// `if col >= in_dim continue` guard. Proves the guarded partial-tile path
/// computes the same result as the dense GEMV over real columns.
fn nf4_gemv_kernel_emulation(
    packed: &[u8],
    scales: &[f32],
    x: &[f32],
    rows: usize,
    cols: usize, // == in_dim
) -> Vec<f64> {
    let tiles = tiles_for(cols);
    let mut out = vec![0f64; rows];
    for r in 0..rows {
        let mut acc = 0f64;
        for t in 0..tiles {
            for g in 0..GPT {
                let scale = scales[r * tiles * GPT + t * GPT + g] as f64;
                for lane in 0..LANES {
                    for i in 0..VPL {
                        let col = t * TILE + g * GROUP + lane * VPL + i;
                        if col >= cols {
                            continue; // guard: zero-padded tail, no in_vector read
                        }
                        let byte = r * tiles * BYTES_TILE + t * BYTES_TILE
                            + g * BYTES_GROUP + lane * 2 + (i / 2);
                        let idx = if i % 2 == 0 {
                            packed[byte] & 0x0F
                        } else {
                            (packed[byte] >> 4) & 0x0F
                        };
                        let weight = NF4_CODEBOOK[idx as usize] as f64 * scale;
                        acc += weight * x[col] as f64;
                    }
                }
            }
        }
        out[r] = acc;
    }
    out
}

fn rng(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*s >> 40) as f32) / ((1u64 << 24) as f32)
}
fn gauss(s: &mut u64) -> f32 {
    let a = rng(s).max(1e-7);
    let b = rng(s);
    (-2.0 * a.ln()).sqrt() * (std::f32::consts::TAU * b).cos()
}

/// Full parity battery for one shape. Returns true if all checks pass.
fn run(rows: usize, cols: usize, label: &str) -> bool {
    let tiles = tiles_for(cols);
    let padded = tiles * TILE;
    println!(
        "\n=== {label}: rows={rows} in_dim={cols} tiles={tiles} (padded width {padded}, tail {} pad cols) ===",
        padded - cols
    );

    let mut s = 0xF00Du64 ^ (cols as u64).wrapping_mul(0x9E3779B97F4A7C15);
    let w: Vec<f32> = (0..rows * cols).map(|_| 0.02 * gauss(&mut s)).collect();
    let x: Vec<f32> = (0..cols).map(|_| gauss(&mut s)).collect();

    let (packed, scales) = pack(&w, rows, cols);

    // (1) fidelity: dequant(pack(w)) vs w
    let recon = dequant_col(&packed, &scales, rows, cols);
    let (mut se, mut den) = (0.0f64, 0.0f64);
    for k in 0..w.len() {
        se += ((w[k] - recon[k]) as f64).powi(2);
        den += (w[k] as f64).powi(2);
    }
    println!("[fidelity] NF4Tile640 recon rel-L2 = {:.4}", (se / den).sqrt());

    // (2) GEMV parity: kernel-emulation (walks padded grid + guard) vs dense.
    let y_dense = dense_gemv(&recon, &x, rows, cols);
    let y_kernel = nf4_gemv_kernel_emulation(&packed, &scales, &x, rows, cols);
    let mut maxd = 0.0f64;
    for r in 0..rows {
        maxd = maxd.max((y_dense[r] - y_kernel[r]).abs());
    }
    let gemv_ok = maxd < 1e-3;
    println!(
        "[gemv parity] max|dense·x - guarded-kernel| = {:.3e}  {}",
        maxd,
        if gemv_ok { "PASS" } else { "FAIL" }
    );

    // (3) layout self-consistency over REAL columns (guard padded region).
    let mut recon_byte = vec![0f32; rows * cols];
    for r in 0..rows {
        for t in 0..tiles {
            for gnum in 0..GPT {
                let scale = scales[r * tiles * GPT + t * GPT + gnum];
                for b in 0..BYTES_GROUP {
                    let lane = b / 2;
                    let wb = b % 2;
                    let byte =
                        packed[r * tiles * BYTES_TILE + t * BYTES_TILE + gnum * BYTES_GROUP + b];
                    for (nib_i, idx) in [(wb * 2, byte & 0x0F), (wb * 2 + 1, (byte >> 4) & 0x0F)] {
                        let gl = lane * VPL + nib_i;
                        let col = t * TILE + gnum * GROUP + gl;
                        if col < cols {
                            recon_byte[r * cols + col] = NF4_CODEBOOK[idx as usize] * scale;
                        }
                    }
                }
            }
        }
    }
    let consistent = recon.iter().zip(&recon_byte).all(|(a, b)| (a - b).abs() < 1e-9);
    println!(
        "[layout] col-iteration == byte-iteration reconstruction: {}",
        if consistent { "PASS" } else { "FAIL" }
    );

    // (4) padded-tail invariant: for a partial tile, all packed nibbles beyond
    // in_dim must decode to code 7 (0.0), i.e. contribute nothing.
    let mut tail_ok = true;
    if padded > cols {
        'rows: for r in 0..rows {
            for col in cols..padded {
                let t = col / TILE;
                let wt = col % TILE;
                let g = wt / GROUP;
                let gl = wt % GROUP;
                let lane = gl / VPL;
                let i = gl % VPL;
                let byte = r * tiles * BYTES_TILE + t * BYTES_TILE
                    + g * BYTES_GROUP + lane * 2 + (i / 2);
                let idx = if i % 2 == 0 { packed[byte] & 0x0F } else { (packed[byte] >> 4) & 0x0F };
                if NF4_CODEBOOK[idx as usize] != 0.0 {
                    tail_ok = false;
                    break 'rows;
                }
            }
        }
        println!(
            "[pad tail] every padded col decodes to 0.0: {}",
            if tail_ok { "PASS" } else { "FAIL" }
        );
    }

    gemv_ok && consistent && tail_ok
}

fn main() {
    let mut all = true;
    all &= run(48, 1280, "exact multiple of 640");
    // Partial last tile: 1290 = 2*640 + 10 → 3 tiles, 630 padded cols.
    all &= run(48, 1290, "partial (in_dim % 640 = 10)");
    // Another partial shape: a realistic odd projection width.
    all &= run(32, 4104, "partial (in_dim % 640 = 264)");

    assert!(all, "NF4Tile640 forward reference FAILED a parity check");
    println!("\nNF4TILE640 FORWARD REFERENCE VERIFIED — kernel layout + partial-tile guard locked");
}
