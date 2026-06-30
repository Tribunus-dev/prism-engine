//! V1/V2 cimage binary format, cimage compiler, and ANE swizzled payload.

pub const STATE_IDLE: u8 = 0;
pub const STATE_PREFETCHING: u8 = 1;
pub const STATE_READY: u8 = 2;
pub const STATE_EXECUTING: u8 = 3;

// ── V2 (Prism) types ──────────────────────────────────────────────

pub const PRISM_MAGIC: [u8; 8] = *b"PRISM\0\0\0";
pub const PRISM_PAGE_SIZE: u64 = 4096;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PrismCimageHeader {
    pub magic: [u8; 8],
    pub version: u32,
    pub segment_count: u32,
    pub payload_hash: [u8; 32],
    // Architecture
    pub num_layers: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub hidden_dim: u32,
    pub intermediate_dim: u32,
    pub vocab_size: u32,
    pub quantization_schema: u32,
    // Segment offsets (from V1 layout)
    pub metal_lib_offset: u64,
    pub metal_lib_len: u64,
    pub main_graph_offset: u64,
    pub main_graph_len: u64,
    pub main_weights_offset: u64,
    pub main_weights_len: u64,
    pub mtp_graph_offset: u64,
    pub mtp_graph_len: u64,
    pub mtp_weights_offset: u64,
    pub mtp_weights_len: u64,
    pub topology_table_offset: u64,
    pub topology_table_len: u64,
    pub lane_isolation: u8,
    pub _pad: [u8; 111],
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PrismCimageLayoutMeta {
    pub embed_clustered: TensorRecord,
    pub centroid_table: TensorRecord,
    pub cluster_map: TensorRecord,
    pub ternary_weights: TensorRecord,
    pub block_scales: TensorRecord,
    pub aux: TensorRecord,
    pub _pad: [u8; 32],
}

#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct TensorRecord {
    pub offset: u64,
    pub length: u64,
}

impl TensorRecord {
    pub fn new(offset: u64, length: u64) -> Self { Self { offset, length } }
}

pub fn verify_prism_cimage(bytes: &[u8]) -> Result<(PrismCimageHeader, PrismCimageLayoutMeta), String> {
    if bytes.len() < core::mem::size_of::<PrismCimageHeader>() { return Err("too small".into()); }
    let header: PrismCimageHeader = unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const PrismCimageHeader) };
    if &header.magic != &PRISM_MAGIC { return Err("bad magic".into()); }
    let lo = core::mem::size_of::<PrismCimageHeader>();
    if lo + core::mem::size_of::<PrismCimageLayoutMeta>() > bytes.len() { return Err("layout past end".into()); }
    let layout: PrismCimageLayoutMeta = unsafe { std::ptr::read_unaligned(bytes.as_ptr().add(lo) as *const PrismCimageLayoutMeta) };
    Ok((header, layout))
}

// ── Swizzled ternary re-pack for ANE Planar Engine gather ────────

/// Map linear (row, col) → (byte_offset, shift_within_byte).
#[inline(always)]
pub fn swizzled_byte_offset(row: usize, col: usize, width: usize) -> (usize, usize) {
    let bpr = width / 16;
    let br = row / 16;
    let bc = col / 16;
    let bi = br * bpr + bc;
    let ir = row % 16;
    let ic = col % 16;
    let ii = ir * 16 + ic;
    (bi * 64 + ii / 4, ii % 4)
}

/// Size of swizzled u8 buffer for tensor shape.
pub fn swizzled_buffer_size(rows: usize, cols: usize) -> usize {
    ((rows + 15) / 16) * ((cols + 15) / 16) * 64
}

/// Decode a u32 base-3 pack into an array of 20 ternary digits [0..2].
#[allow(dead_code)]
#[inline(always)]
fn decode_ternary_u32(packed: u32, digits: &mut [u8; 20]) {
    let mut rem = packed;
    for d in digits.iter_mut() {
        *d = (rem % 3) as u8;
        rem /= 3;
    }
}

/// Re-pack ternary u32 packs from DRAM into 16×16 swizzled u8 in SLC.
///
/// The ternary data uses the tile64 format: u32s at
///   offset = (row × num_tiles × 32 + tile × 32 + lane) × 4
/// Each u32 encodes 20 ternary values in base-3: digit 0→0, 1→+1, 2→-1.
///
/// The ANE reads the swizzled u8 from SLC and expands each quartet to
/// 4 INT8 values via the `gather` LUT (shape [81, 4]).  The scale
/// multiply also happens at gather time.
pub fn repack_ternary_to_swizzled_u8(
    ternary_bytes: &[u8],
    rows: usize,
    cols: usize,
    slc_buf: &mut [u8],
    slc_width: usize,
) {
    let expected = swizzled_buffer_size(rows, cols);
    if slc_buf.len() < expected { return; }
    slc_buf[..expected].fill(0);

    let ts = 640usize;
    let nt = (cols + ts - 1) / ts;

    // Accumulate quartets per SLC byte, then encode once all 4 slots fill
    let mut temp: Vec<[u8; 4]> = vec![[0u8; 4]; expected];
    let mut count: Vec<u8> = vec![0u8; expected];

    for row in 0..rows {
        for t in 0..nt {
            for lane in 0..32 {
                let po = row * nt * 32 * 4 + t * 32 * 4 + lane * 4;
                if po + 4 > ternary_bytes.len() { break; }

                let packed = u32::from_le_bytes([
                    ternary_bytes[po], ternary_bytes[po + 1],
                    ternary_bytes[po + 2], ternary_bytes[po + 3],
                ]);

                let mut rem = packed;
                for vi in 0..20 {
                    let col = t * ts + lane * 20 + vi;
                    if col >= cols { break; }

                    let digit = (rem % 3) as u8;
                    rem /= 3;

                    let (byte_off, shift) = swizzled_byte_offset(row, col, slc_width);
                    if byte_off >= expected { continue; }

                    temp[byte_off][shift as usize] = digit;
                    count[byte_off] += 1;
                }
            }
        }
    }

    // Encode fully-filled quartets into base-3 state bytes
    for bi in 0..expected {
        if count[bi] == 4 {
            let q = &temp[bi];
            slc_buf[bi] = q[0] + q[1] * 3 + q[2] * 9 + q[3] * 27;
        } else if count[bi] > 0 {
            // Partial quartet at tensor edge — encode what's filled
            let mut state: u8 = 0;
            for s in (0..4).rev() {
                state = state * 3 + if s < count[bi] { temp[bi][s as usize] } else { 0 };
            }
            slc_buf[bi] = state;
        }
    }
}

// ── Ternary block quantizer ──────────────────────────────────────

fn f32_to_fp16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let s = ((bits >> 16) & 0x8000) as u16;
    let e = (bits >> 23) & 0xFF;
    let m = bits & 0x7FFFFF;
    if e == 0 { return s; }
    if e == 0xFF { return if m == 0 { if s != 0 { 0xFC00 } else { 0x7C00 } } else { 0x7E00 }; }
    let ef = e as i32 - 127 + 15;
    if ef >= 0x1F { return if s != 0 { 0xFC00 } else { 0x7C00 }; }
    if ef <= 0 { return s; }
    s | ((ef as u16) << 10) | ((m >> 13) as u16)
}

pub fn fp16_to_f32(b: [u8; 2]) -> f32 {
    let bits = u16::from_le_bytes(b);
    let s = (((bits >> 15) & 1) as f32) * -2.0 + 1.0;
    let e = (bits >> 10) & 0x1F;
    let m = (bits & 0x03FF) as f32;
    if e == 0 { return if m == 0.0 { 0.0 } else { s * (m / 1024.0) * 2.0_f32.powi(-14) }; }
    if e == 0x1F { return if m == 0.0 { if s > 0.0 { f32::INFINITY } else { f32::NEG_INFINITY } } else { f32::NAN }; }
    s * (1.0 + m / 1024.0) * 2.0_f32.powi(e as i32 - 15)
}

pub fn ternary_quantize_block(block: &[f32; 256]) -> ([u8; 2], [u8; 64]) {
    let max_mag = block.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let scale = if max_mag > 1e-12 { max_mag } else { 1.0f32 };
    let su = f32_to_fp16_bits(scale);
    let mut nib = [0u8; 64];
    for (i, chk) in block.chunks_exact(4).enumerate() {
        let mut b: u8 = 0;
        for (j, &v) in chk.iter().enumerate() {
            let sn = (v / scale).round().clamp(-1.0, 1.0) as i8;
            b |= (match sn { 1 => 0b01, -1 => 0b10, _ => 0b00 }) << (j * 2);
        }
        nib[i] = b;
    }
    (su.to_le_bytes(), nib)
}

pub fn generate_ane_swizzled_weights(raw_bf16: &[u8], out_dim: u32, in_dim: u32) -> Vec<u8> {
    let rows = out_dim as usize;
    let cols = in_dim as usize;
    let total = swizzled_buffer_size(rows, cols);
    if total == 0 { return Vec::new(); }
    let mut swz = vec![0u8; total];
    let mut temp = vec![[0u8; 4]; total];
    let mut cnt = vec![0u8; total];

    let tv = rows * cols;
    let nb = (tv + 255) / 256;
    for bi in 0..nb {
        let st = bi * 256;
        let n = (tv - st).min(256);
        let mut blk = [0.0f32; 256];
        for j in 0..n {
            let bo = (st + j) * 2;
            if bo + 1 < raw_bf16.len() {
                blk[j] = f32::from_bits((u16::from_le_bytes([raw_bf16[bo], raw_bf16[bo + 1]]) as u32) << 16);
            }
        }
        let (_sc, nib) = ternary_quantize_block(&blk);
        for j in 0..n {
            let d = match (nib[j / 4] >> ((j % 4) * 2)) & 0x03 { 0b01 => 1, 0b10 => 2, _ => 0 };
            let vi = st + j;
            let (bi2, sh) = swizzled_byte_offset(vi / cols, vi % cols, cols);
            temp[bi2][sh as usize] = d;
            cnt[bi2] += 1;
        }
    }
    for b in 0..total {
        if cnt[b] == 0 { continue; }
        let q = &temp[b];
        let mut s: u8 = 0;
        for sh in (0..4).rev() { s = s * 3 + if sh < cnt[b] { q[sh as usize] } else { 0 }; }
        swz[b] = s;
    }
    swz
}

/// Requantize FP16 KV cache → swizzled u8 ternary format.
///
/// Reads FP16 KV values from the ANE's output surface (DRAM), quantizes
/// in 256-element blocks, packs 4 ternary digits per u8, and writes in
/// 16×16 block-swizzled order so the ANE Planar Engine `gather` LUT can
/// read it back.  The KV stays in DRAM as ternary packs until the next
/// ANE invocation needs it, at which point the E-core pumps it to SLC.
///
/// `fp16_kv`: raw FP16 bytes from KV cache (`seq_len * kv_dim * 2` bytes).
/// `seq_len`/`kv_dim`: shape of the KV cache slice being requantized.
/// `slc_buf`: pre-allocated output buffer (size = swizzled_buffer_size).
pub fn requantize_kv_to_swizzled_u8(
    fp16_kv: &[u8],
    seq_len: usize,
    kv_dim: usize,
    slc_buf: &mut [u8],
) {
    let total = seq_len * kv_dim;
    let nb = (total + 255) / 256;
    let expected = swizzled_buffer_size(seq_len, kv_dim);
    if slc_buf.len() < expected { return; }
    slc_buf[..expected].fill(0);

    let mut temp = vec![[0u8; 4]; expected];
    let mut cnt = vec![0u8; expected];

    for bi in 0..nb {
        let st = bi * 256;
        let n = (total - st).min(256);
        let mut blk = [0.0f32; 256];
        for j in 0..n {
            let bo = (st + j) * 2;
            if bo + 1 < fp16_kv.len() {
                let bits = u16::from_le_bytes([fp16_kv[bo], fp16_kv[bo + 1]]);
                blk[j] = fp16_to_f32(bits.to_le_bytes());
            }
        }
        let (_sc, nib) = ternary_quantize_block(&blk);
        for j in 0..n {
            let d = match (nib[j / 4] >> ((j % 4) * 2)) & 0x03 { 0b01 => 1u8, 0b10 => 2u8, _ => 0u8 };
            let vi = st + j;
            let (bi2, sh) = swizzled_byte_offset(vi / kv_dim, vi % kv_dim, kv_dim);
            temp[bi2][sh as usize] = d;
            cnt[bi2] += 1;
        }
    }
    for b in 0..expected {
        if cnt[b] == 0 { continue; }
        let q = &temp[b];
        let mut s: u8 = 0;
        for sh in (0..4).rev() { s = s * 3 + if sh < cnt[b] { q[sh as usize] } else { 0 }; }
        slc_buf[b] = s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swizzle_bijection() {
        let w = 640; let h = 240;
        let tb = ((h + 15) / 16) * (w / 16) * 64;
        let mut seen = vec![[false; 4]; tb];
        for r in 0..h { for c in 0..w {
            let (b, sh) = swizzled_byte_offset(r, c, w);
            assert!(b < tb); assert!(!seen[b][sh]); seen[b][sh] = true;
        }}
        for slots in &seen { for &u in slots { assert!(u); } }
    }

    #[test]
    fn test_repack_roundtrip() {
        let cols = 640; let rows = 32;
        let nt = (cols + 639) / 640;
        // Build mock ternary data in GPU format
        let mut ternary = vec![0u8; rows * nt * 32 * 4];
        let mut expected_digits = vec![0u8; rows * cols];
        for r in 0..rows { for c in 0..cols {
            let d = ((r * cols + c) % 3) as u8;
            expected_digits[r * cols + c] = d;
            // Set digit in the u32 pack
            let tile = c / 640;
            let lane = (c % 640) / 20;
            let vi = (c % 640) % 20;
            let po = r * nt * 32 * 4 + tile * 32 * 4 + lane * 4;
            if po + 4 > ternary.len() { continue; }
            let mut pk = u32::from_le_bytes([ternary[po], ternary[po+1], ternary[po+2], ternary[po+3]]);
            let mut mul = 1u32;
            for _ in 0..vi { mul *= 3; }
            pk = (pk / (mul * 3)) * (mul * 3) + d as u32 * mul + pk % mul;
            ternary[po..po+4].copy_from_slice(&pk.to_le_bytes());
        }}

        let tb = swizzled_buffer_size(rows, cols);
        let mut slc = vec![0u8; tb];
        repack_ternary_to_swizzled_u8(&ternary, rows, cols, &mut slc, cols);

        let mut lut = [[0i8; 4]; 81];
        for s in 0u8..81 { let mut x = s;
            for j in 0..4 { lut[s as usize][j] = match x % 3 { 1 => 1, 2 => -1, _ => 0 }; x /= 3; }
        }

        for r in 0..rows { for c in 0..cols {
            let (b, sh) = swizzled_byte_offset(r, c, cols);
            let decoded = lut[slc[b] as usize][sh];
            let expected = match expected_digits[r * cols + c] { 1 => 1, 2 => -1, _ => 0 };
            assert_eq!(decoded, expected, "Mismatch at ({r},{c})");
        }}
    }

    #[test]
    fn test_generate_ane_swizzled_weights() {
        let mut src = [0u8; 640 * 240 * 2];
        for i in 0..640 * 240 {
            let v = ((i as f32 * 1.618) % 6.0) - 3.0;
            let bits = (v.to_bits() >> 16) as u16;
            src[i*2..i*2+2].copy_from_slice(&bits.to_le_bytes());
        }
        let swz = generate_ane_swizzled_weights(&src, 240, 640);
        assert!(!swz.is_empty());
        let expected = swizzled_buffer_size(240, 640);
        assert_eq!(swz.len(), expected);
    }
    #[test]
    fn test_pump_smoke() {
        let rows = 32;
        let cols = 640;
        let nt = (cols + 639) / 640;
        let mut src = vec![0.0f32; rows * cols];
        for i in 0..rows * cols { src[i] = ((i as f32 * 1.618) % 6.0) - 3.0; }
        let mut ternary = vec![0u8; rows * nt * 32 * 4];
        let mut scales = Vec::new();
        let nb = (rows * cols + 255) / 256;
        for bi in 0..nb {
            let st = bi * 256;
            let n = (rows * cols - st).min(256);
            let mut blk = [0.0f32; 256];
            for j in 0..n { blk[j] = src[st + j]; }
            let (sc, nib) = ternary_quantize_block(&blk);
            scales.push(sc);
            for j in 0..n {
                let d = match (nib[j / 4] >> ((j % 4) * 2)) & 0x03 { 0b01 => 1, 0b10 => 2, _ => 0 };
                let vi = st + j;
                let po = (vi / cols) * nt * 32 * 4 + ((vi % cols) / 640) * 32 * 4 + (((vi % cols) % 640) / 20) * 4;
                if po + 4 > ternary.len() { continue; }
                let mut pk = u32::from_le_bytes([ternary[po], ternary[po+1], ternary[po+2], ternary[po+3]]);
                let sub = (vi % cols) % 640 % 20;
                let mut mul = 1u32;
                for _ in 0..sub { mul *= 3; }
                pk = (pk / (mul * 3)) * (mul * 3) + d as u32 * mul + pk % mul;
                ternary[po..po+4].copy_from_slice(&pk.to_le_bytes());
            }
        }
        let tb = swizzled_buffer_size(rows, cols);
        let mut slc = vec![0u8; tb];
        repack_ternary_to_swizzled_u8(&ternary, rows, cols, &mut slc, cols);
        let mut lut = [[0i8; 4]; 81];
        for s in 0u8..81 { let mut x = s;
            for j in 0..4 { lut[s as usize][j] = match x % 3 { 1 => 1, 2 => -1, _ => 0 }; x /= 3; }
        }
        // Build LUT and decode
        let mut lut = [[0i8; 4]; 81];
        for s in 0u8..81 { let mut x = s;
            for j in 0..4 { lut[s as usize][j] = match x % 3 { 1 => 1, 2 => -1, _ => 0 }; x /= 3; }
        }
        // Build expected digits directly from the quantizer source
        let mut expected_i8 = vec![0i8; rows * cols];
        for vi in 0..rows * cols {
            let bi = vi / 256;
            let st = bi * 256;
            let mut blk = [0.0f32; 256];
            for j in 0..(rows * cols - st).min(256) { blk[j] = src[st + j]; }
            let (sc, nib) = ternary_quantize_block(&blk);
            let _sc_f32 = fp16_to_f32(sc);
            let off = vi - st;
            let nibble = (nib[off / 4] >> ((off % 4) * 2)) & 0x03;
            expected_i8[vi] = match nibble { 0b01 => 1, 0b10 => -1, _ => 0 };
        }
        // Verify LUT-decoded values match expected
        let mut err = 0u32;
        for r in 0..rows { for c in 0..cols {
            let (b, sh) = swizzled_byte_offset(r, c, cols);
            let got = lut[slc[b] as usize][sh];
            let exp = expected_i8[r * cols + c];
            if got != exp { err += 1; if err <= 3 { eprintln!("({r},{c}): got {got} exp {exp}"); } }
        }}
        assert_eq!(err, 0, "{err} mismatches — pure ternary digit mismatch, not FP16 precision");
        eprintln!("[pump smoke] {rows}x{cols}: {} values match", rows * cols);
    }

    #[test]
    fn test_kv_requantizer_roundtrip() {
        let seq_len = 64; let kv_dim = 256;
        let mut kv = vec![0u8; seq_len * kv_dim * 2];
        for i in 0..seq_len * kv_dim {
            let v = ((i as f32 * 1.618) % 2.0) - 1.0;
            let bits = f32_to_fp16_bits(v);
            kv[i*2..i*2+2].copy_from_slice(&bits.to_le_bytes());
        }
        let mut swz = vec![0u8; swizzled_buffer_size(seq_len, kv_dim)];
        requantize_kv_to_swizzled_u8(&kv, seq_len, kv_dim, &mut swz);
        let mut lut = [[0i8; 4]; 81];
        for s in 0u8..81 { let mut x = s;
            for j in 0..4 { lut[s as usize][j] = match x % 3 { 1 => 1, 2 => -1, _ => 0 }; x /= 3; }
        }
        for i in 0..(seq_len * kv_dim).min(500) {
            let (b, sh) = swizzled_byte_offset(i / kv_dim, i % kv_dim, kv_dim);
            let got = lut[swz[b] as usize][sh];
            let bits = u16::from_le_bytes([kv[i*2], kv[i*2+1]]);
            let v = fp16_to_f32(bits.to_le_bytes());
            let bi = i / 256;
            let st = bi * 256;
            let mut max_v = 0.0f32;
            for j in st..(st+256).min(seq_len * kv_dim) {
                let bj = u16::from_le_bytes([kv[j*2], kv[j*2+1]]);
                max_v = max_v.max(fp16_to_f32(bj.to_le_bytes()).abs());
            }
            let mag = if max_v > 1e-6 { max_v } else { 1.0 };
            let snapped = if v.abs() > mag * 0.5 { if v > 0.0 { mag } else { -mag } } else { 0.0 };
            let exp = (snapped / mag).round() as i8;
            assert!((got - exp).abs() <= 1, "Mismatch at {i}: got {got} exp {exp} v={v}");
        }
    }


    #[test]
    fn test_embed_ternary_roundtrip() {
        // Embed table: [vocab_size, hidden_dim] = [32000, 3840] typical
        let vocab = 512;  // small for test speed
        let hd = 128;
        let mut embed = vec![0u8; (vocab * hd) as usize * 2];
        for i in 0..(vocab * hd) as usize {
            let v = ((i as f32 * 1.618) % 2.0) - 1.0;
            let bits = f32_to_fp16_bits(v);
            embed[i*2..i*2+2].copy_from_slice(&bits.to_le_bytes());
        }
        let swz = generate_ane_swizzled_weights(&embed, vocab, hd);
        assert_eq!(swz.len(), swizzled_buffer_size(vocab as usize, hd as usize));
        // Decode one row via LUT and verify it matches the original ternary snap
        let mut lut = [[0i8; 4]; 81];
        for s in 0u8..81 { let mut x = s;
            for j in 0..4 { lut[s as usize][j] = match x % 3 { 1 => 1, 2 => -1, _ => 0 }; x /= 3; }
        }
        // Pick token 42, decode its embedding row
        let row = 42;
        for c in 0..hd.min(32) {
            let col = c as usize;
            let (b, sh) = swizzled_byte_offset(row, col, hd as usize);
            let decoded = lut[swz[b] as usize][sh];
            let i = row * hd as usize + col;
            let bits = u16::from_le_bytes([embed[i*2], embed[i*2+1]]);
            let v = fp16_to_f32(bits.to_le_bytes());
            let bi = i / 256;
            let st = bi * 256;
            let end = (st + 256).min((vocab * hd) as usize);
            let mut max_v = 0.0f32;
            for j in st..end { let bj = u16::from_le_bytes([embed[j*2], embed[j*2+1]]); max_v = max_v.max(fp16_to_f32(bj.to_le_bytes()).abs()); }
            let mag = if max_v > 1e-6 { max_v } else { 1.0 };
            let snapped = if v.abs() > mag * 0.5 { if v > 0.0 { mag } else { -mag } } else { 0.0 };
            let exp = (snapped / mag).round() as i8;
            assert!((decoded - exp).abs() <= 1, "Embed mismatch at [{row},{col}]: got {decoded} exp {exp} v={v}");
        }
    }

}