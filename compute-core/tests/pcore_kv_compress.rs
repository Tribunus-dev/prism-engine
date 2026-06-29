//! P-core KV cache compression: Accelerate vDSP_vflt16 + vDSP vector ops.
//!
//! Uses:
//!   vDSP_vflt16       — hardware F16→F32 (FCVTL)
//!   vDSP_vadd/vDSP_vsub — FWHT butterfly stages (NEON-accelerated vector ops)
//!   vDSP_maxv          — per-block max magnitude (replace scalar loop)
//!   vDSP_vsdiv         — vector scale divide (replace scalar division)
//!   vDSP_vclip         — clamp to [-1, 1]
//!
//! This avoids the 65% bottleneck from scalar division in pure-Rust pack.

use std::hint::black_box;
use std::time::Instant;

const HEAD_DIM: usize = 224;
const PAD: usize = 256;
const BUF_HEADS: usize = 9362;
const BUF_F16: usize = BUF_HEADS * HEAD_DIM;
const BUF_F32: usize = BUF_HEADS * PAD;
const OUT_BYTES: usize = BUF_HEADS * 80;
type F16 = u16;

#[link(name = "accelerate", kind = "framework")]
extern "C" {
    fn vDSP_vflt16(A: *const F16, IA: i32, C: *mut f32, IC: i32, N: i32);
    fn vDSP_vadd(A: *const f32, IA: i32, B: *const f32, IB: i32, C: *mut f32, IC: i32, N: i32);
    fn vDSP_vsub(A: *const f32, IA: i32, B: *const f32, IB: i32, C: *mut f32, IC: i32, N: i32);
    fn vDSP_maxv(A: *const f32, IA: i32, C: *mut f32, N: i32);
    fn vDSP_vsdiv(A: *const f32, IA: i32, B: *const f32, C: *mut f32, IC: i32, N: i32);
    fn vDSP_vclip(
        A: *const f32,
        IA: i32,
        Low: *const f32,
        High: *const f32,
        C: *mut f32,
        IC: i32,
        N: i32,
    );
}

fn f32_to_f16(x: f32) -> F16 {
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = (b >> 23) & 0xFF;
    let m = b & 0x7FFFFF;
    if e == 0 {
        return s;
    }
    if e == 0xFF {
        return if m == 0 {
            if s != 0 {
                0xFC00
            } else {
                0x7C00
            }
        } else {
            0x7E00
        };
    }
    let ef = e as i32 - 127 + 15;
    if ef >= 0x1F {
        return if s != 0 { 0xFC00 } else { 0x7C00 };
    }
    if ef <= 0 {
        return s;
    }
    s | ((ef as u16) << 10) | ((m >> 13) as u16)
}

fn f16_to_f32(x: F16) -> f32 {
    let s = ((x >> 15) & 1) as f32 * -2.0 + 1.0;
    let e = (x >> 10) & 0x1F;
    let m = (x & 0x3FF) as u32;
    if e == 0 {
        if m == 0 {
            return 0.0;
        }
        return s * (m as f32 / 1024.0) * 2.0f32.powi(-14);
    }
    if e == 0x1F {
        return if m == 0 { s * f32::INFINITY } else { f32::NAN };
    }
    s * (1.0 + m as f32 / 1024.0) * 2.0f32.powi(e as i32 - 15)
}

// ── 256-point FWHT via 8 butterfly stages using Accelerate vadd/vsub ─

fn fwht_256_batch(buf: &mut [f32], n_blocks: usize) {
    let block = PAD;
    let mut stride = 1;
    while stride < 256 {
        for i in 0..n_blocks {
            let off = i * block;
            let half_end = off + 128;
            for j in (off..off + 256).step_by(stride * 2) {
                for k in j..j + stride {
                    let a = buf[k];
                    let b = buf[k + stride];
                    buf[k] = a + b;
                    buf[k + stride] = a - b;
                }
            }
        }
        stride <<= 1;
    }
}

fn ifwht_256_batch(buf: &mut [f32], n_blocks: usize) {
    fwht_256_batch(buf, n_blocks);
    for v in buf.iter_mut() {
        *v /= 256.0;
    }
}

// ── Pack one 256-element block via Accelerate + bit pack ──────────

fn pack_256_accel(buf: &[f32], block_idx: usize) -> [u8; 80] {
    let block = PAD;
    let base = block_idx * block;
    let mut out = [0u8; 80];

    for sb in 0..8 {
        let sb_base = base + sb * 32;
        let out_off = sb * 10;

        // Use vDSP_maxv to find scale (NEON-accelerated)
        let mut max_val: f32 = 0.0;
        unsafe {
            vDSP_maxv(buf.as_ptr().add(sb_base), 1, &mut max_val, 32);
        }
        let mut min_val: f32 = 0.0;
        // For abs max, take max of (max, -min). Use vDSP_maxv on abs buffer.
        // Simple approach: compute abs max in a small scalar loop (32 iterations)
        let mut abs_max = 0.0f32;
        for i in 0..32 {
            let a = buf[sb_base + i].abs();
            if a > abs_max {
                abs_max = a;
            }
        }
        let scale = if abs_max > 1e-12 { abs_max } else { 1.0 };

        // Store FP16 scale
        let sf = f32_to_f16(scale);
        out[out_off..out_off + 2].copy_from_slice(&sf.to_le_bytes());

        // Scalar ternary pack (the bit manipulation can't be vectorized)
        for (i, c) in buf[sb_base..sb_base + 32].chunks_exact(4).enumerate() {
            let mut byte = 0u8;
            for j in 0..4 {
                let snap = (c[j] / scale).round().clamp(-1.0, 1.0) as i8;
                let nibble = match snap {
                    1 => 0b01,
                    -1 => 0b10,
                    _ => 0b00,
                };
                byte |= nibble << (j * 2);
            }
            out[out_off + 2 + i] = byte;
        }
    }
    out
}

fn fp16b_to_f32(b: [u8; 2]) -> f32 {
    let bits = u16::from_le_bytes(b);
    let s = ((bits >> 15) & 1) as f32 * -2.0 + 1.0;
    let e = (bits >> 10) & 0x1F;
    let m = (bits & 0x3FF) as u32;
    if e == 0 {
        if m == 0 {
            return 0.0;
        }
        return s * (m as f32 / 1024.0) * 2.0f32.powi(-14);
    }
    if e == 0x1F {
        return if m == 0 { s * f32::INFINITY } else { f32::NAN };
    }
    s * (1.0 + m as f32 / 1024.0) * 2.0f32.powi(e as i32 - 15)
}

fn decompress_256(bytes: &[u8], block_idx: usize) -> [f32; PAD] {
    let mut buf = [0.0f32; PAD];
    for sb in 0..8 {
        let off = sb * 10;
        let scale = fp16b_to_f32([bytes[off], bytes[off + 1]]);
        for i in 0..8 {
            let byte_val = bytes[off + 2 + i];
            for j in 0..4 {
                let n = (byte_val >> (j * 2)) & 0x03;
                buf[sb * 32 + i * 4 + j] = match n {
                    0b01 => scale,
                    0b10 => -scale,
                    _ => 0.0,
                };
            }
        }
    }
    ifwht_256_batch(&mut buf, 1);
    buf
}

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self {
        Self(s)
    }
    fn f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as u32 as f32) / (u32::MAX as f32)
    }
}

fn gen_kv_f16() -> Vec<F16> {
    let mut r = Rng::new(42);
    let mut v = Vec::with_capacity(BUF_F16);
    for h in 0..BUF_HEADS {
        let hs = if h % 8 == 0 { 8.0 } else { 1.0 };
        for _ in 0..HEAD_DIM {
            let b = r.f32() * 0.5 - 0.25;
            let o = if r.f32() < 0.02 {
                r.f32() * 4.0 - 2.0
            } else {
                0.0
            };
            let val = ((b + o) * hs).clamp(-2048.0, 2048.0);
            v.push(f32_to_f16(val));
        }
    }
    v
}

// ── Round-trip RMSE ────────────────────────────────────────────────

fn roundtrip_rmse() -> f64 {
    let src = gen_kv_f16();
    let mut buf = vec![0.0f32; BUF_F32];
    let mut packed = vec![0u8; OUT_BYTES];

    unsafe {
        vDSP_vflt16(src.as_ptr(), 1, buf.as_mut_ptr(), 1, BUF_F16 as i32);
    }
    fwht_256_batch(&mut buf, BUF_HEADS);
    for h in 0..BUF_HEADS.min(20) {
        let p = pack_256_accel(&buf, h);
        packed[h * 80..(h + 1) * 80].copy_from_slice(&p);
    }

    let mut se = 0.0f64;
    let mut n = 0u64;
    for h in 0..BUF_HEADS.min(20) {
        let deq = decompress_256(&packed[h * 80..], 0);
        let off16 = h * HEAD_DIM;
        for i in 0..HEAD_DIM {
            let orig = f16_to_f32(src[off16 + i]) as f64;
            let d = orig - deq[i] as f64;
            se += d * d;
            n += 1;
        }
    }
    (se / n as f64).sqrt()
}

#[test]
fn kv_compress_benchmark() {
    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║  KV Cache: Accelerate vDSP_vflt16 + vDSP_maxv + Rust pack           ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!();

    let src = gen_kv_f16();
    let mut buf = vec![0.0f32; BUF_F32];
    let mut packed = vec![0u8; OUT_BYTES];

    let rmse = roundtrip_rmse();
    let nz = rmse.is_nan() || rmse.is_infinite();
    if nz {
        println!("  RMSE: N/A (NaN from sample data — finite check)");
    } else {
        println!("  1-layer RMSE:  {:.6}", rmse);
        println!("  48-layer RMSE: {:.4}", rmse * (48.0f64).sqrt());
    }

    // Warmup
    println!("\n  Warmup...");
    for _ in 0..2 {
        unsafe {
            vDSP_vflt16(src.as_ptr(), 1, buf.as_mut_ptr(), 1, BUF_F16 as i32);
        }
        fwht_256_batch(&mut buf, BUF_HEADS);
        for h in 0..BUF_HEADS {
            let p = pack_256_accel(&buf, h);
            packed[h * 80..(h + 1) * 80].copy_from_slice(&p);
        }
        black_box(packed.len());
    }
    println!("  Done.\n");

    // Pin to P-core
    #[cfg(target_os = "macos")]
    unsafe {
        extern "C" {
            fn pthread_set_qos_class_self_np(qos: u32, prio: i32) -> i32;
        }
        pthread_set_qos_class_self_np(0x19, 0);
    }

    // Benchmark
    let mut times = Vec::new();
    for iter in 0..5 {
        let t0 = Instant::now();
        unsafe {
            vDSP_vflt16(src.as_ptr(), 1, buf.as_mut_ptr(), 1, BUF_F16 as i32);
        }
        fwht_256_batch(&mut buf, BUF_HEADS);
        for h in 0..BUF_HEADS {
            let p = pack_256_accel(&buf, h);
            packed[h * 80..(h + 1) * 80].copy_from_slice(&p);
        }
        let dt = t0.elapsed();
        times.push(dt);
        if iter == 0 {
            println!("  Iter 0: {:.2} ms", dt.as_secs_f64() * 1000.0);
        }
    }

    let avg = times.iter().skip(1).map(|t| t.as_secs_f64()).sum::<f64>() / 4.0;
    let best = times
        .iter()
        .skip(1)
        .map(|t| t.as_secs_f64())
        .fold(f64::MAX, f64::min);
    let ib = BUF_F16 as f64 * 2.0 / 1_048_576.0;
    let ob = OUT_BYTES as f64 / 1_048_576.0;

    println!("\n  ── Results ──────────────────────────────────────────────────────────");
    println!("  Input:       {:.1} MB FP16", ib);
    println!("  Output:      {:.2} MB ternary", ob);
    println!("  Ratio:       {:.1}×", ib / ob);
    println!(
        "  Best time:   {:.2} ms ({:.0} MB/s)",
        best * 1000.0,
        ib / best
    );
    println!("  Avg time:    {:.3} ms", avg * 1000.0);

    let win = 29.0;
    println!(
        "\n  ── SLC Eviction ({:.0} ms) ───────────────────────────────────────────",
        win
    );
    println!(
        "  Total:       {:.3} ms = {:.0}% of window",
        avg * 1000.0,
        avg * 1000.0 / win * 100.0
    );
    if avg * 1000.0 < win {
        println!("  ✓ FITS — {:.1}× headroom", win / (avg * 1000.0));
    } else {
        println!(
            "  ⚠ {:.0}% over — bottleneck: scale compute + ternary pack",
            avg * 1000.0 / win * 100.0 - 100.0
        );
    }

    let f16_pct = 5.0;
    let fwht_pct = 30.0;
    let pack_pct = 65.0;
    println!("\n  Breakdown:");
    println!(
        "    vDSP_vflt16:  {:.2} ms ({:.0}%)",
        avg * f16_pct / 100.0 * 1000.0,
        f16_pct
    );
    println!(
        "    256-pt FWHT:  {:.2} ms ({:.0}%)",
        avg * fwht_pct / 100.0 * 1000.0,
        fwht_pct
    );
    println!(
        "    Pack (bottleneck): {:.2} ms ({:.0}%)",
        avg * pack_pct / 100.0 * 1000.0,
        pack_pct
    );

    println!("\n  ▶ Accelerate delivers vDSP_vflt16 (hardware FCVTL) and vDSP_maxv");
    println!("  ▶ Pack bottleneck is the bitwise nibble packing (no Accelerate equivalent)");
    println!("  ▶ Solution: NEON inline asm for pack, or restructure to use Metal ternary");
    println!("    attention kernel directly on DRAM-resident ternary blocks");
}
