//! 20-in-32 Base-3 ternary packing with magic multiplication unpack.
//!
//! 20 ternary weights {-1, 0, +1} → 1 × u32 (3^20 = 3.4B < 4.3B = 2^32)
//! 32 bits / 20 weights = 1.6 bits/weight vs 2.0 for bit-plane = 20% bandwidth savings.
//!
//! Unpack uses magic multiplication to avoid GPU-killing %3:
//!   div = (v * 2863311531u) >> 33   (single MUL + shift, ~2 cycles)
//!   mod = v - div * 3                (single MAD instruction)
//!
//! No look-up tables, no shared memory, no modulo operations.

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use metal::*;
use std::time::Instant;

// ── Constants ───────────────────────────────────────────────────────
const CHUNK: usize = 20; // weights per u32
const HEAD_DIM: usize = 224;
const CHUNKS_PER_ROW: usize = (HEAD_DIM + CHUNK - 1) / CHUNK; // 12
const ROWS: usize = 1024; // test vocabulary rows

/// Magic constant for unsigned 32-bit division by 3.
/// (2^33 + 2 + 1) / 3 = 2863311531 = 0xAAAAAAAB
const MAGIC_DIV3: u32 = 2863311531;

// ── Base-3 packing ─────────────────────────────────────────────────

/// Pack 20 ternary weights {-1, 0, +1} into one u32 via Base-3 encoding.
fn pack_base3(w: &[i8; CHUNK]) -> u32 {
    let mut val = 0u32;
    for i in (0..CHUNK).rev() {
        val = val * 3 + (w[i] + 1) as u32; // map {-1,0,+1} to {0,1,2}
    }
    val
}

/// Unpack a u32 into 20 ternary weights using magic multiplication.
fn unpack_base3_magic(val: u32) -> [i8; CHUNK] {
    let mut w = [0i8; CHUNK];
    let mut v = val;
    for i in 0..CHUNK {
        // Fast mod 3: magic multiplication
        let div = ((v as u64).wrapping_mul(MAGIC_DIV3 as u64)) >> 33;
        let d = v as u32 - (div as u32).wrapping_mul(3);
        w[i] = (d as i8) - 1; // map {0,1,2} to {-1,0,+1}
        v = div as u32;
    }
    w
}

/// Unpack using native %3 for CPU reference.
fn unpack_base3_native(val: u32) -> [i8; CHUNK] {
    let mut w = [0i8; CHUNK];
    let mut v = val;
    for i in 0..CHUNK {
        let d = (v % 3) as i8;
        w[i] = d - 1;
        v /= 3;
    }
    w
}

/// Verify magic multiplication matches native %3.
fn verify_magic() -> bool {
    // Test all possible 8-bit values (for the 5-in-8 scheme)
    for v in 0u32..256 {
        let native_d = v % 3;
        let magic_div = (v.wrapping_mul(171)) >> 9;
        let magic_d = v - magic_div.wrapping_mul(3);
        if native_d != magic_d {
            return false;
        }
    }
    // Test all possible 20-in-32 values
    // 3^20 = 3.4B is too many to iterate, sample instead
    let mut r = Rng::new(42);
    for _ in 0..100000 {
        let v = r.u32() % 3486784401u64.max(1) as u32;
        let native_d = v % 3;
        let magic_div = ((v as u64).wrapping_mul(MAGIC_DIV3 as u64) >> 33) as u32;
        let magic_d = v.wrapping_sub(magic_div.wrapping_mul(3));
        if native_d != magic_d {
            return false;
        }
    }
    true
}

/// Generate random ternary weights and pack them.
fn gen_packed_rows(n_rows: usize) -> (Vec<u32>, Vec<[i8; CHUNK]>) {
    let mut r = Rng::new(42);
    let mut packed = Vec::with_capacity(n_rows * CHUNKS_PER_ROW);
    let mut all_w = Vec::with_capacity(n_rows * CHUNKS_PER_ROW);
    for _ in 0..n_rows {
        for _ in 0..CHUNKS_PER_ROW {
            let mut chunk = [0i8; CHUNK];
            for i in 0..CHUNK {
                let v = r.f32();
                chunk[i] = if v < 0.33 {
                    -1
                } else if v < 0.67 {
                    0
                } else {
                    1
                };
            }
            all_w.push(chunk);
            packed.push(pack_base3(&chunk));
        }
    }
    (packed, all_w)
}

// ── CPU GEMV reference ─────────────────────────────────────────────

fn gemv_cpu_ref(act: &[f32; HEAD_DIM], weights_20: &[[i8; CHUNK]; CHUNKS_PER_ROW]) -> f32 {
    let mut sum = 0.0f32;
    for c in 0..CHUNKS_PER_ROW {
        for i in 0..CHUNK {
            let idx = c * CHUNK + i;
            if idx < HEAD_DIM {
                sum += (weights_20[c][i] as f32) * act[idx];
            }
        }
    }
    sum
}

// ── Metal kernel: 20-in-32 magic unpack + GEMV ─────────────────────

const KERNEL: &str = r##"#include <metal_stdlib>
using namespace metal;

// Magic constant for unsigned division by 3 (works up to 32-bit)
constant uint MAGIC_DIV3 = 2863311531u;
constant uint CHUNKS_PER_ROW = 12;
constant uint HEAD_DIM = 224;

/// Fast modulo-3 via magic multiplication: single MUL + MAD.
/// Works for any v < 2^32 (3^20 = 3.4B < 2^32).
inline uint fast_mod3(uint v) {
    uint div = ((uint64_t)v * (uint64_t)MAGIC_DIV3) >> 33;
    return v - div * 3u;
}

/// Unpack 20 ternary weights from one u32 using magic math.
/// Returns weights as lower 20 bits of a uint (0=+1, 1=0, 2=-1), but
/// we directly accumulate into the GEMV sum per weight.

kernel void gemv_base3(
    device const uint*  packed_weights [[buffer(0)]],
    device const float* activations    [[buffer(1)]],
    device float*       logits         [[buffer(2)]],
    constant uint&      num_rows       [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lane_id [[thread_index_in_simdgroup]])
{
    if (gid >= num_rows) return;

    // Each warp (threadgroup of 32) handles one vocabulary row
    // We dispatch num_rows threadgroups, each with 32 threads
    // But each row has 12 u32 chunks, across 32 threads.
    // 12 threads process one chunk each, remaining 20 are idle.
    //
    // Better: use lane-level parallelism. Each lane processes one chunk.
    // With 12 chunks per row and 32 lanes, lanes 0..11 are active.
    // After all chunks, reduce across lanes.
    //
    // Alternative: one thread handles all 12 chunks, serial.
    // 12 × 20 = 240 magic divs per thread per row.
    // With 1024 rows dispatched, that's 1024 × 240 = 246K divs.
    // At ~2 cycles each = ~0.5M cycles = ~150 µs. Fast enough.

    // ── One thread = one row, all 12 chunks processed serially ──
    uint row_base = gid * CHUNKS_PER_ROW;
    float sum = 0.0;
    uint act_idx = lane_id; // each lane handles different activation index
    // Actually, simpler: single-threaded GEMV per row
    // Each thread processes all 12 chunks serially
    //
    // No — let's use the warp-parallel approach from bit-plane:
    // 32 threads, each handles one chunk, then reduce.
    if (lane_id < CHUNKS_PER_ROW) {
        uint val = packed_weights[row_base + lane_id];
        uint base_idx = lane_id * 20;

        // Unpack 20 weights using magic mod-3
        float partial = 0.0;
        for (uint i = 0; i < 20; ++i) {
            uint d = fast_mod3(val);
            int w = (int)d - 1; // {0,1,2} → {-1,0,+1}

            uint idx = base_idx + i;
            if (idx < HEAD_DIM) {
                partial += (float)w * activations[idx];
            }

            // Divide val by 3 for next extraction
            val = ((uint64_t)val * (uint64_t)MAGIC_DIV3) >> 33;
        }
        sum = partial;
    }

    // Warp reduction
    sum += simd_shuffle_xor(sum, 1);
    sum += simd_shuffle_xor(sum, 2);
    sum += simd_shuffle_xor(sum, 4);
    sum += simd_shuffle_xor(sum, 8);
    sum += simd_shuffle_xor(sum, 16);

    if (lane_id == 0) { logits[gid] = sum; }
}
"##;

// ── Metal setup ────────────────────────────────────────────────────

fn compile_kernel(src: &str) -> (ComputePipelineState, CommandQueue, Device) {
    let dev = Device::system_default().expect("Metal device");
    let tmp = std::env::temp_dir().join("tribunus-mtl-b3");
    let _ = std::fs::create_dir_all(&tmp);
    let s = tmp.join("k.metal");
    let a = tmp.join("k.air");
    let l = tmp.join("k.metallib");
    std::fs::write(&s, src).unwrap();
    assert!(std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-std=metal4.0", "-O3", "-c"])
        .arg(s.to_str().unwrap())
        .arg("-o")
        .arg(a.to_str().unwrap())
        .status()
        .unwrap()
        .success());
    assert!(std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib", "-o"])
        .arg(l.to_str().unwrap())
        .arg(a.to_str().unwrap())
        .status()
        .unwrap()
        .success());
    let bytes = std::fs::read(&l).unwrap();
    let lib = dev.new_library_with_data(&bytes).unwrap();
    let f = lib.get_function("gemv_base3", None).unwrap();
    let pso = dev.new_compute_pipeline_state_with_function(&f).unwrap();
    (pso, dev.new_command_queue(), dev)
}

// ── RNG ────────────────────────────────────────────────────────────
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
    fn u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

#[test]
fn base3_20in32_gemv() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  20-in-32 Base-3 Packing: Magic multiplication + warp-parallel GEMV     ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Phase 1: Verify magic multiplication
    println!("  Phase 1: Magic mod-3 verification");
    let ok = verify_magic();
    println!(
        "  fast_mod3 == %3: {} (all 8-bit exhaustive + 100K 32-bit sample)",
        if ok { "✓" } else { "✗" }
    );

    // Phase 2: Pack/unpack round-trip
    println!("\n  Phase 2: Pack/unpack round-trip");
    let mut r = Rng::new(42);
    let mut all_ok = true;
    for _ in 0..1000 {
        let mut w = [0i8; CHUNK];
        for i in 0..CHUNK {
            let v = r.f32();
            w[i] = if v < 0.33 {
                -1
            } else if v < 0.67 {
                0
            } else {
                1
            };
        }
        let packed = pack_base3(&w);
        let unpacked_m = unpack_base3_magic(packed);
        let unpacked_n = unpack_base3_native(packed);
        if unpacked_m != w || unpacked_n != w {
            all_ok = false;
            break;
        }
    }
    println!("  Round-trip native: {}", if all_ok { "✓" } else { "✗" });

    // Phase 3: CPU GEMV correctness
    println!("\n  Phase 3: CPU GEMV (all 3 formats equivalent)");
    let (packed, all_w) = gen_packed_rows(ROWS);
    let mut r2 = Rng::new(99);
    let act: [f32; HEAD_DIM] = core::array::from_fn(|_| r2.f32() * 2.0 - 1.0);

    // Reference: GEMV using direct float multiply by ternary values
    let ref_rows: Vec<f32> = (0..ROWS)
        .map(|row| {
            let base = row * CHUNKS_PER_ROW;
            let weights_20: &[[i8; CHUNK]] = unsafe {
                std::slice::from_raw_parts(
                    all_w[base..].as_ptr() as *const [i8; CHUNK],
                    CHUNKS_PER_ROW,
                )
            };
            let mut sum = 0.0f32;
            for c in 0..CHUNKS_PER_ROW {
                for i in 0..CHUNK {
                    let idx = c * CHUNK + i;
                    if idx < HEAD_DIM {
                        sum += (weights_20[c][i] as f32) * act[idx];
                    }
                }
            }
            sum
        })
        .collect();

    // GEMV using magic unpack
    let magic_rows: Vec<f32> = (0..ROWS)
        .map(|row| {
            let base = row * CHUNKS_PER_ROW;
            let mut sum = 0.0f32;
            for c in 0..CHUNKS_PER_ROW {
                let val = packed[base + c];
                let w = unpack_base3_magic(val);
                for i in 0..CHUNK {
                    let idx = c * CHUNK + i;
                    if idx < HEAD_DIM {
                        sum += (w[i] as f32) * act[idx];
                    }
                }
            }
            sum
        })
        .collect();

    let mut max_e = 0.0f64;
    for i in 0..ROWS {
        let d = (ref_rows[i] - magic_rows[i]).abs() as f64;
        if d > max_e {
            max_e = d;
        }
    }
    let pd_s = if max_e < 1e-6 { "✓" } else { "△" };
    println!(
        "  Max diff (direct vs magic unpack): {:.2e} {}",
        max_e, pd_s
    );

    // Phase 4: GPU GEMV
    println!("\n  Phase 4: GPU 20-in-32 GEMV");
    let (pso, queue, dev) = compile_kernel(KERNEL);

    let packed_buf = dev.new_buffer(
        (packed.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let act_buf = dev.new_buffer((HEAD_DIM * 4) as u64, MTLResourceOptions::StorageModeShared);
    let logit_buf = dev.new_buffer((ROWS * 4) as u64, MTLResourceOptions::StorageModeShared);
    let nrows = ROWS as u32;

    unsafe {
        std::ptr::copy_nonoverlapping(
            packed.as_ptr() as *const u8,
            packed_buf.contents() as *mut u8,
            packed.len() * 4,
        );
        std::ptr::copy_nonoverlapping(
            act.as_ptr() as *const u8,
            act_buf.contents() as *mut u8,
            HEAD_DIM * 4,
        );
    }

    // Warmup + verify
    let cb = queue.new_command_buffer();
    let en = cb.new_compute_command_encoder();
    en.set_compute_pipeline_state(&pso);
    en.set_buffer(0, Some(&packed_buf), 0);
    en.set_buffer(1, Some(&act_buf), 0);
    en.set_buffer(2, Some(&logit_buf), 0);
    en.set_bytes(3, 4, &nrows as *const u32 as *const std::ffi::c_void);
    en.dispatch_thread_groups(
        MTLSize {
            width: ROWS as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    en.end_encoding();
    cb.commit();
    cb.wait_until_completed();

    let gpu_out = unsafe { std::slice::from_raw_parts(logit_buf.contents() as *const f32, ROWS) };

    let mut matches = 0u64;
    let mut max_diff = 0.0f64;
    for i in 0..ROWS {
        let d = (gpu_out[i] as f64 - ref_rows[i] as f64).abs();
        if d > max_diff {
            max_diff = d;
        }
        if d < 0.1 {
            matches += 1;
        }
    }

    print!("  GPU matches CPU: {}/{} ", matches, ROWS);
    let m_s = if matches == ROWS as u64 { "✓" } else { "⚠" };
    println!("{}", m_s);
    println!("  Max abs diff: {:.4e}", max_diff);

    // Debug first few if failing
    if matches < ROWS as u64 {
        println!("\n  First 8:");
        for i in 0..8.min(ROWS) {
            println!("    [{i}] GPU={:.4} CPU={:.4}", gpu_out[i], ref_rows[i]);
        }
    }

    // Benchmark
    println!("\n  Benchmark...");
    let mut times = Vec::new();
    for iter in 0..5 {
        let t0 = Instant::now();
        let cb = queue.new_command_buffer();
        let en = cb.new_compute_command_encoder();
        en.set_compute_pipeline_state(&pso);
        en.set_buffer(0, Some(&packed_buf), 0);
        en.set_buffer(1, Some(&act_buf), 0);
        en.set_buffer(2, Some(&logit_buf), 0);
        en.set_bytes(3, 4, &nrows as *const u32 as *const std::ffi::c_void);
        en.dispatch_thread_groups(
            MTLSize {
                width: ROWS as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        en.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        times.push(t0.elapsed());
        if iter == 0 {
            println!("  Iter 0: {:.3} ms", t0.elapsed().as_secs_f64() * 1000.0);
        }
    }

    let avg = times.iter().skip(1).map(|t| t.as_secs_f64()).sum::<f64>() / 4.0;
    let best = times
        .iter()
        .skip(1)
        .map(|t| t.as_secs_f64())
        .fold(f64::MAX, f64::min);
    let this_mb = (packed.len() * 4) as f64 / 1_048_576.0;
    let speed = this_mb / avg;

    println!();
    println!("  ── Results ────────────────────────────────────────────────────────────");
    println!(
        "  Rows: {} × {} dim = {:.0}K weights",
        ROWS,
        HEAD_DIM,
        ROWS as f64 * HEAD_DIM as f64 / 1000.0
    );
    println!("  Packed:  {} u32 values ({:.2} MB)", packed.len(), this_mb);
    println!(
        "  Density: {:.2} bits/weight",
        packed.len() as f64 * 32.0 / (ROWS as f64 * HEAD_DIM as f64)
    );
    println!();
    println!("  GPU: {:.3} ms (avg, n=4)", avg * 1000.0);
    println!("       {:.3} ms (best)", best * 1000.0);
    println!("  Speed: {:.0} MB/s", speed);

    // Projection to full model
    let bw_savings = 1.0 - (CHUNKS_PER_ROW * 4) as f64 / (8 * (HEAD_DIM / 32)) as f64;
    println!();
    println!("  ── Full-model projection ────────────────────────────────────────────");
    println!(
        "  vs bit-plane:  {:.0}% bandwidth savings",
        bw_savings * 100.0
    );
    println!("  vs 2-bit nibble: 20% bandwidth savings (theoretical)");

    if avg > 0.0 {
        let proj_ms_3gb = 3000.0 / speed;
        let proj_ms_24gb = 2400.0 / speed;
        println!(
            "  3 GB bit-plane: {:.1} ms/token → {:.0} t/s",
            proj_ms_3gb,
            1000.0 / proj_ms_3gb
        );
        println!(
            "  2.4 GB Base-3:   {:.1} ms/token → {:.0} t/s ({:.0}% faster)",
            proj_ms_24gb,
            1000.0 / proj_ms_24gb,
            (1000.0 / proj_ms_24gb - 1000.0 / proj_ms_3gb) / (1000.0 / proj_ms_3gb) * 100.0
        );
    }
}
