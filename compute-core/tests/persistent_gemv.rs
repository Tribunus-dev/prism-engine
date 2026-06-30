//! Production correctness + benchmark for T32 coalesced uint4 GEMV kernel.
//!
//! Run: cargo test --test persistent_gemv --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use metal::*;
use std::io::Write;
use std::time::Instant;
use tribunus_compute_core::compute_image::compile::int4_pack::{
    AlignedTernaryBlock32, TernaryBlock32, quantize_to_ternary_block32,
};
use tribunus_compute_core::compute_image::megakernel::{
    dispatch_persistent_gemv, PERSISTENT_GEMV_SRC, PERSISTENT_GEMV_ROWS_PER_TG,
    PERSISTENT_GEMV_THREADS_PER_TG,
};

const HIDDEN_DIM: usize = 3840;
const BLOCKS_PER_ROW: usize = HIDDEN_DIM / 32; // 120

const TEST_ROWS: usize = 140;
const TOTAL_BLOCKS: usize = TEST_ROWS * BLOCKS_PER_ROW;

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(seed) }
    fn f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let bits = (self.0 >> 40) as u32;
        f32::from_bits((bits >> 9) | 0x3F80_0000) - 1.0
    }
}

fn cpu_matvec(weights: &[TernaryBlock32], activation: &[f32], rows: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows];
    for r in 0..rows {
        let mut acc = 0.0f32;
        for b in 0..BLOCKS_PER_ROW {
            let block = &weights[r * BLOCKS_PER_ROW + b];
            let scale = f32::from(half::f16::from_bits(block.block_scale));
            for elem in 0..32 {
                let byte_idx = elem / 5;
                let byte_val = block.packed_trits[byte_idx];
                let trit = if byte_idx >= 6 {
                    if elem == 30 { byte_val as u32 % 3 } else { (byte_val as u32 / 3) % 3 }
                } else {
                    let mut v = byte_val as u32;
                    for _ in 0..(elem % 5) { v = (v * 86) >> 8; }
                    let q = (v * 86) >> 8;
                    v.wrapping_sub(q * 3)
                };
                acc += (trit as i8 - 1) as f32 * scale * activation[b * 32 + elem];
            }
        }
        out[r] = acc;
    }
    out
}

#[test]
fn persistent_gemv_correctness_and_benchmark() {
    let mut rng = Rng::new(42);
    let dev = Device::system_default().expect("Metal device");
    let shared = MTLResourceOptions::StorageModeShared;

    println!("\n═══ T32 Coalesced GEMV — Correctness + Benchmark ═══\n");
    println!("  Rows: {}", TEST_ROWS);
    println!("  Hidden dim: {}", HIDDEN_DIM);

    // Generate 9-byte weights (for CPU ref) and 16-byte aligned weights (for GPU)
    let mut weights_9 = Vec::with_capacity(TOTAL_BLOCKS);
    let mut aligned_16 = Vec::with_capacity(TOTAL_BLOCKS * 16);

    for _ in 0..TOTAL_BLOCKS {
        let mut f32_block = [0.0f32; 32];
        for i in 0..32 { f32_block[i] = ((rng.f32() * 3.0 - 1.5) as i8).clamp(-1, 1) as f32; }
        let max_abs = f32_block.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1.0);
        for v in &mut f32_block { *v *= max_abs; }

        let tb = quantize_to_ternary_block32(&f32_block);
        // Copy fields manually for CPU ref (9-byte) + GPU aligned (16-byte)
        let mut pt = [0u8; 7];
        pt.copy_from_slice(&tb.packed_trits);
        let bs = tb.block_scale;
        let ab = AlignedTernaryBlock32 {
            packed_trits: tb.packed_trits,
            block_scale: tb.block_scale,
            padding: [0u8; 7],
        };
        weights_9.push(TernaryBlock32 { packed_trits: pt, block_scale: bs });
        aligned_16.extend_from_slice(&ab.packed_trits);
        aligned_16.extend_from_slice(&ab.block_scale.to_le_bytes());
        aligned_16.extend_from_slice(&ab.padding);
    }

    // Activation (f32 for CPU, f16 for GPU)
    let mut activation_f32 = Vec::with_capacity(HIDDEN_DIM);
    for _ in 0..HIDDEN_DIM { activation_f32.push(rng.f32()); }
    
    let act_half: Vec<u16> = activation_f32.iter().map(|&v| half::f16::from_f32(v).to_bits()).collect();

    // CPU reference
    println!("  CPU reference...");
    let cpu_start = Instant::now();
    let cpu_ref = cpu_matvec(&weights_9, &activation_f32, TEST_ROWS);
    println!("  CPU time: {:.1} µs", cpu_start.elapsed().as_secs_f64() * 1e6);

    // Compile kernel
    println!("  Compiling kernel...");
    let pso = compile_gemv_kernel(&dev, PERSISTENT_GEMV_SRC, "matvec_persistent_t32_coalesced");
    let queue = dev.new_command_queue();

    // GPU buffers
    let w_buf = dev.new_buffer(aligned_16.len() as u64, MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeWriteCombined);
    unsafe { std::ptr::copy_nonoverlapping(aligned_16.as_ptr(), w_buf.contents() as *mut u8, aligned_16.len()); }
    let a_buf = dev.new_buffer((HIDDEN_DIM * 2) as u64, MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeWriteCombined);
    unsafe { std::ptr::copy_nonoverlapping(act_half.as_ptr() as *const u8, a_buf.contents() as *mut u8, HIDDEN_DIM * 2); }
    let o_buf = dev.new_buffer((TEST_ROWS * 2) as u64, shared);

    // Warmup + correctness
    println!("  Warmup + correctness...");
    {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        dispatch_persistent_gemv(&enc, &pso, &w_buf, &a_buf, &o_buf, TEST_ROWS);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    let gpu_half = unsafe { std::slice::from_raw_parts(o_buf.contents() as *const u16, TEST_ROWS) };
    let gpu_out: Vec<f32> = gpu_half.iter().map(|&b| half::f16::from_bits(b).to_f32()).collect();

    println!("  First 8 GPU/CPU:");
    for i in 0..8.min(TEST_ROWS) {
        println!("    [{}] GPU={:.6} CPU={:.6}", i, gpu_out[i], cpu_ref[i]);
    }

    let tol = 0.1;
    let mut matches = 0u64;
    let mut max_diff = 0.0f64;
    for i in 0..TEST_ROWS {
        let d = (gpu_out[i] as f64 - cpu_ref[i] as f64).abs();
        if d > max_diff { max_diff = d; }
        if d < tol as f64 { matches += 1; }
    }

    print!("  GPU matches CPU: {}/{} ", matches, TEST_ROWS);
    if matches == TEST_ROWS as u64 { println!("✓"); }
    else { println!("⚠ (max diff {:.2e})", max_diff); }
    assert!(matches as f64 / TEST_ROWS as f64 > 0.95,
        "Match rate too low: {}/{} (max diff {:.2e})", matches, TEST_ROWS, max_diff);

    std::io::stdout().flush().unwrap();

    // Benchmark
    const ITERS: usize = 20;
    println!("  Benchmark ({ITERS} iters)...");

    for _ in 0..3 { // warmup
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        dispatch_persistent_gemv(&enc, &pso, &w_buf, &a_buf, &o_buf, TEST_ROWS);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    let mut times = Vec::new();
    for _ in 0..ITERS {
        let t0 = Instant::now();
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        dispatch_persistent_gemv(&enc, &pso, &w_buf, &a_buf, &o_buf, TEST_ROWS);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        times.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[ITERS / 2];

    println!("\n═══ Results ═══");
    println!("  Median: {:.1} µs", median);
    println!("  Extrapolated decode ({TEST_ROWS} rows → 15360 rows Gate):");
    let gate_est = median * (15360.0 / TEST_ROWS as f64);
    let per_layer = gate_est * 4.0; // rough: Gate+Up+Down ≈ 3×, Q+K+V+O ≈ 1× = 4× Gate equiv
    let per_token = per_layer * 48.0;
    println!("    Gate/Up:  ~{:.1} µs", gate_est);
    println!("    Per layer: ~{:.1} µs", per_layer);
    println!("    Per token: {:.1} ms", per_token / 1000.0);
    println!("    Tokens/s:  ~{:.1}", 1_000_000.0 / per_token);

    let useful_bw = (TEST_ROWS * BLOCKS_PER_ROW * 9) as f64 / (median / 1e6) / 1e6;
    println!("    Useful BW: {:.0} MB/s", useful_bw);
    println!("    M1 peak:   ~70,000 MB/s");
    println!("    Efficiency: {:.1}%", useful_bw / 70000.0 * 100.0);
    println!();
}

fn compile_gemv_kernel(dev: &Device, src: &str, entry: &str) -> ComputePipelineState {
    let tmp = std::env::temp_dir().join("tribunus-gemv-prod");
    let _ = std::fs::create_dir_all(&tmp);
    let s = tmp.join("k.metal");
    let a = tmp.join("k.air");
    let l = tmp.join("k.metallib");
    std::fs::write(&s, src).unwrap();
    assert!(std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-std=metal4.0", "-O3", "-c"])
        .arg(s.to_str().unwrap()).arg("-o").arg(a.to_str().unwrap())
        .status().unwrap().success());
    assert!(std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib", "-o"])
        .arg(l.to_str().unwrap()).arg(a.to_str().unwrap())
        .status().unwrap().success());
    let bytes = std::fs::read(&l).unwrap();
    let lib = dev.new_library_with_data(&bytes).unwrap();
    let f = lib.get_function(entry, None).unwrap();
    dev.new_compute_pipeline_state_with_function(&f).unwrap()
}
