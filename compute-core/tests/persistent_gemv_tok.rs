//! tok/s benchmark: persistent row-batched GEMV in full decode workload.
//!
//! Measures per-layer matvec throughput for all 7 projections:
//!   Q(4096×3840), K(2048×3840), V(2048×3840), O(4096×3840),
//!   Gate(15360×3840), Up(15360×3840), Down(3840×640×24)
//!
//! Reports per-token latency and tok/s for 48-layer Gemma 4 decode.
//!
//! Run: cargo test --test persistent_gemv_tok --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use metal::*;
use std::io::Write;
use std::time::Instant;
use tribunus_compute_core::compute_image::compile::int4_pack::quantize_to_ternary_block32;
use tribunus_compute_core::compute_image::megakernel::{
    dispatch_persistent_gemv, PERSISTENT_GEMV_ROWS_PER_TG, PERSISTENT_GEMV_THREADS_PER_TG, PERSISTENT_GEMV_SRC,
};

const HIDDEN: usize = 3840;
const FFN_INTER: usize = 15360;
const TILE: usize = 640;
const LAYERS: usize = 48;

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Self(s) }
    fn f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        f32::from_bits(((self.0 >> 40) as u32 >> 9) | 0x3F80_0000) - 1.0
    }
}

/// Generate ternary weights for a (rows × hidden_dim) matrix.
fn gen_weights(rng: &mut Rng, rows: usize, hidden_dim: usize) -> Vec<u8> {
    let blocks_per_row = hidden_dim / 32;
    let total_blocks = rows * blocks_per_row;
    let mut bytes = Vec::with_capacity(total_blocks * 9);
    for _ in 0..total_blocks {
        let mut f32_block = [0.0f32; 32];
        for i in 0..32 {
            let t: i8 = (rng.f32() * 3.0 - 1.5) as i8;
            f32_block[i] = t.clamp(-1, 1) as f32;
        }
        // Scale so weights are non-trivial
        let max_abs = f32_block.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1.0);
        for v in &mut f32_block { *v *= max_abs; }
        let block = quantize_to_ternary_block32(&f32_block);
        bytes.extend_from_slice(&block.packed_trits);
        bytes.extend_from_slice(&block.block_scale.to_le_bytes());
    }
    bytes
}

/// Generate activation vector (half-float bytes).
fn gen_activation(rng: &mut Rng, dim: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(dim * 2);
    for _ in 0..dim {
        let v = half::f16::from_f32(rng.f32());
        bytes.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    bytes
}

fn compile_kernel(src: &str, entry: &str) -> (ComputePipelineState, CommandQueue, Device) {
    let dev = Device::system_default().expect("Metal device");
    let tmp = std::env::temp_dir().join("tribunus-gemv-bench");
    let _ = std::fs::create_dir_all(&tmp);
    let s = tmp.join("kernel.metal");
    let a = tmp.join("kernel.air");
    let l = tmp.join("kernel.metallib");
    std::fs::write(&s, src).unwrap();
    assert!(std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-std=metal3.2", "-O3", "-c"])
        .arg(s.to_str().unwrap()).arg("-o").arg(a.to_str().unwrap())
        .status().unwrap().success());
    assert!(std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib", "-o"])
        .arg(l.to_str().unwrap()).arg(a.to_str().unwrap())
        .status().unwrap().success());
    let bytes = std::fs::read(&l).unwrap();
    let lib = dev.new_library_with_data(&bytes).unwrap();
    let f = lib.get_function(entry, None).unwrap();
    let pso = dev.new_compute_pipeline_state_with_function(&f).unwrap();
    (pso, dev.new_command_queue(), dev)
}

// ── 640-dim kernel variant for Down projection ─────────────────────
// Same structure as PERSISTENT_GEMV_SRC but with HIDDEN_DIM=640.
const PERSISTENT_GEMV_SRC_640: &str = r##"#include <metal_stdlib>
using namespace metal;

struct __attribute__((packed)) TernaryBlock32 {
    uchar packed_trits[7];
    half  block_scale;
};

inline void unpack_5_trits(uchar byte, thread half* out_regs) {
    for (int i = 0; i < 5; ++i) {
        uint q = (byte * 86u) >> 8;
        int8_t trit = byte - (q * 3);
        out_regs[i] = static_cast<half>(trit - 1);
        byte = q;
    }
}

kernel void matvec_persistent_batched_640(
    device const TernaryBlock32* weight_stream [[buffer(0)]],
    device const half* activation_vector       [[buffer(1)]],
    device half* output_vector                 [[buffer(2)]],
    uint thread_index                          [[thread_index_in_threadgroup]],
    uint2 tg_position                          [[threadgroup_position_in_grid]])
{
    threadgroup half sram_activations[128];
    const uint base_row  = tg_position.x * 4;
    const uint simd_id   = thread_index / 32;
    const uint lane_id   = thread_index % 32;
    const uint assigned_row = base_row + simd_id;
    const uint total_chunks = 640 / 128;
    float my_accum = 0.0f;
    for (uint chunk = 0; chunk < total_chunks; ++chunk) {
        if (thread_index < 128) {
            sram_activations[thread_index] = activation_vector[chunk * 128 + thread_index];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint b = 0; b < 4; ++b) {
            uint global_block_idx = (assigned_row * (640 / 32)) + (chunk * 4) + b;
            device const TernaryBlock32& block = weight_stream[global_block_idx];
            half scale = block.block_scale;
            uchar byte_val = block.packed_trits[lane_id / 5];
            half w_val;
            if (lane_id >= 30) {
                uint trit = (lane_id == 30) ? ((uint)byte_val % 3) : ((uint)byte_val / 3 % 3);
                w_val = (half)((int)trit - 1) * scale;
            } else {
                uint v = (uint)byte_val;
                for (uint i = 0; i < lane_id % 5; ++i) {
                    v = (v * 86u) >> 8;
                }
                uint q = (v * 86u) >> 8;
                uint trit = v - q * 3;
                w_val = (half)((int)trit - 1) * scale;
            }
            my_accum += static_cast<float>(w_val * sram_activations[b * 32 + lane_id]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total = simd_sum(my_accum);
    if (lane_id == 0) {
        output_vector[assigned_row] = static_cast<half>(total);
    }
}
"##;

#[test]
fn decode_tok_benchmark() {
    let mut rng = Rng::new(42);
    let dev = Device::system_default().expect("Metal device");

    // Projection configs: (name, rows, hidden_dim, repeats_for_down)
    #[derive(Clone, Copy)]
    struct Proj {
        name: &'static str,
        rows: usize,
        hidden: usize,
        repeats: usize, // how many dispatches (24 for Down tiles)
    }
    const PROJS: &[Proj] = &[
        Proj { name: "Q",   rows: 4096,  hidden: HIDDEN,     repeats: 1 },
        Proj { name: "K",   rows: 2048,  hidden: HIDDEN,     repeats: 1 },
        Proj { name: "V",   rows: 2048,  hidden: HIDDEN,     repeats: 1 },
        Proj { name: "O",   rows: 4096,  hidden: HIDDEN,     repeats: 1 },
        Proj { name: "Gate", rows: 15360, hidden: HIDDEN,    repeats: 1 },
        Proj { name: "Up",  rows: 15360, hidden: HIDDEN,     repeats: 1 },
        Proj { name: "Down", rows: HIDDEN, hidden: TILE,     repeats: 24 },
    ];

    println!("
═══ Decode tok/s Benchmark ═══
");
    println!("  Model:     Gemma 4 12B ({} layers)", LAYERS);
    println!("  Dispatches per layer: 7 projections");
    println!("  Kernel:    Persistent row-batched (4 rows/TG, {} threads/TG)",
        PERSISTENT_GEMV_THREADS_PER_TG);
    println!("  SRAM:      {} bytes/TG", PERSISTENT_GEMV_THREADS_PER_TG * 2);

    // Compile both kernel variants
    println!("
  Compiling 3840-dim kernel...");
    let (pso_3840, queue, dev) = compile_kernel(PERSISTENT_GEMV_SRC, "matvec_persistent_batched");
    println!("  Compiling 640-dim kernel (Down)...");
    let (pso_640, _, _) = compile_kernel(PERSISTENT_GEMV_SRC_640, "matvec_persistent_batched_640");

    // Generate all weights and activations
    println!("  Generating weights and activations...");
    struct Prepped {
        weight_buf: Buffer,
        act_buf: Buffer,
        out_buf: Buffer,
        tgs: u64,
    }

    let mut prepped = Vec::new();
    for &p in PROJS {
        let weight_bytes = gen_weights(&mut rng, p.rows, p.hidden);
        let act_bytes = gen_activation(&mut rng, p.hidden);

        let weight_buf = dev.new_buffer(weight_bytes.len() as u64, MTLResourceOptions::StorageModeShared);
        unsafe { std::ptr::copy_nonoverlapping(weight_bytes.as_ptr(), weight_buf.contents() as *mut u8, weight_bytes.len()); }

        let act_buf = dev.new_buffer(act_bytes.len() as u64, MTLResourceOptions::StorageModeShared);
        unsafe { std::ptr::copy_nonoverlapping(act_bytes.as_ptr(), act_buf.contents() as *mut u8, act_bytes.len()); }

        let out_buf = dev.new_buffer((p.rows * 2) as u64, MTLResourceOptions::StorageModeShared);

        let tgs = (p.rows / PERSISTENT_GEMV_ROWS_PER_TG) as u64;
        prepped.push(Prepped { weight_buf, act_buf, out_buf, tgs });
    }

    // ── Warmup: 3 iterations of all projections ──────────────────
    println!("  Warmup...");
    for _ in 0..3 {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        for (i, &p) in PROJS.iter().enumerate() {
            let pso = if p.hidden == TILE { &pso_640 } else { &pso_3840 };
            for _ in 0..p.repeats {
                dispatch_persistent_gemv_generic(&enc, pso,
                    &prepped[i].weight_buf, &prepped[i].act_buf, &prepped[i].out_buf,
                    prepped[i].tgs);
            }
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // ── Measure: 20 iterations of all projections (per-layer cost) ──
    const ITERS: usize = 20;
    let mut layer_times = Vec::with_capacity(ITERS);

    for _ in 0..ITERS {
        let t0 = Instant::now();
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        for (i, &p) in PROJS.iter().enumerate() {
            let pso = if p.hidden == TILE { &pso_640 } else { &pso_3840 };
            for _ in 0..p.repeats {
                dispatch_persistent_gemv_generic(&enc, pso,
                    &prepped[i].weight_buf, &prepped[i].act_buf, &prepped[i].out_buf,
                    prepped[i].tgs);
            }
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        layer_times.push(t0.elapsed().as_secs_f64());
    }

    layer_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_layer_s = layer_times[ITERS / 2];
    let min_layer_s = layer_times[0];
    let max_layer_s = layer_times[ITERS - 1];

    let per_token_s = median_layer_s * LAYERS as f64;
    let tok_per_s = 1.0 / per_token_s;

    // ── Report ─────────────────────────────────────────────────
    println!("
  Per-projection dispatch details:");
    for (i, &p) in PROJS.iter().enumerate() {
        println!("    {:>6}: {:>5} rows × {:>5} dim, {} TGs, {} rep(s)",
            p.name, p.rows, p.hidden, prepped[i].tgs, p.repeats);
    }

    println!("
═══ Per-Layer Latency ({} projections) ═══", PROJS.len());
    println!("  Median:  {:.3} ms", median_layer_s * 1e3);
    println!("  Min:     {:.3} ms", min_layer_s * 1e3);
    println!("  Max:     {:.3} ms", max_layer_s * 1e3);

    println!("
═══ Decode Throughput ({} layers) ═══", LAYERS);
    println!("  Per-token GPU time: {:.1} ms", per_token_s * 1e3);
    println!("  Tokens/sec:         {:.1}", tok_per_s);
    println!();

    // Per-projection breakdown
    // (measure each projection individually for the breakdown)
    println!("  Per-projection timing (individual dispatches):");
    for (i, &p) in PROJS.iter().enumerate() {
        let mut times = Vec::new();
        let pso = if p.hidden == TILE { &pso_640 } else { &pso_3840 };
        for _ in 0..10 {
            let t0 = Instant::now();
            let cb = queue.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            for _ in 0..p.repeats {
                dispatch_persistent_gemv_generic(&enc, pso,
                    &prepped[i].weight_buf, &prepped[i].act_buf, &prepped[i].out_buf,
                    prepped[i].tgs);
            }
            enc.end_encoding();
            cb.commit();
            cb.wait_until_completed();
            times.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = times[5];
        println!("    {:>6}: {:.1} µs/layer ({} disp × {} rep)",
            p.name, med, prepped[i].tgs, p.repeats);
    }

    println!("
  Occupancy estimate (3840-dim):");
    let concurrent_3840 = 64u64;
    let tgs_3840 = prepped[0].tgs;
    println!("    Concurrent TGs:   {}", concurrent_3840);
    println!("    TGs (Q 4096):     {}", (4096 / PERSISTENT_GEMV_ROWS_PER_TG));
    println!("    TGs (Gate 15360): {}", (15360 / PERSISTENT_GEMV_ROWS_PER_TG));
    println!("    Waves (Gate):     ~{}", (15360 / PERSISTENT_GEMV_ROWS_PER_TG) as f64 / concurrent_3840 as f64);
    println!();
}

/// Generic dispatch for any row count.
fn dispatch_persistent_gemv_generic(
    encoder: &ComputeCommandEncoderRef,
    pso: &ComputePipelineState,
    weight_stream: &BufferRef,
    activation_vector: &BufferRef,
    output_vector: &BufferRef,
    threadgroups: u64,
) {
    encoder.set_compute_pipeline_state(pso);
    encoder.set_buffer(0, Some(weight_stream), 0);
    encoder.set_buffer(1, Some(activation_vector), 0);
    encoder.set_buffer(2, Some(output_vector), 0);
    encoder.dispatch_thread_groups(
        MTLSize { width: threadgroups, height: 1, depth: 1 },
        MTLSize { width: PERSISTENT_GEMV_THREADS_PER_TG as u64, height: 1, depth: 1 },
    );
}
