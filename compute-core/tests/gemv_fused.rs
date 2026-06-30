//! Fused projection kernel: Q(4096) K(2048) V(2048) O(4096) Gate(15360) Up(15360)
//! All share the same 3840-dim activation. Single dispatch per layer.
//!
//! Run: cargo test --test gemv_fused --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use metal::*;
use std::time::Instant;
use tribunus_compute_core::compute_image::compile::int4_pack::{AlignedTernaryBlock32, quantize_to_ternary_block32};

const HIDDEN: usize = 3840;
const BPR: usize = HIDDEN / 32; // 120

// Projection dimensions (rows)
const Q_ROWS: usize = 4096;
const K_ROWS: usize = 2048;
const V_ROWS: usize = 2048;
const O_ROWS: usize = 4096;
const GATE_ROWS: usize = 15360;
const UP_ROWS: usize = 15360;
const MAX_ROWS: usize = 15360;
const TGS: usize = MAX_ROWS / 4; // 3840 TGs for 4-row groups

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self { Self(s) }
    fn f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        f32::from_bits(((self.0 >> 40) as u32 >> 9) | 0x3F80_0000) - 1.0
    }
}

/// Generate AlignedTernaryBlock32 weights for a matrix.
fn gen_weights(rng: &mut Rng, rows: usize) -> Vec<u8> {
    let total = rows * BPR;
    let mut bytes = Vec::with_capacity(total * 16);
    for _ in 0..total {
        let mut fb = [0.0f32; 32];
        for i in 0..32 { fb[i] = ((rng.f32() * 3.0 - 1.5) as i8).clamp(-1, 1) as f32; }
        let s = fb.iter().map(|v| v.abs()).fold(0.0f32, f32::max).max(1.0);
        for v in &mut fb { *v *= s; }
        let tb = quantize_to_ternary_block32(&fb);
        let ab: AlignedTernaryBlock32 = tb.into();
        bytes.extend_from_slice(&ab.packed_trits);
        bytes.extend_from_slice(&ab.block_scale.to_le_bytes());
        bytes.extend_from_slice(&ab.padding);
    }
    bytes
}

fn gen_activation(rng: &mut Rng) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(HIDDEN * 2);
    for _ in 0..HIDDEN { bytes.extend_from_slice(&half::f16::from_f32(rng.f32()).to_bits().to_le_bytes()); }
    bytes
}

fn compile_msl(src: &str, entry: &str, dev: &Device) -> ComputePipelineState {
    let tmp = std::env::temp_dir().join("gemv-fused");
    let _ = std::fs::create_dir_all(&tmp);
    let s = tmp.join("k.metal"); let a = tmp.join("k.air"); let l = tmp.join("k.metallib");
    std::fs::write(&s, src).unwrap();
    assert!(std::process::Command::new("xcrun").args(["-sdk","macosx","metal","-std=metal4.0","-O3","-c"]).arg(s.to_str().unwrap()).arg("-o").arg(a.to_str().unwrap()).status().unwrap().success());
    assert!(std::process::Command::new("xcrun").args(["-sdk","macosx","metallib","-o"]).arg(l.to_str().unwrap()).arg(a.to_str().unwrap()).status().unwrap().success());
    let lib = dev.new_library_with_data(&std::fs::read(&l).unwrap()).unwrap();
    let f = lib.get_function(entry, None).unwrap();
    dev.new_compute_pipeline_state_with_function(&f).unwrap()
}

const FUSED_KERNEL: &str = r##"#include <metal_stdlib>
using namespace metal;

constant uint HIDDEN_DIM = 3840;
constant uint BPR = HIDDEN_DIM / 32;

// Row counts per projection (max 15360, quantized to 4-row groups)
constant uint Q_TG     = 4096 / 4;
constant uint KV_TG    = 2048 / 4;
constant uint GATE_TG  = 15360 / 4;

// Weight buffer offsets (in blocks, set from Rust side)
// Q: 0..4096*120, K: 4096*120..(4096+2048)*120, V: .., O: .., Gate: .., Up: ..

kernel void fused_projections(
    device const uint4* weight_stream   [[buffer(0)]],
    device const half*  activation      [[buffer(1)]],
    device half*        out_q           [[buffer(2)]],
    device half*        out_k           [[buffer(3)]],
    device half*        out_v           [[buffer(4)]],
    device half*        out_o           [[buffer(5)]],
    device half*        out_gate        [[buffer(6)]],
    device half*        out_up          [[buffer(7)]],
    constant uint4&     wg_offsets      [[buffer(8)]], // {Q,K,V,O,Gate,Up} × BPR × offset
    uint ti                              [[thread_index_in_threadgroup]],
    uint2 tp                             [[threadgroup_position_in_grid]])
{
    // Load activation once into SRAM (cooperative, all 128 threads)
    threadgroup half sram[HIDDEN_DIM];
    for (uint i = ti; i < HIDDEN_DIM; i += 128) { sram[i] = activation[i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint base_row = tp.x * 4;
    const uint simd_id  = ti / 32;
    const uint lane     = ti % 32;

    half unpacked[32];
    half act_reg[32];

    // Helper: process one projection for this row group
    // proj_offset = offset in blocks from weight_stream start
    // proj_rows = number of rows in this projection
    // out_buf = output buffer
    // 
    // Each SIMD group handles one of 4 rows.
    // branchless masking for TGs beyond the projection's row count.
#define PROCESS_PROJ(offset, proj_rows, out_buf) \
    do { \
        bool proj_valid = (base_row < proj_rows); \
        uint proj_row = base_row + simd_id; \
        bool row_valid = (proj_row < proj_rows); \
        float row_acc = 0.0f; \
        if (proj_valid && row_valid) { \
            uint rb = (offset) + proj_row * BPR; \
            for (uint bg = 0; bg < 128; bg += 32) { \
                uint g = bg / 32; \
                uint act_base = g * 1024 + lane * 32; \
                for (uint e = 0; e < 32; ++e) { act_reg[e] = sram[act_base + e]; } \
                uint lb = bg + lane; \
                uint sb = (lb < BPR) ? lb : (BPR - 1); \
                uint4 vec = weight_stream[rb + sb]; \
                thread const uchar* raw = (thread const uchar*)&vec; \
                ushort sc = ((ushort)raw[7]) | ((ushort)raw[8] << 8); \
                half scale = as_type<half>(sc); \
                for (uint i = 0; i < 7; ++i) { \
                    uint v = raw[i]; uint n = (i < 6) ? 5 : 2; \
                    for (uint j = 0; j < n; ++j) { unpacked[i*5+j] = (half)((int)(v%3)-1); v/=3; } \
                } \
                float ls = 0.0f; \
                for (uint e = 0; e < 32; ++e) { ls += (float)(unpacked[e] * scale * act_reg[e]); } \
                row_acc += (lb < BPR) ? ls : 0.0f; \
            } \
        } \
        float total = simd_sum(row_acc); \
        if (lane == 0 && row_valid) { out_buf[proj_row] = (half)total; } \
        threadgroup_barrier(mem_flags::mem_threadgroup); \
    } while(0)

    // Order: Q K V O Gate Up (all share the same SRAM activation)
    // Offsets would come from a constant buffer in production
    // For test we hardcode: Q=0, K=Q_ROWS*BPR, V=K_ROWS*BPR, etc.
    // But we pass them through wg_offsets buffer

    uint q_off  = 0;
    uint k_off  = q_off + 4096 * BPR;
    uint v_off  = k_off + 2048 * BPR;
    uint o_off  = v_off + 2048 * BPR;
    uint g_off  = o_off + 4096 * BPR;
    uint u_off  = g_off + 15360 * BPR;

    PROCESS_PROJ(q_off, 4096, out_q);
    PROCESS_PROJ(k_off, 2048, out_k);
    PROCESS_PROJ(v_off, 2048, out_v);
    PROCESS_PROJ(o_off, 4096, out_o);
    PROCESS_PROJ(g_off, 15360, out_gate);
    PROCESS_PROJ(u_off, 15360, out_up);
}
"##;

#[test]
fn fused_projections_benchmark() {
    let mut rng = Rng::new(42);
    let dev = Device::system_default().expect("Metal device");
    let shared = MTLResourceOptions::StorageModeShared | MTLResourceOptions::CPUCacheModeWriteCombined;
    let queue = dev.new_command_queue();

    println!("\n═══ Fused Projections Kernel ═══\n");

    // Generate all 6 weight matrices concatenated
    println!("  Generating weights...");
    let w_q     = gen_weights(&mut rng, Q_ROWS);
    let w_k     = gen_weights(&mut rng, K_ROWS);
    let w_v     = gen_weights(&mut rng, V_ROWS);
    let w_o     = gen_weights(&mut rng, O_ROWS);
    let w_gate  = gen_weights(&mut rng, GATE_ROWS);
    let w_up    = gen_weights(&mut rng, UP_ROWS);

    let mut all_weights = Vec::new();
    all_weights.extend_from_slice(&w_q);
    all_weights.extend_from_slice(&w_k);
    all_weights.extend_from_slice(&w_v);
    all_weights.extend_from_slice(&w_o);
    all_weights.extend_from_slice(&w_gate);
    all_weights.extend_from_slice(&w_up);
    println!("  Total weight data: {:.1} MB", all_weights.len() / 1_000_000);

    let act_bytes = gen_activation(&mut rng);

    let w_buf = dev.new_buffer(all_weights.len() as u64, shared);
    unsafe { std::ptr::copy_nonoverlapping(all_weights.as_ptr(), w_buf.contents() as *mut u8, all_weights.len()); }
    let a_buf = dev.new_buffer(act_bytes.len() as u64, shared);
    unsafe { std::ptr::copy_nonoverlapping(act_bytes.as_ptr(), a_buf.contents() as *mut u8, act_bytes.len()); }

    let mk_buf = |size| { let b = dev.new_buffer(size as u64, MTLResourceOptions::StorageModeShared); b };
    let o_q    = mk_buf(Q_ROWS * 2);
    let o_k    = mk_buf(K_ROWS * 2);
    let o_v    = mk_buf(V_ROWS * 2);
    let o_o    = mk_buf(O_ROWS * 2);
    let o_gate = mk_buf(GATE_ROWS * 2);
    let o_up   = mk_buf(UP_ROWS * 2);

    // Compile
    println!("  Compiling kernel...");
    let pso = compile_msl(FUSED_KERNEL, "fused_projections", &dev);

    // Single dispatch: 3840 TGs × 128 threads
    let tg_count = TGS as u64;
    let tg_size = 128u64;

    // Warmup
    println!("  Warmup (3 iters)...");
    for _ in 0..3 {
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&w_buf), 0);
        enc.set_buffer(1, Some(&a_buf), 0);
        enc.set_buffer(2, Some(&o_q), 0);
        enc.set_buffer(3, Some(&o_k), 0);
        enc.set_buffer(4, Some(&o_v), 0);
        enc.set_buffer(5, Some(&o_o), 0);
        enc.set_buffer(6, Some(&o_gate), 0);
        enc.set_buffer(7, Some(&o_up), 0);
        enc.dispatch_thread_groups(
            MTLSize { width: tg_count, height: 1, depth: 1 },
            MTLSize { width: tg_size, height: 1, depth: 1 },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    // Benchmark: single fused dispatch for one layer's 6 projections
    println!("  Benchmark (15 iters)...");
    let mut times = Vec::new();
    for _ in 0..15 {
        let t0 = Instant::now();
        let cb = queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&w_buf), 0);
        enc.set_buffer(1, Some(&a_buf), 0);
        enc.set_buffer(2, Some(&o_q), 0);
        enc.set_buffer(3, Some(&o_k), 0);
        enc.set_buffer(4, Some(&o_v), 0);
        enc.set_buffer(5, Some(&o_o), 0);
        enc.set_buffer(6, Some(&o_gate), 0);
        enc.set_buffer(7, Some(&o_up), 0);
        enc.dispatch_thread_groups(
            MTLSize { width: tg_count, height: 1, depth: 1 },
            MTLSize { width: tg_size, height: 1, depth: 1 },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        times.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[7];

    // Per-projection weight data
    let w_mb = |bytes: &[u8]| bytes.len() as f64 / 1_000_000.0;

    println!("\n═══ Results (1 fused dispatch, 3840 TGs × 128 threads) ═══");
    println!("  Latency: {:.1} µs", med);
    println!("\n  Per-projection:");
    println!("    Q:     {:.1} MB", w_mb(&w_q));
    println!("    K:     {:.1} MB", w_mb(&w_k));
    println!("    V:     {:.1} MB", w_mb(&w_v));
    println!("    O:     {:.1} MB", w_mb(&w_o));
    println!("    Gate:  {:.1} MB", w_mb(&w_gate));
    println!("    Up:    {:.1} MB", w_mb(&w_up));
    println!("    Total: {:.1} MB", w_mb(&all_weights));

    let total_mb = w_mb(&all_weights);
    let bw = total_mb / (med / 1_000_000.0);
    println!("  Bandwidth: {:.1} GB/s", bw);
    println!("  M1 peak:   ~70 GB/s");
    println!("  Efficiency: {:.1}%", bw / 70.0 * 100.0);

    // Estimate per-layer + per-token
    let per_layer = med; // 6 projections in one shot
    let down_est = med * 0.3; // rough: Down is ~30% of Gate
    let per_token = (per_layer + down_est) * 48.0;
    let tok_s = 1_000_000.0 / per_token;
    println!("\n  Extrapolated decode (fused 6 proj + Down separate, 48 layers):");
    println!("    Per layer (fused 6): {:.1} µs", per_layer);
    println!("    Down (est):          {:.1} µs", down_est);
    println!("    Per token (48 lay):  {:.1} ms", per_token / 1000.0);
    println!("    Tokens/s:            ~{:.1}", tok_s);
    println!();
}
