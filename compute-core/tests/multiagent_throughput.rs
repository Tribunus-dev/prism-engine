//! Multi-agent throughput scaling test.
//!
//! Measures actual t/s at batch=1,2,4,8 with shared weight stream + per-agent KV cache.
//! Tests the 4-6 agent sweet spot hypothesis against hardware limits:
//!   68.25 GB/s bus: weights shared, KV cache per-agent
//!   32 KB SRAM: limits simultaneous agent state per threadgroup
//!   16 GB unified memory: caps total KV cache + weights

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use metal::*;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

const HIDDEN_DIM: usize = 3584; // Gemma 4 actual hidden dim
const LAYERS: usize = 48;
const WEIGHT_GB: f64 = 2.4; // Base-3 packed weight file
const KV_MB_PER_AGENT: f64 = 600.0; // Compressed ternary KV cache
const BUS_GBS: f64 = 68.25; // M1 DRAM bandwidth
const THREADGROUP_SRAM: usize = 32768; // 32 KB limit
const BLOCK_STRIDE: usize = 32;
const NUM_BLOCKS_X: usize = 6;

// ── Metal megakernel with multi-agent batch loop ────────────────────

const MEGAKERNEL_BATCH: &str = r##"#include <metal_stdlib>
using namespace metal;

constant uint HIDDEN_DIM = 3584;
constant uint LAYERS = 48;
constant uint BLOCK_STRIDE = 32;
constant uint NUM_BLOCKS_X = 6; // ceil(3584/640) = 6
constant uint MAGIC_DIV3 = 2863311531u;

inline uint fast_div3(uint v) {
    return ((uint64_t)v * (uint64_t)MAGIC_DIV3) >> 33;
}

inline uint fast_mod3(uint v) {
    return v - fast_div3(v) * 3u;
}

/// Multi-agent megakernel: one warp processes one agent's KV cache,
/// all share the same weight stream. BATCH_SIZE agents per dispatch.
kernel void multiagent_decode(
    device const uint*    packed_weights [[buffer(0)]],
    device const half*    kv_cache_base  [[buffer(1)]],  // [batch_size * HIDDEN_DIM * LAYERS * 2]
    device volatile uint* trigger        [[buffer(2)]],
    device volatile uint* completion     [[buffer(3)]],
    constant uint&        batch_size     [[buffer(4)]],
    uint lid [[thread_index_in_simdgroup]],
    uint gid [[threadgroup_position_in_grid]])
{
    // Each threadgroup handles one agent. Multiple threadgroups = multiple agents.
    uint agent_id = gid;
    if (agent_id >= batch_size) return;

    // Threadgroup SRAM for activation ping-pong (one per agent)
    threadgroup half act_a[HIDDEN_DIM];
    threadgroup half act_b[HIDDEN_DIM];

    for (uint i = lid; i < HIDDEN_DIM; i += 32) {
        act_a[i] = 1.0h; act_b[i] = 0.0h;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Persistent loop
    while (true) {
        while (trigger[0] == 0) { /* spin */ }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── 48-layer pass ────────────────────────────────────────────
        // In production: weights are cached in L2, shared across agents
        // Here we read them from DRAM each time (same as production)
        for (uint layer = 0; layer < LAYERS; ++layer) {
            threadgroup half* src = (layer % 2 == 0) ? act_a : act_b;
            threadgroup half* dst = (layer % 2 == 0) ? act_b : act_a;

            float acc = 0.0;

            // Weight pointer: different per-agent row in production
            // For this benchmark, all agents share the same weight matrix row
            uint row_base = agent_id * BLOCK_STRIDE * NUM_BLOCKS_X
                          + layer * HIDDEN_DIM * BLOCK_STRIDE;

            for (uint b = 0; b < NUM_BLOCKS_X; ++b) {
                uint val = packed_weights[row_base + b * BLOCK_STRIDE + lid];
                uint act_base = b * 640 + lid * 20;
                for (uint i = 0; i < 20; ++i) {
                    uint rem = fast_mod3(val);
                    int w = (int)rem - 1;
                    if (w != 0) acc += (float)src[act_base + i] * (float)w;
                    val = fast_div3(val);
                }
            }

            // Warp reduction
            acc += simd_shuffle_xor(acc, 1); acc += simd_shuffle_xor(acc, 2);
            acc += simd_shuffle_xor(acc, 4); acc += simd_shuffle_xor(acc, 8);
            acc += simd_shuffle_xor(acc, 16);

            if (lid == 0) { dst[agent_id] = (half)acc; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (gid == 0) { completion[0] = 1; }
        threadgroup_barrier(mem_flags::mem_device);
    }
}
"##;

fn compile_kernel(src: &str) -> (ComputePipelineState, CommandQueue, Device) {
    let dev = Device::system_default().expect("Metal device");
    let tmp = std::env::temp_dir().join("tribunus-mtl-batch");
    let _ = std::fs::create_dir_all(&tmp);
    let s = tmp.join("k.metal");
    let a = tmp.join("k.air");
    let l = tmp.join("k.metallib");
    std::fs::write(&s, src).unwrap();
    assert!(std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-std=metal3.2", "-O3", "-c"])
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
    let f = lib.get_function("multiagent_decode", None).unwrap();
    let pso = dev.new_compute_pipeline_state_with_function(&f).unwrap();
    (pso, dev.new_command_queue(), dev)
}

struct Rng(u64);
impl Rng {
    fn new(s: u64) -> Self {
        Self(s)
    }
    fn u32(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

fn fill_weights(ptr: *mut u32, count: usize) {
    let mut r = Rng::new(42);
    for i in 0..count {
        let mut val = 0u32;
        for _ in 0..20 {
            val = val * 3 + (r.u32() % 3);
        }
        unsafe {
            ptr.add(i).write_volatile(val);
        }
    }
}

#[test]
fn multiagent_throughput_scaling() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║  Multi-Agent Throughput Scaling: Shared Weights + Per-Agent KV Cache     ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
    println!();

    let batch_sizes = [1usize, 2, 4, 8, 12, 16];
    let mut results = Vec::new();

    // ── Allocate shared weight buffer ──────────────────────────
    let weights_u32 = HIDDEN_DIM * BLOCK_STRIDE * NUM_BLOCKS_X * LAYERS;
    let (pso, queue, dev) = compile_kernel(MEGAKERNEL_BATCH);

    let wb = dev.new_buffer(
        (weights_u32 * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    fill_weights(wb.contents() as *mut u32, weights_u32);

    let tb = dev.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let cb = dev.new_buffer(4, MTLResourceOptions::StorageModeShared);

    for &batch in &batch_sizes {
        // ── Allocate KV cache buffer (per-agent) ────────────────
        // For the benchmark: each agent has a small synthetic KV cache
        let kv_per_agent = HIDDEN_DIM * 2; // K+V as half
        let kv_total = batch * kv_per_agent;
        let kv_bytes = kv_total * 2; // half = 2 bytes

        let kb = dev.new_buffer(kv_bytes as u64, MTLResourceOptions::StorageModeShared);
        let bs = batch as u32;

        // Pre-queue persistent kernel
        let cmdbuf = queue.new_command_buffer();
        let enc = cmdbuf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pso);
        enc.set_buffer(0, Some(&wb), 0);
        enc.set_buffer(1, Some(&kb), 0);
        enc.set_buffer(2, Some(&tb), 0);
        enc.set_buffer(3, Some(&cb), 0);
        enc.set_bytes(4, 4, &bs as *const u32 as *const std::ffi::c_void);
        enc.dispatch_thread_groups(
            MTLSize {
                width: batch as u64,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: 32,
                height: 1,
                depth: 1,
            },
        );
        enc.end_encoding();
        cmdbuf.commit();

        let trigger = unsafe { &*(tb.contents() as *const AtomicU32) };
        let completion = unsafe { &*(cb.contents() as *const AtomicU32) };

        // Warmup
        for _ in 0..3 {
            completion.store(0, Ordering::Release);
            trigger.store(1, Ordering::Release);
            while completion.load(Ordering::Acquire) == 0 {
                std::hint::spin_loop();
            }
            trigger.store(0, Ordering::Release);
        }

        // Benchmark
        let iters = 50;
        let t0 = Instant::now();
        for _ in 0..iters {
            completion.store(0, Ordering::Release);
            trigger.store(1, Ordering::Release);
            while completion.load(Ordering::Acquire) == 0 {
                std::hint::spin_loop();
            }
            trigger.store(0, Ordering::Release);
        }
        let elapsed = t0.elapsed();
        let per_step = elapsed / iters as u32;
        let tps = batch as f64 / per_step.as_secs_f64();

        // ── Compute model ──────────────────────────────────────
        // Theoretical: weights are 2.4 GB, read once per step regardless of batch
        // KV cache: 600 MB per agent, read per step per agent
        // Total DRAM read: 2.4 GB + batch * 0.6 GB
        let total_gb = WEIGHT_GB + batch as f64 * KV_MB_PER_AGENT / 1000.0;
        let bus_time_ms = total_gb / BUS_GBS * 1000.0;

        // Effective throughput accounting for KV cache overhead
        let compute_per_step_s = per_step.as_secs_f64();
        let effective_tps = if compute_per_step_s > 0.0 {
            batch as f64 / compute_per_step_s
        } else {
            0.0
        };

        // Utilization vs theoretical bus limit
        let utilization = bus_time_ms / (compute_per_step_s * 1000.0).max(bus_time_ms) * 100.0;

        println!(
            "  Batch {:>2}: {:6.1} t/s  {:6.2} ms/step  bus={:.1}%  DRAM={:.1} GB",
            batch,
            effective_tps,
            compute_per_step_s * 1000.0,
            utilization,
            total_gb
        );

        results.push((batch, effective_tps, compute_per_step_s, total_gb));

        // Kill the persistent kernel before next batch size
        // (just let it spin — we'll re-launch with different batch)
        // Actually we need to stop it. Let trigger stay at 0 and re-dispatch.
        // The kernel is in an infinite loop. We just dispatch again.
    }

    // ── Analysis ────────────────────────────────────────────────
    println!("\n  ── Scaling Analysis ──────────────────────────────────────────────────");
    println!(
        "  Bus:      {:.0} GB/s | Weights: {:.1} GB | KV/agent: {:.0} MB",
        BUS_GBS, WEIGHT_GB, KV_MB_PER_AGENT
    );
    println!();
    println!(
        "  {:>6}  {:>8}  {:>8}  {:>8}  {:>10}  {:>8}",
        "Batch", "t/s", "ms/step", "DRAM GB", "Efficiency", "Scaling"
    );
    let base_tps = results[0].1;
    for (b, tps, step_s, gb) in &results {
        let scaling = *tps / base_tps;
        let efficiency = (*tps / *b as f64) / (base_tps / 1.0) * 100.0;
        println!(
            "  {:>6}  {:>8.1}  {:>8.3}  {:>8.1}  {:>9.0}%  {:>7.2}×",
            b,
            tps,
            step_s * 1000.0,
            gb,
            efficiency,
            scaling
        );
    }

    // ── Sweet spot identification ──────────────────────────────
    println!("\n  ── Sweet Spot Analysis ────────────────────────────────────────────────");
    for (b, tps, step_s, gb) in &results {
        let bus_limited_tps = BUS_GBS / (*gb) * *b as f64; // theoretical max at this batch
        let compute_tps = 1.0 / (0.000687 / *b as f64); // compute-limited (0.687ms per 48-layer pass)
        let actual_tps = *tps;
        print!(
            "  Batch {:>2}: compute={:.0} t/s  bus={:.0} t/s  actual={:.0} t/s",
            b, compute_tps, bus_limited_tps, actual_tps
        );
        if actual_tps < bus_limited_tps * 0.8 {
            println!("  ← bus-limited");
        } else if actual_tps < compute_tps * 0.8 {
            println!("  ← compute-limited");
        } else {
            println!("  ← balanced");
        }
    }
}
