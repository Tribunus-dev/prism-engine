//! ANE matmul stream parallelism sweep (fused matmul approach).
//!
//! Tests whether running N independent parallel matmuls on the same input
//! improves ANE utilization. Uses the fused matmul + add-tree topology
//! (same as ane_max_throughput) but at larger batch sizes.
//!
//! For each stream count N in [1, 2, 4, 8, 16]:
//!   1. N weight matrices W_i[2048, 256] each (total 4096 output dim)
//!   2. N parallel matmuls: x[batch, 2048] @ W_i -> [batch, 256]
//!   3. Add tree: sum all N results -> [batch, 256]
//!   4. Total FLOPs = N x 2 x batch x 2048 x 256
//!   5. If utilization increases with N, ANE benefits from more parallel streams
//!
//! Note: Unlike concat, sum changes the math (it reduces N channels to 1).
//! The key question is utilization scaling, not arithmetic equivalence.
//!
//! Run: cargo test --test ane_fused_streams --features prism-backend -- --nocapture
//! Requires: macOS 14.0+ on Apple Silicon

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::path::{Path, PathBuf};
use std::time::Instant;

use coreml_proto::proto::mil_spec;
use mlx_rs::Dtype;
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::arena::DataType;
use tribunus_compute_core::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use tribunus_compute_core::coreml_pipeline::compile_mlpackage;
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{write_mlpackage, ModelMeta};

// ── Constants ──────────────────────────────────────────────────────────────

const TEST_DIR: &str = "/tmp/prism_ane_fused_streams";
const H: i64 = 2048;
/// Per-stream output dimension: 256 gives 16 streams for full 4096 FFN.
const K_PER_STREAM: i64 = 256;
const THEORETICAL_PEAK_GFLOPS: f64 = 11_000.0;
const WARMUP: usize = 5;
const SAMPLES: usize = 20;
const BATCH: u32 = 512;
/// Stream counts: each stream is an independent matmul x @ W_i.
/// Total FLOPs scales linearly with stream count.
const STREAMS: &[usize] = &[1, 2, 4, 8, 16];

// ── Helpers ────────────────────────────────────────────────────────────────

fn model_dir(name: &str) -> PathBuf {
    let p = Path::new(TEST_DIR).join(name);
    let _ = std::fs::create_dir_all(&p);
    p
}

fn seeded_weights(seed: u64, rows: i64, cols: i64) -> Vec<f32> {
    use std::hash::{Hash, Hasher};
    let mut w = Vec::with_capacity((rows * cols) as usize);
    for i in 0..((rows * cols) as u64) {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (seed + i).hash(&mut h);
        w.push((h.finish() as f32 % 1000.0 - 500.0) / 500.0);
    }
    w
}

/// Build fused MIL: N parallel matmuls x @ W_i -> add tree -> output.
///
///   x[batch, H] -+-- matmul(x, W_0[H, K]) -> [batch, K] --+
///                 |-- matmul(x, W_1[H, K]) -> [batch, K] --|--- add tree -> [batch, K]
///                 |-- ...                                  |
///                 +-- matmul(x, W_{N-1}[H, K]) -> [batch,K]-+
///
fn build_fused_mil(batch: u32, num_streams: usize) -> Result<(mil_spec::Program, String), String> {
    let k = K_PER_STREAM;
    let mut b = MilBuilder::new("main").input("x", mil_spec::DataType::Float16, &[batch as i64, H]);

    // ── N weight constants ────────────────────────────────────────
    let mut matmul_names: Vec<String> = Vec::with_capacity(num_streams);
    for i in 0..num_streams {
        let w = seeded_weights(i as u64, H, k);
        b = b.const_f16(&format!("w_{}", i), &w, &[H, k]);
        let wn = b
            .last_name()
            .ok_or_else(|| format!("weight_{}", i))?
            .to_string();
        b = b.matmul("x", &wn);
        let mn = b
            .last_name()
            .ok_or_else(|| format!("matmul_{}", i))?
            .to_string();
        matmul_names.push(mn);
    }

    // ── Add tree: ((m0 + m1) + m2) + ... ──────────────────────────
    let mut sum = matmul_names[0].clone();
    for mm in &matmul_names[1..] {
        b = b.add(&sum, mm);
        sum = b.last_name().ok_or("add")?.to_string();
    }

    let out_name = sum;
    let b = b.output(&out_name);
    let prog = b.build().map_err(|e| format!("MIL build error: {}", e))?;
    Ok((prog, out_name))
}

fn compile(tag: &str, prog: mil_spec::Program, meta: ModelMeta) -> Result<PathBuf, String> {
    let dir = model_dir(tag);
    let pkg = write_mlpackage(prog, &dir, &meta).map_err(|e| format!("mlpackage: {}", e))?;
    let od = dir.join("compiled");
    let _ = std::fs::create_dir_all(&od);
    let r = compile_mlpackage(&pkg, &od, tag, "cpuAndNeuralEngine", "macOS26")
        .map_err(|e| format!("compile: {}", e))?;
    Ok(PathBuf::from(&r.compiled_modelc_path))
}

fn fill_arena(arena: &Arena, batch: u32, cols: u32) -> Result<(), String> {
    arena.lock().map_err(|e| format!("arena lock: {}", e))?;
    unsafe {
        let ptr = arena.base_ptr() as *mut u16;
        let count = (batch as usize) * (cols as usize);
        for i in 0..count {
            let val = ((i as u16).wrapping_mul(265).wrapping_add(1234)) & 0x7FFF;
            *ptr.add(i) = val;
        }
    }
    arena.unlock().map_err(|e| format!("arena unlock: {}", e))?;
    Ok(())
}

fn bench_one(
    path: &str,
    cu: CoreMlComputeUnits,
    in_name: &str,
    in_arena: &Arena,
    out_name: &str,
    out_arena: &Arena,
) -> Result<(f64, f64, f64), String> {
    let m = CoreMlModel::load_with_compute_units(path, cu)
        .map_err(|e| format!("load({:?}): {}", cu, e))?;

    for _ in 0..WARMUP {
        m.predict(in_name, &in_arena.info, out_name, &out_arena.info)
            .map_err(|e| format!("warmup: {}", e))?;
    }

    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let t0 = Instant::now();
        m.predict(in_name, &in_arena.info, out_name, &out_arena.info)
            .map_err(|e| format!("run: {}", e))?;
        samples.push(t0.elapsed().as_nanos() as f64);
    }
    samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = samples[samples.len() / 2];
    let p95 = samples[(samples.len() as f64 * 0.95) as usize];
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    Ok((p50, p95, mean))
}

// ═════════════════════════════════════════════════════════════════════════════
// T E S T
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn ane_fused_stream_sweep() {
    println!("\n=== ANE FUSED STREAM PARALLELISM SWEEP ===");
    println!(
        "Model: N parallel matmuls x[{},{}] @ W_i[{},{}], add tree -> [{},{}]",
        BATCH, H, H, K_PER_STREAM, BATCH, K_PER_STREAM
    );
    println!("Total FLOPs scales linearly with N (more work = more ANE exposure)");
    println!("Key metric: GFLOPS utilization at N=16 vs N=1");
    println!(
        "Theoretical peak: {} GFLOPS (M1 ANE FP16)",
        THEORETICAL_PEAK_GFLOPS as u64
    );
    println!("Streams: {:?}", STREAMS);
    println!("batch={}, warmup={}, samples={}", BATCH, WARMUP, SAMPLES);
    println!("{}", "=".repeat(130));

    println!(
        "{:>7} {:>12} {:>12} {:>10} {:>10} {:>8} {:>12} {:>10}",
        "Streams", "FLOPs", "Time(us)", "GFLOPS", "%Peak", "Status", "tok/s", "Compile(ms)"
    );
    println!("{}", "-".repeat(130));

    for &num_streams in STREAMS {
        let tag = format!("fused_{}", num_streams);

        // ── Build MIL ─────────────────────────────────────────────
        let (prog, out_name) = match build_fused_mil(BATCH, num_streams) {
            Ok(v) => v,
            Err(e) => {
                println!(
                    "{:>7} {:>12} {:>12} {:>10} {:>10} {:>8} {:>12} {:>10}",
                    num_streams, "N/A", "BUILD_FAIL", "N/A", "N/A", "ERR", "N/A", "N/A"
                );
                eprintln!("  {} BUILD: {}", tag, e);
                continue;
            }
        };

        let meta = ModelMeta {
            model_name: tag.clone(),
            function_name: "main".into(),
            short_description: format!("ane_fused_stream_{}", num_streams),
            version: "1.0".into(),
            author: "prism".into(),
            output_name: out_name.clone(),
            inputs: vec![("x".into(), vec![BATCH as i64, H])],
            outputs: vec![(out_name.clone(), vec![BATCH as i64, K_PER_STREAM])],

        };

        // ── Compile ───────────────────────────────────────────────
        let compile_start = Instant::now();
        let model_path = match compile(&tag, prog, meta) {
            Ok(p) => p,
            Err(e) => {
                println!(
                    "{:>7} {:>12} {:>12} {:>10} {:>10} {:>8} {:>12} {:>10}",
                    num_streams, "N/A", "COMPILE_FAIL", "N/A", "N/A", "ERR", "N/A", "N/A"
                );
                eprintln!("  {} COMPILE: {}", tag, e);
                continue;
            }
        };
        let compile_ms = compile_start.elapsed().as_millis();
        let path_str = model_path.to_str().expect("valid path");

        // ── Allocate arenas ───────────────────────────────────────
        let in_arena = match Arena::new(BATCH, H as u32, DataType::Float16) {
            Ok(a) => a,
            Err(e) => {
                println!(
                    "{:>7} {:>12} {:>12} {:>10} {:>10} {:>8} {:>12} {:>10}",
                    num_streams, "N/A", "ALLOC_FAIL", "N/A", "N/A", "ERR", "N/A", "N/A"
                );
                eprintln!("  {} arena: {}", tag, e);
                continue;
            }
        };
        let out_arena = match Arena::new(BATCH, K_PER_STREAM as u32, DataType::Float16) {
            Ok(a) => a,
            Err(e) => {
                println!(
                    "{:>7} {:>12} {:>12} {:>10} {:>10} {:>8} {:>12} {:>10}",
                    num_streams, "N/A", "ALLOC_FAIL", "N/A", "N/A", "ERR", "N/A", "N/A"
                );
                eprintln!("  {} output: {}", tag, e);
                continue;
            }
        };
        if let Err(e) = fill_arena(&in_arena, BATCH, H as u32) {
            eprintln!("  {} fill: {}", tag, e);
        }

        // ── ANE benchmark ─────────────────────────────────────────
        let ane_result = match bench_one(
            path_str,
            CoreMlComputeUnits::CpuAndNeuralEngine,
            "x",
            &in_arena,
            &out_name,
            &out_arena,
        ) {
            Ok(r) => r,
            Err(e) => {
                println!(
                    "{:>7} {:>12} {:>12} {:>10} {:>10} {:>8} {:>12} {:>10}",
                    num_streams, "N/A", "ANE_FAIL", "N/A", "N/A", "ERR", "N/A", "N/A"
                );
                eprintln!("  {} ANE: {}", tag, e);
                continue;
            }
        };
        let (ane_p50_ns, _ane_p95_ns, _ane_mean_ns) = ane_result;

        // ── CPU benchmark (fallback detection) ────────────────────
        let cpu_result = match bench_one(
            path_str,
            CoreMlComputeUnits::CpuOnly,
            "x",
            &in_arena,
            &out_name,
            &out_arena,
        ) {
            Ok(r) => r,
            Err(_) => (0.0, 0.0, 0.0),
        };
        let (_cpu_p50, _cpu_p95, cpu_mean_ns) = cpu_result;

        // ── Compute metrics ───────────────────────────────────────
        // FLOPs per matmul = 2 x batch x H x K_PER_STREAM
        // Total = N x 2 x batch x H x K_PER_STREAM
        let flops_per_matmul = 2.0 * BATCH as f64 * H as f64 * K_PER_STREAM as f64;
        let total_flops = num_streams as f64 * flops_per_matmul;
        let time_us = ane_p50_ns / 1000.0;
        let time_s = ane_p50_ns / 1_000_000_000.0;
        let gflops = if time_s > 0.0 {
            total_flops / time_s / 1_000_000_000.0
        } else {
            0.0
        };
        let pct_peak = if THEORETICAL_PEAK_GFLOPS > 0.0 {
            gflops / THEORETICAL_PEAK_GFLOPS * 100.0
        } else {
            0.0
        };

        let ratio = if cpu_mean_ns > 0.0 {
            ane_p50_ns / cpu_mean_ns
        } else {
            0.0
        };
        let status = if ratio > 0.8 { "CPU_FB" } else { "on-ANE" };

        let tok_s = if time_us > 0.0 {
            1_000_000.0 / (time_us * 48.0 / BATCH as f64)
        } else {
            0.0
        };

        println!(
            "{:>7} {:>12.0e} {:>12.1} {:>10.2} {:>10.3}% {:>8} {:>12.1} {:>10}",
            num_streams, total_flops, time_us, gflops, pct_peak, status, tok_s, compile_ms
        );
    }

    println!("{}", "=".repeat(130));
    println!("If utilization increases with N, ANE benefits from parallel streams");
    println!("Flat utilization across N: single matmul already saturates ANE memory bandwidth");
}
