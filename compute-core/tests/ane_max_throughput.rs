//! ANE max throughput — sweep fused model complexity to find peak ANE GFLOPS.
//!
//! For each fusion depth N (number of parallel matmuls):
//!   1. Build MIL: x[1,2048] → N× (x @ W_i) → sum → y[1,4096]
//!   2. Compile for ANE
//!   3. Load with ANE (cpuAndNeuralEngine) and CPU-only
//!   4. Check CPU fallback: if ratio > 0.8, stop and report last working depth
//!   5. Warmup 5, measure 20
//!   6. FLOPS = N × 2048 × 4096 × 2 / time_seconds
//!
//! Run: cargo test --test ane_max_throughput --features prism-backend -- --nocapture
//! Requires: macOS 14.0+ on Apple Silicon

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::Instant;

use coreml_proto::proto::mil_spec;
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::arena::DataType;
use tribunus_compute_core::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use tribunus_compute_core::coreml_pipeline::compile_mlpackage;
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{write_mlpackage, ModelMeta};

const TEST_DIR: &str = "/tmp/prism_ane_max_throughput";
const WARMUP: usize = 5;
const SAMPLES: usize = 20;
const THEORETICAL_PEAK_GFLOPS: f64 = 11_000.0; // 11 TFLOPS theoretical M1 ANE peak
const H: i64 = 2048;
const FFN: i64 = 4096;
const DEPTHS: &[usize] = &[1, 2, 3, 4, 6, 8];

fn model_dir(name: &str) -> PathBuf {
    let p = Path::new(TEST_DIR).join(name);
    let _ = std::fs::create_dir_all(&p);
    p
}

/// Deterministic weight values based on seed for reproducible builds.
fn seeded_weights(seed: u64, rows: i64, cols: i64) -> Vec<f32> {
    let mut w = Vec::with_capacity((rows * cols) as usize);
    for i in 0..((rows * cols) as u64) {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (seed + i).hash(&mut h);
        w.push((h.finish() as f32 % 1000.0 - 500.0) / 500.0);
    }
    w
}

/// Build MIL program for a fused model with N parallel matmuls + sum.
///
/// Graph:
///   x[1, 2048] --+-- matmul(x, W_0) --+
///                |-- matmul(x, W_1) --|--- add --- output y[1, 4096]
///                |-- ...              |
///                +-- matmul(x, W_N-1) -+
///
/// Returns (program, output_name).
fn build_fused_mil(depth: usize) -> Result<(mil_spec::Program, String), String> {
    let mut b = MilBuilder::new("main").input("x", mil_spec::DataType::Float16, &[1, H]);

    // ── Create N weight constants ──────────────────────────────────
    // Each weight is [2048, 4096], stored as const_f16
    let mut weight_names: Vec<String> = Vec::with_capacity(depth);
    for i in 0..depth {
        let w = seeded_weights(i as u64, H, FFN);
        b = b.const_f16("w", &w, &[H, FFN]);
        let wn = b
            .last_name()
            .ok_or_else(|| format!("weight {}", i))?
            .to_string();
        weight_names.push(wn);
    }

    // ── N parallel matmuls: x @ W_i ────────────────────────────────
    // Input is referenced by its original name "x"
    let mut matmul_names: Vec<String> = Vec::with_capacity(depth);
    for wn in &weight_names {
        b = b.matmul("x", wn);
        let mn = b
            .last_name()
            .ok_or_else(|| format!("matmul {}", wn))?
            .to_string();
        matmul_names.push(mn);
    }

    // ── Serial add tree: ((m0 + m1) + m2) + ... ────────────────────
    let mut sum = matmul_names[0].clone();
    for mm in &matmul_names[1..] {
        b = b.add(&sum, mm);
        sum = b.last_name().ok_or("add")?.to_string();
    }

    let out_name = sum;
    b = b.output(&out_name);
    let prog = b.build().map_err(|e| format!("MIL build error: {}", e))?;
    Ok((prog, out_name))
}

/// Compile a MIL program into a .modelc directory.
fn compile_fused(tag: &str, prog: mil_spec::Program, meta: ModelMeta) -> Result<PathBuf, String> {
    let dir = model_dir(tag);
    let pkg = write_mlpackage(prog, &dir, &meta).map_err(|e| format!("mlpackage: {}", e))?;
    let od = dir.join("compiled");
    let _ = std::fs::create_dir_all(&od);
    let r = compile_mlpackage(&pkg, &od, tag, "cpuAndNeuralEngine", "macOS26")
        .map_err(|e| format!("compile: {}", e))?;
    Ok(PathBuf::from(&r.compiled_modelc_path))
}

/// Benchmark one model on one compute policy.
/// Returns (p50_ns, p95_ns, mean_ns).
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

fn make_arena(d0: u32, d1: u32) -> Arena {
    Arena::new(d0, d1, DataType::Float16).expect("arena")
}

// ═════════════════════════════════════════════════════════════════════════════
// T E S T
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn ane_throughput_sweep() {
    println!("\n=== ANE MAX THROUGHPUT (FUSION DEPTH SWEEP) ===");
    println!(
        "H={}, FFN={}, batch=1, theoretical peak={} GFLOPS",
        H, FFN, THEORETICAL_PEAK_GFLOPS as u64
    );
    println!("{}", "-".repeat(80));
    println!(
        "{:>6} {:>10} {:>8} {:>7} {:>8} {:>12}",
        "Depth", "Time(µs)", "GFLOPS", "%Peak", "Status", "tok/s"
    );
    println!("{}", "-".repeat(80));

    let mut last_working_depth: Option<usize> = None;
    let mut max_gflops: f64 = 0.0;
    let mut max_gflops_depth: usize = 0;

    for &depth in DEPTHS {
        let tag = format!("fused_{}", depth);

        // ── Build MIL ──────────────────────────────────────────────
        let (prog, out_name) = match build_fused_mil(depth) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("  {}: BUILD FAIL {}", tag, e);
                continue;
            }
        };

        let meta = ModelMeta {
            model_name: tag.clone(),
            function_name: "main".into(),
            short_description: "ane_fused".into(),
            version: "1.0".into(),
            author: "prism".into(),
            output_name: out_name.clone(),
            inputs: vec![("x".into(), vec![1, H])],
            outputs: vec![(out_name.clone(), vec![1, FFN])],
        };

        // ── Compile ────────────────────────────────────────────────
        let model_path = match compile_fused(&tag, prog, meta) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("  {}: COMPILE FAIL {}", tag, e);
                continue;
            }
        };
        let path_str = model_path.to_str().ok_or("path").unwrap();

        let in_arena = make_arena(1, H as u32);
        let out_arena = make_arena(1, FFN as u32);

        // ── CPU-only benchmark (baseline) ──────────────────────────
        let cpu_result = match bench_one(
            path_str,
            CoreMlComputeUnits::CpuOnly,
            "x",
            &in_arena,
            &out_name,
            &out_arena,
        ) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("  {}: CPU BENCH FAIL {}", tag, e);
                continue;
            }
        };
        let (_cpu_p50, _cpu_p95, cpu_mean_ns) = cpu_result;

        // ── ANE benchmark ──────────────────────────────────────────
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
                eprintln!("  {}: ANE BENCH FAIL {}", tag, e);
                continue;
            }
        };
        let (ane_p50_ns, _ane_p95_ns, ane_mean_ns) = ane_result;

        // ── CPU fallback detection ─────────────────────────────────
        let ratio = if cpu_mean_ns > 0.0 {
            ane_mean_ns / cpu_mean_ns
        } else {
            0.0
        };
        let status = if ratio > 0.8 {
            "CPU_FALLBACK"
        } else {
            "on-ANE"
        };

        // ── Compute metrics ────────────────────────────────────────
        // FLOPS = N × H × FFN × 2 / time_seconds
        let total_flops = depth as f64 * H as f64 * FFN as f64 * 2.0;
        let time_us = ane_p50_ns / 1000.0;
        let time_s = ane_p50_ns / 1_000_000_000.0;
        let gflops = total_flops / time_s / 1_000_000_000.0;
        let pct_peak = gflops / THEORETICAL_PEAK_GFLOPS * 100.0;

        // tokens/sec: 1_000_000 / (time_us × 48_layers)
        let tokens_per_sec = if time_us > 0.0 {
            1_000_000.0 / (time_us * 48.0)
        } else {
            0.0
        };

        println!(
            "{:>6} {:>10.1} {:>8.1} {:>6.2}% {:>8} {:>12.1}",
            depth, time_us, gflops, pct_peak, status, tokens_per_sec
        );

        if gflops > max_gflops {
            max_gflops = gflops;
            max_gflops_depth = depth;
        }

        if status == "on-ANE" {
            last_working_depth = Some(depth);
        } else {
            println!(
                "\nCPU fallback detected at depth={} (ane_mean={:.0}ns, cpu_mean={:.0}ns, ratio={:.2})",
                depth, ane_mean_ns, cpu_mean_ns, ratio
            );
            break;
        }
    }

    // ── Summary ────────────────────────────────────────────────────
    println!("{}", "-".repeat(80));
    println!(
        "Peak throughput: {:.1} GFLOPS at depth={} ({:.2}% of 11 TFLOPS theoretical)",
        max_gflops,
        max_gflops_depth,
        max_gflops / THEORETICAL_PEAK_GFLOPS * 100.0
    );
    match last_working_depth {
        Some(d) => println!("Last working depth (no CPU fallback): {}", d),
        None => println!("All depths fell back to CPU"),
    }
}
