//! ANE matmul-only utilization sweep.
//!
//! Tests whether a single matmul-only operation can saturate the ANE compute
//! units by sweeping batch size: x[batch,2048] @ W[2048,4096] -> [batch,4096].
//!
//! For each batch size:
//!   1. Build MIL, compile for ANE (cpuAndNeuralEngine)
//!   2. Load with CpuAndNeuralEngine and CpuOnly
//!   3. Warmup 5, measure 20
//!   4. FLOPS = 2 x batch x 2048 x 4096 / time_s
//!   5. Utilization = FLOPS / 11 TFLOPS (theoretical M1 ANE peak)
//!
//! Run: cargo test --test ane_matmul_utilization --features prism-backend -- --nocapture
//! Requires: macOS 14.0+ on Apple Silicon

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::path::{Path, PathBuf};
use std::time::Instant;

use coreml_proto::proto::mil_spec;
use mlx_rs::Dtype;
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use tribunus_compute_core::coreml_pipeline::compile_mlpackage;
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{write_mlpackage, ModelMeta};

// ── Constants ──────────────────────────────────────────────────────────────

const TEST_DIR: &str = "/tmp/prism_ane_matmul_util";
const H: i64 = 2048;
const FFN: i64 = 4096;
const THEORETICAL_PEAK_GFLOPS: f64 = 11_000.0;
const WARMUP: usize = 5;
const SAMPLES: usize = 20;
const CPU_FALLBACK_RATIO: f64 = 0.8;
const BATCH_SIZES: &[u32] = &[
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
];

// ── Helpers ────────────────────────────────────────────────────────────────

fn model_dir(name: &str) -> PathBuf {
    let p = Path::new(TEST_DIR).join(name);
    let _ = std::fs::create_dir_all(&p);
    p
}

/// Deterministic weight values based on seed.
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

/// Build MIL: x[batch,2048] @ W[2048,4096] -> [batch,4096].
fn build_mil(batch: u32) -> Result<(mil_spec::Program, String), String> {
    let w = seeded_weights(42, H, FFN);
    let b = MilBuilder::new("main")
        .input("x", mil_spec::DataType::Float16, &[batch as i64, H])
        .const_f16("w", &w, &[H, FFN]);
    let wn = b.last_name().ok_or("weight name")?.to_string();
    let mut b = b.matmul("x", &wn);
    let out_name = b.last_name().ok_or("matmul name")?.to_string();
    b = b.output(&out_name);
    let prog = b.build().map_err(|e| format!("MIL build error: {}", e))?;
    Ok((prog, out_name))
}

/// Compile a MIL program into a .modelc directory.
fn compile_model(tag: &str, prog: mil_spec::Program, meta: ModelMeta) -> Result<PathBuf, String> {
    let dir = model_dir(tag);
    let pkg = write_mlpackage(prog, &dir, &meta).map_err(|e| format!("mlpackage: {}", e))?;
    let od = dir.join("compiled");
    let _ = std::fs::create_dir_all(&od);
    let r = compile_mlpackage(&pkg, &od, tag, "cpuAndNeuralEngine", "macOS26")
        .map_err(|e| format!("compile: {}", e))?;
    Ok(PathBuf::from(&r.compiled_modelc_path))
}

/// Fill an arena with deterministic FP16 data for a given batch size.
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

// ═════════════════════════════════════════════════════════════════════════════
// T E S T
// ═════════════════════════════════════════════════════════════════════════════

#[test]
fn ane_matmul_utilization_sweep() {
    println!("\n=== ANE MATMUL UTILIZATION SWEEP ===");
    println!(
        "Model: x[batch,{}] @ W[{},{}] -> [batch,{}]",
        H, H, FFN, FFN
    );
    println!(
        "Theoretical peak: {} GFLOPS (M1 ANE FP16)",
        THEORETICAL_PEAK_GFLOPS as u64
    );
    println!("Batch sizes: {:?}", BATCH_SIZES);
    println!("{}", "=".repeat(95));

    println!(
        "{:>8} {:>12} {:>10} {:>10} {:>10} {:>8} {:>12}",
        "Batch", "FLOPs", "Time(us)", "GFLOPS", "%Peak", "Status", "tok/s"
    );
    println!("{}", "-".repeat(95));

    let mut max_utilization: f64 = 0.0;
    let mut max_util_batch: u32 = 0;
    let mut max_gflops: f64 = 0.0;

    for &batch in BATCH_SIZES {
        let tag = format!("util_batch_{}", batch);

        // ── Build MIL ─────────────────────────────────────────────
        let (prog, out_name) = match build_mil(batch) {
            Ok(v) => v,
            Err(e) => {
                println!(
                    "{:>8} {:>12} {:>10} {:>10} {:>10} {:>8} {:>12}",
                    batch, "N/A", "BUILD_FAIL", "N/A", "N/A", "ERR", "N/A"
                );
                eprintln!("  {} BUILD: {}", tag, e);
                continue;
            }
        };

        let meta = ModelMeta {
            model_name: tag.clone(),
            function_name: "main".into(),
            short_description: format!("ane_matmul_util_{}", batch),
            version: "1.0".into(),
            author: "prism".into(),
            output_name: out_name.clone(),
            inputs: vec![("x".into(), vec![batch as i64, H])],
            outputs: vec![(out_name.clone(), vec![batch as i64, FFN])],
            spec_version: 9,
        };

        // ── Compile ───────────────────────────────────────────────
        let model_path = match compile_model(&tag, prog, meta) {
            Ok(p) => p,
            Err(e) => {
                println!(
                    "{:>8} {:>12} {:>10} {:>10} {:>10} {:>8} {:>12}",
                    batch, "N/A", "COMPILE_FAIL", "N/A", "N/A", "ERR", "N/A"
                );
                eprintln!("  {} COMPILE: {}", tag, e);
                continue;
            }
        };
        let path_str = model_path.to_str().expect("valid path");

        // ── Allocate arenas ───────────────────────────────────────
        let in_arena = match Arena::new(batch, H as u32, Dtype::Float16) {
            Ok(a) => a,
            Err(e) => {
                println!(
                    "{:>8} {:>12} {:>10} {:>10} {:>10} {:>8} {:>12}",
                    batch, "N/A", "ALLOC_FAIL", "N/A", "N/A", "ERR", "N/A"
                );
                eprintln!("  {} arena alloc: {}", tag, e);
                continue;
            }
        };
        let out_arena = match Arena::new(batch, FFN as u32, Dtype::Float16) {
            Ok(a) => a,
            Err(e) => {
                println!(
                    "{:>8} {:>12} {:>10} {:>10} {:>10} {:>8} {:>12}",
                    batch, "N/A", "ALLOC_FAIL", "N/A", "N/A", "ERR", "N/A"
                );
                eprintln!("  {} output arena: {}", tag, e);
                continue;
            }
        };
        if let Err(e) = fill_arena(&in_arena, batch, H as u32) {
            eprintln!("  {} fill: {}", tag, e);
        }

        // ── CPU-only benchmark ────────────────────────────────────
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
                println!(
                    "{:>8} {:>12} {:>10} {:>10} {:>10} {:>8} {:>12}",
                    batch, "N/A", "CPU_FAIL", "N/A", "N/A", "ERR", "N/A"
                );
                eprintln!("  {} CPU: {}", tag, e);
                continue;
            }
        };
        let (_cpu_p50, _cpu_p95, cpu_mean_ns) = cpu_result;

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
                    "{:>8} {:>12} {:>10} {:>10} {:>10} {:>8} {:>12}",
                    batch, "N/A", "ANE_FAIL", "N/A", "N/A", "ERR", "N/A"
                );
                eprintln!("  {} ANE: {}", tag, e);
                continue;
            }
        };
        let (ane_p50_ns, _ane_p95_ns, ane_mean_ns) = ane_result;

        // ── CPU fallback detection ────────────────────────────────
        let ratio = if cpu_mean_ns > 0.0 {
            ane_mean_ns / cpu_mean_ns
        } else {
            0.0
        };
        let status = if ratio > CPU_FALLBACK_RATIO {
            "CPU_FB"
        } else {
            "on-ANE"
        };

        // ── Compute metrics ───────────────────────────────────────
        // FLOPS = 2 x batch x H x FFN
        let total_flops = 2.0 * batch as f64 * H as f64 * FFN as f64;
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

        // tokens/sec (simulated decode, 48 layers, single token per batch)
        let tok_s = if time_us > 0.0 {
            1_000_000.0 / (time_us * 48.0 / batch as f64)
        } else {
            0.0
        };

        println!(
            "{:>8} {:>12.0e} {:>10.1} {:>10.2} {:>9.3}% {:>8} {:>12.1}",
            batch, total_flops, time_us, gflops, pct_peak, status, tok_s
        );

        if gflops > max_gflops {
            max_gflops = gflops;
            max_util_batch = batch;
            max_utilization = pct_peak;
        }

        if status == "CPU_FB" {
            println!(
                "\nCPU fallback at batch={} (ratio={:.2}), continuing to next batch",
                batch, ratio
            );
        }
    }

    // ── Summary ────────────────────────────────────────────────────
    println!("{}", "=".repeat(95));
    println!(
        "Peak utilization: {:.2}% of {} GFLOPS at batch={} ({:.2} GFLOPS)",
        max_utilization, THEORETICAL_PEAK_GFLOPS as u64, max_util_batch, max_gflops
    );
    println!(
        "Conclusion: matmul-only ops achieved {:.1}% of theoretical ANE peak",
        max_utilization
    );
    if max_utilization > 95.0 {
        println!("Result: ANE CAN be saturated by matmul-only ops");
    } else if max_utilization > 80.0 {
        println!("Result: ANE nearly saturated (above 80%) by matmul-only ops");
    } else if max_utilization > 50.0 {
        println!("Result: ANE partially utilized (50-80%) — bottleneck may be memory bandwidth or ANE dispatch overhead");
    } else {
        println!("Result: ANE poorly utilized (<50%) — topology or Core ML runtime overhead likely limiting factor");
    }
}
