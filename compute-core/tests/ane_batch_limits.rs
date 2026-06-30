//! ANE batch limit stress test — sweep to 1M tokens to find absolute maximum.
//!
//! For each batch size: build matmul model x[batch,2048] @ W[2048,4096] -> [batch,4096],
//! compile for ANE (cpuAndNeuralEngine, iOS15), then load with CpuAndNeuralEngine,
//! measure latency with 3 warmup + 10 measured predicts.
//!
//! Stops at first failure (compile, alloc, load, or predict).
//! Reports throughput in tok/s and ANE utilization (GFLOPS, % of 11 TFLOPS peak).
//!
//! Run: cargo test --test ane_batch_limits --features prism-backend -- --nocapture
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

// ── Constants ──────────────────────────────────────────────────────────────

const TEST_DIR: &str = "/tmp/prism_ane_batch_limits";
const H: i64 = 2048;
const FFN: i64 = 4096;
/// Sweep: powers of two from 2K to 1M.
const BATCH_SIZES: &[u32] = &[
    2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576,
];
const WARMUP: usize = 3;
const SAMPLES: usize = 10;
const THEORETICAL_PEAK_GFLOPS: f64 = 11_000.0; // 11 TFLOPS M1 ANE

// ── Helpers ────────────────────────────────────────────────────────────────

fn model_dir(name: &str) -> PathBuf {
    let p = Path::new(TEST_DIR).join(name);
    let _ = std::fs::create_dir_all(&p);
    p
}

/// Deterministic weight values based on seed.
fn seeded_weights(seed: u64, rows: i64, cols: i64) -> Vec<f32> {
    let mut w = Vec::with_capacity((rows * cols) as usize);
    for i in 0..((rows * cols) as u64) {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (seed + i).hash(&mut h);
        w.push((h.finish() as f32 % 1000.0 - 500.0) / 500.0);
    }
    w
}

/// Build MIL program for batched matmul: x[batch,2048] @ W[2048,4096] -> [batch,4096].
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

/// Benchmark a model: warmup + measured predicts. Returns median latency in seconds.
fn bench_model(
    model: &CoreMlModel,
    in_name: &str,
    in_arena: &Arena,
    out_name: &str,
    out_arena: &Arena,
) -> Result<f64, String> {
    // Warmup
    for _ in 0..WARMUP {
        model
            .predict(in_name, &in_arena.info, out_name, &out_arena.info)
            .map_err(|e| format!("warmup predict: {}", e))?;
    }

    // Measured
    let mut samples: Vec<f64> = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let t0 = Instant::now();
        model
            .predict(in_name, &in_arena.info, out_name, &out_arena.info)
            .map_err(|e| format!("run predict: {}", e))?;
        samples.push(t0.elapsed().as_secs_f64());
    }

    samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(samples[samples.len() / 2])
}

// ═══════════════════════════════════════════════════════════════════════════
// T E S T
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ane_batch_limit_sweep() {
    println!("\n=== ANE BATCH LIMIT SWEEP ===");
    println!(
        "Model: x[batch,{}] @ W[{},{}] -> [batch,{}]",
        H, H, FFN, FFN
    );
    println!("Batch sizes: {:?}", BATCH_SIZES);
    println!(
        "Measurements: {} warmup + {} measured per batch",
        WARMUP, SAMPLES
    );
    println!(
        "Theoretical ANE peak: {:.0} GFLOPS",
        THEORETICAL_PEAK_GFLOPS
    );
    println!("{}", "=".repeat(120));

    // Table header
    println!(
        "{:>8} {:>12} {:>12} {:>10} {:>10} {:>10}",
        "Batch", "Time(s)", "tok/s", "GFLOPS", "Util%", "Status"
    );
    println!("{}", "-".repeat(80));

    let mut last_working_batch: u32 = 0;
    let mut last_working_tok_s: f64 = 0.0;

    for &batch in BATCH_SIZES {
        let tag = format!("batch_{}", batch);

        // ── Step 1: Build MIL ─────────────────────────────────────
        let (prog, out_name) = match build_mil(batch) {
            Ok(v) => v,
            Err(e) => {
                println!(
                    "{:>8} {:>12} {:>12} {:>10} {:>10} {:>10}",
                    batch, "-", "-", "-", "-", "BUILD_FAIL"
                );
                eprintln!("  {} build error: {}", tag, e);
                break;
            }
        };

        let meta = ModelMeta {
            model_name: tag.clone(),
            function_name: "main".into(),
            short_description: format!("ane_batch_limits_{}", batch),
            version: "1.0".into(),
            author: "prism".into(),
            output_name: out_name.clone(),
            inputs: vec![("x".into(), vec![batch as i64, H])],
            outputs: vec![(out_name.clone(), vec![batch as i64, FFN])],

        };

        // ── Step 2: Compile ───────────────────────────────────────
        let model_path = match compile_model(&tag, prog, meta) {
            Ok(p) => p,
            Err(e) => {
                println!(
                    "{:>8} {:>12} {:>12} {:>10} {:>10} {:>10}",
                    batch, "-", "-", "-", "-", "COMPILE_FAIL"
                );
                eprintln!("  {} compile error: {}", tag, e);
                break;
            }
        };
        let path_str = model_path.to_str().expect("valid path");

        // ── Step 3: Allocate IOSurface arenas ─────────────────────
        let in_arena = match Arena::new(batch, H as u32, DataType::Float16) {
            Ok(a) => a,
            Err(e) => {
                println!(
                    "{:>8} {:>12} {:>12} {:>10} {:>10} {:>10}",
                    batch, "-", "-", "-", "-", "ALLOC_FAIL"
                );
                eprintln!("  {} input arena alloc error: {}", tag, e);
                break;
            }
        };
        let out_arena = match Arena::new(batch, FFN as u32, DataType::Float16) {
            Ok(a) => a,
            Err(e) => {
                println!(
                    "{:>8} {:>12} {:>12} {:>10} {:>10} {:>10}",
                    batch, "-", "-", "-", "-", "ALLOC_FAIL"
                );
                eprintln!("  {} output arena alloc error: {}", tag, e);
                break;
            }
        };

        // Fill input with deterministic data
        if let Err(e) = fill_arena(&in_arena, batch, H as u32) {
            eprintln!("  {} fill error: {}", tag, e);
        }

        // ── Step 4: Load with CpuAndNeuralEngine ──────────────────
        let ane_model = match CoreMlModel::load_with_compute_units(
            path_str,
            CoreMlComputeUnits::CpuAndNeuralEngine,
        ) {
            Ok(m) => m,
            Err(e) => {
                println!(
                    "{:>8} {:>12} {:>12} {:>10} {:>10} {:>10}",
                    batch, "-", "-", "-", "-", "LOAD_FAIL"
                );
                eprintln!("  {} load(ANE) error: {}", tag, e);
                break;
            }
        };

        let in_name = "x";

        // ── Step 5: Benchmark (3 warmup + 10 measured) ────────────
        let median_secs = match bench_model(&ane_model, in_name, &in_arena, &out_name, &out_arena) {
            Ok(s) => s,
            Err(e) => {
                println!(
                    "{:>8} {:>12} {:>12} {:>10} {:>10} {:>10}",
                    batch, "-", "-", "-", "-", "PREDICT_FAIL"
                );
                eprintln!("  {} predict error: {}", tag, e);
                break;
            }
        };

        // ── Compute metrics ───────────────────────────────────────
        let tok_s = batch as f64 / median_secs;
        // GFLOPS = 2 * batch * 2048 * 4096 / time_seconds / 1e9
        let gflops = 2.0 * batch as f64 * H as f64 * FFN as f64 / median_secs / 1_000_000_000.0;
        let utilization = (gflops / THEORETICAL_PEAK_GFLOPS) * 100.0;

        println!(
            "{:>8} {:>12.6} {:>12.1} {:>10.0} {:>9.1}% {:>10}",
            batch, median_secs, tok_s, gflops, utilization, "on-ANE",
        );

        last_working_batch = batch;
        last_working_tok_s = tok_s;
    }

    println!("{}", "-".repeat(80));
    println!("=== SWEEP COMPLETE ===");
    if last_working_batch > 0 {
        println!(
            "Last working batch: {} ({:.1} tok/s)",
            last_working_batch, last_working_tok_s
        );
    }
}
