//! ANE batch decode throughput sweep.
//!
//! For each batch size: build matmul model x[batch,2048] @ W[2048,4096] -> [batch,4096],
//! compile for ANE (cpuAndNeuralEngine, iOS15), load with CpuAndNeuralEngine and CpuOnly,
//! measure latency with 5 warmup + 20 measured predicts.
//!
//! Two measurement modes:
//!   - rapid: back-to-back predicts (no gap), measures raw ANE throughput
//!   - keepalive: 50ms gap between predicts, simulates decode-time inter-token latency
//!
//! Tokens/sec = batch / (time_seconds) for each mode.
//!
//! Run: cargo test --test ane_batch_decode --features prism-backend -- --nocapture
//! Requires: macOS 14.0+ on Apple Silicon

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

use coreml_proto::proto::mil_spec;
use mlx_rs::Dtype;
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::arena::DataType;
use tribunus_compute_core::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use tribunus_compute_core::coreml_pipeline::compile_mlpackage;
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{write_mlpackage, ModelMeta};

// ── Constants ──────────────────────────────────────────────────────────────

const TEST_DIR: &str = "/tmp/prism_ane_batch_decode";
const H: i64 = 2048;
const FFN: i64 = 4096;
const BATCH_SIZES: &[u32] = &[4, 8, 16, 32, 64, 128, 512, 1024];
const WARMUP: usize = 5;
const SAMPLES: usize = 20;
const KEEPALIVE_GAP_MS: u64 = 50;
const CPU_FALLBACK_RATIO: f64 = 0.8;

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

/// Run one predict and return latency in microseconds.
/// Benchmark a loaded model: warmup + measured predicts, optionally with a
/// gap between each predict. Returns median latency in microseconds.
fn bench_model(
    model: &CoreMlModel,
    in_name: &str,
    in_arena: &Arena,
    out_name: &str,
    out_arena: &Arena,
    gap_ms: u64,
) -> Result<f64, String> {
    let gap = Duration::from_millis(gap_ms);

    // Warmup
    for _ in 0..WARMUP {
        model
            .predict(in_name, &in_arena.info, out_name, &out_arena.info)
            .map_err(|e| format!("warmup: {}", e))?;
        if gap_ms > 0 {
            thread::sleep(gap);
        }
    }

    // Measured
    let mut samples: Vec<f64> = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let t0 = Instant::now();
        model
            .predict(in_name, &in_arena.info, out_name, &out_arena.info)
            .map_err(|e| format!("run: {}", e))?;
        samples.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        if gap_ms > 0 {
            thread::sleep(gap);
        }
    }

    samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let median = samples[samples.len() / 2];
    Ok(median)
}

// ═══════════════════════════════════════════════════════════════════════════
// T E S T
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn ane_batch_decode_sweep() {
    println!("\n=== ANE BATCH DECODE THROUGHPUT SWEEP ===");
    println!(
        "Model: x[batch,{}] @ W[{},{}] -> [batch,{}]",
        H, H, FFN, FFN
    );
    println!("Batch sizes: {:?}", BATCH_SIZES);
    println!(
        "Measurements: {} warmup + {} measured per mode",
        WARMUP, SAMPLES
    );
    println!("{}", "=".repeat(100));

    // Table header
    println!(
        "{:>6} {:>8} {:>10} {:>10} {:>10} {:>10} {:>12} {:>12} {:>10}",
        "Batch",
        "Tokens",
        "Time(us)",
        "tok/s",
        "Time+50ms",
        "tok/s+50ms",
        "CPUtok/s",
        "Ratio",
        "Status"
    );
    println!("{}", "-".repeat(100));

    for &batch in BATCH_SIZES {
        let tag = format!("batch_{}", batch);

        // ── Build MIL ─────────────────────────────────────────────
        let (prog, out_name) = match build_mil(batch) {
            Ok(v) => v,
            Err(_e) => {
                println!(
                    "{:>6} {:>8} {:>10} {:>10} {:>10} {:>12} {:>12} {:>10} {:>10}",
                    batch, batch, "BUILD_FAIL", "N/A", "N/A", "N/A", "N/A", "N/A", "BUILD_FAIL"
                );
                continue;
            }
        };

        let meta = ModelMeta {
            model_name: tag.clone(),
            function_name: "main".into(),
            short_description: format!("ane_batch_decode_{}", batch),
            version: "1.0".into(),
            author: "prism".into(),
            output_name: out_name.clone(),
            inputs: vec![("x".into(), vec![batch as i64, H])],
            outputs: vec![(out_name.clone(), vec![batch as i64, FFN])],

        };

        // ── Compile ───────────────────────────────────────────────
        let model_path = match compile_model(&tag, prog, meta) {
            Ok(p) => p,
            Err(e) => {
                println!(
                    "{:>6} {:>8} {:>10} {:>10} {:>10} {:>12} {:>12} {:>10} {:>10}",
                    batch, batch, "COMPILE_FAIL", "N/A", "N/A", "N/A", "N/A", "N/A", "COMPILE_FAIL"
                );
                eprintln!("  {} compile error: {}", tag, e);
                continue;
            }
        };
        let path_str = model_path.to_str().expect("valid path");

        // ── Allocate arenas (batch, dim) ──────────────────────────
        let in_arena = match Arena::new(batch, H as u32, DataType::Float16) {
            Ok(a) => a,
            Err(e) => {
                println!(
                    "{:>6} {:>8} {:>10} {:>10} {:>10} {:>12} {:>12} {:>10} {:>10}",
                    batch, batch, "ALLOC_FAIL", "N/A", "N/A", "N/A", "N/A", "N/A", "ALLOC_FAIL"
                );
                eprintln!("  {} arena alloc error: {}", tag, e);
                continue;
            }
        };
        let out_arena = match Arena::new(batch, FFN as u32, DataType::Float16) {
            Ok(a) => a,
            Err(e) => {
                println!(
                    "{:>6} {:>8} {:>10} {:>10} {:>10} {:>12} {:>12} {:>10} {:>10}",
                    batch, batch, "ALLOC_FAIL", "N/A", "N/A", "N/A", "N/A", "N/A", "ALLOC_FAIL"
                );
                eprintln!("  {} output arena alloc error: {}", tag, e);
                continue;
            }
        };

        // Fill input with deterministic data
        if let Err(e) = fill_arena(&in_arena, batch, H as u32) {
            eprintln!("  {} fill error: {}", tag, e);
        }

        // ── Load with CpuAndNeuralEngine ──────────────────────────
        let ane_model = match CoreMlModel::load_with_compute_units(
            path_str,
            CoreMlComputeUnits::CpuAndNeuralEngine,
        ) {
            Ok(m) => m,
            Err(e) => {
                println!(
                    "{:>6} {:>8} {:>10} {:>10} {:>10} {:>12} {:>12} {:>10} {:>10}",
                    batch, batch, "LOAD_FAIL", "N/A", "N/A", "N/A", "N/A", "N/A", "LOAD_FAIL"
                );
                eprintln!("  {} load(ANE) error: {}", tag, e);
                continue;
            }
        };

        // ── Load with CpuOnly (for fallback detection) ────────────
        let cpu_model =
            match CoreMlModel::load_with_compute_units(path_str, CoreMlComputeUnits::CpuOnly) {
                Ok(m) => Some(m),
                Err(e) => {
                    eprintln!(
                        "  {} load(CPU) error: {} (continuing with ANE only)",
                        tag, e
                    );
                    // We don't need CPU fallback detection if CPU model won't load,
                    // but we can still run the ANE model.
                    None
                }
            };

        let in_name = "x";

        // ── Rapid bench (no delay) ────────────────────────────────
        let ane_median_us =
            match bench_model(&ane_model, in_name, &in_arena, &out_name, &out_arena, 0) {
                Ok(us) => us,
                Err(e) => {
                    println!(
                        "{:>6} {:>8} {:>10} {:>10} {:>10} {:>12} {:>12} {:>10} {:>10}",
                        batch,
                        batch,
                        "PREDICT_FAIL",
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        "N/A",
                        "PREDICT_FAIL"
                    );
                    eprintln!("  {} ANE rapid bench error: {}", tag, e);
                    continue;
                }
            };

        // ── CPU benchmark (for fallback detection) ────────────────
        let (cpu_median_us, cpu_tok_s) = match &cpu_model {
            Some(cpu) => match bench_model(cpu, in_name, &in_arena, &out_name, &out_arena, 0) {
                Ok(cpu_us) if cpu_us > 0.0 => (cpu_us, batch as f64 / (cpu_us / 1_000_000.0)),
                _ => (0.0, 0.0),
            },
            None => (0.0, 0.0),
        };

        // ── Keepalive bench (50ms gap) ────────────────────────────
        let ane_keepalive_us = match bench_model(
            &ane_model,
            in_name,
            &in_arena,
            &out_name,
            &out_arena,
            KEEPALIVE_GAP_MS,
        ) {
            Ok(us) => us,
            Err(e) => {
                eprintln!("  {} ANE keepalive bench error: {}", tag, e);
                0.0
            }
        };

        // ── Compute metrics ───────────────────────────────────────
        let tok_s = batch as f64 / (ane_median_us / 1_000_000.0);
        let tok_s_keepalive = if ane_keepalive_us > 0.0 {
            batch as f64 / (ane_keepalive_us / 1_000_000.0)
        } else {
            0.0
        };

        // ── Fallback detection ────────────────────────────────────
        let ratio = if cpu_median_us > 0.0 {
            ane_median_us / cpu_median_us
        } else {
            0.0
        };

        let (status_line, should_stop) = if cpu_model.is_some() && ratio > CPU_FALLBACK_RATIO {
            ("CPU_FALLBACK".to_string(), false)
        } else {
            ("on-ANE".to_string(), false)
        };

        println!(
            "{:>6} {:>8} {:>10.1} {:>10.1} {:>10.1} {:>12.1} {:>12.1} {:>10.2} {:>10}",
            batch,
            batch,
            ane_median_us,
            tok_s,
            ane_keepalive_us,
            tok_s_keepalive,
            cpu_tok_s,
            ratio,
            status_line,
        );

        if should_stop {
            println!(
                "\nCPU fallback detected at batch={} (ratio={:.2}), stopping sweep",
                batch, ratio
            );
            break;
        }
    }

    println!("{}", "-".repeat(100));
    println!("=== SWEEP COMPLETE ===");
}
