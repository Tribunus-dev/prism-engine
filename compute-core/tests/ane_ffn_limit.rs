//! ANE-FFN-LIMIT-0001: Find the maximum FFN (output) width the M1 ANE
//! can execute per model without spilling to CPU.
//!
//! Sweep: FFN = [2048, 3072, 4096, 5120, 6144] at fixed hidden=2048.
//! For each FFN:
//!   1. Build matmul: x[1, 2048] @ W[2048, FFN] → y[1, FFN]
//!   2. Load with CpuAndNeuralEngine and CpuOnly
//!   3. 5 warmup + 15 measured predicts each
//!   4. If ane_us / cpu_us > 0.8 → CPU_FALLBACK
//!   5. Find largest FFN where ANE is genuinely running the model
//!
//! Output format:
//! ```
//! FFN    ANE(us)  CPU(us)  Ratio  Status
//! 2048    XXX      XXX      X.XX   on-ANE
//! ...
//! 6144    XXX      XXX      X.XX   CPU_FALLBACK
//! Max ANE FFN on M1 = N
//! ```
//!
//! Run: cargo test --test ane_ffn_limit --features prism-backend,ane -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend", feature = "ane"))]

use coreml_proto::proto::mil_spec;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::arena::DataType;
use tribunus_compute_core::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use tribunus_compute_core::coreml_pipeline::{compile_mlpackage, CoreMlIslandReceipt};
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{self, ModelMeta};

// ── Constants ───────────────────────────────────────────────────────────────

/// Fixed hidden dimension (input).
const HIDDEN: u32 = 2048;

/// Root directory for compiled model artifacts.
const MODEL_DIR: &str = "/tmp/ane_ffn_limit_models";

/// Sweep: FFN output dimensions to test.
const FFN_SIZES: &[u32] = &[2048, 3072, 4096, 5120, 6144];

/// Warmup predictions before measurement.
const WARMUP: u32 = 5;

/// Measured predictions for latency averaging.
const MEASURED: u32 = 15;

/// ANE/CPU latency ratio threshold for CPU fallback detection.
/// If ANE latency exceeds this fraction of CPU latency, we consider it
/// a CPU fallback. This catches the sudden jump when the model spills
/// out of ANE SRAM and execution silently moves to CPU (ANE ≈ CPU).
///
/// At small FFN dims (e.g. FFN=2048), ANE launch overhead makes ANE look
/// slower than CPU — that's NOT a spill, just small-model overhead. Real
/// spilling shows as a sharp knee where ANE latency becomes ~CPU latency
/// or where ANE unexpectedly degrades relative to a clear on-ANE point.
///
/// Threshold 0.95 matches the established convention in ane_hidden_dim_spill.rs.
const FALLBACK_THRESHOLD: f64 = 0.95;

// ── Model building & compilation ────────────────────────────────────────────

/// Return the expected `.modelc` directory for a given FFN dimension.
fn expected_modelc_dir(ffn: u32) -> PathBuf {
    Path::new(MODEL_DIR).join(format!("ffn_H{}_FFN{}.modelc", HIDDEN, ffn))
}

/// Build, compile, and cache a matmul model for the given FFN dimension.
///
/// The model computes: input[1, HIDDEN] × weight[HIDDEN, FFN] → output[1, FFN].
/// The weight has randomly-initialized FP16 values.
fn build_model(ffn: u32) -> Result<PathBuf, String> {
    let model_dir = Path::new(MODEL_DIR);
    let _ = std::fs::create_dir_all(model_dir);

    let modelc_outer = expected_modelc_dir(ffn);
    if modelc_outer.exists() {
        // Find the inner directory containing metadata.json.
        if let Some(inner) = find_modelc_inner(&modelc_outer) {
            return Ok(inner);
        }
    }

    // ── Build weight data ─────────────────────────────────────────────────
    // Weight shape: [HIDDEN, FFN], FP16. Fill with deterministic pseudo-random
    // values derived from the FFN dimension (avoiding zero weights that
    // could trigger constant-folding or degenerate matmul paths).
    let n_weight = (HIDDEN as usize) * (ffn as usize);
    let mut weight: Vec<f32> = Vec::with_capacity(n_weight);
    for i in 0..n_weight {
        // LCG with seed derived from FFN: keeps output deterministic per FFN.
        let x = (i as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(ffn as u64);
        // Map to uniform [-1.0, 1.0].
        let val = ((x >> 33) as f32) / (1u64 << 31) as f32;
        weight.push(val);
    }

    // ── Build MIL program ─────────────────────────────────────────────────
    let prog = MilBuilder::new("main")
        .input("input", mil_spec::DataType::Float16, &[1, HIDDEN as i64])
        .const_f16("weight", &weight, &[HIDDEN as i64, ffn as i64])
        .matmul("input", "weight_0")
        .output("matmul_1")
        .build()
        .map_err(|e| format!("MIL build failed for FFN={}: {:?}", ffn, e))?;

    let meta = ModelMeta {
        model_name: format!("ffn_H{}_FFN{}", HIDDEN, ffn),
        function_name: "main".into(),
        short_description: format!("ANE FFN limit test: H={} FFN={} matmul", HIDDEN, ffn),
        version: "1.0.0".into(),
        author: "Tribunus Compute".into(),
        output_name: "matmul_1".into(),
        inputs: vec![("input".into(), vec![1, HIDDEN as i64])],
        outputs: vec![("matmul_1".into(), vec![1, ffn as i64])],
    };

    // ── Write .mlpackage ──────────────────────────────────────────────────
    let tmp = tempfile::tempdir().map_err(|e| format!("tempdir: {}", e))?;
    let pkg_path = mlpackage::write_mlpackage(prog, tmp.path(), &meta)
        .map_err(|e| format!("mlpackage write failed for FFN={}: {}", ffn, e))?;

    // ── Compile via xcrun coremlcompiler ──────────────────────────────────
    let island_id = format!("ffn_H{}_FFN{}", HIDDEN, ffn);
    let receipt: CoreMlIslandReceipt = compile_mlpackage(
        &pkg_path,
        model_dir,
        &island_id,
        "cpuAndNeuralEngine",
        "CoreML9",
    )
    .map_err(|e| format!("compile failed for FFN={}: {}", ffn, e))?;

    let modelc_path = PathBuf::from(&receipt.compiled_modelc_path);
    if !modelc_path.exists() {
        return Err(format!("compiled modelc not found at {:?}", modelc_path));
    }

    Ok(modelc_path)
}

/// Walk into a `.modelc` directory to find the inner dir with `metadata.json`.
fn find_modelc_inner(dir: &Path) -> Option<PathBuf> {
    fn walk(d: &Path, depth: u32) -> Option<PathBuf> {
        if depth > 4 {
            return None;
        }
        if d.join("metadata.json").exists() && d.join("model.mil").exists() {
            return Some(d.to_path_buf());
        }
        if let Ok(entries) = std::fs::read_dir(d) {
            for e in entries.filter_map(|e| e.ok()) {
                if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    if let Some(found) = walk(&e.path(), depth + 1) {
                        return Some(found);
                    }
                }
            }
        }
        None
    }
    walk(dir, 0)
}

// ── Latency measurement ────────────────────────────────────────────────────

/// Measure mean prediction latency for a model at a given FFN dimension.
///
/// Returns `(mean_us, all_us)` where `mean_us` is the mean of measured
/// predictions in microseconds, and `all_us` is every measured latency.
fn measure_latency(
    model: &CoreMlModel,
    ffn: u32,
    compute_desc: &str,
) -> Result<(f64, Vec<u64>), String> {
    // Allocate input [1, HIDDEN] and output [1, FFN] arenas (FP16).
    let input_arena = Arena::new(1, HIDDEN, DataType::Float16).map_err(|e| {
        format!(
            "input arena alloc failed ({} FFN={}): {}",
            compute_desc, ffn, e
        )
    })?;
    let output_arena = Arena::new(1, ffn, DataType::Float16).map_err(|e| {
        format!(
            "output arena alloc failed ({} FFN={}): {}",
            compute_desc, ffn, e
        )
    })?;

    let input_name = "input".to_string();
    let output_name = "matmul_1".to_string();

    // Fill input with deterministic FP16 data.
    {
        input_arena
            .lock()
            .map_err(|e| format!("input lock failed: {}", e))?;
        unsafe {
            let ptr = input_arena.base_ptr() as *mut u16;
            let count = input_arena.byte_len() / 2;
            for i in 0..count {
                // Deterministic pattern varying across elements.
                let val = ((i as u16).wrapping_mul(265).wrapping_add(1234)) & 0x7FFF;
                *ptr.add(i) = val;
            }
        }
        input_arena
            .unlock()
            .map_err(|e| format!("input unlock failed: {}", e))?;
    }

    // Warmup predictions.
    for _ in 0..WARMUP {
        model
            .predict(
                &input_name,
                &input_arena.info,
                &output_name,
                &output_arena.info,
            )
            .map_err(|e| {
                format!(
                    "warmup predict failed ({}, FFN={}): {}",
                    compute_desc, ffn, e
                )
            })?;
    }

    // Measured predictions.
    let mut latencies: Vec<u64> = Vec::with_capacity(MEASURED as usize);
    for _ in 0..MEASURED {
        let start = Instant::now();
        model
            .predict(
                &input_name,
                &input_arena.info,
                &output_name,
                &output_arena.info,
            )
            .map_err(|e| {
                format!(
                    "measured predict failed ({}, FFN={}): {}",
                    compute_desc, ffn, e
                )
            })?;
        let elapsed_us = start.elapsed().as_nanos() as u64 / 1000;
        latencies.push(elapsed_us);
    }

    let mean_us = latencies.iter().sum::<u64>() as f64 / latencies.len() as f64;
    Ok((mean_us, latencies))
}

// ── Single-dimension test runner ───────────────────────────────────────────

#[derive(Debug, Clone)]
struct FFNResult {
    ffn: u32,
    ane_us: Option<f64>,
    cpu_us: Option<f64>,
    ratio: Option<f64>,
    status: &'static str,
    ane_samples: Vec<u64>,
    cpu_samples: Vec<u64>,
}

/// Test a single FFN dimension: build (or load cached) model, measure
/// ANE and CPU latencies, and classify the result.
fn test_ffn_dim(ffn: u32) -> FFNResult {
    let modelc_path = match build_model(ffn) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  [WARN] build_model(FFN={}): {}", ffn, e);
            return FFNResult {
                ffn,
                ane_us: None,
                cpu_us: None,
                ratio: None,
                status: "FAIL",
                ane_samples: vec![],
                cpu_samples: vec![],
            };
        }
    };

    // Load with CpuAndNeuralEngine.
    let ane_model = match CoreMlModel::load_with_compute_units(
        &modelc_path.to_string_lossy(),
        CoreMlComputeUnits::CpuAndNeuralEngine,
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("  [WARN] load ANE(FFN={}): {}", ffn, e);
            return FFNResult {
                ffn,
                ane_us: None,
                cpu_us: None,
                ratio: None,
                status: "FAIL",
                ane_samples: vec![],
                cpu_samples: vec![],
            };
        }
    };

    // Load with CpuOnly.
    let cpu_model = match CoreMlModel::load_with_compute_units(
        &modelc_path.to_string_lossy(),
        CoreMlComputeUnits::CpuOnly,
    ) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("  [WARN] load CPU(FFN={}): {}", ffn, e);
            return FFNResult {
                ffn,
                ane_us: None,
                cpu_us: None,
                ratio: None,
                status: "FAIL",
                ane_samples: vec![],
                cpu_samples: vec![],
            };
        }
    };

    // Measure ANE latency.
    let (ane_mean, ane_samples) = match measure_latency(&ane_model, ffn, "ANE") {
        Ok((mean, samples)) => (Some(mean), samples),
        Err(e) => {
            eprintln!("  [WARN] measure ANE(FFN={}): {}", ffn, e);
            (None, vec![])
        }
    };

    // Measure CPU latency.
    let (cpu_mean, cpu_samples) = match measure_latency(&cpu_model, ffn, "CPU") {
        Ok((mean, samples)) => (Some(mean), samples),
        Err(e) => {
            eprintln!("  [WARN] measure CPU(FFN={}): {}", ffn, e);
            (None, vec![])
        }
    };

    let (ratio, status) = match (ane_mean, cpu_mean) {
        (Some(ane), Some(cpu)) if cpu > 0.0 => {
            let r = ane / cpu;
            if r > FALLBACK_THRESHOLD {
                // ANE ≥ 80% of CPU → it's running on CPU (silent spill)
                (Some(r), "CPU_FALLBACK")
            } else {
                // ANE is faster than 80% of CPU → genuine ANE execution
                (Some(r), "on-ANE")
            }
        }
        _ => (None, "FAIL"),
    };

    FFNResult {
        ffn,
        ane_us: ane_mean,
        cpu_us: cpu_mean,
        ratio,
        status,
        ane_samples,
        cpu_samples,
    }
}

// ── Table printing ─────────────────────────────────────────────────────────

fn print_table(results: &[FFNResult], title: &str) {
    println!();
    println!("--- {} ---", title);
    println!(
        "{:<8} {:>10} {:>10} {:>7} {:>20}",
        "FFN", "ANE(us)", "CPU(us)", "Ratio", "Status"
    );
    println!("{}", "─".repeat(59));
    for r in results {
        let ane_str = match r.ane_us {
            Some(us) => format!("{:.0}", us),
            None => "FAIL".to_string(),
        };
        let cpu_str = match r.cpu_us {
            Some(us) => format!("{:.0}", us),
            None => "FAIL".to_string(),
        };
        let ratio_str = match r.ratio {
            Some(r) => format!("{:.2}", r),
            None => "N/A".to_string(),
        };
        println!(
            "{:<8} {:>10} {:>10} {:>7} {:>20}",
            r.ffn, ane_str, cpu_str, ratio_str, r.status
        );
    }
}

// ── Test ────────────────────────────────────────────────────────────────────

#[test]
fn find_ane_ffn_limit() {
    println!("=== ANE FFN Limit Sweep ===");
    println!("Fixed hidden dim: {}", HIDDEN);
    println!("Testing FFN sizes: {:?}", FFN_SIZES);
    println!(
        "Using {} warmup + {} measured predictions, fallback threshold = {}",
        WARMUP, MEASURED, FALLBACK_THRESHOLD
    );

    let mut results: Vec<FFNResult> = Vec::new();

    for &ffn in FFN_SIZES {
        let r = test_ffn_dim(ffn);
        results.push(r);
    }

    print_table(&results, "FFN Limit Sweep");

    // ── Determine max ANE FFN ─────────────────────────────────────────────
    let mut max_ane_ffn: u32 = 0;
    let mut first_fallback: Option<u32> = None;
    for r in &results {
        if r.status == "on-ANE" {
            max_ane_ffn = r.ffn;
        } else if r.status == "CPU_FALLBACK" && first_fallback.is_none() {
            first_fallback = Some(r.ffn);
        }
    }

    // ── Summary ──────────────────────────────────────────────────────────
    println!();
    println!("{}", "=".repeat(60));
    println!("Max ANE FFN on M1 = {}", max_ane_ffn);
    if let Some(fb) = first_fallback {
        println!(
            "First CPU_FALLBACK at FFN={} (ANE latency ~ CPU latency)",
            fb
        );
    } else {
        println!("No CPU_FALLBACK detected in tested range");
    }

    // Print measurement details at the boundary.
    if let Some(fb) = first_fallback {
        if let Some(r) = results.iter().find(|r| r.ffn == fb) {
            if !r.ane_samples.is_empty() {
                let min_ane = r.ane_samples.iter().min().unwrap();
                let max_ane = r.ane_samples.iter().max().unwrap();
                println!(
                    "  At FFN={} ANE: mean={:.0}us, min={}us, max={}us, samples={:?}",
                    fb,
                    r.ane_us.unwrap_or(0.0),
                    min_ane,
                    max_ane,
                    r.ane_samples
                );
            }
            if !r.cpu_samples.is_empty() {
                let min_cpu = r.cpu_samples.iter().min().unwrap();
                let max_cpu = r.cpu_samples.iter().max().unwrap();
                println!(
                    "  At FFN={} CPU: mean={:.0}us, min={}us, max={}us, samples={:?}",
                    fb,
                    r.cpu_us.unwrap_or(0.0),
                    min_cpu,
                    max_cpu,
                    r.cpu_samples
                );
            }
        }
    }

    println!("{}", "=".repeat(60));

    // Assert we found at least one on-ANE run.
    assert!(
        max_ane_ffn > 0,
        "No FFN dimension ran on-ANE — all fell back to CPU?"
    );
}
