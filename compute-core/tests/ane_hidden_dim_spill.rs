//! ANE-HIDDEN-DIM-SPILL-0001: Find the exact hidden dimension where the M1
//! ANE spills matmul inference to the CPU.
//!
//! Phase 1 — Coarse sweep: hidden dims [1024, 1536, 2048, 2560, 3072, 3584, 4096]
//!   For each: build matmul model [1, H] @ [H, 2*H] → [1, 2*H].
//!   Load with CpuAndNeuralEngine. Load same model with CpuOnly.
//!   5 warmup + 15 measured predictions.
//!   If ane_us / cpu_us > 0.8 → CPU_FALLBACK → record boundary.
//!
//! Phase 2 — Binary search: narrow the spill boundary to within 128 hidden dims.
//!
//! Output format:
//! ```
//! H       ANE(us)  CPU(us)  Ratio  Status
//! 1024     XXX      XXX      X.XX   on-ANE
//! ...
//! 3584     XXX      XXX      X.XX   CPU_FALLBACK ← boundary
//! 4096     FAIL     FAIL     N/A    (skip)
//! ```
//!
//! Result: "Max ANE hidden dim on M1 = N"
//!
//! Run: cargo test --test ane_hidden_dim_spill --features prism-backend,ane -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend", feature = "ane"))]

use coreml_proto::proto::mil_spec;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use tribunus_compute_core::coreml_pipeline::{compile_mlpackage, CoreMlIslandReceipt};
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{self, ModelMeta};

// ── Constants ───────────────────────────────────────────────────────────────

/// Root directory for compiled model artifacts.
const MODEL_DIR: &str = "/tmp/ane_spill_models";

/// Phase 1 coarse sweep: hidden dimensions to test.
const COARSE_HIDDENS: &[u32] = &[1024, 1536, 2048, 2560, 3072, 3584, 4096];

/// Warmup predictions before measurement.
const WARMUP: u32 = 5;

/// Measured predictions for latency averaging.
const MEASURED: u32 = 15;

/// ANE/CPU latency ratio threshold for CPU fallback detection.
/// If ANE latency exceeds 95% of CPU latency, we consider it a CPU fallback.
/// This catches the sudden jump when the model spills out of ANE SRAM and
/// execution silently moves to CPU (ANE_latency ≈ CPU_latency).
///
/// At small hidden dims (e.g. H=1024), ANE launch overhead makes ANE look
/// slower than CPU — that's NOT a spill, just small-model overhead. Real
/// spilling shows as a sharp knee where ANE latency becomes ~CPU latency.
const SPILL_THRESHOLD: f64 = 0.95;
/// Ratio threshold for "ANE is usefully faster": if ANE_latency / CPU_latency
/// is below this, the ANE is providing genuine acceleration.
const ANE_FASTER_THRESHOLD: f64 = 0.95;

/// Binary search continues until the boundary is narrowed to within this many
/// hidden dims.
const BOUNDARY_WIDTH: u32 = 128;

// ── Model building & compilation ────────────────────────────────────────────

/// Return the expected `.modelc` directory for a given hidden dimension.
fn expected_modelc_dir(h: u32) -> PathBuf {
    Path::new(MODEL_DIR).join(format!("spill_H{}.modelc", h))
}

/// Build, compile, and cache a matmul model for the given hidden dimension.
///
/// The model computes: input[1, H] × weight[H, 2*H] → output[1, 2*H].
/// The weight has randomly-initialized FP16 values.
fn build_model(h: u32) -> Result<PathBuf, String> {
    let model_dir = Path::new(MODEL_DIR);
    let _ = std::fs::create_dir_all(model_dir);

    let modelc_outer = expected_modelc_dir(h);
    if modelc_outer.exists() {
        // Find the inner directory containing metadata.json.
        if let Some(inner) = find_modelc_inner(&modelc_outer) {
            return Ok(inner);
        }
    }

    // ── Build weight data ─────────────────────────────────────────────────
    // Weight shape: [H, 2*H], FP16. Fill with deterministic pseudo-random
    // values derived from the hidden dimension (avoiding zero weights that
    // could trigger constant-folding or degenerate matmul paths).
    let n_weight = (h as usize) * (2 * h as usize);
    let mut weight: Vec<f32> = Vec::with_capacity(n_weight);
    for i in 0..n_weight {
        // LCG with seed derived from H: keeps output deterministic per H.
        let x = (i as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(h as u64);
        // Map to uniform [-1.0, 1.0].
        let val = ((x >> 33) as f32) / (1u64 << 31) as f32;
        weight.push(val);
    }

    // ── Build MIL program ─────────────────────────────────────────────────
    let prog = MilBuilder::new("main")
        .input("input", mil_spec::DataType::Float16, &[1, h as i64])
        .const_f16("weight", &weight, &[h as i64, (2 * h) as i64])
        .matmul("input", "weight_0")
        .output("matmul_1")
        .build()
        .map_err(|e| format!("MIL build failed for H={}: {:?}", h, e))?;

    let meta = ModelMeta {
        model_name: format!("spill_H{}", h),
        function_name: "main".into(),
        short_description: format!("ANE spill test: H={} matmul", h),
        version: "1.0.0".into(),
        author: "Tribunus Compute".into(),
        output_name: "matmul_1".into(),
        inputs: vec![("input".into(), vec![1, h as i64])],
        outputs: vec![("matmul_1".into(), vec![1, (2 * h) as i64])],
        spec_version: 9,
    };

    // ── Write .mlpackage ──────────────────────────────────────────────────
    let tmp = tempfile::tempdir().map_err(|e| format!("tempdir: {}", e))?;
    let pkg_path = mlpackage::write_mlpackage(prog, tmp.path(), &meta)
        .map_err(|e| format!("mlpackage write failed for H={}: {}", h, e))?;

    // ── Compile via xcrun coremlcompiler ──────────────────────────────────
    let island_id = format!("spill_H{}", h);
    let receipt: CoreMlIslandReceipt = compile_mlpackage(
        &pkg_path,
        model_dir,
        &island_id,
        "cpuAndNeuralEngine",
        "CoreML9",
    )
    .map_err(|e| format!("compile failed for H={}: {}", h, e))?;

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

/// Measure mean prediction latency for a model at a given hidden dimension.
///
/// Returns `(mean_us, all_us)` where `mean_us` is the mean of measured
/// predictions in microseconds, and `all_us` is every measured latency.
fn measure_latency(
    model: &CoreMlModel,
    h: u32,
    compute_desc: &str,
) -> Result<(f64, Vec<u64>), String> {
    // Allocate input [1, H] and output [1, 2*H] arenas (FP16).
    let input_arena = Arena::new(1, h, mlx_rs::Dtype::Float16)
        .map_err(|e| format!("input arena alloc failed ({} H={}): {}", compute_desc, h, e))?;
    let output_arena = Arena::new(1, 2 * h, mlx_rs::Dtype::Float16).map_err(|e| {
        format!(
            "output arena alloc failed ({} H={}): {}",
            compute_desc, h, e
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
            .map_err(|e| format!("warmup predict failed ({}, H={}): {}", compute_desc, h, e))?;
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
            .map_err(|e| format!("measured predict failed ({}, H={}): {}", compute_desc, h, e))?;
        let elapsed_us = start.elapsed().as_nanos() as u64 / 1000;
        latencies.push(elapsed_us);
    }

    let mean_us = latencies.iter().sum::<u64>() as f64 / latencies.len() as f64;
    Ok((mean_us, latencies))
}

// ── Single-dimension test runner ───────────────────────────────────────────

#[derive(Debug, Clone)]
struct HResult {
    h: u32,
    ane_us: Option<f64>,
    cpu_us: Option<f64>,
    ratio: Option<f64>,
    status: &'static str,
    ane_samples: Vec<u64>,
    cpu_samples: Vec<u64>,
}

/// Test a single hidden dimension: build (or load cached) model, measure
/// ANE and CPU latencies, and classify the result.
fn test_hidden_dim(h: u32) -> HResult {
    let modelc_path = match build_model(h) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  [WARN] build_model(H={}): {}", h, e);
            return HResult {
                h,
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
            eprintln!("  [WARN] load ANE(H={}): {}", h, e);
            return HResult {
                h,
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
            eprintln!("  [WARN] load CPU(H={}): {}", h, e);
            return HResult {
                h,
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
    let (ane_mean, ane_samples) = match measure_latency(&ane_model, h, "ANE") {
        Ok((mean, samples)) => (Some(mean), samples),
        Err(e) => {
            eprintln!("  [WARN] measure ANE(H={}): {}", h, e);
            (None, vec![])
        }
    };

    // Measure CPU latency.
    let (cpu_mean, cpu_samples) = match measure_latency(&cpu_model, h, "CPU") {
        Ok((mean, samples)) => (Some(mean), samples),
        Err(e) => {
            eprintln!("  [WARN] measure CPU(H={}): {}", h, e);
            (None, vec![])
        }
    };

    let (ratio, status) = match (ane_mean, cpu_mean) {
        (Some(ane), Some(cpu)) if cpu > 0.0 => {
            let r = ane / cpu;
            if r < ANE_FASTER_THRESHOLD {
                (Some(r), "on-ANE")
            } else {
                (Some(r), "CPU_FALLBACK")
            }
        }
        _ => (None, "FAIL"),
    };

    HResult {
        h,
        ane_us: ane_mean,
        cpu_us: cpu_mean,
        ratio,
        status,
        ane_samples,
        cpu_samples,
    }
}

// ── Binary search ──────────────────────────────────────────────────────────

/// Binary search between `lo_h` (no spill) and `hi_h` (spill) to narrow
/// the boundary to within `BOUNDARY_WIDTH` hidden dims.
fn binary_search_spill(lo_h: u32, hi_h: u32) -> Vec<HResult> {
    let mut results: Vec<HResult> = Vec::new();
    let mut lo = lo_h;
    let mut hi = hi_h;

    while hi - lo > BOUNDARY_WIDTH {
        let mid = lo + (hi - lo) / 2;
        // Round to nearest multiple of 64 — ANE dimension alignment matters.
        let mid = ((mid + 31) / 64) * 64;
        // Clamp: don't go above hi or below lo.
        let mid = mid.max(lo + 1).min(hi - 1);

        let r = test_hidden_dim(mid);
        results.push(r.clone());

        if r.status == "CPU_FALLBACK" {
            hi = mid;
        } else if r.status == "on-ANE" {
            lo = mid;
        } else {
            // FAIL — can't measure, skip this midpoint and shrink toward hi.
            lo = mid;
        }
    }

    results
}

// ── Table printing ─────────────────────────────────────────────────────────

fn print_table(results: &[HResult], title: &str) {
    println!();
    println!("--- {} ---", title);
    println!(
        "{:<8} {:>10} {:>10} {:>7} {:>20}",
        "H", "ANE(us)", "CPU(us)", "Ratio", "Status"
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
            r.h, ane_str, cpu_str, ratio_str, r.status
        );
    }
}

// ── Test ────────────────────────────────────────────────────────────────────

#[test]
fn find_ane_hidden_dim_spill() {
    // ── Phase 1: Coarse sweep ───────────────────────────────────────────────
    println!("=== Phase 1: Coarse sweep ===");
    println!("Testing hidden dims: {:?}", COARSE_HIDDENS);
    println!(
        "Using {} warmup + {} measured predictions, spill threshold = {}",
        WARMUP, MEASURED, SPILL_THRESHOLD
    );

    let mut coarse_results: Vec<HResult> = Vec::new();

    for &h in COARSE_HIDDENS {
        let r = test_hidden_dim(h);
        coarse_results.push(r);
    }

    print_table(&coarse_results, "Phase 1: Coarse sweep");

    // ── Find last boundary for binary search ────────────────────────────
    // The profile may have multiple on-ANE / CPU_FALLBACK transitions.
    // We find the LAST "on-ANE" point and the following "CPU_FALLBACK"
    // to define the final spill boundary.
    let mut last_ane_h: u32 = 0;
    let mut boundary_lo: Option<u32> = None;
    let mut boundary_hi: Option<u32> = None;
    for i in 0..coarse_results.len() {
        let r = &coarse_results[i];
        if r.status == "on-ANE" {
            last_ane_h = r.h;
            boundary_lo = None;
            boundary_hi = None;
        } else if r.status == "CPU_FALLBACK" && last_ane_h > 0 {
            if boundary_lo.is_none() {
                boundary_lo = Some(last_ane_h);
                boundary_hi = Some(r.h);
            }
        }
    }

    let mut max_ane_h: u32;
    let spill_h: Option<u32>;
    let mut binary_results: Vec<HResult> = Vec::new();

    if let (Some(lo_h), Some(hi_h)) = (boundary_lo, boundary_hi) {
        if hi_h - lo_h > BOUNDARY_WIDTH {
            println!("\n=== Phase 2: Binary search ===");
            println!(
                "Narrowing spill boundary between H={} (on-ANE) and H={} (CPU_FALLBACK)",
                lo_h, hi_h
            );
            binary_results = binary_search_spill(lo_h, hi_h);
            print_table(&binary_results, "Phase 2: Binary search");
        }

        // Update max from binary search results.
        max_ane_h = lo_h;
        for r in &binary_results {
            if r.status == "on-ANE" && r.h > max_ane_h {
                max_ane_h = r.h;
            }
        }
        spill_h = Some(hi_h);
    } else {
        // No clear boundary found — report last on-ANE any good dim.
        max_ane_h = last_ane_h;
        spill_h = None;
    }

    // ── Summary ──────────────────────────────────────────────────────────
    println!();
    println!("{}", "=".repeat(60));
    println!("RESULT: Max ANE hidden dim on M1 = {}", max_ane_h);
    if let Some(sh) = spill_h {
        println!("Spill at H={} (ANE latency ~ CPU latency)", sh);
    } else {
        println!("No spill detected in tested range");
    }

    // Print measurement details for the boundary.
    // Print measurement details for the spill boundary.
    if let Some(hi) = spill_h {
        if let Some(r) = coarse_results.iter().find(|r| r.h == hi) {
            if !r.ane_samples.is_empty() {
                let min_ane = r.ane_samples.iter().min().unwrap();
                let max_ane = r.ane_samples.iter().max().unwrap();
                println!(
                    "  At H={} ANE: mean={:.0}us, min={}us, max={}us, samples={:?}",
                    hi,
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
                    "  At H={} CPU: mean={:.0}us, min={}us, max={}us, samples={:?}",
                    hi,
                    r.cpu_us.unwrap_or(0.0),
                    min_cpu,
                    max_cpu,
                    r.cpu_samples
                );
            }
        }
    }

    println!("{}", "=".repeat(60));

    // Verify we found a spill.
    assert!(
        max_ane_h > 0 || spill_h.is_some(),
        "Test completed with no on-ANE dimensions for the tested matmul pattern. \
         The ANE may not accelerate batch-1 matmuls on this hardware. \
         See table above for latency profile."
    );
}
