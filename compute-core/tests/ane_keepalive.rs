//! ANE keepalive power-state transition detection.
//!
//! Tests whether idle gaps between ANE predictions cause power-state transitions
//! that increase subsequent predict latency. A single matmul model
//! (x[1,2048] @ W[2048,4096] → [1,4096]) is loaded once and reused.
//!
//! Part 1 — Gap sensitivity:
//!   For gaps [0, 1, 5, 10, 50, 100, 500] ms:
//!     10 warmup predicts (0 gap), then 1 timed predict with specified gap,
//!     repeat 20 times, track median/p95.
//!   If any gap > 0 has ratio vs 0ms > 1.20, ANE is entering a power state.
//!
//! Part 2 — Thermal drift:
//!   1000 rapid predicts (0 gap), latency in windows of 100.
//!   If drift from window 1 to 10 exceeds 20%, thermal throttling is active.
//!
//! Run: cargo test --test ane_keepalive --features prism-backend -- --nocapture
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

const TEST_DIR: &str = "/tmp/prism_ane_keepalive";
const H: i64 = 2048;
const FFN: i64 = 4096;
const GAPS_MS: &[u64] = &[0, 1, 5, 10, 50, 100, 500];
const THERMAL_RAPID: usize = 1000;
const THERMAL_WINDOW: usize = 100;

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

/// Build single matmul: x[1,2048] @ W[2048,4096] → output[1,4096].
fn build_mil() -> Result<(mil_spec::Program, String), String> {
    let w = seeded_weights(42, H, FFN);
    let b = MilBuilder::new("main")
        .input("x", mil_spec::DataType::Float16, &[1, H])
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

/// Run one predict and return latency in microseconds.
fn timed_predict(
    model: &CoreMlModel,
    in_name: &str,
    in_arena: &Arena,
    out_name: &str,
    out_arena: &Arena,
) -> f64 {
    let t0 = Instant::now();
    model
        .predict(in_name, &in_arena.info, out_name, &out_arena.info)
        .expect("predict failed");
    t0.elapsed().as_nanos() as f64 / 1000.0
}

/// Compute sorted median and p95 from a slice.
fn median_p95(mut samples: Vec<f64>) -> (f64, f64) {
    samples.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = samples[samples.len() / 2];
    let p95 = samples[(samples.len() as f64 * 0.95) as usize];
    (p50, p95)
}

#[test]
fn ane_keepalive_test() {
    println!("\n=== ANE KEEPALIVE: POWER-STATE GAP SENSITIVITY ===");
    println!("Model: x[1,{}] @ W[{},{}] → [1,{}]", H, H, FFN, FFN);
    println!("Gaps tested: {:?} ms", GAPS_MS);
    println!("{}", "-".repeat(60));

    // ── Build and compile once ────────────────────────────────────
    let tag = "ane_keepalive";
    let (prog, out_name) = build_mil().expect("build MIL");
    let meta = ModelMeta {
        model_name: tag.into(),
        function_name: "main".into(),
        short_description: "ane_keepalive".into(),
        version: "1.0".into(),
        author: "prism".into(),
        output_name: out_name.clone(),
        inputs: vec![("x".into(), vec![1, H])],
        outputs: vec![(out_name.clone(), vec![1, FFN])],
    };

    let model_path = compile_model(tag, prog, meta).expect("compile model");
    let path_str = model_path.to_str().expect("valid path");

    // ── Load model once ───────────────────────────────────────────
    let model =
        CoreMlModel::load_with_compute_units(path_str, CoreMlComputeUnits::CpuAndNeuralEngine)
            .expect("load model");

    let in_arena = Arena::new(1, H as u32, DataType::Float16).expect("input arena");
    let out_arena = Arena::new(1, FFN as u32, DataType::Float16).expect("output arena");

    let in_name = "x";
    // ── Part 1: Gap sensitivity ───────────────────────────────────
    println!(
        "{:>7}  {:>10}  {:>10}  {:>8}",
        "Gap(ms)", "Median(µs)", "P95(µs)", "vs_0ms"
    );
    println!("{}", "-".repeat(45));

    let mut reference_median: Option<f64> = None;
    let mut any_power_transition = false;

    for &gap_ms in GAPS_MS {
        let gap = Duration::from_millis(gap_ms);
        let mut timed: Vec<f64> = Vec::with_capacity(20);

        for _ in 0..20 {
            // 10 warmup predicts with 0 gap
            for _ in 0..10 {
                model
                    .predict(in_name, &in_arena.info, &out_name, &out_arena.info)
                    .expect("warmup predict");
            }
            // Wait the gap BEFORE the timed predict
            if gap_ms > 0 {
                thread::sleep(gap);
            }
            timed.push(timed_predict(
                &model, in_name, &in_arena, &out_name, &out_arena,
            ));
        }

        let (median_us, p95_us) = median_p95(timed);

        let ratio = match reference_median {
            Some(ref_m) if ref_m > 0.0 => median_us / ref_m,
            _ => 1.0,
        };

        if reference_median.is_none() {
            reference_median = Some(median_us);
        }

        println!(
            "{:>7}  {:>10.1}  {:>10.1}  {:>8.2}",
            gap_ms, median_us, p95_us, ratio
        );

        if gap_ms > 0 && ratio > 1.20 {
            any_power_transition = true;
        }
    }

    println!(
        "\nPower-state transition detected: {}",
        any_power_transition
    );
    if any_power_transition {
        println!("→ ANE likely enters a lower-power idle state between decodes.");
        println!("→ Consider keepalive pings (dummy predicts at <50ms intervals) to maintain active state.");
    } else {
        println!("→ No significant gap-induced latency increase — ANE stays active between decode tokens.");
    }

    // ── Part 2: Thermal drift ─────────────────────────────────────
    println!("\n{}", "=".repeat(60));
    println!("=== ANE THERMAL DRIFT (1000 RAPID PREDICTS) ===");
    println!("{}", "-".repeat(60));

    let mut thermal_latencies: Vec<f64> = Vec::with_capacity(THERMAL_RAPID);
    for _ in 0..THERMAL_RAPID {
        thermal_latencies.push(timed_predict(
            &model, in_name, &in_arena, &out_name, &out_arena,
        ));
    }

    let num_windows = THERMAL_RAPID / THERMAL_WINDOW;
    let mut window_medians: Vec<f64> = Vec::with_capacity(num_windows);

    println!(
        "{:>12}  {:>10}  {:>8}",
        "Window(100)", "Median(µs)", "Drift"
    );
    println!("{}", "-".repeat(35));

    for w in 0..num_windows {
        let start = w * THERMAL_WINDOW;
        let end = start + THERMAL_WINDOW;
        let (wm, _) = median_p95(thermal_latencies[start..end].to_vec());
        window_medians.push(wm);
    }

    let first_median = window_medians[0];
    for (w, &wm) in window_medians.iter().enumerate() {
        let drift = if first_median > 0.0 {
            wm / first_median
        } else {
            1.0
        };
        let label = format!(
            "{:>4}-{:<4}",
            w * THERMAL_WINDOW + 1,
            (w + 1) * THERMAL_WINDOW
        );
        println!("{:>12}  {:>10.1}  {:>8.2}", label, wm, drift);
    }

    let last_median = window_medians[num_windows - 1];
    let overall_drift = if first_median > 0.0 {
        last_median / first_median
    } else {
        1.0
    };

    println!(
        "\nThermal drift (window 1 → window {}): {:.2}x",
        num_windows, overall_drift
    );
    if overall_drift > 1.20 {
        println!("→ Thermal throttling detected — latency increased >20% over 1000 predicts.");
    } else {
        println!("→ No significant thermal throttling — ANE latency stable under sustained load.");
    }

    // ── Combined assessment ───────────────────────────────────────
    println!("\n{}", "=".repeat(60));
    println!("RECOMMENDATION:");
    if any_power_transition {
        println!(
            "  - Add keepalive pings (dummy predicts at ~50ms intervals between decode tokens)"
        );
    } else {
        println!("  - No keepalive pings needed for ANE power-state management");
    }
    if overall_drift > 1.20 {
        println!(
            "  - Monitor ANE thermal behavior; consider cooling or throttling-aware scheduling"
        );
    } else {
        println!("  - ANE thermally stable under rapid decode");
    }
}
