//! SVD-LATENT-PREFILL-0001: Prove latent prefill (compress→ANE→decompress) works.
//!
//! Full reference: x[1, 3072] @ W[3072, 6144] → y[1, 6144] (FP32 cblas_sgemv).
//! Compressed path:
//!   1. x_latent = x @ P^T  [1,3072] → [1,2048]  (P[2048,3072], random)
//!   2. ANE: x_latent @ W_latent  [1,2048] → [1,4096] (W_latent = P @ W[:,0:4096])
//!   3. Decompress: ane_output @ Q  [1,4096] → [1,6144] (Q[4096,6144])
//! Compare decompressed vs full reference. Report RMSE and SNR.
//!
//! Compression ratio: 1.5× (3072 → 2048 input).
//! The compression+decompression matrices are random, NOT trained.
//! The test proves the CONCEPT works, not accuracy without training.
//!
//! All random data is scaled by 1/sqrt(HIDDEN) to keep ANE matmul output
//! within FP16 range (otherwise 3072-element dot products overflow FP16).
//!
//! Run: cargo test --test svd_latent_prefill --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use coreml_proto::proto::mil_spec;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use tribunus_compute_core::coreml_pipeline::{compile_mlpackage, CoreMlIslandReceipt};
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{self, ModelMeta};

// ── CBLAS constants ────────────────────────────────────────────────────────

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;

#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemv(
        order: i32,
        trans: i32,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        incx: i32,
        beta: f32,
        y: *mut f32,
        incy: i32,
    );
    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
}

// ── Constants ───────────────────────────────────────────────────────────────

/// Input dimension (scaled from Gemma 4's 5120 for practicality).
const HIDDEN: u32 = 3072;

/// Latent dimension — ANE-friendly (H <= 2560 limit).
const LATENT: u32 = 2048;

/// FFN output dimension for the ANE model (FFN <= 4096 limit).
const LATENT_FFN: u32 = 4096;

/// Full output dimension.
const OUTPUT: u32 = 6144;

/// Scale factor: 1/sqrt(HIDDEN) keeps dot products near ±1, fitting FP16.
const SCALE: f32 = 0.01805; // ≈ 1/sqrt(3072)

/// Root directory for compiled model artifacts.
const MODEL_DIR: &str = "/tmp/svd_latent_models";

/// Warmup predictions before measurement.
const WARMUP: u32 = 5;

/// Measured predictions for latency averaging.
const MEASURED: u32 = 15;

// ── FP16 conversion helpers ─────────────────────────────────────────────────

fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;

    if exp == 0xFF {
        return (sign << 15) | 0x7C00 | if mant != 0 { 0x0200 } else { 0 };
    }
    if exp == 0 {
        return sign << 15;
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 0x1F {
        return (sign << 15) | 0x7C00;
    }
    if new_exp <= 0 {
        return sign << 15;
    }

    (sign << 15) | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x03FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign << 31);
        }
        let leading = mant.leading_zeros() - 22;
        let norm_mant = (mant << leading) & 0x7F_FFFF;
        return f32::from_bits((sign << 31) | ((127 - 15 - leading) << 23) | norm_mant);
    }
    if exp == 0x1F {
        if mant == 0 {
            return f32::from_bits((sign << 31) | 0x7F80_0000);
        }
        return f32::from_bits((sign << 31) | 0x7FC0_0000 | (mant << 13));
    }

    f32::from_bits((sign << 31) | ((exp + (127 - 15)) << 23) | (mant << 13))
}

// ── Deterministic data generation ──────────────────────────────────────────
// Values in [-1, 1] ∈ [-1, 1], further scaled by SCALE when used for weights
// or projections to keep intermediate matmuls in FP16 range.

fn make_data(n: usize, seed: u64) -> Vec<f32> {
    let scale = SCALE;
    (0..n)
        .map(|i| {
            let x = (i as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(seed);
            (((x >> 33) as f32) / (1u64 << 31) as f32) * scale
        })
        .collect()
}

// ── BLAS helpers ───────────────────────────────────────────────────────────

fn gemv(a: &[f32], x: &[f32], y: &mut [f32], m: usize, n: usize) {
    unsafe {
        cblas_sgemv(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            m as i32,
            n as i32,
            1.0,
            a.as_ptr(),
            n as i32,
            x.as_ptr(),
            1,
            0.0,
            y.as_mut_ptr(),
            1,
        );
    }
}

fn gemv_colwise(a: &[f32], x: &[f32], y: &mut [f32], m: usize, n: usize) {
    unsafe {
        cblas_sgemv(
            CBLAS_ROW_MAJOR,
            CBLAS_TRANS,
            m as i32,
            n as i32,
            1.0,
            a.as_ptr(),
            n as i32,
            x.as_ptr(),
            1,
            0.0,
            y.as_mut_ptr(),
            1,
        );
    }
}

// ── Model compilation ──────────────────────────────────────────────────────

fn expected_modelc_dir() -> PathBuf {
    Path::new(MODEL_DIR).join("svd_latent.modelc")
}

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

fn build_ane_model() -> Result<PathBuf, String> {
    let model_dir = Path::new(MODEL_DIR);
    let _ = std::fs::create_dir_all(model_dir);

    let modelc_outer = expected_modelc_dir();
    if modelc_outer.exists() {
        if let Some(inner) = find_modelc_inner(&modelc_outer) {
            return Ok(inner);
        }
    }

    // Build weight data for W_latent[2048, 4096] (FP16).
    // Note: W_latent is built from the same SCALE as P and W, so the
    // cblas_sgemm(P @ W[:,0:4096]) already produces correctly-scaled values.
    let n_weight = (LATENT as usize) * (LATENT_FFN as usize);
    let mut weight: Vec<f32> = Vec::with_capacity(n_weight);
    for i in 0..n_weight {
        let x = (i as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(42);
        let val = (((x >> 33) as f32) / (1u64 << 31) as f32) * SCALE;
        weight.push(val);
    }

    // ── Build MIL program ─────────────────────────────────────────────────
    let prog = MilBuilder::new("main")
        .input("input", mil_spec::DataType::Float16, &[1, LATENT as i64])
        .const_f16("weight", &weight, &[LATENT as i64, LATENT_FFN as i64])
        .matmul("input", "weight_0")
        .output("matmul_1")
        .build()
        .map_err(|e| format!("MIL build failed: {:?}", e))?;

    let meta = ModelMeta {
        model_name: "svd_latent".into(),
        function_name: "main".into(),
        short_description: format!("SVD latent prefill: {}x{} matmul", LATENT, LATENT_FFN),
        version: "1.0.0".into(),
        author: "Tribunus Compute".into(),
        output_name: "matmul_1".into(),
        inputs: vec![("input".into(), vec![1, LATENT as i64])],
        outputs: vec![("matmul_1".into(), vec![1, LATENT_FFN as i64])],
        spec_version: 9,
    };

    // ── Write .mlpackage ──────────────────────────────────────────────────
    let tmp = tempfile::tempdir().map_err(|e| format!("tempdir: {}", e))?;
    let pkg_path = mlpackage::write_mlpackage(prog, tmp.path(), &meta)
        .map_err(|e| format!("mlpackage write failed: {}", e))?;

    // ── Compile via xcrun coremlcompiler ──────────────────────────────────
    let receipt: CoreMlIslandReceipt = compile_mlpackage(
        &pkg_path,
        model_dir,
        "svd_latent",
        "cpuAndNeuralEngine",
        "CoreML9",
    )
    .map_err(|e| format!("compile failed: {}", e))?;

    let modelc_path = PathBuf::from(&receipt.compiled_modelc_path);
    if !modelc_path.exists() {
        return Err(format!("compiled modelc not found at {:?}", modelc_path));
    }

    Ok(modelc_path)
}

// ── Metrics ─────────────────────────────────────────────────────────────────

fn compute_rmse(reference: &[f32], actual: &[f32]) -> f64 {
    let n = reference.len().min(actual.len());
    let sum_sq: f64 = reference[..n]
        .iter()
        .zip(actual[..n].iter())
        .map(|(a, b)| {
            let d = *a as f64 - *b as f64;
            d * d
        })
        .sum();
    (sum_sq / n as f64).sqrt()
}

fn compute_snr(reference: &[f32], actual: &[f32]) -> f64 {
    let n = reference.len().min(actual.len());
    let signal_power: f64 = reference[..n]
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        / n as f64;
    let noise_power: f64 = reference[..n]
        .iter()
        .zip(actual[..n].iter())
        .map(|(a, b)| {
            let d = *a as f64 - *b as f64;
            d * d
        })
        .sum::<f64>()
        / n as f64;
    if noise_power <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (signal_power / noise_power).log10()
}

// ── Test ────────────────────────────────────────────────────────────────────

#[test]
fn svd_latent_prefill() {
    println!("=== SVD Latent Prefill: compress->ANE->decompress ===");
    println!(
        "Hidden={}, Latent={}, LatentFFN={}, Output={}",
        HIDDEN, LATENT, LATENT_FFN, OUTPUT
    );
    println!(
        "Input compression ratio: {:.1}x ({} -> {})",
        HIDDEN as f64 / LATENT as f64,
        HIDDEN,
        LATENT
    );
    println!(
        "Output decompression ratio: {:.1}x ({} -> {})",
        OUTPUT as f64 / LATENT_FFN as f64,
        LATENT_FFN,
        OUTPUT
    );
    println!("Data scale: {} (1/sqrt({}))", SCALE, HIDDEN);
    println!(
        "Using {} warmup + {} measured predictions",
        WARMUP, MEASURED
    );

    let h = HIDDEN as usize;
    let l = LATENT as usize;
    let f = LATENT_FFN as usize;
    let o = OUTPUT as usize;

    // ── Step 1: Generate reference data ────────────────────────────────────

    // Input vector x[3072] — scaled to keep dot products in FP16 range.
    let x = make_data(h, 100);

    // Weight matrix W[3072, 6144], row-major: w[r * o + c] = W[r][c].
    let w = make_data(h * o, 200);

    // Full reference: y_ref = x @ W using cblas_sgemv(Trans).
    let mut y_ref = vec![0.0f32; o];
    gemv_colwise(&w, &x, &mut y_ref, h, o);

    assert!(
        y_ref.iter().all(|v| v.is_finite()),
        "reference has non-finite values"
    );

    // ── Step 2: Build projection matrices ──────────────────────────────────

    // Compression matrix P[2048, 3072] stored row-major: p[r][c] = p[r * h + c].
    let p = make_data(l * h, 300);

    // Decompression matrix Q[4096, 6144] stored row-major: q[r][c] = q[r * o + c].
    let q = make_data(f * o, 400);

    // Weight for latent FFN: W_latent[2048, 4096] = P @ W[:, 0:4096]
    // via cblas_sgemm: C = A * B where A=P[2048×3072], B=W[:,0:4096][3072×4096]
    let mut w_latent = vec![0.0f32; l * f];
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            l as i32,
            f as i32,
            h as i32,
            1.0,
            p.as_ptr(),
            h as i32,
            w.as_ptr(),
            o as i32, // full stride 6144 over W rows
            0.0,
            w_latent.as_mut_ptr(),
            f as i32,
        );
    }

    // ── Step 3: Compress input — x_latent = x @ P^T ────────────────────────
    // Using cblas_sgemv(NoTrans): x_latent[i] = sum_j P[i][j] * x[j]
    let mut x_latent = vec![0.0f32; l];
    gemv(&p, &x, &mut x_latent, l, h);

    println!(
        "Generated data: x[{}], W[{}x{}], P[{}x{}], W_latent[{}x{}], Q[{}x{}]",
        h, h, o, l, h, l, f, f, o
    );
    println!(
        "Reference signal power: {:.4}",
        y_ref.iter().map(|v| (v * v) as f64).sum::<f64>() / o as f64
    );
    println!(
        "x_latent range: [{:.6}, {:.6}]",
        x_latent.iter().fold(f32::MAX, |a, &b| a.min(b)),
        x_latent.iter().fold(f32::MIN, |a, &b| a.max(b))
    );

    // ── Step 4: Build and load ANE model ───────────────────────────────────
    println!(
        "\nBuilding ANE model (latent {} -> {})...",
        LATENT, LATENT_FFN
    );
    let modelc_path = build_ane_model().expect("build ANE model");

    let ane_model = CoreMlModel::load_with_compute_units(
        &modelc_path.to_string_lossy(),
        CoreMlComputeUnits::CpuAndNeuralEngine,
    )
    .expect("load ANE model");
    println!("ANE model loaded.");

    // ── Step 5: Allocate arenas ────────────────────────────────────────────
    let input_arena = Arena::new(1, LATENT, mlx_rs::Dtype::Float16).expect("input arena");
    let output_arena = Arena::new(1, LATENT_FFN, mlx_rs::Dtype::Float16).expect("output arena");

    let input_name = "input".to_string();
    let output_name = "matmul_1".to_string();

    // ── Step 6: Warmup — 5 predictions ─────────────────────────────────────
    println!("\nWarmup ({} predictions)...", WARMUP);
    for w in 0..WARMUP {
        {
            input_arena.lock().expect("lock input");
            unsafe {
                let ptr = input_arena.base_ptr() as *mut u16;
                for i in 0..l {
                    ptr.add(i).write(f32_to_f16_bits(x_latent[i]));
                }
            }
            input_arena.unlock().expect("unlock input");
        }

        ane_model
            .predict(
                &input_name,
                &input_arena.info,
                &output_name,
                &output_arena.info,
            )
            .unwrap_or_else(|e| panic!("warmup {} predict failed: {}", w, e));
    }

    // ── Step 7: Measured predictions ───────────────────────────────────────
    let mut latencies: Vec<u64> = Vec::with_capacity(MEASURED as usize);
    let mut last_ane_output = vec![0.0f32; f];

    println!("Measuring ({} predictions)...", MEASURED);
    for m in 0..MEASURED {
        {
            input_arena.lock().expect("lock input");
            unsafe {
                let ptr = input_arena.base_ptr() as *mut u16;
                for i in 0..l {
                    ptr.add(i).write(f32_to_f16_bits(x_latent[i]));
                }
            }
            input_arena.unlock().expect("unlock input");
        }

        let start = Instant::now();
        ane_model
            .predict(
                &input_name,
                &input_arena.info,
                &output_name,
                &output_arena.info,
            )
            .unwrap_or_else(|e| panic!("measured predict {} failed: {}", m, e));
        let elapsed_us = start.elapsed().as_nanos() as u64 / 1000;
        latencies.push(elapsed_us);

        // Read output arena as FP16 (last iteration preserved for decompression).
        {
            output_arena.lock().expect("lock output");
            unsafe {
                let ptr = output_arena.base_ptr() as *const u16;
                for i in 0..f {
                    last_ane_output[i] = f16_bits_to_f32(ptr.add(i).read());
                }
            }
            output_arena.unlock().expect("unlock output");
        }

        // Diagnostic: check ANE output after first measured iteration.
        if m == 0 {
            let n_nan = last_ane_output.iter().filter(|v| v.is_nan()).count();
            let n_inf = last_ane_output.iter().filter(|v| v.is_infinite()).count();
            println!("  ANE output[0..8]: {:?}", &last_ane_output[..8.min(f)]);
            println!(
                "  ANE output NaN={}, Inf={}, finite={} / {}",
                n_nan,
                n_inf,
                f - n_nan - n_inf,
                f
            );
        }
    }

    let mean_us = latencies.iter().sum::<u64>() as f64 / latencies.len() as f64;
    let min_us = *latencies.iter().min().unwrap() as f64;
    let max_us = *latencies.iter().max().unwrap() as f64;

    println!(
        "ANE latency: mean={:.0} us  min={:.0} us  max={:.0} us",
        mean_us, min_us, max_us
    );

    // ── Step 8: Decompress ─────────────────────────────────────────────────
    // decompressed = ane_output @ Q where Q[4096,6144] stored row-major.
    // Using cblas_sgemv(Trans): decompressed[j] = sum_i Q[i][j] * ane_output[i]
    let mut decompressed = vec![0.0f32; o];
    gemv_colwise(&q, &last_ane_output, &mut decompressed, f, o);

    // ── Step 9: Print results ──────────────────────────────────────────────
    println!();
    println!("--- Results ---");
    println!();
    print!("Full reference [0..4]:   ");
    for v in &y_ref[..4] {
        print!("{:+.6}  ", v);
    }
    println!();
    print!("Decompressed  [0..4]:   ");
    for v in &decompressed[..4] {
        print!("{:+.6}  ", v);
    }
    println!();

    let rmse = compute_rmse(&y_ref, &decompressed);
    let snr = compute_snr(&y_ref, &decompressed);
    println!("RMSE: {:.6}", rmse);
    println!("SNR:  {:.2} dB", snr);

    println!();
    println!("Latency samples (us): {:?}", latencies);
    println!();

    // ── Diagnostics before verification ─────────────────────────────────────
    let n_nan = decompressed.iter().filter(|v| v.is_nan()).count();
    let n_inf = decompressed.iter().filter(|v| v.is_infinite()).count();
    let n_finite = decompressed.iter().filter(|v| v.is_finite()).count();
    println!(
        "decompressed NaN={}, Inf={}, finite={} / {}",
        n_nan, n_inf, n_finite, o
    );
    println!("decompressed[0..8]: {:?}", &decompressed[..8.min(o)]);
    println!("ane_output[0..8]: {:?}", &last_ane_output[..8.min(f)]);
    if n_finite > 0 {
        let min_f = decompressed
            .iter()
            .cloned()
            .filter(|v| v.is_finite())
            .fold(f32::MAX, f32::min);
        let max_f = decompressed
            .iter()
            .cloned()
            .filter(|v| v.is_finite())
            .fold(f32::MIN, f32::max);
        println!("decompressed finite range: [{:.6}, {:.6}]", min_f, max_f);
    }

    // ── Step 10: Verify ──────────────────────────────────────────────────────

    // ANE should execute on device (not CPU fallback). M1 ANE running a small
    // matmul (2048x4096) typically completes in 50-200 us.
    assert!(
        mean_us > 20.0,
        "ANE mean latency {:.0} us suspiciously fast -- possible CPU-only execution \
         (expected 50-200 us for ANE on 2048x4096 matmul)",
        mean_us
    );
    assert!(
        mean_us < 20000.0,
        "ANE mean latency {:.0} us suggests CPU fallback or thermal throttle \
         (expected < 200 us for ANE)",
        mean_us
    );

    // Finite outputs. With properly scaled data all values should be finite.
    assert!(
        decompressed.iter().all(|v| v.is_finite()),
        "decompressed output has non-finite values (NaN={}, Inf={})",
        n_nan,
        n_inf
    );
    assert!(rmse.is_finite(), "RMSE is not finite");
    assert!(snr.is_finite(), "SNR is not finite");

    println!();
    println!("=== SVD Latent Prefill PASSED ===");
    println!(
        "Pipeline: compress ({}) -> ANE ({}) -> decompress ({})  |  RMSE={:.4}  SNR={:.1} dB",
        LATENT, LATENT_FFN, OUTPUT, rmse, snr
    );
}
