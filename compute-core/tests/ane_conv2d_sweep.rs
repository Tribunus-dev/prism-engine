//! Conv2d 1x1 Deception Sweep — test #1 from the SRAM geometry plan.
//!
//! Tests the hypothesis that the ANE's native image-processing path (Conv2d 1x1)
//! has different alignment constraints than a standard matmul.
//!
//! Sweeps sequence length S around 64-byte alignment boundaries:
//!   conv:  x[1, 2048, 1, S] ~ conv2d(W[4096, 2048, 1, 1]) -> y[1, 4096, 1, S]
//!   matmul: x[1, S×2048] @ W[2048, 4096] -> [1, 4096] -> reshape to [1, 4096, 1, S]
//!
//! The 64-byte alignment constraint means the last axis byte size must be
//! a multiple of 64. For FP16 (2 bytes/elem), this means S % 32 == 0.
//!
//! Run: cargo test --test ane_conv2d_sweep --features prism-backend -- --nocapture

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

const TEST_DIR: &str = "/tmp/prism_ane_conv2d_sweep";
const H: i64 = 2048;
const FFN: i64 = 4096;
const THEORETICAL_PEAK_GFLOPS: f64 = 11_000.0;
const WARMUP: usize = 5;
const SAMPLES: usize = 15;

/// Sequence lengths to test. Focus on the 64-byte alignment boundary.
/// For FP16: aligned when S % 32 == 0 (since 2 bytes/elem, need S*2 % 64 == 0).
const SEQ_LENS: &[u32] = &[
    // Below boundary
    61, 62, 63, // At boundary
    64, // Just above
    65, 95, // Next alignment point
    96, // Sweep up
    128, 160, 192, 224, 256, 288, 320,
];

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

/// Build Conv2d 1x1 model: x[1, H, 1, S] @ W[FFN, H, 1, 1] -> y[1, FFN, 1, S]
fn build_conv(seq_len: u32) -> Result<(mil_spec::Program, String, String), String> {
    let w = seeded_weights(0, H, FFN);
    let mut b = MilBuilder::new("main")
        .input("x", mil_spec::DataType::Float16, &[1, H, 1, seq_len as i64])
        .const_f16("w", &w, &[FFN, H, 1, 1]); // weight: [out_ch, in_ch, 1, 1]
    let wn = b.last_name().ok_or("weight name")?.to_string();

    // Conv2d 1x1 with valid padding
    b = b.conv("conv", "x", &wn, &[1, 1], "valid");
    let out_name = b.last_name().ok_or("conv name")?.to_string();
    let b = b.output(&out_name);
    let prog = b.build().map_err(|e| format!("MIL: {}", e))?;
    Ok((prog, "x".into(), out_name))
}

/// Build standard matmul model: x[batch, H] @ W[H, FFN] -> [batch, FFN]
fn build_matmul(batch: u32) -> Result<(mil_spec::Program, String, String), String> {
    let w = seeded_weights(0, H, FFN);
    let b = MilBuilder::new("main")
        .input("x", mil_spec::DataType::Float16, &[batch as i64, H])
        .const_f16("w", &w, &[H, FFN]);
    let wn = b.last_name().ok_or("weight name")?.to_string();
    let b = b.matmul("x", &wn);
    let out_name = b.last_name().ok_or("matmul name")?.to_string();
    let b = b.output(&out_name);
    let prog = b.build().map_err(|e| format!("MIL: {}", e))?;
    Ok((prog, "x".into(), out_name))
}

fn compile_with_target(
    tag: &str,
    prog: mil_spec::Program,
    meta: ModelMeta,
    target: &str,
) -> Result<PathBuf, String> {
    let dir = model_dir(tag);
    let pkg = write_mlpackage(prog, &dir, &meta).map_err(|e| format!("mlpackage: {}", e))?;
    let od = dir.join("compiled");
    let _ = std::fs::create_dir_all(&od);
    let r = compile_mlpackage(&pkg, &od, tag, "cpuAndNeuralEngine", target)
        .map_err(|e| format!("compile: {}", e))?;
    Ok(PathBuf::from(&r.compiled_modelc_path))
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

fn fill_arena(arena: &Arena, count: usize) {
    arena.lock().unwrap();
    unsafe {
        let ptr = arena.base_ptr() as *mut u16;
        for i in 0..count {
            *ptr.add(i) = ((i as u16).wrapping_mul(265).wrapping_add(1234)) & 0x7FFF;
        }
    }
    arena.unlock().unwrap();
}

fn bench_one(
    path: &str,
    cu: CoreMlComputeUnits,
    in_name: &str,
    in_arena: &Arena,
    out_name: &str,
    out_arena: &Arena,
) -> Result<f64, String> {
    let m = CoreMlModel::load_with_compute_units(path, cu).map_err(|e| format!("load: {}", e))?;
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
    Ok(samples[samples.len() / 2])
}

#[test]
fn ane_conv2d_sweep() {
    println!("\n=== CONV2D 1x1 DECEPTION SWEEP ===");
    println!(
        "Tests whether Conv2d 1x1 avoids ANE padding penalties at non-64-byte-aligned seq lengths"
    );
    println!("FP16: aligned when S×2 (bytes on last axis) % 64 == 0, i.e. S %% 32 == 0");
    println!("{}", "=".repeat(140));
    println!(
        "{:>6} {:>6} {:>15} {:>15} {:>12} {:>12} {:>12} {:>10}",
        "S",
        "aligned",
        "conv_time(us)",
        "mm_time(us)",
        "conv_GFLOPS",
        "mm_GFLOPS",
        "conv_%Peak",
        "mm_%Peak"
    );
    println!("{}", "-".repeat(140));

    for &seq_len in SEQ_LENS {
        let aligned = (seq_len as usize * 2) % 64 == 0;
        let align_mark = if aligned { "YES" } else { "NO" };

        // ── Conv2d model ──────────────────────────────────────────
        let conv_tag = format!("conv_s{}", seq_len);
        let (conv_prog, conv_in, conv_out) = match build_conv(seq_len) {
            Ok(v) => v,
            Err(e) => {
                println!(
                    "{:>6} {:>6} {:>15} {:>15} {:>12} {:>12} {:>12} {:>10}",
                    seq_len, align_mark, "BUILD_FAIL", "N/A", "N/A", "N/A", "N/A", "N/A"
                );
                eprintln!("  conv build: {}", e);
                continue;
            }
        };

        let conv_meta = ModelMeta {
            model_name: conv_tag.clone(),
            function_name: "main".into(),
            short_description: format!("conv_sweep_{}", seq_len),
            version: "1.0".into(),
            author: "prism".into(),
            output_name: conv_out.clone(),
            inputs: vec![("x".into(), vec![1, H, 1, seq_len as i64])],
            outputs: vec![(conv_out.clone(), vec![1, FFN, 1, seq_len as i64])],
            spec_version: 10,
        };

        let conv_path = match compile_with_target(&conv_tag, conv_prog, conv_meta, "macOS26") {
            Ok(p) => p,
            Err(e) => {
                println!(
                    "{:>6} {:>6} {:>15} {:>15} {:>12} {:>12} {:>12} {:>10}",
                    seq_len, align_mark, "COMPILE_FAIL", "N/A", "N/A", "N/A", "N/A", "N/A"
                );
                eprintln!("  conv compile: {}", e);
                continue;
            }
        };

        // Conv2d input arena: [1, H, 1, S] -> elem count = 1*H*1*S
        let conv_in_arena =
            Arena::new(1, (H as u32) * seq_len, Dtype::Float16).expect("conv in arena");
        let conv_out_arena =
            Arena::new(1, (FFN as u32) * seq_len, Dtype::Float16).expect("conv out arena");
        fill_arena(&conv_in_arena, 1 * H as usize * 1 * seq_len as usize);

        let conv_time = match bench_one(
            conv_path.to_str().unwrap(),
            CoreMlComputeUnits::CpuAndNeuralEngine,
            &conv_in,
            &conv_in_arena,
            &conv_out,
            &conv_out_arena,
        ) {
            Ok(t) => t / 1000.0, // ns -> us
            Err(e) => {
                println!(
                    "{:>6} {:>6} {:>15} {:>15} {:>12} {:>12} {:>12} {:>10}",
                    seq_len, align_mark, "ANE_FAIL", "N/A", "N/A", "N/A", "N/A", "N/A"
                );
                eprintln!("  conv ANE: {}", e);
                continue;
            }
        };

        // ── Matmul model ──────────────────────────────────────────
        // Equivalent matmul: x[seq_len, H] @ W[H, FFN] -> [seq_len, FFN]
        let mm_tag = format!("mm_s{}", seq_len);
        let (mm_prog, mm_in, mm_out) = match build_matmul(seq_len) {
            Ok(v) => v,
            Err(e) => {
                println!(
                    "{:>6} {:>6} {:>15} {:>15} {:>12} {:>12} {:>12} {:>10}",
                    seq_len, align_mark, conv_time, "BUILD_FAIL", "N/A", "N/A", "N/A", "N/A"
                );
                eprintln!("  mm build: {}", e);
                continue;
            }
        };

        let mm_meta = ModelMeta {
            model_name: mm_tag.clone(),
            function_name: "main".into(),
            short_description: format!("mm_sweep_{}", seq_len),
            version: "1.0".into(),
            author: "prism".into(),
            output_name: mm_out.clone(),
            inputs: vec![("x".into(), vec![seq_len as i64, H])],
            outputs: vec![(mm_out.clone(), vec![seq_len as i64, FFN])],
            spec_version: 9,
        };

        let mm_path = match compile(&mm_tag, mm_prog, mm_meta) {
            Ok(p) => p,
            Err(e) => {
                println!(
                    "{:>6} {:>6} {:>15} {:>15} {:>12} {:>12} {:>12} {:>10}",
                    seq_len, align_mark, conv_time, "COMPILE_FAIL", "N/A", "N/A", "N/A", "N/A"
                );
                eprintln!("  mm compile: {}", e);
                continue;
            }
        };

        let mm_in_arena = Arena::new(seq_len, H as u32, Dtype::Float16).expect("mm in arena");
        let mm_out_arena = Arena::new(seq_len, FFN as u32, Dtype::Float16).expect("mm out arena");
        fill_arena(&mm_in_arena, (seq_len as usize) * (H as usize));

        let mm_time = match bench_one(
            mm_path.to_str().unwrap(),
            CoreMlComputeUnits::CpuAndNeuralEngine,
            &mm_in,
            &mm_in_arena,
            &mm_out,
            &mm_out_arena,
        ) {
            Ok(t) => t / 1000.0,
            Err(e) => {
                println!(
                    "{:>6} {:>6} {:>15} {:>15} {:>12} {:>12} {:>12} {:>10}",
                    seq_len, align_mark, conv_time, "ANE_FAIL", "N/A", "N/A", "N/A", "N/A"
                );
                eprintln!("  mm ANE: {}", e);
                continue;
            }
        };

        // ── Compute metrics ───────────────────────────────────────
        let total_flops = 2.0 * seq_len as f64 * H as f64 * FFN as f64;
        let t_s = conv_time / 1_000_000.0;
        let g = total_flops / t_s / 1_000_000_000.0;
        let conv_gflops_f = g;
        let conv_pct_f = conv_gflops_f / THEORETICAL_PEAK_GFLOPS * 100.0;

        let t_s2 = mm_time / 1_000_000.0;
        let g2 = total_flops / t_s2 / 1_000_000_000.0;
        let mm_gflops_f = g2;
        let mm_pct_f = mm_gflops_f / THEORETICAL_PEAK_GFLOPS * 100.0;

        println!(
            "{:>6} {:>6} {:>15.1} {:>15.1} {:>10.1} GFLOPS {:>10.1} GFLOPS {:>9.2}% {:>9.2}%",
            seq_len,
            align_mark,
            conv_time,
            mm_time,
            conv_gflops_f,
            mm_gflops_f,
            conv_pct_f,
            mm_pct_f
        );

        if !aligned {
            println!("  → NON-ALIGNED: seq_len={} ({} bytes on last axis). Conv time vs expected baseline?",
                seq_len, seq_len as usize * 2);
        }
    }

    println!("{}", "=".repeat(140));
    println!("Key: aligned = S*2 (bytes on last axis) % 64 == 0");
    println!("If conv_time shows no penalty at non-aligned S while mm_time shows a cliff,");
    println!("the Conv2d 1x1 path avoids the ANE's 64-byte alignment padding.");
}
