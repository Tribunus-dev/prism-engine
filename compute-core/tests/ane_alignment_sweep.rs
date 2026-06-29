//! Direct 64-byte alignment test: sweep sequence lengths across alignment boundaries.
//!
//! Tests whether the ANE silently pads tensors when the last axis byte size
//! isn't 64-byte aligned. For FP16: S × 2 % 64 == 0 means S % 32 == 0.
//!
//! Model: x[S, 2048] @ W[2048, 4096] -> [S, 4096]
//! Sweeps S across alignment boundaries: 61..128, plus up to 320.
//!
//! If a latency spike appears at non-aligned S, the ANE is doing hidden padding.
//!
//! Run: cargo test --test ane_alignment_sweep --features prism-backend -- --nocapture

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

const TEST_DIR: &str = "/tmp/prism_ane_alignment";
const H: i64 = 2048;
const FFN: i64 = 4096;
const THEORETICAL_PEAK_GFLOPS: f64 = 11_000.0;
const WARMUP: usize = 5;
const SAMPLES: usize = 15;

/// Sweep sequence lengths, focusing on 64-byte alignment boundaries.
/// For FP16 (2 bytes/elem), aligned when S × 2 % 64 == 0 => S % 32 == 0.
const SEQ_LENS: &[u32] = &[
    61, 62, 63, 64, 65, 66, 67, // cross the 64 boundary
    95, 96, 97, // cross the 96 boundary
    127, 128, 129, // cross the 128 boundary
    160, 192, 224, 256, 288, 320, // larger
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

fn build_matmul(batch: u32) -> Result<(mil_spec::Program, String, String), String> {
    let w = seeded_weights(42, H, FFN);
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

fn bench(
    path: &str,
    in_name: &str,
    in_arena: &Arena,
    out_name: &str,
    out_arena: &Arena,
) -> Result<f64, String> {
    let m = CoreMlModel::load_with_compute_units(path, CoreMlComputeUnits::CpuAndNeuralEngine)
        .map_err(|e| format!("load: {}", e))?;
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
fn ane_alignment_sweep() {
    println!("\n=== ANE 64-BYTE ALIGNMENT SWEEP ===");
    println!("Model: x[S,{}] @ W[{},{}] -> [S,{}]", H, H, FFN, FFN);
    println!("FP16: aligned when S*2 (bytes on last axis) % 64 == 0, i.e. S % 32 == 0");
    println!("{}", "=".repeat(120));
    println!(
        "{:>6} {:>6} {:>6} {:>12} {:>12} {:>10} {:>10}",
        "S", "bytes", "align", "Time(us)", "GFLOPS", "%Peak", "tok/s"
    );
    println!("{}", "-".repeat(120));

    let mut prev_time: Option<f64> = None;
    let mut prev_s: Option<u32> = None;

    for &seq_len in SEQ_LENS {
        let tag = format!("align_s{}", seq_len);
        let bytes_last = (seq_len as u32 * 2) as i64; // FP16: 2 bytes/elem
        let aligned = bytes_last % 64 == 0;
        let align_str = if aligned { "YES" } else { "NO" };

        let (prog, in_name, out_name) = match build_matmul(seq_len) {
            Ok(v) => v,
            Err(_e) => {
                println!(
                    "{:>6} {:>6} {:>6} {:>12} {:>10} {:>10} {:>10}",
                    seq_len, bytes_last, align_str, "BUILD_FAIL", "N/A", "N/A", "N/A"
                );
                continue;
            }
        };

        let meta = ModelMeta {
            model_name: tag.clone(),
            function_name: "main".into(),
            short_description: format!("align_sweep_{}", seq_len),
            version: "1.0".into(),
            author: "prism".into(),
            output_name: out_name.clone(),
            inputs: vec![("x".into(), vec![seq_len as i64, H])],
            outputs: vec![(out_name.clone(), vec![seq_len as i64, FFN])],

        };

        let model_path = match compile(&tag, prog, meta) {
            Ok(p) => p,
            Err(_e) => {
                println!(
                    "{:>6} {:>6} {:>6} {:>12} {:>10} {:>10} {:>10}",
                    seq_len, bytes_last, align_str, "COMPILE_FAIL", "N/A", "N/A", "N/A"
                );
                continue;
            }
        };

        let in_arena = Arena::new(seq_len, H as u32, DataType::Float16).expect("in arena");
        let out_arena = Arena::new(seq_len, FFN as u32, DataType::Float16).expect("out arena");
        fill_arena(&in_arena, (seq_len as usize) * (H as usize));

        let time_ns = match bench(
            model_path.to_str().unwrap(),
            &in_name,
            &in_arena,
            &out_name,
            &out_arena,
        ) {
            Ok(t) => t,
            Err(_e) => {
                println!(
                    "{:>6} {:>6} {:>6} {:>12} {:>10} {:>10} {:>10}",
                    seq_len, bytes_last, align_str, "ANE_FAIL", "N/A", "N/A", "N/A"
                );
                continue;
            }
        };
        let time_us = time_ns / 1000.0;

        let total_flops = 2.0 * seq_len as f64 * H as f64 * FFN as f64;
        let time_s = time_ns / 1_000_000_000.0;
        let gflops = total_flops / time_s / 1_000_000_000.0;
        let pct = gflops / THEORETICAL_PEAK_GFLOPS * 100.0;
        let tok_s = if time_us > 0.0 {
            1_000_000.0 / (time_us * 48.0 / seq_len as f64)
        } else {
            0.0
        };

        println!(
            "{:>6} {:>6} {:>6} {:>12.1} {:>10.2} {:>9.3}% {:>10.1}",
            seq_len, bytes_last, align_str, time_us, gflops, pct, tok_s
        );

        // Check for alignment cliff: compare to previous time scaled by FLOPs ratio
        if let (Some(prev), Some(prev_s_val)) = (prev_time, prev_s) {
            let flops_ratio = seq_len as f64 / prev_s_val as f64;
            let expected_us = prev * flops_ratio;
            let ratio = time_us / expected_us;
            if ratio > 1.3 {
                println!(
                    "  ← CLIFF: {:.1}× slower than linear scaling from S={}",
                    ratio, prev_s_val
                );
            }
        }
        prev_time = Some(time_us);
        prev_s = Some(seq_len);
    }

    println!("{}", "=".repeat(120));
    println!("If time jumps at non-aligned S, the ANE is padding the tensor internally.");
    println!("Smooth scaling across boundaries means no hidden padding.");
}
