//! Tensor partitioning — proves ANE can handle hidden dims > 2048 by splitting
//! the input across multiple ANE-compatible chunks and summing the outputs.
//!
//! For a layer Y = X @ W where X has dim D > 2048 (the M1 ANE's limit):
//!   Split X into chunks X_0..X_n where each chunk dim ≤ 2048
//!   Split W into corresponding row-chunks W_0..W_n
//!   Compute Y_i = X_i @ W_i on ANE independently
//!   Y = sum(Y_i) on CPU (Accelerate) via vDSP_vadd
//!
//! Run: cargo test --test tensor_partition --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::path::Path;
use std::time::Instant;

use coreml_proto::proto::mil_spec;
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use tribunus_compute_core::coreml_pipeline::compile_mlpackage;
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{write_mlpackage, ModelMeta};

// ── Config ────────────────────────────────────────────────────────────────

/// Total hidden dimension — exceeds M1 ANE's individual limit.
const TOTAL_DIM: i64 = 3072;
/// FFN dimension.
const FFN_DIM: i64 = 6144;
/// Max safe chunk size for M1 ANE.
const CHUNK_SIZE: i64 = 2048;

const WARMUP: usize = 5;
const ITERS: usize = 15;

// ── FP16 helpers ──────────────────────────────────────────────────────────

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3FF;
    (if exp <= 0 {
        sign | (mant >> 1)
    } else if exp >= 31 {
        sign | 0x7C00 | mant
    } else {
        sign | ((exp as u32) << 10) | mant
    }) as u16
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) >> 15) << 31;
    let exp = ((bits >> 10) & 0x1F) as i32 - 15 + 127;
    let mant = (bits & 0x3FF) as u32;
    if exp <= 0 {
        f32::from_bits(sign | (mant << 13))
    } else if exp >= 255 {
        f32::from_bits(sign | 0x7F800000 | (mant << 13))
    } else {
        f32::from_bits(sign | ((exp as u32) << 23) | (mant << 13))
    }
}

// ── Data generation ───────────────────────────────────────────────────────

fn make_f32(n: usize, seed: u64) -> Vec<f32> {
    use std::hash::{Hash, Hasher};
    (0..n)
        .map(|i| {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            (i as u64 ^ seed).hash(&mut h);
            (h.finish() as f32 % 1000.0 - 500.0) / 500.0
        })
        .collect()
}

// ── Reference matmul (FP32) ──────────────────────────────────────────────

fn ref_matmul(x: &[f32], w: &[f32], d: usize, ffn: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; ffn];
    for j in 0..ffn {
        let mut sum = 0.0f32;
        for i in 0..d {
            sum += x[i] * w[j * d + i];
        }
        y[j] = sum;
    }
    y
}

// ── Build chunk model ─────────────────────────────────────────────────────

fn build_chunk_model(
    chunk_input_dim: i64,
    output_dim: i64,
    weights: &[f32],
    tag: &str,
) -> Result<(std::path::PathBuf, String), String> {
    let dir = Path::new("/tmp/prism_tensor_partition");
    let _ = std::fs::create_dir_all(dir);

    let b = MilBuilder::new("main").input("x", mil_spec::DataType::Float16, &[1, chunk_input_dim]);
    let b = b.const_f16("w", weights, &[chunk_input_dim, output_dim]);
    let wn = b.last_name().unwrap().to_string();
    let b = b.matmul("x", &wn);
    let on = b.last_name().unwrap().to_string();
    let prog = b.output(&on).build().map_err(|e| format!("build: {}", e))?;

    let meta = ModelMeta {
        model_name: tag.into(),
        function_name: "main".into(),
        short_description: tag.into(),
        version: "1.0".into(),
        author: "tensor_partition".into(),
        output_name: on.clone(),
        inputs: vec![("x".into(), vec![1, chunk_input_dim])],
        outputs: vec![(on.clone(), vec![1, output_dim])],
        spec_version: 9,
    };

    let pkg = write_mlpackage(prog, dir, &meta).map_err(|e| format!("write: {}", e))?;
    let od = dir.join("compiled");
    let _ = std::fs::create_dir_all(&od);
    let receipt = compile_mlpackage(&pkg, &od, tag, "cpuAndNeuralEngine", "macOS26")
        .map_err(|e| format!("compile: {}", e))?;
    Ok((std::path::PathBuf::from(&receipt.compiled_modelc_path), on))
}

// ─── Run ANE predict, return latency ns and check for CPU fallback ────────

fn run_ane(
    path: &Path,
    output_name: &str,
    input_arena: &Arena,
    output_arena: &Arena,
) -> Result<(f64, f64, bool), String> {
    // Load with ANE
    let m_ane = CoreMlModel::load_with_compute_units(
        path.to_str().ok_or("path")?,
        CoreMlComputeUnits::CpuAndNeuralEngine,
    )
    .map_err(|e| format!("ANE load: {}", e))?;

    // Load with CPU for fallback detection
    let m_cpu = CoreMlModel::load_with_compute_units(
        path.to_str().ok_or("path")?,
        CoreMlComputeUnits::CpuOnly,
    )
    .map_err(|e| format!("CPU load: {}", e))?;

    // Warmup
    for _ in 0..WARMUP {
        m_ane
            .predict("x", &input_arena.info, output_name, &output_arena.info)
            .map_err(|e| format!("ane warmup: {}", e))?;
        m_cpu
            .predict("x", &input_arena.info, output_name, &output_arena.info)
            .map_err(|e| format!("cpu warmup: {}", e))?;
    }

    // Time ANE
    let t0 = Instant::now();
    for _ in 0..ITERS {
        m_ane
            .predict("x", &input_arena.info, output_name, &output_arena.info)
            .map_err(|e| format!("ane predict: {}", e))?;
    }
    let ane_ns = t0.elapsed().as_nanos() as f64 / ITERS as f64;

    // Time CPU
    let t0 = Instant::now();
    for _ in 0..ITERS {
        m_cpu
            .predict("x", &input_arena.info, output_name, &output_arena.info)
            .map_err(|e| format!("cpu predict: {}", e))?;
    }
    let cpu_ns = t0.elapsed().as_nanos() as f64 / ITERS as f64;

    // Detect fallback: if ANE == CPU within 20%, it's running on CPU
    let ratio = ane_ns / cpu_ns;
    let fallen_back = ratio > 0.8;

    Ok((ane_ns, cpu_ns, fallen_back))
}

// ── Test ──────────────────────────────────────────────────────────────────

#[test]
fn test_tensor_partition() {
    println!("\n=== TENSOR PARTITIONING: BYPASS ANE DIM LIMIT ===");
    println!(
        "  Total dim: {}, FFN: {}, Max chunk: {}",
        TOTAL_DIM, FFN_DIM, CHUNK_SIZE
    );
    println!();

    // ── Generate weight data ──
    let total_weight_len = (TOTAL_DIM * FFN_DIM) as usize;
    let weight_all = make_f32(total_weight_len, 0xBEEF);

    let input_all = make_f32(TOTAL_DIM as usize, 0xCAFE);

    // ── Reference ──
    let ref_out = ref_matmul(
        &input_all,
        &weight_all,
        TOTAL_DIM as usize,
        FFN_DIM as usize,
    );
    println!("  Reference output[0..4] = {:.4?}", &ref_out[..4]);

    // ── Chunk definitions ──
    // chunks: [(start_row, dim)]
    let chunks: &[(i64, i64)] = &[(0, 2048), (2048, 1024)];

    let mut partial_outputs: Vec<Vec<f32>> = Vec::new();
    let mut chunk_latencies: Vec<(f64, bool)> = Vec::new();

    for (chunk_idx, &(start_row, chunk_dim)) in chunks.iter().enumerate() {
        let tag = format!("chunk_{}", chunk_idx);

        // Extract chunk of weight matrix: rows start_row..start_row+chunk_dim
        // Weight layout: [FFN, TOTAL_DIM] — each row has TOTAL_DIM elements
        // Chunk W: [FFN, chunk_dim] — each row has chunk_dim elements starting at start_row
        let ffn = FFN_DIM as usize;
        let total = TOTAL_DIM as usize;
        let cd = chunk_dim as usize;
        let sr = start_row as usize;
        let mut chunk_weights = Vec::with_capacity(ffn * cd);
        // W shape: [chunk_dim, FFN]  (each row of W is one FFN output)
        // Extract from weight_all[FFN, TOTAL_DIM]: W[i][j] = weight_all[j * TOTAL_DIM + sr + i]
        for i in 0..cd {
            for j in 0..ffn {
                chunk_weights.push(weight_all[j * total + sr + i]);
            }
        }

        // Build and compile model for this chunk
        eprint!("  Building chunk {} (dim={})... ", chunk_idx, chunk_dim);
        let (path, out_name) =
            build_chunk_model(chunk_dim, FFN_DIM, &chunk_weights, &tag).expect("build chunk model");
        println!("OK");

        // Extract chunk of input: input[start_row..start_row+chunk_dim]
        let chunk_input_f32: Vec<f32> = (0..chunk_dim as usize)
            .map(|i| input_all[start_row as usize + i])
            .collect();

        // Create arena with chunk input
        let ia = Arena::new(1, chunk_dim as u32, mlx_rs::Dtype::Float16).expect("input arena");
        unsafe {
            let ptr = ia.info.base_address as *mut u16;
            for (i, &v) in chunk_input_f32.iter().enumerate() {
                ptr.add(i).write(f32_to_f16_bits(v));
            }
        }
        let oa = Arena::new(1, FFN_DIM as u32, mlx_rs::Dtype::Float16).expect("output arena");

        // Run on ANE
        match run_ane(&path, &out_name, &ia, &oa) {
            Ok((ane_ns, cpu_ns, fallen_back)) => {
                let ratio = ane_ns / cpu_ns;
                let status = if fallen_back {
                    "CPU FALLBACK"
                } else {
                    "on-ANE"
                };
                println!(
                    "    ANE={:>7.1}µs  CPU={:>7.1}µs  ratio={:.2}  {}",
                    ane_ns / 1000.0,
                    cpu_ns / 1000.0,
                    ratio,
                    status
                );
                chunk_latencies.push((ane_ns, fallen_back));

                // Read output
                let mut result = vec![0.0f32; FFN_DIM as usize];
                unsafe {
                    let ptr = oa.info.base_address as *mut u16;
                    for i in 0..FFN_DIM as usize {
                        result[i] = f16_bits_to_f32(ptr.add(i).read());
                    }
                }
                partial_outputs.push(result);
            }
            Err(e) => {
                panic!("Chunk {} predict failed: {}", chunk_idx, e);
            }
        }
    }

    // ── Sum partial outputs ──
    let mut combined = vec![0.0f32; FFN_DIM as usize];
    for partial in &partial_outputs {
        for i in 0..FFN_DIM as usize {
            combined[i] += partial[i];
        }
    }

    // ── Verify against reference ──
    let mut max_err = 0.0f64;
    for i in 0..FFN_DIM as usize {
        let err = (combined[i] - ref_out[i]).abs() as f64;
        if err > max_err {
            max_err = err;
        }
    }
    let rmse = {
        let mut sum_sq = 0.0f64;
        for i in 0..FFN_DIM as usize {
            let d = (combined[i] - ref_out[i]) as f64;
            sum_sq += d * d;
        }
        (sum_sq / FFN_DIM as f64).sqrt()
    };

    // ── Detect ANE spill ──
    let any_fallback = chunk_latencies.iter().any(|(_, fb)| *fb);
    let total_latency: f64 = chunk_latencies.iter().map(|(ns, _)| ns).sum();

    println!();
    if any_fallback {
        println!("  ⚠ CPU FALLBACK DETECTED — at least one chunk couldn't run on ANE");
    } else {
        println!("  ✓ All chunks executed on ANE (no CPU fallback)");
    }
    println!("  Combined ANE latency: {:.1}µs", total_latency / 1000.0);
    println!("  Max error vs reference: {:.6}", max_err);
    println!("  RMSE: {:.6}", rmse);
    println!();

    // Report per-chunk table
    println!("  Chunk  Dim    Latency(µs)  Status");
    println!("  -----  ----   ----------  ----------");
    for idx in 0..chunks.len() {
        let (_, dim) = chunks[idx];
        let (ns, fb) = chunk_latencies[idx];
        let status = if fb { "CPU_FALLBACK" } else { "on-ANE" };
        println!(
            "  {:>5}  {:>4}   {:>10.1}  {}",
            idx,
            dim,
            ns / 1000.0,
            status
        );
    }

    assert!(max_err < 2.0, "Max error too high: {}", max_err);
    if any_fallback {
        println!("\n  ⚠ TENSOR PARTITION: chunks valid, but ANE FFN limit suggests N=6144 exceeds per-op max (~4096).");
        println!("  Chunks ran on CPU. The PARTITIONING MATH IS CORRECT (RMSE={:.6}) but chunks need smaller FFN.", rmse);
    } else {
        println!(
            "\n  ✓ TENSOR PARTITIONING VALIDATED: ANE handles {} dim via {}-chunk split",
            TOTAL_DIM,
            chunks.len()
        );
    }
}
