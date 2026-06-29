//! Shape bucketing: enumerated batch sizes with zero-pad execution.
//!
//! Compiles ANE models at batch sizes 1, 2, 4 for a [B, 2048] @ [2048, 4096] matmul.
//! Measures latency for 3 individual requests (batch=1 each) vs a single
//! zero-padded batch-4 predict. Verifies correctness by comparing individual
//! row results against their batched counterparts.
//!
//! Run: cargo test --test shape_bucketing --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::path::{Path, PathBuf};
use std::time::Instant;

use coreml_proto::proto::mil_spec;
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::arena::DataType;
use tribunus_compute_core::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use tribunus_compute_core::coreml_pipeline::compile_mlpackage;
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{write_mlpackage, ModelMeta};

const MODEL_DIR: &str = "/tmp/prism_shape_bucketing";
const IN_DIM: u32 = 2048;
const OUT_DIM: u32 = 4096;
const NUM_REQUESTS: usize = 3;
const BATCH_4: u32 = 4;
const WARMUP: usize = 5;
const ITERS: usize = 50;

/// Deterministic weight data [IN_DIM, OUT_DIM] as f32 values.
/// Same seed produces identical weights for all compiled models.
fn generate_weights() -> Vec<f32> {
    use std::hash::{Hash, Hasher};
    let len = (IN_DIM as usize) * (OUT_DIM as usize);
    let mut w = Vec::with_capacity(len);
    for i in 0..len {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        i.hash(&mut h);
        w.push((h.finish() as f32 % 1000.0 - 500.0) / 500.0);
    }
    w
}

/// Deterministic input data for request `req_idx`, shape [1, IN_DIM].
fn generate_input(req_idx: usize) -> Vec<f32> {
    use std::hash::{Hash, Hasher};
    let len = IN_DIM as usize;
    let mut x = Vec::with_capacity(len);
    for i in 0..len {
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (req_idx * 10000 + i).hash(&mut h);
        x.push((h.finish() as f32 % 1000.0 - 500.0) / 500.0);
    }
    x
}

/// Build and compile a matmul model: y = x @ W where x is [batch, IN_DIM],
/// W is [IN_DIM, OUT_DIM], y is [batch, OUT_DIM]. Uses FP16 throughout.
fn build_model(batch: i64, weights: &[f32], tag: &str) -> Result<(PathBuf, String), String> {
    let model_dir = Path::new(MODEL_DIR);
    let _ = std::fs::create_dir_all(model_dir);

    let b = MilBuilder::new("main");
    let b = b.input(
        "input",
        mil_spec::DataType::Float16,
        &[batch, IN_DIM as i64],
    );
    let b = b.const_f16("weight", weights, &[IN_DIM as i64, OUT_DIM as i64]);
    let w_name = b.last_name().ok_or("no weight name")?.to_string();
    let b = b.matmul("input", &w_name);
    let out_name = b.last_name().ok_or("no out name")?.to_string();
    let prog = b
        .output(&out_name)
        .build()
        .map_err(|e| format!("MIL build: {:?}", e))?;

    let meta = ModelMeta {
        model_name: tag.into(),
        function_name: "main".into(),
        short_description: format!("batch={} matmul", batch),
        version: "1.0.0".into(),
        author: "shape_bucketing".into(),
        output_name: out_name.clone(),
        inputs: vec![("input".into(), vec![batch, IN_DIM as i64])],
        outputs: vec![(out_name.clone(), vec![batch, OUT_DIM as i64])],
    };

    let tmp = tempfile::tempdir().map_err(|e| format!("tempdir: {}", e))?;
    let pkg_path = write_mlpackage(prog, tmp.path(), &meta)?;
    let output_dir = model_dir.join("compiled");
    std::fs::create_dir_all(&output_dir).ok();
    let receipt = compile_mlpackage(&pkg_path, &output_dir, tag, "cpuAndNeuralEngine", "CoreML9")
        .map_err(|e| format!("compile {}: {}", tag, e))?;

    Ok((PathBuf::from(&receipt.compiled_modelc_path), out_name))
}

/// Build and compile a matmul model with explicit input/output dimensions.
/// Unlike `build_model` (which uses global IN_DIM/OUT_DIM), this lets the
/// caller specify any (in_dim, out_dim) pair, e.g. (4096, 8192).
fn build_model_with_dims(
    batch: i64,
    in_dim: i64,
    out_dim: i64,
    weights: &[f32],
    tag: &str,
) -> Result<(PathBuf, String), String> {
    let model_dir = Path::new(MODEL_DIR);
    let _ = std::fs::create_dir_all(model_dir);

    let b = MilBuilder::new("main");
    let b = b.input("input", mil_spec::DataType::Float16, &[batch, in_dim]);
    let b = b.const_f16("weight", weights, &[in_dim, out_dim]);
    let w_name = b.last_name().ok_or("no weight name")?.to_string();
    let b = b.matmul("input", &w_name);
    let out_name = b.last_name().ok_or("no out name")?.to_string();
    let prog = b
        .output(&out_name)
        .build()
        .map_err(|e| format!("MIL build: {:?}", e))?;

    let meta = ModelMeta {
        model_name: tag.into(),
        function_name: "main".into(),
        short_description: format!("batch={} matmul", batch),
        version: "1.0.0".into(),
        author: "shape_bucketing".into(),
        output_name: out_name.clone(),
        inputs: vec![("input".into(), vec![batch, in_dim])],
        outputs: vec![(out_name.clone(), vec![batch, out_dim])],
    };

    let tmp = tempfile::tempdir().map_err(|e| format!("tempdir: {}", e))?;
    let pkg_path = write_mlpackage(prog, tmp.path(), &meta)?;
    let output_dir = model_dir.join("compiled");
    std::fs::create_dir_all(&output_dir).ok();
    let receipt = compile_mlpackage(&pkg_path, &output_dir, tag, "cpuAndNeuralEngine", "CoreML9")
        .map_err(|e| format!("compile {}: {}", tag, e))?;

    Ok((PathBuf::from(&receipt.compiled_modelc_path), out_name))
}

/// Write f32 values as FP16 into an arena at a given element offset.
unsafe fn write_fp16_arena(arena: &Arena, data: &[f32], offset: usize) {
    let ptr = arena.base_ptr() as *mut u16;
    for (i, &v) in data.iter().enumerate() {
        ptr.add(offset + i).write(half::f16::from_f32(v).to_bits());
    }
}

/// Read a contiguous region from an FP16 arena into f32 values.
unsafe fn read_fp16_arena(arena: &Arena, offset: usize, len: usize) -> Vec<f32> {
    let ptr = arena.base_ptr() as *const u16;
    (0..len)
        .map(|i| half::f16::from_bits(ptr.add(offset + i).read()).to_f32())
        .collect()
}

#[test]
fn test_shape_bucketing() {
    // 1. Generate weight data (deterministic, same for all models)
    let weights = generate_weights();

    // 2. Build and compile models at batch=1, batch=2, batch=4
    let (path_b1, out_b1) = build_model(1, &weights, "batch_1").expect("build batch=1");
    let (_path_b2, _out_b2) = build_model(2, &weights, "batch_2").expect("build batch=2");
    let (path_b4, out_b4) = build_model(4, &weights, "batch_4").expect("build batch=4");

    // 3. Load models
    let model_b1 = CoreMlModel::load_with_compute_units(
        &path_b1.to_string_lossy(),
        tribunus_compute_core::coreml_bridge::CoreMlComputeUnits::CpuAndNeuralEngine,
    )
    .expect("load batch=1");
    let model_b4 = CoreMlModel::load_with_compute_units(
        &path_b4.to_string_lossy(),
        tribunus_compute_core::coreml_bridge::CoreMlComputeUnits::CpuAndNeuralEngine,
    )
    .expect("load batch=4");

    // 4. Generate 3 request inputs, each shape [1, IN_DIM]
    let requests: Vec<Vec<f32>> = (0..NUM_REQUESTS).map(|i| generate_input(i)).collect();

    // ── Test A: individual (batch=1) ───────────────────────────────

    let ia_b1 = Arena::new(1, IN_DIM, mlx_rs::Dtype::Float16).expect("input arena b1");
    let oa_b1 = Arena::new(1, OUT_DIM, mlx_rs::Dtype::Float16).expect("output arena b1");

    // Warmup: run each request through batch=1 model (WARMUP times each)
    for _ in 0..WARMUP {
        for req in &requests {
            unsafe {
                write_fp16_arena(&ia_b1, req, 0);
            }
            model_b1
                .predict("input", &ia_b1.info, &out_b1, &oa_b1.info)
                .expect("warmup predict");
        }
    }

    // Measured: sum over all 3 requests per iteration
    let t0 = Instant::now();
    for _ in 0..ITERS {
        for req in &requests {
            unsafe {
                write_fp16_arena(&ia_b1, req, 0);
            }
            model_b1
                .predict("input", &ia_b1.info, &out_b1, &oa_b1.info)
                .expect("predict");
        }
    }
    let batch1_total_ns = t0.elapsed().as_nanos();

    // Capture individual results for correctness comparison
    let mut individual_results: Vec<Vec<f32>> = Vec::with_capacity(NUM_REQUESTS);
    for req in &requests {
        unsafe {
            write_fp16_arena(&ia_b1, req, 0);
        }
        model_b1
            .predict("input", &ia_b1.info, &out_b1, &oa_b1.info)
            .expect("capture predict");
        let row = unsafe { read_fp16_arena(&oa_b1, 0, OUT_DIM as usize) };
        individual_results.push(row);
    }

    // ── Test B: batched (batch=4, zero-padded) ─────────────────────

    let ia_b4 = Arena::new(BATCH_4, IN_DIM, mlx_rs::Dtype::Float16).expect("input arena b4");
    let oa_b4 = Arena::new(BATCH_4, OUT_DIM, mlx_rs::Dtype::Float16).expect("output arena b4");

    // Fill batch-4 input: rows 0..=2 from requests, row 3 stays zero
    // (IOSurface memory is zero-initialized on allocation)
    unsafe {
        for (row, req) in requests.iter().enumerate() {
            write_fp16_arena(&ia_b4, req, row * IN_DIM as usize);
        }
        // Row 3 is already zeroed by IOSurface zero-init
    }

    // Warmup
    for _ in 0..WARMUP {
        model_b4
            .predict("input", &ia_b4.info, &out_b4, &oa_b4.info)
            .expect("warmup batch-4");
    }

    // Measured
    let t0 = Instant::now();
    for _ in 0..ITERS {
        model_b4
            .predict("input", &ia_b4.info, &out_b4, &oa_b4.info)
            .expect("batch-4 predict");
    }
    let batch4_total_ns = t0.elapsed().as_nanos();

    // Capture batched output for correctness
    model_b4
        .predict("input", &ia_b4.info, &out_b4, &oa_b4.info)
        .expect("capture batch-4");

    // Verify rows 0-2 match individual results (tolerate FP16 rounding)
    let mut batched_rows_ok = true;
    for row in 0..NUM_REQUESTS {
        let batched = unsafe { read_fp16_arena(&oa_b4, row * OUT_DIM as usize, OUT_DIM as usize) };
        for c in 0..OUT_DIM as usize {
            let diff = (batched[c] - individual_results[row][c]).abs();
            let max_abs = batched[c].abs().max(individual_results[row][c].abs());
            let rel_err = if max_abs > 0.0 { diff / max_abs } else { diff };
            // FP16 matmul has limited precision; allow small tolerance
            if rel_err > 0.01 && diff > 0.001 {
                batched_rows_ok = false;
                break;
            }
        }
        if !batched_rows_ok {
            break;
        }
    }

    // Verify row 3 is all zero (zero input -> zero output)
    let row3_data = unsafe { read_fp16_arena(&oa_b4, 3 * OUT_DIM as usize, OUT_DIM as usize) };
    let row3_zero = row3_data.iter().all(|&v| v == 0.0);

    // ── Report ─────────────────────────────────────────────────────

    let individual_total_ns = batch1_total_ns as f64 / ITERS as f64;
    let per_batch_ns = batch4_total_ns as f64 / ITERS as f64;
    let speedup = individual_total_ns / per_batch_ns;

    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                Shape Bucketing Results                       ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!(
        "║ Individual latency (batch=1, sum of 3): {:>12.0} ns  ║",
        individual_total_ns
    );
    println!(
        "║ Batched latency   (batch=4, one predict): {:>12.0} ns  ║",
        per_batch_ns
    );
    println!(
        "║ Speedup:                                   {:>8.2}x    ║",
        speedup
    );
    println!(
        "║ Row 0-2 correctness:                       {:>8}       ║",
        if batched_rows_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "║ Row 3 zero output:                         {:>8}       ║",
        if row3_zero { "PASS" } else { "FAIL" }
    );
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    assert!(
        batched_rows_ok,
        "batched rows 0-2 must match individual results"
    );
    assert!(row3_zero, "row 3 must be zero output");
}

#[test]
fn test_batch_crossover_sweep() {
    const WARMUP: usize = 5;
    const ITERS: usize = 10;
    const BATCH_SIZES: &[i64] = &[1, 2, 4, 8];
    const MATRIX_SIZES: &[(i64, i64)] = &[(2048, 4096), (4096, 8192)];

    println!();
    println!(
        "{:>6} {:>5} {:>14} {:>14} {:>8}  {:>6}",
        "Hidden", "Batch", "Individual(ns)", "Batched(ns)", "Speedup", "Winner"
    );
    println!("{}", "-".repeat(62));

    for &(hidden, ffn) in MATRIX_SIZES {
        let in_dim = hidden as usize;
        let out_dim = ffn as usize;
        let weights_len = in_dim * out_dim;

        // Deterministic weights for this matrix size
        let weights = {
            use std::hash::{Hash, Hasher};
            let mut w = Vec::with_capacity(weights_len);
            for i in 0..weights_len {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                i.hash(&mut h);
                w.push((h.finish() as f32 % 1000.0 - 500.0) / 500.0);
            }
            w
        };

        // Build and load batch=1 model once for individual measurements
        let tag_b1 = format!("cross_h{}_f{}_b1", hidden, ffn);
        let (path_b1, out_b1) =
            build_model_with_dims(1, hidden, ffn, &weights, &tag_b1).expect("build batch=1");
        let model_b1 = CoreMlModel::load_with_compute_units(
            &path_b1.to_string_lossy(),
            tribunus_compute_core::coreml_bridge::CoreMlComputeUnits::CpuAndNeuralEngine,
        )
        .expect("load batch=1");

        let ia_b1 = Arena::new(1, hidden as u32, mlx_rs::Dtype::Float16).expect("input arena b1");
        let oa_b1 = Arena::new(1, ffn as u32, mlx_rs::Dtype::Float16).expect("output arena b1");

        for &batch in BATCH_SIZES {
            // Generate deterministic inputs for this batch
            let requests: Vec<Vec<f32>> = (0..batch as usize)
                .map(|i| {
                    use std::hash::{Hash, Hasher};
                    let mut x = Vec::with_capacity(in_dim);
                    for j in 0..in_dim {
                        let mut h = std::collections::hash_map::DefaultHasher::new();
                        (i * 10000 + j).hash(&mut h);
                        x.push((h.finish() as f32 % 1000.0 - 500.0) / 500.0);
                    }
                    x
                })
                .collect();

            // -- Individual timing: B separate batch=1 predictions ---------
            for _ in 0..WARMUP {
                for req in &requests {
                    unsafe {
                        write_fp16_arena(&ia_b1, req, 0);
                    }
                    model_b1
                        .predict("input", &ia_b1.info, &out_b1, &oa_b1.info)
                        .expect("warmup indiv");
                }
            }
            let t0 = Instant::now();
            for _ in 0..ITERS {
                for req in &requests {
                    unsafe {
                        write_fp16_arena(&ia_b1, req, 0);
                    }
                    model_b1
                        .predict("input", &ia_b1.info, &out_b1, &oa_b1.info)
                        .expect("indiv predict");
                }
            }
            let indiv_ns = t0.elapsed().as_nanos() as f64 / ITERS as f64;

            // -- Batched timing: one batch=B prediction -------------------
            if batch == 1 {
                // Single-row: individual == batched, report directly
                let speedup = 1.0;
                println!(
                    "{:>6} {:>5} {:>14.0} {:>14.0} {:>8.2}  {:>6}",
                    hidden, batch, indiv_ns, indiv_ns, speedup, "tie"
                );
                continue;
            }

            let tag = format!("cross_h{}_f{}_b{}", hidden, ffn, batch);
            let (path_b_b, out_b_b) = build_model_with_dims(batch, hidden, ffn, &weights, &tag)
                .expect("build batched model");
            let model_b_b = CoreMlModel::load_with_compute_units(
                &path_b_b.to_string_lossy(),
                tribunus_compute_core::coreml_bridge::CoreMlComputeUnits::CpuAndNeuralEngine,
            )
            .expect("load batched model");

            let ia_b_b = Arena::new(batch as u32, hidden as u32, mlx_rs::Dtype::Float16)
                .expect("input arena batch");
            let oa_b_b = Arena::new(batch as u32, ffn as u32, mlx_rs::Dtype::Float16)
                .expect("output arena batch");

            unsafe {
                for (row, req) in requests.iter().enumerate() {
                    write_fp16_arena(&ia_b_b, req, row * in_dim);
                }
            }

            for _ in 0..WARMUP {
                model_b_b
                    .predict("input", &ia_b_b.info, &out_b_b, &oa_b_b.info)
                    .expect("warmup batched");
            }
            let t0 = Instant::now();
            for _ in 0..ITERS {
                model_b_b
                    .predict("input", &ia_b_b.info, &out_b_b, &oa_b_b.info)
                    .expect("batched predict");
            }
            let batched_ns = t0.elapsed().as_nanos() as f64 / ITERS as f64;

            let speedup = indiv_ns / batched_ns;
            let winner = if speedup > 1.5 {
                "batch"
            } else if speedup < 0.95 {
                "indiv"
            } else {
                "tie"
            };

            println!(
                "{:>6} {:>5} {:>14.0} {:>14.0} {:>8.2}  {:>6}",
                hidden, batch, indiv_ns, batched_ns, speedup, winner
            );
        }
    }
    println!();
}

/// Sweep batch sizes [1, 2, 4, 8, 16, 32, 64, 128] with hidden=4096, ffn=8192
/// to find the maximum batch the ANE can handle before spilling to CPU.
///
/// Detects: compile failure, load failure, predict failure, and silent CPU fallback
/// (ANE latency within 20% of CPU-only latency).
#[test]
fn test_ane_batch_capacity() {
    const WARMUP: usize = 5;
    const ITERS: usize = 10;
    const HIDDEN: i64 = 4096;
    const FFN: i64 = 8192;
    const BATCH_SIZES: &[i64] = &[1, 2, 4, 8, 16, 32, 64, 128];

    let in_dim = HIDDEN as usize;
    let out_dim = FFN as usize;
    let weights_len = in_dim * out_dim;

    // Deterministic weights for hidden=4096, ffn=8192
    let weights = {
        use std::hash::{Hash, Hasher};
        let mut w = Vec::with_capacity(weights_len);
        for i in 0..weights_len {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            i.hash(&mut h);
            w.push((h.finish() as f32 % 1000.0 - 500.0) / 500.0);
        }
        w
    };

    // Table header
    println!();
    println!(
        "{:>5}  {:>10}  {:>10}  {:>8}  {:>12}",
        "Batch", "ANE(us)", "CPU(us)", "Ratio", "Status"
    );
    println!("{}", "-".repeat(52));

    let mut max_ane_batch: i64 = 0;

    for &batch in BATCH_SIZES {
        let tag = format!("ane_cap_b{}", batch);

        // ── Step (a): Build (compile) ──────────────────────────────────────
        let (model_path, out_name) = match build_model_with_dims(batch, HIDDEN, FFN, &weights, &tag)
        {
            Ok(pair) => pair,
            Err(e) => {
                println!(
                    "{:>5}  {:>10}  {:>10}  {:>8}  {:>12}",
                    batch, "FAIL", "FAIL", "N/A", "COMPILE_FAIL"
                );
                eprintln!("COMPILE FAIL batch={}: {}", batch, e);
                break;
            }
        };

        // ── Step (b): Load with CpuAndNeuralEngine ────────────────────────
        let model_ane = match CoreMlModel::load_with_compute_units(
            &model_path.to_string_lossy(),
            CoreMlComputeUnits::CpuAndNeuralEngine,
        ) {
            Ok(m) => m,
            Err(e) => {
                println!(
                    "{:>5}  {:>10}  {:>10}  {:>8}  {:>12}",
                    batch, "FAIL", "FAIL", "N/A", "LOAD_FAIL"
                );
                eprintln!("LOAD FAIL batch={}: {}", batch, e);
                break;
            }
        };

        // ── Load same compiled model with CpuOnly ─────────────────────────
        let model_cpu = match CoreMlModel::load_with_compute_units(
            &model_path.to_string_lossy(),
            CoreMlComputeUnits::CpuOnly,
        ) {
            Ok(m) => m,
            Err(e) => {
                println!(
                    "{:>5}  {:>10}  {:>10}  {:>8}  {:>12}",
                    batch, "FAIL", "FAIL", "N/A", "CPU_LOAD_FAIL"
                );
                eprintln!("CPU LOAD FAIL batch={}: {}", batch, e);
                break;
            }
        };

        // Generate deterministic inputs for this batch
        let requests: Vec<Vec<f32>> = (0..batch as usize)
            .map(|i| {
                use std::hash::{Hash, Hasher};
                let mut x = Vec::with_capacity(in_dim);
                for j in 0..in_dim {
                    let mut h = std::collections::hash_map::DefaultHasher::new();
                    (i * 10000 + j).hash(&mut h);
                    x.push((h.finish() as f32 % 1000.0 - 500.0) / 500.0);
                }
                x
            })
            .collect();

        let arena_in =
            Arena::new(batch as u32, HIDDEN as u32, mlx_rs::Dtype::Float16).expect("input arena");
        let arena_out =
            Arena::new(batch as u32, FFN as u32, mlx_rs::Dtype::Float16).expect("output arena");

        unsafe {
            for (row, req) in requests.iter().enumerate() {
                write_fp16_arena(&arena_in, req, row * in_dim);
            }
        }

        // ── Step (c): Benchmark ANE (5 warmup + 10 measured) ──────────────
        let mut ane_failed = false;
        let mut ane_t0 = Instant::now();
        for _ in 0..WARMUP {
            if let Err(e) = model_ane.predict("input", &arena_in.info, &out_name, &arena_out.info) {
                eprintln!("PREDICT FAIL batch={} (ANE warmup): {}", batch, e);
                ane_failed = true;
                break;
            }
        }
        if !ane_failed {
            ane_t0 = Instant::now();
            for _ in 0..ITERS {
                if let Err(e) =
                    model_ane.predict("input", &arena_in.info, &out_name, &arena_out.info)
                {
                    eprintln!("PREDICT FAIL batch={} (ANE measured): {}", batch, e);
                    ane_failed = true;
                    break;
                }
            }
        }

        if ane_failed {
            println!(
                "{:>5}  {:>10}  {:>10}  {:>8}  {:>12}",
                batch, "FAIL", "FAIL", "N/A", "PREDICT_FAIL"
            );
            break;
        }
        let ane_ns = ane_t0.elapsed().as_nanos();
        let ane_us = ane_ns as f64 / ITERS as f64 / 1_000.0;

        // ── Step (d): Benchmark CPU on same model (5 warmup + 10 measured) ─
        let mut cpu_failed = false;
        let mut cpu_t0 = Instant::now();
        for _ in 0..WARMUP {
            if let Err(e) = model_cpu.predict("input", &arena_in.info, &out_name, &arena_out.info) {
                eprintln!("CPU PREDICT FAIL batch={} (warmup): {}", batch, e);
                cpu_failed = true;
                break;
            }
        }
        if !cpu_failed {
            cpu_t0 = Instant::now();
            for _ in 0..ITERS {
                if let Err(e) =
                    model_cpu.predict("input", &arena_in.info, &out_name, &arena_out.info)
                {
                    eprintln!("CPU PREDICT FAIL batch={} (measured): {}", batch, e);
                    cpu_failed = true;
                    break;
                }
            }
        }

        if cpu_failed {
            println!(
                "{:>5}  {:>10.1}  {:>10}  {:>8}  {:>12}",
                batch, ane_us, "FAIL", "N/A", "CPU_PREDICT_FAIL"
            );
            break;
        }
        let cpu_ns = cpu_t0.elapsed().as_nanos();
        let cpu_us = cpu_ns as f64 / ITERS as f64 / 1_000.0;

        // ── Step (e): Detect spill ────────────────────────────────────────
        let ratio = ane_us / cpu_us;
        if ratio > 0.8 {
            println!(
                "{:>5}  {:>10.1}  {:>10.1}  {:>8.2}  {:>12}",
                batch, ane_us, cpu_us, ratio, "CPU_FALLBACK"
            );
            eprintln!(
                "CPU FALLBACK DETECTED batch={}: ane={:.1}us cpu={:.1}us",
                batch, ane_us, cpu_us,
            );
            break;
        }

        max_ane_batch = batch;
        println!(
            "{:>5}  {:>10.1}  {:>10.1}  {:>8.2}  {:>12}",
            batch, ane_us, cpu_us, ratio, "on-ANE"
        );
    }

    println!("{}", "-".repeat(52));
    println!("Max ANE batch = {}", max_ane_batch);
    println!();
}
