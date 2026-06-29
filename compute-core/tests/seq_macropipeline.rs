//! Sequence-chunked macropipelining test — ANE dispatch overhead is hidden
//! behind CPU accumulation of the previous chunk.
//!
//! Architecture:
//!   Long prefill: 1024 tokens, chunked into 8 × 128-token chunks.
//!   Each chunk: ANE matmul [128, 2048] @ [2048, 4096] → [128, 4096].
//!
//! Two modes (both measure wall-clock time for all 8 chunks):
//!   Sequential — for each chunk: ANE predict() → wait → CPU accumulate.
//!   Pipelined  — main thread calls predict() (blocks on ANE) while a
//!                worker thread accumulates the *previous* chunk's output.
//!                Overlap hides the ~95 µs ANE dispatch overhead from
//!                the critical path for all but the final accumulation.
//!
//! Expected: pipelined < sequential (overlap shrinks total wall time).
//!
//! Run: cargo test --test seq_macropipeline --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use coreml_proto::proto::mil_spec;
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use tribunus_compute_core::coreml_pipeline::compile_mlpackage;
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{write_mlpackage, ModelMeta};

// ── Constants ────────────────────────────────────────────────────────────

const N_CHUNKS: usize = 8;
const CHUNK_SIZE: usize = 128;
const HIDDEN: usize = 2048;
const FFN_DIM: usize = 4096;
const N_WEIGHT: usize = HIDDEN * FFN_DIM;
const CHUNK_ELTS: usize = CHUNK_SIZE * HIDDEN;
const OUTPUT_ELTS: usize = CHUNK_SIZE * FFN_DIM;

const WARMUP_ITERS: usize = 5;
const TIMED_ITERS: usize = 10;

const MODEL_DIR: &str = "/tmp/seq_macropipeline_models";

// ── FP16 conversion helpers ──────────────────────────────────────────────
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) >> 15) << 31;
    let exp = ((bits >> 10) & 0x1F) as i32 - 15 + 127;
    let mant = (bits & 0x3FF) as u32;
    if exp <= 0 {
        f32::from_bits(sign | mant << 13)
    } else if exp >= 255 {
        f32::from_bits(sign | 0x7F800000 | (mant << 13))
    } else {
        f32::from_bits(sign | ((exp as u32) << 23) | (mant << 13))
    }
}

// ── Deterministic data generation ────────────────────────────────────────

/// LCG with per-element seed for deterministic FP16 values.
fn fill_deterministic(buf: &mut [u16], base_seed: u64) {
    for (i, v) in buf.iter_mut().enumerate() {
        let x = (i as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(base_seed);
        *v = ((x >> 33) as u16) & 0x7FFF; // positive only
    }
}

/// Generate weight values in f32, used at model-build time.
fn make_weight_f32() -> Vec<f32> {
    let mut w = Vec::with_capacity(N_WEIGHT);
    for i in 0..N_WEIGHT {
        let x = (i as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(42);
        let val = ((x >> 33) as f32) / (1u64 << 31) as f32;
        w.push(val);
    }
    w
}

/// Generate N_CHUNKS input chunks, each CHUNK_ELTS FP16 values.
fn generate_chunks() -> Vec<Vec<u16>> {
    let mut chunks = Vec::with_capacity(N_CHUNKS);
    for c in 0..N_CHUNKS {
        let mut buf = vec![0u16; CHUNK_ELTS];
        fill_deterministic(&mut buf, 100 + c as u64);
        chunks.push(buf);
    }
    chunks
}

// ── Model building ───────────────────────────────────────────────────────

fn build_model() -> Result<PathBuf, String> {
    let model_dir = Path::new(MODEL_DIR);
    let _ = std::fs::create_dir_all(model_dir);

    let modelc_dir = model_dir.join("seq_macropipeline.modelc");
    if let Some(inner) = find_modelc_inner(&modelc_dir) {
        return Ok(inner);
    }

    let weight_f32 = make_weight_f32();

    let prog = MilBuilder::new("main")
        .input(
            "input",
            mil_spec::DataType::Float16,
            &[CHUNK_SIZE as i64, HIDDEN as i64],
        )
        .const_f16("weight", &weight_f32, &[HIDDEN as i64, FFN_DIM as i64])
        .matmul("input", "weight_0")
        .output("matmul_1")
        .build()
        .map_err(|e| format!("MIL build: {:?}", e))?;

    let meta = ModelMeta {
        model_name: "seq_macropipeline".into(),
        function_name: "main".into(),
        short_description: "ANE sequence macropipeline matmul".into(),
        version: "1.0.0".into(),
        author: "Tribunus Compute".into(),
        output_name: "matmul_1".into(),
        inputs: vec![("input".into(), vec![CHUNK_SIZE as i64, HIDDEN as i64])],
        outputs: vec![("matmul_1".into(), vec![CHUNK_SIZE as i64, FFN_DIM as i64])],
        spec_version: 9,
    };

    let tmp = tempfile::tempdir().map_err(|e| format!("tempdir: {}", e))?;
    let pkg_path =
        write_mlpackage(prog, tmp.path(), &meta).map_err(|e| format!("mlpackage write: {}", e))?;

    let receipt = compile_mlpackage(
        &pkg_path,
        model_dir,
        "seq_macropipeline",
        "cpuAndNeuralEngine",
        "CoreML9",
    )
    .map_err(|e| format!("compile: {}", e))?;

    let modelc_path = PathBuf::from(&receipt.compiled_modelc_path);
    if !modelc_path.exists() {
        return Err(format!("compiled modelc not found at {:?}", modelc_path));
    }
    Ok(modelc_path)
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

// ── Per-iteration helpers ────────────────────────────────────────────────

/// Write an FP16 chunk into the input arena.
fn write_input(arena: &Arena, chunk: &[u16]) {
    arena.lock().expect("input lock");
    unsafe {
        std::ptr::copy_nonoverlapping(chunk.as_ptr(), arena.base_ptr() as *mut u16, chunk.len());
    }
    arena.unlock().expect("input unlock");
}

/// Read the output arena into a Vec<u16>.
fn read_output(arena: &Arena) -> Vec<u16> {
    let mut buf = vec![0u16; OUTPUT_ELTS];
    arena.lock().expect("output lock");
    unsafe {
        std::ptr::copy_nonoverlapping(
            arena.base_ptr() as *const u16,
            buf.as_mut_ptr(),
            OUTPUT_ELTS,
        );
    }
    arena.unlock().expect("output unlock");
    buf
}

/// Accumulate one FP16 output buffer into the f32 accumulator.
fn accumulate(output: &[u16], acc: &mut [f32]) {
    assert_eq!(output.len(), OUTPUT_ELTS);
    for (j, &v) in output.iter().enumerate() {
        acc[j % FFN_DIM] += f16_bits_to_f32(v);
    }
}

// ── Sequential baseline ──────────────────────────────────────────────────

/// Sequential: for each of 8 chunks, ANE predict → wait → CPU accumulate.
/// Returns wall-clock seconds for all 8 chunks.
fn run_sequential(
    model: &CoreMlModel,
    input_arena: &Arena,
    output_arena: &Arena,
    chunks: &[Vec<u16>],
    input_name: &str,
    output_name: &str,
) -> f64 {
    let start = Instant::now();
    let mut acc = vec![0.0f32; FFN_DIM];

    for chunk in chunks {
        write_input(input_arena, chunk);
        model
            .predict(
                input_name,
                &input_arena.info,
                output_name,
                &output_arena.info,
            )
            .expect("sequential predict");
        let out = read_output(output_arena);
        accumulate(&out, &mut acc);
    }

    let elapsed = start.elapsed().as_secs_f64();
    std::hint::black_box(&acc);
    elapsed
}

// ── Pipelined benchmark ──────────────────────────────────────────────────

/// Pipelined: main thread calls ANE predict() for chunk i while a worker
/// thread accumulates chunk i-1.  Overlap hides dispatch overhead.
///
/// Returns wall-clock seconds for all 8 chunks.
fn run_pipelined(
    model: &CoreMlModel,
    input_arena: &Arena,
    output_arena: &Arena,
    chunks: &[Vec<u16>],
    input_name: &str,
    output_name: &str,
) -> f64 {
    let start = Instant::now();
    let acc = vec![0.0f32; FFN_DIM];

    // Channel: main thread sends output copies to worker for accumulation.
    let (tx, rx) = mpsc::channel::<Vec<u16>>();

    let worker = thread::Builder::new()
        .name("seq-pipeline-worker".into())
        .spawn(move || {
            // We use a local accumulator — no mutex contention.
            let mut local_acc = vec![0.0f32; FFN_DIM];
            for _ in 0..N_CHUNKS {
                let buf = rx.recv().expect("worker recv");
                accumulate(&buf, &mut local_acc);
            }
            // Prevent the compiler from eliding the accumulation.
            std::hint::black_box(&local_acc);
        })
        .expect("spawn worker");

    // Chunk 0: predict first, then send for accumulation.
    write_input(input_arena, &chunks[0]);
    model
        .predict(
            input_name,
            &input_arena.info,
            output_name,
            &output_arena.info,
        )
        .expect("pipelined predict[0]");
    tx.send(read_output(output_arena)).expect("send[0]");

    // Chunks 1..N-1: predict next chunk while worker accumulates previous.
    for chunk in &chunks[1..] {
        write_input(input_arena, chunk);
        model
            .predict(
                input_name,
                &input_arena.info,
                output_name,
                &output_arena.info,
            )
            .expect("pipelined predict");
        tx.send(read_output(output_arena)).expect("send");
    }

    // Wait for worker to finish the final accumulation.
    worker.join().expect("worker join");

    let elapsed = start.elapsed().as_secs_f64();
    // Touch the shared accumulator to keep it alive.
    std::hint::black_box(&acc);
    elapsed
}

// ── Test ─────────────────────────────────────────────────────────────────

#[test]
fn test_seq_macropipeline_overlap() {
    // ── Build / load model ───────────────────────────────────────────────
    let modelc_path = build_model().expect("model build & compile should succeed");

    let model = CoreMlModel::load_with_compute_units(
        &modelc_path.to_string_lossy(),
        CoreMlComputeUnits::CpuAndNeuralEngine,
    )
    .expect("load ANE model");

    // ── Allocate arenas ──────────────────────────────────────────────────
    let input_arena =
        Arena::new(CHUNK_SIZE as u32, HIDDEN as u32, mlx_rs::Dtype::Float16).expect("input arena");
    let output_arena = Arena::new(CHUNK_SIZE as u32, FFN_DIM as u32, mlx_rs::Dtype::Float16)
        .expect("output arena");

    let input_name = "input".to_string();
    let output_name = "matmul_1".to_string();
    let chunks = generate_chunks();

    // ── Warmup ──────────────────────────────────────────────────────────
    for _ in 0..WARMUP_ITERS {
        // Both modes to prime ANE caches.
        std::hint::black_box(run_sequential(
            &model,
            &input_arena,
            &output_arena,
            &chunks,
            &input_name,
            &output_name,
        ));
        std::hint::black_box(run_pipelined(
            &model,
            &input_arena,
            &output_arena,
            &chunks,
            &input_name,
            &output_name,
        ));
    }

    // ── Timed iterations ────────────────────────────────────────────────
    let mut seq_times: Vec<f64> = Vec::with_capacity(TIMED_ITERS);
    let mut pipe_times: Vec<f64> = Vec::with_capacity(TIMED_ITERS);

    for i in 0..TIMED_ITERS {
        // Alternate order to cancel any ordering bias.
        if i % 2 == 0 {
            seq_times.push(run_sequential(
                &model,
                &input_arena,
                &output_arena,
                &chunks,
                &input_name,
                &output_name,
            ));
            pipe_times.push(run_pipelined(
                &model,
                &input_arena,
                &output_arena,
                &chunks,
                &input_name,
                &output_name,
            ));
        } else {
            pipe_times.push(run_pipelined(
                &model,
                &input_arena,
                &output_arena,
                &chunks,
                &input_name,
                &output_name,
            ));
            seq_times.push(run_sequential(
                &model,
                &input_arena,
                &output_arena,
                &chunks,
                &input_name,
                &output_name,
            ));
        }
    }

    // ── Statistics ──────────────────────────────────────────────────────
    fn mean(v: &[f64]) -> f64 {
        v.iter().sum::<f64>() / v.len() as f64
    }

    let seq_mean = mean(&seq_times);
    let pipe_mean = mean(&pipe_times);
    let speedup = seq_mean / pipe_mean;
    let overlap_savings = seq_mean - pipe_mean;

    println!("─── Sequential-chunked Macropipeline ───");
    println!(
        "  Chunks: {} × {}@{} → {}",
        N_CHUNKS, CHUNK_SIZE, HIDDEN, FFN_DIM
    );
    println!("  Warmup: {} | Measured: {}", WARMUP_ITERS, TIMED_ITERS);
    println!();
    println!("  Sequential  (mean): {:.6}s", seq_mean);
    println!("  Pipelined   (mean): {:.6}s", pipe_mean);
    println!("  Speedup:           {:.4}×", speedup);
    println!(
        "  Overlap savings:   {:.6}s ({:.1}%)",
        overlap_savings,
        (1.0 - pipe_mean / seq_mean) * 100.0,
    );
    println!();
    println!("  Per-iteration sequential times: {:?}", seq_times);
    println!("  Per-iteration pipelined  times: {:?}", pipe_times);

    // Assertion: pipelined is measurably faster
    assert!(
        pipe_mean < seq_mean * 0.99,
        "pipelined ({:.6}s) should be faster than sequential ({:.6}s) — dispatch overhead hidden behind accumulation",
        pipe_mean,
        seq_mean,
    );

    println!(
        "  ✓ PASS: pipelined {:.6}s < sequential {:.6}s (speedup {:.4}×)",
        pipe_mean, seq_mean, speedup
    );
}
