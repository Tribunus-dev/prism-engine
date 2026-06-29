//! CHANNEL-FACTORIZED-0001: Prove that a channel-factorized transformer
//! (permanently split across multiple independent ANE models) produces
//! correct results within FP16 precision.
//!
//! Architecture:
//!   Total hidden H=3072, FFN=6144.
//!   Split into 3 independent branches A(1024), B(1024), C(1024).
//!   Each branch: x_chunk[1, 1024] @ W_chunk[1024, 2048] → [1, 2048].
//!   Output: concat(A, B, C) → [1, 6144].
//!
//! Key design: For the concat approach to equal the full matmul, the weight
//! matrix must be block-diagonal: W[r, c] = 0 when r and c belong to
//! different channel blocks. This matches the channel-factorized transformer
//! architecture where each channel has independent parameters (e.g. per-head
//! projections in multi-head attention, or per-expert weights in MoE).
//!
//! With block-diagonal weights:
//!   x @ W = concat(x_A @ W_A, x_B @ W_B, x_C @ W_C) exactly.
//!
//! Run: cargo test --test channel_factorized --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

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

/// Root directory for compiled model artifacts.
const MODEL_DIR: &str = "/tmp/channel_factorized_models";

/// Total hidden dimension (input width).
const H: u32 = 3072;

/// Total FFN dimension (output width).
const FFN: u32 = 6144;

/// Number of independent channel branches.
const N_BRANCHES: u32 = 3;

/// Per-branch hidden dimension.
const BRANCH_H: u32 = H / N_BRANCHES; // 1024

/// Per-branch FFN dimension (output width per branch).
const BRANCH_FFN: u32 = FFN / N_BRANCHES; // 2048

/// Warmup predictions before measuring latency.
const WARMUP: u32 = 3;

/// Measured predictions for fallback detection.
const MEASURED: u32 = 5;

/// ANE/CPU latency ratio threshold for CPU fallback detection.
const FALLBACK_THRESHOLD: f64 = 0.95;

// ── FP16 conversion helpers ─────────────────────────────────────────────────

fn f32_to_f16_bits(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    if exp == 0 {
        return sign << 15;
    }
    if exp == 255 {
        return (sign << 15) | 0x7C00;
    }
    let new_exp = exp - 127 + 15;
    if new_exp <= 0 {
        return sign << 15;
    }
    if new_exp >= 31 {
        return (sign << 15) | 0x7C00;
    }
    let new_mant = (mant >> 13) as u16;
    (sign << 15) | ((new_exp as u16) << 10) | new_mant
}

fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) >> 15) << 31;
    let exp = ((h >> 10) & 0x1F) as i32 - 15 + 127;
    let mant = (h & 0x3FF) as u32;
    if exp <= 0 {
        let value = (mant as f32) * 2.0f32.powi(-24);
        if sign != 0 {
            -value
        } else {
            value
        }
    } else if exp >= 255 {
        f32::INFINITY
    } else {
        let normalized = 1.0f32 + (mant as f32) / 1024.0f32;
        let exponent = 2.0f32.powi(exp - 127);
        let value = normalized * exponent;
        if sign != 0 {
            -value
        } else {
            value
        }
    }
}

// ── RMSE ────────────────────────────────────────────────────────────────────

fn rmse(computed: &[f32], reference: &[f32]) -> f64 {
    let n = computed.len().min(reference.len());
    let mut sum_sq = 0.0f64;
    for i in 0..n {
        let diff = (computed[i] as f64) - (reference[i] as f64);
        sum_sq += diff * diff;
    }
    (sum_sq / n as f64).sqrt()
}

// ── Model building ──────────────────────────────────────────────────────────

/// Return the expected `.modelc` directory for a given branch index.
fn expected_modelc_dir(branch: u32) -> PathBuf {
    Path::new(MODEL_DIR).join(format!(
        "branch_br{}_H{}_F{}.modelc",
        branch, BRANCH_H, BRANCH_FFN
    ))
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

/// Build, compile, and cache an ANE model for one branch.
///
/// The model computes: input[1, BRANCH_H] × weight[BRANCH_H, BRANCH_FFN] → output[1, BRANCH_FFN].
/// The weight is extracted from the global weight tensor for the given branch.
fn build_branch_model(branch: u32, global_weight_f32: &[f32]) -> Result<PathBuf, String> {
    let model_dir = Path::new(MODEL_DIR);
    let _ = std::fs::create_dir_all(model_dir);

    let modelc_outer = expected_modelc_dir(branch);
    if modelc_outer.exists() {
        if let Some(inner) = find_modelc_inner(&modelc_outer) {
            return Ok(inner);
        }
    }

    // ── Extract weight chunk for this branch ───────────────────────────
    // global_weight has shape [H, FFN] = [3072, 6144].
    // Branch b uses rows [b*1024 : (b+1)*1024] and cols [b*2048 : (b+1)*2048].
    // So weight_chunk[b] = global_weight[1024*b..1024*(b+1), 2048*b..2048*(b+1)].
    // In row-major: weight_chunk[r,c] = global_weight[(br*BRANCH_H + r) * FFN + (bc*BRANCH_FFN + c)]
    let ffn_usize = FFN as usize;
    let bh_usize = BRANCH_H as usize;
    let bf_usize = BRANCH_FFN as usize;
    let branch_idx = branch as usize;

    let mut chunk_weight: Vec<f32> = Vec::with_capacity(bh_usize * bf_usize);
    for r in 0..bh_usize {
        for c in 0..bf_usize {
            let global_row = branch_idx * bh_usize + r;
            let global_col = branch_idx * bf_usize + c;
            chunk_weight.push(global_weight_f32[global_row * ffn_usize + global_col]);
        }
    }

    // ── Build MIL program ──────────────────────────────────────────────
    let prog = MilBuilder::new("main")
        .input("input", mil_spec::DataType::Float16, &[1, BRANCH_H as i64])
        .const_f16(
            "weight",
            &chunk_weight,
            &[BRANCH_H as i64, BRANCH_FFN as i64],
        )
        .matmul("input", "weight_0")
        .output("matmul_1")
        .build()
        .map_err(|e| format!("MIL build failed for branch {}: {:?}", branch, e))?;

    let meta = ModelMeta {
        model_name: format!("channel_branch_{}", branch),
        function_name: "main".into(),
        short_description: format!(
            "Channel factorized branch {}: H={} FFN={}",
            branch, BRANCH_H, BRANCH_FFN
        ),
        version: "1.0.0".into(),
        author: "Tribunus Compute".into(),
        output_name: "matmul_1".into(),
        inputs: vec![("input".into(), vec![1, BRANCH_H as i64])],
        outputs: vec![("matmul_1".into(), vec![1, BRANCH_FFN as i64])],
    };

    // ── Write .mlpackage ───────────────────────────────────────────────
    let tmp = tempfile::tempdir().map_err(|e| format!("tempdir: {}", e))?;
    let pkg_path = mlpackage::write_mlpackage(prog, tmp.path(), &meta)
        .map_err(|e| format!("mlpackage write failed for branch {}: {}", branch, e))?;

    // ── Compile via xcrun coremlcompiler ──────────────────────────────
    let island_id = format!("channel_branch_{}", branch);
    let receipt: CoreMlIslandReceipt = compile_mlpackage(
        &pkg_path,
        model_dir,
        &island_id,
        "cpuAndNeuralEngine",
        "CoreML9",
    )
    .map_err(|e| format!("compile failed for branch {}: {}", branch, e))?;

    let modelc_path = PathBuf::from(&receipt.compiled_modelc_path);
    if !modelc_path.exists() {
        return Err(format!("compiled modelc not found at {:?}", modelc_path));
    }

    Ok(modelc_path)
}

// ── CPU matmul reference ────────────────────────────────────────────────────

/// Compute y[1, FFN] = x[1, H] @ W[H, FFN] in FP32.
fn ref_matmul(x: &[f32], w: &[f32], h: usize, ffn: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; ffn];
    for c in 0..ffn {
        let mut acc = 0.0f32;
        for r in 0..h {
            acc += x[r] * w[r * ffn + c];
        }
        y[c] = acc;
    }
    y
}

// ── Branch prediction helper ────────────────────────────────────────────────

/// Run one branch through its ANE model and return the output FP32 values.
fn predict_branch(model: &CoreMlModel, input_chunk: &[f32]) -> Result<Vec<f32>, String> {
    let h_chunk = BRANCH_H;
    let ffn_chunk = BRANCH_FFN;

    // Allocate FP16 arenas.
    let input_arena = Arena::new(1, h_chunk, mlx_rs::Dtype::Float16)
        .map_err(|e| format!("input arena alloc: {}", e))?;
    let output_arena = Arena::new(1, ffn_chunk, mlx_rs::Dtype::Float16)
        .map_err(|e| format!("output arena alloc: {}", e))?;

    // Fill input arena with FP16 data.
    {
        input_arena
            .lock()
            .map_err(|e| format!("input lock: {}", e))?;
        unsafe {
            let ptr = input_arena.base_ptr() as *mut u16;
            for i in 0..h_chunk as usize {
                ptr.add(i).write(f32_to_f16_bits(input_chunk[i]));
            }
        }
        input_arena
            .unlock()
            .map_err(|e| format!("input unlock: {}", e))?;
    }

    // Predict.
    model
        .predict("input", &input_arena.info, "matmul_1", &output_arena.info)
        .map_err(|e| format!("predict: {}", e))?;

    // Read output.
    let mut out_f32 = vec![0.0f32; ffn_chunk as usize];
    unsafe {
        let ptr = output_arena.base_ptr() as *mut u16;
        for i in 0..ffn_chunk as usize {
            out_f32[i] = f16_bits_to_f32(ptr.add(i).read());
        }
    }

    Ok(out_f32)
}

// ── CPU fallback detection ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct BranchLatency {
    branch: u32,
    ane_ns: f64,
    cpu_ns: f64,
    ratio: f64,
    is_cpu_fallback: bool,
}

/// Detect whether the given branch model is running on ANE or falling back to CPU.
fn detect_cpu_fallback(
    model_path: &str,
    branch: u32,
    input_chunk: &[f32],
) -> Result<BranchLatency, String> {
    let h_chunk = BRANCH_H;
    let ffn_chunk = BRANCH_FFN;

    // Load with CpuAndNeuralEngine.
    let ane_model =
        CoreMlModel::load_with_compute_units(model_path, CoreMlComputeUnits::CpuAndNeuralEngine)
            .map_err(|e| format!("load ANE branch {}: {}", branch, e))?;

    // Load with CpuOnly.
    let cpu_model = CoreMlModel::load_with_compute_units(model_path, CoreMlComputeUnits::CpuOnly)
        .map_err(|e| format!("load CPU branch {}: {}", branch, e))?;

    // Allocate arenas.
    let input_arena = Arena::new(1, h_chunk, mlx_rs::Dtype::Float16)
        .map_err(|e| format!("input arena: {}", e))?;
    let output_arena = Arena::new(1, ffn_chunk, mlx_rs::Dtype::Float16)
        .map_err(|e| format!("output arena: {}", e))?;

    // Fill input.
    {
        input_arena
            .lock()
            .map_err(|e| format!("input lock: {}", e))?;
        unsafe {
            let ptr = input_arena.base_ptr() as *mut u16;
            for i in 0..h_chunk as usize {
                ptr.add(i).write(f32_to_f16_bits(input_chunk[i]));
            }
        }
        input_arena
            .unlock()
            .map_err(|e| format!("input unlock: {}", e))?;
    }

    // Warmup both.
    for _ in 0..WARMUP {
        ane_model
            .predict("input", &input_arena.info, "matmul_1", &output_arena.info)
            .map_err(|e| format!("ANE warmup branch {}: {}", branch, e))?;
        cpu_model
            .predict("input", &input_arena.info, "matmul_1", &output_arena.info)
            .map_err(|e| format!("CPU warmup branch {}: {}", branch, e))?;
    }

    // Measure ANE latency (nanoseconds).
    let ane_start = Instant::now();
    for _ in 0..MEASURED {
        ane_model
            .predict("input", &input_arena.info, "matmul_1", &output_arena.info)
            .map_err(|e| format!("ANE measured branch {}: {}", branch, e))?;
    }
    let ane_ns = ane_start.elapsed().as_nanos() as f64 / MEASURED as f64;

    // Measure CPU latency (nanoseconds).
    let cpu_start = Instant::now();
    for _ in 0..MEASURED {
        cpu_model
            .predict("input", &input_arena.info, "matmul_1", &output_arena.info)
            .map_err(|e| format!("CPU measured branch {}: {}", branch, e))?;
    }
    let cpu_ns = cpu_start.elapsed().as_nanos() as f64 / MEASURED as f64;

    let ratio = if cpu_ns > 0.0 {
        ane_ns / cpu_ns
    } else {
        f64::INFINITY
    };
    let is_cpu_fallback = ratio > FALLBACK_THRESHOLD;

    Ok(BranchLatency {
        branch,
        ane_ns,
        cpu_ns,
        ratio,
        is_cpu_fallback,
    })
}

// ── Test ────────────────────────────────────────────────────────────────────

#[test]
fn channel_factorized_correctness() {
    println!("=== Channel Factorized Correctness Test ===");
    println!(
        "H={}, FFN={}, {} branches of H={}, FFN={} each",
        H, FFN, N_BRANCHES, BRANCH_H, BRANCH_FFN
    );
    println!(
        "Warmup={}, Measured={}, Fallback threshold={}",
        WARMUP, MEASURED, FALLBACK_THRESHOLD
    );

    let h_usize = H as usize;
    let ffn_usize = FFN as usize;
    let bh_usize = BRANCH_H as usize;
    let bf_usize = BRANCH_FFN as usize;

    // ── 1. Generate random input and weight ──────────────────────────────
    println!(
        "\n--- Generating random input x[1, {}] and W[{}, {}] ---",
        H, H, FFN
    );
    let mut x: Vec<f32> = Vec::with_capacity(h_usize);
    let mut w: Vec<f32> = Vec::with_capacity(h_usize * ffn_usize);
    for i in 0..h_usize {
        let seed = i as u64;
        let v = (seed as f32) / (h_usize as f32) * 2.0 - 1.0;
        x.push(v);
    }

    // Build block-diagonal weight matrix.
    // For the concat approach to equal the full matmul, the weight matrix
    // must be block-diagonal: only the diagonal block for each channel
    // has non-zero entries. Off-diagonal blocks are zero.
    //
    // Block A: W[0:1024, 0:2048]     ← rows 0-1023, cols 0-2047
    // Block B: W[1024:2048, 2048:4096] ← rows 1024-2047, cols 2048-4095
    // Block C: W[2048:3072, 4096:6144] ← rows 2048-3071, cols 4096-6143
    w.resize(h_usize * ffn_usize, 0.0f32);
    for b in 0..N_BRANCHES as usize {
        let br = b;
        let row_off = br * bh_usize;
        let col_off = br * bf_usize;
        let mut idx = 0u64;
        for r in 0..bh_usize {
            for c in 0..bf_usize {
                let seed = idx
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add((br as u64) * 1000 + 42);
                let v = ((seed >> 33) as f32) / (1u64 << 31) as f32;
                w[(row_off + r) * ffn_usize + (col_off + c)] = v;
                idx += 1;
            }
        }
    }
    println!(
        "  x range: [{:.4}, {:.4}]",
        x.iter().cloned().fold(f32::INFINITY, f32::min),
        x.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );
    println!(
        "  W range: [{:.4}, {:.4}]",
        w.iter().cloned().fold(f32::INFINITY, f32::min),
        w.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    );

    // ── 2. Reference: x @ W (FP32) ──────────────────────────────────────
    println!("\n--- Computing FP32 reference matmul ---");
    let ref_out = ref_matmul(&x, &w, h_usize, ffn_usize);
    let ref_range = (
        ref_out.iter().cloned().fold(f32::INFINITY, f32::min),
        ref_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );
    println!(
        "  Reference range: [{:.4}, {:.4}]",
        ref_range.0, ref_range.1
    );

    // ── 3. Build models for each branch ──────────────────────────────────
    println!("\n--- Building 3 ANE models (one per branch) ---");
    let mut model_paths: Vec<String> = Vec::with_capacity(N_BRANCHES as usize);
    for b in 0..N_BRANCHES {
        match build_branch_model(b, &w) {
            Ok(p) => {
                println!("  Branch {} model: {:?}", b, p);
                model_paths.push(p.to_string_lossy().to_string());
            }
            Err(e) => {
                panic!("Build failed for branch {}: {}", b, e);
            }
        }
    }

    // ── 4. Extract chunks and predict ────────────────────────────────────
    println!("\n--- Predicting on all 3 branches ---");
    let mut branch_outputs: Vec<Vec<f32>> = Vec::with_capacity(N_BRANCHES as usize);
    let mut fallback_results: Vec<BranchLatency> = Vec::new();

    for b in 0..N_BRANCHES {
        let br_idx = b as usize;

        // Extract input chunk: x[br_idx * BRANCH_H .. (br_idx+1) * BRANCH_H]
        let x_chunk: Vec<f32> = x[br_idx * bh_usize..(br_idx + 1) * bh_usize].to_vec();

        // Load model and run prediction.
        let model = CoreMlModel::load(&model_paths[br_idx])
            .unwrap_or_else(|e| panic!("Load branch {} model: {}", b, e));

        let out = predict_branch(&model, &x_chunk)
            .unwrap_or_else(|e| panic!("Predict branch {}: {}", b, e));
        println!(
            "  Branch {} output: {} elements, range=[{:.4}, {:.4}], sum={:.4}",
            b,
            out.len(),
            out.iter().cloned().fold(f32::INFINITY, f32::min),
            out.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            out.iter().sum::<f32>()
        );
        branch_outputs.push(out);

        // Detect CPU fallback.
        match detect_cpu_fallback(&model_paths[br_idx], b, &x_chunk) {
            Ok(lat) => {
                let label = if lat.is_cpu_fallback {
                    "CPU_FALLBACK"
                } else {
                    "on-ANE"
                };
                println!(
                    "  Branch {} CPU fallback: ane={:.0}ns, cpu={:.0}ns, ratio={:.3} => {}",
                    b, lat.ane_ns, lat.cpu_ns, lat.ratio, label
                );
                fallback_results.push(lat);
            }
            Err(e) => {
                eprintln!(
                    "  [WARN] CPU fallback detection failed for branch {}: {}",
                    b, e
                );
            }
        }
    }

    // ── 5. Concatenate outputs ───────────────────────────────────────────
    println!("\n--- Concatenating branch outputs ---");
    let mut full_out: Vec<f32> = Vec::with_capacity(ffn_usize);
    for bo in &branch_outputs {
        full_out.extend_from_slice(bo);
    }
    println!(
        "  Full output: {} elements, range=[{:.4}, {:.4}], sum={:.4}",
        full_out.len(),
        full_out.iter().cloned().fold(f32::INFINITY, f32::min),
        full_out.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        full_out.iter().sum::<f32>()
    );

    // ── 6. Compare against reference ────────────────────────────────────
    println!("\n--- Comparing against FP32 reference ---");
    let rmse_v = rmse(&full_out, &ref_out);
    let ref_range_w = ref_range.1 - ref_range.0;

    println!("  RMSE = {:.8}", rmse_v);
    println!("  Reference range width = {:.4}", ref_range_w);
    println!("  RMSE / range = {:.8}", rmse_v / ref_range_w as f64);

    // Per-branch error analysis.
    println!("\n--- Per-branch RMSE ---");
    for b in 0..N_BRANCHES {
        let br_idx = b as usize;
        let bo = &branch_outputs[br_idx];
        let ref_chunk = &ref_out[br_idx * bf_usize..(br_idx + 1) * bf_usize];
        let ch_rmse = rmse(bo, ref_chunk);
        println!("  Branch {} RMSE = {:.8}", b, ch_rmse);
    }

    // ── 7. Assert correctness ─────────────────────────────────────────────
    // FP16 matmul through ANE should produce results very close to FP32
    // reference. RMSE should be small relative to the output range.
    // Allow for FP16 precision loss: typical max error per element in FP16
    // is ~0.1% of the range or ~5e-3 in absolute terms.
    let max_acceptable_rmse = (ref_range_w as f64) * 0.005; // 0.5% of range
    println!("\n  Max acceptable RMSE = {:.8}", max_acceptable_rmse);

    assert!(
        rmse_v < max_acceptable_rmse || rmse_v < 0.1,
        "RMSE {} exceeds threshold {} (or absolute 0.1). \
         Full output differs significantly from FP32 reference.",
        rmse_v,
        max_acceptable_rmse
    );

    // ── 8. Summary ────────────────────────────────────────────────────────
    println!("\n{}", "=".repeat(60));
    println!("CHANNEL FACTORIZED TEST RESULT");
    println!("  H={}, FFN={}, Branches={}", H, FFN, N_BRANCHES);
    println!(
        "  Reference range: [{:.4}, {:.4}]",
        ref_range.0, ref_range.1
    );
    println!("  Concatenated RMSE: {:.8}", rmse_v);
    println!("  Status: PASS (factorized matmul matches reference)");

    for lat in &fallback_results {
        let label = if lat.is_cpu_fallback {
            "CPU_FALLBACK"
        } else {
            "on-ANE"
        };
        println!(
            "  Branch {}: {} (ane={:.0}ns, cpu={:.0}ns, ratio={:.3})",
            lat.branch, label, lat.ane_ns, lat.cpu_ns, lat.ratio
        );
    }
    println!("{}", "=".repeat(60));
}
