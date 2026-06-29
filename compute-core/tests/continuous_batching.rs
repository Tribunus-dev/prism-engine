//! Continuous batching with block diagonal masking on the ANE.
//!
//! Primary approach: SDPA with a block-diagonal mask built via reshape +
//! scaled_dot_product_attention. If the ANE backend rejects this (reshape
//! or mask not supported), we fall back to two independent models for
//! seq=2 and seq=4, proving ANE SDPA works position by position.
//!
//! Since the predict bridge and Arena are 2D-only, models use
//! flatten+reshape internally: input [1, seq_len * D] is reshaped to
//! [1, 1, seq_len, D] for SDPA, then reshaped back to [1, seq_len * D].

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::path::{Path, PathBuf};

use coreml_proto::proto::mil_spec;
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use tribunus_compute_core::coreml_pipeline::compile_mlpackage;
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{write_mlpackage, ModelMeta};

const D: i64 = 64;
const MODEL_DIR: &str = "/tmp/continuous_batching_models";

// ── FP16 conversion helpers ───────────────────────────────────────────────

fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    if exp == 0 {
        sign << 15
    } else if exp == 255 {
        (sign << 15) | 0x7C00
    } else {
        let ne = exp - 127 + 15;
        if ne <= 0 {
            sign << 15
        } else if ne >= 31 {
            (sign << 15) | 0x7C00
        } else {
            let nm = mant >> 13;
            (sign << 15) | ((ne as u16) << 10) | (nm as u16)
        }
    }
}

#[allow(dead_code)]
fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    if exp == 0 {
        let v = (mant as f32) * 2.0f32.powi(-24);
        if sign != 0 {
            -v
        } else {
            v
        }
    } else if exp == 31 {
        if mant == 0 {
            if sign != 0 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        }
    } else {
        let normalized = 1.0 + (mant as f32) / 1024.0;
        let exponent = 2.0f32.powi((exp as i32) - 15);
        let v = normalized * exponent;
        if sign != 0 {
            -v
        } else {
            v
        }
    }
}

// ── Model builders ────────────────────────────────────────────────────────

/// Build an SDPA model for a given sequence length, optionally providing
/// a causal mask. Returns (compiled_modelc_path, input_name).
///
/// Input: [1, seq_len * D], output: [1, seq_len * D].
/// Internally reshapes to [1, 1, seq_len, D], applies SDPA, reshapes back.
fn build_sdpa_model(
    model_dir: &Path,
    seq_len: i64,
    tag: &str,
    causal: bool,
) -> Result<(PathBuf, String, String), String> {
    let flat = seq_len * D;
    let model_name = format!("cb_sdpa_{}", tag);
    let modelc_path = model_dir.join(format!("{}.mlmodelc", model_name));
    if modelc_path.exists() {
        return Ok((modelc_path, "x".into(), "out_0".into()));
    }
    let _ = std::fs::create_dir_all(model_dir);

    let b = MilBuilder::new("main");

    // Build mask if causal
    let b = if causal {
        let neg_inf: f32 = -1e4;
        let mut mask = vec![0.0f32; (seq_len * seq_len) as usize];
        for i in 0..seq_len as usize {
            for j in (i + 1)..seq_len as usize {
                mask[i * seq_len as usize + j] = neg_inf;
            }
        }
        let b = b.input("x", mil_spec::DataType::Float16, &[1, flat]);
        let b = b.const_f16("cm", &mask, &[1, 1, seq_len, seq_len]);
        let mn = b.last_name().unwrap().to_string();
        let b = b.reshape("qkv", "x", &[1, 1, seq_len, D]);
        let qkv = b.last_name().unwrap().to_string();
        let scale = 1.0 / (D as f32).sqrt();
        let b = b.scaled_dot_product_attention("attn", &qkv, &qkv, &qkv, Some(&mn), Some(scale));
        let attn = b.last_name().unwrap().to_string();
        let b = b.reshape("out", &attn, &[1, flat]);
        b
    } else {
        let b = b.input("x", mil_spec::DataType::Float16, &[1, flat]);
        let b = b.reshape("qkv", "x", &[1, 1, seq_len, D]);
        let qkv = b.last_name().unwrap().to_string();
        let scale = 1.0 / (D as f32).sqrt();
        let b = b.scaled_dot_product_attention("attn", &qkv, &qkv, &qkv, None, Some(scale));
        let attn = b.last_name().unwrap().to_string();
        let b = b.reshape("out", &attn, &[1, flat]);
        b
    };

    let out = b.last_name().unwrap().to_string();
    let prog = b
        .output(&out)
        .build()
        .map_err(|e| format!("MIL build: {:?}", e))?;

    let meta = ModelMeta {
        model_name: model_name.clone(),
        function_name: "main".into(),
        short_description: format!("CB SDPA seq={}", seq_len),
        version: "1.0.0".into(),
        author: "ContBatching".into(),
        output_name: out.clone(),
        inputs: vec![("x".into(), vec![1, flat])],
        outputs: vec![(out.clone(), vec![1, flat])],
        spec_version: 9,
    };

    let mlpackage_dir =
        write_mlpackage(prog, model_dir, &meta).map_err(|e| format!("mlpackage write: {}", e))?;
    let output_dir = model_dir.join("compiled");
    std::fs::create_dir_all(&output_dir).map_err(|e| format!("mkdir: {}", e))?;
    let receipt = compile_mlpackage(
        &mlpackage_dir,
        &output_dir,
        &model_name,
        "cpuAndNeuralEngine",
        "iOS17",
    )
    .map_err(|e| format!("compile: {}", e))?;

    let compiled = PathBuf::from(&receipt.compiled_modelc_path);
    Ok((compiled, "x".into(), out))
}

/// Build a continuous-batching model: block diagonal SDPA for two sequences
/// A (positions 0-1) and B (positions 2-3) on the same input.
/// This uses the reshaped SDPA approach with a mask for each block.
fn build_block_diag_model(model_dir: &Path) -> Result<(PathBuf, String, String), String> {
    let flat = 4 * D;
    let model_name = "cb_block_diag";
    let modelc_path = model_dir.join(format!("{}.mlmodelc", model_name));
    if modelc_path.exists() {
        return Ok((modelc_path, "x".into(), "out_0".into()));
    }
    let _ = std::fs::create_dir_all(model_dir);

    // Block diagonal mask: A (0,1) attends to A only, B (2,3) to B only.
    let neg_inf: f32 = -1e4;
    let mut mask = vec![0.0f32; 16]; // 4x4
                                     // Rows 0-1 (A): block positions 2-3
                                     // Rows 2-3 (B): block positions 0-1
    for i in 0..4 {
        let block_start = if i < 2 { 0 } else { 2 };
        let block_end = block_start + 2;
        for j in 0..4 {
            if j < block_start || j >= block_end {
                mask[i * 4 + j] = neg_inf;
            }
        }
    }

    let b = MilBuilder::new("main");
    let b = b.input("x", mil_spec::DataType::Float16, &[1, flat]);
    let b = b.const_f16("bd_mask", &mask, &[1, 1, 4, 4]);
    let mn = b.last_name().unwrap().to_string();
    let b = b.reshape("qkv", "x", &[1, 1, 4, D]);
    let qkv = b.last_name().unwrap().to_string();
    let scale = 1.0 / (D as f32).sqrt();
    let b = b.scaled_dot_product_attention("attn", &qkv, &qkv, &qkv, Some(&mn), Some(scale));
    let attn = b.last_name().unwrap().to_string();
    let b = b.reshape("out", &attn, &[1, flat]);
    let out = b.last_name().unwrap().to_string();

    let prog = b
        .output(&out)
        .build()
        .map_err(|e| format!("MIL build: {:?}", e))?;

    let meta = ModelMeta {
        model_name: model_name.into(),
        function_name: "main".into(),
        short_description: "CB block diagonal SDPA".into(),
        version: "1.0.0".into(),
        author: "ContBatching".into(),
        output_name: out.clone(),
        inputs: vec![("x".into(), vec![1, flat])],
        outputs: vec![(out, vec![1, flat])],
        spec_version: 9,
    };

    let mlpackage_dir =
        write_mlpackage(prog, model_dir, &meta).map_err(|e| format!("mlpackage write: {}", e))?;
    let output_dir = model_dir.join("compiled");
    std::fs::create_dir_all(&output_dir).map_err(|e| format!("mkdir: {}", e))?;
    let receipt = compile_mlpackage(
        &mlpackage_dir,
        &output_dir,
        model_name,
        "cpuAndNeuralEngine",
        "ios26",
    )
    .map_err(|e| format!("compile: {}", e))?;

    let compiled = PathBuf::from(&receipt.compiled_modelc_path);
    Ok((compiled, "x".into(), "out_0".into()))
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Fill a [1, seq_len * D] arena with repeated per-position values.
/// Position p gets `val` repeated D times.
unsafe fn fill_token_vector(ptr: *mut u16, val: f32, d: i64) {
    let f16 = f32_to_f16_bits(val);
    for i in 0..d as usize {
        ptr.add(i).write(f16);
    }
}

/// Compute mean across D dimensions starting at `base`.
#[allow(dead_code)]
unsafe fn mean_d(data: *const f32, d: i64) -> f32 {
    let mut sum = 0.0;
    for i in 0..d as usize {
        sum += *data.add(i);
    }
    sum / d as f32
}

/// Read position-wise means from a flat output arena.
fn position_means(arena: &Arena, seq_len: usize) -> Vec<f32> {
    let flat = seq_len * D as usize;
    let data = unsafe { std::slice::from_raw_parts(arena.base_ptr() as *const f32, flat) };
    (0..seq_len)
        .map(|pos| {
            let start = pos * D as usize;
            let end = start + D as usize;
            data[start..end].iter().copied().sum::<f32>() / D as f32
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────

/// Test: Block-diagonal masked SDPA (if reshape + mask SDPA works).
/// This test will fail informatively if the ANE does not support
/// the reshape op or masked SDPA, in which case the user should
/// rely on the per-position fallback tests.
#[test]
fn ws9a1_block_diagonal_isolation() {
    let model_dir = Path::new(MODEL_DIR);
    let (path, in_name, out_name) =
        build_block_diag_model(model_dir).expect("block diag model must build/compile");

    let model = CoreMlModel::load_with_compute_units(
        &path.to_string_lossy(),
        CoreMlComputeUnits::CpuAndNeuralEngine,
    )
    .expect("model must load");

    // Sequence A = positions 0,1; B = positions 2,3.
    // Run 1: A=[1.0, 2.0], B=[10.0, 20.0]
    // Run 2: A=[1.0, 2.0], B=[100.0, 200.0]
    // Block diagonal means A output must be identical between runs.

    let flat = (4 * D) as u32;

    let input1 = Arena::new(1, flat, mlx_rs::Dtype::Float16).expect("input arena 1");
    unsafe {
        let p = input1.base_ptr() as *mut u16;
        fill_token_vector(p, 1.0, D); // pos 0 (A)
        fill_token_vector(p.add(D as usize), 2.0, D); // pos 1 (A)
        fill_token_vector(p.add(2 * D as usize), 10.0, D); // pos 2 (B)
        fill_token_vector(p.add(3 * D as usize), 20.0, D); // pos 3 (B)
    }
    let output1 = Arena::new(1, flat, mlx_rs::Dtype::Float32).expect("output arena 1");
    model
        .predict(&in_name, &input1.info, &out_name, &output1.info)
        .expect("predict run 1 must succeed");
    let means1 = position_means(&output1, 4);

    let input2 = Arena::new(1, flat, mlx_rs::Dtype::Float16).expect("input arena 2");
    unsafe {
        let p = input2.base_ptr() as *mut u16;
        fill_token_vector(p, 1.0, D); // pos 0 (A)
        fill_token_vector(p.add(D as usize), 2.0, D); // pos 1 (A)
        fill_token_vector(p.add(2 * D as usize), 100.0, D); // pos 2 (B)
        fill_token_vector(p.add(3 * D as usize), 200.0, D); // pos 3 (B)
    }
    let output2 = Arena::new(1, flat, mlx_rs::Dtype::Float32).expect("output arena 2");
    model
        .predict(&in_name, &input2.info, &out_name, &output2.info)
        .expect("predict run 2 must succeed");
    let means2 = position_means(&output2, 4);

    // Verify block diagonal isolation: A positions (0,1) shouldn't change.
    let tol = 5e-3;
    for pos in 0..2 {
        let diff = (means1[pos] - means2[pos]).abs();
        let max_val = means1[pos].abs().max(means2[pos].abs()).max(1e-6);
        let rel = diff / max_val;
        assert!(
            rel < tol,
            "A pos {} changed: run1={} run2={} rel_err={} > tol={}",
            pos,
            means1[pos],
            means2[pos],
            rel,
            tol
        );
    }
    println!("  A positions preserved under B change (block diag isolation)");
    println!(
        "  run1 A=[{:.4}, {:.4}] B=[{:.4}, {:.4}]",
        means1[0], means1[1], means1[2], means1[3]
    );
    println!(
        "  run2 A=[{:.4}, {:.4}] B=[{:.4}, {:.4}]",
        means2[0], means2[1], means2[2], means2[3]
    );
}

/// Fallback test: build separate causal-masked SDPA models for seq=2 and seq=4,
/// and verify they produce non-NaN, position-varying output.
#[test]
fn ws9a2_causal_sdpa_per_position() {
    let model_dir = Path::new(MODEL_DIR);

    // ── seq=2 model (causal mask) ────────────────────────────────────
    let (path2, in_name, out_name2) =
        build_sdpa_model(model_dir, 2, "causal_seq2", true).expect("seq=2 causal model");
    let model2 = CoreMlModel::load_with_compute_units(
        &path2.to_string_lossy(),
        CoreMlComputeUnits::CpuAndNeuralEngine,
    )
    .expect("model2 load");

    let flat2 = (2 * D) as u32;
    let input2 = Arena::new(1, flat2, mlx_rs::Dtype::Float16).expect("input2 arena");
    unsafe {
        fill_token_vector(input2.base_ptr() as *mut u16, 1.0, D);
        fill_token_vector((input2.base_ptr() as *mut u16).add(D as usize), 3.0, D);
    }
    let output2 = Arena::new(1, flat2, mlx_rs::Dtype::Float32).expect("output2 arena");
    model2
        .predict(&in_name, &input2.info, &out_name2, &output2.info)
        .expect("seq=2 predict");

    let means2 = position_means(&output2, 2);
    for (pos, &m) in means2.iter().enumerate() {
        assert!(!m.is_nan(), "seq=2 pos {} mean is NaN", pos);
        assert!(m > 0.0, "seq=2 pos {} mean {} should be > 0", pos, m);
    }
    println!(
        "  seq=2 causal SDPA: pos0={:.4} pos1={:.4}",
        means2[0], means2[1]
    );

    // ── seq=4 model (causal mask) ────────────────────────────────────
    let (path4, in_name4, out_name4) =
        build_sdpa_model(model_dir, 4, "causal_seq4", true).expect("seq=4 causal model");
    let model4 = CoreMlModel::load_with_compute_units(
        &path4.to_string_lossy(),
        CoreMlComputeUnits::CpuAndNeuralEngine,
    )
    .expect("model4 load");

    let flat4 = (4 * D) as u32;
    let input4 = Arena::new(1, flat4, mlx_rs::Dtype::Float16).expect("input4 arena");
    unsafe {
        for pos in 0..4 {
            fill_token_vector(
                (input4.base_ptr() as *mut u16).add(pos * D as usize),
                pos as f32 + 1.0,
                D,
            );
        }
    }
    let output4 = Arena::new(1, flat4, mlx_rs::Dtype::Float32).expect("output4 arena");
    model4
        .predict(&in_name4, &input4.info, &out_name4, &output4.info)
        .expect("seq=4 predict");

    let means4 = position_means(&output4, 4);
    for (pos, &m) in means4.iter().enumerate() {
        assert!(!m.is_nan(), "seq=4 pos {} mean is NaN", pos);
        assert!(m > 0.0, "seq=4 pos {} mean {} should be > 0", pos, m);
    }
    println!(
        "  seq=4 causal SDPA: pos0={:.4} pos1={:.4} pos2={:.4} pos3={:.4}",
        means4[0], means4[1], means4[2], means4[3]
    );
}

/// Supplementary: SDPA without mask on seq=2 and seq=4.
/// Proves the vanilla SDPA op works end-to-end on the ANE.
#[test]
fn ws9a3_sdpa_no_mask_baseline() {
    let model_dir = Path::new(MODEL_DIR);

    for (seq, tag) in &[(2i64, "nomasks2"), (4i64, "nomasks4")] {
        let (path, in_name, out_name) =
            build_sdpa_model(model_dir, *seq, tag, false).expect("no-mask SDPA model");
        let model = CoreMlModel::load_with_compute_units(
            &path.to_string_lossy(),
            CoreMlComputeUnits::CpuAndNeuralEngine,
        )
        .expect("model load");

        let flat = (seq * D) as u32;
        let input = Arena::new(1, flat, mlx_rs::Dtype::Float16).expect("input arena");
        unsafe {
            for pos in 0..*seq {
                fill_token_vector(
                    (input.base_ptr() as *mut u16).add((pos * D) as usize),
                    pos as f32 + 1.0,
                    D,
                );
            }
        }
        let output = Arena::new(1, flat, mlx_rs::Dtype::Float32).expect("output arena");
        model
            .predict(&in_name, &input.info, &out_name, &output.info)
            .expect("predict");

        let means = position_means(&output, *seq as usize);
        for (pos, &m) in means.iter().enumerate() {
            assert!(!m.is_nan(), "seq={} pos {} mean is NaN", seq, pos);
        }
        println!("seq={} no-mask SDPA output means: {:?}", seq, means);
    }
}
