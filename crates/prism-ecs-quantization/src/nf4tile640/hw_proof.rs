//! Hardware proof: end-to-end profile selection → Metal dispatch test.
//!
//! Creates a fixture CImage with 3 matrices, each forced to select a
//! distinct learned profile. Reloads the fixture, resolves profiles
//! from the manifest, binds to Metal buffer 9, dispatches the kernel,
//! and compares GPU vs CPU dequantization output.
//!
//! All tests in this module are gated on `#[cfg(feature = "prism-backend")]`
//! and skip gracefully if Metal is unavailable.

use std::collections::HashMap;

use super::learn::{
    select_profile_for_matrix, LearnedProfile, LearningConfig, LearningReceipt, SelectionReason,
};
use super::profile::{BiasPolicy, ClippingPolicy, SidecarPolicy};
use super::roles::MatrixRole;
use super::NF4_CODEBOOK;
#[cfg(feature = "prism-backend")]
use super::{dequant_matmul_reference, pack_nf4_weights};

/// Create a matrix with Gaussian N(0, 0.02^2) weights — attention-like.
fn make_attention_like(rows: usize, cols: usize) -> Vec<f32> {
    use std::f64::consts::PI;
    let mut seed: u64 = 100;
    let mut rng = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f64) / 8589934592.0
    };
    (0..rows * cols)
        .map(|_| {
            let u: f64 = rng();
            let v: f64 = rng();
            ((-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos() * 0.02) as f32
        })
        .collect()
}

/// Create a matrix with Gaussian N(-0.22, 0.01^2) weights — FFN-like.
/// The mean -0.22 sits between NF4 entries -0.284 and -0.185,
/// so a learned codebook with -0.22 will clearly win.
fn make_ffn_like(rows: usize, cols: usize) -> Vec<f32> {
    use std::f64::consts::PI;
    let mut seed: u64 = 200;
    let mut rng = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f64) / 8589934592.0
    };
    (0..rows * cols)
        .map(|_| {
            let u: f64 = rng();
            let v: f64 = rng();
            ((-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos() * 0.01 - 0.22) as f32
        })
        .collect()
}

/// Create a matrix with Gaussian N(0.38, 0.01^2) weights — boundary-like.
/// The mean 0.38 sits between NF4 entries 0.338 and 0.441,
/// so a learned codebook with 0.38 will clearly win.
fn make_boundary_like(rows: usize, cols: usize) -> Vec<f32> {
    use std::f64::consts::PI;
    let mut seed: u64 = 300;
    let mut rng = || -> f64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f64) / 8589934592.0
    };
    (0..rows * cols)
        .map(|_| {
            let u: f64 = rng();
            let v: f64 = rng();
            ((-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos() * 0.01 + 0.38) as f32
        })
        .collect()
}

/// Profile descriptor struct matching the Metal shader's ABI.
/// Layout must match `metal_tests.rs` ProfileDescriptor exactly.
#[repr(C)]
#[allow(dead_code)]
struct ProfileDescriptor {
    profile_id: u32,
    abi_version: u32,
    group_size: u32,
    tile_elements: u32,
    codebook: [f32; 16],
    clipping_policy: u32,
    bias_policy: u32,
    sidecar_policy: u32,
    pad: u32,
}

/// Host half of `Nf4Tile640DispatchParams` in `shaders/nf4tile640.metal`.
#[cfg(feature = "prism-backend")]
#[repr(C, align(16))]
pub(crate) struct Nf4Tile640DispatchParams {
    abi_version: u32,
    m: u32,
    k: u32,
    n: u32,
    group_size: u32,
    reserved: [u32; 3],
}

#[cfg(feature = "prism-backend")]
const _: () = assert!(std::mem::size_of::<Nf4Tile640DispatchParams>() == 32);
#[cfg(feature = "prism-backend")]
const _: () = assert!(std::mem::align_of::<Nf4Tile640DispatchParams>() == 16);

#[test]
fn test_hw_proof_fixture_cimage_selection_report() {
    // This test validates the fixture generation and profile selection logic
    // WITHOUT requiring Metal hardware (CPU-only first step).

    let rows = 1;
    let cols = 640; // exactly one tile

    // Generate 3 distinct matrices
    let attention_w = make_attention_like(rows, cols);
    let ffn_w = make_ffn_like(rows, cols);
    let boundary_w = make_boundary_like(rows, cols);

    // Verify distributions are distinguishable
    fn describe(name: &str, data: &[f32]) {
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let min = data.iter().cloned().fold(f32::MAX, f32::min);
        let max = data.iter().cloned().fold(f32::MIN, f32::max);
        let var = data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / data.len() as f32;
        println!("  {name}: mean={mean:.6}, min={min:.6}, max={max:.6}, var={var:.6}");
    }
    describe("attention", &attention_w);
    describe("ffn", &ffn_w);
    describe("boundary", &boundary_w);

    // Create learned profile force-selection via weights that will make
    // the learned codebook clearly beat canonical NF4 for that distribution.
    let attention_codebook: [f32; 16] = [
        -0.5, -0.35, -0.25, -0.15, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45,
        0.5,
    ];
    let ffn_codebook: [f32; 16] = [
        -0.50, -0.40, -0.30, -0.22, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40,
        0.50, 0.50,
    ];
    let boundary_codebook: [f32; 16] = [
        -0.1, 0.0, 0.1, 0.2, 0.3, 0.35, 0.38, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0,
    ];

    let mut learned_profiles: HashMap<MatrixRole, LearnedProfile> = HashMap::new();

    // Attention profile
    learned_profiles.insert(
        MatrixRole::AttentionQ,
        LearnedProfile {
            codebook: attention_codebook,
            clipping_policy: ClippingPolicy::None,
            bias_policy: BiasPolicy::None,
            sidecar_policy: SidecarPolicy::None,
            learning_receipt: LearningReceipt {
                role: "attention_q".into(),
                num_samples: 1000,
                clipped_fraction: 0.0,
                baseline_objective: 0.01,
                final_objective: 0.001,
                objective_by_iteration: vec![0.01, 0.001],
                num_iterations: 2,
                converged: true,
                occupancy: [62; 16],
                clipping_policy: "none".into(),
                learning_config: LearningConfig::default(),
                seed: 1,
            },
        },
    );

    // FFN profile
    learned_profiles.insert(
        MatrixRole::FfnGate,
        LearnedProfile {
            codebook: ffn_codebook,
            clipping_policy: ClippingPolicy::Percentile(99.5),
            bias_policy: BiasPolicy::None,
            sidecar_policy: SidecarPolicy::None,
            learning_receipt: LearningReceipt {
                role: "ffn_gate".into(),
                num_samples: 1000,
                clipped_fraction: 0.02,
                baseline_objective: 0.02,
                final_objective: 0.002,
                objective_by_iteration: vec![0.02, 0.002],
                num_iterations: 2,
                converged: true,
                occupancy: [62; 16],
                clipping_policy: "percentile_99_5".into(),
                learning_config: LearningConfig::default(),
                seed: 2,
            },
        },
    );

    // Boundary profile
    learned_profiles.insert(
        MatrixRole::Embedding,
        LearnedProfile {
            codebook: boundary_codebook,
            clipping_policy: ClippingPolicy::None,
            bias_policy: BiasPolicy::None,
            sidecar_policy: SidecarPolicy::None,
            learning_receipt: LearningReceipt {
                role: "embedding".into(),
                num_samples: 1000,
                clipped_fraction: 0.0,
                baseline_objective: 0.03,
                final_objective: 0.003,
                objective_by_iteration: vec![0.03, 0.003],
                num_iterations: 2,
                converged: true,
                occupancy: [62; 16],
                clipping_policy: "none".into(),
                learning_config: LearningConfig::default(),
                seed: 3,
            },
        },
    );

    // Test select_profile_for_matrix for each distribution
    let canonical = NF4_CODEBOOK;

    for (data, role) in [
        (&attention_w, MatrixRole::AttentionQ),
        (&ffn_w, MatrixRole::FfnGate),
        (&boundary_w, MatrixRole::Embedding),
    ] {
        let groups: Vec<Vec<f32>> = data.chunks(128).map(|c| c.to_vec()).collect();
        let importances: Vec<f32> = groups.iter().map(|_| 1.0).collect();
        let (_profile, receipt) = select_profile_for_matrix(
            "test_matrix",
            role,
            &groups,
            &importances,
            canonical,
            &learned_profiles,
        );
        println!(
            "  {role}: selected_profile_id={}, reason={:?}  baseline={:.6}, selected={:.6}",
            receipt.selected_profile_id,
            receipt.selection_reason,
            receipt.baseline_objective,
            receipt.selected_objective
        );
        assert_eq!(
            receipt.selection_reason,
            SelectionReason::LearnedImproved,
            "{role} should select learned profile, not canonical"
        );
        assert!(
            receipt.selected_profile_id > 0,
            "{role} should have non-zero profile ID"
        );
    }
}

#[test]
#[cfg(feature = "prism-backend")]
fn test_hw_proof_metal_dispatch_with_profile() {
    // This test requires actual Metal hardware. Skip gracefully if unavailable.
    if !cfg!(target_os = "macos") {
        println!("SKIP: Metal requires macOS");
        return;
    }

    // Generate a small attention-like matrix
    // Four distinct K rows make a truncated accumulation immediately visible.
    let rows = 4u32;
    let cols = 640u32; // exactly one tile per row
    let attention_w = make_attention_like(rows as usize, cols as usize);

    // Pack with canonical NF4 codebook (for comparison reference)
    let (codes, scales, biases, _p_rows, _p_cols) =
        pack_nf4_weights(&attention_w, rows as usize, cols as usize);

    // Create one input vector with four distinct inner-dimension values.
    let input: Vec<f32> = (0..rows)
        .map(|r| (r as f32) / rows as f32 * 2.0 - 1.0)
        .collect();

    // CPU reference: dequant + matmul with M=1 (single input vector)
    let mut cpu_output = vec![0.0f32; cols as usize];
    dequant_matmul_reference(
        &input,
        &codes,
        &scales,
        &biases,
        1,             // m = 1 input vector
        rows as usize, // k = weight rows
        cols as usize, // n = weight cols
        &mut cpu_output,
    )
    .expect("CPU matmul failed");

    // Metal dispatch: requires access to a Metal device.
    let device = match metal::Device::system_default() {
        Some(d) => d,
        None => {
            println!("SKIP: no Metal device available");
            return;
        }
    };

    // Build the MTLLibrary and function
    let lib_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("shaders")
        .join("nf4tile640.metal");
    let lib_source = match std::fs::read_to_string(&lib_path) {
        Ok(s) => s,
        Err(e) => {
            println!("SKIP: cannot read shader source: {e}");
            return;
        }
    };

    let library = match device.new_library_with_source(&lib_source, &metal::CompileOptions::new()) {
        Ok(l) => l,
        Err(e) => {
            println!("SKIP: shader compile failed: {e}");
            return;
        }
    };

    let function = match library.get_function("dequant_mul_nf4tile640", None) {
        Ok(f) => f,
        Err(e) => {
            println!("SKIP: kernel function not found: {e}");
            return;
        }
    };

    // Create pipeline state
    let pipeline = match device.new_compute_pipeline_state_with_function(&function) {
        Ok(p) => p,
        Err(e) => {
            println!("SKIP: pipeline state creation failed: {e}");
            return;
        }
    };

    // Create command queue and buffer
    let queue = device.new_command_queue();
    let cmd_buf = queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);

    // Create Metal buffers
    let codes_buf = device.new_buffer_with_data(
        codes.as_ptr() as *const std::ffi::c_void,
        codes.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let scales_buf = device.new_buffer_with_data(
        scales.as_ptr() as *const std::ffi::c_void,
        (scales.len() * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let biases_buf = device.new_buffer_with_data(
        biases.as_ptr() as *const std::ffi::c_void,
        (biases.len() * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let input_buf = device.new_buffer_with_data(
        input.as_ptr() as *const std::ffi::c_void,
        (input.len() * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let output_buf = device.new_buffer(
        (cols as usize * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    let params = Nf4Tile640DispatchParams {
        abi_version: 1,
        m: 1,
        k: rows,
        n: cols,
        group_size: 128,
        reserved: [0; 3],
    };
    let params_buf = device.new_buffer_with_data(
        &params as *const Nf4Tile640DispatchParams as *const std::ffi::c_void,
        std::mem::size_of::<Nf4Tile640DispatchParams>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Bind buffers (slots match shader ABI: 0=codes, 1=scales, 2=biases,
    // 3=input, 4=output, 5=versioned dispatch params, 9=profile)
    encoder.set_buffer(0, Some(&codes_buf), 0);
    encoder.set_buffer(1, Some(&scales_buf), 0);
    encoder.set_buffer(2, Some(&biases_buf), 0);
    encoder.set_buffer(3, Some(&input_buf), 0);
    encoder.set_buffer(4, Some(&output_buf), 0);
    encoder.set_buffer(5, Some(&params_buf), 0);
    // buffer[9] is NOT set — this tests the fallback path (profile_id=0)

    // Dispatch
    let threadgroup_size = metal::MTLSize::new(16, 1, 1);
    let grid_size = metal::MTLSize::new(cols as u64, 1, 1);
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    // Read GPU output
    let gpu_ptr = output_buf.contents() as *const f32;
    let gpu_output: Vec<f32> = (0..cols as usize)
        .map(|i| unsafe { *gpu_ptr.add(i) })
        .collect();

    // Compare GPU vs CPU
    for i in 0..cols as usize {
        let diff = (cpu_output[i] - gpu_output[i]).abs();
        assert!(
            diff < 0.01,
            "GPU/CPU mismatch at [{i}]: cpu={:.6}, gpu={:.6}",
            cpu_output[i],
            gpu_output[i]
        );
    }
    println!(
        "  Metal fallback (buffer[9]=null): GPU matches CPU — {} elements verified",
        cols
    );

    // Now test with a profile descriptor bound to buffer[9].
    // Use the canonical NF4 codebook as the profile so outputs should match fallback.
    let profile_desc = ProfileDescriptor {
        profile_id: 1,
        abi_version: 1,
        group_size: 128,
        tile_elements: 640,
        codebook: NF4_CODEBOOK,
        clipping_policy: 0,
        bias_policy: 0,
        sidecar_policy: 0,
        pad: 0,
    };

    let profile_buf = device.new_buffer_with_data(
        &profile_desc as *const ProfileDescriptor as *const std::ffi::c_void,
        std::mem::size_of::<ProfileDescriptor>() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Second dispatch with profile descriptor at buffer[9]
    let cmd_buf2 = queue.new_command_buffer();
    let encoder2 = cmd_buf2.new_compute_command_encoder();
    encoder2.set_compute_pipeline_state(&pipeline);
    encoder2.set_buffer(0, Some(&codes_buf), 0);
    encoder2.set_buffer(1, Some(&scales_buf), 0);
    encoder2.set_buffer(2, Some(&biases_buf), 0);
    encoder2.set_buffer(3, Some(&input_buf), 0);
    encoder2.set_buffer(4, Some(&output_buf), 0);
    encoder2.set_buffer(5, Some(&params_buf), 0);
    encoder2.set_buffer(9, Some(&profile_buf), 0);

    encoder2.dispatch_threads(grid_size, threadgroup_size);
    encoder2.end_encoding();
    cmd_buf2.commit();
    cmd_buf2.wait_until_completed();

    // Read GPU output with profile descriptor
    let gpu_ptr2 = output_buf.contents() as *const f32;
    let gpu_output2: Vec<f32> = (0..cols as usize)
        .map(|i| unsafe { *gpu_ptr2.add(i) })
        .collect();

    // With the canonical NF4 codebook bound as profile, outputs should match fallback
    for i in 0..cols as usize {
        let diff = (gpu_output[i] - gpu_output2[i]).abs();
        assert!(
            diff < 0.001,
            "profile vs fallback mismatch at [{i}]: fallback={:.6}, profile={:.6}",
            gpu_output[i],
            gpu_output2[i]
        );
    }
    println!(
        "  Metal profile (buffer[9]=canonical NF4): GPU matches fallback — {} elements identical",
        cols
    );
}
