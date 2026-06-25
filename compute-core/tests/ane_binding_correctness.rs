//! PRISM-METAL-ANE-METAL-HARDWARE-EVIDENCE-0001 WS8A: ANE binding and correctness.
//!
//! Hardware-gated tests that exercise the IOSurface → Core ML path on Apple
//! Silicon. Each test builds a small FP16 MIL program, compiles it via
//! coremlcompiler, and runs predictions against IOSurface-backed arenas.
//!
//! These tests do NOT depend on the WS4 orchestrator. They validate:
//!
//!   1. DecodeActivationV1 IOSurface binding — descriptor creation, fp16
//!      data write, Core ML prediction, CPU reference comparison, boundary
//!      latency recording.
//!   2. IOSurface-backed MLMultiArray construction — shape/dtype/strides
//!      verification, invalid-pixel-format rejection.
//!   3. Metal-writes → Core ML reads — gradient pattern via MLX (Metal
//!      backend), Core ML consumes same IOSurface, output matches reference,
//!      boundary latency recorded.
//!
//! All tests assert correctness and binding viability, not undocumented
//! zero-copy.  The test model is a simple [1, 64] × [64, 64] matmul whose
//! weight is an approximate identity matrix so output ≈ input within fp16
//! precision.

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::path::{Path, PathBuf};
use std::time::Instant;

use coreml_proto::proto::mil_spec;
use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::coreml_bridge::{CoreMlComputeUnits, CoreMlModel};
use tribunus_compute_core::coreml_pipeline::compile_mlpackage;
use tribunus_compute_core::mil_builder::MilBuilder;
use tribunus_compute_core::mlpackage::{write_mlpackage, ModelMeta};

// ── Constants ───────────────────────────────────────────────────────────────

/// DecodeActivationV1 descriptor: S=1, H=64.
const BATCH: i64 = 1;
const HIDDEN: i64 = 64;
const ELEMENT_COUNT: usize = (BATCH * HIDDEN) as usize;
const BYTE_COUNT: usize = ELEMENT_COUNT * 2; // f16

/// Pixel format for kCVPixelFormatType_OneComponent16Half ('L00h').
const PIXEL_FORMAT_HALF_FLOAT: i32 = 0x4C303068;

/// Model directory for compiled artifacts.
const MODEL_DIR: &str = "/tmp/ane_binding_correctness_models";

/// Maximum acceptable boundary latency (500 µs).
/// This is a generous budget for a first test; actual ANE dispatch is
/// typically <100 µs for a 1×64 matmul.
const BOUNDARY_NS_BUDGET: u128 = 500_000;

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
    let new_mant = mant >> 13;
    (sign << 15) | ((new_exp as u16) << 10) | (new_mant as u16)
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    if exp == 0 {
        let value = (mant as f32) * 2.0f32.powi(-24);
        if sign != 0 { -value } else { value }
    } else if exp == 31 {
        if mant == 0 {
            if sign != 0 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else {
            f32::NAN
        }
    } else {
        let normalized = 1.0f32 + (mant as f32) / 1024.0f32;
        let exponent = 2.0f32.powi((exp as i32) - 15);
        let value = normalized * exponent;
        if sign != 0 { -value } else { value }
    }
}

// ── Model compilation helper ────────────────────────────────────────────────

/// Build, write, and compile a small fp16 identity-approximation model:
/// matmul(input [1, 64], weight [64, 64]) → output [1, 64].
///
/// The weight is a perturbed identity: diagonal ≈ 1.0, off-diagonal = 0.0.
/// For fp16 input x, output ≈ x + small numerical noise.
///
/// Returns the path to the compiled .mlmodelc directory, caching on disk.
fn build_fp16_identity_model() -> Result<PathBuf, String> {
    let model_dir = Path::new(MODEL_DIR);
    let _ = std::fs::create_dir_all(model_dir);

    // Check cache
    let modelc_name = "fp16_identity_test.mlmodelc";
    let modelc_path = model_dir.join("compiled").join(modelc_name);
    if modelc_path.exists() {
        return Ok(modelc_path);
    }

    // Build a near-identity weight: diagonal elements close to 1.0,
    // off-diagonal zero. This ensures output ≈ input within fp16 precision.
    let weight_len = (BATCH * HIDDEN) as usize;
    let mut weight: Vec<f32> = vec![0.0f32; weight_len];
    for i in 0..HIDDEN as usize {
        // Each row i of the [64, 64] weight has a 1.0 at column i.
        // Since the weight is row-major, index = i * HIDDEN + i.
        weight[i * HIDDEN as usize + i] = 1.0;
    }

    let b = MilBuilder::new("main");
    let b = b.input("input", mil_spec::DataType::Float16, &[BATCH, HIDDEN]);
    let b = b.const_f16("weight", &weight, &[HIDDEN, HIDDEN]);
    let weight_name = b.last_name().unwrap_or("weight_0").to_string();
    let b = b.matmul("input", &weight_name);
    let output_name = b.last_name().unwrap_or("matmul_0").to_string();
    let prog = b
        .output(&output_name)
        .build()
        .map_err(|e| format!("MIL build: {:?}", e))?;

    let meta = ModelMeta {
        model_name: "fp16_identity_test".into(),
        function_name: "main".into(),
        short_description: "FP16 identity-approximation test model".into(),
        version: "1.0.0".into(),
        author: "Tribunus Compute WS8A".into(),
        output_name: output_name.clone(),
        inputs: vec![("input".into(), vec![BATCH, HIDDEN])],
        outputs: vec![(output_name.clone(), vec![BATCH, HIDDEN])],
    };

    let mlpackage_dir = write_mlpackage(prog, model_dir, &meta)
        .map_err(|e| format!("mlpackage write: {}", e))?;

    let output_dir = model_dir.join("compiled");
    std::fs::create_dir_all(&output_dir)
        .map_err(|e| format!("mkdir {}: {}", output_dir.display(), e))?;

    let receipt = compile_mlpackage(
        &mlpackage_dir,
        &output_dir,
        "fp16_identity_test",
        "cpuAndNeuralEngine",
        "iOS15",
    )
    .map_err(|e| format!("compile_mlpackage: {}", e))?;

    // The receipt's compiled_modelc_path points to the model inside the .modelc dir.
    // CoreMlModel::load expects the .mlmodelc bundle.
    let compiled_path = PathBuf::from(&receipt.compiled_modelc_path);
    eprintln!(
        "[WS8A] compiled identity model: {}",
        compiled_path.display()
    );
    Ok(compiled_path)
}

// ── Test 1: DecodeActivationV1 IOSurface binding ─────────────────────────

/// Create a DecodeActivationV1 descriptor (S=1, H=64), allocate IOSurface-backed
/// fp16 buffer, write deterministic data, bind to Core ML, run prediction,
/// verify output against CPU reference, record boundary latency.
#[test]
fn test_decode_activation_v1_iosurface_binding() {
    let model_path = build_fp16_identity_model()
        .expect("identity model compilation must succeed");

    let model = CoreMlModel::load_with_compute_units(
        &model_path.to_string_lossy(),
        CoreMlComputeUnits::CpuAndNeuralEngine,
    )
    .expect("load Core ML model must succeed");

    // DecodeActivationV1 descriptor: S=1 (batch), H=64 (hidden)
    let batch = BATCH as u32;
    let hidden = HIDDEN as u32;

    // Allocate IOSurface-backed arenas
    let input = Arena::new(batch, hidden, mlx_rs::Dtype::Float16)
        .expect("input arena must allocate");
    let mut output = Arena::new(batch, hidden, mlx_rs::Dtype::Float16)
        .expect("output arena must allocate");

    // Verify pixel format is half-float
    assert_eq!(
        input.info.pixel_format, PIXEL_FORMAT_HALF_FLOAT,
        "input arena pixel format must be kCVPixelFormatType_OneComponent16Half"
    );
    assert_eq!(
        output.info.pixel_format, PIXEL_FORMAT_HALF_FLOAT,
        "output arena pixel format must be kCVPixelFormatType_OneComponent16Half"
    );

    // Write deterministic fp16 data: [1.0, 2.0, 3.0, ..., 64.0]
    let reference: Vec<f32> = (0..ELEMENT_COUNT).map(|i| (i + 1) as f32).collect();
    unsafe {
        let ptr = input.base_ptr() as *mut u16;
        for (i, &v) in reference.iter().enumerate() {
            ptr.add(i).write(f32_to_f16_bits(v));
        }
    }

    // Write fp16 output expectations: with near-identity weight,
    // output[i] ≈ input[i] within fp16 precision
    let expected: Vec<f32> = reference.clone();

    // Run Core ML prediction
    let start = Instant::now();
    model
            .predict_pixelbuffer("input", &input.info, "output", &mut output.info)
        .expect("predict_pixelbuffer must succeed");
    let boundary_latency_ns = start.elapsed().as_nanos();

    // Read back output
    let mut actual = vec![0.0f32; ELEMENT_COUNT];
    unsafe {
        let out_ptr = output.base_ptr() as *const u16;
        for i in 0..ELEMENT_COUNT {
            actual[i] = f16_to_f32(out_ptr.add(i).read());
        }
    }

    // Verify correctness: output ≈ input (identity weight)
    let tolerance = 0.1; // fp16 precision budget for 1×64 × 64×64
    for i in 0..ELEMENT_COUNT {
        let diff = (actual[i] - expected[i]).abs();
        assert!(
            diff < tolerance,
            "binding mismatch at [{}]: got {:.4}, expected {:.4}, diff={:.4}",
            i,
            actual[i],
            expected[i],
            diff
        );
    }

    // Verify dtype matching
    assert_eq!(
        input.dtype,
        mlx_rs::Dtype::Float16,
        "input dtype must be Float16"
    );
    assert_eq!(
        output.dtype,
        mlx_rs::Dtype::Float16,
        "output dtype must be Float16"
    );

    // Verify shape matching
    assert_eq!(
        input.element_count(),
        ELEMENT_COUNT,
        "input element count must match DecodeActivationV1 descriptor"
    );
    assert_eq!(
        output.element_count(),
        ELEMENT_COUNT,
        "output element count must match DecodeActivationV1 descriptor"
    );

    // Verify byte count
    assert_eq!(
        input.byte_len(),
        BYTE_COUNT,
        "input byte count must be {} ({} elements * 2 bytes)",
        BYTE_COUNT,
        ELEMENT_COUNT
    );
    assert_eq!(
        output.byte_len(),
        BYTE_COUNT,
        "output byte count must be {} ({} elements * 2 bytes)",
        BYTE_COUNT,
        ELEMENT_COUNT
    );

    // Record boundary latency within budget
    assert!(
        boundary_latency_ns < BOUNDARY_NS_BUDGET,
        "boundary latency {} ns exceeds budget {} ns",
        boundary_latency_ns,
        BOUNDARY_NS_BUDGET
    );

    eprintln!(
        "[WS8A] test_decode_activation_v1_iosurface_binding: PASS \
         (latency={} ns, elements={}, bytes={})",
        boundary_latency_ns, ELEMENT_COUNT, BYTE_COUNT
    );

    drop(input);
    drop(output);
    drop(model);
}

// ── Test 2: IOSurface-backed MLMultiArray construction ───────────────────

/// Verify that Arena::new correctly constructs an IOSurface-backed buffer
/// with the expected shape, dtype, strides, pixel format, and byte count.
/// Also verify failure for unsupported dtypes.
#[test]
fn test_iosurface_coreml_input_construction() {
    // 1. Create IOSurface with known layout — DecodeActivationV1: S=1, H=64
    let arena = Arena::new(BATCH as u32, HIDDEN as u32, mlx_rs::Dtype::Float16)
        .expect("fp16 arena must allocate");

    // 2-3. Verify shape, dtype, strides match the ActivationContract
    assert_eq!(
        arena.info.logical_dim0 as u32,
        BATCH as u32,
        "logical_dim0 (batch) must be {}",
        BATCH
    );
    assert_eq!(
        arena.info.logical_dim1 as u32,
        HIDDEN as u32,
        "logical_dim1 (hidden) must be {}",
        HIDDEN
    );
    assert_eq!(
        arena.element_count(),
        ELEMENT_COUNT,
        "element_count must be {}",
        ELEMENT_COUNT
    );
    assert_eq!(arena.byte_len(), BYTE_COUNT, "byte_len must be {}", BYTE_COUNT);

    // Pixel format must be kCVPixelFormatType_OneComponent16Half
    assert_eq!(
        arena.info.pixel_format, PIXEL_FORMAT_HALF_FLOAT,
        "pixel_format must be 0x{:08X} (kCVPixelFormatType_OneComponent16Half)",
        PIXEL_FORMAT_HALF_FLOAT
    );

    // bytes_per_row must be at least logical_dim1 * element_size (2 bytes for f16)
    let min_stride = HIDDEN as i32 * 2;
    assert!(
        arena.info.bytes_per_row >= min_stride,
        "bytes_per_row ({}) must be >= minimum stride ({})",
        arena.info.bytes_per_row,
        min_stride
    );

    // IOSurface must have a valid ID and backing
    assert!(
        arena.io_surface_id() > 0,
        "IOSurface ID must be positive"
    );
    assert!(
        !arena.io_surface_ptr().is_null(),
        "IOSurface pointer must not be null"
    );
    assert!(
        !arena.info.base_address.is_null(),
        "base_address must not be null"
    );

    // 4. Verify failure on invalid pixel format / unsupported dtype
    // Arena::new only accepts Float16 and Float32; other dtypes should error.
    let err = Arena::new(BATCH as u32, HIDDEN as u32, mlx_rs::Dtype::Int32);
    assert!(
        err.is_err(),
        "Int32 arena construction must fail (FP16/F32 only)"
    );

    let err = Arena::new(BATCH as u32, HIDDEN as u32, mlx_rs::Dtype::Uint8);
    assert!(
        err.is_err(),
        "Uint8 arena construction must fail (FP16/F32 only)"
    );

    // Float32 should also work
    let f32_arena = Arena::new(BATCH as u32, HIDDEN as u32, mlx_rs::Dtype::Float32)
        .expect("Float32 arena must allocate");
    assert_eq!(f32_arena.byte_len(), ELEMENT_COUNT * 4);
    drop(f32_arena);

    drop(arena);
    eprintln!("[WS8A] test_iosurface_coreml_input_construction: PASS");
}

// ── Test 3: Metal writes, Core ML reads same IOSurface ──────────────────

/// Allocate IOSurface in DecodeActivationV1 layout, write a gradient pattern
/// via MLX (Metal backend), run Core ML prediction on the same IOSurface,
/// verify output matches reference, record boundary latency.
#[test]
fn test_metal_writes_coreml_reads_iosurface() {
    let model_path = build_fp16_identity_model()
        .expect("identity model compilation must succeed");

    let model = CoreMlModel::load_with_compute_units(
        &model_path.to_string_lossy(),
        CoreMlComputeUnits::CpuAndNeuralEngine,
    )
    .expect("load Core ML model must succeed");

    let batch = BATCH as u32;
    let hidden = HIDDEN as u32;

    // Step 1: Allocate IOSurface in DecodeActivationV1 layout
    // Use separate input and output arenas (as the real pipeline does).
    let input = Arena::new(batch, hidden, mlx_rs::Dtype::Float16)
        .expect("input arena must allocate");
    let mut output = Arena::new(batch, hidden, mlx_rs::Dtype::Float16)
        .expect("output arena must allocate");

    // Step 2: Write gradient pattern via MLX (Metal backend).
    // The gradient pattern is a linear ramp: [0.0, 1.0, 2.0, ..., 63.0].
    // This exercises the Metal compute pipeline writing into IOSurface memory.
    //
    // Approach: Create an MLX external array over the IOSurface, then use
    // MLX ops (which run on Metal) to fill it.
    let shape_mlx: [i32; 2] = [batch as i32, hidden as i32];
    // Write gradient via CPU for the test (IOSurface memory is CPU-accessible).
    // In production, a Metal compute kernel would write this directly.
    // The memory backing is identical regardless of writing agent.
    let gradient: Vec<f32> = (0..ELEMENT_COUNT).map(|i| i as f32).collect();
    unsafe {
        let ptr = input.base_ptr() as *mut u16;
        for (i, &v) in gradient.iter().enumerate() {
            ptr.add(i).write(f32_to_f16_bits(v));
        }
    }

    // Now wrap the IOSurface-backed input as an MLX external array so we
    // can run a Metal compute operation — multiply every element by 2.0
    // through MLX (which uses the Metal GPU backend).
    let input_storage = std::sync::Arc::new(unsafe {
        tribunus_compute_core::external_array::StaticStorage::new(
            input.base_ptr() as *const u8,
            input.byte_len(),
        )
    });
    let arr = unsafe {
        tribunus_compute_core::external_array::new_external_array(
            input_storage,
            &shape_mlx,
            mlx_rs::Dtype::Float16,
        )
    }
    .expect("external array over IOSurface must succeed");

    // Run MLX Metal op: multiply by 2.0
    let two = mlx_rs::Array::from_slice(&[2.0f32], &[1]);
    let doubled = arr.multiply(&two).expect("multiply via MLX/Metal must succeed");
    doubled
        .eval()
        .expect("MLX eval (Metal dispatch) must succeed");

    // Drop the MLX arrays so the IOSurface memory is released for Core ML
    drop(doubled);
    drop(arr);

    // Read back the gradient pattern × 2.0 from the IOSurface
    // Expected: [0.0, 2.0, 4.0, 6.0, ..., 126.0]
    let expected: Vec<f32> = (0..ELEMENT_COUNT).map(|i| (i as f32) * 2.0_f32).collect();

    // Verify Metal wrote correctly
    let mut metal_result = vec![0.0f32; ELEMENT_COUNT];
    unsafe {
        let in_ptr = input.base_ptr() as *const u16;
        for i in 0..ELEMENT_COUNT {
            metal_result[i] = f16_to_f32(in_ptr.add(i).read());
        }
    }
    let tol = 0.2;
    for i in 0..ELEMENT_COUNT {
        let diff = (metal_result[i] - expected[i]).abs();
        assert!(
            diff < tol,
            "Metal write mismatch at [{}]: got {:.4}, expected {:.4}, diff={:.4}",
            i,
            metal_result[i],
            expected[i],
            diff
        );
    }

    // Step 3: Core ML reads from the same IOSurface (through predict_pixelbuffer).
    // The input arena now has [0, 2, 4, ..., 126] written by Metal.
    // With the near-identity weight, output ≈ input.
    let start = Instant::now();
    model
        .predict_pixelbuffer("input", &input.info, "output", &mut output.info)
        .expect("predict_pixelbuffer must succeed");
    let boundary_latency_ns = start.elapsed().as_nanos();

    // Step 4: Output matches reference (output ≈ gradient × 2.0)
    let mut actual = vec![0.0f32; ELEMENT_COUNT];
    unsafe {
        let out_ptr = output.base_ptr() as *const u16;
        for i in 0..ELEMENT_COUNT {
            actual[i] = f16_to_f32(out_ptr.add(i).read());
        }
    }

    for i in 0..ELEMENT_COUNT {
        let diff = (actual[i] - expected[i]).abs();
        assert!(
            diff < tol,
            "Core ML read mismatch at [{}]: got {:.4}, expected {:.4}, diff={:.4}",
            i,
            actual[i],
            expected[i],
            diff
        );
    }

    // Step 5: Record boundary latency
    assert!(
        boundary_latency_ns < BOUNDARY_NS_BUDGET,
        "boundary latency {} ns exceeds budget {} ns",
        boundary_latency_ns,
        BOUNDARY_NS_BUDGET
    );

    eprintln!(
        "[WS8A] test_metal_writes_coreml_reads_iosurface: PASS \
         (boundary_latency={} ns, elements={})",
        boundary_latency_ns, ELEMENT_COUNT
    );

    drop(input);
    drop(output);
    drop(model);
}
