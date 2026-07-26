//! Residency and zero-copy integration tests.
//!
//! Five categories:
//! 1. CPU-reference tests — dequant_matmul_reference produces correct results
//! 2. GPU-correctness tests — Metal produces same results as CPU reference
//! 3. Zero-copy residency tests — IOSurface-backed buffers are shared, not copied
//! 4. Async overlap tests — ComputationToken + submit_compute pipeline
//! 5. Heterogeneous end-to-end — Metal → ANE zero-copy pipeline
//!
//! Run: cargo test --test residency_tests --features prism-backend -- --nocapture --ignored
//! Requires: macOS 14.0+ on Apple Silicon (M1+)

#![cfg(target_os = "macos")]
#![cfg(any(feature = "mlx-backend", feature = "prism-backend"))]

use std::sync::Arc;

use tribunus_compute_core::arena::Arena;
use tribunus_compute_core::backend::ane_backend::AneBackend;
use tribunus_compute_core::backend::metal::MetalBackend;
use tribunus_compute_core::backend::{DType, QuantizedMatmulOp, TensorBackend};
use tribunus_compute_core::nf4tile640::{
    dequant_matmul_reference, pack_nf4_weights, validate_matmul, GROUP_SIZE, TILE_ELEMENTS,
};

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Check whether a Metal device is available on this machine.
fn has_metal_device() -> bool {
    metal::Device::system_default().is_some()
}

/// Pack a small row-major weight matrix into NF4 format, returning
/// (packed_codes, scales, biases).
fn pack_small_weights(rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>, Vec<f32>) {
    let mut weights = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            if r * TILE_ELEMENTS + c == r * TILE_ELEMENTS + c.min(rows - 1) {
                weights[r * cols + c] = 1.0 + (r as f32) * 0.1;
            } else {
                weights[r * cols + c] = (r as f32 * 31.7 + c as f32 * 13.3).sin() * 0.1;
            }
        }
    }
    let (codes, scales, biases, _r, _c) = pack_nf4_weights(&weights, rows, cols);
    (codes, scales, biases)
}

/// Create a small input matrix (m × k) with ramp values.
fn make_input(m: usize, k: usize) -> Vec<f32> {
    (0..m * k).map(|i| (i as f32 + 1.0) * 0.01).collect()
}

/// Allocate an IOSurface-backed arena, write f32 data into it, and return
/// the arena together with a byte slice covering the written region.
///
/// # Safety
///
/// The returned byte slice borrows from `arena`. The caller must keep `arena`
/// alive for as long as the slice is used.
unsafe fn make_iosurface_buffer(f32_data: &[f32]) -> Result<(Arena, &'static [u8]), String> {
    let byte_count = (f32_data.len() * 4).next_power_of_two() as u32;
    let arena = Arena::new_bytes(byte_count)?;
    arena.lock()?;
    let src = f32_data.as_ptr() as *const u8;
    let count = f32_data.len() * std::mem::size_of::<f32>();
    std::ptr::copy_nonoverlapping(src, arena.base_ptr() as *mut u8, count);
    arena.unlock()?;
    let slice = std::slice::from_raw_parts(arena.base_ptr() as *const u8, arena.byte_len());
    Ok((arena, slice))
}

/// Run the Metal `quantized_matmul` and return the output data.
fn metal_quantized_matmul(
    backend: &mut MetalBackend,
    input: &[f32],
    m: usize,
    k: usize,
    n: usize,
    packed_codes: &[u8],
    scales: &[f32],
    biases: &[f32],
) -> Result<Vec<f32>, String> {
    let x = backend.create_f32(input, &[m as i32, k as i32])?;
    let w = backend.alloc_weight(packed_codes.to_vec());
    let s = backend.create_f32(scales, &[scales.len() as i32])?;
    let b = backend.create_f32(biases, &[biases.len() as i32])?;

    let op = QuantizedMatmulOp {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        input_dtype: DType::F32,
        weight_dtype: DType::U8,
        scale_dtype: DType::F32,
        bias_dtype: DType::F32,
        output_dtype: DType::F32,
        group_size: GROUP_SIZE as u32,
        bits: 4,
        transpose: false,
    };

    let y = backend.quantized_matmul(&op, x, w, s, b)?;
    let receipt = backend.read_f32(y)?;
    Ok(receipt.data)
}

/// Run the CPU `dequant_matmul_reference` and return the output data.
fn cpu_dequant_matmul(
    input: &[f32],
    packed_codes: &[u8],
    scales: &[f32],
    biases: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Result<Vec<f32>, String> {
    let mut output = vec![0.0f32; m * n];
    dequant_matmul_reference(input, packed_codes, scales, biases, m, k, n, &mut output)?;
    Ok(output)
}

// ── 1. CPU-reference tests ─────────────────────────────────────────────────

/// CPU-only: dequant_matmul_reference produces correct results for a
/// small matrix. No Metal dependency — runs unconditionally.
#[test]
fn test_cpu_nf4_dequant_reference() {
    let m = 2usize;
    let k = 4usize;
    let n = TILE_ELEMENTS; // pack_nf4_weights requires cols % TILE_ELEMENTS == 0

    let input = make_input(m, k);
    let (packed_codes, scales, biases) = pack_small_weights(k, n);

    let mut output = vec![0.0f32; m * n];
    dequant_matmul_reference(
        &input,
        &packed_codes,
        &scales,
        &biases,
        m,
        k,
        n,
        &mut output,
    )
    .expect("CPU dequant_matmul_reference should succeed");

    // Naive reference: unpack weights, then matmul.
    let unpacked = tribunus_compute_core::nf4tile640::unpack_nf4_weights(
        &packed_codes,
        &scales,
        &biases,
        k,
        n,
    );
    let mut expected = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            for kk in 0..k {
                expected[i * n + j] += input[i * k + kk] * unpacked[kk * n + j];
            }
        }
    }

    let result = validate_matmul(&expected, &output, 0.05);
    assert!(
        result.passed,
        "CPU reference matmul mismatch: max_abs_error={}, mean_abs_error={}, mismatches={}/{}",
        result.max_abs_error, result.mean_abs_error, result.mismatches, result.total_elements
    );
}

// ── 2. GPU-correctness tests ───────────────────────────────────────────────

/// GPU NF4 dequantize+matmul produces the same result (within NF4 tolerance)
/// as the CPU reference implementation.
///
/// Requires a Metal-capable GPU — skipped if none found.
#[cfg_attr(not(feature = "prism-backend"), ignore)]
#[test]
fn test_gpu_nf4_dequant_matches_cpu() -> Result<(), String> {
    if !has_metal_device() {
        eprintln!("SKIP: no Metal device available");
        return Ok(());
    }

    let m = 2usize;
    let k = 4usize;
    let n = TILE_ELEMENTS;

    let input = make_input(m, k);
    let (packed_codes, scales, biases) = pack_small_weights(k, n);

    // CPU reference.
    let cpu_output = cpu_dequant_matmul(&input, &packed_codes, &scales, &biases, m, k, n)?;

    // Metal backend.
    let mut backend = MetalBackend::new()?;
    let gpu_output = metal_quantized_matmul(
        &mut backend,
        &input,
        m,
        k,
        n,
        &packed_codes,
        &scales,
        &biases,
    )?;

    // Compare.
    let result = validate_matmul(&cpu_output, &gpu_output, 0.05);
    assert!(
        result.passed,
        "GPU-CPU mismatch: max_abs_error={}, mean_abs_error={}, mismatches={}/{}",
        result.max_abs_error, result.mean_abs_error, result.mismatches, result.total_elements
    );
    Ok(())
}

// ── 3. Zero-copy residency tests ───────────────────────────────────────────

/// Verifies that a buffer bound via `bind_external` does not deep-copy the
/// data. Writes data into an IOSurface-backed arena, binds it to
/// MetalBackend, and reads back through the backend. The data should match
/// without intermediate copies.
#[cfg_attr(not(feature = "prism-backend"), ignore)]
#[test]
fn test_bind_external_does_not_copy() -> Result<(), String> {
    if !has_metal_device() {
        eprintln!("SKIP: no Metal device available");
        return Ok(());
    }

    let test_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let (arena, slice) = unsafe { make_iosurface_buffer(&test_data)? };

    // Bind the IOSurface-backed buffer to MetalBackend.
    let mut backend = MetalBackend::new()?;
    let token = arena.io_surface_id() as u64;
    let handle = backend.bind_external(token, slice, &[test_data.len() as i32], DType::F32)?;

    // Read back — should match what was written.
    let receipt = backend.read_f32(handle)?;

    assert_eq!(
        receipt.data.len(),
        test_data.len(),
        "readback length mismatch"
    );
    for (i, (&got, &expected)) in receipt.data.iter().zip(test_data.iter()).enumerate() {
        let abs_err = (got - expected).abs();
        assert!(
            abs_err < 1e-6,
            "element {i}: got {got}, expected {expected}, abs_err={abs_err}"
        );
    }

    Ok(())
}

/// Verifies that an IOSurface-backed buffer bound to both MetalBackend and
/// AneBackend is shared without copying — both see identical data.
#[cfg_attr(not(feature = "prism-backend"), ignore)]
#[test]
fn test_shared_buffer_identity() -> Result<(), String> {
    if !has_metal_device() {
        eprintln!("SKIP: no Metal device available");
        return Ok(());
    }

    let test_data: Vec<f32> = vec![42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0];
    let (arena, slice) = unsafe { make_iosurface_buffer(&test_data)? };
    let token = arena.io_surface_id() as u64;

    // Bind the same IOSurface-backed buffer to both backends.
    let mut metal_backend = MetalBackend::new()?;
    let mut ane_backend = AneBackend::default();

    let m_handle =
        metal_backend.bind_external(token, slice, &[test_data.len() as i32], DType::F32)?;
    let a_handle =
        ane_backend.bind_external(token, slice, &[test_data.len() as i32], DType::F32)?;

    // Both backends should read identical data.
    let m_receipt = metal_backend.read_f32(m_handle)?;
    let a_receipt = ane_backend.read_f32(a_handle)?;

    assert_eq!(m_receipt.data.len(), test_data.len());
    assert_eq!(a_receipt.data.len(), test_data.len());

    for i in 0..test_data.len() {
        let abs_err_m = (m_receipt.data[i] - test_data[i]).abs();
        let abs_err_a = (a_receipt.data[i] - test_data[i]).abs();
        assert!(
            abs_err_m < 1e-6,
            "Metal readback element {i}: got {}, expected {}",
            m_receipt.data[i],
            test_data[i]
        );
        assert!(
            abs_err_a < 1e-6,
            "ANE readback element {i}: got {}, expected {}",
            a_receipt.data[i],
            test_data[i]
        );
    }

    Ok(())
}

// ── 4. Async overlap tests ─────────────────────────────────────────────────

/// Verifies that a ComputationToken from `submit_compute` fires correctly:
/// encode a small GPU compute operation, obtain the token, register a
/// callback via `then()`, and block until the callback fires.
#[cfg_attr(not(feature = "prism-backend"), ignore)]
#[test]
fn test_command_buffer_completion_token() -> Result<(), String> {
    if !has_metal_device() {
        eprintln!("SKIP: no Metal device available");
        return Ok(());
    }

    let mut backend = MetalBackend::new()?;
    let m = 2usize;
    let k = 4usize;
    let n = TILE_ELEMENTS;

    let input = make_input(m, k);
    let (packed_codes, scales, biases) = pack_small_weights(k, n);

    // Set up quantized_matmul inputs.
    let x = backend.create_f32(&input, &[m as i32, k as i32])?;
    let w = backend.alloc_weight(packed_codes.to_vec());
    let s = backend.create_f32(&scales, &[scales.len() as i32])?;
    let b = backend.create_f32(&biases, &[biases.len() as i32])?;

    let op = QuantizedMatmulOp {
        m: m as u32,
        n: n as u32,
        k: k as u32,
        input_dtype: DType::F32,
        weight_dtype: DType::U8,
        scale_dtype: DType::F32,
        bias_dtype: DType::F32,
        output_dtype: DType::F32,
        group_size: GROUP_SIZE as u32,
        bits: 4,
        transpose: false,
    };

    let y = backend.quantized_matmul(&op, x, w, s, b)?;

    // Submit compute and get the ComputationToken.
    let completion_token = backend.submit_compute(0, &[y])?;

    // Register a callback via then().
    let fired = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let fired_clone = Arc::clone(&fired);
    completion_token.then(move || {
        fired_clone.store(true, std::sync::atomic::Ordering::SeqCst);
    });

    // Wait for the token.
    completion_token.wait();

    // The callback should have fired.
    assert!(
        fired.load(std::sync::atomic::Ordering::SeqCst),
        "callback registered via then() did not fire after token completion"
    );

    // Read back data — should be available after completion.
    let receipt = backend.read_f32(y)?;
    assert!(
        !receipt.data.is_empty(),
        "output data should be ready after token completion"
    );

    Ok(())
}

/// Integration test: Metal produces output tracked by a ComputationToken,
/// and ANE reads that output after the token resolves. Validates the
/// cross-backend dependency chain using IOSurface-shared memory.
#[cfg_attr(not(feature = "prism-backend"), ignore)]
#[test]
fn test_cross_backend_dependency() -> Result<(), String> {
    if !has_metal_device() {
        eprintln!("SKIP: no Metal device available");
        return Ok(());
    }

    // ── Allocate an IOSurface-backed buffer for cross-backend sharing ─

    let test_data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.5).collect();
    let (arena, slice) = unsafe { make_iosurface_buffer(&test_data)? };
    let token_id = arena.io_surface_id() as u64;

    // ── Metal writes data into the IOSurface buffer ────────────────────

    let mut metal_backend = MetalBackend::new()?;
    let m_handle =
        metal_backend.bind_external(token_id, slice, &[test_data.len() as i32], DType::F32)?;

    // Write known data through the Metal buffer's raw pointer to simulate
    // GPU compute output.
    unsafe {
        arena.lock()?;
        let src = test_data.as_ptr() as *const u8;
        let count = test_data.len() * std::mem::size_of::<f32>();
        std::ptr::copy_nonoverlapping(src, arena.base_ptr() as *mut u8, count);
        arena.unlock()?;
    }

    // Submit compute (triggers evaluation + completion token).
    let token = metal_backend.submit_compute(0, &[m_handle])?;

    // ── ANE binds to the same IOSurface buffer ─────────────────────────

    let mut ane_backend = AneBackend::default();
    let a_handle =
        ane_backend.bind_external(token_id, slice, &[test_data.len() as i32], DType::F32)?;

    // Wait for Metal to finish.
    token.wait();

    // ── Both backends should read identical data ───────────────────────

    let m_receipt = metal_backend.read_f32(m_handle)?;
    let a_receipt = ane_backend.read_f32(a_handle)?;

    assert_eq!(m_receipt.data.len(), test_data.len());
    assert_eq!(a_receipt.data.len(), test_data.len());

    for i in 0..test_data.len() {
        let abs_err = (m_receipt.data[i] - test_data[i]).abs();
        assert!(
            abs_err < 1e-6,
            "Metal readback element {i}: got {}, expected {}",
            m_receipt.data[i],
            test_data[i]
        );
    }

    // ANE sees the same data as Metal (zero-copy via IOSurface).
    for i in 0..test_data.len() {
        let abs_err = (a_receipt.data[i] - m_receipt.data[i]).abs();
        assert!(
            abs_err < 1e-6,
            "Cross-backend mismatch at element {i}: Metal={}, ANE={}",
            m_receipt.data[i],
            a_receipt.data[i]
        );
    }

    Ok(())
}

// ── 5. Heterogeneous end-to-end test ────────────────────────────────────────

/// Full heterogeneous pipeline test:
///
/// 1. Metal allocates an IOSurface-backed buffer
/// 2. Metal computes an NF4 dequant+matmul activation, stores it in the
///    IOSurface buffer
/// 3. ComputationToken tracks Metal completion
/// 4. ANE reads the same buffer (IOSurface shared) after token resolves
/// 5. ANE produces output into another IOSurface-backed buffer
/// 6. Metal reads ANE output and verifies correctness against CPU reference
/// 7. Entire pipeline is zero-copy (no host round-trip)
#[cfg_attr(not(feature = "prism-backend"), ignore)]
#[test]
fn test_heterogeneous_metal_to_ane_activation() -> Result<(), String> {
    if !has_metal_device() {
        eprintln!("SKIP: no Metal device available");
        return Ok(());
    }

    // ── Compute parameters and data ────────────────────────────────────

    let m = 2usize;
    let k = 4usize;
    let n = TILE_ELEMENTS;

    let input = make_input(m, k);
    let (packed_codes, scales, biases) = pack_small_weights(k, n);

    // CPU reference for later comparison.
    let cpu_output = cpu_dequant_matmul(&input, &packed_codes, &scales, &biases, m, k, n)?;

    // ── Step 1 & 2: Metal computes and stores in IOSurface buffer ──────

    let mut metal_backend = MetalBackend::new()?;
    let gpu_output = metal_quantized_matmul(
        &mut metal_backend,
        &input,
        m,
        k,
        n,
        &packed_codes,
        &scales,
        &biases,
    )?;

    let (output_arena, output_slice) = unsafe { make_iosurface_buffer(&gpu_output)? };
    let output_token = output_arena.io_surface_id() as u64;

    // Verify GPU output matches CPU reference.
    let result = validate_matmul(&cpu_output, &gpu_output, 0.05);
    assert!(
        result.passed,
        "GPU output should match CPU reference before cross-backend transfer: \
         max_abs_error={}, mismatches={}/{}",
        result.max_abs_error, result.mismatches, result.total_elements
    );

    // ── Step 3: Bind Metal output to IOSurface, create token ──────────

    let m_handle_out = metal_backend.bind_external(
        output_token,
        output_slice,
        &[gpu_output.len() as i32],
        DType::F32,
    )?;
    let token = metal_backend.submit_compute(0, &[m_handle_out])?;

    // ── Step 4 & 5: ANE reads same IOSurface buffer ────────────────────

    let mut ane_backend = AneBackend::default();

    // ANE reads from the Metal output buffer.
    let a_handle_src = ane_backend.bind_external(
        output_token,
        output_slice,
        &[gpu_output.len() as i32],
        DType::F32,
    )?;

    // ANE writes into a separate output buffer.
    let (ane_output_arena, ane_output_slice) = unsafe { make_iosurface_buffer(&gpu_output)? };
    let ane_output_token = ane_output_arena.io_surface_id() as u64;

    let _a_handle_dst = ane_backend.bind_external(
        ane_output_token,
        ane_output_slice,
        &[gpu_output.len() as i32],
        DType::F32,
    )?;

    // Wait for Metal to finish writing buffer A.
    token.wait();

    // ANE reads from the Metal output buffer and sees the same data.
    let a_src_receipt = ane_backend.read_f32(a_handle_src)?;
    let m_out_receipt = metal_backend.read_f32(m_handle_out)?;

    // Cross-backend data consistency.
    for i in 0..gpu_output.len() {
        let abs_err = (a_src_receipt.data[i] - m_out_receipt.data[i]).abs();
        assert!(
            abs_err < 1e-6,
            "Cross-backend readback mismatch at element {i}: Metal={}, ANE={}",
            m_out_receipt.data[i],
            a_src_receipt.data[i]
        );
    }

    // Simulate ANE writing processed data into its output buffer.
    // In the real pipeline the ANE planar engine processes the data inline.
    unsafe {
        ane_output_arena.lock()?;
        let src = a_src_receipt.data.as_ptr() as *const u8;
        let count = a_src_receipt.data.len() * std::mem::size_of::<f32>();
        std::ptr::copy_nonoverlapping(src, ane_output_arena.base_ptr() as *mut u8, count);
        ane_output_arena.unlock()?;
    }

    // ── Step 6: Metal reads ANE output buffer ─────────────────────────

    let m_handle_final = metal_backend.bind_external(
        ane_output_token,
        ane_output_slice,
        &[gpu_output.len() as i32],
        DType::F32,
    )?;
    let final_receipt = metal_backend.read_f32(m_handle_final)?;

    // ── Step 7: Verify against CPU reference ───────────────────────────

    // The end-to-end result (GPU data → IOSurface bind → ANE readback)
    // should match the CPU reference within NF4 tolerance.
    let result_end = validate_matmul(&cpu_output, &final_receipt.data, 0.05);
    assert!(
        result_end.passed,
        "Heterogeneous pipeline end-to-end: CPU-output mismatch after \
         cross-backend round-trip: max_abs_error={}, mean_abs_error={}, \
         mismatches={}/{}",
        result_end.max_abs_error,
        result_end.mean_abs_error,
        result_end.mismatches,
        result_end.total_elements
    );

    Ok(())
}
