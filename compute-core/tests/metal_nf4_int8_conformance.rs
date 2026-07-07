//! Metal NF4/INT8 tile640 GEMV conformance test.
//!
//! Packs random weight matrices, dispatches NF4 and INT8 Metal kernels,
//! and verifies outputs match CPU reference within tolerance.
//!
//! Includes scaled reduction-axis tests (buffer 7 = reduction_scales) for
//! both NF4 (manually encoded) and INT8 (via Int8Tile640GEMVDispatcher).
//!
//! Also covers non-multiple-of-640 and multi-tile cases.
//!
//! Run:  cargo test --test metal_nf4_int8_conformance --features metal-dispatch,prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "metal-dispatch"))]

use metal::*;
use half::f16;
use parking_lot::Mutex;
use rand::Rng;
use std::sync::Arc;
use tribunus_compute_core::compute_image::compile::kernel_dispatch::{
    Int8Tile640GEMVDispatcher, Int8Tile640Offsets, Nf4Tile640ProjectionDispatcher,
    Nf4Tile640Offsets, RegistryRef,
};
use tribunus_compute_core::compute_image::compile::kernel_registry::KernelRegistry;
use tribunus_compute_core::compute_image::compile::kernel_types::{
    KernelReceipt, ProjectionParams,
};
use tribunus_compute_core::nf4tile640::{
    pack_int8_weights, pack_nf4_weights, unpack_int8_weights, unpack_nf4_weights, TILE_ELEMENTS,
};

const ROWS: usize = 128;
const COLS: usize = TILE_ELEMENTS;
const BATCH: usize = 4;

fn zero_receipt() -> KernelReceipt {
    KernelReceipt {
        kernel_id: 0,
        phase_id: 0,
        page_count: 0,
        sidecar_hits: 0,
        sidecar_entries_read: 0,
        threadgroups: 0,
        threads_per_threadgroup: 0,
        output_elements: 0,
        flags: 0,
        logical_weight_bytes: 0,
        logical_sidecar_bytes: 0,
        logical_activation_bytes: 0,
    }
}

fn ref_matmul(weights: &[f32], input: &[f32], rows: usize, cols: usize, batch: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * rows];
    for b in 0..batch {
        for r in 0..rows {
            let mut sum = 0.0f32;
            for c in 0..cols {
                sum += input[b * cols + c] * weights[r * cols + c];
            }
            out[b * rows + r] = sum;
        }
    }
    out
}

/// Compute ref output for scaled reduction: Y = (X ⊙ S)W'^T.
/// S is a per-column (reduction-axis) scale vector of length `cols`.
fn ref_scaled_matmul(
    weights: &[f32],
    input: &[f32],
    reduction_scales: &[f32],
    rows: usize,
    cols: usize,
    batch: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * rows];
    for b in 0..batch {
        for r in 0..rows {
            let mut sum = 0.0f32;
            for c in 0..cols {
                sum += input[b * cols + c] * reduction_scales[c] * weights[r * cols + c];
            }
            out[b * rows + r] = sum;
        }
    }
    out
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    if na > 1e-10 && nb > 1e-10 {
        dot / (na * nb)
    } else {
        1.0
    }
}

fn nrmse(a: &[f32], b: &[f32]) -> f32 {
    let rms_a = (a.iter().map(|v| v * v).sum::<f32>() / a.len() as f32).sqrt();
    let diff: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    (diff / a.len() as f32).sqrt() / rms_a
}

fn make_params(rows: u32, cols: u32) -> ProjectionParams {
    ProjectionParams {
        in_dim: cols,
        out_dim: rows,
        page_count: 1,
        page_width: TILE_ELEMENTS as u32,
        mode_flags: 0,
        probe_seed: 0,
        reserved: [0; 5],
    }
}

/// Manual NF4 scaled-reduction dispatch: binds buffers 0-6 like the base
/// NF4 kernel and buffer 7 = reduction_scales (FP16).  There is no
/// dispatcher struct for the scaled NF4 kernel yet.
fn dispatch_nf4_scaled_gemv(
    registry: &RegistryRef,
    cmd_buf: &CommandBufferRef,
    codes_buf: &Buffer,
    scales_buf: &Buffer,
    biases_buf: &Buffer,
    in_buf: &Buffer,
    out_buf: &Buffer,
    reduction_scales_buf: &Buffer,
    params: &ProjectionParams,
    receipt: &mut KernelReceipt,
) {
    let (pso, dev) = {
        let mut reg = registry.lock();
        let fcv = FunctionConstantValues::new();
        let pso = reg.get_or_create("fused_gemv_nf4_scaled_reduction_tile640_fp32", &fcv, 0);
        (pso, reg.device().clone())
    };

    let encoder = cmd_buf.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pso);
    encoder.set_buffer(0, Some(codes_buf), 0);
    encoder.set_buffer(1, Some(scales_buf), 0);
    encoder.set_buffer(2, Some(biases_buf), 0);
    encoder.set_buffer(3, Some(in_buf), 0);
    encoder.set_buffer(4, Some(out_buf), 0);

    let num_macro_tiles = params.page_count.max(1);
    let num_macro_tiles_buf = dev.new_buffer_with_data(
        &num_macro_tiles as *const u32 as *const std::ffi::c_void,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    encoder.set_buffer(5, Some(&num_macro_tiles_buf), 0);

    let in_dim_val = params.in_dim;
    let in_dim_buf = dev.new_buffer_with_data(
        &in_dim_val as *const u32 as *const std::ffi::c_void,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    encoder.set_buffer(6, Some(&in_dim_buf), 0);

    encoder.set_buffer(7, Some(reduction_scales_buf), 0);

    encoder.dispatch_thread_groups(
        MTLSize {
            width: params.out_dim as u64,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    encoder.end_encoding();

    receipt.kernel_id = 14; // NF4_TILE640_SCALED_REDUCTION
    receipt.phase_id = 0;
    receipt.page_count = num_macro_tiles;
    receipt.sidecar_hits = 0;
    receipt.sidecar_entries_read = 0;
    receipt.threadgroups = params.out_dim;
    receipt.threads_per_threadgroup = 32;
    receipt.output_elements = params.out_dim;
    receipt.flags = 0;
    receipt.logical_weight_bytes = (params.out_dim as u64) * (num_macro_tiles as u64) * 320;
    receipt.logical_sidecar_bytes =
        (params.out_dim as u64) * (num_macro_tiles as u64) * 5 * 2 * 4;
    receipt.logical_activation_bytes = (params.in_dim as u64) * 4 + (params.out_dim as u64) * 4;
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_metal_nf4_gemv() {
    let mut rng = rand::thread_rng();
    let src: Vec<f32> = (0..ROWS * COLS).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let input: Vec<f32> = (0..BATCH * COLS).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let (codes, scales, biases, ..) = pack_nf4_weights(&src, ROWS, COLS);
    let ref_out = ref_matmul(&src, &input, ROWS, COLS, BATCH);

    let device = Device::system_default().expect("no Metal device");
    let registry: RegistryRef = Arc::new(Mutex::new(KernelRegistry::new(&device)));
    let dispatcher = Nf4Tile640ProjectionDispatcher::new(registry.clone());

    let codes_buf = device.new_buffer_with_data(
        codes.as_ptr() as *const std::ffi::c_void,
        codes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let scales_buf = device.new_buffer_with_data(
        scales.as_ptr() as *const std::ffi::c_void,
        (scales.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let biases_buf = device.new_buffer_with_data(
        biases.as_ptr() as *const std::ffi::c_void,
        (biases.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let params = make_params(ROWS as u32, COLS as u32);
    let mut all_gpu = Vec::with_capacity(BATCH * ROWS);
    for b in 0..BATCH {
        let input_slice = &input[b * COLS..(b + 1) * COLS];
        let in_buf = device.new_buffer_with_data(
            input_slice.as_ptr() as *const std::ffi::c_void,
            (COLS * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = device.new_buffer(
            (ROWS * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let cmd_queue = device.new_command_queue();
        let cmd_buf = cmd_queue.new_command_buffer();
        let mut receipt = zero_receipt();

        dispatcher.dispatch_with_offsets(
            &cmd_buf, &codes_buf, &scales_buf, &biases_buf,
            &in_buf, &out_buf, &params,
            Nf4Tile640Offsets::default(), &mut receipt,
        );
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let ptr = out_buf.contents() as *const f32;
        let result = unsafe { std::slice::from_raw_parts(ptr, ROWS).to_vec() };
        all_gpu.extend(result);
    }

    let cos = cosine_similarity(&ref_out, &all_gpu);
    let nrms = nrmse(&ref_out, &all_gpu);
    assert!(cos > 0.99, "NF4 Metal cosine {:.8} <= 0.99", cos);
    assert!(nrms < 0.15, "NF4 Metal NRMSE {:.6} >= 0.15", nrms);
}

#[test]
fn test_metal_int8_gemv() {
    let mut rng = rand::thread_rng();
    let src: Vec<f32> = (0..ROWS * COLS).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let input: Vec<f32> = (0..BATCH * COLS).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let (codes, scales, biases) = pack_int8_weights(&src, ROWS, COLS);
    let ref_out = ref_matmul(&src, &input, ROWS, COLS, BATCH);

    let device = Device::system_default().expect("no Metal device");
    let registry: RegistryRef = Arc::new(Mutex::new(KernelRegistry::new(&device)));
    let dispatcher = Int8Tile640GEMVDispatcher::new(registry.clone());

    let codes_buf = device.new_buffer_with_data(
        codes.as_ptr() as *const std::ffi::c_void,
        codes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let scales_buf = device.new_buffer_with_data(
        scales.as_ptr() as *const std::ffi::c_void,
        (scales.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let biases_buf = device.new_buffer_with_data(
        biases.as_ptr() as *const std::ffi::c_void,
        (biases.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let params = make_params(ROWS as u32, COLS as u32);
    let mut all_gpu = Vec::with_capacity(BATCH * ROWS);
    for b in 0..BATCH {
        let input_slice = &input[b * COLS..(b + 1) * COLS];
        let in_buf = device.new_buffer_with_data(
            input_slice.as_ptr() as *const std::ffi::c_void,
            (COLS * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = device.new_buffer(
            (ROWS * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let cmd_queue = device.new_command_queue();
        let cmd_buf = cmd_queue.new_command_buffer();
        let mut receipt = zero_receipt();

        dispatcher.dispatch_with_offsets(
            &cmd_buf, &codes_buf, &scales_buf, &biases_buf,
            &in_buf, &out_buf, None, &params,
            Int8Tile640Offsets::default(), &mut receipt,
        );
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let ptr = out_buf.contents() as *const f32;
        let result = unsafe { std::slice::from_raw_parts(ptr, ROWS).to_vec() };
        all_gpu.extend(result);
    }

    let cos = cosine_similarity(&ref_out, &all_gpu);
    let nrms = nrmse(&ref_out, &all_gpu);
    assert!(cos > 0.99, "INT8 Metal cosine {:.8} <= 0.99", cos);
    assert!(nrms < 0.02, "INT8 Metal NRMSE {:.6} >= 0.02", nrms);
}

/// NF4 scaled reduction-axis GEMV: dispatches the
/// `fused_gemv_nf4_scaled_reduction_tile640_fp32` kernel with an FP16
/// reduction-axis scale vector in buffer 7.  The kernel computes
/// Y = (X ⊙ S)W'^T where S is the per-column reduction scale.
#[test]
fn test_metal_nf4_scaled_reduction_gemv() {
    let mut rng = rand::thread_rng();
    let src: Vec<f32> = (0..ROWS * COLS).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let input: Vec<f32> = (0..BATCH * COLS).map(|_| rng.gen_range(-0.5..0.5)).collect();

    // Random reduction-axis scales in [0.1, 2.0].
    let reduction_scales: Vec<f32> = (0..COLS)
        .map(|_| rng.gen_range(0.1f32..2.0))
        .collect();

    // Pack NF4 weights for CPU reference.
    let (codes, scales, biases, ..) = pack_nf4_weights(&src, ROWS, COLS);

    // CPU reference: dequantize weights, then compute Y = (X ⊙ S)W'^T.
    let dequant_src = unpack_nf4_weights(&codes, &scales, &biases, ROWS, COLS);
    let ref_out = ref_scaled_matmul(&dequant_src, &input, &reduction_scales, ROWS, COLS, BATCH);

    // GPU path.
    let device = Device::system_default().expect("no Metal device");
    let registry: RegistryRef = Arc::new(Mutex::new(KernelRegistry::new(&device)));

    // Pack reduction scales as FP16 (half*) for the GPU.
    let reduction_scales_bytes: Vec<u8> = reduction_scales
        .iter()
        .copied()
        .flat_map(|v| f16::from_f32(v).to_bits().to_le_bytes())
        .collect();

    let codes_buf = device.new_buffer_with_data(
        codes.as_ptr() as *const std::ffi::c_void,
        codes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let scales_buf = device.new_buffer_with_data(
        scales.as_ptr() as *const std::ffi::c_void,
        (scales.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let biases_buf = device.new_buffer_with_data(
        biases.as_ptr() as *const std::ffi::c_void,
        (biases.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let rs_buf = device.new_buffer_with_data(
        reduction_scales_bytes.as_ptr() as *const std::ffi::c_void,
        reduction_scales_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let params = make_params(ROWS as u32, COLS as u32);
    let mut all_gpu = Vec::with_capacity(BATCH * ROWS);
    for b in 0..BATCH {
        let input_slice = &input[b * COLS..(b + 1) * COLS];
        let in_buf = device.new_buffer_with_data(
            input_slice.as_ptr() as *const std::ffi::c_void,
            (COLS * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = device.new_buffer(
            (ROWS * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let cmd_queue = device.new_command_queue();
        let cmd_buf = cmd_queue.new_command_buffer();
        let mut receipt = zero_receipt();

        dispatch_nf4_scaled_gemv(
            &registry,
            &cmd_buf,
            &codes_buf,
            &scales_buf,
            &biases_buf,
            &in_buf,
            &out_buf,
            &rs_buf,
            &params,
            &mut receipt,
        );
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let ptr = out_buf.contents() as *const f32;
        let result = unsafe { std::slice::from_raw_parts(ptr, ROWS).to_vec() };
        all_gpu.extend(result);
    }

    let cos = cosine_similarity(&ref_out, &all_gpu);
    let nrms = nrmse(&ref_out, &all_gpu);
    assert!(cos > 0.99, "NF4 scaled reduction cosine {:.8} <= 0.99", cos);
    assert!(nrms < 0.15, "NF4 scaled reduction NRMSE {:.6} >= 0.15", nrms);
}

/// INT8 scaled reduction-axis GEMV via Int8Tile640GEMVDispatcher with
/// reduction_scales_buffer = Some(...).  The kernel computes
/// Y = (X ⊙ S)W'^T where S is the per-column reduction scale.
#[test]
fn test_metal_int8_scaled_reduction_gemv() {
    let mut rng = rand::thread_rng();
    let src: Vec<f32> = (0..ROWS * COLS).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let input: Vec<f32> = (0..BATCH * COLS).map(|_| rng.gen_range(-0.5..0.5)).collect();

    let reduction_scales: Vec<f32> = (0..COLS)
        .map(|_| rng.gen_range(0.1f32..2.0))
        .collect();

    let (codes, scales, biases) = pack_int8_weights(&src, ROWS, COLS);

    // CPU reference: dequantize, then compute Y = (X ⊙ S)W'^T.
    let dequant_src = unpack_int8_weights(&codes, &scales, &biases, ROWS, COLS);
    let ref_out = ref_scaled_matmul(&dequant_src, &input, &reduction_scales, ROWS, COLS, BATCH);

    // GPU path.
    let device = Device::system_default().expect("no Metal device");
    let registry: RegistryRef = Arc::new(Mutex::new(KernelRegistry::new(&device)));
    let dispatcher = Int8Tile640GEMVDispatcher::new(registry.clone());

    let reduction_scales_bytes: Vec<u8> = reduction_scales
        .iter()
        .copied()
        .flat_map(|v| f16::from_f32(v).to_bits().to_le_bytes())
        .collect();

    let codes_buf = device.new_buffer_with_data(
        codes.as_ptr() as *const std::ffi::c_void,
        codes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let scales_buf = device.new_buffer_with_data(
        scales.as_ptr() as *const std::ffi::c_void,
        (scales.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let biases_buf = device.new_buffer_with_data(
        biases.as_ptr() as *const std::ffi::c_void,
        (biases.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let rs_buf = device.new_buffer_with_data(
        reduction_scales_bytes.as_ptr() as *const std::ffi::c_void,
        reduction_scales_bytes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );

    let params = make_params(ROWS as u32, COLS as u32);
    let mut all_gpu = Vec::with_capacity(BATCH * ROWS);
    for b in 0..BATCH {
        let input_slice = &input[b * COLS..(b + 1) * COLS];
        let in_buf = device.new_buffer_with_data(
            input_slice.as_ptr() as *const std::ffi::c_void,
            (COLS * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = device.new_buffer(
            (ROWS * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let cmd_queue = device.new_command_queue();
        let cmd_buf = cmd_queue.new_command_buffer();
        let mut receipt = zero_receipt();

        dispatcher.dispatch_with_offsets(
            &cmd_buf, &codes_buf, &scales_buf, &biases_buf,
            &in_buf, &out_buf, Some(&rs_buf), &params,
            Int8Tile640Offsets::default(), &mut receipt,
        );
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let ptr = out_buf.contents() as *const f32;
        let result = unsafe { std::slice::from_raw_parts(ptr, ROWS).to_vec() };
        all_gpu.extend(result);
    }

    let cos = cosine_similarity(&ref_out, &all_gpu);
    let nrms = nrmse(&ref_out, &all_gpu);
    assert!(cos > 0.99, "INT8 scaled reduction cosine {:.8} <= 0.99", cos);
    assert!(nrms < 0.02, "INT8 scaled reduction NRMSE {:.6} >= 0.02", nrms);
}

/// NF4 GEMV with non-multiple-of-640 input width (700 columns).
/// The packer pads to 1280.  The kernel's buffer(6) in_dim guard ensures
/// activation reads don't exceed the real 700-element input.
#[test]
fn test_metal_nf4_non_multiple_of_640() {
    const COLS_700: usize = 700;
    const ROWS_64: usize = 64;
    const BATCH_2: usize = 2;

    let mut rng = rand::thread_rng();
    let src: Vec<f32> = (0..ROWS_64 * COLS_700)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let input: Vec<f32> = (0..BATCH_2 * COLS_700)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let (codes, scales, biases, ..) = pack_nf4_weights(&src, ROWS_64, COLS_700);
    let ref_out = ref_matmul(&src, &input, ROWS_64, COLS_700, BATCH_2);

    let device = Device::system_default().expect("no Metal device");
    let registry: RegistryRef = Arc::new(Mutex::new(KernelRegistry::new(&device)));
    let dispatcher = Nf4Tile640ProjectionDispatcher::new(registry.clone());

    let codes_buf = device.new_buffer_with_data(
        codes.as_ptr() as *const std::ffi::c_void,
        codes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let scales_buf = device.new_buffer_with_data(
        scales.as_ptr() as *const std::ffi::c_void,
        (scales.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let biases_buf = device.new_buffer_with_data(
        biases.as_ptr() as *const std::ffi::c_void,
        (biases.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Non-multiple-of-640: page_count = ceil(700/640) = 2.
    let params = ProjectionParams {
        in_dim: COLS_700 as u32,
        out_dim: ROWS_64 as u32,
        page_count: 2,
        page_width: TILE_ELEMENTS as u32,
        mode_flags: 0,
        probe_seed: 0,
        reserved: [0; 5],
    };
    let mut all_gpu = Vec::with_capacity(BATCH_2 * ROWS_64);
    for b in 0..BATCH_2 {
        let input_slice = &input[b * COLS_700..(b + 1) * COLS_700];
        let in_buf = device.new_buffer_with_data(
            input_slice.as_ptr() as *const std::ffi::c_void,
            (COLS_700 * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = device.new_buffer(
            (ROWS_64 * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let cmd_queue = device.new_command_queue();
        let cmd_buf = cmd_queue.new_command_buffer();
        let mut receipt = zero_receipt();

        dispatcher.dispatch_with_offsets(
            &cmd_buf, &codes_buf, &scales_buf, &biases_buf,
            &in_buf, &out_buf, &params,
            Nf4Tile640Offsets::default(), &mut receipt,
        );
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let ptr = out_buf.contents() as *const f32;
        let result = unsafe { std::slice::from_raw_parts(ptr, ROWS_64).to_vec() };
        all_gpu.extend(result);
    }

    let cos = cosine_similarity(&ref_out, &all_gpu);
    let nrms = nrmse(&ref_out, &all_gpu);
    assert!(
        cos > 0.99,
        "NF4 non-mult-of-640 cosine {:.8} <= 0.99",
        cos
    );
    assert!(nrms < 0.15, "NF4 non-mult-of-640 NRMSE {:.6} >= 0.15", nrms);
}

/// INT8 GEMV with non-multiple-of-640 input width (700 columns).
#[test]
fn test_metal_int8_non_multiple_of_640() {
    const COLS_700: usize = 700;
    const ROWS_64: usize = 64;
    const BATCH_2: usize = 2;

    let mut rng = rand::thread_rng();
    let src: Vec<f32> = (0..ROWS_64 * COLS_700)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let input: Vec<f32> = (0..BATCH_2 * COLS_700)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let (codes, scales, biases) = pack_int8_weights(&src, ROWS_64, COLS_700);
    let ref_out = ref_matmul(&src, &input, ROWS_64, COLS_700, BATCH_2);

    let device = Device::system_default().expect("no Metal device");
    let registry: RegistryRef = Arc::new(Mutex::new(KernelRegistry::new(&device)));
    let dispatcher = Int8Tile640GEMVDispatcher::new(registry.clone());

    let codes_buf = device.new_buffer_with_data(
        codes.as_ptr() as *const std::ffi::c_void,
        codes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let scales_buf = device.new_buffer_with_data(
        scales.as_ptr() as *const std::ffi::c_void,
        (scales.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let biases_buf = device.new_buffer_with_data(
        biases.as_ptr() as *const std::ffi::c_void,
        (biases.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Non-multiple-of-640: page_count = ceil(700/640) = 2.
    let params = ProjectionParams {
        in_dim: COLS_700 as u32,
        out_dim: ROWS_64 as u32,
        page_count: 2,
        page_width: TILE_ELEMENTS as u32,
        mode_flags: 0,
        probe_seed: 0,
        reserved: [0; 5],
    };
    let mut all_gpu = Vec::with_capacity(BATCH_2 * ROWS_64);
    for b in 0..BATCH_2 {
        let input_slice = &input[b * COLS_700..(b + 1) * COLS_700];
        let in_buf = device.new_buffer_with_data(
            input_slice.as_ptr() as *const std::ffi::c_void,
            (COLS_700 * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = device.new_buffer(
            (ROWS_64 * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let cmd_queue = device.new_command_queue();
        let cmd_buf = cmd_queue.new_command_buffer();
        let mut receipt = zero_receipt();

        dispatcher.dispatch_with_offsets(
            &cmd_buf, &codes_buf, &scales_buf, &biases_buf,
            &in_buf, &out_buf, None, &params,
            Int8Tile640Offsets::default(), &mut receipt,
        );
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let ptr = out_buf.contents() as *const f32;
        let result = unsafe { std::slice::from_raw_parts(ptr, ROWS_64).to_vec() };
        all_gpu.extend(result);
    }

    let cos = cosine_similarity(&ref_out, &all_gpu);
    let nrms = nrmse(&ref_out, &all_gpu);
    assert!(
        cos > 0.99,
        "INT8 non-mult-of-640 cosine {:.8} <= 0.99",
        cos
    );
    assert!(
        nrms < 0.02,
        "INT8 non-mult-of-640 NRMSE {:.6} >= 0.02",
        nrms
    );
}

/// NF4 multi-tile GEMV: 1280 columns = 2 tiles per row.
#[test]
fn test_metal_nf4_multi_tile() {
    const COLS_1280: usize = 1280;
    const ROWS_64: usize = 64;
    const BATCH_2: usize = 2;

    let mut rng = rand::thread_rng();
    let src: Vec<f32> = (0..ROWS_64 * COLS_1280)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let input: Vec<f32> = (0..BATCH_2 * COLS_1280)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let (codes, scales, biases, ..) = pack_nf4_weights(&src, ROWS_64, COLS_1280);
    let ref_out = ref_matmul(&src, &input, ROWS_64, COLS_1280, BATCH_2);

    let device = Device::system_default().expect("no Metal device");
    let registry: RegistryRef = Arc::new(Mutex::new(KernelRegistry::new(&device)));
    let dispatcher = Nf4Tile640ProjectionDispatcher::new(registry.clone());

    let codes_buf = device.new_buffer_with_data(
        codes.as_ptr() as *const std::ffi::c_void,
        codes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let scales_buf = device.new_buffer_with_data(
        scales.as_ptr() as *const std::ffi::c_void,
        (scales.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let biases_buf = device.new_buffer_with_data(
        biases.as_ptr() as *const std::ffi::c_void,
        (biases.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Multi-tile: page_count = ceil(1280/640) = 2.
    let params = ProjectionParams {
        in_dim: COLS_1280 as u32,
        out_dim: ROWS_64 as u32,
        page_count: 2,
        page_width: TILE_ELEMENTS as u32,
        mode_flags: 0,
        probe_seed: 0,
        reserved: [0; 5],
    };
    let mut all_gpu = Vec::with_capacity(BATCH_2 * ROWS_64);
    for b in 0..BATCH_2 {
        let input_slice = &input[b * COLS_1280..(b + 1) * COLS_1280];
        let in_buf = device.new_buffer_with_data(
            input_slice.as_ptr() as *const std::ffi::c_void,
            (COLS_1280 * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = device.new_buffer(
            (ROWS_64 * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let cmd_queue = device.new_command_queue();
        let cmd_buf = cmd_queue.new_command_buffer();
        let mut receipt = zero_receipt();

        dispatcher.dispatch_with_offsets(
            &cmd_buf, &codes_buf, &scales_buf, &biases_buf,
            &in_buf, &out_buf, &params,
            Nf4Tile640Offsets::default(), &mut receipt,
        );
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let ptr = out_buf.contents() as *const f32;
        let result = unsafe { std::slice::from_raw_parts(ptr, ROWS_64).to_vec() };
        all_gpu.extend(result);
    }

    let cos = cosine_similarity(&ref_out, &all_gpu);
    let nrms = nrmse(&ref_out, &all_gpu);
    assert!(cos > 0.99, "NF4 multi-tile cosine {:.8} <= 0.99", cos);
    assert!(nrms < 0.15, "NF4 multi-tile NRMSE {:.6} >= 0.15", nrms);
}

/// INT8 multi-tile GEMV: 1280 columns = 2 tiles per row.
#[test]
fn test_metal_int8_multi_tile() {
    const COLS_1280: usize = 1280;
    const ROWS_64: usize = 64;
    const BATCH_2: usize = 2;

    let mut rng = rand::thread_rng();
    let src: Vec<f32> = (0..ROWS_64 * COLS_1280)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let input: Vec<f32> = (0..BATCH_2 * COLS_1280)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    let (codes, scales, biases) = pack_int8_weights(&src, ROWS_64, COLS_1280);
    let ref_out = ref_matmul(&src, &input, ROWS_64, COLS_1280, BATCH_2);

    let device = Device::system_default().expect("no Metal device");
    let registry: RegistryRef = Arc::new(Mutex::new(KernelRegistry::new(&device)));
    let dispatcher = Int8Tile640GEMVDispatcher::new(registry.clone());

    let codes_buf = device.new_buffer_with_data(
        codes.as_ptr() as *const std::ffi::c_void,
        codes.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let scales_buf = device.new_buffer_with_data(
        scales.as_ptr() as *const std::ffi::c_void,
        (scales.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let biases_buf = device.new_buffer_with_data(
        biases.as_ptr() as *const std::ffi::c_void,
        (biases.len() * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );

    // Multi-tile: page_count = ceil(1280/640) = 2.
    let params = ProjectionParams {
        in_dim: COLS_1280 as u32,
        out_dim: ROWS_64 as u32,
        page_count: 2,
        page_width: TILE_ELEMENTS as u32,
        mode_flags: 0,
        probe_seed: 0,
        reserved: [0; 5],
    };
    let mut all_gpu = Vec::with_capacity(BATCH_2 * ROWS_64);
    for b in 0..BATCH_2 {
        let input_slice = &input[b * COLS_1280..(b + 1) * COLS_1280];
        let in_buf = device.new_buffer_with_data(
            input_slice.as_ptr() as *const std::ffi::c_void,
            (COLS_1280 * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let out_buf = device.new_buffer(
            (ROWS_64 * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let cmd_queue = device.new_command_queue();
        let cmd_buf = cmd_queue.new_command_buffer();
        let mut receipt = zero_receipt();

        dispatcher.dispatch_with_offsets(
            &cmd_buf, &codes_buf, &scales_buf, &biases_buf,
            &in_buf, &out_buf, None, &params,
            Int8Tile640Offsets::default(), &mut receipt,
        );
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        let ptr = out_buf.contents() as *const f32;
        let result = unsafe { std::slice::from_raw_parts(ptr, ROWS_64).to_vec() };
        all_gpu.extend(result);
    }

    let cos = cosine_similarity(&ref_out, &all_gpu);
    let nrms = nrmse(&ref_out, &all_gpu);
    assert!(cos > 0.99, "INT8 multi-tile cosine {:.8} <= 0.99", cos);
    assert!(nrms < 0.02, "INT8 multi-tile NRMSE {:.6} >= 0.02", nrms);
}
