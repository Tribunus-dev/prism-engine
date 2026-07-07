//! Metal NF4/INT8 tile640 GEMV conformance test.
//!
//! Packs random weight matrices, dispatches NF4 and INT8 Metal kernels,
//! and verifies outputs match CPU reference within tolerance.
//!
//! Run:  cargo test --test metal_nf4_int8_conformance --features metal-dispatch,prism-backend -- --nocapture
//!
//! Note: Scaled NF4 (reduction-axis sidecar) is not yet supported by the
//! NF4 Metal kernel — format 1 validation is CPU-only in test_mixed_artifact_e2e.

#![cfg(all(target_os = "macos", feature = "metal-dispatch"))]

use metal::*;
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
use tribunus_compute_core::nf4tile640::{pack_int8_weights, pack_nf4_weights, TILE_ELEMENTS};

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
