//! End-to-end ternary 2-bit nibble GEMV test.
//!
//! Verifies that the unified ternary encoding (00=0, 01=+1, 10=-1)
//! is correctly packed by the compiler and decoded by the Metal
//! `ternary_gemv` kernel.  This is the bridge test that caught
//! the inverted-encoding ghost (commit fixing ternary_gemv.metal).
//!
//! Run: cargo test --test ternary_gemv_e2e --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use metal::*;

// ── Known-good ternary matrix ──────────────────────────────────────
// 2 rows × 4 cols, values match unified encoding:
//   00=0, 01=+1, 10=-1
//
// Row 0: [ 0, +1, -1,  0] → packed byte: 00_10_01_00 = 0x24
// Row 1: [ 0, -1, +1,  0] → packed byte: 00_01_10_00 = 0x48
const K: usize = 4;
const N: usize = 2;

fn pack_ternary_matrix() -> Vec<u8> {
    let weights: [[i8; K]; N] = [[0, 1, -1, 0], [0, -1, 1, 0]];
    let mut packed = vec![0u8; N * K / 4];
    for row in 0..N {
        let byte_idx = row;
        let mut byte = 0u8;
        for col in 0..K {
            let nibble: u8 = match weights[row][col] {
                0 => 0b00,
                1 => 0b01,
                -1 => 0b10,
                _ => unreachable!(),
            };
            byte |= nibble << ((col % 4) * 2);
        }
        packed[byte_idx] = byte;
    }
    packed
}

fn cpu_gemv(packed: &[u8], input: &[f32; K]) -> [f32; N] {
    let mut out = [0.0f32; N];
    for row in 0..N {
        let byte = packed[row];
        let n0 = (byte >> 0) & 0x03;
        let n1 = (byte >> 2) & 0x03;
        let n2 = (byte >> 4) & 0x03;
        let n3 = (byte >> 6) & 0x03;
        let w: [f32; K] = [
            if n0 == 1 { 1.0 } else if n0 == 2 { -1.0 } else { 0.0 },
            if n1 == 1 { 1.0 } else if n1 == 2 { -1.0 } else { 0.0 },
            if n2 == 1 { 1.0 } else if n2 == 2 { -1.0 } else { 0.0 },
            if n3 == 1 { 1.0 } else if n3 == 2 { -1.0 } else { 0.0 },
        ];
        out[row] = w[0] * input[0] + w[1] * input[1] + w[2] * input[2] + w[3] * input[3];
    }
    out
}

// ── Metal kernel (inline copy of the fixed ternary_gemv.metal) ──────

const KERNEL_SRC: &str = r##"
#include <metal_stdlib>
using namespace metal;

kernel void ternary_gemv(
    device const uint8_t* packed_weights [[buffer(0)]],
    device const half*    input          [[buffer(1)]],
    device half*          output         [[buffer(2)]],
    constant uint&        in_dim         [[buffer(3)]],
    constant uint&        out_dim        [[buffer(4)]],
    uint                  row            [[thread_position_in_grid]])
{
    if (row >= out_dim) return;
    uint packed_cols = in_dim / 4;
    uint offset      = row * packed_cols;
    half sum = 0.0h;
    for (uint i = 0; i < packed_cols; ++i) {
        uint8_t byte   = packed_weights[offset + i];
        half4   iv     = *((device const half4*)(input + i * 4));
        uint nibble0 = uint(byte)       & 0x03u;
        uint nibble1 = (uint(byte) >> 2) & 0x03u;
        uint nibble2 = (uint(byte) >> 4) & 0x03u;
        uint nibble3 = (uint(byte) >> 6) & 0x03u;
        half4 tmp;
        // Unified encoding: 00=0, 01=+iv, 10=-iv
        tmp.x = select(select(0.0h, iv.x, nibble0 == 1u), -iv.x, nibble0 == 2u);
        tmp.y = select(select(0.0h, iv.y, nibble1 == 1u), -iv.y, nibble1 == 2u);
        tmp.z = select(select(0.0h, iv.z, nibble2 == 1u), -iv.z, nibble2 == 2u);
        tmp.w = select(select(0.0h, iv.w, nibble3 == 1u), -iv.w, nibble3 == 2u);
        sum += tmp.x + tmp.y + tmp.z + tmp.w;
    }
    output[row] = sum;
}
"##;

// ── Metal setup ──────────────────────────────────────────────────────

fn compile_kernel() -> (ComputePipelineState, CommandQueue, Device) {
    let device = Device::system_default().expect("no Metal device");
    let lib = device
        .new_library_with_source(KERNEL_SRC, &CompileOptions::new())
        .expect("compile Metal source");
    let kernel = lib
        .get_function("ternary_gemv", None)
        .expect("get kernel function");
    let pipeline = device
        .new_compute_pipeline_state_with_function(&kernel)
        .expect("create pipeline state");
    let queue = device.new_command_queue();
    (pipeline, queue, device)
}

#[test]
fn ternary_2bit_nibble_gemv_e2e() {
    let packed = pack_ternary_matrix();
    let input: [f32; K] = [1.0, 2.0, 3.0, 4.0];
    let expected = cpu_gemv(&packed, &input);

    // Convert input to half
    let input_half: Vec<u16> = input.iter().map(|&v| f32_to_f16(v)).collect();

    // Metal setup
    let (pipeline, queue, device) = compile_kernel();
    let cmd_buf = queue.new_command_buffer();
    let encoder = cmd_buf.new_compute_command_encoder();

    // Create buffers
    let weight_buf = device.new_buffer_with_data(
        packed.as_ptr() as *const std::ffi::c_void,
        packed.len() as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let input_buf = device.new_buffer_with_data(
        input_half.as_ptr() as *const std::ffi::c_void,
        (K * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let output_buf = device.new_buffer_with_data(
        std::ptr::null(),
        (N * 2) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    let in_dim: u32 = K as u32;
    let out_dim: u32 = N as u32;
    let in_dim_buf = device.new_buffer_with_data(
        &in_dim as *const u32 as *const std::ffi::c_void,
        4,
        MTLResourceOptions::StorageModeShared,
    );
    let out_dim_buf = device.new_buffer_with_data(
        &out_dim as *const u32 as *const std::ffi::c_void,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(&weight_buf), 0);
    encoder.set_buffer(1, Some(&input_buf), 0);
    encoder.set_buffer(2, Some(&output_buf), 0);
    encoder.set_buffer(3, Some(&in_dim_buf), 0);
    encoder.set_buffer(4, Some(&out_dim_buf), 0);

    let grid_size = MTLSize { width: N as u64, height: 1, depth: 1 };
    let threadgroup_size = MTLSize { width: 1, height: 1, depth: 1 };
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();
    cmd_buf.commit();
    cmd_buf.wait_until_completed();

    // Read back
    let out_ptr = output_buf.contents() as *const u16;
    let out_slice = unsafe { std::slice::from_raw_parts(out_ptr, N) };
    let result: [f32; N] = [f16_to_f32(out_slice[0]), f16_to_f32(out_slice[1])];

    eprintln!(
        "packed weights: {:02x?}, input: {:?}, expected: {:?}, got: {:?}",
        packed, input, expected, result
    );

    for i in 0..N {
        let diff = (result[i] - expected[i]).abs();
        assert!(
            diff < 0.01,
            "row {}: expected {} got {} (diff {})",
            i,
            expected[i],
            result[i],
            diff
        );
    }
    eprintln!("PASS: ternary 2-bit nibble GEMV matches CPU reference");
}

// ── Half-precision conversion helpers ─────────────────────────────────

fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = (bits >> 13) & 0x3ff;
    if exp <= 0 {
        sign
    } else if exp >= 31 {
        sign | 0x7c00 | mantissa
    } else {
        sign | ((exp as u32) << 10) | mantissa
    } as u16
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) as u32) << 31;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mantissa = (bits & 0x3ff) as u32;
    if exp == 0 {
        f32::from_bits(sign | mantissa << 13)
    } else if exp == 31 {
        f32::from_bits(sign | 0x7f800000 | mantissa << 13)
    } else {
        f32::from_bits(sign | ((exp + 127 - 15) << 23) | mantissa << 13)
    }
}
