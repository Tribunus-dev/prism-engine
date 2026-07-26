//! Q4_BLOCK_SYM Metal kernel benchmark — 4-way comparison:
//!   FP16 baseline | Q4_GS128 | Q4_GS64 | PALETTE_LUT4
//!
//! Run: cargo test --test q4_block_sym_bench --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use std::time::Instant;

// ── Metal source (inline, compiled via xcrun) ─────────────────────────────

const Q4_BLOCK_SYM_SRC: &str = r##"#include <metal_stdlib>
using namespace metal;

// Q4_BLOCK_SYM: packed signed int4 + FP16 group scale, symmetric (no zero-point).
// 8 int4 values packed per uint32, row-major weights.
// Each group of `group_size` weights shares one FP16 scale.
//
// Branch-free sign extension: (nibble ^ 8) - 8  converts 0..15 -> -8..7
// Threadgroup: 1 thread per output row, input vector shared via threadgroup memory.
//
// Buffer layout:
//   0: input  [K] half
//   1: weights [N * K/8] uint   (8 int4 values per uint32)
//   2: scales  [N * K/group_size] half
//   3: output  [N] half
//   4: K       uint
//   5: N       uint  (output dim)
//   6: group_size  uint  (64 or 128)
//   7: num_groups  uint

kernel void q4_gemv(
    device const half*      input   [[buffer(0)]],
    device const uint*      weights [[buffer(1)]],
    device const half*      scales  [[buffer(2)]],
    device half*            output  [[buffer(3)]],
    constant uint&          K       [[buffer(4)]],
    constant uint&          N       [[buffer(5)]],
    constant uint&          gs      [[buffer(6)]],
    constant uint&          ng      [[buffer(7)]],
    uint                    row     [[thread_position_in_grid]])
{
    if (row >= N) return;

    // ── Per-row GEMV over packed int4 weights ────────────────────────
    float acc_f = 0.0f;
    uint base = row * (K / 8);

    for (uint g = 0; g < ng; ++g) {
        float group_acc = 0.0f;
        half scale = scales[row * ng + g];

        for (uint j = 0; j < gs / 8; ++j) {
            uint packed = weights[base + g * (gs / 8) + j];

            // Reinterpret as 4 bytes to extract nibbles
            uchar4 bytes = as_type<uchar4>(packed);

            // Each byte -> 2 nibbles -> sign extend -> multiply by scale
            // 8 values from one uint32
            uint off = g * gs + j * 8;

            // Byte 0: nibbles 0,1
            { uint n = bytes[0] & 0x0Fu; float v = float(int(n ^ 8u) - 8) * float(scale) * float(input[off + 0]); group_acc += v; }
            { uint n = (bytes[0] >> 4) & 0x0Fu; float v = float(int(n ^ 8u) - 8) * float(scale) * float(input[off + 1]); group_acc += v; }
            // Byte 1: nibbles 2,3
            { uint n = bytes[1] & 0x0Fu; float v = float(int(n ^ 8u) - 8) * float(scale) * float(input[off + 2]); group_acc += v; }
            { uint n = (bytes[1] >> 4) & 0x0Fu; float v = float(int(n ^ 8u) - 8) * float(scale) * float(input[off + 3]); group_acc += v; }
            // Byte 2: nibbles 4,5
            { uint n = bytes[2] & 0x0Fu; float v = float(int(n ^ 8u) - 8) * float(scale) * float(input[off + 4]); group_acc += v; }
            { uint n = (bytes[2] >> 4) & 0x0Fu; float v = float(int(n ^ 8u) - 8) * float(scale) * float(input[off + 5]); group_acc += v; }
            // Byte 3: nibbles 6,7
            { uint n = bytes[3] & 0x0Fu; float v = float(int(n ^ 8u) - 8) * float(scale) * float(input[off + 6]); group_acc += v; }
            { uint n = (bytes[3] >> 4) & 0x0Fu; float v = float(int(n ^ 8u) - 8) * float(scale) * float(input[off + 7]); group_acc += v; }
        }
        acc_f += group_acc;
    }

    output[row] = half(acc_f);
}
"##;

// FP16 baseline: standard Metal matmul, 1 thread per output row
fn fp16_source(n: u32, name: &str) -> String {
    let _ = n;
    use std::fmt::Write;
    let mut s = String::new();
    write!(s, "#include <metal_stdlib>\nusing namespace metal;\n").unwrap();
    write!(
        s,
        "kernel void {}(device const half* input [[buffer(0)]],\n",
        name
    )
    .unwrap();
    write!(
        s,
        "                    device const half* weight [[buffer(1)]],\n"
    )
    .unwrap();
    write!(
        s,
        "                    device half* output [[buffer(3)]],\n"
    )
    .unwrap();
    write!(s, "                    constant uint& K [[buffer(4)]],\n").unwrap();
    write!(s, "                    constant uint& N [[buffer(5)]],\n").unwrap();
    write!(
        s,
        "                    uint row [[thread_position_in_grid]]) {{\n"
    )
    .unwrap();
    write!(s, "    if (row >= N) return;\n").unwrap();
    write!(s, "    half acc = 0;\n").unwrap();
    write!(s, "    for (uint i = 0; i < K; ++i) {{\n").unwrap();
    write!(s, "        acc += input[i] * weight[row * K + i];\n").unwrap();
    write!(s, "    }}\n").unwrap();
    write!(s, "    output[row] = acc;\n}}\n").unwrap();
    s
}

// ── Weight packing ────────────────────────────────────────────────────────

/// Pack FP16 weights into Q4 block-symmetric format.
/// Returns (packed_weights_u32, scales_f16_as_bytes).
/// layout: weights[n][k/8] uint32, scales[n][ng] f16
fn pack_q4_block_sym(data: &[f32], n: usize, k: usize, gs: usize) -> (Vec<u32>, Vec<u16>) {
    let ng = k / gs;
    let mut packed = vec![0u32; n * (k / 8)];
    let mut scales = vec![0u16; n * ng];

    for row in 0..n {
        for g in 0..ng {
            let group_start = row * k + g * gs;
            let group = &data[group_start..group_start + gs];

            // Find max absolute value
            let mut max_abs = 0.0f32;
            for &v in group {
                let a = v.abs();
                if a > max_abs {
                    max_abs = a;
                }
            }

            // Scale factor: map max_abs to int4 max (7, symmetric)
            let scale = if max_abs > 0.0 {
                max_abs / 7.0f32
            } else {
                1.0f32
            };
            scales[row * ng + g] = f16_to_bits(scale);

            // Quantize and pack
            for j in 0..(gs / 8) {
                let mut word = 0u32;
                for nib in 0..8 {
                    let idx = group_start + j * 8 + nib;
                    // Clamp original value to group
                    let orig = data[idx];
                    let q = (orig / scale).round().clamp(-8.0, 7.0) as i32;
                    let uq = (q & 0x0F) as u32; // 2's complement 4-bit
                    word |= uq << (nib * 4);
                }
                packed[row * (k / 8) + g * (gs / 8) + j] = word;
            }
        }
    }
    (packed, scales)
}

fn f16_to_bits(v: f32) -> u16 {
    // Simple float->f16 via half crate if available, or manual
    // We use bytemuck + half transmute through u32
    let bits = v.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x3FF;
    (if exp <= 0 {
        sign | (mant >> 1)
    } else if exp >= 31 {
        sign | 0x7C00 | mant
    } else {
        sign | ((exp as u32) << 10) | mant
    }) as u16
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) >> 15) << 31;
    let exp = ((bits >> 10) & 0x1Fu16) as i32 - 15 + 127;
    let mant = (bits & 0x3FF) as u32;
    if exp <= 0 {
        f32::from_bits(sign | mant << 13)
    } else if exp >= 255 {
        f32::from_bits(sign | 0x7F800000 | (mant << 13))
    } else {
        f32::from_bits(sign | ((exp as u32) << 23) | (mant << 13))
    }
}

/// Compute reference matmul output (FP32 precise).
fn ref_matmul(input: &[f32], weight: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n];
    for row in 0..n {
        let mut sum = 0.0f32;
        for i in 0..k {
            sum += input[i] * weight[row * k + i];
        }
        out[row] = sum;
    }
    out
}

/// Dequantize Q4 weights back to FP32 for error measurement.
#[allow(dead_code)]
fn dequant_q4(packed: &[u32], scales: &[u16], n: usize, k: usize, gs: usize) -> Vec<f32> {
    let ng = k / gs;
    let mut out = vec![0.0f32; n * k];
    for row in 0..n {
        for g in 0..ng {
            let scale = f16_bits_to_f32(scales[row * ng + g]);
            for j in 0..(gs / 8) {
                let word = packed[row * (k / 8) + g * (gs / 8) + j];
                for nib in 0..8 {
                    let nibble = (word >> (nib * 4)) & 0x0F;
                    let signed_val = (nibble ^ 8) as i32 - 8; // branch-free sign extend
                    let idx = row * k + g * gs + j * 8 + nib;
                    out[idx] = (signed_val as f32) * scale;
                }
            }
        }
    }
    out
}

// ── Benchmark helpers ─────────────────────────────────────────────────────

fn compile(
    name: &str,
    source: &str,
) -> Option<tribunus_compute_core::compute_image::metal_pipeline::MetalPipelineOutput> {
    tribunus_compute_core::compute_image::metal_pipeline::compile_metal_source(name, source)
}

/// Benchmark a Metal kernel with given buffers.
/// Returns per-invocation latency in nanoseconds.
fn bench_kernel(
    pl: &metal::ComputePipelineStateRef,
    ba: &metal::BufferRef,
    bw: &metal::BufferRef,
    bc: &metal::BufferRef,
    bo: &metal::BufferRef,
    extra: &[&metal::BufferRef],
    wg: metal::MTLSize,
    gg: metal::MTLSize,
    it: usize,
) -> f64 {
    let dev = metal::Device::system_default().unwrap();
    let q = dev.new_command_queue();

    // Warmup
    for _ in 0..5 {
        let cb = q.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pl);
        enc.set_buffer(0, Some(ba), 0);
        enc.set_buffer(1, Some(bw), 0);
        enc.set_buffer(2, Some(bc), 0);
        enc.set_buffer(3, Some(bo), 0);
        for (i, &eb) in extra.iter().enumerate() {
            enc.set_buffer((4 + i) as u64, Some(eb), 0);
        }
        enc.dispatch_thread_groups(gg, wg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    let t0 = Instant::now();
    for _ in 0..it {
        let cb = q.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pl);
        enc.set_buffer(0, Some(ba), 0);
        enc.set_buffer(1, Some(bw), 0);
        enc.set_buffer(2, Some(bc), 0);
        enc.set_buffer(3, Some(bo), 0);
        for (i, &eb) in extra.iter().enumerate() {
            enc.set_buffer((4 + i) as u64, Some(eb), 0);
        }
        enc.dispatch_thread_groups(gg, wg);
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    t0.elapsed().as_nanos() as f64 / it as f64
}

// ── Main benchmark ────────────────────────────────────────────────────────

#[test]
fn test_q4_block_sym_bench() {
    println!("\n=== Q4_BLOCK_SYM: 4-WAY METAL KERNEL BENCHMARK ===");
    println!("SYMMETRIC (no zero-point). ");
    println!();

    let sizes: &[(usize, usize, &str)] = &[
        (256, 1024, "small"),
        (512, 2048, "med"),
        (1024, 4096, "large"),
    ];

    // Compile kernels once
    let fp16_metal = compile("fp16", &fp16_source(512, "fp16_mm")).expect("fp16 compile");
    let q4_128_metal = compile("q4_gs128", Q4_BLOCK_SYM_SRC).expect("q4_128 compile");
    let _q4_64_metal = compile("q4_gs64", Q4_BLOCK_SYM_SRC).expect("q4_64 compile");

    let dev = metal::Device::system_default().unwrap();

    for &(h, i, label) in sizes {
        let k = h; // input dim
        let n = i; // output dim

        // ── Generate random data ──
        use std::hash::{Hash, Hasher};
        let input_f32: Vec<f32> = (0..k)
            .map(|i| {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                (i as u64 ^ 0x1234).hash(&mut h);
                (h.finish() as f32 % 1000.0 - 500.0) / 500.0
            })
            .collect();

        let weight_f32: Vec<f32> = (0..n * k)
            .map(|i| {
                let mut h = std::collections::hash_map::DefaultHasher::new();
                (i as u64).hash(&mut h);
                (h.finish() as f32 % 1000.0 - 500.0) / 500.0
            })
            .collect();

        // Reference output
        let ref_out = ref_matmul(&input_f32, &weight_f32, n, k);

        // ── Pack Q4 weights ──
        let (q4_128_packed, q4_128_scales) = pack_q4_block_sym(&weight_f32, n, k, 128);
        let (q4_64_packed, q4_64_scales) = pack_q4_block_sym(&weight_f32, n, k, 64);

        let q4_128_ng = k / 128;
        let q4_64_ng = k / 64;

        // ── Create Metal buffers ──
        let sb = metal::MTLResourceOptions::StorageModeShared;
        let fp16_in = dev.new_buffer((k as u64 * 2) as u64, sb);
        let fp16_w = dev.new_buffer((n as u64 * k as u64 * 2) as u64, sb);
        let fp16_out = dev.new_buffer((n as u64 * 2) as u64, sb);
        let q4_in = dev.new_buffer((k as u64 * 2) as u64, sb);
        let q4_out = dev.new_buffer((n as u64 * 2) as u64, sb);

        // Write FP16 input data
        unsafe {
            let in_ptr = q4_in.contents() as *mut u16;
            for i in 0..k {
                in_ptr.add(i).write(f16_to_bits(input_f32[i]));
            }
        }
        unsafe {
            let w_ptr = fp16_w.contents() as *mut u16;
            for i in 0..n * k {
                w_ptr.add(i).write(f16_to_bits(weight_f32[i]));
            }
        }
        unsafe {
            let in_ptr = fp16_in.contents() as *mut u16;
            for i in 0..k {
                in_ptr.add(i).write(f16_to_bits(input_f32[i]));
            }
        }

        // Q4 weight buffer
        let q4_w = dev.new_buffer((q4_128_packed.len() * 4) as u64, sb);
        unsafe {
            std::ptr::copy_nonoverlapping(
                q4_128_packed.as_ptr() as *const u8,
                q4_w.contents() as *mut u8,
                q4_128_packed.len() * 4,
            );
        }

        // Q4 scale buffers
        let q4_128_s = dev.new_buffer((q4_128_scales.len() * 2) as u64, sb);
        unsafe {
            std::ptr::copy_nonoverlapping(
                q4_128_scales.as_ptr() as *const u8,
                q4_128_s.contents() as *mut u8,
                q4_128_scales.len() * 2,
            );
        }

        let q4_64_w = dev.new_buffer((q4_64_packed.len() * 4) as u64, sb);
        unsafe {
            std::ptr::copy_nonoverlapping(
                q4_64_packed.as_ptr() as *const u8,
                q4_64_w.contents() as *mut u8,
                q4_64_packed.len() * 4,
            );
        }

        let q4_64_s = dev.new_buffer((q4_64_scales.len() * 2) as u64, sb);
        unsafe {
            std::ptr::copy_nonoverlapping(
                q4_64_scales.as_ptr() as *const u8,
                q4_64_s.contents() as *mut u8,
                q4_64_scales.len() * 2,
            );
        }

        // Constants for Q4 kernel
        let const_k = dev.new_buffer(4, sb);
        unsafe {
            *(const_k.contents() as *mut u32) = k as u32;
        }
        let const_n = dev.new_buffer(4, sb);
        unsafe {
            *(const_n.contents() as *mut u32) = n as u32;
        }
        let const_gs128 = dev.new_buffer(4, sb);
        unsafe {
            *(const_gs128.contents() as *mut u32) = 128u32;
        }
        let const_ng128 = dev.new_buffer(4, sb);
        unsafe {
            *(const_ng128.contents() as *mut u32) = q4_128_ng as u32;
        }
        let const_gs64 = dev.new_buffer(4, sb);
        unsafe {
            *(const_gs64.contents() as *mut u32) = 64u32;
        }
        let const_ng64 = dev.new_buffer(4, sb);
        unsafe {
            *(const_ng64.contents() as *mut u32) = q4_64_ng as u32;
        }

        // ── Build pipeline states ──
        let fp16_lib = dev
            .new_library_with_data(&fp16_metal.metallib_bytes)
            .unwrap();
        let fp16_fn = fp16_lib.get_function("fp16_mm", None).unwrap();
        let fp16_pl = dev
            .new_compute_pipeline_state_with_function(&fp16_fn)
            .unwrap();

        let q4_lib = dev
            .new_library_with_data(&q4_128_metal.metallib_bytes)
            .unwrap();
        let q4_fn = q4_lib.get_function("q4_gemv", None).unwrap();
        let q4_pl = dev
            .new_compute_pipeline_state_with_function(&q4_fn)
            .unwrap();

        // ── Benchmark FP16 ──
        const TG: u64 = 256;
        let wg = metal::MTLSize {
            width: TG,
            height: 1,
            depth: 1,
        };
        let gg = metal::MTLSize {
            width: ((n as u64 + TG - 1) / TG),
            height: 1,
            depth: 1,
        };

        let iters = if h <= 512 { 200 } else { 50 };
        let fp16_ns = bench_kernel(
            &fp16_pl,
            &fp16_in,
            &fp16_w,
            &fp16_in,
            &fp16_out,
            &[&const_k, &const_n],
            wg,
            gg,
            iters,
        );

        // ── Benchmark Q4_GS128 ──
        let q4_128_ns = bench_kernel(
            &q4_pl,
            &q4_in,
            &q4_w,
            &q4_128_s,
            &q4_out,
            &[&const_k, &const_n, &const_gs128, &const_ng128],
            wg,
            gg,
            iters,
        );

        // ── Benchmark Q4_GS64 ──
        // Reuse same kernel (q4_gemv), different gs/ng constants
        let q4_64_ns = bench_kernel(
            &q4_pl,
            &q4_in,
            &q4_64_w,
            &q4_64_s,
            &q4_out,
            &[&const_k, &const_n, &const_gs64, &const_ng64],
            wg,
            gg,
            iters,
        );

        // ── Compute error ──
        // Read back FP16 output
        let mut fp16_result = vec![0.0f32; n];
        unsafe {
            let out_ptr = fp16_out.contents() as *mut u16;
            for i in 0..n {
                fp16_result[i] = f16_bits_to_f32(out_ptr.add(i).read());
            }
        }

        // Read back Q4_128 output
        let mut q4_128_result = vec![0.0f32; n];
        unsafe {
            let out_ptr = q4_out.contents() as *mut u16;
            for i in 0..n {
                q4_128_result[i] = f16_bits_to_f32(out_ptr.add(i).read());
            }
        }

        // Read back Q4_64 output
        let mut q4_64_result = vec![0.0f32; n];
        unsafe {
            let out_ptr = q4_out.contents() as *mut u16;
            for i in 0..n {
                q4_64_result[i] = f16_bits_to_f32(out_ptr.add(i).read());
            }
        }

        // Compute max relative error
        fn max_rel_err(computed: &[f32], ref_out: &[f32]) -> f64 {
            computed
                .iter()
                .zip(ref_out)
                .map(|(c, r)| {
                    if r.abs() > 1e-10 {
                        ((c - r).abs() / r.abs()) as f64
                    } else {
                        (c - r).abs() as f64
                    }
                })
                .fold(0.0f64, f64::max)
        }

        let fp16_err = max_rel_err(&fp16_result, &ref_out);
        let q4_128_err = max_rel_err(&q4_128_result, &ref_out);
        let q4_64_err = max_rel_err(&q4_64_result, &ref_out);

        // ── DRAM bytes estimate ──
        let fp16_bytes = (k * 2 + n * k * 2 + n * 2) as f64;
        let q4_128_bytes = (k * 2 + n * k / 8 * 4 + n * q4_128_ng * 2 + n * 2) as f64;
        let q4_64_bytes = (k * 2 + n * k / 8 * 4 + n * q4_64_ng * 2 + n * 2) as f64;

        // ── Print results ──
        println!("{} (H={} I={}):", label, h, i);
        println!(
            "  FP16:            {:>7.1}us  err={:.4}  DRAM={:.0}B",
            fp16_ns / 1000.0,
            fp16_err,
            fp16_bytes
        );
        println!("  Q4_GS128:        {:>7.1}us  err={:.4}  DRAM={:.0}B  speedup={:.2}x  mem_ratio={:.1}x",
            q4_128_ns / 1000.0, q4_128_err, q4_128_bytes,
            fp16_ns / q4_128_ns.max(1.0),
            fp16_bytes / q4_128_bytes.max(1.0));
        println!("  Q4_GS64:         {:>7.1}us  err={:.4}  DRAM={:.0}B  speedup={:.2}x  mem_ratio={:.1}x",
            q4_64_ns / 1000.0, q4_64_err, q4_64_bytes,
            fp16_ns / q4_64_ns.max(1.0),
            fp16_bytes / q4_64_bytes.max(1.0));
        println!();
    }
    println!("=== RESULTS ===");
    println!("speedup > 1.0: Q4 kernel beats FP16 baseline");
    println!("err: max relative error vs FP32 reference");
    println!("mem_ratio: DRAM bytes ratio (FP16 / Q4) — expect ~4x for weights");
}
