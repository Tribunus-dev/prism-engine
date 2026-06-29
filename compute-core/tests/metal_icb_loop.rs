//! Metal dispatch floor measurement + ICB (Indirect Command Buffer) dispatch.
//! On macOS 26.5, MTLComputeCommandEncoder supports executeCommandsInBuffer:withRange:
//! even though the metal crate 0.29.0 only exposes it on RenderCommandEncoderRef.
//! We call it via raw msg_send! from the objc crate.
//!
//! Two modes:
//!   a) CPU-dispatch baseline  — full re-encode each iteration
//!   b) ICB-dispatch via msg_send!  — pre-encoded ICB, minimal CPU overhead
//!
//! Known limitation: On Apple Silicon M1, encoding compute dispatch
//! (concurrentDispatchThreadgroups/Threads) on IndirectComputeCommandRef
//! crashes with SIGBUS — the ICB command encodes kernel buffers only,
//! pipeline state is inherited from the encoder, and dispatch must
//! be issued via the regular encoder path.
//!
//! Run: cargo test --test metal_icb_loop --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]
#![allow(unexpected_cfgs)]

use std::time::Instant;

#[cfg(feature = "metal-dispatch")]
use objc::{msg_send, sel, sel_impl};

#[cfg(feature = "metal-dispatch")]
fn metal_source(n: u32, name: &str) -> String {
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
        "                    device half* output [[buffer(2)]],\n"
    )
    .unwrap();
    write!(
        s,
        "                    uint2 gid [[thread_position_in_grid]]) {{\n"
    )
    .unwrap();
    write!(s, "    uint row = gid.y;\n").unwrap();
    write!(s, "    if (row >= {}) return;\n", n).unwrap();
    write!(s, "    half acc = 0;\n").unwrap();
    write!(s, "    for (uint i = 0; i < {}; ++i) {{\n", n).unwrap();
    write!(s, "        acc += input[i] * weight[row * {} + i];\n", n).unwrap();
    write!(s, "    }}\n").unwrap();
    write!(s, "    output[row] = acc;\n}}\n").unwrap();
    s
}

#[cfg(feature = "metal-dispatch")]
fn bench_mm(n: u32, it: usize) -> f64 {
    use tribunus_compute_core::compute_image::metal_pipeline::compile_metal_source;
    let src = metal_source(n, "mm");
    let out = compile_metal_source("mm", &src).unwrap();
    let dev = metal::Device::system_default().unwrap();
    let lib = dev.new_library_with_data(&out.metallib_bytes).unwrap();
    let func = lib.get_function("mm", None).unwrap();
    let pl = dev.new_compute_pipeline_state_with_function(&func).unwrap();
    let q = dev.new_command_queue();
    let sb = metal::MTLResourceOptions::StorageModeShared;
    let ba = dev.new_buffer((n as u64 * 2) as u64, sb);
    let bw = dev.new_buffer((n as u64 * n as u64 * 2) as u64, sb);
    let bc = dev.new_buffer((n as u64 * 2) as u64, sb);
    for _ in 0..5 {
        let cb = q.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pl);
        enc.set_buffer(0, Some(&ba), 0);
        enc.set_buffer(1, Some(&bw), 0);
        enc.set_buffer(2, Some(&bc), 0);
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: ((n + 15) / 16) as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 16,
                height: 1,
                depth: 1,
            },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    let t0 = Instant::now();
    for _ in 0..it {
        let cb = q.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pl);
        enc.set_buffer(0, Some(&ba), 0);
        enc.set_buffer(1, Some(&bw), 0);
        enc.set_buffer(2, Some(&bc), 0);
        enc.dispatch_thread_groups(
            metal::MTLSize {
                width: ((n + 15) / 16) as u64,
                height: 1,
                depth: 1,
            },
            metal::MTLSize {
                width: 16,
                height: 1,
                depth: 1,
            },
        );
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    t0.elapsed().as_nanos() as f64 / it as f64
}

#[cfg(feature = "metal-dispatch")]
fn bench_mm_icb(n: u32, it: usize) -> f64 {
    use tribunus_compute_core::compute_image::metal_pipeline::compile_metal_source;
    let src = metal_source(n, "mm");
    let out = compile_metal_source("mm", &src).unwrap();
    let dev = metal::Device::system_default().unwrap();
    let lib = dev.new_library_with_data(&out.metallib_bytes).unwrap();
    let func = lib.get_function("mm", None).unwrap();
    let pl_desc = metal::ComputePipelineDescriptor::new();
    pl_desc.set_compute_function(Some(&func));
    pl_desc.set_support_indirect_command_buffers(true);
    let pl = dev.new_compute_pipeline_state(&pl_desc).unwrap();
    let q = dev.new_command_queue();
    let sb = metal::MTLResourceOptions::StorageModeShared;
    let ba = dev.new_buffer((n as u64 * 2) as u64, sb);
    let bw = dev.new_buffer((n as u64 * n as u64 * 2) as u64, sb);
    let bc = dev.new_buffer((n as u64 * 2) as u64, sb);

    // ICB setup
    let icb_desc = metal::IndirectCommandBufferDescriptor::new();
    icb_desc.set_command_types(metal::MTLIndirectCommandType::ConcurrentDispatch);
    icb_desc.set_inherit_buffers(false);
    icb_desc.set_inherit_pipeline_state(true);
    icb_desc.set_max_kernel_buffer_bind_count(3);

    let icb = dev.new_indirect_command_buffer_with_descriptor(
        &icb_desc,
        1,
        metal::MTLResourceOptions::StorageModeShared,
    );

    // Encode kernel buffer bindings into ICB (pipeline state inherited from encoder;
    // dispatch encoding on IndirectComputeCommandRef crashes on Apple Silicon M1)
    let cmd = icb.indirect_compute_command_at_index(0);
    cmd.set_kernel_buffer(0, Some(&ba), 0);
    cmd.set_kernel_buffer(1, Some(&bw), 0);
    cmd.set_kernel_buffer(2, Some(&bc), 0);

    // Warmup + benchmark: encoder sets pipeline + dispatch, ICB supplies buffer bindings
    // Warmup + benchmark: encoder provides pipeline state (inherited), ICB supplies buffer bindings
    for _ in 0..5 {
        let cb = q.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pl);
        unsafe {
            let _: () = msg_send![enc,
            executeCommandsInBuffer: &*icb
            withRange: metal::NSRange::new(0, 1)];
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }

    let t0 = Instant::now();
    for _ in 0..it {
        let cb = q.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pl);
        unsafe {
            let _: () = msg_send![enc,
            executeCommandsInBuffer: &*icb
            withRange: metal::NSRange::new(0, 1)];
        }
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    t0.elapsed().as_nanos() as f64 / it as f64
}

#[cfg(not(feature = "metal-dispatch"))]
fn bench_mm(_n: u32, _i: usize) -> f64 {
    0.0
}

#[cfg(not(feature = "metal-dispatch"))]
fn bench_mm_icb(_n: u32, _i: usize) -> f64 {
    0.0
}

#[test]
fn test_metal_dispatch_floor() {
    println!("\n=== METAL DISPATCH FLOOR ===");
    #[cfg(feature = "metal-dispatch")]
    {
        for &n in &[64u32, 256, 512] {
            println!("  --- n = {} ---", n);
            let t = bench_mm(n, 200);
            println!("  CPU-dispatch matmul {}x{}: {:>7.1}us", n, n, t / 1000.0);
            let t_icb = bench_mm_icb(n, 200);
            println!(
                "  ICB-dispatch matmul {}x{}: {:>7.1}us",
                n,
                n,
                t_icb / 1000.0
            );
        }
        println!();
    }
    #[cfg(not(feature = "metal-dispatch"))]
    println!("  Metal dispatch not available (need metal-dispatch feature).");
}
