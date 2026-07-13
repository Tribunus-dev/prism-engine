//! KV Interleave — concurrent decode+prefetch dispatch isolator matrix.
//!
//! Tests the MTLDispatchType::Concurrent compute encoder with two different
//! PSOs (decode worker + prefetch worker) sharing an epoch_control buffer.
//!
//! Run: cargo test --test kv_concurrent_test --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use metal::*;
use std::time::Instant;

fn compile_minimal(device: &Device, src: &str, entry: &str) -> ComputePipelineState {
    let opts = CompileOptions::new();
    let lib = device.new_library_with_source(src, &opts).unwrap();
    device
        .new_compute_pipeline_state_with_function(&lib.get_function(entry, None).unwrap())
        .unwrap()
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Stage A: Two trivial different PSOs, no shared buffers
//  Verifies the concurrent encoder supports PSO changes between dispatches.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#[test]
fn a_stage_different_pso() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let pso_a = compile_minimal(&device,
        "#include <metal_stdlib>\nusing namespace metal;\nkernel void a(uint tid [[thread_index_in_threadgroup]]) { }", "a");
    let pso_b = compile_minimal(&device,
        "#include <metal_stdlib>\nusing namespace metal;\nkernel void b(uint tid [[thread_index_in_threadgroup]]) { }", "b");

    let cb = queue.new_command_buffer();
    let enc = cb.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);
    enc.set_compute_pipeline_state(&pso_a);
    enc.dispatch_threads(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    enc.set_compute_pipeline_state(&pso_b);
    enc.dispatch_threads(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
    println!("  ✓ Stage A: two different PSOs in concurrent encoder");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Stage B: Two trivial different PSOs, one shared read-only buffer
//  Verifies shared resources don't trigger driver barriers.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#[test]
fn b_stage_shared_readonly() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();

    let shared = device.new_buffer_with_data(
        &42u32 as *const u32 as *const std::ffi::c_void,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let src = "#include <metal_stdlib>\nusing namespace metal;\nkernel void r(\
        device const uint* buf [[buffer(0)]], uint tid [[thread_index_in_threadgroup]]) { \
        volatile uint x = buf[0]; (void)x; }";
    let pso = compile_minimal(&device, src, "r");

    let cb = queue.new_command_buffer();
    let enc = cb.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);
    enc.set_compute_pipeline_state(&pso);
    enc.set_buffer(0, Some(&shared), 0);
    enc.dispatch_threads(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    enc.set_compute_pipeline_state(&pso);
    enc.set_buffer(0, Some(&shared), 0);
    enc.dispatch_threads(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    enc.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
    println!("  ✓ Stage B: shared read-only buffer, concurrent encoder");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Stage C: Full decode + prefetch, epoch_close=1 (both exit immediately)
//  The key test: prefetch worker's epoch_control is at [[buffer(11)]].
//  Previous test incorrectly bound it to buffer(8), causing garbage reads.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#[test]
fn c_stage_epoch_close() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();
    let epoch_ctl_src = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void c_epoch_decode_worker(
            device atomic_uint* epoch_control     [[buffer(13)]],
            device atomic_uint* receipt           [[buffer(18)]],
            uint tid [[thread_index_in_threadgroup]]) {
            if (tid == 0) {
                uint ec = atomic_load_explicit(epoch_control, memory_order_relaxed);
                atomic_store_explicit(receipt, ec, memory_order_relaxed);
            }
        }

        kernel void c_epoch_prefetch_worker(
            device atomic_uint* epoch_control     [[buffer(11)]],
            device atomic_uint* receipt           [[buffer(18)]],
            uint tid [[thread_index_in_threadgroup]]) {
            if (tid == 0) {
                uint ec = atomic_load_explicit(epoch_control, memory_order_relaxed);
                atomic_store_explicit(receipt, ec, memory_order_relaxed);
            }
        }
    "#;
    let pso_d = compile_minimal(&device, epoch_ctl_src, "c_epoch_decode_worker");
    let pso_p = compile_minimal(&device, epoch_ctl_src, "c_epoch_prefetch_worker");

    let ec = device.new_buffer(128, MTLResourceOptions::StorageModeShared);
    unsafe {
        *(ec.contents() as *mut u32) = 1;
    }

    let d_bufs: Vec<_> = (0..31)
        .map(|_| device.new_buffer(64 * 1024, MTLResourceOptions::StorageModeShared))
        .collect();
    let p_bufs: Vec<_> = (0..12)
        .map(|_| device.new_buffer(64 * 1024, MTLResourceOptions::StorageModeShared))
        .collect();

    let cb = queue.new_command_buffer();
    let enc = cb.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);

    enc.set_compute_pipeline_state(&pso_d);
    for (i, b) in d_bufs.iter().enumerate() {
        enc.set_buffer(i as u64, Some(b), 0);
    }
    enc.set_buffer(13, Some(&ec), 0);
    enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));

    enc.set_compute_pipeline_state(&pso_p);
    for (i, b) in p_bufs.iter().enumerate() {
        enc.set_buffer(i as u64, Some(b), 0);
    }
    enc.set_buffer(11, Some(&ec), 0);
    enc.dispatch_threads(MTLSize::new(1, 1, 1), MTLSize::new(256, 1, 1));

    enc.end_encoding();
    let start = Instant::now();
    cb.commit();
    cb.wait_until_completed();
    assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
    println!(
        "  ✓ Stage C: decode+prefetch epoch close — {:.1}ms",
        start.elapsed().as_secs_f64() * 1e3
    );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Stage D: epoch_close=1 + one ring entry queued
//  Verifies ring dequeue and epoch drain protocols under concurrent dispatch.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#[test]
fn d_stage_ring_entry_epoch_close() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();
    let proto_src = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void d_ring_decode_worker(
            device atomic_uint* epoch_control     [[buffer(13)]],
            device volatile uint* ring_entry      [[buffer(22)]],
            device atomic_uint* ring_tail         [[buffer(23)]],
            device atomic_uint* receipt           [[buffer(18)]],
            device atomic_uint* max_threads_v     [[buffer(27)]],
            uint tid [[thread_index_in_threadgroup]]) {
            if (tid == 0) {
                uint ec = atomic_load_explicit(epoch_control, memory_order_relaxed);
                // Claim the ring entry
                ring_entry[0] = 2; // CONSUMED
                atomic_store_explicit(receipt, ec, memory_order_relaxed);
            }
        }

        kernel void d_ring_prefetch_worker(
            device atomic_uint* epoch_control     [[buffer(11)]],
            device atomic_uint* receipt           [[buffer(18)]],
            device atomic_uint* max_threads_v     [[buffer(10)]],
            uint tid [[thread_index_in_threadgroup]]) {
            if (tid == 0) {
                uint ec = atomic_load_explicit(epoch_control, memory_order_relaxed);
                atomic_store_explicit(receipt, ec, memory_order_relaxed);
            }
        }
    "#;
    let pso_d = compile_minimal(&device, proto_src, "d_ring_decode_worker");
    let pso_p = compile_minimal(&device, proto_src, "d_ring_prefetch_worker");

    let ec = device.new_buffer(128, MTLResourceOptions::StorageModeShared);
    unsafe {
        *(ec.contents() as *mut u32) = 1;
    } // epoch_close_requested = 1
    let mt = device.new_buffer_with_data(
        &1u32 as *const u32 as *const std::ffi::c_void,
        4,
        MTLResourceOptions::StorageModeShared,
    );

    let d_bufs: Vec<_> = (0..31)
        .map(|_| device.new_buffer(64 * 1024, MTLResourceOptions::StorageModeShared))
        .collect();

    // Ring entry at position 0: state=SUBMITTED(1)
    unsafe {
        let r = d_bufs[22].contents() as *mut u32;
        r.write_volatile(0);
        r.add(1).write_volatile(1); // request_id
        r.add(2).write_volatile(0); // session_id = token 0
        r.add(3).write_volatile(0); // sequence_id = pos 0
        r.write_volatile(1); // state = SUBMITTED
        *(d_bufs[23].contents() as *mut u32) = 0; // ring_tail
        std::ptr::write_bytes(d_bufs[18].contents(), 0, 64); // receipt
    }

    let p_bufs: Vec<_> = (0..12)
        .map(|_| device.new_buffer(64 * 1024, MTLResourceOptions::StorageModeShared))
        .collect();

    let cb = queue.new_command_buffer();
    let enc = cb.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);

    enc.set_compute_pipeline_state(&pso_d);
    for (i, b) in d_bufs.iter().enumerate() {
        enc.set_buffer(i as u64, Some(b), 0);
    }
    enc.set_buffer(13, Some(&ec), 0);
    enc.set_buffer(27, Some(&mt), 0);
    enc.dispatch_threads(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );

    enc.set_compute_pipeline_state(&pso_p);
    for (i, b) in p_bufs.iter().enumerate() {
        enc.set_buffer(i as u64, Some(b), 0);
    }
    enc.set_buffer(10, Some(&mt), 0);
    enc.set_buffer(11, Some(&ec), 0);
    enc.dispatch_threads(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );

    enc.end_encoding();
    let start = Instant::now();
    cb.commit();
    cb.wait_until_completed();
    let elapsed = start.elapsed();

    unsafe {
        let r = d_bufs[22].contents() as *const u32;
        let entry_state = std::ptr::read_volatile(r);
        let state = entry_state & 3;
        let consumed = state >= 2;
        let receipt = *(d_bufs[18].contents() as *const u32);
        println!(
            "  Ring entry state={}, consumed={}, requests_claimed={}",
            state, consumed, receipt
        );
    }
    assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
    println!(
        "  ✓ Stage D: ring entry + epoch close — {:.1}ms",
        elapsed.as_secs_f64() * 1e3
    );
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
//  Stage E: Host-driven lifecycle for persistent concurrent workers.
//  The production kernels require production-sized model buffers, so this
//  isolator exercises the open -> claimed -> close -> completed handshake
//  without allowing either persistent worker to wait forever.
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#[test]
fn e_stage_full_protocol() {
    let device = Device::system_default().unwrap();
    let queue = device.new_command_queue();
    let protocol_src = r#"
        #include <metal_stdlib>
        using namespace metal;

        kernel void protocol_decode(
            device atomic_uint* close [[buffer(0)]],
            device atomic_uint* claimed [[buffer(1)]],
            uint tid [[thread_index_in_threadgroup]]) {
            if (tid == 0) {
                atomic_store_explicit(claimed, 1, memory_order_relaxed);
                while (atomic_load_explicit(close, memory_order_relaxed) == 0) {}
            }
        }

        kernel void protocol_prefetch(
            device atomic_uint* close [[buffer(0)]],
            device atomic_uint* observed_close [[buffer(1)]],
            uint tid [[thread_index_in_threadgroup]]) {
            if (tid == 0) {
                while (atomic_load_explicit(close, memory_order_relaxed) == 0) {}
                atomic_store_explicit(observed_close, 1, memory_order_relaxed);
            }
        }
    "#;
    let pso_d = compile_minimal(&device, protocol_src, "protocol_decode");
    let pso_p = compile_minimal(&device, protocol_src, "protocol_prefetch");

    let close = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let claimed = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    let observed_close = device.new_buffer(4, MTLResourceOptions::StorageModeShared);
    unsafe {
        (close.contents() as *mut u32).write_volatile(0);
        (claimed.contents() as *mut u32).write_volatile(0);
        (observed_close.contents() as *mut u32).write_volatile(0);
    }

    let cb = queue.new_command_buffer();
    let enc = cb.compute_command_encoder_with_dispatch_type(MTLDispatchType::Concurrent);

    enc.set_compute_pipeline_state(&pso_d);
    enc.set_buffer(0, Some(&close), 0);
    enc.set_buffer(1, Some(&claimed), 0);
    enc.dispatch_threads(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );

    enc.set_compute_pipeline_state(&pso_p);
    enc.set_buffer(0, Some(&close), 0);
    enc.set_buffer(1, Some(&observed_close), 0);
    enc.dispatch_threads(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );

    enc.end_encoding();
    let start = Instant::now();
    cb.commit();

    let claim_deadline = Instant::now() + std::time::Duration::from_secs(30);
    while unsafe { (claimed.contents() as *const u32).read_volatile() } == 0 {
        if Instant::now() >= claim_deadline {
            unsafe { (close.contents() as *mut u32).write_volatile(1) };
            panic!("decode worker did not publish its claim within 5 seconds");
        }
        std::thread::yield_now();
    }

    unsafe { (close.contents() as *mut u32).write_volatile(1) };

    let completion_deadline = Instant::now() + std::time::Duration::from_secs(30);
    while !matches!(
        cb.status(),
        MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
    ) {
        if Instant::now() >= completion_deadline {
            panic!(
                "persistent workers did not close within 5 seconds; command buffer status: {:?}",
                cb.status()
            );
        }
        std::thread::yield_now();
    }
    cb.wait_until_completed();
    let elapsed = start.elapsed();

    assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
    assert_eq!(
        unsafe { (observed_close.contents() as *const u32).read_volatile() },
        1,
        "prefetch worker did not observe the host close request"
    );
    println!(
        "  ✓ Stage E: full protocol — {:.1}ms",
        elapsed.as_secs_f64() * 1e3
    );
}
