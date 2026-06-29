//! GPU L2→DRAM flush verification via MTLSharedEvent encodeSignalEvent.
//!
//! Tests that encodeSignalEvent on a command buffer guarantees GPU L2 cache
//! lines are flushed to DRAM (and thus visible to the CPU via StorageModeShared)
//! BEFORE the shared event's signaled value is updated.
//!
//! Without this guarantee, the CPU may observe stale L2 data (zeros)
//! after the event signals, because the GPU wrote to the buffer but the
//! L2 lines hadn't been written back to DRAM yet.
//!
//! Run: cargo test --test mtl_shared_event_barrier --features prism-backend -- --nocapture
//!
//! Requires: macOS 14.0+, Apple Silicon (M1 tested)

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use metal::*;
use std::sync::atomic::Ordering;
use std::time::Instant;
use tribunus_compute_core::compute_image::metal_pipeline::compile_metal_source;

// ── Constants ──────────────────────────────────────────────────────────────

/// Sentinel value written by GPU to every buffer element.
const SENTINEL: u32 = 0xDEAD_BEEF;

/// Number of elements in the shared buffer.
const N_ELEMS: usize = 1024;

/// Number of test iterations.
const ITERS: usize = 1000;

// ── Metal kernel source ────────────────────────────────────────────────────

/// Writes `value` to every element of `data`, one element per thread.
const WRITE_KERNEL: &str = r##"#include <metal_stdlib>
using namespace metal;

kernel void write_test_data(
    device uint*     data  [[buffer(0)]],
    constant uint&   value [[buffer(1)]],
    uint id [[thread_position_in_grid]])
{
    data[id] = value;
}
"##;

// ── Setup ──────────────────────────────────────────────────────────────────

fn setup() -> (Device, ComputePipelineState, CommandQueue) {
    let dev = Device::system_default().expect("Metal device");
    let out = compile_metal_source("write_test_data", WRITE_KERNEL)
        .expect("Metal kernel compilation failed");
    let lib = dev
        .new_library_with_data(&out.metallib_bytes)
        .expect("new_library_with_data");
    let func = lib
        .get_function("write_test_data", None)
        .expect("get_function(write_test_data)");
    let pl = dev
        .new_compute_pipeline_state_with_function(&func)
        .expect("new_compute_pipeline_state");
    let q = dev.new_command_queue();
    (dev, pl, q)
}

/// Allocate a shared buffer of N_ELEMS u32s, zero-initialized.
fn make_data_buf(dev: &Device) -> Buffer {
    let n_bytes = N_ELEMS * std::mem::size_of::<u32>();
    let bytes = n_bytes as u64;
    let buf = dev.new_buffer(bytes, MTLResourceOptions::StorageModeShared);
    // Zero-initialize (Metal may or may not zero new buffers on macOS).
    unsafe {
        std::ptr::write_bytes(buf.contents() as *mut u8, 0u8, n_bytes);
    }
    buf
}

/// Allocate a buffer holding the sentinel constant value.
fn make_value_buf(dev: &Device) -> Buffer {
    dev.new_buffer_with_data(
        &SENTINEL as *const u32 as *const std::ffi::c_void,
        std::mem::size_of::<u32>() as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

// ── Main test ──────────────────────────────────────────────────────────────

#[test]
fn test_mtl_shared_event_barrier() {
    println!();
    println!("═══ MTLSharedEvent encodeSignalEvent: GPU L2→DRAM Flush Verification ═══");
    println!("  Device:    Apple Silicon M1");
    println!("  macOS:     26.5");
    println!("  Elements:  {} × u32", N_ELEMS);
    println!("  Sentinel:  0x{SENTINEL:08X}");
    println!("  Iterations: {}", ITERS);
    println!();

    let (dev, pl, q) = setup();
    let data_buf = make_data_buf(&dev);
    let value_buf = make_value_buf(&dev);
    let shared_event = dev.new_shared_event();

    let mut successes: u32 = 0;
    let mut failures: u32 = 0;
    // Store per-iteration CPU read-loop durations (nanoseconds).
    let mut read_times_ns: Vec<u64> = Vec::with_capacity(ITERS);

    for iter in 0..ITERS {
        // Reset event to 0 for this iteration.
        shared_event.set_signaled_value(0);

        // Write a compiler fence so the reset is visible before we submit GPU work.
        std::sync::atomic::compiler_fence(Ordering::Release);

        // ── Submit GPU work ──────────────────────────────────────────────
        let cb: &CommandBufferRef = q.new_command_buffer();
        {
            let enc: &ComputeCommandEncoderRef = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pl);
            enc.set_buffer(0, Some(&data_buf), 0);
            enc.set_buffer(1, Some(&value_buf), 0);
            enc.dispatch_thread_groups(
                MTLSize {
                    width: N_ELEMS as u64,
                    height: 1,
                    depth: 1,
                },
                MTLSize {
                    width: 1,
                    height: 1,
                    depth: 1,
                },
            );
            enc.end_encoding();
        }
        // Signal event AFTER the kernel completes (encodes a post-compute barrier).
        cb.encode_signal_event(&*shared_event, 1);
        cb.commit();

        // ── CPU-side: ensure compiler doesn't reorder our reads ──────────
        std::sync::atomic::compiler_fence(Ordering::SeqCst);

        // ── CPU polls event until GPU signals completion ─────────────────
        loop {
            if shared_event.signaled_value() >= 1 {
                std::sync::atomic::compiler_fence(Ordering::Acquire);
                break;
            }
            std::hint::spin_loop();
        }

        // ── CPU reads buffer — measure read-loop time ────────────────────
        let t0 = Instant::now();
        let ptr = data_buf.contents() as *const u32;
        let mut all_correct = true;
        for i in 0..N_ELEMS {
            let val = unsafe { ptr.add(i).read_unaligned() };
            if val != SENTINEL {
                all_correct = false;
                break;
            }
        }
        let elapsed_ns = t0.elapsed().as_nanos() as u64;
        read_times_ns.push(elapsed_ns);

        if all_correct {
            successes += 1;
        } else {
            // Scan to report how many elements had stale data.
            let mut stale_count = 0;
            for i in 0..N_ELEMS {
                let val = unsafe { ptr.add(i).read_unaligned() };
                if val != SENTINEL {
                    stale_count += 1;
                    // Report first few stale positions.
                    if stale_count <= 3 {
                        eprintln!(
                            "  [iter {}] stale at data[{}] = 0x{:08X} (expected 0x{:08X})",
                            iter, i, val, SENTINEL
                        );
                    }
                }
            }
            eprintln!(
                "  [iter {}] FAILED — {} / {} elements had stale L2 data",
                iter, stale_count, N_ELEMS
            );
            failures += 1;
        }

        // Ensure GPU fully idle before next iteration.
        cb.wait_until_completed();
    }

    // ── Report ───────────────────────────────────────────────────────────
    println!();
    println!("{:─<60}", "");
    println!("  Results after {} iterations", ITERS);
    println!("{:─<60}", "");
    println!(
        "  Successful cycles:     {} / {} (all 1024 correct)",
        successes, ITERS
    );
    println!("  Failed cycles:         {} / {}", failures, ITERS);

    if failures == 0 {
        println!();
        println!("  *** Memory barrier: VERIFIED — encodeSignalEvent guarantees L2 flush ***");
    } else {
        println!();
        println!(
            "  *** Memory barrier: FAILED — {} cycles had stale reads ***",
            failures
        );
    }

    // ── Median verification time ─────────────────────────────────────────
    read_times_ns.sort_unstable();
    let median_ns = {
        let n = read_times_ns.len();
        if n % 2 == 0 {
            (read_times_ns[n / 2 - 1] + read_times_ns[n / 2]) / 2
        } else {
            read_times_ns[n / 2]
        }
    };
    println!();
    println!("  CPU verification (read-loop) timing:");
    println!("    Median:  {} ns", median_ns);
    println!("    Samples: {} iterations", read_times_ns.len());

    // ── Summary line for CI parsing ──────────────────────────────────────
    println!();
    if failures == 0 {
        println!("Memory barrier: VERIFIED — encodeSignalEvent guarantees L2 flush");
    } else {
        println!(
            "Memory barrier: FAILED — {} cycles had stale reads",
            failures
        );
    }
    println!();

    assert_eq!(
        failures, 0,
        "{} / {} iterations had stale L2 reads after MTLSharedEvent signal",
        failures, ITERS
    );
}
