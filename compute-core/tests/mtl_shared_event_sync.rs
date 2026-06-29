//! GPU→CPU synchronization benchmark: AtomicBool in shared buffer vs MTLSharedEvent.
//!
//! Compares wake latency and CPU usage for four mechanisms:
//!   1. AtomicBool (tight spin)   — CPU pegs at 100% polling buffer
//!   2. AtomicBool (yield)        — CPU yields via thread::sleep(0)
//!   3. MTLSharedEvent (poll)     — CPU polls event.signaled_value()
//!   4. MTLSharedEvent (notify)   — CPU blocked, callback fires via GCD
//!
//! Run: cargo test --test mtl_shared_event_sync --features prism-backend -- --nocapture
//!
//! Requires: macOS 14.0+, Apple Silicon (M1 tested)

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use block::ConcreteBlock;
use metal::*;
use std::ffi::CString;
use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::time::{Duration, Instant};
use tribunus_compute_core::compute_image::metal_pipeline::compile_metal_source;

// ── Constants ──────────────────────────────────────────────────────────────

/// Sentinel value written by GPU kernel to flag buffer (split as two u32 halves).
const SENTINEL_LO: u32 = 0x0000_0001;
const SENTINEL_HI: u32 = 0xDEAD_BEEF;

/// Number of benchmark iterations per mode.
const ITERS: usize = 500;

/// Warmup iterations before measurement.
const WARMUP: usize = 20;

/// Dispatch queue label for MTLSharedEventListener.
const QUEUE_LABEL: &str = "com.tribunus.mtl-shared-event";

// ── Metal kernel source ────────────────────────────────────────────────────

/// Trivial GPU compute kernel: writes a sentinel value to buffer[0].
///
/// Buffer 0: flag (two u32 values = sentinel halves).
/// Buffer 1: sentinel constant as two u32 values.
/// MSL only supports memory_order_relaxed; on Apple Silicon unified memory
/// the write propagates to the CPU-coherent domain via the L2 cache.
const SENTINEL_KERNEL: &str = r##"#include <metal_stdlib>
using namespace metal;

kernel void write_sentinel(
    device uint*     flag     [[buffer(0)]],
    constant uint2&  sentinel [[buffer(1)]])
{
    flag[0] = sentinel.x;
    flag[1] = sentinel.y;
}
"##;

// ── Raw FFI for GCD dispatch queue creation ────────────────────────────────

extern "C" {
    fn dispatch_queue_create(
        label: *const std::ffi::c_char,
        attr: *const std::ffi::c_void,
    ) -> *mut std::ffi::c_void;
    fn dispatch_release(object: *mut std::ffi::c_void);
}

// ── Statistics helpers ─────────────────────────────────────────────────────

fn stats(ns: &mut Vec<f64>) -> (f64, f64, f64) {
    ns.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let n = ns.len();
    let mean = ns.iter().sum::<f64>() / n as f64;
    let median = if n % 2 == 0 {
        (ns[n / 2 - 1] + ns[n / 2]) / 2.0
    } else {
        ns[n / 2]
    };
    let p95_idx = ((n as f64) * 0.95).ceil() as usize - 1;
    let p95 = ns[p95_idx.min(n - 1)];
    (median, p95, mean)
}

// ── Device and kernel setup (shared by all four modes) ─────────────────────

fn setup() -> (Device, ComputePipelineState, CommandQueue) {
    let dev = Device::system_default().expect("Metal device");
    let out =
        compile_metal_source("sentinel", SENTINEL_KERNEL).expect("Metal kernel compilation failed");
    let lib = dev
        .new_library_with_data(&out.metallib_bytes)
        .expect("new_library_with_data");
    let func = lib
        .get_function("write_sentinel", None)
        .expect("get_function(write_sentinel)");
    let pl = dev
        .new_compute_pipeline_state_with_function(&func)
        .expect("new_compute_pipeline_state");
    let q = dev.new_command_queue();
    (dev, pl, q)
}

/// Allocate a Metal buffer for the sentinel flag (16 bytes = two uint32 halves).
fn make_flag_buf(dev: &Device) -> Buffer {
    dev.new_buffer(16, MTLResourceOptions::StorageModeShared)
}

/// Allocate a Metal buffer with the sentinel constant value (8 bytes = two u32).
fn make_sentinel_buf(dev: &Device) -> Buffer {
    let encoded: [u32; 2] = [SENTINEL_LO, SENTINEL_HI];
    dev.new_buffer_with_data(
        encoded.as_ptr() as *const std::ffi::c_void,
        8,
        MTLResourceOptions::StorageModeShared,
    )
}

// ── Benchmark: AtomicBool tight spin ───────────────────────────────────────

fn bench_atomic_tight(
    _dev: &Device,
    pl: &ComputePipelineState,
    q: &CommandQueue,
    flag_buf: &BufferRef,
    sentinel_buf: &BufferRef,
) -> Vec<f64> {
    let mut samples = Vec::with_capacity(ITERS);

    // Warmup
    {
        let flag_ptr = flag_buf.contents() as *mut u32;
        for _ in 0..WARMUP {
            unsafe {
                flag_ptr.write_unaligned(0u32);
                flag_ptr.add(1).write_unaligned(0u32);
            }
            std::sync::atomic::compiler_fence(Ordering::Release);

            let cb = q.new_command_buffer();
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pl);
            enc.set_buffer(0, Some(flag_buf), 0);
            enc.set_buffer(1, Some(sentinel_buf), 0);
            enc.dispatch_thread_groups(
                MTLSize {
                    width: 1,
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
            cb.commit();
            cb.wait_until_completed();
        }
    }

    // Timed iterations
    for _ in 0..ITERS {
        let flag_ptr = flag_buf.contents() as *mut u32;
        unsafe {
            flag_ptr.write_unaligned(0u32);
            flag_ptr.add(1).write_unaligned(0u32);
        }
        std::sync::atomic::compiler_fence(Ordering::Release);

        let cb = q.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pl);
        enc.set_buffer(0, Some(flag_buf), 0);
        enc.set_buffer(1, Some(sentinel_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize {
                width: 1,
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
        cb.commit();
        std::sync::atomic::compiler_fence(Ordering::SeqCst);

        let t0 = Instant::now();

        // Tight spin: read flag until it matches sentinel
        let ptr = flag_buf.contents() as *const u32;
        loop {
            let lo = unsafe { ptr.read_unaligned() };
            let hi = unsafe { ptr.add(1).read_unaligned() };
            if lo == SENTINEL_LO && hi == SENTINEL_HI {
                std::sync::atomic::compiler_fence(Ordering::Acquire);
                break;
            }
            std::hint::spin_loop();
        }

        let elapsed = t0.elapsed().as_nanos() as f64;
        samples.push(elapsed);

        // Drain GPU before next iteration
        cb.wait_until_completed();
    }

    samples
}

// ── Benchmark: AtomicBool with sleep(0) yield ──────────────────────────────

fn bench_atomic_yield(
    _dev: &Device,
    pl: &ComputePipelineState,
    q: &CommandQueue,
    flag_buf: &BufferRef,
    sentinel_buf: &BufferRef,
) -> Vec<f64> {
    let mut samples = Vec::with_capacity(ITERS);

    for _ in 0..WARMUP {
        let flag_ptr = flag_buf.contents() as *mut u32;
        unsafe {
            flag_ptr.write_unaligned(0u32);
            flag_ptr.add(1).write_unaligned(0u32);
        }
        std::sync::atomic::compiler_fence(Ordering::Release);

        let cb = q.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pl);
        enc.set_buffer(0, Some(flag_buf), 0);
        enc.set_buffer(1, Some(sentinel_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize {
                width: 1,
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
        cb.commit();
        cb.wait_until_completed();
    }

    for _ in 0..ITERS {
        let flag_ptr = flag_buf.contents() as *mut u32;
        unsafe {
            flag_ptr.write_unaligned(0u32);
            flag_ptr.add(1).write_unaligned(0u32);
        }
        std::sync::atomic::compiler_fence(Ordering::Release);

        let cb = q.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(pl);
        enc.set_buffer(0, Some(flag_buf), 0);
        enc.set_buffer(1, Some(sentinel_buf), 0);
        enc.dispatch_thread_groups(
            MTLSize {
                width: 1,
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
        cb.commit();
        std::sync::atomic::compiler_fence(Ordering::SeqCst);

        let t0 = Instant::now();

        // Spin with yield
        let ptr = flag_buf.contents() as *const u32;
        loop {
            let lo = unsafe { ptr.read_unaligned() };
            let hi = unsafe { ptr.add(1).read_unaligned() };
            if lo == SENTINEL_LO && hi == SENTINEL_HI {
                std::sync::atomic::compiler_fence(Ordering::Acquire);
                break;
            }
            std::thread::sleep(Duration::ZERO);
        }

        let elapsed = t0.elapsed().as_nanos() as f64;
        samples.push(elapsed);
        cb.wait_until_completed();
    }

    samples
}

// ── Benchmark: MTLSharedEvent poll ─────────────────────────────────────────

fn bench_event_poll(
    dev: &Device,
    pl: &ComputePipelineState,
    q: &CommandQueue,
    flag_buf: &BufferRef,
    sentinel_buf: &BufferRef,
) -> Vec<f64> {
    let mut samples = Vec::with_capacity(ITERS);
    let shared_event = dev.new_shared_event();

    for _ in 0..WARMUP {
        shared_event.set_signaled_value(0);
        let cb: &CommandBufferRef = q.new_command_buffer();
        {
            let enc: &ComputeCommandEncoderRef = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pl);
            enc.set_buffer(0, Some(flag_buf), 0);
            enc.set_buffer(1, Some(sentinel_buf), 0);
            enc.dispatch_thread_groups(
                MTLSize {
                    width: 1,
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
        cb.encode_signal_event(&*shared_event, 1);
        cb.commit();
        cb.wait_until_completed();
    }

    for _ in 0..ITERS {
        shared_event.set_signaled_value(0);
        std::sync::atomic::compiler_fence(Ordering::Release);

        let cb: &CommandBufferRef = q.new_command_buffer();
        {
            let enc: &ComputeCommandEncoderRef = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pl);
            enc.set_buffer(0, Some(flag_buf), 0);
            enc.set_buffer(1, Some(sentinel_buf), 0);
            enc.dispatch_thread_groups(
                MTLSize {
                    width: 1,
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
        cb.encode_signal_event(&*shared_event, 1);
        cb.commit();
        std::sync::atomic::compiler_fence(Ordering::SeqCst);

        let t0 = Instant::now();

        // Poll shared event value
        loop {
            if shared_event.signaled_value() >= 1 {
                std::sync::atomic::compiler_fence(Ordering::Acquire);
                break;
            }
            std::hint::spin_loop();
        }

        let elapsed = t0.elapsed().as_nanos() as f64;
        samples.push(elapsed);
        cb.wait_until_completed();
    }

    samples
}

// ── Benchmark: MTLSharedEvent notify callback ──────────────────────────────

fn bench_event_notify(
    dev: &Device,
    pl: &ComputePipelineState,
    q: &CommandQueue,
    flag_buf: &BufferRef,
    sentinel_buf: &BufferRef,
) -> Vec<f64> {
    let mut samples = Vec::with_capacity(ITERS);
    let shared_event = dev.new_shared_event();

    // Create a serial GCD dispatch queue for the event listener
    let queue_label = CString::new(QUEUE_LABEL).expect("CString");
    let raw_queue = unsafe { dispatch_queue_create(queue_label.as_ptr(), std::ptr::null()) };
    assert!(!raw_queue.is_null(), "dispatch_queue_create failed");

    // Create listener from raw queue handle
    let listener = unsafe { SharedEventListener::from_queue_handle(raw_queue as *mut _) };

    // Warmup (without notification, just to get GPU warmed up)
    for _ in 0..WARMUP {
        shared_event.set_signaled_value(0);
        let cb: &CommandBufferRef = q.new_command_buffer();
        {
            let enc: &ComputeCommandEncoderRef = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pl);
            enc.set_buffer(0, Some(flag_buf), 0);
            enc.set_buffer(1, Some(sentinel_buf), 0);
            enc.dispatch_thread_groups(
                MTLSize {
                    width: 1,
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
        cb.encode_signal_event(&*shared_event, 1);
        cb.commit();
        cb.wait_until_completed();
    }

    // Timed iterations with notification
    for _ in 0..ITERS {
        shared_event.set_signaled_value(0);
        let (tx, rx) = mpsc::channel::<u64>();

        // Build the notification block (move tx into block closure)
        let notify_block = ConcreteBlock::new(move |_event: &SharedEventRef, value: u64| {
            let _ = tx.send(value);
        });
        let rc_block = notify_block.copy();

        // Register notification for value >= 1
        shared_event.notify(&listener, 1, rc_block);
        std::sync::atomic::compiler_fence(Ordering::Release);

        // Submit work and encode signal
        let cb: &CommandBufferRef = q.new_command_buffer();
        {
            let enc: &ComputeCommandEncoderRef = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pl);
            enc.set_buffer(0, Some(flag_buf), 0);
            enc.set_buffer(1, Some(sentinel_buf), 0);
            enc.dispatch_thread_groups(
                MTLSize {
                    width: 1,
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
        cb.encode_signal_event(&*shared_event, 1);
        cb.commit();
        std::sync::atomic::compiler_fence(Ordering::SeqCst);

        let t0 = Instant::now();

        // Wait for notification from callback
        let _ = rx.recv().expect("notify channel closed");

        let elapsed = t0.elapsed().as_nanos() as f64;
        samples.push(elapsed);

        // Ensure GPU done before next iteration
        cb.wait_until_completed();
    }

    unsafe {
        dispatch_release(raw_queue);
    }
    samples
}

// ── Report helper ──────────────────────────────────────────────────────────

fn print_row(label: &str, samples: &mut Vec<f64>) {
    if samples.is_empty() {
        println!("  {:<35}  SKIPPED", label);
        return;
    }
    let (median, p95, mean) = stats(samples);
    let polls = match label {
        l if l.contains("notify") => "1     ".to_string(),
        l if l.contains("yield") => "∞ (yield)".to_string(),
        _ => "∞ (busy)".to_string(),
    };
    println!(
        "  {:<35}  {:>8.0}  {:>9.0}  {:>9.0}  {}",
        label, median, p95, mean, polls
    );
}

// ── Main test ──────────────────────────────────────────────────────────────

#[test]
fn test_mtl_shared_event_sync() {
    println!();
    println!("═══ MTLSharedEvent vs AtomicBool: GPU→CPU Synchronization ═══");
    println!("  Device:  Apple Silicon M1");
    println!("  macOS:   26.5");
    println!("  Iterations: {} per mode, {} warmup", ITERS, WARMUP);
    println!();

    let (dev, pl, q) = setup();
    let flag_buf = make_flag_buf(&dev);
    let sentinel_buf = make_sentinel_buf(&dev);

    println!("{:─<77}", "");
    println!(
        "  {:<35}  {:>8}  {:>9}  {:>9}  {}",
        "Mode", "Median(ns)", "P95(ns)", "Mean(ns)", "Polls/iter"
    );
    println!("{:─<77}", "");

    // ── 1. AtomicBool (tight spin) ──
    let mut t1 = bench_atomic_tight(&dev, &pl, &q, &flag_buf, &sentinel_buf);
    print_row("AtomicBool (tight spin)", &mut t1);

    // ── 2. AtomicBool (yield) ──
    let mut t2 = bench_atomic_yield(&dev, &pl, &q, &flag_buf, &sentinel_buf);
    print_row("AtomicBool (sleep(0) yield)", &mut t2);

    // ── 3. MTLSharedEvent (poll) ──
    let mut t3 = bench_event_poll(&dev, &pl, &q, &flag_buf, &sentinel_buf);
    print_row("MTLSharedEvent (poll signaled_value)", &mut t3);

    // ── 4. MTLSharedEvent (notify callback) ──
    let mut t4 = bench_event_notify(&dev, &pl, &q, &flag_buf, &sentinel_buf);
    print_row("MTLSharedEvent (notify callback)", &mut t4);

    println!("{:─<77}", "");
    println!();
    println!("Notes:");
    println!("  - Polls/iter: ∞ means CPU pegs at 100%% during wait.");
    println!("  - P95 is the 95th percentile latency.");
    println!("  - All measurements include GPU completion latency.");
    println!();
}
