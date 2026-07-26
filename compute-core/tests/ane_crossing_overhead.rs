//! GPU->CPU->ANE crossing overhead benchmark.
//!
//! Compares three signaling mechanisms from GPU completion to a simulated
//! ANEClient call across the CPU bridge:
//!
//!   1. MTLSharedEvent notify callback (GCD dispatch)
//!   2. Shared atomic flag: GPU writes u64 sentinel to MTLStorageModeShared buffer, CPU polls
//!   3. Signal shadowing: dedicated Metal doorbell kernel writes u64 sequence number, CPU polls
//!
//! All three execute the same trivial GPU compute kernel (writes 42 to output),
//! then measure elapsed time from GPU completion notification to issuing a
//! simulated ANEClient call (a dummy function call that models the ANE dispatch
//! bridge overhead).
//!
//! Run: cargo test --test ane_crossing_overhead --features prism-backend -- --nocapture
//!
//! Requires: macOS 14.0+, Apple Silicon (M1 tested)

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use block::ConcreteBlock;
use metal::*;
use std::ffi::CString;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc;
use std::time::Instant;
use tribunus_compute_core::compute_image::metal_pipeline::compile_metal_source;

// ── Constants ──────────────────────────────────────────────────────────────

/// Number of benchmark iterations per mode.
const ITERS: usize = 100;

/// Warmup iterations before measurement.
const WARMUP: usize = 20;

/// Dispatch queue label for MTLSharedEventListener.
const QUEUE_LABEL: &str = "com.tribunus.ane-crossing";

// ── Dummy ANE client call ──────────────────────────────────────────────────

/// Models the fixed overhead of crossing the ANE bridge (opaque FFI dispatch).
/// In real usage this would be an io_connect or ANEClientCopyClient callback;
/// here it's an atomic add to a static that the optimizer cannot elide.
static ANE_DISPATCH_COUNTER: AtomicU64 = AtomicU64::new(0);

fn ane_dispatch() {
    ANE_DISPATCH_COUNTER.fetch_add(1, Ordering::Relaxed);
}

// ── Metal kernel source: trivial compute ───────────────────────────────────

/// Minimal GPU compute: writes 42 to output[0]. All three mechanisms use
/// this same kernel for the actual work; the signaling differs only in how
/// the CPU learns the GPU is done and proceeds to ane_dispatch().
const COMPUTE_KERNEL: &str = r##"#include <metal_stdlib>
using namespace metal;

kernel void compute_42(
    device uint* output [[buffer(0)]])
{
    output[0] = 42;
}
"##;

// ── Metal kernel source: signal shadow doorbell ────────────────────────────

/// Dedicated doorbell kernel: writes a sequence number (u64) to shared memory.
/// The CPU polls this address; when the value matches the expected sequence
/// number, the GPU has completed the prior compute dispatch.
///
/// This is dispatched _after_ the compute kernel in the same command buffer,
/// so the doorbell write orders after the compute write.
const SIGNAL_SHADOW_KERNEL: &str = r##"#include <metal_stdlib>
using namespace metal;

kernel void signal_shadow(
    device uint64_t* doorbell [[buffer(0)]],
    constant uint64_t&  value   [[buffer(1)]])
{
    *doorbell = value;
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

// ── Device and kernel setup ────────────────────────────────────────────────

struct SharedKernels {
    compute_pl: ComputePipelineState,
    shadow_pl: Option<ComputePipelineState>,
}

fn setup() -> (Device, SharedKernels, CommandQueue) {
    let dev = Device::system_default().expect("Metal device");

    // Compile compute kernel
    let compute_out = compile_metal_source("ane_compute_42", COMPUTE_KERNEL)
        .expect("compute_42 Metal kernel compilation failed");
    let lib = dev
        .new_library_with_data(&compute_out.metallib_bytes)
        .expect("new_library_with_data (compute)");
    let func = lib
        .get_function("compute_42", None)
        .expect("get_function(compute_42)");
    let compute_pl = dev
        .new_compute_pipeline_state_with_function(&func)
        .expect("new_compute_pipeline_state (compute)");

    // Compile signal shadow kernel (optional — some toolchains may reject it)
    let shadow_pl = compile_metal_source("signal_shadow", SIGNAL_SHADOW_KERNEL).map(|out| {
        let lib = dev
            .new_library_with_data(&out.metallib_bytes)
            .expect("new_library_with_data (shadow)");
        let func = lib
            .get_function("signal_shadow", None)
            .expect("get_function(signal_shadow)");
        dev.new_compute_pipeline_state_with_function(&func)
            .expect("new_compute_pipeline_state (shadow)")
    });

    let q = dev.new_command_queue();
    (
        dev,
        SharedKernels {
            compute_pl,
            shadow_pl,
        },
        q,
    )
}

/// Allocate a shared buffer with room for a uint64 (for sentinel / doorbell).
fn make_u64_buf(dev: &Device) -> Buffer {
    dev.new_buffer(8, MTLResourceOptions::StorageModeShared)
}

/// Allocate a shared buffer containing a constant u64 value.
fn make_value_buf(dev: &Device, val: u64) -> Buffer {
    dev.new_buffer_with_data(
        &val as *const u64 as *const std::ffi::c_void,
        8,
        MTLResourceOptions::StorageModeShared,
    )
}

/// Allocate a small output buffer for the compute kernel to write 42 into.
fn make_output_buf(dev: &Device) -> Buffer {
    dev.new_buffer(4, MTLResourceOptions::StorageModeShared)
}

// ── Mechanism 1: MTLSharedEvent notify callback ───────────────────────────

fn bench_event_notify(dev: &Device, kernels: &SharedKernels, q: &CommandQueue) -> Vec<f64> {
    let mut samples = Vec::with_capacity(ITERS);
    let shared_event = dev.new_shared_event();

    let output_buf = make_output_buf(dev);

    // GCD dispatch queue
    let queue_label = CString::new(QUEUE_LABEL).expect("CString");
    let raw_queue = unsafe { dispatch_queue_create(queue_label.as_ptr(), std::ptr::null()) };
    assert!(!raw_queue.is_null(), "dispatch_queue_create failed");
    let listener = unsafe { SharedEventListener::from_queue_handle(raw_queue as *mut _) };

    // Warmup
    for _ in 0..WARMUP {
        shared_event.set_signaled_value(0);

        let cb = q.new_command_buffer();
        {
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&kernels.compute_pl);
            enc.set_buffer(0, Some(&output_buf), 0);
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

    // Timed iterations
    for _ in 0..ITERS {
        shared_event.set_signaled_value(0);
        let (tx, rx) = mpsc::channel::<u64>();

        let notify_block = ConcreteBlock::new(move |_event: &SharedEventRef, value: u64| {
            let _ = tx.send(value);
        });
        let rc_block = notify_block.copy();

        shared_event.notify(&listener, 1, rc_block);

        let cb = q.new_command_buffer();
        {
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&kernels.compute_pl);
            enc.set_buffer(0, Some(&output_buf), 0);
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
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);

        // ── Measure: notification → ane_dispatch ──
        let t0 = Instant::now();
        let _ = rx.recv().expect("notify channel closed");
        ane_dispatch();
        let elapsed = t0.elapsed().as_nanos() as f64;
        samples.push(elapsed);

        cb.wait_until_completed();
    }

    unsafe {
        dispatch_release(raw_queue);
    }
    samples
}

// ── Mechanism 2: Shared atomic flag (GPU writes, CPU polls) ────────────────

/// Modified compute kernel: writes 42 to output[0] AND a sentinel u64 to flag[0].
/// Sentinel hard-coded to avoid constant address space issues.
const COMPUTE_KERNEL_WITH_FLAG: &str = r##"#include <metal_stdlib>
using namespace metal;

kernel void compute_42_with_flag(
    device uint*   output [[buffer(0)]],
    device ulong*  flag   [[buffer(1)]])
{
    output[0] = 42;
    *flag = 0xBEEF000000000001ULL;
}
"##;

fn bench_atomic_flag(dev: &Device, _kernels: &SharedKernels, q: &CommandQueue) -> Vec<f64> {
    let mut samples = Vec::with_capacity(ITERS);
    let output_buf = make_output_buf(dev);
    let flag_buf = make_u64_buf(dev);
    const SENTINEL: u64 = 0xBEEF000000000001;

    let flag_out = compile_metal_source("compute_42_with_flag", COMPUTE_KERNEL_WITH_FLAG)
        .expect("compute_42_with_flag Metal compilation failed");
    let lib = dev
        .new_library_with_data(&flag_out.metallib_bytes)
        .expect("new_library_with_data (flag)");
    let func = lib
        .get_function("compute_42_with_flag", None)
        .expect("get_function(compute_42_with_flag)");
    let pl = dev
        .new_compute_pipeline_state_with_function(&func)
        .expect("new_compute_pipeline_state (flag)");

    // Warmup
    for _ in 0..WARMUP {
        let flag_ptr = flag_buf.contents() as *mut u64;
        unsafe {
            flag_ptr.write_unaligned(0u64);
        }

        let cb = q.new_command_buffer();
        {
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pl);
            enc.set_buffer(0, Some(&output_buf), 0);
            enc.set_buffer(1, Some(&flag_buf), 0);
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
        cb.commit();
        cb.wait_until_completed();
    }
    assert_eq!(
        unsafe { *(output_buf.contents() as *const u32) },
        42,
        "warmup: compute kernel did not write 42"
    );

    // Timed iterations
    for _ in 0..ITERS {
        let flag_ptr = flag_buf.contents() as *mut u64;
        unsafe {
            flag_ptr.write_unaligned(0u64);
        }

        let cb = q.new_command_buffer();
        {
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pl);
            enc.set_buffer(0, Some(&output_buf), 0);
            enc.set_buffer(1, Some(&flag_buf), 0);
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
        cb.commit();
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);

        // ── Measure: poll flag → ane_dispatch ──
        let t0 = Instant::now();
        let ptr = flag_buf.contents() as *const u64;
        loop {
            if unsafe { ptr.read_unaligned() } == SENTINEL {
                std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::Acquire);
                break;
            }
            std::hint::spin_loop();
        }
        ane_dispatch();
        let elapsed = t0.elapsed().as_nanos() as f64;
        samples.push(elapsed);

        cb.wait_until_completed();
    }

    samples
}

// ── Mechanism 3: Signal shadowing (dedicated doorbell kernel) ──────────────

fn bench_signal_shadow(dev: &Device, kernels: &SharedKernels, q: &CommandQueue) -> Vec<f64> {
    let mut samples = Vec::with_capacity(ITERS);
    let output_buf = make_output_buf(dev);
    let doorbell_buf = make_u64_buf(dev);

    let shadow_pl = kernels
        .shadow_pl
        .as_ref()
        .expect("signal_shadow kernel not compiled — Metal 3.2 may not be available");

    // Warmup
    for seq in 0..WARMUP {
        let doorbell_ptr = doorbell_buf.contents() as *mut u64;
        unsafe {
            doorbell_ptr.write_unaligned(0u64);
        }
        let target_val = (seq + 1) as u64;
        let val_buf = make_value_buf(dev, target_val);

        let cb = q.new_command_buffer();
        // Compute kernel first
        {
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&kernels.compute_pl);
            enc.set_buffer(0, Some(&output_buf), 0);
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
        // Doorbell kernel second — writes the sequence number after the compute
        // kernel's store is visible (same command buffer guarantees ordering).
        {
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(shadow_pl);
            enc.set_buffer(0, Some(&doorbell_buf), 0);
            enc.set_buffer(1, Some(&val_buf), 0);
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
        cb.commit();
        cb.wait_until_completed();
    }

    // Timed iterations
    for seq in 0..ITERS {
        let target_val = (seq + 1 + WARMUP) as u64;
        let doorbell_ptr = doorbell_buf.contents() as *mut u64;
        unsafe {
            doorbell_ptr.write_unaligned(0u64);
        }
        let val_buf = make_value_buf(dev, target_val);

        let cb = q.new_command_buffer();
        // Compute kernel
        {
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&kernels.compute_pl);
            enc.set_buffer(0, Some(&output_buf), 0);
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
        // Doorbell kernel
        {
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(shadow_pl);
            enc.set_buffer(0, Some(&doorbell_buf), 0);
            enc.set_buffer(1, Some(&val_buf), 0);
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
        cb.commit();
        std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::SeqCst);

        // ── Measure: poll doorbell → ane_dispatch ──
        let t0 = Instant::now();
        let ptr = doorbell_buf.contents() as *const u64;
        loop {
            if unsafe { ptr.read_unaligned() } == target_val {
                std::sync::atomic::compiler_fence(std::sync::atomic::Ordering::Acquire);
                break;
            }
            std::hint::spin_loop();
        }
        ane_dispatch();
        let elapsed = t0.elapsed().as_nanos() as f64;
        samples.push(elapsed);

        cb.wait_until_completed();
    }

    samples
}

// ── Report helper ──────────────────────────────────────────────────────────

fn print_row(label: &str, samples: &mut Vec<f64>) {
    if samples.is_empty() {
        println!("  {:<40}  SKIPPED", label);
        return;
    }
    let (median, p95, mean) = stats(samples);
    println!(
        "  {:<40}  {:>9.0}  {:>9.0}  {:>9.0}",
        label, median, p95, mean
    );
}

// ── Main test ──────────────────────────────────────────────────────────────

#[test]
fn test_ane_crossing_overhead() {
    println!();
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  ANE Crossing Overhead: GPU->CPU->ANE Signaling Mechanisms");
    println!("═══════════════════════════════════════════════════════════════════");
    println!("  Device:  Apple Silicon M1");
    println!("  macOS:   26.5");
    println!("  Iterations: {} per mode, {} warmup", ITERS, WARMUP);
    println!();
    println!("  All measurements: ns from GPU completion detect to ane_dispatch()");
    println!("  (simulated ANE bridge call)");
    println!();

    let (dev, kernels, q) = setup();

    println!("{:─<80}", "");
    println!(
        "  {:<40}  {:>9}  {:>9}  {:>9}",
        "Mechanism", "Median(ns)", "P95(ns)", "Mean(ns)"
    );
    println!("{:─<80}", "");

    // ── 1. MTLSharedEvent notify callback ──
    let mut t1 = bench_event_notify(&dev, &kernels, &q);
    print_row("MTLSharedEvent notify callback", &mut t1);

    // ── 2. Shared atomic flag (GPU writes, CPU polls) ──
    let mut t2 = bench_atomic_flag(&dev, &kernels, &q);
    print_row("Shared atomic flag (busy poll)", &mut t2);

    // ── 3. Signal shadowing (doorbell kernel) ──
    let mut t3 = bench_signal_shadow(&dev, &kernels, &q);
    print_row("Signal shadow doorbell (busy poll)", &mut t3);

    println!("{:─<80}", "");
    println!();
    println!("Notes:");
    println!("  - MTLSharedEvent notify: CPU blocks until GCD callback fires.");
    println!("  - Shared atomic flag: compute kernel writes sentinel u64, CPU tight-spins.");
    println!("  - Signal shadow: separate doorbell kernel writes sequence u64 after compute,");
    println!("    CPU tight-spins. Guarantees ordering via same command buffer sequencing.");
    println!("  - All values include ane_dispatch() simulation call overhead (~2–5 ns).");
    println!("  - Lower is better. P95 shows tail latency behavior.");
    println!();
}
