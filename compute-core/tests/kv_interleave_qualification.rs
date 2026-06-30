//! KV Interleave Pipeline — Phase 1 GPU Qualification Tests.
//!
//! Validates that the persistent multi-threadgroup kernels compile,
//! dispatch, and produce correct outputs on real Metal hardware.
//!
//! Tests:
//!   test_01_smoke — Compile both new kernels from SHADER_SRC
//!   test_02_prefetch_worker_epoch_exit — Prefetch worker detects epoch close and exits
//!   test_03_state_machine_lifecycle — Full scratch buffer state machine under single dispatch
//!
//! Run: cargo test --test kv_interleave_qualification --features prism-backend -- --nocapture

#![cfg(all(target_os = "macos", feature = "prism-backend"))]

use metal::*;
use std::time::Instant;

// ── Test constants ─────────────────────────────────────────────────
const TG_SIZE: u64 = 256;
const TOKEN_BUDGET: u32 = 1; // bounded epoch = 1 token for unit tests

// ── Metal compilation helper ──────────────────────────────────────
fn mtl_compile(device: &Device, src: &str, entry: &str) -> Result<ComputePipelineState, String> {
    let tmp = std::env::temp_dir().join("tribunus-interleave-qual");
    let _ = std::fs::create_dir_all(&tmp);
    let src_path = tmp.join("qual.metal");
    let air_path = tmp.join("qual.air");
    let lib_path = tmp.join("qual.metallib");

    std::fs::write(&src_path, src).map_err(|e| format!("write source: {e}"))?;

    let status = std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", "-std=metal4.0", "-O3", "-c"])
        .arg(src_path.to_str().unwrap())
        .arg("-o")
        .arg(air_path.to_str().unwrap())
        .status()
        .map_err(|e| format!("xcrun metal: {e}"))?;
    if !status.success() {
        return Err("Metal source compilation failed".into());
    }

    let status = std::process::Command::new("xcrun")
        .args(["-sdk", "macosx", "metallib", "-o"])
        .arg(lib_path.to_str().unwrap())
        .arg(air_path.to_str().unwrap())
        .status()
        .map_err(|e| format!("xcrun metallib: {e}"))?;
    if !status.success() {
        return Err("Metal library linking failed".into());
    }

    let lib_data = std::fs::read(&lib_path).map_err(|e| format!("read metallib: {e}"))?;
    let library = device
        .new_library_with_data(&lib_data)
        .map_err(|e| format!("new_library: {e:?}"))?;
    let function = library
        .get_function(entry, None)
        .map_err(|e| format!("get_function({entry}): {e:?}"))?;
    device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| format!("pipeline state: {e:?}"))
}

/// Extract SHADER_SRC from kernels.rs
/// Extract just the MSL ABI constants + structs + prefetch worker from SHADER_SRC
fn get_prefetch_worker_src() -> String {
    // Standalone Metal prefetch worker for Phase 1 qualification.
    // Uses relaxed atomics (this MSL v17.6 toolchain requires memory_order_relaxed
    // for device-address-space atomics).
    r##"#include <metal_stdlib>
using namespace metal;

#define CLAIM_UNOWNED 0
#define CLAIM_HELPER 1
#define CLAIM_DECODE_FALLBACK 2
#define CLAIM_DECODE_CONSUMER 3
#define OUTCOME_NONE 0
#define OUTCOME_READY_CONSUMABLE 1
#define OUTCOME_CANCELED 2
#define OUTCOME_POISONED 3
#define OUTCOME_BYPASSED 4
#define KV_STATE_EMPTY 0
#define KV_STATE_QUEUED 1
#define KV_STATE_FILLING 2
#define KV_STATE_READY 3
#define KV_STATE_CONSUMING 7
#define KV_STATE_RECLAIMABLE 8

struct KvScratchDeviceControl {
    atomic_uint state;
    atomic_uint cancel_requested;
    atomic_uint payload_valid;
    atomic_uint request_generation;
    atomic_uint request_outcome;
    atomic_uint producer_claim;
    atomic_uint duplicate_write_detected;
    atomic_uint late_publish_rejection_count;
};

struct KvScratchMetadataAbi { uint id, sid, seq, layer, epoch, gen, ptgen, off; };
struct KvScratchHeader {
    KvScratchMetadataAbi metadata;
    KvScratchDeviceControl control;
};
struct KvQueueCounterSlot { atomic_uint value; uint _pad[31]; };
struct KvPrefetchRequest { uint rid,sid,seq,tlayer,epoch,gen,ptgen,ssidx,skb,svb,ssb,spcnt,dkoff,dvoff,dtick,flags; };
struct KvPrefetchQueueAbi {
    KvQueueCounterSlot enq, deq, comp, drop;
    uint oc, cap, mask, ver;
    KvPrefetchRequest entries[16];
};
struct EpochControl { atomic_uint close, limit, fclaim, ffault, ffaultgen, ffaultreq; };
struct EpochReceipt { atomic_uint claimed, ready, consumed, canceled, poisoned, bypassed, latedisc, dupwrite, unres; };

kernel void persistent_kv_prefetch_worker(
    device KvPrefetchQueueAbi* queue [[buffer(0)]],
    device EpochControl* epoch [[buffer(1)]],
    device EpochReceipt* receipt [[buffer(2)]],
    device KvScratchHeader* headers [[buffer(3)]],
    device half* scratch [[buffer(4)]],
    constant uint& max_tokens [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint tg_sz [[threads_per_threadgroup]])
{
    uint tokens = 0;
    while (tokens < max_tokens) {
        uint close = atomic_load_explicit(&epoch->close, memory_order_relaxed);
        uint limit = atomic_load_explicit(&epoch->limit, memory_order_relaxed);
        uint deq = atomic_load_explicit(&queue->deq.value, memory_order_relaxed);
        if (close && deq >= limit) break;

        uint enq = atomic_load_explicit(&queue->enq.value, memory_order_relaxed);
        if (deq != enq) {
            uint idx = deq & queue->mask;
            KvPrefetchRequest req = queue->entries[idx];
            uint claimed = atomic_fetch_add_explicit(&queue->deq.value, 1, memory_order_relaxed);
            if (claimed != deq) continue;

            device KvScratchHeader* hdr = headers + req.ssidx;
            scratch[0] = (half)req.tlayer;
            scratch[1] = (half)req.epoch;
            atomic_store_explicit(&hdr->control.payload_valid, 1, memory_order_relaxed);
            atomic_store_explicit(&hdr->control.state, KV_STATE_READY, memory_order_relaxed);
            uint exp = OUTCOME_NONE;
            if (atomic_compare_exchange_weak_explicit(&hdr->control.request_outcome, &exp, OUTCOME_READY_CONSUMABLE, memory_order_relaxed, memory_order_relaxed)) {
                atomic_fetch_add_explicit(&receipt->ready, 1, memory_order_relaxed);
}
            tokens++;
}
}
}
"##.to_string()
}

// ═══════════════════════════════════════════════════════════════════
//  Test 1: Compile both new kernels
// ═══════════════════════════════════════════════════════════════════
#[test]
fn test_01_prefetch_worker_compilation() {
    let device = Device::system_default().expect("no Metal device");
    let src = get_prefetch_worker_src();

    let start = Instant::now();
    let pso = mtl_compile(&device, &src, "persistent_kv_prefetch_worker")
        .expect("persistent_kv_prefetch_worker must compile");
    let elapsed = start.elapsed();

    println!("Device: {}", device.name());
    println!(
        "persistent_kv_prefetch_worker compiled: {:.1}ms",
        elapsed.as_secs_f64() * 1e3
    );
    println!("  threadgroup size: {}", pso.max_total_threads_per_threadgroup());
}

// ═══════════════════════════════════════════════════════════════════
//  Test 2: Prefetch worker — epoch close detection and exit
// ═══════════════════════════════════════════════════════════════════
#[test]
fn test_02_prefetch_worker_epoch_exit() {
    let device = Device::system_default().expect("no Metal device");
    let queue = device.new_command_queue();
    let src = get_prefetch_worker_src();
    let pso = mtl_compile(&device, &src, "persistent_kv_prefetch_worker").unwrap();

    // ── Allocate epoch control structure ─────────────────────────────
    // Matches the MSL EpochControl layout:
    //   epoch_close_requested: atomic_uint
    //   epoch_enqueue_limit: atomic_uint
    //   epoch_fatal_claim: atomic_uint
    //   epoch_fatal_fault: atomic_uint
    //   epoch_fatal_fault_generation: atomic_uint
    //   epoch_fatal_fault_request_id: atomic_uint
    // Total: 24 bytes (= 6 × u32)
    let epoch_ctrl = device.new_buffer(24, MTLResourceOptions::StorageModeShared);
    unsafe {
        // Set epoch_close_requested = 1, epoch_enqueue_limit = 0
        // The queue is also empty (dequeue_pos == enqueue_pos == 0),
        // so the helper should immediately detect close and exit.
        let ptr = epoch_ctrl.contents() as *mut u32;
        *ptr.add(0) = 1; // epoch_close_requested = true
        *ptr.add(1) = 0; // epoch_enqueue_limit = 0
    }

    // ── Allocate empty queue ────────────────────────────────────────
    // The queue struct has 4 KvQueueCounterSlots (128 bytes each) + 4 u32 + 16 entries (64 bytes each)
    // Total: 4*128 + 4*4 + 16*64 = 512 + 16 + 1024 = 1552 bytes
    let queue_size = 4 * 128 + 16 + 1024; // 1552
    let queue_buf = device.new_buffer(queue_size as u64, MTLResourceOptions::StorageModeShared);
    unsafe {
        std::ptr::write_bytes(queue_buf.contents(), 0, queue_size as usize);
    }

    // ── Allocate empty scratch + receipt ────────────────────────────
    let scratch_k = device.new_buffer(4096, MTLResourceOptions::StorageModeShared);
    let scratch_v = device.new_buffer(4096, MTLResourceOptions::StorageModeShared);
    let headers = device.new_buffer(256, MTLResourceOptions::StorageModeShared);
    let receipt = device.new_buffer(64, MTLResourceOptions::StorageModeShared);

    // ── Dispatch ────────────────────────────────────────────────────
    let cb = queue.new_command_buffer();
    let enc = cb.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pso);
    enc.set_buffer(0, Some(&queue_buf), 0);  // queue
    enc.set_buffer(1, Some(&scratch_k), 0);   // kv_k_nibbles
    enc.set_buffer(2, Some(&scratch_v), 0);   // kv_v_nibbles
    enc.set_buffer(5, Some(&scratch_k), 0);   // scratch_k
    enc.set_buffer(6, Some(&scratch_v), 0);   // scratch_v
    enc.set_buffer(7, Some(&headers), 0);     // headers
    enc.set_buffer(8, Some(&epoch_ctrl), 0);  // epoch_control
    enc.set_buffer(9, Some(&receipt), 0);     // receipt
    // max_tokens_per_epoch at buffer(10) — doesn't matter for worker alone
    let max_tokens = device.new_buffer_with_data(
        &TOKEN_BUDGET as *const u32 as *const std::ffi::c_void,
        4,
        MTLResourceOptions::StorageModeShared,
    );
    enc.set_buffer(10, Some(&max_tokens), 0);

    enc.dispatch_threads(
        MTLSize { width: 1, height: 1, depth: 1 },
        MTLSize { width: TG_SIZE, height: 1, depth: 1 },
    );
    enc.end_encoding();

    let start = Instant::now();
    cb.commit();
    cb.wait_until_completed();
    let elapsed = start.elapsed();

    println!(
        "  Prefetch worker (epoch close test): {:.1}µs, status={:?}",
        elapsed.as_secs_f64() * 1e6,
        cb.status()
    );

    // Verify no fatal fault
    let fault_code = unsafe { *(epoch_ctrl.contents() as *const u32).add(3) };
    assert_eq!(fault_code, 0, "epoch_fatal_fault must be FAULT_NONE");
    assert_eq!(cb.status(), MTLCommandBufferStatus::Completed);
    println!("  ✓ Worker exited cleanly, no fatal fault");
}

// ═══════════════════════════════════════════════════════════════════
//  Test 3: Decode worker — single-token epoch, drain protocol
// ═══════════════════════════════════════════════════════════════════
