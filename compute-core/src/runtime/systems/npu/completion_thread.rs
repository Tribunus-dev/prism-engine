//! Background thread helper for NPU completion port.
//!
//! Spawns a dedicated thread that blocks on the NPU's native wait() and
//! writes the completed submission sequence number into the shared atomic
//! counter.  The ECS observer reads this counter with Ordering::Acquire
//! during Stage::Maintenance — zero lock contention.

use std::sync::Arc;

use crate::runtime::resources::NpuCompletionPort;

/// Spawn a background thread that continuously submits to and waits for
/// the NPU, writing completion notifications into the shared atomic port.
///
/// This is the architectural seam where the FFI npu_wait call lives.  The
/// actual NPU backend binding is not yet wired; the unsafe block is the
/// intended call site.
pub fn spawn_npu_completion_thread(
    _session: *mut std::ffi::c_void,
    _target: crate::backend::npu::ffi::TargetNpu,
    port: Arc<NpuCompletionPort>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || loop {
        // Allocate the next submission ID.  The submitter side also calls
        // next_submission, so the IDs are globally ordered across all
        // submitters and this waiter thread.
        let seq = port.next_submission();

        // Block on the NPU's native wait() — zero CPU while waiting.
        // The FFI path for npu_wait is not yet wired;
        // this is the architectural seam.
        // When the NPU backend binding lands, the call will be:
        //   unsafe { crate::backend::npu::ffi::npu_wait(target, session); }
        std::thread::yield_now();
        // ^ placeholder — replace with the actual FFI wait call.

        // Publish the completed sequence number.
        port.completed_atomic()
            .store(seq, std::sync::atomic::Ordering::Release);
    })
}
