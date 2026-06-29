//! PRISM-PRODUCTION-HETEROGENEOUS-EXECUTOR-0001 — ANE/Core ML lane executor.

use std::sync::Arc;
use std::time::Instant;

use parking_lot::Mutex;
use tokio::sync::mpsc;

use crate::backend::coreml_iosurface::{
    CoreMlComputePolicy, CoreMlIOSurfaceBinding, CoreMlIOSurfaceExecutable,
};
use crate::backend::placement::ExecutionLane;
use crate::compilation::tri_lane::NumericalStatus;
use crate::compute_image::apple_shared_arena::AppleSharedArena;
use crate::coreml_bridge::CoreMlModel;
use crate::scheduling::lane_work::{
    BackendExecutionTiming, LaneExecutionError, LaneExecutor, LaneWorkRequest, TimestampQuality,
    WorkCompletion, WorkSubmission,
};
use std::ffi::c_void;

// ── ANE lane executor ───────────────────────────────────────────────────
/// Wrapper to mark `*mut T` as `Send` for use in `spawn_blocking`.
/// # Safety
/// The caller guarantees exclusive or serialized access to the pointee
/// across the closure's lifetime.
struct SendRawPtr<T>(*mut T);
unsafe impl<T> Send for SendRawPtr<T> {}

/// Allocate a page-aligned zero-initialized buffer via anonymous mmap.
/// The returned pointer is always 16 KB aligned (Apple Silicon page size),
/// which avoids kernel shadow copies during IOSurface creation.
/// The caller is responsible for freeing the memory with `libc::munmap`.
fn allocate_page_aligned_buffer(byte_len: usize) -> *mut u8 {
    if byte_len == 0 {
        return std::ptr::null_mut();
    }
    unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            byte_len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
        if ptr == libc::MAP_FAILED {
            // Fall back to heap allocation if mmap fails
            let mut heap: Vec<u8> = vec![0u8; byte_len];
            let ptr = heap.as_mut_ptr();
            std::mem::forget(heap);
            ptr
        } else {
            ptr as *mut u8
        }
    }
}

/// Real ANE lane executor that runs Core ML predictions on a worker thread.
///
/// Uses `tokio::task::spawn_blocking` to run synchronous Core ML predictions
/// without blocking the Tokio scheduler.  Completion is sent through the
/// channel after the prediction returns.
pub struct AneLaneExecutor {
    /// Core ML executable bound to the arena.
    pub artifact_id: String,
    pub model_path: String,
    pub compute_policy: CoreMlComputePolicy,
    pub input_bindings: Vec<CoreMlIOSurfaceBinding>,
    pub output_bindings: Vec<CoreMlIOSurfaceBinding>,
    pub model: Option<Arc<Mutex<CoreMlModel>>>,
    /// Arena reference for slot access.
    pub arena: *mut AppleSharedArena,
    /// Name for diagnostics.
    pub name: String,
    /// Runtime handle for spawning blocking work.
    pub runtime_handle: tokio::runtime::Handle,
}

// SAFETY: arena access is serialized through the orchestrator.
unsafe impl Send for AneLaneExecutor {}
unsafe impl Sync for AneLaneExecutor {}

impl AneLaneExecutor {
    pub fn new(
        mut coreml_exec: CoreMlIOSurfaceExecutable,
        arena: &mut AppleSharedArena,
        name: &str,
        runtime_handle: tokio::runtime::Handle,
    ) -> Self {
        let model = coreml_exec.model.take().map(|m| Arc::new(Mutex::new(m)));
        Self {
            artifact_id: coreml_exec.artifact_id.clone(),
            model_path: coreml_exec.model_path.clone(),
            compute_policy: coreml_exec.compute_policy,
            input_bindings: coreml_exec.input_bindings.clone(),
            output_bindings: coreml_exec.output_bindings.clone(),
            model,
            arena: arena as *mut AppleSharedArena,
            name: name.to_string(),
            runtime_handle,
        }
    }
}

impl LaneExecutor for AneLaneExecutor {
    fn submit(
        &mut self,
        request: LaneWorkRequest,
        completion_tx: mpsc::UnboundedSender<WorkCompletion>,
    ) -> Result<WorkSubmission, LaneExecutionError> {
        let submit_time = Instant::now();
        let submit_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        // Clone what the worker thread needs.
        let work_id = request.work_id;
        let phase_id = request.phase_id;
        let variant_id = request.variant_id;
        let output_slot = request.output_slot;
        let in_bindings = self.input_bindings.clone();
        let out_bindings = self.output_bindings.clone();
        let _model_path = self.model_path.clone();
        let _compute_policy = self.compute_policy;
        let model_arc = self.model.clone();
        let _worker_name = self.name.clone();
        let handle = self.runtime_handle.clone();
        let arena_addr = self.arena as usize;

        // Spawn blocking worker thread for the Core ML prediction.
        handle.spawn_blocking(move || {
            // Run prediction using the arena-backed IOSurface.
            let arena_ptr = SendRawPtr(arena_addr as *mut AppleSharedArena);
            let prediction_start_ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;

            let arena = unsafe { &mut *arena_ptr.0 };
            let prediction_ok = if in_bindings.is_empty() || out_bindings.is_empty() {
                false
            } else {
                let in_slot_id = in_bindings[0].slot_id;
                let out_slot_id = out_bindings[0].slot_id;
                let in_name = &in_bindings[0].tensor_id;
                let out_name = &out_bindings[0].tensor_id;

                let in_info = arena
                    .slot(in_slot_id)
                    .and_then(|s| s.backing_arena.as_ref())
                    .map(|a| a.info)
                    .unwrap_or_else(|| {
                        let s = arena.slot(in_slot_id).expect("slot exists");
                        let byte_len = s.manifest.byte_length.max(1) as usize;
                        crate::arena_info::ArenaInfo {
                            width: 1,
                            height: 1,
                            logical_dim0: s.manifest.logical_shape.first().copied().unwrap_or(1)
                                as i32,
                            logical_dim1: s.manifest.logical_shape.get(1).copied().unwrap_or(1)
                                as i32,
                            pixel_format: 0,
                            byte_size: byte_len as i32,
                            bytes_per_row: s
                                .manifest
                                .strides_bytes
                                .first()
                                .copied()
                                .unwrap_or(byte_len as u64)
                                as i32,
                            dtype: 9,
                            base_address: allocate_page_aligned_buffer(byte_len) as *mut c_void,
                            cv_buffer: std::ptr::null_mut(),
                            io_surface: std::ptr::null_mut(),
                        }
                    });
                let out_info = arena
                    .slot(out_slot_id)
                    .and_then(|s| s.backing_arena.as_ref())
                    .map(|a| a.info)
                    .unwrap_or_else(|| {
                        let s = arena.slot(out_slot_id).expect("slot exists");
                        let byte_len = s.manifest.byte_length.max(1) as usize;
                        crate::arena_info::ArenaInfo {
                            width: 1,
                            height: 1,
                            logical_dim0: s.manifest.logical_shape.first().copied().unwrap_or(1)
                                as i32,
                            logical_dim1: s.manifest.logical_shape.get(1).copied().unwrap_or(1)
                                as i32,
                            pixel_format: 0,
                            byte_size: byte_len as i32,
                            bytes_per_row: s
                                .manifest
                                .strides_bytes
                                .first()
                                .copied()
                                .unwrap_or(byte_len as u64)
                                as i32,
                            dtype: 9,
                            base_address: allocate_page_aligned_buffer(byte_len) as *mut c_void,
                            cv_buffer: std::ptr::null_mut(),
                            io_surface: std::ptr::null_mut(),
                        }
                    });

                if let Some(model_arc) = &model_arc {
                    let model = model_arc.lock();
                    model
                        .predict(in_name, &in_info, out_name, &out_info)
                        .is_ok()
                } else {
                    false
                }
            };

            let prediction_end_ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;

            let timing = BackendExecutionTiming {
                submit_ns,
                backend_start_ns: prediction_start_ns,
                backend_end_ns: prediction_end_ns,
                completion_callback_ns: prediction_end_ns,
                timestamp_quality: TimestampQuality::WorkerThreadBoundary,
            };

            let _ = completion_tx.send(WorkCompletion {
                work_id,
                phase_id,
                variant_id,
                lane: ExecutionLane::CoreMlAne,
                success: prediction_ok,
                output_slot,
                backend_status: if prediction_ok {
                    crate::scheduling::lane_work::BackendStatus::Completed
                } else {
                    crate::scheduling::lane_work::BackendStatus::Failed("prediction failed".into())
                },
                numerical_status: if prediction_ok {
                    NumericalStatus::Pass
                } else {
                    NumericalStatus::Fail("prediction failed".into())
                },
                timing,
            });
        });

        Ok(WorkSubmission {
            work_id: request.work_id,
            lane: ExecutionLane::CoreMlAne,
            submission_time: submit_time,
        })
    }
}
