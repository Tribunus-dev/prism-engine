//! AMD DS4 persistent megakernel — one HIP kernel launch per decode session.
//!
//! Uses a ring-buffer dispatch protocol matching the existing Metal
//! [`Megakernel`](crate::ecs::compute_image::megakernel::Megakernel) pattern.
//! Weights, KV cache, and ring buffers are pre-allocated at session startup;
//! the kernel runs persistently (polling its ring slot) until a shutdown
//! flag is raised.
//!
//! # Feature gate
//!
//! This module is compiled only when `feature = "amd-rocm"` is enabled.
//!
//! # Module registration
//!
//! In `backend/mod.rs`, AFTER `pub mod amd_rocm;` (line 35), add:
//! ```ignore
//! #[cfg(feature = "amd-rocm")]
//! pub mod amd_megakernel;
//! ```
//!
//! # HIP FFI
//!
//! Raw `extern "C"` stubs against the HIP runtime (`libamdhip64.so`).
//! No external crate dependency — this avoids build-system coupling to
//! AMD's ROCm toolchain until the project-standard `hip-sys` crate is
//! introduced.

#![cfg(feature = "amd-rocm")]

use std::collections::HashMap;
use std::ffi::CString;
use std::path::Path;
use std::sync::atomic::{fence, AtomicU32, Ordering};

use crate::ecs::cimage::{CImageLoader, CImagePayloadEntry, CImagePayloadRef, LoadedCImageV0};

// ── HIP FFI stubs ──────────────────────────────────────────────────────────
//
// These match the ROCm HIP runtime API signatures.
// See https://rocm.docs.amd.com/projects/HIP/en/latest/

type hipDevice_t = i32;
type hipStream_t = *mut std::ffi::c_void;
type hipModule_t = *mut std::ffi::c_void;
type hipFunction_t = *mut std::ffi::c_void;

const HIP_SUCCESS: i32 = 0;

const hipMemcpyHostToDevice: i32 = 1;
const hipMemcpyDeviceToHost: i32 = 2;

extern "C" {
    fn hipSetDevice(deviceId: hipDevice_t) -> i32;
    fn hipGetDeviceCount(count: *mut i32) -> i32;
    fn hipDeviceSynchronize() -> i32;

    fn hipMalloc(ptr: *mut *mut std::ffi::c_void, size: u64) -> i32;
    fn hipFree(ptr: *mut std::ffi::c_void) -> i32;

    fn hipHostMalloc(ptr: *mut *mut std::ffi::c_void, size: u64, flags: u32) -> i32;
    fn hipHostFree(ptr: *mut std::ffi::c_void) -> i32;

    fn hipMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: u64,
        kind: i32,
    ) -> i32;

    fn hipMemset(ptr: *mut std::ffi::c_void, value: i32, count: u64) -> i32;

    fn hipStreamCreate(stream: *mut hipStream_t) -> i32;
    fn hipStreamDestroy(stream: hipStream_t) -> i32;
    fn hipStreamSynchronize(stream: hipStream_t) -> i32;

    fn hipModuleLoadData(module: *mut hipModule_t, image: *const std::ffi::c_void) -> i32;
    fn hipModuleGetFunction(
        function: *mut hipFunction_t,
        module: hipModule_t,
        name: *const std::ffi::c_char,
    ) -> i32;
    fn hipModuleLaunchKernel(
        f: hipFunction_t,
        grid_dim_x: u32,
        grid_dim_y: u32,
        grid_dim_z: u32,
        block_dim_x: u32,
        block_dim_y: u32,
        block_dim_z: u32,
        shared_mem_bytes: u32,
        stream: hipStream_t,
        kernel_params: *mut *mut std::ffi::c_void,
        extra: *mut *mut std::ffi::c_void,
    ) -> i32;
    fn hipModuleUnload(module: hipModule_t) -> i32;
}

// ── Constants ──────────────────────────────────────────────────────────────

/// Ring buffer size (must match GPU kernel constant).
pub const RING_SIZE: usize = 512;

/// Work slot states (must match GPU kernel enum).
#[repr(u32)]
pub enum SlotState {
    Empty = 0,
    TokenReady = 1,
    Processing = 2,
    Completed = 3,
}

/// Entry stride: 5 × u32 per ring slot.
const ENTRY_STRIDE: usize = 5;

/// Per-ring-entry field offsets.
const OFFSET_STATE: usize = 0;
const OFFSET_TOKEN_ID: usize = 1;
const OFFSET_SEQ_POS: usize = 2;
const OFFSET_KV_SLOT_ID: usize = 3;
const OFFSET_RESERVED: usize = 4;

// ── Device weight buffer (RAII) ────────────────────────────────────────────

/// One weight tensor resident in VRAM.
///
/// Drops `device_ptr` via `hipFree` when the buffer goes out of scope.
struct DeviceWeightBuffer {
    device_ptr: *mut std::ffi::c_void,
    bytes: u64,
}

// Safety: `device_ptr` is an exclusive allocation owned by this struct.
unsafe impl Send for DeviceWeightBuffer {}

impl Drop for DeviceWeightBuffer {
    fn drop(&mut self) {
        if !self.device_ptr.is_null() {
            unsafe {
                let _ = hipFree(self.device_ptr);
            }
        }
    }
}

// ── AmdDs4Megakernel ────────────────────────────────────────────────────

/// Host-side controller for the AMD DS4 persistent decode megakernel.
///
/// Owns all GPU resources: ring buffers (pinned host memory visible to GPU),
/// weight tensors in VRAM, KV cache in VRAM, the compiled HIP module, and
/// an asynchronous stream for submissions.
///
/// When this struct is dropped (or `shutdown()` is called), all GPU resources
/// are freed.  Weight buffers are freed via `DeviceWeightBuffer::drop`
/// as the HashMap is drained.
pub struct AmdDs4Megakernel {
    // ── Ring buffer (pinned host memory, coherent with GPU) ──────────
    ring_entries: *mut u32,
    completion_counter: *mut u32,
    shutdown_flag: *mut u32,

    // ── Device buffers ──────────────────────────────────────────────
    logits_out: *mut std::ffi::c_void,
    kv_cache: *mut std::ffi::c_void,
    weight_buffers: HashMap<String, DeviceWeightBuffer>,

    // ── Kernel resources ────────────────────────────────────────────
    module: hipModule_t,
    kernel: hipFunction_t,

    // ── Asynchronous stream ─────────────────────────────────────────
    stream: hipStream_t,

    // ── Metadata ────────────────────────────────────────────────────
    num_slots: u32,
    device_id: hipDevice_t,
    last_completed: AtomicU32,
    vocab_size: u32,
    hidden_dim: u32,
    num_layers: u32,
    kv_cache_bytes: u64,

    _private: (),
}

// Safety: all raw pointers are owned exclusively by this struct and
// freed in `shutdown()` or `Drop`.  No shared aliasing.
unsafe impl Send for AmdDs4Megakernel {}

impl AmdDs4Megakernel {
    // ── Construction ────────────────────────────────────────────────────

    /// Build a fresh persistent megakernel session.
    ///
    /// 1. Selects `device_id` (or device 0)
    /// 2. Loads the CImage file and uploads every tensor to VRAM
    /// 3. Allocates the ring buffer (pinned host), KV cache, and logits buffer
    /// 4. Loads the compiled HIP code object and resolves the kernel entry point
    /// 5. Launches the persistent kernel (one threadblock per slot)
    ///
    /// # Parameters
    ///
    /// * `cimage_path` — path to the `.cimage` file containing weights + metadata.
    /// * `device_id`   — target AMD GPU device index (0-based).
    /// * `num_slots`   — number of decode slots (= number of threadblocks in
    ///   the persistent kernel).
    /// * `vocab_size`, `hidden_dim`, `num_layers` — model architecture dimensions
    ///   ({{PLACEHOLDER}}: these could be extracted from CImage manifest metadata
    ///   in a future refactor).
    pub fn new(
        cimage_path: impl AsRef<Path>,
        device_id: i32,
        num_slots: u32,
        vocab_size: u32,
        hidden_dim: u32,
        num_layers: u32,
    ) -> Result<Self, String> {
        let cimage_path = cimage_path.as_ref();
        let device_id: hipDevice_t = device_id;

        // ── 1. Select device ─────────────────────────────────────────
        unsafe {
            let err = hipSetDevice(device_id);
            if err != HIP_SUCCESS {
                return Err(format!("hipSetDevice({device_id}) returned {err}"));
            }
        }

        // ── 2. Load CImage and upload weights to VRAM ────────────────
        let loaded = CImageLoader::load_v0(cimage_path)
            .map_err(|e| format!("failed to load cimage '{}': {e}", cimage_path.display()))?;

        let weight_buffers = Self::upload_weights(&loaded)?;

        // ── 3. Allocate ring buffer (pinned host memory) ─────────────
        let ring_entries = unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let size = (RING_SIZE * ENTRY_STRIDE * 4) as u64;
            // hipHostMallocCoherent = 0x1 — coherent memory visible to both CPU and GPU
            let err = hipHostMalloc(&mut ptr, size, 1);
            if err != HIP_SUCCESS {
                return Err(format!(
                    "hipHostMalloc(ring_entries, {size}) returned {err}"
                ));
            }
            std::ptr::write_bytes(ptr as *mut u8, 0, size as usize);
            ptr as *mut u32
        };

        let completion_counter = unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let err = hipHostMalloc(&mut ptr, 4, 1);
            if err != HIP_SUCCESS {
                Self::free_pinned(ring_entries, std::ptr::null_mut(), std::ptr::null_mut());
                return Err(format!("hipHostMalloc(completion_counter) returned {err}"));
            }
            std::ptr::write(ptr as *mut u32, 0);
            ptr as *mut u32
        };

        let shutdown_flag = unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let err = hipHostMalloc(&mut ptr, 4, 1);
            if err != HIP_SUCCESS {
                Self::free_pinned(ring_entries, completion_counter, std::ptr::null_mut());
                return Err(format!("hipHostMalloc(shutdown_flag) returned {err}"));
            }
            std::ptr::write(ptr as *mut u32, 0);
            ptr as *mut u32
        };

        // ── 4. Allocate device buffers ───────────────────────────────

        // Logits: RING_SIZE * vocab_size * sizeof(half)
        let logits_bytes = (RING_SIZE as u64) * (vocab_size as u64) * 2;
        let logits_out = unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let err = hipMalloc(&mut ptr, logits_bytes);
            if err != HIP_SUCCESS {
                Self::free_pinned(ring_entries, completion_counter, shutdown_flag);
                return Err(format!(
                    "hipMalloc(logits_out, {logits_bytes}) returned {err}"
                ));
            }
            ptr
        };

        let kv_block_bytes = Self::compute_kv_cache_bytes(num_slots as u64, hidden_dim, num_layers);
        let kv_cache = unsafe {
            let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let err = hipMalloc(&mut ptr, kv_block_bytes);
            if err != HIP_SUCCESS {
                Self::free_pinned(ring_entries, completion_counter, shutdown_flag);
                let _ = hipFree(logits_out);
                return Err(format!(
                    "hipMalloc(kv_cache, {kv_block_bytes}) returned {err}"
                ));
            }
            ptr
        };

        // ── 5. Create stream ─────────────────────────────────────────
        let stream = unsafe {
            let mut s: hipStream_t = std::ptr::null_mut();
            let err = hipStreamCreate(&mut s);
            if err != HIP_SUCCESS {
                Self::free_pinned(ring_entries, completion_counter, shutdown_flag);
                let _ = hipFree(logits_out);
                let _ = hipFree(kv_cache);
                return Err(format!("hipStreamCreate returned {err}"));
            }
            s
        };

        // ── 6. Load HIP module and kernel ────────────────────────────
        let module = Self::load_hip_module(&loaded)?;
        let kernel = Self::get_kernel_function(module)?;

        // ── 7. Launch persistent kernel ──────────────────────────────
        //
        // Grid: num_slots threadblocks × 1 × 1
        // Each threadblock owns one ring-buffer slot, passed via blockIdx.x.
        //
        // Kernel argument layout (matching ds4_persistent_decode):
        //   [0] ring_entries       — uint*     [RING_SIZE * 5]
        //   [1] ring_tail           — uint*     reserved for host polling
        //   [2] embed_weight        — half*     embedding table
        //   [3] kv_cache            — void*     [LAYERS * compressed KV]
        //   [4] logits_out          — half*     [RING_SIZE * VOCAB_SIZE]
        //   [5] completion_counter  — uint*
        //   [6] shutdown_flag       — uint*

        let block_threads = 256u32; // {{PLACEHOLDER}} — tune per model

        // ── Build kernel argument array ───────────────────────────────
        //
        // hipModuleLaunchKernel expects a void** array where each element
        // points to the storage for one kernel argument.  All storage must
        // be live for the duration of the call (which enqueues the kernel
        // on the stream — synchronous in terms of setup, but does not wait
        // for completion).
        //
        // We stack-allocate the argument storage inline so the references
        // are valid through the launch call.

        let arg_ring_entries = ring_entries as *mut std::ffi::c_void;
        let arg_ring_tail = completion_counter as *mut std::ffi::c_void;
        let arg_embed_weight = weight_buffers
            .get("embed_weight")
            .map(|b| b.device_ptr)
            .unwrap_or(std::ptr::null_mut());
        let arg_kv_cache = kv_cache;
        let arg_logits_out = logits_out;
        let arg_completion_counter = completion_counter as *mut std::ffi::c_void;
        let arg_shutdown_flag = shutdown_flag as *mut std::ffi::c_void;

        let mut kernel_args: [*mut std::ffi::c_void; 7] = [
            &mut arg_ring_entries as *mut *mut std::ffi::c_void as *mut std::ffi::c_void,
            &mut arg_ring_tail as *mut *mut std::ffi::c_void as *mut std::ffi::c_void,
            &mut arg_embed_weight as *mut *mut std::ffi::c_void as *mut std::ffi::c_void,
            &mut arg_kv_cache as *mut *mut std::ffi::c_void as *mut std::ffi::c_void,
            &mut arg_logits_out as *mut *mut std::ffi::c_void as *mut std::ffi::c_void,
            &mut arg_completion_counter as *mut *mut std::ffi::c_void as *mut std::ffi::c_void,
            &mut arg_shutdown_flag as *mut *mut std::ffi::c_void as *mut std::ffi::c_void,
        ];

        let err = unsafe {
            hipModuleLaunchKernel(
                kernel,
                num_slots,
                1,
                1, // grid dim
                block_threads,
                1,
                1, // block dim
                0, // shared mem
                stream,
                kernel_args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if err != HIP_SUCCESS {
            let _ = hipModuleUnload(module);
            Self::free_pinned(ring_entries, completion_counter, shutdown_flag);
            let _ = hipFree(logits_out);
            let _ = hipFree(kv_cache);
            let _ = hipStreamDestroy(stream);
            // weight_buffers freed by Drop as they go out of scope
            return Err(format!("hipModuleLaunchKernel returned {err}"));
        }

        Ok(Self {
            ring_entries,
            completion_counter,
            shutdown_flag,
            logits_out,
            kv_cache,
            weight_buffers,
            module,
            kernel,
            stream,
            num_slots,
            device_id,
            last_completed: AtomicU32::new(0),
            vocab_size,
            hidden_dim,
            num_layers,
            kv_cache_bytes: kv_block_bytes,
            _private: (),
        })
    }

    // ── Work submission ─────────────────────────────────────────────────

    /// Submit a decode work item into the ring buffer.
    ///
    /// Writes a single 5-u32 entry at `slot_id` with state `TOKEN_READY`.
    /// The GPU threadblock polling that slot will claim it and run decode.
    ///
    /// Returns immediately after the store — the GPU executes asynchronously.
    ///
    /// # Panics
    ///
    /// Panics if `slot_id >= num_slots`.
    pub fn submit_work(&self, slot_id: u32, token_id: u32, seq_pos: u32) {
        assert!(
            slot_id < self.num_slots,
            "submit_work: slot_id {slot_id} >= num_slots {}",
            self.num_slots
        );
        unsafe {
            let idx = slot_id as usize;
            let entry = self.ring_entries.add(idx * ENTRY_STRIDE);
            *entry.add(OFFSET_STATE) = SlotState::TokenReady as u32;
            entry.add(OFFSET_TOKEN_ID).write(token_id);
            entry.add(OFFSET_SEQ_POS).write(seq_pos);
            entry.add(OFFSET_KV_SLOT_ID).write(slot_id);
            entry.add(OFFSET_RESERVED).write(0);
            fence(Ordering::SeqCst);
        }
    }

    /// Poll whether the GPU has completed work for this slot.
    ///
    /// The GPU atomically stores `COMPLETED` into the ring entry's state
    /// field after finishing decode.  We spin-read that field.
    ///
    /// # Panics
    ///
    /// Panics if `slot_id >= num_slots`.
    pub fn poll_work(&self, slot_id: u32) -> bool {
        assert!(
            slot_id < self.num_slots,
            "poll_work: slot_id {slot_id} >= num_slots {}",
            self.num_slots
        );
        unsafe {
            let idx = slot_id as usize;
            let entry = self.ring_entries.add(idx * ENTRY_STRIDE);
            let state = *entry.add(OFFSET_STATE);
            state == SlotState::Completed as u32
        }
    }

    /// Read logits for a completed slot.
    ///
    /// Copies `vocab_size` fp16 values from the device logits buffer
    /// into a host `Vec<u16>`.
    ///
    /// # Panics
    ///
    /// Panics if `slot_id >= num_slots`.
    pub fn read_logits(&self, slot_id: u32) -> Vec<u16> {
        assert!(
            slot_id < self.num_slots,
            "read_logits: slot_id {slot_id} >= num_slots {}",
            self.num_slots
        );
        let n = self.vocab_size as usize;
        let mut host_buf = vec![0u16; n];
        unsafe {
            let offset = (slot_id as u64) * (self.vocab_size as u64) * 2;
            let src =
                (self.logits_out as *const u8).add(offset as usize) as *const std::ffi::c_void;
            let dst = host_buf.as_mut_ptr() as *mut std::ffi::c_void;
            let bytes = (self.vocab_size as u64) * 2;
            let err = hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost);
            if err != HIP_SUCCESS {
                return vec![0u16; n];
            }
        }
        host_buf
    }

    /// Reset a ring slot back to `EMPTY` after reading its results.
    ///
    /// This recycles the slot so the host can write new work into it.
    ///
    /// # Panics
    ///
    /// Panics if `slot_id >= num_slots`.
    pub fn reset_work_slot(&self, slot_id: u32) {
        assert!(
            slot_id < self.num_slots,
            "reset_work_slot: slot_id {slot_id} >= num_slots {}",
            self.num_slots
        );
        unsafe {
            let idx = slot_id as usize;
            let entry = self.ring_entries.add(idx * ENTRY_STRIDE);
            *entry.add(OFFSET_STATE) = SlotState::Empty as u32;
            fence(Ordering::SeqCst);
        }
    }

    /// Spin until the completion counter advances past the last known value.
    ///
    /// The GPU kernel increments `completion_counter` once per completed slot.
    /// This is a coarse progress indicator; use `poll_work` for per-slot checks.
    pub fn wait_for_completion(&self) {
        let known = self.last_completed.load(Ordering::Acquire);
        loop {
            let completed = unsafe { *self.completion_counter };
            if completed > known {
                self.last_completed.store(completed, Ordering::Release);
                return;
            }
            std::hint::spin_loop();
        }
    }

    // ── Shutdown ────────────────────────────────────────────────────────

    /// Signal the persistent kernel to shut down and free all GPU resources.
    ///
    /// 1. Writes `1` to the shutdown flag (GPU polls this at loop top).
    /// 2. Synchronizes the stream / device.
    /// 3. Frees all VRAM allocations and pinned host memory.
    /// 4. Destroys the module and stream.
    pub fn shutdown(&mut self) {
        unsafe {
            *self.shutdown_flag = 1;
            fence(Ordering::SeqCst);

            let _ = hipStreamSynchronize(self.stream);
            let _ = hipDeviceSynchronize();

            // Free device buffers.
            // Weight buffers are freed by DeviceWeightBuffer::drop as we drain.
            self.weight_buffers.clear();
            let _ = hipFree(self.logits_out);
            let _ = hipFree(self.kv_cache);

            // Free pinned host memory.
            let _ = hipHostFree(self.ring_entries as *mut std::ffi::c_void);
            let _ = hipHostFree(self.completion_counter as *mut std::ffi::c_void);
            let _ = hipHostFree(self.shutdown_flag as *mut std::ffi::c_void);

            let _ = hipModuleUnload(self.module);
            let _ = hipStreamDestroy(self.stream);
        }

        self.module = std::ptr::null_mut();
        self.kernel = std::ptr::null_mut();
        self.stream = std::ptr::null_mut();
        self.ring_entries = std::ptr::null_mut();
        self.completion_counter = std::ptr::null_mut();
        self.shutdown_flag = std::ptr::null_mut();
        self.logits_out = std::ptr::null_mut();
        self.kv_cache = std::ptr::null_mut();
    }

    /// Number of configured decode slots.
    pub fn num_slots(&self) -> u32 {
        self.num_slots
    }

    // ── Private helpers ─────────────────────────────────────────────────

    /// Iterate every tensor in the CImage manifest and copy its payload
    /// bytes from the payload blob into a freshly allocated device buffer.
    fn upload_weights(
        loaded: &LoadedCImageV0,
    ) -> Result<HashMap<String, DeviceWeightBuffer>, String> {
        let blob = &loaded.payload_blob;
        let payload_dir = &loaded.payload_directory;
        let mut buffers = HashMap::new();

        // Build a lookup from payload_id -> CImagePayloadEntry
        let payload_map: HashMap<&str, &CImagePayloadEntry> = payload_dir
            .payloads
            .iter()
            .map(|e| (e.payload_id.as_str(), e))
            .collect();

        for tensor in &loaded.manifest.tensors {
            let payload_id = match &tensor.payload_ref {
                CImagePayloadRef::Single { payload_id } => payload_id.as_str(),
                CImagePayloadRef::MixedPrecision {
                    base_payload_id, ..
                } => base_payload_id.as_str(),
            };

            let entry = payload_map.get(payload_id).ok_or_else(|| {
                format!(
                    "tensor '{}' references payload '{}' not found in directory",
                    tensor.tensor_id, payload_id
                )
            })?;

            let offset = entry.offset as usize;
            let len = entry.len as usize;

            if offset + len > blob.len() {
                return Err(format!(
                    "payload '{}' range [{}, {}) exceeds blob size {}",
                    payload_id,
                    offset,
                    offset + len,
                    blob.len()
                ));
            }

            let device_ptr = unsafe {
                let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
                let err = hipMalloc(&mut ptr, len as u64);
                if err != HIP_SUCCESS {
                    // Already-allocated buffers freed by DeviceWeightBuffer::drop
                    // as the HashMap is dropped on return.
                    return Err(format!(
                        "hipMalloc(weight '{}', {}) returned {err}",
                        tensor.tensor_id, len
                    ));
                }

                let src = blob.as_ptr().add(offset) as *const std::ffi::c_void;
                let err2 = hipMemcpy(ptr, src, len as u64, hipMemcpyHostToDevice);
                if err2 != HIP_SUCCESS {
                    let _ = hipFree(ptr);
                    return Err(format!(
                        "hipMemcpy(weight '{}', {}) returned {err2}",
                        tensor.tensor_id, len
                    ));
                }

                ptr
            };

            buffers.insert(
                tensor.tensor_id.clone(),
                DeviceWeightBuffer {
                    device_ptr,
                    bytes: len as u64,
                },
            );
        }

        Ok(buffers)
    }

    /// Load the compiled HIP code object (HSACO) from the CImage payload.
    fn load_hip_module(loaded: &LoadedCImageV0) -> Result<hipModule_t, String> {
        let module_payload_id = "ds4_persistent_decode.hip";
        let entry = loaded
            .payload_directory
            .payloads
            .iter()
            .find(|e| e.payload_id == module_payload_id)
            .ok_or_else(|| {
                format!(
                    "no payload '{}' found in cimage payload directory (available: {})",
                    module_payload_id,
                    loaded
                        .payload_directory
                        .payloads
                        .iter()
                        .map(|e| e.payload_id.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?;

        let offset = entry.offset as usize;
        let len = entry.len as usize;

        if offset + len > loaded.payload_blob.len() {
            return Err(format!(
                "module payload range [{}, {}) exceeds blob size {}",
                offset,
                offset + len,
                loaded.payload_blob.len()
            ));
        }

        let module_bytes = &loaded.payload_blob[offset..offset + len];

        unsafe {
            let mut module: hipModule_t = std::ptr::null_mut();
            let err = hipModuleLoadData(
                &mut module,
                module_bytes.as_ptr() as *const std::ffi::c_void,
            );
            if err != HIP_SUCCESS {
                return Err(format!("hipModuleLoadData returned {err}"));
            }
            Ok(module)
        }
    }

    /// Resolve the kernel entry point from a loaded HIP module.
    fn get_kernel_function(module: hipModule_t) -> Result<hipFunction_t, String> {
        let name = CString::new("ds4_persistent_decode")
            .map_err(|_| "kernel function name contains NUL".to_string())?;
        unsafe {
            let mut func: hipFunction_t = std::ptr::null_mut();
            let err = hipModuleGetFunction(&mut func, module, name.as_ptr());
            if err != HIP_SUCCESS {
                return Err(format!(
                    "hipModuleGetFunction('ds4_persistent_decode') returned {err}"
                ));
            }
            Ok(func)
        }
    }

    /// Compute the KV cache device allocation size from model config.
    fn compute_kv_cache_bytes(num_slots: u64, hidden_dim: u64, num_layers: u64) -> u64 {
        // {{PLACEHOLDER}} — tune for actual model architecture.
        let kv_block_size: u64 = 128;
        let compress_ratio: u64 = 4;
        let max_context: u64 = 1_024_000;
        let bytes_per_element: u64 = 2; // fp16

        // Compressed KV per-layer per-slot:
        // (max_context / compress_ratio) positions × kv_block_size × 2 (K+V) × sizeof(half)
        let kv_per_layer_per_slot =
            (max_context / compress_ratio) * kv_block_size * 2 * bytes_per_element;
        num_slots * num_layers * kv_per_layer_per_slot
    }

    /// Free the three pinned-host allocations in one shot (error-path helper).
    /// Accepts null pointers gracefully (allows partial cleanup).
    fn free_pinned(ring_entries: *mut u32, completion_counter: *mut u32, shutdown_flag: *mut u32) {
        unsafe {
            if !ring_entries.is_null() {
                let _ = hipHostFree(ring_entries as *mut std::ffi::c_void);
            }
            if !completion_counter.is_null() {
                let _ = hipHostFree(completion_counter as *mut std::ffi::c_void);
            }
            if !shutdown_flag.is_null() {
                let _ = hipHostFree(shutdown_flag as *mut std::ffi::c_void);
            }
        }
    }
}

impl Drop for AmdDs4Megakernel {
    fn drop(&mut self) {
        if !self.module.is_null() {
            unsafe {
                let _ = hipStreamSynchronize(self.stream);
                let _ = hipDeviceSynchronize();
                let _ = hipModuleUnload(self.module);
                let _ = hipStreamDestroy(self.stream);

                // Weight buffers freed by DeviceWeightBuffer::drop via clear().
                self.weight_buffers.clear();

                if !self.logits_out.is_null() {
                    let _ = hipFree(self.logits_out);
                }
                if !self.kv_cache.is_null() {
                    let _ = hipFree(self.kv_cache);
                }

                if !self.ring_entries.is_null() {
                    let _ = hipHostFree(self.ring_entries as *mut std::ffi::c_void);
                }
                if !self.completion_counter.is_null() {
                    let _ = hipHostFree(self.completion_counter as *mut std::ffi::c_void);
                }
                if !self.shutdown_flag.is_null() {
                    let _ = hipHostFree(self.shutdown_flag as *mut std::ffi::c_void);
                }
            }
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(RING_SIZE, 512);
        assert_eq!(SlotState::Empty as u32, 0);
        assert_eq!(SlotState::TokenReady as u32, 1);
        assert_eq!(SlotState::Processing as u32, 2);
        assert_eq!(SlotState::Completed as u32, 3);
        assert_eq!(ENTRY_STRIDE, 5);
    }

    #[test]
    fn test_slot_state_repr() {
        assert!(SlotState::Empty as u32 < SlotState::TokenReady as u32);
        assert!(SlotState::TokenReady as u32 < SlotState::Processing as u32);
        assert!(SlotState::Processing as u32 < SlotState::Completed as u32);
    }
}
