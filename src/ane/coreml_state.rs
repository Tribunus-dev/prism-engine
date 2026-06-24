//! Core ML stateful prediction bridge — Rust FFI bindings.

use crate::ane::arena::Arena;
use crate::ane::arena_info::ArenaInfo;

extern "C" {
    fn tribunus_coreml_state_create(
        out_state: *mut *mut std::ffi::c_void,
        model: *mut std::ffi::c_void,
    ) -> i32;

    fn tribunus_coreml_state_destroy(state: *mut std::ffi::c_void);

    fn tribunus_coreml_predict_stateful(
        model: *mut std::ffi::c_void,
        state: *mut std::ffi::c_void,
        input_name: *const i8,
        input_arena: *const ArenaInfo,
        output_name: *const i8,
        output_arena: *mut ArenaInfo,
    ) -> i32;

    fn tribunus_coreml_predict_stateful_async(
        out_request: *mut *mut std::ffi::c_void,
        model: *mut std::ffi::c_void,
        state: *mut std::ffi::c_void,
        input_name: *const i8,
        input_arena: *const ArenaInfo,
        output_name: *const i8,
        output_arena: *mut ArenaInfo,
    ) -> i32;

    fn tribunus_coreml_stateful_request_is_complete(request: *mut std::ffi::c_void) -> i32;
    fn tribunus_coreml_stateful_request_set_waker(
        request: *mut std::ffi::c_void,
        waker: *mut std::ffi::c_void,
    );
    fn tribunus_coreml_stateful_request_wait(request: *mut std::ffi::c_void) -> i32;
    fn tribunus_coreml_stateful_request_destroy(request: *mut std::ffi::c_void);
}

/// Dynamic callback for the C bridge to wake a Rust task.
#[no_mangle]
pub unsafe extern "C" fn tribunus_coreml_wake_waker(waker_ptr: *mut std::ffi::c_void) {
    if !waker_ptr.is_null() {
        let waker = Box::from_raw(waker_ptr as *mut std::task::Waker);
        waker.wake();
    }
}

/// Stateful prediction request handle.
pub struct CoreMlStatefulRequest {
    ptr: *mut std::ffi::c_void,
}

impl CoreMlStatefulRequest {
    pub fn is_complete(&self) -> bool {
        if self.ptr.is_null() {
            return true;
        }
        unsafe { tribunus_coreml_stateful_request_is_complete(self.ptr) == 1 }
    }

    pub fn wait(mut self) -> Result<(), String> {
        if self.ptr.is_null() {
            return Err("Null request pointer".to_string());
        }
        let ptr = self.ptr;
        self.ptr = std::ptr::null_mut();
        let status = unsafe { tribunus_coreml_stateful_request_wait(ptr) };
        unsafe { tribunus_coreml_stateful_request_destroy(ptr) };
        if status != 0 {
            return Err(format!("async prediction failed with status: {}", status));
        }
        Ok(())
    }
}

impl std::future::Future for CoreMlStatefulRequest {
    type Output = Result<(), String>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if self.is_complete() {
            let ptr = self.ptr;
            self.ptr = std::ptr::null_mut();
            if ptr.is_null() {
                return std::task::Poll::Ready(Ok(()));
            }
            let status = unsafe { tribunus_coreml_stateful_request_wait(ptr) };
            unsafe { tribunus_coreml_stateful_request_destroy(ptr) };
            if status != 0 {
                std::task::Poll::Ready(Err(format!(
                    "async prediction failed with status: {}",
                    status
                )))
            } else {
                std::task::Poll::Ready(Ok(()))
            }
        } else {
            let waker = Box::new(cx.waker().clone());
            let waker_ptr = Box::into_raw(waker) as *mut std::ffi::c_void;
            unsafe {
                tribunus_coreml_stateful_request_set_waker(self.ptr, waker_ptr);
            }
            std::task::Poll::Pending
        }
    }
}

impl Drop for CoreMlStatefulRequest {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { tribunus_coreml_stateful_request_destroy(self.ptr) };
        }
    }
}

/// Owned Core ML state handle.
pub struct CoreMlStateHandle {
    ptr: *mut std::ffi::c_void,
}

impl CoreMlStateHandle {
    /// Create a new state from a loaded model.
    pub fn new(model_ptr: *mut std::ffi::c_void) -> Result<Self, String> {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let status = unsafe { tribunus_coreml_state_create(&mut ptr, model_ptr) };
        if status != 0 {
            return Err(format!("tribunus_coreml_state_create failed: {}", status));
        }
        Ok(CoreMlStateHandle { ptr })
    }

    /// Run stateful prediction with IOSurface-backed arenas.
    pub fn predict_stateful(
        &self,
        model_ptr: *mut std::ffi::c_void,
        input_name: &str,
        input_arena: &ArenaInfo,
        output_name: &str,
        output_arena: &mut ArenaInfo,
    ) -> Result<(), String> {
        let c_in = std::ffi::CString::new(input_name).map_err(|e| format!("CString: {}", e))?;
        let c_out = std::ffi::CString::new(output_name).map_err(|e| format!("CString: {}", e))?;
        let status = unsafe {
            tribunus_coreml_predict_stateful(
                model_ptr,
                self.ptr,
                c_in.as_ptr(),
                input_arena,
                c_out.as_ptr(),
                output_arena,
            )
        };
        if status != 0 {
            return Err(format!(
                "tribunus_coreml_predict_stateful failed: {}",
                status
            ));
        }
        Ok(())
    }

    /// Start an async stateful prediction with IOSurface-backed arenas.
    pub fn predict_stateful_async(
        &self,
        model_ptr: *mut std::ffi::c_void,
        input_name: &str,
        input_arena: &ArenaInfo,
        output_name: &str,
        output_arena: &mut ArenaInfo,
    ) -> Result<CoreMlStatefulRequest, String> {
        let c_in = std::ffi::CString::new(input_name).map_err(|e| format!("CString: {}", e))?;
        let c_out = std::ffi::CString::new(output_name).map_err(|e| format!("CString: {}", e))?;
        let mut req_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let status = unsafe {
            tribunus_coreml_predict_stateful_async(
                &mut req_ptr,
                model_ptr,
                self.ptr,
                c_in.as_ptr(),
                input_arena,
                c_out.as_ptr(),
                output_arena,
            )
        };
        if status != 0 {
            return Err(format!(
                "tribunus_coreml_predict_stateful_async failed: {}",
                status
            ));
        }
        Ok(CoreMlStatefulRequest { ptr: req_ptr })
    }
}

impl Drop for CoreMlStateHandle {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { tribunus_coreml_state_destroy(self.ptr) };
        }
    }
}

// Safety: MLState is documented as thread-safe for prediction in isolated sessions.
unsafe impl Send for CoreMlStateHandle {}
unsafe impl Sync for CoreMlStateHandle {}

/// Context for stateful ANE prefill with IOSurface-backed KV outputs.
///
/// Holds the MLState handle + output arenas for K and V chunks.
/// After `prefill_chunk()`, callers extract the IOSurface from
/// `k_arena` / `v_arena` and bind to Metal for GPU decode.
pub struct StatefulPrefillContext {
    pub state: CoreMlStateHandle,
    pub k_arena: Option<Arena>,
    pub v_arena: Option<Arena>,
}

impl StatefulPrefillContext {
    /// Create a new context with a fresh MLState from a loaded model.
    pub fn new(model_ptr: *mut std::ffi::c_void) -> Result<Self, String> {
        let state = CoreMlStateHandle::new(model_ptr)?;
        Ok(StatefulPrefillContext {
            state,
            k_arena: None,
            v_arena: None,
        })
    }

    /// Run prefill for one chunk, extracting K and V into IOSurface-backed arenas.
    ///
    /// Allocates K/V output arenas on first call. Subsequent calls reuse them.
    pub fn prefill_chunk(
        &mut self,
        model_ptr: *mut std::ffi::c_void,
        input_arena: &ArenaInfo,
        output_arena: &mut ArenaInfo,
        chunk_size: u32,
        n_kv_heads: u32,
        head_dim: u32,
    ) -> Result<(), String> {
        // Allocate K/V output arenas on first call
        if self.k_arena.is_none() {
            let kv_elements = (chunk_size * n_kv_heads * head_dim) as u32;
            use crate::ane::arena::Dtype;
            let k_arena = Arena::new(1, kv_elements, Dtype::Float16)
                .map_err(|e| format!("k_arena alloc: {e}"))?;
            let v_arena = Arena::new(1, kv_elements, Dtype::Float16)
                .map_err(|e| format!("v_arena alloc: {e}"))?;
            self.k_arena = Some(k_arena);
            self.v_arena = Some(v_arena);
        }

        // Predict: model writes attention output + statefully updates K/V cache
        self.state
            .predict_stateful(model_ptr, "input", input_arena, "output", output_arena)?;

        // Extract K and V chunks as separate model outputs
        if let Some(k_arena) = &self.k_arena {
            let mut k_info = k_arena.info.clone();
            self.state
                .predict_stateful(model_ptr, "input", input_arena, "k_chunk", &mut k_info)?;
        }
        if let Some(v_arena) = &self.v_arena {
            let mut v_info = v_arena.info.clone();
            self.state
                .predict_stateful(model_ptr, "input", input_arena, "v_chunk", &mut v_info)?;
        }

        Ok(())
    }
}

unsafe impl Send for StatefulPrefillContext {}
unsafe impl Sync for StatefulPrefillContext {}
