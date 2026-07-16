//! Core ML stateful prediction bridge — Rust FFI bindings.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::arena::Arena;
use crate::arena_info::ArenaInfo;

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
    if waker_ptr.is_null() {
        return;
    }
    let waker = &*(waker_ptr as *const std::task::Waker);
    waker.wake_by_ref();
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
        unsafe { tribunus_coreml_stateful_request_is_complete(self.ptr) != 0 }
    }
}

impl Future for CoreMlStatefulRequest {
    type Output = Result<(), String>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.is_complete() {
            let ptr = self.ptr;
            self.ptr = std::ptr::null_mut();
            if ptr.is_null() {
                return Poll::Ready(Ok(()));
            }
            let rc = unsafe { tribunus_coreml_stateful_request_wait(ptr) };
            unsafe { tribunus_coreml_stateful_request_destroy(ptr) };
            if rc != 0 {
                return Poll::Ready(Err(format!(
                    "tribunus_coreml_stateful_request_wait failed: {}",
                    rc
                )));
            }
            Poll::Ready(Ok(()))
        } else {
            let waker = cx.waker().clone();
            let waker_ptr = Box::into_raw(Box::new(waker)) as *mut std::ffi::c_void;
            unsafe { tribunus_coreml_stateful_request_set_waker(self.ptr, waker_ptr) };
            Poll::Pending
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
    /// Create a state handle from a model.
    pub fn create(model: &super::coreml_bridge::CoreMlModel) -> Result<Self, String> {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let rc = unsafe { tribunus_coreml_state_create(&mut ptr, model.ptr) };
        if rc != 0 {
            return Err(format!("tribunus_coreml_state_create failed: {}", rc));
        }
        if ptr.is_null() {
            return Err("tribunus_coreml_state_create returned null".to_string());
        }
        Ok(CoreMlStateHandle { ptr })
    }

    /// Synchronous stateful prediction.
    pub fn predict(
        &self,
        model: &super::coreml_bridge::CoreMlModel,
        input_name: &str,
        input_arena: &ArenaInfo,
        output_name: &str,
        output_arena: &mut ArenaInfo,
    ) -> Result<(), String> {
        let input_cname =
            std::ffi::CString::new(input_name).map_err(|e| format!("CString: {}", e))?;
        let output_cname =
            std::ffi::CString::new(output_name).map_err(|e| format!("CString: {}", e))?;
        let rc = unsafe {
            tribunus_coreml_predict_stateful(
                model.ptr,
                self.ptr,
                input_cname.as_ptr(),
                input_arena,
                output_cname.as_ptr(),
                output_arena,
            )
        };
        if rc != 0 {
            return Err(format!("tribunus_coreml_predict_stateful failed: {}", rc));
        }
        Ok(())
    }

    /// Async stateful prediction.
    pub fn predict_async(
        &self,
        model: &super::coreml_bridge::CoreMlModel,
        input_name: &str,
        input_arena: &ArenaInfo,
        output_name: &str,
        output_arena: &mut ArenaInfo,
    ) -> Result<CoreMlStatefulRequest, String> {
        let input_cname =
            std::ffi::CString::new(input_name).map_err(|e| format!("CString: {}", e))?;
        let output_cname =
            std::ffi::CString::new(output_name).map_err(|e| format!("CString: {}", e))?;
        let mut request_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let rc = unsafe {
            tribunus_coreml_predict_stateful_async(
                &mut request_ptr,
                model.ptr,
                self.ptr,
                input_cname.as_ptr(),
                input_arena,
                output_cname.as_ptr(),
                output_arena,
            )
        };
        if rc != 0 {
            return Err(format!(
                "tribunus_coreml_predict_stateful_async failed: {}",
                rc
            ));
        }
        if request_ptr.is_null() {
            return Err("tribunus_coreml_predict_stateful_async returned null".to_string());
        }
        Ok(CoreMlStatefulRequest { ptr: request_ptr })
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
    pub fn new(model: &super::coreml_bridge::CoreMlModel) -> Result<Self, String> {
        let state = CoreMlStateHandle::create(model)?;
        Ok(StatefulPrefillContext {
            state,
            k_arena: None,
            v_arena: None,
        })
    }

    /// Run a single chunk through the stateful model.
    ///
    /// `chunk_size` is the number of tokens in this chunk (e.g. 64).
    /// `n_kv_heads` and `head_dim` define the KV cache tile dimensions.
    ///
    /// The K/V output arenas are lazily allocated on the first call.
    pub fn prefill_chunk(
        &mut self,
        model: &super::coreml_bridge::CoreMlModel,
        input_arena: &ArenaInfo,
        chunk_size: u32,
        n_kv_heads: u32,
        head_dim: u32,
    ) -> Result<(), String> {
        // Allocate K/V output arenas on first call
        if self.k_arena.is_none() {
            let kv_elements = (chunk_size * n_kv_heads * head_dim) as u32;
            let k_arena = Arena::new(1, kv_elements, crate::arena::Dtype::Float16)
                .map_err(|e| format!("k_arena alloc: {e}"))?;
            let v_arena = Arena::new(1, kv_elements, crate::arena::Dtype::Float16)
                .map_err(|e| format!("v_arena alloc: {e}"))?;
            self.k_arena = Some(k_arena);
            self.v_arena = Some(v_arena);
        }

        // Predict: model writes attention output + statefully updates K/V cache
        self.state
            .predict(
                model,
                "input",
                input_arena,
                "output",
                &mut self.k_arena.as_mut().unwrap().info,
            )
            .map_err(|e| format!("stateful predict failed: {e}"))?;

        Ok(())
    }
}

unsafe impl Send for StatefulPrefillContext {}
unsafe impl Sync for StatefulPrefillContext {}
