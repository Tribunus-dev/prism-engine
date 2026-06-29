//! Core ML execution bridge — Rust FFI bindings.

use crate::arena_info::ArenaInfo;

/// Compute unit policy for Core ML model loading.
/// Maps to MLComputeUnits in the ObjC bridge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i64)]
pub enum CoreMlComputeUnits {
    CpuOnly = 0,
    CpuAndGpu = 1,
    CpuAndNeuralEngine = 2,
    All = 3,
}

impl CoreMlComputeUnits {
    pub fn name(&self) -> &'static str {
        match self {
            CoreMlComputeUnits::CpuOnly => "cpuOnly",
            CoreMlComputeUnits::CpuAndGpu => "cpuAndGPU",
            CoreMlComputeUnits::CpuAndNeuralEngine => "cpuAndNeuralEngine",
            CoreMlComputeUnits::All => "all",
        }
    }
}

extern "C" {
    fn tribunus_coreml_load_model(
        out_model: *mut *mut std::ffi::c_void,
        path: *const i8,
        compute_units: i64,
    ) -> i32;
    fn tribunus_coreml_free_model(model: *mut std::ffi::c_void);
    fn tribunus_coreml_predict(
        model: *mut std::ffi::c_void,
        input_name: *const i8,
        input_arena: *const ArenaInfo,
        output_name: *const i8,
        output_arena: *const ArenaInfo,
    ) -> i32;
    fn tribunus_coreml_predict_pixelbuffer(
        model: *mut std::ffi::c_void,
        input_name: *const i8,
        input_arena: *const ArenaInfo,
        output_name: *const i8,
        output_arena: *mut ArenaInfo,
    ) -> i32;
    fn tribunus_coreml_predict_multi(
        model: *mut std::ffi::c_void,
        input_names: *mut *const i8,
        input_arenas: *mut *const ArenaInfo,
        num_inputs: i32,
        output_names: *mut *const i8,
        output_arenas: *mut *mut ArenaInfo,
        num_outputs: i32,
    ) -> i32;
}

/// Owned Core ML model handle.
pub struct CoreMlModel {
    pub(crate) ptr: *mut std::ffi::c_void,
}

impl CoreMlModel {
    /// Return the raw underlying FFI pointer.
    pub fn raw_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr
    }

    /// Load a compiled Core ML model from disk with the given compute unit policy.
    pub fn load(path: &str) -> Result<Self, String> {
        Self::load_with_compute_units(path, CoreMlComputeUnits::CpuAndNeuralEngine)
    }

    /// Load with explicit compute unit policy.
    pub fn load_with_compute_units(
        path: &str,
        compute_units: CoreMlComputeUnits,
    ) -> Result<Self, String> {
        let c_path = std::ffi::CString::new(path).map_err(|e| format!("CString: {}", e))?;
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let status =
            unsafe { tribunus_coreml_load_model(&mut ptr, c_path.as_ptr(), compute_units as i64) };
        if status != 0 {
            return Err(format!("tribunus_coreml_load_model failed: {}", status));
        }
        Ok(CoreMlModel { ptr })
    }

    /// Run prediction: input arena → model → output arena.
    /// Both arenas must remain alive until after the prediction completes.
    /// The output arena's data is valid after this call returns.
    pub fn predict(
        &self,
        input_name: &str,
        input_arena: &ArenaInfo,
        output_name: &str,
        output_arena: &ArenaInfo,
    ) -> Result<(), String> {
        let c_in_name =
            std::ffi::CString::new(input_name).map_err(|e| format!("CString: {}", e))?;
        let c_out_name =
            std::ffi::CString::new(output_name).map_err(|e| format!("CString: {}", e))?;
        let status = unsafe {
            tribunus_coreml_predict(
                self.ptr,
                c_in_name.as_ptr(),
                input_arena,
                c_out_name.as_ptr(),
                output_arena,
            )
        };
        if status != 0 {
            return Err(format!("tribunus_coreml_predict failed: {}", status));
        }
        Ok(())
    }

    /// Run prediction using the IOSurface/CVPixelBuffer path.
    ///
    /// Both arenas must be IOSurface-backed (created via `Arena::new`).
    /// The output arena's `ArenaInfo` may be updated with the output CVPixelBuffer
    /// metadata; the original IOSurface backing remains the same.
    pub fn predict_pixelbuffer(
        &self,
        input_name: &str,
        input_arena: &ArenaInfo,
        output_name: &str,
        output_arena: &mut ArenaInfo,
    ) -> Result<(), String> {
        let c_in_name =
            std::ffi::CString::new(input_name).map_err(|e| format!("CString: {}", e))?;
        let c_out_name =
            std::ffi::CString::new(output_name).map_err(|e| format!("CString: {}", e))?;
        let status = unsafe {
            tribunus_coreml_predict_pixelbuffer(
                self.ptr,
                c_in_name.as_ptr(),
                input_arena,
                c_out_name.as_ptr(),
                output_arena,
            )
        };
        if status != 0 {
            return Err(format!(
                "tribunus_coreml_predict_pixelbuffer failed: {}",
                status
            ));
        }
        Ok(())
    }
}

impl Drop for CoreMlModel {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { tribunus_coreml_free_model(self.ptr) };
        }
    }
}

impl CoreMlModel {
    /// Run prediction with multiple named inputs and outputs.
    /// All inputs are set up as a feature dictionary, all outputs as backings.
    /// The model is evaluated once with all I/O bound.
    pub fn predict_multi(
        &self,
        input_names: &[&str],
        input_infos: &[&ArenaInfo],
        output_names: &[&str],
        output_infos: &mut [&mut ArenaInfo],
    ) -> Result<(), String> {
        let c_input_names: Vec<std::ffi::CString> = input_names
            .iter()
            .map(|n| std::ffi::CString::new(*n).map_err(|e| format!("CString: {}", e)))
            .collect::<Result<Vec<_>, _>>()?;
        let c_output_names: Vec<std::ffi::CString> = output_names
            .iter()
            .map(|n| std::ffi::CString::new(*n).map_err(|e| format!("CString: {}", e)))
            .collect::<Result<Vec<_>, _>>()?;

        let mut c_in_ptrs: Vec<*const i8> = c_input_names.iter().map(|s| s.as_ptr()).collect();
        let mut c_out_ptrs: Vec<*const i8> = c_output_names.iter().map(|s| s.as_ptr()).collect();

        let mut in_arena_ptrs: Vec<*const ArenaInfo> =
            input_infos.iter().map(|a| &**a as *const ArenaInfo).collect();
        let mut out_arena_ptrs: Vec<*mut ArenaInfo> =
            output_infos.iter_mut().map(|a| &mut **a as *mut ArenaInfo).collect();

        let status = unsafe {
            tribunus_coreml_predict_multi(
                self.ptr,
                c_in_ptrs.as_mut_ptr(),
                in_arena_ptrs.as_mut_ptr(),
                input_names.len() as i32,
                c_out_ptrs.as_mut_ptr(),
                out_arena_ptrs.as_mut_ptr(),
                output_names.len() as i32,
            )
        };
        if status != 0 {
            return Err(format!("tribunus_coreml_predict_multi failed: {}", status));
        }
        Ok(())
    }
}

// Safety: MLModel is documented as thread-safe for prediction.
unsafe impl Send for CoreMlModel {}
unsafe impl Sync for CoreMlModel {}
