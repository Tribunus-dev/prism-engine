use crate::ane::arena_info::ArenaInfo;

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
            CoreMlComputeUnits::CpuAndGpu => "cpuAndGpu",
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
}

/// Owned Core ML model handle.
pub struct CoreMlModel {
    pub(crate) ptr: *mut std::ffi::c_void,
}

impl CoreMlModel {
    /// Load a Core ML model from a .mlmodelc directory.
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Result<Self, String> {
        let path_str = path
            .as_ref()
            .to_str()
            .ok_or_else(|| "non-UTF-8 path".to_string())?;
        let c_path = std::ffi::CString::new(path_str).map_err(|e| format!("CString: {}", e))?;
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let status = unsafe {
            tribunus_coreml_load_model(
                &mut ptr,
                c_path.as_ptr(),
                CoreMlComputeUnits::CpuAndNeuralEngine as i64,
            )
        };
        if status != 0 {
            return Err(format!("tribunus_coreml_load_model failed: {}", status));
        }
        if ptr.is_null() {
            return Err("tribunus_coreml_load_model returned null pointer".to_string());
        }
        Ok(CoreMlModel { ptr })
    }
}

impl Drop for CoreMlModel {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { tribunus_coreml_free_model(self.ptr) };
        }
    }
}

// Safety: MLModel is documented as thread-safe for prediction.
unsafe impl Send for CoreMlModel {}
unsafe impl Sync for CoreMlModel {}
