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
            CoreMlComputeUnits::CpuAndGpu => "cpuAndGpu",
            CoreMlComputeUnits::CpuAndNeuralEngine => "cpuAndNeuralEngine",
            CoreMlComputeUnits::All => "all",
        }
    }
}
#[allow(dead_code)]
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
    fn tribunus_coreml_predict_two(
        model: *mut std::ffi::c_void,
        input_name_a: *const i8,
        input_a: *const ArenaInfo,
        input_name_b: *const i8,
        input_b: *const ArenaInfo,
        output_name: *const i8,
        output_arena: *mut ArenaInfo,
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

    pub fn predict(
        &self,
        input_name: &str,
        input: &ArenaInfo,
        output_name: &str,
        output: &mut ArenaInfo,
    ) -> Result<(), String> {
        let input_name = std::ffi::CString::new(input_name).map_err(|e| e.to_string())?;
        let output_name = std::ffi::CString::new(output_name).map_err(|e| e.to_string())?;
        let status = unsafe {
            tribunus_coreml_predict(
                self.ptr,
                input_name.as_ptr(),
                input,
                output_name.as_ptr(),
                output,
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(format!("Core ML prediction failed: {status}"))
        }
    }

    pub fn predict_two(
        &self,
        a_name: &str,
        a: &ArenaInfo,
        b_name: &str,
        b: &ArenaInfo,
        output_name: &str,
        output: &mut ArenaInfo,
    ) -> Result<(), String> {
        let a_name = std::ffi::CString::new(a_name).map_err(|e| e.to_string())?;
        let b_name = std::ffi::CString::new(b_name).map_err(|e| e.to_string())?;
        let output_name = std::ffi::CString::new(output_name).map_err(|e| e.to_string())?;
        let status = unsafe {
            tribunus_coreml_predict_two(
                self.ptr,
                a_name.as_ptr(),
                a,
                b_name.as_ptr(),
                b,
                output_name.as_ptr(),
                output,
            )
        };
        if status == 0 {
            Ok(())
        } else {
            Err(format!("Core ML two-input prediction failed: {status}"))
        }
    }

    pub fn predict_two_int8(
        &self,
        a_name: &str,
        a: &crate::Arena,
        b_name: &str,
        b: &crate::Arena,
        output_name: &str,
        output: &mut crate::Arena,
    ) -> Result<(), String> {
        self.predict_two(
            a_name,
            &a.info,
            b_name,
            &b.info,
            output_name,
            &mut output.info,
        )
    }

    pub fn predict_two_int8_planar(
        &self,
        a_name: &str,
        a: &crate::Arena,
        b_name: &str,
        b: &crate::Arena,
        output_name: &str,
        output: &mut crate::Arena,
    ) -> Result<(), String> {
        self.predict_two_int8(a_name, a, b_name, b, output_name, output)
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
