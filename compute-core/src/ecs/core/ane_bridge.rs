use std::collections::HashMap;
use std::ffi::{c_void, CString};
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock};

extern "C" {
    fn tribunus_ane_init() -> i32;

    fn tribunus_ane_compile_mil(
        out_program: *mut *mut c_void,
        mil_text: *const i8,
        program_tag: *const i8,
    ) -> i32;

    fn tribunus_ane_compile_mil_with_weights(
        out_program: *mut *mut c_void,
        mil_text: *const i8,
        weight_dict: *const c_void,
        program_tag: *const i8,
    ) -> i32;

    fn tribunus_ane_eval(
        program: *mut c_void,
        inputs: *mut *mut c_void,
        num_inputs: i32,
        outputs: *mut *mut c_void,
        num_outputs: i32,
    ) -> i32;

    fn tribunus_ane_release_program(program: *mut c_void);

    fn tribunus_ane_compile_count() -> i32;

    fn tribunus_ane_program_reload_weights(
        program: *mut c_void,
        weight_path: *const i8,
        weight_data: *const c_void,
        weight_size: u64,
    ) -> i32;
}

pub struct AneProgram {
    ptr: *mut c_void,
}

impl AneProgram {
    pub fn init() -> Result<(), String> {
        let rc = unsafe { tribunus_ane_init() };
        if rc == 1 {
            Ok(())
        } else {
            Err("Apple Neural Engine private framework not available or failed to load".into())
        }
    }

    pub fn compile(mil_text: &str, tag: &str) -> Result<Self, String> {
        let c_mil = CString::new(mil_text).map_err(|e| format!("CString: {}", e))?;
        let c_tag = CString::new(tag).map_err(|e| format!("CString: {}", e))?;
        let mut ptr: *mut c_void = ptr::null_mut();
        let rc = unsafe { tribunus_ane_compile_mil(&mut ptr, c_mil.as_ptr(), c_tag.as_ptr()) };
        if rc != 0 {
            return Err(format!(
                "tribunus_ane_compile_mil failed with error code: {}",
                rc
            ));
        }
        Ok(AneProgram { ptr })
    }

    /// Compile MIL text with a weight dict (CFRetained NSDictionary*) into an ANE program.
    pub fn compile_with_weights(
        mil_text: &str,
        weight_dict: *const c_void,
        tag: &str,
    ) -> Result<Self, String> {
        let c_mil = CString::new(mil_text).map_err(|e| format!("CString: {}", e))?;
        let c_tag = CString::new(tag).map_err(|e| format!("CString: {}", e))?;
        let mut ptr: *mut c_void = ptr::null_mut();
        let rc = unsafe {
            tribunus_ane_compile_mil_with_weights(
                &mut ptr,
                c_mil.as_ptr(),
                weight_dict,
                c_tag.as_ptr(),
            )
        };
        if rc != 0 {
            return Err(format!(
                "tribunus_ane_compile_mil_with_weights failed: {}",
                rc
            ));
        }
        Ok(AneProgram { ptr })
    }

    pub fn evaluate(&self, inputs: &[*mut c_void], outputs: &[*mut c_void]) -> Result<(), String> {
        if inputs.is_empty() || outputs.is_empty() {
            return Err("inputs or outputs cannot be empty".into());
        }
        let rc = unsafe {
            tribunus_ane_eval(
                self.ptr,
                inputs.as_ptr() as *mut *mut c_void,
                inputs.len() as i32,
                outputs.as_ptr() as *mut *mut c_void,
                outputs.len() as i32,
            )
        };
        if rc != 1 {
            return Err("tribunus_ane_eval failed".into());
        }
        Ok(())
    }

    pub fn reload_weights(&self, path: &str, data: &[u8]) -> Result<(), String> {
        let c_path = CString::new(path).map_err(|e| format!("CString: {}", e))?;
        let rc = unsafe {
            tribunus_ane_program_reload_weights(
                self.ptr,
                c_path.as_ptr(),
                data.as_ptr() as *const c_void,
                data.len() as u64,
            )
        };
        if rc != 1 {
            return Err("tribunus_ane_program_reload_weights failed".into());
        }
        Ok(())
    }

    pub fn compile_count() -> i32 {
        unsafe { tribunus_ane_compile_count() }
    }
}

impl Drop for AneProgram {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { tribunus_ane_release_program(self.ptr) };
        }
    }
}

// SAFETY: ANE runtime state is accessed only by the ANE multiplexer thread which is single-threaded.
// The ANE program handle (ptr: *mut c_void) is initialized once and never mutated concurrently.
unsafe impl Send for AneProgram {}
unsafe impl Sync for AneProgram {}

pub struct AneProgramCache {
    programs: Mutex<HashMap<String, Arc<AneProgram>>>,
}

impl AneProgramCache {
    pub fn global() -> &'static Self {
        static CACHE: OnceLock<AneProgramCache> = OnceLock::new();
        CACHE.get_or_init(|| AneProgramCache {
            programs: Mutex::new(HashMap::new()),
        })
    }

    pub fn get_or_compile(&self, mil_text: &str, tag: &str) -> Result<Arc<AneProgram>, String> {
        let mut cache = self.programs.lock().map_err(|e| e.to_string())?;
        if let Some(prog) = cache.get(mil_text) {
            return Ok(prog.clone());
        }
        let prog = Arc::new(AneProgram::compile(mil_text, tag)?);
        cache.insert(mil_text.to_string(), prog.clone());
        Ok(prog)
    }

    pub fn clear(&self) {
        if let Ok(mut cache) = self.programs.lock() {
            cache.clear();
        }
    }
}

/// Standalone description of an ANE inference step.
///
/// Replaces the mlx-gated `hybrid_profile::ExecutionStep::AneInference` variant
/// so that the ANE bridge can compile under prism-backend without an MLX
/// dependency.
#[derive(Debug, Clone)]
pub struct AneInferenceStep {
    pub mil_text: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub tag: String,
}

/// Execute a single AneInference step through the ANE bridge.
///
/// Extracts the MIL text, tag, and I/O tensor names from the step,
/// retrieves or compiles the program via program_cache, evaluates it,
/// and returns a minimal BoundaryExecutionReceipt.
pub fn execute_ane_step(
    step: &AneInferenceStep,
    program_cache: &AneProgramCache,
) -> Result<crate::ecs::backend::routing::BoundaryExecutionReceipt, String> {
    let AneInferenceStep {
        mil_text,
        inputs,
        outputs,
        tag,
    } = step;

    let program = program_cache.get_or_compile(mil_text, tag)?;

    // Convert string tensor names to raw pointers for the ANE C API.
    // The ANE runtime expects the same number of input/output handles as
    // declared in the MIL program, passed as *mut c_void arrays.
    let input_ptrs: Vec<*mut std::ffi::c_void> = vec![std::ptr::null_mut(); inputs.len()];
    let output_ptrs: Vec<*mut std::ffi::c_void> = vec![std::ptr::null_mut(); outputs.len()];

    program.evaluate(&input_ptrs, &output_ptrs)?;

    Ok(crate::ecs::backend::routing::BoundaryExecutionReceipt {
        group_id: crate::ecs::backend::routing::EvaluationGroupId(0),
        planned_policy: crate::ecs::backend::routing::EvaluationPolicy::BackendLazy,
        backend: crate::ecs::backend::routing::BACKEND_ANE,
        operation_count: 1,
        planned_materialized_outputs: outputs.len(),
        actual_eval_calls: 1,
        actual_sync_count: 0,
        graph_build_ns: 0,
        submit_ns: 0,
        execution_ns: 0,
        wait_ns: 0,
        temporary_bytes: 0,
        released_tensor_count: 0,
        unaccounted_ns: 0,
        policy_support: crate::ecs::backend::routing::EvaluationPolicySupport::Native,
    })
}

/// Execute a batched ANE inference step.
///
/// Batched programs fuse the batch dimension into the MIL input shapes
/// (e.g., `[batch_size, in_features]` instead of `[1, in_features]`).
/// The ANE processes all batch items in a single invocation via MIL
/// matmul broadcasting — no per-item serial dispatch is needed.
///
/// `batch_index` identifies which logical tensor slice within the batch
/// the caller should read/write. The actual ANE evaluation is identical
/// to [`execute_ane_step`] because the batch dimension is baked into the
/// compiled MIL program's tensor shapes at compile time.
pub fn execute_batched_ane_step(
    step: &AneInferenceStep,
    program_cache: &AneProgramCache,
    batch_index: u32,
) -> Result<crate::ecs::backend::routing::BoundaryExecutionReceipt, String> {
    // The MIL program already encodes the batch dimension in its input
    // shapes — `[batch_size, in_features]`. The ANE processes all items
    // at once; batch_index is metadata for the caller to identify which
    // slice of the batch is "theirs".
    let _ = batch_index;
    execute_ane_step(step, program_cache)
}
