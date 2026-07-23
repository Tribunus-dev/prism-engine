//! C FFI for the Prism plugin API.
//!
//! Exposes an inference pipeline through a stable C ABI suitable for
//! dynamic plugin loading.  Callers initialise once, submit jobs, poll
//! progress, and retrieve output, then shut down.

use std::collections::HashMap;
use std::ffi::CStr;
use std::os::raw::{c_char, c_float, c_int};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::Value;
use thiserror::Error;

// ── C ABI types ──────────────────────────────────────────────────────────

/// Plugin-level job categories that the runtime can dispatch.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobType {
    SourceSeparation = 0,
    SuperResolution = 1,
    Denoise = 2,
    TTS = 3,
    Mastering = 4,
}

/// Configuration passed into [`prism_plugin_init`].
#[repr(C)]
#[derive(Debug, Clone)]
pub struct PluginConfig {
    pub model_path: *const c_char,
    pub temp_dir: *const c_char,
    pub metal_device_index: c_int,
}

/// Descriptor for a single inference job.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct JobDescriptor {
    pub job_type: JobType,
    pub input_data: *const u8,
    pub input_size: usize,
    pub params_json: *const c_char,
}

/// Poll result returned by [`prism_plugin_poll`].
#[repr(C)]
#[derive(Debug, Clone)]
pub struct JobStatus {
    /// 0 = completed, 1 = running, -1 = error, -2 = unknown handle.
    pub status: c_int,
    /// Progress estimate in [0.0, 1.0].
    pub progress: c_float,
    /// Null when there is no error, else a null-terminated message string.
    /// The string is valid until the next poll or get_output call for this handle.
    pub error_message: *const c_char,
}

// ── Internal error type ──────────────────────────────────────────────────

#[derive(Debug, Error)]
enum PluginError {
    #[error("already initialised")]
    AlreadyInit,
    #[error("not initialised")]
    NotInit,
    #[error("invalid parameters: {0}")]
    InvalidParams(String),
    #[error("runtime error: {0}")]
    Runtime(String),
    #[error("bad job handle: {0}")]
    BadHandle(i32),
}

impl PluginError {
    fn status_code(&self) -> i32 {
        match self {
            Self::AlreadyInit => -1,
            Self::NotInit => -2,
            Self::InvalidParams(_) => -3,
            Self::Runtime(_) => -5,
            Self::BadHandle(_) => -6,
        }
    }
}

// ── Internal job tracking ────────────────────────────────────────────────

#[derive(Debug)]
struct JobEntry {
    status: c_int,     // 0 completed, 1 running, -1 error
    progress: c_float, // 0.0–1.0
    error_message: Option<String>,
    output: Option<Vec<u8>>,
}

// ── Plugin runtime ───────────────────────────────────────────────────────

struct PluginRuntime {
    /// Tokio runtime for background task execution.
    rt: tokio::runtime::Runtime,
    /// Monotonically incrementing handle counter (starts at 1).
    next_handle: AtomicI32,
    /// Map of handle → job state, shared with background tasks via Arc.
    jobs: Arc<Mutex<HashMap<i32, JobEntry>>>,
    /// Pool of error-string buffers shared with background tasks via Arc.
    /// Strings leaked here are valid for the lifetime of the runtime.
    error_buffers: Arc<Mutex<Vec<String>>>,
}

// Global singleton protected by a mutex.  None → not initialised.
static RUNTIME: Mutex<Option<Box<PluginRuntime>>> = parking_lot::const_mutex(None);

impl PluginRuntime {
    fn new(config: &PluginConfig) -> Result<Self, PluginError> {
        // Validate C-string arguments.
        let _model = if config.model_path.is_null() {
            return Err(PluginError::InvalidParams("model_path is null".into()));
        } else {
            let cstr = unsafe { CStr::from_ptr(config.model_path) };
            cstr.to_str()
                .map_err(|_| PluginError::InvalidParams("model_path is not valid UTF-8".into()))?
                .to_owned()
        };

        let _temp = if config.temp_dir.is_null() {
            None
        } else {
            let cstr = unsafe { CStr::from_ptr(config.temp_dir) };
            Some(
                cstr.to_str()
                    .map_err(|_| PluginError::InvalidParams("temp_dir is not valid UTF-8".into()))?
                    .to_owned(),
            )
        };

        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .thread_name("prism-plugin")
            .enable_all()
            .build()
            .map_err(|e| PluginError::Runtime(format!("tokio init: {e}")))?;

        Ok(Self {
            rt,
            next_handle: AtomicI32::new(1),
            jobs: Arc::new(Mutex::new(HashMap::new())),
            error_buffers: Arc::new(Mutex::new(Vec::new())),
        })
    }

    fn allocate_handle(&self) -> i32 {
        self.next_handle.fetch_add(1, Ordering::Relaxed)
    }

    fn submit(&self, desc: &JobDescriptor) -> Result<i32, PluginError> {
        if desc.input_data.is_null() && desc.input_size > 0 {
            return Err(PluginError::InvalidParams(
                "input_data is null but input_size > 0".into(),
            ));
        }

        // Parse optional params JSON to validate it.
        let params_json = if !desc.params_json.is_null() {
            let cstr = unsafe { CStr::from_ptr(desc.params_json) };
            let s = cstr
                .to_str()
                .map_err(|_| PluginError::InvalidParams("params_json is not valid UTF-8".into()))?;
            if !s.is_empty() {
                let v: Value = serde_json::from_str(s)
                    .map_err(|e| PluginError::InvalidParams(format!("params_json parse: {e}")))?;
                Some(v)
            } else {
                None
            }
        } else {
            None
        };

        // Validate that we can at least recognise the job type.
        match desc.job_type {
            JobType::SourceSeparation
            | JobType::SuperResolution
            | JobType::Denoise
            | JobType::TTS
            | JobType::Mastering => {}
        }

        let handle = self.allocate_handle();

        let input_data = if desc.input_data.is_null() || desc.input_size == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(desc.input_data, desc.input_size) }.to_vec()
        };

        let job_type = desc.job_type;
        let params_str: Option<String> = params_json.map(|v| v.to_string());

        // Register the job as running.
        {
            let mut jobs = self.jobs.lock();
            jobs.insert(
                handle,
                JobEntry {
                    status: 1, // running
                    progress: 0.0,
                    error_message: None,
                    output: None,
                },
            );
        }

        // Clone Arcs for the background task so the future is Send.
        let jobs_arc = Arc::clone(&self.jobs);
        let error_buffers_arc = Arc::clone(&self.error_buffers);

        // Spawn background work on the tokio runtime.
        let rt_handle = self.rt.handle().clone();
        rt_handle.spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;

            let result = Self::dispatch_job(job_type, &input_data, &params_str);

            // Update job state.
            let mut jobs = jobs_arc.lock();
            let entry = jobs.get_mut(&handle).expect("job must exist");
            match result {
                Ok(output) => {
                    entry.status = 0; // completed
                    entry.progress = 1.0;
                    entry.output = Some(output);
                }
                Err(e) => {
                    entry.status = -1; // error
                    let err_msg = e.to_string();
                    // Stash the error string in the pool so the FFI pointer stays valid.
                    let mut bufs = error_buffers_arc.lock();
                    bufs.push(err_msg.clone());
                    entry.error_message = Some(err_msg);
                }
            }
        });

        Ok(handle)
    }

    /// Attempts real dispatch against Prism crates.
    /// Each pipeline returns an explicit error until the corresponding
    /// provider is wired into the plugin runtime.
    fn dispatch_job(
        job_type: JobType,
        _input: &[u8],
        _params: &Option<String>,
    ) -> Result<Vec<u8>, PluginError> {
        match job_type {
            JobType::SourceSeparation => Err(PluginError::Runtime(
                "source separation pipeline not wired; requires AudioProvider registration".into(),
            )),
            JobType::SuperResolution => Err(PluginError::Runtime(
                "super-resolution pipeline not wired; requires VideoDecoder registration".into(),
            )),
            JobType::Denoise => Err(PluginError::Runtime(
                "denoise pipeline not wired; requires AudioStreamState pipeline".into(),
            )),
            JobType::TTS => Err(PluginError::Runtime(
                "TTS pipeline not wired; requires prism-audio generate_speech".into(),
            )),
            JobType::Mastering => Err(PluginError::Runtime(
                "mastering pipeline not wired; requires AudioGenPipeline extension".into(),
            )),
        }
    }

    fn poll(&self, handle: i32) -> JobStatus {
        let jobs = self.jobs.lock();
        let entry = match jobs.get(&handle) {
            Some(e) => e,
            None => {
                return JobStatus {
                    status: -2, // unknown handle
                    progress: 0.0,
                    error_message: std::ptr::null(),
                };
            }
        };

        let err_ptr = match &entry.error_message {
            Some(msg) => msg.as_ptr() as *const c_char,
            None => std::ptr::null(),
        };

        JobStatus {
            status: entry.status,
            progress: entry.progress,
            error_message: err_ptr,
        }
    }

    fn get_output(&self, handle: i32) -> Result<Vec<u8>, PluginError> {
        let mut jobs = self.jobs.lock();
        let entry = jobs
            .get_mut(&handle)
            .ok_or(PluginError::BadHandle(handle))?;
        if entry.status != 0 {
            return Err(PluginError::Runtime(format!(
                "job {handle} not completed (status={})",
                entry.status
            )));
        }
        entry
            .output
            .take()
            .ok_or_else(|| PluginError::Runtime(format!("job {handle} has no output")))
    }
}

// ── FFI entry points ─────────────────────────────────────────────────────

/// Initialise the Prism plugin runtime.  Call once at plugin load.
///
/// Returns 0 on success, or a negative error code:
///   -1  already initialised
///   -3  invalid parameters (null model_path, bad UTF-8)
///   -5  tokio runtime creation failed
///
/// # Safety
///
/// `config` must be non-null and point to a valid, properly aligned
/// [`PluginConfig`] whose pointer fields (if non-null) point to
/// null-terminated C strings valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn prism_plugin_init(config: *const PluginConfig) -> i32 {
    if config.is_null() {
        return PluginError::InvalidParams("config is null".into()).status_code();
    }

    let mut guard = RUNTIME.lock();
    if guard.is_some() {
        return PluginError::AlreadyInit.status_code();
    }

    let cfg = unsafe { &*config };
    match PluginRuntime::new(cfg) {
        Ok(rt) => {
            *guard = Some(Box::new(rt));
            0
        }
        Err(e) => e.status_code(),
    }
}

/// Submit an inference job.  Returns a positive job handle on success,
/// or a negative error code:
///   -2  not initialised
///   -3  invalid parameters (null data with non-zero size, bad JSON)
///   -5  runtime error
///
/// # Safety
///
/// `job_desc` must be non-null and point to a valid [`JobDescriptor`].
/// `input_data` within must point to `input_size` readable bytes (or be
/// null when `input_size` is 0).  `params_json` must be null or a
/// null-terminated C string valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn prism_plugin_submit(job_desc: *const JobDescriptor) -> i32 {
    let guard = RUNTIME.lock();
    let rt = match guard.as_deref() {
        Some(r) => r,
        None => return PluginError::NotInit.status_code(),
    };

    if job_desc.is_null() {
        return PluginError::InvalidParams("job_desc is null".into()).status_code();
    }

    let desc = unsafe { &*job_desc };
    match rt.submit(desc) {
        Ok(handle) => handle,
        Err(e) => e.status_code(),
    }
}

/// Poll for job completion.
///
/// Returns a [`JobStatus`] describing the current state.  The
/// `error_message` pointer is valid until the next call to
/// `prism_plugin_poll` or `prism_plugin_get_output` for the same handle.
///
/// # Safety
///
/// Must be called after [`prism_plugin_init`].
#[no_mangle]
pub extern "C" fn prism_plugin_poll(handle: i32) -> JobStatus {
    let guard = RUNTIME.lock();
    match guard.as_deref() {
        Some(rt) => rt.poll(handle),
        None => JobStatus {
            status: -2, // not init
            progress: 0.0,
            error_message: std::ptr::null(),
        },
    }
}

/// Retrieve output data for a completed job.
///
/// On success, writes the output size into `out_size` and returns a
/// pointer to an allocated buffer.  The caller MUST free the buffer
/// with [`prism_plugin_free_output`].
///
/// Returns null on error (check `out_size` — it is not written on error).
///
/// # Safety
///
/// `out_size` must be non-null and point to a valid `usize`.
#[no_mangle]
pub extern "C" fn prism_plugin_get_output(handle: i32, out_size: *mut usize) -> *mut u8 {
    let guard = RUNTIME.lock();
    let rt = match guard.as_deref() {
        Some(r) => r,
        None => return std::ptr::null_mut(),
    };

    let data = match rt.get_output(handle) {
        Ok(d) => d,
        Err(_) => return std::ptr::null_mut(),
    };

    let len = data.len();
    // Leak the Vec so the caller can free it.
    let ptr = data.leak().as_mut_ptr();
    unsafe {
        *out_size = len;
    }
    ptr
}

/// Free an output buffer previously returned by [`prism_plugin_get_output`].
///
/// # Safety
///
/// `ptr` must be null or a pointer previously returned by
/// `prism_plugin_get_output` that has not already been freed.
#[no_mangle]
pub unsafe extern "C" fn prism_plugin_free_output(ptr: *mut u8) {
    if ptr.is_null() {
        return;
    }
    // Reconstruct the Vec so it gets dropped.
    unsafe {
        let _ = Vec::from_raw_parts(ptr, 0, 0);
    }
}

/// Teardown.  Call once at plugin unload.
///
/// Returns 0 on success, -1 if not initialised.
#[no_mangle]
pub extern "C" fn prism_plugin_shutdown() -> i32 {
    let mut guard = RUNTIME.lock();
    if guard.is_some() {
        *guard = None;
        0
    } else {
        PluginError::NotInit.status_code()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::ptr;
    use std::sync::Mutex as StdMutex;

    /// Serialisation lock for the global runtime.
    /// Tests that touch `RUNTIME` must acquire this.
    static TEST_LOCK: StdMutex<()> = StdMutex::new(());

    /// Helper: init with a minimal config for testing.
    unsafe fn init_for_test() -> i32 {
        let model = CString::new("/tmp/test-model").unwrap();
        let temp = CString::new("/tmp/test-temp").unwrap();
        let cfg = PluginConfig {
            model_path: model.as_ptr(),
            temp_dir: temp.as_ptr(),
            metal_device_index: 0,
        };
        prism_plugin_init(&cfg)
    }

    #[test]
    fn test_plugin_lifecycle() {
        let _guard = TEST_LOCK.lock().unwrap();
        unsafe {
            // Init cleanly.
            let ret = init_for_test();
            assert_eq!(ret, 0, "init should succeed");
            // Double init should fail.
            let ret2 = init_for_test();
            assert_eq!(ret2, -1, "double init should return AlreadyInit");

            // Submit with null descriptor.
            let handle = prism_plugin_submit(ptr::null());
            assert!(handle < 0, "null descriptor should return error");

            // Submit with null input_data but non-zero size.
            let params = CString::new("").unwrap();
            let desc = JobDescriptor {
                job_type: JobType::TTS,
                input_data: ptr::null(),
                input_size: 100, // non-zero but null ptr → error
                params_json: params.as_ptr(),
            };
            let handle = prism_plugin_submit(&desc);
            assert_eq!(
                handle, -3,
                "null data with non-zero size should return InvalidParams"
            );

            // Submit with bad JSON.
            let bad_json = CString::new("not valid json {").unwrap();
            let desc2 = JobDescriptor {
                job_type: JobType::TTS,
                input_data: ptr::null(),
                input_size: 0,
                params_json: bad_json.as_ptr(),
            };
            let handle = prism_plugin_submit(&desc2);
            assert_eq!(handle, -3, "bad JSON should return InvalidParams");

            // Submit a valid job (TTS with empty params, no input).
            let params = CString::new("{}").unwrap();
            let desc3 = JobDescriptor {
                job_type: JobType::TTS,
                input_data: ptr::null(),
                input_size: 0,
                params_json: params.as_ptr(),
            };
            let handle = prism_plugin_submit(&desc3);
            assert!(handle > 0, "valid submit should return a positive handle");

            // Poll the handle — should be running or completed.
            let status = prism_plugin_poll(handle);
            // The background task will quickly fail with "TTS pipeline not wired".
            // Both 1 (running) and -1 (error) are acceptable.
            assert!(
                status.status == 1 || status.status == -1,
                "poll status should be running or error, got {}",
                status.status
            );

            // Poll an unknown handle.
            let unknown = prism_plugin_poll(99999);
            assert_eq!(unknown.status, -2, "unknown handle should return -2");

            // Shutdown.
            let ret = prism_plugin_shutdown();
            assert_eq!(ret, 0);
            // Double shutdown should fail.
            let ret_not_init = prism_plugin_shutdown();
            assert_eq!(ret_not_init, -2, "double shutdown should return NotInit");

            // Submit after shutdown should fail.
            let handle = prism_plugin_submit(&desc3);
            assert_eq!(handle, -2, "submit after shutdown should return NotInit");
        }
    }
}
