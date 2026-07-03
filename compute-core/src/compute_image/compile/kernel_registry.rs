//! Pipeline state registry — caches `ComputePipelineState` objects keyed by
//! (kernel_name, function_constants_digest) to avoid recompilation per dispatch.
//!
//! The registry loads the compiled `.metallib` from the path baked by build.rs
//! into `TRIBUNUS_METALLIB` at compile time.  Callers in `kernel_dispatch.rs`
//! look up or create pipeline objects and then encode dispatches.

#[cfg(feature = "metal-dispatch")]
use metal::*;
#[cfg(feature = "metal-dispatch")]
use std::collections::HashMap;

/// Thread-safe pipeline state cache.
///
/// `kernel_dispatch.rs` wraps this in `Arc<Mutex<KernelRegistry>>` and passes
/// it to every dispatcher, so one cache serves the entire compilation.
#[cfg(feature = "metal-dispatch")]
pub struct KernelRegistry {
    device: Device,
    library: Library,
    /// Cache keyed by (kernel_entry_point_name, digest_of_function_constants).
    /// The digest captures page_width, sidecar_enabled, instrumented, etc.
    cache: HashMap<(String, u64), ComputePipelineState>,
}

#[cfg(feature = "metal-dispatch")]
impl KernelRegistry {
    /// Create a new registry by loading the `.metallib` from the path baked
    /// into `TRIBUNUS_METALLIB` at compile time (set by `build.rs`).
    ///
    /// # Panics
    /// Panics if the metallib cannot be loaded — this is a build-time
    /// integration failure, not a recoverable runtime condition.
    pub fn new(device: &Device) -> Self {
        let metallib_path = env!("TRIBUNUS_METALLIB");
        let library_data = std::fs::read(metallib_path)
            .unwrap_or_else(|e| panic!(
                "KernelRegistry: failed to read metallib at {}: {}",
                metallib_path, e,
            ));
        let library = device
            .new_library_with_data(&library_data)
            .unwrap_or_else(|e| panic!(
                "KernelRegistry: failed to load metallib from {}: {:?}",
                metallib_path, e,
            ));

        KernelRegistry {
            device: device.clone(),
            library,
            cache: HashMap::new(),
        }
    }

    /// Return a reference to the underlying `Device`.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Look up or create a compute pipeline state.
    ///
    /// `name` is the Metal kernel entry-point name (e.g. `"ternary_tile640_gemv"`).
    /// `constants` carries the function constant values.
    /// `digest` is caller-computed — it must be reproducible so the cache
    /// returns the same pipeline for equivalent constants.
    pub fn get_or_create(
        &mut self,
        name: &str,
        constants: &FunctionConstantValues,
        digest: u64,
    ) -> ComputePipelineState {
        let key = (name.to_string(), digest);
        if let Some(pso) = self.cache.get(&key) {
            return pso.clone();
        }

        let function = self
            .library
            .get_function(name, Some(constants.clone()))
            .unwrap_or_else(|e| panic!(
                "KernelRegistry: entry point '{}' not found: {:?}",
                name, e,
            ));

        let pso = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .unwrap_or_else(|e| panic!(
                "KernelRegistry: pipeline state for '{}' failed: {:?}",
                name, e,
            ));

        self.cache.insert(key, pso.clone());
        pso
    }
}

/// Build standard projection function constants and return them along with a
/// deterministic digest that can be used as a cache key.
///
/// Three constants are set:
///   - `page_width`        (MTLDataType::UInt,  index 0)
///   - `sidecar_enabled`   (MTLDataType::Bool,  index 1)
///   - `instrumented`      (MTLDataType::Bool,  index 2)
///
/// The digest is computed from the same three values so it matches the
/// pipeline state cache key.
#[cfg(feature = "metal-dispatch")]
pub fn projection_constants(
    page_width: u32,
    sidecar_enabled: bool,
    instrumented: bool,
) -> (FunctionConstantValues, u64) {
    use std::hash::{Hash, Hasher};

    let fcv = FunctionConstantValues::new();
    fcv.set_constant_value_at_index(
        &page_width as *const u32 as *const std::ffi::c_void,
        MTLDataType::UInt,
        0,
    );
    let sidecar_u8: u8 = sidecar_enabled as u8;
    fcv.set_constant_value_at_index(
        &sidecar_u8 as *const u8 as *const std::ffi::c_void,
        MTLDataType::Bool,
        1,
    );
    let instrumented_u8: u8 = instrumented as u8;
    fcv.set_constant_value_at_index(
        &instrumented_u8 as *const u8 as *const std::ffi::c_void,
        MTLDataType::Bool,
        2,
    );

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    page_width.hash(&mut hasher);
    sidecar_enabled.hash(&mut hasher);
    instrumented.hash(&mut hasher);

    (fcv, hasher.finish())
}
