//! Metal PSO cache — real-Metal [`PsoCache`] implementation.
//!
//! Compiles and caches Metal pipeline states keyed by [`KernelSpecializationKey`].
//! Uses the `metal` crate for PSO compilation.

#![cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]

use std::collections::HashMap;

use crate::execution_plan::pso_cache::{PsoCache, PsoCacheKey, PsoError};
use crate::execution_plan::{FunctionConstantSet, KernelSpecializationKey};

/// A real-Metal [`PsoCache`] implementation that delegates to the
/// `metal` crate for PSO compilation and caching.
#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
#[derive(Debug)]
pub struct MetalPsoCache {
    device: metal::Device,
    library: metal::Library,
    cache: HashMap<PsoCacheKey, metal::ComputePipelineState>,
}

#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
impl MetalPsoCache {
    /// Create a new PSO cache for the given device and shader source.
    pub fn new(device: &metal::Device, shader_source: &str) -> Result<Self, String> {
        let compile_opts = metal::CompileOptions::new();
        let lib = device
            .new_library_with_source(shader_source, &compile_opts)
            .map_err(|e| format!("failed to compile Metal library: {e}"))?;
        Ok(Self {
            device: device.clone(),
            library: lib,
            cache: HashMap::new(),
        })
    }

    /// Compile a new pipeline state for the given key, using the stored library.
    fn compile_pso(
        &self,
        key: &KernelSpecializationKey,
        constants: &FunctionConstantSet,
    ) -> Result<metal::ComputePipelineState, PsoError> {
        let pso_key = PsoCacheKey::from(key);
        let func_name = format!("kernel_{:?}", key.template_id);

        // Build function constant values to pass to get_function.
        let fcs = metal::FunctionConstantValues::new();
        fcs.set_constant_value_at_index(
            &constants.page_width as *const u32 as *const std::ffi::c_void,
            metal::MTLDataType::UInt,
            0,
        );
        fcs.set_constant_value_at_index(
            &constants.tile_m as *const u32 as *const std::ffi::c_void,
            metal::MTLDataType::UInt,
            1,
        );
        fcs.set_constant_value_at_index(
            &constants.tile_n as *const u32 as *const std::ffi::c_void,
            metal::MTLDataType::UInt,
            2,
        );
        fcs.set_constant_value_at_index(
            &constants.tile_k as *const u32 as *const std::ffi::c_void,
            metal::MTLDataType::UInt,
            3,
        );
        fcs.set_constant_value_at_index(
            &constants.group_size as *const u32 as *const std::ffi::c_void,
            metal::MTLDataType::UInt,
            4,
        );

        let func = self
            .library
            .get_function(&func_name, Some(fcs))
            .map_err(|e| PsoError::CompilationFailed(format!("function {func_name}: {e}")))?;

        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| PsoError::CompilationFailed(format!("PSO {:?}: {e}", pso_key)))?;

        Ok(pipeline)
    }
}

#[cfg(all(target_os = "macos", feature = "metal-dispatch"))]
impl PsoCache for MetalPsoCache {
    type PipelineState = metal::ComputePipelineState;

    fn get_or_create(
        &mut self,
        key: &KernelSpecializationKey,
        constants: &FunctionConstantSet,
    ) -> Result<Self::PipelineState, PsoError> {
        let pso_key = PsoCacheKey::from(key);
        if let Some(pso) = self.cache.get(&pso_key) {
            return Ok(pso.clone());
        }
        let pso = self.compile_pso(key, constants)?;
        let pso_clone = pso.clone();
        self.cache.insert(pso_key, pso_clone);
        Ok(pso)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg_attr(not(target_os = "macos"), ignore = "Metal is macOS-only")]
    fn create_empty_cache() {
        let device = metal::Device::system_default().expect("Metal device should be available");
        let source = "// empty library — not used\n";
        let cache = MetalPsoCache::new(&device, source).expect("empty shader compiles");
        assert!(cache.cache.is_empty());
    }
}
