//! Unified PSO (pipeline state object) cache keyed by artifact identity.
//!
//! Replaces ad-hoc per-slot pipeline caches with a single cache that
//! compiles from compiled artifact bytes (pre-compiled .metallib data),
//! not from source text. Cache identity includes the artifact digest,
//! entry point, and device name so that same-provenance artifacts with
//! different targets produce different PSOs.

#![cfg(all(
    target_os = "macos",
    any(feature = "mlx-backend", feature = "prism-backend")
))]

use std::collections::HashMap;

use metal::{ComputePipelineState, Device, Library};

use crate::ecs::canonical::kernel_abi::{ArtifactProvenance, CompiledKernelArtifact};
use crate::execution_plan::pso_cache::{PsoCache as PsoCacheTrait, PsoError};
use crate::execution_plan::{FunctionConstantSet, KernelSpecializationKey};

/// A PSO (pipeline state object) keyed by full artifact identity.
///
/// The key combines the implementation digest, compilation parameters,
/// and device identity so that same-provenance artifacts with different
/// targets produce different PSOs.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct PsoKey {
    /// SHA-256 digest of the compiled artifact bytes.
    pub artifact_digest: String,
    /// Metal function entry point name.
    pub entry_point: String,
    /// GPU device name (e.g. "Apple M2 Max").
    pub device_name: String,
}

/// Cached compiled pipeline state and its provenance.
pub struct PsoEntry {
    /// The compiled Metal compute pipeline state.
    pub pipeline: ComputePipelineState,
    /// Provenance record for this pipeline's artifact.
    pub provenance: ArtifactProvenance,
    /// The Metal library that owns the function; must outlive the pipeline.
    pub library: Library,
}

/// Thread-safe PSO cache keyed by artifact identity.
pub struct PsoCache {
    pub device: Device,
    entries: HashMap<PsoKey, PsoEntry>,
}

impl PsoCache {
    /// Create a new empty PSO cache.
    pub fn new(device: Device) -> Self {
        Self {
            device,
            entries: HashMap::new(),
        }
    }

    /// Get or compile a pipeline from a compiled kernel artifact.
    ///
    /// Returns a reference to the cached `PsoEntry`. If no entry exists for
    /// the artifact's digest + entry point + device combination, the library
    /// is loaded from the compiled bytes and the pipeline is compiled.
    pub fn get_or_compile(
        &mut self,
        artifact: &CompiledKernelArtifact,
        provenance: &ArtifactProvenance,
        entry_point: &str,
    ) -> Result<&PsoEntry, String> {
        let key = PsoKey {
            artifact_digest: artifact.sha256.clone(),
            entry_point: entry_point.to_string(),
            device_name: self.device.name().to_string(),
        };
        if !self.entries.contains_key(&key) {
            // Compile from pre-compiled bytes (not source text).
            let lib = self
                .device
                .new_library_with_data(&artifact.compiled_bytes)
                .map_err(|e| format!("PSO cache: failed to load library: {:?}", e))?;
            let func = lib.get_function(entry_point, None).map_err(|e| {
                format!(
                    "PSO cache: entry point '{}' not found: {:?}",
                    entry_point, e
                )
            })?;
            let pipeline = self
                .device
                .new_compute_pipeline_state_with_function(&func)
                .map_err(|e| format!("PSO cache: failed to create PSO: {:?}", e))?;
            self.entries.insert(
                key.clone(),
                PsoEntry {
                    pipeline,
                    provenance: provenance.clone(),
                    library: lib,
                },
            );
        }
        Ok(self.entries.get(&key).unwrap())
    }

    /// Invalidate all PSOs that match a specific implementation provenance.
    pub fn invalidate_provenance(&mut self, implementation_id: &str) {
        self.entries
            .retain(|_, entry| entry.provenance.implementation_id.0 != implementation_id);
    }

    /// Number of cached PSOs.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Clear all cached PSOs.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal::Device;

    #[test]
    #[cfg_attr(not(target_os = "macos"), ignore = "Metal is macOS-only")]
    fn create_empty_cache() {
        let device = Device::system_default().expect("Metal device required for test");
        let cache = PsoCache::new(device);
        assert_eq!(cache.len(), 0);
    }

    #[test]
    #[cfg_attr(not(target_os = "macos"), ignore = "Metal is macOS-only")]
    fn clear_removes_entries() {
        let device = Device::system_default().expect("Metal device required for test");
        let mut cache = PsoCache::new(device);
        // empty-to-empty is fine
        cache.clear();
        assert_eq!(cache.len(), 0);
    }
}

impl PsoCacheTrait for PsoCache {
    type PipelineState = ComputePipelineState;

    fn get_or_create(
        &mut self,
        key: &KernelSpecializationKey,
        _constants: &FunctionConstantSet,
    ) -> Result<Self::PipelineState, PsoError> {
        let _cache_key = crate::execution_plan::pso_cache::PsoCacheKey::from(key);
        // Phase 1: pre-compiled artifact pipeline not yet wired through this
        // interface.  The old source-text compilation path is replaced by the
        // BackendCompiler pipeline; get_or_compile() is the contemporary API.
        Err(PsoError::UnsupportedConfiguration(format!(
            "PSO from KernelSpecializationKey not yet wired: template={:?}, codec={:?}",
            key.template_id, key.codec
        )))
    }
}
