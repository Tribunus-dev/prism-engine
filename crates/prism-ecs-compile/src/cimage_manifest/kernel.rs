//! Metal kernel dispatch and artifact contracts — the per-kernel evidence
//! bound to a CImage manifest.
//!
//! This module owns the constitutional authority for the
//! [`MetalDispatchRecipe`] and [`MetalKernelArtifact`] types — the
//! per-kernel contract the manifest records for every pre-compiled Metal
//! library embedded in the image. The dispatch recipe pins the binding
//! layout (buffer slots, scalar slots, threadgroup size) for a kernel;
//! the artifact record points to the `.metallib` blob on disk and the
//! dispatch recipe for it.
//!
//! The module does **not** own the per-tensor table (see
//! [`super::types`]) or the manifest itself (see [`super::header`]).
//! The dispatch recipe is the bridge from the manifest to the
//! per-kernel runtime evidence.
//!
//! The `BTreeMap` discipline is honored: the buffer/scalar slot maps
//! inside [`MetalDispatchRecipe`] are `BTreeMap<String, _>` so the
//! binding order is deterministic for replay and rebuild.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use super::types::ArtifactKind;

// ── Dispatch recipe ───────────────────────────────────────────────────────

/// Dispatch configuration for a compiled Metal kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalDispatchRecipe {
    /// Entry point function name within the compiled Metal library.
    pub entry_point: String,
    /// Human-readable kernel name for identification.
    pub kernel_name: String,
    /// Threadgroup size (threads per threadgroup).
    pub threads_per_threadgroup: [u32; 3],
    /// Grid size (number of threadgroups).
    pub threadgroups_per_grid: [u32; 3],
    /// Metal buffer indices for each logical binding, keyed by binding
    /// name. Stored as `BTreeMap` so iteration order is stable for
    /// the manifest hash and the replay path.
    pub buffer_slot_map: BTreeMap<String, u32>,
    /// Scalar binding indices with their Metal type string, keyed by
    /// binding name. Stored as `BTreeMap` for the same reason.
    pub scalar_index_map: BTreeMap<String, (u32, String)>,
    /// K (input channel) dimension from the export.
    pub k: u64,
    /// N (output channel) dimension from the export.
    pub n: u64,
    /// Block quantization group size.
    pub group_size: u32,
    /// Quantization bits.
    pub bits: u8,
    /// Kernel ABI version — must match between compiler and runtime.
    pub kernel_abi_version: u32,
}

// ── Kernel artifact ───────────────────────────────────────────────────────

/// A pre-compiled Metal kernel embedded in the ComputeImage.
/// The .metallib is stored under `metal/kernels/<artifact_id>.metallib`
/// in the image directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetalKernelArtifact {
    /// Unique identifier for this artifact (e.g., "q_proj_nf4_layer0").
    pub artifact_id: String,
    /// Which logical operation this artifact implements.
    pub logical_operation: String,
    /// Target artifact kind (MlxNf4U32, MlxAf8U32, etc.).
    pub kind: ArtifactKind,
    /// Path to the .metallib relative to the image root.
    pub metallib_relpath: String,
    /// BLAKE3 hash of the .metallib for integrity.
    pub metallib_blake3: String,
    /// Byte size of the .metallib.
    pub metallib_byte_length: u64,
    /// Dispatch recipe.
    pub dispatch: MetalDispatchRecipe,
    /// Logical shape of the weight tensor (e.g., [896, 896]).
    pub logical_shape: Vec<u32>,
    /// Storage shape of the packed weight (e.g., [896, 112]).
    pub storage_shape: Vec<u32>,
    /// Quantization bits (4 for NF4, 8 for AF8).
    pub bits: u8,
    /// Block quantization group size.
    pub group_size: u32,
    /// Name of the companion scale tensor.
    pub scale_tensor: String,
    /// Name of the companion bias tensor.
    pub bias_tensor: String,
    /// GPU family this artifact was compiled for.
    pub gpu_family: String,
    /// SHA-256 checksum of the entire artifact descriptor.
    pub checksum: String,
}

impl MetalDispatchRecipe {
    /// Construct an empty dispatch recipe for the given entry point.
    /// All other fields default to neutral values; callers populate
    /// them in the order they are computed.
    pub fn new(entry_point: impl Into<String>, kernel_name: impl Into<String>) -> Self {
        Self {
            entry_point: entry_point.into(),
            kernel_name: kernel_name.into(),
            threads_per_threadgroup: [1, 1, 1],
            threadgroups_per_grid: [1, 1, 1],
            buffer_slot_map: BTreeMap::new(),
            scalar_index_map: BTreeMap::new(),
            k: 0,
            n: 0,
            group_size: 0,
            bits: 0,
            kernel_abi_version: 1,
        }
    }

    /// Bind a Metal buffer slot to a logical binding name.
    pub fn bind_buffer(&mut self, name: impl Into<String>, slot: u32) {
        self.buffer_slot_map.insert(name.into(), slot);
    }

    /// Bind a scalar slot (with Metal type string) to a logical binding name.
    pub fn bind_scalar(&mut self, name: impl Into<String>, slot: u32, ty: impl Into<String>) {
        self.scalar_index_map
            .insert(name.into(), (slot, ty.into()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dispatch_recipe_buffer_slot_map_is_btreemap() {
        let mut recipe = MetalDispatchRecipe::new("gemv_main", "ternary_tile640_gemv");
        recipe.bind_buffer("weights", 0);
        recipe.bind_buffer("scales", 1);
        recipe.bind_buffer("out", 2);
        assert_eq!(recipe.buffer_slot_map.len(), 3);
        // Iteration order is sorted by key — the BTreeMap invariant.
        let names: Vec<&String> = recipe.buffer_slot_map.keys().collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }

    #[test]
    fn dispatch_recipe_scalar_index_map_is_btreemap() {
        let mut recipe = MetalDispatchRecipe::new("kernel", "name");
        recipe.bind_scalar("group_size", 0, "uint");
        recipe.bind_scalar("bits", 1, "uint");
        let names: Vec<&String> = recipe.scalar_index_map.keys().collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }

    #[test]
    fn kernel_artifact_carries_dispatch_and_digest() {
        let mut recipe = MetalDispatchRecipe::new("gemv_main", "ternary_tile640_gemv");
        recipe.bind_buffer("weights", 0);
        recipe.k = 4096;
        recipe.n = 4096;
        recipe.kernel_abi_version = 2;
        let artifact = MetalKernelArtifact {
            artifact_id: "q_proj_nf4_layer0".into(),
            logical_operation: "q_proj".into(),
            kind: ArtifactKind::Nf4Tile640Shared,
            metallib_relpath: "metal/kernels/q_proj_nf4_layer0.metallib".into(),
            metallib_blake3: "deadbeef".into(),
            metallib_byte_length: 16384,
            dispatch: recipe,
            logical_shape: vec![4096, 4096],
            storage_shape: vec![4096, 1280],
            bits: 4,
            group_size: 128,
            scale_tensor: "q_proj.scales".into(),
            bias_tensor: "q_proj.biases".into(),
            gpu_family: "apple9".into(),
            checksum: "00".into(),
        };
        assert_eq!(artifact.logical_shape, vec![4096, 4096]);
        assert_eq!(artifact.bits, 4);
        assert_eq!(artifact.dispatch.kernel_abi_version, 2);
    }
}
