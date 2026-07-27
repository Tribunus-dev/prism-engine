//! Canonical profile metadata and pure measurement types for compiled
//! profiled models.
//!
//! Single authority: the constitutional surface that exposes a profiled
//! model's *facts* to the rest of the compile pipeline and downstream
//! callers, plus the typed port the engine implements to map its
//! `LoadedProfiledModel` runtime onto that surface.
//!
//! The engine's `compute-core/src/ecs/core/profiled_model.rs` is
//! execution-boundary: it owns MLX arrays, IOSurface arenas, mmap regions,
//! ANE CoreML handles, Metal kernel artifacts, thread-local state, and
//! raw FFI to the kernel (mmap, sysctl, IOSurface allocation, MLX
//! external-array construction, ANE DMA). The constitutional layer
//! captures the *same facts* (path, layer count, byte breakdown, handle
//! baseline, per-layer routing, peak estimate, RoPE budget, IOSurface
//! pool sizing) as pure data and pure functions. The boundary is:
//!
//! - engine produces a `LoadedProfiledModel` runtime, exposes canonical
//!   fields through the [`ProfiledModelPort`] trait.
//! - constitutional code calls `port.profile_metadata()` (or
//!   `port.layer_weight_descriptor(i)`) to reason about the model
//!   without ever touching the engine's hardware state.
//! - constitutional peak-bytes, RoPE-budget, and IOSurface-pool
//!   calculations operate on canonical [`data::TensorEntryMeta`] and
//!   [`measurement::ArchitectureMeta`] only.
//!
//! # Module layout
//!
//! - [`data`] — canonical data types: `ProfileMetadata`,
//!   `LayerWeightDescriptor`, `ProjectionDescriptor`,
//!   `TensorEntryMeta`, `LayerRouting`, `ByteCount`, and the small
//!   pure helpers (`format_bytes`, `sibling_cimage_path`,
//!   `tensor_table_has_prefix_meta`).
//! - [`measurement`] — pure peak-bytes and RoPE-table budget
//!   calculations: `peak_bytes_for_manifest`, `rope_table_bytes`,
//!   `embedding_dequant_bytes`, `iosurface_pool_bytes`,
//!   `attention_scores_bytes`, `kv_per_token_bytes`,
//!   `peak_within_budget`, plus the `PeakBytesEstimate` newtype.
//! - the typed port trait [`ProfiledModelPort`] lives in this file.

pub mod data;
pub mod measurement;

// Re-exports so callers can `use crate::ecs::profiled_model::*` without
// reaching into the sub-modules.
pub use data::{
    format_bytes, sibling_cimage_path, tensor_table_has_prefix_meta, tensor_table_has_suffix_meta,
    ByteCount, LayerRouting, LayerWeightDescriptor, ProfileMetadata, ProjectionDescriptor,
    TensorEntryMeta,
};
pub use measurement::{
    admission_safe_budget, attention_scores_bytes, computed_iosurface_pool,
    embedding_dequant_bytes, iosurface_pool_bytes, kv_headroom_bytes, kv_per_token_bytes,
    peak_bytes_for_manifest, peak_within_budget, rope_table_bytes, scratch_bytes,
    ArchitectureMeta, PeakBytesEstimate,
};

// ---------------------------------------------------------------------------
// Typed port: engine exposes canonical profile fields through this trait
// ---------------------------------------------------------------------------

/// Typed port that the engine implements to expose its
/// `LoadedProfiledModel` runtime as canonical [`ProfileMetadata`].
///
/// The engine is the only authority for the runtime state (MLX arrays,
/// IOSurface arenas, mmap regions, ANE CoreML handles, Metal kernel
/// artifacts, thread-local state, raw FFI). Constitutional code never
/// imports the engine; instead, the engine adapts its runtime into the
/// pure-data shape [`ProfileMetadata`] and implements this trait.
///
/// One trait method per fact the constitutional layer needs. Each method
/// is `&self` so the engine can hand out a `&LoadedProfiledModel` adapter
/// without cloning the runtime.
pub trait ProfiledModelPort {
    /// The canonical [`ProfileMetadata`] for this loaded model.
    fn profile_metadata(&self) -> ProfileMetadata;

    /// Number of layers in the loaded model.
    fn layer_count(&self) -> u32;

    /// Canonical descriptor for the `layer_index`-th layer's weight set.
    /// Returns `None` if the index is out of range.
    fn layer_weight_descriptor(&self, layer_index: u32) -> Option<LayerWeightDescriptor>;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// A test double that the tests in this module use to verify the typed
    /// port contract. It holds only canonical fields, so the test
    /// exercises the contract without touching the engine.
    struct TestProfiledModel {
        metadata: ProfileMetadata,
        descriptors: Vec<LayerWeightDescriptor>,
    }

    impl ProfiledModelPort for TestProfiledModel {
        fn profile_metadata(&self) -> ProfileMetadata {
            self.metadata.clone()
        }
        fn layer_count(&self) -> u32 {
            self.metadata.layer_count
        }
        fn layer_weight_descriptor(&self, layer_index: u32) -> Option<LayerWeightDescriptor> {
            self.descriptors
                .iter()
                .find(|d| d.layer_index == layer_index)
                .cloned()
        }
    }

    fn fixture() -> TestProfiledModel {
        TestProfiledModel {
            metadata: ProfileMetadata {
                image_path: PathBuf::from("/cache/models/qwen3"),
                layer_count: 2,
                mapped_weight_bytes: ByteCount(1000),
                copied_weight_bytes: ByteCount(500),
                materialized_bytes: ByteCount(100),
                handle_baseline: 42,
                namespace_root: "model".into(),
            },
            descriptors: (0..2)
                .map(|i| LayerWeightDescriptor {
                    layer_index: i,
                    projections: vec![ProjectionDescriptor {
                        name: format!("model.layers.{i}.self_attn.q_proj.weight"),
                        storage_dtype: "BF16".into(),
                        logical_shape: vec![4096, 4096],
                        byte_length: 33_554_432,
                        has_scales: true,
                        has_biases: true,
                    }],
                    total_bytes: ByteCount(33_554_432),
                    attention_k_eq_v: i % 2 == 0,
                })
                .collect(),
        }
    }

    #[test]
    fn port_exposes_canonical_metadata() {
        let model = fixture();
        let pm = model.profile_metadata();
        assert_eq!(pm.layer_count, 2);
        assert_eq!(pm.namespace_root, "model");
        assert_eq!(
            pm.total_bytes(),
            ByteCount(1000)
                .saturating_add(ByteCount(500))
                .saturating_add(ByteCount(100))
        );
    }

    #[test]
    fn port_returns_per_layer_descriptor_by_index() {
        let model = fixture();
        assert!(model.layer_weight_descriptor(0).is_some());
        assert!(model.layer_weight_descriptor(1).is_some());
        assert!(model.layer_weight_descriptor(99).is_none());
        let d0 = model.layer_weight_descriptor(0).unwrap();
        assert!(d0.attention_k_eq_v);
        assert_eq!(d0.projection_count(), 1);
    }

    #[test]
    fn port_layer_count_matches_metadata() {
        let model = fixture();
        assert_eq!(model.layer_count(), model.profile_metadata().layer_count);
    }
}
