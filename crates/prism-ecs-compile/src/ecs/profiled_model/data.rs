//! Canonical profile metadata types for compiled-model introspection.
//!
//! Single authority: the pure-data shape of profile metadata for a compiled
//! model — what a [`ProfileMetadata`] exposes, what a [`LayerWeightDescriptor`]
//! describes, and the small pure helpers that operate on those shapes. This
//! module is MLX-free, ANE-free, IOSurface-free, FFI-free, and free of any
//! hardware handle, file descriptor, or process-local state. It is the
//! canonical home for the *facts* the engine's `LoadedProfiledModel`
//! represents; the engine's runtime state stays in
//! `compute-core/src/ecs/core/profiled_model.rs`.

use std::fmt;
use std::path::{Path, PathBuf};

use prism_ecs_kernel::BackendKind;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Byte count newtype
// ---------------------------------------------------------------------------

/// A byte count measured during model profiling or weight materialization.
///
/// Newtype wrapper so that `u64` is never silently used as a weight or
/// memory size. Arithmetic uses `saturating_*` so caller code cannot panic
/// from an overflow at the constitutional layer.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct ByteCount(pub u64);

impl ByteCount {
    /// Zero bytes.
    pub const ZERO: Self = Self(0);

    /// Saturating add.
    #[must_use]
    pub const fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    /// Saturating subtraction, clamped at zero.
    #[must_use]
    pub const fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }

    /// Inner value.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

impl Default for ByteCount {
    fn default() -> Self {
        Self::ZERO
    }
}

impl fmt::Display for ByteCount {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&format_bytes(self.0))
    }
}

impl From<u64> for ByteCount {
    fn from(value: u64) -> Self {
        Self(value)
    }
}

// ---------------------------------------------------------------------------
// Tensor entry metadata (canonical, engine-free)
// ---------------------------------------------------------------------------

/// Canonical metadata for one tensor entry in a compiled-image tensor table.
///
/// Mirrors the engine's `TensorEntry` for the fields that the constitutional
/// layer needs to make routing and measurement decisions, without any
/// dependency on the engine's `CompiledImageReader` or `MappedImage`. The
/// engine adapts its runtime tensor table into `Vec<TensorEntryMeta>` when it
/// calls into the constitutional crate.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorEntryMeta {
    /// Tensor name (e.g. `model.layers.0.self_attn.q_proj.weight`).
    pub name: String,
    /// Segment id this tensor lives in.
    pub segment: String,
    /// Byte offset within the segment.
    pub offset: u64,
    /// Byte length of the tensor.
    pub byte_length: u64,
    /// Storage dtype tag (e.g. `F32`, `BF16`, `I8`, `U8`, `U32`).
    pub storage_dtype: String,
    /// Physical shape as emitted into the manifest.
    pub physical_shape: Vec<u64>,
    /// Logical shape exposed to the runtime.
    pub logical_shape: Vec<u64>,
}

impl TensorEntryMeta {
    /// Byte size of this tensor entry.
    #[must_use]
    pub fn byte_size(&self) -> ByteCount {
        ByteCount(self.byte_length)
    }
}

/// Pure query: does any tensor in `table` have a name starting with one of
/// the given prefixes?
///
/// This is the constitutional counterpart of the engine's private
/// `tensor_table_has_prefix`. It is generic over the canonical
/// [`TensorEntryMeta`] so it has no dependency on the engine.
#[must_use]
pub fn tensor_table_has_prefix_meta(table: &[TensorEntryMeta], prefixes: &[&str]) -> bool {
    table
        .iter()
        .any(|entry| prefixes.iter().any(|prefix| entry.name.starts_with(prefix)))
}

/// Pure query: does any tensor in `table` have a name ending with one of
/// the given suffixes?
#[must_use]
pub fn tensor_table_has_suffix_meta(table: &[TensorEntryMeta], suffixes: &[&str]) -> bool {
    table
        .iter()
        .any(|entry| suffixes.iter().any(|suffix| entry.name.ends_with(suffix)))
}

// ---------------------------------------------------------------------------
// Per-layer weight descriptor (canonical, hardware-free)
// ---------------------------------------------------------------------------

/// Canonical descriptor for a single projection within a layer's weight set.
///
/// Holds name + dtype + shape + size — not the actual weight array. The
/// engine's `LayerWeights` carries `Arc<Array>` (MLX) tensors, which is
/// execution-boundary; the constitutional layer only needs the metadata
/// shape to make routing, measurement, and admission decisions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ProjectionDescriptor {
    /// Tensor name (e.g. `model.layers.0.self_attn.q_proj.weight`).
    pub name: String,
    /// Storage dtype tag.
    pub storage_dtype: String,
    /// Logical shape exposed to the runtime.
    pub logical_shape: Vec<u64>,
    /// Byte length of the projection.
    pub byte_length: u64,
    /// True if the projection also carries explicit scales / biases (quant).
    pub has_scales: bool,
    /// True if the projection also carries explicit biases.
    pub has_biases: bool,
}

/// Canonical descriptor for one transformer layer's weight set.
///
/// Lists the projections that the engine's `LayerWeights` carries, plus
/// derived metadata that the constitutional layer can compute without
/// touching MLX. A layer with no q_norm / k_norm (e.g. LLaMA-2) leaves
/// those fields `None`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerWeightDescriptor {
    /// Layer index (zero-based).
    pub layer_index: u32,
    /// Projections (q/k/v/o, gate/up/down, norms) keyed by canonical role.
    pub projections: Vec<ProjectionDescriptor>,
    /// Total bytes across all projections in this layer.
    pub total_bytes: ByteCount,
    /// Whether the manifest marks this layer as `attention_k_eq_v` (GQA
    /// models share the K projection with V).
    pub attention_k_eq_v: bool,
}

impl LayerWeightDescriptor {
    /// Count of projections in this layer.
    #[must_use]
    pub fn projection_count(&self) -> usize {
        self.projections.len()
    }
}

// ---------------------------------------------------------------------------
// Profile metadata
// ---------------------------------------------------------------------------

/// Canonical profile metadata for a compiled model.
///
/// This is the *shape* of the facts that the engine's `LoadedProfiledModel`
/// reports. It is intentionally free of any hardware handle, MLX array,
/// IOSurface arena, mmap region, ANE model, or Metal kernel — those live
/// in the engine and are exposed through the [`ProfiledModelPort`] trait
/// (see `mod.rs`), not through this struct.
///
/// [`ProfiledModelPort`]: super::ProfiledModelPort
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileMetadata {
    /// Path to the compiled-image directory.
    pub image_path: PathBuf,
    /// Number of transformer layers materialized.
    pub layer_count: u32,
    /// Bytes that were accessed as mapped (no-copy) tensors.
    pub mapped_weight_bytes: ByteCount,
    /// Bytes that required a copy into an IOSurface arena.
    pub copied_weight_bytes: ByteCount,
    /// Bytes materialized as standalone buffers (e.g. RoPE tables).
    pub materialized_bytes: ByteCount,
    /// Engine handle count observed before model load.
    pub handle_baseline: u64,
    /// Detected tensor namespace root (e.g. `model`).
    pub namespace_root: String,
}

impl ProfileMetadata {
    /// Total bytes accounted for across mapped, copied, and materialized
    /// tensors.
    #[must_use]
    pub fn total_bytes(&self) -> ByteCount {
        self.mapped_weight_bytes
            .saturating_add(self.copied_weight_bytes)
            .saturating_add(self.materialized_bytes)
    }
}

// ---------------------------------------------------------------------------
// Per-layer routing
// ---------------------------------------------------------------------------

/// Canonical routing for one transformer layer: which backends the engine
/// may dispatch the layer onto.
///
/// The engine's `LayerPlan::route.set_dominant_backend` writes a `u8`; this
/// type captures the same fact as a typed enum so callers do not have to
/// parse raw integers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct LayerRouting {
    /// The dominant (primary) backend the engine will dispatch onto.
    pub dominant: BackendKind,
    /// Whether the engine's ANE island replication applies to this layer.
    pub ane_island_replicated: bool,
}

impl LayerRouting {
    /// Default routing (MLX/Metal dominant, no ANE replication).
    #[must_use]
    pub const fn metal_default() -> Self {
        Self {
            dominant: BackendKind::Metal,
            ane_island_replicated: false,
        }
    }

    /// ANE dominant with island replication enabled.
    #[must_use]
    pub const fn ane_island() -> Self {
        Self {
            dominant: BackendKind::ANE,
            ane_island_replicated: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Pure helpers (moved from engine)
// ---------------------------------------------------------------------------

/// Format a byte count as a human-readable string with binary-prefix units.
///
/// Canonical replacement for the engine's `pub(crate) fn format_bytes`.
/// Returns `"{:.1}GB"` for ≥ 1 GiB, `"{:.1}MB"` for ≥ 1 MiB, else `"{N}B"`.
#[must_use]
pub fn format_bytes(b: u64) -> String {
    if b >= 1_073_741_824 {
        format!("{:.1}GB", b as f64 / 1_073_741_824.0)
    } else if b >= 1_048_576 {
        format!("{:.1}MB", b as f64 / 1_048_576.0)
    } else {
        format!("{b}B")
    }
}

/// Derive the sibling `<stem>.cimage` path for a given `<stem>` image
/// directory. Canonical replacement for the engine's
/// `pub(crate) fn sibling_cimage_path`.
#[must_use]
pub fn sibling_cimage_path(image_dir: &Path) -> Option<PathBuf> {
    let stem = image_dir.file_name()?.to_str()?;
    Some(image_dir.with_file_name(format!("{stem}.cimage")))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_count_arithmetic_is_saturating() {
        let a = ByteCount(u64::MAX);
        let b = ByteCount(1);
        assert_eq!(a.saturating_add(b), ByteCount(u64::MAX));
        assert_eq!(ByteCount(3).saturating_sub(ByteCount(5)), ByteCount::ZERO);
    }

    #[test]
    fn byte_count_display_uses_format_bytes() {
        assert_eq!(ByteCount(0).to_string(), "0B");
        assert_eq!(ByteCount(1024).to_string(), "1.0KB");
        // (Note: format_bytes below 1 MiB returns "<N>B".)
        assert_eq!(ByteCount(1_048_576).to_string(), "1.0MB");
        assert_eq!(ByteCount(1_073_741_824).to_string(), "1.0GB");
    }

    #[test]
    fn format_bytes_thresholds_match_engine() {
        assert_eq!(format_bytes(0), "0B");
        assert_eq!(format_bytes(1_048_575), "1048575B");
        assert_eq!(format_bytes(1_048_576), "1.0MB");
        assert_eq!(format_bytes(1_073_741_823), "1024.0MB");
        assert_eq!(format_bytes(1_073_741_824), "1.0GB");
    }

    #[test]
    fn sibling_cimage_path_replaces_stem() {
        let dir = PathBuf::from("/cache/models/qwen3-8b");
        assert_eq!(
            sibling_cimage_path(&dir),
            Some(PathBuf::from("/cache/models/qwen3-8b.cimage"))
        );
        let root = PathBuf::from("/");
        assert_eq!(sibling_cimage_path(&root), None);
    }

    #[test]
    fn tensor_table_prefix_query_matches_engine_semantics() {
        let table = vec![
            TensorEntryMeta {
                name: "model.layers.0.input_layernorm.weight".into(),
                segment: "seg-0".into(),
                offset: 0,
                byte_length: 4096,
                storage_dtype: "F32".into(),
                physical_shape: vec![4096],
                logical_shape: vec![4096],
            },
            TensorEntryMeta {
                name: "vision_encoder.patch_embedding.weight".into(),
                segment: "seg-1".into(),
                offset: 0,
                byte_length: 16_777_216,
                storage_dtype: "BF16".into(),
                physical_shape: vec![3, 14, 14, 2048],
                logical_shape: vec![3, 14, 14, 2048],
            },
        ];
        assert!(tensor_table_has_prefix_meta(
            &table,
            &["model.layers."]
        ));
        assert!(tensor_table_has_prefix_meta(&table, &["vision_encoder."]));
        assert!(!tensor_table_has_prefix_meta(&table, &["audio_encoder."]));
        assert!(tensor_table_has_suffix_meta(&table, &[".weight"]));
        assert!(!tensor_table_has_suffix_meta(&table, &[".scales"]));
    }

    #[test]
    fn profile_metadata_total_bytes_sums_three_components() {
        let pm = ProfileMetadata {
            image_path: PathBuf::from("/cache/models/qwen"),
            layer_count: 32,
            mapped_weight_bytes: ByteCount(100),
            copied_weight_bytes: ByteCount(50),
            materialized_bytes: ByteCount(25),
            handle_baseline: 7,
            namespace_root: "model".into(),
        };
        assert_eq!(pm.total_bytes(), ByteCount(175));
    }

    #[test]
    fn layer_weight_descriptor_counts_projections() {
        let desc = LayerWeightDescriptor {
            layer_index: 0,
            projections: (0..5)
                .map(|i| ProjectionDescriptor {
                    name: format!("p{i}"),
                    storage_dtype: "F32".into(),
                    logical_shape: vec![1],
                    byte_length: 4,
                    has_scales: false,
                    has_biases: false,
                })
                .collect(),
            total_bytes: ByteCount(20),
            attention_k_eq_v: false,
        };
        assert_eq!(desc.projection_count(), 5);
    }

    #[test]
    fn layer_routing_default_is_metal() {
        assert_eq!(LayerRouting::metal_default().dominant, BackendKind::Metal);
        assert!(!LayerRouting::metal_default().ane_island_replicated);
        assert_eq!(LayerRouting::ane_island().dominant, BackendKind::ANE);
        assert!(LayerRouting::ane_island().ane_island_replicated);
    }
}
