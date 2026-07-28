//! Shape-specialized program variant.

use serde::{Deserialize, Serialize};

use crate::compute_image_runtime::ContentHash;
use crate::compute_image_runtime::program::phase_program::SerializedPhaseProgram;

/// Opaque identifier for a shape-specialized variant.
pub type ShapeSpecializedVariantId = String;

/// Shape-specialized program variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeSpecializedProgram {
    /// Variant identifier.
    pub variant_id: ShapeSpecializedVariantId,
    /// Shape profile this variant is specialized for.
    pub shape_profile: ShapeProfile,
    /// Serialized phase program.
    pub phase_program: SerializedPhaseProgram,
    /// Content hash of the program.
    pub program_hash: ContentHash,
}

/// Shape profile a variant is specialized for.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeProfile {
    /// Maximum batch size.
    pub max_batch: u32,
    /// Maximum tokens.
    pub max_tokens: u32,
    /// Human-readable label.
    pub label: String,
}
