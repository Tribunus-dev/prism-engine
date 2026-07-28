//! Shape variant definition — pure data types and pure algorithms for
//! shape-specialized variants.

use serde::{Deserialize, Serialize};

use crate::compute_image_runtime::ExecutionShapeClass;

/// Opaque identifier for a shape variant.
pub type ShapeVariantId = String;

/// A shape-specialized variant definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapeVariantDefinition {
    /// Variant identifier.
    pub variant_id: ShapeVariantId,
    /// Execution shape class this variant is specialized for.
    pub shape_class: ExecutionShapeClass,
    /// Maximum batch size.
    pub max_batch: u32,
    /// Maximum context tokens.
    pub max_context_tokens: u32,
    /// Whether this variant is the default for the shape class.
    pub is_default: bool,
}
