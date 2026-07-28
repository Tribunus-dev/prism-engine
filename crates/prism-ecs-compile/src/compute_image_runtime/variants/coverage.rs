//! Variant coverage report — pure data types and pure algorithms.

use serde::{Deserialize, Serialize};

use super::shape_class::ShapeVariantId;

/// A variant coverage report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariantCoverageReport {
    /// Variants included in the report.
    pub variant_ids: Vec<ShapeVariantId>,
    /// Total shapes covered.
    pub shapes_covered: u32,
    /// Total shapes requested.
    pub shapes_requested: u32,
    /// Overlap descriptors for redundant variants.
    pub overlaps: Vec<OverlapDescriptor>,
}

/// A descriptor for an overlap between two variants.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapDescriptor {
    /// First variant id.
    pub variant_a: ShapeVariantId,
    /// Second variant id.
    pub variant_b: ShapeVariantId,
    /// Estimated overlap fraction (0.0–1.0).
    pub overlap_fraction: f64,
}
