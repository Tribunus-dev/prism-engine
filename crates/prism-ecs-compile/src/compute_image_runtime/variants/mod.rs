//! Variant configurations — pure data types and pure algorithms for
//! variant selection, coverage, and compatibility.

pub mod compatibility;
pub mod coverage;
pub mod selection;
pub mod shape_class;

pub use compatibility::{
    CompatibilityViolation, RuntimeCapabilitySnapshot, VariantCompatibilityReport,
};
pub use coverage::{OverlapDescriptor, VariantCoverageReport};
pub use selection::VariantSelectionRefusal;
pub use shape_class::{ShapeVariantDefinition, ShapeVariantId};
