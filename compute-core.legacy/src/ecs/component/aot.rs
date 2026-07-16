//! AOT-specific component types for the kernel catalog pipeline.
//!
//! These components are created and consumed by the ECS systems that port
//! the old AOT catalog logic (catalog.rs, generator.rs, selector.rs,
//! validate.rs, receipts.rs) into the ECS pipeline.

use crate::ecs::plan::KernelTemplateId;
use crate::ecs::Component;
use serde::{Deserialize, Serialize};

/// Catalog schema validation result for a kernel binary.
///
/// Attached by `KernelCatalogSystem` (Phase E) to each `KernelEntity`
/// that holds a `CompiledBinary`. Records whether the binary passes
/// structural checks (data non-empty, fingerprint present, format recognized).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogEntry {
    pub valid: bool,
    pub errors: Vec<String>,
}
impl Component for CatalogEntry {}

/// Data attached to each `KernelVariant` entity identifying its (profile × template)
/// combination and linking back to the parent kernel.
///
/// Created by `VariantGenerationSystem` (Phase E) for every (profile, template)
/// pair derived from a dispatch's `FusionGroup`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelVariantEntityData {
    /// The target Apple Silicon profile (stringified Display form, e.g. "m1_max").
    pub profile_id: String,
    /// Which kernel template this variant uses.
    pub template_id: KernelTemplateId,
    /// Which parent kernel entity this variant belongs to (for grouping in selection).
    pub parent_kernel: CompEntityRef,
}
impl Component for KernelVariantEntityData {}

/// A stable entity reference (opaque u64 wrapper) usable inside component data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CompEntityRef(pub u64);

/// Marks the selected variant on a parent `KernelEntity`, written by
/// `VariantSelectionSystem` (Phase E). Carries the winning profile id and
/// its heuristic score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectedVariant {
    pub profile_id: String,
    pub score: f64,
}
impl Component for SelectedVariant {}

/// Held-out shape validation result for a compiled variant.
///
/// Attached by `CatalogValidationSystem` (Phase F) to each `KernelEntity`
/// that has a `SelectedVariant`. Records whether the compiled binary passes
/// correctness checks against held-out tensor shapes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReceipt {
    pub passed: bool,
    pub nrmse: f64,
    pub perplexity_delta: f64,
}
impl Component for ValidationReceipt {}
