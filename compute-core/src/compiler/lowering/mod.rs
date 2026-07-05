//! Real-backend lowering adapters — prove the new compiler spine preserves
//! the already-qualified MLX, Accelerate, and Core ML routes.
//!
//! The Core ML lowering module provides the general-purpose
//! [`CoreAiLowering`] implementing [`BackendLowering`], replacing
//! the hardcoded `build_matmul_region` bypass.

pub mod accelerate;
pub mod coreai;
pub mod dataset;
#[cfg(feature = "mlx-backend")]
pub mod mlx;
pub mod params;
pub mod receipts;

#[cfg(test)]
mod tests;

use crate::compiler::LoweringReceipt;
use crate::coreai_pipeline::CoreAiIslandReceipt;

/// Receipt produced by the Core ML lowering path (legacy compatibility).
#[derive(Debug)]
pub struct CoreAiLoweringReceipt {
    /// The compiler-level lowering receipt.
    pub lowering: LoweringReceipt,
    /// The full compilation island receipt from the pipeline.
    pub island_receipt: CoreAiIslandReceipt,
    /// Whether the `.mlmodelc` artifact exists on disk.
    pub artifact_exists: bool,
}
